import argparse
import collections
import concurrent.futures
import os

import torch
from PIL import Image
import numpy as np

from ema_pytorch import EMA
from torchvision import transforms
from tqdm import tqdm

from source.models.commons import BroadCastDecoder
from source.models.objs.knet import AKOrN
from source.models.savi import Learned, TransformerPredictor, Corrector
from source.models.savi.model import AkornSAVi
from source.models.slot_attention.decoders import MLPDecoder
from source.models.slot_attention.networks import MLP
from source.utils import str2bool, to_one_hot, grid

TQDM_MIN_INTERVAL = 5
DEVICE = 'cuda'


def get_visualization(model: AkornSAVi, vis_images: torch.Tensor, gt_masks: torch.Tensor, num_slots: int, has_gt_masks: bool):
    output = model(images=vis_images, actions=torch.empty((0, vis_images.shape[1])), prior_slots=None, reconstruct=True,
                       masks=True)
    visualizations = {}
    if has_gt_masks:
        vis_gt_masks = gt_masks.to(torch.int64).squeeze(1)
        vis_gt_masks = to_one_hot(vis_gt_masks, num_classes=num_slots)
        visualizations['ground_truth'] = grid(vis_images, vis_gt_masks)

    for key in ['decoder_masks', 'decoder_masks_hard', 'slot_attention_masks', 'slot_attention_masks_hard',
                'images_reconstruction', 'images_reconstruction_masks',
                'images_reconstruction_masks_hard']:
        field = f'{key}_sequence'
        if field in output:
            visualizations[key] = grid(vis_images, output[field], is_reconstruction='mask' not in field)

    return visualizations


def read_episode_images(episode_id, episode_images_path):
    images = []
    for image_file_name in sorted(os.listdir(episode_images_path), key=lambda x: int(x.split('.')[0])):
        pil_image = Image.open(os.path.join(episode_images_path, image_file_name))
        image = transforms.ToTensor()(pil_image)
        images.append(image)

    return episode_id, torch.stack(images)


class EpisodeImagesReader:
    def __init__(self, split_path, image_size_folder, max_workers, start_index=0, end_index=None):
        self.split_path = split_path
        self.image_size_folder = image_size_folder
        self.max_workers = max_workers
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        all_episode_ids = sorted(os.listdir(self.split_path), key=lambda x: int(x))
        if end_index is None:
            all_episode_ids = all_episode_ids[start_index:]
        else:
            all_episode_ids = all_episode_ids[start_index:end_index]

        self._all_episode_ids = collections.deque(all_episode_ids)
        self._n_episodes = len(self._all_episode_ids)
        self._futures = collections.deque()

    @property
    def n_episodes(self):
        return self._n_episodes

    def shutdown(self):
        self._executor.shutdown()

    def _enqueue_task(self):
        episode_id = self._all_episode_ids.popleft()
        self._futures.append(self._executor.submit(
            read_episode_images, episode_id, os.path.join(self.split_path, episode_id, self.image_size_folder)
        ))

    def read_images(self):
        if len(self._futures) == 0:
            if len(self._all_episode_ids) == 0:
                return None, None

            self._enqueue_task()
            assert len(self._futures) > 0

        # enqueue one task the task at the head of the queue is done and two tasks if it is not done
        # it means that we process images faster than we read them for disk
        for _ in range(min(len(self._all_episode_ids), int(not self._futures[0].done()) + 1)):
            self._enqueue_task()

        return self._futures.popleft().result()

    def read_images_sync(self):
        if len(self._all_episode_ids) == 0:
            return None, None

        episode_id = self._all_episode_ids.popleft()
        return read_episode_images(episode_id, os.path.join(self.split_path, episode_id, self.image_size_folder))


def write_slots(split_path, episode_id, slots_file_name, slots):
    slots = torch.stack(slots).numpy()
    write_episode_path = os.path.join(split_path, episode_id)
    os.makedirs(write_episode_path, exist_ok=True)
    np.save(os.path.join(write_episode_path, slots_file_name), slots)


class EpisodeSlotsWriter:
    def __init__(self, split_path, slots_file_name, max_workers):
        self.split_path = split_path
        self.slots_file_name = slots_file_name
        self.max_workers = max_workers
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        self._futures = collections.deque()

    def shutdown(self):
        self._executor.shutdown()

    def write_slots(self, episode_id, slots):
        self._futures.append(self._executor.submit(
            write_slots, self.split_path, episode_id, self.slots_file_name, slots
        ))

        n = 0
        while len(self._futures) > 0 and self._futures[0].done():
            self._futures.popleft().result()
            n += 1

        return n

    def wait(self):
        for future in self._futures:
            future.result()

        return len(self._futures)


@torch.no_grad()
def extract_akornsavi_slots(akorn_savi: AkornSAVi, device, split, read_root_path, image_size_folder, write_root_path,
                            slots_file_name, max_workers, start_index=0, end_index=None):
    read_split_path = os.path.join(read_root_path, split)
    write_split_path = os.path.join(write_root_path, split)
    episode_images_reader = EpisodeImagesReader(
        read_split_path, image_size_folder, max_workers=max_workers, start_index=start_index, end_index=end_index
    )
    episode_slots_writer = EpisodeSlotsWriter(write_split_path, slots_file_name, max_workers=max_workers)
    pbar_read_episodes = tqdm(total=episode_images_reader.n_episodes, desc="Read Episodes")
    pbar_saved_slots = tqdm(total=episode_images_reader.n_episodes, desc="Saved Slots")
    while True:
        # episode_id, images = episode_images_reader.read_images()
        episode_id, images = episode_images_reader.read_images_sync()
        pbar_read_episodes.update()
        if episode_id is None:
            break

        images = images.to(device)
        prior_slots = None
        slots = []
        for image in images:
            prior_slots = akorn_savi.forward(image.unsqueeze(0).unsqueeze(0), actions=torch.empty(0, 1), prior_slots=prior_slots,
                                       step_offset=0 if prior_slots is None else 1, reconstruct=False)['slots_sequence'][:, 0]
            slots.append(prior_slots[0].cpu())

        n_saved_slots = episode_slots_writer.write_slots(episode_id, slots)
        pbar_saved_slots.update(n_saved_slots)

    pbar_read_episodes.close()
    episode_images_reader.shutdown()
    n_saved_slots = episode_slots_writer.wait()
    pbar_saved_slots.update(n_saved_slots)
    pbar_saved_slots.close()
    episode_slots_writer.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Eval options
    parser.add_argument("--model_path", type=str, help="path to the model")

    # General model options
    parser.add_argument("--L", type=int, default=2, help="num of layers")
    parser.add_argument("--ch", type=int, default=256, help="num of channels")
    parser.add_argument(
        "--model_imsize",
        type=int,
        default=None,
        help="""
        Model's imsize that was set when it was initialized. 
        This is used when evaluating or when finetuning a pretrained model.
        """,
    )
    parser.add_argument("--autorescale", type=str2bool, default=False)
    parser.add_argument("--psize", type=int, default=8, help="patch size")
    parser.add_argument("--T", type=int, default=8, help="num of recurrence")
    parser.add_argument(
        "--maxpool", type=str2bool, default=True, help="max pooling or avg pooling"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="num of heads in self-attention"
    )
    parser.add_argument(
        "--gta",
        type=str2bool,
        default=True,
        help="""
        use Geometric Transform Attention (https://github.com/autonomousvision/gta) as positional encoding.
        If False, use standard absolute positional encoding
        """,
    )

    # AKOrN options
    parser.add_argument("--N", type=int, default=4, help="num of rotating dimensions")
    parser.add_argument("--J", type=str, default="conv", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=False)
    parser.add_argument("--global_omg", type=str2bool, default=False)
    parser.add_argument(
        "--c_norm",
        type=str,
        default="gn",
        help="normalization. gn, sandb(scale and bias), or none",
    )

    parser.add_argument(
        "--use_ro_x",
        type=str2bool,
        default=False,
        help="apply linear transform to oscillators between consecutive layers",
    )

    # ablation of some components in the AKOrN's block
    parser.add_argument(
        "--no_ro", type=str2bool, default=False, help="ablation: no use readout module"
    )
    parser.add_argument(
        "--project",
        type=str2bool,
        default=True,
        help="use projection or not in the Kuramoto layer",
    )
    parser.add_argument('--num_slots', type=int, default=11)
    parser.add_argument('--slot_size', type=int, default=256)
    parser.add_argument('--image_reconstruction_loss_coef', type=float, default=0)
    parser.add_argument('--image_decoder', choices=['mlp', 'spatial_broadcast'], type=str, default='mlp')
    parser.add_argument('--image_decoder_hidden_dim', type=int, default=64)
    parser.add_argument("--from_checkpoint", type=str, required=False)
    parser.add_argument("--load_checkpoint_strict", type=str2bool, default=True)
    parser.add_argument("--read_root_path", type=str, default=True)
    parser.add_argument("--write_root_path", type=str, default=True)
    parser.add_argument("--slots_file_name", type=str, default='slots_akornsavi.npy')
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--use_train_split", type=str2bool, default=True)
    parser.add_argument("--use_val_split", type=str2bool, default=True)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    if args.model_imsize % 8 != 0:
        raise ValueError('Image size:', args.model_imsize, 'Patch size:', 8)

    n_patches = (args.model_imsize // 8) ** 2

    encoder = AKOrN(
        args.N,
        ch=args.ch,
        L=args.L,
        T=args.T,
        J=args.J,  # "conv" or "attn",
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        c_norm=args.c_norm,
        psize=args.psize,
        imsize=args.model_imsize,
        autorescale=args.autorescale,
        maxpool=args.maxpool,
        project=args.project,
        heads=args.heads,
        use_ro_x=args.use_ro_x,
        no_ro=args.no_ro,
        gta=args.gta,
    )

    encoder = EMA(encoder)
    encoder.load_state_dict(torch.load(args.model_path, weights_only=True)["model_state_dict"])
    encoder = encoder.ema_model
    encoder = encoder.eval()
    encoder.requires_grad_(False)

    features_projector = MLP(
        inp_dim=args.ch,
        outp_dim=args.slot_size,
        hidden_dims=[2 * 256],
        initial_layer_norm=True,)

    initializer = Learned(num_slots=args.num_slots, slot_dim=args.slot_size)
    slot_attention = Corrector(
        num_slots=args.num_slots,
        slot_dim=args.slot_size,
        feature_dim=args.slot_size,
        num_iterations=1,
        num_initial_iterations=3,
        hidden_dim=4 * args.slot_size,
    )

    decoder = MLPDecoder(inp_dim=args.slot_size, outp_dim=args.ch, hidden_dims=[512, 512, 512], n_patches=n_patches)
    if args.image_reconstruction_loss_coef > 0:
        if args.image_decoder == 'mlp':
            image_decoder = MLPDecoder(inp_dim=args.slot_size, outp_dim=3, hidden_dims=[args.image_decoder_hidden_dim] * 3, n_patches=args.model_imsize ** 2)
        else:
            image_decoder = BroadCastDecoder(obs_size=args.model_imsize, obs_channels=3, hidden_size=args.image_decoder_hidden_dim, slot_size=args.slot_size)
    else:
        image_decoder = None

    predictor = TransformerPredictor(slot_dim=args.slot_size, action_dim=-1,)
    akornsavi = AkornSAVi(encoder, features_projector, initializer, slot_attention, decoder, predictor, image_decoder, is_encoder_frozen=True).to('cuda')
    sd = torch.load(args.from_checkpoint)
    missing, unexpected = akornsavi.load_state_dict(
        torch.load(args.from_checkpoint)['model'], strict=args.load_checkpoint_strict
    )
    print('Loading weights from checkpoint:', args.from_checkpoint)
    print('Missing parameters:', missing)
    print('Unexpected parameters:', unexpected)

    akornsavi = akornsavi.eval().to(DEVICE)
    image_size_folder = f"obs_{'x'.join([str(args.model_imsize)] * 2)}"
    if args.use_train_split:
        extract_akornsavi_slots(
            akornsavi, DEVICE, "train", args.read_root_path, image_size_folder, args.write_root_path,
            args.slots_file_name, args.max_workers, args.start_index, args.end_index
        )

    if args.use_val_split:
        extract_akornsavi_slots(
            akornsavi, DEVICE, "val", args.read_root_path, image_size_folder, args.write_root_path,
            args.slots_file_name, args.max_workers
        )




