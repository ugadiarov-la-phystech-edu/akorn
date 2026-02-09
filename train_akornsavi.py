import argparse
import math
import os
import sys
from collections import defaultdict

import comet_ml
import torch
from comet_ml import CometExperiment
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from ema_pytorch import EMA
from tqdm import tqdm

from source.logging.local_logger import LocalLogger
from source.models.commons import BroadCastDecoder
from source.models.objs.knet import AKOrN
from source.models.savi import Learned, TransformerPredictor, Corrector
from source.models.savi.model import AkornSAVi
from source.models.slot_attention.decoders import MLPDecoder
from source.models.slot_attention.networks import MLP
from source.training_utils import ExpDecayWithLinearWarmupScheduler
from source.utils import str2bool, to_one_hot, AdjustedRandIndex, grid

TQDM_MIN_INTERVAL = 5
DEVICE = 'cuda'


def set_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_loader(data, data_root, episode_folder_pattern, imsize, batchsize, ddp_config, drop_last=False, num_workers=0, is_eval=False,
               image_file_extension=None, kind='image', sequence_length=1):
    from source.data.datasets.objs.load_data import load_data

    dataset, imsize, collate_fn = load_data(data, data_root, episode_folder_pattern, imsize, is_eval=is_eval, kind=kind,
                                            image_file_extension=image_file_extension, sequence_length=sequence_length)

    kwargs = {'batch_size': batchsize, 'num_workers': num_workers, 'drop_last': drop_last, 'shuffle': True}
    if data in ("clevrtex_full", "clevrtex_outd", "clevrtex_camo", "coco"):
        kwargs['collate_fn'] = collate_fn

    if ddp_config['world_size'] > 1:
        sampler = DistributedSampler(dataset, num_replicas=ddp_config['world_size'],
                                     rank=ddp_config['local_rank'], shuffle=True, drop_last=drop_last)
        del kwargs['shuffle']
        del kwargs['drop_last']
        kwargs['sampler'] = sampler

    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader, imsize


def maybe_log_wandb(experiment: CometExperiment, local_logger: LocalLogger, step, record, visualizations_dict=None):
    if local_logger is not None:
        local_logger.log_metrics(record, step)
        if visualizations_dict is not None:
            local_logger.log_images(visualizations_dict, step)

    if experiment is None:
        return

    experiment.log_metrics(record, step=step)
    if visualizations_dict is not None:
        for key, visualization in visualizations_dict.items():
            experiment.log_image(visualization, name=key, step=step)


def step(model: AkornSAVi, images: torch.Tensor, do_train: bool, do_need_log_ari: bool = False,
         gt_masks: torch.Tensor = None, ari_slot_attention = None, ari_decoder = None):
    output = model(images=images, actions=torch.empty((0, images.shape[1])), prior_slots=None, reconstruct=True)

    features_reconstruction_loss = torch.nn.functional.mse_loss(output['features_sequence'],
                                                                output['features_reconstruction_sequence'])
    if args.image_reconstruction_loss_coef > 0:
        images_reconstruction_loss = torch.nn.functional.mse_loss(images, output['images_reconstruction_sequence'])
    else:
        images_reconstruction_loss = torch.as_tensor(0)

    loss = features_reconstruction_loss + args.image_reconstruction_loss_coef * images_reconstruction_loss
    output = dict(loss=loss.item(), features_reconstruction_loss=features_reconstruction_loss.item(),
                  images_reconstruction_loss=images_reconstruction_loss.item())

    if do_train:
        loss.backward()

    if do_need_log_ari:
        gt_masks = to_one_hot(gt_masks.to(torch.int64).squeeze(1)).to(torch.bool)
        ari_slot_attention.update(gt_masks, output["slot_attention_masks_hard_sequence"])
        ari_decoder.update(gt_masks, output["decoder_masks_hard_sequence"])

    return output


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
            visualizations[key] = grid(vis_images, output[field])

    return visualizations


def ddp_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def ddp_cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Eval options
    parser.add_argument("--model_path", type=str, help="path to the model")

    # Data loading
    parser.add_argument("--data", type=str, default="clevrtex_full")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="optional. you can specify the dir path if the default path of each dataset is not appropritate one. Currently only applied to ImageNet",
    )
    parser.add_argument("--data_episode_folder_pattern", type=str, default="*")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--data_imsize",
        type=int,
        default=None,
        help="Image size. If None, use the default size of each dataset",
    )

    # General model options
    parser.add_argument("--L", type=int, default=1, help="num of layers")
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
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
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
        "--init_omg", type=float, default=0.01, help="initial omega length"
    )
    parser.add_argument("--learn_omg", type=str2bool, default=False)
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
    parser.add_argument('--per_slot_initialization', type=str2bool, default=False)
    parser.add_argument('--image_reconstruction_loss_coef', type=float, default=0)
    parser.add_argument('--image_decoder', choices=['mlp', 'spatial_broadcast'], type=str, default='mlp')
    parser.add_argument('--image_decoder_hidden_dim', type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument('--warmup_iters', type=int, default=2500)
    parser.add_argument('--decay_steps', type=int, default=100000)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument("--grad_norm_clip", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--log_ari_every_n_epochs", type=int, default=1)
    parser.add_argument("--visualize_every_n_epochs", type=int, default=1)
    parser.add_argument("--visualize_n_images", type=int, default=2)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument(
        "--wandb_project", type=str, required=False,
    )
    parser.add_argument(
        "--wandb_group", type=str, required=False,
    )
    parser.add_argument(
        "--wandb_run_name", type=str, required=False,
    )
    parser.add_argument(
        "--wandb_run_id", type=str, required=False,
    )
    parser.add_argument("--image_file_extension", type=str, required=False)
    parser.add_argument("--from_checkpoint", type=str, required=False)
    parser.add_argument("--load_checkpoint_strict", type=str2bool, default=True)
    parser.add_argument("--freeze_loaded_weights", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_local_logger", type=str2bool, default=False)

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)
    set_seed(args.seed)

    if args.model_imsize % 8 != 0:
        raise ValueError('Image size:', args.model_imsize, 'Patch size:', 8)

    n_patches = (args.model_imsize // 8) ** 2

    encoder = AKOrN(
        args.N,
        ch=args.ch,
        L=args.L,
        T=args.T,
        gamma=args.gamma,
        J=args.J,  # "conv" or "attn",
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        c_norm=args.c_norm,
        psize=args.psize,
        imsize=args.model_imsize,
        autorescale=args.autorescale,
        init_omg=args.init_omg,
        learn_omg=args.learn_omg,
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
    optimizer = torch.optim.Adam(akornsavi.parameters(), lr=args.lr)
    scheduler = ExpDecayWithLinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters,
                                                  decay_steps=args.decay_steps, decay_rate=args.decay_rate)
    start_epoch = -1
    global_step = 0
    param_names_to_freeze = set()
    if args.from_checkpoint is not None:
        sd = torch.load(args.from_checkpoint)
        missing, unexpected = akornsavi.load_state_dict(
            torch.load(args.from_checkpoint)['model'], strict=args.load_checkpoint_strict
        )
        print('Loading weights from checkpoint:', args.from_checkpoint)
        print('Missing parameters:', missing)
        print('Unexpected parameters:', unexpected)
        optimizer.load_state_dict(sd['optimizer'])
        scheduler.load_state_dict(sd['scheduler'])
        start_epoch = sd['epoch']
        global_step = sd['global_step']
        if args.freeze_loaded_weights:
            param_names_to_freeze = set(torch.load(args.from_checkpoint)['model'].keys())
            for name, param in akornsavi.named_parameters():
                if name in param_names_to_freeze:
                    param.requires_grad = False

                param = None

    use_ddp = torch.cuda.device_count() > 1 and DEVICE == 'cuda'
    if use_ddp:
        rank, local_rank, world_size = ddp_setup()
        ddp_config = {'world_size': world_size, 'rank': rank, 'local_rank': local_rank}
        DEVICE = local_rank
        print(f"Using {torch.cuda.device_count()} GPUs!")
        akornsavi.to(ddp_config['local_rank'])
        akornsavi = DDP(akornsavi, device_ids=[ddp_config['local_rank']])
    else:
        ddp_config = {'world_size': 1, 'rank': 0, 'local_rank': 0}
        akornsavi.to(DEVICE)

    train_dataloader, _ = get_loader(args.data, args.data_root, args.data_episode_folder_pattern, args.model_imsize,
                                     args.batchsize, ddp_config, drop_last=True, num_workers=args.num_workers,
                                     is_eval=False, kind='video',
                                     image_file_extension=args.image_file_extension, sequence_length=args.sequence_length)
    val_dataloader, _ = get_loader(args.data, args.data_root, args.data_episode_folder_pattern, args.model_imsize,
                                   args.batchsize, ddp_config, drop_last=False, num_workers=args.num_workers,
                                   is_eval=True, kind='video', image_file_extension=args.image_file_extension,
                                   sequence_length=args.sequence_length)

    experiment: CometExperiment = None
    if args.wandb_project is not None and len(args.wandb_project) > 0 and ddp_config['rank'] == 0:
        mode = 'create' if args.wandb_run_id is None else 'get'
        experiment = comet_ml.start(project_name=args.wandb_project, experiment_key=args.wandb_run_id, mode=mode)
        experiment.add_tag(args.wandb_run_name)
        experiment.set_name(args.wandb_run_name)
        experiment.log_system_info('command', ' '.join(sys.argv))

        import socket
        experiment.log_system_info('hostname', socket.gethostname())
    
    local_logger: LocalLogger = None
    if args.use_local_logger:
        local_logger = LocalLogger(args.save_path)

    best_val_loss = math.inf
    for epoch in range(start_epoch + 1, args.epochs):
        akornsavi.train(True)
        not_frozen_params = [name for name, param in akornsavi.named_parameters() if name in param_names_to_freeze and param.requires_grad]
        assert len(not_frozen_params) == 0, f'These parameters are expected to be frozen: {not_frozen_params}'

        epoch_loss = 0
        epoch_features_reconstruction_loss = 0
        epoch_images_reconstruction_loss = 0
        n_batches = len(train_dataloader)
        n_updates = n_batches // args.gradient_accumulation_steps
        optimizer.zero_grad()
        if use_ddp:
            train_dataloader.sampler.set_epoch(epoch)

        if ddp_config['rank'] == 0:
            train_pbar = tqdm(enumerate(train_dataloader), desc='Training', mininterval=TQDM_MIN_INTERVAL)
        else:
            train_pbar = enumerate(train_dataloader)

        record = defaultdict(float)
        for i, batch in train_pbar:
            if isinstance(batch, tuple):
                assert len(batch) == 4, f'Expected 4 elements, but has: {len(batch)}'
                _, images, _, _ = batch
            else:
                images = batch

            train_step_output = step(akornsavi, images.to(DEVICE), do_train=True)
            record['train/step_loss'] += train_step_output['loss']
            record['train/step_features_reconstruction_loss'] += train_step_output['features_reconstruction_loss']
            record['train/step_images_reconstruction_loss'] += train_step_output['images_reconstruction_loss']
            if (i + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                grad_norm = torch.nn.utils.clip_grad_norm_(akornsavi.parameters(), args.grad_norm_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                record = {k: v / args.gradient_accumulation_steps for k, v in record.items() if 'loss' in k}
                epoch_loss += train_step_output['loss']
                epoch_features_reconstruction_loss += train_step_output['features_reconstruction_loss']
                epoch_images_reconstruction_loss += train_step_output['images_reconstruction_loss']
                record['train/grad_norm'] = grad_norm.item()
                record['global_step'] = global_step
                record.update({f'lr_{k}': lr for k, lr in enumerate(scheduler.get_lr())})

                if global_step % n_updates == 0:
                    record['train/epoch_loss'] = epoch_loss / n_updates
                    record['train/epoch_features_reconstruction_loss'] = epoch_features_reconstruction_loss / n_updates
                    record['train/epoch_images_reconstruction_loss'] = epoch_images_reconstruction_loss / n_updates

                if ddp_config['rank'] == 0:
                    train_pbar.set_postfix(record, refresh=False)
                    if global_step % n_updates == 0 or global_step % args.log_every_n_steps == 0:
                        maybe_log_wandb(experiment, local_logger, global_step, record)

                record = defaultdict(float)

        if ddp_config['rank'] == 0:
            train_pbar.close()

        val_loss = 0
        val_features_reconstruction_loss = 0
        val_images_reconstruction_loss = 0
        k = 0
        val_n_batches = len(val_dataloader)
        akornsavi.train(False)
        visualizations = {}
        record = {'global_step': global_step, 'epoch': epoch}
        ari_slot_attention = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        ari_decoder = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        if ddp_config['rank'] == 0:
            val_pbar = tqdm(enumerate(val_dataloader), desc='Validation', mininterval=TQDM_MIN_INTERVAL)
        else:
            val_pbar = enumerate(val_dataloader)

        for i, batch in val_pbar:
            if isinstance(batch, tuple):
                assert len(batch) == 4, f'Expected 4 elements, but has: {len(batch)}'
                _, images, gt_masks, _ = batch
                # treat segmentations with class_id <= 0 as background
                gt_masks = torch.as_tensor(gt_masks > 0, dtype=gt_masks.dtype) * gt_masks
                has_gt_masks = True
            else:
                images = batch
                gt_masks = None
                has_gt_masks = False

            do_need_log_ari = has_gt_masks and epoch % args.log_ari_every_n_epochs == 0
            batch_size = images.size()[0]
            k += batch_size
            val_step_output = step(akornsavi, images=images.to(DEVICE), do_train=False,
                                   do_need_log_ari=do_need_log_ari, gt_masks=gt_masks.to(DEVICE) if has_gt_masks else None,
                                   ari_slot_attention=ari_slot_attention, ari_decoder=ari_decoder)
            val_loss += val_step_output['loss'] * batch_size
            val_features_reconstruction_loss += val_step_output['features_reconstruction_loss'] * batch_size
            val_images_reconstruction_loss += val_step_output['images_reconstruction_loss'] * batch_size
            if i == val_n_batches - 1:
                val_loss /= k
                val_features_reconstruction_loss /= k
                val_images_reconstruction_loss /= k
                record['val/loss'] = val_loss
                record['val/val_features_reconstruction_loss'] = val_features_reconstruction_loss
                record['val/val_images_reconstruction_loss'] = val_images_reconstruction_loss
                if do_need_log_ari:
                    record['val/ari_slot_attention'] = ari_slot_attention.compute().item()
                    record['val/ari_decoder'] = ari_decoder.compute().item()

        if ddp_config['rank'] == 0:
            val_pbar.set_postfix(record)
            val_pbar.close()

            do_need_visualize = epoch % args.visualize_every_n_epochs == 0
            if do_need_visualize:
                batch = next(iter(val_dataloader))
                if isinstance(batch, tuple):
                    assert len(batch) == 4, f'Expected 4 elements, but has: {len(batch)}'
                    _, images, gt_masks, _ = batch
                    # treat segmentations with class_id <= 0 as background
                    gt_masks = torch.as_tensor(gt_masks > 0, dtype=gt_masks.dtype) * gt_masks
                    has_gt_masks = True
                else:
                    images = batch
                    gt_masks = None
                    has_gt_masks = False

                visualizations = get_visualization(akornsavi, images[:args.visualize_n_images].to(DEVICE),
                                                   gt_masks[:args.visualize_n_images].to(DEVICE) if has_gt_masks else None,
                                                   args.num_slots, has_gt_masks)

            maybe_log_wandb(experiment, local_logger, global_step, record, visualizations)
            if use_ddp:
                state_dict = akornsavi.module.state_dict()
            else:
                state_dict = akornsavi.state_dict()

            if epoch % args.save_every_n_epochs == 0:
                checkpoint = {'model': state_dict, 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'global_step': global_step, 'epoch': epoch,
                              'val_loss': val_loss, 'best_val_loss': best_val_loss}
                checkpoint_folder = os.path.join(args.save_path, args.wandb_run_name)
                os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(checkpoint, os.path.join(checkpoint_folder, 'checkpoint.pt'))

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                checkpoint = {'model': state_dict, 'optimizer': optimizer.state_dict(),
                              'scheduler': scheduler.state_dict(), 'global_step': global_step, 'epoch': epoch,
                              'val_loss': val_loss, 'best_val_loss': best_val_loss}
                checkpoint_folder = os.path.join(args.save_path, args.wandb_run_name)
                os.makedirs(checkpoint_folder, exist_ok=True)
                torch.save(checkpoint, os.path.join(checkpoint_folder, 'best_checkpoint.pt'))

    if use_ddp:
        ddp_cleanup()
