import argparse
import math
import os

import comet_ml
from comet_ml import CometExperiment

import torch
from ema_pytorch import EMA
from tqdm import tqdm

from source.models.objs.knet import AKOrN
from source.models.slotcontrast.loss import SlotSlotContrastiveLoss
from source.models.slotcontrast.model import SlotContrastAkornSAur
from source.models.slotcontrast.modules.decoders import MLPDecoder
from source.models.slotcontrast.modules.initializer import FixedLearnedInit
from source.models.videosaur.modules.groupers import SlotAttention
from source.models.videosaur.modules.networks import MLP, TransformerEncoder
from source.models.videosaur.modules.video import LatentProcessor
from source.training_utils import ExpDecayWithLinearWarmupScheduler
from source.utils import str2bool, to_one_hot, AdjustedRandIndex, grid

TQDM_MIN_INTERVAL = 5
DEVICE = 'cuda'


def get_loader(data, data_root, imsize, batchsize, drop_last=False, num_workers=0, is_eval=False,
               image_file_extension=None, kind='image', sequence_length=1):
    from source.data.datasets.objs.load_data import load_data

    dataset, imsize, collate_fn = load_data(data, data_root, imsize, is_eval=is_eval, kind=kind,
                                            image_file_extension=image_file_extension, sequence_length=sequence_length)

    kwargs = {'batch_size': batchsize, 'num_workers': num_workers, 'drop_last': drop_last, 'shuffle': True}
    if data in ("clevrtex_full", "clevrtex_outd", "clevrtex_camo", "coco"):
        kwargs['collate_fn'] = collate_fn

    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader, imsize


def maybe_log_wandb(experiment: CometExperiment, step, record, visualizations_dict=None):
    if experiment is None:
        return

    experiment.log_metrics(record, step=step)
    if visualizations_dict is not None:
        for key, visualization in visualizations_dict.items():
            experiment.log_image(visualization, name=key, step=step)


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
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--data_imsize",
        type=int,
        default=None,
        help="Image size. If None, use the default size of each dataset",
    )

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
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--contrast_loss_coeff", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument('--warmup_iters', type=int, default=2500)
    parser.add_argument('--decay_steps', type=int, default=100000)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument("--grad_norm_clip", type=float, default=0.05)
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
    parser.add_argument("--wandb_run_id", type=str, required=False)
    parser.add_argument("--image_file_extension", type=str, required=False)
    parser.add_argument("--from_checkpoint", type=str, required=False)

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

    encoder_output_transform = MLP(
        inp_dim=args.ch, outp_dim=args.slot_size, hidden_dims=[2 * args.ch], initial_layer_norm=True,
    )
    initializer = FixedLearnedInit(n_slots=args.num_slots, dim=args.slot_size)
    slot_attention = SlotAttention(
        inp_dim=args.slot_size, slot_dim=args.slot_size, n_iters=2, use_mlp=True,
    )
    decoder = MLPDecoder(inp_dim=args.slot_size, outp_dim=args.ch, hidden_dims=[1024, 1024, 1024],
                         n_patches=n_patches)
    predictor = TransformerEncoder(dim=args.slot_size, n_blocks=1, n_heads=4,)
    latent_processor = LatentProcessor(corrector=slot_attention, predictor=predictor, first_step_corrector_args={'n_iters': 3})
    slot_contrast_akornsaur = SlotContrastAkornSAur(
        encoder=encoder, encoder_output_transform=encoder_output_transform, initializer=initializer, decoder=decoder,
        latent_processor=latent_processor, is_encoder_frozen=True,
    ).to(DEVICE)
    ss_contrast_loss = SlotSlotContrastiveLoss(temperature=args.temperature,)

    optimizer = torch.optim.Adam(slot_contrast_akornsaur.parameters(), lr=args.lr)
    scheduler = ExpDecayWithLinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters,
                                                  decay_steps=args.decay_steps, decay_rate=args.decay_rate)

    global_step = 0
    start_epoch = -1
    best_val_loss = math.inf
    val_loss = math.inf
    if args.from_checkpoint is not None:
        sd = torch.load(args.from_checkpoint)
        global_step = sd['global_step']
        start_epoch = sd['epoch']
        best_val_loss = sd['val_loss']
        val_loss = sd['val_loss']

        slot_contrast_akornsaur.load_state_dict(sd['model'])
        optimizer.load_state_dict(sd['optimizer'])
        scheduler = ExpDecayWithLinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters,
                                                      decay_steps=args.decay_steps, decay_rate=args.decay_rate,
                                                      last_iter=global_step)

    train_dataloader, _ = get_loader(args.data, args.data_root, args.model_imsize, args.batchsize, drop_last=True,
                                     num_workers=args.num_workers, is_eval=False, kind='video',
                                     image_file_extension=args.image_file_extension, sequence_length=args.sequence_length,)
    val_dataloader, _ = get_loader(args.data, args.data_root, args.model_imsize, args.batchsize, drop_last=False,
                                     num_workers=args.num_workers, is_eval=True, kind='video',
                                     image_file_extension=args.image_file_extension, sequence_length=args.sequence_length,)

    experiment: CometExperiment = None
    if args.wandb_project is not None and len(args.wandb_project) > 0:
        mode = 'create' if args.wandb_run_id is None else 'get'
        experiment = comet_ml.start(project_name=args.wandb_project, experiment_key=args.wandb_run_id, mode=mode)
        experiment.add_tag(args.wandb_run_name)
        experiment.set_name(args.wandb_run_name)
        experiment.log_parameters(vars(args))

    for epoch in range(start_epoch + 1, args.epochs):
        slot_contrast_akornsaur.train(True)
        epoch_loss = 0
        epoch_loss_mse = 0
        epoch_loss_contrast = 0
        n_batches = len(train_dataloader)
        train_pbar = tqdm(enumerate(train_dataloader), desc='Training', mininterval=TQDM_MIN_INTERVAL)
        for i, batch in train_pbar:
            if isinstance(batch, tuple):
                assert len(batch) == 4, f'Expected 4 elements, but has: {len(batch)}'
                _, images, _, _ = batch
            else:
                images = batch

            images = images.to(DEVICE)
            global_step += 1
            output = slot_contrast_akornsaur(images=images, actions=torch.empty((0, batch.shape[1])), prior_slots=None, reconstruct=True)
            optimizer.zero_grad()
            mse_loss = torch.nn.functional.mse_loss(output['features_reconstruction_sequence'], output['features_sequence'])
            contrast_loss = ss_contrast_loss(output['slots_sequence'])
            total_loss = mse_loss + args.contrast_loss_coeff * contrast_loss
            total_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(slot_contrast_akornsaur.parameters(), args.grad_norm_clip)
            optimizer.step()
            scheduler.step()

            mse_loss = mse_loss.item()
            contrast_loss = contrast_loss.item()
            total_loss = total_loss.item()
            epoch_loss += total_loss
            epoch_loss_mse += mse_loss
            epoch_loss_contrast += contrast_loss
            record = {}
            record['global_step'] = global_step
            record['train/step_loss'] = total_loss
            record['train/step_mse_loss'] = mse_loss
            record['train/step_contrast_loss'] = contrast_loss
            record['train/grad_norm'] = grad_norm.item()
            record.update({f'lr_{k}': lr for k, lr in enumerate(scheduler.get_lr())})

            if i == n_batches - 1:
                record['global_step'] = global_step
                record['train/epoch_loss'] = epoch_loss / n_batches
                record['train/epoch_loss_mse'] = epoch_loss_mse / n_batches
                record['train/epoch_loss_contrast'] = epoch_loss_contrast / n_batches

            train_pbar.set_postfix(record, refresh=False)
            if i == n_batches - 1 or global_step % args.log_every_n_steps == 0:
                maybe_log_wandb(experiment, global_step, record)

        train_pbar.close()


        val_loss = 0
        val_loss_mse = 0
        val_loss_contrast = 0
        k = 0
        val_n_batches = len(val_dataloader)
        slot_contrast_akornsaur.train(False)
        visualizations = {}
        record = {'global_step': global_step, 'epoch': epoch}
        ari_slot_attention = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        ari_decoder = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        val_pbar = tqdm(enumerate(val_dataloader), desc='Validation', mininterval=TQDM_MIN_INTERVAL)
        for i, batch in val_pbar:
            if isinstance(batch, tuple):
                assert len(batch) == 4, f'Expected 4 elements, but has: {len(batch)}'
                _, images, gt_masks, _ = batch
                gt_masks = gt_masks.to(DEVICE)
                # treat segmentations with class_id <= 0 as background
                gt_masks = torch.as_tensor(gt_masks > 0, dtype=gt_masks.dtype) * gt_masks
                has_gt_masks = True
            else:
                images = batch
                gt_masks = None
                has_gt_masks = False

            images = images.to(DEVICE)
            k += images.size()[0]
            do_need_log_ari = has_gt_masks and epoch % args.log_ari_every_n_epochs == 0
            do_need_visualize = epoch % args.visualize_every_n_epochs == 0
            if (i == 0 and do_need_visualize) or do_need_log_ari:
                output = slot_contrast_akornsaur(images=images, actions=torch.empty((0, batch.shape[1])), prior_slots=None, reconstruct=True, masks=True)
                vis_images = images[:args.visualize_n_images]
                slot_attention_masks = output["slot_attention_masks_hard_sequence"][:args.visualize_n_images]
                decoder_masks = output["decoder_masks_hard_sequence"][:args.visualize_n_images]
                if has_gt_masks:
                    vis_gt_masks = gt_masks[:args.visualize_n_images].to(torch.int64).squeeze(1)
                    vis_gt_masks = to_one_hot(vis_gt_masks, num_classes=args.num_slots)
                    visualizations['ground_truth'] = grid(vis_images, vis_gt_masks)
                else:
                    vis_gt_masks = None

                visualizations['decoder_masks'] = grid(vis_images,
                                                   output["decoder_masks_sequence"][:args.visualize_n_images])
                visualizations['decoder_masks_hard'] = grid(vis_images,
                                                   output["decoder_masks_hard_sequence"][:args.visualize_n_images])
                visualizations['slot_attention_masks'] = grid(vis_images,
                                                   output["slot_attention_masks_sequence"][:args.visualize_n_images])
                visualizations['slot_attention_masks_hard'] = grid(vis_images,
                                                        output["slot_attention_masks_hard_sequence"][:args.visualize_n_images])
                del slot_attention_masks, decoder_masks
            else:
                output = slot_contrast_akornsaur(images=images, actions=torch.empty((0, batch.shape[1])), prior_slots=None, reconstruct=True)

            mse_loss = torch.nn.functional.mse_loss(output['features_reconstruction_sequence'], output['features_sequence'])
            contrast_loss = ss_contrast_loss(output['slots_sequence'])
            val_loss_mse += mse_loss.item() * images.size()[0]
            val_loss_contrast += contrast_loss.item() * images.size()[0]
            val_loss += (mse_loss + args.contrast_loss_coeff * contrast_loss).item() * images.size()[0]
            if do_need_log_ari:
                gt_masks = to_one_hot(gt_masks.to(torch.int64).squeeze(1)).to(torch.bool)
                ari_slot_attention.update(gt_masks, output["slot_attention_masks_hard_sequence"])
                ari_decoder.update(gt_masks, output["decoder_masks_hard_sequence"])
            if i == val_n_batches - 1:
                val_loss /= k
                val_loss_mse /= k
                val_loss_contrast /= k
                record['val/loss'] = val_loss
                record['val/loss_mse'] = val_loss_mse
                record['val/loss_contrast'] = val_loss_contrast
                if do_need_log_ari:
                    record['val/ari_slot_attention'] = ari_slot_attention.compute().item()
                    record['val/ari_decoder'] = ari_decoder.compute().item()
                val_pbar.set_postfix(record)

        val_pbar.close()

        maybe_log_wandb(experiment, global_step, record, visualizations)
        if epoch % args.save_every_n_epochs == 0:
            checkpoint = {'model': slot_contrast_akornsaur.state_dict(), 'optimizer': optimizer.state_dict(),
                          'global_step': global_step, 'epoch': epoch, 'val_loss': val_loss, }
            checkpoint_folder = os.path.join(args.save_path, args.wandb_run_name)
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_folder, 'checkpoint.pt'))

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            checkpoint = {'model': slot_contrast_akornsaur.state_dict(), 'optimizer': optimizer.state_dict(),
                          'global_step': global_step, 'epoch': epoch, 'val_loss': val_loss, }
            checkpoint_folder = os.path.join(args.save_path, args.wandb_run_name)
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_folder, 'best_checkpoint.pt'))
