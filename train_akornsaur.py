import argparse
import os

import torch
import wandb
from ema_pytorch import EMA
from tqdm import tqdm

from source.models.objs.knet import AKOrN
from source.models.slot_attention.akornsaur import AkornSAur
from source.models.slot_attention.decoders import MLPDecoder
from source.models.slot_attention.initializers import RandomInit
from source.models.slot_attention.networks import MLP
from source.models.slot_attention.slot_attention import SlotAttention
from source.training_utils import ExpDecayWithLinearWarmupScheduler
from source.utils import str2bool, to_one_hot, grid_numpy, AdjustedRandIndex


TQDM_MIN_INTERVAL = 5


def get_loader(data, data_root, imsize, batchsize, drop_last=False, num_workers=0, is_eval=False):
    from source.data.datasets.objs.load_data import load_data

    dataset, imsize, collate_fn = load_data(data, data_root, imsize, is_eval=is_eval, kind='image')

    kwargs = {'batch_size': batchsize, 'num_workers': num_workers, 'drop_last': drop_last, 'shuffle': True}
    if data in ("clevrtex_full", "clevrtex_outd", "clevrtex_camo", "coco"):
        kwargs['collate_fn'] = collate_fn

    loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return loader, imsize


def maybe_log_wandb(record, wandb_project, wandb_group, wandb_run_name, path=None):
    if len(record) == 0 or wandb_project is None:
        return

    if wandb.run is None:
        if path is None:
            path = os.path.join('wandb', args.wandb_run_name)

        wandb.init(project=args.wandb_project, group=wandb_group, name=wandb_run_name, dir=path, config=vars(args),)

    wandb.log(record)


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
    parser.add_argument("--batchsize", type=int, default=250)
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
    parser.add_argument("--lr", type=float, default=0.0004)
    parser.add_argument('--warmup_iters', type=int, default=10000)
    parser.add_argument('--decay_steps', type=int, default=100000)
    parser.add_argument('--decay_rate', type=float, default=0.5)
    parser.add_argument("--grad_norm_clip", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--log_every_n_steps", type=int, default=200)
    parser.add_argument("--log_metrics_every_n_epochs", type=int, default=1)
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

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)


    n_patches = (128 // 8) ** 2

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
        inp_dim=256,
        outp_dim=args.slot_size,
        hidden_dims=[2 * 256],
        initial_layer_norm=True,)

    initializer = RandomInit(n_slots=args.num_slots, dim=args.slot_size)
    slot_attention = SlotAttention(
        inp_dim=args.slot_size,
        slot_dim=args.slot_size,
        n_iters=3,
        use_mlp=True,)

    decoder = MLPDecoder(inp_dim=args.slot_size, outp_dim=256, hidden_dims=[512, 512, 512], n_patches=n_patches)
    akornsaur = AkornSAur(encoder, features_projector, initializer, slot_attention, decoder, is_encoder_frozen=True).to('cuda')
    optimizer = torch.optim.Adam(akornsaur.parameters(), lr=args.lr)
    scheduler = ExpDecayWithLinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters,
                                                  decay_steps=args.decay_steps, decay_rate=args.decay_rate)

    train_dataloader, _ = get_loader(args.data, args.data_root, args.model_imsize, args.batchsize, drop_last=True,
                                     num_workers=args.num_workers, is_eval=False)
    val_dataloader, _ = get_loader(args.data, args.data_root, args.model_imsize, args.batchsize, drop_last=False,
                                     num_workers=args.num_workers, is_eval=True)
    if args.wandb_project is not None:
        path = os.path.join('wandb', args.wandb_run_name)
        os.makedirs(path, exist_ok=True)
        wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.wandb_run_name, dir=path,
                   config=vars(args),)

    global_step = 0
    for epoch in range(args.epochs):
        akornsaur.train(True)
        epoch_loss = 0
        n_batches = len(train_dataloader)
        train_pbar = tqdm(enumerate(train_dataloader), desc='Training', mininterval=TQDM_MIN_INTERVAL)
        for i, (_, images, _, _) in train_pbar:
            images = images.to('cuda')
            global_step += 1
            loss, aux_output = akornsaur.step(images, do_predict_masks=False)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(akornsaur.parameters(), args.grad_norm_clip)
            optimizer.step()
            scheduler.step()

            loss = loss.item()
            epoch_loss += loss
            record = {}
            record['global_step'] = global_step
            record['train/step_loss'] = loss / args.batchsize
            record['train/grad_norm'] = grad_norm.item()
            record.update({f'lr_{k}': lr for k, lr in enumerate(scheduler.get_lr())})

            if i == n_batches - 1:
                record['global_step'] = global_step
                record['train/epoch_loss'] = epoch_loss / n_batches / args.batchsize

            train_pbar.set_postfix(record, refresh=False)
            if i == n_batches - 1 or global_step % args.log_every_n_steps == 0:
                maybe_log_wandb(record, args.wandb_project, args.wandb_group, args.wandb_run_name)

        train_pbar.close()


        val_loss = 0
        k = 0
        val_n_batches = len(val_dataloader)
        akornsaur.train(False)
        vis_grid = None
        record = {'global_step': global_step, 'epoch': epoch}
        ari_slot_attention = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        ari_decoder = AdjustedRandIndex(ignore_background=True, ignore_overlaps=False)
        val_pbar = tqdm(enumerate(val_dataloader), desc='Validation', mininterval=TQDM_MIN_INTERVAL)
        for i, (_, images, gt_masks, _) in val_pbar:
            images = images.to('cuda')
            gt_masks = gt_masks.to(images.device)
            # treat segmentations with class_id <= 0 as background
            gt_masks = torch.as_tensor(gt_masks > 0, dtype=gt_masks.dtype) * gt_masks
            k += images.size()[0]
            do_log_metrics = epoch % args.log_metrics_every_n_epochs == 0
            if i == 0 or do_log_metrics:
                loss, aux_output = akornsaur.step(images, do_predict_masks=True)
                vis_images = images[:args.visualize_n_images]
                vis_gt_masks = gt_masks[:args.visualize_n_images].to(torch.int64).squeeze(1)
                vis_gt_masks = to_one_hot(vis_gt_masks, num_classes=args.num_slots)
                slot_attention_masks = aux_output["slot_attention_masks_hard"][:args.visualize_n_images]
                decoder_masks = aux_output["decoder_masks_hard"][:args.visualize_n_images]
                vis_grid = grid_numpy(vis_images, vis_gt_masks, decoder_masks, slot_attention_masks)
            else:
                loss, aux_output = akornsaur.step(images, do_predict_masks=False)

            val_loss += loss.item()
            if do_log_metrics:
                gt_masks = to_one_hot(gt_masks.to(torch.int64).squeeze(1)).to(torch.bool)
                ari_slot_attention.update(gt_masks, aux_output["slot_attention_masks_hard"])
                ari_decoder.update(gt_masks, aux_output["decoder_masks_hard"])
            if i == val_n_batches - 1:
                record['val/loss'] = val_loss / k
                if do_log_metrics:
                    record['val/ari_slot_attention'] = ari_slot_attention.compute().item()
                    record['val/ari_decoder'] = ari_decoder.compute().item()
                val_pbar.set_postfix(record)

        val_pbar.close()
        record['val/visualization'] = wandb.Image(vis_grid)
        maybe_log_wandb(record, args.wandb_project, args.wandb_group, args.wandb_run_name)

        if epoch % args.save_every_n_epochs == 0:
            checkpoint = {'model': akornsaur.state_dict(), 'optimizer': optimizer.state_dict(),
                          'global_step': global_step, 'epoch': epoch,}
            checkpoint_folder = os.path.join(args.save_path, args.wandb_run_name)
            os.makedirs(checkpoint_folder, exist_ok=True)
            torch.save(checkpoint, os.path.join(checkpoint_folder, 'checkpoint.pt'))
