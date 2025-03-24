import torch
import os
from torch.optim.lr_scheduler import _LRScheduler

def add_gradient_histograms(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(name + "/grad", param.grad, epoch)


def save_model(model, epoch, checkpoint_dir, prefix="checkpoint"):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}_{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    print(f"Model saved: {checkpoint_path}")


def save_checkpoint(
    model, optimizer, epoch, loss, checkpoint_dir, max_checkpoints=None
):

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")

    manage_checkpoints(checkpoint_dir, max_checkpoints)


def manage_checkpoints(checkpoint_dir, max_checkpoints):
    if max_checkpoints is None:
        return
    else:
        checkpoints = [
            f
            for f in os.listdir(checkpoint_dir)
            if f.startswith("checkpoint_") and f.endswith(".pth")
        ]
        checkpoints.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))

        while len(checkpoints) > max_checkpoints:
            old_checkpoint = checkpoints.pop(0)
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print(f"Old checkpoint removed: {old_checkpoint}")


class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_iter=-1):
        self.warmup_iters = warmup_iters
        self.current_iter = 0 if last_iter == -1 else last_iter
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            # Linear warmup phase
            return [
                base_lr * (self.current_iter + 1) / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            # Maintain the base learning rate after warmup
            return [base_lr for base_lr in self.base_lrs]

    def step(self, it=None):
        if it is None:
            it = self.current_iter + 1
        self.current_iter = it
        super(LinearWarmupScheduler, self).step(it)


class ExpDecayWithLinearWarmupScheduler(LinearWarmupScheduler):
    def __init__(self, optimizer, warmup_iters, decay_steps, decay_rate, last_iter=-1):
        super().__init__(optimizer, warmup_iters, last_iter)
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

    def get_lr(self):
        lrs = super().get_lr()
        if self.current_iter > self.warmup_iters:
            it = self.current_iter - self.warmup_iters
            decay_steps = self.decay_steps - self.warmup_iters
            lrs = [lr * self.decay_rate ** (it / decay_steps) for lr in lrs]

        return lrs
