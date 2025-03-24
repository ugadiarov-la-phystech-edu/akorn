import torch
import os
import numpy as np
import torch.nn.functional as F
from torchvision.utils import make_grid

PLOTCOLORS = {
    "blue": "#377eb8",
    "orange": "#ff7f00",
    "green": "#4daf4a",
    "pink": "#f781bf",
    "brown": "#a65628",
    "purple": "#984ea3",
    "gray": "#999999",
    "red": "#e41a1c",
    "lightgray": "#d3d3d3",
    "lightgreen": "#90ee90",
    "yellow": "#dede00",
}


def ConvSingularValues(kernel, input_shape):
    transforms = torch.fft.fft2(kernel.permute(2, 3, 0, 1), input_shape, dim=[0, 1])
    print(transforms.shape)
    return torch.linalg.svd(transforms)


def ConvSingularValuesNumpy(kernel, input_shape):
    kernel = kernel.detach().cpu().numpy()
    transforms = np.fft.fft2(kernel.transpose(2, 3, 0, 1), input_shape, axes=[0, 1])
    # transforms = np.fft.fft2(kernel.permute(2,3,0,1), input_shape, dim=[0,1])
    print(transforms.shape)
    return np.linalg.svd(transforms)


def EigenValues(kernel, input_shape):
    # transforms = np.fft.fft2(kernel, input_shape, axes=[0, 1])
    transforms = torch.fft.fft2(kernel.permute(2, 3, 0, 1), input_shape, dim=[0, 1])
    print(transforms.shape)
    return torch.linalg.eig(transforms)


def load_state_dict_ignore_size_mismatch(model, state_dict):
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    for key, param in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == param.shape:
                matched_state_dict[key] = param
            else:
                print(
                    f"Size mismatch for key '{key}': model {model_state_dict[key].shape}, checkpoint {param.shape}"
                )
        else:
            print(f"Key '{key}' not found in model state dict.")

    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)


def compare_optimizer_state_dicts(original, modified):
    diff = {}
    for key in original.keys():
        if key not in modified:
            diff[key] = "Removed"
        elif original[key] != modified[key]:
            diff[key] = {"Original": original[key], "Modified": modified[key]}
    for key in modified.keys():
        if key not in original:
            diff[key] = "Added"
    return diff


def get_worker_init_fn(start, end):
    return lambda worker_id: os.sched_setaffinity(0, range(start, end))


def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ["0", "n", "f"]:
        return False
    elif x[0] in ["1", "y", "t"]:
        return True
    raise ValueError("Invalid value: {}".format(x))


def apply_pca(x, n_components=3):
    # x.shape = [B, C, H, W]
    from sklearn.decomposition import PCA

    pca = PCA(n_components)
    nx = []
    d = x.shape[1]
    for _x in x:
        _x = _x.permute(1, 2, 0).reshape(-1, d)
        _x = pca.fit_transform(_x)
        _x = _x.transpose(1, 0).reshape(n_components, x.shape[2], x.shape[3])
        nx.append(torch.tensor(_x))
    nx = torch.stack(nx, 0)
    # normalize to [0, 1]
    nx = (nx - nx.min()) / (nx.max() - nx.min())
    return nx


def apply_pca_torch(x, n_components=3):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    N = H * W

    if n_components >= C:
        return x

    # Reshape to [B, N, C]
    x = x.permute(0, 2, 3, 1).reshape(B, N, C)

    # Center the data per sample
    x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, C]
    x_centered = x - x_mean  # [B, N, C]

    # Compute covariance matrix per sample: [B, C, C]
    cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (N - 1)

    # Compute eigenvalues and eigenvectors per sample
    eigenvalues, eigenvectors = torch.linalg.eigh(
        cov
    )  # eigenvalues: [B, C], eigenvectors: [B, C, C]

    # Reverse the order of eigenvalues and eigenvectors to get descending order
    eigenvalues = eigenvalues.flip(dims=[1])
    eigenvectors = eigenvectors.flip(dims=[2])

    # Select the top 'dim' eigenvectors
    top_eigenvectors = eigenvectors[:, :, :n_components]  # [B, C, dim]

    # Project the centered data onto the top eigenvectors
    x_pca = torch.bmm(x_centered, top_eigenvectors)  # [B, N, dim]

    # Reshape back to [B, dim, H, W]
    x_pca = x_pca.transpose(1, 2).reshape(B, n_components, H, W)

    return x_pca


def gen_saccade_imgs(img, psize, r):
    H, W = img.shape[-2:]
    img = F.interpolate(img, (H + psize - r, W + psize - r), mode="bicubic")
    imgs = []
    for h in range(0, psize, r):
        for w in range(0, psize, r):
            imgs.append(img[:, :, h : h + H, w : w + W])
    return imgs, img[:, :, psize // 2 : H + psize // 2, psize // 2 : W + psize // 2]


def to_one_hot(tsr, num_classes=-1):
    assert len(tsr.size()) == 3
    assert num_classes == -1 or num_classes > tsr.max()
    return torch.nn.functional.one_hot(tsr, num_classes=num_classes).movedim(-1, 1)


def vis_mask(images, masks):
    images = images.unsqueeze(1)
    masks = masks.unsqueeze(2)
    return images * masks + (1 - masks)


def grid_numpy(images, gt_masks, decoder_masks, slot_attention_masks):
    attn_gt = vis_mask(images, gt_masks)
    attn_decoder = vis_mask(images, decoder_masks)
    attn_sa = vis_mask(images, slot_attention_masks)
    log_image = torch.stack([attn_gt, attn_sa, attn_decoder], dim=1)
    log_image = log_image.flatten(end_dim=2)
    log_image = make_grid(log_image, nrow=gt_masks.size()[1], pad_value=0.5).movedim(0, -1).cpu().numpy()

    return log_image


def adjusted_rand_index(
    true_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Adapted to Pytorch from SAVi Jax implementation:
    https://github.com/google-research/slot-attention-video/blob/main/savi/lib/metrics.py

    Args:
        true_mask: A binary tensor of shape (batch_size, n_points, n_true_clusters). The true cluster
            assignment encoded as one-hot with missing values allowed.
        pred_mask: A binary tensor of shape (batch_size, n_points, n_pred_clusters). The predicted
            cluster assignment encoded as one-hot.

    Returns:
        ARI scores as a tensor of shape (batch_size,).
    """
    N = torch.einsum("bpc, bpk -> bck", true_mask.to(torch.float64), pred_mask.to(torch.float64))
    A = torch.sum(N, axis=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, axis=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, axis=1)

    rindex = torch.sum(N * (N - 1), axis=[1, 2])
    aindex = torch.sum(A * (A - 1), axis=1)
    bindex = torch.sum(B * (B - 1), axis=1)
    expected_rindex = aindex * bindex / torch.clip(num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0.0, ari, 1.0)


class AdjustedRandIndex:
    """Abstract ARI metric."""

    def __init__(
        self,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
    ):
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.total_ari = 0
        self.n_images = 0

    def update(self, true_mask: torch.Tensor, pred_mask: torch.Tensor):
        """Update metric.

        Args:
            true_mask: Binary true masks of shape (batch, n_true_classes, height, width)
            pred_mask: One-hot predicted masks of shape (batch, n_pred_classes, height, width)
        """
        true_mask = true_mask.flatten(start_dim=2).movedim(1, 2)
        pred_mask = pred_mask.flatten(start_dim=2).movedim(1, 2)
        assert true_mask.ndim == 3
        assert pred_mask.ndim == 3
        if torch.any((true_mask != 0.0) & (true_mask != 1.0)):
            raise ValueError("`true_mask` is not binary")
        if torch.any((pred_mask != 0.0) & (pred_mask != 1.0)):
            raise ValueError("`pred_mask` is not binary")
        if torch.any(pred_mask.sum(dim=-1) != 1.0):
            raise ValueError("`pred_mask` is not one-hot")

        n_true_classes_per_point = true_mask.sum(dim=-1)
        if not self.ignore_overlaps and torch.any(n_true_classes_per_point > 1.0):
            raise ValueError("There are overlaps in `true_mask`.")
        if self.ignore_background and torch.any(n_true_classes_per_point != 1.0):
            raise ValueError("`true_mask` is not one-hot")

        if self.ignore_overlaps:
            overlaps = n_true_classes_per_point > 1.0
            true_mask = true_mask.clone()
            true_mask[overlaps] = 0.0  # ARI ignores pixels where all ground truth clusters are zero

        if self.ignore_background:
            true_mask = true_mask[..., 1:]  # Remove the background mask

        values = adjusted_rand_index(true_mask, pred_mask)

        # Special case: skip samples without any ground truth mask
        non_empty = n_true_classes_per_point.sum(dim=-1) > 0
        values = values[non_empty]

        self.total_ari += values.sum()
        self.n_images += len(values)

    def compute(self):
        return self.total_ari / self.n_images
