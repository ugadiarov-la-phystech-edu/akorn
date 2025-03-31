from collections import OrderedDict
from typing import Dict, Callable

import torch
from torchvision.utils import make_grid
from torch.nn import functional as F
import numpy as np

from source.models.objs.knet import AKOrN
from source.models.objs.vit import ViT
from source.utils import gen_saccade_imgs


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


def model_preds(model, org_images):
    activation = {}
    imsize_h, imsize_w =  org_images.shape[-2], org_images.shape[-1]

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    if isinstance(model, AKOrN):
        model.out[0].register_forward_hook(get_activation("z"))
    elif isinstance(model, ViT):
        model.out[0].register_forward_hook(get_activation("z"))

    else:
        raise Exception()

    model.eval()
    imgs = org_images.cuda()

    with torch.no_grad():
        if (
            isinstance(model, AKOrN)
            or isinstance(model, ViT)
        ):
            output, _xs = model(imgs, return_xs=True)
        else:
            output = model(imgs)
            _xs = None
    v = activation["z"]

    if isinstance(model, AKOrN) or isinstance(model, ViT):
        v = F.normalize(v, dim=1)
    #elif isinstance(model, ViTWrapper):
    #    v = F.normalize(v, dim=2)
    #    v = v.permute(0, 2, 1)[..., 1:]
    #    h, w = int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1]))  # estimated inpsize
    #    v = v.unflatten(-1, (h, w))
    remove_all_forward_hooks(model)
    return v


def clustering(x, h, w, method="spectral", n_clusters=3):
    from sklearn.cluster import KMeans

    if method == "agglomerative":
        import fastcluster
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage

        x = x.view(x.shape[0], -1).transpose(-2, -1).to("cpu").detach()
        Z = fastcluster.average(x)
        label = fcluster(Z, t=n_clusters, criterion="maxclust")
        return label.reshape(h, w)
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            x.view(x.shape[0], -1).transpose(-2, -1).to("cpu").detach()
        )
        label = kmeans.labels_
        return label.reshape(h, w)

    else:
        raise ValueError("Clustering method not found")


def to_one_hot(tsr, num_classes):
    tsr = tsr - tsr.min()
    return torch.nn.functional.one_hot(tsr, num_classes=num_classes).movedim(-1, 0)


def get_image(
    model,
    images,
    method="agglomerative",
    n_clusters=7,
    saccade_r=1,
    pca=False,
    pca_dim=128,
):
    preds = []
    N = images.shape[0]
    _imgs, _ = gen_saccade_imgs(images, model.psize, model.psize // saccade_r)
    outputs = []
    for img in _imgs:
        v = model_preds(model, img)
        outputs.append(v.detach().cpu())

    nh, nw = int(np.sqrt(len(_imgs))), int(np.sqrt(len(_imgs)))
    ho, wo = outputs[0].shape[-2], outputs[0].shape[-1]
    nimg = torch.zeros(N, outputs[0].shape[1], ho, nh, wo, nw)
    for h in range(nh):
        for w in range(nw):
            nimg[:, :, :, h, :, w] = outputs[h * (nh) + w]
    nimg = nimg.view(N, -1, ho * nh, wo * nw)

    from source.utils import apply_pca_torch

    with torch.no_grad():
        if pca:
            pcaimg_ = apply_pca_torch(nimg, n_components=pca_dim)
            x = pcaimg_
        else:
            x = nimg

    for idx in range(N):
        _x = x[idx]
        pred = clustering(_x, *_x.shape[1:], method, n_clusters)
        pred = torch.nn.Upsample(
            scale_factor=(images.shape[-2]/pred.shape[-2], images.shape[-1]/pred.shape[-1]),
            mode='nearest')(torch.Tensor(pred[None, None]).float())[0, 0]
        preds.append(pred)

    preds = torch.stack(preds, 0).long()

    s_img = images[0].unsqueeze(0).cpu()
    s_p = to_one_hot(preds[0], num_classes=n_clusters).unsqueeze(1)
    attn_p = s_img * s_p + (1 - s_p)

    return make_grid(torch.cat([s_img, attn_p]), nrow=attn_p.shape[0] + 1, pad_value=0.5).cpu()
