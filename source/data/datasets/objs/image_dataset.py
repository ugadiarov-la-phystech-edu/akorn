import time

import PIL.Image
from torch.utils.data import Dataset
import glob
import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms

from source.data.augs import simclr_augmentation

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(self, root, mode, res=128, extension='png', return_tensor=True,):
        assert mode in ['train', 'val', 'valid', 'test']
        if mode in ('valid', 'test'):
            mode = 'val'

        self.root = root
        self.res = res
        self.mode = mode
        self.extension = extension
        self.return_tensor = return_tensor
        self.to_tensor = transforms.ToTensor()
        start = time.time()
        self.image_paths = [p for p in glob.iglob(os.path.join(self.root, self.mode, '**'), recursive=True) if
                            p.endswith(f'.{extension}')]
        print(f'Dataset contains {len(self.image_paths)} images. Indexing took {time.time() - start} seconds.')

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index])
        if img.size[0] * img.size[1] <= self.res * self.res:
            resample = PIL.Image.Resampling.NEAREST
        else:
            resample = PIL.Image.Resampling.BILINEAR

        img = img.resize((self.res, self.res), resample=resample)

        if self.return_tensor:
            return self.to_tensor(img)

        return img

    def __len__(self):
        return len(self.image_paths)


class AugmentedPairImageDataset(ImageDataset):
    def __init__(self, root, mode, res=128, extension='png', hflip=False):
        super().__init__(root, mode, res, extension, return_tensor=False)
        self.transform = simclr_augmentation(imsize=self.res, hflip=hflip)

    def __getitem__(self, index):
        pil_image = super().__getitem__(index)
        augmented_pair = [self.transform(pil_image), self.transform(pil_image)]
        return torch.stack(augmented_pair)
