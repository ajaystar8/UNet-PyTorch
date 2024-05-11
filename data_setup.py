import os.path
from glob import glob
from typing import *
from PIL import Image

import torch
import torchvision.transforms.v2 as transforms

from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform=None) -> None:

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # load image
        img = Image.open(self.image_paths[idx]).convert("L")

        # load mask
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            return self.transform(img, mask)
        else:
            return img, mask

    def __len__(self) -> int:
        return len(self.image_paths)


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
):
    # Create train dataset
    train_data = SegmentationDataset(image_paths=glob(os.path.join(train_dir, "images", "*.png")),
                                     mask_paths=glob(os.path.join(train_dir, "masks", "*.png")),
                                     transform=transform)

    # Create test dataset
    test_data = SegmentationDataset(image_paths=glob(os.path.join(test_dir, "images", "*.png")),
                                    mask_paths=glob(os.path.join(test_dir, "masks", "*.png")),
                                    transform=transform)

    # train dataloader
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)

    # test dataloader
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader
