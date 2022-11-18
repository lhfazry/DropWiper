import os
import pathlib
import collections
import numpy as np
import torch
import torch.utils.data
import cv2  # pytype: disable=attribute-error
import random
import albumentations as album
from glob import glob

class RaindropDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, masks_dir, augmented=False):
        self.augmented = augmented

        if not os.path.exists(data_dir) or not os.path.exists(masks_dir):
            raise ValueError(f"Path does not exist: {data_dir} or {masks_dir}")

        self.data = glob(os.path.join(data_dir, "*.png"))
        self.masks = glob(os.path.join(masks_dir, "*.png"))

        train_transform = [    
            album.RandomCrop(height=64, width=64, always_apply=True),
            album.OneOf(
                [
                    album.HorizontalFlip(p=1),
                    album.VerticalFlip(p=1),
                    album.RandomRotate90(p=1),
                ],
                p=0.75,
            ),
        ]

        self.augmentation = album.Compose(train_transform)
            
    def __getitem__(self, index):
        img_path = self.data[index]
        mask_path = self.masks[index]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

        if self.augmented:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = np.where(mask > 0, 1, 0)
        mask = np.expand_dims(mask, axis=0)

        image = image.transpose(2, 0, 1).astype('float32')
        mask = mask.transpose(2, 0, 1)#.astype('float32')
        image /= 255.

        return image, mask
            
    def __len__(self):
        return len(self.data)
