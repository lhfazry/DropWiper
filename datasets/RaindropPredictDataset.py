import os
import pathlib
import collections
import numpy as np
import torch
import torch.utils.data
import cv2  # pytype: disable=attribute-error
import random
from glob import glob

class RaindropPredictDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):

        if not os.path.exists(data_dir):
            raise ValueError(f"Path does not exist: {data_dir}")

        self.data = glob(os.path.join(data_dir, "*.png"))
            
    def __getitem__(self, index):
        img_path = self.data[index]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        if self.augmented:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        mask = np.where(mask > 0, 1, 0)
        mask = np.expand_dims(mask, axis=0)

        image = image.transpose(2, 0, 1).astype('float32')
        #mask = mask.transpose(2, 0, 1)#.astype('float32')
        image /= 255.

        return image, mask
            
    def __len__(self):
        return len(self.data)
