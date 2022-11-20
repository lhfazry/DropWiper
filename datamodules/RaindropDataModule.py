import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader
from datasets.RaindropDataset import RaindropDataset
from glob import glob

class RaindropDataModule(pl.LightningDataModule):
    def __init__(self, root: str = "datasets/RainDrop", 
            batch_size: int = 32, 
            num_workers: int = 8):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage = None):
        print(f'setup: {self.root}, stage: {stage}')

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = RaindropDataset(
                                data_dir=os.path.join(self.root, 'train', 'data'),
                                masks_dir=os.path.join(self.root, 'train', 'mask'),
                                augmented=True)
            
            self.val_set   = RaindropDataset(
                                data_dir=os.path.join(self.root, 'val', 'data'),
                                masks_dir=os.path.join(self.root, 'val', 'mask'))

        if stage == "validate" or stage is None:
            self.val_set   = RaindropDataset(
                                data_dir=os.path.join(self.root, 'val', 'data'),
                                masks_dir=os.path.join(self.root, 'val', 'mask'))

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set   = RaindropDataset(
                                data_dir=os.path.join(self.root, 'val', 'data'),
                                masks_dir=os.path.join(self.root, 'val', 'mask'))

        if stage == "predict" or stage is None:
            self.predict_set   = RaindropDataset(
                                data_dir=os.path.join(self.root, 'test', 'data'))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=self.batch_size, num_workers=self.num_workers)