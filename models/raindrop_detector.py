from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np

from torch.nn import functional as F
from backbones.ard_cnn import ARDCNN
from backbones.unet import UNet

class RaindropDetector(pl.LightningModule):
    def __init__(self, in_channels=3):
        super().__init__()
        self.save_hyperparameters()
        #self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()

        self.raindrop_detector = ARDCNN(in_channels)
        #self.raindrop_detector = UNet(out_classes=1)

    def forward(self, x):
        return self.raindrop_detector(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        prediction = self(image)

        #loss = self.dice(prediction, mask)
        loss = self.bce(prediction, mask.float())

        self.log("loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalars('loss', {'train': loss}, global_step=self.current_epoch) 

        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        prediction = self(image)
        #loss = self.dice(prediction, mask)
        loss = self.bce(prediction, mask.float())

        self.log("val_loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False)
        self.logger.experiment.add_scalars('loss', {'val': loss}, global_step=self.current_epoch) 

    def predict_step(self, batch, batch_idx):
        image = batch
        prediction = torch.where(F.sigmoid(self(image)) > 0.5, 1, 0)
        print(f"prediction: {prediction.shape}")

    def configure_optimizers(self):
        # define optimizer
        optimizer = torch.optim.Adam([ 
            dict(params=self.parameters(), lr=0.0008),
        ])

        # define learning rate scheduler (not used in this NB)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2, eta_min=5e-5,
        )
        #optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer], [lr_scheduler]