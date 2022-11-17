from tabnanny import verbose
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F

from torch.nn import functional as F
from backbones.ard_cnn import ARDCNN
from losses.rmse import RMSE


class RaindropDetector(pl.LightningModule):
    def __init__(self, in_channels=3):
        super().__init__()
        self.save_hyperparameters()

        self.raindrop_detector = ARDCNN(in_channels)

    def forward(self, x):
        return self.raindrop_detector(x)

    def training_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch

        #print(f'ejection before: {ejection}')
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'ejection after: {ejection}')
        ef_label = ejection.type(torch.float32) / 100.

        #print(f'nlabel: {nlabel.shape}')
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')

        #classes_vec, ef_pred = self(nvideo)
        #print(f'classes_vec: {classes_vec.shape}')
        ef_pred = self(nvideo)
        #print(f'ef_pred size: {ef_pred.shape}')

        #print(f'ejection: {ejection.data}')
        #print(f'y_hat: {y_hat.data}')
        #loss1 = F.cross_entropy(classes_vec.view(-1, 3), nlabel.view(-1))
        #loss2 = F.mse_loss(ef_pred, ef_label)

        #loss = loss1 + loss2
        #loss = F.huber_loss(y_hat, ejection)
        loss = F.mse_loss(ef_pred, ef_label)
        
        #self.train_mse(y_hat, ejection)
        #self.train_mae(y_hat, ejection)
        #r2loss = r2_loss(y_hat, ejection)

        self.log('loss', loss, on_epoch=True, batch_size=self.batch_size)
        #self.log('train_mse', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        #self.log('train_mae', self.train_mse, on_step=True, on_epoch=False, batch_size=self.batch_size)
        #self.log('train_r2', r2loss, on_step=True, on_epoch=False, batch_size=self.batch_size)
        
        #tensorboard_logs = {'loss':{'train': loss.detach() } }
        #return {"loss": loss, 'log': tensorboard_logs }
        return loss

    def validation_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo)
        loss = F.mse_loss(ef_pred, ef_label)

        self.val_rmse(ef_pred, ef_label)
        self.val_mae(ef_pred, ef_label)
        self.val_r2(ef_pred, ef_label)
        #r2loss = r2_score(y_hat, ejection)

        self.log('val_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_rmse', self.val_rmse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_mae', self.val_mae, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('val_r2', self.val_r2, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        #tensorboard_logs = {'loss':{'val': loss.detach() } }
        #return {"val_loss": loss, 'log': tensorboard_logs }
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        #ejection = (2 * ejection.type(torch.float32) / 100.) - 1
        #print(f'nvideo.shape: {nvideo.shape}')
        #print(f'ejection: {ejection}')
        #print(f'nvideo.shape: f{nvideo.shape}')
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo) 
        loss = F.mse_loss(ef_pred, ef_label)
        
        self.test_rmse(ef_pred, ef_label)
        self.test_mse(ef_pred, ef_label)
        self.test_mae(ef_pred, ef_label)
        self.test_r2(ef_pred, ef_label)
        #r2loss = r2_score(y_hat, ejection)

        self.log('test_loss', loss, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_rmse', self.test_rmse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_mse', self.test_mse, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_mae', self.test_mae, on_epoch=True, batch_size=self.batch_size, prog_bar=True)
        self.log('test_r2', self.test_r2, on_epoch=True, batch_size=self.batch_size, prog_bar=True)

        return {"test_loss": loss}

    def predict_step(self, batch, batch_idx):
        filename, nvideo, nlabel, ejection, repeat, fps = batch
        #ejection = ejection.type(torch.float32) / 100.
        ef_label = ejection.type(torch.float32) / 100.

        ef_pred = self(nvideo) 

        loss = F.mse_loss(ef_pred, ejection)
        return {'filename': filename, 'EF': ef_label * 100., 'Pred': ef_pred * 100., 'loss': loss * 100.}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.85, verbose=True)

        return [optimizer], [lr_scheduler]