import argparse
import pytorch_lightning as pl
from models.raindrop_detector import RaindropDetector
from datamodules.RaindropDataModule import RaindropDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="Train or test")
parser.add_argument("--data_dir", type=str, default="datasets/EchoNet", help="Path ke datasets")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")
parser.add_argument("--num_workers", type=int, default=8, help="num_workers")
parser.add_argument("--accelerator", type=str, default='cpu', help="Accelerator")
parser.add_argument("--logs_dir", type=str, default='lightning_logs', help="Log dir")
parser.add_argument("--log", action='store_true', help="log")

params = parser.parse_args()

if __name__ == '__main__':
    mode = params.mode
    data_dir = params.data_dir
    batch_size = params.batch_size
    max_epochs = params.max_epochs
    num_workers = params.num_workers
    accelerator = params.accelerator
    logs_dir = params.logs_dir
    log = params.log

    logger = TensorBoardLogger(save_dir=logs_dir, name="raindrop_detector")

    data_module = RaindropDataModule(root=data_dir, 
                        batch_size=batch_size, 
                        num_workers=num_workers)

    raindrop_detector = RaindropDetector()

    trainer = pl.Trainer(accelerator=accelerator, 
                max_epochs=max_epochs, 
                num_sanity_val_steps=1, 
                #auto_scale_batch_size=True, 
                enable_model_summary=True,
                logger=logger,
                precision=16,
                #accumulate_grad_batches=2,
                callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)])

    if mode == 'train':
        trainer.fit(model=raindrop_detector, datamodule=data_module)

    if mode == 'validate':
        if not log:
            trainer.logger = False

        trainer.validate(model=raindrop_detector, datamodule=data_module)

    if mode == 'test':
        if not log:
            trainer.logger = False

        trainer.test(model=raindrop_detector, datamodule=data_module)

    if mode == 'predict':
        if not log:
            trainer.logger = False

        predicts = trainer.predict(model=raindrop_detector, datamodule=data_module)

        for predict in predicts:
            print(predict)
            print('\n')