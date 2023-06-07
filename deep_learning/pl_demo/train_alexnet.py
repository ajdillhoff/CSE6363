import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from data_modules.cifar10 import CIFAR10DataModule
from AlexNetModel import AlexNetModel

def main(args):
    # Add EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5)

    # Configure Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback], max_epochs=-1)
    cifar10 = CIFAR10DataModule("~/Data/CIFAR10/", batch_size=512, num_workers=8)

    model = AlexNetModel(10)
    trainer.fit(model=model, datamodule=cifar10)
    trainer.test(model, datamodule=cifar10)

if __name__ == "__main__":
    main(sys.argv)