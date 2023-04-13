import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.tuner import Tuner

from data_modules.food101 import Food101DataModule
from ResNetModel import ResNetModel

def main(args):

    model = ResNetModel(101, 1e-3)

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
    food101 = Food101DataModule("~/Data/Food101/", batch_size=256, num_workers=8)
    # tuner = Tuner(trainer)
    # tuner.lr_find(model, food101)
    trainer.fit(model=model, datamodule=food101)
    trainer.test(model, datamodule=food101)

if __name__ == "__main__":
    main(sys.argv)