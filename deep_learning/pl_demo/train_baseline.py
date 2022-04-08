import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from BaselineModel import BaselineModel


def main():
    # Prepare the dataset
    train_transforms = transforms.Compose([
        transforms.RandAugment(),
        transforms.ToTensor(),
        cifar10_normalization()
    ])

    dm = CIFAR10DataModule("/home/alex/Data/CIFAR10/",
                           val_split=0.1,
                           num_workers=8,
                           normalize=True,
                           batch_size=256,
                           train_transforms=train_transforms)

    model = BaselineModel()

    # Add EarlyStopping
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=5)

    # Configure Checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min"
    )

    trainer = pl.Trainer(callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
