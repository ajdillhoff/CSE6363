import sys
import torch
from torch import nn
from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl

from LeNetModel import LeNetModel


def main(args):
    # Prepare the dataset
    dm = CIFAR10DataModule("/home/alex/Data/CIFAR10/",
                           val_split=0.1,
                           num_workers=8,
                           normalize=True,
                           batch_size=256)

    model = LeNetModel.load_from_checkpoint(
        checkpoint_path=args[1]
    )

    trainer = pl.Trainer()
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main(sys.argv)
