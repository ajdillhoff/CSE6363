from lightning.pytorch.cli import LightningCLI

from data_modules.food101 import Food101DataModule
from ResNetModel import ResNetModel


def cli_main():

    cli = LightningCLI(ResNetModel, Food101DataModule)

if __name__ == "__main__":
    cli_main()