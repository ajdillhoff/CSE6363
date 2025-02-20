import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
import lightning as L


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str="path/to/dir",
                batch_size: int=128,
                num_workers: int=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            train_transforms = transforms.Compose([
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            train_dataset = CIFAR10(self.data_dir, train=True, download=True, transform=train_transforms)

            # Use 10% of the training set for validation
            train_set_size = int(len(train_dataset) * 0.9)
            val_set_size = len(train_dataset) - train_set_size

            seed = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_set_size, val_set_size], generator=seed)
            val_dataset.dataset.transform = test_transforms

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            test_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])

            self.test_dataset = CIFAR10(self.data_dir, train=False, download=True, transform=test_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )