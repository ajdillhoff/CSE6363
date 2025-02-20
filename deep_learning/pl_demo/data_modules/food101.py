import torch
from torchvision import transforms
from torchvision.datasets import Food101
import lightning as L


class Food101DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str="path/to/dir",
                batch_size: int=128,
                num_workers: int=8,
                augment: bool=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.augment:
                train_transforms = transforms.Compose([
                    transforms.Resize(224), 
	                transforms.CenterCrop(224), 
	                transforms.RandomHorizontalFlip(),
	                transforms.RandomVerticalFlip(),
	                transforms.RandomRotation(45),
	                transforms.RandomAffine(45),
	                transforms.ColorJitter(),
	                transforms.ToTensor(),
	                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            else:
                train_transforms = transforms.Compose([
                    transforms.Resize(256), 
                    transforms.CenterCrop(224), 
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            test_transforms = transforms.Compose([
                transforms.Resize(256), 
	            transforms.CenterCrop(224), 
                transforms.ToTensor(),
	            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            train_dataset = Food101(self.data_dir, split="train", download=True, transform=train_transforms)

            # Use 5% of the training set for validation
            train_set_size = int(len(train_dataset) * 0.95)
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
	            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            self.test_dataset = Food101(self.data_dir, split="test", download=True, transform=test_transforms)

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