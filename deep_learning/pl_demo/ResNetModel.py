import torchvision.models as models
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torch.nn as nn
import torch


class ResNetModel(pl.LightningModule):
    def __init__(self, num_classes, finetune="none"):
        super().__init__()
        self.num_classes = num_classes
        self.finetune = finetune

        if self.finetune == "none":
            # init a resnet from scratch
            self.feature_extractor = models.resnet50(weights=None, num_classes=num_classes)
            self.classifier = nn.Identity()
        else:
            # init a pretrained resnet
            backbone = models.resnet50(weights="DEFAULT")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)

            # use the pretrained model to classify food101 (101 image classes)
            self.classifier = nn.Linear(num_filters, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        if self.finetune == "none":
            return self.feature_extractor(x)
        else:
            return self.classifier(self.feature_extractor(x).flatten(1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("val_accuracy", self.accuracy)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)

        self.log("test_accuracy", self.accuracy)
        self.log("test_loss", loss)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler,
    #         "monitor": "val_loss"
    #     }