import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Accuracy

class ClassificationTask(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            optimizer: any, # recebe o otimizador configurado via Hydra
            loss_fn: nn.Module
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_factory = optimizer
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = self.optimizer_factory(params=self.model.parameters())
        return optimizer