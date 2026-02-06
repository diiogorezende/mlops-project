import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
from torchvision import transforms

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./data",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False):
        super().__init__()
        self.save_hyperparameters() # salva args para log
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])

    def prepare_data(self) -> None:
        FashionMNIST(self.data_dir, train=True, download=True)
        FashionMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            mnist_full = FashionMNIST(self.data_dir, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(mnist_full, [55000, 5000])
        if stage == 'test' or stage is None:
            self.test_set = FashionMNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )