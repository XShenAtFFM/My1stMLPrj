import copy
import random

import lightning as L
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset


L.seed_everything(42)

class CifarDataModule(L.LightningDataModule):
    def __init__(self, root_path = './'):
        super().__init__()
        self.root_path = root_path
        self._test_dataset = None
        self._train_dataset = None
        self._val_dataset = None
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])  # manual or calculated
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])

    def setup(self, stage: str = None):
        full_train_dataset  = CIFAR10(root = self.root_path, train = True, download = False, transform = None )
        # try to split the train dataset to train and val dataset
        n = len(full_train_dataset)
        picked = random.sample(range(n), int(n * 0.95))
        picked_complement = [i for i in range(n) if i not in set(picked)]

        # split both sources so transforms are correct
        self._train_dataset = Subset(full_train_dataset, picked)
        val_dataset = copy.copy(full_train_dataset)
        self._val_dataset =  Subset(val_dataset, picked_complement)

        self._train_dataset.dataset.transform = self.train_transform
        self._val_dataset.dataset.transform = self.test_transform

        self._test_dataset = CIFAR10(root = self.root_path, train = False, download= False, transform= self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self._train_dataset,
            batch_size = 128,
            num_workers = 0,
            shuffle = True,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_dataset,
            num_workers = 0,
            batch_size = 128,
            shuffle = False,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_dataset,
            num_workers = 0,
            batch_size = 128,
            shuffle = False,
        )




