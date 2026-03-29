import lightning as L
from torch.utils.data import Dataset
import torch
import random
from torch.utils.data import Subset, Dataset,Subset, TensorDataset, DataLoader, RandomSampler, BatchSampler,SequentialSampler

class MyTransformerDataModule(L.LightningDataModule):
    def __init__(self, seq_length):
        super().__init__()
        self.dataset = TinyShakespeareDataset(seq_length)
        self.batch_size = 32

    def setup(self, stage: str) -> None:
        train_size = int(self.dataset.size * 0.9)
        eval_size = int(self.dataset.size * 0.05)
        self._trainset = Subset(self.dataset, range(0, train_size))
        self._evalset = Subset(self.dataset, range(train_size, train_size + eval_size))
        self._testset = Subset(self.dataset, range(train_size + eval_size, self.dataset.size))

    def train_dataloader(self):
        return DataLoader(
            self._trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self._evalset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle = False,
        )

    def test_dataloader(self):
        return DataLoader(
            self._testset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length

        # read text
        with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # token <-> index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        # convert text to token

        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        self.data_length = len(self.data)
        self.size = self.data_length - self.seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1:  idx + self.seq_length +1]
        return x, y

