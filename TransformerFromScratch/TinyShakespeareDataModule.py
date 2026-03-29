import lightning as L
import torch
import random
from torch.utils.data import Subset, Dataset,Subset, TensorDataset, DataLoader, RandomSampler, BatchSampler,SequentialSampler


class TinyShakespeareDataModule(L.LightningDataModule):
    def __init__(self, seq_length, batch_size = 32):
        super().__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size

        # read text
        with open("tiny_shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # token <-> index mapping
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)

    def setup(self, stage = None) -> None:
        total_len = len(self.data)
        train_end = int(total_len * 0.9)
        val_end = int(total_len * 0.95)

        self.trainset = TinyShakespeareDataset(self.data[:train_end], self.seq_length)
        self.valset = TinyShakespeareDataset(self.data[train_end:val_end], self.seq_length)
        self.testset = TinyShakespeareDataset(self.data[val_end:], self.seq_length)

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle = False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

class TinyShakespeareDataset(Dataset):
    def __init__(self, data, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.data = data
        self.size = (len(self.data) - 1) // self.seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = idx * self.seq_length
        x = self.data[start : start + self.seq_length]
        y = self.data[start + 1 : start + self.seq_length +1]
        return x, y

# log
# My architecture
# Dataset = load text + build vocab + create windows
# DataModule = split dataset indices
#
# In the 1st draft the tiny shakespeare text (file) is loaded within the class TinyShakespeareDataset.
# The dataset is built with an overlapped a moving window. Late on the dataset is split to train, eval and test
# dataset. A minor issue, deu to the overlapped window, the last chars at then end of train dataset are part in the eva
# dataset. It is not very clean building. but for the mount of samples. it is fine.
#
# GPT proposed
# DataModule = load text + build vocab + split raw data
# Dataset = only create sliding windows
# GPT think each class has its single responsibility. This is very important
#
# Last but not at least argument
# the vocab_size, char_ti_idx and idx_to char can be accessed better
