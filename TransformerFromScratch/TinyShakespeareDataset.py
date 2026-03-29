import torch
from plotly.validators.streamtube import starts
from tensorflow.python.debug.lib.dumping_callback import disable_dump_debug_info
from torch.utils.data import Dataset

class TinyShakespeareData(Dataset):
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
        self.data_length = len(data)
        self.size = self.data_length - self.seq_length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1:  idx + self.seq_length +1]
        return x, y

