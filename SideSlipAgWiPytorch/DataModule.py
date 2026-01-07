import lightning as L
import torch
import random
from torch.utils.data import Dataset,Subset, TensorDataset, DataLoader, RandomSampler, BatchSampler,SequentialSampler
from SideSlipAgWiPytorch.VehMeasurdData import VehMeasurdData

def collate_fn_pad(batch):
    """
    This function is currently unused. Because the batch consists of only 1 sequence.

    Note: This function is be called after collate_fn is called. It takes the datasets of the batch created by the dataloader and
    pad the sequences accordingly in order that all sequences in the batch match the max length in that batch.
    Because DataLoader builds a batch as a single tensor. A tensor requires a rectangular shape.
    Pad would be sufficient for a batch of sequences. pack_padded_sequence is required by LSTM

    :param batch:
    :return: batched feature and label tensors
    """
    lengths = torch.tensor([t[0].shape[0] for t in batch])
    feature_batch = [t[0] for t in batch]
    label_batch = [t[1] for t in batch]
    feature_batch = torch.nn.utils.rnn.pad_sequence(feature_batch, batch_first=True)
    label_batch = torch.nn.utils.rnn.pad_sequence(label_batch, batch_first=True)
    feature_batch = torch.nn.utils.rnn.pack_padded_sequence(
        feature_batch, lengths, batch_first=True, enforce_sorted=False
    )
    label_batch = torch.nn.utils.rnn.pack_padded_sequence(
        label_batch, lengths, batch_first=True, enforce_sorted=False
    )
    return feature_batch, label_batch

class DataModule(L.LightningDataModule):
    """
    The DataModule is sub class of the lightning. LightningDataModule.
    It encapsulates training, validation, testing, and prediction dataloaders

    Note: Decision of the batch size for train_dataloader and validation_dataloader.
    I have a couple of vehicle measurements. In engineer world, these called time series data.
    In the sequence model world a time series data is a sequence.
    A batch consists of only 1 sequence. Because the sequences are independent of each other. If a batch has more than
    one sequence. It has to be connected together. In LSTM layer the first sequence “runs into” the 2nd sequence.
    There are some work around, but it requires a lot of effort or computing power.

    :attributes:
        _data_set: instance of VehMeasurdData, pointing to all found measurements

    """
    def __init__(self, filelist: list, is_standardized = False, is_corrsys_preferred = True, **kwargs):
        """
        Data module initialization function,
        :param filelist: filelist of the entire dataset
        :param is_standardized: bool, required for the init class VehMeasurdData
        :param is_corrsys_preferred: bool, required for the init class VehMeasurdData
        """
        super().__init__()
        self._dataset = VehMeasurdData(filelist, None, is_standardized, is_corrsys_preferred)

    def setup(self, stage: str = None):
        """
        override the parent hook, setup datasets for fit (train + validate), validate, test, or predict
        :param stage: unused
        """
        measurements = len(self._dataset)

        # Due to low amount ot data, no data splitting strategy is created.
        self._train_dataset = self._dataset
        # self._train_dataset.lengths = self._dataset.lengths

        self._val_dataset = self._dataset
        # self._val_dataset.lengths = self._train_dataset.lengths

        # It was considered in case of a lot of datasets, 75% data shall used for training
        # The rest for validation

        # if measurements >= 10:
        #     all_measurements = set(range(measurements))
        #     train_set= set(random.sample(list(all_measurements), int(measurements*0.75)))
        #     val_set = list(set(all_measurements - train_set))
        #     train_set = list(train_set)
        #
        #     self._train_dataset = Subset(self._dataset, train_set)
        #     self._val_dataset = Subset(self._dataset, val_set)
        #     #self._train_dataset.lengths = self._dataset.lengths[train_set]
        #     #self._val_dataset.lengths = self._dataset.lengths[val_set]
        # else:
        #     self._train_dataset = self._dataset
        #     #self._train_dataset.lengths = self._dataset.lengths
        #
        #     self._val_dataset = self._train_dataset
        #     #self._val_dataset.lengths = self._train_dataset.lengths

    def train_dataloader(self):
        sp = RandomSampler(self._train_dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1, drop_last = False)
        return DataLoader(
            self._train_dataset,
            num_workers=0,
            #collate_fn = collate_fn_pad,
            batch_sampler= bs,
            shuffle = False,
        )

    def val_dataloader(self):
        sp = SequentialSampler(self._val_dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1, drop_last = False)
        return DataLoader(
            self._val_dataset,
            num_workers = 0,
            #collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

    def test_dataloader(self):
        sp = SequentialSampler(self._dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1, drop_last = False)
        return DataLoader(
            self._dataset,
            num_workers=1,
            #collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

    def predict_dataloader(self):
        sp = SequentialSampler(self._dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1, drop_last = False)
        return DataLoader(
            self._dataset,
            num_workers=1,
            #collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

