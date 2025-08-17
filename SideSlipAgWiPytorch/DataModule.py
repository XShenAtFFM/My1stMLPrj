import lightning as L
import torch
import random
from torch.utils.data import Dataset,Subset, TensorDataset, DataLoader, RandomSampler, BatchSampler,SequentialSampler
from SideSlipAgWiPytorch.VehMeasurdData import VehMeasurdData

def collate_fn_pad(batch):
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
        bs = BatchSampler(sampler = sp, batch_size = 2, drop_last = False)
        return DataLoader(
            self._train_dataset,
            collate_fn = collate_fn_pad,
            batch_sampler= bs,
            shuffle = False,
        )

    def val_dataloader(self):
        sp = SequentialSampler(self._val_dataset)
        bs = BatchSampler(sampler = sp, batch_size = 10000, drop_last = False)
        return DataLoader(
            self._val_dataset,
            collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

    def test_dataloader(self):
        sp = SequentialSampler(self._dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1000, drop_last = False)
        return DataLoader(
            self._dataset,
            collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

    def predict_dataloader(self):
        sp = SequentialSampler(self._dataset)
        bs = BatchSampler(sampler = sp, batch_size = 1000, drop_last = False)
        return DataLoader(
            self._dataset,
            collate_fn = collate_fn_pad,
            batch_sampler = bs,
            shuffle = False,
        )

