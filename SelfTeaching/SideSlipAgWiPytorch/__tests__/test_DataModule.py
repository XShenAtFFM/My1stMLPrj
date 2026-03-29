import unittest
import pandas as pd
import numpy as np
import torch
from asammdf import MDF, Signal
import sys

from torch.nn.utils.rnn import pad_packed_sequence

sys.path.insert(0, '../../')
from pathlib import Path
from SideSlipAgWiPytorch.DataModule import DataModule
from torch.utils.data import DataLoader

# function to generate test mf4 file
def create_test_mf4_file():
    """
    Generate a random panda dataframe, The columns names are related to signals which might be used for the ai model
    Store the dataframe in a .mf4 file
    """

    # Generate test mf4 file
    signal_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'VEL_RR',
                              'FSTANGLE', 'VseFx_FL', 'VseFx_FR', 'VseFx_RL',
                              'VseFx_RR', 'SSAInLongAcc', 'YR', 'SSAInLatAcc',
                              'SsaRollRate', 'RT_Msg604_VelForward', 'RT_Msg604_VelLateral',]
    num_mf4_file = 12
    for idx in range(num_mf4_file):
        df = pd.DataFrame(np.random.rand(np.random.randint(5,16), len(signal_list)), columns = signal_list)
        df.iloc[0,] = idx
        # Create time stamp column, this is required for the mf4 file
        df['time'] = np.arange(df.shape[0])*0.01
        # Create an empty MDF file
        mdf = MDF()
        # Add each signal from the DataFrame
        for column in df.columns:
            if column != 'time':
                signal = Signal(
                    samples=df[column].values,
                    timestamps=df['time'].values,
                    name=column,
                    unit="",
                )
                mdf.append(signal)
        file_name = './testdata/testdata_' + str(idx) + '.mf4'
        # Save the file
        mdf.save(file_name, overwrite=True)

class TestDataModule(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("setUp, generte random datasets, initialize the data_module...")
        create_test_mf4_file()
        file_list = [p.as_posix() for p in Path('./testdata/').iterdir() if p.suffix == '.mf4']
        self.data_module = DataModule(file_list,False,True)

    def test_setup(self):
        print("Test that setup() correctly initializes datasets...")
        # the setup method need be called only once. This test case must be run as 1st
        self.data_module.setup()
        self.assertIsNotNone(self.data_module._train_dataset, "Train dataset not initialized")
        self.assertIsNotNone(self.data_module._val_dataset, "Validation dataset not initialized")

    def test_train_dataloader(self):
        print("Check that train_dataloader returns a DataLoader with correct batch size")
        train_loader = self.data_module.train_dataloader()
        # Check return of the train_dataloader
        self.assertIsInstance(train_loader, DataLoader)
        batch = next(iter(train_loader))
        self.assertIsInstance(batch, (list, tuple))
        x, y = batch
        x, lengths = pad_packed_sequence(x, batch_first=True)
        self.assertEqual(x.shape, torch.Size([2, max(lengths), 13]))

    def test_val_dataloader(self):
        print("Check validation dataloader consistency.")
        val_loader = self.data_module.val_dataloader()
        self.assertIsInstance(val_loader, DataLoader)

    #def test_test_dataloader(self):
        #"""Check test dataloader consistency."""
        # self.data_module.setup()
        # test_loader = self.data_module.test_dataloader()
        # self.assertIsInstance(test_loader, DataLoader)

    @classmethod
    def tearDownClass(cls):
        print("Clean up once")
        # delete the generated test data files
        for p in Path("./testdata/").glob("*.mf4"):
            p.unlink()


if __name__ == '__main__':
    unittest.main()







