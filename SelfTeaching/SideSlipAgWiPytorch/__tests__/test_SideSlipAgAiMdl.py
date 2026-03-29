import unittest
import pandas as pd
import sys
sys.path.insert(0, '../../')
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from SideSlipAgWiPytorch.SideSlipAgAiMdl import SideSlipAgAiMdl
from SideSlipAgWiPytorch.DataModule import DataModule
import lightning as L
import numpy as np
from asammdf import MDF, Signal

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
    num_mf4_file = 1
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

class TestSideSlipAgAiMdl(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print("SetUpClass, create test data, define data module, ")
        create_test_mf4_file()
        file_list = [p.as_posix() for p in Path('./testdata/').iterdir() if p.suffix == '.mf4']
        self.data_module = DataModule(file_list, False, True)

    def test_forward_pass(self):
        print("test_forward_pass, check the forward pass...")
        in_size = np.random.randint(3, high = None)
        length = np.random.randint(510,1000)
        ai_model = SideSlipAgAiMdl(in_size)
        x = torch.randn(1, length, in_size)
        out = ai_model(x)
        self.assertEqual(out.shape, torch.Size([length, 4]),"The output shape is different than expected")


    def test_trainer(self):
        print("test_trainer")
        ai_model = SideSlipAgAiMdl(13)
        trainer = L.Trainer(overfit_batches = True, max_epochs = 5)
        trainer.fit(ai_model, self.data_module)
        final_loss = trainer.callback_metrics.get("train_loss")
        self.assertLess(final_loss,0.5,"Loss doesn't become smaller than 0.5")


    @classmethod
    def tearDownClass(cls):
        print("Clean up once")
        # delete the generated test data files
        for p in Path("./testdata/").glob("*.mf4"):
            p.unlink()

if __name__ == '__main__':
    unittest.main()



