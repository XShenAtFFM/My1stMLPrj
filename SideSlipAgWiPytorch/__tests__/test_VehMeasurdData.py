import math
import unittest
import pandas as pd
from asammdf import MDF, Signal
import torch
import sys
sys.path.insert(0, '../../')
from pathlib import Path
from SideSlipAgWiPytorch.VehMeasurdData import VehMeasurdData
import numpy as np

# function to generate test mf4 file
def create_test_mf4_file(signal_list, is_corrsys_valid = True):
    """
    Generate a random panda dataframe, The columns names are related to signals which might be used for the ai model
    Store the dataframe in a .mf4 file
    """
    df = pd.DataFrame(np.random.rand(10, 13), columns = signal_list)

    # mean value of corrsys has to be higher than 3
    if 'Corrsys_VL' in signal_list and is_corrsys_valid:
        df['Corrsys_VL'] = df['Corrsys_VL'] + 5

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

    # Save the file
    mdf.save('testdata.mf4', overwrite=True)

class TestVehMeasurdData(unittest.TestCase):
    def setUp(self):
        print('setUp...')


    def test_data_loading(self):
        """
        Test if the same number of measurements are found
        Test if the features are picked and standardized as expected
        Test if the labels are picked and transferred correctly, when is_corrsys_preferred == False
        """
        print('test_data_loading...')
        # Generate test mf4 file
        signal_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'VEL_RR', 'YawRate', 'FSTANGLE', 'SsaRollRate', 'LACC', 'YR',
                         'RT_Msg604_VelLateral', 'RT_Msg604_VelForward', 'Corrsys_VL', 'Corrsys_VQ']
        create_test_mf4_file(signal_list)

        # list all mf4 files within current folder
        file_list = [p.as_posix() for p in Path('./').iterdir() if p.suffix == '.mf4']

        #Define the signal list, which shall be picked from the mf4 files
        feature_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'UsedYawRate','UsedLatA']

        # instantiate class VehMeasuredata
        vehmeasurddata = VehMeasurdData(file_list, feature_list, True, is_corrsys_preferred = False)

        # Check if the object has same number of mf4 as available
        self.assertEqual(vehmeasurddata._data_list, file_list,'all file found')

        # get the feature and label via from vehmeasurddata
        fd, lb = vehmeasurddata[0]

        # Load the mf4 directly and keep only columns same as the featurelist
        src = MDF(vehmeasurddata._data_list[0]).to_dataframe()
        adaptfeaturelist = ['VEL_FL', 'VEL_FR', 'VEL_RL','YR', 'LACC']
        src_feature = src.drop(columns=src.columns.difference(adaptfeaturelist))
        src_feature = src_feature.reindex(columns = adaptfeaturelist)
        src_feature =  src_feature.to_numpy()

        # inverse standardize scaler
        fd_orig = vehmeasurddata._standardizer.inverse_transform(fd.numpy())
        # Compare the data with original dataset
        self.assertEqual(np.allclose(fd_orig, src_feature, atol=1e-05, equal_nan=False), True, 'feature checked')

        src_label = src.drop(columns=src.columns.difference(['RT_Msg604_VelLateral','RT_Msg604_VelForward']))
        src_label = src_label.reindex(columns=['RT_Msg604_VelLateral','RT_Msg604_VelForward'])
        src_label = src_label.to_numpy()
        src_label[:, 0] *= -1
        # Compare the data with original dataset
        self.assertEqual(np.allclose(lb, src_label, atol=1e-05, equal_nan=False), True, 'feature checked')



    def test_label_processing_case1(self):
        """
        Test if the labels are picked and transferred correctly, when is_corrsys_preferred == True
        :return:
        """
        print('test_label_processing case1...')
        # Generate test mf4 file
        signal_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'VEL_RR', 'YawRate', 'FSTANGLE', 'SsaRollRate', 'LACC', 'YR',
                         'RT_Msg604_VelLateral', 'RT_Msg604_VelForward', 'Corrsys_VL', 'Corrsys_VQ']
        create_test_mf4_file(signal_list, True)

        # list all mf4 files within current folder
        file_list = [p.as_posix() for p in Path('./').iterdir() if p.suffix == '.mf4']

        #Define the signal list, which shall be picked from the mf4 files
        feature_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'UsedYawRate','UsedLatA']

        # instantiate class VehMeasuredata
        vehmeasurddata = VehMeasurdData(file_list, feature_list, True, is_corrsys_preferred=True)

        # get the feature and label via from vehmeasurddata
        fd, lb = vehmeasurddata[0]

        # Load the mf4 directly and keep only columns same as the featurelist
        src = MDF(vehmeasurddata._data_list[0]).to_dataframe()
        adaptfeaturelist = ['Corrsys_VL', 'Corrsys_VQ', 'YR']
        src_feature = src.drop(columns=src.columns.difference(adaptfeaturelist))
        src_feature = src_feature.reindex(columns=adaptfeaturelist)
        src_feature = src_feature.to_numpy()
        src_feature[:, 1] = src_feature[:, 1] / 3.6 + (1.5 + 0.74) * src_feature[:, 2] / 180 * math.pi
        src_feature[:, 0] = src_feature[:, 0] / 3.6 - (0.47) * src_feature[:, 2] / 180 * math.pi
        src_feature = np.delete(src_feature, 2, 1)
        # Compare the data with original dataset
        self.assertEqual(np.allclose(lb, src_feature, atol=1e-03, equal_nan=False), True, 'feature checked')

    def test_label_processing_case2(self):
        """
        Test if the labels are picked and transferred correctly, when Corrsys is invalid
        :return:
        """
        print('test_label_processing case2...')
        # Generate test mf4 file
        signal_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'VEL_RR', 'YawRate', 'FSTANGLE', 'SsaRollRate', 'LACC', 'YR',
                         'RT_Msg604_VelLateral', 'RT_Msg604_VelForward', 'Corrsys_VL', 'Corrsys_VQ_fake']
        create_test_mf4_file(signal_list, True)

        # list all mf4 files within current folder
        file_list = [p.as_posix() for p in Path('./').iterdir() if p.suffix == '.mf4']

        #Define the signal list, which shall be picked from the mf4 files
        feature_list = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'UsedYawRate','UsedLatA']

        # instantiate class VehMeasuredata
        vehmeasurddata = VehMeasurdData(file_list, feature_list, True, is_corrsys_preferred=True)

        # get the feature and label via from vehmeasurddata
        fd, lb = vehmeasurddata[0]

        # Load the mf4 directly and keep only columns same as the featurelist
        src = MDF(vehmeasurddata._data_list[0]).to_dataframe()
        adaptfeaturelist = ['RT_Msg604_VelLateral', 'RT_Msg604_VelForward']
        src_label = src.drop(columns=src.columns.difference(adaptfeaturelist))
        src_label = src_label.reindex(columns=adaptfeaturelist)
        src_label = src_label.to_numpy()
        src_label[:, 0] *= -1
        # Compare the data with original dataset
        self.assertEqual(np.allclose(lb, src_label, atol=1e-03, equal_nan=False), True, 'feature checked')

if __name__ == '__main__':
    unittest.main()