"""
This file defines a class VehMeassurdData, it is a subcloass of the torch.utils.dataset.Dataset class.
The goal of this class is to prepare the dataset for lightning datamodule
"""
import torch
from torch.utils.data import Dataset
from asammdf import MDF
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import math


class VehMeasurdData(Dataset):
    """
    The VehMeasurdData is sub class of pytorch dataset
    :attributes:
        _data_list: list, measurement mf4 file list
        _is_label_available: bool, shall give if the label is available in the measurement, not used now
        _lengths: length of found measurements
        _features: list, features of the data which also shall be picked from the measurement
        _standardizer: sklearn.preprocessing StandardScaler object, to standardize the data
        _signal_dictionary: dictionary, shall list all possible name of certain signal in the measurement,
                        the signal is named differently in the measurements. it shall support the signal pick from measurement
        _label_dictionary:  dictionary, the measurements can have two different label signals, corrsys or rt_ sometime both are
                available, the key, CorrsysPreferred indicate if corrsys shall be preferred or not
    """
    def __init__(self, file_list: list, feature_list : list[any] | None = None, is_standardized = False, is_corrsys_preferred = True):
        """
        :param file_list: mf4 file list
        :param feature_list: signal list which shall be picked from the measurement data(mf4 file)
        :param is_standardized: True, load standard scaler parameter from json file otherwise run a new standardscaler
        :param is_corrsys_preferred: True, Corrsys signals shall be preferred than RT signals
        """
        self._data_list = file_list
        self._is_label_available = [False] * len(file_list)
        self._lengths = list(range(len(self)))

        # Specify the feature list, if the feature_list isn't available as argument, default list is used
        if feature_list is not None:
            self._features = feature_list
        else:
            self._features = ['VEL_FL', 'VEL_FR', 'VEL_RL', 'VEL_RR',
                              'FSTANGLE', 'VseFx_FL', 'VseFx_FR', 'VseFx_RL',
                              'VseFx_RR', 'UsedLgtA', 'UsedLatA', 'UsedYawRate',
                              'UsedRollRate']

        self._signal_dictionary = {'UsedLgtA': ['SSAInLongAcc', 'LGASHRES'],
                                  'UsedLatA': ['SSAInLatAcc', 'NPL', 'LACC'],
                                  'UsedRollRate': ['SSAInRollRate', 'SsaRollRate', 'SSI_RollRate'],
                                  'UsedYawRate': ['SSAInYawRate', 'YR', 'YawRate']}

        self._label_dictionary = {'Corrsys': ['Corrsys_VL', 'Corrsys_VQ'],
                       'RtBox':['RT_Msg604_VelForward', 'RT_Msg604_VelLateral'],
                       'CorrsysPreferred': is_corrsys_preferred }

        # Initial or new generation a StandardScaler
        self._standardizer = None
        if is_standardized:
            self._standardizer = self._configStandardizer()
        else:
            self._standardizer = self._standardizing()


    def _configStandardizer(self):
        """
        The method create a StandardScaler and configure it with parameters from _standardizParam.json
        :Return:
            _standard_scaler: StandardScaler object
        """
        # Load parameter values from jons file
        with open("_standardizParam.json", "r") as f:
            params = json.load(f)

        # Convert load data to numpy array
        mean = np.array(params["mean"])
        scale = np.array(params["scal"])
        var = np.array(params["var"])

        # Define a new StandardScaler object, configure it with load data
        standard_scaler = StandardScaler()
        standard_scaler.mean_ = mean
        standard_scaler.scale_ = scale
        standard_scaler.var_ = var
        standard_scaler.n_features_in_ = len(mean)
        standard_scaler.n_samples_seen_ = 1  # dummy value (not important unless partial_fit is used)

        return standard_scaler


    def _standardizing(self):
        """
        The method create a StandardScaler and compute the mean and std based on entire data sets
        To compute the mean and std value, the entire data are load and concat together

        :Return:
          _standard_scaler: StandardScaler object
        """
        print('standardizing...')
        # Load measurements data and concat together for StandardScaler fit
        data = []
        for idx in range(len(self)):
            feature, _ = self[idx]
            data.append(feature.numpy())
        data = np.concatenate(data, axis=0)
        # Create Stan
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)
        print(f"mean: {list(standard_scaler.mean_)}")
        print(f"scale: {list(standard_scaler.scale_)}")
        print(f"var: {list(standard_scaler.var_)}")

        # Store the computed mean, scal and var value into a json file for reusing late on
        params = {"mean": list(standard_scaler.mean_), "scal": list(standard_scaler.scale_), "var": list(standard_scaler.var_)}
        with open('_standardizParam.json', 'w') as f:
            json.dump(params, f)

        return standard_scaler

    def __getitem__(self, idx):
        """
        Overwirte the get function, it returns the load measurement according to the index of the _data_list
        :param idx: the index of the _data_list
        :return: torch tensor dataset
        """
        feature, label = self._load_mf4(idx)
        # Standardize the load data
        if self._standardizer is not None:
            feature = self._standardizer.transform(feature)

        return torch.tensor(feature, dtype = torch.float32), torch.tensor(label, dtype = torch.float32)

    def __len__(self):
        return len(self._data_list)

    def _load_mf4(self, idx):
        """
        The method load the measurement data from a mf4 file, drop unused columns. Return feature and label
        :param idx: the index of the _data_list, point to the related mf4 file
        :return: feature data, label data
        """
        loaded_measurement = MDF(self._data_list[idx]).to_dataframe()

        feature_dataframe = loaded_measurement
        # Generate picksignal_dictionary the keys shall be available in _signal_dictionary and _features
        picksignal_dictionary = {key: value for key, value in self._signal_dictionary.items() if key in self._features}

        # Pick required features(signals) from dataframe, rename to the key name as defined in dictionary
        # Raise error if any feature/signal is unavailable
        if not all([self._pick_signal({renameto: signals.copy()}, feature_dataframe) for renameto, signals in picksignal_dictionary.items()]):
            raise Exception("Some feature signal is unavailable")
        # Sort column according to the _feature and drop unnamed in the _features
        feature_dataframe = feature_dataframe.reindex(columns=self._features)
        # Pick correct label(reference signal) from dataframe
        # aise error if the label is unavailable
        label_dataframe = loaded_measurement
        # The yawrate is required for signal transforming
        label_dataframe['UsedYawRate'] = feature_dataframe['UsedYawRate']
        label_available, label_dataframe = self._build_label(label_dataframe)
        if label_available is not True:
            raise Exception("label is unavailable, pure prediction is unimplemented")
        return feature_dataframe.to_numpy(), label_dataframe.to_numpy()

    def _pick_signal(self, signal_dict, measurement_data_frame):
        """
        The method pick the same name signal as value name in the sig_dict from mdf and rename it to the key
        :param signal_dict: dictionary, possible name of the expected signal
        :param measurement_data_frame: measurement data frame
        :return: bool, True the signal is found
        """
        rename_to, signals = next(iter(signal_dict.items()))
        # Search value(signal) in the data frame columns
        for signal in signals:
            if signal in measurement_data_frame.columns:
                signals.remove(signal)
                measurement_data_frame.drop(columns=signals, inplace=True, errors='ignore')
                # Rename signal to key in measurement data frame
                measurement_data_frame.rename(columns={signal: rename_to}, inplace=True)
                return True
        return False

    def _build_label(self, measurement_data_frame):
        """
        This method try to find the labels/reference signals (Corrsys, RtBox) from the measurement data frame. Return label.
        If both are available and CorrsysPreferred is True, the Corrsys signals are preferred.
        :param measurement_data_frame:
        :return: labels
        """
        corrsys_avl = rtbox_avl = False
        # Search corrsys signals
        if all(sig in measurement_data_frame.columns for sig in self._label_dictionary['Corrsys']):
            # The Corrsys signals are only valid whenn mean of Corrsys_VL is not close to 0
            if measurement_data_frame['Corrsys_VL'].mean() > 3:
                # Transfer the Corrsys signal to cog and convert it to m/s, orginal km/h
                measurement_data_frame['Corrsys_VL'] = measurement_data_frame['Corrsys_VL'] / 3.6 - (0.47) * measurement_data_frame['UsedYawRate'] / 180 * math.pi
                measurement_data_frame['Corrsys_VQ'] = measurement_data_frame['Corrsys_VQ'] / 3.6 + (1.5 + 0.74) * measurement_data_frame['UsedYawRate'] / 180 * math.pi
                corrsys_avl = True
        # Search Rt Box signals
        if all(sig in measurement_data_frame.columns for sig in self._label_dictionary['RtBox']):
                # Reverse the sign of lateral velocity, left shall be positive, originally positive is to right
                measurement_data_frame['RT_Msg604_VelLateral'] = measurement_data_frame['RT_Msg604_VelLateral'] * -1
                measurement_data_frame['RT_Msg604_VelForward'] = measurement_data_frame['RT_Msg604_VelForward']
                rtbox_avl = True
        # Drop unneeded columns in measurement_data_frame
        if self._label_dictionary['CorrsysPreferred'] and corrsys_avl:
            measurement_data_frame = measurement_data_frame.reindex(columns=self._label_dictionary['Corrsys'])
            return True, measurement_data_frame
        elif ~self._label_dictionary['CorrsysPreferred'] and rtbox_avl:
            measurement_data_frame = measurement_data_frame.reindex(columns=self._label_dictionary['RtBox'])
            return True, measurement_data_frame
        else:
            return None, None
