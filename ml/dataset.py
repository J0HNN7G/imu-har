"""Load ODGT files for PDIoT dataset"""
import numpy as np
import json
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def drop_unwanted_columns(dataframe):
    drop_list = dataframe.filter(['timestamp', 'mag_x', 'mag_y', 'mag_z', 'ind', 'Unnamed: 0'])
    return dataframe.drop(drop_list, axis=1)


def get_sliding_windows(data, window_size, overlap_size):
    return sliding_window_view(data, window_size, axis=0)[::(overlap_size * window_size)]


def odgt2data(odgt_fp, window_size, overlap_size, num_classes):
    odgt = [json.loads(x.rstrip()) for x in open(odgt_fp, 'r')]
    
    X = []
    y = []
    for recording in odgt:
        df = pd.read_csv(recording['filepath'])
        df = drop_unwanted_columns(df)
        samples = get_sliding_windows(df.to_numpy(), window_size, overlap_size)
        for sample in samples:
            X.append(sample)
            y.append(recording['annotation'])

    X = np.array(X)

    # Convert labels to one-hot encoding using numpy
    y_onehot = np.zeros((len(y), num_classes))
    y_onehot[np.arange(len(y)), y] = 1
    y = np.array(y_onehot)

    return X, y