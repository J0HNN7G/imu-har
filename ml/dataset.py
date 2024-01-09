"""Load ODGT files for PDIoT dataset"""
import numpy as np
import json
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def drop_unwanted_columns(dataframe):
    drop_list = dataframe.filter(['timestamp', 'mag_x', 'mag_y', 'mag_z', 'ind', 'Unnamed: 0'])
    return dataframe.drop(drop_list, axis=1)


def get_sliding_windows(data, window_size, overlap_size):
    # Ensure the step size is positive and smaller than the window size
    step_size = window_size - overlap_size
    if step_size <= 0:
        raise ValueError("Overlap size must be smaller than the window size.")

    # Create the sliding windows using sliding_window_view
    sliding_windows = sliding_window_view(data, window_shape=(window_size,), axis=0)

    # Use the `::` operator to select every nth window, where n is the step size
    windows = sliding_windows[::step_size]

    # Transpose the array to match the shape expected by the model
    windows = windows.transpose(0, 2, 1)

    return windows


def odgt2train(odgt_fp, window_size, overlap_size):
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
    y = np.array(y)

    # Convert labels to one-hot encoding using numpy
    #y_onehot = np.zeros((len(y), num_classes))
    #y_onehot[np.arange(len(y)), y] = 1
    #y = np.array(y_onehot)

    return X, y


def odgt2test(odgt_fp, window_size, overlap_size):
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
    y = np.array(y)

    return X, y