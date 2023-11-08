"""Load ODGT files for PDIoT dataset"""
import json
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def drop_unwanted_columns(dataframe):
    drop_list = dataframe.filter(['timestamp', 'mag_x', 'mag_y', 'mag_z', 'ind', 'Unnamed: 0'])
    return dataframe.drop(drop_list, axis=1)


def get_sliding_windows(data, window_size, overlap, refresh_rate):
    return sliding_window_view(data, int(refresh_rate * window_size), axis=0)[::int(overlap * refresh_rate * window_size)]


def odgt2data(odgt_fp, window_size=1.0, overlap=0.5, refresh_rate=25.0):
    odgt = [json.loads(x.rstrip()) for x in open(odgt_fp, 'r')]
    
    X = []
    y = []
    for recording in odgt:
        df = pd.read_csv(recording['filepath'])
        df = drop_unwanted_columns(df)
        samples = get_sliding_windows(df.to_numpy(), window_size, overlap, refresh_rate)
        for sample in samples:
            X.append(sample)
            y.append(recording['annotation'])
    return X, y