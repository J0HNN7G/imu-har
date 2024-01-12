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


def odgt2test(odgt_fp, task, subject_id, window_size, overlap_size):
    test_dict = {
        'train' : {
            'X' : [],
            'y' : [],
        },
        'val' : {
            'X' : [],
            'y' : [], 
        }
    }

    for data_set in ['train', 'val']:
        if task == 1:
            test_dict[data_set]['motion'] = []
            test_dict[data_set]['dynamic'] = []
            test_dict[data_set]['static'] = []
        elif task == 2: 
            test_dict[data_set]['static'] = []
            test_dict[data_set]['resp'] = []
        elif task == 3:
            test_dict[data_set]['static'] = []
            test_dict[data_set]['breath'] = []
        elif task == 4:
            test_dict[data_set]['static'] = []
            test_dict[data_set]['breath'] = []
        else:
            raise ValueError(f'Unrecognized task: {task}')

    odgt = [json.loads(x.rstrip()) for x in open(odgt_fp, 'r')]

    for recording in odgt:
        df = pd.read_csv(recording['filepath'])
        df = drop_unwanted_columns(df)
        samples = get_sliding_windows(df.to_numpy(), window_size, overlap_size)

        data_set = 'train' if recording['subject'] != subject_id else 'val'
        for sample in samples:
            test_dict[data_set]['X'].append(sample)
            test_dict[data_set]['y'].append(recording['annotation'])
            if task == 1:
                test_dict[data_set]['motion'].append(recording['labels'][0])
                test_dict[data_set]['dynamic'].append(recording['labels'][1])
                test_dict[data_set]['static'].append(recording['labels'][2])
            elif task == 2:
                test_dict[data_set]['static'].append(recording['labels'][0])
                test_dict[data_set]['resp'].append(recording['labels'][1])
            elif task == 3:
                test_dict[data_set]['static'].append(recording['labels'][0])
                test_dict[data_set]['breath'].append(recording['labels'][1])
            elif task == 4:
                test_dict[data_set]['static'].append(recording['labels'][0])
                test_dict[data_set]['breath'].append(recording['labels'][1])
        
    test_dict['train']['X'] = np.array(test_dict['train']['X'])
    test_dict['train']['y'] = np.array(test_dict['train']['y'])
    test_dict['val']['X'] = np.array(test_dict['val']['X'])
    test_dict['val']['y'] = np.array(test_dict['val']['y'])

    for data_set in ['train', 'val']:
        if task == 1:
            test_dict[data_set]['motion'] = np.array(test_dict[data_set]['motion'])
            test_dict[data_set]['dynamic'] = np.array(test_dict[data_set]['dynamic'])
            test_dict[data_set]['static'] = np.array(test_dict[data_set]['static'])
        elif task == 2:
            test_dict[data_set]['static'] = np.array(test_dict[data_set]['static'])
            test_dict[data_set]['resp'] = np.array(test_dict[data_set]['resp'])
        elif task == 3:
            test_dict[data_set]['static'] = np.array(test_dict[data_set]['static'])
            test_dict[data_set]['breath'] = np.array(test_dict[data_set]['breath'])
        elif task == 4:
            test_dict[data_set]['static'] = np.array(test_dict[data_set]['static'])
            test_dict[data_set]['breath'] = np.array(test_dict[data_set]['breath'])

    return test_dict