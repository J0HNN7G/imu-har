"""Load ODGT files for PDIoT dataset"""
import numpy as np
import json
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


TRAIN_FRAC = 0.95

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


def odgt2train(odgt_fp, task, part, window_size, overlap_size):
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

    return train_X, val_X, train_y, val_y




def odgt2train(odgt_fp, task, part, window_size, overlap_size):
    odgt = [json.loads(x.rstrip()) for x in open(odgt_fp, 'r')]

    X = []
    y = []

    class_indices = {}  # Dictionary to store indices for each class

    for recording in odgt:
        df = pd.read_csv(recording['filepath'])
        df = drop_unwanted_columns(df)

        # Determine the correct index from recording['labels'] based on the 'task' and 'part' arguments
        if task == 1:
            if part == 'motion':
                label_index = 0
            elif part == 'dynamic':
                label_index = 1
            elif part == 'static':
                label_index = 2
            else:
                raise ValueError(f'Invalid part: {part} for task: {task}')
        elif task == 2:
            if part == 'static':
                label_index = 0
            elif part == 'resp':
                label_index = 1
            else:
                raise ValueError(f'Invalid part: {part} for task: {task}')
        elif task in [3, 4]:
            if part == 'static':
                label_index = 0
            elif part == 'breath':
                label_index = 1
            else:
                raise ValueError(f'Invalid part: {part} for task: {task}')
        else:
            raise ValueError(f'Unrecognized task: {task}')

        # Check if the label is valid before proceeding
        if recording['labels'][label_index] != -1:

            label = recording['labels'][label_index]
            samples = get_sliding_windows(df.to_numpy(), window_size, overlap_size)

            for sample in samples:
                X.append(sample)
                y.append(label)
                if label not in class_indices:
                    class_indices[label] = []
                
                class_indices[label].append(len(X) - 1)  # Store the index for the corresponding class

    # Calculate the train and validation split
    train_indices = []
    val_indices = []

    for class_label, indices in class_indices.items():
        num_samples = len(indices)
        limit = int(num_samples * TRAIN_FRAC)

        # shuffle users
        indices = np.random.permutation(indices)

        train_indices.extend(indices[:limit])
        val_indices.extend(indices[limit:])

    # Shuffle classes
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    # Extract features and labels for train and validation sets
    train_X = np.array([X[i] for i in train_indices])
    val_X = np.array([X[i] for i in val_indices])

    train_y = np.array([y[i] for i in train_indices])
    val_y = np.array([y[i] for i in val_indices])

    return train_X, val_X, train_y, val_y


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

        data_set = 'train' if (recording['subject'] != subject_id) else 'val'
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