"""Generate ODGT label files for PDIoT dataset"""
# general
import os
import glob
import json
import argparse

# boxes
import numpy as np

# progress bar
from tqdm import tqdm


# constants
TRAIN_NAME = 'train'
VAL_NAME = 'val'
DATASET_NAME = 'pdiot-data'
ODGT_NAME = f'{DATASET_NAME}.odgt'

# List of breathing types
breathing_list = [
    'normal',
    'coughing',
    'singing',
    'eating',
    'hyperventilating',
    'laughing',
    'talking'
]

# List of activity types
activity_list = [
    'sitting',
    'standing',
    'lying down back',
    'lying down right',
    'lying down on left',
    'lying down on stomach',
    'ascending stairs',
    'descending stairs',
    'running',
    'normal walking',
    'shuffle walking',
    'miscellaneous movements'
]

static_activities = activity_list[:6]
dynamic_activities = activity_list[6:]


def parse_metadata(data_fp):
    parts = data_fp.split('/')[-1].split('_')
    return parts[0], parts[1], parts[2], parts[3]


def get_label(task, data_fp):
    user, device, activity, breathing = parse_metadata(data_fp)

    # Exclude certain users or devices
    if (device != 'respeck') or (user == 'S37'):
        return -1

    # Check if task is valid
    if task not in ['all', 'dynamic', 'static', 'breath']:
        raise ValueError(f'Invalid task: {task}')

    # Check if activity is valid
    if activity not in dynamic_activities and activity not in static_activities:
        raise ValueError(f'Invalid activity: {activity}')
    
    # Check if breathing is valid
    if breathing not in breathing_list:
        raise ValueError(f'Invalid breathing: {breathing}')

    if task == 'dynamic':
        # Dynamic task returns the dynamic activity label if activity is dynamic
        if activity in dynamic_activities:
            return dynamic_activities.index(activity)
        else:
            return -1  # This ensures static activities don't get processed here

    elif task == 'static':
        # Static task creates a combined label for static activity and breathing if activity is static
        if activity in static_activities:
            static_index = static_activities.index(activity)
            breath_index = breathing_list.index(breathing)
            return static_index * len(breathing_list) + breath_index
        else:
            return -1  # This ensures dynamic activities don't get processed here

    elif task == 'breath':
        # Breath task returns breathing label directly if activity is static
        if activity in static_activities:
            return breathing_list.index(breathing)
        else:
            return -1  # Breath labels only apply to static activities

    elif task == 'all':
        # 'All' task generates a unique label for all combinations
        if activity in dynamic_activities:
            # Dynamic activities have a unique label, offset by the total count of static-breath combinations
            return (len(static_activities) * len(breathing_list)) + dynamic_activities.index(activity)
        elif activity in static_activities:
            # Static activities are combined with breathing types
            static_index = static_activities.index(activity)
            breath_index = breathing_list.index(breathing)
            return (static_index * len(breathing_list)) + breath_index
        else:
            return -1  # In case activity is neither dynamic nor static

    else:
        # If none of the above cases are met, an invalid task is provided
        raise ValueError("Unrecognized task.")


def indices2odgt(odgt_fp, indices, data_fps, labels):
    """
    Generate ODGT label files for PennFudanPed dataset using given indices and file paths.

    Parameters:
    - odgt_fp (str): File path to save the ODGT label file.
    - dir_p (str): Directory path to the dataset.
    - indices (list): List of indices to process.
    - data_fps (list): List of data file paths.
    - labels (list): List of mask file paths.

    Returns:
    None
    """
    for idx in tqdm(indices): 
        sample = {
            'filepath': data_fps[idx],
            'annotation': labels[idx]
        }

        with open(odgt_fp, 'a') as f:
            json.dump(sample, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PDIoT Classification Dataset"
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        metavar="PATH",
        help="absolute path to intended PennFudan dataset directory",
        type=str,
    )
    parser.add_argument(
        "-t", "--task",
        default='all',
        metavar="STR",
        help="task to generate labels for: all, dynamic, static, breath",
        type=str,
    )
    parser.add_argument(
        "-f", "--frac",
        default=0.8,
        metavar="FLOAT",
        help="fraction of samples to put in training set vs validation set",
        type=float,
    )
    args = parser.parse_args()
    if not os.path.exists(args.dir):
        raise ValueError('Cannot find data directory')
    if not 0 <= args.frac <= 1:
        raise ValueError('train fraction out of range [0,1]!')


    print(f'Starting odgt file creation')
    dataset_dir_p = os.path.join(args.dir, DATASET_NAME)
    data_fps = glob.glob(dataset_dir_p + '/anonymized_dataset_2023/*/*/*' )


    data_fps_valid = []
    labels = []
    for data_fp in data_fps:
        label = get_label(args.task, data_fp)
        if label != -1:
            data_fps_valid.append(data_fp)
            labels.append(label)
    data_fps = data_fps_valid
            

    indices = np.random.permutation(len(data_fps)).tolist()
    limit = int(len(data_fps) * args.frac) 

    print(f'Train indexing')
    odgt_fp_train = os.path.join(dataset_dir_p, f'{TRAIN_NAME}_{args.task}_{ODGT_NAME}')
    if os.path.exists(odgt_fp_train):
        print('Train indexing already done!')
    else:
        open(odgt_fp_train, 'w').close() 
        indices2odgt(odgt_fp_train, indices[:limit], data_fps, labels)
        print(f'Train file saved at: {odgt_fp_train}')

    print(f'Validation indexing')
    odgt_fp_val = os.path.join(dataset_dir_p, f'{VAL_NAME}_{args.task}_{ODGT_NAME}')
    if os.path.exists(odgt_fp_val):
        print('Validation indexing already done!')
    else:
        open(odgt_fp_val, 'w').close() 
        indices2odgt(odgt_fp_val, indices[limit:], data_fps, labels)
        print(f'Validation file saved at: {odgt_fp_val}')




    

