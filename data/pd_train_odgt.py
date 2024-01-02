"""Generate ODGT label files for PDIoT dataset"""
import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm


# configuration for dataset split ratios and task names
TRAIN_FRAC = 0.95
TASK_NAMES = ['train', 'val', 'full']
DATASET_NAME = 'pdiot-data'
ODGT_FILE_FORMAT = f'{DATASET_NAME}.odgt'
TASK_CATEGORIES = {
    'train': ['motion', 'dynamic', 'static', 'breath'],
    'test': ['task_1', 'task_2', 'task_3', 'all']
}


# maps and max indices for activity and breathing types
activity_list = [
    'sitting',
    'standing',
    'lyingLeft',
    'lyingRight',    
    'lyingBack',
    'lyingStomach',
    'normalWalking',
    'running',
    'descending',
    'ascending',
    'shuffleWalking',
    'miscMovement'
]
activity_idxs = [0] + list(range(len(activity_list)-1))
activity_dict = dict(zip(activity_list, activity_idxs))
activity_max_len = max(activity_dict.values()) + 1


breathing_list = [
    'breathingNormal',
    'coughing',
    'hyperventilating',
    'singing',
    'eating',
    'laughing',
    'talking'
]
breathing_idxs = list(range(len(breathing_list)-4)) + [3] * 4
breathing_dict = dict(zip(breathing_list, breathing_idxs))
breathing_max_len = max(breathing_dict.values()) + 1


static_list = activity_list[:6]
static_dict = {}
for activity in static_list:
    static_dict[activity] = activity_dict[activity]
static_max_len = max(static_dict.values()) + 1


dynamic_list = activity_list[6:]
dynamic_dict = {}
for i, activity in enumerate(dynamic_list):
    dynamic_dict[activity] = i
dynamic_max_len = max(dynamic_dict.values()) + 1


def parse_metadata(data_fp):
    parts = data_fp.split('/')[-1].split('_')
    return parts[0], parts[1], parts[2], parts[3].replace('.csv','')


def get_label(task, data_fp):
    user, device, activity, breathing = parse_metadata(data_fp)

    # Exclude certain users or devices
    if (device != 'respeck'):
        return -1
    
    # Check if activity is valid
    if activity not in dynamic_list and activity not in static_list:
        raise ValueError(f'Invalid activity: {activity}')
    # Check if breathing is valid
    if breathing not in breathing_list:
        raise ValueError(f'Invalid breathing: {breathing}')
    
    if task == 'motion':
        if activity in static_list:
            return 0
        elif activity in dynamic_list:
            return 1
    if task == 'dynamic':
        if activity in dynamic_list:
            return dynamic_dict[activity]
    elif task == 'static':
        if activity in static_list:
            return static_dict[activity]
    elif task == 'breath':
        if activity in static_list:
            return breathing_dict[breathing]
    elif task == 'all':
        # 'all' task generates a unique label for all combinations
        if activity in dynamic_list:
            return (static_max_len * breathing_max_len) + dynamic_dict[activity]
        elif activity in static_list:
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 'task_1':
        if activity in activity_list:
            return activity_dict[activity]
    elif task == 'task_2':
        if (activity in static_list) and (breathing in breathing_list[:3]):
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 'task_3':
        if activity in static_list:
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    else:
        raise ValueError("Unrecognized task.")
    
    return -1


def indices2odgt(odgt_fp, indices, data_fps, labels):
    """
    Write ODGT label files for dataset given indices, file paths, and labels.
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
    parser = argparse.ArgumentParser(description="PDIoT Classification Dataset")
    parser.add_argument("-d", "--dir", required=True, help="Directory path to the dataset", type=str)
    parser.add_argument("-t", "--task", default='breath', help="Task for which to generate labels", type=str)
    parser.add_argument("-s", "--split", action='store_false', help="Flag to not split into train and val sets")
    args = parser.parse_args()

    # Validate input directory and task
    if not os.path.exists(args.dir):
        raise ValueError('Data directory not found')
    if args.task not in TASK_CATEGORIES['train'] + TASK_CATEGORIES['test']:
        raise ValueError(f'Invalid task: {args.task}')

    # Prepare dataset file paths
    print('Starting ODGT file creation')
    dataset_dir = os.path.join(args.dir, DATASET_NAME)
    data_fps = glob.glob(f'{dataset_dir}/updated_anonymized_dataset_2023/*/*/*')

    # Filter and label data file paths
    data_fps_valid, labels = [], []
    for data_fp in data_fps:
        label = get_label(args.task, data_fp)
        if label != -1:
            data_fps_valid.append(data_fp)
            labels.append(label)

    # Perform dataset split if required
    if args.split:
        indices = np.random.permutation(len(data_fps_valid)).tolist()
        limit = int(len(data_fps_valid) * TRAIN_FRAC)
        split_indices = [
            indices[:limit],
            indices[limit:],
        ]

        # Generate ODGT files for train and validation splits
        for i, split in enumerate(TASK_NAMES[:2]):
            odgt_fp = os.path.join(dataset_dir, f'{split}_{args.task}_{ODGT_FILE_FORMAT}')
            if not os.path.exists(odgt_fp):
                open(odgt_fp, 'w').close()
                indices2odgt(odgt_fp, split_indices[i], data_fps_valid, labels)
                print(f'{split.capitalize()} file saved at: {odgt_fp}')
            else:
                print(f'{split.capitalize()} indexing already done!')

    # Generate a single ODGT file without splitting
    else:
        odgt_fp = os.path.join(dataset_dir, f'{TASK_NAMES[2]}_{args.task}_{ODGT_FILE_FORMAT}')
        if not os.path.exists(odgt_fp):
            open(odgt_fp, 'w').close()
            indices2odgt(odgt_fp, range(len(data_fps_valid)), data_fps_valid, labels)
            print(f'File saved at: {odgt_fp}')
        else:
            print('Indexing already done!')
