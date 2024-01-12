"""Generate ODGT label files for PDIoT dataset"""
import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm


# configuration for dataset split ratios and task names
TRAIN_FRAC = 0.95
DATA_SPLIT = ['train', 'val', 'full']
DATASET_NAME = 'pdiot-data'
ODGT_FILE_FORMAT = f'{DATASET_NAME}.odgt'
TASK_NAMES = {
    'train': ['motion', 'dynamic', 'static', 'breath', 'resp'],
    'test': ['t1', 't2', 't3', 't4', 'all']
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

def get_subject(data_fp):
    return parse_metadata(data_fp)[0]

def get_label(task, data_fp):
    user, device, activity, breathing = parse_metadata(data_fp)

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
    elif task == 'resp':
        if (activity in static_list) and (breathing in breathing_list[:3]):
            return breathing_dict[breathing]
    elif task == 't1':
        if (activity in activity_list) and (breathing == breathing_list[0]):
            return activity_dict[activity]
    elif task == 't2':
        if (activity in static_list) and (breathing in breathing_list[:3]):
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 't3' or task == 't4':
        if activity in static_list:
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 'all':
        # 't4' task generates a unique label for all combinations
        if activity in dynamic_list:
            return (static_max_len * breathing_max_len) + dynamic_dict[activity]
        elif activity in static_list:
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    else:
        raise ValueError("Unrecognized task.")
    
    return -1


def indices2odgt(odgt_fp, indices, data_fps, annotations, subjects=None, labels=None):
    """
    Write ODGT label files for dataset given indices, file paths, and labels.
    """
    if labels is None:
        for idx in tqdm(indices): 
            sample = {
                'filepath': data_fps[idx],
                'annotation': annotations[idx],
            }

            with open(odgt_fp, 'a') as f:
                json.dump(sample, f)
                f.write('\n')
    else:
        for idx in tqdm(indices): 
            sample = {
                'filepath': data_fps[idx],
                'subject': subjects[idx],
                'annotation': annotations[idx],
                'labels': labels[idx]
            }

            with open(odgt_fp, 'a') as f:
                json.dump(sample, f)
                f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDIoT Classification Dataset")
    parser.add_argument("-d", "--dir", required=True, help="Directory path to the dataset", type=str)
    parser.add_argument("-t", "--task", default='breath', help="Task for which to generate labels", type=str)
    parser.add_argument("-s", "--split", action='store_true', help="Flag to split into train and val sets")
    parser.add_argument("-o", "--overwrite", action='store_true', help="Flag to overwrite existing files")
    args = parser.parse_args()

    # Validate input directory and task
    if not os.path.exists(args.dir):
        raise ValueError('Data directory not found')
    if args.task not in TASK_NAMES['train'] + TASK_NAMES['test']:
        raise ValueError(f'Invalid task: {args.task}')

    # Prepare dataset file paths
    print('Starting ODGT file creation')
    dataset_dir = os.path.join(args.dir, DATASET_NAME)
    data_fps = glob.glob(f'{dataset_dir}/updated_anonymized_dataset_2023/Respeck/*/*')

    # get subject ids
    all_subjects = set([get_subject(data_fp) for data_fp in data_fps])
    subject_ordering = sorted(all_subjects, key=lambda x: int(x[1:]))
    subject_lookup = {subject: i for i, subject in enumerate(subject_ordering)}

    # Filter and label data file paths
    data_fps_valid, annotations = [], []
    if args.task in TASK_NAMES['train']:
        labels = None
        subjects = None
    else:
        labels = []
        subjects = []

    for data_fp in data_fps:
        annotation = get_label(args.task, data_fp)
        if annotation != -1:
            data_fps_valid.append(data_fp)
            annotations.append(annotation)

            if args.task in TASK_NAMES['test']:
                subject_id = subject_lookup[get_subject(data_fp)]
                subjects.append(subject_id)

            if args.task == 't1':
                label = [get_label(task, data_fp) for task in ['motion', 'dynamic', 'static']]
                labels.append(label)
            elif args.task == 't2':
                label = [get_label(task, data_fp) for task in ['static', 'resp']]
                labels.append(label)
            elif (args.task == 't3') or (args.task == 't4'):
                label = [get_label(task, data_fp) for task in ['static', 'breath']]
                labels.append(label)
            elif args.task == 'all':
                label = [get_label(task, data_fp) for task in ['motion', 'dynamic', 'static', 'breath']]
                labels.append(label)
    
    
    if args.split:
        # Store indices by class
        indices = np.random.permutation(range(len(data_fps_valid))).tolist()
        class_indices = {}
        for i in indices:
            annotation = annotations[i]
            if annotation not in class_indices:
                class_indices[annotation] = []
            class_indices[annotation].append(i)

        # Construct split_indices by adding proportional to the number of each class in the overall dataset
        split_indices = [[], []]  # Train and validation splits
        for class_label, indices in class_indices.items():
            num_samples = len(indices)
            limit = int(num_samples * TRAIN_FRAC)
            train_indices = indices[:limit]
            val_indices = indices[limit:]

            split_indices[0].extend(train_indices)
            split_indices[1].extend(val_indices)

            np.random.shuffle(split_indices[0])
            np.random.shuffle(split_indices[1])

        # Generate ODGT files for train and validation splits
        for i, split in enumerate(DATA_SPLIT[:2]):
            odgt_fp = os.path.join(dataset_dir, f'{split}_{args.task}_{ODGT_FILE_FORMAT}')
            if (not os.path.exists(odgt_fp)) or args.overwrite:
                open(odgt_fp, 'w').close()
                indices2odgt(odgt_fp, split_indices[i], data_fps_valid, annotations, subjects, labels)
                print(f'{split.capitalize()} file saved at: {odgt_fp}')
            elif not args.overwrite:
                print(f'{split.capitalize()} indexing already done!')

    # Generate a single ODGT file without splitting
    else:
        odgt_fp = os.path.join(dataset_dir, f'{DATA_SPLIT[2]}_{args.task}_{ODGT_FILE_FORMAT}')
        if (not os.path.exists(odgt_fp)) or args.overwrite:
            open(odgt_fp, 'w').close()
            indices2odgt(odgt_fp, list(range(len(data_fps_valid))), data_fps_valid, annotations, subjects, labels)
            print(f'File saved at: {odgt_fp}')
        elif not args.overwrite:
            print('Indexing already done!')
