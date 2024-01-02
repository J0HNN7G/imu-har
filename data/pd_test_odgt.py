"""Generate ODGT label files for PDIoT dataset"""
import os
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm


# configuration for cross-validation split
TASKS = [1,2,3]
DATASET_NAME = 'pdiot-data'
ODGT_EXT = '.odgt'


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


def generate_data(student_fps, data_type):
    data_fps = []
   
    for student_fp in student_fps:
        data_fps = data_fps + glob.glob(f'{student_fp}/*.csv')
    labels = [get_label(data_type, fp) for fp in data_fps]
 
    return data_fps, labels

def parse_metadata(data_fp):
    parts = data_fp.split('/')[-1].split('_')
    return parts[0], parts[1], parts[2], parts[3].replace('.csv','')


def get_label(task, data_fp):
    user, device, activity, breathing = parse_metadata(data_fp)
    
    # Check if activity is valid
    if activity not in dynamic_list and activity not in static_list:
        raise ValueError(f'Invalid activity: {activity}')
    # Check if breathing is valid
    if breathing not in breathing_list:
        raise ValueError(f'Invalid breathing: {breathing}')
    
    if task == 'motion':
        if activity in activity_list:
            return activity_dict[activity]
    elif task == 'static':
        if (activity in static_list) and (breathing in breathing_list[:3]):
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 'dynamic':
        if activity in static_list:
            static_index = static_dict[activity]
            breath_index = breathing_dict[breathing]
            return static_index + breath_index * static_max_len
    elif task == 'resp':
        pass
    elif task == 'breath':
        pass
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
    parser.add_argument("-t", "--task", default=1, help="Task for which to generate labels", type=int)
    parser.add_argument("-s", "--student", default=0, help="Student to use for test set, ordered by name, alphabetically", type=int)
    args = parser.parse_args()

    # Validate input directory and task
    if not os.path.exists(args.dir):
        raise ValueError('Data directory not found')
    if args.task not in TASKS:
        raise ValueError(f'Invalid task: {args.task}')

    # Prepare dataset file paths
    print('Starting ODGT file creation')
    dataset_dir = os.path.join(args.dir, DATASET_NAME)
    all_student_fps = glob.glob(f'{dataset_dir}/updated_anonymized_dataset_2023/Respeck/*')
    if (args.student > len(all_student_fps)) or (args.student < 0):
        raise ValueError(f'Invalid student index: {args.student}')
    
    all_student_fps.sort()
    train_student_fps = all_student_fps[:args.student] + all_student_fps[args.student+1:] 
    test_student_fp = all_student_fps[args.student]

    if args.task == 1:
        train_motion_fps, train_motion_labels = generate_data(train_student_fps, 'motion')
        train_static_fps, train_static_labels = generate_data(train_student_fps, 'static')
        train_dynamic_fps, train_dynamic_labels = generate_data(train_student_fps, 'dynamic')
        
        test_motion_fps, test_motion_labels = generate_data(test_student_fp, 'motion')
        test_static_fps, test_static_labels = generate_data(test_student_fp, 'static')
        test_dynamic_fps, test_dynamic_labels = generate_data(test_student_fp, 'dynamic')
        
    elif args.task == 2:
        train_static_fps, train_static_labels = generate_data(train_student_fps, 'static')
        train_resp_fps, train_resp_labels = generate_data(train_student_fps, 'resp')
        
        test_static_fps, test_static_labels = generate_data(test_student_fp, 'static')
        test_resp_fps, test_resp_labels = generate_data(test_student_fp, 'resp')
        
    elif args.task == 3:
        train_static_fps, train_static_labels = generate_data(train_student_fps, 'static')
        train_breath_fps, train_breath_labels = generate_data(train_student_fps, 'breath')
        
        test_static_fps, test_static_labels = generate_data(test_student_fp, 'static')
        test_breath_fps, test_breath_labels = generate_data(test_student_fp, 'breath')
    else:
        raise ValueError("Unrecognized task.")
    