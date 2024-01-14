"""Train all model components for all tasks on slurm"""
import os
import json
import argparse
from itertools import product


TASK_MODEL_DICT = {
    1: {'motion' : 2, 
        'dynamic': 6, 
        'static': 5},
    2: {'static': 5, 
        'resp': 3},
    3: {'static' : 5, 
        'breath' : 4},
    4: {'static' : 5, 
        'breath' : 4}
}

# constants
TASKS = [1,2,3]
LOC_OPTS = ['PERSONAL', 'EDI']

PARAM_LIST = ['--task', '--part']
EXP_DN_FORMAT = 'task_{}_{}'
EXP_NAME = 'name'
CMD_NAME = 'cmd'
SEP = ','


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm trainig HAR components for all tasks"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="PATH",
        help="Absolute path to path config file",
        type=str,
    )
    parser.add_argument(
        "-l", "--loc",
        required=True,
        metavar="STR",
        choices=LOC_OPTS,
        help="Working directory [PERSONAL,EDI]",
        type=str,
    )
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError('Config file not found!')
    cfg = {}
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    MAIN_HOME = cfg[args.loc]['HOME']
    MAIN_USER = cfg[args.loc]['USER']
    MAIN_PROJECT = cfg[args.loc]['PROJECT']

    # node details 
    if args.loc == LOC_OPTS[1]:
        NODE_HOME = cfg['SCRATCH']['HOME']
        NODE_USER = cfg['SCRATCH']['USER']
        NODE_PROJECT = cfg['SCRATCH']['PROJECT']
    elif args.loc == LOC_OPTS[0]:
        NODE_HOME = MAIN_HOME
        NODE_USER = MAIN_USER
        NODE_PROJECT = MAIN_PROJECT
    else:
        raise ValueError('Unsupported choice!')


    main_project_path = os.path.join(MAIN_HOME, MAIN_USER, MAIN_PROJECT)
    train_path = os.path.join(main_project_path, cfg['TRAIN_FN'])
    config_path = os.path.join(main_project_path, cfg['CONFIG_DN'])
    
    node_project_path = os.path.join(NODE_HOME, NODE_USER, NODE_PROJECT)
    data_path = os.path.join(node_project_path, cfg['DATA_DN'], cfg['DATASET'])
    ckpt_path = os.path.join(node_project_path, cfg['CKPT_DN'], 'train')

    base_call = f"python {train_path} -c {config_path} -i {data_path} -o {ckpt_path}"

    nr_expts = 7
    print(f'Total experiments = {nr_expts}')

    # generation
    main_slurm_path = os.path.join(main_project_path, cfg['SLURM_DN'], 'train')
    main_exp_path = os.path.join(main_slurm_path, cfg['EXP']['CSV']['DEFAULT_FN'])
    # clear csv and create header
    with open(main_exp_path, 'w') as f:
        header =  SEP.join(PARAM_LIST + [EXP_NAME, CMD_NAME]) + '\n'
        f.write(header)

    # generation
    main_slurm_path = os.path.join(main_project_path, cfg['SLURM_DN'], 'train')
    main_exp_path = os.path.join(main_slurm_path, cfg['EXP']['CSV']['DEFAULT_FN'])
    # clear csv and create header
    with open(main_exp_path, 'w') as f:
        header =  SEP.join(PARAM_LIST + [EXP_NAME, CMD_NAME]) + '\n'
        f.write(header)

    for task in TASKS:
        for component in TASK_MODEL_DICT[task].keys():
            params = [task, component]
            param_call_str = ' '.join(f"{param_call} {param}" for param_call, param in zip(PARAM_LIST, params))
            dn = EXP_DN_FORMAT.format(task, component)
            expt_call = f"{base_call}/{dn} {param_call_str}"
            with open(main_exp_path, 'a') as f:
                line = SEP.join([str(x) for x in params] + [os.path.join('train', dn), expt_call]) + '\n'
                f.write(line)