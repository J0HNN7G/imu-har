"""Generate slurm batch experiments"""
import os
import json
import argparse

# grid generation
import numpy as np


# constants
LOC_OPTS = ['PERSONAL', 'EDI']

PARAM_LIST = ['DATA.batch_size', 'OPTIM.lr', 'OPTIM.momentum', 'OPTIM.weight_decay', 'LR.step_size', 'LR.gamma']
PARAM_CALLS = ['TRAIN.' + x for x in PARAM_LIST]
EXP_NAME = 'name'
CMD_NAME = 'cmd'
SEP = '\t'

def cartesian(arrays):
    """
    Generate a Cartesian product of input arrays.
    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = [x.dtype for x in arrays]

    n = np.prod([x.size for x in arrays])
    product = []

    m = int(n / arrays[0].size)
    first_column = np.repeat(arrays[0], m)
    if len(arrays) > 1:
        rest_of_product = cartesian(arrays[1:])
        for j in range(arrays[0].size):
            product.extend([(first_column[j*m],) + t for t in rest_of_product])
    else:
        product.extend([(x,) for x in first_column])
    return product


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Grid Search Experiment Setup for APRTF"
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

        
    exp_name = f"{cfg['ARCH']}-{cfg['DATASET'].lower()}" 
    main_project_path = os.path.join(MAIN_HOME, MAIN_USER, MAIN_PROJECT)
    train_path = os.path.join(main_project_path, cfg['TRAIN_FN'])
    config_path = os.path.join(main_project_path, cfg['CONFIG_DN'], f"{exp_name}.yaml" )
    
    node_project_path = os.path.join(NODE_HOME, NODE_USER, NODE_PROJECT)
    data_path = os.path.join(node_project_path, cfg['DATA_DN'], cfg['DATASET'])
    ckpt_path = os.path.join(node_project_path, cfg['CKPT_DN'], exp_name)

    base_call = f"python {train_path} -c {config_path} -i {data_path} -o {ckpt_path}"

    # parameters
    batch_size = [1, 2, 4, 8, 16, 32, 64]
    lrs = [0.005]
    step_sizes = [3]
    gammas = [0.1]
    momentums = [0.9]
    weight_decays = [0.0005] 

    param_list = [batch_size, lrs, momentums, weight_decays, step_sizes, gammas]
    settings = cartesian(param_list)

    # generation
    nr_expts = len(settings)
    print(f'Total experiments = {nr_expts}')

    main_slurm_path = os.path.join(main_project_path, cfg['SLURM_DN'])
    main_exp_path = os.path.join(main_slurm_path, cfg['EXP']['TSV']['DEFAULT_FN'])
    # clear tsv and create header
    with open(main_exp_path, 'w') as f:
        header =  SEP.join(PARAM_LIST + [EXP_NAME, CMD_NAME]) + '\n'
        f.write(header)

    for i, params in enumerate(settings, start=1):
        param_call_str = ' '.join(f"{param_call} {param}" for param_call, param in zip(PARAM_CALLS, params))
        # Note that we don't set a seed for rep - a seed is selected at random
        # and recorded in the output data by the python script
        expt_call = f"{base_call}_{i} {param_call_str}"
        dn = f"{exp_name}_{i}"
        
        with open(main_exp_path, 'a') as f:
            line = SEP.join([str(x) for x in params] + [dn, expt_call]) + '\n'
            f.write(line)