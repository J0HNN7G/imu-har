"""Generate slurm batch experiments"""
import os
import json
import argparse

from itertools import product

# grid generation
import numpy as np


# constants
LOC_OPTS = ['PERSONAL', 'EDI']

PARAM_LIST = ['MODEL.INPUT.window_size', 
              'TRAIN.DATA.overlap_size', 
              'MODEL.INPUT.sensor', 
              'MODEL.INPUT.format', 
              'MODEL.ARCH.LSTM.num_layers', 
              'MODEL.ARCH.LSTM.hidden_size',
              'MODEL.ARCH.MLP.num_layers',
              'MODEL.ARCH.MLP.hidden_size',
              'MODEL.ARCH.MLP.dropout']
EXP_NAME = 'name'
CMD_NAME = 'cmd'
SEP = ','


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Slurm Grid Search Experiment Setup for PDIoT"
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

        
    exp_name = cfg['TASK'] 
    main_project_path = os.path.join(MAIN_HOME, MAIN_USER, MAIN_PROJECT)
    train_path = os.path.join(main_project_path, cfg['TRAIN_FN'])
    config_path = os.path.join(main_project_path, cfg['CONFIG_DN'], f"{exp_name}.yaml" )
    
    node_project_path = os.path.join(NODE_HOME, NODE_USER, NODE_PROJECT)
    data_path = os.path.join(node_project_path, cfg['DATA_DN'], cfg['DATASET'])
    ckpt_path = os.path.join(node_project_path, cfg['CKPT_DN'], exp_name)

    base_call = f"python {train_path} -c {config_path} -i {data_path} -o {ckpt_path}"

    # parameters
    settings = []

    window_size = [(5,4), (10,8), (15,12), (20,16), (25,20)]
    sensor = ['all']


    # log model
    format = ['normal', 'summary']
    lstm_num_layers = [0]
    lstm_hidden_sizes = [0]
    mlp_num_layers = [0]
    mlp_hidden_sizes = [0]
    mlp_dropout = [1.0]

    param_list = [window_size, 
                  sensor, 
                  format, 
                  lstm_num_layers, 
                  lstm_hidden_sizes, 
                  mlp_num_layers, 
                  mlp_hidden_sizes, 
                  mlp_dropout]
    settings = settings + list(product(*param_list))


    # mlp model
    format = ['normal', 'summary']
    lstm_num_layers = [0]
    lstm_hidden_sizes = [0]
    mlp_num_layers = [2, 3, 4]
    mlp_hidden_sizes = [32, 64, 128]
    mlp_dropout = [0.0, 0.2, 0.5]

    param_list = [window_size, 
                  sensor, 
                  format, 
                  lstm_num_layers, 
                  lstm_hidden_sizes, 
                  mlp_num_layers, 
                  mlp_hidden_sizes, 
                  mlp_dropout]
    settings = settings + list(product(*param_list))


    # lstm model
    format = ['window']
    lstm_num_layers = [1, 2]
    lstm_hidden_sizes = [16, 32, 64]
    mlp_num_layers = [0]
    mlp_hidden_sizes = [0]
    mlp_dropout = [1.0]

    param_list = [window_size, 
                  sensor, 
                  format, 
                  lstm_num_layers, 
                  lstm_hidden_sizes, 
                  mlp_num_layers, 
                  mlp_hidden_sizes, 
                  mlp_dropout]
    settings = settings + list(product(*param_list))

    nr_expts = len(settings)
    print(f'Total experiments = {nr_expts}')


    # generation
    main_slurm_path = os.path.join(main_project_path, cfg['SLURM_DN'])
    main_exp_path = os.path.join(main_slurm_path, cfg['EXP']['CSV']['DEFAULT_FN'])
    # clear csv and create header
    with open(main_exp_path, 'w') as f:
        header =  SEP.join(PARAM_LIST + [EXP_NAME, CMD_NAME]) + '\n'
        f.write(header)

    nr_expts = 0
    for i, params in enumerate(settings, start=1):
        params = [params[0][0], params[0][1]] + list(params[1:]) 
        param_call_str = ' '.join(f"{param_call} {param}" for param_call, param in zip(PARAM_LIST, params))
        expt_call = f"{base_call}_{i} {param_call_str}"
        dn = f"{exp_name}_{i}"
        
        with open(main_exp_path, 'a') as f:
            line = SEP.join([str(x) for x in params] + [dn, expt_call]) + '\n'
            f.write(line)
