"""Log experiment checkpoint to wandb"""
# args
import os
import argparse

from har.config.train import cfg
from yacs.config import CfgNode as CN

# help
import logging

# wandb, api key should be give prior to script 
import wandb

# Constants
MODEL_NAME  = 'model'
TRAIN_NAME  = 'train'
VAL_NAME = 'val'
VALUE_NAME = 'acc'
VAL_VALUE_NAME = f'{VAL_NAME}_{VALUE_NAME}'
BEST_VALUE_NAME = f'best_{VAL_VALUE_NAME}'
SEP = ','

def add_to_wandb_config(wandb_config, yacs_config, prefix=""):
    for key, value in yacs_config.items():
        if isinstance(value, CN):
            add_to_wandb_config(wandb_config, value, f"{prefix}/{key}" if prefix else key)
        else:
            wandb_config[f"{prefix}/{key}" if prefix else key] = value


def main(cfg):
    """
    Main function for performing logging with Wandb for PDIoT classification training.

    Parameters:
    - cfg (object): A configuration object containing experiment parameters.

    Returns:
    None
    """
    exp_name = f't{cfg.DATASET.task}-{cfg.DATASET.component}'

    # Assuming exp_name is defined elsewhere in your code
    run = wandb.init(
        project='pdiot-ml',
        name=exp_name,
        config={}
    )
    add_to_wandb_config(run.config, cfg)


    with open(cfg.TRAIN.history, 'r') as f:
            lines = f.readlines()
            # ignore epoch
            headers = lines[0].split(SEP)[1:]

            best_val = -1
            val_idx = headers.index(VAL_VALUE_NAME)

            for content in lines[1:]:
                content = content.split(SEP)[1:]
                vals = [float(x) for x in content]
                # log row
                val_dict = dict(list(zip(headers,vals)))
                run.log(val_dict)
                # update summary
                if best_val < vals[val_idx]:
                    best_val = vals[val_idx]
    
    run.summary.update({BEST_VALUE_NAME : best_val})
    run.finish()
    
    logging.info('Wandb Logging Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wandb Logging for PDIoT ML"
    )
    parser.add_argument(
        "-c", "--ckpt",
        metavar="PATH",
        help="path to model checkpoint directory",
        type=str,
    )
    args = parser.parse_args()
    cfg_fp = os.path.join(args.ckpt, 'config.yaml')
    assert os.path.exists(cfg_fp), 'config.yaml does not exist!'
    cfg.merge_from_file(cfg_fp)
    cfg.TRAIN.path = args.ckpt

    # setup logger
    cfg.TRAIN.log = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.log)
    cfg.TRAIN.history = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.history)
    assert os.path.exists(cfg.TRAIN.log), 'logs do not exist!'
    assert os.path.exists(cfg.TRAIN.history), 'history does not exist!'
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(cfg.TRAIN.log)])
    logging.info(f"Starting Wandb Logging for experiment {cfg.TRAIN.path.split('/')[-1]}")

    main(cfg)