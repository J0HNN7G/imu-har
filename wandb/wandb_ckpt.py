"""Log experiment checkpoint to wandb"""
# args
import os
import argparse

from ml.config import cfg

# help
import logging
from train import TRAIN_NAME, VAL_AP_NAME, SEP

# wandb, api key should be give prior to script 
import wandb

# Constants
BEST_VAL_NAME = VAL_AP_NAME + '_best'


def main(cfg):
    """
    Main function for performing logging with Wandb for PDIoT classification training.

    Parameters:
    - cfg (object): A configuration object containing experiment parameters.

    Returns:
    None
    """
    # remove batch run from name:
    dataset_name = cfg.DATASET.path.split('/')[-1].lower()
    exp_name = f'{cfg.MODEL.arch}_{dataset_name}'

    run = wandb.init(
        project='pdiot-ml',
        name=exp_name,
        config = {
            'architecture' : cfg.MODEL.arch,
            'dataset' : dataset_name,
            f'{TRAIN_NAME}/data/batch_size' : cfg.TRAIN.DATA.batch_size, 
            f'{TRAIN_NAME}/len/epochs' : cfg.TRAIN.LEN.num_epoch,
            f'{TRAIN_NAME}/len/early_stop' : cfg.TRAIN.LEN.early_stop,
            f'{TRAIN_NAME}/optim/optimizer' : cfg.TRAIN.OPTIM.optim,
            f'{TRAIN_NAME}/optim/momentum' : cfg.TRAIN.OPTIM.momentum, 
            f'{TRAIN_NAME}/optim/weight_decay' : cfg.TRAIN.OPTIM.weight_decay, 
            f'{TRAIN_NAME}/optim/initial': cfg.TRAIN.OPTIM.lr,
            f'{TRAIN_NAME}/lr/schedule': cfg.TRAIN.LR.schedule,
            f'{TRAIN_NAME}/lr/step_size' : cfg.TRAIN.LR.step_size,
            f'{TRAIN_NAME}/lr/gamma' : cfg.TRAIN.LR.gamma,
            f'{TRAIN_NAME}/seed' : cfg.TRAIN.seed, 
        }
    )

    with open(cfg.TRAIN.history, 'r') as f:
            lines = f.readlines()
            # ignore epoch
            headers = lines[0].split(SEP)[1:]

            best_val_ap = -1
            val_ap_idx = headers.index(VAL_AP_NAME)

            for content in lines[1:]:
                content = content.split(SEP)[1:]
                vals = [float(x) for x in content]
                # log row
                val_dict = dict(list(zip(headers,vals)))
                run.log(val_dict)
                # update summary
                if best_val_ap < vals[val_ap_idx]:
                    best_val_ap = vals[val_ap_idx]
    
    run.summary.update({BEST_VAL_NAME : best_val_ap})
    run.finish()
    
    logging.info('Wandb Logging Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wandb Logging for PyTorch Pedestrian Detection Finetuning"
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="ckpt/fasterrcnn_resnet50_fpn-pennfudan",
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
    cfg.TRAIN.history = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.hist)
    assert os.path.exists(cfg.TRAIN.log), 'logs do not exist!'
    assert os.path.exists(cfg.TRAIN.history), 'history does not exist!'
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(cfg.TRAIN.log)])
    logging.info(f"Starting Wandb Logging for experiment {cfg.TRAIN.path.split('/')[-1]}")

    main(cfg)