"""Log experiment checkpoint to wandb"""
# args
import os
import argparse

from har.config.train import cfg

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

def main(cfg):
    """
    Main function for performing logging with Wandb for PDIoT classification training.

    Parameters:
    - cfg (object): A configuration object containing experiment parameters.

    Returns:
    None
    """
    # remove batch run from name:
    if cfg.MODEL.ARCH.MLP.num_layers + cfg.MODEL.ARCH.LSTM.num_layers == 0:
         arch_name = 'log'
    elif cfg.MODEL.ARCH.MLP.num_layers > 0:
         arch_name = 'mlp'
    elif cfg.MODEL.ARCH.LSTM.num_layers > 0:
         arch_name = 'lstm'
    else:
        raise ValueError('Invalid architecture!')
    
    dataset_name = cfg.DATASET.path.split('/')[-1].lower()
    task_name = cfg.DATASET.LIST.train.split('_')[1].lower()
    exp_name = f'{task_name}-{arch_name}'

    run = wandb.init(
        project='pdiot-ml',
        name=exp_name,
        config = {
            'task' : task_name,
            'architecture': arch_name,
            'dataset' : dataset_name,
            'num_classes' : cfg.DATASET.num_classes,
            f'{TRAIN_NAME}/data/batch_size' : cfg.TRAIN.DATA.batch_size, 
            f'{TRAIN_NAME}/data/overlap_size' : cfg.TRAIN.DATA.overlap_size, 
            f'{TRAIN_NAME}/len/epochs' : cfg.TRAIN.LEN.num_epoch,
            f'{TRAIN_NAME}/len/early_stop' : cfg.TRAIN.LEN.early_stop,
            f'{TRAIN_NAME}/optim/optimizer' : cfg.TRAIN.OPTIM.optim,
            f'{TRAIN_NAME}/optim/momentum' : cfg.TRAIN.OPTIM.momentum, 
            f'{TRAIN_NAME}/optim/weight_decay' : cfg.TRAIN.OPTIM.weight_decay, 
            f'{TRAIN_NAME}/optim/lr': cfg.TRAIN.OPTIM.lr,
            f'{TRAIN_NAME}/lr/schedule': cfg.TRAIN.LR.schedule,
            f'{TRAIN_NAME}/lr/step_size' : cfg.TRAIN.LR.step_size,
            f'{MODEL_NAME}/input/format' : cfg.MODEL.INPUT.format,
            f'{MODEL_NAME}/input/sensor' : cfg.MODEL.INPUT.sensor,
            f'{MODEL_NAME}/input/window_size' : cfg.MODEL.INPUT.window_size,
            f'{MODEL_NAME}/arch/mlp/num_layers' : cfg.MODEL.ARCH.MLP.num_layers,
            f'{MODEL_NAME}/arch/mlp/hidden_size' : cfg.MODEL.ARCH.MLP.hidden_size,
            f'{MODEL_NAME}/arch/mlp/dropout' : cfg.MODEL.ARCH.MLP.dropout,
            f'{MODEL_NAME}/arch/lstm/num_layers' : cfg.MODEL.ARCH.LSTM.num_layers,
            f'{MODEL_NAME}/arch/lstm/hidden_size' : cfg.MODEL.ARCH.LSTM.hidden_size,
        }
    )

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