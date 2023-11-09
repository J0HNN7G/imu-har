"""Train classification model"""

# system libs
import os
import sys
import argparse

# logging
import logging
import traceback
from contextlib import redirect_stdout

# ml
import tensorflow as tf

# seeding randomness
import random
import numpy as np

# training
from ml.config import cfg
from ml.models import ModelBuilder, OptimizerBuilder, LRScheduleBuilder
from ml.dataset import odgt2data

# constants
TRAIN_NAME = 'train'
VAL_NAME = 'val'
TRAIN_EPOCH_NAME = f'{TRAIN_NAME}/epoch'
TRAIN_LOSS_NAME = f'{TRAIN_NAME}/loss'
TRAIN_LR_NAME = f'{TRAIN_NAME}/lr'
TRAIN_TIME = f'{TRAIN_NAME}/time'
VAL_ACC_NAME = f'{VAL_NAME}/acc'
VAL_PR_NAME = f'{VAL_NAME}/pr'
VAL_RE_NAME = f'{VAL_NAME}/re'
VAL_F1_NAME = f'{VAL_NAME}/f1'

BEST_EPOCH_NAME = 'best_' + TRAIN_EPOCH_NAME
BEST_F1_NAME = 'best_' + VAL_F1_NAME

WEIGHT_FN = 'weights.hdf5'

TRAIN_HEADERS = [TRAIN_EPOCH_NAME, TRAIN_TIME, TRAIN_LR_NAME]
VAL_HEADERS = [VAL_ACC_NAME, VAL_PR_NAME, VAL_RE_NAME, VAL_F1_NAME]
SEP = '\t'


def main(cfg, device):
    """
    Main function for training a pedestrian detection model.

    Parameters:
    - cfg (object): A configuration object containing training parameters.
    - device (object): Device (CPU or GPU) to perform training on.

    Returns:
    None
    """
    # model
    model = ModelBuilder.build_detector(args=cfg.MODEL, weights=cfg.TRAIN.weights)
    model.to(device)

    # dataset
    train_path = os.path.join(cfg.DATASET.path,cfg.DATASET.LIST.train)
    train_dataset = PedestrianDetectionDataset(train_path, transforms=get_transform(train=True))
    val_path = os.path.join(cfg.DATASET.path,cfg.DATASET.LIST.val)
    val_dataset = PedestrianDetectionDataset(val_path, transforms=get_transform(train=False))

    # dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.DATA.batch_size, shuffle=True, num_workers=cfg.TRAIN.DATA.num_workers,
        collate_fn=collate_fn)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.TRAIN.DATA.batch_size, shuffle=False, num_workers=cfg.TRAIN.DATA.num_workers,
        collate_fn=collate_fn)

    # optimizer
    optimizer = OptimizerBuilder.build_optimizer(cfg.TRAIN.OPTIM, model)
    lr_scheduler = LRScheduleBuilder.build_scheduler(cfg.TRAIN.LR, optimizer)
    
    # track metrics
    history = {
        BEST_EPOCH_NAME: 0,
        BEST_AP_NAME: -1
    }
    # catch up
    if cfg.TRAIN.LEN.start_epoch > 0:
        # catch up on learning rate
        for i in range(0, cfg.TRAIN.LEN.start_epoch):
            lr_scheduler.step()
        # catch up on history
        setup_previous_history(history, cfg)

    # training
    for epoch in range(cfg.TRAIN.LEN.start_epoch, cfg.TRAIN.LEN.num_epoch):
        # early stopping
        if cfg.TRAIN.LEN.early_stop < epoch - history[BEST_EPOCH_NAME]:
            logging.info(f'Early stop! No improvement in validation set for {cfg.TRAIN.LEN.early_stop} epochs')
            # rename checkpoint to final weights
            curr_weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(epoch-1))
            final_weights_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FINAL_FN)
            os.rename(curr_weights_fp, final_weights_fp)
            break
        else:
            logging.info(f'Starting Epoch {epoch}')
            
        # train + evaluate
        with redirect_stdout(logging):
            train_log = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=cfg.TRAIN.DATA.disp_iter)
            eval_log = evaluate(model, val_data_loader, device=device)
        
        # make loss headers in first epoch of run 
        if epoch == cfg.TRAIN.LEN.start_epoch:
            loss_names, loss_headers = setup_loss_details(train_log, cfg)

        # update history
        history[TRAIN_EPOCH_NAME] = epoch
        history[TRAIN_TIME] = train_log.meters['time'].value
        history[TRAIN_LR_NAME] = lr_scheduler.get_last_lr()[-1]

        # add training metrics
        for i in range(len(loss_names)):
            loss_name = loss_names[i]
            loss_header = loss_headers[i]
            loss = train_log.meters[loss_name].total
            history[loss_header] = loss 

        # add validation metrics
        history[VAL_AP_NAME] = eval_log.coco_eval['bbox'].stats[0]
        history[VAL_AR_NAME] = eval_log.coco_eval['bbox'].stats[6]
        if history[BEST_AP_NAME] < history[VAL_AP_NAME]:
            history[BEST_EPOCH_NAME] = epoch
            history[BEST_AP_NAME] = history[VAL_AP_NAME]

        # save model + history + visual
        checkpoint(model, history, cfg, epoch)
        visual_evaluate(model, val_data_loader, cfg, device=device)

        # update learning rate
        lr_scheduler.step()

    logging.info('Training Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Pedestrian Detection Finetuning"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="FILENAME",
        help="absolute path to config file",
        type=str,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="PATH",
        help="absolute path to directory with train and validation lists",
        type=str,
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="PATH",
        help="absolute path to checkpoint directory",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.DATASET.path = args.input
    cfg.TRAIN.path = args.output

    # check if already done
    weight_fp = os.path.join(cfg.TRAIN.path, WEIGHT_FN)
    if os.path.exists(final_weight_fp):
        print(f'Training was done already! Final weights: {final_weight_fp}')
        exit()

    # make output directory
    if not os.path.isdir(cfg.TRAIN.path):
        os.makedirs(cfg.TRAIN.path)
    elif cfg.TRAIN.LEN.start_epoch == 0:
        # starting from scratch
        for f in os.listdir(cfg.TRAIN.path):
            os.remove(os.path.join(cfg.TRAIN.path,f))

    # set/save random seed
    if cfg.TRAIN.seed < 0: 
        seed = torch.seed()
        cfg.TRAIN.seed = int(seed)
    else:
        cfg.TRAIN.SEED = int(cfg.TRAIN.seed)
        torch.manual_seed(cfg.TRAIN.seed)
    random.seed(cfg.TRAIN.seed) 
    # seed must be between 0 and 2**32 -1
    np.random.seed(cfg.TRAIN.seed % (2**32 - 1))

    # make config 
    config_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.cfg)
    if not os.path.exists(config_fp):
        with open(config_fp, 'w') as f:
            f.write(str(cfg))

    # setup logger
    log_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.log)
    if not os.path.exists(log_fp):
        open(log_fp, 'a').close()
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_fp)])
    # for redirecting stdout
    logging.write = lambda msg: logging.info(msg) if msg != '\n' else None
    # log details
    logging.info("Loaded configuration file: {}".format(args.config))
    logging.info("Running with config:\n{}".format(cfg))
    logging.info("Outputting to: {}".format(cfg.TRAIN.path))

    # start from checkpoint
    cfg.TRAIN.history = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.hist)
    cfg.TRAIN.weights = ""
    if cfg.TRAIN.LEN.start_epoch > 0:
        cfg.TRAIN.weights = os.path.join(cfg.TRAIN.path, WEIGHT_FN.format(cfg.TRAIN.LEN.start_epoch-1))
        assert os.path.exists(cfg.TRAIN.weights), "weight checkpoint does not exist!"
        assert os.path.exists(cfg.TRAIN.history), "history checkpoint does not exist!"

    # make visual filepath
    cfg.TRAIN.visual = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.vis)

    # train on the GPU or on the CPU, if a GPU is not available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        logging.info('No GPU found! Training on CPU')
        device = torch.device('cpu')

    try:
        main(cfg, device)
    except Exception:
        logging.error(traceback.format_exc())
        # document everything
        with open(log_fp, 'r') as f:
            print(f.read())
        # for bash script
        sys.exit(1)
    sys.exit(0)


