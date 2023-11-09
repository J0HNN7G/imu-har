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
import numpy as np

# training
from ml.config import cfg
from ml.models import ModelBuilder, OptimizerBuilder, LRScheduleBuilder
from ml.dataset import odgt2data


def main(cfg):
    """
    Main function for training a pedestrian detection model.

    Parameters:
    - cfg (object): A configuration object containing training parameters.
    
    Returns:
    None
    """
    # model
    train_odgt_fp = os.path.join(cfg.DATASET.path, cfg.DATASET.LIST.train)
    val_odgt_fp = os.path.join(cfg.DATASET.path, cfg.DATASET.LIST.val)

    train_X, train_y = odgt2data(train_odgt_fp, cfg.MODEL.INPUT.window_size, 
                                                cfg.TRAIN.DATA.overlap_size, 
                                                cfg.DATASET.num_classes)
    val_X, val_y = odgt2data(val_odgt_fp, cfg.MODEL.INPUT.window_size, 
                                          cfg.TRAIN.DATA.overlap_size,
                                          cfg.DATASET.num_classes)

        
    model = ModelBuilder.build_classifier(cfg.MODEL, '', cfg.DATASET.num_classes)
    optimizer = OptimizerBuilder.build_optimizer(cfg.TRAIN.OPTIM)
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.Precision(name='prec'),
        tf.keras.metrics.Recall(name='rec'),
        tf.keras.metrics.AUC(name='auc')
    ]

    lr_scheduler = LRScheduleBuilder.build_scheduler(cfg.TRAIN.LR)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=cfg.TRAIN.LEN.early_stop)

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.weight),
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)

    history_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.history), 
        separator=',', 
        append=False)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # train + evaluate
    with redirect_stdout(logging):
        model.fit(train_X, train_y, validation_data=(val_X, val_y), 
                epochs=cfg.TRAIN.LEN.num_epoch, 
                batch_size=cfg.TRAIN.DATA.batch_size,
                callbacks=[lr_callback, 
                            early_stop_callback,
                            checkpoint_callback,
                            history_callback])

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
    weight_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.weight)
    if os.path.exists(weight_fp):
        print(f'Training was done already! Final weights: {weight_fp}')
        exit()
    # make output directory
    if not os.path.isdir(cfg.TRAIN.path):
        os.makedirs(cfg.TRAIN.path)

    # make config 
    config_fp = os.path.join(cfg.TRAIN.path, cfg.TRAIN.FN.config)
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

    try:
        main(cfg)
    except Exception:
        logging.error(traceback.format_exc())
        # document everything
        with open(log_fp, 'r') as f:
            print(f.read())
        # for bash script
        sys.exit(1)
    sys.exit(0)


