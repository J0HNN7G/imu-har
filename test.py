"""Test classification model"""

# system libs
import os
import sys
import argparse

# logging
import csv
import logging
import traceback

# ml
import tensorflow as tf
from sklearn.metrics import classification_report

# training
from har.config.train import default_cfg
from har.config.test import cfg as cfg_test
from har.models import TASK_MODEL_DICT, ModelBuilder, OptimizerBuilder, LRScheduleBuilder, BestModelCallback
from har.dataset import odgt2test

# task odgt filename format
DATA_ODGT_FORMAT = 'full_t{}_pdiot-data.odgt'
# config dir task format
TASK_CFG_DIR = 'task_{}'
CFG_EXT = '.yaml'

def main(cfg_test):
    """
    Main function for testing IMU HAR model.

    Parameters:
    - cfg_test (object): A configuration object containing testing parameters.
    
    Returns:
    None
    """
    logging.info('Testing Starting!')
    # model
    test_odgt_fp = os.path.join(cfg_test.DATASET.path, cfg_test.DATASET.odgt)
    test_dict = odgt2test(test_odgt_fp, cfg_test.DATASET.task,
                                        cfg_test.TEST.subject, 
                                        cfg_test.MODEL.INPUT.window_size, 
                                        cfg_test.TEST.DATA.overlap_size)
    

    components = TASK_MODEL_DICT[cfg_test.DATASET.task].keys()
    model_dict = {}
    for component in components:
        logging.info(component)

        # filter data for relevant samples for component
        valid_idx = test_dict['train'][component] != -1
        train_X = test_dict['train']['X'][valid_idx]
        train_y = test_dict['train'][component][valid_idx]

        valid_idx = test_dict['val'][component] != -1
        val_X = test_dict['val']['X'][valid_idx]
        val_y = test_dict['val'][component][valid_idx]

        component_cfg_fp = os.path.join(cfg_test.MODEL.path, cfg_test.MODEL.CONFIG[component])
        cfg_train = default_cfg()
        cfg_train.merge_from_file(component_cfg_fp)
        logging.info("config:\n{}".format(cfg_train))


        model = ModelBuilder.build_classifier(cfg_train.MODEL, '', cfg_train.DATASET.num_classes)
        optimizer = OptimizerBuilder.build_optimizer(cfg_train.TRAIN.OPTIM)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        ]
        
        lr_scheduler = LRScheduleBuilder.build_scheduler(cfg_train.TRAIN.LR)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=cfg_train.TRAIN.LEN.early_stop)
        best_model_callback = BestModelCallback(model)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(train_X, train_y, 
                  validation_data=(val_X, val_y),
                  epochs=cfg_train.TRAIN.LEN.num_epoch, 
                  batch_size=cfg_test.TEST.DATA.batch_size,
                  callbacks=[lr_callback, 
                             early_stop_callback, 
                             best_model_callback])

        model_dict[component] = model

    model = ModelBuilder.build_hierarchical_classifier(cfg_test.DATASET.task, model_dict)

    pred = model(test_dict['val']['X']).numpy()
    actual = test_dict['val']['y']

    # Write pred and true annotation to CSV file
    result_fp = os.path.join(cfg_test.TEST.path, cfg_test.TEST.FN.result)
    with open(result_fp, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pred', 'true'])
        writer.writerows(zip(pred, actual)) 

    report = classification_report(actual, pred)

    logging.info(report)
    logging.info('Testing Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tensorflow IMU HAR Model Testing"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="PATH",
        help="absolute path to config directory",
        type=str,
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        metavar="PATH",
        help="absolute path to directory with test list",
        type=str,
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        metavar="PATH",
        help="absolute path to output directory",
        type=str,
    )
    parser.add_argument(
        "-t", "--task",
        required=True,
        metavar="INT",
        help="which task to test on",
        type=int,
    )
    parser.add_argument(
        "-s", "--subject",
        required=True,
        metavar="INT",
        help="subject id left out of training for validation",
        type=int,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    
    if not args.task in TASK_MODEL_DICT.keys():
        raise ValueError(f'Invalid task: {args.task}')

    # get config file and check directory set up is correct
    if not os.path.isdir(args.config):
        raise ValueError(f'Config directory does not exist: {args.config}')

    test_cfg_fp = os.path.join(args.config, 'test', TASK_CFG_DIR.format(args.task) + CFG_EXT)
    if not os.path.isfile(test_cfg_fp):
        raise ValueError(f'Test config file does not exist: {test_cfg_fp}')
    cfg_test.merge_from_file(test_cfg_fp)
    cfg_test.merge_from_list(args.opts)

    if cfg_test.DATASET.task != args.task:
        raise ValueError(f'Task number in config does not match task number in args!: (cfg) {cfg_test.DATASET.task} != (arg) {args.task}')

    train_cfg_fp = os.path.join(args.config, 'train', TASK_CFG_DIR.format(args.task))
    for component in TASK_MODEL_DICT[cfg_test.DATASET.task].keys():
        cfg_fp = os.path.join(train_cfg_fp, cfg_test.MODEL.CONFIG[component])
        if not os.path.isfile(cfg_fp):
            raise ValueError(f'Train config file for component {component} does not exist!')
    cfg_test.MODEL.path = train_cfg_fp

    # input args
    if args.task in [1, 2, 3]:
        cfg_test.DATASET.odgt = DATA_ODGT_FORMAT.format(args.task)
    elif args.task == 4:
        cfg_test.DATASET.odgt = DATA_ODGT_FORMAT.format(3)
    cfg_test.DATASET.path = args.input
    cfg_test.TEST.subject = args.subject
    cfg_test.TEST.path = args.output

    # make output directory
    if not os.path.isdir(cfg_test.TEST.path):
        os.makedirs(cfg_test.TEST.path)

    # make config 
    config_fp = os.path.join(cfg_test.TEST.path, cfg_test.TEST.FN.config)
    if not os.path.exists(config_fp):
        with open(config_fp, 'w') as f:
            f.write(str(cfg_test))

    # setup logger
    log_fp = os.path.join(cfg_test.TEST.path, cfg_test.TEST.FN.log)
    if not os.path.exists(log_fp):
        open(log_fp, 'a').close()
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s %(levelname)s %(filename)s] %(message)s',
                        datefmt='%H:%M:%S',
                        handlers=[logging.FileHandler(log_fp)])
    # for redirecting stdout
    logging.write = lambda msg: logging.info(msg) if msg != '\n' else None
    # log details
    logging.info("Loaded configuration file: {}".format(test_cfg_fp))
    logging.info("Running with config:\n{}".format(cfg_test))
    logging.info("Outputting to: {}".format(cfg_test.TEST.path))

    try:
        main(cfg_test)
    except Exception:
        logging.error(traceback.format_exc())
        # document everything
        with open(log_fp, 'r') as f:
            print(f.read())
        # for bash script
        sys.exit(1)
    sys.exit(0)


