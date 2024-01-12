"""Test classification model"""

# system libs
import os
import sys
import argparse

# logging
import logging
import traceback

# ml
import tensorflow as tf
from sklearn.metrics import classification_report

# training
from ml.config.train import default_cfg
from ml.config.train import cfg as cfg_train
from ml.config.test import cfg as cfg_test
from ml.models import TASK_MODEL_DICT, ModelBuilder, OptimizerBuilder, LRScheduleBuilder, BestModelCallback
from ml.dataset import odgt2test

ODGT_FN = 'full_t{}_pdiot-data.odgt'


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
    test_odgt_fp = os.path.join(cfg_test.DATASET.path, cfg_test.DATASET.odgt.format(cfg_test.DATASET.task))
    test_dict = odgt2test(test_odgt_fp, cfg_test.DATASET.task,
                                        cfg_test.TEST.subject, 
                                        cfg_test.MODEL.INPUT.window_size, 
                                        cfg_test.TEST.DATA.overlap_size)
    

    components = TASK_MODEL_DICT[cfg_test.DATASET.task]
    model_dict = {}
    for component in components:
        print(component)

        # filter data for relevant samples for component
        valid_idx = test_dict['train'][component] != -1
        train_X = test_dict['train']['X'][valid_idx]
        train_y = test_dict['train'][component][valid_idx]

        valid_idx = test_dict['val'][component] != -1
        val_X = test_dict['val']['X'][valid_idx]
        val_y = test_dict['val'][component][valid_idx]

        project_dp = os.path.join(cfg_test.DATASET , component)
        component_cfg_fp = os.path.join(project_dp, cfg_test.MODEL.CONFIG[component])
        cfg_train = default_cfg()
        cfg_train.merge_from_file(component_cfg_fp)

        model = ModelBuilder.build_classifier(cfg_train.MODEL, '', cfg_train.DATASET.num_classes)
        optimizer = OptimizerBuilder.build_optimizer(cfg_train.TRAIN.OPTIM)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        ]
        
        lr_scheduler = LRScheduleBuilder.build_scheduler(cfg_test.TRAIN.LR)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=cfg_train.TRAIN.LEN.early_stop)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(train_X, train_y, 
                  validation_data=(val_X, val_y),
                  epochs=cfg_train.TRAIN.LEN.num_epoch, 
                  batch_size=cfg_test.TEST.DATA.batch_size,
                  callbacks=[lr_callback, early_stop_callback])

        model_dict[component] = model

    model = ModelBuilder.build_hierarchical_classifier(cfg_test.DATASET.task, model_dict)

    pred = model(test_dict['val']['X'])

    report = classification_report(test_dict['val']['y'], pred)

    print(f'Task {cfg_test.DATASET.task} - Subject {cfg_test.TEST.subject}') 
    print(report)

    logging.info('Testing Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Tensorflow IMU HAR Model Testing"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="FILENAME",
        help="absolute path to config directory",
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
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    # get config file and check directory set up is correct
    args.config


    cfg_test.merge_from_file(args.config)
    cfg_test.merge_from_list(args.opts)
    cfg_test.DATASET.subject = args.subject
    cfg_test.DATASET.path = args.input
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
    logging.info("Loaded configuration file: {}".format(args.config))
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


