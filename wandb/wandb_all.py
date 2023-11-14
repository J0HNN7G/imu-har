"""Log all checkpoints to wandb"""
# args
import os
import argparse

from ml.config import cfg

# misc
import logging
from importlib import reload
from tqdm import tqdm

# main function
from wandb_ckpt import main


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wandb Logging for PDIoT Classification Training"
    )
    parser.add_argument(
        "-c", "--ckpt",
        default="ckpt",
        metavar="PATH",
        help="path to checkpoint directories",
        type=str,
    )
    args = parser.parse_args()
    assert os.path.isdir(args.ckpt)
    ckpt_dns = os.listdir(args.ckpt)
    for ckpt_dn in tqdm(ckpt_dns):
        try:
            ckpt_path = os.path.join(args.ckpt, ckpt_dn)
            cfg_fp = os.path.join(ckpt_path, 'config.yaml')
            assert os.path.exists(cfg_fp), 'config.yaml does not exist!'
            cfg.merge_from_file(cfg_fp)
            cfg.TRAIN.path = ckpt_path

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

            logging.shutdown()
            reload(logging)
        except Exception as e:
            print(f"Checkpointing for {ckpt_dn} failed:")
            print(e)