from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

cfg = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cfg.DATASET = CN()
# path to the root of the dataset directory
cfg.DATASET.path = ""
# task number
cfg.DATASET.task = 1

# -----------------------------------------------------------------------------
# HAR model
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
cfg.MODEL.CONFIG = CN()
# motion classifier configuration file
cfg.MODEL.CONFIG.motion = ""
# dynamic classifier configuration file
cfg.MODEL.CONFIG.dynamic = ""
# static classifier configuration file
cfg.MODEL.CONFIG.static = ""
# breathing classifier configuration file
cfg.MODEL.CONFIG.breath = ""
# respiratory classifier configuration file
cfg.MODEL.CONFIG.resp = ""

cfg.MODEL.INPUT = CN()
# window size
cfg.MODEL.INPUT.window_size = 50

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
cfg.TEST = CN()
# path to checkpoint
cfg.TEST.path = ""
# subject id
cfg.TEST.subject = -1

cfg.TEST.DATA = CN()
# overlap size
cfg.TEST.DATA.overlap_size = 0
# batch size
cfg.TEST.DATA.batch_size = 128
