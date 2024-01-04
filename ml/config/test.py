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
# subject id
cfg.DATASET.subject = -1

# -----------------------------------------------------------------------------
# HAR model
# -----------------------------------------------------------------------------
cfg.HAR = CN()
# motion classifier configuration file
cfg.HAR.motion = ""
# dynamic classifier configuration file
cfg.HAR.dynamic = ""
# static classifier configuration file
cfg.HAR.static = ""
# breathing classifier configuration file
cfg.HAR.breathing = ""