# code altered from MIT semantic segmentation repository
# https://github.com/CSAILVision/semantic-segmentation-pytorch

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# absolute path to dataset folder
_C.DATASET.path = ""

_C.DATASET.LIST = CN()
# training list ODGT
_C.DATASET.LIST.train = ""
# validation list ODGT
_C.DATASET.LIST.val = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# name of model architecture
_C.MODEL.arch = "fasterrcnn_resnet50_fpn"

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

# absolute path to training checkpoint folder
_C.TRAIN.path = ""
# manual seed
_C.TRAIN.seed = -1

_C.TRAIN.DATA = CN()
# training batch size
_C.TRAIN.DATA.batch_size = 32
# number of data loading workers
_C.TRAIN.DATA.num_workers = 4
# frequency to display
_C.TRAIN.DATA.disp_iter = 20

_C.TRAIN.LEN = CN()
# epoch to start training. useful if continue from a checkpoint 
_C.TRAIN.LEN.start_epoch = 0
# epochs to train for
_C.TRAIN.LEN.num_epoch = 10
# early stopping if no improvements in this many epochs
_C.TRAIN.LEN.early_stop = 2

_C.TRAIN.OPTIM = CN()
# optimizer
_C.TRAIN.OPTIM.optim = "sgd"
# initial learning rate
_C.TRAIN.OPTIM.lr = 0.005
# momentum for sgd, beta1 for adam
_C.TRAIN.OPTIM.momentum = 0.9
# weights regularizer
_C.TRAIN.OPTIM.weight_decay = 0.0005

_C.TRAIN.LR = CN()
# lr schedule regime
_C.TRAIN.LR.schedule = 'step'
# period of learning rate decay
_C.TRAIN.LR.step_size = 3
# multiplicative factor of learning rate decay
_C.TRAIN.LR.gamma = 0.1

_C.TRAIN.FN = CN()
# config filename
_C.TRAIN.FN.cfg = "config.yaml"
# history filename
_C.TRAIN.FN.hist = "history.tsv"
# log filename
_C.TRAIN.FN.log = "log.txt"
# visual filename
_C.TRAIN.FN.vis = "visual.png"