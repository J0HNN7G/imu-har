from yacs.config import CfgNode as CN


def default_cfg():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern

    return cfg_backup.clone()

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
# task odgt filename
cfg.DATASET.odgt = ""
# task number
cfg.DATASET.task = -1
# model component
cfg.DATASET.component = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
# l2 weight regularization
cfg.MODEL.l2 = 0.0

# model input configuration
cfg.MODEL.INPUT = CN()
# type of sensors used for input data
cfg.MODEL.INPUT.sensor = "all"
# input data format
cfg.MODEL.INPUT.format = "normal"
# size of the window used for input data segmentation
cfg.MODEL.INPUT.window_size = 50
# include fourier transform of input data
cfg.MODEL.INPUT.fft = False
# include norm of accel and gyro data
cfg.MODEL.INPUT.norm = False


# model architecture specification
cfg.MODEL.ARCH = CN()

# CNN architecture specification
cfg.MODEL.ARCH.CNN = CN()
# residual connections for CNN
cfg.MODEL.ARCH.CNN.residual = False
# number of hidden layers for CNN
cfg.MODEL.ARCH.CNN.num_layers = 0
# size of hidden layers for CNN
cfg.MODEL.ARCH.CNN.hidden_size = 0
# layer-wise increase factor in hidden units
cfg.MODEL.ARCH.CNN.depth_scaling = 1.0
# dropout rate for regularization for CNN
cfg.MODEL.ARCH.CNN.dropout = 1.0
# kernel size for CNN
cfg.MODEL.ARCH.CNN.kernel_size = 0
# pooling size for CNN
cfg.MODEL.ARCH.CNN.pool_size = 0

# LSTM architecture specification
cfg.MODEL.ARCH.LSTM = CN()
# number of hidden layers for LSTM
cfg.MODEL.ARCH.LSTM.num_layers = 0
# size of hidden layers for LSTM
cfg.MODEL.ARCH.LSTM.hidden_size = 0
# dropout rate for regularization for LSTM
cfg.MODEL.ARCH.LSTM.dropout = 1.0

# MLP architecture specification
cfg.MODEL.ARCH.MLP = CN()
# number of hidden layers for MLP
cfg.MODEL.ARCH.MLP.num_layers = 0
# size of hidden layers for MLP
cfg.MODEL.ARCH.MLP.hidden_size = 0
# dropout rate for regularization for MLP
cfg.MODEL.ARCH.MLP.dropout = 1.0

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
cfg.TRAIN = CN()
# directory to save training checkpoints
cfg.TRAIN.path = ""

cfg.TRAIN.DATA = CN()
# window overlap size in sliding window analysis
cfg.TRAIN.DATA.overlap_size = 40
# number of samples per gradient update
cfg.TRAIN.DATA.batch_size = 128

cfg.TRAIN.LEN = CN()
# total number of training epochs
cfg.TRAIN.LEN.num_epoch = 10
# stop training if validation hasn't improved for this many epochs
cfg.TRAIN.LEN.early_stop = 3

cfg.TRAIN.OPTIM = CN()
# algorithm to use for optimization
cfg.TRAIN.OPTIM.optim = "adam"
# initial learning rate for training
cfg.TRAIN.OPTIM.lr = 0.001
# momentum factor for SGD, or beta1 for Adam optimizer
cfg.TRAIN.OPTIM.momentum = 0.9
# L2 penalty (regularization term) parameter
cfg.TRAIN.OPTIM.weight_decay = 0.0005

cfg.TRAIN.LR = CN()
# learning rate scheduling method
cfg.TRAIN.LR.schedule = 'step'
# number of epochs between learning rate reductions
cfg.TRAIN.LR.step_size = 100
# factor to reduce the learning rate
cfg.TRAIN.LR.gamma = 0.1

cfg.TRAIN.FN = CN()
# weights filename
cfg.TRAIN.FN.weight = "weights.hdf5"
# history filename
cfg.TRAIN.FN.history = "history.csv"
# config filename
cfg.TRAIN.FN.config = "config.yaml"
# log filename
cfg.TRAIN.FN.log = "log.txt"


# for resets
cfg_backup = cfg.clone()