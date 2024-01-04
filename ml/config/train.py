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
# total number of unique labels model output
cfg.DATASET.num_classes = 2
cfg.DATASET.LIST = CN()
# file path to the training data list
cfg.DATASET.LIST.train = ""
# file path to the validation data list
cfg.DATASET.LIST.val = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
cfg.MODEL = CN()
# model input configuration
cfg.MODEL.INPUT = CN()
# type of sensors used for input data
cfg.MODEL.INPUT.sensor = "all"
# input data format
cfg.MODEL.INPUT.format = "normal"
# size of the window used for input data segmentation
cfg.MODEL.INPUT.window_size = 25

# model architecture specification
cfg.MODEL.ARCH = CN()

# LSTM architecture specification
cfg.MODEL.ARCH.LSTM = CN()
# number of hidden layers for LSTM
cfg.MODEL.ARCH.LSTM.num_layers = 0
# size of hidden layers for LSTM
cfg.MODEL.ARCH.LSTM.hidden_size = 16

# MLP architecture specification
cfg.MODEL.ARCH.MLP = CN()
# number of hidden layers for MLP
cfg.MODEL.ARCH.MLP.num_layers = 0
# size of hidden layers for MLP
cfg.MODEL.ARCH.MLP.hidden_size = 32
# dropout rate for regularization for MLP
cfg.MODEL.ARCH.MLP.dropout = 0.2

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
cfg.TRAIN = CN()
# directory to save training checkpoints
cfg.TRAIN.path = ""

cfg.TRAIN.DATA = CN()
# window overlap size in sliding window analysis
cfg.TRAIN.DATA.overlap_size = 10
# number of samples per gradient update
cfg.TRAIN.DATA.batch_size = 128

cfg.TRAIN.LEN = CN()
# total number of training epochs
cfg.TRAIN.LEN.num_epoch = 10
# stop training if validation hasn't improved for this many epochs
cfg.TRAIN.LEN.early_stop = 2

cfg.TRAIN.OPTIM = CN()
# algorithm to use for optimization
cfg.TRAIN.OPTIM.optim = "sgd"
# initial learning rate for training
cfg.TRAIN.OPTIM.lr = 0.005
# momentum factor for SGD, or beta1 for Adam optimizer
cfg.TRAIN.OPTIM.momentum = 0.9
# L2 penalty (regularization term) parameter
cfg.TRAIN.OPTIM.weight_decay = 0.0005

cfg.TRAIN.LR = CN()
# learning rate scheduling method
cfg.TRAIN.LR.schedule = 'step'
# number of epochs between learning rate reductions
cfg.TRAIN.LR.step_size = 3
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
