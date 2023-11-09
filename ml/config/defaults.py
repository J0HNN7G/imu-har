from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
# path to the root of the dataset directory
_C.DATASET.path = ""
# total number of unique labels model output
_C.DATASET.num_classes = 2
_C.DATASET.LIST = CN()
# file path to the training data list
_C.DATASET.LIST.train = ""
# file path to the validation data list
_C.DATASET.LIST.val = ""
# file path to the test data list
_C.DATASET.LIST.test = ""

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# model input configuration
_C.MODEL.INPUT = CN()
# type of sensors used for input data
_C.MODEL.INPUT.sensor = "all"
# input data format
_C.MODEL.INPUT.format = "normal"
# size of the window used for input data segmentation
_C.MODEL.INPUT.window_size = 25

# model architecture specification
_C.MODEL.ARCH = CN()
# type of model architecture (log, mlp, lstm)
_C.MODEL.ARCH.type = "mlp"
# size of hidden layers for MLP or LSTM
_C.MODEL.ARCH.hidden_size = 256
# number of hidden layers for MLP or LSTM
_C.MODEL.ARCH.num_layers = 2
# dropout rate for regularization for MLP
_C.MODEL.ARCH.dropout = 0.5
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# directory to save training checkpoints
_C.TRAIN.path = ""

_C.TRAIN.DATA = CN()
# window overlap size in sliding window analysis
_C.TRAIN.DATA.overlap_size = 10
# number of samples per gradient update
_C.TRAIN.DATA.batch_size = 32

_C.TRAIN.LEN = CN()
# total number of training epochs
_C.TRAIN.LEN.num_epoch = 10
# stop training if validation hasn't improved for this many epochs
_C.TRAIN.LEN.early_stop = 2

_C.TRAIN.OPTIM = CN()
# algorithm to use for optimization
_C.TRAIN.OPTIM.optim = "sgd"
# initial learning rate for training
_C.TRAIN.OPTIM.lr = 0.005
# momentum factor for SGD, or beta1 for Adam optimizer
_C.TRAIN.OPTIM.momentum = 0.9
# L2 penalty (regularization term) parameter
_C.TRAIN.OPTIM.weight_decay = 0.0005

_C.TRAIN.LR = CN()
# learning rate scheduling method
_C.TRAIN.LR.schedule = 'step'
# number of epochs between learning rate reductions
_C.TRAIN.LR.step_size = 3
# factor to reduce the learning rate
_C.TRAIN.LR.gamma = 0.1

_C.TRAIN.FN = CN()
# weights filename
_C.TRAIN.FN.weight = "weights.hdf5"
# history filename
_C.TRAIN.FN.history = "history.csv"
# config filename
_C.TRAIN.FN.config = "config.yaml"
# log filename
_C.TRAIN.FN.log = "log.txt"
