# pdiot-ml
PDIoT 2023-2024 A1 ML model

## Structure

![Diagram](docs/diagram.jpg)

##  Installation
```
conda create -n <env_name>
conda activate <env_name>

# tensorflow (change to desired version)
conda install tensorflow==2.11.0 

# data processing
conda install pandas numpy scikit-learn

# config, logging and metrics (not on conda)
pip install wandb yacs --upgrade-strategy only-if-needed
```

## Training
1. Format activity classification dataset. Example scripts found in `data`.

Custom dataset files are expected to be formatted as follows:
```
{ 'filepath': <data_csv_filepath>, 'annotation': <label>}
...
```


2. Make configuration YAML file. 

Example `log-motion.yaml`:
```
DATASET:
  path: ""
  num_classes: 2
  LIST:
    train: "train_motion_pdiot-data.odgt"
    val: "val_motion_pdiot-data.odgt"
    test: "test_motion_pdiot-data.odgt"


MODEL:
  INPUT:
    sensor: "all"
    format: "normal"
    window_size: 25
  ARCH:
    type: "log"


TRAIN:
  path: ""
  seed: -1
  DATA:
    overlap_size: 10
    batch_size: 64
  LEN:
    num_epoch: 100
    start_epoch: 0
    early_stop: 10
  OPTIM:
    optim: "sgd"
    lr: 0.001
    momentum: 0.9
    weight_decay: 0.0005
  LR:
    schedule: 'step'
    step_size: 40
    gamma: 0.1
```

3. Run the training
```
python train.py -c <config_filepath> -i <train_val_odgt_dirpath> -o <checkpoint_dirpath>
```

4. Results are stored at the checkpoint directory. By default your directory will be set up as follows:
```
<DIR>
├── weights.hdf5            # checkpoint with best validation accuracy
├── history.csv             # training and validation metrics history
├── config.yaml             # configuration file (updated with train.py arguments)
└── log.txt                 # model training logs
```

## Evaluation

```
python test.py -c <config_filepath> -i <test_odgt_dirpath>
```