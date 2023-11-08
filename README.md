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

## Activity Classification Training
1. Format activity classification dataset. Example scripts found in `data`.

Custom dataset files are expected to be formatted as follows:
```
{ 'filepath': <data_csv_filepath>, 'annotation': <label>}
...
```


2. Make configuration YAML file. 

Example `mlp-dynamic.yaml`:
```
DATASET:
  path: ""
  LIST:
    train: "train_dynamic.odgt"
    val: "val_dynamic.odgt"

MODEL:
  arch: "mlp"
  sensor: "all"
  input: "normal"

TRAIN:
  path: ""
  seed: -1
  DATA:
    batch_size: 32
    num_workers: 4
    disp_iter: 20
  LEN:
    num_epoch: 10
    start_epoch: 0
    early_stop: 2
  OPTIM:
    optim: "sgd"
    lr: 0.005
    momentum: 0.9
    weight_decay: 0.0005
  LR:
    schedule: 'step'
    step_size: 3
    gamma: 0.1
```

3. Run the training
```
python train.py -c <config_filepath> -i <train_val_odgt_dirpath> -o <checkpoint_dirpath>
```

4. Results are stored at the checkpoint directory. By default your directory will be set up as follows:
```
<DIR>
├── weights_best.tflite        # checkpoint with best validation mAP
├── weights_epoch_<n>.tflite   # last checkpoint whilst running
├── weights_final.tflite       # final checkpoint if run finished
├── history.tsv             # training and validation metrics history
├── config.yaml             # configuration file (updated with train.py arguments)
└── log.txt                 # model training logs
```

## Evaluation

```
python test.py -c <config_filepath> -i <test_odgt_dirpath>
```