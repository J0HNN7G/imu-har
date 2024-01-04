# IMU Activity Recognition for Android

IMU Activity Data: https://github.com/specknet/pdiot-data

## Structure

![Diagram](docs/system_architecture.png)

![Diagram](docs/har_model.png)


#### TASK 1: CLASSIFICATION OF GENERAL HUMAN ACTIVITIES
*Only use data collected during normal breathing!*
```
class 0: sitting/standing
class 1: lying down on your left side
class 2: lying down on your right side
class 3: lying down on your back
class 4: lying down on your stomach
class 5: walking
class 6: running/jogging
class 7: descending stairs
class 8: ascending stairs
class 9: shuffle walking
class 10: miscellaneous movements 
```

#### TASK 2: CLASSIFICATION OF STATIONARY ACTIVITIES WITH RESPIRATORY SYMPTOMS
*Do not include any data files/recordings that contain laughing, singing, eating, or talking!*
```
class 0: sitting/standing + breathing normally
class 1: lying down on your left side + breathing normally
class 2: lying down on your right side + breathing normally
class 3: lying down on your back + breathing normally
class 4: lying down on your stomach + breathing normally 
class 5: sitting/standing + coughing
class 6: lying down on your left side + coughing
class 7: lying down on your right side + coughing
class 8: lying down on your back + coughing
class 9: lying down on your stomach + coughing
class 10: sitting/standing + hyperventilating
class 11: lying down on your left side + hyperventilating
class 12: lying down on your right side + hyperventilating
class 13: lying down on your back + hyperventilating
class 14: lying down on your stomach + hyperventilating
```

#### TASK 3: CLASSIFICATION OF STATIONARY ACTIVITIES WITH RESPIRATORY SYMPTOMS AND OTHER BEHAVIORS
```
class 0: sitting/standing + breathing normally
class 1: lying down on your left side + breathing normally
class 2: lying down on your right side + breathing normally
class 3: lying down on your back + breathing normally
class 4: lying down on your stomach + breathing normally 
class 5: sitting/standing + coughing
class 6: lying down on your left side + coughing
class 7: lying down on your right side + coughing
class 8: lying down on your back + coughing
class 9: lying down on your stomach + coughing
class 10: sitting/standing + hyperventilating
class 11: lying down on your left side + hyperventilating
class 12: lying down on your right side + hyperventilating
class 13: lying down on your back + hyperventilating
class 14: lying down on your stomach + hyperventilating
class 15: sitting/standing + other
class 16: lying down on your left side + other
class 17: lying down on your right side + other
class 18: lying down on your back + other
class 19: lying down on your stomach + other
**other refers to singing/talking/laughing/eating
```

![Diagram](docs/task_models.png)


##  Installation
Using Python 3.10.13:
```
conda create -n <env_name>
conda activate <env_name>

# tensorflow (change to desired version)
conda install tensorflow==2.11.0 

# data processing
conda install numpy pandas scikit-learn tqdm

# config, logging (not on conda)
pip install yacs wandb --upgrade-strategy only-if-needed

# for fully functionality
cd <pdiot-ml_directory>
python setup.py install
```

## Training
1. Format activity classification dataset. Example scripts found in `data`.

Custom dataset files are expected to be formatted as follows:
```
{ 'filepath': <data_csv_filepath>, 'annotation': <label>}
...
```


2. Make configuration YAML file. 

Example `mlp-motion.yaml`:
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
    format: "summary"
    window_size: 15
  ARCH:
    LSTM:
      num_layers: 0
    MLP:
      num_layers: 3
      hidden_size: 32
      dropout: 0.2

TRAIN:
  path: ""
  DATA:
    overlap_size: 5
    batch_size: 128
  LEN:
    num_epoch: 100
    early_stop: 10
  OPTIM:
    optim: "adam"
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

## Leave-One-Out (LOO) Testing

1. Format activity classification dataset for your test task. Example scripts found in `data`.

Custom dataset files are expected to be formatted as follows:
```
{ 'filepath': <data_csv_filepath>, 'annotation': <class>, 'labels': [<motion_label>, 
                                                                     <dynamic_label>,
                                                                     <static_label>,
                                                                     <breath_label>]}
...
```
The number of labels will vary based on the model components required for the task ([Structure](#structure)).

2. Make configuration YAML file. 

Example `task_1.yaml`:
```
DATASET:
  path: ""
  task: 1

HAR:
  MOTION:
    config: "config/train/motion.yaml"
  DYNAMIC:
    config: "config/train/dynamic.yaml"
  STATIC:
    config: "config/train/static.yaml"
  BREATH:
    config: "config/train/breath.yaml"
```

3. Run the testing
```
python test.py -c <config_filepath> -i <test_odgt_dirpath> -o <checkpoint_dirpath>
```

4. Results are stored at the checkpoint directory. By default your directory will be set up as follows:
```
<DIR>
├── confusion.txt           # LOO test confusion matrix 
├── config.yaml             # configuration file (updated with test.py arguments)
└── log.txt                 # LOO test logs
```

