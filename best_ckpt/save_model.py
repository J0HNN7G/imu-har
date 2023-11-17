import os
import numpy as np
import tensorflow as tf
import sys
import argparse
import glob

from ml.config import cfg
from ml.dataset import odgt2data
from ml.models import ModelBuilder, OptimizerBuilder, LRScheduleBuilder, TimingCallback

print("All packages imported!")

data = glob.glob('*')

for d in data:
    if d == 'save_model.py' or d == 'temp_model':
        continue
    print(d)
    config_file = d + '/config.yaml'
    model_file = d + '/model.tflite'
    
    cfg.merge_from_file(config_file)
    model = ModelBuilder.build_classifier(cfg.MODEL, '', cfg.DATASET.num_classes)
    if cfg.MODEL.ARCH.LSTM.num_layers > 0:
      	model.build((1,) + (15, 6))  
    else:
    	model.build((None,) + (15, 6))
    model.save('temp_model')
    model.summary()
    converter = tf.lite.TFLiteConverter.from_saved_model('temp_model')
    
    converter.target_spec.supported_ops = [
  		tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  		tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
		]
    converter.experimental_new_converter=True
    converter.allow_custom_ops=True
    
    tflite_model = converter.convert()
    
    with open(model_file, 'wb') as f:
    	f.write(tflite_model)
    	
