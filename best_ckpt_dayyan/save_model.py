import os
import numpy as np
import tensorflow as tf
import sys
import argparse
import glob

from ml.config import cfg
from ml.dataset import odgt2data
from ml.models import ModelBuilder

print("All packages imported!")

data = glob.glob('*')

for d in data:
    if d == 'save_model.py' or d == 'temp_model':
        continue
    print(d)
    config_file = d + '/config.yaml'
    model_file = d + '/model.tflite'
    weight_file = d + '/model.keras'
    
    cfg.merge_from_file(config_file)
    model = tf.keras.models.load_model(weight_file, compile=False, safe_mode = False)
    
    model.summary()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    converter.target_spec.supported_ops = [
  		tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  		tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
		]
    converter.experimental_new_converter=True
    converter.allow_custom_ops=True
    
    tflite_model = converter.convert()
    
    with open(model_file, 'wb') as f:
    	f.write(tflite_model)
    	
