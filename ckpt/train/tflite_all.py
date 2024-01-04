import os
import glob
import tensorflow as tf

from ml.config.train import cfg
from ml.models import ModelBuilder

def main():
    data = glob.glob('*')

    for d in data:
        if d == 'tflite_all.py':
            continue

        config_file = os.path.join(d, 'config.yaml')
        cfg.merge_from_file(config_file)
        weight_file = os.path.join(d, cfg.TRAIN.FN.weight)

        model_file = os.path.join(d, 'model.tflite')

        model = ModelBuilder.build_classifier(cfg.MODEL, weight_file, cfg.DATASET.num_classes)
        model.summary()

        # directly convert the Keras model to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.experimental_new_converter = True
        converter.allow_custom_ops = True
        
        tflite_model = converter.convert()
        
        with open(model_file, 'wb') as f:
            f.write(tflite_model)

if __name__ == "__main__":
    main()
