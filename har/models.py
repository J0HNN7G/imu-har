"""Build classficiation model components"""

# architectures
import tensorflow as tf
import numpy as np

# timing
from timeit import default_timer as timer

# Sampling rate
SAMPLING_RATE = 25
SAMPLING_TIME = 1 / SAMPLING_RATE

# number of sensors
NUM_SENSORS = 6

# task details
TASK_MODEL_DICT = {
    1: ['motion', 'dynamic', 'static'],
    2: ['static', 'resp'],
    3: ['static', 'breath'],
    4: ['static', 'breath']
}
STATIC_MAX_LEN = 5


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Keep track of time per epoch.
    """
    def on_epoch_begin(self, epoch, logs):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs):
        logs['time'] = timer()-self.starttime


class BestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super(BestModelCallback, self).__init__()
        self.best_model_weights = None
        self.best_val_acc = -1

    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get('val_acc')
        if val_acc >= self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_model_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        self.model.set_weights(self.best_model_weights)


###################
# CUSTOM FEATURES #
###################

def add_norm(imu_data):
    features = [imu_data]
    for i in range(0, imu_data.shape[-1], 3):
        norm = tf.norm(imu_data[:, :, i:i+3], axis=-1)
        features.append(tf.expand_dims(norm, axis=-1))
    
    features = tf.concat(features, axis=-1)

    return features


def add_fft(imu_data):
    # only on innermost axis
    fft_real = tf.transpose(imu_data, perm=[0, 2, 1])
    fft_real = tf.signal.rfft(fft_real)
    fft_real = tf.transpose(fft_real, perm=[0, 2, 1])

    fft_feats = tf.abs(fft_real)

    feats = tf.concat((fft_feats, imu_data), axis=1)
    
    return feats

class InputTransformBuilder:
    """
    Builder class for transforming input to features for classfication models.

    Methods:
    - build_transform(args): Build a transform builder for classifier model.
    """

    @staticmethod
    def build_transform(args, needs_window=False):
        """
        Build a transform for the input.

        Args:
        - args: input configuration

        Returns:
        - a transform function
        """
        if needs_window and (args.format != 'window'):
            raise Exception('CNN or LSTM require window format!')
        elif (not needs_window) and (args.format == 'window'):
            raise Exception('only CNN or LSTM can use window format!')

        transform = tf.keras.Sequential()

        # what sensors to use
        if args.sensor == 'all':
            pass
        elif args.sensor == 'accel':
            sensors = tf.keras.layers.Lambda(lambda x: x[..., 0:3])
            transform.add(sensors)
        elif args.sensor == 'gyro':
            sensors = tf.keras.layers.Lambda(lambda x: x[..., 3:6])
            transform.add(sensors)
        else:
            raise Exception('Sensor undefined!')
        
        # what features to add to data
        if args.norm:
            norm = tf.keras.layers.Lambda(lambda x: add_norm(x))
            transform.add(norm)
        if args.fft:
            fft = tf.keras.layers.Lambda(lambda x: add_fft(x))
            transform.add(fft)

        # how to process that data
        if (args.format == 'window'):
            input_shape = (None, None, NUM_SENSORS)
        elif (args.format == 'normal'):
            input_shape = (None, args.window_size, NUM_SENSORS)
            format = tf.keras.layers.Flatten()
            transform.add(format)
        elif args.format == 'summary':
            input_shape = (None, args.window_size, NUM_SENSORS)
            format = tf.keras.layers.Lambda(lambda x: tf.concat([tf.math.reduce_mean(x, axis=1),
                                                                 tf.math.reduce_std(x, axis=1)], 
                                                                 axis=1))
            transform.add(format)
            transform.add(tf.keras.layers.Flatten())
        else:
            raise Exception('Transform undefined!')
        
        return transform, input_shape


class OptimizerBuilder:
    """
    Builder class for creating optimizers.

    Methods:
    - build_optimizer(args, model): Build an optimizer for a given model.
    """
    @staticmethod
    def build_optimizer(args):
        """
        Build an optimizer for a given model.

        Args:
        - args: Optimizer configuration.

        Returns:
        - optimizer: The optimizer.
        """
        if args.optim == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr,
                                                momentum=args.momentum, 
                                                weight_decay=args.weight_decay)
        elif args.optim == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr,
                                                 beta_1=args.momentum,
                                                 weight_decay=args.weight_decay)
        else:
            raise Exception('Optimizer undefined!')
        return optimizer


class LRScheduleBuilder:
    """
    Builder class for creating learning rate schedulers.

    Methods:
    - build_scheduler(args, optimizer): Build a learning rate scheduler for a given optimizer.
    """
    @staticmethod
    def build_scheduler(args):
        """
        Build a learning rate scheduler for a given optimizer.

        Args:
        - args: Learning rate scheduler configuration.

        Returns:
        - lr_scheduler: The learning rate scheduler.
        """
        if args.schedule == 'constant':
            def lr_scheduler(epoch, lr):
                return lr
        elif args.schedule == 'step':
            def lr_scheduler(epoch, lr):
                if epoch > 0 and epoch % args.step_size == 0:
                    return lr * args.gamma
                else:
                    return lr
        else:
            raise Exception('LR Scheduler undefined!')
        return lr_scheduler


class ModelBuilder:
    """
    Builder class for creating classfication models.

    Methods:
    - build_classifier(args, weights, num_classes): Build a classifier model.
    """
    @staticmethod
    def build_classifier(args, weights, num_classes):
        """
        Build a classifier model.

        Args:
        - args: Model architecture and other arguments.
        - weights (str): Path to pre-trained weights file.
        - num_classes (int): Number of classes.

        Returns:
        - classifier: A classifier model.
        """
        
        classifier = tf.keras.Sequential()

        # format input
        needs_window = (args.ARCH.LSTM.num_layers > 0) or (args.ARCH.CNN.num_layers > 0)
        transforms, input_size = InputTransformBuilder.build_transform(args.INPUT, needs_window)
        if len(transforms.layers) > 0:
            classifier.add(transforms)

        # CNN
        for i in range(args.ARCH.CNN.num_layers):
            hidden_size = int(args.ARCH.CNN.hidden_size * (args.ARCH.CNN.depth_scaling ** i))
            if i < args.ARCH.CNN.num_layers - 1:
                if args.ARCH.CNN.residual:
                    classifier.add(ResidualBlock(filters=hidden_size,
                                                 kernel_size=args.ARCH.CNN.kernel_size,
                                                 strides=args.ARCH.CNN.pool_size,
                                                 l2_reg=args.l2))
                else:
                    classifier.add(tf.keras.layers.Conv1D(filters=hidden_size,
                                                        kernel_size=args.ARCH.CNN.kernel_size,
                                                        activation='relu',
                                                        kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
                    
                #if args.ARCH.CNN.pool_size > 1:
                #    classifier.add(tf.keras.layers.MaxPooling1D(args.ARCH.CNN.pool_size))
            else:
                if args.ARCH.CNN.residual:
                    classifier.add(ResidualBlock(filters=hidden_size,
                                                 kernel_size=args.ARCH.CNN.kernel_size,
                                                 strides=args.ARCH.CNN.pool_size,
                                                 l2_reg=args.l2))
                else:
                    classifier.add(tf.keras.layers.Conv1D(filters=hidden_size,
                                                        kernel_size=args.ARCH.CNN.kernel_size,
                                                        activation='relu',
                                                        kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
                    
                if (args.ARCH.CNN.dropout > 0) and (args.ARCH.CNN.dropout != 1.0):
                    classifier.add(tf.keras.layers.Dropout(args.ARCH.CNN.dropout))
                if args.ARCH.CNN.pool_size > 1:
                    classifier.add(tf.keras.layers.MaxPooling1D(args.ARCH.CNN.pool_size))
                if args.ARCH.LSTM.num_layers == 0:
                    classifier.add(tf.keras.layers.GlobalAveragePooling1D())

        # LSTM
        for i in range(args.ARCH.LSTM.num_layers):
            if i < args.ARCH.LSTM.num_layers - 1:
                classifier.add(tf.keras.layers.LSTM(args.ARCH.LSTM.hidden_size, 
                                                    return_sequences=True,
                                                    kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
            else:
                classifier.add(tf.keras.layers.LSTM(args.ARCH.LSTM.hidden_size, 
                                                    return_sequences=False,
                                                    kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
                if (args.ARCH.LSTM.dropout > 0) and (args.ARCH.LSTM.dropout != 1.0):
                    classifier.add(tf.keras.layers.Dropout(args.ARCH.LSTM.dropout))

        # MLP
        for _ in range(args.ARCH.MLP.num_layers):
            classifier.add(tf.keras.layers.Dense(units=args.ARCH.MLP.hidden_size, 
                                                 activation='relu',
                                                 kernel_regularizer=tf.keras.regularizers.l2(args.l2)))
        if (args.ARCH.MLP.dropout > 0) and (args.ARCH.MLP.dropout != 1.0):
            classifier.add(tf.keras.layers.Dropout(args.ARCH.MLP.dropout))

        # Final Activation
        classifier.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

        pretrained = (len(weights) > 0)
        if pretrained:
            classifier.build(input_size)
            classifier.load_weights(weights)

        return classifier

    @staticmethod
    def build_hierarchical_classifier(task, component_dict):
        """
        Build a non-trainable hierarchical classifier model.

        Args:
        - args: Model architecture and other arguments.
        - component_dict (dict): A dictionary of component models.

        Returns:
        - classifier: A classifier model (not trainable).
        """
        if set(component_dict.keys()) != set(TASK_MODEL_DICT[task]):
            raise Exception(f'Invalid component dictionary {component_dict.keys()} for task {task}')

        if task == 1:
            def classify(x):
                motion_output = component_dict['motion'](x)
                motion_pred = tf.math.argmax(motion_output, axis=1)

                dynamic_indices = tf.where(motion_pred == 1)[:, 0]
                static_indices = tf.where(motion_pred == 0)[:, 0]

                dynamic_output = component_dict['dynamic'](tf.gather(x, dynamic_indices))
                dynamic_pred =  STATIC_MAX_LEN + tf.math.argmax(dynamic_output, axis=1)

                static_output = component_dict['static'](tf.gather(x, static_indices))
                static_pred = tf.math.argmax(static_output, axis=1)

                batch_size = tf.shape(x)[0]
                pred = tf.fill((batch_size,), -1.0)
                pred = tf.cast(pred, tf.int64)

                pred = tf.tensor_scatter_nd_update(pred, tf.expand_dims(dynamic_indices, axis=1), dynamic_pred)
                pred = tf.tensor_scatter_nd_update(pred, tf.expand_dims(static_indices, axis=1), static_pred)

                return pred
            
        elif task == 2:
            def classify(x):
                static_pred = component_dict['static'](x)
                static_pred = tf.math.argmax(static_pred, axis=1)

                resp_pred = component_dict['resp'](x)
                resp_pred = tf.math.argmax(resp_pred, axis=1)

                pred = static_pred + resp_pred * STATIC_MAX_LEN

                return pred

        elif (task == 3) or (task == 4):
            def classify(x):
                static_pred = component_dict['static'](x)
                static_pred = tf.math.argmax(static_pred, axis=1)

                breath_pred = component_dict['breath'](x)
                breath_pred = tf.math.argmax(breath_pred, axis=1)

                pred = static_pred + breath_pred * STATIC_MAX_LEN

                return pred
            
        else:
            raise Exception(f'Model undefined for task: {task}')
        
        return classify


###########################
# Custom model components #
###########################

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, l2_reg=0.001):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv1D(
            self.filters, 
            self.kernel_size, 
            strides=self.strides,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv1D(
            self.filters, 
            kernel_size=1, 
            strides=self.strides,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.add_layer = tf.keras.layers.Add()
        self.relu2 = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.relu1(x)

        shortcut = self.conv2(inputs)
        shortcut = self.batch_norm2(shortcut)

        x = self.add_layer([x, shortcut])
        x = self.relu2(x)

        return x
