"""Build classficiation model components"""

# architectures
import tensorflow as tf

# timing
from timeit import default_timer as timer


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


class InputTransformBuilder:
    """
    Builder class for transforming input to features for classfication models.

    Methods:
    - build_transform(args): Build a transform builder for classifier model.
    """

    @staticmethod
    def build_transform(args, is_rnn=False):
        """
        Build a transform for the input.

        Args:
        - args: input configuration

        Returns:
        - a transform function
        """
        if is_rnn and (args.format != 'window'):
            raise Exception('RNNs require window format!')
        elif (not is_rnn) and (args.format == 'window'):
            raise Exception('only RNNs can use window format!')

        transform = tf.keras.Sequential()

        # what sensors to use
        if args.sensor == 'all':
            pass
        elif args.sensor == 'accel':
            sensors = tf.keras.layers.Lambda(lambda x: x[:, 0:3])
            transform.add(sensors)
        elif args.sensor == 'gyro':
            sensors = tf.keras.layers.Lambda(lambda x: x[:, 3:6])
            transform.add(sensors)
        else:
            raise Exception('Sensor undefined!')

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
        transforms, input_size = InputTransformBuilder.build_transform(args.INPUT, args.ARCH.LSTM.num_layers > 0)
        if len(transforms.layers) > 0:
            classifier.add(transforms)

        # build layers
        for i in range(args.ARCH.LSTM.num_layers):
            if i < args.ARCH.LSTM.num_layers - 1:
                classifier.add(tf.keras.layers.LSTM(args.ARCH.LSTM.hidden_size, 
                                                    return_sequences=True))
            else:
                classifier.add(tf.keras.layers.LSTM(args.ARCH.LSTM.hidden_size, 
                                                    return_sequences=False))

        for _ in range(args.ARCH.MLP.num_layers):
            classifier.add(tf.keras.layers.Dense(units=args.ARCH.MLP.hidden_size, 
                                                 activation='relu'))
            classifier.add(tf.keras.layers.Dropout(args.ARCH.MLP.dropout))

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