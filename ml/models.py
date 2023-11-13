"""Build classficiation model components"""

# architectures
import tensorflow as tf

# timing
from timeit import default_timer as timer


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
            pass
        elif (args.format == 'normal'):
            format = tf.keras.layers.Flatten()
            transform.add(format)
        elif args.format == 'summary':
            format = tf.keras.layers.Lambda(lambda x: tf.concat([tf.math.reduce_mean(x, axis=1),
                                                                 tf.math.reduce_std(x, axis=1)], 
                                                                 axis=1))
            transform.add(format)
            transform.add(tf.keras.layers.Flatten())
        else:
            raise Exception('Transform undefined!')
        
        return transform
    


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
        transforms = InputTransformBuilder.build_transform(args.INPUT, args.ARCH.LSTM.num_layers > 0)
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
            classifier.load_weights(weights)

        return classifier


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