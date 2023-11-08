"""Build classficiation model components"""

# architectures
import tensorflow as tf


def get_num_sensors(args):
    if args.sensor == 'all':
        num_sensors = 6
    elif args.sensor == 'accel':
        num_sensors = 3
    elif args.sensor == 'gyro':
        num_sensors = 3
    else:
        raise Exception(f'Sensor undefined!: {args.sensor}')
    return num_sensors

def get_per_sensor_size(args):
    if args.format == 'normal':
        num_per_sensor = args.window_size
    elif args.format == 'summary':
        num_per_sensor =  2
    else:
        raise Exception(f'Input format undefined!: {args.format}')
    return num_per_sensor

def get_input_shape(args):
    return (get_num_sensors(args) * get_per_sensor_size(args),)



class InputTransformBuilder:
    """
    Builder class for transforming input to features for classfication models.

    Methods:
    - build_transform(args): Build a transform builder for classifier model.
    """
    @staticmethod
    def build_transform(args):
        """
        Build a transform for the input.

        Args:
        - args: input configuration

        Returns:
        - a transform function
        """
        transform = tf.keras.Sequential()

        # what sensors to use
        if args.sensor == 'all':
            pass
        elif args.sensor == 'accel':
            transform.add(tf.keras.layers.Lambda(lambda x: x[:, 0:3]))
        elif args.sensor == 'gyro':
            transform.add(tf.keras.layers.Lambda(lambda x: x[:, 3:6]))
        else:
            raise Exception('Sensor undefined!')

        # how to process that data
        if args.format == 'normal':
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
        transforms = InputTransformBuilder.build_transform(args.INPUT)

        classifier = tf.keras.Sequential()
        classifier.add(transforms)

        if args.ARCH.type == 'mlp':
            for _ in range(args.ARCH.num_layers):
                classifier.add(tf.keras.layers.Dense(units=args.ARCH.hidden_size, 
                                                     activation='relu'))
                classifier.add(tf.keras.layers.Dropout(args.ARCH.dropout))
        elif args.ARCH.type == 'lstm':
            classifier.add(tf.keras.layers.LSTM(args.ARCH.hidden_size))
        elif args.ARCH.type == 'log':
            pass
        else:
            raise Exception('Architecture undefined!')
        
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
        lr_scheduler = LRScheduleBuilder.build_scheduler(args.LR)

        if args.OPTIM.optim == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr_scheduler,
                                                momentum=args.OPTIM.momentum, 
                                                weight_decay=args.OPTIM.weight_decay)
        elif args.OPTIM.optim == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler,
                                                 beta_1=args.OPTIM.momentum,
                                                 weight_decay=args.OPTIM.weight_decay)
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
        - optimizer: The optimizer to schedule the learning rate for.

        Returns:
        - lr_scheduler: The learning rate scheduler.
        """
        if args.schedule == 'constant':
            lr_scheduler = args.lr
        elif args.schedule == 'step':
            lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
                args.lr, 
                args.step_size, 
                args.gamma, 
                staircase=True)
        else:
            raise Exception('LR Scheduler undefined!')
        return lr_scheduler