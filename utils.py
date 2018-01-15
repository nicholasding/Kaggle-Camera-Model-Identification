import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras import backend as K


def build_generator(train_folder, batch_size, image_size):
    datagen_train = ImageDataGenerator(
        # horizontal_flip=True,
        # vertical_flip=True,
        # fill_mode='reflect',
        rescale=1. / 255
    )

    datagen_validation = ImageDataGenerator(
        rescale=1. / 255
    )

    train_generator = datagen_train.flow_from_directory(
        os.path.join(train_folder, 'train'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        # random_crop=True
    )

    validation_generator = datagen_validation.flow_from_directory(
        os.path.join(train_folder, 'validation'),
        target_size=(image_size, image_size),
        batch_size=batch_size,
        shuffle=True,
        # center_crop=True
    )

    return train_generator, validation_generator


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 0.00000002
    if epoch > 120:
        lr *= 0.0001
    elif epoch > 60:
        lr *= 0.001
    elif epoch > 30:
        lr *= 0.01
    elif epoch > 10:
        lr *= 0.1 
    
    return lr


class LearningRateHistory(Callback):

    def on_epoch_begin(self, batch, logs):
        lr = K.get_value(self.model.optimizer.lr)
        print('LR: %.2f' % lr)

