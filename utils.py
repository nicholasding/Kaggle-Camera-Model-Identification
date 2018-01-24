import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras import backend as K
from config import IMAGE_SIZE


def build_generator(train_folder, batch_size, image_size):
    datagen_train = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        # zoom_range=0.2,
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


def build_test_generator(test_folder, batch_size, image_size):
    datagen = ImageDataGenerator(rescale=1. / 255)
    return datagen.flow_from_directory(test_folder, 
                        target_size=(image_size, image_size),
                        batch_size=batch_size,
                        shuffle=True)


INITIAL_RATE = 0.002

def exp_decay(epoch):
    initial_lrate = INITIAL_RATE
    k = 0.1
    lrate = initial_lrate * np.exp(-k * epoch)
    return lrate


class LearningRateHistory(Callback):

    def on_epoch_begin(self, batch, logs):
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        sgd = self.model.optimizer
        lr = sgd.lr
        lr *= (1. / (1. + sgd.decay * K.cast(sgd.iterations, K.dtype(sgd.decay))))
        print('LR:', K.eval(lr))

