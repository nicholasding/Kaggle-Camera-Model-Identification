import keras
import numpy as np
import os

from PIL import Image
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.applications import ResNet50, InceptionV3

from utils import build_generator, lr_schedule
from config import BATCH_SIZE, IMAGE_SIZE, list_classes

IMAGE_SIZE = 224

def build_model(num_classes):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet')
    
    # Connect last layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    outputs = Dense(len(list_classes), activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(input=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=lr_schedule(0), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    train_folder = '/home/nicholas/Workspace/Resources/Camera'
    epochs = 1000

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.xfr.best.camera.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpointer]

    model = build_model(len(list_classes))

    train_generator, validation_generator = build_generator(train_folder, BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=2000 // BATCH_SIZE, epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=800 // BATCH_SIZE,
                        callbacks=callbacks_list)


def predict():
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    model = build_model(len(list_classes))
    model.load_weights('saved_models/weights.xfr.best.camera.hdf5')

    print('fname,camera')

    for filename in os.listdir(test_folder):
        im = Image.open(os.path.join(test_folder, filename))
        resized = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NORMAL)
        arr = np.array(resized)
        y_hat = model.predict(np.array([arr]))
        print(filename + "," + list_classes[y_hat[0].argmax()])


# train_model()
predict()