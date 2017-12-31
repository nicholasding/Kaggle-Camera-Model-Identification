# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, SeparableConv2D, Dropout, LSTM, BatchNormalization, Flatten, GlobalMaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint  
from keras import optimizers, utils
from keras.layers.normalization import BatchNormalization

IMAGE_SIZE = 256

list_classes = [
    'HTC-1-M7', 
    'LG-Nexus-5x', 
    'Motorola-Droid-Maxx', 
    'Motorola-Nexus-6', 
    'Motorola-X', 
    'Samsung-Galaxy-Note3', 
    'Samsung-Galaxy-S4', 
    'Sony-NEX-7', 
    'iPhone-4s', 
    'iPhone-6'
]

def build_model(num_classes, input_shape):
    model = Sequential()
    
    # Conv layers
    model.add(SeparableConv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(SeparableConv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Activation
    model.add(GlobalMaxPool2D())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    model.compile(optimizer='rmsprop', 
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    
    model.summary()

    return model


def load_image_array(filepath):
    im = Image.open(filepath)
    im = im.convert('RGB')
    im = im.resize((IMAGE_SIZE, IMAGE_SIZE))
    im = np.asarray(im, dtype=np.float32) / 255
    im = im[:, :, :3]
    return im


def load_training_generator():
    train_folder = '/home/nicholas/Workspace/Resources/Camera/train'

    labels = utils.to_categorical(range(len(list_classes)))

    for i, category in enumerate(list_classes):
        for filename in os.listdir(os.path.join(train_folder, category)):
            X = load_image_array(os.path.join(train_folder, category, filename))
            y = labels[i]
            print(X.shape)
            yield X, y


def build_generator():
    # generator = ImageDataGenerator(rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     rescale=1./255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')
    
    generator = ImageDataGenerator()

    return generator


def train_model():
    train_folder = '/home/nicholas/Workspace/Resources/Camera'

    epochs = 1000
    batch_size = 16

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.camera.hdf5', verbose=1, save_best_only=True)
    callbacks_list = [checkpointer]

    model = build_model(len(list_classes), (IMAGE_SIZE, IMAGE_SIZE, 3))

    train_datagen = build_generator()
    train_generator = train_datagen.flow_from_directory(os.path.join(train_folder, 'train'), target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=batch_size)
    validation_generator = train_datagen.flow_from_directory(os.path.join(train_folder, 'validation'), target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=batch_size)

    generator = load_training_generator()

    model.fit_generator(train_generator,
                        steps_per_epoch=2000 // batch_size, epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=800 // batch_size,
                        callbacks=callbacks_list)

    # X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=callbacks_list)
    

def predict():
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    model = build_model(len(list_classes), (IMAGE_SIZE, IMAGE_SIZE, 3))
    model.load_weights('saved_models/weights.best.camera.hdf5')

    print('fname,camera')

    for filename in os.listdir(test_folder):
        im = Image.open(os.path.join(test_folder, filename))
        resized = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NORMAL)
        arr = np.array(resized)
        y_hat = model.predict(np.array([arr]))
        print(filename + "," + list_classes[y_hat[0].argmax()])
        # print(filename, y_hat)


# train_model()
predict()