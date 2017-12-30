# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Any results you write to the current directory are saved as output.

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, LSTM, BatchNormalization, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint  
from keras import optimizers
from keras.layers.normalization import BatchNormalization

list_classes = ['Samsung-Galaxy-Note3', 'HTC-1-M7', 'Sony-NEX-7', 'Samsung-Galaxy-S4', 'iPhone-4s', 'Motorola-Nexus-6', 'LG-Nexus-5x', 'Motorola-X', 'iPhone-6', 'Motorola-Droid-Maxx']

def build_model(num_classes, input_shape):
    model = Sequential()
    
    # Conv layers
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    # Activation
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile
    model.compile(optimizer=optimizers.RMSprop(lr=1e-4), 
        loss='categorical_crossentropy', metrics=['acc'])
    
    model.summary()

    return model

def build_generator():
    generator = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    
    return generator


train_folder = '/home/nicholas/Workspace/Resources/Camera'

batch_size = 32

train_datagen = build_generator()
train_generator = train_datagen.flow_from_directory(os.path.join(train_folder, 'train'), target_size=(256, 256), batch_size=batch_size)

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.camera.hdf5', verbose=1, save_best_only=True)

model = build_model(len(list_classes), (256, 256, 3))
model.fit_generator(train_generator, steps_per_epoch=2000 // batch_size, epochs=10, callbacks=[checkpointer])
