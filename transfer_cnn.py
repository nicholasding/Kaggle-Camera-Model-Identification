import keras
import numpy as np
import os

from PIL import Image
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.optimizers import Adam, SGD, Adagrad
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.applications import ResNet50, InceptionV3, Xception

from utils import build_generator, lr_schedule
from config import BATCH_SIZE, IMAGE_SIZE, list_classes

IMAGE_SIZE = 224

def build_model(num_classes):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet')
    # base_model = InceptionV3(include_top=False, weights='imagenet')

    # Connect last layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = False
    # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_model():
    train_folder = '/home/nicholas/Workspace/Resources/Camera/patches'
    epochs = 10000

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.xfr.patches.best.camera.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                            cooldown=0,
    #                            patience=5,
    #                            min_lr=0.5e-6)
    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=100)

    callbacks_list = [checkpointer, lr_reducer, lr_scheduler, early_stopping, TensorBoard()]

    model = build_model(len(list_classes))

    train_generator, validation_generator = build_generator(train_folder, BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=10000 // BATCH_SIZE, epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=5000 // BATCH_SIZE,
                        callbacks=callbacks_list)


def predict():
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    model = build_model(len(list_classes))
    model.load_weights('saved_models/weights.xfr.patches.best.camera.hdf5')

    print('fname,camera')

    for filename in os.listdir(test_folder):
        im = Image.open(os.path.join(test_folder, filename))
        resized = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)
        arr = np.array(resized)
        y_hat = model.predict(np.array([arr]) / 255)
        print(filename + "," + list_classes[y_hat[0].argmax()])


train_model()
# predict()
