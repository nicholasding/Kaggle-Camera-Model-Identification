import keras
import numpy as np
import os

from PIL import Image
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
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

def build_model(num_classes, weights_file=None):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet')
    # base_model = InceptionV3(include_top=False, weights='imagenet')

    # Connect last layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers[0:40]: layer.trainable = False

    if weights_file:
        model.load_weights(weights_file)
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    
    sgd = SGD(lr=0.002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=RMSprop(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    return base_model, model

MODEL_FILE = 'saved_models/weights.xfr.patches.best.camera.hdf5'

def train_model(load_weights=False):
    train_folder = '/home/nicholas/Workspace/Resources/Camera/random_patch'
    epochs = 10000

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.xfr.random.best.camera.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    lr_scheduler = LearningRateScheduler(lr_schedule)
    # lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
    #                            cooldown=0,
    #                            patience=5,
    #                            min_lr=0.5e-6)
    lr_reducer = ReduceLROnPlateau(factor=0.1, patience=3)
    early_stopping = EarlyStopping(patience=100)

    callbacks_list = [checkpointer, lr_scheduler, early_stopping, TensorBoard()]
    # callbacks_list = [checkpointer, lr_reducer, early_stopping, TensorBoard()] # Adagrad doesn't need lr scheduler

    # base_model, model = build_model(len(list_classes), MODEL_FILE)
    base_model, model = build_model(len(list_classes))
    model.summary()

    train_generator, validation_generator = build_generator(train_folder, BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=5000 // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=4000 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        verbose=1)


def predict():
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    base_model, model = build_model(len(list_classes))
    model.load_weights('saved_models/weights.xfr.random.best.camera.hdf5')

    print('fname,camera')

    for filename in os.listdir(test_folder):
        im = Image.open(os.path.join(test_folder, filename))
        resized = im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
        arr = np.array(resized)
        y_hat = model.predict(np.array([arr]) / 255)
        print(filename + "," + list_classes[y_hat[0].argmax()])


#train_model(load_weights=False)
predict()
