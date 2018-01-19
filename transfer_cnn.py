import keras
import numpy as np
import os
import sys

from PIL import Image
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.applications import ResNet50, InceptionV3, Xception, InceptionResNetV2

from utils import build_generator, lr_schedule, LearningRateHistory, exp_decay, build_test_generator
from config import BATCH_SIZE, IMAGE_SIZE, list_classes

IMAGE_SIZE = 299

def build_model(num_classes, weights_file=None):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Connect last layer
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = False

    if weights_file: model.load_weights(weights_file)
        # for layer in base_model.layers[-20:]:
        #     layer.trainable = True
    
    # sgd = SGD(lr=0.0002, decay=0.0005, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adagrad(), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=RMSprop(lr=0.045, decay=0.9, epsilon=1.0), loss='categorical_crossentropy', metrics=['accuracy'])

    return base_model, model


def train_model(load_weights=False):
    train_folder = '/media/nicholas/Data/Resources/Camera/random_patch' # '/home/nicholas/Workspace/Resources/Camera/random_patch'
    epochs = 10000

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.xfr.random.best.camera.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(patience=150)

    callbacks_list = [checkpointer, LearningRateHistory(), early_stopping, TensorBoard()]

    if load_weights:
        base_model, model = build_model(len(list_classes), MODEL_FILE)
    else:
        base_model, model = build_model(len(list_classes))
    
    # model.summary()

    train_generator, validation_generator = build_generator(train_folder, BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=10000 // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=4000 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        verbose=1)


def predict(model):
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    base_model, model = build_model(len(list_classes), weights_file=model)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print('fname,camera')

    for filename in os.listdir(test_folder):
        img = load_img(os.path.join(test_folder, filename))
        arr = img_to_array(img) / 255.
        y_hat = model.predict(np.asarray([arr]))
        print(filename + "," + list_classes[y_hat[0].argmax()])


def fine_tune(model, output_file, epochs=500):
    base_model, model = build_model(model, weights_file=model)

    # Unfreeze the n last layers for fine tuning
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    sgd = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    train_folder = '/media/nicholas/Data/Resources/Camera/random_patch'

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(__file__), output_file),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    
    early_stopping = EarlyStopping(patience=30)
    callbacks_list = [checkpointer, LearningRateHistory(), early_stopping, TensorBoard()]

    train_generator, validation_generator = build_generator(train_folder, BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=10000 // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=4000 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        verbose=1)


def evaluate(model, test_folder):
    base_model, model = build_model(model, weights_file=model)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    test_generator = build_test_generator(test_folder, BATCH_SIZE, IMAGE_SIZE)

    result = model.evaluate_generator(test_generator, use_multiprocessing=True)
    print(result)


MODEL_FILE = 'saved_models/weights.xfr.random_load.best.camera.hdf5'


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train_model()
    elif cmd == 'tune':
        fine_tune(model='saved_models/weights.xfr.random_load.best.camera.hdf5', output_file='saved_models/weights.finetune.resnet.hdf5')
    elif cmd == 'predict':
        predict(model=sys.argv[2])
    elif cmd == 'eval':
        evaluate(model=sys.argv[2], test_folder='/media/nicholas/Data/Resources/Camera/random_patch/test')
