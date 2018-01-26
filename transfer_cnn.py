import keras
import numpy as np
import os
import sys
import time

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

from utils import build_generator, LearningRateHistory, exp_decay, build_test_generator
from config import BATCH_SIZE, IMAGE_SIZE, list_classes
from random_crop import center_crop, random_crop, RandomCropSequence


def build_model_resnet_small_good_LB_782(num_classes, weights_file=None):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Connect last layer
    x = base_model.output
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = True

    if weights_file: model.load_weights(weights_file)

    return base_model, model


def build_model_resnet_small_LB_884(num_classes, weights_file=None):
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # Connect last layer
    x = base_model.output
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = True

    if weights_file: model.load_weights(weights_file)

    return base_model, model


def build_model_resnet(num_classes, weights_file=None):
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

    return base_model, model


def build_model_InceptionResNetV2(num_classes, weights_file=None):
    # Base model from pre-trained network
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

    # Connect last layer
    x = base_model.output
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    outputs = Dense(len(list_classes), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    for layer in base_model.layers: layer.trainable = False

    if weights_file: model.load_weights(weights_file)
    
    return base_model, model


def lr_schedule(epoch):
    """
    Learning Rate Scheduler for Adam
    """
    lr = 0.0003
    if epoch > 7:
        lr = 0.0001
    elif epoch > 9:
        lr = 0.00005
    elif epoch > 11:
        lr = 0.00001
    
    return lr


build_model = build_model_resnet_small_LB_884


def train_model(base_name=False):
    epochs = 500

    checkpointer = ModelCheckpoint(filepath='saved_models/weights.%s.base.hdf5' % base_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(patience=20)
    lr_reducer = ReduceLROnPlateau(factor=0.3, patience=10)

    callbacks_list = [checkpointer, LearningRateScheduler(lr_schedule), early_stopping, TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d_%H%M'))]

    base_model, model = build_model(len(list_classes))
    
    model.compile(optimizer=Adam(lr=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

    # Dynamic Image Cropping
    train_folder = '/media/nicholas/Data/Resources/Camera/train'
    train_generator = RandomCropSequence(train_folder)
    validation_generator = build_test_generator('/media/nicholas/Data/Resources/Camera/center_val_final_512/train', BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=15000 // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=2800 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        workers=4,
                        verbose=1)


def fine_tune(model, output_file, epochs=200):
    base_model, model = build_model(len(list_classes), weights_file=model)

    # Unfreeze the n last layers for fine tuning
    for layer in base_model.layers: layer.trainable = True

    model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    train_folder = '/media/nicholas/Data/Resources/Camera/train'

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(os.path.dirname(__file__), output_file),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min')
    
    # Callbacks
    early_stopping = EarlyStopping(patience=20)
    lr_reducer = ReduceLROnPlateau(factor=0.3, patience=10)
    callbacks_list = [checkpointer, early_stopping, TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d_%H%M'))]
    
    train_generator = RandomCropSequence(train_folder) # random_crop_generator(train_folder)
    validation_generator = build_test_generator('/media/nicholas/Data/Resources/Camera/center_val_final/train', BATCH_SIZE, IMAGE_SIZE)

    model.fit_generator(train_generator,
                        steps_per_epoch=20000 // BATCH_SIZE,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=5600 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        use_multiprocessing=True, workers=4,
                        verbose=1,
                        initial_epoch=94)


def evaluate(model, test_folder):
    base_model, model = build_model(model, weights_file=model)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    test_generator = build_test_generator(test_folder, BATCH_SIZE, IMAGE_SIZE)

    result = model.evaluate_generator(test_generator, use_multiprocessing=True)
    print(result)



def predict(model, average=False):
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    base_model, model = build_model(len(list_classes), weights_file=model)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print('fname,camera')

    for filename in os.listdir(test_folder):
        if average:
            im = load_img(os.path.join(test_folder, filename))
            w, h = im.size

            # Cut five patches: four corners and center
            X = []
            
            # Center
            crop = center_crop(im, (IMAGE_SIZE, IMAGE_SIZE))
            X.append(img_to_array(crop) / 255.)

            # Upper-left
            crop = im.crop((0, 0, IMAGE_SIZE, IMAGE_SIZE))
            X.append(img_to_array(crop) / 255.)
            # Upper-right
            crop = im.crop((w - IMAGE_SIZE, 0, w, IMAGE_SIZE))
            X.append(img_to_array(crop) / 255.)
            # Bottom-left
            crop = im.crop((0, h - IMAGE_SIZE, IMAGE_SIZE, h))
            X.append(img_to_array(crop) / 255.)
            # Bottom-right
            crop = im.crop((w - IMAGE_SIZE, h - IMAGE_SIZE, w, h))
            X.append(img_to_array(crop) / 255.)
            
            y_hat = model.predict(np.asarray(X, dtype=np.float32))
            y_hat = np.sum(y_hat, axis=0) / 5.
            print(filename + "," + list_classes[y_hat.argmax()])
        else:
            img = load_img(os.path.join(test_folder, filename))
            img = center_crop(img, (IMAGE_SIZE, IMAGE_SIZE))
            arr = img_to_array(img) / 255.
            y_hat = model.predict(np.asarray([arr], dtype=np.float32))
            print(filename + "," + list_classes[y_hat[0].argmax()])


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train_model(base_name='resnet_s')
    elif cmd == 'tune':
        fine_tune(model='saved_models/weights.resnet_s.base.hdf5', output_file='saved_models/weights.finetune.resnet_s.hdf5')
    elif cmd == 'predict':
        predict(model=sys.argv[2], average=True)
    elif cmd == 'eval':
        evaluate(model=sys.argv[2], test_folder='/media/nicholas/Data/Resources/Camera/random_patch/test')
