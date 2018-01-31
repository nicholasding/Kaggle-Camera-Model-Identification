import keras
import numpy as np
import os
import sys
import time
import cv2

from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import AveragePooling2D, Input, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D, Dropout, concatenate
from keras.optimizers import Adam, SGD, Adagrad, RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.applications import ResNet50, InceptionV3, Xception, InceptionResNetV2

from utils import build_generator, LearningRateHistory, exp_decay, build_test_generator
from config import BATCH_SIZE, IMAGE_SIZE, list_classes
from random_crop import center_crop, ImageLoadSequence, RandomCropMergedSequence


def build_model(num_classes, weights_file=None):
    """
    Build concatenated model that uses all the information, especially the manip/unalt attribute.
    """
    # Base model from pre-trained network
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')

    # ResNet50 output layer
    x = base_model.output
    
    # Input input layer (OpenCV BGR)
    image_input = base_model.input

    # Information about whether the image was manipulated or not
    manipulated_input = Input(shape=(1,))

    # Concatenated two inputs
    x = concatenate([x, manipulated_input])

    # Adding FC layers
    x = Dense(512, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name='dropout_fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5, name='dropout_fc2')(x)

    # Adding final layer
    outputs = Dense(len(list_classes), activation='softmax', name='output')(x)

    # Build the model
    model = Model(inputs=[image_input, manipulated_input], outputs=outputs)

    # Compile the model and freeze the pre-trained layers
    # for layer in base_model.layers: layer.trainable = True

    # Load weights
    if weights_file: model.load_weights(weights_file)

    return base_model, model


def train_model(base_name=False, weights_file=None, initial_epoch=0):
    # Callbacks
    if weights_file is None:
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.%s.base.hdf5' % base_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    else:
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.%s.tune.hdf5' % base_name, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    early_stopping = EarlyStopping(patience=25)
    lr_reducer = ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-10)
    callbacks_list = [checkpointer, lr_reducer, early_stopping, TensorBoard(log_dir='./logs/' + time.strftime('%Y%m%d_%H%M'))]

    # Model build & compile
    base_model, model = build_model(len(list_classes), weights_file=weights_file)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    epochs = 500
    train_batches = 8000 // BATCH_SIZE

    # Dynamic Image Cropping
    train_folder = '/media/nicholas/Data/Resources/Camera/train'
    validation_folder = '/media/nicholas/Data/Resources/Camera/center_val_final/train'
    train_generator = RandomCropMergedSequence(train_folder, train_batches)
    validation_generator = ImageLoadSequence(validation_folder, return_manipulated=True)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_batches,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=2800 // BATCH_SIZE,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        workers=4,
                        initial_epoch=initial_epoch,
                        verbose=1)


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
            # img = load_img(os.path.join(test_folder, filename))
            # img = center_crop(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = cv2.imread(os.path.join(test_folder, filename))
            arr = img / 255.
            y_hat = model.predict(np.asarray([arr], dtype=np.float32))
            print(filename + "," + list_classes[y_hat[0].argmax()])


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train_model(base_name='resnet_m')
    elif cmd == 'tune':
        # fine_tune(model='saved_models/weights.resnet_s.base.hdf5.LB.890', output_file='saved_models/weights.finetune.resnet_s.hdf5')
        train_model(base_name='resnet_m', weights_file='saved_models/weights.resnet_s.base.hdf5', initial_epoch=52)
    elif cmd == 'predict':
        predict(model=sys.argv[2], average=False)
