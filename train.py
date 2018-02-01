import keras
import numpy as np
import os
import sys
import time
import cv2

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

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
from random_crop import center_crop, ImageLoadSequence, RandomCropMergedSequence, CenterCropMergedSequence


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


def load_all_files(folder):
    files = []
    for clazz in list_classes:
        files.extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
    return files


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
    validation_batches = 2800 // BATCH_SIZE

    # Dynamic Image Cropping
    train_folder = '/media/nicholas/Data/Resources/Camera/train'
    # validation_folder = '/media/nicholas/Data/Resources/Camera/center_val_final_512/train'
    # train_generator = RandomCropMergedSequence(train_folder, train_batches)
    # validation_generator = ImageLoadSequence(validation_folder, return_manipulated=True)

    files = load_all_files(train_folder)
    train_files, validation_files = train_test_split(files, test_size=0.1, random_state=42)
    print('%d files for training, %d files for validation' % (len(train_files), len(validation_files)))

    train_generator = RandomCropMergedSequence(train_files, train_batches)
    validation_generator = CenterCropMergedSequence(validation_files, validation_batches)

    # Balance the class weights
    calculated_weights = class_weight.compute_class_weight('balanced', np.unique(list_classes), [os.path.dirname(name).split('/')[-1] for name in files])
    print('Weights', calculated_weights)

    model.fit_generator(train_generator,
                        steps_per_epoch=train_batches,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=validation_batches,
                        callbacks=callbacks_list,
                        use_multiprocessing=True,
                        workers=6,
                        initial_epoch=initial_epoch,
                        class_weight=calculated_weights,
                        verbose=1)


def predict(model, average=False):
    test_folder = '/home/nicholas/Workspace/Resources/Camera/test'

    base_model, model = build_model(len(list_classes), weights_file=model)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    print('fname,camera')

    for filename in os.listdir(test_folder):
        if average:
            im = cv2.imread(os.path.join(test_folder, filename))
            w, h = im.shape[1], im.shape[0]
            
            # Normalize
            im = im / 255.
            manip = np.float32([1. if 'manip' in filename else 0.])

            sw, sh = w // IMAGE_SIZE, h // IMAGE_SIZE
            X = np.zeros((sw * sh, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
            m = np.zeros((sw * sh, 1), dtype=np.float32)

            idx = 0
            for x in range(sw):
                for y in range(sh):
                    X[idx] = np.copy(im[y * IMAGE_SIZE : (y+1) * IMAGE_SIZE, x * IMAGE_SIZE : (x+1) * IMAGE_SIZE])
                    m[idx] = manip
                    idx += 1
            
            y_hat = model.predict([X, m])
            y_hat = np.sum(y_hat, axis=0)
            print(filename + "," + list_classes[y_hat.argmax()])
        else:
            X, m = [], []
            img = cv2.imread(os.path.join(test_folder, filename))
            manip = 'manip' in filename
            if manip:
                manip = 1.
            else:
                manip = 0.
            
            X.append(img / 255.)
            m.append(manip)
            
            y_hat = model.predict([np.asarray(X, dtype=np.float32), np.asarray(m, dtype=np.float32)])
            print(filename + "," + list_classes[y_hat[0].argmax()])


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'train':
        train_model(base_name='resnet_m')
    elif cmd == 'tune':
        # fine_tune(model='saved_models/weights.resnet_s.base.hdf5.LB.890', output_file='saved_models/weights.finetune.resnet_s.hdf5')
        train_model(base_name='resnet_m', weights_file='saved_models/weights.resnet_m.base.hdf5', initial_epoch=70)
    elif cmd == 'predict':
        predict(model=sys.argv[2], average=True)
