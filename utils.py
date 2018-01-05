import os

from keras.preprocessing.image import ImageDataGenerator


def build_generator(train_folder, batch_size, image_size):
    # generator = ImageDataGenerator(rotation_range=40,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     rescale=1./255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     fill_mode='nearest')
    generator = ImageDataGenerator(rescale=1. / 255)

    train_generator = generator.flow_from_directory(os.path.join(train_folder, 'train'), target_size=(image_size, image_size), batch_size=batch_size)
    validation_generator = generator.flow_from_directory(os.path.join(train_folder, 'validation'), target_size=(image_size, image_size), batch_size=batch_size)
    
    return train_generator, validation_generator


def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-4
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 150:
        lr *= 1e-3
    elif epoch > 100:
        lr *= 1e-2
    elif epoch > 50:
        lr *= 1e-1
    return lr
