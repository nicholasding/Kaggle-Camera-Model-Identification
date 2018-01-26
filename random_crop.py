import os
import numpy as np
import random

from io import BytesIO
from skimage import exposure
from PIL import Image
from config import list_classes, BATCH_SIZE, IMAGE_SIZE
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import load_img
from collections import defaultdict
from img_utils import random_transformation, random_crop, center_crop


PATCHES = BATCH_SIZE


class RandomCropSequence(Sequence):

    def __init__(self, folder):
        self.folder = folder
        self.prepare(folder)
    
    def prepare(self, folder):
        self.files = defaultdict(list)

        for clazz in list_classes:
            self.files[clazz].extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
        
        self.samples_per_class = 1
        self.patches_per_image = PATCHES // len(list_classes)
        self.values = list(self.files.values())
    
    def __len__(self):
        return 10000000
    
    def __getitem__(self, idx):
        crop_size = IMAGE_SIZE
        num_classes = len(list_classes)

        X, y = [], []

        for i in range(num_classes):
            samples = np.random.choice(self.values[i], self.samples_per_class)

            for sample in samples:
                im = load_img(sample)

                name = os.path.dirname(sample).split('/')[-1]
                idx = list_classes.index(name)

                for i in range(self.patches_per_image):
                    if np.random.random() < 0.5:
                        try:
                            # Make twice as big because of the resize 
                            crop = random_crop(im, (crop_size * 2.5, crop_size * 2.5))
                            crop = random_transformation(crop)
                            crop = center_crop(crop, (crop_size, crop_size))
                        except Exception as e:
                            print('Error:', e)
                            crop = random_crop(im, (crop_size, crop_size))
                    else:
                        crop = random_crop(im, (crop_size, crop_size))
                    
                    X.append(np.asarray(crop, dtype=np.float32))
                    y.append(idx)
        
        if crop_size > 224:
            # To overcome GPU memory limit, divide the batch into two
            for i in range(2):
                return (np.asarray(X[i*5:(i+1)*5], dtype=np.float32) / 255., to_categorical(y[i*5:(i+1)*5], len(list_classes)))
        else:
            return (np.asarray(X, dtype=np.float32) / 255., to_categorical(y, len(list_classes)))


# def random_crop_generator(folder):
#     files = defaultdict(list)
#     for clazz in list_classes:
#         files[clazz].extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
    
#     crop_size = 224
#     num_classes = len(list_classes)
#     samples_per_class = 1
#     patches_per_image = PATCHES // num_classes
#     values = list(files.values())

#     while True:
#         X, y = [], []

#         for i in range(num_classes):
#             samples = np.random.choice(values[i], samples_per_class)

#             for sample in samples:
#                 im = load_img(sample)
                
#                 # Random transformation on 50% chance
#                 if np.random.random() < 0.5: im = random_transformation(im)

#                 name = os.path.dirname(sample).split('/')[-1]
#                 idx = list_classes.index(name)

#                 for i in range(patches_per_image):
#                     crop = random_crop(im, (crop_size, crop_size))
#                     X.append(np.asarray(crop, dtype=np.float32))
#                     y.append(idx)
        
#         yield (np.asarray(X, dtype=np.float32) / 255., to_categorical(y, len(list_classes)))
