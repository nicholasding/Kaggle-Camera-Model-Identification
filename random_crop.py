import os
import numpy as np
import random
import cv2

from config import list_classes, BATCH_SIZE, IMAGE_SIZE
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import load_img
from collections import defaultdict
from img_utils import random_transformation, random_crop, center_crop

PATCHES = BATCH_SIZE


class ImageLoadSequence(Sequence):
    
    def __init__(self, folder, return_manipulated=False):
        self.folder = folder
        self.return_manipulated = return_manipulated
        self.prepare(folder)
    
    def prepare(self, folder):
        self.files = []

        for clazz in list_classes:
            self.files.extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
        
        print('Found %d samples in %s classes' % (len(self.files), len(list_classes)))
    
    def __len__(self):
        return len(self.files) // BATCH_SIZE
    
    def __getitem__(self, idx):
        samples = np.random.choice(self.files, BATCH_SIZE)

        X, y, manipulated = [], [], []

        for sample in samples:
            im = cv2.imread(sample)

            name = os.path.dirname(sample).split('/')[-1]
            idx = list_classes.index(name)

            if 'manip' in sample:
                manipulated.append(1.)
            else:
                manipulated.append(0.)
            
            X.append(np.asarray(im, dtype=np.float32))
            y.append(idx)
        
        if self.return_manipulated:
            return ([np.asarray(X, dtype=np.float32) / 255., np.asarray(manipulated, dtype=np.float32)], to_categorical(y, len(list_classes)))
        else:
            return (np.asarray(X, dtype=np.float32) / 255., to_categorical(y, len(list_classes)))


class RandomCropSequence(Sequence):

    def __init__(self, folder, length):
        self.folder = folder
        self.length = length
        self.prepare(folder)
    
    def prepare(self, folder):
        self.files = defaultdict(list)

        for clazz in list_classes:
            self.files[clazz].extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
        
        self.samples_per_class = PATCHES // len(list_classes)
        self.patches_per_image = PATCHES // len(list_classes)
        self.values = list(self.files.values())
        self.all_files = []

        for i in range(len(list_classes)):
            self.all_files.extend(self.values[i])
        
        print('Found %d samples in %s classes' % (len(self.all_files), len(list_classes)))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        crop_size = IMAGE_SIZE
        num_classes = len(list_classes)

        X, y = [], []

        # samples = []

        # for i in range(num_classes):
        #     samples.extend(np.random.choice(self.values[i], self.samples_per_class))

        # np.random.shuffle(samples)
        # samples = np.random.choice(samples, PATCHES)

        samples = np.random.choice(self.all_files, BATCH_SIZE)

        for sample in samples:
            im = cv2.imread(sample)

            name = os.path.dirname(sample).split('/')[-1]
            idx = list_classes.index(name)
            
            if np.random.random() < 0.5:
                # Make twice as big because of the resize 
                crop = random_crop(im, (crop_size * 3, crop_size * 3))
                crop = random_transformation(crop)
                crop = center_crop(crop, (crop_size, crop_size))
            else:
                crop = random_crop(im, (crop_size, crop_size))

            X.append(np.asarray(crop, dtype=np.float32))
            y.append(idx)
        
        return (np.asarray(X, dtype=np.float32) / 255., to_categorical(y, len(list_classes)))


class RandomCropMergedSequence(Sequence):
    
    def __init__(self, files, length):
        self.files = files
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        X, y, manipulated = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32), [], np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        samples = np.random.choice(self.files, BATCH_SIZE)

        for i, sample in enumerate(samples):
            im = cv2.imread(sample)

            name = os.path.dirname(sample).split('/')[-1]
            idx = list_classes.index(name)
            
            if np.random.random() < 0.5:
                # Make twice as big because of the resize 
                crop = random_crop(im, (IMAGE_SIZE * 3, IMAGE_SIZE * 3))
                crop = random_transformation(crop)
                crop = center_crop(crop, (IMAGE_SIZE, IMAGE_SIZE))
                manipulated[i] = np.float32([1.])
            else:
                crop = random_crop(im, (IMAGE_SIZE, IMAGE_SIZE))
                manipulated[i] = np.float32([0.])

            X[i] = crop
            y.append(idx)
        
        return (
                    # Image & Manipulation (Boolean)
                    [X / 255., manipulated],
                    # Labels
                    to_categorical(y, len(list_classes))
                )


class CenterCropMergedSequence(Sequence):
    
    def __init__(self, files, length):
        self.files = files
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        X, y, manipulated = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32), [], np.zeros((BATCH_SIZE, 1), dtype=np.float32)
        samples = np.random.choice(self.files, BATCH_SIZE)

        for i, sample in enumerate(samples):
            im = cv2.imread(sample)

            name = os.path.dirname(sample).split('/')[-1]
            idx = list_classes.index(name)
            
            if np.random.random() < 0.5:
                crop = center_crop(im, (IMAGE_SIZE * 2, IMAGE_SIZE * 2))
                crop = random_transformation(crop)
                crop = center_crop(crop, (IMAGE_SIZE, IMAGE_SIZE))
                manipulated[i] = np.float32([1.])
            else:
                crop = center_crop(im, (IMAGE_SIZE, IMAGE_SIZE))
                manipulated[i] = np.float32([0.])
            
            X[i] = crop
            y.append(idx)
        
        return (
                    # Image & Manipulation (Boolean)
                    [X / 255., manipulated],
                    # Labels
                    to_categorical(y, len(list_classes))
                )
