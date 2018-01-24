import os
import numpy as np

from PIL import Image
from config import list_classes, BATCH_SIZE
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from collections import defaultdict

PATCHES = BATCH_SIZE


def random_crop(img, random_crop_size, sync_seed=None):
    np.random.seed(sync_seed)
    w, h = img.size[0], img.size[1]
    rangew = (w - random_crop_size[0])
    rangeh = (h - random_crop_size[1])
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return img.crop((offsetw, offseth, offsetw + random_crop_size[0], offseth + random_crop_size[1]))


def center_crop(img, center_crop_size):
    center_w, center_h = img.size[0] // 2, img.size[1] // 2
    half_w, half_h = center_crop_size[0] // 2, center_crop_size[1] // 2
    return img.crop((center_w - half_w, center_h - half_h, center_w + half_w, center_h + half_h))


def random_crop_generator(folder):
    files = defaultdict(list)
    for clazz in list_classes:
        files[clazz].extend([os.path.join(folder, clazz, name) for name in os.listdir(os.path.join(folder, clazz))])
    
    crop_size = 224
    num_classes = len(list_classes)
    samples_per_class = 1
    patches_per_image = PATCHES // num_classes
    values = list(files.values())

    while True:
        X, y = [], []

        for i in range(num_classes):
            samples = np.random.choice(values[i], samples_per_class)

            for sample in samples:
                im = load_img(sample)

                name = os.path.dirname(sample).split('/')[-1]
                idx = list_classes.index(name)

                for i in range(patches_per_image):
                    crop = random_crop(im, (crop_size, crop_size))
                    X.append(np.asarray(crop, dtype=np.float32))
                    y.append(idx)
        
        yield (np.asarray(X, dtype=np.float32) / 255., to_categorical(y, len(list_classes)))
