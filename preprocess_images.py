import math
import os
import random

from PIL import Image

train_folder = '/home/nicholas/Workspace/Resources/Camera/train'
validation_folder = '/home/nicholas/Workspace/Resources/Camera/validation'

def crop_split(filepath, grid_size=512):
    filename = os.path.basename(filepath)
    name, ext = filename.split('.')
    print(name, ext)

    im = Image.open(filepath)
    crops = []

    for col in range(im.size[0] // grid_size):
        for row in range(im.size[1] // grid_size):
            x1 = col * grid_size
            y1 = row * grid_size
            x2 = x1 + grid_size
            y2 = y1 + grid_size
            
            crops.append(im.crop((x1, y1, x2, y2)))
    
    dirname = os.path.dirname(filepath)

    for idx, crop in enumerate(crops):
        print('Saving', os.path.join(dirname, '.'.join([name, str(idx), ext])))
        crop.save(os.path.join(dirname, '.'.join([name, str(idx), ext])))

def batch_crop():
    # Walk into each directory and process the images
    for folder in os.listdir(train_folder):
        for filename in os.listdir(os.path.join(train_folder, folder)):
            print('Processing', folder, filename)
            crop_split(os.path.join(train_folder, folder, filename))
            os.remove(os.path.join(train_folder, folder, filename))

def generate_validation_set():
    for folder in os.listdir(train_folder):
        files = os.listdir(os.path.join(train_folder, folder))
        validation_set = random.sample(files, k=int(len(files) * 0.2))

        print('Moving %d files out of %d' % (len(validation_set), len(files)))

        assert len(validation_set) == len(set(validation_set))

        # Create target folder
        target_folder = os.path.join(validation_folder, folder)
        if not os.path.exists(target_folder):os.mkdir(target_folder)

        # Move files
        for filename in validation_set:
            os.rename(os.path.join(train_folder, folder, filename), os.path.join(target_folder, filename))

# batch_crop()
generate_validation_set()
