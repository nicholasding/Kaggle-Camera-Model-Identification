import math
import os

from PIL import Image

train_folder = '/home/nicholas/Workspace/Resources/Camera/train'

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

# Walk into each directory and process the images
for folder in os.listdir(train_folder):
    for filename in os.listdir(os.path.join(train_folder, folder)):
        print('Processing', folder, filename)
        crop_split(os.path.join(train_folder, folder, filename))
        os.remove(os.path.join(train_folder, folder, filename))
