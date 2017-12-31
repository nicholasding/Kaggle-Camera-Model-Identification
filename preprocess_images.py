import math
import os
import random

from multiprocessing import Process
from PIL import Image

train_folder = '/home/nicholas/Workspace/Resources/Camera/train'
validation_folder = '/home/nicholas/Workspace/Resources/Camera/validation'

def resize_crop(im, ratio, grid_size=1024):
    resized = im.resize((int(im.size[0] * ratio), int(im.size[1] * ratio)), Image.BICUBIC)

    center_x = resized.size[0] // 2
    center_y = resized.size[1] // 2

    return resized.crop((center_x - grid_size // 2, center_y - grid_size // 2, center_x + grid_size // 2, center_y + grid_size // 2))

def gamma_correction(im, gamma):
    """
    Fast gamma correction with PIL's image.point() method
    """
    invert_gamma = 1.0 / gamma
    lut = [pow(x/255., invert_gamma) * 255 for x in range(256)]
    lut = lut * 3 # need one set of data for each band for RGB
    im = im.point(lut)
    return im

def crop_split(filepath, grid_size=1024):
    """
    JPEG compression with quality factor = 70
    JPEG compression with quality factor = 90
    resizing (via bicubic interpolation) by a factor of 0.5
    resizing (via bicubic interpolation) by a factor of 0.8
    resizing (via bicubic interpolation) by a factor of 1.5
    resizing (via bicubic interpolation) by a factor of 2.0
    gamma correction using gamma = 0.8
    gamma correction using gamma = 1.2
    """
    filename = os.path.basename(filepath)
    name, ext = filename.split('.')
    dirname = os.path.dirname(filepath)

    im = Image.open(filepath)
    crops = []

    center_x = im.size[0] // 2
    center_y = im.size[1] // 2

    centered_im = im.crop((center_x - grid_size // 2, center_y - grid_size // 2, center_x + grid_size // 2, center_y + grid_size // 2))

    # Original
    centered_im.save((os.path.join(dirname, '.'.join([name, 'centered', ext]))))

    # JPEG compression
    centered_im.save((os.path.join(dirname, '.'.join([name, 'jpeg70', ext]))), "JPEG", quality=70)
    centered_im.save((os.path.join(dirname, '.'.join([name, 'jpeg90', ext]))), "JPEG", quality=90)

    # Resize
    resized_im = resize_crop(im, 0.5)
    resized_im.save((os.path.join(dirname, '.'.join([name, 'r0.5', ext]))))
    resized_im = resize_crop(im, 0.8)
    resized_im.save((os.path.join(dirname, '.'.join([name, 'r0.8', ext]))))
    resized_im = resize_crop(im, 1.5)
    resized_im.save((os.path.join(dirname, '.'.join([name, 'r1.5', ext]))))
    resized_im = resize_crop(im, 2.0)
    resized_im.save((os.path.join(dirname, '.'.join([name, 'r2.0', ext]))))

    # Gamma correction
    corrected_im = gamma_correction(centered_im, 0.8)
    corrected_im.save((os.path.join(dirname, '.'.join([name, 'g0.8', ext]))))
    corrected_im = gamma_correction(centered_im, 1.2)
    corrected_im.save((os.path.join(dirname, '.'.join([name, 'g1.2', ext]))))


def batch_crop():
    def crop_in_folder(folder):
        for filename in os.listdir(os.path.join(train_folder, folder)):
            print('Processing', folder, filename)
            crop_split(os.path.join(train_folder, folder, filename))
            os.remove(os.path.join(train_folder, folder, filename))

    # Walk into each directory and process the images
    procs = []

    for folder in os.listdir(train_folder):
        p = Process(target=crop_in_folder, args=(folder,))
        p.start()
        procs.append(p)
    
    for proc in procs:
        proc.join()


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
