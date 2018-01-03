import math
import os
import random

from multiprocessing import Process
from PIL import Image


validation_folder = '/home/nicholas/Workspace/Resources/Camera/validation'

VALIDATION_SPLIT = 0.4
CROP_SIZE = 512


def resize_crop(im, ratio, grid_size=CROP_SIZE):
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


class BasePlan(object):
    
    def __init__(self, train_folder, output_folder):
        self.train_folder = train_folder
        self.output_train = os.path.join(output_folder, 'train')
        self.output_val = os.path.join(output_folder, 'validation')

        self.grid_size = CROP_SIZE

        if not os.path.exists(output_folder): os.mkdir(output_folder)
        if not os.path.exists(self.output_train): os.mkdir(self.output_train)
        if not os.path.exists(self.output_val): os.mkdir(self.output_val)
    
    def start(self):
        self.process_images()
        self.generate_validation_set()
    
    def process_images(self):
        def crop_in_folder(folder):
            for filename in os.listdir(os.path.join(self.train_folder, folder)):
                output_folder = os.path.join(self.output_train, folder)
                if not os.path.exists(output_folder): os.mkdir(output_folder)
                print('Processing', folder, filename)
                self.crop(os.path.join(self.train_folder, folder, filename), output_folder)
                # os.remove(os.path.join(self.train_folder, folder, filename))
        
        # Walk into each directory and process the images
        procs = []

        for folder in os.listdir(self.train_folder):
            p = Process(target=crop_in_folder, args=(folder,))
            p.start()
            procs.append(p)
        
        for proc in procs:
            proc.join()
    
    def generate_validation_set(self):
        for folder in os.listdir(self.output_train):
            files = os.listdir(os.path.join(self.output_train, folder))
            validation_set = random.sample(files, k=int(len(files) * VALIDATION_SPLIT))

            print('Moving %d files out of %d' % (len(validation_set), len(files)))

            assert len(validation_set) == len(set(validation_set))

            # Create target folder
            target_folder = os.path.join(self.output_val, folder)
            if not os.path.exists(target_folder): os.mkdir(target_folder)

            # Move files
            for filename in validation_set:
                os.rename(os.path.join(self.output_train, folder, filename), os.path.join(target_folder, filename))

    def crop(self, image_file, output_folder):
        raise NotImplementedError


class CenterPatchPlan(BasePlan):
    """
    Crop the center patch 512 x 512 without any augmentations
    """
    def crop(self, filepath, output_folder):
        filename = os.path.basename(filepath)
        name, ext = filename.split('.')
        dirname = os.path.dirname(filepath)

        im = Image.open(filepath)

        center_x = im.size[0] // 2
        center_y = im.size[1] // 2

        centered_im = im.crop((center_x - self.grid_size // 2, center_y - self.grid_size // 2, center_x + self.grid_size // 2, center_y + self.grid_size // 2))

        # Original
        centered_im.save(os.path.join(output_folder, '.'.join([name, 'centered', ext])), 'JPEG', quality=100)


class CenterPatchAugPlan(BasePlan):
    """
    Center patch with all the augmentations
    """
    def crop(self, filepath, output_folder):
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

        centered_im = im.crop((center_x - self.grid_size // 2, center_y - self.grid_size // 2, center_x + self.grid_size // 2, center_y + self.grid_size // 2))

        # Original
        centered_im.save(os.path.join(output_folder, '.'.join([name, 'centered', ext])), 'JPEG', quality=100)

        # JPEG compression
        centered_im.save((os.path.join(output_folder, '.'.join([name, 'jpeg70', ext]))), "JPEG", quality=70)
        centered_im.save((os.path.join(output_folder, '.'.join([name, 'jpeg90', ext]))), "JPEG", quality=90)

        # Resize
        resized_im = resize_crop(im, 0.5)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'r0.5', ext])), 'JPEG', quality=100)
        resized_im = resize_crop(im, 0.8)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'r0.8', ext])), 'JPEG', quality=100)
        resized_im = resize_crop(im, 1.5)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'r1.5', ext])), 'JPEG', quality=100)
        resized_im = resize_crop(im, 2.0)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'r2.0', ext])), 'JPEG', quality=100)

        # Gamma correction
        corrected_im = gamma_correction(centered_im, 0.8)
        corrected_im.save(os.path.join(output_folder, '.'.join([name, 'g0.8', ext])), 'JPEG', quality=100)
        corrected_im = gamma_correction(centered_im, 1.2)
        corrected_im.save(os.path.join(output_folder, '.'.join([name, 'g1.2', ext])), 'JPEG', quality=100)


class GridPatchPlan(BasePlan):
    """
    Use patch size 512 x 512 to have all the variations
    """
    def crop(self, filepath, output_folder):
        crop_size = 512

        filename = os.path.basename(filepath)
        name, ext = filename.split('.')
        dirname = os.path.dirname(filepath)

        im = Image.open(filepath)
        col = im.size[0] // crop_size
        row = im.size[1] // crop_size

        print(filename + ' image Size', im.size)

        for i in range(row):
            for j in range(col):
                x1 = crop_size * j
                x2 = crop_size * j + crop_size
                y1 = crop_size * i
                y2 = crop_size * i + crop_size
                crop = im.crop((x1, y1, x2, y2))
                # print('Saving', output_folder, '%s.%d.%d.jpg' % (name, i, j))
                crop.save(os.path.join(output_folder, '%s.%d.%d.jpg' % (name, i, j)), 'JPEG', quality=100)


if __name__ == '__main__':
    train_folder = '/home/nicholas/Workspace/Resources/Camera/train'
    # plan = CenterPatchAugPlan(train_folder, '/home/nicholas/Workspace/Resources/Camera/center_patch')
    plan = GridPatchPlan(train_folder, '/home/nicholas/Workspace/Resources/Camera/patches')
    plan.start()
