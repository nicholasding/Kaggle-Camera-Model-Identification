import math
import os
import random
import numpy as np

from multiprocessing import Process
from PIL import Image
from keras.preprocessing.image import load_img


validation_folder = '/home/nicholas/Workspace/Resources/Camera/validation'

VALIDATION_SPLIT = 0.2
CROP_SIZE = 224


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
        # self.generate_validation_set()
    
    def process_images(self):
        def crop_in_folder(folder):
            for filename in os.listdir(os.path.join(self.train_folder, folder)):
                output_folder = os.path.join(self.output_train, folder)
                if not os.path.exists(output_folder): os.mkdir(output_folder)
                print('Processing', folder, filename)
                try:
                    self.crop(os.path.join(self.train_folder, folder, filename), output_folder)
                except Exception as error:
                    print('Error in processing', error)
                    
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


class DefaultPlan(BasePlan):
    """
    Do not crop anything
    """
    def crop(self, filepath, output_folder):
        filename = os.path.basename(filepath)
        name, ext = filename.split('.')
        dirname = os.path.dirname(filepath)

        im = Image.open(filepath)
        im.save(os.path.join(output_folder, '.'.join([name, 'original', ext])), 'JPEG', quality=100)


class CenterPatchPlan(BasePlan):
    """
    Crop the center patch 512 x 512 without any augmentations
    """
    def crop(self, filepath, output_folder):
        filename = os.path.basename(filepath)
        name, ext = filename.split('.')
        dirname = os.path.dirname(filepath)

        im = load_img(filepath)

        center_x = im.size[0] // 2
        center_y = im.size[1] // 2

        centered_im = im.crop((center_x - self.grid_size // 2, center_y - self.grid_size // 2, center_x + self.grid_size // 2, center_y + self.grid_size // 2))

        # Original
        ext = 'png'
        centered_im.save(os.path.join(output_folder, '.'.join([name, 'centered', ext])), 'PNG')


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


class RandomPatchPlan(BasePlan):
    """
    Crop random patches but keep center patches for validation
    """
    PATCHES = 100

    def random_crop(self, img, random_crop_size, ratio=1.0, sync_seed=None):
        np.random.seed(sync_seed)
        w, h = img.size[0], img.size[1]
        rangew = (w - random_crop_size[0] * ratio)
        rangeh = (h - random_crop_size[1] * ratio)
        offsetw = 0 if rangew == 0 else np.random.randint(rangew)
        offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
        return img.crop((offsetw, offseth, offsetw + random_crop_size[0], offseth + random_crop_size[1]))

    def center_crop(self, img, center_crop_size):
        center_w, center_h = img.size[0] // 2, img.size[1] // 2
        half_w, half_h = center_crop_size[0] // 2, center_crop_size[1] // 2
        return img.crop((center_w - half_w, center_h - half_h, center_w + half_w, center_h + half_h))

    def crop(self, filepath, output_folder):
        crop_size = 224 # ResNet50

        filename = os.path.basename(filepath)
        name, ext = filename.split('.')
        dirname = os.path.dirname(filepath)

        im = Image.open(filepath)
        col = im.size[0] // crop_size
        row = im.size[1] // crop_size

        # Random Patches
        for i in range(self.PATCHES * 3):
            crop = self.random_crop(im, (crop_size, crop_size))
            crop.save(os.path.join(output_folder, '%s.%d.jpg' % (name, i)), 'JPEG')
        
        # Random Patches with transformation
        for i in range(self.PATCHES // 3):
            ratio = np.random.choice([0.5, 0.8, 1.5, 2.0])
            quality = np.random.choice([70, 90, 100])
            crop = self.random_crop(im, (crop_size * ratio, crop_size * ratio))
            crop = crop.resize((crop_size, crop_size), Image.BICUBIC)
            crop.save(os.path.join(output_folder, '%s.%d.jpg' % (name, i)), 'JPEG', quality=int(quality))
        
        # Random Gamma Correction
        for i in range(self.PATCHES // 3):
            gamma = np.random.choice([0.8, 1.2])
            crop = self.random_crop(im, (crop_size, crop_size))
            corrected = gamma_correction(crop, gamma)
            corrected.save(os.path.join(output_folder, '%s.g%d.jpg' % (name, i)), 'JPEG')
        
        crop = self.center_crop(im, (crop_size, crop_size))
        crop.save(os.path.join(output_folder, '%s.center.jpg' % name), 'JPEG', quality=100)
        crop.save(os.path.join(output_folder, '%s.center.jpep90.jpg' % name), 'JPEG', quality=90)
        crop.save(os.path.join(output_folder, '%s.center.jpeg70.jpg' % name), 'JPEG', quality=70)
        
        # Enlarge validation set
        # Resize
        ext = 'jpg'
        resized_im = resize_crop(im, 0.5, grid_size=crop_size)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'center.r0.5', ext])), 'JPEG')
        resized_im = resize_crop(im, 0.8, grid_size=crop_size)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'center.r0.8', ext])), 'JPEG')
        resized_im = resize_crop(im, 1.5, grid_size=crop_size)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'center.r1.5', ext])), 'JPEG')
        resized_im = resize_crop(im, 2.0, grid_size=crop_size)
        resized_im.save(os.path.join(output_folder, '.'.join([name, 'center.r2.0', ext])), 'JPEG')

        # Gamma correction
        corrected_im = gamma_correction(crop, 0.8)
        corrected_im.save(os.path.join(output_folder, '.'.join([name, 'center.g0.8', ext])), 'TIFF', compression='None')
        corrected_im = gamma_correction(crop, 1.2)
        corrected_im.save(os.path.join(output_folder, '.'.join([name, 'center.g1.2', ext])), 'TIFF', compression='None')


    # def generate_validation_set(self):
    #     for folder in os.listdir(self.output_train):
    #         files = [filename for filename in os.listdir(os.path.join(self.output_train, folder)) if 'center' in filename]
    #         validation_set = files

    #         print('Moving %d files out of %d' % (len(validation_set), len(files)))

    #         assert len(validation_set) == len(set(validation_set))

    #         # Create target folder
    #         target_folder = os.path.join(self.output_val, folder)
    #         if not os.path.exists(target_folder): os.mkdir(target_folder)

    #         # Move files
    #         for filename in validation_set:
    #             os.rename(os.path.join(self.output_train, folder, filename), os.path.join(target_folder, filename))


if __name__ == '__main__':
    train_folder = '/media/nicholas/Data/Resources/Camera/train_merged'
    plan = CenterPatchPlan(train_folder, '/media/nicholas/Data/Resources/Camera/center_merged_val')
    # plan = CenterPatchAugPlan(train_folder, '/home/nicholas/Workspace/Resources/Camera/center_patch')
    # plan = GridPatchPlan(train_folder, '/home/nicholas/Workspace/Resources/Camera/patches')
    # plan = DefaultPlan(train_folder, '/home/nicholas/Workspace/Resources/Camera/default')
    # plan = RandomPatchPlan(train_folder, '/media/nicholas/Data/Resources/Camera/merged_patches_1')
    plan.start()
