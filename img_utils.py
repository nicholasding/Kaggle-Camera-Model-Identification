import random

from PIL import Image
from io import BytesIO


TRANSFORMATIONS = ['jpeg_70', 'jpeg_90', 'resize_0.5', 'resize_0.8', 'resize_1.5', 'resize_2.0', 'gamma_0.8', 'gamma_1.2']


def gamma_correction(im, gamma):
    """
    Fast gamma correction with PIL's image.point() method
    """
    invert_gamma = 1.0 / gamma
    lut = [pow(x/255., invert_gamma) * 255 for x in range(256)]
    lut = lut * 3 # need one set of data for each band for RGB
    im = im.point(lut)
    return im


def trans_jpeg(img, quality):
    buffer = BytesIO()
    img.save(buffer, 'JPEG', quality=int(quality))
    return Image.open(buffer)


def trans_resize(img, ratio):
    w, h = int(img.size[0] * float(ratio)), int(img.size[1] * float(ratio))
    return img.resize((w, h), Image.BICUBIC)


def trans_gamma(img, gamma):
    try:
        return gamma_correction(img, float(gamma))
    except Exception as e:
        print('Error in gamma correction', e)
    return img


def random_transformation(img):
    """
    Apply one of the transformation

    JPEG compression with quality factor = 70
    JPEG compression with quality factor = 90
    resizing (via bicubic interpolation) by a factor of 0.5
    resizing (via bicubic interpolation) by a factor of 0.8
    resizing (via bicubic interpolation) by a factor of 1.5
    resizing (via bicubic interpolation) by a factor of 2.0
    gamma correction using gamma = 0.8
    gamma correction using gamma = 1.2
    """
    action = random.choice(TRANSFORMATIONS)
    op, val = action.split('_')
    if op == 'jpeg':
        return trans_jpeg(img, val)
    elif op == 'resize':
        return trans_resize(img, val)
    else:
        return trans_gamma(img, val)


def random_crop(im, random_crop_size, sync_seed=None):
    w, h = im.size
    
    if w <= random_crop_size[0] or h <= random_crop_size[1]:
        print(w, h, random_crop_size)
    
    rangew = (w - random_crop_size[0])
    rangeh = (h - random_crop_size[1])
    offsetw = random.randint(0, rangew)
    offseth = random.randint(0, rangeh)
    return im.crop((offsetw, offseth, offsetw + random_crop_size[0], offseth + random_crop_size[1]))


def center_crop(im, center_crop_size):
    center_w, center_h = im.size[0] // 2, im.size[1] // 2
    half_w, half_h = center_crop_size[0] // 2, center_crop_size[1] // 2
    return im.crop((center_w - half_w, center_h - half_h, center_w + half_w, center_h + half_h))


def resize_crop(im, ratio, grid_size):
    resized = im.resize((int(im.size[0] * ratio), int(im.size[1] * ratio)), Image.BICUBIC)

    center_x = resized.size[0] // 2
    center_y = resized.size[1] // 2

    return resized.crop((center_x - grid_size // 2, center_y - grid_size // 2, center_x + grid_size // 2, center_y + grid_size // 2))
