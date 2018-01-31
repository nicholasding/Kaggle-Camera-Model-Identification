import random
import cv2
import numpy as np

# https://mmeysenburg.github.io/image-processing/02-opencv-images/

TRANSFORMATIONS = ['jpeg_70', 'jpeg_90', 'resize_0.5', 'resize_0.8', 'resize_1.5', 'resize_2.0', 'gamma_0.8', 'gamma_1.2']

def trans_jpeg(img, quality):
    """
    In memory compression. Ref: https://stackoverflow.com/questions/40768621/python-opencv-jpeg-compression-in-memory
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_img = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(encoded_img, 1)


def trans_resize(img, ratio):
    return cv2.resize(img, (0,0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def trans_gamma(img, gamma):
    return np.uint8(cv2.pow(img / 255., gamma) * 255.)


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
        return trans_jpeg(img, int(val))
    elif op == 'resize':
        return trans_resize(img, float(val))
    else:
        return trans_gamma(img, float(val))


def random_crop(im, random_crop_size, sync_seed=None):
    w, h = im.shape[1], im.shape[0]
    
    rangew = (w - random_crop_size[0])
    rangeh = (h - random_crop_size[1])

    offsetw = random.randint(0, rangew)
    offseth = random.randint(0, rangeh)

    return im[offseth : offseth + random_crop_size[1], offsetw : offsetw + random_crop_size[0]]


def center_crop(im, center_crop_size):
    center_w, center_h = im.shape[1] // 2, im.shape[0] // 2
    half_w, half_h = center_crop_size[0] // 2, center_crop_size[1] // 2
    return im[center_h - half_h : center_h + half_h, center_w - half_w : center_w + half_w]
