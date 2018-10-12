import cv2
import numpy as np

from config import IMG_SIZE
from config import IMG_MODEL_SIZE


def apply_mask(img, mask, color):
    mask_repeat = np.repeat(mask, img.shape[2]).reshape(img.shape)
    overlay = np.full(img.shape, color, dtype=np.uint8)
    overlay = np.multiply(overlay, mask_repeat).astype(np.uint8)
    masked_img = img.copy()
    cv2.addWeighted(masked_img, 0.8, overlay, 0.2, 0, masked_img)

    return masked_img


def upsample(img):
    if IMG_SIZE == IMG_MODEL_SIZE:
        return img

    return cv2.resize(img, (IMG_MODEL_SIZE, IMG_MODEL_SIZE))


def downsample(img):
    if IMG_SIZE == IMG_MODEL_SIZE:
        return img

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
