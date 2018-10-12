import os
import sys

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img

from image_proc import apply_mask
from image_proc import downsample
from image_proc import upsample
from metrics import iou_metric
from config import TEST_IMGS_DIR
from config import IMG_SIZE
from config import PREDICT_THRESHOLD
from config import MASK_COLOR
from config import IMG_MODEL_SIZE


def main(model_path, img_id):
    model = load_model(model_path, custom_objects={'iou_metric': iou_metric})

    img_path = os.path.join(TEST_IMGS_DIR, '{}.png'.format(img_id))
    input_img = np.array(load_img(img_path, color_mode='grayscale')) / 255
    input_img = upsample(input_img)
    input_img = input_img.reshape(1, IMG_MODEL_SIZE, IMG_MODEL_SIZE, 1)
    input_img = np.repeat(input_img, 3, axis=3)

    pred = model.predict(input_img)
    mask = downsample(pred[0])
    mask = np.where(mask > PREDICT_THRESHOLD, 1, 0)

    img = cv2.imread(img_path)
    masked_img = apply_mask(img, mask, MASK_COLOR)

    cv2.imshow('Display', masked_img)
    cv2.waitKey()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} model_path img_id'.format(sys.argv[0]))
        sys.exit(-1)

    model_path = sys.argv[1]
    img_id = sys.argv[2]

    main(model_path, img_id)
