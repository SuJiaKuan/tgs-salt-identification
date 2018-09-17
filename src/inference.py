import os
import sys

import cv2
import numpy as np
from keras.models import load_model

from image_proc import apply_mask
from config import TEST_IMGS_DIR
from config import IMG_SIZE
from config import PREDICT_THRESHOLD
from config import MASK_COLOR


def main(model_path, img_id):
    model = load_model(model_path)

    img_path = os.path.join(TEST_IMGS_DIR, '{}.png'.format(img_id))
    img = cv2.imread(img_path)
    input_img = img[:, :, 0].reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255

    pred = model.predict(input_img)
    mask = np.where(pred[0] > PREDICT_THRESHOLD, 1, 0)

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
