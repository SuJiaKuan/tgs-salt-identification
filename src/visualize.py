import os
import sys

import cv2
import numpy as np

from image_proc import apply_mask
from config import TRAIN_IMGS_DIR
from config import TRAIN_MASKS_DIR
from config import MASK_COLOR


def main(img_id):
    img_path = os.path.join(TRAIN_IMGS_DIR, '{}.png'.format(img_id))
    mask_path = os.path.join(TRAIN_MASKS_DIR, '{}.png'.format(img_id))

    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    mask = np.where(mask[:, :, 0] == 255, 1, 0)

    masked_img = apply_mask(img, mask, MASK_COLOR)

    cv2.imshow('Dispaly', masked_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 {} img_id'.format(sys.argv[0]))
        sys.exit(-1)

    img_id = sys.argv[1]

    main(img_id)
