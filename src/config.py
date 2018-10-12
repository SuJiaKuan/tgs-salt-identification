import os

DATA_ROOT_DIR = '../input'
TRAIN_DF_PATH = os.path.join(DATA_ROOT_DIR, 'train.csv')
TRAIN_IMGS_DIR = os.path.join(DATA_ROOT_DIR, 'train/images')
TRAIN_MASKS_DIR = os.path.join(DATA_ROOT_DIR, 'train/masks')
TEST_MASKS_DIR = os.path.join(DATA_ROOT_DIR, 'test/images')

MODELS_DIR = '../models'

IMG_SIZE = 101
IMG_MODEL_SIZE = 128

MASK_COLOR = (33, 50, 230)

PREDICT_THRESHOLD = 0.5
