import os

# DATA_ROOT_DIR = '/data/feabries/tgs-salt-identification-challenge'
DATA_ROOT_DIR = '../input'
TRAIN_DF_PATH = os.path.join(DATA_ROOT_DIR, 'train.csv')
TRAIN_IMGS_DIR = os.path.join(DATA_ROOT_DIR, 'train/images')
TRAIN_MASKS_DIR = os.path.join(DATA_ROOT_DIR, 'train/masks')
# TEST_MASKS_DIR = os.path.join(DATA_ROOT_DIR, 'test/images')
TEST_IMGS_DIR = os.path.join(DATA_ROOT_DIR, 'train/images')

MODELS_DIR = '../models'

IMG_SIZE = 101

MASK_COLOR = (33, 50, 230)

PREDICT_THRESHOLD = 0.3
