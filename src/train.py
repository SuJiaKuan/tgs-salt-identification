import os
import time

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from models import unet
from config import TRAIN_DF_PATH
from config import TRAIN_IMGS_DIR
from config import TRAIN_MASKS_DIR
from config import MODELS_DIR
from config import IMG_SIZE


COVERAGE_CLASS_THRESHOLDS = [x / 10.0 for x in range(11)]


def coverage_to_class(coverage):
    for threshold in COVERAGE_CLASS_THRESHOLDS:
        if coverage <= threshold:
            return COVERAGE_CLASS_THRESHOLDS.index(threshold)


def main(epochs=100, batch_size=32):
    train_df = pd.read_csv(TRAIN_DF_PATH)

    imgs = []
    masks = []
    coverage_classes = []

    print('Loading training data')
    for img_id in tqdm(train_df.id):
        img_path = os.path.join(TRAIN_IMGS_DIR, '{}.png'.format(img_id))
        img = np.array(load_img(img_path,  color_mode='grayscale')) / 255
        imgs.append(img)

        mask_path = os.path.join(TRAIN_MASKS_DIR, '{}.png'.format(img_id))
        mask = np.array(load_img(mask_path, color_mode='grayscale')) / 255
        masks.append(mask)

        coverage = np.sum(mask) / (IMG_SIZE * IMG_SIZE)
        coverage_class = coverage_to_class(coverage)
        coverage_classes.append(coverage_class)
    print('Training data loaded')

    x_train, x_val, y_train, y_val = train_test_split(
        np.array(imgs).reshape(-1, IMG_SIZE, IMG_SIZE, 1),
        np.array(masks).reshape(-1, IMG_SIZE, IMG_SIZE, 1),
        test_size=0.2,
        stratify=coverage_classes,
    )

    model = unet((IMG_SIZE, IMG_SIZE, 1))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    model_output_path = '{}/unet_{}.model'.format(MODELS_DIR, int(time.time()))

    model_checkpoint = ModelCheckpoint(model_output_path,
                                       monitor='val_acc',
                                       mode='max',
                                       save_best_only=True,
                                       verbose=1)

    model.fit(x_train,
              y_train,
              validation_data=[x_val, y_val],
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[model_checkpoint],
              shuffle=True,
              verbose=1)


if __name__ == '__main__':
    main()
