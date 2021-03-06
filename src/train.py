import os
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from image_proc import upsample
from models import unet
from metrics import iou_metric
from config import TRAIN_DF_PATH
from config import TRAIN_IMGS_DIR
from config import TRAIN_MASKS_DIR
from config import RESULTS_DIR
from config import IMG_SIZE
from config import IMG_MODEL_SIZE


COVERAGE_CLASS_THRESHOLDS = [x / 10.0 for x in range(11)]


def coverage_to_class(coverage):
    for threshold in COVERAGE_CLASS_THRESHOLDS:
        if coverage <= threshold:
            return COVERAGE_CLASS_THRESHOLDS.index(threshold)


def data_augmentation(x_train, y_train):
    x_train_aug = np.append(x_train, [np.fliplr(x) for x in x_train], axis=0)
    y_train_aug = np.append(y_train, [np.fliplr(x) for x in y_train], axis=0)

    return x_train_aug, y_train_aug


def save_history_raw(history, raw_output_path):
    with open(raw_output_path, 'wb') as file_out:
        pickle.dump(history.history, file_out)
    print('History raw saved to {}'.format(raw_output_path))


def save_history_plot(history, plot_output_path):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))

    ax_loss.plot(history.epoch,
                 history.history['loss'],
                 label='Train loss')
    ax_loss.plot(history.epoch,
                 history.history['val_loss'],
                 label='Validation loss')
    ax_loss.legend()

    ax_score.plot(history.epoch,
                  history.history['iou_metric'],
                  label='Train score')
    ax_score.plot(history.epoch,
                  history.history['val_iou_metric'],
                  label='Validation score')
    ax_score.legend()

    fig.savefig(plot_output_path, dpi=fig.dpi)
    print('History plot saved to {}'.format(plot_output_path))


def main(epochs=100, batch_size=32):
    train_df = pd.read_csv(TRAIN_DF_PATH)

    imgs = []
    masks = []
    coverage_classes = []

    print('Loading training data')
    for img_id in tqdm(train_df.id):
        img_path = os.path.join(TRAIN_IMGS_DIR, '{}.png'.format(img_id))
        img = np.array(load_img(img_path, color_mode='grayscale')) / 255
        imgs.append(img)

        mask_path = os.path.join(TRAIN_MASKS_DIR, '{}.png'.format(img_id))
        mask = np.array(load_img(mask_path, color_mode='grayscale')) / 255
        masks.append(mask)

        coverage = np.sum(mask) / (IMG_SIZE * IMG_SIZE)
        coverage_class = coverage_to_class(coverage)
        coverage_classes.append(coverage_class)
    print('Training data loaded')

    x_train, x_val, y_train, y_val = train_test_split(
        np.array(list(map(upsample, imgs))).reshape(-1, IMG_MODEL_SIZE, IMG_MODEL_SIZE, 1),
        np.array(list(map(upsample, masks))).reshape(-1, IMG_MODEL_SIZE, IMG_MODEL_SIZE, 1),
        test_size=0.2,
        stratify=coverage_classes,
    )

    x_train, y_train = data_augmentation(x_train, y_train)

    x_train = np.repeat(x_train, 3, axis=3)
    x_val = np.repeat(x_val, 3, axis=3)

    model = unet((IMG_MODEL_SIZE, IMG_MODEL_SIZE, 3))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=[iou_metric])

    result_dir = os.path.join(RESULTS_DIR, str(time.time()))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    model_output_path = os.path.join(result_dir, 'unet_resnet50.model')
    raw_output_path = os.path.join(result_dir, 'history.pickle')
    plot_output_path = os.path.join(result_dir, 'plot.png')

    model_checkpoint = ModelCheckpoint(model_output_path,
                                       monitor='val_iou_metric',
                                       mode='max',
                                       save_best_only=True,
                                       verbose=1)

    history = model.fit(x_train,
                        y_train,
                        validation_data=[x_val, y_val],
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[model_checkpoint],
                        shuffle=True,
                        verbose=1)

    save_history_raw(history, raw_output_path)
    save_history_plot(history, plot_output_path)


if __name__ == '__main__':
    main()
