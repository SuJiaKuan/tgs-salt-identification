import numpy as np
import tensorflow as tf

from config import PREDICT_THRESHOLD


IOU_METRIC_THRESHOLDS = list(np.arange(0.5, 1, 0.05))


def _seg_iou(seg1, seg2):
    batch_size = seg1.shape[0]
    metrics = []

    for idx in range(batch_size):
        ans1 = seg1[idx] > 0
        ans2 = seg2[idx] > 0
        intersection = np.logical_and(ans1, ans2)
        union = np.logical_or(ans1, ans2)
        iou = (np.sum(intersection > 0) + 1e-10 ) / (np.sum(union > 0) + 1e-10)
        scores = [iou > threshold for threshold in IOU_METRIC_THRESHOLDS]
        metrics.append(np.mean(scores))

    return np.mean(metrics)


def iou_metric(y_true, y_pred):
    return tf.py_func(_seg_iou, [y_true, y_pred > PREDICT_THRESHOLD], np.float64)
