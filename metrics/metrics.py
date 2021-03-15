from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow as tf
import constants as c
import numpy as np


def _prep_predictions(y_true, y_pred):
    y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
    y_true_rank = ops.convert_to_tensor(y_true).shape.ndims
    # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
    if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            K.int_shape(y_true)) == len(K.int_shape(y_pred))):
        y_true = array_ops.squeeze(y_true, [-1])
    y_pred = math_ops.argmax(y_pred, axis=-1)
    
    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if K.dtype(y_pred) != K.dtype(y_true):
        y_pred = math_ops.cast(y_pred, K.dtype(y_true))
        
    return y_true, y_pred


def macro_recall(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return recall_score(y_true_filter.eval(session=tf.Session()), y_pred_filter.eval(session=tf.Session()),
                        average='macro')


def macro_precision(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return precision_score(y_true_filter.eval(session=tf.Session()), y_pred_filter.eval(session=tf.Session()),
                           average='macro')


def micro_f1(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    
    return f1_score(y_true_filter.eval(session=tf.Session()), y_pred_filter.eval(session=tf.Session()), average='micro')


def macro_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def get_classification_report(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return classification_report(y_true_filter.eval(session=tf.Session()), y_pred_filter.eval(session=tf.Session()), digits=len(c.LABELS), labels=c.LABELS)
