from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow as tf
import constants as c


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

    return y_true.eval(session=tf.Session()), y_pred.eval(session=tf.Session())


def macro_recall(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return recall_score(y_true_filter, y_pred_filter, average='macro')


def macro_precision(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return precision_score(y_true_filter, y_pred_filter, average='macro')


def micro_f1(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    
    return f1_score(y_true_filter, y_pred_filter, average='micro')


def macro_f1(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return f1_score(y_true_filter, y_pred_filter, average='macro')


def get_classification_report(y_true, y_pred):
    y_true_filter, y_pred_filter = _prep_predictions(y_true, y_pred)
    return classification_report(y_true_filter, y_pred_filter, digits=len(c.LABELS), labels=c.LABELS)
