from tensorflow.python.ops import array_ops
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow as tf
from tensorflow_core.python.keras.callbacks import Callback

import constants as c
import numpy as np


class EvalMetrics(Callback):
    
    def __init__(self, val_data, labels, batch_size):
        super().__init__()
        self.validation_X = val_data
        self.label_data = labels
        self.batch_size = batch_size
    
    def on_train_begin(self, logs={}):
        print(self.validation_X)
        self.val_f1s = []
        self.val_f1s_micro = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        predictions = np.argmax(self.model.predict(self.validation_X, batch_size=self.batch_size, verbose=1).logits,
                                axis=-1)
        print(np.shape(predictions))
        sk_report, macro_f1_score, micro_f1_score, macro_recall_score, macro_precision_score = calculate_pred_metrics(
            self.label_data, predictions)
        print(sk_report)
        
        self.val_f1s.append(macro_f1_score)
        self.val_recalls.append(macro_recall_score)
        self.val_precisions.append(macro_precision_score)
        self.val_f1s_micro.append(micro_f1_score)
        print(f'-Micro F1: {micro_f1_score}, Macro F1: {macro_f1_score}, Recall: {macro_recall_score},'
              f' Precision: {macro_precision_score}')


# val_predict = self.model.predict(self.val_input)
# val_targ = self.val_output
# _val_f1 = f1_score(val_targ, val_predict, average="macro")
# _val_recall = recall_score(val_targ, val_predict)
# _val_precision = precision_score(val_targ, val_predict)
# self.val_f1s.append(_val_f1)
# self.val_recalls.append(_val_recall)
# self.val_precisions.append(_val_precision)
# print(" — val_f1_v2: % f — val_precision_v2: % f — val_recall_v2 % f" % (_val_f1, _val_precision, _val_recall))


def calculate_pred_metrics(y_true, y_pred):
    # true_f, pred_f = _prep_predictions(y_true, y_pred)
    print(y_true.shape, y_pred.shape)
    true_f, pred_f = np.reshape(y_true, (len(y_true) * c.MAX_SEQ_LENGTH,)), np.reshape(y_pred, (len(y_pred),))
    print(true_f.shape, pred_f.shape)
    return classification_report(y_true=true_f, y_pred=pred_f), f1_score(
        true_f, pred_f, average='macro'), f1_score(true_f, pred_f, average='micro'), recall_score(
        true_f, pred_f, average='macro'), precision_score(true_f, pred_f, average='macro')


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
    return classification_report(y_true_filter.eval(session=tf.Session()), y_pred_filter.eval(session=tf.Session()),
                                 digits=len(c.LABELS), labels=c.LABELS)
