import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class F1Metric(Callback):
    
    def __init__(self, val_data):
        super().__init__()
        self.validation_data = val_data
    
    def on_train_begin(self, logs={}):
        print(self.validation_data)
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
    
    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
