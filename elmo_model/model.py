import logging
import os
import pickle
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras import Input

from ner_utils import extract_features
from metrics.metrics import recall_m, precision_m, f1_m
import constants as c


def elmo_embedding_layer(x):
    return c.ELMO_MODEL.signatures['tokens'](
        tokens=tf.squeeze(tf.cast(x, tf.string)),
        sequence_len=tf.constant(c.BATCH_SIZE * [c.MAX_SEQ_LENGTH])
    )['elmo']


def elmo_model():
    input_text = Input(shape=(c.MAX_SEQ_LENGTH,), dtype=tf.string)
    embedding = Lambda(elmo_embedding_layer, output_shape=(c.MAX_SEQ_LENGTH, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(embedding)
    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(len(c.LABELS), activation="softmax"))(x)
    return Model(input_text, out)


def train_test(epochs, epsilon=1e-7, init_lr=2e-5, beta_1=0.9, beta_2=0.999):
    """Create Features & Tokenize"""
    logging.getLogger().setLevel(logging.INFO)
    
    # Build Model
    model = elmo_model()
    model.summary()
    
    # Add Optimizer and loss metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
    metrics = [keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32), recall_m, precision_m, f1_m]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    logging.info("Compiling Model ...")
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logging.info("Model has been compiled")
    
    # Retrieve Features
    train_tokens, train_labels = extract_features.retrieve_features(c.TRAIN_FILE, c.LABELS, c.MAX_SEQ_LENGTH)
    val_tokens, val_labels = extract_features.retrieve_features(c.VALIDATION_FILE, c.LABELS, c.MAX_SEQ_LENGTH)
    test_tokens, test_labels = extract_features.retrieve_features(c.TEST_FILE, c.LABELS, c.MAX_SEQ_LENGTH)
    
    # Training the model
    logging.info("Test Validation features are ready")
    model.fit(np.array(train_tokens), train_labels, epochs=epochs, batch_size=c.BATCH_SIZE,
              validation_data=(np.array(val_tokens), val_labels))
    logging.info("Model Fitting is done")
    
    # Save Model
    save_dir_path = os.path.join(c.MODEL_OUTPUT_DIR, str(datetime.utcnow()))
    os.mkdir(save_dir_path)
    tf.saved_model.save(model, export_dir=save_dir_path)
    logging.info("Model Saved")
    
    # Compute Scores
    test_loss, test_acc, test_recall, test_precision, test_f_score = model.evaluate(np.array(test_tokens), test_labels,
                                                                                    batch_size=c.BATCH_SIZE)
    logging.info("****TEST METRICS****")
    logging.info(f'Test Loss: {test_loss}')
    logging.info(f'Test Accuracy: {test_acc}')
    logging.info(f'Test Recall: {test_recall}')
    logging.info(f'Test Precision: {test_precision}')
    logging.info(f'Test F1_Score: {test_f_score}')
    return save_dir_path


def serve_with_saved_model(formatted_data, saved_classifier):
    result = saved_classifier(formatted_data, training=False)
    formatted_result = tf.argmax(result).numpy()
    label2id_map = pickle.load(open(c.LABEL_ID_PKL_FILE, "r"))
    id2label_map = {v: k for k, v in label2id_map.items()}
    return np.vectorize(id2label_map.get)[formatted_result]
