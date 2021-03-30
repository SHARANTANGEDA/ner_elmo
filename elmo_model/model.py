import logging
import os
import time
import uuid

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Lambda
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

import constants as c
from metrics.metrics import macro_f1, get_classification_report, micro_f1, macro_precision, macro_recall, EvalMetrics, \
    calculate_pred_metrics
from ner_utils import extract_features


def elmo_embedding_layer(x):
    return c.ELMO_MODEL(inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": tf.constant(c.BATCH_SIZE*[c.MAX_SEQ_LENGTH])
                      },
                      signature="tokens",
                      as_dict=True)["elmo"]


def elmo_model():
    input_text = Input(shape=(c.MAX_SEQ_LENGTH,), dtype=tf.string)
    embedding = Lambda(elmo_embedding_layer, output_shape=(c.MAX_SEQ_LENGTH, 1024))(input_text)
    x = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(embedding)
    x_rnn = Bidirectional(LSTM(units=512, return_sequences=True, recurrent_dropout=0.2, dropout=0.2))(x)
    x = add([x, x_rnn])  # residual connection to the first biLSTM
    out = TimeDistributed(Dense(len(c.LABELS), activation="softmax"))(x)
    return Model(input_text, out)


def train_test(epochs, epsilon=1e-7, init_lr=2e-5, beta_1=0.9, beta_2=0.999):
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
   
    # IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras!
    # Otherwise, their weights will be unavailable in the threads after the session there has been set
    """Create Features & Tokenize"""
    logging.getLogger().setLevel(logging.INFO)
    #
    # Build Model
    model = elmo_model()
    model.summary()
    
    # Add Optimizer and loss metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=epsilon, beta_1=beta_1, beta_2=beta_2)
    
    metrics = [keras.metrics.SparseCategoricalAccuracy('micro_f1/cat_accuracy', dtype=tf.float32), macro_f1]
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
    save_dir_path = os.path.join(c.MODEL_OUTPUT_DIR, "model_"+str(time.time()))
    os.mkdir(save_dir_path)
    # tf.saved_model.save(model, export_dir=save_dir_path)
    model.save_pretrained(save_dir_path, saved_model=True)
    logging.info("Model Saved at: {}".format(save_dir_path))
    
    # Compute Scores
    test_loss, test_acc = model.evaluate(x=np.array(test_tokens), y=test_labels, batch_size=c.BATCH_SIZE)

    logging.info({"Loss": test_loss, "Accuracy": test_acc})

    # evaluate model with sklearn
    predictions = np.argmax(model.predict(np.array(test_tokens), batch_size=c.BATCH_SIZE, verbose=1).logits, axis=-1)
    print(np.shape(predictions), np.shape(test_labels))
    sk_report, macro_f1_score, micro_f1_score, macro_recall_score, macro_precision_score = calculate_pred_metrics(
        test_labels, predictions)

    print('\n')
    print(sk_report)
    logging.info(sk_report)

    logging.info("****TEST METRICS****")
    metrics_dict = {"Loss": test_loss, "CatAcc": test_acc, "Macro_F1": macro_f1_score, "Micro_F1": micro_f1_score,
                    "Macro_Precision": macro_precision_score, "Macro_Recall": macro_recall_score}
    logging.info(str(metrics_dict))
    mlflow.log_metrics(metrics_dict)

    return save_dir_path, [
        f'epochs:{epochs}', f'batch_size: {c.BATCH_SIZE}', f'epsilon: {epsilon}', f'init_lr: {init_lr}',
        f'beta_1: {beta_1}', f'beta_2: {beta_2}'], f'elmo_{test_acc}_{macro_f1_score}_{uuid.uuid4()}'
