import os
import tensorflow_hub as hub

"""Initialize ENV Variables"""
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH"))
PROCESSED_DATASET_DIR = os.getenv("PROCESSED_DATASET_DIR")
MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
ML_FLOW_SAVE_DIR = os.getenv("ML_FLOW_SAVE_DIR")
LOGS_DIR = os.getenv("LOGS_DIR")
PRE_TRAINED_MODEL_DIR = os.getenv("PRE_TRAINED_MODEL_DIR")
ML_FLOW_EXPERIMENT_ID = os.getenv("ML_FLOW_EXPERIMENT_ID")


"""
here "X" used to represent "##eer","##soo" and so on!
"[PAD]" for padding
:return:
"""

LABELS = ["[PAD]", "B-NAME", "B-LOC", "O", "B-ORG", "B-MISC", "[CLS]", "[SEP]", "X"]
TRAIN_FILE = "train.csv"
VALIDATION_FILE = "validation.csv"
TEST_FILE = "test.csv"
TOTAL_FILE = "total.csv"

ELMO_MODEL = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)

BATCH_SIZE = 32
