import os
from tqdm import tqdm

from tensorflow.keras.preprocessing.sequence import pad_sequences

from ner_utils.pre_process import read_dataset
import constants as c


def convert_to_input(sentences, tags, label_map, max_seq_length):
    tokens_list, label_id_list = [], []
    
    for x, y in tqdm(zip(sentences, tags), total=len(tags)):
        
        tokens = []
        label_ids = []
        seq_list = x.split(' ')
        seq_label_list = y.split(' ')
        
        for word, label in zip(seq_list, seq_label_list):
            tokens.extend(word)
            label_ids.append(label_map[label])
        
        if len(tokens) > max_seq_length:
            tokens = tokens[: max_seq_length]
            label_ids = label_ids[: max_seq_length]
        
        tokens_list.append(tokens)
        label_id_list.append(label_ids)
    
    return tokens_list, label_id_list


def retrieve_features(data_type, label_list, max_seq_length):
    label_map = {}
    X, y = read_dataset(os.path.join(c.PROCESSED_DATASET_DIR, data_type))
    
    # here start with zero this means that "[PAD]" is zero
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    tokens_list, label_id_list = convert_to_input(X, y, label_map, max_seq_length)
    tokens = pad_sequences(tokens_list, maxlen=max_seq_length, dtype="long", truncating="post", padding="post")
    labels = pad_sequences(label_id_list, maxlen=max_seq_length, dtype="long", truncating="post", padding="post")
    
    return tokens, labels
