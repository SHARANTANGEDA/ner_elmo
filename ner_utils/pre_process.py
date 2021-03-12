import pandas as pd


def _create_example(df):
    sentences, labels = [], []
    for (idx, line) in df.iterrows():
        sentences.append(line[0])
        labels.append(line[1])
    return sentences, labels


def read_dataset(file_path):
    return _create_example(pd.read_csv(file_path))
