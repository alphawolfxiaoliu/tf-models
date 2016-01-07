import os
import pandas as pd
import numpy as np
import tfmodels.data.utils
from sklearn.cross_validation import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MR_DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, "../../data/mr"))

POSITIVE_SENTENCE_FILE = os.path.join(MR_DATA_DIR, "rt-polarity.pos")
NEGATIVE_SENTENCE_FILE = os.path.join(MR_DATA_DIR, "rt-polarity.neg")


def load():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load data from files
    with open(POSITIVE_SENTENCE_FILE) as f:
        positive_examples = [s.strip() for s in f]
    with open(NEGATIVE_SENTENCE_FILE) as f:
        negative_examples = [s.strip() for s in f]
    # Split by words
    sentences = positive_examples + negative_examples
    labels = [1 for _ in positive_examples] + [0 for _ in positive_examples]
    return [sentences, labels]


def build_train_dev(sentences, labels, random_seed=42, padding=True):
    split_sentences = [s.split(" ") for s in sentences]
    if padding:
        sentences = tfmodels.data.utils.pad_sequences(split_sentences, pad_location="LEFT")
    vocab, vocab_inv = tfmodels.data.utils.build_vocabulary(sentences)
    x_train_all = np.array([[vocab[token] for token in sent] for sent in sentences])
    y_train_all = pd.get_dummies(labels).values
    x_train, x_test, y_train, y_test = train_test_split(
        x_train_all, y_train_all, test_size=0.1, random_state=random_seed)
    return [x_train, x_test, y_train, y_test, vocab, vocab_inv]


def train_iter(x_train, y_train, batch_size=64, num_epochs=100, random_seed=42):
    train_iter = tfmodels.data.utils.batch_iter(
        list(zip(x_train, y_train)), batch_size, num_epochs, fill=True, seed=random_seed)
    return map(lambda batch: zip(*batch), train_iter)


def dev_iter(x_dev, y_dev, batch_size=64, num_epochs=1, random_seed=42):
    dev_iter = tfmodels.data.utils.batch_iter(
        list(zip(x_dev, y_dev)), batch_size, num_epochs, fill=True, seed=random_seed)
    return map(lambda batch: zip(*batch), dev_iter)
