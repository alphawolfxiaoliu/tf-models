import os
import pandas as pd
import numpy as np
import tfmodels.data.utils
from sklearn.cross_validation import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
MR_DATA_DIR = os.path.abspath(os.path.join(FILE_DIR, "../../data/mr"))

POSITIVE_SENTENCE_FILE = os.path.join(MR_DATA_DIR, "rt-polarity.pos")
NEGATIVE_SENTENCE_FILE = os.path.join(MR_DATA_DIR, "rt-polarity.neg")


class MRData:
    """
    Loads and preprocessed data for the MR dataset.
    """
    def __init__(self, padding=True, clean_str=True):
        # Load data from files
        with open(POSITIVE_SENTENCE_FILE) as f:
            positive_examples = [s.strip() for s in f]
        with open(NEGATIVE_SENTENCE_FILE) as f:
            negative_examples = [s.strip() for s in f]

        # Append examples
        self.sentences = positive_examples + negative_examples
        if clean_str:
            self.sentences = [tfmodels.data.utils.clean_str(s) for s in self.sentences]

        # Split and pad sentences
        self.sentences_tokenized = [s.split(" ") for s in self.sentences]
        if padding:
            self.sentences_tokenized = tfmodels.data.utils.pad_sequences(
                self.sentences_tokenized, pad_location="LEFT")

        # Labels
        self.y = [1 for _ in positive_examples] + [0 for _ in positive_examples]
        self.y = pd.get_dummies(self.y).values

        # Build the vocabulary
        self.vocab, self.vocab_inv = tfmodels.data.utils.build_vocabulary(self.sentences_tokenized)
        # Vectorize sentences
        self.x = np.array([[self.vocab[token] for token in sent] for sent in self.sentences_tokenized])

    def build_train_dev(self, test_size=0.1, random_seed=42):
        return train_test_split(
            self.x,
            self.y,
            test_size=test_size,
            random_state=random_seed)
