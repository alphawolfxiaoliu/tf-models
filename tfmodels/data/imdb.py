import os
import tarfile
import re
import pandas as pd
import numpy as np
import tfmodels.data.utils
from sklearn.cross_validation import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
IMDB_DATA_FILE = os.path.abspath(os.path.join(FILE_DIR, "../../data/imdb.gz"))
FILE_RATING_RE = re.compile(r"_(\d+)")


class IMDBData:
    """
    Loads and preprocessed data for the MR dataset.
    """
    def __init__(self, padding=True, clean_str=True, max_length=512):
        train, test, unlabeled = IMDBData.read_tarfile()
        self.train_x, self.train_y, self.train_y2 = zip(*train)
        self.test_x, self.test_y, self.test_y2 = zip(*test)
        self.unlabeled = unlabeled

        # Append examples
        if clean_str:
            self.train_x = [tfmodels.data.utils.clean_str(s) for s in self.train_x]
            self.test_x = [tfmodels.data.utils.clean_str(s) for s in self.test_x]
            self.unlabeled = [tfmodels.data.utils.clean_str(s) for s in self.unlabeled]

        # Split and pad sentences
        self.train_x = [s.split(" ") for s in self.train_x]
        self.test_x = [s.split(" ") for s in self.test_x]
        self.unlabeled = [s.split(" ") for s in self.unlabeled]
        if padding:
            self.train_x = tfmodels.data.utils.pad_sequences(
                self.train_x, pad_location="LEFT", max_length=max_length)
            self.test_x = tfmodels.data.utils.pad_sequences(
                self.test_x, pad_location="LEFT", max_length=max_length)
            self.unlabeled = tfmodels.data.utils.pad_sequences(
                self.unlabeled, pad_location="LEFT", max_length=max_length)

        # Labels
        self.train_y = pd.get_dummies(self.train_y).values
        self.train_y2 = pd.get_dummies(self.train_y2).values
        self.test_y = pd.get_dummies(self.test_y).values
        self.test_y2 = pd.get_dummies(self.test_y2).values

        # Build the vocabulary
        self.vocab, self.vocab_inv = tfmodels.data.utils.build_vocabulary(
            self.train_x + self.test_x + self.unlabeled)

        # Vectorize sentences
        self.train_x = np.array([[self.vocab[token] for token in sent] for sent in self.train_x])
        self.test_x = np.array([[self.vocab[token] for token in sent] for sent in self.test_x])
        self.unlabeled = np.array([[self.vocab[token] for token in sent] for sent in self.unlabeled])

    @staticmethod
    def read_tarfile():
        tar = tarfile.open(IMDB_DATA_FILE, "r:gz")
        archive_members = tar.getmembers()

        # Helper function to parse the rating from the filename
        def get_rating(name): return int(FILE_RATING_RE.search(name).groups()[0])

        # Extract training data
        train = []
        for m in filter(lambda m: "aclImdb/train/pos/" in m.name, archive_members):
            train.append([tar.extractfile(m).read().decode(), 1, get_rating(m.name)])
        for m in filter(lambda m: "aclImdb/train/neg/" in m.name, archive_members):
            train.append([tar.extractfile(m).read().decode(), 0, get_rating(m.name)])
        # Extract test data
        test = []
        for m in filter(lambda m: "aclImdb/test/pos/" in m.name, archive_members):
            test.append([tar.extractfile(m).read().decode(), 1, get_rating(m.name)])
        for m in filter(lambda m: "aclImdb/test/neg/" in m.name, archive_members):
            test.append([tar.extractfile(m).read().decode(), 0, get_rating(m.name)])
        # Extract unlabeled data
        unlabeled = []
        for m in filter(lambda m: "aclImdb/train/unsup/" in m.name, archive_members):
            unlabeled.append(tar.extractfile(m).read().decode())
        tar.close()
        return [train, test, unlabeled]

    def build_train_dev(self, test_size=0.1, random_seed=42):
        return train_test_split(
            self.train_x,
            self.train_y,
            test_size=test_size,
            random_state=random_seed)
