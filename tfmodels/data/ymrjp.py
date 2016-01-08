import os
import pandas as pd
import numpy as np
import tarfile
import tfmodels.data.utils
from sklearn.cross_validation import train_test_split

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
YMR_DATA_PATH = os.path.abspath(os.path.join(FILE_DIR, "../../data/ymrjp/yahoo-movie-reviews.json.tar.gz"))


class YMRJPData:
    """
    Loads and preprocessed data for the MR dataset.
    """
    def __init__(self, padding=True, max_length=512, balance=False):
        # Load data from file
        with tarfile.open(YMR_DATA_PATH, "r:gz") as tf:
            f = tf.extractfile(tf.getmembers()[0])
            data = pd.read_json(f)
            data.movieName = data.movieName.str.strip()
            data.text = data.text.str.strip()
            data.title = data.title.str.strip()
            data.url = data.url.str.strip()
            data = data[data.text.str.len() > 0]

        # Create polar data
        data_polar = data.loc[data.rating != 3].copy()
        data_polar.loc[data_polar.rating <= 2, 'rating'] = 0
        data_polar.loc[data_polar.rating >= 4, 'rating'] = 1

        # Optionally balance the data set classes
        if balance:
            grouped_ratings = data_polar.groupby('rating')
            K = grouped_ratings.rating.count().min()
            indices = itertools.chain(
                *[np.random.choice(v, K, replace=False) for k, v in grouped_ratings.groups.items()])
            data_polar = data_polar.reindex(indices).copy()

        # Split and pad sentences
        documents = [list(s) for s in data_polar.text]
        if padding:
            documents = tfmodels.data.utils.pad_sequences(documents, pad_location="LEFT", max_length=max_length)

        # Labels
        y = pd.get_dummies(data_polar.rating).values

        # Build the vocabulary and vectorize
        self.vocab, self.vocab_inv = tfmodels.data.utils.build_vocabulary(documents)
        x = np.array([[self.vocab[token] for token in d] for d in documents])

        # Create an "official" test set
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x, y, test_size=0.1, random_state=1337, stratify=data_polar.rating)

    def build_train_dev(self, test_size=0.1, random_seed=42):
        return train_test_split(
            self.x_train,
            self.y_train,
            test_size=test_size,
            random_state=random_seed)
