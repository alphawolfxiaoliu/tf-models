import os
import tfmodels.data.utils

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
    sentences = [s.split(" ") for s in sentences]
    labels = [1 for _ in positive_examples] + [0 for _ in positive_examples]
    return [sentences, labels]
