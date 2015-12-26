import os
import tarfile
import re

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
IMDB_DATA_FILE = os.path.abspath(os.path.join(FILE_DIR, "../../data/imdb.gz"))
FILE_RATING_RE = re.compile(r"_(\d+)")


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
        unlabeled.append([tar.extractfile(m).read().decode()])
    tar.close()
    return [train, test, unlabeled]


def load():
    train, test, unlabeled = read_tarfile()
    train_x, train_y, train_y2 = zip(*train)
    test_x, test_y, test_y2 = zip(*test)
    return [train_x, train_y, train_y2, test_x, test_y, test_y2, unlabeled]
