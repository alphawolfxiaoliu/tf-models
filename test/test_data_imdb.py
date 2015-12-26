import unittest
import tfmodels.data.imdb


class TestDataImdb(unittest.TestCase):
    def test_load_data(self):
        train_x, train_y, train_y2, test_x, test_y, test_y2, unlabeled = tfmodels.data.imdb.load()
        self.assertEqual(len(train_x), 25000)
        self.assertEqual(len(train_y), 25000)
        self.assertEqual(len(train_y2), 25000)
        self.assertEqual(len(test_x), 25000)
        self.assertEqual(len(test_y), 25000)
        self.assertEqual(len(test_y2), 25000)
        self.assertEqual(len(unlabeled), 50000)
        self.assertEqual(sum(train_y), 12500)
        self.assertEqual(sum(test_y), 12500)