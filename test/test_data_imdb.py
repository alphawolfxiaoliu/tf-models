import unittest
from tfmodels.data.imdb import IMDBData
import tfmodels.data.imdb
import numpy as np


class TestDataImdb(unittest.TestCase):
    def test_load_data(self):
        data = IMDBData(max_length=512)

        # Train
        self.assertEqual(data.train_x.shape, (25000, 512))
        self.assertEqual(data.train_y.shape, (25000, 2))
        self.assertEqual(data.train_y2.shape, (25000, 8))

        # Test
        self.assertEqual(data.test_x.shape, (25000, 512))
        self.assertEqual(data.test_y.shape, (25000, 2))
        self.assertEqual(data.test_y2.shape, (25000, 8))

        # Unlabeled Data
        self.assertEqual(data.unlabeled.shape, (50000, 512))
