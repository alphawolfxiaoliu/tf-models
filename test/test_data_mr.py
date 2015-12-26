import unittest
import tfmodels.data.mr


class TestDataMR(unittest.TestCase):
    def test_load_data(self):
        sentences, labels = tfmodels.data.mr.load()
        self.assertEqual(len(sentences), 10662)
        self.assertEqual(len(labels), 10662)
        self.assertEqual(sum(1 for _ in labels if _ == 1), 5331)
        self.assertEqual(sum(1 for _ in labels if _ == 0), 5331)
