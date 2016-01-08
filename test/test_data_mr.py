import unittest
from tfmodels.data.mr import MRData


class TestDataMR(unittest.TestCase):
    def test_load_data(self):
        data = MRData(padding=True, clean_str=True)
        self.assertEqual(data.x.shape, (10662, 56))
        self.assertEqual(data.y.shape, (10662, 2))
