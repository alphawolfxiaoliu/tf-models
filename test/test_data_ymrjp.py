import unittest
from tfmodels.data.ymrjp import YMRJPData


class TestDataYMRJP(unittest.TestCase):
    def test_load_data(self):
        data = YMRJPData(padding=True, max_length=512)
        # Training data
        self.assertEqual(data.x_train.shape, (67988, 512))
        self.assertEqual(data.y_train.shape, (67988, 2))
        self.assertEqual(sum(1 for _ in data.y_train if _[0] == 1), 17182)
        self.assertEqual(sum(1 for _ in data.y_train if _[1] == 1), 50806)
        # Testing Data
        self.assertEqual(data.x_test.shape, (7555, 512))
        self.assertEqual(data.y_test.shape, (7555, 2))
        self.assertEqual(sum(1 for _ in data.y_test if _[0] == 1), 1909)
        self.assertEqual(sum(1 for _ in data.y_test if _[1] == 1), 5646)
