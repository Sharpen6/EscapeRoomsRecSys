import unittest

import numpy as np

from recsys.esc_room_rec_sys import *

class TestRecSysMethods(unittest.TestCase):

    def test_split_dataset(self):
        recsys = esc_room_rec_sys()
        recsys.split_dataset(np.datetime64('2018-05-01'),
                             '../resources/dataset_8-9-2018.xlsx',
                             '../resources/dataset_8-9-2018_train.csv',
                             '../resources/dataset_8-9-2018_test.csv')
if __name__ == '__main__':
    unittest.main()