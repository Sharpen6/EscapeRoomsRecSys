import numpy as np
import unittest
from recsys.esc_room_rec_sys import *

class TestRecSysMethods(unittest.TestCase):

    def test_scan_rooms(self):
        print('Scanning rooms')
        recsys = esc_room_rec_sys()
        recsys.read_data('..//resources//dataset_8-9-2018.csv')
        pass

    def test_prediction(self):
        recsys = esc_room_rec_sys()
        recsys.read_data('..//resources//dataset_8-9-2018.csv')
        recsys.set_params({'date_split_threshold': np.datetime64('2018-05-01')})
        recsys.predict_rating()

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()