import unittest
from recsys.esc_room_rec_sys import *

class TestRecSysMethods(unittest.TestCase):

    def test_print_test_a(self):
        print("ok")

    def test_prediction(self):
        #grid_params = {'SVD (base model)' : {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
        #                                    'reg_all': [0.4, 0.6]}}
        algorithms = {#'SVD (base model)': SVD(),
                      #'SVD++': SVDpp(),
                      #'NMF': NMF(),
                      #'SlopeOne': SlopeOne(),
                      #'KNNBaseline': KNNBaseline(),
                      #'KNNBasic': KNNBasic(),
                      #'KNNWithMeans': KNNWithMeans(),
                      #'KNNWithZScore': KNNWithZScore(),
                      #'CoClustering': CoClustering(),
                      #'BaselineOnly': BaselineOnly(),
                      'NormalPredictor': NormalPredictor()
                      }
        recsys = esc_room_rec_sys()
        recsys.read_data('..//resources//dataset_8-9-2018.csv')
        recsys.predict_rating_using_cross_validation(algorithms)

    def test_one_fold_by_time(self):
        recsys = esc_room_rec_sys()
        recsys.predict_rating_split_by_time(('..//resources//dataset_8-9-2018_train.csv',
                                             '..//resources//dataset_8-9-2018_test.csv'))
        pass

if __name__ == '__main__':
    unittest.main()
