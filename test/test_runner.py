import unittest
from tqdm import tqdm
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

    def test_all_algorithms(self):
        #algorithms = {
        #    'SVD (base model)': SVD(),
        #    'SVD++': SVDpp(),
        #    'NMF': NMF(),
        #    'SlopeOne': SlopeOne(),
        #    'KNNBaseline': KNNBaseline(),
        #    'KNNBasic cosine user min = 1': KNNBasic(sim_options={'name': 'cosine', 'user_based': True}),
        #    'KNNBasic pearson user': KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': True}),
        #    'KNNBasic cosine item': KNNBasic(sim_options={'name': 'cosine', 'user_based': False}),
        #    'KNNBasic pearson item': KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': False}),
        #    'KNNWithMeans': KNNWithMeans(),
        #    'KNNWithZScore': KNNWithZScore(),
        #    'CoClustering': CoClustering(),
        #    'BaselineOnly': BaselineOnly(),
        #    'NormalPredictor': NormalPredictor()
        #}

        # Add KNNBasic
        name = ['cosine', 'pearson', 'pearson_baseline']
        user_based = [True, False]
        min_support = [1, 3, 5, 10]
        KNNBasic_variations = [('KNNBasic_'+str(n)+'_'+str(u)+'_'+str(m), KNNBasic(sim_options={'name': n, 'user_based': u, 'min_support': m}))
                           for n in name
                           for u in user_based
                           for m in min_support]

        recsys = esc_room_rec_sys()
        train_set = '..//resources//dataset_8-9-2018_train.csv'
        test_set = '..//resources//dataset_8-9-2018_test.csv'
        results = pd.DataFrame(columns=['Algorithm', 'RMSE'])
        pbar = tqdm(total=len(KNNBasic_variations))
        for algoName, algoObj in KNNBasic_variations:
            try:
                print(algoName)
                rmse_results = []
                for i in range(10):
                    rmse_results.append(recsys.predict_rating_split_by_time((train_set, test_set), algoObj))
                rmse = np.mean(rmse_results)
                results = results.append({'Algorithm': algoName, 'RMSE': rmse}, ignore_index=True)
            except:
                print(algoName + ' failed')
            pbar.update(1)
        results = results.sort_values('RMSE')
        pbar.close()
        print(results)

    def test_recsys(self):
        from recsys.algorithm.factorize import SVD
        svd = SVD()
        svd.load_data(filename='./data/movielens/ratings.dat',
                      sep='::',
                      format={'col': 0, 'row': 1, 'value': 2, 'ids': int})
if __name__ == '__main__':
    unittest.main()
