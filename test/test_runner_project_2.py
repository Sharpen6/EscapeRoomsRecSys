import unittest
from tqdm import tqdm
from recsys.esc_room_rec_sys import *
from k_markov_new import k_markov_rc
from scipy.stats import *
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

class TestUserProfiling(unittest.TestCase):

    def test_all_algorithms(self):
        algorithms = {
                  'SVD (base model)': (SVD(),True),
        }


        train_set = '..//resources//dataset_8-9-2018_train.csv'
        test_set = '..//resources//dataset_8-9-2018_test.csv'

        results = pd.DataFrame(columns=['Algorithm', 'RMSE'])
        #results = results.append(self.test_old_method(train_set, test_set, algorithms), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 1), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 2), ignore_index=True)
        results = results.append(self.test_profiling_method(train_set, test_set, 3), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 4), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 5), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 6), ignore_index=True)
        #results = results.append(self.test_profiling_method(train_set, test_set, 7), ignore_index=True)
        results = results.append(self.test_profiling_method(train_set, test_set, 8), ignore_index=True)
        print(results)

    def test_old_method(self, train_set, test_set, algorithms):
        results = pd.DataFrame(columns=['Algorithm', 'RMSE'])
        recsys = esc_room_rec_sys()
        pbar = tqdm(total=len(algorithms))
        for algoName, algoObj in algorithms.items():
            try:
                print(algoName)
                rmse = recsys.predict_rating_split_by_time((train_set, test_set), algoObj)
                results = results.append({'Algorithm': algoName, 'RMSE': rmse}, ignore_index=True)
            except Exception as e:
                print(algoName + ' failed on: ' +str(e))
            pbar.update(1)
        results = results.sort_values('RMSE')
        pbar.close()
        return results

    def test_profiling_method(self, train_set_path, test_set_path, n_clusters):
        train_set = pd.read_csv(train_set_path, names=['userID','itemID','rating'])
        test_set = pd.read_csv(test_set_path, names=['userID','itemID','rating'])

        # train regular SVD
        default_algo = SVDpp()
        fold_files = [(train_set_path, test_set_path)]

        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)

        for trainset, testset in PredefinedKFold().split(data):
            default_algo.fit(trainset)

        users_data = train_set.groupby('userID')['rating'].agg(['count', 'mean', 'mad', 'median', 'min', 'max', 'std',
                                                      ('skew', lambda value: skew(value)),
                                                      ('kurtosis', lambda value: kurtosis(value)),
                                                      ]
                                                     ).fillna(0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(users_data)
        users_data = users_data.reset_index()
        cluster_mapping = dict(zip(users_data['userID'], kmeans.labels_))
        train_set['cluster'] = train_set['userID'].map(cluster_mapping)

        print('Total dataset size: ' + str(users_data.shape))
        for n in range(0, n_clusters):
            train_set[train_set['cluster'] == n].drop('cluster', axis=1).to_csv('..\\resources\\tmp\\train_set_cluster_'
                                                                                    + str(n) + '.csv', index=False,
                                                                                header=False)
            print('Generated cluster ' + str(train_set[train_set['cluster'] == n].shape))
        pass

        clusters_predictors = {}

        # create svd predictor for each cluster
        for n in range(0, n_clusters):
            algo = SVDpp()
            train_set_path = '..\\resources\\tmp\\train_set_cluster_' + str(n) + '.csv'
            dummy_path = train_set_path
            fold_files = [(train_set_path, dummy_path)]

            reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
            data = Dataset.load_from_folds(fold_files, reader=reader)

            for trainset, testset in PredefinedKFold().split(data):
                algo.fit(trainset)
                clusters_predictors[n] = algo

        pass

        # predict for test
        predictions = []
        for index, row  in test_set.iterrows():
            userID = row['userID']
            itemID = row['itemID']
            rating = row['rating']

            prediction = None

            if userID in cluster_mapping.keys():
                cluster = cluster_mapping[userID]
                prediction = clusters_predictors[cluster].predict(userID, itemID, rating)
            else:
                prediction = default_algo.predict(userID, itemID, rating)
            pass
            predictions.append(prediction)

        rmse = accuracy.rmse(predictions, verbose=False)
        result = pd.DataFrame(columns=['Algorithm', 'RMSE'])
        result = result.append({'Algorithm': 'Cluster-based-{}'.format(n_clusters), 'RMSE': rmse}, ignore_index=True)
        return result


if __name__ == '__main__':
    unittest.main()
