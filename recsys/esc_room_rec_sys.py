import numpy as np
import pandas as pd

from surprise import SVD
from surprise import AlgoBase
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import *

from surprise.model_selection import PredefinedKFold



class esc_room_rec_sys:
    dataset = None
    parameters = None

    def __init__(self):
        pass

    def read_data(self, file_path):
        self.dataset = pd.read_csv(file_path).set_index(['index'])
        self.dataset['timestamp'] = self.dataset['timestamp'].astype('datetime64[ns]')
        pass

    def analyse_data(self, dataset, description):
        print('~~~~~~~   Analysing dataset: ' + description + '~~~~~~~')
        print('Shape: ' + str(dataset.shape))
        n_users = dataset.userID.unique().shape[0]
        n_items = dataset.itemID.unique().shape[0]
        print('Number of users = ' + str(n_users) + ' | Number of rooms = ' + str(n_items))
        #user_counts = dataset.userID.value_counts()
        #items_counts = dataset.itemID.value_counts()
        pass

    def predict_rating_using_cross_validation(self, algorithms):
        res_df = pd.DataFrame(columns=['Algorithm', 'RMSE'])
        for algoKey, algoVal in algorithms.items():
            algo = algoVal
            reader = Reader(rating_scale=(1, 10), line_format='user item rating')
            columns = ['userID', 'itemID', 'rating']
            dataset = self.dataset[columns].dropna(how='any')
            all_data = Dataset.load_from_df(dataset, reader)

            results = cross_validate(algo, all_data, measures=['RMSE'], cv=10, n_jobs=-1, verbose=False)
            rmse = results['test_rmse'].mean()
            print('algo {} : RMSE: {}'.format(algoKey,rmse))
            res_df = res_df.append({'Algorithm': algoKey, 'RMSE': rmse}, ignore_index=True)
        print(res_df)

    def predict_rating_split_by_time(self, files_pair, algo):
        fold_files = [(files_pair)]
        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)

        for trainset, testset in PredefinedKFold().split(data):
            algo.fit(trainset)
            predictions = algo.test(testset)
            rmse = accuracy.rmse(predictions, verbose=False)
            return rmse




    def split_dataset(self, threshold, input, output_train, output_test):
        pass
        #columns = ['userID', 'itemID', 'rating']
        #self.dataset = pd.read_excel(input, sheet_name='dataset_8-9-2018')
        #self.dataset['timestamp'] = self.dataset['timestamp'].astype('datetime64[ns]')
        #self.dataset = self.dataset.dropna(how='any')
        #train_data = self.dataset[self.dataset['timestamp'] < threshold][columns]
        #test_data = self.dataset[self.dataset['timestamp'] > threshold][columns]

        #train_data.to_csv(output_train, header=False, index=False)
        #test_data.to_csv(output_test, header=False, index=False)
