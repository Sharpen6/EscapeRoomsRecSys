import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader

from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import *

class esc_room_rec_sys:

    dataset = None
    parameters = None

    def read_data(self, file_path):
        self.dataset = pd.read_csv(file_path).set_index(['index'])
        self.dataset['aprox_review_date'] = self.dataset['aprox_review_date'].astype('datetime64[ns]')

        self.analyse_data(self.dataset, 'All records')

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
    def set_params(self, params):
        if 'date_split_threshold' in params:
            try:
                asdate = params['date_split_threshold'].astype('datetime64[ns]')
            except:
                print('FAILED PARSING DATETIME THRESHOLD IN PARAMETERS')
                pass
            date_split_threshold = asdate

        self.parameters = {'date_split_threshold' : date_split_threshold,
                           }
    def predict_rating_using_cross_validation(self):

        algo = SVD()
        reader = Reader(rating_scale=(1, 10))
        columns = ['userID', 'itemID', 'rating']
        dataset = self.dataset[columns].dropna(how='any')
        all_data = Dataset.load_from_df(dataset, reader)

        # Run 5-fold cross-validation and print results
        results = cross_validate(algo, all_data, measures=['RMSE'], cv=10, verbose=True)
        print(results['test_rmse'].mean())

    def predict_rating(self):

        threshold = self.parameters['date_split_threshold']
        columns = ['userID', 'itemID', 'rating']

        train_data = self.dataset[self.dataset['aprox_review_date'] < threshold][columns]
        test_data = self.dataset[self.dataset['aprox_review_date'] > threshold][columns]

        train_data = train_data.dropna(how='any')
        test_data = test_data.dropna(how='any')

        self.analyse_data(train_data, 'Train set')
        self.analyse_data(test_data, 'Test set')

        reader = Reader(rating_scale=(1, 10))

        train_data = Dataset.load_from_df(train_data, reader)
        test_data = Dataset.load_from_df(test_data, reader)

        built_trainset = train_data.build_full_trainset()
        built_testset = test_data.build_testset()

        algo = SVD()
        algo.fit(built_trainset)
        predictions = algo.test(built_testset)
        results = accuracy.rmse(predictions)
        print(results)
