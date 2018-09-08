import numpy as np
import pandas as pd
from sklearn import cross_validation as cv


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
        n_users = dataset.user_id.unique().shape[0]
        n_items = dataset.item_id.unique().shape[0]
        print('Number of users = ' + str(n_users) + ' | Number of rooms = ' + str(n_items))
        user_counts = dataset.user_id.value_counts()
        items_counts = dataset.item_id.value_counts()
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

    def predict_rating(self):
        threshold = self.parameters['date_split_threshold']

        train_data = self.dataset[self.dataset['aprox_review_date'] < threshold]
        test_data = self.dataset[self.dataset['aprox_review_date'] > threshold]

        self.analyse_data(train_data, 'Train set')
        self.analyse_data(test_data, 'Test set')