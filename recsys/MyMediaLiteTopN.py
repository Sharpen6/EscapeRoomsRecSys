import sys, string, os
import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np
import operator
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import PredefinedKFold
from surprise.prediction_algorithms import *
from top_n_algorithms import TopNRecsys

class MyMdediaLiteRecMethod(TopNRecsys):

    def __init__(self, method, params):
        self.method = method
        self.params = params
        pass

    def fit(self, train_set):

        self.train_set = train_set

    def get_top_n_recommendations(self, test_set, top_n):
        self.test_set = test_set

        test_users_path_tmp = "..\\resources\\tmp\\test_users_file.csv"
        train_path_tmp = "..\\resources\\tmp\\train_file.csv"
        output_path_tmp = "..\\resources\\tmp\\output_file.csv"

        with open(test_users_path_tmp, 'w') as f:
            for item in self.test_set.userID.unique():
                f.write("%s\n" % item)

        self.train_set.to_csv(train_path_tmp, index=False, header=False)


        recommender_arg = '--recommender={}'.format(self.method)
        train_path_arg = '--training-file={}'.format(train_path_tmp)
        output_path_arg = '--prediction-file={}'.format(output_path_tmp)
        test_users_arg = '--test-users={}'.format(test_users_path_tmp)

        options = ''
        if self.params != '':
            options = '--recommender-options={}'.format(self.params)

        exe_line = ".\..\MyMediaLite\item_recommendation.exe  --no-id-mapping --file-format=ignore_first_line" \
                                                       " {} {} {} {} {}".format(recommender_arg,
                                                                             train_path_arg,
                                                                             test_users_arg,
                                                                             output_path_arg,
                                                                             options)

        os.system(exe_line)

        user_recommendations = {}
        with open(output_path_tmp) as f:
            count = 0
            count_skipped = 1
            for line in f:
                count += 1
                user_id = line.split('[')[0].replace('\t', '')
                recommendations = line.split('[')[1].replace(']', '').replace('\n', '')
                if recommendations == '':
                    rec_list = []
                else:
                    rec_list = recommendations.split(',')

                rec_for_user = []
                for rec in rec_list[:top_n]:
                    rec_for_user.append(rec.split(':')[0])
                if len(rec_for_user) == 0:
                    count_skipped += 1
                user_recommendations[user_id] = rec_for_user
            print('skipped: ' + str(count_skipped) + '/' + str(count))
        return user_recommendations
