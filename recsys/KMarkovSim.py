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

class k_markov_rc_user_similarity(TopNRecsys):

    def __init__(self, k):
        self.k = k
        self.dict_count_k = {}
        self.dict_count_k_m_1 = {}
        self.train_set = None
        self.sim_method = None

        self.threshold = 0.98
    def fit(self, train_set):
        self.train_set = train_set

    def calc_similarities(self):
        test_path_tmp = "..\\resources\\tmp\\test_file.csv"
        train_path_tmp = "..\\resources\\tmp\\train_file.csv"

        self.train_set.to_csv(train_path_tmp, index=False, header=False)
        self.test_set.to_csv(test_path_tmp, index=False, header=False)

        fold_files = [(train_path_tmp, test_path_tmp)]
        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)
        self.sim_method = KNNBasic(sim_options={'name': 'cosine', 'user_based': True})
        for trainset, testset in PredefinedKFold().split(data):
            self.sim_method.fit(trainset)


    def refit_by_users(self, users):
        self.dict_count_k = {}
        self.dict_count_k_m_1 = {}
        #pbar = tqdm(total=len(users))
        for userID in users:
            #pbar.update(1)
            df_user_ratings = self.train_set[self.train_set.userID == userID]
            df_user_ratings.sort_values('timestamp')
            grouped_by_time = df_user_ratings.groupby('timestamp')['itemID'].apply(list)
            grouped_items = [x for x in grouped_by_time]

            if len(grouped_items) <= self.k - 1:
                continue

            combinations = list(itertools.combinations(grouped_items, r=self.k))
            for combination in combinations:
                for k in itertools.product(*combination):
                    if k not in self.dict_count_k:
                        self.dict_count_k[k] = 0
                    self.dict_count_k[k] += 1

            combinations = list(itertools.combinations(grouped_items, r=self.k - 1))
            for combination in combinations:
                for k in itertools.product(*combination):
                    if k not in self.dict_count_k_m_1:
                        self.dict_count_k_m_1[k] = 0
                    self.dict_count_k_m_1[k] += 1
            pass
        #pbar.close()


    def get_top_n_recommendations(self, test_set, top_n):
        self.test_set = test_set
        self.calc_similarities()
        if self.train_set is None:
            return []

        result = {}

        count_failed = 0

        pbar = tqdm(total=len(test_set.userID.unique()))
        for userID in test_set.userID.unique():
            pbar.update(1)
            # get most similar users
            selected_usrs = {}
            if userID >= len(self.sim_method.sim):
                result[str(userID)] = []
                count_failed += 1
                continue

            for i, candidate_user in enumerate(self.sim_method.sim[userID]):
                if candidate_user > self.threshold:
                    selected_usrs[i] = candidate_user
            users_sim = [x[0] for x in sorted(selected_usrs.items(), key=operator.itemgetter(1), reverse=True)]
            self.refit_by_users(users_sim)


            df_user_previous_ratings = self.train_set[self.train_set.userID == userID]

            df_user_previous_ratings.sort_values('timestamp')
            grouped_by_time = df_user_previous_ratings.groupby('timestamp')['itemID'].apply(list)
            grouped_items = [x for x in grouped_by_time]

            combinations = list(itertools.combinations(grouped_items, r=self.k - 1))

            # Calc for each item
            item_prob = {}
            for itemID in self.train_set.itemID.unique():
                pass
                item_prob_sum = [0]
                for comb in combinations:
                    for k in itertools.product(*comb + ([itemID],)):
                        if k in self.dict_count_k:
                            numerator = self.dict_count_k[k]
                        else:
                            continue
                        k_without_target_item = k[:-1]
                        if k_without_target_item in self.dict_count_k_m_1:
                            denumerator = self.dict_count_k_m_1[k_without_target_item]
                        else:
                            continue
                        if denumerator != 0 and numerator != 0:
                            prob = numerator / float(denumerator)
                            item_prob_sum.append(prob)

                item_prob[itemID] = np.sum(item_prob_sum)

            top_list = sorted(item_prob.items(), key=operator.itemgetter(1), reverse=True)

            # is prob for best item is 0 - return empty list
            if top_list[0][1] == 0:
                result[str(userID)] = []
                count_failed += 1
            else:
                result[str(userID)] = [str(i[0]) for i in top_list[:top_n]]
        print('skipped: ' + str(count_failed) + '/' + str(len(test_set.userID.unique())))
        pbar.close()
        return result
