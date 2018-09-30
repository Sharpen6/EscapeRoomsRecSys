import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np
import operator
from sklearn.metrics import precision_score

class k_markov_rc:

    def __init__(self, k=3):
        self.k = k
        self.dict_count_k = {}
        self.dict_count_k_m_1 = {}
        self.train_set = None

    def fit(self, train_set):
        self.train_set = train_set
        pbar = tqdm(total=len(train_set.userID.unique()))

        count = 0

        for userID in train_set.userID.unique():
            pbar.update(1)

            count+=1
            if count > 1000:
                break

            df_user_ratings = train_set[train_set.userID == userID]
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
        pbar.close()
        print('Done fitting')

    def get_top_n_recommendations(self, test_set, top_n=10):

        if self.train_set is None:
            return []

        pbar = tqdm(total=len(test_set.userID.unique()))

        rec_list = {}

        for userID in test_set.userID.unique():
            pbar.update(1)
            df_user_previous_ratings = self.train_set[self.train_set.userID == userID]

            df_user_previous_ratings.sort_values('timestamp')
            grouped_by_time = df_user_previous_ratings.groupby('timestamp')['itemID'].apply(list)
            grouped_items = [x for x in grouped_by_time]

            combinations = list(itertools.combinations(grouped_items, r=self.k - 1))

            # Calc for each item
            item_prob = {}
            for itemID in self.train_set.itemID.unique():
                item_prob_sum = [0]
                for comb in combinations:
                    for k in itertools.product(*comb + ([itemID],)):

                        numerator = None
                        denumerator = None

                        if k in self.dict_count_k:
                            numerator = self.dict_count_k[k]
                        k_without_target_item = k[:-1]
                        if k_without_target_item in self.dict_count_k_m_1:
                            denumerator = self.dict_count_k_m_1[k_without_target_item]

                        if denumerator is not None and numerator is not None and denumerator != 0 and numerator != 0:
                            prob = numerator / float(denumerator)
                            item_prob_sum.append(prob)

                item_prob[itemID] = np.sum(item_prob_sum)

            top_list = sorted(item_prob.items(), key=operator.itemgetter(1), reverse=True)
            rec_list = [i[0] for i in top_list[:top_n]]

            # Test set not to be used until now..
            df_user_ratings = test_set[test_set.userID == userID]
            test_real = df_user_ratings.itemID.unique().tolist()
            try:
                print('precision: ' + precision_score([test_real], [rec_list]))
            except Exception as e:
                print(e)
        pbar.close()


