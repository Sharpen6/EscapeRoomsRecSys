import sys, string, os
import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np
import operator
from sklearn.metrics import precision_score
class TopNRecsys:

    def __init__(self):
        pass

    def fit(self, train_set):
        pass

    def get_top_n_recommendations(self, test_set, top_n):
        pass


class k_markov_rc(TopNRecsys):

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

            count += 1
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

    def get_top_n_recommendations(self, test_set, top_n):

        if self.train_set is None:
            return []

        result = {}

        pbar = tqdm(total=len(test_set.userID.unique()))

        for userID in test_set.userID.unique():
            if userID == 15 or userID == '15':
                pass
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
            result[str(userID)] = [str(i[0]) for i in top_list[:top_n]]

        pbar.close()
        return result

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
            for line in f:
                user_id = line.split('[')[0].replace('\t', '')
                recommendations = line.split('[')[1].replace(']', '').replace('\n', '')
                rec_list = recommendations.split(',')
                rec_for_user = []
                for rec in rec_list[:top_n]:
                    rec_for_user.append(rec.split(':')[0])
                user_recommendations[user_id] = rec_for_user
        return user_recommendations
