import itertools
from tqdm import tqdm
import numpy as np
import operator
from top_n_algorithms import TopNRecsys

class k_markov_rc(TopNRecsys):

    def __init__(self, k):
        self.k = k
        self.dict_count_k = {}
        self.dict_count_k_m_1 = {}
        self.train_set = None

    def fit(self, train_set, clean_fake):

        self.train_set = train_set

        if clean_fake:
            user_ratings = train_set.groupby('userID')['itemID'].apply(list)
            fake_users = user_ratings[user_ratings.apply(lambda x: len(x) <= 1)]
            train_set_filtered = train_set[~train_set['userID'].isin(fake_users.keys().tolist())]
        else:
            train_set_filtered = train_set

        count = 0
        pbar = tqdm(total=len(train_set_filtered.userID.unique()))
        for userID in train_set_filtered.userID.unique():
            pbar.update(1)
            count += 1

            df_user_ratings = train_set_filtered[train_set_filtered.userID == userID]
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

        already_ranked_items_by_users = self.train_set.groupby('userID')['itemID'].apply(list)

        result = {}

        pbar = tqdm(total=len(test_set.userID.unique()))

        count_failed = 0

        for userID in test_set.userID.unique():
            pbar.update(1)

            if userID not in self.train_set.userID.unique():
                result[userID] = []
                count_failed += 1
                continue

            df_user_previous_ratings = self.train_set[self.train_set.userID == userID]

            df_user_previous_ratings.sort_values('timestamp')
            grouped_by_time = df_user_previous_ratings.groupby('timestamp')['itemID'].apply(list)
            grouped_items = [x for x in grouped_by_time]

            combinations = list(itertools.combinations(grouped_items, r=self.k - 1))

            # Calc for each item
            item_prob = {}
            for itemID in self.train_set.itemID.unique():
                if itemID in already_ranked_items_by_users[userID]:
                    continue

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

            #if is prob for best item is 0 - return empty list
            if top_list[0][1] == 0:
                result[str(userID)] = []
                count_failed += 1
            else:
                result[str(userID)] = [str(i[0]) for i in top_list[:top_n]]
        print('skipped: ' + str(count_failed) + '/' + str(len(test_set.userID.unique())))
        pbar.close()
        return result
