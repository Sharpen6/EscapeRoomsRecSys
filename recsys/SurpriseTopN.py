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


class SurpriseRecMethod(TopNRecsys):

    def __init__(self, method):
        self.method = method

    def fit(self, train_set):

        self.train_set = train_set

    def get_rating_predictions(self, test_set, cluster_user_mapping=None):
        self.test_set = test_set
        test_path_tmp = "..\\resources\\tmp\\test_file.csv"
        train_path_tmp = "..\\resources\\tmp\\train_file.csv"

        self.train_set.to_csv(train_path_tmp, index=False, header=False)
        self.test_set.to_csv(test_path_tmp, index=False, header=False)

        fold_files = [(train_path_tmp, test_path_tmp)]
        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)

        for trainset, testset in PredefinedKFold().split(data):

            if cluster_user_mapping is None:
                self.method.fit(trainset)
            else:
                df_users_in_clusters = pd.DataFrame.from_dict(cluster_user_mapping)
                df_cluser_users = df_users_in_clusters.groupby('')
                #Distinct clusters:
                clusters = list(set(cluster_user_mapping.values()))

                #for cluster in clusters:
                    #cluster_train_data = trainset[trainset.userID.isin() userID]
                pass


        results = pd.DataFrame(columns=['userID', 'itemID', 'real', 'est'])

        pbar = tqdm(total=len(self.test_set.index))

        for key, val in self.test_set.iterrows():
            prediction = self.method.predict(str(val.userID), str(val.itemID), clip=False)
            results = results.append({"userID": int(val.userID),
                                      "itemID": int(val.itemID),
                                      "real": int(val.rating),
                                      "est": int(prediction.est)}, ignore_index=True)
            pbar.update(1)
        pbar.close()
        return results

    def get_top_n_recommendations(self, test_set, top_n):
        self.test_set = test_set

        test_path_tmp = "..\\resources\\tmp\\test_file.csv"
        train_path_tmp = "..\\resources\\tmp\\train_file.csv"

        self.train_set.to_csv(train_path_tmp, index=False, header=False)
        self.test_set.to_csv(test_path_tmp, index=False, header=False)

        fold_files = [(train_path_tmp, test_path_tmp)]
        reader = Reader(rating_scale=(1, 10), line_format='user item rating', sep=',')
        data = Dataset.load_from_folds(fold_files, reader=reader)

        for trainset, testset in PredefinedKFold().split(data):
            self.method.fit(trainset)

        already_ranked_items_by_users = self.train_set.groupby('userID')['itemID'].apply(list)

        recommendations = {}
        pbar = tqdm(total=len(self.test_set.userID.unique()))
        for userID in self.test_set.userID.unique():
            pbar.update(1)

            if userID not in self.train_set.userID.unique():
                recommendations[str(userID)] = []
                continue

            items_expected_ranking = {}
            for itemID in self.train_set.itemID.unique():
                if itemID in already_ranked_items_by_users[userID]:
                    continue
                # Calc prediction for item for user
                predicted = self.method.predict(str(userID), str(itemID), clip=False)
                items_expected_ranking[itemID] = predicted[3]
            sorted_predictions = sorted(items_expected_ranking.items(), key=operator.itemgetter(1))
            sorted_predictions.reverse()
            sorted_predictions = [str(x[0]) for x in sorted_predictions]
            user_recommendations = sorted_predictions[:top_n]
            recommendations[str(userID)] = user_recommendations
        pbar.close()
        return recommendations
