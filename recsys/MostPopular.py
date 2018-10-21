import itertools
from tqdm import tqdm
import numpy as np
import operator
from top_n_algorithms import TopNRecsys


class MostPopular(TopNRecsys):

    def __init__(self):
        self.item_ratings_sorted = None

    def fit(self, train_set):
        self.item_ratings_sorted = \
        train_set.groupby(['itemID'])['userID'].agg(['count']).sort_values(by='count', ascending=False)[
            'count'].tolist()

    def get_top_n_recommendations(self, test_set, top_n):
        result = {}

        for userID in test_set.userID.unique():
            top_list = self.item_ratings_sorted
            result[str(userID)] = [str(i) for i in top_list[:top_n]]
        return result

    def clone(self):
        pass
