import itertools
from tqdm import tqdm
import numpy as np
import operator
from top_n_algorithms import TopNRecsys


class AveragePredictor(TopNRecsys):

    def __init__(self):
        self.ratings_average = 5
        self.items_average = {}

    def fit(self, train_set):
        train_set['rating'] = train_set['rating'].astype(int)
        self.ratings_average = train_set['rating'].mean()
        x = train_set.groupby(by=['itemID'])['rating'].mean()
        for key, val in x.items():
            self.items_average[str(key)] = val

    def predict_for_item(self, IID):
        IID = str(IID)
        if IID in self.items_average:
            return self.items_average[IID]
        else:
            return self.ratings_average

    def clone(self):
        pass
