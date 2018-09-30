from surprise import AlgoBase
from surprise import Dataset
import numpy as np
import itertools
from collections import defaultdict
from tqdm import tqdm

class k_markov_recsys(AlgoBase):

    def __init__(self, k=3, sim_options={}, bsl_options={}):
        # Always call base method before doing anything.
        AlgoBase.__init__(self, sim_options=sim_options,
                          bsl_options=bsl_options)
        self.k = k

    def estimate(self, u, i):
        return 3

    def fit(self, trainset):
        # Here again: call base method before doing anything.
        AlgoBase.fit(self, trainset)

        cnt_k = {}
        cnt_k_m_1 = {}
        pbar = tqdm(total=trainset.n_users)

        for uid in trainset.all_users():
            try:
                pbar.update(1)
                user_ratings = trainset.ur[uid]

                if len(user_ratings) <= self.k - 1:
                    continue

                user_pickings = [x[0] for x in user_ratings]
                user_pickings.sort()

                combinations_k = list(itertools.combinations(user_pickings, r=self.k))
                combinations_k_m_1 = list(itertools.combinations(user_pickings, r=self.k - 1))

                for x in combinations_k:
                    if x not in cnt_k:
                        cnt_k[x] = 0
                    cnt_k[x] += 1
                for x in combinations_k_m_1:
                    if x not in cnt_k_m_1:
                        cnt_k_m_1[x] = 0
                    cnt_k_m_1[x] += 1
            except Exception as e:
                print(e)
            pass


        pbar.close()
        pass


        return self