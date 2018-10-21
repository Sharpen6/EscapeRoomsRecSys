import unittest
from scipy.stats import *
from CleaningAlgorithms.CleanFakeUsersRankedOneItem import CleanFakeUsersRankedOne
from CleaningAlgorithms.CleanFakeUsersRankedOnly10 import CleanFakeUsersRankedOnlyTen
from CleaningAlgorithms.CleanFakeUsersNoClean import CleanFakeUsersNone
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
from surprise.prediction_algorithms import *
from recsys import SurpriseTopN
from recsys import MyMediaLiteTopN
from recsys import KMarkov
from recsys import KMarkovLatest
from sklearn.base import clone
from MostPopular import MostPopular
import datetime
from sklearn.cluster import KMeans

from AveragePredictor import AveragePredictor


class TestRecSysMethods(unittest.TestCase):

    def reset_algs(self):

        rec_sys_algorithms = {}

        # rec_sys_algorithms = self.add_mymedialite_algorithms(rec_sys_algorithms)
        rec_sys_algorithms = self.add_surprise_algorithms(rec_sys_algorithms)
        # rec_sys_algorithms = self.add_k_markov_algorithms(rec_sys_algorithms)

        return rec_sys_algorithms

    def test_prediction_algorithms(self):

        train_set_path = '..//resources//aggregated//train_numerized_with_anon.csv'
        test_set_path = '..//resources//aggregated//test_numerized_with_anon.csv'
        train_set = pd.read_csv(train_set_path, parse_dates=[3], index_col='index')
        test_set = pd.read_csv(test_set_path, parse_dates=[3], index_col='index')

        clean_fake_methods = [
            CleanFakeUsersNone(),
            CleanFakeUsersRankedOne(),
            CleanFakeUsersRankedOnlyTen()
        ]
        fill_with_popular = [
            True,
            False
        ]
        n_clusters = 3
        use_clustering = [
            True,
            #False
        ]
        if fill_with_popular:
            # train most popular for default
            average_predictor = AveragePredictor()
            average_predictor.fit(train_set)

        algs_results = pd.DataFrame(columns=['Algorithm', 'Filled with pop', 'Clean fake method', 'RMSE'])

        users_data = train_set.groupby('userID')['rating'].agg(
            ['count', 'mean', 'mad', 'median', 'min', 'max', 'std',
             ('skew', lambda value: skew(value)),
             ('kurtosis', lambda value: kurtosis(value)),
             ]
        ).fillna(0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(users_data)
        users_data = users_data.reset_index()
        cluster_user_mapping = dict(zip(users_data['userID'], kmeans.labels_))


        for fill_with_pop in fill_with_popular:
            for clean_fake_method in clean_fake_methods:

                # Re-init algorithms objects
                rec_sys_algs = self.reset_algs()
                with tqdm(total=len(rec_sys_algs.keys())) as pbar:
                    for name, model in rec_sys_algs.items():
                        custom_train_set = clean_fake_method.clean(train_set)
                        custom_train_set['itemID'] = custom_train_set['itemID'].astype(int)
                        custom_train_set['userID'] = custom_train_set['userID'].astype(int)

                        try:
                            if use_clustering:

                                model.fit(custom_train_set)
                                predictions = model.get_rating_predictions(test_set, cluster_user_mapping)

                                rmse = self.calc_rmse_from_predictions(predictions)

                                algs_results = algs_results.append({"Algorithm": name, "Filled with pop": fill_with_pop,
                                                                    "Clean fake method": clean_fake_method.print(),
                                                                    "RMSE": rmse},
                                                                   ignore_index=True)

                            else:
                                model.fit(custom_train_set)
                                predictions = model.get_rating_predictions(test_set)

                                rmse = self.calc_rmse_from_predictions(predictions)

                                algs_results = algs_results.append({"Algorithm": name, "Filled with pop": fill_with_pop,
                                                                    "Clean fake method": clean_fake_method.print(), "RMSE": rmse},
                                                                   ignore_index=True)
                        except:
                            algs_results = algs_results.append({"Algorithm": name, "Filled with pop": fill_with_pop,
                                                                "Clean fake method": clean_fake_method.print(),
                                                                "RMSE": "FAILED"},
                                                               ignore_index=True)
                        pbar.update(1)
        filename = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        algs_results.to_csv('..\\resources\\results\\predictions_results_' + filename + '.csv')
        print('Done')

    def calc_rmse_from_predictions(self, predictions):
        return np.sqrt(((predictions.real - predictions.est) ** 2).mean())



    def add_mymedialite_algorithms(self, rec_sys_algorithms):
        algos = {
            # "(MyMediaLite) BiPolarSlopeOne":MyMediaLiteTopN.MyMdediaLiteRecMethod('BiPolarSlopeOne',''),
            # "(MyMediaLite) FactorWiseMatrixFactorization":MyMediaLiteTopN.MyMdediaLiteRecMethod('FactorWiseMatrixFactorization',''),
            # "(MyMediaLite) GlobalAverage":MyMediaLiteTopN.MyMdediaLiteRecMethod('GlobalAverage',''),
            # "(MyMediaLite) ItemAttributeKNN":MyMediaLiteTopN.MyMdediaLiteRecMethod('ItemAttributeKNN',''),
            # "(MyMediaLite) ItemAverage":MyMediaLiteTopN.MyMdediaLiteRecMethod('ItemAverage',''),
            # "(MyMediaLite) ItemKNN":MyMediaLiteTopN.MyMdediaLiteRecMethod('ItemKNN',''),
            # "(MyMediaLite) MatrixFactorization":MyMediaLiteTopN.MyMdediaLiteRecMethod('MatrixFactorization',''),
            # "(MyMediaLite) SlopeOne":MyMediaLiteTopN.MyMdediaLiteRecMethod('SlopeOne',''),
            # "(MyMediaLite) UserAttributeKNN":MyMediaLiteTopN.MyMdediaLiteRecMethod('UserAttributeKNN',''),
            # "(MyMediaLite) UserAverage":MyMediaLiteTopN.MyMdediaLiteRecMethod('UserAverage',''),
            # "(MyMediaLite) UserItemBaseline":MyMediaLiteTopN.MyMdediaLiteRecMethod('UserItemBaseline',''),
            # "(MyMediaLite) UserKNN": MyMediaLiteTopN.MyMdediaLiteRecMethod('UserKNN',''),
            # "(MyMediaLite) TimeAwareBaseline":MyMediaLiteTopN.MyMdediaLiteRecMethod('TimeAwareBaseline',''),
            # "(MyMediaLite) TimeAwareBaselineWithFrequencies":MyMediaLiteTopN.MyMdediaLiteRecMethod('TimeAwareBaselineWithFrequencies',''),
            # "(MyMediaLite) CoClustering":MyMediaLiteTopN.MyMdediaLiteRecMethod('CoClustering',''),
            # "(MyMediaLite) Random":MyMediaLiteTopN.MyMdediaLiteRecMethod('Random',''),
            # "(MyMediaLite) Constant":MyMediaLiteTopN.MyMdediaLiteRecMethod('Constant',''),
            # "(MyMediaLite) LatentFeatureLogLinearModel":MyMediaLiteTopN.MyMdediaLiteRecMethod('LatentFeatureLogLinearModel',''),
            # "(MyMediaLite) BiasedMatrixFactorization":MyMediaLiteTopN.MyMdediaLiteRecMethod('BiasedMatrixFactorization',''),
            # "(MyMediaLite) SVDPlusPlus":MyMediaLiteTopN.MyMdediaLiteRecMethod('SVDPlusPlus',''),
            # "(MyMediaLite) SigmoidSVDPlusPlus":MyMediaLiteTopN.MyMdediaLiteRecMethod('SigmoidSVDPlusPlus',''),
            # "(MyMediaLite) SocialMF":MyMediaLiteTopN.MyMdediaLiteRecMethod('SocialMF',''),
            # "(MyMediaLite) SigmoidItemAsymmetricFactorModel":MyMediaLiteTopN.MyMdediaLiteRecMethod('SigmoidItemAsymmetricFactorModel',''),
            # "(MyMediaLite) SigmoidUserAsymmetricFactorModel":MyMediaLiteTopN.MyMdediaLiteRecMethod('SigmoidUserAsymmetricFactorModel',''),
            # "(MyMediaLite) SigmoidCombinedAsymmetricFactorModel":MyMediaLiteTopN.MyMdediaLiteRecMethod('SigmoidCombinedAsymmetricFactorModel',''),
            # "(MyMediaLite) NaiveBayes":MyMediaLiteTopN.MyMdediaLiteRecMethod('NaiveBayes',''),
            # "(MyMediaLite) ExternalRatingPredictor":MyMediaLiteTopN.MyMdediaLiteRecMethod('ExternalRatingPredictor',''),
            # "(MyMediaLite) GSVDPlusPlus":MyMediaLiteTopN.MyMdediaLiteRecMethod('GSVDPlusPlus','')
        }

        new_algo_list = {**rec_sys_algorithms, **algos}
        return new_algo_list

    def add_surprise_algorithms(self, rec_sys_algorithms):
        algos = {
            #"(Surprise) KNNBasic pearson item": SurpriseTopN.SurpriseRecMethod(
            #    KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': False})),
            "(Surprise) SVD (base model)": SurpriseTopN.SurpriseRecMethod(SVD()),
            #"(Surprise) SVD++": SurpriseTopN.SurpriseRecMethod(SVDpp()),
            #"(Surprise) NMF": SurpriseTopN.SurpriseRecMethod(NMF()),
            #"(Surprise) SlopeOne": SurpriseTopN.SurpriseRecMethod(SlopeOne()),
            #"(Surprise) KNNBaseline": SurpriseTopN.SurpriseRecMethod(KNNBaseline()),
            #"(Surprise) KNNBasic cosine user min = 1": SurpriseTopN.SurpriseRecMethod(
            #    KNNBasic(sim_options={'name': 'cosine', 'user_based': True})),
            #"(Surprise) KNNBasic pearson user": SurpriseTopN.SurpriseRecMethod(
            #    KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': True})),
            #"(Surprise) KNNBasic cosine item": SurpriseTopN.SurpriseRecMethod(
            #    KNNBasic(sim_options={'name': 'cosine', 'user_based': False})),
            #"(Surprise) KNNWithMeans": SurpriseTopN.SurpriseRecMethod(KNNWithMeans()),
            #"(Surprise) KNNWithZScore": SurpriseTopN.SurpriseRecMethod(KNNWithZScore()),
            #"(Surprise) CoClustering": SurpriseTopN.SurpriseRecMethod(CoClustering()),
            #"(Surprise) BaselineOnly": SurpriseTopN.SurpriseRecMethod(BaselineOnly()),
            #"(Surprise) NormalPredictor": SurpriseTopN.SurpriseRecMethod(NormalPredictor())
        }

        new_algo_list = {**rec_sys_algorithms, **algos}
        return new_algo_list
