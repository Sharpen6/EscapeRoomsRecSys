import unittest


from recsys.top_n_algorithms import *
import pandas as pd
import os
import numpy as np
from surprise.prediction_algorithms import *
from recsys import SurpriseTopN
from recsys import MyMediaLiteTopN
from recsys import KMarkov
from recsys import KMarkovSim
from recsys import KMarkovClusters
from recsys import KMarkovLatest

class TestRecSysMethods(unittest.TestCase):


    def test_top_n_algorithms(self):
        train_set_path = '..//resources//aggregated//train_numerized.csv'
        test_set_path = '..//resources//aggregated//test_numerized.csv'
        train_set = pd.read_csv(train_set_path, parse_dates=[3])
        test_set = pd.read_csv(test_set_path, parse_dates=[3])

        rec_sys_algs = {
            #"(Surprise) SVD (base model)": SurpriseTopN.SurpriseRecMethod(SVD()),
            #"(Surprise) SVD++": SurpriseTopN.SurpriseRecMethod(SVDpp()),
            #"(Surprise) NMF": SurpriseTopN.SurpriseRecMethod(NMF()),
            #"(Surprise) SlopeOne": SurpriseTopN.SurpriseRecMethod(SlopeOne()),
            #"(Surprise) KNNBaseline": SurpriseTopN.SurpriseRecMethod(KNNBaseline()),
            #"(Surprise) KNNBasic cosine user min = 1": SurpriseTopN.SurpriseRecMethod(KNNBasic(sim_options={'name': 'cosine', 'user_based': True})),
            #"(Surprise) KNNBasic pearson user": SurpriseTopN.SurpriseRecMethod(KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': True})),
            #"(Surprise) KNNBasic cosine item": SurpriseTopN.SurpriseRecMethod(KNNBasic(sim_options={'name': 'cosine', 'user_based': False})),
            #"(Surprise) KNNBasic pearson item": SurpriseTopN.SurpriseRecMethod(KNNBasic(sim_options={'name': 'pearson_baseline', 'user_based': False})),
            #"(Surprise) KNNWithMeans": SurpriseTopN.SurpriseRecMethod(KNNWithMeans()),
            #"(Surprise) KNNWithZScore": SurpriseTopN.SurpriseRecMethod(KNNWithZScore()),
            #"(Surprise) CoClustering": SurpriseTopN.SurpriseRecMethod(CoClustering()),
            #"(Surprise) BaselineOnly": SurpriseTopN.SurpriseRecMethod(BaselineOnly()),
            #"(Surprise) NormalPredictor": SurpriseTopN.SurpriseRecMethod(NormalPredictor()),
#
           #"(MyMediaLite) UserKNN": MyMediaLiteTopN.MyMdediaLiteRecMethod('UserKNN', 'correlation=Pearson'),
           #"(MyMediaLite) BPRMF": MyMediaLiteTopN.MyMdediaLiteRecMethod('BPRMF', ''),
           #"(MyMediaLite) ItemAttributeKNN ": MyMediaLiteTopN.MyMdediaLiteRecMethod('ItemAttributeKNN ', ''),
           #"(MyMediaLite) ItemKNN": MyMediaLiteTopN.MyMdediaLiteRecMethod('ItemKNN', ''),
           #"(MyMediaLite) MostPopular": MyMediaLiteTopN.MyMdediaLiteRecMethod('MostPopular', ''),
           #"(MyMediaLite) Random": MyMediaLiteTopN.MyMdediaLiteRecMethod('Random', ''),
           #"(MyMediaLite) UserAttributeKNN": MyMediaLiteTopN.MyMdediaLiteRecMethod('UserAttributeKNN', ''),
           #"(MyMediaLite) WRMF": MyMediaLiteTopN.MyMdediaLiteRecMethod('WRMF', ''),
           #"(MyMediaLite) Zero": MyMediaLiteTopN.MyMdediaLiteRecMethod('Zero', ''),
           #"(MyMediaLite) MultiCoreBPRMF ": MyMediaLiteTopN.MyMdediaLiteRecMethod('MultiCoreBPRMF', ''),
           #"(MyMediaLite) SoftMarginRankingMF": MyMediaLiteTopN.MyMdediaLiteRecMethod('SoftMarginRankingMF', ''),
           #"(MyMediaLite) WeightedBPRMF": MyMediaLiteTopN.MyMdediaLiteRecMethod('WeightedBPRMF', ''),
           #"(MyMediaLite) MostPopularByAttributes": MyMediaLiteTopN.MyMdediaLiteRecMethod('MostPopularByAttributes', ''),
           #"(MyMediaLite) BPRSLIM": MyMediaLiteTopN.MyMdediaLiteRecMethod('BPRSLIM', ''),
#
            #"K-markov(k=1)": KMarkov.k_markov_rc(k=1),
            "K-markov(k=2)": KMarkov.k_markov_rc(k=2),
            #"K-markov(k=3)": KMarkov.k_markov_rc(k=3),
            #"K-markov(k=4)": KMarkov.k_markov_rc(k=4),
            "K-markov-latest(k=2)": KMarkovLatest.KMarkovLatest(k=2),
            #"K-markov-similarity":  KMarkovSim.k_markov_rc_user_similarity(k=2),
            #"K-markov-clustered": KMarkovClusters.k_markov_clusters(k=2, clusters=3)
            }

        algs_results = {}

        for name, model in rec_sys_algs.items():
            print(name)
            model.fit(train_set, False)
            results = model.get_top_n_recommendations(test_set, top_n=10)
            algs_results[name+'(clean_fake=False)'] = results

        for name, model in rec_sys_algs.items():
            print(name)
            model.fit(train_set, True)
            results = model.get_top_n_recommendations(test_set, top_n=10)
            algs_results[name+'(clean_fake=True)'] = results


        stats = self.calc_precision_auc(train_set, test_set, algs_results)
        stats.to_csv('..\\resources\\tmp\\results2.csv')
        print('Done')

    def calc_precision_auc(self, train_set, test_set, algs_results):
        test_path_tmp = "..\\resources\\tmp\\test_file.csv"
        train_path_tmp = "..\\resources\\tmp\\train_file.csv"
        prediction_path_tmp = "..\\resources\\tmp\\recommendations_output_for_method.csv"
        test_set.to_csv(test_path_tmp, index=False, header=False)
        train_set.to_csv(train_path_tmp, index=False, header=False)

        df_stats = pd.DataFrame(columns=['alg', 'AUC', 'prec@5', 'prec@10', 'MAP', 'recall@5', 'recall@10', 'NDCG',
                                         'MRR'])

        for method, recommendations in algs_results.items():
            with open(prediction_path_tmp, 'w') as f:
                for key, value in recommendations.items():
                    for rate in value:
                        f.write(str(key) + ',' + str(rate) + ',1\n')

            train_path_arg = '--training-file={}'.format(train_path_tmp)
            test_path_arg = '--test-file={}'.format(test_path_tmp)
            predictions_path_arg = '--recommender-options=prediction_file={}'.format(prediction_path_tmp)

            exe_line = ".\..\MyMediaLite\item_recommendation.exe --recommender=ExternalItemRecommender " \
                       "{} {} {}  --measures=AUC,prec@5,prec@10,MAP,recall@5,recall@10,NDCG,MRR" \
                       "".format(train_path_arg, test_path_arg, predictions_path_arg)

            result = os.popen(exe_line).read()
            parts = result.split('\n')[3].split(' ')

            df_stats = df_stats.append({'alg': method, 'AUC': parts[3], 'prec@5': parts[5], 'prec@10': parts[7],
                                        'MAP': parts[9],
                             'recall@5': parts[11], 'recall@10': parts[13], 'NDCG': parts[15], 'MRR': parts[17]},
                            ignore_index=True)
        return df_stats


    def numerize_data(self):

        # Datasets

        train_set = '..//resources//aggregated//train.csv'
        test_set = '..//resources//aggregated//test.csv'

        train_set = pd.read_csv(train_set, parse_dates=[3])
        test_set = pd.read_csv(test_set, parse_dates=[3])

        item_to_id_mapping = {}
        user_to_id_mapping = {}

        item_index = 0
        user_index = 0
        all_sets = pd.concat([train_set, test_set])
        for item in all_sets['itemID']:
            if item not in item_to_id_mapping.keys():
                item_to_id_mapping[item] = item_index
                item_index += 1
        for user in all_sets['userID']:
            if user not in user_to_id_mapping.keys():
                user_to_id_mapping[user] = user_index
                user_index += 1

        train_set['itemID'] = train_set['itemID'].map(item_to_id_mapping)
        test_set['itemID'] = test_set['itemID'].map(item_to_id_mapping)
        train_set['userID'] = train_set['userID'].map(user_to_id_mapping)
        test_set['userID'] = test_set['userID'].map(user_to_id_mapping)

        train_set.to_csv('..//resources//aggregated/train_numerized.csv')
        test_set.to_csv('..//resources//aggregated/test_numerized.csv')
