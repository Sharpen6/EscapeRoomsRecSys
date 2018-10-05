import unittest
from recsys.top_n_algorithms import *
import pandas as pd
import numpy as np


class TestRecSysMethods(unittest.TestCase):


    def test_top_n_algorithms(self):
        train_set_path = '..//resources//aggregated//train_numerized.csv'
        test_set_path = '..//resources//aggregated//test_numerized.csv'
        train_set = pd.read_csv(train_set_path, parse_dates=[3])
        test_set = pd.read_csv(test_set_path, parse_dates=[3])

        rec_sys_algs = {
            "UserKNN": MyMdediaLiteRecMethod('UserKNN','correlation=Pearson'),
            #"BPRMF": MyMdediaLiteRecMethod('BPRMF', ''),
            #"ItemAttributeKNN ": MyMdediaLiteRecMethod('ItemAttributeKNN ', ''),
            #"ItemKNN": MyMdediaLiteRecMethod('ItemKNN', ''),
            #"MostPopular": MyMdediaLiteRecMethod('MostPopular', ''),
            #"Random": MyMdediaLiteRecMethod('Random', ''),
            #"UserAttributeKNN": MyMdediaLiteRecMethod('UserAttributeKNN', ''),
            #"WRMF": MyMdediaLiteRecMethod('WRMF', ''),
            #"Zero": MyMdediaLiteRecMethod('Zero', ''),
            #"MultiCoreBPRMF ": MyMdediaLiteRecMethod('MultiCoreBPRMF', ''),
            #"SoftMarginRankingMF": MyMdediaLiteRecMethod('SoftMarginRankingMF', ''),
            #"WeightedBPRMF": MyMdediaLiteRecMethod('WeightedBPRMF', ''),
            #"MostPopularByAttributes": MyMdediaLiteRecMethod('MostPopularByAttributes', ''),
            #"BPRSLIM": MyMdediaLiteRecMethod('BPRSLIM', ''),
            #"K-markov": k_markov_rc()
                        }

        algs_results = {}

        for name, model in rec_sys_algs.items():
            print(name)
            model.fit(train_set)
            results = model.get_top_n_recommendations(test_set, top_n=10)
            algs_results[name] = results

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

    def calc_precision_for_methods(self, test_set, algs_results):
        precision_ranks = {}
        df_user_ratings = test_set.groupby('userID')['itemID'].apply(list)
        for method, recs in algs_results.items():
            precision_scores = []
            for user, ratings in df_user_ratings.items():
                recommended = recs[str(user)]
                precision_scores.append(self.precision_at_k(ratings, recommended, 10))
            precision = np.mean(precision_scores)
            precision_ranks[method] = precision
        return precision_ranks

    def precision_at_k(self, actual, predicted, k=10):
        actual = [str(x) for x in actual]
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

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
