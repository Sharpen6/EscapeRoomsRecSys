import pandas as pd

# Datasets

train_set = '..//resources//aggregated//train_with_anon.csv'
test_set = '..//resources//aggregated//test_with_anon.csv'

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

train_set.to_csv('..//resources//aggregated/train_numerized_with_anon.csv')
test_set.to_csv('..//resources//aggregated/test_numerized_with_anon.csv')

