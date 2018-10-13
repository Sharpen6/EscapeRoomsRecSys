from CleaningAlgorithms.CleanFakeUsers import AbstCleanFakeUsers

class CleanFakeUsersRankedOne(AbstCleanFakeUsers):
    def __init__(self):
        pass

    def clean(self, train_set):
        user_ratings = train_set.groupby('userID')['itemID'].apply(list)
        fake_users = user_ratings[user_ratings.apply(lambda x: len(x) <= 1)]
        train_set_new = train_set[~train_set['userID'].isin(fake_users.keys().tolist())]
        return train_set_new

    def print(self):
        return 'Ranked 1 Item'