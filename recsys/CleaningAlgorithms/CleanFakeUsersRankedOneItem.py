from CleaningAlgorithms.CleanFakeUsers import AbstCleanFakeUsers

class CleanFakeUsersRankedOne(AbstCleanFakeUsers):
    def __init__(self):
        self.cached_training_set = None

    def clean(self, train_set):
        if self.cached_training_set is None:
            user_ratings = train_set.groupby('userID')['itemID'].apply(list)
            fake_users = user_ratings[user_ratings.apply(lambda x: len(x) <= 1)]
            train_set_new = train_set[~train_set['userID'].isin(fake_users.keys().tolist())]
            self.cached_training_set = train_set_new.copy()
            return train_set_new
        else:
            return self.cached_training_set

    def print(self):
        return 'Ranked 1 Item'