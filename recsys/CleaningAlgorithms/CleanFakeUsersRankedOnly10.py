from CleaningAlgorithms.CleanFakeUsers import AbstCleanFakeUsers

class CleanFakeUsersRankedOnlyTen(AbstCleanFakeUsers):
    def __init__(self):
        self.cached_training_set = None
        pass

    def clean(self, train_set):
        if self.cached_training_set is None:
            user_ratings = train_set.groupby('userID')['rating'].apply(list)
            fake_users = user_ratings[user_ratings.apply(lambda x: all(e == '10' for e in x))]
            train_set_new = train_set[~train_set['userID'].isin(fake_users.keys().tolist())]
            self.cached_training_set = train_set_new.copy()
            return self.cached_training_set
        else:
            return self.cached_training_set

    def print(self):
        return 'Ranked Only 10'