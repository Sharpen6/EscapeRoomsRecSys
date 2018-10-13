from CleaningAlgorithms.CleanFakeUsers import AbstCleanFakeUsers

class CleanFakeUsersNone(AbstCleanFakeUsers):
    def __init__(self):
        pass

    def clean(self, train_set):
        return train_set

    def print(self):
        return 'No Clean'