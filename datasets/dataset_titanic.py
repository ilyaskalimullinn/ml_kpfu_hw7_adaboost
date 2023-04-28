import pandas as pd


class Titanic:

    def __init__(self, train_path, test_path):
        self.train_set_csv = pd.read_csv(train_path)
        self.test_set_csv = pd.read_csv(test_path)

    def __call__(self):
        return {'train_input': self.train_set_csv.values[:, 3:],
                'train_target': 2 * self.train_set_csv.values[:, 2] - 1,
                'test_input': self.test_set_csv.values[:, 3:],
                'test_target': 2 * self.test_set_csv.values[:, 2] - 1
                }
