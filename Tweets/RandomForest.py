import math
class RandomForest:
    def __init__(self, max_features, n_features, min_leaf, n_estimators, depth, train_x, test_x, train_y, test_y):
        if max_features == 'log':
            self.n_features = math.log2(train_x.shape[0])