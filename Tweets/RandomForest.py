import math
import random
import numpy as np
class RandomForest:
    def __init__(self, max_features, n_features, min_leaf, n_estimators, depth, train_x, test_x, train_y, test_y, sample_size):
        #data passed in will consist of 1 dimensional array of averaged word vectors representing a tweet
        if max_features == 'log':
            self.n_features = math.log2(train_x.shape[0])
        elif max_features == 'sqrt':
            self.n_features = math.sqrt(train_x.shape[0])
        else:
            self.n_features = n_features
        self.train_x, self.test_x, self.train_y, self.test_y, self.min_leaf, self.depth, self.sample_size = \
            train_x, test_x, train_y, test_y, min_leaf, depth, sample_size
        self.trees = []
        for i in range(n_estimators):
            self.trees.append(self.generateTree())

    def generateTree(self):
        rows = np.random.permuation(self.train_x.shape[0])[:self.sample_size] # random row indexes to use
        feature_indexes = np.random.permuation(self.train_x.shape[0])[:self.n_features] # random feature indexes
        return DecisionTree(x=self.train_x[rows], y=self.train_y[rows], n_features=self.n_features,
                     feature_indexes= feature_indexes, indexes=np.arrange(self.sample_size), depth=self.depth, min_leaf=self.min_leaf)
#need: map from
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.left = None
        self.right = None
        self.split_index = None
        self.category = None  # category if node is a leaf

class Model:
    def __init__(self, x, y, n_features, max_depth, min_leaf, n_trees):
        self.x, self.y, self.n_features, self.depth, self.min_leaf = x, n_features, depth, min_leaf
        trees = []
        for i in range(n_trees):
            trees.append(self.build_tree(0, x, y, n_features, max_depth, min_leaf))

    # recursively builds the decision tree
    def build_tree(self, depth, x, y, n_features, max_depth, min_leaf):
        root = Node(x, y)
        root.split_index = self.get_split_point(n_features)
        left_indexes, right_indexes = self.split(x, root.split_index)
        left_x = list(x[i] for i in left_indexes)
        right_x = list(x[i] for i in right_indexes)
        left_y = list(y[i] for i in left_indexes)
        right_y = list(y[i] for i in right_indexes)
        if(depth >= max_depth or len(left_indexes) == 0 or len(right_indexes) == 0):
            root.category = self.get_most_common_category(y)
        else:
            if(len(left_indexes) < min_leaf):
                self.left = Node(left_x, left_y)
                self.left.category = self.get_most_common_category(left_y)
            else:
                self.left = self.build_tree(depth+1, left_x, left_y, n_features, max_depth, min_leaf)
            if(len(right_indexes) < min_leaf):
                self.right = Node(right_x, right_y)
                self.right.category = self.get_most_common_category(right_y)
            else:
                self.right = self.build_tree()(depth+1, right_x, right_y, n_features, max_depth, min_leaf)

    # returns class with most common occurence in binary classification
    def get_most_common_category(self, y):
        categories = set(y)
        return max(y.count(categories[0]), y.count(categories[1]))

    def get_feature_indexes(n_selected_features, self):
        selected_features = [i for i in range(self.n_features)]
        random.shuffle(selected_features)
        return selected_features[:n_selected_features]

    def get_split_point(n_selected_features, self):
        categories = [0, 4]
        feature_indexes = self.get_feature_indexes(n_selected_features)
        split_feature, gini_index = None, None
        for feature_index in feature_indexes:       # feature_index is the index of the tweet vector to split by
            left, right = self.split(self.x, feature_index)
            curr_gini = self.get_gini(left, right)
            if(curr_gini < gini_index):
                gini_index = curr_gini
                split_feature = feature_index
        return split_feature

    #splits the data based on a specific feature and returns left and right lists of indexes
    def split(self, x, feature_index):
        left = []
        right = []
        selected_val = x[feature_index]
        for index in range(len(x)):
            if x[index] < selected_val:
                left.append(index)
            else:
                right.append(index)
        return left, right

    # Returns gini index of given tree
    # left and right are lists of indexes of data for the proposed split
    def get_gini(self, left_split, right_split):
        gini_index = 0
        for split in left_split, right_split:
            score = 0
            for polarity in [0,4]:
                p = self.get_count(self.y[split], polarity)
                score += p*p
            gini_index += (1-score)*len(split)/(len(left_split) + len(right_split))
        return gini_index
    # Helper function returning number of times value appears in list
    def get_count(list, val):
        count = 0
        for i in list:
            if(list[i] == val):
                count += 1
        return count