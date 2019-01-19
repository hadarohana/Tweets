import math
import random
import numpy as np
class RandomForest:

    # initializes values and trains the model
    def __init__(self, train_x, test_x, train_y, test_y, max_features, max_depth, min_leaf, n_trees ):
        #data passed in will consist of 1 dimensional array of averaged word vectors representing a tweet
        if max_features == 'log':
            self.n_features = (int)(math.log2(len(train_x)))
        elif max_features == 'sqrt':
            self.n_features = (int)(math.sqrt(len(train_x)))
        else:
            self.n_features = max_features
        self.test_x,  self.test_y= test_x, test_y
        model = Model(train_x, train_y, self.n_features)
        self.trees = model.build_model(max_depth, min_leaf, n_trees)

    # returns accuracy of test set on trained model
    def predict(self):
        trees = self.trees
        predictions = []
        for feature in self.test_x:
            predictions.append(Model.predict(trees, feature))
        correct = 0
        for i in range(len(predictions)):
            if(predictions[i] == self.test_y[i]):
                correct += 1
        return (float)(correct/len(predictions))


class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.left = None
        self.right = None
        self.split_index = None
        self.category = None  # category if node is a leaf

class Model:

    def __init__(self, x, y, n_features):
        self.x, self.y, self.n_features= x, y, n_features
    def build_model(self, max_depth, min_leaf, n_trees):
        trees = []
        for i in range(n_trees):
            trees.append(self.build_tree(0, self.x, self.y, max_depth, min_leaf))
        return trees

    # returns category predicted by most of the trees
    def predict(self, trees, feature):
        predictions = []
        for tree in trees:
            predictions.append(self.predict_single_tree(tree, feature))
        return max(set(predictions), key=predictions.count)

    # returns predicted category given a tree and feature
    def predict_single_tree(self, tree, feature):
        if tree.category is not None:
            return tree.category
        else:
            if feature < tree.x:
                return self.predict_single_tree(tree.left, feature)
            else:
                return self.predict_single_tree(tree.right, feature)

    # recursively builds the decision tree
    def build_tree(self, depth, x, y, max_depth, min_leaf):
        root = Node(x, y)
        root.split_index = self.get_split_point()
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
                self.left = self.build_tree(depth+1, left_x, left_y, self.n_features, max_depth, min_leaf)
            if(len(right_indexes) < min_leaf):
                self.right = Node(right_x, right_y)
                self.right.category = self.get_most_common_category(right_y)
            else:
                self.right = self.build_tree()(depth+1, right_x, right_y, self.n_features, max_depth, min_leaf)

    # returns class with most common occurence in binary classification
    def get_most_common_category(self, y):
        categories = set(y)
        return max(y.count(categories[0]), y.count(categories[1]))

    def get_feature_indexes(self):
        selected_indexes = [i for i in range(len(self.x))]
        random.shuffle(selected_indexes)
        return selected_indexes[:self.n_features]

    def get_split_point(self):
        feature_indexes = self.get_feature_indexes()
        split_feature, gini_index = None, None
        for feature_index in feature_indexes:       # feature_index is the index of the tweet vector to split by
            left, right = self.split(self.x, feature_index) # TODO: how to split this data????
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
                p = self.get_vals(self.y, split).count(polarity)
                score += p*p
            gini_index += (1-score)*len(split)/(len(left_split) + len(right_split))
        return gini_index

    # Helper function returning list of values at list of indexes
    def get_vals(list, indexes):
        vals = []
        for index in indexes:
            vals.append(list[index])