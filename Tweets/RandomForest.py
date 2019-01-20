import math
import random
import numpy as np
class RandomForest:

    # initializes values and trains the model
    def __init__(self, train_x, test_x, train_y, test_y, max_features, max_depth, min_leaf, n_trees ):
        #data passed in will consist of 1 dimensional array of averaged word vectors representing a tweet
        if max_features == 'log':
            self.n_features = (int)(math.log2(len(train_x[0])))
        elif max_features == 'sqrt':
            self.n_features = (int)(math.sqrt(len(train_x[0])))
        else:
            self.n_features = max_features
        self.test_x,  self.test_y= test_x, test_y
        self.model = Model(train_x, train_y, self.n_features)
        self.trees = self.model.build_model(max_depth, min_leaf, n_trees)

    # returns accuracy of test set on trained model
    def evaluate(self):
        trees = self.trees
        predictions = []
        for document in self.test_x:
            predictions.append(Model.predict(self = self.model, trees=trees, document=document))
        correct = 0
        for i in range(len(predictions)):
            if(predictions[i] == self.test_y[i]):
                correct += 1
        return ((float)(correct))/len(predictions)



class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.left = None
        self.right = None
        self.split_point = None # (tweet index, feature index)
        self.category = None  # category if node is a leaf

class Model:

    def __init__(self, x, y, n_features):
        self.x, self.y, self.n_features= x, y, n_features

    def build_model(self, max_depth, min_leaf, n_trees):
        trees = []
        for i in range(n_trees):
            tree = self.build_tree(0, self.x, self.y, max_depth, min_leaf)
            trees.append(tree)
        return trees

    # returns category predicted by most of the trees
    def predict(self, trees, document):
        predictions = []
        print(len(trees))
        for tree in trees:
            predictions.append(self.predict_single_tree(tree, document))
        return max(set(predictions), key=predictions.count)

    # returns predicted category given a tree and feature
    def predict_single_tree(self, tree, document):
        if tree.category is not None:
            return tree.category
        row, col = tree.split_point
        split_val = tree.x[row][col]
        if document[col] < split_val:
            return self.predict_single_tree(tree.left, document)
        else:
            return self.predict_single_tree(tree.right, document)

    # recursively builds the decision tree
    def build_tree(self, depth, x, y, max_depth, min_leaf):
        root = Node(x, y)
        root.split_index = self.get_split_point(x)
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
                self.left = self.build_tree(depth+1, left_x, left_y, max_depth, min_leaf)
            if(len(right_indexes) < min_leaf):
                self.right = Node(right_x, right_y)
                self.right.category = self.get_most_common_category(right_y)
            else:
                self.right = self.build_tree(depth+1, right_x, right_y, max_depth, min_leaf)
        return root

    # returns class with most common occurence in binary classification
    def get_most_common_category(self, y):
        return max(set(y), key =y.count)

    def get_feature_indexes(self, x):
        selected_indexes = [i for i in range(len(x[0]))]
        random.shuffle(selected_indexes)
        return selected_indexes[:self.n_features]

    def get_split_point(self, x):
        feature_indexes = self.get_feature_indexes(x)
        row, col, gini_index = None, None, None
        for row_index in range(len(x)):
            for feature_index in feature_indexes:       # feature_index is the index of the tweet vector to split by
                left, right = self.split(x, row_index, feature_index) # TODO: how to split this data????
                curr_gini = self.get_gini(left, right)
                if(gini_index == None or curr_gini < gini_index):
                    gini_index = curr_gini
                    row = row_index
                    col = feature_index
        return row, col

    #splits the data based on a specific feature and returns left and right lists of indexes
    def split(self, x, row_index, feature_index):
        left = []
        right = []
        selected_val = x[row_index][feature_index]
        print(feature_index)
        print(len(x))
        for index in range(len(x)):
            print(index)
            if x[index][feature_index] < selected_val:
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
                p = (self.get_vals(list=self.y, indexes=split)).count(polarity)
                score += p*p
            gini_index += (1-score)*len(split)/(len(left_split) + len(right_split))
        return gini_index

    # Helper function returning list of values at list of indexes
    def get_vals(self, list, indexes):
        vals = []
        for index in indexes:
            vals.append(list[index])
        return vals