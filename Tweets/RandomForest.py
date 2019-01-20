import math
import random
from sklearn.ensemble import RandomForestClassifier

# Random forest classifier implemented both from scratch and using the scikit-learn model
class RandomForest:

    # Initializes values and trains the model
    def __init__(self, train_x, test_x, train_y, test_y, max_features, max_depth, min_leaf, n_trees, model_type):
        #data passed in will consist of 300 dimensional array of averaged word vectors representing a tweet

        self.test_x, self.test_y = test_x, test_y
        self.model_type = model_type
        if model_type == 'my_model':
            if max_features == 'log2':
                self.n_features = (int)(math.log2(len(train_x[0])))
            elif max_features == 'sqrt':
                self.n_features = (int)(math.sqrt(len(train_x[0])))
            else:
                self.n_features = max_features
            self.model = Model(train_x, train_y, self.n_features)
            self.trees = self.model.build_model(max_depth, min_leaf, n_trees)
        if model_type == 'scikit_model':
            self.model = RandomForestClassifier(n_estimators=n_trees, min_leaf = min_leaf, max_depth=max_depth, max_features=max_features)
            self.trees = self.model.fit(X=train_x, y=train_y)


    # Returns accuracy of test set on trained model
    def evaluate(self):
        trees = self.trees
        if self.model_type == 'my_model':
            predictions = []
            for document in self.test_x:
                predictions.append(Model.predict(self = self.model, trees=trees, document=document))
            correct = 0.0
            for i in range(len(predictions)):
                if(predictions[i] == self.test_y[i]):
                    correct += 1
            return ((float)(correct))/len(predictions)

        if self.model_type == 'scikit_model':
            return trees.score(X=self.test_x, y=self.test_y)
# Decision tree node
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.left = None
        self.right = None
        self.split_point = None # (tweet index, feature index)
        self.category = None  # category if node is a leaf

# Random forest model implemented using gini index for impurity measure
class Model:
    def __init__(self, x, y, n_features):
        self.x, self.y, self.n_features= x, y, n_features

    # Initializes the forest
    def build_model(self, max_depth, min_leaf, n_trees):
        trees = []
        for i in range(n_trees):
            tree = self.build_tree(0, self.x, self.y, max_depth, min_leaf)
            trees.append(tree)
            print('tree' + str(i) + 'of' + str(n_trees))
        return trees

    # Returns category predicted by most of the trees
    def predict(self, trees, document):
        predictions = []
        for tree in trees:
            predictions.append(self.predict_single_tree(tree, document))
        return max(set(predictions), key=predictions.count)

    # Returns predicted category given a tree and feature
    def predict_single_tree(self, tree, document):
        if tree.category is not None:
            return tree.category
        row, col = tree.split_point
        split_val = tree.x[row][col]
        if document[col] < split_val:
            return self.predict_single_tree(tree.left, document)
        else:
            return self.predict_single_tree(tree.right, document)

    # Recursively builds the decision tree
    def build_tree(self, depth, x, y, max_depth, min_leaf):
        root = Node(x, y)
        row, col = self.get_split_point(x)
        left_indexes, right_indexes = self.split(x, row, col)
        left_x = list(x[i] for i in left_indexes)
        right_x = list(x[i] for i in right_indexes)
        left_y = list(y[i] for i in left_indexes)
        right_y = list(y[i] for i in right_indexes)
        if(depth >= max_depth or len(left_indexes) == 0 or len(right_indexes) == 0):
            root.category = self.get_most_common_category(y)
        else:
            root.split_point = (row, col)
            if(len(left_indexes) < min_leaf):
                root.left = Node(left_x, left_y)
                root.left.category = self.get_most_common_category(left_y)
            else:
                root.left = self.build_tree(depth+1, left_x, left_y, max_depth, min_leaf)
            if(len(right_indexes) < min_leaf):
                root.right = Node(right_x, right_y)
                root.right.category = self.get_most_common_category(right_y)
            else:
                root.right = self.build_tree(depth+1, right_x, right_y, max_depth, min_leaf)
        return root

    # Returns class with most common occurence in binary classification
    def get_most_common_category(self, y):
        return max(set(y), key =y.count)

    # Returns random selection of specified number of indexes for features to use
    def get_feature_indexes(self, x):
        selected_indexes = [i for i in range(len(x[0]))]
        random.shuffle(selected_indexes)
        return selected_indexes[:self.n_features]

    # Uses brute force to identify the optimal split point in terms of gini index
    def get_split_point(self, x):
        feature_indexes = self.get_feature_indexes(x)
        row, col, gini_index = None, None, None
        for row_index in range(len(x)):
            for feature_index in feature_indexes:       # feature_index is the index of the tweet vector to split by
                left, right = self.split(x, row_index, feature_index)
                curr_gini = self.get_gini(left, right)
                if(gini_index == None or curr_gini < gini_index):
                    gini_index = curr_gini
                    row = row_index
                    col = feature_index
        return row, col

    # Splits the data based on a specific feature and returns left and right lists of indexes
    def split(self, x, row_index, feature_index):
        left = []
        right = []
        selected_val = x[row_index][feature_index]
        for index in range(len(x)):
            if x[index][feature_index] < selected_val:
                left.append(index)
            else:
                right.append(index)
        return left, right

    # Returns gini index of given tree
    # left_split and right_split are lists of indexes of data for the proposed split
    def get_gini(self, left_split, right_split):
        gini_index = 0
        for split in left_split, right_split:
            score = 0
            if(len(split) == 0):
                continue
            for category in set(self.y):
                p = (float)(self.get_vals(list=self.y, indexes=split).count(category))/(len(split))
                score += p*p
            gini_index += (1-score)*len(split)/(len(left_split) + len(right_split))
        return gini_index

    # Helper function returning list of values at list of indexes
    def get_vals(self, list, indexes):
        vals = []
        for index in indexes:
            vals.append(list[index])
        return vals