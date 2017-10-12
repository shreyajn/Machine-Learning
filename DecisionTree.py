from __future__ import division, print_function
import numpy as np
from sklearn import datasets
from itertools import combinations_with_replacement
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import os
import math
import scipy.io
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import itertools
import progressbar

#shuffle train and test data 
def shuff_d(X, y, seed=1):
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]
#Divide dataset based on if sample value on feature index is larger than the given threshold 
def divide_on_feature(X, feature_i, threshold):
    
    #print(X,feature_i,threshold)
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    #else:
        #split_func = lambda sample: sample[feature_i] == threshold --not needed

    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    #print(X_1,X_2)
    return np.array([X_1, X_2])

#splitting the data and we can shuffle as needed

def traintestsplit(X, y, test_size=0.5, shuffle=True,seed=None):
    
    if shuffle:
        X, y = shuff_d(X, y, seed)
    
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


#uncertinaty measure 
def entropy(y):
    """ Calculate the entropy of label array y """
    #print("a")
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy


#to check accuracy of predictions
def accuracy_score(y_true, y_pred):

    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

'''def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}
    rows = list(itertools.chain(*rows))
    #print(rows)  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    #print(counts)
    
    return counts

def gini(y):
    count = class_counts(y)
    impurity = 1
    for unique in count:
        prob = count[unique]/float(len(y))
        impurity -= prob ** 2
    return impurity'''



#========================================
class DecisionNode():
  
    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
 
    def __init__(self, min_samples_split=10, min_impurity=1e-7,
                 max_depth=100, loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss=None

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""
        acc_train =[]
        acc_test=[]
        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        #print(y1,y2)
                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)
                        #print(impurity)
                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }


        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)

            print(True)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)

        return DecisionNode(value=leaf_value)


    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print (tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print ("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print ("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print ("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)
#================================================================

class ClassificationTree(DecisionTree):
    def _calculate_information_gain(self, y, y1, y2):
        #print("entered")
        # Calculate information gain
        p = len(y1) / len(y)
        e = entropy(y)
        info_gain = e - p * \
            entropy(y1) - (1 - p) * \
            entropy(y2)

        return info_gain

    def _majority_vote(self, y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        self._impurity_calculation = self._calculate_information_gain
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

#=====================================================

def main():
    K = sys.argv[0]
    mat = scipy.io.loadmat('/Users/shreyajain/Downloads/hw1data.mat')
    X = mat["X"].astype(float)
    Y = mat["Y"].astype(float)
   # 
    
    '''print ("Starting")
    std_list = []
    for row in range(X.shape[1]):
        print("Iterate")
        std = np.std(X[row:])
        std_list.append([row, std])

    std_list = sorted( std_list, key=lambda x : x[1], reverse = True)

    print(std_list)
'''

    X = SelectKBest(chi2, k=200).fit_transform(X, Y)


    '''print ("Starting")
    std_list = []
    for row in range(X.shape[1]):
        print("Iterate")
        std = np.std(X[row:])
        std_list.append([row, std])

    std_list = sorted( std_list, key=lambda x : x[1], reverse = True)

    print(std_list)'''

    print(X.shape[0])
    for row in range(X.shape[0]):
        for col in range(X.shape[1]):
            if X[row,col] > 0:
                X[row,col] = 1
    #

    y = list(itertools.chain(*Y))
    y = np.array(y)
	#X_new = X_new[100,:]
	#print(X_new.shape)
	#print(y.shape)
    print(X)
    X_train, X_test, y_train, y_test = traintestsplit(X, y, test_size=0.2)
    print(X_train,X_test)
    #print(X_new,y)
    clf = ClassificationTree()
    clf.fit(X_train, y_train)
	
    clf.print_tree()
    y_pred1 = clf.predict(X_test)
    accuracy1 = accuracy_score(y_test, y_pred1)
    print("Accuracy test:", accuracy1)
    y_pred2 = clf.predict(X_train)
    accuracy2 = accuracy_score(y_train, y_pred2)
    print("Accuracy train:", accuracy2)

if __name__ == "__main__":
    main()
