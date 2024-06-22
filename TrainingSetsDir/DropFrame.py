import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import DecisionTree as tree
import DeepNeuralNetwork as neural

target_name = 'Hazardous'


class DataFrame:
    metrics_tree = None
    metrics_dnn = None

    def __init__(self, X_tree_train, X_tree_test, y_tree_train, y_tree_test, X_dnn_train, X_dnn_test, y_dnn_train,
                 y_dnn_test, df, i, features):

        self.tree1 = tree.Tree(df=None, X_train=X_tree_train, X_test=X_tree_test, y_train=y_tree_train,
                          y_test=y_tree_test, model_name=f'dropped{i}_tree', features=features)
        self.metrics_tree = self.tree1.decision_tree_training(False)

        self.dnn = neural.DNN(df=df, X_train=X_dnn_train, X_test=X_dnn_test, y_train=y_dnn_train,
                         y_test=y_dnn_test, model_name=f'dropped{i}_dnn', target_name='Hazardous')
        self.metrics_dnn = self.dnn.deep_neural_netw(False)


class Metrics:
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0
    auc = 0

    def __init__(self, accuracy, precision, recall, f1_score, auc):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.auc = auc
