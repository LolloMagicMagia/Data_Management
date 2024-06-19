import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import ML_model_tree as tree
import Deep_neural_netw as neural

target_name = 'Hazardous'


class DataFrame:
    metrics_tree = None
    metrics_dnn = None

    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        accuracy, precision, recall, f1_score, auc = tree.ml_albero(self.x_train, self.x_test, self.y_train, self.y_test, name_modello='',
                                                                    print_info=False)
        self.metrics_tree = Metrics(accuracy, precision, recall, f1_score, auc)

        accuracy, precision, recall, f1_score = neural.deep_neural_netw(self.x_train, self.x_test, self.y_train, self.y_test,
                                                                        name_modello='', print_info=False)
        self.metrics_dnn = Metrics(accuracy, precision, recall, f1_score, 0)


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
