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
    df = None
    error_type = None
    percentage = 0
    metrics_tree = None
    metrics_dnn = None

    def __init__(self, df, error_type, percentage, x_test, y_test):
        self.df = df
        self.error_type = error_type
        self.percentage = percentage

        X_train = df.copy()
        X_train = df.drop(target_name, axis=1)
        y_train = df[target_name]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        accuracy, precision, recall, f1_score, auc = tree.ml_albero(X_train, x_test, y_train, y_test, name_modello='',
                                                                    print_info=False)
        self.metrics_tree = Metrics(accuracy, precision, recall, f1_score, auc)

        accuracy, precision, recall, f1_score = neural.deep_neural_netw(X_train, x_test, y_train, y_test,
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
