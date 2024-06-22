import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, make_scorer, auc
)
from tensorflow.keras import backend as K
import tensorflow as tf

import Shap
import Metrics

def _recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.cast(y_true, tf.float32) * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(tf.cast(y_true, tf.float32), 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def _precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(tf.cast(y_true, tf.float32) * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def _f1(y_true, y_pred):
    precision = _precision_m(y_true, y_pred)
    recall = _recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class DNN:

    def __init__(self, df, X_train, X_test, y_train, y_test, target_name, model_name='deep_neural_network'):
        self.df = df
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_name = target_name
        self.features = df.columns.tolist()

        if self.target_name in self.features:
            self.features.remove(self.target_name)

        self.model_name = model_name
        self.shap = None

        columns_to_remove = ['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Hazardous',
                             'Orbiting Body', 'Equinox']

        for col in columns_to_remove:
            if col in self.features:
                self.features.remove(col)



    def deep_neural_netw(self, print_info=True):
        # Preparazione dei dati
        input_shape = (self.X_train.shape[1],)
        num_classes = 1  # Per una classificazione binaria

        # Definizione del modello
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='sigmoid')
        ])

        # Compilazione del modello
        self.model.compile(loss='binary_crossentropy',
                           optimizer="adam",
                           metrics=['accuracy'])

        # Addestramento del modello
        self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=32, verbose=0)

        # Valutazione del modello
        self.model.evaluate(self.X_test, self.y_test)
        y_pred = (self.model.predict(self.X_test) > 0.5).astype("int32")

        accuracy_train_test, precision, recall, f1 = self.metrics_model(y_pred)

        auc, fpr_dt, tpr_dt = self.roc_curve_method(self.model)

        self.metrics = Metrics.Metrics(accuracy_train_test, precision, recall, f1, auc)

        if print_info:
            self.print_metrics(y_pred, fpr_dt, tpr_dt)
            return

        return self.metrics

    def metrics_model(self, y_pred):
        # Valuta il modello utilizzando i dati di test
        accuracy_train_test = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return accuracy_train_test, precision, recall, f1

    def print_metrics(self, y_pred, fpr_dt, tpr_dt):
        print('Accuracy:', self.metrics.accuracy)
        print('Precision:', self.metrics.precision)
        print('Recall:', self.metrics.recall)
        print('F1-score:', self.metrics.f1_score)
        print(classification_report(y_pred, self.y_test))
        print("\n")

        print('Matrice di confusione')
        self.confmatrix_plot()

        print("\n")

        self._save_model()

        print("ROC curve")
        # Plotta la curva ROC dei due modelli per comparare le performance
        plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {self.metrics.auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def confmatrix_plot(self):
        # Prevedi le classi con il modello
        model_pred = self.model.predict(self.X_test)
        model_pred_classes = (model_pred > 0.5).astype("int32").flatten()  # Flatten per trasformarlo in una dimensione

        # Ottenere le classi previste uniche
        classes = np.unique(np.concatenate((self.y_test, model_pred_classes)))

        cm = confusion_matrix(self.y_test, model_pred_classes, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

        disp.plot(values_format='')  # values_format='' sopprime la notazione scientifica
        plt.grid(False)
        plt.show()

    def roc_curve_method(self, model):
        # Calcola le probabilità di predizione del modello di rete neurale utilizzando predict
        y_dt_pred_prob = model.predict(self.X_test).ravel()

        fpr_dt, tpr_dt, thresholds_dt = roc_curve(self.y_test, y_dt_pred_prob)
        auc = roc_auc_score(self.y_test, y_dt_pred_prob)

        return auc, fpr_dt, tpr_dt

    def _save_model(self):
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_filename = os.path.join(model_dir, self.model_name + '.pkl')

        joblib.dump(self.model, model_filename)

    def shap_init(self):

        if not os.path.exists("models\\" + self.model_name + ".pkl"):
            self._save_model()

        self.shap = Shap.Shap(joblib.load("models/" + self.model_name + ".pkl"), self.X_train, self.X_test,
                              self.features, tree=False)
