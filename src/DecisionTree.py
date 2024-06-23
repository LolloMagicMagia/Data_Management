import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from tensorflow.keras.models import save_model
import joblib

import Metrics
import Shap

class Tree:

    def __init__(self, df=None, X_train=None, X_test=None, y_train=None, y_test=None, target_name='Hazardous',
                 features=[], model_name='decision_tree'):

        self.metrics = None
        self.model = None
        self.target_name = target_name
        self.features = features
        self.model_name = model_name

        if df is not None:
            self.df = df
            self.generate_subsets()

        if df is None:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

        self.shap = None

    def generate_subsets(self, test_size=0.1):
        """
        Method to get the Training and Test Set from DataFrame
        :param test_size: Percentage of DataFrame in Test Sets
        :return: Calculate x_train, x_test, y_train, y_test for the class
        """
        y = self.df[self.target_name]

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.df[self.features], self.df[self.target_name], test_size=test_size, stratify=y,
                             random_state=42)

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def decision_tree_training(self, print_info=True):

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # ccp_alphas = potatura(X_train, y_train)
        best_tree = self.grid_search()
        best_tree_params = best_tree.best_params_

        # Valuta il modello utilizzando i dati di test
        y_pred = best_tree.predict(self.X_test)
        accuracy_train_test, precision, recall, f1 = self.metrics_model(y_pred)

        # Stampa le prestazioni del modello
        self.model = DecisionTreeClassifier(**best_tree_params)
        self.model.fit(self.X_train, self.y_train)

        auc, fpr_dt, tpr_dt = self.roc_curve_method(best_tree)

        self.metrics = Metrics.Metrics(accuracy_train_test, precision, recall, f1, auc)

        if print_info:
            self.print_metrics(best_tree, y_pred, fpr_dt, tpr_dt)
            return

        return self.metrics

    def print_metrics(self, model, y_pred, fpr_dt, tpr_dt):
        print('Accuracy:', self.metrics.accuracy)
        print('Precision:', self.metrics.precision)
        print('Recall:', self.metrics.recall)
        print('F1-score:', self.metrics.f1_score)
        print(classification_report(y_pred, self.y_test))
        print("\n")

        print('Matrice di confusione')
        self.confmatrix_plot(model)

        print("\n")
        fig, ax = plt.subplots(figsize=(15, 10))
        plot_tree(self.model, filled=True, ax=ax)
        plt.show()

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

    def confmatrix_plot(self, model):

        # Accepts as argument model object, x data (test or validate), and y data (test or validate).
        # Return a plot of confusion matrix for predictions on y data.
        model_pred = model.predict(self.X_test)

        # Ottenere le classi previste uniche
        classes = np.unique(np.concatenate((self.y_test, model_pred)))

        cm = confusion_matrix(self.y_test, model_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classes)

        disp.plot(values_format='')  # values_format='' suppresses scientific notation
        plt.grid(False)
        plt.show()

    def grid_search(self):
        # Definisci la griglia degli iperparametri
        param_distributions = {
            # 'ccp_alpha':ccp_alphas,
            'criterion': ['gini', 'entropy'],
            'max_depth': [50, 40, 30, 100, 200],
            'min_samples_split': [2, 5, 10, 20, 50],
            'max_features': [None, 'sqrt', 'log2'],
            # 'min_samples_leaf': [1, 2, 4],
            # 'min_impurity_decrease': [0.0, 0.01, 0.1],
            # 'max_leaf_nodes': [None, 50, 100],
            'random_state': [42],
        }

        # Usato in GridSearchCV
        best_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_distributions,
                                 cv=10,
                                 scoring="f1")

        # Ad esempio, supponiamo che tu abbia i dati x_train e y_train
        best_tree.fit(self.X_train, self.y_train)

        return best_tree

    def metrics_model(self, y_pred):
        # Valuta il modello utilizzando i dati di test
        accuracy_train_test = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        return accuracy_train_test, precision, recall, f1

    def potatura(self):
        # Genera il path di potatura
        model_tree = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=50, max_features=None,
                                            min_samples_split=2)
        path = model_tree.cost_complexity_pruning_path(self.X_train, self.y_train)
        ccp_alphas = path.ccp_alphas

        clfs = []
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(random_state=0, criterion='gini', max_depth=50, max_features=None,
                                         min_samples_split=2, ccp_alpha=ccp_alpha)
            clf.fit(self.X_train, self.y_train)
            clfs.append(clf)
        return ccp_alphas

    def roc_curve_method(self, model_tree):
        # Calcola le probabilità di predizione del modello dell'albero decisionale utilizzando predict_proba
        y_dt_pred_prob = model_tree.predict_proba(self.X_test)[:, 1]

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

        self.shap = Shap.Shap(joblib.load("models\\" + self.model_name + ".pkl"), self.X_train, self.X_test,
                              self.features, tree=True)
