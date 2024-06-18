import Ml_model_tree as tree
import NoisyDataGeneration as ndg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import DataCleaner as dc

def generate_dirty_trainset(x_train, x_test, y_train, y_test):

    for percentage in range(10, 100, 10):
        percentage_decimal = percentage / 100

        # Creazione dataset con outliers
        df_outliers = ndg.create_outliers(x_train, percentage_decimal)
        tree.ml_albero(df_outliers, x_test, y_train, y_test)

        # Creazione dataset con valori inconsistenti
        df_inconsistents = ndg.create_inconsistents(x_train, percentage_decimal)
        tree.ml_albero(df_inconsistents, x_test, y_train, y_test)

        # Creazione dataset con valori nulli in ogni colonna numerica
        df_nulls = x_train.copy()
        for col in x_train.select_dtypes(include=[np.number]).columns:
            df_nulls = ndg.create_null(df_nulls, percentage_decimal, col)
        tree.ml_albero(dc.clean_dataset_with_learn2clean(df_nulls), x_test, y_train, y_test)

        # Creazione dataset con outliers e valori inconsistenti
        df_outliers_inconsistents = ndg.create_outliers(x_train, percentage_decimal)
        df_outliers_inconsistents = ndg.create_inconsistents(df_outliers_inconsistents, percentage_decimal)
        tree.ml_albero(df_outliers_inconsistents, x_test, y_train, y_test)

        # Creazione dataset con outliers e valori nulli
        df_outliers_nulls = ndg.create_outliers(x_train, percentage_decimal)
        for col in df_outliers_nulls.select_dtypes(include=[np.number]).columns:
            df_outliers_nulls = ndg.create_null_column(df_outliers_nulls, percentage_decimal, col)
        tree.ml_albero(df_outliers_nulls, x_test, y_train, y_test)

        # Creazione dataset con valori inconsistenti e valori nulli
        df_inconsistents_nulls = ndg.create_inconsistents(x_train, percentage_decimal)
        for col in df_inconsistents_nulls.select_dtypes(include=[np.number]).columns:
            df_inconsistents_nulls = ndg.create_null_column(df_inconsistents_nulls, percentage_decimal, col)
        tree.ml_albero(df_inconsistents_nulls, x_test, y_train, y_test)

    print("Tutti i dataset sporchi sono stati creati e salvati con successo!")



file_path = '../input/nasa.csv'
df = pd.read_csv(file_path)
columns_to_remove = ['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Hazardous', 'Orbiting Body', 'Equinox']


target = df["Hazardous"]
features = df.drop(columns=columns_to_remove, axis=1)

x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

generate_dirty_trainset(x_train, x_test, y_train, y_test)