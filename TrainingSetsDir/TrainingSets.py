import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import DirtyFrame
import DropFrame

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import NoisyDataGeneration as ndg


def dirty_train(df_tree, df_dnn, target_name='Hazardous', test_size=0.15):

    features_tree = df_tree.columns.tolist()
    features_tree.remove(target_name)
    features_dnn = df_dnn.columns.tolist()
    features_dnn.remove(target_name)

    dataframe = None

    y = df_dnn[target_name]

    X_tree_train_base, X_tree_test, y_tree_train, y_tree_test = \
        train_test_split(df_tree[features_tree], df_tree[target_name], test_size=test_size,
                         stratify=y,
                         random_state=42)

    X_dnn_train_base, X_dnn_test, y_dnn_train, y_dnn_test = \
        train_test_split(df_dnn[features_dnn], df_dnn[target_name], test_size=test_size,
                         stratify=y,
                         random_state=42)

    datasets_tree = ndg.create_dirty_dataset(pd.DataFrame(X_tree_train_base, columns=features_tree))
    datasets_dnn = ndg.create_dirty_dataset(pd.DataFrame(X_dnn_train_base, columns=features_dnn))

    columns = ['Dataset Name', 'Tree Accuracy', 'Tree Precision', 'Tree Recall', 'Tree F1 Score',
               'Tree Clean Auc', 'Tree Clean Accuracy', 'Tree Clean Precision', 'Tree Clean Recall',
               'Tree Clean F1 Score', 'Tree Clean Auc',
               'DNN Accuracy', 'DNN Precision', 'DNN Recall', 'DNN F1 Score', 'DNN Auc']

    for key, value in datasets_tree.items():
        error_type = key[(key.find('-') + 1):key.find('/')]
        percentage = key[(key.rfind('_') + 1):]

        if error_type == 'null' or error_type == 'outlier':
            if percentage == 10 or percentage == '10':
                continue

        print(f'Start of {error_type}_{percentage} Dataset')

        X_tree_train = datasets_tree[key].values
        X_dnn_train = datasets_dnn[key].values

        df_clean = pd.DataFrame(X_tree_train, columns=features_tree)
        df_clean[target_name] = y_tree_train

        df_clean = clean_data(df_clean)

        X_tree_clean = df_clean[features_tree].values

        trained = DirtyFrame.DataFrame(error_type, percentage, X_tree_train, X_tree_test, y_tree_train, y_tree_test,
                                       X_tree_clean, X_dnn_test, y_dnn_train, y_dnn_test, X_tree_clean, y_tree_train,
                                       df=df_dnn)

        dir_name = key[0:key.find('/')]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        trained.tree1.shap_init()
        trained.tree1.shap.print_beeswarm(dir_name + f'/{error_type}_{percentage}_tree')
        trained.tree2.shap_init()
        trained.tree2.shap.print_beeswarm(dir_name + f'/{error_type}_{percentage}_clean')
        trained.dnn.shap_init()
        trained.dnn.shap.print_beeswarm(dir_name + f'/{error_type}_{percentage}_dnn')

        txt_file_path = os.path.join(dir_name, f"{error_type}_{percentage}.txt")
        with open(txt_file_path, 'w') as file:
            file.write(f"{error_type} {percentage}%\n")
            file.write('Tree Metrics:\n')
            file.write(f'\tAccuracy: {trained.metrics_tree.accuracy}\n')
            file.write(f'\tPrecision: {trained.metrics_tree.precision}\n')
            file.write(f'\tRecall: {trained.metrics_tree.recall}\n')
            file.write(f'\tF1 Score: {trained.metrics_tree.f1_score}\n')
            file.write(f'\tAUC: {trained.metrics_tree.auc}\n')
            file.write('Tree Clean Metrics:\n')
            file.write(f'\tAccuracy: {trained.metrics_tree_clean.accuracy}\n')
            file.write(f'\tPrecision: {trained.metrics_tree_clean.precision}\n')
            file.write(f'\tRecall: {trained.metrics_tree_clean.recall}\n')
            file.write(f'\tF1 Score: {trained.metrics_tree_clean.f1_score}\n')
            file.write(f'\tAUC: {trained.metrics_tree_clean.auc}\n')
            file.write('Deep Neural Network Metrics:\n')
            file.write(f'\tAccuracy: {trained.metrics_dnn.accuracy}\n')
            file.write(f'\tPrecision: {trained.metrics_dnn.precision}\n')
            file.write(f'\tRecall: {trained.metrics_dnn.recall}\n')
            file.write(f'\tF1 Score: {trained.metrics_dnn.f1_score}\n')
            file.write(f'\tAUC: {trained.metrics_dnn.auc}\n')
        if dataframe is None:
            data = {
                'Dataset Name': [f"{error_type}{percentage}"],
                'Tree Accuracy': [trained.metrics_tree.accuracy],
                'Tree Precision': [trained.metrics_tree.precision],
                'Tree Recall': [trained.metrics_tree.recall],
                'Tree F1 Score': [trained.metrics_tree.f1_score],
                'Tree Auc': [trained.metrics_tree.auc],
                'Tree Clean Accuracy': [trained.metrics_tree_clean.accuracy],
                'Tree Clean Precision': [trained.metrics_tree_clean.precision],
                'Tree Clean Recall': [trained.metrics_tree_clean.recall],
                'Tree Clean F1 Score': [trained.metrics_tree_clean.f1_score],
                'Tree Clean Auc': [trained.metrics_tree_clean.auc],
                'DNN Accuracy': [trained.metrics_dnn.accuracy],
                'DNN Precision': [trained.metrics_dnn.precision],
                'DNN Recall': [trained.metrics_dnn.recall],
                'DNN F1 Score': [trained.metrics_dnn.f1_score],
                'DNN Auc': [trained.metrics_dnn.auc]
            }
            dataframe = pd.DataFrame(data=data, columns=columns)
        else:
            data = {
                'Dataset Name': f"{error_type}{percentage}",
                'Tree Accuracy': trained.metrics_tree.accuracy,
                'Tree Precision': trained.metrics_tree.precision,
                'Tree Recall': trained.metrics_tree.recall,
                'Tree F1 Score': trained.metrics_tree.f1_score,
                'Tree Auc': trained.metrics_tree.auc,
                'Tree Clean Accuracy': trained.metrics_tree_clean.accuracy,
                'Tree Clean Precision': trained.metrics_tree_clean.precision,
                'Tree Clean Recall': trained.metrics_tree_clean.recall,
                'Tree Clean F1 Score': trained.metrics_tree_clean.f1_score,
                'Tree Clean Auc': trained.metrics_tree_clean.auc,
                'DNN Accuracy': trained.metrics_dnn.accuracy,
                'DNN Precision': trained.metrics_dnn.precision,
                'DNN Recall': trained.metrics_dnn.recall,
                'DNN F1 Score': trained.metrics_dnn.f1_score,
                'DNN Auc': trained.metrics_dnn.auc
            }
            dataframe.loc[len(dataframe)] = data

        print(f'End of {error_type}_{percentage} Dataset')

    dataframe.to_csv('../TrainingSetsDir/dirtysets_trained_metrics.csv', index=False)


def drop_train(df_tree, df_dnn, target_name='Hazardous', test_size=0.15):
    features_tree = df_tree.columns.to_list()
    features_tree.remove(target_name)
    features_dnn = df_dnn.columns.to_list()
    features_dnn.remove(target_name)

    dataframe = None

    y = df_dnn[target_name]

    X_tree_train, X_tree_test, y_tree_train, y_tree_test = \
        train_test_split(df_tree[features_tree], df_tree[target_name], test_size=0.15,
                         stratify=y,
                         random_state=42)

    X_dnn_train, X_dnn_test, y_dnn_train, y_dnn_test = \
        train_test_split(df_dnn[features_dnn], df_dnn[target_name], test_size=0.15,
                         stratify=y,
                         random_state=42)

    columns = ['Dropped Features', 'Tree Accuracy', 'Tree Precision', 'Tree Recall', 'Tree F1 Score',
               'Tree Clean Auc', 'DNN Accuracy', 'DNN Precision', 'DNN Recall', 'DNN F1 Score', 'DNN Auc']

    columns_to_drop = ['Absolute Magnitude', 'Minimum Orbit Intersection', 'Est Dia in KM(min)']
    drop_column = get_columns(columns_to_drop)

    i = 0
    for combo in drop_column:
        X_tree_train = drop_features(X_tree_train_base, features_tree[:], combo)
        X_tree_test = drop_features(X_tree_test_base, features_tree[:], combo)
        X_dnn_train = drop_features(X_dnn_train_base, features_dnn[:], combo)
        X_dnn_test = drop_features(X_dnn_test_base, features_dnn[:], combo)

        features_drop_tree = features_tree[:]
        features_drop_dnn = features_dnn[:]
        for f in combo:
            if f in features_drop_tree:
                features_drop_tree.remove(f)
            if f in features_drop_dnn:
                features_drop_dnn.remove(f)

        trained = DropFrame.DataFrame(X_tree_train, X_tree_test, y_tree_train, y_tree_test,
                                      X_dnn_train, X_dnn_test, y_dnn_train, y_dnn_test, df=df_dnn, i=i,
                                      features=features_drop_dnn)

        i += 1

        dir_name = 'dropsets'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        trained.tree1.shap_init()
        trained.tree1.shap.print_beeswarm(dir_name + f'/model_{i}_tree')
        trained.dnn.shap_init()
        trained.dnn.shap.print_beeswarm(dir_name + f'/model_{i}_dnn')

        if dataframe is None:
            data = {
                'Dropped Features': [f"{combo}"],
                'Tree Accuracy': [trained.metrics_tree.accuracy],
                'Tree Precision': [trained.metrics_tree.precision],
                'Tree Recall': [trained.metrics_tree.recall],
                'Tree F1 Score': [trained.metrics_tree.f1_score],
                'Tree Auc': [trained.metrics_tree.auc],
                'DNN Accuracy': [trained.metrics_dnn.accuracy],
                'DNN Precision': [trained.metrics_dnn.precision],
                'DNN Recall': [trained.metrics_dnn.recall],
                'DNN F1 Score': [trained.metrics_dnn.f1_score],
                'DNN Auc': [trained.metrics_dnn.auc]
            }
            dataframe = pd.DataFrame(data=data, columns=columns)
        else:
            data = {
                'Dropped Features': f"{combo}",
                'Tree Accuracy': trained.metrics_tree.accuracy,
                'Tree Precision': trained.metrics_tree.precision,
                'Tree Recall': trained.metrics_tree.recall,
                'Tree F1 Score': trained.metrics_tree.f1_score,
                'Tree Auc': trained.metrics_tree.auc,
                'DNN Accuracy': trained.metrics_dnn.accuracy,
                'DNN Precision': trained.metrics_dnn.precision,
                'DNN Recall': trained.metrics_dnn.recall,
                'DNN F1 Score': trained.metrics_dnn.f1_score,
                'DNN Auc': trained.metrics_dnn.auc
            }
            dataframe.loc[len(dataframe)] = data

    dataframe.to_csv('../TrainingSetsDir/dropsets_trained_metrics.csv', index=False)


def get_columns(features):
    columns = []
    for f1 in features:
        for f2 in features:
            for f3 in features:
                col = []
                col.append(f1)
                if f1 != f2:
                    col.append(f2)
                if f3 != f2 and f3 != f1:
                    col.append(f3)

                col.sort()

                if col not in columns:
                    columns.append(col)

    return columns

def drop_features(X, columns, columns_to_remove):
    X1 = pd.DataFrame(data=X, columns=columns)
    X1 = X1.drop(columns_to_remove, axis=1)
    return X1.values


def load_csvs_from_disk(base_dir='../dirty_datasets'):
    """
    Carica tutti i file CSV presenti nelle sottocartelle della cartella 'dirty_datasets',
    associandoli con il nome della directory di provenienza.

    Args:
        base_dir (str): Il percorso della cartella 'dirty_datasets'. Default è '../dirty_datasets'.

    Returns:
        dict: Un dizionario con i nomi delle directory e i percorsi dei file CSV come chiavi, e i DataFrame caricati come valori.
        Pd.DataFrame: Il DataFrame di Test
    """
    csv_files = {}
    testset = None

    for root, dirs, files in os.walk(base_dir):
        # Ignora la directory base, considera solo le sottodirectory
        if root != base_dir:
            dir_name = os.path.basename(root)
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    key = f"{dir_name}/{file}"
                    csv_files[key] = pd.read_csv(file_path)

        else:
            for file in files:
                if file.endswith('test.csv'):
                    file_path = os.path.join(root, file)
                    testset = pd.read_csv(file_path)

    return csv_files, testset


def handle_missing_values(df, method='mean', value=None):
    """
    Gestisce i valori nulli nel dataframe.

    Parametri:
    df (DataFrame): Il dataframe da pulire.
    method (str): Il metodo per gestire i valori nulli ('drop', 'mean', 'median', 'mode', 'value').
    value (optional): Il valore con cui sostituire i valori nulli se method è 'value'.

    Ritorna:
    DataFrame: Il dataframe con i valori nulli gestiti.
    """
    if method == 'drop':
        df = df.dropna()
    elif method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    elif method == 'median':
        df = df.fillna(df.median(numeric_only=True))
    elif method == 'mode':
        df = df.fillna(df.mode().iloc[0])
    elif method == 'value':
        df = df.fillna(value)
    return df


def handle_outliers(df, method='cap', factor=1.5):
    """
    Gestisce gli outliers nel dataframe.

    Parametri:
    df (DataFrame): Il dataframe da pulire.
    method (str): Il metodo per gestire gli outliers ('remove', 'cap').
    factor (float): Il fattore per il calcolo dell'IQR per il metodo 'cap'.

    Ritorna:
    DataFrame: Il dataframe con gli outliers gestiti.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'remove':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    elif method == 'cap':
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df


def handle_inconsistent_values(df, method='median', value=None):
    """
    Gestisce i valori inconsistenti nel dataframe.

    Parametri:
    df (DataFrame): Il dataframe da pulire.
    method (str): Il metodo per gestire i valori inconsistenti ('drop', 'mean', 'median', 'mode', 'value').
    value (optional): Il valore con cui sostituire i valori inconsistenti se method è 'value'.

    Ritorna:
    DataFrame: Il dataframe con i valori inconsistenti gestiti.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'drop':
        for col in numeric_cols:
            df = df[np.isfinite(df[col])]
    elif method == 'mean':
        for col in numeric_cols:
            df[col] = np.where(~np.isfinite(df[col]), df[col].mean(), df[col])
    elif method == 'median':
        for col in numeric_cols:
            df[col] = np.where(~np.isfinite(df[col]), df[col].median(), df[col])
    elif method == 'mode':
        for col in numeric_cols:
            df[col] = np.where(~np.isfinite(df[col]), df[col].mode().iloc[0], df[col])
    elif method == 'value':
        for col in numeric_cols:
            df[col] = np.where(~np.isfinite(df[col]), value, df[col])
    return df


def clean_data(df, missing_method='mean', outlier_method='cap', inconsistent_method='median', value=None):
    """
    Pulisce i dati nel file CSV specificato.

    Parametri:
    df: Il dataframe da pulire
    missing_method (str): Il metodo per gestire i valori nulli.
    outlier_method (str): Il metodo per gestire gli outliers.
    inconsistent_method (str): Il metodo per gestire i valori inconsistenti.
    value (optional): Il valore con cui sostituire i valori nulli o inconsistenti se i metodi sono 'value'.

    Ritorna:
    DataFrame: Il dataframe pulito.
    """

    df_clean = df.copy()

     # Gestisce gli outliers
    df_clean = handle_outliers(df_clean, method=outlier_method)

    # Gestisce i valori inconsistenti
    df_clean = handle_inconsistent_values(df_clean, method=inconsistent_method, value=value)

    # Gestisce i valori nulli
    df_clean = handle_missing_values(df_clean, method=missing_method, value=value)

    return df_clean


if __name__ == "__main__":

    target_name = 'Hazardous'

    """Generazione Dataset per Decision Tree"""
    df_tree = pd.read_csv('../input/nasaClean.csv')

    features_tree = df_tree.columns.to_list()
    features_tree.remove(target_name)

    """Generazione Dataset per Depp Neural Network"""
    df_dnn = pd.read_csv('../input/nasaClean.csv')

    features_dnn = df_dnn.columns.to_list()
    features_dnn.remove(target_name)

    columns_to_remove = ['Neo Reference ID', 'Name', 'Close Approach Date', 'Orbit Determination Date', 'Hazardous',
                         'Orbiting Body', 'Equinox']

    for col in columns_to_remove:
        if col in features_dnn:
            features_dnn.remove(col)
            df_dnn = df_dnn.drop(col, axis=1)

    dirty_train(df_tree=df_tree, df_dnn=df_dnn, target_name=target_name, test_size=0.2)
    #drop_train(df_tree=df_tree, df_dnn=df_dnn, target_name=target_name, test_size=0.15)
