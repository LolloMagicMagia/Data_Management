import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import DirtyFrame
import DropFrame

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_path)

import NoisyDataGeneration as ndg


def dirty_train(load=True):
    target_name = 'Hazardous'
    datasets = {}
    trained_set = []
    testset = None
    columns = ['Dataset Name', 'Tree Accuracy', 'Tree Precision', 'Tree Recall', 'Tree F1 Score', 'Tree Auc',
               'DNN Accuracy', 'DNN Precision', 'DNN Recall', 'DNN F1 Score']
    dataframe = None
    dir_name = None

    if load:
        datasets, testset = load_csvs_from_disk()
    else:
        datasets, testset = ndg.create_dirty_datasets()

    x_test = testset.copy()
    x_test = x_test.drop(target_name, axis=1)
    scaler = StandardScaler()
    x_test = scaler.fit_transform(x_test)
    y_test = testset[target_name]

    for key, value in datasets.items():
        error_type = key[(key.find('-') + 1):key.find('/')]
        percentage = key[(key.rfind('_') + 1):key.rfind('.')]
        trained = DataSetFrame.DataFrame(value, error_type, percentage, x_test, y_test)
        trained_set.append(trained)

        dir_name = key[0:key.find('/')]
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        txt_file_path = os.path.join(dir_name, f"{error_type}_{percentage}.txt")
        with open(txt_file_path, 'w') as file:
            file.write(f"{error_type} {percentage}%\n")
            file.write('Tree Metrics:\n')
            file.write(f'\tAccuracy: {trained.metrics_tree.accuracy}\n')
            file.write(f'\tPrecision: {trained.metrics_tree.precision}\n')
            file.write(f'\tRecall: {trained.metrics_tree.recall}\n')
            file.write(f'\tF1 Score: {trained.metrics_tree.f1_score}\n')
            file.write(f'\tAUC: {trained.metrics_tree.auc}\n')
            file.write('Deep Neural Network Metrics:\n')
            file.write(f'\tAccuracy: {trained.metrics_dnn.accuracy}\n')
            file.write(f'\tPrecision: {trained.metrics_dnn.precision}\n')
            file.write(f'\tRecall: {trained.metrics_dnn.recall}\n')
            file.write(f'\tF1 Score: {trained.metrics_dnn.f1_score}')

        if dataframe is None:
            data = {
                'Dataset Name': [f"{error_type}{percentage}"],
                'Tree Accuracy': [trained.metrics_tree.accuracy],
                'Tree Precision': [trained.metrics_tree.precision],
                'Tree Recall': [trained.metrics_tree.recall],
                'Tree F1 Score': [trained.metrics_tree.f1_score],
                'Tree Auc': [trained.metrics_tree.auc],
                'DNN Accuracy': [trained.metrics_dnn.accuracy],
                'DNN Precision': [trained.metrics_dnn.precision],
                'DNN Recall': [trained.metrics_dnn.recall],
                'DNN F1 Score': [trained.metrics_dnn.f1_score]
            }
            dataframe = pd.DataFrame(data=data)
        else:
            data = {
                'Dataset Name': f"{error_type}{percentage}",
                'Tree Accuracy': trained.metrics_tree.accuracy,
                'Tree Precision': trained.metrics_tree.precision,
                'Tree Recall': trained.metrics_tree.recall,
                'Tree F1 Score': trained.metrics_tree.f1_score,
                'Tree Auc': trained.metrics_tree.auc,
                'DNN Accuracy': trained.metrics_dnn.accuracy,
                'DNN Precision': trained.metrics_dnn.precision,
                'DNN Recall': trained.metrics_dnn.recall,
                'DNN F1 Score': trained.metrics_dnn.f1_score
            }
            dataframe.loc[len(dataframe)] = data

    dataframe.to_csv('../TrainingSetsDir/dirtysets_trained_metrics.csv', index=False)

def drop_train():
    test = pd.read_csv('../dirty_datasets/nasa_test.csv')
    train = pd.read_csv('../dirty_datasets/nasa_train.csv')
    target_feature = 'Hazardous'
    features = train.columns.tolist()
    features.remove(target_feature)

    dataframe = None

    columns = ['Dropped Features', 'Tree Accuracy', 'Tree Precision', 'Tree Recall', 'Tree F1 Score', 'Tree Auc',
               'DNN Accuracy', 'DNN Precision', 'DNN Recall', 'DNN F1 Score']

    features = ['Absolute Magnitude', 'Orbit Uncertainity', 'Est Dia in KM(min)']
    drop_column = get_columns(features)

    for combo in drop_column:
        df_test = test.drop(combo, axis=1)
        df_train = train.drop(combo, axis=1)

        X_train = df_train
        X_train = X_train.drop(target_feature, axis=1)
        y_train = df_train[target_feature]

        X_test = df_test
        X_test = X_test.drop(target_feature, axis=1)
        y_test = df_test[target_feature]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        trained = DropFrame.DataFrame(X_train, y_train, X_test, y_test)

        if dataframe is None:
            data = {
                'Dropped Features': [combo],
                'Tree Accuracy': [trained.metrics_tree.accuracy],
                'Tree Precision': [trained.metrics_tree.precision],
                'Tree Recall': [trained.metrics_tree.recall],
                'Tree F1 Score': [trained.metrics_tree.f1_score],
                'Tree Auc': [trained.metrics_tree.auc],
                'DNN Accuracy': [trained.metrics_dnn.accuracy],
                'DNN Precision': [trained.metrics_dnn.precision],
                'DNN Recall': [trained.metrics_dnn.recall],
                'DNN F1 Score': [trained.metrics_dnn.f1_score]
            }
            dataframe = pd.DataFrame(data=data)
        else:
            data = {
                'Dropped Features': combo,
                'Tree Accuracy': trained.metrics_tree.accuracy,
                'Tree Precision': trained.metrics_tree.precision,
                'Tree Recall': trained.metrics_tree.recall,
                'Tree F1 Score': trained.metrics_tree.f1_score,
                'Tree Auc': trained.metrics_tree.auc,
                'DNN Accuracy': trained.metrics_dnn.accuracy,
                'DNN Precision': trained.metrics_dnn.precision,
                'DNN Recall': trained.metrics_dnn.recall,
                'DNN F1 Score': trained.metrics_dnn.f1_score
            }
            dataframe.loc[len(dataframe)] = data

    dataframe.to_csv('../TrainingSetsDir/dropsets_trained_metrics.csv', index=False)

def get_columns(features) :
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

#dirty_train()
drop_train()