import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Carica il dataset (funzione separata per modularità)
def load_dataset(file_path='../input/nasaClean.csv'):
    df = pd.read_csv(file_path)
    feature_names = df.columns.tolist()
    target_name = 'Hazardous'

    feature_names.remove(target_name)
    y = df[target_name]
    x_train, x_test, y_train, y_test = train_test_split(df[feature_names], df[target_name], test_size=0.1,
                                                        stratify=y, random_state=42)

    train_data = pd.concat([x_train, y_train], axis=1)  # Concatena le features di addestramento con il target
    test_data = pd.concat([x_test, y_test], axis=1)

    train_data.to_csv('../dirty_datasets/nasa_train.csv', index=False)
    test_data.to_csv('../dirty_datasets/nasa_test.csv', index=False)

    return train_data


# Funzione per creare valori nulli
def create_null(df, percentage=0.5, columns=[]):
    """
    Riempie una colonna numerica casuale o specificata con una percentuale specificata di valori nulli (NaN).

    Args:
        df (pd.DataFrame): Il DataFrame in cui inserire i valori nulli.
        percentage (float): La percentuale di valori da impostare a NaN nella colonna scelta.
        columns (list): Lista di colonne su cui operare.

    Returns:
        pd.DataFrame: Una copia del DataFrame originale con valori nulli introdotti.
    """
    df_null = df.copy()
    for col in columns:
        if col != 'Hazardous':
            n_total_values = len(df_null[col])
            n_nulls = int(n_total_values * percentage)
            null_indices = np.random.choice(df_null.index, n_nulls, replace=False)
            df_null.loc[null_indices, col] = np.nan
    return df_null


def create_outliers(df, percentage=0.5):
    """
    Introduce outlier plausibili in un DataFrame Pandas.

    Gli outlier sono valori estremi ma comunque possibili nel contesto dei dati, generati al di fuori del range interquartile (IQR) ma entro limiti ragionevoli.

    Args:
        df (pd.DataFrame): Il DataFrame in cui inserire gli outlier.
        percentage (float): La percentuale di valori da trasformare in outlier per ogni colonna numerica.

    Returns:
        pd.DataFrame: Una copia del DataFrame originale con outlier introdotti.

    Note:
        - La funzione opera solo sulle colonne numeriche del DataFrame.
        - Gli outlier vengono generati utilizzando una distribuzione uniforme all'interno di un intervallo definito in base ai quantili della colonna.
        - I valori outlier vengono convertiti in interi se la colonna è di tipo int64.
    """

    df_outliers = df.copy()
    numeric_cols = df_outliers.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'Hazardous':
            # Calcola quantili
            Q1 = df_outliers[col].quantile(0.25)
            Q3 = df_outliers[col].quantile(0.75)
            IQR = Q3 - Q1

            # Definisci limiti per gli outlier
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Trova gli indici dei valori che non sono già outlier
            non_outlier_indices = df_outliers[
                (df_outliers[col] >= lower_bound) & (df_outliers[col] <= upper_bound)
                ].index

            # Numero di outliers da inserire nella colonna
            n_outliers_col = int(percentage * len(non_outlier_indices))

            # Scegli casualmente gli indici delle righe da modificare tra quelli non outlier
            outlier_indices = np.random.choice(non_outlier_indices, n_outliers_col, replace=False)

            # Genera outliers casuali al di fuori dell'intervallo, ma entro limiti ragionevoli
            for idx in outlier_indices:
                if df_outliers.at[idx, col] < Q1:
                    # Se il valore originale è sotto Q1, genera un outlier più piccolo
                    if df_outliers[col].dtype == np.int64:  # Controlla il tipo di dato della colonna
                        df_outliers.at[idx, col] = int(np.random.uniform(lower_bound, Q1))
                    else:
                        df_outliers.at[idx, col] = np.random.uniform(lower_bound, Q1)
                else:
                    # Se il valore originale è sopra Q3, genera un outlier più grande
                    if df_outliers[col].dtype == np.int64:  # Controlla il tipo di dato della colonna
                        df_outliers.at[idx, col] = int(np.random.uniform(Q3, upper_bound))
                    else:
                        df_outliers.at[idx, col] = np.random.uniform(Q3, upper_bound)

    return df_outliers


# Funzione per inserire valori inconsistenti
def create_inconsistents(df, percentage=0.5):
    """
    Introduce dati inconsistenti in un DataFrame Pandas.

    I dati inconsistenti sono valori estremi e improbabili, generati al di fuori di un intervallo definito dalla media e dalla deviazione standard della colonna.

    Args:
        df (pd.DataFrame): Il DataFrame in cui inserire i dati inconsistenti.
        percentage (float): La percentuale di valori da trasformare in dati inconsistenti per ogni colonna numerica.

    Returns:
        pd.DataFrame: Una copia del DataFrame originale con dati inconsistenti introdotti.

    Note:
        - La funzione opera solo sulle colonne numeriche del DataFrame.
        - I dati inconsistenti vengono generati casualmente scegliendo tra valori molto piccoli o molto grandi rispetto alla media e alla deviazione standard.
        - I valori inconsistenti vengono convertiti in interi se la colonna è di tipo int64.
    """

    df_inconsistents = df.copy()
    numeric_cols = df_inconsistents.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col != 'Hazardous':
            mean_val = df_inconsistents[col].mean()
            std_dev = df_inconsistents[col].std()

            # Definisci limiti per i dati inconsistenti (molto al di fuori della distribuzione normale)
            lower_bound = mean_val - 3 * std_dev
            upper_bound = mean_val + 3 * std_dev

            # Trova gli indici dei valori che non sono già inconsistenti
            non_inconsistent_indices = df_inconsistents[
                (df_inconsistents[col] >= lower_bound) & (df_inconsistents[col] <= upper_bound)
            ].index

            # Numero di dati inconsistenti da inserire nella colonna
            n_inconsistents_col = int(percentage * len(non_inconsistent_indices))

            # Scegli casualmente gli indici delle righe da modificare tra quelli non inconsistenti
            inconsistent_indices = np.random.choice(non_inconsistent_indices, n_inconsistents_col, replace=False)

            # Genera dati inconsistenti casualmente scegliendo tra valori molto piccoli o molto grandi
            for idx in inconsistent_indices:
                # Scelta casuale tra valore molto piccolo o molto grande
                choice = np.random.choice([0, 1])
                if choice == 0:
                    if df_inconsistents[col].dtype == np.int64:
                        df_inconsistents.at[idx, col] = int(np.random.uniform(lower_bound / 5, lower_bound))
                    else:
                        df_inconsistents.at[idx, col] = np.random.uniform(lower_bound / 5, lower_bound)
                else:
                    if df_inconsistents[col].dtype == np.int64:
                        df_inconsistents.at[idx, col] = int(np.random.uniform(upper_bound, upper_bound * 5))
                    else:
                        df_inconsistents.at[idx, col] = np.random.uniform(upper_bound, upper_bound * 5)

    return df_inconsistents


def drop_column(df):
    """
    Elimina una colonna numerica casuale dal DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame da cui eliminare la colonna.

    Returns:
        pd.DataFrame: Una copia del DataFrame originale con la colonna rimossa.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    column_to_drop = np.random.choice(numeric_cols)

    df_dropped = df.drop(columns=[column_to_drop])
    return df_dropped


def introduce_errors_globally(df, percentage, error_type='all', target_name='Hazardous'):
    """
    Crea un dataset sporcato secondo uno tra gli error type disponibili

    :type df: pandas.DataFrame
    :type percentage: float
    :type error_type: str
    :type target_name: string
    :param df: Il DataFrame da sporcare
    :param percentage: Percentuale di valori sporcati
    :param error_type: Tipo di errore che si vuole introdurre
    :param target_name: Nome della colonna target da escludere dalla corruzione
    :return: Un nuovo dataset sporcato

    Raises:
        ValueError: Se error_type non è uno tra 'null', 'outlier', 'inconsistent' o 'all'.
    """
    if error_type not in ['null', 'outlier', 'inconsistent', 'all', 'on', 'oi', 'in', 'no', 'io', 'ni']:
        raise ValueError(
            "error_type deve essere uno tra 'null', 'outlier', 'inconsistent', 'all', 'on', 'oi', 'in', 'no', 'io', 'ni'")

    df_corrupted = df.copy()
    columns = df.columns.tolist()

    if target_name is not None and target_name in columns:
        columns.remove(target_name)

    n_total_values = df[columns].size
    n_errors = int(n_total_values * percentage)

    if error_type == 'all':
        df_corrupted = create_null(df_corrupted, percentage=percentage / 3, columns=columns)
        df_corrupted = create_outliers(df_corrupted, percentage=percentage / 3)
        df_corrupted = create_inconsistents(df_corrupted, percentage=percentage / 3)
    elif error_type == 'on' or 'no':
        df_corrupted = create_null(df_corrupted, percentage=percentage / 2, columns=columns)
        df_corrupted = create_outliers(df_corrupted, percentage=percentage / 2)
    elif error_type == 'in' or 'ni':
        df_corrupted = create_null(df_corrupted, percentage=percentage / 2, columns=columns)
        df_corrupted = create_inconsistents(df_corrupted, percentage=percentage / 2)
    elif error_type == 'oi' or 'io':
        df_corrupted = create_outliers(df_corrupted, percentage=percentage / 2)
        df_corrupted = create_inconsistents(df_corrupted, percentage=percentage / 2)
    else:
        if error_type == 'null':
            df_corrupted = create_null(df_corrupted, percentage, columns=columns)
        elif error_type == 'outlier':
            df_corrupted = create_outliers(df_corrupted, percentage)
        elif error_type == 'inconsistent':
            df_corrupted = create_inconsistents(df_corrupted, percentage)

    return df_corrupted


def generate_dirty_dataset(df=load_dataset()):
    os.makedirs('../dirty_datasets/1-null', exist_ok=True)
    os.makedirs('../dirty_datasets/1-outlier', exist_ok=True)
    os.makedirs('../dirty_datasets/1-inconsistent', exist_ok=True)
    os.makedirs('../dirty_datasets/2-outlier_null', exist_ok=True)
    os.makedirs('../dirty_datasets/2-outlier_inconsistent', exist_ok=True)
    os.makedirs('../dirty_datasets/2-inconsistent_null', exist_ok=True)
    os.makedirs('../dirty_datasets/3-all', exist_ok=True)

    for i in range(10, 100, 10):
        percentage = i / 100

        for error_type in ['null', 'outlier', 'inconsistent', 'all', 'on', 'oi', 'in']:
            df_corrupted = introduce_errors_globally(df, percentage, error_type)

            if error_type == 'null':
                df_corrupted.to_csv(f'../dirty_datasets/1-null/nasa_{error_type}_{i}.csv', index=False)
            elif error_type == 'outlier':
                df_corrupted.to_csv(f'../dirty_datasets/1-outlier/nasa_{error_type}_{i}.csv', index=False)
            elif error_type == 'inconsistent':
                df_corrupted.to_csv(f'../dirty_datasets/1-inconsistent/nasa_{error_type}_{i}.csv', index=False)
            elif error_type == 'all':
                df_corrupted.to_csv(f'../dirty_datasets/3-all/nasa_{error_type}_{i}.csv', index=False)
            elif error_type == 'on':
                df_corrupted.to_csv(f'../dirty_datasets/2-outlier_null/nasa_outlier_null_{i}.csv', index=False)
            elif error_type == 'oi':
                df_corrupted.to_csv(f'../dirty_datasets/2-outlier_inconsistent/nasa_outlier_inconsistent_{i}.csv',
                                    index=False)
            elif error_type == 'in':
                df_corrupted.to_csv(f'../dirty_datasets/2-inconsistent_null/nasa_inconsistent_null_{i}.csv',
                                    index=False)

    print("Tutti i dataset sporchi sono stati creati e salvati con successo!")

def create_dirty_dataset(df=load_dataset()):

    dfs = {}

    for i in range(10, 100, 10):
        percentage = i / 100

        for error_type in ['null', 'outlier', 'inconsistent', 'all', 'on', 'oi', 'in']:
            df_corrupted = introduce_errors_globally(df, percentage, error_type)

            if error_type == 'null':
                dfs[f'1-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'outlier':
                dfs[f'1-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'inconsistent':
                dfs[f'1-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'all':
                dfs[f'3-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'on':
                dfs[f'2-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'oi':
                dfs[f'2-{error_type}/{error_type}_{i}'] = df_corrupted
            elif error_type == 'in':
                dfs[f'2-{error_type}/{error_type}_{i}'] = df_corrupted

    return dfs
