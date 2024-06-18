import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Carica il dataset (funzione separata per modularità)
def load_dataset(file_path='../input/nasa.csv'):
    return pd.read_csv(file_path)


def create_outliers(df, outlier_percentage=0.5):
    """
    Introduce outlier plausibili in un DataFrame Pandas.

    Gli outlier sono valori estremi ma comunque possibili nel contesto dei dati, generati al di fuori del range interquartile (IQR) ma entro limiti ragionevoli.

    Args:
        df (pd.DataFrame): Il DataFrame in cui inserire gli outlier.
        outlier_percentage (float): La percentuale di valori da trasformare in outlier per ogni colonna numerica.

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
        # Calcola quantili
        Q1 = df_outliers[col].quantile(0.25)
        Q3 = df_outliers[col].quantile(0.75)
        IQR = Q3 - Q1

        # Definisci limiti per gli outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Numero di outliers da inserire nella colonna
        n_outliers_col = int(outlier_percentage * len(df_outliers))

        # Trova gli indici dei valori che non sono già outlier
        non_outlier_indices = df_outliers[
            (df_outliers[col] >= lower_bound) & (df_outliers[col] <= upper_bound)
        ].index

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
def create_inconsistents(df, inconsistent_percentage=0.05):
    """
    Introduce valori inconsistenti in un DataFrame Pandas.

    Le inconsistenze possono essere:
        - Valori numerici fuori scala, generati casualmente da una distribuzione uniforme.
        - Valori di tipo errato (stringhe casuali) inseriti in colonne non numeriche.

    Args:
        df (pd.DataFrame): Il DataFrame in cui inserire le inconsistenze.
        inconsistent_percentage (float): La percentuale di valori da rendere inconsistenti per ogni colonna.

    Returns:
        pd.DataFrame: Una copia del DataFrame originale con inconsistenze introdotte.
    """

    df_inconsistents = df.copy()

    # 1. Valori fuori scala (con distribuzione casuale)
    numeric_cols = df_inconsistents.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_val = df_inconsistents[col].mean()
        std_dev = df_inconsistents[col].std()

        # Definisci lower_bound e upper_bound FUORI dal ciclo
        lower_bound = mean_val - 3 * std_dev
        upper_bound = mean_val + 3 * std_dev

        # Numero di valori inconsistenti da inserire nella colonna
        n_inconsistents_col = int(inconsistent_percentage * len(df_inconsistents))

        # Genera valori inconsistenti casuali fuori dall'intervallo
        inconsistent_values = np.random.choice(
            [
                np.random.uniform(lower_bound / 5, lower_bound),
                np.random.uniform(upper_bound, upper_bound * 5),
            ],
            n_inconsistents_col,
        )

        # Indici casuali delle righe da modificare
        inconsistent_row_indices = np.random.choice(df_inconsistents.index, n_inconsistents_col, replace=False)

        # Conversione a intero se la colonna è di tipo int64
        if df_inconsistents[col].dtype == np.int64:
            inconsistent_values = inconsistent_values.astype(int)

        # Inserisci i valori inconsistenti
        df_inconsistents.loc[inconsistent_row_indices, col] = inconsistent_values


    # 2. Valori di tipo errato (stringhe casuali)
    object_cols = df_inconsistents.select_dtypes(include=[object]).columns
    for col in object_cols:
        n_inconsistents_col = int(inconsistent_percentage * len(df_inconsistents))
        inconsistent_values = np.random.choice(
            ["valore_errato_1", "valore_errato_2", "valore_errato_3"], n_inconsistents_col
        )
        inconsistent_row_indices = np.random.choice(df_inconsistents.index, n_inconsistents_col, replace=False)
        df_inconsistents.loc[inconsistent_row_indices, col] = inconsistent_values

    return df_inconsistents


# Funzione per creare valori nulli
def create_null_column(df, null_percentage=0.5, column_to_null=None):  # Aggiungi parametro per la percentuale di null
    """
        Riempie una colonna numerica casuale con una percentuale specificata di valori nulli (NaN).

        Args:
            df (pd.DataFrame): Il DataFrame in cui inserire i valori nulli.
            null_percentage (float): La percentuale di valori da impostare a NaN nella colonna scelta.

        Returns:
            pd.DataFrame: Una copia del DataFrame originale con valori nulli introdotti.
        """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if not column_to_null:
        column_to_null = np.random.choice(numeric_cols)

    df_nulls = df.copy()

    # Calcola il numero di valori nulli da inserire
    n_nulls = int(null_percentage * len(df_nulls))

    # Scegli casualmente gli indici delle righe da modificare
    null_indices = np.random.choice(df_nulls.index, n_nulls, replace=False)

    # Inserisci i valori nulli
    df_nulls.loc[null_indices, column_to_null] = np.nan

    return df_nulls


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



# --- Esecuzione ---

df = load_dataset()

# Creazione dataset sporchi (chiamate alle funzioni)
df_outliers = create_outliers(df, 0.07)
df_inconsistents = create_inconsistents(df)
df_nulls = create_null_column(df, 1, 'Absolute Magnitude')
df_dropped = drop_column(df)

# Salvataggio (eventualmente in una funzione separata)
df_outliers.to_csv('../dirty_datasets/nasa_outliers.csv', index=False)
df_inconsistents.to_csv('../dirty_datasets/nasa_inconsistents.csv', index=False)
df_nulls.to_csv('../dirty_datasets/nasa_null_column.csv', index=False)
df_dropped.to_csv('../dirty_datasets/nasa_dropped_column.csv', index=False)

print("Dataset sporchi creati e salvati con successo!")
