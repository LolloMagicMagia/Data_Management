import pandas as pd
from learn2clean.clean import (CleanData)


def clean_dataset_with_learn2clean(
    df: pd.DataFrame,
    missing_method: str = 'mean',
    outlier_method: str = 'iqr',
    string_normalization: dict = None,
    dtype_corrections: dict = None
) -> pd.DataFrame:
    """
    Pulisce un DataFrame utilizzando Learn2Clean.

    Args:
        df (pd.DataFrame): Il DataFrame da pulire.
        missing_method (str): Metodo per gestire i valori nulli ('mean', 'median', 'most_frequent', ecc.).
        outlier_method (str): Metodo per gestire gli outlier ('iqr', 'zscore', ecc.).
        string_normalization (dict): Dizionario con le colonne da normalizzare e le operazioni da eseguire.
        dtype_corrections (dict): Dizionario con le colonne e i tipi di dati da correggere.

    Returns:
        pd.DataFrame: Il DataFrame pulito.
    """
    # Creazione dell'oggetto CleanData
    cleaner = CleanData(df)

    # Gestione dei valori nulli
    if missing_method in ['mean', 'median', 'most_frequent']:
        cleaner.handle_missing(method=missing_method)
    else:
        raise ValueError(f"Metodo per la gestione dei valori nulli non valido: {missing_method}")

    # Rimozione dei duplicati
    cleaner.remove_duplicates()

    # Correzione dei tipi di dati
    if dtype_corrections:
        for column, dtype in dtype_corrections.items():
            if column in cleaner.df.columns:
                cleaner.df[column] = cleaner.df[column].astype(dtype)
            else:
                raise KeyError(f"Colonna non trovata nel DataFrame: {column}")

    # Gestione degli outlier
    if outlier_method in ['iqr', 'zscore']:
        cleaner.handle_outliers(method=outlier_method)
    else:
        raise ValueError(f"Metodo per la gestione degli outlier non valido: {outlier_method}")

    # Normalizzazione delle stringhe
    if string_normalization:
        for column, operations in string_normalization.items():
            if column in cleaner.df.columns:
                for operation in operations:
                    if operation == 'strip':
                        cleaner.df[column] = cleaner.df[column].str.strip()
                    elif operation == 'lower':
                        cleaner.df[column] = cleaner.df[column].str.lower()
                    elif operation == 'upper':
                        cleaner.df[column] = cleaner.df[column].str.upper()
                    else:
                        raise ValueError(f"Operazione di normalizzazione non valida: {operation}")
            else:
                raise KeyError(f"Colonna non trovata nel DataFrame: {column}")

    # Restituisce il DataFrame pulito
    return cleaner.df

# Esempio di utilizzo della funzione
if __name__ == "__main__":
    # Caricamento del dataset sporco
    df_dirty = pd.read_csv('path_to_dirty_file.csv')

    # Pulizia del dataset utilizzando Learn2Clean
    df_clean = clean_dataset_with_learn2clean(df_dirty)

    # Stampa del DataFrame pulito e del report sulla qualità dei dati
    print(df_clean)
