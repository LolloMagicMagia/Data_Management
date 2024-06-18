import pandas as pd
from .l
earn2clean.clean import (CleanData)

def clean_dataset_with_learn2clean(df: pd.DataFrame) -> pd.DataFrame:
    # Creazione dell'oggetto CleanData
    cleaner = CleanData(df)

    # Gestione dei valori nulli
    cleaner.handle_missing(method='mean')  # Puoi cambiare il metodo a 'median', 'most_frequent', ecc.

    # Rimozione dei duplicati
    cleaner.remove_duplicates()

    # Correzione dei tipi di dati (se necessario)
    # Qui dovrai specificare le conversioni necessarie
    # cleaner.df['column1'] = cleaner.df['column1'].astype(int)  # esempio

    # Gestione degli outlier
    cleaner.handle_outliers(method='iqr')  # Puoi cambiare il metodo a 'zscore', ecc.

    # Normalizzazione delle stringhe (se necessario)
    # cleaner.df['column2'] = cleaner.df['column2'].str.strip().str.lower()  # esempio

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
