import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
file_path = '../input/nasa.csv'
df = pd.read_csv(file_path)

# Filtra solo le colonne numeriche
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Imposta la percentuale di outliers da inserire
outlier_percentage = 0.05
n_outliers = int(outlier_percentage * len(df))

# Scegli casualmente le righe e le colonne numeriche dove inserire outliers
outlier_indices = np.random.choice(df.index, n_outliers, replace=False)
outlier_columns = np.random.choice(numeric_cols, n_outliers, replace=True)

# Moltiplica i valori scelti per un fattore alto per creare outliers
df_outliers = df.copy()
for idx, col in zip(outlier_indices, outlier_columns):
    df_outliers.at[idx, col] *= 10

# Salva il dataset con outliers
outliers_file_path = '../dirty_datasets/nasa_outliers.csv'
df_outliers.to_csv(outliers_file_path, index=False)

# Scegli un valore costante fuori dalla distribuzione dei dati numerici
inconsistent_value = df[numeric_cols].mean().mean() * 10

# Imposta la percentuale di valori inconsistenti da inserire
inconsistent_percentage = 0.05
n_inconsistents = int(inconsistent_percentage * len(df) * len(numeric_cols))

# Scegli casualmente le posizioni nel DataFrame dove inserire valori inconsistenti
inconsistent_row_indices = np.random.randint(0, df.shape[0], n_inconsistents)
inconsistent_col_indices = np.random.choice(numeric_cols, n_inconsistents, replace=True)

df_inconsistents = df.copy()
for row_idx, col in zip(inconsistent_row_indices, inconsistent_col_indices):
    # Cast the inconsistent value to the appropriate type
    inconsistent_val = np.array(inconsistent_value).astype(df_inconsistents[col].dtype)
    df_inconsistents.at[row_idx, col] = inconsistent_val

# Salva il dataset con valori inconsistenti
inconsistents_file_path = '../dirty_datasets/nasa_inconsistents.csv'
df_inconsistents.to_csv(inconsistents_file_path, index=False)

# Scegli una colonna numerica da riempire completamente con valori nulli
column_to_null = np.random.choice(numeric_cols)
df_100_nulls = df.copy()
df_100_nulls[column_to_null] = np.nan

# Oppure droppa direttamente una colonna
df_dropped_column = df.drop(columns=[column_to_null])

# Salva i dataset con colonne nulle o droppate
nulls_file_path = '../dirty_datasets/nasa_100_nulls.csv'
dropped_column_file_path = '../dirty_datasets/nasa_dropped_column.csv'
df_100_nulls.to_csv(nulls_file_path, index=False)
df_dropped_column.to_csv(dropped_column_file_path, index=False)

(outliers_file_path, inconsistents_file_path, nulls_file_path, dropped_column_file_path)
