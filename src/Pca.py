import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_pca(df_analysis, features, target, variance_threshold=0.95):
    """
    Presuppongo che il dataset passato in ingresso abbia solo variabili numeriche.

    Args:
        df_analysis (pd.DataFrame): Il dataset di input con sole variabili numeriche.
        features (list): Lista delle colonne da utilizzare per la PCA.
        target (str): Nome della colonna della variabile target.
        variance_threshold (float): La soglia di varianza spiegata per selezionare le componenti principali.

    Returns:
        pd.DataFrame: Un dataset contenente le componenti principali selezionate che spiegano fino al 95% della varianza e la variabile target.

    Note:
        La funzione scala i dati prima di applicare la PCA.
    """

    # Separa le features dalla variabile target
    X = df_analysis[features]
    y = df_analysis[target]
    
    # Scala i dati
    scaled_data = StandardScaler().fit_transform(X)

    # Esegue la PCA
    pca = PCA().fit(scaled_data)
    
    # Calcola il numero di componenti necessarie per raggiungere la soglia di varianza
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Applica la PCA con il numero ottimale di componenti
    pca_optimal = PCA(n_components=num_components).fit(scaled_data)
    principal_components = pca_optimal.transform(scaled_data)
    
    # Crea un DataFrame con le componenti principali
    df_principal_components = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(num_components)])
    
    # Aggiungi la variabile target al DataFrame delle componenti principali
    df_principal_components[target] = y.reset_index(drop=True)
    
    return df_principal_components

