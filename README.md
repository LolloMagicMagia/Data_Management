# DataManagement for Machine Learning: Asteroids Dataset

- Biancini Mattia 865966
- Gerardi Marco 869138
- Monti Lorenzo 869960

## Struttura del progetto 

* Cartella input: contiene i file di sorgente dati e materiale relativo al dataset del progetto
  
* Cartella scr: contiene tutti i file di lavoro vero e proprio.
  * train.py: lo script che si preoccuperà di addestrare il modello scelto da tune_model.py. Restituirà un modello salvato su disco in formato pickle.
  
  * predict.py: è lo script da chiamare per effettuare una predizione con il modello generato da train.py
  
  * model_selection.py: in questo script inseriremo il codice che andrà a darci un riferimento per il modello più performante per il nostro dataset.
  
  * tune_model.py: servirà a passare il risultato di model_selection.py alla pipeline di ottimizzazione degli iperparametri
  
  * utils.py: conterrà tutte le nostre funzioni helper che ci serviranno durante il progetto

* Cartella models: contiene i modelli che salveremo restituiti da train.py in formato pickle.
  
* Cartella notebooks: è "l'ambiente di esplorazione" dove risiedono tutti notebook .ipynb.

### Struttura del progetto (Tia)
1) PCA e pulizia del Dataset
2) Addestrare i modelli e vedere le performance
3) Sporcare il Dataset con valori nulli, outliers, valori inconsistenti fuori dalla distribuzione (Violazione 4 principi di Data Quality)
    1) Riempire una o più feature con valori 100% mancanti o dropparla direttamente
    2) Inconsistenza usando i quartili come valori nuovi
4) Addestrare gli stessi modelli sporcati e vedere le performance
5) Confrontare modelli puliti e sporchi (sottolineando quanto la Data Quality influisca)
6) Conclusione


## Il dataset

Il dataset riguarda gli asteroidi ed è fornito da NEOWS (Near-Earth Object Web Service).

#### Panoramica del dataset

- **Numero di istanze:** 4687
- **Numero di features:** 40
- **Valori nulli:** Nessuno

#### Descrizione di alcune features

- **Neo Reference ID:** ID di riferimento assegnato a un asteroide.
- **Name:** Nome dato a un asteroide.
- **Absolute Magnitude:** Magnitudine assoluta di un asteroide, ovvero la magnitudine visiva che un osservatore registrerebbe se l'asteroide fosse posto a 1 Unità Astronomica (UA) di distanza dalla Terra e dal Sole e a un angolo di fase zero.
- **Est Dia in KM(min):** Diametro stimato dell'asteroide in chilometri (KM).
- **Est Dia in M(min):** Diametro stimato dell'asteroide in metri (M).
- **Relative Velocity km per sec:** Velocità relativa dell'asteroide in chilometri al secondo.
- **Relative Velocity km per hr:** Velocità relativa dell'asteroide in chilometri all'ora.
- **Orbiting Body:** Pianeta attorno al quale l'asteroide sta ruotando.
- **Jupiter Tisserand Invariant:** Parametro di Tisserand per l'asteroide, un valore calcolato da diversi elementi orbitali (semiasse maggiore, eccentricità orbitale e inclinazione) di un oggetto relativamente piccolo e un 'corpo perturbatore' più sostanziale. Viene utilizzato per distinguere diversi tipi di orbite.
- **Eccentricity:** Valore di eccentricità dell'orbita dell'asteroide.
- **Semi Major Axis:** Valore del semiasse maggiore dell'orbita dell'asteroide.
- **Orbital Period:** Valore del periodo orbitale dell'asteroide, ovvero il tempo impiegato dall'asteroide per compiere una rivoluzione completa attorno al suo corpo orbitante.
- **Perihelion Distance:** Valore della distanza del perielio dell'asteroide. Per un corpo in orbita attorno al Sole, il punto di minima distanza è il perielio.
- **Aphelion Dist:** Valore della distanza dell'afelio dell'asteroide. Per un corpo in orbita attorno al Sole, il punto di massima distanza è l'afelio.
- **Hazardous:** Indica se l'asteroide è pericoloso o meno.

## Fasi del progetto 


**1. Definizione del Problema e Obiettivi:**

*   **Comprensione del problema:** Cosa si vuole ottenere con il machine learning? Si tratta di classificazione, regressione, clustering, o altro?
*   **Definizione degli obiettivi:** Quali sono le metriche di successo? Ad esempio, accuratezza, precisione, recall, errore quadratico medio, ecc.

**2. Raccolta e Preparazione dei Dati:**

*   **Raccolta dati** 
  
*   **Pulizia dei dati:** Gestire valori mancanti, outlier, duplicati.
*   **Analisi Esplorativa dei Dati (EDA):**
    *   Visualizzazione dei dati (istogrammi, scatter plot, box plot) per comprenderne la distribuzione, relazioni e anomalie.
    *   Calcolo di statistiche descrittive (media, mediana, deviazione standard) per riassumere le caratteristiche dei dati.
    *   Identificazione di pattern, correlazioni e tendenze nei dati.
*   **Feature Engineering:**
    *   PCA (Principal Component Analysis)
    *   Creazione di nuove feature significative a partire da quelle esistenti (es. combinazioni, trasformazioni, estrazioni).
    *   Normalizzazione o standardizzazione delle feature per migliorarne la compatibilità con gli algoritmi.

**3. Scelta e Addestramento del Modello:**

*   **Scelta del modello:** In base al tipo di problema, agli obiettivi e ai dati, selezionare l'algoritmo più adatto (es. regressione lineare, alberi decisionali, reti neurali, SVM).
*   **Divisione dei dati:** Separare i dati in un set di addestramento (per costruire il modello) e un set di test (per valutarlo).
*   **Addestramento del modello:** Utilizzare il set di addestramento per far apprendere al modello i pattern e le relazioni nei dati.
*   **Validazione del modello:** Utilizzare tecniche come la cross-validation per valutare le prestazioni del modello durante l'addestramento e prevenire l'overfitting.
*   **Ottimizzazione degli iperparametri:** Regolare i parametri del modello (es. tasso di apprendimento, numero di strati in una rete neurale) per migliorarne le prestazioni.

**4. Valutazione del Modello:**

*   **Valutazione delle prestazioni:** Utilizzare il set di test per valutare le prestazioni del modello su dati non visti durante l'addestramento.
*   **Confronto con baseline:** Confrontare le prestazioni del modello con modelli più semplici o con risultati ottenuti in precedenza.
*   **Analisi degli errori:** Esaminare i tipi di errori commessi dal modello per identificare aree di miglioramento.

**5. Implementazione e Monitoraggio del Modello:**

*   **Implementazione in produzione:** Integrare il modello in un sistema reale per fare previsioni o prendere decisioni.
*   **Monitoraggio delle prestazioni:** Continuare a valutare le prestazioni del modello nel tempo e aggiornarlo se necessario.
*   **Raccolta di feedback:** Raccogliere feedback dagli utenti per migliorare il modello e adattarlo a nuovi dati o scenari.

