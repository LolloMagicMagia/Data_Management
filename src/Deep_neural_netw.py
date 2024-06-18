from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import backend as K
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, auc


def deep_neural_netw(x_train, x_test, y_train, y_test, name_modello):    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    # Preparazione dei dati
    # Assume che train_data e train_labels siano i tuoi dati di addestramento
    input_shape = (X_train.shape[1],)
    num_classes = 1  # Per una classificazione binaria

    # Definizione del modello
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    # Definizione della metrica F1-score personalizzata
    def f1(y_true, y_pred):
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        
        precision = precision_m(y_true, y_pred)
        recall = recall_m(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # Compilazione del modello
    model.compile(loss='binary_crossentropy',
            optimizer= "adam",
            metrics=[f1])

    # Addestramento del modello
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose = 0)

    # Valutazione del modello
    test_loss, test_f1 = model.evaluate(X_test, y_test)
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    accuracy_train_test, precision, recall, f1 = metrics_model(y_test, predictions)

    # Stampa le prestazioni del modello
    print('Accuracy:', accuracy_train_test)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print(classification_report(predictions, y_test))
    print("\n")

    #Matrice di confusione
    print('Matrice di confusione')
    confmatrix_plot(model, X_test, y_test)
    print("\n")

    # Sostituisci con il percorso della tua cartella
    model_dir = 'models'  
    os.makedirs(model_dir, exist_ok=True)
    model_filename = os.path.join(model_dir, name_modello + '.pkl')

    joblib.dump(model, model_filename)

    print(f"Modello salvato correttamente come {model_filename}")


def metrics_model(y_test, y_pred):
    # Valuta il modello utilizzando i dati di test
    accuracy_train_test = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy_train_test, precision, recall, f1

def confmatrix_plot(model, x_data, y_data):
    # Prevedi le classi con il modello
    model_pred = model.predict(x_data)
    model_pred_classes = (model_pred > 0.5).astype("int32").flatten()  # Flatten per trasformarlo in una dimensione

    # Ottenere le classi previste uniche
    classes = np.unique(np.concatenate((y_data, model_pred_classes)))

    cm = confusion_matrix(y_data, model_pred_classes, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot(values_format='')  # values_format='' sopprime la notazione scientifica
    plt.show()