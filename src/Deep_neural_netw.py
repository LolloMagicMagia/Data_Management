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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, auc


def deep_neural_netw(x_train, x_test, y_train, y_test):    
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
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Valutazione del modello
    test_loss, test_f1 = model.evaluate(X_test, y_test)
    predictions = (model.predict(X_test) > 0.5).astype("int32")

    # Calcolo delle metriche di valutazione
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    print("Test f1 Algo:", test_f1)
    print("F1-score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion matrix:\n", conf_matrix)
