from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from time import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, auc

def ml_albero(x_train, x_test, y_train, y_test):    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    #ccp_alphas = potatura(X_train, y_train)
    best_tree = grid_search(X_train, y_train)
    best_tree_params = best_tree.best_params_
    # Valuta il modello utilizzando i dati di test
    y_pred = best_tree.predict(X_test)
    accuracy_train_test, precision, recall, f1 = metrics_model(y_test, y_pred)

    # Stampa le prestazioni del modello
    print('Accuracy:', accuracy_train_test)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print(classification_report(y_pred, y_test))
    print("\n")

    print('Matrice di confusione')
    confmatrix_plot(best_tree, X_test, y_test)

    print("\n")

    model_tree = DecisionTreeClassifier(**best_tree_params)
    model_tree.fit(X_train, y_train)    

    # Visualizza l'albero decisionale
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(model_tree, filled=True, ax=ax)
    plt.show()

    print("\n")

    print("ROC curve")
    roc_curve_method(best_tree, X_test, y_test)



def potatura(X_train, y_train):
    # Genera il path di potatura
    model_tree = DecisionTreeClassifier(random_state=0, criterion= 'gini', max_depth = 50, max_features= None, min_samples_split= 2)
    path = model_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas

    ccp_alphas

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, criterion= 'gini', max_depth = 50, max_features= None, min_samples_split= 2, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    return ccp_alphas

def grid_search(X_train, y_train):
    # Definisci la griglia degli iperparametri
    param_distributions = {
        #'ccp_alpha':ccp_alphas,
        'criterion': ['gini', 'entropy'],
        'max_depth': [50, 40, 30 , 100, 200],
        'min_samples_split': [2, 5 ,10, 20, 50],
        'max_features': [None, 'sqrt', 'log2'],
        #'min_samples_leaf': [1, 2, 4],
        #'min_impurity_decrease': [0.0, 0.01, 0.1],
        #'max_leaf_nodes': [None, 50, 100],
        'random_state': [42],
    }

    # Usato in GridSearchCV
    best_tree = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid= param_distributions , cv=10, scoring="f1")

    # Ad esempio, supponiamo che tu abbia i dati x_train e y_train
    best_tree.fit(X_train, y_train)

    return best_tree

def metrics_model(y_test, y_pred):
    # Valuta il modello utilizzando i dati di test
    accuracy_train_test = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy_train_test, precision, recall, f1


def confmatrix_plot(model, x_data, y_data):

    # Accepts as argument model object, x data (test or validate), and y data (test or validate).
    # Return a plot of confusion matrix for predictions on y data.
    model_pred = model.predict(x_data)

    # Ottenere le classi previste uniche
    classes = np.unique(np.concatenate((y_data, model_pred)))

    cm = confusion_matrix(y_data, model_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=classes)

    disp.plot(values_format='')  # values_format='' suppresses scientific notation
    plt.show()

def roc_curve_method(model_tree, X_test, y_test):
    # Calcola le probabilità di predizione del modello dell'albero decisionale utilizzando predict_proba
    y_dt_pred_prob = model_tree.predict_proba(X_test)[:, 1]

    # Calcola la curva ROC e l'AUC dell'albero decisionale
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, y_dt_pred_prob)
    auc_dt = roc_auc_score(y_test, y_dt_pred_prob)

    # Plotta la curva ROC dei due modelli per comparare le performance
    plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()