from enum import bin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, auc



class AsteroidsLib:
    CONTINUE_FEATURE = 0
    CATEGORICAL_FEATURE = 1
    COUNT_FEATURE = 2

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.target_name = 'Hazardous'
        self.features = self.df.columns.tolist()
        self.all_features = self.features[:]

        if self.target_name in self.features:
            self.features.remove(self.target_name)

        # Boolean values conversion to 0 and 1
        self.df_analysis = self.df
        self.df_analysis[self.target_name] = self.df_analysis[self.target_name].astype(int)

    """ PREPARAZIONE DATASET """

    def remove_features(self, features):
        for feature in features:
            if feature in self.features:
                self.df_analysis = self.df_analysis.drop(feature, axis=1)
                self.features.remove(feature)
                self.all_features.remove(feature)
        return self.df_analysis

    def remove_duplicates(self):
        return self.df_analysis.drop_duplicates(keep=False)

    def remove_correlated_features(self, correlated_features):
        return self.remove_features(correlated_features[1:])

    """ANALISI ESPLORATIVA """

    def get_feature_type(self, feature):
        if feature in self.features:
            if self.df_analysis[feature].dtype == 'object':
                return AsteroidsLib.CATEGORICAL_FEATURE
            elif self.df_analysis[feature].dtype == 'int64' and self.df_analysis[feature].min() >= 0:
                return AsteroidsLib.COUNT_FEATURE
            else:
                return AsteroidsLib.CONTINUE_FEATURE

    def get_features_group(self, continue_feature=False, categorical_feature=False, count_feature=False, target=False):
        """
        Return a list of feature that is classified as one of the following params if set to True.
        :param continue_feature: True for Continue & Discrete Features
        :param categorical_feature: True for Categorical Features
        :param count_feature: True for Count Features
        :return: A list of features
        """
        features_group = []
        for feature in self.all_features:
            if self.get_feature_type(feature) is AsteroidsLib.CONTINUE_FEATURE and continue_feature:
                features_group.append(feature)
            if self.get_feature_type(feature) is AsteroidsLib.CATEGORICAL_FEATURE and categorical_feature:
                features_group.append(feature)
            if self.get_feature_type(feature) is AsteroidsLib.COUNT_FEATURE and count_feature:
                features_group.append(feature)
        if target and self.target_name not in features_group:
            features_group.append(self.target_name)
        return features_group

    # Funzione per normalizzare i dati e creare un boxplot per ogni feature con colori diversi
    def box_plot_all(self, all_features=False, target=True):
        # Normalizza i dati utilizzando la normalizzazione min-max
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(self.df_analysis)

        # Converte i dati normalizzati in un DataFrame
        if all_features:
            normalized_df = pd.DataFrame(normalized_data, columns=self.all_features)
        else:
            continue_features = self.get_features_group(continue_feature=True, target=target)
            feature_indices = [self.df_analysis.columns.get_loc(col) for col in continue_features]
            normalized_df = pd.DataFrame(normalized_data[:, feature_indices], columns=continue_features)

        # Crea il boxplot con colori diversi per ogni feature
        plt.figure(figsize=(18, 8))

        # Definisci una palette di colori che va dal blu al verde per i boxplot
        cmap = plt.cm.GnBu

        box_data = plt.boxplot(normalized_df.values, patch_artist=True)

        # Assegna colori che vanno dal blu al verde ai boxplot
        for i, box in enumerate(box_data['boxes']):
            box_color = cmap((i + 1) / len(normalized_df.columns))
            box.set_facecolor(box_color)

        # Imposta il colore dei mediani
        for median in plt.findobj(match=lambda x: isinstance(x, plt.Line2D) and 'median' in x.get_label()):
            median.set_color('gray')

        # Crea un elenco separato per i boxplot per utilizzarlo per la legenda
        box_list = [plt.Rectangle((0, 0), 1, 1, fc=cmap((i + 1) / len(normalized_df.columns))) for i in
                    range(len(normalized_df.columns))]

        # Aggiungi titolo e label agli assi
        plt.title('Boxplot per ogni Feature Normalizzata')
        plt.xlabel('Feature')
        plt.ylabel('Valore Normalizzato in [0, 1]')
        plt.grid(axis='y')

        # Aggiungi una legenda per i colori
        plt.legend(box_list, normalized_df.columns, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)

        plt.show()

    def bar_chart_all(self, all_features=False, target=True):
        # Seleziona le feature categoriche
        if all_features:
            categorical_features = self.all_features
        else:
            categorical_features = self.get_features_group(categorical_feature=True, count_feature=True, target=target)

        # Crea un DataFrame con le frequenze delle feature categoriche
        freq_df = pd.DataFrame({feature: self.df_analysis[feature].value_counts() for feature in categorical_features})

        # Crea il bar chart
        plt.figure(figsize=(12, 15))
        for i, feature in enumerate(categorical_features):
            bins = self.freedman_diaconis_bins(feature)
            plt.subplot(len(categorical_features), 1, i + 1)
            plt.hist(self.df_analysis[feature], bins=bins)
            plt.title(feature)
            plt.xlabel('Categoria')
            plt.ylabel('Frequenza')

        # Aggiungi titolo e label agli assi
        plt.suptitle('Bar Chart per ogni Feature Categorica\n')
        plt.tight_layout()

        # Mostra il grafico
        plt.show()

    # Calcola il numero ottimale di classi secondo il metodo di Freedman-Diaconis
    def freedman_diaconis_bins(self, feature):
        data = self.df[feature]
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25

        # Controllo se l'IQR è zero
        if iqr == 0:
            # Metodo alternativo: metodo di Sturges
            n_bins = int(np.ceil(np.log2(len(data)) + 1))
        else:
            h = 2 * iqr / (len(data) ** (1 / 3))
            n_bins = int((data.max() - data.min()) / h)

        if n_bins >= 100:
            return 100
        return n_bins

    def pie_chart(self):
        # Calcola i conteggi delle due classi
        count_hazardous_0 = self.df_analysis[self.target_name].value_counts()[0]
        count_hazardous_1 = self.df_analysis[self.target_name].value_counts()[1]

        plt.pie(self.df_analysis[self.target_name].value_counts(), labels=["0", "1"], autopct='%1.1f%%')
        plt.legend()
        plt.title("Distribuzione della variabile " + self.target_name)
        plt.show()

        print('Valori False (= 0): ', count_hazardous_0)
        print('Valori True (= 1): ', count_hazardous_1)

    def scatter_plot(self, feature1, feature2):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(x=feature1, y=feature2, hue=self.target_name, data=self.df_analysis, palette="Dark2", ax=ax)
        plt.title("Relazione tra " + feature1 + ", " + feature2 + " e " + self.target_name)
        plt.show()

    def scatter_plot_all(self, continue_feature=True):
        if continue_feature:
            features = self.get_features_group(continue_feature=True)
        else:
            features = self.all_features[:]

        for pair in itertools.combinations(features, 2):
            self.scatter_plot(feature1=pair[0], feature2=pair[1])

    def stacked_column_chart(self, feature1, feature2):
        # Crea una tabella pivot per ottenere i dati necessari per il grafico a colonne impilate
        pivot_table = self.df_analysis.pivot_table(index=feature1, columns=feature2, aggfunc='size', fill_value=0)

        # Crea il grafico a colonne impilate
        pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6))

        plt.title(f'Stacked Column Chart per {feature1} e {feature2}')
        plt.xlabel(feature1)
        plt.ylabel('Count')
        plt.legend(title=feature2)
        plt.show()

    def stacked_column_chart_all(self, categorical_feature=True):
        if categorical_feature:
            features = self.get_features_group(categorical_feature=True, count_feature=True)
        else:
            features = self.all_features[:]

        for fea in itertools.combinations(features, 2):
            self.stacked_column_chart(feature1=pair[0], feature2=pair[1])

    def bar_plot(self, a, b, df1, n_class):
        # Supponendo che df["relative_velocity"] contenga valori float e df["hazardous"] contenga valori booleani

        # Definisci i bin per la suddivisione di relative_velocity
        bins = np.linspace(df1[a].min(), df1[a].max(), n_class)

        # Crea una colonna 'Bin' nel DataFrame basata sui bin
        df1['bin'] = pd.cut(df1[a], bins=bins, labels=False)

        # Raggruppa i dati in base a 'Bin' e 'hazardous', ottenendo la frequenza di True e False in ciascun bin
        grouped = df1.groupby(['bin', b]).size().unstack(fill_value=0)

        # Creiamo il grafico a barre stackato
        ax = grouped.plot(kind='bar', stacked=True)

        # Imposta i ticks sull'asse x
        num_ticks = min(20, len(bins))  # Imposta il numero desiderato di ticks
        plt.locator_params(axis='x', nbins=num_ticks)

        ax.set_title('Distribuzione di ' + b + ' in base a ' + a)
        ax.set_xlabel('Bin di ' + a)
        ax.set_ylabel('Frequenza')

        # Aggiungiamo una legenda
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=b)

        df1.drop(["bin"], axis=1)

        plt.show()

    def bar_plot_test(self):
        n_class_diameter = 100

        for a in self.features:
            print(a + " e hazardous\n")
            self.bar_plot(a, "hazardous", self.df_analysis, self.freedman_diaconis_bins(self.df_analysis[a]))
            print("\n\n")

        self.df_analysis = self.df_analysis.drop('bin', axis=1)

    def print_outliers(self, feature):
        Q1 = self.df_analysis[feature].quantile(0.25)
        Q3 = self.df_analysis[feature].quantile(0.75)
        # Finding IQR
        IQR = Q3 - Q1
        da = (self.df_analysis[feature] < (Q1 - 1.5 * IQR)) | (self.df_analysis[feature] > (Q3 + 1.5 * IQR))
        print("Outliers")
        print(da.value_counts())

    def confmatrix_plot(self, model, x_data, y_data):
    
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
    
    def metrics_model(self, y_test, y_pred):
        # Valuta il modello utilizzando i dati di test
        accuracy_train_test = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return accuracy_train_test, precision, recall, f1

