from enum import bin

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score, make_scorer, auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from IPython.display import display, HTML


class AsteroidsLib:
    CONTINUE_FEATURE = 0
    CATEGORICAL_FEATURE = 1
    COUNT_FEATURE = 2

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.target_name = 'Hazardous'
        self.features = self.df.columns.tolist()
        self.all_features = self.features[:]

        self.corr_matrix = None
        self.scaled_data = None
        self.pca = None
        self.pca_components = 0
        self.df_optimal = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

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
        for feature in self.features:
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

        # Crea il bar chart
        plt.figure(figsize=(12, 15))
        for i, feature in enumerate(categorical_features):
            bins = self.freedman_diaconis_bins(feature, max=100)
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
    def freedman_diaconis_bins(self, feature, max=100):
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

        if n_bins >= max:
            return max
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

    def correlation_matrix(self):
        numeric_columns = self.df_analysis.select_dtypes(include=['number'])
        self.corr_matrix = numeric_columns.corr()
        plt.figure(figsize=(32, 15))
        sns.heatmap(self.corr_matrix, annot=True, cmap="coolwarm")
        plt.show()

    def correlated_features(self, threshold=0.999):
        correlated_groups = {}

        for col in self.corr_matrix.columns:
            group_name = None
            for group, features in correlated_groups.items():
                if any(abs(self.corr_matrix.loc[col, feature]) >= threshold for feature in features):
                    group_name = group
                    break
            if group_name is None:
                group_name = f"Group_{len(correlated_groups) + 1}"
                correlated_groups[group_name] = []
            correlated_groups[group_name].append(col)

        correlated_groups_list = [sublist for sublist in correlated_groups.values() if len(sublist) > 1]
        print('Gruppi di feature correlate:')
        for groups in correlated_groups_list:
            print('\t', groups)

        for correlated_features in correlated_groups_list:
            self.remove_correlated_features(correlated_features)

        return self.df_analysis

    def pair_plot(self):
        sns.pairplot(self.df_analysis)

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

    def stacked_column_chart(self, feature):
        if self.df_analysis[feature].max() - self.df_analysis[feature].min() > 100:
            bins = self.freedman_diaconis_bins(feature, max=30)
            # Raggruppiamo i dati in bins
            df_grouped = self.df_analysis.groupby(
                [pd.cut(self.df_analysis[feature], bins=bins), self.target_name]).size().reset_index(name='Frequenza')
        else:
            df_grouped = self.df_analysis.groupby([feature, self.target_name]).size().reset_index(name='Frequenza')

            # Rinominiamo la colonna con i bins
        df_grouped.columns.values[0] = feature

        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_grouped, x=feature, y='Frequenza', hue=self.target_name, palette="Set1", dodge=False)
        plt.title(f'Grouped Bar Chart per {feature} e {self.target_name}')
        plt.xlabel(feature)
        plt.ylabel('Frequenza')
        plt.legend(title=self.target_name, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def stacked_column_chart_all(self, categorical_feature=True):
        if categorical_feature:
            features = self.get_features_group(categorical_feature=True, count_feature=True)
        else:
            features = self.all_features[:]

        for feature in features:
            self.stacked_column_chart(feature)

    def print_outliers(self, feature):
        Q1 = self.df_analysis[feature].quantile(0.25)
        Q3 = self.df_analysis[feature].quantile(0.75)
        # Finding IQR
        IQR = Q3 - Q1
        da = (self.df_analysis[feature] < (Q1 - 1.5 * IQR)) | (self.df_analysis[feature] > (Q3 + 1.5 * IQR))
        print("Outliers")
        print(da.value_counts())

    """ PCA """

    def scale_data(self):
        self.scaled_data = StandardScaler().fit_transform(self.df_analysis[self.features])
        return self.scaled_data

    def pca_init(self):
        self.scale_data()
        self.pca = PCA().fit(self.scaled_data)
        return self.pca

    def explained_variance_plot(self, limit=0.95):
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o', linestyle='--')
        plt.xlabel('Numero di Componenti Principali')
        plt.ylabel('Varianza Spiegata Cumulativa')
        plt.title('Curva della Varianza Spiegata')
        plt.axhline(y=limit, color='r', linestyle='--')
        plt.grid()
        plt.show()

    def components_to_keep(self, limit=0.95):
        explained_variance = self.pca.explained_variance_ratio_
        cumulative_variance = explained_variance.cumsum()

        self.pca_components = next(i for i, cumulative in enumerate(cumulative_variance) if cumulative >= limit) + 1
        print(f'Numero di componenti principali da conservare: {self.pca_components}')
        return self.pca_components

    def pca_swarm_plot(self, df=None):
        if df is None:
            df = self.df_analysis[:]
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='PC1', y='PC2', hue=self.target_name, palette='Set1')
        plt.title('Primo e Secondo Componente Principale')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend(title=self.target_name)
        plt.grid()
        plt.show()

    def kaiser_components_rule(self):
        """
        Filtra le componenti principali con autovalori > 1
        :return:
        """
        components_to_keep = np.sum(self.pca.explained_variance_ > 1)
        print(f"Componenti principali da mantenere secondo la regola di Kaiser: {components_to_keep}")
        return components_to_keep

    def pca_pipeline(self):
        # Creazione Pipeline
        pipeline = Pipeline([
            ('pca', self.pca),
            ('logistic', LogisticRegression())
        ])

        # Lista per memorizzare i punteggi di validazione incrociata
        scores = []

        # Testare un range di componenti principali
        for n_components in range(1, self.scaled_data.shape[1] + 1):
            pipeline.set_params(pca__n_components=n_components)
            score = cross_val_score(pipeline, self.scaled_data, self.df_analysis[self.target_name], cv=5).mean()
            scores.append(score)

        # Grafico dei punteggi di validazione incrociata
        plt.plot(range(1, self.scaled_data.shape[1] + 1), scores, marker='o')
        plt.xlabel('Numero di componenti principali')
        plt.ylabel('Accuratezza di cross-validation')
        plt.title('Accuratezza di cross-validation vs Numero di componenti principali')
        plt.axhline(y=0.95, color='r', linestyle='--')
        plt.show()

        # Numero di componenti principali con il punteggio migliore
        optimal_components = np.argmax(scores) + 1
        print(f"Numero ottimale di componenti principali: {optimal_components}")
        return optimal_components

    def principal_components(self):
        pcs = self.pca.components_
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])

        for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
            # plot line between origin and point (x, y)
            ax.plot([0, x], [0, y], color='k')
            # display the label of the point
            ax.text(x, y, self.df_analysis.columns[i], fontsize='5')

    def optimal_features(self, features):
        pca = PCA(n_components=features)
        pca.fit(self.df_analysis)

        # Ottieni gli indici delle feature più significative nei primi 'features' componenti principali
        top_feature_indices = np.argsort(np.abs(pca.components_))[:, ::-1][:, :features].flatten()

        # Seleziona i nomi delle colonne associate alle feature più significative
        optimal_features = self.df_analysis.columns[top_feature_indices]

        self.df_optimal = self.df_analysis[optimal_features]

        return self.df_optimal

    def pca_component_loadings(self):
        # Ottieni i componenti principali e i carichi delle feature
        components = self.pca.components_
        loadings = pd.DataFrame(components, columns=self.features)

        html_table = "<table><tr><th>Componente Principale</th>"
        for feature in self.features:
            html_table += f"<th>{feature}</th>"
        html_table += "</tr>"

        for i, component in enumerate(loadings.values, start=1):
            html_table += f"<tr><td>Componente {i}</td>"
            for loading in component:
                html_table += f"<td>{loading:.3f}</td>"
            html_table += "</tr>"

        html_table += "</table>"

        # Stampa la tabella HTML
        display(HTML(html_table))

    """ TRAINING DATA """

    def get_subsets(self, test_size=0.1):
        """
        Method to get the Training and Test Set from DataFrame
        :param test_size: Percentage of DataFrame in Test Sets
        :return: Calculate x_train, x_test, y_train, y_test for the class
        """
        self.y = self.df_analysis[self.target_name]

        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(df_model[self.features], df_model[self.target_name], test_size=test_size, stratify=y,
                             random_state=42)

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
