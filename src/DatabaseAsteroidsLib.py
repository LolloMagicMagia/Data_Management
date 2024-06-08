import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class AsteroidsLib:

    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.target_name = 'Hazardous'
        self.features = self.df.columns.tolist()

        if self.target_name in self.features:
            self.features.remove(self.target_name)

        # Boolean values conversion to 0 and 1
        self.df_analysis = self.df
        self.df_analysis[self.target_name] = self.df_analysis[self.target_name].astype(int)

    def remove_features(self, features):
        for feature in features:
            if feature in self.features:
                self.df_analysis = self.df_analysis.drop(feature, axis=1)
                self.features.remove(feature)
        return self.df_analysis

    def remove_duplicates(self):
        return self.df_analysis.drop_duplicates(keep=False)

    def remove_correlated_features(self, correlated_features):
        return self.remove_features(correlated_features[1:])

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

        return n_bins

    def bar_plot(self, feature):

        # Definisci i bin per la suddivisione di relative_velocity
        bins = np.linspace(self.df_analysis[feature].min(), self.df_analysis[feature].max(), self.freedman_diaconis_bins(feature))

        # Crea una colonna 'Bin' nel DataFrame basata sui bin
        self.df_analysis['bin'] = pd.cut(self.df_analysis[feature], bins=bins, labels=False)

        # Raggruppa i dati in base a 'Bin' e 'hazardous', ottenendo la frequenza di True e False in ciascun bin
        grouped = self.df_analysis.groupby(['bin', self.target_name]).size().unstack(fill_value=0)

        # Creiamo il grafico a barre stackato
        ax = grouped.plot(kind='bar', stacked=True)

        # Imposta i ticks sull'asse x
        num_ticks = min(20, len(bins))  # Imposta il numero desiderato di ticks
        plt.locator_params(axis='x', nbins=num_ticks)

        ax.set_title('Distribuzione di ' + self.target_name + ' in base a ' + feature)
        ax.set_xlabel('Bin di ' + feature)
        ax.set_ylabel('Frequenza')

        # Aggiungiamo una legenda
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=self.target_name)

        self.df_analysis.drop(["bin"], axis=1)

        plt.show()

    def bar_distribution(self, feature):

        # Esempio di utilizzo con dati contenenti outliers
        num_bins_fd = self.freedman_diaconis_bins(feature)

        #  Visualizziamo la distribuzione della distanza di miss dei NEO usando la funzione histplot()
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df_analysis[feature], bins=num_bins_fd, kde=True)
        plt.xlabel('Bin di ' + feature)
        plt.ylabel('Frequenza')
        plt.title('Distribuzione di ' + self.target_name + ' in base a ' + feature)
        plt.show()

    def box_plot(self, feature):
        sns.catplot(x=self.target_name, y= feature, data=self.df_analysis, kind="box", aspect=1.5)
        plt.title("Boxplot per " + feature + " e " + self.target_name)
        plt.show()

    def scatter_plot(self, feature1, feature2):
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.scatterplot(x=feature1, y=feature2, hue=self.target_name, data=self.df_analysis, palette="Dark2", ax=ax)
        plt.title("Relazione tra " + feature1 + ", " + feature2 + " e " + self.target_name)
        plt.show()

        # Finding the count of outliers based on those instances which are out of iqr

    def print_outliers(self, feature):
        Q1 = self.df_analysis[feature].quantile(0.25)
        Q3 = self.df_analysis[feature].quantile(0.75)
        # Finding IQR
        IQR = Q3 - Q1
        da = (self.df_analysis[feature] < (Q1 - 1.5 * IQR)) | (self.df_analysis[feature] > (Q3 + 1.5 * IQR))
        print("Outliers")
        print(da.value_counts())
