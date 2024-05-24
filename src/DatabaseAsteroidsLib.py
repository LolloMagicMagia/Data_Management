import pandas as pd


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
                self.features = self.df_analysis.columns.tolist()
        return self.df_analysis

    def remove_duplicates(self):
        return self.df_analysis.drop_duplicates(keep=False)

    def remove_correlated_features(self, correlated_features):
        return self.remove_features(correlated_features[1:])
