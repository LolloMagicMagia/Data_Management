import shap
import matplotlib.pyplot as plt
import numpy as np


class Shap:

    def __init__(self, model, x_train, x_test, features, tree=True, num_background_samples=100):
        self.model = model
        self.X_train = x_train
        self.X_test = x_test
        self.X_test = self.X_test[:250]

        # Initialize explainer based on tree or kernel
        if tree:
            self.explainer = shap.TreeExplainer(self.model, self.X_train)
        else:
            background = shap.sample(self.X_train, num_background_samples)
            self.explainer = shap.KernelExplainer(self.model.predict, background)

        # Compute SHAP values for X_test
        self.shap_values = self.explainer.shap_values(self.X_test)

        # Extract the SHAP values for the true class (assuming this is a classification task)
        self.values_true = self.shap_values[:, :, 0]

        # Create Explanation object
        self.explanation = shap.Explanation(
            values=self.values_true,
            base_values=self.explainer.expected_value,  # Adjust according to your model and explainer
            data=self.X_test,
            feature_names=features
        )

    def waterfall(self, index=0):
        shap.plots.waterfall(self.explanation[index])

    def plot_force(self, index=0):
        shap.initjs()
        display(shap.plots.force(self.explanation[index], matplotlib=False))

    def plot_force_range(self, min=0, max=0):
        shap.initjs()
        display(shap.plots.force(self.explanation[min:max]))

    def plots_bar(self):
        shap.plots.bar(self.explanation)

    def beeswarm(self):
        shap.plots.beeswarm(self.explanation)

    def print_beeswarm(self, name='beeswarm'):
        shap.plots.beeswarm(self.explanation, show=False)

        fig = plt.gcf()

        # Salva la figura come immagine
        fig.savefig(f'{name}.png', bbox_inches='tight', dpi=300)

        # Chiudi la figura per liberare memoria
        plt.close(fig)