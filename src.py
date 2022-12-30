from pyEDM import *
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd


class SimplexPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.X_ = None
        self.res = None
        self.embed_dim = -1
        self.lib_start = -1
        self.lib_end = -1
        self.embed_start = -1
        self.embed_end = -1
        self.pred_start = -1
        self.pred_end = -1

    def fit(self, X):
        self.X_ = pd.DataFrame({'Time': list(range(len(X.values()))), 'x': X.values().flatten()})
        self.lib_start = 1
        self.lib_end = int(X.n_timesteps * 0.8)
        self.embed_start = self.lib_end + 1
        self.embed_end = X.n_timesteps
        self.pred_start = X.n_timesteps + 1
        return self

    def predict(self, fh):
        self.pred_end = self.pred_start + fh
        embeddimtable = EmbedDimension(
            dataFrame=self.X_, lib=f"{self.lib_start} {self.lib_end}", pred=f"{self.embed_start} {self.embed_end}",
            columns="x", showPlot=False)
        self.embed_dim = embeddimtable[embeddimtable['rho'] == embeddimtable['rho'].max()]['E']
        self.res = Simplex(dataFrame=self.X_, lib=f"{self.lib_start} {self.embed_end}",
                           pred=f"{self.pred_start} {self.pred_end}",
                           columns=['x'], E=self.embed_dim, showPlot=False,
                           generateSteps=self.pred_end - self.pred_start)
        return self.res['Predictions'][2:]


class SMapPredictor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.X_ = None
        self.res = None
        self.embed_dim = -1
        self.theta = -1
        self.lib_start = -1
        self.lib_end = -1
        self.embed_start = -1
        self.embed_end = -1
        self.pred_start = -1
        self.pred_end = -1

    def fit(self, X):
        self.X_ = pd.DataFrame({'Time': list(range(len(X.values()))), 'x': X.values().flatten()})
        self.lib_start = 1
        self.lib_end = int(X.n_timesteps * 0.8)
        self.embed_start = self.lib_end + 1
        self.embed_end = X.n_timesteps
        self.pred_start = X.n_timesteps + 1
        return self

    def predict(self, fh):
        self.pred_end = self.pred_start + fh
        embeddimtable = EmbedDimension(
            dataFrame=self.X_, lib=f"{self.lib_start} {self.lib_end}", pred=f"{self.embed_start} {self.embed_end}",
            columns="x", showPlot=False)
        self.embed_dim = embeddimtable[embeddimtable['rho'] == embeddimtable['rho'].max()]['E']
        thetatable = PredictNonlinear(dataFrame=self.X_, lib=f"{self.lib_start} {self.lib_end}",
                                      pred=f"{self.embed_start} {self.embed_end}",
                                      columns="x", showPlot=False, E=self.embed_dim)
        self.theta = thetatable[thetatable['rho'] == thetatable['rho'].max()]['Theta']
        self.res = SMap(dataFrame=self.X_, lib=f"{self.lib_start} {self.embed_end}",
                        pred=f"{self.pred_start} {self.pred_end}",
                        columns=['x'], E=self.embed_dim, theta=self.theta, showPlot=False,
                        generateSteps=self.pred_end - self.pred_start)
        return self.res['predictions']['Predictions'][2:]
