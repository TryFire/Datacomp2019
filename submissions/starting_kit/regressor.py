from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np


class Regressor(BaseEstimator):
    def __init__(self):
        #self.reg = RandomForestRegressor(n_estimators=10, max_depth=30, max_features=10)
        rng = np.random.RandomState(1)
        self.reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=40),learning_rate=0.1,n_estimators =35,random_state=rng)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)