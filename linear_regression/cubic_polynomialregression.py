import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import random

class Regression:
    def __init__(self):
        self.x = np.linspace(0, 1, 200)
        self.noise = np.asarray(random.sample((range(200)), 200))
        self.y = self.x**3 * self.noise

    def linreg(self):
        poly3 = PolynomialFeatures(degree=3)
        X = poly3.fit_transform(self.x[:, np.newaxis])
        self.clf3 = LinearRegression()
        self.clf3.fit(X,self.y)



    def visualization(self):
        Xplot = poly3.fit_transform(self.x[:, np.newaxis])


if __name__ == '__main__':
    inst = Regression()
    inst.linreg()
