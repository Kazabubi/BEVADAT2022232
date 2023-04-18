import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        iris = load_iris()
        self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.L = lr
        self.Epoch = epochs
        X = self.data['petal width (cm)'].values
        y = self.data['sepal length (cm)'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def fit(self, X: np.array, y: np.array):
        self.m = 0
        self.c = 0

        n = float(len(X)) # Number of elements in X

        # Performing Gradient Descent 
        losses = []
        for i in range(self.Epoch): 
            y_pred = self.m*X + self.c  # The current predicted value of Y

            residuals = y - y_pred
            loss = np.sum(residuals ** 2)
            losses.append(loss)
            D_m = (-2/n) * sum(X * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.L * D_m  # Update m
            self.c = self.c - self.L * D_c  # Update c

    def predict(self, X):
        self.y_pred = self.m*X + self.c

    def plotLR(self, X, y):
        fig, ax = plt.subplots()
        ax.scatter(X,y)
        ax.plot([min(X), max(X)], [min(self.y_pred), max(self.y_pred)], color='red')
        return fig
    
    def evaluate(self, X, y):
        self.mse = np.mean((X - y)**2)


LR = LinearRegression(2000,0.001)
LR.fit(LR.X_train, LR.y_train)
LR.predict(LR.X_test)
LR.plotLR(LR.X_test, LR.y_test).show()
LR.evaluate(LR.y_pred, LR.y_test)
print(LR.mse)
input()
