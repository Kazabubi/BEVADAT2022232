import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits as dig


class KMeansOnDigits():

    def __init__(self,  n_clusters : int, random_state : int) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self):
        self.digits = dig()

    def predict(self):
        kmeans = KMeans(n_clusters= self.n_clusters, random_state=self.random_state)
        out = kmeans.fit_predict(X=self.digits.data, y=self.digits.target)
        self.clusters = out

    def get_labels(self):
        results = np.zeros((self.clusters.shape))
        for i in range(0,self.n_clusters):
            mask = self.clusters == i
            sub = self.digits.target[mask]
            m = mode(sub)
            results[mask] = m.mode
        self.labels = results

    def calc_accuracy(self):
        self.accuracy = accuracy_score(y_pred=self.labels, y_true=self.digits.target)

    def confusion_matrix(self):
        self.mat = confusion_matrix(y_pred=self.labels, y_true=self.digits.target)


"""
km = KMeansOnDigits(n_clusters=10,random_state=0)
km.load_dataset()
km.predict()
km.get_labels()
km.calc_accuracy()
km.confusion_matrix()
"""