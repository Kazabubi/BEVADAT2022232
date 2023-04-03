import pandas as pd
import numpy as np
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix as conf_mat


csv_path = "diabetes.csv"

class KNNClassifier:

    def __init__(self, k:int, test_set_ratio : float) -> None:
        self.k =k
        self.test_split_ratio = test_set_ratio

    @property
    def k_neighbors(self):
        return self.k
    
    
    @staticmethod
    def load_csv(csv_path:str) ->Tuple[pd.DataFrame,pd.DataFrame]:
        dataset = pd.read_csv(filepath_or_buffer=csv_path, sep=',')
        dataset = dataset.sample(frac=1, random_state=42)
        x,y = dataset.iloc[:,:7],dataset.iloc[:,-1]
        return x,y
    
    
    def train_test_split(self, features:pd.DataFrame,
                     labels:pd.DataFrame) -> None:
        
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train,y_train = features.iloc[:train_size,:],labels.iloc[:train_size]
        x_test,y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    
    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
        return (((self.x_train.loc[:,] - element_of_x.iloc[0])**2).sum(axis=1))**(1/2)
    

    def predict(self) -> None:
        labels_pred = []
        for x_test_element in self.x_test.iterrows():
            distances = self.euclidean(pd.DataFrame(x_test_element[1]).transpose())
            distances = pd.DataFrame(sorted(zip(distances,self.y_train)))
            label_pred = mode(distances[1].head(self.k),keepdims=False).mode
            labels_pred.append(label_pred)

        self.y_preds = pd.Series(labels_pred, dtype = int)

    
    def accuracy(self) -> float:
        true_positive = pd.DataFrame(zip(self.y_test, self.y_preds))
        true_positive = true_positive[0] == true_positive[1]
        return true_positive.sum() / len(self.y_test) * 100
    

    def confusion_matrix(self) -> np.ndarray:
        conf_matrix = conf_mat(self.y_test,self.y_preds)
        return conf_mat

    def best_k(self) -> Tuple[int,float]:
        acc = 0
        r = 0
        for k in range(1,21):
            tmp = self.accuracy(y_test=self.y_test, y_preds=self.predict(self.x_train, self.y_train, self.x_test, k)) 
            if(tmp> acc):
                r = k
                acc = tmp
        return (r, acc)