import pandas as pd
import seaborn as sns
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