"""
1.  Értelmezd az adatokat!!!
    A feladat megoldásához használd a NJ transit + Amtrack csv-t a moodle-ból.
    A NJ-60k az a megoldott. Azt fogom használni a modellek teszteléséhez, illetve össze tudod hasonlítani az eredményedet.    

2. Írj egy osztályt a következő feladatokra:  
     2.1 Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     2.2 Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     2.3 Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el  
     2.4 Dobjuk el a from és a to oszlopokat, illetve azokat a sorokat ahol van nan és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el  
     2.5 A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     2.6 Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    2.7 A késéseket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0min <= x < 5min   --> 0  
         5min <= x          --> 1  
    2.8 Dobd el a felesleges oszlopokat 'train_id' 'scheduled_time' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    2.9 Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    2.10 Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re. 
    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik (0), ha 5p <= x --> késik (1).
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI-
    HAZI06-
        -NJCleaner.py
        -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""



import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#from src.DecisionTreeClassifier import DecisionTreeClassifier

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain 
        
        # for leaf node
        self.value = value


#from src.Node import Node

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        
        self.root = None
        
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split["info_gain"]>0: 
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["info_gain"])
        

        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds: 
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode=="gini":
            gain = self.gini_index(parent) - (weight_l*self.gini_index(l_child) + weight_r*self.gini_index(r_child))
        else:
            gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
    def gini_index(self, y):
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini
        
    def calculate_leaf_value(self, Y):
        
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, tree=None, indent=" "):
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        



col_name = ['stop_sequence','from_id','to_id','status','line','type','day','part_of_the_day','delay']
data = pd.read_csv("res.csv", header=None, skiprows=1, names=col_name)


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.2, random_state=41)


classifier = DecisionTreeClassifier(min_samples_split=9, max_depth=8)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(accuracy_score(Y_test, Y_pred))

"""

#Legjobb értékek keresése során érzékelt probléma a best_split["info_gain"] által dobott hiba, ugyanis nem jön létre ez a best_split, ha kisebb a hátralévő info gainünk.
#Ezt a problémát try except blokkokkal oldottam meg

#Grid search min_samples_split -re és max_depth -re
#Numpy array-ben tárolom accuracy értékeket. Ha errort dob, 0 a pontosság

#2D array, aminek sorindexei a min_samples_split értékek -2, oszlopindexei pedig a max_depth értékek -2
grid = np.zeros((10,10))

for i in range(0,10):
    for j in range(0, 10):
        #Kivételkezelés
        try:
            #fitelés
            classifier = DecisionTreeClassifier(min_samples_split=i+2, max_depth=j+2)
            classifier.fit(X_train, Y_train)
            Y_pred = classifier.predict(X_test)
            grid[i,j] = accuracy_score(Y_test, Y_pred)
        except:
            print("Error on min_sample_split = " + str(i+2) + " and max_depth = " + str(j+2))


#Lementem accuracy értékeket
np.savetxt(fname="accuracy.csv", X=grid, delimiter=",")

#Maximum érték
m = np.max(grid)
print("Max accuracy = " + str(m))

#maximum értékhez tartozó paraméterek
t = np.where(grid == m)
t = (t[0][0], t[1][0])
print("min_samples_split = " + str(t[0] +2) + ", max_depth = " + str(t[1]+2))


#A kapott táblázat

#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.000833333333333686e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.001666666666666927e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.001666666666666927e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.002500000000000169e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.002500000000000169e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00
#7.794999999999999707e-01,7.886666666666666270e-01,7.889166666666667105e-01,7.934999999999999831e-01,7.946666666666666323e-01,7.979166666666667185e-01,8.002500000000000169e-01,0.000000000000000000e+00,0.000000000000000000e+00,0.000000000000000000e+00

# Ezekből a kiválasztott legnagyobb 0.80025, min_samples_split = 9 (sor + 2) és max_depth = 8 (oszlop + 2)

"""