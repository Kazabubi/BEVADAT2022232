# %%
import numpy as np

# %%
#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)

# %%
#Készíts egy függvényt ami létre hoz egy nullákkal teli numpy array-t.
#Paraméterei: mérete (tuple-ként), default mérete pedig legyen egy (2,2)
#Be: (2,2)
#Ki: [[0,0],[0,0]]
#create_array()

# %%
def create_array(size = (2,2)):
    return np.zeros(size)

# %%
#Készíts egy függvényt ami a paraméterként kapott array-t főátlóját feltölti egyesekkel
#Be: [[1,2],[3,4]]
#Ki: [[1,2],[3,1]]
#set_one()

# %%
def set_one(arr : np.ndarray):
    np.fill_diagonal(arr, 1)
    return arr


# %%
# Készíts egy függvényt ami transzponálja a paraméterül kapott mártix-ot:
# Be: [[1, 2], [3, 4]]
# Ki: [[1, 3], [2, 4]]
# do_transpose()

# %%
def do_transpose(arr : np.ndarray):
    return arr.transpose()

# %%
# Készíts egy olyan függvényt ami az array-ben lévő értékeket N tizenedjegyik kerekíti, ha nincs megadva ez a paraméter, akkor legyen az alapértelmezett a kettő 
# Be: [0.1223, 0.1675], 2
# Ki: [0.12, 0.17]
# round_array()

# %%
def round_array(arr : np.ndarray, r = 2):
    return arr.round(2)

# %%
# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 0 - False-ra, az 1 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# bool_array()

# %%
def bool_array(arr : np.ndarray):
    return arr.astype(bool)

# %%


# %%
# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 1 - False-ra az 0 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ False True True], [ False  False  False], [True True True]]
# invert_bool_array()

# %%
def rt_bool_array(arr : np.ndarray):
    x =arr.astype(bool)
    return np.invert(x)

# %%
# Készíts egy olyan függvényt ami a paraméterként kapott array-t kilapítja
# Be: [[1,2], [3,4]]
# Ki: [1,2,3,4]
# flatten()


# %%
def flatten(arr : np.ndarray):
    return arr.reshape(-1)


