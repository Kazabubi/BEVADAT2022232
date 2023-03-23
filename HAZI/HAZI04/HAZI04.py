# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''

# %%
'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''

# %%
def csv_to_df(path : str):
    df = pd.read_csv(path)
    return df
#df_data = csv_to_df("StudentsPerformance.csv")
#df_data

# %%
'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''

# %%
def cp_condition(x : str):
    if not x.__contains__('e'):
        x = x.upper()
    return x

def capitalize_columns(df :pd.DataFrame):
    df_out = df.copy()
    df_out = df_out.rename(columns=cp_condition)
    return df_out
#capitalize_columns(df_data)

# %%
'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''

# %%
def math_passed_count(df : pd.DataFrame):
    df_out = df.copy()
    x = np.where(df_out["math score"] > 49)
    return np.shape(x)[1]
#math_passed_count(df_data)

# %%
'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''

# %%
def did_pre_course(df : pd.DataFrame):
    df_out = df.copy()
    x = df_out.loc[df_out["test preparation course"] == "completed"]
    return x
#did_pre_course(df_data)

# %%
'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''

# %%


# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''

# %%
def add_age(df : pd.DataFrame):
    df_out = df.copy()
    np.random.seed = 42
    x = np.random.randint(low=18, high=67, size=np.shape(df_out)[0])
    df_out["age"] = x
    return df_out
#add_age(df_data)

# %%
'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''

# %%
def female_top_score(df : pd.DataFrame):
    df_out = df.copy()
    x = df_out.loc[df_out["gender"] == "female"]
    x["highscore"] = x["math score"] + x["reading score"] + x["writing score"]
    y = x.loc[x["highscore"] == x["highscore"].max()].iloc[0]
    return(y["math score"], y["reading score"], y["writing score"])
#female_top_score(df_data)

# %%
'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''

# %%
def add_grade(df : pd.DataFrame):
    df_out = df.copy()
    x = (df_out["math score"] + df_out["reading score"] + df_out["writing score"]) / 300
    criteria =[x.between(0.9, 1),x.between(0.8, 0.89),x.between(0.7, 0.79),x.between(0.6, 0.69),x.between(0, 0.6)]
    values = ["A", "B", "C", "D", "F"]
    df_out["grade"] = np.select(criteria, values, 0)
    return df_out
#add_grade(df_data)

# %%
'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''

# %%
def math_bar_plot(df : pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_title("Avarage Math Score by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Math Score")
    ax.bar(df["gender"], np.average(df["math score"]))
    return fig
#math_bar_plot(df_data)

# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''

# %%
def writing_hist(df : pd.DataFrame):
    fig, ax = plt.subplots()
    ax.set_label("Distribution of Writing Scores")
    ax.set_xlabel("Writing Score")
    ax.set_ylabel("Number of Student")
    ax.hist(df["writing score"])
    return fig
#writing_hist(df_data)


# %%
''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

# %%
def ethnicity_pie_chart(df : pd.DataFrame):
    df_out = df.copy()
    fig, ax = plt.subplots()
    ax.set_title('Proportion of Students by Race/Ethnicity')
    x = df.groupby(["race/ethnicity"])
    ax.pie(x["race/ethnicity"].count(), labels=x.groups.keys(), autopct='%1.1f%%')
    return fig
#ethnicity_pie_chart(df_data)


