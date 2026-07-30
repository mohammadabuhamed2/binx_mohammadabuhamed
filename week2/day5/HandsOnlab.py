import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
titanic=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week1\day5\tested.csv")
print(titanic.isnull().sum())
titanic.drop(columns="Cabin",inplace=True)#اخترت الحذف لانو رقم الكبينه او الغرفه مش راح يفرق على الدراسة باشي ولانو اكثر من نصو فاضي صعب نعبيه
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
print(titanic.isnull().sum())


