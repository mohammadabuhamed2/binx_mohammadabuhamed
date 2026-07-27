import pandas as pd
import numpy as np
titanic=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week2\day1\tested.csv")
age=titanic['Age']
age= age.dropna()
npage=np.array(age)
print(np.mean(npage))
print(np.median(npage))
print(age.mode())
print(np.std(age))
q1,q3=np.percentile(age,[25,75])
IQR=q3-q1
print(IQR)

