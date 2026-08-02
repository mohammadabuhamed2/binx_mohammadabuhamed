import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
df= pd.read_csv(r'C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week3\day1\housing.csv')
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
print(df.isna().sum())
print(df.columns)
print(df.dtypes)
X=df.drop(['median_house_value','ocean_proximity'],axis=1)
y=df['median_house_value']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

