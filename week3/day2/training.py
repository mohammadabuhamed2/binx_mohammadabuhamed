import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df= pd.read_csv(r'C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week3\day1\housing.csv')
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
X=df.drop(['median_house_value','ocean_proximity'],axis=1)
y=df['median_house_value']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
prin