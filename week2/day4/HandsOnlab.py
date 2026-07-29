import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#1
df=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week2\day4\tested.csv")
df=df[['Survived','Name', 'Sex', 'Age',
         'Fare','Embarked']]
fig,axis=plt.subplots(1,3, figsize=(9,5))
sns.histplot(data=df,x='Age',ax=axis[0])
axis[0].set_tilte="age"
sns.histplot(data=df,x='Fare',ax=axis[1])
axis[1].set_tilte="fare"
sns.histplot(data=df,x='Survived',ax=axis[2])
axis[2].set_tilte="survived"
plt.show()

#2

#The box plot shows several outliers in the Fare variable.
#These points represent passengers who paid very high ticket prices.
#Further investigation is required before deciding how to handle them.
sns.boxplot(data=df,x='Fare')
plt.show()
#3
q1=df['Fare'].quantile(0.25)
q3=df['Fare'].quantile(0.75)
IQR=q3-q1
outliers=df[(df['Fare']<q1-IQR*1.5)|(df['Fare']>q3+IQR*1.5)]
print(outliers)
print(outliers.count())
#Outlier Handling
#The IQR method detected several outliers in the Fare column.
#These outliers were kept because they represent real passengers who paid expensive ticket prices rather than data entry errors.
#Removing them could result in losing valuable information.

#4
fig,axis=plt.subplots(1,2,figsize=(9,5))
sns.countplot(data=df,x='Sex',ax=axis[0])
sns.countplot(data=df,x='Embarked',ax=axis[1])
plt.show()
#Sex
#There are more male passengers than female passengers.
#This indicates a class imbalance between the two categories.

#Embarked
#Most passengers embarked from Southampton (S).
#Fewer passengers embarked from Cherbourg (C) and Queenstown (Q).