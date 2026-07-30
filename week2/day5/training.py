import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
#1
titanic=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week1\day5\tested.csv")
print(titanic.isnull().sum())
titanic.drop(columns="Cabin",inplace=True)#اخترت الحذف لانو رقم الكبينه او الغرفه مش راح يفرق على الدراسة باشي ولانو اكثر من نصو فاضي صعب نعبيه
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
print(titanic.isnull().sum())

q1= titanic['Fare'].quantile(0.25)
q3=titanic['Fare'].quantile(0.75)
IQR=q3-q1
outliers=titanic[(titanic['Fare']>=q3+IQR*1.5)|(titanic['Fare']<q1-IQR*1.5)]
print(titanic.columns)
#sns.scatterplot(data=titanic,x='Age',y='',hue='Sex')
#plt.show()
#sns.boxplot(data=titanic,x='Embarked',y='Fare')
#plt.show()
corr=titanic.corr(numeric_only=True)
#sns.heatmap(corr,annot=True,cmap="coolwarm")
#plt.show()
sns.pairplot(data=titanic ,hue='Survived')
# حذف PassengerId قبل الرسم لتخفيف الزحام
sns.pairplot(
    data=titanic.drop(columns=['PassengerId']), 
    hue="Survived"
)
plt.tight_layout()
plt.show()