#1
import pandas as pd
titanic=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week1\day4\tested.csv")
print(titanic)
print(titanic.shape)
print(titanic.columns)
print(titanic.dtypes)

#2
print(titanic.isnull().sum())
titanic.drop(columns="Cabin",inplace=True)#اخترت الحذف لانو رقم الكبينه او الغرفه مش راح يفرق على الدراسة باشي ولانو اكثر من نصو فاضي صعب نعبيه
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
print(titanic.isnull().sum())
print("****************************")
#3
udnerage=titanic['Age']<18
print(titanic[udnerage])
print("****************************")

#4
grp=titanic.groupby('Pclass')['Survived'].sum()#عدد يلي نجو من كل فئة حيث مقاعد درجة اولى الى اخره
print(grp)