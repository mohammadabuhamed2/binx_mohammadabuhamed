import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#1
titanic=pd.read_csv(r"C:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week1\day5\tested.csv")
print(titanic.isnull().sum())
titanic.drop(columns="Cabin",inplace=True)#اخترت الحذف لانو رقم الكبينه او الغرفه مش راح يفرق على الدراسة باشي ولانو اكثر من نصو فاضي صعب نعبيه
titanic['Age']=titanic['Age'].fillna(titanic['Age'].mean())
titanic['Fare']=titanic['Fare'].fillna(titanic['Fare'].median())
print(titanic.isnull().sum())
#2
ishavefamily=(titanic.SibSp>0) | (titanic.Parch>0)
titanic['have_family']=ishavefamily
number_WhoHave_Family=np.array(ishavefamily)
print(number_WhoHave_Family.sum())

#3
fig ,axis=plt.subplots(1,3 ,figsize=(9,3))
axis[0].hist(titanic.Age, bins=np.arange(titanic.Age.min(),titanic.Age.max(),5))
axis[0].set_title("ages chart")
axis[0].set_xlabel("age")
axis[0].set_ylabel('number')
axis[1].scatter(titanic.Age,titanic.Fare)
axis[1].set_title("Relation between age and fare")
axis[1].set_xlabel("age")
axis[1].set_ylabel("fare")
survive = titanic['Survived'].value_counts()
axis[2].bar(['survived','not survivied'],survive.values,color=["red",'green'])
axis[2].set_title('number of survived')
axis[2].set_ylabel('the number')
plt.tight_layout()
plt.show()