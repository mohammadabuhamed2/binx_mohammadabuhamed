#1
import numpy as np
#first column age
#second column experince
#third column salary
#shape (3,3)
employees=np.array([
    [22,3,5000],
    [26,5,8000],
    [28,10,15000]
]
)
#2
weight=np.array([0.5,2,0.002])
res=np.dot(weight,employees[0])
print(res)
#22x0.5=11
#2x3=6
#5000x0.002=10
#11+6+10=27

#3
res2=employees@weight
#(3,3).(3,)valid 
print(res2)

#4
weightE=np.array([1,2])
#res3=employees@weightE
#the erorr in tha shape (3,3).(2,) not valid to fix it we need to add one more weight for last index
weightE=np.append(weightE,3)
res3=employees@weightE
print(res3)
