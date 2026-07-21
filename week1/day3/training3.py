import numpy as np
l1=np.array([10,2,15])
l2=np.array([15,2,4])
z=np.zeros((2,3))
o=np.ones((4,4))
r=np.arange(0,15,5)
li=np.linspace(0,1,3)
print(f"the zeros matrix is {z}")
print(f"the ones matrix is {o}")
print(f"the range matrix is {r}")
print(f"the line matrix is {li}")
print(np.dot(l1,l2))
print((l1*l2).sum())
print(l1.shape)
l3=np.array([
    [1,2,3],
    
    [4,5,6],
            
    [7,8,9],
    [0,1,2]
])
print(l3)
print(l1)
l4=l3@l1
print("*************")
print(l4)
climate_data=np.genfromtxt("week1/day3/climate.csv",delimiter=",",skip_header=1)
print(climate_data)
print(climate_data.shape)
wh=np.array([0.3,0.7,0.5])
res=np.dot(climate_data,wh)
print(res)
print(res.shape)
fres=np.concatenate((climate_data,res.reshape(10000,1)),axis=1)
print("*********************************")
print(fres)