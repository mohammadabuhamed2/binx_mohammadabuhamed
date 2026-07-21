import numpy as np
#1
randarr=np.arange(1,17).reshape(4,4)
print(randarr)
print(randarr.shape)
print(randarr.dtype)
#2
#ما عرفتش قصدك بالزبط فراح احلها بطريقتين
subarr=randarr[3,2]
print(subarr)
###################
print(randarr[0:,1].reshape(4,1))
print(randarr[3])
#3
means=randarr.mean()
gt=randarr>means
print(randarr[gt])
addedrow=np.random.randint(1,20,size=4)
finalarr=randarr+addedrow
print(randarr)
print("**********************")
print(addedrow)
print("**********************")
print(finalarr)
check1=randarr[0,0]+addedrow[0]
print(f"""
      the number in the matrix[0,0]= {randarr[0,0]}
      the addnumber from the addrow (0)= {addedrow[0]}
       {randarr[0,0]} + {addedrow[0]} = {finalarr[0,0]} 
      """)

