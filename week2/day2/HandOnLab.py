import pandas as dp
import matplotlib.pyplot as plt
import numpy as np

#1
coin=np.random.choice(['head','tail'],size=10000)
heads=np.sum(coin=='head')
#print((heads/10000))

#2
nor=np.random.normal(loc=0,scale=1,size=1000)
plt.hist(nor)
plt.title("normal")
#plt.show()

#3
stud=np.array(['male']*60+['female']*40)
grades=np.array(
    ['yes']*30+
    ['no']*30+
    ['yes']*20+
    ['no']*20
)
total=len(stud)
males=stud=='male'
malescount = np.sum(males)
onlymales=grades[males]
pass_male=np.sum(onlymales=='yes')
pOFb=malescount/total
anb=pass_male/total
pOFanb=anb/pOFb
print(pOFanb)


