import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
apples=[0.957,0.967,0.976,0.980,0.985,0.990]
orange=[0.997,0.987,0.966,0.950,0.945,0.930]
years=[2012,2013,2014,2015,2016,2017]
#plt.plot(years,apples, marker='o')
#plt.plot(years,orange,marker='x')
plt.xlabel('year')
plt.ylabel("frutes in tons")
plt.title("frutes growth")
plt.legend(['orange','apples'])
flow=pd.read_csv(r"c:\Users\Hp\Desktop\binx_mohammadabuhamed\binx_mohammadabuhamed\week1\day5\iris_148.csv")
print(flow.species.unique())
#plt.scatter(flow.sepal_length,flow.sepal_width)
dfsetosa=flow[flow.species=='setosa']
dfversicolor=flow[flow.species=='versicolor']
dfvirginica=flow[flow.species=='virginica']
plt.hist(dfsetosa.sepal_width, bins=np.arange(2,5,0.25))
plt.hist(dfversicolor.sepal_width,bins=np.arange(2,5,0.25))
plt.hist(dfvirginica.sepal_width,bins=np.arange(2,5,0.25))
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.show()

