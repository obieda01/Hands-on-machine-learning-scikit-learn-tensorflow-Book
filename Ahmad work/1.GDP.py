import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

#load data
oecd_bli=pd.read_csv("C:\Deep Learning Course\handson-ml-master\github\datasets\lifesat\oecd_bli_2015.csv",thousands=',')
gdp_per_capita=pd.read_csv("C:\Deep Learning Course\handson-ml-master\github\datasets\lifesat\gdp_per_capita.csv"
                    ,thousands=',',delimiter='\t',encoding='latin1',na_values='n/a')


#prepare the data 
country_stats=prepare_country_stats(oecd_bli,gdp_per_capita)
x=np.c_[country_stats["GPD per Capita"]]
y=np.c_[country_stats["Life satisfaction"]]

#visualize the data 
country_stats.plot(kind='scatter',x="GPD per Capita",y="Life satisfaction")
plt.show()

#Select a linear model
model=sklearn.linear_model.LinearRegression()

#Train the model
model.fit(x,y)


#Make a prediction for Cyprus
X_new=[[22587]]
print(model.predict(X_new))