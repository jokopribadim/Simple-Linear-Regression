#import package
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#input data
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))  #x_data
y = np.array([5, 20, 14, 32, 22, 38])                   #y_data

#create model and fit it
model = LinearRegression()  #create model
model.fit(x,y)	            #fit proses

#get result
r_sq = model.score(x,y)               #get R square
print('coef. determination: ', r_sq)  #print R square

#predict respon
y_pred = model.predict(x)
print('prediction response: ', y_pred, sep='\n')

#CREATE VISUALIZATION
plt.scatter(x, y, color="red")
plt.plot(x, model.predict(x), color="green")
plt.title("RealPython Simple Linear Regression")
plt.xlabel("x_data")
plt.ylabel("y_data")
plt.show()
