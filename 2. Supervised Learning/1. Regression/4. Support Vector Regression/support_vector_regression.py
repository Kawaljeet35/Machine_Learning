#Support Vector Regression

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Position_Salaries.csv
dataset = pd.read_csv('Position_Salaries.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,1:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

#Turning y into a 2-d array
#So as to Print it vertically so using reshape function which takes 2 parameters
#no of elements in array which we get by len() and second parameter is no of
#column which is 1
y = y.reshape(len(y), 1)

#Feature scaling
#Importing StandardScaler class from sklearn
from sklearn.preprocessing import StandardScaler
#Using two different scales for X and y because they 
#have different range and hence different mean
sc_X = StandardScaler()
sc_y = StandardScaler()
#Creates a new scaled X
X = sc_X.fit_transform(X)
#Creates a new scaled y
y = sc_y.fit_transform(y)

#Training the SVR model on the whole dataset
from sklearn.svm import SVR
#Using the most popular kernel 'rbf' made for SVR model
regressor = SVR(kernel= 'rbf')
#Training the model with X and y (scaled)
regressor.fit(X, y)

#Predicting a new result
#As the model was trained on scaled value so it is necessary to
#scale 6.5 with the same scalar and turning into a 2d array
#Using inverse transform to convert output to original scale as 
#y was also scaled and transformed before being used for training
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))

#Visualizing the SVR result
#Scaling back the X and y to original scale
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()