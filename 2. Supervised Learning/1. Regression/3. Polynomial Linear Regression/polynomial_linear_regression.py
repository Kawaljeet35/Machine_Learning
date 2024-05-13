#Polynomial Linear Regression

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

#Training the Simple Regression Model on the whole set
from sklearn.linear_model import LinearRegression
#Calling the LinearRegression class and saving to a variable named lin_reg
lin_reg = LinearRegression() 
#Connecting the method with our features
lin_reg.fit(X, y)

#Training the Polynomial Linear Regression Model on the whole set
from sklearn.preprocessing import PolynomialFeatures
#Calling the PolynomialFeatures class and saving to a variable named poly_reg
#Degree refers to the power we want
poly_reg = PolynomialFeatures(degree = 4)
#Training the model on polynomial regression
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
#Connecting the method with our new features
lin_reg_2.fit(X_poly, y)

#Visualizing the Linear Regression result
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualizing the Polynomial Regression result
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting salary from Linear Regression model
print(lin_reg.predict([[6.5]]))

#Predicting salary from Polynomial Regression model
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))

