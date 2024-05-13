#Simple Linear Regression

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Salary_Data.csv
dataset = pd.read_csv('Salary_Data.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the Simple Regression Model on the training set
from sklearn.linear_model import LinearRegression
#Calling the LinearRegression class and saving to a variable named regressor
regressor = LinearRegression() 
#Training the model with our training set
regressor.fit(X_train, y_train)

#Predicting the test set result
y_pred = regressor.predict(X_test)

#Visualising the training set result
#Plotting the actual salary vs experience of training set as red dots
plt.scatter(X_train, y_train, color = 'red')
#Plotting the regression line derived from training the model on training set
#Vs predicted y_train in blue color
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#Displays title for the graph
plt.title('Salary Vs Experience (Training Set)')
#Label for x-axis
plt.xlabel('Years of Experience')
#Label for y-axis
plt.ylabel('Salary')
#Displays the final graph
plt.show()

#Visualising the test set result
#Plotting the actual salary vs experience of test set as red dots
plt.scatter(X_test, y_test, color = 'red')
#Plotting the regression line derived from training the model on training set
#Vs predicted y_train in blue color
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#Displays title for the graph
plt.title('Salary Vs Experience (Test Set)')
#Label for x-axis
plt.xlabel('Years of Experience')
#Label for y-axis
plt.ylabel('Salary')
#Displays the final graph
plt.show()



