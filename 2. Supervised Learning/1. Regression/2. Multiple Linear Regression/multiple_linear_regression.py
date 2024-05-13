#Multiple Linear Regression

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. 50_Startups.csv
dataset = pd.read_csv('50_Startups.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

#Encoding categorical data like names
#Encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Column Transformer takes 2 params, transformers and remainder
#transformer params = what to do, the instance of OneHotEncoder and the index of row
#remainder = what to do with rest of the rows
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
#Transforming X into a numpy array after fit_transform
X = np.array(ct.fit_transform(X))

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
#Connecting the method with our training set
regressor.fit(X_train, y_train)

#Predicting and displaying the test set result
#Comparing y_pred to y_test
y_pred = regressor.predict(X_test)
#Trying to display numeric values upto 2 decimal places
np.set_printoptions(precision=2)
#Accepts tuple of arrays we want to concatenate and second parameter 1
#meaning axis 0 or 1 (1 for horizontal and 0 for vertical concatenation)
#Printing it vertically so using reshape function which takes 2 parameters
#no of elements in array which we get by len() and second parameter is no of
#column which is 1
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Making a single prediction (for example the profit of a startup with R&D Spend = 160000, 
#Administration Spend = 130000, Marketing Spend = 300000 and State = 'California')
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

