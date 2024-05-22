#Artificial Neural Networks ( Case Study - Power Plant)

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing pandas to work with datasets
import pandas as pd
#Importing the tensorflow library for ANN
import tensorflow as tf

#Importing the data
#Calling the read_excel function from pandas to read excel file i.e. Data.csv
dataset = pd.read_excel('Folds5x2_pp.xlsx')
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

#Building the Artificial Neural Network
#Initializing the ANN as sequence of layers
ann = tf.keras.models.Sequential()
#Adding the input and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1))

#Training the Artificial Neural Network
#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
#Training the ANN on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Making the predictions and evaluating the model
#Predicting the test set results
#Creating the prediction of X_test
y_pred = ann.predict(X_test)
np.set_printoptions(precision = 2)
#Numpy Concat fn accepts tuple of arrays we want to concatenate and second parameter 1
#meaning axis 0 or 1 (1 for horizontal and 0 for vertical concatenation)
#Printing it vertically so using reshape function which takes 2 parameters
#no of elements in array which we get by len() and second parameter is no of
#column which is 1
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
