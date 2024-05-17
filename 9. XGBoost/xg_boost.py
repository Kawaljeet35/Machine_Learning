# XGBoost Implementation

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Data.csv
dataset = pd.read_csv('Data.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

# Mapping features [2, 4] to [0, 1]
y = np.where(y == 2, 0, 1)

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the XGBoost on training set
from xgboost import XGBClassifier
#Calling the XGBClassifier class and saving to a variable named regressor
classifier = XGBClassifier() 
#Training the model with our training set
classifier.fit(X_train, y_train)

#Making the confusion matrix
#To show the accuracy of the prediction
from sklearn.metrics import confusion_matrix, accuracy_score
#Predicting the test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#Output format [[tn, fp][fn, tp]]
print(cm) 
#Outputs accuracy % in decimel
print(accuracy_score(y_test, y_pred)) 

#Computing the accuracy with K-fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))