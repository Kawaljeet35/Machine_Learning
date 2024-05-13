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

#Taking care of missing data
#Importing SimpleImputer class from sklearn.impute module
from sklearn.impute import SimpleImputer
#imputer variable calling the class with params what to replace and with what to replace
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
#Connecting imputer with only numerical values of X (row index 1 and 2)
imputer.fit(X[:,1:3])
#Replacing and updating the value of modified values of X with transform method
#Transform returns the new updated values
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data like names, country names etc
#Encoding independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Column Transformer takes 2 params, transformers and remainder
#transformer params = what to do, the instance of OneHotEncoder and the index of row
#remainder = what to do with rest of the rows
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#Transforming X into a numpy array after fit_transform
X = np.array(ct.fit_transform(X))
#Encoding dependent variable (Yes and No in this case)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling
#Importing StandardScaler class from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Scaling only the numeric values
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
#using the same scale or scalar for both test and train set so
#no need to fit the test columns only using transform
X_test[:,3:] = sc.transform(X_test[:,3:])



