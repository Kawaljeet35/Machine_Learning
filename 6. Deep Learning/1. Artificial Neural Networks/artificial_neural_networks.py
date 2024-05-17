#Artificial Neural Networks

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing pandas to work with datasets
import pandas as pd
#Importing the tensorflow library for ANN
import tensorflow as tf

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Data.csv
dataset = pd.read_csv('Churn_Modelling.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,3:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

#Encoding categorical data like names, country names etc
#Encoding dependent variable (Gender in this case)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
#Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Column Transformer takes 2 params, transformers and remainder
#transformer params = what to do, the instance of OneHotEncoder and the index of row
#remainder = what to do with rest of the rows
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
#Transforming X into a numpy array after fit_transform
X = np.array(ct.fit_transform(X))

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature scaling (Necessary in DL for all columns)
#Importing StandardScaler class from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Scaling only the numeric values
X_train = sc.fit_transform(X_train)
#using the same scale or scalar for both test and train set so
#no need to fit the test columns only using transform
X_test = sc.transform(X_test)

#Building the Artificial Neural Network
#Initializing the ANN as sequence of layers
ann = tf.keras.models.Sequential()
#Adding the input and first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
#Adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#Training the Artificial Neural Network
#Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#Training the ANN on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#Making the predictions and evaluating the model
#Predicting a single output
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#Predicting the test set results
#Creating the prediction of X_test
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
#Numpy Concat fn accepts tuple of arrays we want to concatenate and second parameter 1
#meaning axis 0 or 1 (1 for horizontal and 0 for vertical concatenation)
#Printing it vertically so using reshape function which takes 2 parameters
#no of elements in array which we get by len() and second parameter is no of
#column which is 1
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

#Making the confusion matrix
#To show the accuracy of the prediction
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
#Output format [[tn, fp][fn, tp]]
print(cm) 
#Outputs accuracy % in decimel
print(accuracy_score(y_test, y_pred))