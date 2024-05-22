# Recurrent Neural Networks

# Part 1 - Data Preprocessing

# Importing the libraries
# Importing numpy to work with arrays of all dimensions
import numpy as np
# Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
# Importing pandas to work with datasets
import pandas as pd

# Importing the training set
# Calling the read_csv function from pandas to read csv file i.e. Google_Stock_Price_Train.csv
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# Setting the training set (feature which will help in prediction)
# Values of all rows, and only 2nd column with index 1 (iloc = locate index)
training_set = dataset_train.iloc[:,1:2].values

# Feature scaling (Applying Normalization)
# Importing MinMaxScaler class from sklearn
from sklearn.preprocessing import MinMaxScaler
# Feature scale = (0,1) for all values to fall under 0 and 1
sc = MinMaxScaler(feature_range = (0,1))
# Scaling the rows of training set and saving to a new variable
training_set_scaled = sc.fit_transform(training_set)

# Creating a new data structure with 60 timesteps and 1 output
# 60 timesteps means at each time "t" the RNN will look at total 60 stock prices
# before time "t" (stock prices between t-60 and t) and based on those trends
# it will try to predict the next output i.e. stock price at time t+1
# X_train to hold the input to the neural network containing stock prices of previous 60 financial days
X_train = []
# y_train to hold the stockprice of the next financial day
y_train = []
# Using a for loop to populate both the input/output lists
# Range starts at 60th row as we have to look back till t-60
for i in range(60, 1258):
    # Appending stock prices of previous 60 financial days per row i in range & for column index 0
    X_train.append(training_set_scaled[i-60:i, 0])
    # Appending stock prices at t+1 financial day i.e. i (because idx starts from 0)
    y_train.append(training_set_scaled[i, 0])
# Converting X_train and y_train lists into numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping the training set (Creating a new dimension)
# 2nd parameter is the shape we want i.e. 3d
# 3-D tensor with shape (batch_size, timesteps, features) taken from keras api
# batch size = total no of observations in X_train, timesteps = 60, features = indicators or predictors
# X_train.shape[0] gives the number of rows in X_train (batch_size)
# X_train.shape[1] gives the number of columns in X_train (timesteps)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential # type: ignore
from keras.layers import Dense  # type: ignore
from keras.layers import LSTM # type: ignore
from keras.layers import Dropout # type: ignore

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation to avoid overfitting
# Adding the LSTM layer with 3 params - no of memory cells (units), return sequences = true,
# input shape that is the shape of input in X_train i.e. in 3D but only the last 2
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# Adding the Dropout regularisation with Dropout rate as param (neurons to drop/ignore)
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation to avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation to avoid overfitting
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation to avoid overfitting
# return_sequences = True removed because no more layers to add
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
# units = dimension of output layer or no of neurons
regressor.add(Dense(units = 1))

# Compiling the RNN
# optimizer = adam for stochastic gradient descent
# loss = mean_squared_error for regression
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
# Training the ANN on the training set
# batch size = 32 because its a classic value in mathematics
# epochs = 100 (number of back propagations and forward propagations)
regressor.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
# Importing the test set for predictions and
# Converting them to a numpy array
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

# Getting the predicted stock price of 2017
# Concatinating training set and test set to get 60 financial days back 
# of before time "t" as done in the RNN, axis = 0 means vertical concat
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# Getting the data from 60 financial days back for prediction
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# Reshaping to get the right numpy array
inputs = inputs.reshape(-1,1)
# Scaling the inputs array because RNN was trained on scaled data
inputs = sc.transform(inputs)
# Creating a new data structure with test set
X_test = []
for i in range(60, 80):
    # Appending stock prices of previous 60 financial days per row i in range & for column index 0
    X_test.append(inputs[i-60:i, 0])
# Converting X_train list into a numpy array
X_test = np.array(X_test)
# Reshaping the test set (Creating a new dimension)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Making the prediction
predicted_stock_price = regressor.predict(X_test)
# Inversing the scaled as the training was done on scaled data
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()