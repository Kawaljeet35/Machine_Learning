# Self Organizing Maps

# Importing the libraries
# Importing numpy to work with arrays of all dimensions
import numpy as np
# Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
# Importing pandas to work with datasets
import pandas as pd

# Importing the data
# Calling the read_csv function from pandas to read csv file i.e. Credit_Card_Applications.csv
dataset = pd.read_csv('Credit_Card_Applications.csv')
# Setting the independent variable (feature which will help in clustering)
# Values of all rows, and all columns except the last one 
X = dataset.iloc[:,:-1].values
# Setting the dependent variable with only the last columns and all rows
y = dataset.iloc[:,-1].values

# Feature scaling (Applying Normalization)
# Importing MinMaxScaler class from sklearn
from sklearn.preprocessing import MinMaxScaler
# Feature scale = (0,1) for all values to fall under 0 and 1
sc = MinMaxScaler(feature_range = (0,1))
# Scaling the rows of X and saving to a new variable
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
# Creating an object from Minisom class
# X and y = 10 is the grid size 10x10, input_len = no of features in X
# sigma is the radius of different neighbors, learning_rate = how
# much weights are updated per iteration
som = MiniSom(x=10, y=10, input_len= 15, sigma= 1.0, learning_rate = 0.5)
# Initialisation the values of weights vector closer to 0
som.random_weights_init(X)
# Training the SOM on X, num_iterations = number of iterations
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
# Initializing the window that will contain the map with bone fn
bone()
# Putting the different winning nodes on the map, T = transpose of the matrix
pcolor(som.distance_map().T)
# Adding a legend to show which colors correspond to what
colorbar()
# Creating a vector of markers, 'o' for circle and 's' for square
markers = ['o', 's']
# Creating the color of markers, 'r' for red and 'g' for green
# red if customer not approved & green if approved
colors = ['r', 'g']
# Looping over all customers to get each winning node
# i = vaues of different values, x = customers
for i, x in enumerate(X):
    w = som.winner(x)
    # 0.5 added to put on middle of square
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
# Getting the coordinates of winning nodes
frauds = np.concatenate((mappings[(8,2)], mappings[(8, 5)]), axis = 0)
frauds = sc.inverse_transform(frauds)
