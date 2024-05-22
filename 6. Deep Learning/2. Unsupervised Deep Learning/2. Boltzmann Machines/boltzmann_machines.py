# Boltzmann Machines

# Importing the libraries
# Importing numpy to work with arrays of all dimensions
import numpy as np
# Importing pandas to work with datasets
import pandas as pd
# Importing the torch libraries
import torch
# Importing torch module to implement the neural networks
import torch.nn as nn
# Importing torch module for parallel computing
import torch.nn.parallel
# Importing torch module for optimiser
import torch.optim as optim
import torch.utils.data
# Importing torch module for stochastic gradient descent
from torch.autograd import Variable

# Importing the dataset
# param1 = filepath, param2 = separator other than comma as movie names have commas in them
# param3 = header as the movie file has no header, param4 = engine to import movies correctly
# param5 = encoding as some movie names have symbols less known
movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine= 'python', encoding = 'latin-1')
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine= 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine= 'python', encoding = 'latin-1')

# Preparing the training set and the test set
# Importing the training and test set, param1 = path, param2 = separator i.e. tab in this case
# and converting them into a numpy array, dtype = data type
training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

# Getting the total number of users and movies
# Taking the max of max user id and movie id from training and test set
# As max user id and movie id can be any of training or test set because
# of the split that happens by a random seed
nb_users = int(max(max(training_set[:,0]),max(test_set[:,0])))
nb_movies = int(max(max(training_set[:,1]),max(test_set[:,1])))

# Coverting the data into an array with users in rows and movies in columns
# Creating a function to do the same for each of the traning and test set
def convert(data):
    # Creating a list of lists, 943 lists for each user and each list will
    # have 1682 elements, 1 for each movie
    new_data = []
    # Using a for loop to poulate the list
    for id_users in range(1,nb_users+1):
        # List of all movies id rated by the user
        id_movies = data[:,1][data[:,0] == id_users]
        # Ratings of movies the user rated
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies-1] = id_ratings
        new_data.append(list(ratings))
    return new_data

# Applying the function to both the training and test set
training_set = convert(training_set)
test_set = convert(test_set)

# Converting data (list of lists) into torch tensors
# for both the training and test set
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 
# 1 (Liked) and 0 (Not Liked)
# First changing the non rated value 0 to -1
training_set[training_set == 0] = -1
# Changing the ratings 1 & 2 to 0 meaning movies not liked
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
# Changing the ratings 3 and above to 1 meaning movies liked
training_set[training_set >= 3] = 1
# Doing the same for test set
test_set[test_set == 0] = -1
# Changing the ratings 1 & 2 to 0 meaning movies not liked
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
# Changing the ratings 3 and above to 1 meaning movies liked
test_set[test_set >= 3] = 1

# Creating the architecture of the neural network
# Creating a RBM class
class RBM():
    # nv & nh = no of visible and hidden nodes respectively
    def __init__(self, nv, nh):
        # W = initialised weight
        self.W = torch.randn(nh, nv)
        # Creating a bias for hidden nodes
        self.a = torch.randn(1, nh)
        # Creating a bias for visible nodes
        self.b = torch.randn(1, nv)
    
    # Function to sample hidden nodes according to probability
    def sample_h(self, x):
        # Computing probability of h given v (t = transpose)
        wx = torch.mm(x, self.W.t())
        # Computing the sigmoid activation function
        activation = wx + self.a.expand_as(wx)
        # Computing the probability of h given v
        p_h_given_v = torch.sigmoid(activation)
        # Returning the probability of h given v and some samples
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    # Function to sample visible nodes according to probability
    def sample_v(self, y):
        # Computing probability of v given h
        wy = torch.mm(y, self.W)
        # Computing the sigmoid activation function
        activation = wy + self.b.expand_as(wy)
        # Computing the probability of v given h
        p_v_given_h = torch.sigmoid(activation)
        # Returning the probability of v given h and some samples
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    # Function to approxiamate the likelihood gradient
    # v0 = input vector, vk = visible nodes after k contrastive divergence
    # ph0 = prob hidden node = 1 given visible node 0, 
    # phk = prob of hidden nodes after k sampling given visible
    def train(self, v0, vk, ph0, phk):
        self.W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Creating an RBM object
# Setting up the parameters nv & nh = no of visible and hidden nodes
nv = len(training_set[0])
nh =  100
# Experimental batch size to update weights after n iteratins 
# where n = batch size
batch_size = 100
# Creating the object
rbm = RBM(nv, nh)

# Training the RBM
# Choosing the number of epochs
nb_epoch = 10
# Looping over each epoch to update the weights
for epoch in range(1,nb_epoch+1):
    # Using a simple difference loss variable
    train_loss = 0
    # Making a float counter
    s = 0.
    # Looping over all users in nb_users with a step = batch_size
    for id_user in range(0, nb_users - batch_size, batch_size):
        # Making the input batch
        vk = training_set[id_user:id_user + batch_size]
        # Making the target batch
        v0 = training_set[id_user:id_user + batch_size]
        # Initial Probabilities
        # ,_ after variable name returns only the first element of the function
        ph0,_ = rbm.sample_h(v0)
        # Looping for k steps
        for k in range(10):
            # ,_ before variable name returns only the first element of the function
            _,hk = rbm.sample_h(vk)
            # Updating vk after gibbs sampling
            _,vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk,_ = rbm.sample_h(vk)
        # Calling the train function to train
        rbm.train(v0, vk, ph0, phk)
        # Updating the loss function
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('test_loss: '+str(test_loss/s))



