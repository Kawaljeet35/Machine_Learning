# AutoEncoders

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

# Creating the architecture of the neural network
# Creating an AutoEncoder class
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+' loss: '+ str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))
