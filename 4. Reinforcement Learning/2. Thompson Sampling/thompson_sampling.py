#Thompson Sampling

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Ads_CTR_Optimisation.csv
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
import random
#Total no of users or rounds i.e. 10,000
N = 10000
#Total no of ads i.e. 10
d = 10
#Contains the no of selected ads after each round (len=10000)
ads_selected = []
#Number of times ad i got reward 1 upto round n
numbers_of_rewards_1 = [0] * d
#Number of times ad i got reward 0 upto round n
numbers_of_rewards_0 = [0] * d
#Total rewards accumulated
total_reward = 0
#Outer for loop to iterarte all the rounds
for n in range(0,N):
    ad = 0
    max_random = 0
    #Inner for loop to iterate over all the ads
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1
    total_reward = total_reward + reward

#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

