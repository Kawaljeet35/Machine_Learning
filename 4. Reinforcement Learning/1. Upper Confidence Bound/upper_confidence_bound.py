#Upper Confidence Bound (UCB)

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

#Implementing Upper Confidence Bound
#Importing math for calculation
import math
#Total no of users or rounds i.e. 10,000
N = 10000
#Total no of ads i.e. 10
d = 10
#Full list of ads selected i.e. nth element = ad selected at round n
ads_selected = []
#No of times ad was selected
numbers_of_selection = [0] * d
#Sum of total rewards
sum_of_rewards = [0] * d
#Total rewards accumulated
total_reward = 0
#Outer for loop to iterarte all the rounds
for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    #Inner for loop to iterate over all the ads
    for i in range(0,d):
        if numbers_of_selection[i] > 0:
            average_reward = sum_of_rewards[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2 *math.log(n+1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] += reward
    total_reward += reward

#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()