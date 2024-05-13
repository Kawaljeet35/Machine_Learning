#Apriori (Association Rule)

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing and preprocessing the data
#Calling the read_csv function from pandas to read csv file i.e. Market_Basket_Optimisation.csv
#header = None because 1st row in dataset is not a header but actual data
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
#Apriori only accepts a lists as argument so to train it,
#we create a list called transaction using a for loop
transactions = []
#1st for loop to loop over all 7501 rows
#2nd for loop to loop over columns i.e. 0-20
for i in range(1,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training the Apriori model on the dataset
from apyori import apriori
#The function will train and return the rules
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

#Visualising the results
#Displaying the first results coming directly from the Apriori function
results = list(rules)
#Putting the results well organised into a Pandas DataFrame\
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
#Displaying the results non sorted
print(resultsinDataFrame)
#Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
