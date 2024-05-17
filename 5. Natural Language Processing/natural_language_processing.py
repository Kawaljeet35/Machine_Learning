#Natural Language Processing (Bag of Words)

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read tsv file i.e. Restaurant_Reviews.tsv
#Using parameter quoting = 3 to ask model to ignore double quotes in texts
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#Cleaning the texts
#Importing the Python regular expression (regex) module
import re
#Importing the nltk, python library for for NLP tasks
import nltk
#Downloading the NLTK stopwords corpus. 
#Stopwords are common words in a language (e.g., "the", "is", "and").
nltk.download('stopwords')
#Importing the stopwords corpus from NLTK.
from nltk.corpus import stopwords
#To apply stemming on our reviews we import PorterStemmer
#Stemming is taking only the root of the word that can tell the meaning
from nltk.stem.porter import PorterStemmer
#Creating a new list to contain all cleaned reviews 
corpus = []
#Loop to itearate over all reviews to populate the corpus
for i in range(0,1000):
    #review holds individual review and is updated in each step as we clean it
    #First we remove replace non alphabets by a space in ith row of column "Review"
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #Changing every text to lowercase
    review = review.lower()
    #Splitting each word of review
    review = review.split()
    #Creating a class from PorterSteammer
    ps = PorterStemmer()
    #Applying stemming to all words not in stepwords
    all_stopwords = stopwords.words('english')
    #Not including the "not" word in the stopwords
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    #Joining the review again
    review = ' '.join(review)
    #Appending the review to corpus
    corpus.append(review)

#Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
#Creating an object from CountVectorizer class with 1500 words
cv = CountVectorizer(max_features = 1500)
#Creating the matrix of features
X = cv.fit_transform(corpus).toarray()
#Creating the dependent variable
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# #Training the K-Nearest Neighbor model on the training set (64.5%)
# from sklearn.neighbors import KNeighborsClassifier
# #Calling the LinearRegression class and saving to a variable named regressor
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) 
# #Training the model with our training set
# classifier.fit(X_train, y_train)

# #Training the Random Forest Classification model on the training set (72.5%)
# from sklearn.ensemble import RandomForestClassifier
# #Calling the LinearRegression class and saving to a variable named regressor
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# #Training the model with our training set
# classifier.fit(X_train, y_train)

# #Training the Naive Bayes model on the training set (73.0%)
# from sklearn.naive_bayes import GaussianNB
# #Calling the GaussianNB class and saving to a variable named regressor
# classifier = GaussianNB()
# #Training the model with our training set
# classifier.fit(X_train, y_train)

# #Training the Decision Tree model on the training set (75.0%)
# from sklearn.tree import DecisionTreeClassifier
# #Calling the LinearRegression class and saving to a variable named regressor
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# #Training the model with our training set
# classifier.fit(X_train, y_train)

# #Training the Logistic Regression model on the training set (77.5%)
# from sklearn.linear_model import LogisticRegression
# #Calling the LinearRegression class and saving to a variable named regressor
# classifier = LogisticRegression(random_state = 0) 
# #Training the model with our training set
# classifier.fit(X_train, y_train)

# #Training the Kernel SVM model on the training set (78.0%)
# from sklearn.svm import SVC
# #Calling the LinearRegression class and saving to a variable named regressor
# classifier = SVC(kernel = 'rbf', random_state = 0)
# #Training the model with our training set
# classifier.fit(X_train, y_train)

#Training the Support Vector Machine model on the training set (79.0%)
from sklearn.svm import SVC
#Calling the LinearRegression class and saving to a variable named regressor
classifier = SVC(kernel = 'linear', random_state = 0)
#Training the model with our training set
classifier.fit(X_train, y_train)

#Predicting the test set result
#Creating the prediction of X_test
y_pred = classifier.predict(X_test)
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

