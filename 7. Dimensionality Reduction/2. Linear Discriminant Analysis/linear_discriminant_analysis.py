#Linear Discriminant Analysis

#Importing the libraries
#Importing numpy to work with arrays of all dimensions
import numpy as np
#Importing only the pyplot module from matplotlib library
import matplotlib.pyplot as plt 
#Importing pandas to work with datasets
import pandas as pd

#Importing the data
#Calling the read_csv function from pandas to read csv file i.e. Wine.csv
dataset = pd.read_csv('Wine.csv')
#Setting the independent variable (feature which will help in prediction)
#Values of all rows, all columns except the last one (iloc = locate index)
X = dataset.iloc[:,:-1].values
#Setting the dependent variable (which is to be predicted)
#All rows, only the last column
y = dataset.iloc[:,-1].values

#Splitting dataset into training and test set
#Importing train_set_split function from model_selection class of sklearn
from sklearn.model_selection import train_test_split
#Using the imported function to split into 2 pairs of training and test set
#Splitting happens at a random point usually according to timestamp so
#giving a random_state an int value for same split always
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Feature scaling
#Importing StandardScaler class from sklearn
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Scaling only the numeric values
X_train = sc.fit_transform(X_train)
#using the same scale or scalar for both test and train set so
#no need to fit the test columns only using transform
X_test = sc.transform(X_test)

#Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#Training the logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
#Calling the LinearRegression class and saving to a variable named regressor
classifier = LogisticRegression(random_state = 0) 
#Training the model with our training set
classifier.fit(X_train, y_train)

#Making the confusion matrix
#To show the accuracy of the prediction
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
#Output format [[tn, fp][fn, tp]]
print(cm) 
#Outputs accuracy % in decimel
print(accuracy_score(y_test, y_pred)) 

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
