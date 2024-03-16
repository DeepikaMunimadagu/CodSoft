# Task - 3
# TOPIC - IRIS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# now let's load the data into the variable iris
iris = pd.read_csv("IRIS.csv",header = 0,sep=",")
# printing first 3 rows to help in visualization
iris.head(3)
# now for data analysis let's see the structure of data frame for any missing values
iris.info()
# let's first see total number of missing values
iris.isnull().sum()
# let's make a count plot to visualize the numbers
sns.countplot(x="species", data=iris)
# our model only understands numeric values so we have to convert the strings into numeric
iris.replace({"species":{"Iris-setosa":0,"Iris-versicolor":1,"Iris-virginica":2}}, inplace=True)
# now let's separate features and target values
# we drop all values that are strings, not required and value that is the target
X = iris.drop(columns=["species"],axis=1)
Y = iris["species"]
# Now let's make 4 arrays
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
# now define the model and fit the training data into it
model = LogisticRegression()
model.fit(X_train, Y_train)
# now let the model predict the outcome using the train data
X_train_prediction = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, X_train_prediction)
print("Training model accuracy = ",train_accuracy)
# now let the model predict the outcome using the test data
X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Testing model accuracy = ",test_accuracy)
# Here training accuracy and testing accuracy are close enough
# So there is neither over-fitting nor under-fitting
