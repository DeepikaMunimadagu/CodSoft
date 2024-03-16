# Task - 1
# TOPIC - TITANIC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# now let's load the data into the variable titanic_data
titanic_data = pd.read_csv("titanic.csv",header = 0,sep=",")
# printing first 3 rows to help in visualization
titanic_data.head(3)
# now for data analysis let's see the structure of dat frame for any misiing values
titanic_data.info()
# here we see there are missing values in cabin, embarked and age columns
# to fix we either delete records with missing values or add values.
# due to large number of missing values the data set will be in- effective
# so we add the most repeated values to empty sets to balance it out
# but in case of cabin, missing value is very large and the column is not important for analysis so we simple drop the cabin column
# let's first see total number of missing values
titanic_data.isnull().sum()
# drop column cabin
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
# for embarked and age columns, we replace mean age with missing values and most repeated value in embarked column with the missing values
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
# now let's see total empty values
titanic_data.isnull().sum()
# let's make a count plot to visualize the numbers
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
# our model only understands numeric values so we have to convert the strings into numeric
titanic_data.replace({"Sex":{"male":0,"female":1},"Embarked":{"S":0,"C":1,"Q":2}}, inplace=True)
# now let's separate features and target values
# we drop all values that are strings, not required and value that is the target
X = titanic_data.drop(columns=["PassengerId","Name","Ticket","Survived"],axis=1)
Y = titanic_data["Survived"]
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
