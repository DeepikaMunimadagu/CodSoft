# Task - 5
# TOPIC - CREDIT CARD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# now let's load the data into the variable credit
credit = pd.read_csv("creditcard.csv")
# printing first 3 rows to help in visualization
credit.head(3)
# now for data analysis let's see the structure of data frame for any missing values
credit.info()
# let's first see total number of missing values
credit.isnull().sum()
# let's make a count plot to visualize the numbers
sns.countplot(x="Class", data=credit)
# so now we see that there are a lot more true values than the fraud ones.
# we can either use over-sampling and undersampling to make the numbers equal.
# beacuse if we use a data set with this big of a difference there will be less recall value as model won't be able to detect 
# minority class effectively 
# so to train it effectively we need to make a fair input.
# here undersampling is used,because firstly it reduces the computational time and also because making a model learn from legit
# value is more effective than training it with synthetic values and with legit values our model will be better accustomed to
# predict the outcome
# define two separate classes
correct = credit[credit.Class == 0]
fraud = credit[credit.Class == 1]
# let's see no. of fraud transactions
# this number will help us after we undersample
credit["Class"].value_counts()
# so fraud value is 492 and correct value is 284315
# so now we take same number of correct values as the fraud values and then using that dataframe train our model
correct_value = correct.sample(n=492)
credit_card_new = pd.concat([correct_value,fraud],axis=0)
# now lets see the number of fraud and correct values
credit_card_new["Class"].value_counts()
# now let's separate features and target values
# we drop all values that are strings, not required and value that is the target
X = credit_card_new.drop(columns=["Class"],axis=1)
Y = credit_card_new["Class"]
# Now let's make 4 arrays
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
# now define the model and fit the training data into it
model = LogisticRegression()
model.fit(X_train, Y_train)
from sklearn.metrics import precision_score, recall_score, f1_score
# now let the model predict the outcome using the train data
X_train_prediction = model.predict(X_train)
# let's calculate precision, recall and f1 score for training data
train_precision = precision_score(Y_train, X_train_prediction)
train_recall = recall_score(Y_train, X_train_prediction)
train_f1 = f1_score(Y_train, X_train_prediction)
print("Precision: ",train_precision)
print("Recall: ",train_recall)
print("F1 Score: ",train_f1)
# now let the model predict the outcome using the test data
X_test_prediction = model.predict(X_test)
# let's calculate precision, recall and f1 score for testing data
test_precision = precision_score(Y_test, X_test_prediction)
test_recall = recall_score(Y_test, X_test_prediction)
test_f1 = f1_score(Y_test, X_test_prediction)
print("Precision: ",test_precision)
print("Recall: ",test_recall)
print("F1 Score: ",test_f1)
# as we can see the training and testing datasets yielded approximately same values
# so we conclud our model is effective and neither over-fitted nor under-fitted
