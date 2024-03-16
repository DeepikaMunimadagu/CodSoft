# Task - 4
# TOPIC - SALES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# now let's load the data into the variable sales
sales = pd.read_csv("ads.csv")
# printing first 3 rows to help in visualization
sales.head(3)
# now for data analysis let's see the structure of data frame for any missing values
sales.info()
# let's first see total number of missing values
sales.isnull().sum()
# now let's separate features and target values
# we drop all values that are strings, not required and value that is the target
X = sales[['TV', 'Radio', 'Newspaper']]
Y = sales['Sales']
# Now let's make 4 arrays
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# now define the model and fit the training data into it
model = LinearRegression()
model.fit(X_train, Y_train)
# here we use mse to evaluate the model, lower the mse closer are the predicted and actual output values
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
# now here i used random inputs into tv, radio and newspapers to see what the sales output will be
random_sales_input = pd.DataFrame({'TV': [50], 'Radio': [50], 'Newspaper': [50]})
predicted_sales = model.predict(random_sales_input)
print("Sales:", predicted_sales)
# here we see the sales will be 12.5 units if the inputs for tv,radio and newspapers as 50.
# we can use a scatter plot to understand this better
tv = sales["TV"]
radio = sales["Radio"]
news = sales["Newspaper"]
sal = sales["Sales"]

plt.figure(figsize=(10, 6))

# tv VS Sales
plt.subplot(1, 3, 1)
plt.scatter(tv, sal, color="darkcyan")
plt.title("TV advertising vs sales")
plt.xlabel("TV")
plt.ylabel('Sales')

# radio VS sales
plt.subplot(1, 3, 2)
plt.scatter(radio, sal, color="coral")
plt.title("Radio advertising vs Sales")
plt.xlabel("Radio")
plt.ylabel("Sales")

#newspaper VS sales
plt.subplot(1, 3, 3)
plt.scatter(news, sal, color="darkviolet")
plt.title("Newspaper advertising vs Sales")
plt.xlabel("Newspaper")
plt.ylabel("Sales")
