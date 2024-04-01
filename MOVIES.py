import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
movies = pd.read_csv("movies.csv",header = 0,sep=",")
movies.head(3)
movies.info()
movies.isnull().sum()
movies = movies.drop(columns=["name","released","director","writer","star","company","budget","gross","runtime","rating"], axis=1)
movies.isnull().sum()
movies["country"].fillna(movies["country"].mode()[0], inplace=True)
movies["score"].fillna(movies["score"].mean(), inplace=True)
movies["votes"].fillna(movies["votes"].mean(), inplace=True)
movies.isnull().sum()
genre = movies["genre"]
score = movies["score"]
plt.figure(figsize=(10, 6))
plt.scatter(genre,score,color="darkcyan")
plt.title("Genre vs Score")
plt.xlabel("Genre")
plt.ylabel("Score")
plt.show()
frequency = movies["genre"].value_counts()
print(frequency)
movies.replace({"genre":{"Comedy":0,"Action":1,"Drama":2,"Crime":3,"Biography":4,"Adventure":5,"Animation":6,"Horror":7,"Fantasy":8,"Mystery":9,"Thriller":10,"Family":11,"Sci-Fi":12,"Romance":13,"Western":14,"Musical":15,"Music":16,"History":17,"Sport":18}},inplace=True)
X = movies.drop(columns=["country","score"],axis=1)
Y = movies["score"]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
X_prediction = model.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(Y_test, X_prediction)
print("Mean Squared Error:", mse)
