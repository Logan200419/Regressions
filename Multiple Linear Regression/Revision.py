import pandas as pd
import numpy as np

data=pd.read_csv('50_Startups.csv')
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough')
X = ct.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)


# Comparing Predicted and Test Values
Y_pred = reg.predict(X_test)
np.set_printoptions(precision=2)
Compare = np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)),axis=1)
print(Compare)
