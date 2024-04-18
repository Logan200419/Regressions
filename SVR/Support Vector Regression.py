import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('Position_Salaries1.csv')
x = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
y = y.reshape(len(y),1)
print(x)
print(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)
print(x)
sc2 = StandardScaler()
y = sc2.fit_transform(y)
print(y)

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
reg.fit(x,y)

np.set_printoptions(precision=2)
print(sc2.inverse_transform(reg.predict(sc.transform([[6.5]])).reshape(-1, 1)))


