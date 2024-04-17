import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=10)
poly_x = pf.fit_transform(x)
reg2 = LinearRegression()
reg2.fit(poly_x, y)

plt.scatter(x,y,color='blue')
plt.plot(x, reg.predict(x), color='red')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(x,y,color='blue')
plt.plot(x, reg2.predict(poly_x), color='red')
plt.title('Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

print(reg.predict([[6.5]]))
print(reg2.predict(pf.fit_transform([[6.5]])))



