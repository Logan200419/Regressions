import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Salary_Data.csv')
X= data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, Y_train)

# Visualizing Training Set
plt.scatter(X_train, Y_train,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Experience vs Salary (Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing Test Set
plt.scatter(X_test, Y_test,color='red')
plt.plot(X_train, reg.predict(X_train), color='blue')
plt.title('Experience vs Salary (Testing set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

print(reg.predict([[1]]))

