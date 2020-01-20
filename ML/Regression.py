import matplotlib.pyplot as plt
import numpy as np
import random as rand
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# Generate some data and plot it
# -> Blue
def fun(x):
    return 5 - 5 / x**0.01 + rand.random() / 10000.0
x = np.array([e / 20 for e in range(1, 50)])
y = np.array([fun(f) for f in x])
sp = plt.subplot()
sp.scatter(x, y, marker='.')

# Fit a linear regression model using the generated data, predict values for generated input and plot
# -> Red
reg = linear_model.LinearRegression()
xarr = x.reshape(-1, 1)
reg.fit(xarr, y)
y2 = np.array([reg.predict(f.reshape(-1, 1)) for f in x])
sp = plt.subplot()
sp.scatter(x, y2, marker='.', c='r')

# Fit a model using Polynomial model using the generated data, predict values for generated input and plot
# -> Yellow
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
model = Pipeline([('poly', PolynomialFeatures(degree=3)), ('linear', LinearRegression(fit_intercept=False))])
model = model.fit(x[:, np.newaxis], y)
y3 = np.array([model.predict(f.reshape(-1, 1)) for f in x])
sp = plt.subplot()
sp.scatter(x, y3, marker='.', c='y')

plt.show()
