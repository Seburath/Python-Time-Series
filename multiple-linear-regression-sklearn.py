import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

list_x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
list_y = [5, 20, 14, 32, 22, 38, 43, 55]

array_x = np.array(list_x)
array_y = np.array(list_y)

print('array_x:\n', array_x)
print('array_y:\n', array_y)

model = LinearRegression()
model.fit(array_x, array_y)

print('R2: ' + str(model.score(array_x, array_y)))
print('intercept: ' + str(model.intercept_))
print('slop: ' + str(model.coef_))

x_estimated = np.array([[65, 45]])
y_estimated = model.predict(x_estimated)

print('prediction for', x_estimated, ':', y_estimated)
