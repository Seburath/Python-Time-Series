import numpy as np
from sklearn.linear_model import LinearRegression

list_x = [5, 15, 25, 35, 45, 55]
list_y = [5, 20, 14, 32, 22, 38]

#we need to transform lists to arrays to use them with sklearn 
array_x = np.array(list_x)
array_y = np.array(list_y)

print('list_x:\n', array_x)
print('list_y:\n', array_y)

array_x = array_x.reshape((-1, 1))

model = LinearRegression()
model.fit(array_x, array_y)

print('R2: ' + str(model.score(array_x, array_y)))
print('intercept: ' + str(model.intercept_))
print('slope: ' + str(model.coef_[0]))

x_estimated = np.array([65])
x_estimated = x_estimated.reshape((-1, 1))
y_estimated = model.predict(x_estimated)

print('prediction for', x_estimated, ':', y_estimated[0])
