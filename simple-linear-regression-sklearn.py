import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import metrics  

list_x = [5, 15, 25, 35, 45, 55]
list_y = [5, 20, 14, 32, 22, 38]

array_x = np.array(list_x)
array_x = array_x.reshape((-1, 1))
array_y = np.array(list_y)

model = LinearRegression()
model.fit(array_x, array_y)

print('R2: ' + str(model.score(array_x, array_y)))
print('intercept: ' + str(model.intercept_))
print('slope: ' + str(model.coef_))

y_estimated = model.predict(array_x)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_estimated, array_y))  
print('Mean Squared Error:', metrics.mean_squared_error(y_estimated, array_y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_estimated, array_y)))  
