import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#In the dataset col7=age col5=BMI(BodyMassIndex) col8=outcome/diabetes
array_y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
array_x = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 

model = LinearRegression()
model.fit(array_x, array_y)
results = model.fit(array_x, array_y)

y_estimated = model.predict(array_x)
y_estimated_round = []
for i in y_estimated:
    if i < 0.5:
        y_estimated_round.append(0)
    else:
        y_estimated_round.append(1)

r_sq = model.score(array_x, array_y)
intercept, coefficients = model.intercept_, model.coef_
print('coefficient of determination:', r_sq)
print('intercept:', intercept)
print('coefficients:', coefficients, sep='\n')

print('Mean Absolute Error:', metrics.mean_absolute_error(y_estimated, array_y))  
print('Mean Squared Error:', metrics.mean_squared_error(y_estimated, array_y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_estimated, array_y))) 
#plot
colors = ListedColormap(['#0000FF', '#FF0000'])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Diabetes 1/0')
ax.scatter(array_x[:,0], array_x[:,1], array_y, c=array_y, cmap=colors)
plt.suptitle("Pima Indians Diabetes Database")

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('Age')
ax2.set_ylabel('BMI')
ax2.set_zlabel('Diabetes 1/0')
ax2.scatter(array_x[:,0], array_x[:,1], y_estimated, c=array_y, cmap=colors)
plt.suptitle("Multiple Linear Regression")

plt.show()
