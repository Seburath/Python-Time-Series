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
colors = ListedColormap(['#0000FF', '#FF0000'])

model = LinearRegression()
model.fit(array_x, array_y)

y_estimated = model.predict(array_x)
y_estimated_round = []
for i in y_estimated:
    if i < 0.5:
        y_estimated_round.append(0)
    else:
        y_estimated_round.append(1)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_estimated_round, array_y))  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_x[:,0], array_x[:,1], array_y, c=array_y, cmap=colors)

figu = plt.figure()
ay = figu.add_subplot(111, projection='3d')
ay.scatter(array_x[:,0], array_x[:,1], y_estimated, c=array_y, cmap=colors)

plt.show()
