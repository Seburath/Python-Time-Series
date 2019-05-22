import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

array_y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
array_x = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 


x, y = array_x, array_y 

# Step 2b: Transform input data
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

# Step 3: Create a model and fit it
model = LinearRegression().fit(x_, y) 
# Step 5: Predict
y_estimated_round = []
y_pred = model.predict(x_) 
for i in y_pred:
    if i < 0.5:
        y_estimated_round.append(0)
    else:
        y_estimated_round.append(1)

print('Mean Absolute Error:(Logistic Regression)', metrics.mean_absolute_error(y_estimated_round, array_y))  


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


#In the dataset col7=age col5=BMI(BodyMassIndex) col8=outcome/diabetes
colors = ListedColormap(['#0000FF', '#FF0000'])


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_x[:,0], array_x[:,1], array_y, c=array_y, cmap=colors)

figu = plt.figure()
ay = figu.add_subplot(111, projection='3d')
ay.scatter(array_x[:,0], array_x[:,1], y_pred, c=array_y, cmap=colors)

plt.show()

