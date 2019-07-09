import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.colors import ListedColormap
from sklearn import metrics

y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
x = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 

x_ = PolynomialFeatures(degree=3, include_bias=False).fit_transform(x)

# Step 3: =Create a model and fit it
model = LinearRegression().fit(x_, y) 
# Step 5: Predict
y_estimated_round = []
y_pred = model.predict(x_) 
for i in y_pred:
    if i < 0.5:
        y_estimated_round.append(0)
    else:
        y_estimated_round.append(1)

r_sq = model.score(x_, y)
intercept, coefficients = model.intercept_, model.coef_
print('coefficient of determination:', r_sq)
print('intercept:', intercept)
print('coefficients:', coefficients, sep='\n')

print('Mean Absolute Error:(Polynomial Regression)', metrics.mean_absolute_error(y_pred, y))  
print('Mean Squared Error:', metrics.mean_squared_error(y_pred, y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred, y))) 


#In the dataset col7=age col5=BMI(BodyMassIndex) col8=outcome/diabetes
colors = ListedColormap(['#0000FF', '#FF0000'])

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.set_xlabel('Age')
ax2.set_ylabel('BMI')
ax2.set_zlabel('Diabetes 1/0')
ax2.scatter(x[:,0], x[:,1], y_pred, c=y, cmap=colors)
plt.suptitle("Polynomial Regression")

plt.show()

