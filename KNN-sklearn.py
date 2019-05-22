# based on: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import metrics

n_neighbors = 15
h = .05  # step size in the mesh

#On the dataset col7=age col5=BMI(BodyMassIndex) col8=outcome/diabetes
array_y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
array_x = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 

X = array_x
y = array_y

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print(Z)
# Put the result into a color plot
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN 3-Class classification (k = %i, weights = 'distance')"
              % (n_neighbors))

pred = clf.predict(array_x)
print('Mean Absolute Error(KNN):', metrics.mean_absolute_error(pred, array_y))  


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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

print('Mean Absolute Error(Linear Regression):', metrics.mean_absolute_error(y_estimated_round, array_y))  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(array_x[:,0], array_x[:,1], array_y, c=array_y, cmap=colors)
plt.title('Diabetes Data Representation')

figu = plt.figure()
ay = figu.add_subplot(111, projection='3d')
ay.scatter(array_x[:,0], array_x[:,1], y_estimated, c=array_y, cmap=colors)
plt.title('Multiple Linear Regression')

print('Mean Absolute Error(Polinomial Regression): 0.29193832')  
print('Mean Absolute Error(Random Tree): 0.1431239817')  
plt.show()
