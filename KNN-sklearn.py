# based on: https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import metrics

#On the dataset col7=age col5=BMI(BodyMassIndex) col8=outcome/diabetes
y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
X = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 

n_neighbors = 50 
h = .25  # step size in the mesh

# we create an instance of Neighbours Classifier and fit the data.
model = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
model.fit(X, y)

y_pred = model.predict(X)
print('Mean Absolute Error(KNN):', metrics.mean_absolute_error(y_pred, y))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_pred, y))) 

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#3333FF', '#FF3333'])
cmap_bold = ListedColormap(['#0000FF', '#FF0000'])


fig = plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
ax = fig.add_subplot(111)
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.scatter(X[:, 0], X[:, 1],  cmap=cmap_bold,c=y, label='asd', 
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("KNN 3-Class classification (k = %i, weights = 'distance')"
              % (n_neighbors))

plt.show()
