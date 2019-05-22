import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn import metrics  
from sklearn import tree
#%matplotlib inline

array_y = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[8]) 
array_x = np.loadtxt('diabetes.csv', delimiter=",", skiprows=1, usecols=[7, 5]) 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(array_x, array_y)

y_pred = clf.predict(array_x)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred, array_y))  

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters
n_classes = 3
plot_colors = "rb"
plot_step = 0.10

# Plot the decision boundary
plt.plot()

x_min, x_max = array_x[:, 0].min() - 1, array_x[:, 0].max() + 1
y_min, y_max = array_x[:, 1].min() - 1, array_x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    idx = np.where(array_y == i)
    plt.scatter(array_x[idx, 0], array_x[idx, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.show()
