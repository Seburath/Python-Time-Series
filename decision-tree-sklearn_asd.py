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
tree.plot_tree(clf.fit(array_x, array_y)
)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_pred, array_y))  

