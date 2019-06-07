import numpy as np
import matplotlib.pyplot as plt
import pandas
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

x = np.linspace(0, 20, 20)
X, Y = np.meshgrid(x, x)

np.random.seed(1)

Z = -5 + 3*X - 3*Y + 8 * np.random.normal(size=X.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm,
                       rstride=1, cstride=1)
ax.view_init(20, -120)
ax.set_xlabel('Tiempo')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Convert the data into a Pandas DataFrame to use the formulas framework
# in statsmodels
# First we need to flatten the data: it's 2D layout is not relevent.

X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

data = pandas.DataFrame({'x': X, 'y': Y, 'z': Z})

model = ols("z ~ x + y", data).fit()

print(model.summary())
print("\nParameter estimates:")
print(model._results.params)

anova_results = anova_lm(model)

print('\nANOVA results')
print(anova_results)

plt.show()
