import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
train = pd.read_csv('train.csv')
X=train['GarageArea']
Y=train['SalePrice']
Z=train[['GarageArea','SalePrice']]
print(Z)
plt.scatter(X, Y, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('Outliers')
plt.show()
Z=Z.as_matrix()
hull = ConvexHull(Z)
print(Z[hull.vertices])
plt.plot(train["GarageArea"], train["SalePrice"], 'ok')
plt.plot(Z[hull.vertices, 0], Z[hull.vertices,1], 'r--', lw = 2)
plt.plot(Z[hull.vertices, 0], Z[hull.vertices,1], 'ro', lw = 2)
plt.plot(np.delete(Z[:,0], hull.vertices), np.delete(Z[:,1], hull.vertices), 'ok')
plt.xlabel("GarageArea")
plt.ylabel("SalePrice")
plt.title("Figure 3 -  with no outliers")
plt.show()
