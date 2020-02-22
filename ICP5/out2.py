import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('train.csv')
gaurage_area = train["GarageArea"]
sale_price = train["SalePrice"]
plt.scatter(gaurage_area, sale_price, alpha=.75,
            color='b')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

outliers = train['GarageArea']<=1200
train = train[outliers]
outliers = train['GarageArea']!=0
train = train[outliers]

gaurage_area = train["GarageArea"]
sale_price = train["SalePrice"]
plt.scatter(gaurage_area, sale_price, alpha=.75,
            color='b')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

train.SalePrice.describe()
