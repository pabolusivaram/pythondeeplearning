import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train = pd.read_csv('winequality-red.csv')

train.quality.describe()
print ("Skew is:", train.quality.skew())
plt.hist(train.quality, color='blue')
plt.show()


#Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
np.fill_diagonal(corr.values, -2)
print (corr['quality'].sort_values(ascending=False)[:3], '\n')
corr_res= corr['quality'].sort_values(ascending=False)[:4]
#for i in corr_res:
 #   train.drop(train[i], axis=1)
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'

data = train.select_dtypes(include=[np.number]).interpolate().dropna()

y = np.log(train.quality)
X = data.drop(['quality'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)
print ("R^2 is: \n", model.score(X_test, y_test))
print(model)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Quality')
plt.ylabel('Actual Quality')
plt.title('Linear Regression Model')
plt.show() 