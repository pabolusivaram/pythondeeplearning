from sklearn.decomposition import PCA

# Make an instance of the Model

import pandas as pd

import numpy as np

train = pd.read_csv('CC.csv')

desired_width=320



pd.set_option('display.width', desired_width)



# pd.set_option(linewidth=300)



pd.set_option('display.max_columns',20)

import matplotlib.pyplot as plt



data = train.select_dtypes(include=[np.number]).interpolate().fillna(train.select_dtypes(include=[np.number]).interpolate().mean(axis=0))

from sklearn.model_selection import train_test_split

X_train = data







from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(X_train)

X_train=scaler.transform(X_train)

#X_scaled=pd.DataFrame(X_scaled_array, columns =X_train.columns)



pca = PCA(.95)

pca.fit(X_train)

X_train = pca.transform(X_train)



print(X_train)

print(X_train.shape)