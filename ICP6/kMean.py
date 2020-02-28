import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import seaborn as sns

#read dataset
dataset = pd.read_csv('CC.csv')

# numeric_features = dataset.select_dtypes(include=[np.number])
# corr = numeric_features.corr()
#
# print (corr['TENURE'].sort_values(ascending=False)[:4], '\n')
# dataset.apply(lambda x: x.fillna(x.mean()),axis=0)

##Null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
# print(nulls)

##handling the missing value
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()
# print(sum(data.isnull().sum() != 0))


# #assign
x_train = data.iloc[:,[2,-5,-6]]

#Preprocessing the data
scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
X_scaled_array = scaler.transform(x_train)
X_scaled = pd.DataFrame(X_scaled_array, columns = x_train.columns)

#
# from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x_train)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(x_train, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
#
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

# from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,12):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
#
plt.plot(range(1,11),wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

pca = PCA(2)
x_pca = pca.fit_transform(X_scaled)
df2 = pd.DataFrame(data=x_pca)
# finaldf = pd.concat([df2,dataset[['TENURE']]],axis=1)
# print(finaldf)

from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(df2)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(df2, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))
plt.plot(range(1, 4), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()