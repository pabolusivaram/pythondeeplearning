from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# You can add the parameter data_home to wherever to where you want to download your data
dataset = pd.read_csv('CC.csv')
# numeric_features = dataset.select_dtypes(include=[np.number])
# corr = numeric_features.corr()
# print (corr['TENURE'].sort_values(ascending=False)[:4], '\n')

##Null values
nulls = pd.DataFrame(dataset.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
# print(nulls)

##handling the missing value
data = dataset.select_dtypes(include=[np.number]).interpolate().dropna()

x_train = dataset.iloc[:,[2,-5,-6]]
# y = dataset.iloc[:,-1]

scaler = StandardScaler()
scaler.fit(x_train)
# Apply transform.
x_scaler = scaler.transform(x_train)
X_scaled = pd.DataFrame(x_scaler, columns = x_train.columns)


from sklearn import metrics
wcss = []
# ##elbow method to know the number of clusters
for i in range(2,5):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X_scaled)
   # print(kmeans.inertia_,'-------------------')
    wcss.append(kmeans.inertia_)
    score = silhouette_score(X_scaled, kmeans.labels_, metric='euclidean')
    print("For n_clusters = {}, silhouette score is {})".format(i, score))

plt.plot(range(1,4),wcss)
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