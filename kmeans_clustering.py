# This file is used to cluster  the data with using kmeans method.
# load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import cluster
import scipy.cluster.hierarchy as sch

parameter2D = pd.read_csv('parameter2D.csv', sep=',')

# set parameters for PCA and make the principle component analysis of Parameter2D
n = len(parameter2D.columns)
pca = PCA(n_components=n, whiten=False)
new_parameter2D = pca.fit_transform(parameter2D)
kmeans_data = new_parameter2D[:, 0:3]

k_means = cluster.KMeans(n_clusters=3)


kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(kmeans_data)
cluster_result = kmeans.labels_
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.figure('K-means with 3 clusters')
# ax.scatter(kmeans_data[:, 0], kmeans_data[:, 1], kmeans_data[:, 2], c=kmeans.labels_)
plt.scatter(kmeans_data[:, 0], kmeans_data[:, 1], c=kmeans.labels_)
plt.savefig('K-means cluster results with using PCA')
plt.show()

# Hierarchical classifying
# Computing distance
disMat = sch.distance.pdist(new_parameter2D, 'euclidean')
# Cluter tree
Z = sch.linkage(disMat, method='average')
# save image plot_dendrogram.png
P = sch.dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram of Synapses')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.savefig('plot_dendrogram_sklearn_pca.png')
plt.show()









