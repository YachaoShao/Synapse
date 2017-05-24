# This script is used to analise the principle component of the synaptic data
# after that use kmeans to cluster the data.

# load library
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets,cluster
from sklearn.decomposition import PCA
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


# load data from excel and save it to csv file
parameter2D = pd.read_csv('parameter2D.csv', sep=',')

# set parameters for PCA, use PCA to analyze Parameter2D
n = len(parameter2D.columns)
pca = PCA(n_components=n, whiten=False)
new_parameter2D = pca.fit_transform(parameter2D)

# Get the figure of variance ratio
plt.bar(range(1, n+1), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.title('Principle component ratio')
plt.xlabel('principle component')
plt.ylabel('explained variance ratio')
plt.savefig('Synaptic_explained_variance_ratio.png')
plt.show()

# choose the first three components to cluster the synapses
kmeans_data = new_parameter2D[:, 0:3]
# set the number of clusters
kmeans = KMeans(n_clusters=3, random_state=111)
kmeans.fit(kmeans_data)
# plot and save the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(kmeans_data[:, 0], kmeans_data[:, 1], kmeans_data[:, 2], c=kmeans.labels_)
plt.savefig('K-means cluster results with using PCA')
plt.show()

# save the results
cluser_results = np.c_[kmeans.labels_.T, parameter2D]
np.savetxt('Parameter2D_cluster.csv', cluser_results, delimiter=',')








