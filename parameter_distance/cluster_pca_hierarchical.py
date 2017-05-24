# import libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA

# load data from csv file
synapse_data = pd.read_csv('synape_matrix.csv', sep=',')
print synapse_data.shape

# pca
n = len(synapse_data)
print n
pca = PCA(n_components='mle', whiten=False)
synapse_data_new = pca.fit_transform(synapse_data)
print synapse_data_new.shape

plt.bar(range(1, 9), pca.explained_variance_ratio_, alpha=0.5, align='center', label='individual explained variance')
plt.title('Principle component ratio')
plt.xlabel('principle component')
plt.ylabel('explained variance ratio')
plt.savefig('plot_explained_variance_ratio.png')
plt.show()

# Hierarchical classifying
# Computing distance
disMat = sch.distance.pdist(synapse_data_new,'euclidean')
# Cluter tree
Z = sch.linkage(disMat,method='average')
# set the cluster number k
k = 6
# calculate color threshold
cl = Z[-(k-1), 2]
# save image plot_dendrogram.png
P = sch.dendrogram(Z,color_threshold=cl)
plt.title('Hierarchical Clustering Dendrogram of Synapses')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.savefig('plot_dendrogram_sklearn_pca.png')
plt.show()

# results
cluster = sch.fcluster(Z, t=1.14)

# add cluster label to the original data and save it to csv file
synapse_data_cluster = np.c_[cluster.T, synapse_data]
np.savetxt('synapse_cluster.csv', synapse_data_cluster, delimiter=',')
# print "Original cluster by hierarchy clustering:\n",cluster












