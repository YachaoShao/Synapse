### import libraries
import scipy
import pandas as pd
import scipy.cluster.hierarchy as sch
import numpy as np
import matplotlib.pylab as plt


# load data from excel
synapse_data = pd.read_csv('synape_matrix.csv', sep=',')
# print synapse_data.shape
# print synapse_data.describe()
# Hierarchical classifying
# Computing distance
disMat = sch.distance.pdist(synapse_data,'euclidean')
# Cluter tree
Z = sch.linkage(disMat, method='average')
c, coph_dist = sch.cophenet(Z, disMat)
print c
# save image plot_dendrogram.png
P = sch.dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram of Synapses')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.savefig('plot_dendrogram.png')
plt.show()

# results
cluster = sch.fcluster(Z, t=1)
print "Original cluster by hierarchy clustering:\n", cluster
