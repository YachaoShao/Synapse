# import libraries
import numpy as np
import xlrd
import csv
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
# load data from excel and save it to csv file
with xlrd.open_workbook('pre_synapse.xlsx') as wb:
    sh = wb.sheet_by_index(0)  # or wb.sheet_by_name('name_of_the_sheet_here')
    with open('pre_synapse_data_new.csv', 'wb') as f:
        c = csv.writer(f)
        for r in range(sh.nrows):
            c.writerow(sh.row_values(r))

pre_synapse_data = pd.read_csv('pre_synapse_data_new.csv', sep=',')

print pre_synapse_data.shape
print pre_synapse_data.describe()
# pre_synapse_data_new = np.delete(pre_synapse_data, [0, 1], axis=1)


# print pre_synapse_data_new.shape
# Hierarchical classifying
# Computing distance
disMat = sch.distance.pdist(pre_synapse_data, 'euclidean')
# Cluster tree
Z = sch.linkage(disMat, method='average')
# save image plot_dendrogram.png
P = sch.dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram of Synapses')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.savefig('plot_pre_synapse_cluster.png')
# plt.show()

# results
cluster = sch.fcluster(Z, t=1.154)
print "Original cluster by hierarchy clustering:\n", cluster

# add cluster label to the original data and save it to csv file
synapse_data_cluster = np.c_[cluster.T, pre_synapse_data]
np.savetxt('pre_synapse_cluster.csv', synapse_data_cluster, delimiter=',')