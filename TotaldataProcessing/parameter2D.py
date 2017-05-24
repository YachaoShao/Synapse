# import libraries
import numpy as np
import xlrd
import csv
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
# load data from excel and save it to csv file
with xlrd.open_workbook('Synapticparameters 2D.xlsx') as wb:
    sh = wb.sheet_by_index(0)  # or wb.sheet_by_name('name_of_the_sheet_here')
    with open('parameter2D.csv', 'wb') as f:
        c = csv.writer(f)
        for r in range(sh.nrows):
            c.writerow(sh.row_values(r))

parameter2D = pd.read_csv('parameter2D.csv', sep=',')

print parameter2D.shape
print parameter2D.describe()
# pre_synapse_data_new = np.delete(pre_synapse_data, [0, 1], axis=1)


# print pre_synapse_data_new.shape
# Hierarchical classifying
# Computing distance
disMat = sch.distance.pdist(parameter2D, 'euclidean')
# Cluster tree
Z = sch.linkage(disMat, method='average')
# save image plot_dendrogram.png
P = sch.dendrogram(Z)

plt.title('Hierarchical Clustering Dendrogram in Parameter2D')
plt.xlabel('sample index')
plt.ylabel('distance')
plt.savefig('parameter2D.png')
plt.show()

# results
cluster = sch.fcluster(Z, t=1.152)
print "Original cluster by hierarchy clustering:\n", cluster

# add cluster label to the original data and save it to csv file
parameter2D_cluster = np.c_[cluster.T, parameter2D]
np.savetxt('parameter2D_cluster.csv', parameter2D_cluster, delimiter=',')
