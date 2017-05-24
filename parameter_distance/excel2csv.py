# load library
import numpy as np
import xlrd
import csv
import pandas as pd

with xlrd.open_workbook('synapdata.xlsx') as wb:
    sh = wb.sheet_by_index(0)  # or wb.sheet_by_name('name_of_the_sheet_here')
    with open('synape_matrix.csv', 'wb') as f:
        c = csv.writer(f)
        for r in range(sh.nrows):
            c.writerow(sh.row_values(r))

# read data from excel and write in the matrix
# book = xlrd.open_workbook("Mappe1.xlsx")
# table = book.sheet_by_index(0)


# synape_matrix = np.zeros((table.nrows-3,table.ncols-2))

# for row in range(3,table.nrows):
#     for col in range(1,table.ncols-1):
#         synape_matrix[row-3][col-1] = int(table.cell_value(row,col))
#
#save to txt

# synape_data= synape_matrix.T

# np.savetxt('synape_data.csv', synape_data, delimiter=',')