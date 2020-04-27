import os
import csv
import numpy as np
import pdb

src = 'data/meta_ch2x2_uncertainty_sim.csv'
dest = 'data/dataset1.csv'

grp_interval = 100

dset_arr = np.genfromtxt(src,delimiter=',')

counter = 0

groups = []
for i in range(len(dset_arr)):
	if i % grp_interval == 0:
		counter += 1

	groups.append(counter)

groups = np.array(groups)

# add the group column at the end
new_dset = np.hstack((dset_arr,groups.reshape(-1,1)))
np.savetxt(dest, new_dset, delimiter=',')
