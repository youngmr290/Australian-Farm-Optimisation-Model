
'''
Combines two feed supplies to create a new fs. 
Takes a slice/s from the source fs and overrides a slice/s in the destination fs. Then saves the resulting fs with a new fs_number.
For example if you wanted to examine the importance of feeding multiples (twins and triplets) more because you know there are some triplets
vs feeding them based solely on twins you could run a fs_opt for scan 3 which generates a different fs for singles, twins and triplets
and then update the twin scan 2 fs with the twin fs from scan 3.
Different conventions have been used to define the pkl_fs.
The current convention (Feb 26) for naming the feed supply is that 12 digits are used to describe the scenario
    digit 1   - feeding equations (AFS-CSIRO, GFS-hybrid for GEPEP, RFS-Hutton)
          2   - number of offspring sale opportunities
          3   - Region
          4   - Breed
          5&6 - 2 digits to describe WBE (EVG tweak & the age adjusted)
          7   - TOL
          8   - Scan level
          9   - Meat price pointer
          10  - Crop grazing inc
          11 - Pasture % pointer
          12 - Age EL joined

The convention required to carry out the example above of switching a twin pattern from a scan 3 run requires including
  digit 1 - scan number
        2 - empty feed supply
        3 - singles feed supply
        4 - twins feed supply
        5 - triplets feed supply
The value used to describe the feed supply for each class of sheep is determined from the scanning level that
generated the feed supply and the class of sheep that was copied.
   Scan 0 - All sheep 1
   Scan 1 - Empty 2
            Pregnant 3
   Scan 2 - Empty 4
            Single 5
            Multiples 6
   Scan 3 - Empty 7
            Singles 8
            Twins 9
            Triplets A
'''


import pickle as pkl
import sys
import copy
import ast
import itertools
import numpy as np

import relativeFile


##setup
source_axis_slice = {}
dest_axis_slice = {}

s_pos = -17
t_pos = -16
p_pos = -15
a1_pos = -14
e1_pos = -13
b1_pos = -12
n_pos = -11
w_pos = -10
z_pos = -9
i_pos = -8
d_pos = -7
a0_pos = -6
e0_pos = -5
b0_pos = -4
x_pos = -3
y_pos = -2
g_pos = -1


#################
## User inputs  #
#################

#Read the group (dams or offs) as an arg (read in as a string).
group = sys.argv[1]  #the script path is the first value (0) hence take the second (1)

## define the levels for the destination of each of the 11 characteristics that define the feed supply.
### Note: the source is 'hardwired' in the loop below and can be any of the 11 characteristics
### alter these to suit the pkl_fs to be changed. Use lists because levels are looped
### arguments can be passed when the module is called if this is useful. variable = sys.argv[x]
equations = [1]
off_sales = [4]
region = ast.literal_eval(sys.argv[2])   #the region(s) arg needs to be a list, literal_eval converts it from string to a list
breed = [1, 2, 4]
wbe = [0]  #note this is converted to a 2 digit string later
tol = [1, 2, 3]
scan = [2]
meat_price = [1, 2, 3, 4]
crop_grazing = [2]
past_area = [0]
age_joined = [1, 2, 3, 4, 5, 6]

## Define the slices of the feedsupply that are to be copied
### Note: Only one axis can be changed per run of the module, but multiple slices of that axis can be copied
### The slices to copy are specified as tuples [start, stop, step].
### There needs to be the same number of tuples for source and destination

# ## the code below overwrites the triplet fs in the destination with the twin feedsupply from the source
# ###the new feed supply is if scanned for triplets but didn't change triplet nutrition.
# ###source is twins b1[3] and destination is LSLN 33 b1[4], 32 b1[6], 31 b1[7] and 30 b1[10]
# axis = b1_pos
# dest_axis_slice[axis]   = [(4, 5, 1), (6, 8, 1), (10, 11, 1)]  # must be unique slices.
# source_axis_slice[axis] = [(3, 4, 1), (3, 4, 1), (3, 4, 1)]  # same number of slices as destination (a slice can be repeated and copied to multiple destination slices).

# the code below overwrites the mated fs of ewe lambs in the destination with mated fs from the source
## the source is the fs for mated ewe lambs and the destination is the fs when ewe lambs are not mated (age_join==6)
###source & destination is mated b1[1:] for the periods up to 2 tooth joining (0 to 100ish)
copy_axis1 = b1_pos
copy_axis2 = p_pos
dest_axis_slice[copy_axis1]   = [(1, None, 1)]  # If not unique the later of the matching source slice will prevail. Can add more tuples to the list
source_axis_slice[copy_axis1] = [(1, None, 1)]  # same number of slices as destination (a slice can be repeated and copied to multiple destination slices).

# The date of 2 tooth joining and hence the slices for the p axis (copy_axis_2) vary with region and tol.
p_slices = [[100,108,114],[102,107,112],[100,105,110]]

count = 0

#Loop through the 11 characteristics (defined above) to define the source, destination and new feedsupply
param_lists = [equations, off_sales, region, breed, wbe, tol, scan, meat_price, crop_grazing, past_area, age_joined]
for n_eqn, n_os, n_region, n_breed, n_wbe, n_tol, n_scan, n_mp, n_cg, n_past, n_age in itertools.product(*param_lists):
    p_slice = p_slices[n_region-1][n_tol-1]
    dest_axis_slice[copy_axis2] = [(0, p_slice, 1)]
    source_axis_slice[copy_axis2] = [(0, p_slice, 1)]

    ## 'Hard wire' the difference for the name of the destination feedsupply in this code.
    ## the code below uses age_joined 6 as the destination
    fs_dest_number = (str(n_eqn) + str(n_os) + str(n_region) + str(n_breed)
                      + f'{n_wbe:02d}' + str(n_tol) + str(n_scan) + str(n_mp)
                      + str(n_cg) + str(n_past) + str(6))
    fs_source_number = (str(n_eqn) + str(n_os) + str(n_region) + str(n_breed)
                        + f'{n_wbe:02d}' + str(n_tol) + str(n_scan) + str(n_mp)
                        + str(n_cg) + str(n_past) + str(n_age))
    ## The new name is similar to the source name with '2' appended
    fs_new_number = (str(n_eqn) + str(n_os) + str(n_region) + str(n_breed)
                     + f'{n_wbe:02d}' + str(n_tol) + str(n_scan) + str(n_mp)
                     + str(n_cg) + str(n_past) + str(n_age)) + str(2)

    #################
    ## calcs        #
    #################

    ##read in both fs and retain the feedsupply for the target group
    pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_source_number}.pkl")
    with open(pkl_fs_path, "rb") as f:
        fs_source = pkl.load(f)['fs'][group]
    pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_dest_number}.pkl")
    with open(pkl_fs_path, "rb") as f:
        full_dest = pkl.load(f)
        fs_dest = full_dest['fs'][group]
        # create a shallow copy of the full dictionary, then deep copy only the target array
        fs_new = copy.copy(full_dest)
        fs_new['fs'] = copy.copy(full_dest['fs'])
        fs_dest = np.copy(fs_dest)  # work on a copy of the array to modify

    ##manipulate fs to create new fs
    axes = list(source_axis_slice.keys())
    if axes:
        num_copies = len(source_axis_slice[axes[0]])
        # All axes must have the same number of slice tuples otherwise this will throw and error
        for i in range(num_copies):
            sl_dest = [slice(None)] * fs_dest.ndim
            sl_source = [slice(None)] * fs_source.ndim
            for axis in axes:
                start, stop, step = dest_axis_slice[axis][i]
                sl_dest[axis] = slice(start, stop, step)
                start, stop, step = source_axis_slice[axis][i]
                sl_source[axis] = slice(start, stop, step)
            ###update
            fs_dest[tuple(sl_dest)] = fs_source[tuple(sl_source)]
    fs_new['fs'][group] = fs_dest

    ##save new fs
    pkl_fs_path = relativeFile.find(__file__, "../../pkl", 'pkl_fs{0}.pkl'.format(fs_new_number))
    with open(pkl_fs_path, "wb") as f:
        pkl.dump(fs_new, f)
    count = count + 1

print(f'{count} new pkl_fs created.')

