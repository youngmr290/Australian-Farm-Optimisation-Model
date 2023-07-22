'''
Combines two feed supplies to create a new fs. 
Takes a slice/s from the source fs and overrides a slice/s in the destination fs. Then saves the resulting fs with a new fs_number.
For example if you wanted to examine the importance of feeding multiples (twins and triplets) more because you know there are some triplets
vs feeding them based solely on twins you could run scan 3 which generates a different fs for singles, twins and triplets
and then update the twin scan 2 fs with the twin fs from scan 3.
The convention for naming the feed supply is that 5 digits are used to describe the scanning and feed supply for dams with different litter size
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

import relativeFile

##setup
source_axis_slice = {}
dest_axis_slice = {}

#################
## User inputs  #
#################
#reads in as string so need to convert to int, the script path is the first value hence take the second
region = sys.argv[1]  # 1 - swv (nimbus instances) 2 - gsw (google instances)
genotype = sys.argv[2] # 2 - medium wool merino (nimbus and google instance 1)  4 - maternal (nimbus and google instance 2)

## the code below uses a scan3 run and overwrites the triplet fs with the twin feedsupply and calls it a scan 2 feedsupply
###the new feed supply is the management if scanned for multiples and no allowance was made for the triplets in the mob.

if region == 1: #SWV
    tol = [2,3]   #SWV is winter and spring lambing
else:
    tol = [1,3]   #GSW is autumn and spring lambing
scan = [3] #scan 3 for all cases
meat_price = [1,2,3,4]
group='dams'

source_axis_slice[-12] = [3, 4, 1], [3, 4, 1], [3, 4, 1]  # can be repeated. twin fs b1[3]
dest_axis_slice[-12] = [4, 5, 1], [6, 8, 1], [10, 11, 1]  # must be unique slices. triplets 33 [4], 32 [6], 31 [7] & 30 [10]

#Loop through tol, scan & meat_price scenarios that are defined above
for n_tol in tol:
    for n_scan in scan:
        for n_mp in meat_price:

            # fs_pkl number is defined by region, genotype, TOL, Scan & meat price
            fs_source_number = str(region) + str(genotype) + str(n_tol) + str(n_mp) + str(n_scan) + '0000'
            fs_dest_number = str(region) + str(genotype) + str(n_tol) + str(n_mp) + str(n_scan) + '0000'
            #Naming is based on changing scan number to scan 2 in new fs. Code for the empty, single & twins are their
            ## respective values from a scan 3 trial (7, 8 & 9). Triplets are based on the scan3 twin feed supply (9).
            fs_new_number = str(region) + str(genotype) + str(n_tol) + str(n_mp) + str(27899)

            #################
            ## calcs        #
            #################

            ##read in both fs and retain the feeds supply for the target group
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_source_number}.pkl")
            with open(pkl_fs_path,
                      "rb") as f:
                fs_source = pkl.load(f)['fs'][group]
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_dest_number}.pkl")
            with open(pkl_fs_path,
                      "rb") as f:
                fs_dest = pkl.load(f)
                # create a copy of the full dictionary prior to retaining only the target group
                fs_new = copy.deepcopy(fs_dest)
                fs_dest = fs_dest['fs'][group]

            ##manipulate fs to create new fs
            for axis in source_axis_slice:
                for slc_source, slice_dest in zip(source_axis_slice[axis], dest_axis_slice[axis]):
                    sl_dest = [slice(None)] * fs_dest.ndim
                    sl_source = [slice(None)] * fs_source.ndim
                    start = slc_source[0]
                    stop = slc_source[1]
                    step = slc_source[2]
                    sl_source[axis] = slice(start, stop, step)
                    start = slice_dest[0]
                    stop = slice_dest[1]
                    step = slice_dest[2]
                    sl_dest[axis] = slice(start, stop, step)
                    ###update
                    fs_dest[tuple(sl_dest)] = fs_source[tuple(sl_source)]
            fs_new['fs'][group] = fs_dest


            ##save new fs
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", 'pkl_fs{0}.pkl'.format(fs_new_number))
            with open(pkl_fs_path, "wb") as f:
                pkl.dump(fs_new, f)
