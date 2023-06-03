'''
Combines two feed supplies to create a new fs. 
Takes a slice/s from the source fs and overrides a slice/s in the destination fs. Then saves the resulting fs with a new fs_number.
For example if you wanted to examine the importance of feeding multiples (twins and tripplets) more because you know there are some tripplets
vs feeding them based soley on twins you could run scan 3 which generates a different fs for singles, twins and tripplets
and then update the twin scan 2 fs with the twin fs from scan 3.
'''
import pickle as pkl
import sys

import relativeFile

##setup
source_axis_slice = {}
dest_axis_slice = {}

#################
## User inputs  #
#################
#reads in as string so need to convert to int, the script path is the first value hence take the second
region = sys.argv[2]  #swv -nimbus instance
region = sys.argv[2] #gsm - google instance
genotype = sys.argv[3] #medium wool merino - nimbus instance 1 and google instance 1
genotype = sys.argv[3] #mat - nimbus instance 2 and google instance 2

tol = [1,3]
scan = [3] #scan 3 for all cases
meat_price = [1,2,3,4]
group='dams'
#change scan number to scan 4 in new fs.
for n_tol in tol:
    for n_scan in scan:
        for n_mp in meat_price:


            fs_source_number = str(region) + str(genotype) + str(n_tol) + str(n_scan) + str(n_mp)
            fs_dest_number = str(region) + str(genotype) + str(n_tol) + str(n_scan) + str(n_mp)
            fs_new_number = str(region) + str(genotype) + str(n_tol) + str(4) + str(n_mp)

            source_axis_slice[-12] = [3, 4, 1], [3, 4, 1], [3, 4, 1] #twin fs b1[3]
            dest_axis_slice[-12] = [4, 5, 1], [6, 8, 1], [10,11, 1] #tripplets 33, 32, 31 & 30




            #################
            ## calcs        #
            #################

            ##read in both fs
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_source_number}.pkl")
            with open(pkl_fs_path,
                      "rb") as f:
                fs_source = pkl.load(f)['fs'][group]
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_dest_number}.pkl")
            with open(pkl_fs_path,
                      "rb") as f:
                fs_dest = pkl.load(f)['fs'][group]

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
                    fs_dest[tuple(sl_dest)] = fs_source[tuple(sl_source)]*100
            fs_new = fs_dest


            ##save new fs
            pkl_fs_path = relativeFile.find(__file__, "../../pkl", 'pkl_fs{0}.pkl'.format(fs_new_number))
            with open(pkl_fs_path, "wb") as f:
                pkl.dump(fs_new, f)
