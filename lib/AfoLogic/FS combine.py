'''
Combines two feed supplies to create a new fs. 
Takes a slice/s from the source fs and overrides a slice/s in the destination fs. Then saves the resulting fs with a new fs_number.
For example if you wanted to examine the importance of feeding multiples (twins and tripplets) more because you know there are some tripplets
vs feeding them based soley on twins you could run scan 3 which generates a different fs for singles, twins and tripplets
and then update the twin scan 2 fs with the twin fs from scan 3.
'''
import pickle as pkl

import relativeFile

##setup
source_axis_slice = {}
dest_axis_slice = {}

#################
## User inputs  #
#################
group='dams'
fs_source_number = 0
fs_dest_number = 0
fs_new_number = 00

source_axis_slice[-12] = [3, 4, 1] #twin fs b1[3]
dest_axis_slice[-12] = [4, 5, 1] #tripplets 33
dest_axis_slice[-12] = [6, 8, 1] #tripplets 32, 31
dest_axis_slice[-12] = [10,11, 1] #tripplets 30




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
###source
sl_source = [slice(None)] * fs_source.ndim
for axis, slc in source_axis_slice.items():
    start = slc[0]
    stop = slc[1]
    step = slc[2]
    sl_source[axis] = slice(start, stop, step)
###dest
sl_dest = [slice(None)] * fs_dest.ndim
for axis, slc in dest_axis_slice.items():
    start = slc[0]
    stop = slc[1]
    step = slc[2]
    sl_dest[axis] = slice(start, stop, step)
###update
fs_dest[tuple(sl_dest)] = fs_source[tuple(sl_source)]*100
fs_new = fs_source


##save new fs
pkl_fs_path = relativeFile.find(__file__, "../../pkl", 'pkl_fs{0}.pkl'.format(fs_new_number))
with open(pkl_fs_path, "wb") as f:
    pkl.dump(fs_new, f)
