
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


def slices_to_str(array_name, slices):
    """Convert a list of slice objects into a nice indexing string like arr1[2:5,1:4,:]"""

    def fmt(s):
        if s == slice(None):  # the most common case
            return ':'

        start = s.start if s.start is not None else ''
        stop = s.stop if s.stop is not None else ''
        step = s.step if s.step is not None else 1

        if step == 1:  # step=1 is the default → omit it
            if start == 0 and stop == '':
                return ':'
            elif start == 0:
                return f':{stop}'
            elif stop == '':
                return f'{start}:'
            else:
                return f'{start}:{stop}'
        else:  # explicit step
            start_str = str(start) if start != 0 else ''
            return f'{start_str}:{stop}:{step}'.strip(':') or ':'

    parts = [fmt(sl) for sl in slices]
    return f"{array_name}[{','.join(parts)}]"


##setup
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

## define the levels for the destination of each of the 11 characteristics that define the feed supply.
### Note: the source is 'hardwired' in the loop below and can be any of the 11 characteristics
### alter these to suit the pkl_fs to be changed. Use lists because levels are looped
### arguments can be passed when the module is called if this is useful. variable = sys.argv[x]
equations = [1]
off_sales = [4]
region = ast.literal_eval(sys.argv[1])   #the region(s) arg needs to be a list, literal_eval converts it from string to a list
breed = [1, 2, 4]
wbe = [0]  #note this is converted to a 2 digit string later
tol = [1, 2, 3]
scan = [2]
meat_price = [1, 2, 3, 4]
crop_grazing = [2]
past_area = [0]
age_joined = [1, 2, 3, 4, 5, 6]

# ─────────────────────────────────────────────────────────────
#   Configuration per group, defines the slices of the feedsupply that are to be copied.
#   Blocks to be copied can be defined by multiple axes and multiple slices of the axes can be copied
#   Multiple collections of axes and slices can be specified by appending numbers to the group name in GROUP_CONFIG
#   Multiple collections are required if the axes to copy differ
#   The slices of each axis to copy are specified as tuples (start, stop, step).
#   Can provide a list of tuples, but there needs to be the same number of tuples for source and destination
#   The slices can be defined to vary depending on the feedsupply eg vary with region and tol. This is controlled using lambda
#   The number of slices specified for the source and destination need to be broadcastable.
# ─────────────────────────────────────────────────────────────

# the config below copies from source to destination:
# 1. the not mated fs of ewe lamb dams b[0:1] in all p[0:]
# 2. mated dams b[1:] in periods after 100ish, with the period varying with region & tol based on p_slices.
# 3. offspring for 2 tooths and adults d[1:] without specifying p.
#
# The date of 2 tooth joining and hence the slices for the p axis (second axis) varies with region and tol.
#Note: The pointer to the array is region and TOL value, not the position in the list. i.e. Region 4 is the 4th slice

GROUP_CONFIG = {
    "dams1": {
        "axes": [b1_pos, p_pos],
        "slice_defs": {
            b1_pos: {
                "dest": [(0, 1, 1), (1, None, 1)],
                "source": [(0, 1, 1), (1, None, 1)],
            },
            p_pos: {
                "dest": lambda region, tol: [(0, None, 1), (p_slices[region-1][tol-1], None, 1)],
                "source": lambda region, tol: [(0, None, 1), (p_slices[region-1][tol-1], None, 1)],
            }
        },
        "p_slices": [[100, 108, 114], [102, 107, 112], [100, 105, 110], [100, 105, 110]],
    },
    "offs1": {
        "axes": [d_pos],
        "slice_defs": {
            d_pos: {
                "dest": [(1, None, 1)],
                "source": [(1, None, 1)],
            }
        },
        # no p_slices needed
    }
}

# ─────────────────────────────────────────────────────────────────────────
#   Main logic – one loop for both groups & only one read/write per pkl_fs
# ─────────────────────────────────────────────────────────────────────────

if not GROUP_CONFIG:
    print(f"No groups defined in GROUP_CONFIG.")
    sys.exit(1)

count = 0

param_lists = [equations, off_sales, region, breed, wbe, tol, scan, meat_price, crop_grazing, past_area, age_joined]

for params in itertools.product(*param_lists):
    n_eqn, n_os, n_region, n_breed, n_wbe, n_tol, n_scan, n_mp, n_cg, n_past, n_age = params

    ## Build pkl_fs numbers. Change destination and/or source
    ### 'Hard wire' the difference for the name of the destination feedsupply in this code.
    ### the code below uses age_joined 6 as the source and appends '2' to create the new name
    fs_dest_number = (str(n_eqn) + str(n_os) + str(n_region) + str(n_breed)
                      + f'{n_wbe:02d}' + str(n_tol) + str(n_scan) + str(n_mp)
                      + str(n_cg) + str(n_past) + str(n_age))
    fs_source_number = (str(n_eqn) + str(n_os) + str(n_region) + str(n_breed)
                        + f'{n_wbe:02d}' + str(n_tol) + str(n_scan) + str(n_mp)
                        + str(n_cg) + str(n_past) + str(6))
    fs_new_number = fs_dest_number + str(2)

    # ── Load files ─────────────────────────────────────────────────────────────────────────────────────
    src_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_source_number}.pkl")
    with open(src_path, "rb") as f:
        full_source = pkl.load(f)

    dest_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_dest_number}.pkl")
    with open(dest_path, "rb") as f:
        full_dest = pkl.load(f)

    #create a copy to work on
    full_new = copy.deepcopy(full_dest)

    # ── Process each group using the loaded data ────────────────────────
    for group_name, config in GROUP_CONFIG.items():
        group = group_name[:4]   #remove any identifier from end of the name. The identifier allows multiple collections of axes per group
        fs_new = full_new['fs'][group]
        fs_source = full_source['fs'][group]
        axes_to_copy = config["axes"]
        slice_defs = config["slice_defs"]
        p_slices = config.get("p_slices")  # may be None for some groups

        num_operations = None
        axis_slices_dest = {}
        axis_slices_src  = {}

        for axis in axes_to_copy:
            spec = slice_defs[axis]

            if callable(spec["dest"]):  # dynamic (e.g. p_pos in dams)
                dest_list = spec["dest"](n_region, n_tol)
                src_list  = spec["source"](n_region, n_tol)
            else:
                dest_list = spec["dest"]
                src_list  = spec["source"]

            axis_slices_dest[axis] = dest_list
            axis_slices_src[axis]  = src_list

        if num_operations is None:
            num_operations = len(dest_list)
        if len(dest_list) != num_operations or len(src_list) != num_operations:
            raise ValueError(f"Inconsistent number of slices for axis {axis} in {group_name}")

        # ── Apply all copy operations ─────────────────────────────────────────────
        for i in range(num_operations):
            sl_dest = [slice(None)] * fs_new.ndim
            sl_source = [slice(None)] * fs_source.ndim

            for axis in axes_to_copy:
                sl_dest[axis] = slice(*axis_slices_dest[axis][i])
                sl_source[axis] = slice(*axis_slices_src[axis][i])

            fs_new[tuple(sl_dest)] = fs_source[tuple(sl_source)]

            # Optional progress update
            print(f"{count} {group_name.upper()} Copied  pkl_fs{fs_source_number} "
                  f"{slices_to_str('fs_source', sl_source)}  →  "
                  f"pkl_fs{fs_dest_number} {slices_to_str('fs_dest', sl_dest)}")

    # ── Save ─────────────────────────────────────────────────────────────────────────────────────────
    new_path = relativeFile.find(__file__, "../../pkl", f"pkl_fs{fs_new_number}.pkl")
    with open(new_path, "wb") as f:
        pkl.dump(full_new, f)

    count += 1

print(f"{count} new pkl_fs created.")

