import numpy as np
from timeit import default_timer as timer
import sys
import os

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")

#sets the path to the root directory so the relative imports in the other files work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lib.RawVersion import LoadExcelInputs as dxl
from lib.RawVersion import LoadExp as exp
from lib.RawVersion import RawVersionExtras as rve
from lib.AfoLogic import StructuralInputs as sinp
from lib.AfoLogic import PropertyInputs as pinp
from lib.AfoLogic import UniversalInputs as uinp
from lib.AfoLogic import Periods as per
from lib.AfoLogic import Functions as fun
from lib.AfoLogic import SeasonalFunctions as zfun
from lib.AfoLogic import Sensitivity as sen

from lib.AfoLogic import Pasture as pas
from lib.AfoLogic import SupFeed as sup
from lib.AfoLogic import Saltbush as slp
from lib.AfoLogic import CropGrazing as cgz
from lib.AfoLogic import CropResidue as stub
from lib.AfoLogic import StockGenerator as stock
from lib.AfoLogic import Trees as trees

time_list.append(timer()) ; time_was.append("import Modules")

params={}
r_vals={}

###############
#User control #
###############
trial = 3   #3 is quick test
test_pas = False
test_sup = False
test_slp = False
test_cropgraze = False
test_stub = False
test_stock = False
test_trees = False

######
#Run #
######
##load excel data and experiment data
exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
exp_data = exp.f_group_exp(exp_data, exp_group_bool)
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp.iloc[[trial]])
d_rot_info = dxl.f_load_phases()
cat_propn_s1_ks2 = dxl.f_load_stubble()

##select property for the current trial
property = trial_pinp.iloc[trial]

##process user SA
user_sa = rve.f_process_user_sa(exp_data, trial)

##select property and reset default inputs for the current trial. Must occur first.
sinp.f_select_n_reset_sinp(sinp_defaults)
sinp.f_landuse_sets()
uinp.f_select_n_reset_uinp(uinp_defaults)
pinp.f_select_n_reset_pinp(property, pinp_defaults)

##update sensitivity values
sen.create_sa()
fun.f_update_sen(user_sa,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)

##call sa functions - assigns sa variables to relevant inputs
sinp.f_structural_inp_sa(sinp_defaults)
uinp.f_universal_inp_sa(uinp_defaults)
pinp.f_property_inp_sa(pinp_defaults)


##expand p6 axis to include nodes
sinp.f1_expand_p6()
pinp.f1_expand_p6()

##check the rotations and inputs align - this means rotation method can be controlled using a SA
d_rot_info = pinp.f1_phases(d_rot_info)

##mask lmu
pinp.f1_mask_lmu()

##mask land use
pinp.f1_mask_landuse()
uinp.f1_mask_landuse()



##Populate the nv dict with the input values for the nv cutoffs (normally are from StockGenerator) - used for some of the test modules.
### create nv dict
nv={}
### read values from the pasture_inputs dictionary
pas_inc = np.array(pinp.general['pas_inc_t'])
pastures = sinp.general['pastures'][pas_inc]
exceldata = pinp.pasture_inputs[pastures[0]]           # assign to exceldata the pasture data for the first pasture type (annuals)
i_me_maintenance_vf = exceldata['MaintenanceEff'][:, 1:].T
##add nv params to dict for use in pasture.py
n_non_confinement_pools=4
confinement_inc = False
index_f = np.arange(n_non_confinement_pools+confinement_inc)
##create the upper and lower cutoffs. If there is a confinement slice then it will be populated with values but they never get used.
###get association between the input fp and the node adjusted fp
a_p6std_p6z = per.f_feed_periods(option=2)
###apply association
####stock
sinp.structuralsa['i_nv_upper_p6z'] = np.take_along_axis(sinp.structuralsa['i_nv_upper_p6'][:,None],a_p6std_p6z,axis=0)
sinp.structuralsa['i_nv_lower_p6z'] = np.take_along_axis(sinp.structuralsa['i_nv_lower_p6'][:,None],a_p6std_p6z,axis=0)

nv_upper_p6fz = sinp.structuralsa['i_nv_upper_p6z'][:,None,:]
nv_upper_p6fz = zfun.f_seasonal_inp(nv_upper_p6fz,numpy=True,axis=-1)
nv_lower_p6fz = sinp.structuralsa['i_nv_lower_p6z'][:,None,:]
nv_lower_p6fz = zfun.f_seasonal_inp(nv_lower_p6fz,numpy=True,axis=-1)
nv_cutoff_lower_p6fz = nv_lower_p6fz + (
            nv_upper_p6fz - nv_lower_p6fz) / n_non_confinement_pools * index_f[:,None]
nv_cutoff_upper_p6fz = nv_lower_p6fz + (nv_upper_p6fz - nv_lower_p6fz) / n_non_confinement_pools * (
            index_f[:,None] + 1)
###Average these values to be passed to Pasture.py for efficiency of utilising ME and add to the dict
nv_cutoff_ave_p6fz = (nv_cutoff_lower_p6fz + nv_cutoff_upper_p6fz) / 2
nv['nv_cutoff_ave_p6fz'] = nv_cutoff_ave_p6fz
nv['confinement_inc'] = confinement_inc
nv['len_nv'] = n_non_confinement_pools+confinement_inc

time_list.append(timer()) ; time_was.append("startup complete")

#call desired test modules
if test_pas:
    pas.f_pasture(params, r_vals, nv)
    time_list.append(timer()) ; time_was.append("Pasture complete")
if test_sup:
    sup.f_sup_params(params,r_vals, nv)
    time_list.append(timer()) ; time_was.append("SupFeed complete")
if test_slp:    
    slp.f_saltbush_precalcs(params, r_vals, nv)
    time_list.append(timer()) ; time_was.append("Saltbush complete")
if test_cropgraze:
    cgz.f1_cropgraze_params(params, r_vals, nv)
    time_list.append(timer()) ; time_was.append("CropGraze complete")
if test_stub:
    stub.crop_residue_all(params, r_vals, nv, cat_propn_s1_ks2)
    time_list.append(timer()) ; time_was.append("Stubble complete")
if test_stock:
    stock.generator(params, r_vals, nv)
    time_list.append(timer()) ; time_was.append("Stock complete")
if test_trees:
    trees.f1_trees(params, r_vals)
    time_list.append(timer()) ; time_was.append("Trees complete")

#report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for test master module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in seconds
