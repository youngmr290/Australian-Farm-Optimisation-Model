import numpy as np
from timeit import default_timer as timer

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")

import StructuralInputs as sinp
import PropertyInputs as pinp
import UniversalInputs as uinp
import Periods as per
import Functions as fun
import SeasonalFunctions as zfun
import Sensitivity as sen
import CropGrazing as cgz



params={}
r_vals={}

exp_data, exp_group_bool = fun.f_read_exp()
exp_data = fun.f_group_exp(exp_data, exp_group_bool)
##update sensitivity values
fun.f_update_sen(4,exp_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav) #4 is quick test
##call sa functions - assigns sa variables to relevant inputs
sinp.f_structural_inp_sa()
uinp.f_universal_inp_sa()
pinp.f_property_inp_sa()
##expand p6 axis to include nodes
sinp.f1_expand_p6()
pinp.f1_expand_p6()

##Populate the nv dict with the input values for the nv cutoffs (normally are from StockGenerator)
### create nv dict
nv={}
### read values from the pasture_inputs dictionary
pas_inc = np.array(pinp.general['pas_inc'])
pastures = sinp.general['pastures'][pas_inc]
exceldata = pinp.pasture_inputs[pastures[0]]           # assign to exceldata the pasture data for the first pasture type (annuals)
i_me_maintenance_vf = exceldata['MaintenanceEff'][:, 1:].T
##add nv params to dict for use in pasture.py
n_non_confinement_pools=4
confinement_inc = False
index_f = np.arange(n_non_confinement_pools+confinement_inc)
###create the upper and lower cutoffs. If there is a confinement slice then it will be populated with values but they never get used.
####get association between the input fp and the node adjusted fp
a_p6std_p6z = per.f_feed_periods(option=2)
####apply association
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
####Average these values to be passed to CropGrazing.py for efficiency of utilising ME and add to the dict
nv_cutoff_ave_p6fz = (nv_cutoff_lower_p6fz + nv_cutoff_upper_p6fz) / 2
nv['nv_cutoff_ave_p6fz'] = nv_cutoff_ave_p6fz
nv['confinement_inc'] = confinement_inc
nv['len_nv'] = n_non_confinement_pools+confinement_inc

##call pasture module
cgz.f1_cropgraze_params(params, r_vals, nv)

time_list.append(timer()) ; time_was.append("CropGrazing complete")