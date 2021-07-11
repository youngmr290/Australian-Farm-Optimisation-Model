# -*- coding: utf-8 -*-
'''
Created on Mon Nov 11 10:15:40 2019

Version Control:
Version     Date        Person  Change
   1.1      30Dec19     JMY     Added timing to check elapsed time for each function in pasture functions

Known problems:
Fixed   Date    ID by   Problem


@author: John
'''
import numpy as np
from timeit import default_timer as timer

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")

import StructuralInputs as sinp
import PropertyInputs as pinp
import Periods as per

time_list.append(timer()) ; time_was.append("import Universal")

import Pasture as pas

time_list.append(timer()) ; time_was.append("import Pasture")

params={}
r_vals={}

#call this to adjust inputs with p6 axis for season nodes
pinp.property_inp_sa()

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
##create the upper and lower cutoffs. If there is a confinement slice then it will be populated with values but they never get used.
###get association between the input fp and the node adjusted fp
a_p6std_p6z = per.f_feed_periods(option=2)
###apply association
####stock
sinp.structuralsa['i_nv_upper_p6z'] = np.take_along_axis(sinp.structuralsa['i_nv_upper_p6'][:,None],a_p6std_p6z,axis=0)
sinp.structuralsa['i_nv_lower_p6z'] = np.take_along_axis(sinp.structuralsa['i_nv_lower_p6'][:,None],a_p6std_p6z,axis=0)

nv_upper_p6fz = sinp.structuralsa['i_nv_upper_p6z'][:,None,:]
nv_upper_p6fz = pinp.f_seasonal_inp(nv_upper_p6fz,numpy=True,axis=-1)
nv_lower_p6fz = sinp.structuralsa['i_nv_lower_p6z'][:,None,:]
nv_lower_p6fz = pinp.f_seasonal_inp(nv_lower_p6fz,numpy=True,axis=-1)
nv_cutoff_lower_p6fz = nv_lower_p6fz + (
            nv_upper_p6fz - nv_lower_p6fz) / n_non_confinement_pools * index_f[:,None]
nv_cutoff_upper_p6fz = nv_lower_p6fz + (nv_upper_p6fz - nv_lower_p6fz) / n_non_confinement_pools * (
            index_f[:,None] + 1)
###Average these values to be passed to Pasture.py for efficiency of utilising ME and add to the dict
nv_cutoff_ave_p6fz = (nv_cutoff_lower_p6fz + nv_cutoff_upper_p6fz) / 2
nv['nv_cutoff_ave_p6fz'] = nv_cutoff_ave_p6fz
nv['confinement_inc'] = confinement_inc
nv['len_nv'] = n_non_confinement_pools+confinement_inc

##call pasture module
pas.f_pasture(params, r_vals, nv)


time_list.append(timer()) ; time_was.append("Pasture complete")


# pas.map_excel(params,r_vals)
#pas.map_excel('Property.xlsx')                         # read inputs from Excel file and map to the python variables
# time_list.append(timer()) ; time_was.append("init & read inputs from Excel")


# pas.calculate_germ_and_reseed(params)                          # calculate the germination for each rotation phase
# a = pas.foo_grn_reseeding_p6lrt
# b = a[:,4,...]
# c = np.sum(b, axis = 1)
# time_list.append(timer()) ; time_was.append("germination & reseeding")

# pas.green_and_dry(params, r_vals, nv)                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
# time_list.append(timer()) ; time_was.append("green feed & dry feed")

# poc_con_p6t = pas.poc(params)                            # calculate the pasture on crop paddocks
# poc_md_p6t = pas.poc_md()                              # calculate the pasture on crop paddocks
# poc_vol_p6t = pas.poc_vol()                            # calculate the pasture on crop paddocks
# print(poc_vol_ft)
# time_list.append(timer()) ; time_was.append("poc")




#report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in seconds


#test times
#def test1():
#    annual.germ_phase_data.columns.values[range(phase_len)] = [*range(phase_len)]
#def test2():
#    annual.germ_phase_data.columns.values[0:phase_len] = [*range(phase_len)]
#
#print(timeit.repeat(test1,number=5,repeat=10))
#print(timeit.repeat(test2,number=5,repeat=10))
