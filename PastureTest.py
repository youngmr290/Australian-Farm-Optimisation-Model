# -*- coding: utf-8 -*-
'''
Created on Mon Nov 11 10:15:40 2019

Version Control:
Version     Date        Person  Change
   1.1      30Dec19     JMY     Added timing to check elapased time for each function in pasture functions

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

time_list.append(timer()) ; time_was.append("import Universal")

import Pasture as pas

time_list.append(timer()) ; time_was.append("import Pasture")

params={}
r_vals={}

##Populate the ev dict with the input values for the ev cutoffs (normally are from StockGenerator)
### create ev dict
ev={}
### read values from the pasture_inputs dictionary
pas_inc = np.array(pinp.general['pas_inc'])
pastures = sinp.general['pastures'][pas_inc]
exceldata = pinp.pasture_inputs[pastures[0]]           # assign to exceldata the pasture data for the first pasture type (annuals)
i_me_maintenance_vf = exceldata['MaintenanceEff'][:, 1:].T
##add ev params to dict for use in pasture.py
ev['ev_cutoff_p6fz'] = i_me_maintenance_vf[0:-1, ...].T[..., None]
ev['ev_max_p6z'] = i_me_maintenance_vf[-1, :, None]

pas.f_pasture(params, r_vals, ev)


time_list.append(timer()) ; time_was.append("Pasture complete")


# pas.map_excel(params,r_vals)
#pas.map_excel('Property.xlsx')                         # read inputs from Excel file and map to the python variables
# time_list.append(timer()) ; time_was.append("init & read inputs from Excel")


# pas.calculate_germ_and_reseed(params)                          # calculate the germination for each rotation phase
# a = pas.foo_grn_reseeding_flrt
# b = a[:,4,...]
# c = np.sum(b, axis = 1)
# time_list.append(timer()) ; time_was.append("germination & reseeding")

# pas.green_and_dry(params, r_vals, ev)                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
# time_list.append(timer()) ; time_was.append("green feed & dry feed")

# poc_con_ft = pas.poc(params)                            # calculate the pasture on crop paddocks
# poc_md_ft = pas.poc_md()                              # calculate the pasture on crop paddocks
# poc_vol_ft = pas.poc_vol()                            # calculate the pasture on crop paddocks
# print(poc_vol_ft)
# time_list.append(timer()) ; time_was.append("poc")




#report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in secondsfirst


#test times
#def test1():
#    annual.germ_phase_data.columns.values[range(phase_len)] = [*range(phase_len)]
#def test2():
#    annual.germ_phase_data.columns.values[0:phase_len] = [*range(phase_len)]
#
#print(timeit.repeat(test1,number=5,repeat=10))
#print(timeit.repeat(test2,number=5,repeat=10))
