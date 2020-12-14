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

import UniversalInputs as uinp

time_list.append(timer()) ; time_was.append("import Universal")

import Pasture as pas

params={}
r_vals={}

time_list.append(timer()) ; time_was.append("import Pasture")

pas.map_excel(params,r_vals)
#pas.map_excel('Property.xlsx')                         # read inputs from Excel file and map to the python variables
time_list.append(timer()) ; time_was.append("init & read inputs from Excel")
params=dict()
pas.calculate_germ_and_reseed(params)                          # calculate the germination for each rotation phase
a = pas.foo_grn_reseeding_flrt
b = a[:,4,...]
c = np.sum(b, axis = 1)
time_list.append(timer()) ; time_was.append("germination & reseeding")

pas.green_and_dry(params, r_vals)                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
time_list.append(timer()) ; time_was.append("green feed & dry feed")

poc_con_ft = pas.poc(params)                            # calculate the pasture on crop paddocks
# poc_md_ft = pas.poc_md()                              # calculate the pasture on crop paddocks
# poc_vol_ft = pas.poc_vol()                            # calculate the pasture on crop paddocks
# print(poc_vol_ft)
time_list.append(timer()) ; time_was.append("poc")




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
