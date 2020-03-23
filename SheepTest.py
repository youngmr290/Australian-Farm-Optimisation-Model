# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John
"""

from timeit import default_timer as timer
import datetime as dt


import SheepSim as ssim
import PropertyInputs as pinp
import FeedBudget as fdb


## map the Excel inputs to the python variables
ssim.init_and_map_excel()

## define the periods




### to time function calls in a module
time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")
#call function
time_list.append(timer()) ; time_was.append("import other modules")
## then repeat for other modules
##report the timer results
time_prev=time_list[0]
for time_step, time in enumerate(time_list):
    time_elapsed = time-time_prev
    if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
    time_prev=time
print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in secondsfirst
