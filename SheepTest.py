# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John
"""

from timeit import default_timer as timer

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")

import StockPyomo as spy
import CreateModel as model
import StockGenerator as sgen

params=dict() #^this will come from exp once everything is done
r_vals=dict() #^this will come from exp once everything is done
nv={}

# time_list.append(timer()) ; time_was.append("import other modules")

## call the main simulation loop & update the timer
model.sets()
# spy.stock_precalcs(params,report)
sgen.generator(params, r_vals, nv, plots=True)  # call generator direct so StockPyomo code is not cluttered
spy.stockpyomo_local(params)


# time_list.append(timer()) ; time_was.append("simulation loops")

## call the function to create the parameters & update the timer
# sgen.parameters()
# time_list.append(timer()) ; time_was.append("masks & parameters")



# ##report the timer results
# time_prev=time_list[0]
# for time_step, time in enumerate(time_list):
#     time_elapsed = time-time_prev
#     if time_elapsed > 0: print(time_was[time_step], f"{time_elapsed:0.4f}", "secs")
#     time_prev=time
# print("elapsed total time for pasture module", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in seconds
