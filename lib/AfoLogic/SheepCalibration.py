# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John
"""

from timeit import default_timer as timer

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")

from scipy import optimize as spo

from . import StockGenerator as sgen

params=dict() #^this will come from exp once everything is done
r_vals=dict() #^this will come from exp once everything is done
nv={}

# time_list.append(timer()) ; time_was.append("import other modules")

##calibration
###The calibration traits and the calibration coefficients are:
# CFW  -                  sfw
# FD   -                  sfd
# SS   -                  cw[16]
# SL   -                  cw[11]
# conception   -          cu2[25,0]
# litter size  -          cl0[25,2]
# ewe rearing ability -
# LW    -                srw
# proportion fat  -      cg[8] and cg[9] varied together
# mortality    -
# weaning weight   -

##The calibration objective function is the weighted sum of the mean square error WSMSE
###


## weightings for the calibration objective function
### these are defined for all teams and don't vary
calibration_weights = []

## the targets vary for each team. Would be good to control these from exp.xls
calibration_targets = []

## create the bounds for the calibration coefficients. Would be good to control these from exp so they can be tweaked for each team
bounds = [(),()]
##specify the best starting conditions. Again good to be from exp.xls
bestbet = []

#Set some of the control variables (that might want to be tweaked later)
maxiter = 2    #1000    The number of iterations of 'popsize' that can be carried out
popsize = 4    #15     The number of simulations being selected from
disp = True     #False   Display the result each iteration
polish = False  #True   After the differential evolution carry out some further refining
workers = 2     #1       The number of multi-processes. #todo perhaps could access this from the RunAFORaw arg


## call the optimise routine
calibration = spo.differential_evolution(sgen.generator, bounds, args = (params, r_vals, nv, pkl_fs_info, pkl_fs, gepep = True)
                                         ,maxiter=maxiter,popsize=popsize, disp=disp, polish=polish, workers=workers, x0 = bestbet)



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
