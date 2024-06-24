# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John

Multi-process the teams if there are sufficient teams to occupy the computer resource
If not, use multiple workers. The maximum useful number of workers is the size of the selection population
Multiprocessing teams should be more efficient because it can use 'immediate' updating

sys.argv: Experiment number (will use the first trial in the experiment). If blank uses QT (trial 31)
          Number of multi processes. If blank will not process but will use workers
"""


from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy import optimize as spo
import multiprocessing as mp
import os
import sys
import multiprocessing


time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")


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
from lib.AfoLogic import StockGenerator as sgen
from lib.AfoLogic import relativeFile


params={}
r_vals={}

###############
#User control #
###############
try:
    exp_number = int(sys.argv[1])  #reads in as string so need to convert to int, the script path is the first value hence take the second.
    trial = 0    #If an experiment was passed as an argument then take the first trial in the experiment
except (IndexError, ValueError) as e:  #in case no arg passed to python specify a trial number
    ## the trial number is value in Col A of target trial in exp.xls. Default is QT (trial 31) but can point to any trial
    trial = 31   #31 is QT

######
#Run #
######
##load excel data and experiment data
exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
d_rot_info = dxl.f_load_phases()
cat_propn_s1_ks2 = dxl.f_load_stubble()

##cut exp_data based on the experiment group
exp_data = exp.f_group_exp(exp_data, exp_group_bool)

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

##mask lmu
pinp.f1_mask_lmu()

##expand p6 axis to include nodes
sinp.f1_expand_p6()
pinp.f1_expand_p6()

##check the rotations and inputs align - this means rotation method can be controlled using a SA
d_rot_info = pinp.f1_phases(d_rot_info)

# time_list.append(timer()) ; time_was.append("import other modules")


targets_tp = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Targets",index_col=[0],header=[0], engine='openpyxl')
weights_p = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Weights",index_col=[0],header=[0], engine='openpyxl')
bestbet_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="BestBet",index_col=[0],header=[0], engine='openpyxl')
bnd_lo_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Low",index_col=[0],header=[0], engine='openpyxl')
bnd_up_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="High",index_col=[0],header=[0], engine='openpyxl')

keys_t = targets_tp.index
keys_c = bestbet_tc.columns
n_coef = len(keys_c)
n_teams = len(keys_t)

##############
##processors #
##############
## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
try:
    maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the trial is the first value hence take the second.
except IndexError:  # in case no arg passed to python
    maximum_processes = 1  # available memory / value determined by size of the model being run (~5GB for the small model)
## number of agents (processes) should be min of the num of cpus, number of teams or the user specified limit due to memory capacity
n_processes = min(multiprocessing.cpu_count(),n_teams, maximum_processes)

###convert to np
targets_tp = targets_tp.values
weights_p = weights_p.values
bestbet_tc = bestbet_tc.values
bnd_lo_tc = bnd_lo_tc.values
bnd_up_tc = bnd_up_tc.values

##sgen args
nv={}
pkl_fs_info={}
pkl_fs={}
gepep = True
stubble=False

## create empty arrays to accept the output
coefficients_tc = np.zeros((n_teams,n_coef))
success_t = np.zeros(n_teams, dtype=bool)
wsmse_t = np.zeros(n_teams)
message_t = np.empty(n_teams, dtype = object)

##loop through teams and save output

def f_run_calibration(t,coefficients_dict, success_dict, wsmse_dict, message_dict):
    ## weightings for the calibration objective function
    ### these are defined for all teams and don't vary
    calibration_weights = weights_p
    ## the targets vary for each team. Would be good to control these from exp.xls
    calibration_targets = targets_tp[t]
    ## create the bounds for the calibration coefficients. Would be good to control these from exp so they can be tweaked for each team
    bounds = list(zip(bnd_lo_tc[t], bnd_up_tc[t]))
    ##specify the best starting conditions. Again good to be from exp.xls
    bestbet = bestbet_tc[t]

    ##Set some of the control variables (that might want to be tweaked later)
    maxiter = 1000  #1000      The maximum number of iterations. # calls = (maxiter + 1) * selection population
    popsize = 5     #15        The selection population is (popsize * n coefficients)
    tol = 0.01       #0.01      The optimisation relative tolerance
    disp = True     #False     Display the result each iteration
    polish = True   #True      After the differential evolution carry out some further refining
    workers = 1     #10        Must be equal to 1 if multiprocessing the teams
    updating = 'immediate'  #  Use deferred if workers > 1 to suppress warning

    ## call the optimise routine
    result = spo.differential_evolution(sgen.generator, bounds
        , args = (params, r_vals, nv, pkl_fs_info, pkl_fs, stubble, gepep, calibration_weights, calibration_targets)
        , maxiter=maxiter, popsize=popsize, tol=tol, disp=disp, polish=polish, updating=updating, workers=workers, x0=bestbet)
    #assign the team results to dicts
    coefficients_dict[t] = result.x
    success_dict[t] = result.success
    wsmse_dict[t] = result.fun
    message_dict[t] = result.message
    print(f"Team {t} coefficients are {result.x} obj: {result.fun} evaluations {result.nfev}")

teams = list(range(n_teams))
if __name__ == '__main__':
    if n_processes != 1:    # read as a string so need to convert to int
        print (f"multiprocess across {n_processes} teams")
        manager = multiprocessing.Manager()
        coefficients_dict = manager.dict()
        success_dict = manager.dict()
        wsmse_dict = manager.dict()
        message_dict = manager.dict()
        from functools import partial
        ##start multiprocessing
        ### number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
        agents = n_processes

        with multiprocessing.Pool(processes=agents) as pool:
            # results = pool.map(f_run_calibration, teams, return_dict, chunksize=1)
            results = pool.map(partial(f_run_calibration, coefficients_dict=coefficients_dict, success_dict=success_dict
                                       , wsmse_dict=wsmse_dict, message_dict=message_dict), teams, chunksize=1)

        ##save output by trial - just so that user can check (this is not for AFO)
        ### Set up the dataframes
        keys_t = coefficients_dict.keys() #incase teams are out of order
        coefficients_tc = np.array(coefficients_dict.values())
        success_t = np.array(success_dict.values())
        wsmse_t = np.array(wsmse_dict.values())
        message_t = np.array(message_dict.values())

    else:
        for t in np.arange(n_teams):
            ## weightings for the calibration objective function
            ### these are defined for all teams and don't vary
            calibration_weights = weights_p
            ## the targets vary for each team. Would be good to control these from exp.xls
            calibration_targets = targets_tp[t]
            ## create the bounds for the calibration coefficients. Would be good to control these from exp so they can be tweaked for each team
            bounds = list(zip(bnd_lo_tc[t], bnd_up_tc[t]))
            ##specify the best starting conditions. Again good to be from exp.xls
            bestbet = bestbet_tc[t]

            ##Set some of the control variables (that might want to be tweaked later)
            maxiter = 1000  #1000      The maximum number of iterations. # calls = (maxiter + 1) * selection population
            popsize = 6  #15        The selection population is (popsize * n coefficients)
            tol = 0.01  #0.01      The optimisation relative tolerance
            disp = True  #False     Display the result each iteration
            polish = True  #True      After the differential evolution carry out some further refining
            population = popsize * n_coef
            max_workers = 15  #1         The number of multi-processes, while calculating the population. Relate to size of population
            workers = min(multiprocessing.cpu_count(), population, max_workers)
            if workers != 1:
                updating = 'deferred'  #   Use deferred if workers > 1 to suppress warning
            else:
                updating = 'immediate'

            print(f"multiprocess the population of {population} with {workers} workers")

            mp.freeze_support()
            ## call the optimise routine
            result = spo.differential_evolution(sgen.generator, bounds
                , args = (params, r_vals, nv, pkl_fs_info, pkl_fs, stubble, gepep, calibration_weights, calibration_targets)
                , maxiter=maxiter, popsize=popsize, tol=tol, disp=disp, polish=polish, updating=updating, workers=workers, x0=bestbet)
            #assign the team results to arrays
            coefficients_tc[t, :] = result.x
            success_t[t] = result.success
            wsmse_t[t] = result.fun
            message_t[t] = result.message
            time_list.append(timer()); time_was.append(t)
            print(f"Team {t} coefficients are {result.x} obj: {result.fun} evaluations {result.nfev} {time_list[-1] - time_list[-2]:0.4f}secs")

    coefficients = pd.DataFrame(coefficients_tc, index=keys_t, columns=keys_c)
    success = pd.DataFrame(success_t, index=keys_t, columns=["Optimal"])
    wsmse  = pd.DataFrame(wsmse_t, index=keys_t, columns=["WSMSE"])
    message = pd.DataFrame(message_t, index=keys_t, columns=["Message"])

    ### Write to Excel
    calibration_path = relativeFile.findExcel('CalibrationResults.xlsx')
    writer = pd.ExcelWriter(calibration_path, engine='xlsxwriter')
    coefficients.to_excel(writer,"result", index=True, header=True, startrow=0, startcol=0)
    success.to_excel(writer,"result", index=False, header=True, startrow=0, startcol=n_coef+1)
    wsmse.to_excel(writer,"result", index=False, header=True, startrow=0, startcol=n_coef+2)
    message.to_excel(writer,"result", index=False, header=True, startrow=0, startcol=n_coef+3)
    writer.close()

    time_list.append(timer()) ; time_was.append("end")
    print(f"elapsed total time for calibration: {time_list[-1] - time_list[0]:0.4f} secs") # Time in seconds
