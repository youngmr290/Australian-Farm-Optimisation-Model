# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John
"""

from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy import optimize as spo


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
trial = 31   #31 is quick test

######
#Run #
######
##load excel data and experiment data
exp_data, exp_group_bool, trial_pinp = exp.f_read_exp()
sinp_defaults, uinp_defaults, pinp_defaults = dxl.f_load_excel_default_inputs(trial_pinp=trial_pinp)
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

##mask lmu
pinp.f1_mask_lmu()

##expand p6 axis to include nodes
sinp.f1_expand_p6()
pinp.f1_expand_p6()

##check the rotations and inputs align - this means rotation method can be controlled using a SA
d_rot_info = pinp.f1_phases(d_rot_info)

# time_list.append(timer()) ; time_was.append("import other modules")


targets_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Targets",index_col=[0],header=[0], engine='openpyxl')
weights_c = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Weights",index_col=[0],header=[0], engine='openpyxl')
bestbet_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="BestBet",index_col=[0],header=[0], engine='openpyxl')
bnd_lo_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="Low",index_col=[0],header=[0], engine='openpyxl')
bnd_up_tc = pd.read_excel(relativeFile.findExcel("GEPEP_calibration.xlsx"), sheet_name="High",index_col=[0],header=[0], engine='openpyxl')

keys_t = targets_tc.index
keys_c = targets_tc.columns
n_coef = len(keys_c)
n_teams = len(keys_t)

###convert to np
targets_tc = targets_tc.values
weights_c = weights_c.values
bestbet_tc = bestbet_tc.values
bnd_lo_tc = bnd_lo_tc.values
bnd_up_tc = bnd_up_tc.values

##Set some of the control variables (that might want to be tweaked later)
maxiter = 1    #1000    The number of iterations of 'popsize' that can be carried out
popsize = 4    #15     The number of simulations being selected from
disp = True     #False   Display the result each iteration
polish = False  #True   After the differential evolution carry out some further refining
workers = 1     #1       The number of multi-processes. #todo perhaps could access this from the RunAFORaw arg

##sgen args
nv={}
pkl_fs_info={}
pkl_fs={}
gepep = True
stubble=False

##loop through teams and save output
calibration_tc = np.zeros((n_teams,n_coef))
for t in np.arange(n_teams):
    ## weightings for the calibration objective function
    ### these are defined for all teams and don't vary
    calibration_weights = weights_c
    ## the targets vary for each team. Would be good to control these from exp.xls
    calibration_targets = targets_tc[t]
    ## create the bounds for the calibration coefficients. Would be good to control these from exp so they can be tweaked for each team
    bounds = list(zip(bnd_lo_tc[t], bnd_up_tc[t]))
    ##specify the best starting conditions. Again good to be from exp.xls
    bestbet = bestbet_tc[t]

    ## call the optimise routine
    calibration_tc[t,:] = spo.differential_evolution(sgen.generator, bounds, args = (params, r_vals, nv, pkl_fs_info, pkl_fs, stubble, gepep, calibration_weights, calibration_targets)
                                             ,maxiter=maxiter,popsize=popsize, disp=disp, polish=polish, workers=workers, x0 = bestbet)["x"]

    print(calibration_tc[t])


##save output by trial - just so that user can check (this is not for AFO)
calibration = pd.DataFrame(calibration_tc, index=keys_t, columns=keys_c)
calibration_path = relativeFile.findExcel('calibration.xlsx')
writer = pd.ExcelWriter(calibration_path, engine='xlsxwriter')
calibration.to_excel(writer,index=True, header=True)
writer.close()

time_list.append(timer()) ; time_was.append("end")
print("elapsed total time for calibration", f"{time_list[-1] - time_list[0]:0.4f}", "secs") # Time in seconds
