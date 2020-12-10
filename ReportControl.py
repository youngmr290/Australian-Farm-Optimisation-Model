"""
Created on Thu Dec 2 09:35:26 2020

@author: Young
"""

import pickle as pkl
import pandas as pd

import ReportFunctions as rep
import Functions as fun


##read in exp log
exp_data_nosort = pd.read_excel('exp.xlsx',index_col=[0,1,2], header=[0,1,2,3])
exp_data_index = exp_data_nosort.index #need to use this so user can specify the trial number as per exp.xlsx
exp_data = exp_data_nosort.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx
exp_data['run']=False



##load pickle
with open('pkl_lp_vars.pkl', "rb") as f:
    lp_vars = pkl.load(f)
with open('pkl_r_vals.pkl', "rb") as f:
    r_vals = pkl.load(f)
###prev exp - used to determine if the report is using up to date data.
with open('pkl_exp.pkl', "rb") as f:
    prev_exp = pkl.load(f)

##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xlsx has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial
exp_data = fun.f_run_required(prev_exp, exp_data, check_pyomo=False)
trial_outdated = exp_data['run'] #returns true if trial is out of date

run_pnl = True #table of profit and loss
run_profitarea = False #graph profit by crop area
run_saleprice = False #table of saleprices


##run report functions
if run_pnl:
    rep.f_pnl(lp_vars, r_vals, trial_outdated, exp_data_index, trials=[1,2])

if run_profitarea:
    rep.f_croparea_profit(lp_vars, r_vals, trial_outdated, exp_data_index, trials=[1,2], option=0)

if run_saleprice:
    ##select which grids, weights and fat scores to return
    grid = [0,5,6] #has to be int between 0 and 7 inclusive
    weight = [22,40,25] #can be anything, weight is interpolated
    fs = [2,3,2] #must be int between 1 and 5 inclusive
    rep.f_saleprice(lp_vars, r_vals, trial_outdated, exp_data_index, trials=[1,2], grid=grid, weight=weight, fs=fs)




