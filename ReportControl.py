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
run_saleprice = True #table of saleprices
run_cfw_dams = False #table of cfw
run_grnfoo = True #table of green foo at end of fp


##run report functions
if run_pnl:
    func = rep.f_profitloss_table
    trials = [1, 1]
    pnl = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials)

if run_profitarea:
    profit_option = 0
    area_option = 4
    trials = [1]
    rep.f_croparea_profit(lp_vars, r_vals, trial_outdated, exp_data_index, trials, area_option, profit_option)

if run_saleprice:
    func = rep.f_price_summary
    trials = [1, 1]
    option = 0
    grid = [0,5,6]
    weight = [22,40,25]
    fs = [2,3,2]
    rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, option=option, grid=grid, weight=weight, fs=fs)

if run_cfw_dams:
    func = rep.f_stock_summary
    trials = [1, 1]
    prod = 'cfw_hdmob_k2tva1nwziyg1' #keys for r_vals
    weights = 'numbers_k2tva1nwziyg1' #keys for r_vals
    keys = 'dams_keys_k2tvanwziy1g1' #keys for r_vals
    arith = 1
    arith_axis = [6]
    index =[0,1,5]
    cols =[2]
    axis_slice = {}
    axis_slice[0] = [0, 2, 1]
    cfw_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)

if run_grnfoo:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [1]
    prod = 'foo_end_grnha_goflt' #keys for r_vals
    weights = 'greenpas_ha_vgoflt' #keys for r_vals
    keys = 'keys_vgoflt' #keys for r_vals
    arith = 2
    arith_axis = [0,1,2,4,5]
    index =[3]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    grnfoo_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)

    # ##dry transfer eg tonnes of dry at end of period
    # drypas_transfer_dft = pas_vars['drypas_transfer_dft']
    #
    # ##dry transfer eg tonnes of dry at end of period
    # nap_transfer_dft = pas_vars['nap_transfer_dft']
