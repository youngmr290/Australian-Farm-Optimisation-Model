"""
Created on Thu Dec 2 09:35:26 2020

@author: Young
"""

import pickle as pkl
import pandas as pd

import ReportFunctions as rep
import Functions as fun

## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('Output\Report.xlsx', engine='xlsxwriter')

##read in exp log
exp_data_nosort = fun.f_read_exp()
exp_data_index = exp_data_nosort.index #need to use this so user can specify the trial number as per exp.xlsx
exp_data = exp_data_nosort.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx




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
run_profitarea = True #graph profit by crop area
run_saleprice = True #table of saleprices
run_cfw_dams = True #table of cfw
run_dse = True #table of dse
run_grnfoo = True #table of green foo at end of fp
run_dryfoo = True #table of dry foo at end of fp
run_napfoo = True #table of nap foo at end of fp
run_grncon = True #table of green con at end of fp
run_drycon = True #table of dry con at end of fp
run_napcon = True #table of nap con at end of fp
run_poccon = True #table of poc con at end of fp


##run report functions
if run_pnl:
    func = rep.f_profitloss_table
    trials = [0, 25,26,27,28]
    pnl = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials)
    pnl.to_excel(writer, 'pnl')

if run_profitarea:
    func0 = rep.f_area_summary
    func1 = rep.f_profit
    func0_option = 4
    func1_option = 0
    trials = [25,26,27,28]
    plt = rep.f_xy_graph(func0, func1, lp_vars, r_vals, trial_outdated, exp_data_index, trials, func0_option, func1_option)
    plt.savefig('Output\profitarea_curve.png')

if run_saleprice:
    func = rep.f_price_summary
    trials = [0, 25,26,27,28]
    option = 0
    grid = [0,5,6]
    weight = [22,40,25]
    fs = [2,3,2]
    saleprice = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, option=option, grid=grid, weight=weight, fs=fs)
    saleprice.to_excel(writer, 'saleprice')

if run_cfw_dams:
    func = rep.f_stock_summary
    trials = [0, 25,26,27,28]
    prod = 'cfw_hdmob_k2tva1nwziyg1'
    weights = 'dams_numbers_k2tvanwziy1g1'
    keys = 'dams_keys_k2tvanwziy1g1'
    arith = 1
    arith_axis = [6]
    index =[0,1,5]
    cols =[2]
    axis_slice = {}
    axis_slice[0] = [0, 2, 1]
    cfw_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    cfw_dams.to_excel(writer, 'cfw_dams')

if run_dse:
    func = rep.f_dse
    trials = [0]
    method = 0
    per_ha = True
    dse = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, method = method, per_ha = per_ha)
    dse.to_excel(writer, 'dse')

if run_grnfoo:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 'foo_end_grnha_goflt'
    weights = 'greenpas_ha_vgoflt'
    keys = 'keys_vgoflt'
    arith = 2
    arith_axis = [0,1,2,4,5]
    index =[3]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    grnfoo = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    grnfoo.to_excel(writer, 'grnfoo')

if run_dryfoo:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    weights = 'drypas_transfer_dft'
    keys = 'keys_dft'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    dryfoo = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    dryfoo.to_excel(writer, 'dryfoo')

if run_napfoo:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    weights = 'nap_transfer_dft'
    keys = 'keys_dft'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    napfoo = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    napfoo.to_excel(writer, 'napfoo')

if run_grncon:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 'cons_grnha_t_goflt'
    weights = 'greenpas_ha_vgoflt'
    keys = 'keys_vgoflt'
    arith = 2
    arith_axis = [0,1,2,4,5]
    index =[3]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    grncon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    grncon.to_excel(writer, 'grncon')

if run_drycon:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    weights = 'drypas_consumed_vdft'
    keys = 'keys_vdft'
    arith = 2
    arith_axis = [0,1,3]
    index =[2]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    drycon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    drycon.to_excel(writer, 'drycon')

if run_napcon:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    weights = 'nap_consumed_vdft'
    keys = 'keys_vdft'
    arith = 2
    arith_axis = [0,1,3]
    index =[2]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    napcon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    napcon.to_excel(writer, 'napcon')

if run_poccon:
    #returns foo at end of each fp
    func = rep.f_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    weights = 'poc_consumed_vfl'
    keys = 'keys_vfl'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    poccon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    poccon.to_excel(writer, 'poccon')




writer.save()