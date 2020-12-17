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

run_pnl = False #table of profit and loss
run_profitarea = False #graph profit by crop area
run_saleprice = False #table of saleprices
run_cfw_dams = False #table of cfw
run_daily_mei_dams = True #table of mei
run_daily_pi_dams = False #table of mei
run_numbers_dams = False #table of numbers
run_numbers_offs = False #table of numbers
run_dse = False #table of dse
run_grnfoo = False #table of green foo at end of fp
run_dryfoo = False #table of dry foo at end of fp
run_napfoo = False #table of nap foo at end of fp
run_grncon = True #table of green con at end of fp
run_drycon = True #table of dry con at end of fp
run_napcon = True #table of nap con at end of fp
run_poccon = True #table of poc con at end of fp
run_supcon = True #table of sup con at end of fp
run_stubcon = True #table of sup con at end of fp


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
    plot = rep.f_xy_graph(func0, func1, lp_vars, r_vals, trial_outdated, exp_data_index, trials, func0_option, func1_option)
    plot.savefig('Output\profitarea_curve.png')

if run_saleprice:
    func = rep.f_price_summary
    trials = [0, 25,26,27,28]
    option = 2
    grid = [0,5,6]
    weight = [22,40,25]
    fs = [2,3,2]
    saleprice = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, option=option, grid=grid, weight=weight, fs=fs)
    saleprice.to_excel(writer, 'saleprice')

if run_cfw_dams:
    func = rep.f_stock_pasture_summary
    trials = [0, 25,26,27,28]
    type = 'stock'
    prod = 'cfw_hdmob_k2tva1nwziyg1'
    weights = 'dams_numbers_k2tvanwziy1g1'
    keys = 'dams_keys_k2tvanwziy1g1'
    arith = 1
    arith_axis = [6]
    index =[0,1,5]
    cols =[2]
    axis_slice = {}
    axis_slice[0] = [0, 2, 1]
    cfw_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    cfw_dams.to_excel(writer, 'cfw_dams')

if run_daily_mei_dams:
    func = rep.f_stock_pasture_summary
    trials = [0]
    type = 'stock'
    prod = 'dams_mei_k2p6ftva1nwziyg1'
    weights = 'dams_numbers_k2tvanwziy1g1'
    na_weights = [1, 2]
    den_weights = 'stock_days_k2p6ftva1nwziyg1'
    keys = 'dams_keys_k2p6ftvanwziy1g1'
    arith = 1
    arith_axis = [2,3,4,5,6,7,8,9,10,11]
    index =[1]
    cols =[0]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    daily_mei_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                           arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    daily_mei_dams.to_excel(writer, 'daily_mei_dams')

if run_daily_pi_dams:
    func = rep.f_stock_pasture_summary
    trials = [0]
    type = 'stock'
    prod = 'dams_pi_k2p6ftva1nwziyg1'
    weights = 'dams_numbers_k2tvanwziy1g1'
    na_weights = [1, 2]
    den_weights = 'stock_days_k2p6ftva1nwziyg1'
    keys = 'dams_keys_k2p6ftvanwziy1g1'
    arith = 1
    arith_axis = [2,3,4,5,6,7,8,9,10,11]
    index =[1]
    cols =[0]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    daily_pi_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                           arith_axis=arith_axis, index=index, cols=cols,
                           axis_slice=axis_slice)
    daily_pi_dams.to_excel(writer, 'daily_pi_dams')

if run_numbers_dams:
    func = rep.f_stock_pasture_summary
    trials = [0]
    type = 'stock'
    weights = 'dams_numbers_k2tvanwziy1g1'
    keys = 'dams_keys_k2tvanwziy1g1'
    arith = 2
    arith_axis = [1,3,4,5,6,7,8,9]
    index =[2]
    cols =[0]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    numbers_dams = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                           axis_slice=axis_slice)
    numbers_dams.to_excel(writer, 'numbers_dams')

if run_numbers_offs:
    func = rep.f_stock_pasture_summary
    trials = [0]
    type = 'stock'
    weights = 'offs_numbers_k3k5tvnwziaxyg3'
    keys = 'offs_keys_k3k5tvnwziaxyg3'
    arith = 2
    arith_axis = [0,1,4,5,6,7,8,9,10,11]
    index =[3]
    cols =[2]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    numbers_offs = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                           axis_slice=axis_slice)
    numbers_offs.to_excel(writer, 'numbers_offs')

if run_dse:
    func = rep.f_dse
    trials = [0]
    method = 0
    per_ha = True
    dse = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, method = method, per_ha = per_ha)
    dse.to_excel(writer, 'dse')

if run_grnfoo:
    #returns foo at end of each fp
    func = rep.f_stock_pasture_summary
    trials = [0,25,26,27,28]
    type = 'stock'
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
    func = rep.f_stock_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    type = 'pas'
    weights = 'drypas_transfer_dft'
    keys = 'keys_dft'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    dryfoo = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    dryfoo.to_excel(writer, 'dryfoo')

if run_napfoo:
    #returns foo at end of each fp
    func = rep.f_stock_pasture_summary
    trials = [0,25,26,27,28]
    prod = 1000
    type = 'pas'
    weights = 'nap_transfer_dft'
    keys = 'keys_dft'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    napfoo = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    napfoo.to_excel(writer, 'napfoo')

if run_grncon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [0]
    prod = 'cons_grnha_t_goflt'
    type = 'pas'
    weights = 'greenpas_ha_vgoflt'
    keys = 'keys_vgoflt'
    arith = 2
    arith_axis = [0,1,2,4,5]
    index =[3]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    grncon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    grncon.to_excel(writer, 'grncon')

if run_drycon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [0]
    prod = 1000
    type = 'pas'
    weights = 'drypas_consumed_vdft'
    keys = 'keys_vdft'
    arith = 2
    arith_axis = [0,1,3]
    index =[2]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    drycon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    drycon.to_excel(writer, 'drycon')

if run_napcon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [0]
    prod = 1000
    type = 'pas'
    weights = 'nap_consumed_vdft'
    keys = 'keys_vdft'
    arith = 2
    arith_axis = [0,1,3]
    index =[2]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    napcon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    napcon.to_excel(writer, 'napcon')

if run_poccon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [0]
    prod = 1000
    type = 'pas'
    weights = 'poc_consumed_vfl'
    keys = 'keys_vfl'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    poccon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    poccon.to_excel(writer, 'poccon')

if run_supcon:
    #returns consumption in each fp
    func = rep.f_grain_sup_summary
    trials = [0]
    option = 1
    supcon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, option=option)
    supcon.to_excel(writer, 'supcon')

if run_stubcon:
    #returns consumption in each fp
    func = rep.f_stubble_summary
    trials = [0]
    stubcon = rep.f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials)
    stubcon.to_excel(writer, 'stubcon')




writer.save()