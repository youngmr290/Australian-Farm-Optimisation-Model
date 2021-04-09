"""
Created on Thu Dec 2 09:35:26 2020


How this module works:
The trials to report are controlled in exp.xl.
The reports to run are controlled in exp.xl
The code loops through each 'on' trial and calculates the reports. The resulting table from each trial is stacked
 together. Once the loop is completed the between trial reports (reports that require info from all trials) are
 calculated.

To add a report:
1. Make the relevant code.
2. add it to exp.xl
3. add an empty df to stack results
4. add if statement to store final report to excel

Note: If reporting dates from a numpy array it is necessary to convert to datetime64[ns] prior to converting to a DataFrame

For example:
data_df3 = pd.DataFrame(fvp_fdams.astype('datetime64[ns]'))  # conversion to dataframe only works with this datatype

@author: Young
"""

import numpy as np
import pandas as pd


import ReportFunctions as rep
import Functions as fun

## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('Output/Report.xlsx',engine='xlsxwriter')

##read in exp log
exp_data_nosort = fun.f_read_exp()
exp_data_index = exp_data_nosort.index #need to use this so user can specify the trial number as per exp.xls
exp_data = exp_data_nosort.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xls


##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xls has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial
exp_data = fun.f_run_required(exp_data, check_pyomo=False)
trial_outdated = exp_data['run'] #returns true if trial is out of date

## enter the trials to summarise and the reports to include
trials = np.array(range(len(exp_data_index)))[pd.Series(exp_data_index.get_level_values(2)).fillna(0).astype(bool)]  #this is slightly complicated because blank rows in exp.xl result in nan, so nan must be converted to 0.

##check the trials you want to run exist and are up to date
rep.f_errors(exp_data_index, trial_outdated, trials)

##read in excel that controls which reports to run
report_run = pd.read_excel('exp.xlsm', sheet_name='Run Report', index_col=[0], header=[0], engine='openpyxl')

##create empty df to stack each trial results into
stacked_summary = pd.DataFrame()  # create df to append table from each trial
stacked_areasum = pd.DataFrame()  # create df to append table from each trial
stacked_pnl = pd.DataFrame()  # create df to append table from each trial
stacked_profitarea = pd.DataFrame()  # create df to append table from each trial
stacked_saleprice = pd.DataFrame()  # create df to append table from each trial
stacked_saledate_offs = pd.DataFrame()  # create df to append table from each trial
stacked_cfw_dams = pd.DataFrame()  # create df to append table from each trial
stacked_fd_dams = pd.DataFrame()  # create df to append table from each trial
stacked_wbe_dams = pd.DataFrame()  # create df to append table from each trial
stacked_wbe_offs = pd.DataFrame()  # create df to append table from each trial
stacked_lw_dams = pd.DataFrame()  # create df to append table from each trial
stacked_ffcfw_dams = pd.DataFrame()  # create df to append table from each trial
stacked_fec_dams = pd.DataFrame()  # create df to append table from each trial
stacked_ffcfw_prog = pd.DataFrame()  # create df to append table from each trial
stacked_fec_offs = pd.DataFrame()  # create df to append table from each trial
stacked_weanper = pd.DataFrame()  # create df to append table from each trial
stacked_scanper = pd.DataFrame()  # create df to append table from each trial
stacked_dry_propn = pd.DataFrame()  # create df to append table from each trial
stacked_lamb_survival = pd.DataFrame()  # create df to append table from each trial
stacked_daily_mei_dams = pd.DataFrame()  # create df to append table from each trial
stacked_daily_pi_dams = pd.DataFrame()  # create df to append table from each trial
stacked_numbers_dams = pd.DataFrame()  # create df to append table from each trial
stacked_numbers_prog = pd.DataFrame()  # create df to append table from each trial
stacked_numbers_offs = pd.DataFrame()  # create df to append table from each trial
stacked_dse = pd.DataFrame()  # create df to append table from each trial
stacked_dse1 = pd.DataFrame()  # create df to append table from each trial
stacked_grnfoo = pd.DataFrame()  # create df to append table from each trial
stacked_dryfoo = pd.DataFrame()  # create df to append table from each trial
stacked_napfoo = pd.DataFrame()  # create df to append table from each trial
stacked_grncon = pd.DataFrame()  # create df to append table from each trial
stacked_drycon = pd.DataFrame()  # create df to append table from each trial
stacked_napcon = pd.DataFrame()  # create df to append table from each trial
stacked_poccon = pd.DataFrame()  # create df to append table from each trial
stacked_supcon = pd.DataFrame()  # create df to append table from each trial
stacked_stubcon = pd.DataFrame()  # create df to append table from each trial

##read in the pickled results
report_data = {}
for row in trials:
    trial_name = exp_data_index[row][3]
    lp_vars,r_vals = rep.load_pkl(trial_name)

    ##run report functions
    if report_run.loc['run_summary','Run']:
        summary = rep.f_summary(lp_vars,r_vals,trial_name)
        stacked_summary = stacked_summary.append(summary)
    
    if report_run.loc['run_areasum', 'Run']:
        option = 2
        areasum = rep.f_area_summary(lp_vars, r_vals, option=option)
        areasum = pd.concat([areasum],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_areasum = stacked_areasum.append(areasum)
    
    if report_run.loc['run_pnl', 'Run']:
        pnl = rep.f_profitloss_table(lp_vars, r_vals)
        pnl = pd.concat([pnl],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_pnl = stacked_pnl.append(pnl)

    
    if report_run.loc['run_profitarea', 'Run']:
        area_option = 4
        profit_option = 0
        profitarea = pd.DataFrame(index=[trial_name], columns=['area','profit'])
        profitarea.loc[trial_name, 'area'] = rep.f_area_summary(lp_vars,r_vals,area_option)
        profitarea.loc[trial_name,'profit'] = rep.f_profit(lp_vars,r_vals,profit_option)
        stacked_profitarea = stacked_profitarea.append(profitarea)

    
    if report_run.loc['run_saleprice', 'Run']:
        option = 2
        grid = [0,5,6]
        weight = [22,40,25]
        fs = [2,3,2]
        saleprice = rep.f_price_summary(lp_vars, r_vals, option=option, grid=grid, weight=weight, fs=fs)
        saleprice = pd.concat([saleprice],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_saleprice = stacked_saleprice.append(saleprice)

    
    if report_run.loc['run_saledate_offs', 'Run']:
        #todo Currently this can return a date prior to the start of the generator. Some animals must not have a date
        type = 'stock'
        prod = 'saledate_k3k5tvnwziaxyg3'
        weights = 'offs_numbers_k3k5tvnwziaxyg3'
        keys = 'offs_keys_k3k5tvnwziaxyg3'
        arith = 1
        arith_axis = [4,6,7,8,10,11]
        index =[3,5]
        cols =[0,1,2,9]
        saledate_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols).astype('datetime64[D]')
        saledate_offs = pd.concat([saledate_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_saledate_offs = stacked_saledate_offs.append(saledate_offs)

    if report_run.loc['run_cfw_dams', 'Run']:
        type = 'stock'
        prod = 'cfw_hdmob_k2tva1nwziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        keys = 'dams_keys_k2tvanwziy1g1'
        arith = 1
        arith_axis = [3,4,5,6,7,8,9]
        index =[2]
        cols =[0,1]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        cfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        cfw_dams = pd.concat([cfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_cfw_dams = stacked_cfw_dams.append(cfw_dams)

    if report_run.loc['run_fd_dams', 'Run']:
        type = 'stock'
        prod = 'fd_hdmob_k2tva1nwziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        keys = 'dams_keys_k2tvanwziy1g1'
        arith = 1
        arith_axis = [3,4,5,6,7,8,9]
        index =[2]
        cols =[0,1]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        fd_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        fd_dams = pd.concat([fd_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_fd_dams = stacked_fd_dams.append(fd_dams)

    if report_run.loc['run_wbe_dams', 'Run']:
        type = 'stock'
        prod = 'wbe_k2tva1nwziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        keys = 'dams_keys_k2tvanwziy1g1'
        arith = 1
        arith_axis = [1,3,4,5,6,7,8,9]
        index =[2]
        cols =[0]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        wbe_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        wbe_dams = pd.concat([wbe_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_wbe_dams = stacked_wbe_dams.append(wbe_dams)

    if report_run.loc['run_wbe_offs', 'Run']:
        prod = 'wbe_k3k5tvnwziaxyg3'
        weights = 'offs_numbers_k3k5tvnwziaxyg3'
        keys = 'offs_keys_k3k5tvnwziaxyg3'
        arith = 1
        arith_axis = [4,5,6,7,8,9,10,11]
        index =[3]
        cols =[0,1,2]
        axis_slice = {}
        wbe_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        wbe_offs = pd.concat([wbe_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_wbe_offs = stacked_wbe_offs.append(wbe_offs)

    if report_run.loc['run_lw_dams', 'Run']:
        type = 'stock'
        prod = 'lw_dams_k2vpa1e1b1nw8ziyg1'
        na_prod = [1]
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [3,5,6]
        den_weights = 'pe1b1_denom_weights_k2tvpa1e1b1nw8ziyg1'
        keys = 'dams_keys_k2tvpaebnwziy1g1'
        arith = 1
        arith_axis = [0,1,2,4,5,7,8,9,10,11,12]  #reporting p(3) & b1(6)
        index =[3]
        cols =[6]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        lw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights
                                 , den_weights=den_weights, na_prod=na_prod, na_weights=na_weights
                                 , keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        lw_dams = pd.concat([lw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_lw_dams = stacked_lw_dams.append(lw_dams)

    
    if report_run.loc['run_ffcfw_dams', 'Run']:
        type = 'stock'
        prod = 'ffcfw_dams_k2vpa1e1b1nw8ziyg1'
        na_prod = [1]
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [3, 5, 6]
        den_weights = 'pe1b1_denom_weights_k2tvpa1e1b1nw8ziyg1'
        keys = 'dams_keys_k2tvpaebnwziy1g1'
        arith = 1
        arith_axis = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 12]  #reporting p(3) & b1(6)
        index = [3]
        cols = [6]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        ffcfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights
                                 , den_weights=den_weights, na_prod=na_prod, na_weights=na_weights
                                 , keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        ffcfw_dams = pd.concat([ffcfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_ffcfw_dams = stacked_ffcfw_dams.append(ffcfw_dams)

    
    if report_run.loc['run_fec_dams', 'Run']:
        type = 'stock'
        prod = 'fec_dams_k2vpa1e1b1nw8ziyg1'
        na_prod = [1]
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [3,5,6]
        den_weights = 'pe1b1_denom_weights_k2tvpa1e1b1nw8ziyg1'
        keys = 'dams_keys_k2tvpaebnwziy1g1'
        arith = 1
        arith_axis = [0,1,2,4,5,7,8,9,10,11,12]  #reporting p(3) & b1(6)
        index =[3]
        cols =[6]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        fec_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               den_weights=den_weights, na_prod=na_prod, na_weights=na_weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        fec_dams = pd.concat([fec_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_fec_dams = stacked_fec_dams.append(fec_dams)

    
    if report_run.loc['run_ffcfw_prog', 'Run']:
        type = 'stock'
        prod = 'ffcfw_prog_zia0xg2w9'
        weights = 1
        keys = 'prog_keys_zia0xg2w9'
        arith = 0
        arith_axis = []  #reporting p(3) & b1(6)
        index = [5]
        cols = [0,1,2,3,4]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        ffcfw_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights
                                 , keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        ffcfw_prog = pd.concat([ffcfw_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_ffcfw_prog = stacked_ffcfw_prog.append(ffcfw_prog)

    
    
    if report_run.loc['run_fec_offs', 'Run']:
        type = 'stock'
        prod = 'fec_offs_k3k5vpnw8zida0e0b0xyg3'
        na_prod = [2]
        weights = 'offs_numbers_k3k5tvnwziaxyg3'
        na_weights = [4,9,11,12]
        den_weights = 'pde0b0_denom_weights_k3k5tvpnw8zida0e0b0xyg3'
        keys = 'offs_keys_k3k5tvpnwzidaebxyg3'
        arith = 1
        arith_axis = [0,1,3,5,6,7,8,9,10,11,13,14,15]  # reporting p(4) & b0(12)
        index =[4]
        cols =[2,12]
        axis_slice = {}
        axis_slice[11] = [0,1,1] #first cycle
        axis_slice[9] = [2,-1,1] #Adult
        axis_slice[15] = [0,1,1] #BBB
        fec_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               den_weights=den_weights, na_prod=na_prod, na_weights=na_weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        fec_offs = pd.concat([fec_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_fec_offs = stacked_fec_offs.append(fec_offs)

    
    if report_run.loc['run_lamb_survival', 'Run']:
        option = 0
        arith_axis = [0,1,3,4,6,7,8,9,10]
        index =[2]
        cols =[5]
        axis_slice = {}
        lamb_survival = rep.f_lambing_status(lp_vars, r_vals, option=option, arith_axis=arith_axis,
                                    index=index, cols=cols, axis_slice=axis_slice)
        lamb_survival = pd.concat([lamb_survival],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_lamb_survival = stacked_lamb_survival.append(lamb_survival)

    
    if report_run.loc['run_weanper', 'Run']:
        option = 1
        arith_axis = [0,1,3,4,5,6,7,8,9]
        index =[2]
        cols =[]
        axis_slice = {}
        weanper = rep.f_lambing_status(lp_vars, r_vals, option=option, arith_axis=arith_axis,
                                    index=index, cols=cols, axis_slice=axis_slice)
        weanper = pd.concat([weanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_weanper = stacked_weanper.append(weanper)

    
    if report_run.loc['run_scanper', 'Run']:
        option = 2
        arith_axis = [0,1,3,4,5,6,7,8,9]
        index =[2]
        cols =[]
        axis_slice = {}
        scanper = rep.f_lambing_status(lp_vars, r_vals, option=option, arith_axis=arith_axis,
                                    index=index, cols=cols, axis_slice=axis_slice)
        scanper = pd.concat([scanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_scanper = stacked_scanper.append(scanper)

    if report_run.loc['run_dry_propn', 'Run']:
        option = 3
        arith_axis = [0,1,3,4,5,6,7,8,9]
        index =[2]
        cols =[]
        axis_slice = {}
        dry_propn = rep.f_lambing_status(lp_vars, r_vals, option=option, arith_axis=arith_axis,
                                    index=index, cols=cols, axis_slice=axis_slice)
        dry_propn = pd.concat([dry_propn],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_dry_propn = stacked_dry_propn.append(dry_propn)

    if report_run.loc['run_daily_mei_dams', 'Run']:
        type = 'stock'
        prod = 'mei_dams_k2p6ftva1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [1, 2]
        den_weights = 'stock_days_k2p6ftva1nwziyg1'
        keys = 'dams_keys_k2p6ftvanwziy1g1'
        arith = 1
        arith_axis = [3,5,6,7,8,9,10,11]    # for FP only [2,3,4,5,6,7,8,9,10,11]
        index =[4,1]                                    # [1]
        cols =[0,2]                                     # [0]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        daily_mei_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                               arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        daily_mei_dams = pd.concat([daily_mei_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_daily_mei_dams = stacked_daily_mei_dams.append(daily_mei_dams)

    
    if report_run.loc['run_daily_pi_dams', 'Run']:
        type = 'stock'
        prod = 'pi_dams_k2p6ftva1nw8ziyg1'
        weights = 'dams_numbers_k2tvanwziy1g1'
        na_weights = [1, 2]
        den_weights = 'stock_days_k2p6ftva1nwziyg1'
        keys = 'dams_keys_k2p6ftvanwziy1g1'
        arith = 1
        arith_axis = [3,5,6,7,8,9,10,11]    # for FP only [2,3,4,5,6,7,8,9,10,11]
        index =[4,1]                                    # [1]
        cols =[0,2]                                     # [0]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        daily_pi_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                                   arith_axis=arith_axis, index=index, cols=cols,
                                   axis_slice=axis_slice)
        daily_pi_dams = pd.concat([daily_pi_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_daily_pi_dams = stacked_daily_pi_dams.append(daily_pi_dams)

    
    if report_run.loc['run_numbers_dams', 'Run']:
        type = 'stock'
        weights = 'dams_numbers_k2tvanwziy1g1'
        keys = 'dams_keys_k2tvanwziy1g1'
        arith = 2
        arith_axis = [3,4,5,6,7,8,9]
        index =[2]
        cols =[0,1]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        numbers_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                               axis_slice=axis_slice)
        numbers_dams = pd.concat([numbers_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_numbers_dams = stacked_numbers_dams.append(numbers_dams)

    
    if report_run.loc['run_numbers_prog', 'Run']:
        type = 'stock'
        weights = 'prog_numbers_k5twzida0xg2'
        keys = 'prog_keys_k5twzida0xg2'
        arith = 2
        arith_axis = [3,4,5,6,7,8]
        index =[2]
        cols =[0,1]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        numbers_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                               axis_slice=axis_slice)
        numbers_prog = pd.concat([numbers_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_numbers_prog = stacked_numbers_prog.append(numbers_prog)

    
    if report_run.loc['run_numbers_offs', 'Run']:
        type = 'stock'
        weights = 'offs_numbers_k3k5tvnwziaxyg3'
        keys = 'offs_keys_k3k5tvnwziaxyg3'
        arith = 2
        arith_axis = [4,5,6,7,8,9,10,11]
        index =[3]
        cols =[0,1,2]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        numbers_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                               axis_slice=axis_slice)
        numbers_offs = pd.concat([numbers_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_numbers_offs = stacked_numbers_offs.append(numbers_offs)

    
    if report_run.loc['run_dse', 'Run']:
        method = 0
        per_ha = True
        dse = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
        method = 1
        dse1 = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
        dse = pd.concat([dse],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_dse = stacked_dse.append(dse)
        dse1 = pd.concat([dse1],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_dse1 = stacked_dse1.append(dse1)

    
    if report_run.loc['run_grnfoo', 'Run']:
        #returns foo at end of each fp
        type = 'pas'
        prod = 'foo_end_grnha_goflzt'
        weights = 'greenpas_ha_vgoflzt'
        keys = 'keys_vgoflzt'
        arith = 2
        arith_axis = [0,1,2,4,5]
        index =[3]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        grnfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        grnfoo = pd.concat([grnfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_grnfoo = stacked_grnfoo.append(grnfoo)

    
    if report_run.loc['run_dryfoo', 'Run']:
        #returns foo at end of each fp
        type = 'pas'
        prod = 1000
        weights = 'drypas_transfer_dfzt'
        keys = 'keys_dfzt'
        arith = 2
        arith_axis = [0,2]
        index =[1]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        dryfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        dryfoo = pd.concat([dryfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_dryfoo = stacked_dryfoo.append(dryfoo)

    
    if report_run.loc['run_napfoo', 'Run']:
        #returns foo at end of each fp
        prod = 1000
        type = 'pas'
        weights = 'nap_transfer_dfzt'
        keys = 'keys_dfzt'
        arith = 2
        arith_axis = [0,2]
        index =[1]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        napfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        napfoo = pd.concat([napfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_napfoo = stacked_napfoo.append(napfoo)

    
    if report_run.loc['run_grncon', 'Run']:
        #returns consumption in each fp
        prod = 'cons_grnha_t_goflzt'
        type = 'pas'
        weights = 'greenpas_ha_vgoflzt'
        keys = 'keys_vgoflzt'
        arith = 2
        arith_axis = [0,1,2,4,5]
        index =[3]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        grncon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        grncon = pd.concat([grncon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_grncon = stacked_grncon.append(grncon)

    
    if report_run.loc['run_drycon', 'Run']:
        #returns consumption in each fp
        prod = 1000
        type = 'pas'
        weights = 'drypas_consumed_vdfzt'
        keys = 'keys_vdfzt'
        arith = 2
        arith_axis = [0,1,3]
        index =[2]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        drycon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        drycon = pd.concat([drycon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_drycon = stacked_drycon.append(drycon)

    
    if report_run.loc['run_napcon', 'Run']:
        #returns consumption in each fp
        prod = 1000
        type = 'pas'
        weights = 'nap_consumed_vdfzt'
        keys = 'keys_vdfzt'
        arith = 2
        arith_axis = [0,1,3]
        index =[2]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        napcon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        napcon = pd.concat([napcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_napcon = stacked_napcon.append(napcon)

    
    if report_run.loc['run_poccon', 'Run']:
        #returns consumption in each fp
        prod = 1000
        type = 'pas'
        weights = 'poc_consumed_vflz'
        keys = 'keys_vflz'
        arith = 2
        arith_axis = [0,2]
        index =[1]
        cols =[3]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        poccon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
        poccon = pd.concat([poccon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_poccon = stacked_poccon.append(poccon)

    
    if report_run.loc['run_supcon', 'Run']:
        #returns consumption in each fp
        option = 1
        supcon = rep.f_grain_sup_summary(lp_vars, r_vals, option=option)
        supcon = pd.concat([supcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_supcon = stacked_supcon.append(supcon)

    if report_run.loc['run_stubcon', 'Run']:
        #returns consumption in each fp
        stubcon = rep.f_stubble_summary(lp_vars, r_vals)
        stubcon = pd.concat([stubcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
        stacked_stubcon = stacked_stubcon.append(stubcon)

####################################
#run between trial reports and save#
####################################
if report_run.loc['run_summary', 'Run']:
    rep.f_df2xl(writer, stacked_summary, 'summary', option=1)
if report_run.loc['run_areasum', 'Run']:
    rep.f_df2xl(writer, stacked_areasum, 'areasum', option=1)
if report_run.loc['run_pnl', 'Run']:
    rep.f_df2xl(writer, stacked_pnl, 'pnl', option=1)
if report_run.loc['run_profitarea', 'Run']:
    #todo need to fix the plot function
    plot = rep.f_xy_graph(stacked_profitarea)
    plot.savefig('Output/profitarea_curve.png')
if report_run.loc['run_saleprice', 'Run']:
    rep.f_df2xl(writer, stacked_saleprice, 'saleprice', option=1)
if report_run.loc['run_saledate_offs', 'Run']:
    stacked_saledate_offs = stacked_saledate_offs.astype(object)
    stacked_saledate_offs[stacked_saledate_offs==np.datetime64('1970-01-01')] = 0
    rep.f_df2xl(writer, stacked_saledate_offs, 'saledate_offs', option=1)
if report_run.loc['run_cfw_dams', 'Run']:
    rep.f_df2xl(writer, stacked_cfw_dams, 'cfw_dams', option=1)
if report_run.loc['run_fd_dams', 'Run']:
    rep.f_df2xl(writer, stacked_fd_dams, 'fd_dams', option=1)
if report_run.loc['run_wbe_dams', 'Run']:
    rep.f_df2xl(writer, stacked_wbe_dams, 'wbe_dams', option=1)
if report_run.loc['run_wbe_offs', 'Run']:
    rep.f_df2xl(writer, stacked_wbe_offs, 'wbe_offs', option=1)
if report_run.loc['run_lw_dams', 'Run']:
    rep.f_df2xl(writer, stacked_lw_dams, 'lw_dams', option=1)
if report_run.loc['run_ffcfw_dams', 'Run']:
    rep.f_df2xl(writer, stacked_ffcfw_dams, 'ffcfw_dams', option=1)
if report_run.loc['run_fec_dams', 'Run']:
    rep.f_df2xl(writer, stacked_fec_dams, 'fec_dams', option=1)
if report_run.loc['run_ffcfw_prog', 'Run']:
    rep.f_df2xl(writer, stacked_ffcfw_prog, 'ffcfw_prog', option=1)
if report_run.loc['run_fec_offs', 'Run']:
    rep.f_df2xl(writer, stacked_fec_offs, 'fec_offs', option=1)
if report_run.loc['run_lamb_survival', 'Run']:
    rep.f_df2xl(writer, stacked_lamb_survival, 'lamb_survival', option=1)
if report_run.loc['run_weanper', 'Run']:
    rep.f_df2xl(writer, stacked_weanper, 'wean_per', option=1)
if report_run.loc['run_scanper', 'Run']:
    rep.f_df2xl(writer, stacked_scanper, 'scan_per', option=1)
if report_run.loc['run_dry_propn', 'Run']:
    rep.f_df2xl(writer, stacked_dry_propn, 'dry_propn', option=1)
if report_run.loc['run_daily_mei_dams', 'Run']:
    rep.f_df2xl(writer, stacked_daily_mei_dams, 'daily_mei_dams', option=1)
if report_run.loc['run_daily_pi_dams', 'Run']:
    rep.f_df2xl(writer, stacked_daily_pi_dams, 'daily_pi_dams', option=1)
if report_run.loc['run_numbers_dams', 'Run']:
    rep.f_df2xl(writer, stacked_numbers_dams, 'numbers_dams', option=1)
if report_run.loc['run_numbers_prog', 'Run']:
    rep.f_df2xl(writer, stacked_numbers_prog, 'numbers_prog', option=1)
if report_run.loc['run_numbers_offs', 'Run']:
    rep.f_df2xl(writer, stacked_numbers_offs, 'numbers_offs', option=1)
if report_run.loc['run_dse', 'Run']:
    rep.f_df2xl(writer, stacked_dse, 'dse_wt', option=1)
    rep.f_df2xl(writer, stacked_dse1, 'dse_mei', option=1)
if report_run.loc['run_grnfoo', 'Run']:
    rep.f_df2xl(writer, stacked_grnfoo, 'grnfoo', option=1)
if report_run.loc['run_dryfoo', 'Run']:
    rep.f_df2xl(writer, stacked_dryfoo, 'dryfoo', option=1)
if report_run.loc['run_napfoo', 'Run']:
    rep.f_df2xl(writer, stacked_napfoo, 'napfoo', option=1)
if report_run.loc['run_grncon', 'Run']:
    rep.f_df2xl(writer, stacked_grncon, 'grncon', option=1)
if report_run.loc['run_drycon', 'Run']:
    rep.f_df2xl(writer, stacked_drycon, 'drycon', option=1)
if report_run.loc['run_napcon', 'Run']:
    rep.f_df2xl(writer, stacked_napcon, 'napcon', option=1)
if report_run.loc['run_poccon', 'Run']:
    rep.f_df2xl(writer, stacked_poccon, 'poccon', option=1)
if report_run.loc['run_supcon', 'Run']:
    rep.f_df2xl(writer, stacked_supcon, 'supcon', option=1)
if report_run.loc['run_stubcon', 'Run']:
    rep.f_df2xl(writer, stacked_stubcon, 'stubcon', option=1)


writer.save()
