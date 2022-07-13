"""

Background and usage
---------------------
Reports are generated in a two step process. Firstly, each trial undergoes some ‘within trial’ calculations
(e.g. calc the sale price for a trial), the results are stacked together and stored in a table
(each report has its own table). The resulting table can then undergo some ‘between trial’ calculations
(e.g. graphing profit by sale price).

Tweaking the output of a given report is done in the ReportControl.py. Here the user specifies the report
properties. For example, they can specify which axes to report in the table (as either rows or columns in the table)
and which to average or another example. The user can also specify report options. For example, what type of profit
(e.g. is asset opportunity cost included or not) they want to use in the profit by area curve.

To run execute this module with optional args <exp group> <processor number> <report number> <excel display mode>.
If no arguments are passed, all exp groups are reported, 1 processor is used, the 'default' report group is run
and rows/cols that contain only 0's are collapsed when written to excel.

The trials to report are controlled in exp.xl.
The reports to run for a given report number are controlled in exp.xl.

How to add a report:
--------------------
#. Add the required report variables to the ``r_vals`` dictionary - this is done in precalcs
#. Build a 'within trial' report function in Report.py (if an existing one does not meet your criteria)
#. Build a 'between trial' function (if an existing one does not meet your criteria). This is done
   in the 'Final reports' subsection of ReportFunctions.py
#. Add it to exp.xlsx
#. in ReportControl.py add an empty df to stack results
#. in ReportControl.py build the 'within trial' and 'between trial' sections (this can easily be done by copying
   existing code and making the relevant changes).

.. tip:: When creating r_vals values try and do it in obvious spots so it is easier to understand later.

.. note:: For livestock: livestock is slightly more complicated. If you add a r_val or lp_vars you
    will also need to add it to f_stock_reshape the allows you to get the shape correct
    (remove singleton axis and converts lp_vars from dict to numpy).


author: Young
"""


import numpy as np
import pandas as pd
import os
import sys
import multiprocessing
import glob
import time

import ReportFunctions as rep
import Functions as fun


#report the clock time that the experiment was started
print(f'Reporting commenced at: {time.ctime()}')
start = time.time()


##read in excel that controls which reports to run and slice for the selected experiment.
## If no arg passed in or the experiment is not set up with custom col in report_run then default col is used
report_run = pd.read_excel('exp.xlsx', sheet_name='Run Report', index_col=[0], header=[0,1], engine='openpyxl')
try:
    exp_group = int(sys.argv[3])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
except:  # in case no arg passed to python
    exp_group = "Default"
try:
    report_run = report_run.loc[:,('Run',exp_group)]
except KeyError:  # in case the experiment is not set up with custom report_run
    report_run = report_run.loc[:,('Run',"Default")]
report_run = report_run.to_frame()
report_run = report_run.droplevel(1, axis=1)

#todo Reports to add:
# 1. todo add a second mortality report that the weighted average mortality for the animals selected (to complement the current report that is mortality for each w axis)

def f_report(processor, trials, non_exist_trials):
    '''Function to wrap ReportControl.py so that multiprocessing can be used.'''
    # print('Start processor: {0}'.format(processor))
    # print('Start trials: {0}'.format(trials))

    ## A control to switch between reporting the optimised production level True) and the production assumptions (False)
    ### Note this is only active for some of the reports. It also changes the axes that are reported.
    lp_vars_inc = True

    ##create empty df to stack each trial results into
    stacked_infeasible = pd.DataFrame().rename_axis('Trial')  # name of any infeasible trials
    stacked_non_exist = pd.DataFrame(non_exist_trials).rename_axis('Trial')  # name of any infeasible trials
    stacked_summary = pd.DataFrame()  # 1 line summary of each trial
    stacked_areasum = pd.DataFrame()  # area summary
    stacked_pnl = pd.DataFrame()  # profit and loss statement
    stacked_wc = pd.DataFrame()  # max bank overdraw
    stacked_profitarea = pd.DataFrame()  # profit by land area
    stacked_feed = pd.DataFrame()  # feed budget
    stacked_season_nodes = pd.DataFrame()  # season periods
    stacked_feed_periods = pd.DataFrame()  # feed periods
    stacked_dam_dvp_dates = pd.DataFrame()  # dam dvp dates
    stacked_repro_dates = pd.DataFrame()  # dam repro dates
    stacked_offs_dvp_dates = pd.DataFrame()  # offs dvp dates
    stacked_saleprice = pd.DataFrame()  # sale price
    stacked_salegrid_dams = pd.DataFrame()  # sale grid
    stacked_salegrid_yatf = pd.DataFrame()  # sale grid
    stacked_salegrid_offs = pd.DataFrame()  # sale grid
    stacked_salevalue_dams = pd.DataFrame()  # average sale value dams
    stacked_salevalue_offs = pd.DataFrame()  # average sale value offs
    stacked_salevalue_prog = pd.DataFrame()  # average sale value offs
    stacked_woolvalue_dams = pd.DataFrame()  # average wool value dams
    stacked_woolvalue_offs = pd.DataFrame()  # average wool value offs
    stacked_saledate_offs = pd.DataFrame()  # offs sale date
    stacked_saledateEL_offs = pd.DataFrame()  # offs sale date
    stacked_cfw_dams = pd.DataFrame()  # clean fleece weight dams
    stacked_fd_dams = pd.DataFrame()  # fibre diameter dams
    stacked_cfw_offs = pd.DataFrame()  # clean fleece weight dams
    stacked_fd_offs = pd.DataFrame()  # fibre diameter dams
    stacked_wbe_dams = pd.DataFrame()  # whole body energy content dams
    stacked_wbe_offs = pd.DataFrame()  # whole body energy content offs
    stacked_lw_dams = pd.DataFrame()  # live weight dams (large array with p, e and b axis)
    stacked_ffcfw_dams = pd.DataFrame()  # fleece free conceptus free weight dams (large array with p, e and b axis)
    stacked_ffcfw_cut_dams = pd.DataFrame()  # fleece free conceptus free weight dams (large array with p, e and b axis)
    stacked_nv_dams = pd.DataFrame()  # diet nutritive value for dams (large array with p, e and b axis)
    stacked_ffcfw_yatf = pd.DataFrame()  # fleece free conceptus free weight yatf (large array with p, e and b axis)
    stacked_ffcfw_prog = pd.DataFrame()  # fleece free conceptus free weight prog (large array with p, e and b axis)
    stacked_ffcfw_offs = pd.DataFrame()  # fleece free conceptus free weight offs (large array with p, e and b axis)
    stacked_nv_offs = pd.DataFrame()  # diet nutritive value for offs (large array with p, e and b axis)
    stacked_weanper = pd.DataFrame()  # weaning percent
    stacked_scanper = pd.DataFrame()  # scan percent
    stacked_dry_propn = pd.DataFrame()  # dry ewe proportion
    stacked_lamb_survival = pd.DataFrame()  # lamb survival
    stacked_daily_mei_dams = pd.DataFrame()  # mei dams
    stacked_daily_pi_dams = pd.DataFrame()  # potential intake dams
    stacked_daily_mei_offs = pd.DataFrame()  # mei offs
    stacked_daily_pi_offs = pd.DataFrame()  # potential intake offs
    stacked_numbers_dams = pd.DataFrame()  # numbers dams
    stacked_numbers_dams_p = pd.DataFrame()  # numbers dams with p axis (large array)
    stacked_numbers_prog = pd.DataFrame()  # numbers prog
    stacked_numbers_offs = pd.DataFrame()  # numbers offs
    stacked_numbers_offs_p = pd.DataFrame()  # numbers offs with p axis (large array)
    stacked_mort_dams = pd.DataFrame()  # mort dams with p axis (large array)
    stacked_mort_offs = pd.DataFrame()  # mort offs with p axis (large array)
    stacked_dse_sire = pd.DataFrame()  # dse based on normal weight
    stacked_dse_dams = pd.DataFrame()  # dse based on normal weight
    stacked_dse_offs = pd.DataFrame()  # dse based on normal weight
    stacked_dse1_sire = pd.DataFrame()  # dse based on mei
    stacked_dse1_dams = pd.DataFrame()  # dse based on mei
    stacked_dse1_offs = pd.DataFrame()  # dse based on mei
    stacked_pgr = pd.DataFrame()  # pasture growth
    stacked_grnfoo = pd.DataFrame()  # green foo
    stacked_dryfoo = pd.DataFrame()  # dry foo
    stacked_napfoo = pd.DataFrame()  # non-arable pasture foo
    stacked_grncon = pd.DataFrame()  # green pasture consumed
    stacked_drycon = pd.DataFrame()  # dry pasture consumed
    stacked_napcon = pd.DataFrame()  # non-arable pasture feed consumed
    stacked_poccon = pd.DataFrame()  # pasture on crop paddocks feed consumed
    stacked_supcon = pd.DataFrame()  # supplement feed consumed
    stacked_stubcon = pd.DataFrame()  # stubble feed consumed
    stacked_grnnv = pd.DataFrame()  # NV of green pas
    stacked_grndmd = pd.DataFrame()  # dmd of green pas
    stacked_avegrnfoo = pd.DataFrame()  # Average Foo of green pas
    stacked_drynv = pd.DataFrame()  # NV of dry pas
    stacked_drydmd = pd.DataFrame()  # dmd of dry pas
    stacked_avedryfoo = pd.DataFrame()  # Average Foo of dry pas
    stacked_mvf = pd.DataFrame()  # Average Foo of dry pas

    ##read in the pickled results
    for trial_name in trials:
        lp_vars,r_vals = rep.load_pkl(trial_name)

        ##handle infeasible trials
        if os.path.isfile('Output/infeasible/{0}.txt'.format(trial_name)):
            stacked_infeasible = rep.f_append_dfs(stacked_infeasible, pd.DataFrame([trial_name]).rename_axis('Trial'))
            lp_vars = fun.f_clean_dict(lp_vars) #if a trial is infeasible or doesn't solve all the lp values are None. This function converts them to 0 so the report can still run.

        ##run report functions
        if report_run.loc['run_summary', 'Run']:
            summary = rep.f_summary(lp_vars,r_vals,trial_name)
            summary.index.name = 'Trial'
            stacked_summary = rep.f_append_dfs(stacked_summary, summary)

        if report_run.loc['run_areasum', 'Run']:
            option = 0
            areasum = rep.f_area_summary(lp_vars, r_vals, option=option)
            areasum = pd.concat([areasum],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_areasum = rep.f_append_dfs(stacked_areasum, areasum)

        if report_run.loc['run_pnl', 'Run']:
            pnl = rep.f_profitloss_table(lp_vars, r_vals)
            pnl = pd.concat([pnl],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_pnl = rep.f_append_dfs(stacked_pnl, pnl)

        if report_run.loc['run_wc', 'Run']:
            wc = rep.f_wc_summary(lp_vars, r_vals)
            wc = pd.concat([wc],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_wc = rep.f_append_dfs(stacked_wc, wc)

        if report_run.loc['run_profitarea', 'Run']:
            area_option = 3
            profit_option = 0
            profitarea = pd.DataFrame(index=[trial_name], columns=['area','profit'])
            profitarea.loc[trial_name, 'area'] = rep.f_area_summary(lp_vars,r_vals,area_option)
            profitarea.loc[trial_name,'profit'] = rep.f_profit(lp_vars,r_vals,profit_option)
            stacked_profitarea = rep.f_append_dfs(stacked_profitarea, profitarea)

        if report_run.loc['run_feedbudget', 'Run']:
            option = 0
            nv_option = 0
            dams_cols = [6] #birth opp
            offs_cols = [7] #shear opp
            feed = rep.f_feed_budget(lp_vars, r_vals, option=option, nv_option=nv_option, dams_cols=dams_cols, offs_cols=offs_cols)
            feed = pd.concat([feed],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_feed = rep.f_append_dfs(stacked_feed, feed)

        if report_run.loc['run_period_dates', 'Run']:
            ###season nodes (p7)
            type = 'zgen'
            prod = 'date_season_node_p7z'
            keys = 'keys_p7z'
            arith = 0
            index =[0] #p7
            cols = [1] #z
            season_nodes = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                       keys=keys, arith=arith, index=index, cols=cols)
            season_nodes = pd.concat([season_nodes],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_season_nodes = rep.f_append_dfs(stacked_season_nodes, season_nodes)

            ###feed periods (p6)
            type = 'pas'
            prod = 'fp_date_start_p6z'
            keys = 'keys_p6z'
            arith = 0
            index =[0] #p6
            cols = [1] #z
            feed_periods = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                       keys=keys, arith=arith, index=index, cols=cols)
            feed_periods = pd.concat([feed_periods],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_feed_periods = rep.f_append_dfs(stacked_feed_periods, feed_periods)

            ###dams dvp
            type = 'stock'
            prod = 'dvp_start_vezg1'
            keys = 'dams_keys_vezg1'
            arith = 0
            index =[0] #v
            cols = [1,3,2] #e, g, z
            dam_dvp_dates = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                       keys=keys, arith=arith, index=index, cols=cols)
            dam_dvp_dates = pd.concat([dam_dvp_dates],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dam_dvp_dates = rep.f_append_dfs(stacked_dam_dvp_dates, dam_dvp_dates)

            ###dams repro dates
            type = 'stock'
            prod = 'r_repro_dates_roe1g1'
            keys = 'dams_keys_roeg1'
            arith = 0
            index =[1] #o
            cols = [0,2,3] #r, e, g
            repro_dates = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                       keys=keys, arith=arith, index=index, cols=cols)
            repro_dates = pd.concat([repro_dates],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_repro_dates = rep.f_append_dfs(stacked_repro_dates, repro_dates)

            ###offs dvp
            type = 'stock'
            prod = 'dvp_start_vzdxg3'
            keys = 'offs_keys_vzdxg3'
            arith = 0
            index =[0] #v
            cols = [4,3,2,1] #g, x, d, z
            offs_dvp_dates = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                       keys=keys, arith=arith, index=index, cols=cols)
            offs_dvp_dates = pd.concat([offs_dvp_dates],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_offs_dvp_dates = rep.f_append_dfs(stacked_offs_dvp_dates, offs_dvp_dates)

        if report_run.loc['run_saleprice', 'Run']:
            option = 2
            grid = [0,5,6]
            weight = [22,40,25]
            fs = [2,3,2]
            saleprice = rep.f_price_summary(lp_vars, r_vals, option=option, grid=grid, weight=weight, fs=fs)
            saleprice = pd.concat([saleprice],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_saleprice = rep.f_append_dfs(stacked_saleprice, saleprice)

        if report_run.loc['run_salegrid_dams', 'Run']:
            type = 'stock'
            prod = 'salegrid_tva1e1b1nwziyg1'
            keys = 'dams_keys_tva1e1b1nwziyg1'
            arith = 0
            index =[1]
            cols = [10,0,4,6] #g,t,b,w
            salegrid_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, 
                                   keys=keys, arith=arith, index=index, cols=cols)
            salegrid_dams = pd.concat([salegrid_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salegrid_dams = rep.f_append_dfs(stacked_salegrid_dams, salegrid_dams)

        if report_run.loc['run_salegrid_yatf', 'Run']:
            type = 'stock'
            prod = 'salegrid_Tva1e1b1nwzixyg2'
            keys = 'yatf_keys_Tvaebnwzixy1g2'
            arith = 0
            index =[1]
            cols = [11,9,0,4,6] #g,x,t,b,w
            salegrid_yatf = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                   keys=keys, arith=arith, index=index, cols=cols)
            salegrid_yatf = pd.concat([salegrid_yatf],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salegrid_yatf = rep.f_append_dfs(stacked_salegrid_yatf, salegrid_yatf)

        if report_run.loc['run_salegrid_offs', 'Run']:
            type = 'stock'
            prod = 'salegrid_tvnwzida0e0b0xyg3'
            keys = 'offs_keys_tvnwzida0e0b0xyg3'
            arith = 0
            index =[1]
            cols = [0,3,8,9,10,12] #t,w,e,b,x,g
            salegrid_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, 
                                   keys=keys, arith=arith, index=index, cols=cols)
            salegrid_offs = pd.concat([salegrid_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salegrid_offs = rep.f_append_dfs(stacked_salegrid_offs, salegrid_offs)

        if report_run.loc['run_salevalue_dams', 'Run']:
            type = 'stock'
            prod = 'salevalue_k2p7tva1nwziyg1'
            na_prod = [0,1]  # q,s
            keys = 'dams_keys_qsk2p7tvanwziy1g1'
            if lp_vars_inc:
                weights = 'dams_numbers_qsk2tvanwziy1g1'
                na_weights = [3]  #p7
                arith = 1
                index =[5]      #v
                cols =[2,3,4]    #k2, p7, t
            else:
                weights = None
                na_weights = []
                arith = 4
                index =[5] #v
                cols =[12,3,4,2,8] #g,p7,t,k2,w
            # axis_slice = {}
            # axis_slice[3] = [0, 1, 1]   #c0: stk
            salevalue_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            salevalue_dams = pd.concat([salevalue_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salevalue_dams = rep.f_append_dfs(stacked_salevalue_dams, salevalue_dams)

        if report_run.loc['run_salevalue_offs', 'Run']:
            type = 'stock'
            prod = 'salevalue_k3k5p7tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            keys = 'offs_keys_qsk3k5p7tvnwziaxyg3'
            if lp_vars_inc:
                weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
                na_weights = [4]  #p7
                arith = 1
                index =[6, 12]      #v, x
                cols =[4, 14, 5]    #cashflow period, g3, t
            else:
                weights = None
                na_weights = []
                arith = 4
                index = [6, 12]     #DVP, gender
                cols = [4, 14, 5, 8]   #cashflow period, g3, t, w
            # axis_slice = {}
            # axis_slice[4] = [0, 1, 1]   #c0: stk
            salevalue_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            salevalue_offs = pd.concat([salevalue_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salevalue_offs = rep.f_append_dfs(stacked_salevalue_offs, salevalue_offs)

        if report_run.loc['run_salevalue_prog', 'Run']:
            type = 'stock'
            prod = 'salevalue_k3k5p7twzia0xg2'
            na_prod = [0,1]  # q,s
            keys = 'prog_keys_qsk3k5p7twzia0xg2'
            if lp_vars_inc:
                weights = 'prog_numbers_qsk3k5twzia0xg2'
                na_weights = [4]  #p7
                arith = 1
                index =[6]      #w
                cols =[4, 11, 5]    #cashflow period, g2, t
            else:
                weights = None
                na_weights = []
                arith = 4
                index = [6]     #w
                cols = [4, 11, 5]   #cashflow period, g2, t
            # axis_slice = {}
            # axis_slice[4] = [0, 1, 1]   #c0: stk
            salevalue_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            salevalue_prog = pd.concat([salevalue_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_salevalue_prog = rep.f_append_dfs(stacked_salevalue_prog, salevalue_prog)

        if report_run.loc['run_woolvalue_dams', 'Run']:
            type = 'stock'
            prod = 'woolvalue_k2p7tva1nwziyg1'
            na_prod = [0,1]  # q,s
            keys = 'dams_keys_qsk2p7tvanwziy1g1'
            if lp_vars_inc:
                weights = 'dams_numbers_qsk2tvanwziy1g1'
                na_weights = [3]  #p7
                arith = 1
                index =[5]      #v
                cols =[2,3,4]    #k2, p7, t
            else:
                weights = None
                na_weights = []
                arith = 4
                index =[5] #v
                cols =[12,3,4,2,8] #g,p7,t,k2,w
            # axis_slice = {}
            # axis_slice[3] = [0, 1, 1]   #c0: stk
            woolvalue_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            woolvalue_dams = pd.concat([woolvalue_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_woolvalue_dams = rep.f_append_dfs(stacked_woolvalue_dams, woolvalue_dams)

        if report_run.loc['run_woolvalue_offs', 'Run']:
            type = 'stock'
            prod = 'woolvalue_k3k5p7tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [4] #p7
            keys = 'offs_keys_qsk3k5p7tvnwziaxyg3'
            arith = 1
            index = [6, 12]     #DVP, gender
            cols = [4, 14, 5]   #cashflow period, g3, t
            # axis_slice = {}
            # axis_slice[4] = [0, 1, 1]   #c0: stk
            woolvalue_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
            woolvalue_offs = pd.concat([woolvalue_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_woolvalue_offs = rep.f_append_dfs(stacked_woolvalue_offs, woolvalue_offs)

        if report_run.loc['run_saledate_offs', 'Run']:
            type = 'stock'
            prod = 'saledate_k3k5tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 1
            index = [5, 7]              #DVP, w
            cols = [13, 2, 3, 4, 11]     #g3, dam age, BTRT, t, gender
            saledate_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols)
            saledate_offs = pd.concat([saledate_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_saledate_offs = rep.f_append_dfs(stacked_saledate_offs, saledate_offs)

        if report_run.loc['run_saledateEL_offs', 'Run']:
            type = 'stock'
            prod = 'saledate_k3k5tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 1               #weighted average on numbers
            index = [0]             #q so that it only has 1 row
            cols = [13]             #g3
            axis_slice = {}
            axis_slice[4] = [1,None,1] #t: only the sale slices
            axis_slice[11] = [1,2,1] #x: Castrate

            saledateEL_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice).astype('datetime64[D]')
            saledateEL_offs = pd.concat([saledateEL_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_saledateEL_offs = rep.f_append_dfs(stacked_saledateEL_offs, saledateEL_offs)

        if report_run.loc['run_cfw_dams', 'Run']:
            type = 'stock'
            prod = 'cfw_hdmob_k2tva1nwziyg1'
            na_prod = [0,1]  # q,s
            keys = 'dams_keys_qsk2tvanwziy1g1'
            if lp_vars_inc:
                weights = 'dams_numbers_qsk2tvanwziy1g1'
                na_weights = []
                arith = 1
                index =[4]      #v
                cols =[2,3]    #k2, t
            else:
                weights = None
                na_weights = []
                arith = 4
                index =[4] #v
                cols =[11,3,2,8] #g,t,k2,w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]   #LSLN: NM & 00
            cfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            cfw_dams = pd.concat([cfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_cfw_dams = rep.f_append_dfs(stacked_cfw_dams, cfw_dams)

        if report_run.loc['run_fd_dams', 'Run']:
            type = 'stock'
            prod = 'fd_hdmob_k2tva1nwziyg1'
            na_prod = [0,1]  # q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            keys = 'dams_keys_qsk2tvanwziy1g1'
            arith = 1
            index =[4] #v
            cols =[2,3] #k,t
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            fd_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            fd_dams = pd.concat([fd_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_fd_dams = rep.f_append_dfs(stacked_fd_dams, fd_dams)

        if report_run.loc['run_cfw_offs', 'Run']:
            type = 'stock'
            prod = 'cfw_hdmob_k3k5tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 1
            index = [5]          #DVP
            cols = [13, 2, 3, 4, 11]  #g3, dam age, BTRT, t, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            cfw_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            cfw_offs = pd.concat([cfw_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_cfw_offs = rep.f_append_dfs(stacked_cfw_offs, cfw_offs)

        if report_run.loc['run_fd_offs', 'Run']:
            type = 'stock'
            prod = 'fd_hdmob_k3k5tvnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 1
            index = [5]                 #DVP
            cols = [13, 2, 3, 4, 11]     #g3, dam age, BTRT, t, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            fd_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            fd_offs = pd.concat([fd_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_fd_offs = rep.f_append_dfs(stacked_fd_offs, fd_offs)

        if report_run.loc['run_wbe_dams', 'Run']:
            type = 'stock'
            prod = 'wbe_k2tva1nwziyg1'
            na_prod = [0,1] #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            keys = 'dams_keys_qsk2tvanwziy1g1'
            arith = 1
            index = [4] #v
            cols = [2] #k2
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            wbe_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            wbe_dams = pd.concat([wbe_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_wbe_dams = rep.f_append_dfs(stacked_wbe_dams, wbe_dams)

        if report_run.loc['run_wbe_offs', 'Run']:
            type = 'stock'
            prod = 'wbe_k3k5tvnwziaxyg3'
            na_prod = [0,1] #q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 1
            index = [5]             #DVP
            cols = [13, 2, 3, 4]    #g3, dam age, BTRT, t
            axis_slice = {}
            wbe_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            wbe_offs = pd.concat([wbe_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_wbe_offs = rep.f_append_dfs(stacked_wbe_offs, wbe_offs)

        if report_run.loc['run_lw_dams', 'Run']:
            ##Average dam lw with p, e & b axis. Lw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'lw_dams_k2Tvpa1e1b1nw8ziyg1'
            na_prod = [0,1] #q,s
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0,1] #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5,7,8]  #p,e,b
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0,1] #q,s
            keys = 'dams_keys_qsk2tvpaebnwziy1g1'
            arith = 1
            index = [5] #p
            cols = [8] #b
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            lw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                     , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights, den_weights=den_weights, na_denweights=na_denweights
                                     , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            lw_dams = pd.concat([lw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_lw_dams = rep.f_append_dfs(stacked_lw_dams, lw_dams)

        if report_run.loc['run_ffcfw_dams', 'Run']:
            ##Average dam ffcfw with p, e & b axis. ffcfw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'ffcfw_dams_k2Tvpa1e1b1nw8ziyg1'
            na_prod = [0,1] #q,s
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0,1] #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5, 7, 8]
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0,1] #q,s
            keys = 'dams_keys_qsk2tvpaebnwziy1g1'
            arith = 1
            if lp_vars_inc:
                index = [5] #p
                cols = [8] #b
            else:    #adding the w axis while still reporting with lp_vars weighting
                index = [5] #p
                cols = [2,3,10] #b,w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            ffcfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                     , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights
                                     , den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith
                                     , index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_dams = pd.concat([ffcfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_dams = rep.f_append_dfs(stacked_ffcfw_dams, ffcfw_dams)

        #todo remove after ewelamb analysis
        if report_run.loc['run_ffcfw_cut_dams', 'Run']:
            type = 'stock'
            prod = 'ffcfw_dams_k2tvPa1nw8ziyg1'
            na_prod = [0,1] #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5]#p
            keys = 'dams_keys_qsk2tvPanwziy1g1'
            arith = 1
            index = [5] #p
            cols = [4] #v
            axis_slice = {}
            axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
            axis_slice[4] = [0, 7, 1]     #DVPs 0 to 6 inclusive
            ffcfw_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                     , na_weights=na_weights, keys=keys, arith=arith
                                     , index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_dams = pd.concat([ffcfw_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_cut_dams = rep.f_append_dfs(stacked_ffcfw_cut_dams, ffcfw_dams)

        if report_run.loc['run_nv_dams', 'Run']:
            ##Average dam NV with p, e & b axis. NV is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'nv_dams_k2Tvpa1e1b1nw8ziyg1'
            na_prod = [0,1] #q,s
            prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0, 1]  #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5, 7, 8]
            den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0, 1]  #q,s
            keys = 'dams_keys_qsk2tvpaebnwziy1g1'
            arith = 1
            if lp_vars_inc:
                index = [5]  #p
                cols = [14, 7, 3, 8]  #g1, e, t & b1
            else:
                index =[5] #p
                cols =[14,7,3,8,10] #g,e,t,b1,w
            axis_slice = {}
            # axis_slice[5] = [0, 1, 1]   #only the first cycle of e1
            nv_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod,
                                   prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights, na_weights=na_weights,
                                   den_weights=den_weights, na_denweights=na_denweights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            nv_dams = pd.concat([nv_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_nv_dams = rep.f_append_dfs(stacked_nv_dams, nv_dams)

        if report_run.loc['run_ffcfw_yatf', 'Run']:
            ##Average yatf ffcfw with p, e & b axis. ffcfw is not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            ##For yatf the denom weight also includes a weighting for nyatf. The numerator also gets weighted by this.
            ##v_dam must be used because v_prog has a different w axis than yatf.
            type = 'stock'
            prod = 'ffcfw_yatf_k2Tvpa1e1b1nw8zixyg1'
            na_prod = [0,1] #q,s
            prod_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0,1] #q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5, 7, 8, 13]                  #p, e1, b1, x
            den_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0,1] #q,s
            keys = 'yatf_keys_qsk2tvpaebnwzixy1g1'
            arith = 1
            index = [5]     #p
            cols = [15, 10]  #g2, w8
            axis_slice = {}
            ffcfw_yatf = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                     , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys
                                     , arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_yatf = pd.concat([ffcfw_yatf],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_yatf = rep.f_append_dfs(stacked_ffcfw_yatf, ffcfw_yatf)

        if report_run.loc['run_ffcfw_prog', 'Run']:
            type = 'stock'
            prod = 'ffcfw_prog_k3k5wzida0e0b0xyg2'
            na_prod = [0,1,4] #q,s,t
            prod_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight prod for propn of animals in e and b slice
            na_prodweights = [0,1] #q,s
            weights = 'prog_numbers_qsk3k5twzia0xg2'
            na_weights = [8,10,11,13] #d, e,b,y
            den_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight numbers for propn of animals in e and b slice
            na_denweights = [0,1] #q,s
            keys = 'prog_keys_qsk3k5twzida0e0b0xyg2'
            arith = 1
            index = [5]             #w9
            cols = [3,6,12,14]  #k2, z, gender, g2
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            ffcfw_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod
                                                     , prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights
                                                     , na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                                     , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_prog = pd.concat([ffcfw_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_prog = rep.f_append_dfs(stacked_ffcfw_prog, ffcfw_prog)

        if report_run.loc['run_ffcfw_offs', 'Run']:
            ##Average offs ffcfw with p, e & b axis. ffcfw is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'ffcfw_offs_k3k5Tvpnw8zida0e0b0xyg3'
            na_prod = [0,1] #q,s
            prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0,1] #q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [6, 11, 13, 14]
            den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0,1] #q,s
            keys = 'offs_keys_qsk3k5tvpnwzidaebxyg3'
            arith = 1
            index = [3, 6]      #k5, p. k5 here to save columns when many w
            cols = [17, 15, 4, 8]   #g3, x, t, w
            axis_slice = {}
            axis_slice[13] = [0,1,1] #e: first cycle
            axis_slice[11] = [2,-1,1] #dam age: Adult
            # axis_slice[15] = [0,1,1] #g3: BBB
            ffcfw_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                     , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                     , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            ffcfw_offs = pd.concat([ffcfw_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_ffcfw_offs = rep.f_append_dfs(stacked_ffcfw_offs, ffcfw_offs)

        if report_run.loc['run_nv_offs', 'Run']:
            ##Average offs NV with p, e & b axis. NV is adjusted for animals that are sold but not adjusted by mortality
            ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
            ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
            type = 'stock'
            prod = 'nv_offs_k3k5Tvpnw8zida0e0b0xyg3'
            na_prod = [0,1] #q,s
            prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_prodweights = [0,1] #q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [6,11,13,14] #p,d,e,b
            den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
            na_denweights = [0,1] #q, s
            keys = 'offs_keys_qsk3k5tvpnwzidaebxyg3'
            arith = 1
            index =[6] #p
            cols =[2,15,3,4,8]  #k3,x,k5,t,w
            axis_slice = {}
            axis_slice[13] = [0,1,1] #first cycle
            axis_slice[11] = [2,-1,1] #Adult
            axis_slice[17] = [0,1,1] #BBB
            nv_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                   , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                   , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            nv_offs = pd.concat([nv_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_nv_offs = rep.f_append_dfs(stacked_nv_offs, nv_offs)

        if report_run.loc['run_lamb_survival', 'Run']:
            #todo the calculations underpinning this report are still not correct (23May22). Particularly for Maternals
            #axes are qsk2tvaeb9nwziy1g1      b9 axis is shorten b axis: [0,1,2,3]
            option = 0
            if lp_vars_inc:
                index =[4]      #v
                cols =[13,11,0,1,10,7]    #g,i,q,s,z & b9 #report must include the b axis otherwise an error is caused because the axis added after the arith.
            else:
                index = [4]     #v
                cols =[13,11,0,1,10,7,9]  #g,i,q,s,z,b & w
            axis_slice = {}
            lamb_survival = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                                 , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
            lamb_survival = pd.concat([lamb_survival],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_lamb_survival = rep.f_append_dfs(stacked_lamb_survival, lamb_survival)

        if report_run.loc['run_weanper', 'Run']:
            #todo there is an error here if drys are sold at scanning. We can't think of an easy way to fix it. (note if scan=4 then birth dvp may be different across e axis)
            #with the current structure w CANNOT be reported. 23Apr22 - seems to be working when not using lp_vars
            # problem could be that dams can change w slice between joining (nfoet) and lambing (nyatf)
            #axes are qsk2tvanwziy1g1
            option = 1
            if lp_vars_inc:
                index =[4]      #v
                cols =[11,9,0,1,8]   #g,i,q,s & z [11,2]      #g & k2 (needs k2 in the current form).
            else:
                index = [4]     #v
                cols = [11,9,0,1,8,7]    #g,i,q,s,z & w  Makes most sense to report all the axes that are individual animals (k2 optional here)
            axis_slice = {}
            weanper = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                           , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
            weanper = pd.concat([weanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_weanper = rep.f_append_dfs(stacked_weanper, weanper)

        if report_run.loc['run_scanper', 'Run']:
            #axes are qsk2tvanwziy1g1
            option = 2
            if lp_vars_inc:
                index =[4]      #v
                cols =[11,9,0,1,8]   #g,i,q,s & z [11,2]      #g & k2 (needs k2 in the current form).
            else:
                index = [4]     #v
                cols = [11,9,0,1,8,7]    #g,i,q,s,z & w  Makes most sense to report all the axes that are individual animals
            axis_slice = {}
            scanper = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                           , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
            scanper = pd.concat([scanper],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_scanper = rep.f_append_dfs(stacked_scanper, scanper)

        if report_run.loc['run_dry_propn', 'Run']:
            #axes are qsk2tvanwziy1g1
            option = 3
            if lp_vars_inc:
                index =[4]      #v
                cols =[11,9,0,1,8]   #g,i,q,s & z [11,2]      #g & k2 (needs k2 in the current form).
            else:
                index = [4]     #v
                cols = [11,9,0,1,8,7]    #g,i,q,s,z & w  Makes most sense to report all the axes that are individual animals
            axis_slice = {}
            dry_propn = rep.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                             , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
            dry_propn = pd.concat([dry_propn],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dry_propn = rep.f_append_dfs(stacked_dry_propn, dry_propn)

        if report_run.loc['run_daily_mei_dams', 'Run']:
            type = 'stock'
            prod = 'mei_dams_k2p6ftva1nw8ziyg1'
            na_prod = [0,1]  # q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [3, 4]
            den_weights = 'stock_days_k2p6ftva1nwziyg1'
            na_denweights = [0,1] #q,s
            keys = 'dams_keys_qsk2p6ftvanwziy1g1'
            arith = 1               # for FP only
            index = [6, 3]          # [1]
            cols = [2, 4]           # [0]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_mei_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
            daily_mei_dams = pd.concat([daily_mei_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_mei_dams = rep.f_append_dfs(stacked_daily_mei_dams, daily_mei_dams)

        if report_run.loc['run_daily_pi_dams', 'Run']:
            type = 'stock'
            prod = 'pi_dams_k2p6ftva1nw8ziyg1'
            na_prod = [0,1]  # q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [3, 4]
            den_weights = 'stock_days_k2p6ftva1nwziyg1'
            na_denweights = [0,1] #q,s
            keys = 'dams_keys_qsk2p6ftvanwziy1g1'
            arith = 1
            index =[6, 3] #v,p6
            cols =[2, 4] #k2, f
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_pi_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                       index=index, cols=cols, axis_slice=axis_slice)
            daily_pi_dams = pd.concat([daily_pi_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_pi_dams = rep.f_append_dfs(stacked_daily_pi_dams, daily_pi_dams)

        if report_run.loc['run_daily_mei_offs', 'Run']:
            type = 'stock'
            prod = 'mei_offs_k3k5p6ftvnw8ziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [4, 5]
            den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
            na_denweights = [0,1] #q,s
            keys = 'offs_keys_qsk3k5p6ftvnwziaxyg3'
            arith = 1
            index =[7, 4]       #DVP, fp
            cols =[15, 3, 5]    #g3, BTRT, nv
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_mei_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
            daily_mei_offs = pd.concat([daily_mei_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_mei_offs = rep.f_append_dfs(stacked_daily_mei_offs, daily_mei_offs)

        if report_run.loc['run_daily_pi_offs', 'Run']:
            type = 'stock'
            prod = 'pi_offs_k3k5p6ftvnw8ziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [4, 5]
            den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
            na_denweights = [0,1] #q,s
            keys = 'offs_keys_qsk3k5p6ftvnwziaxyg3'
            arith = 1
            index =[7, 4]       #DVP, fp
            cols =[15, 3, 5]    #g3, BTRT, nv
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            daily_pi_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                       index=index, cols=cols, axis_slice=axis_slice)
            daily_pi_offs = pd.concat([daily_pi_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_daily_pi_offs = rep.f_append_dfs(stacked_daily_pi_offs, daily_pi_offs)

        if report_run.loc['run_numbers_dams', 'Run']:
            type = 'stock'
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            keys = 'dams_keys_qsk2tvanwziy1g1'
            arith = 2
            if lp_vars_inc:
                index =[4]      #v
                cols =[0,1,8, 2, 3] #q, s, z, k2, t
            else:
                index =[4] #v
                cols =[0,1,8, 2, 3, 7] #q, s, z, k2, t, w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_dams = pd.concat([numbers_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_dams = rep.f_append_dfs(stacked_numbers_dams, numbers_dams)

        if report_run.loc['run_numbers_dams_p', 'Run']:
            type = 'stock'
            prod = 'on_hand_mort_k2tvpa1nwziyg1'
            na_prod = [0,1]  # q,s
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [5]
            keys = 'dams_keys_qsk2tvpanwziy1g1'
            arith = 2
            index =[4, 5]
            cols =[2, 3, 8]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_dams_p = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_dams_p = pd.concat([numbers_dams_p],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_dams_p = rep.f_append_dfs(stacked_numbers_dams_p, numbers_dams_p)

        if report_run.loc['run_numbers_prog', 'Run']:
            type = 'stock'
            weights = 'prog_numbers_qsk3k5twzia0xg2'
            keys = 'prog_keys_qsk3k5twzia0xg2'
            arith = 2
            index =[0]   #q (as a dummy variable)
            cols =[10, 9, 2, 4] #dam age(2), birth type(3), t slice(4), gender(9), genotype(10)
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_prog = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_prog = pd.concat([numbers_prog],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_prog = rep.f_append_dfs(stacked_numbers_prog, numbers_prog)

        if report_run.loc['run_numbers_offs', 'Run']:
            type = 'stock'
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            keys = 'offs_keys_qsk3k5tvnwziaxyg3'
            arith = 2
            index =[5]                  #DVP
            cols =[13, 11, 2, 4]   #g3, Gender, dam age, BTRT, t, w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            numbers_offs = pd.concat([numbers_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_offs = rep.f_append_dfs(stacked_numbers_offs, numbers_offs)
    #todo numbers_p has a bit of an error for dams and offs (numbers seem to reduce a bit quickly)
        if report_run.loc['run_numbers_offs_p', 'Run']:
            type = 'stock'
            prod = 'on_hand_mort_k3k5tvpnwziaxyg3'
            na_prod = [0,1]  # q,s
            weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
            na_weights = [6]
            keys = 'offs_keys_qsk3k5tvpnwziaxyg3'
            arith = 2
            index =[6]              #p
            cols =[14, 2,12,3,4,8]  #genotype, dam age, gender, BTRT, t, w
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            numbers_offs_p = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            numbers_offs_p = pd.concat([numbers_offs_p],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_numbers_offs_p = rep.f_append_dfs(stacked_numbers_offs_p, numbers_offs_p)

        if report_run.loc['run_mort_dams', 'Run']:
            ##note t axis will be singleton unless stock generator was run with active t axis.
            type = 'stock'
            prod = 'mort_Tpa1e1b1nwziyg1' #uses b axis instead of k for extra detail when scan=0
            weights = None
            na_weights = []
            keys = 'dams_keys_Tpaebnwziy1g1'
            arith = 4
            index =[1]          #period
            cols =[10, 4, 6]     #genotype, LSLN, w
            axis_slice = {}
            # axis_slice[0] = [2, 3, 1]
            # axis_slice[1] = [2, 4, 1]
            mort_dams = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            mort_dams = pd.concat([mort_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_mort_dams = rep.f_append_dfs(stacked_mort_dams, mort_dams)

        if report_run.loc['run_mort_offs', 'Run']:
            ##note t axis will be singleton unless stock generator was run with active t axis.
            type = 'stock'
            prod = 'mort_Tpnwzida0e0b0xyg3' #uses b axis instead of k for extra detail when scan=0
            weights = None
            na_weights = []
            keys = 'offs_keys_Tpnwzidaebxyg3'
            arith = 4
            index =[1]              #p
            cols =[12, 9, 3, 10]     #g3, BTRT, w, gender
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            mort_offs = rep.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                                   na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                                   axis_slice=axis_slice)
            mort_offs = pd.concat([mort_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_mort_offs = rep.f_append_dfs(stacked_mort_offs, mort_offs)

        if report_run.loc['run_dse', 'Run']:
            ##you can go into f_dse to change the axis being reported.
            per_ha = True

            method = 0
            dse_sire, dse_dams, dse_offs = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
            dse_sire = pd.concat([dse_sire],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse_dams = pd.concat([dse_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse_offs = pd.concat([dse_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dse_sire = rep.f_append_dfs(stacked_dse_sire, dse_sire)
            stacked_dse_dams = rep.f_append_dfs(stacked_dse_dams, dse_dams)
            stacked_dse_offs = rep.f_append_dfs(stacked_dse_offs, dse_offs)

            method = 1
            dse1_sire, dse1_dams, dse1_offs = rep.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
            dse1_sire = pd.concat([dse1_sire],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse1_dams = pd.concat([dse1_dams],keys=[trial_name],names=['Trial'])  # add trial name as index level
            dse1_offs = pd.concat([dse1_offs],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dse1_sire = rep.f_append_dfs(stacked_dse1_sire, dse1_sire)
            stacked_dse1_dams = rep.f_append_dfs(stacked_dse1_dams, dse1_dams)
            stacked_dse1_offs = rep.f_append_dfs(stacked_dse1_offs, dse1_offs)

        if report_run.loc['run_grnfoo', 'Run']:
            #returns foo at end of each FP
            type = 'pas'
            prod = 'foo_end_grnha_gop6lzt'
            na_prod = [0,1,2] #q,s,f
            weights = 'greenpas_ha_qsfgop6lzt'
            keys = 'keys_qsfgop6lzt'
            arith = 2
            index =[5]
            cols =[6]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            grnfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnfoo = pd.concat([grnfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grnfoo = rep.f_append_dfs(stacked_grnfoo, grnfoo)

        if report_run.loc['run_pgr', 'Run']:
            #returns average pgr per day per ha.
            #to get total PG change arith to 2 and remove den_weights
            type = 'pas'
            prod = 'pgr_grnha_gop6lzt'
            na_prod = [0,1,2] #q,s,f
            weights = 'greenpas_ha_qsfgop6lzt'
            den_weights = 'days_p6z'
            na_denweights = [1,3]
            keys = 'keys_qsfgop6lzt'
            arith = 1
            index =[5]
            cols =[6]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            pgr = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights, den_weights=den_weights,
                                   na_denweights=na_denweights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            pgr = pd.concat([pgr],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_pgr = rep.f_append_dfs(stacked_pgr, pgr)

        if report_run.loc['run_dryfoo', 'Run']:
            #returns foo at end of each FP
            type = 'pas'
            prod = np.array([1000])
            weights = 'drypas_transfer_qsdp6zlt'
            keys = 'keys_qsdp6zlt'
            arith = 2
            index =[3]
            cols =[2]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            dryfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            dryfoo = pd.concat([dryfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_dryfoo = rep.f_append_dfs(stacked_dryfoo, dryfoo)

        if report_run.loc['run_napfoo', 'Run']:
            #returns foo at end of each FP
            type = 'pas'
            prod = np.array([1000])
            weights = 'nap_transfer_qsdp6zt'
            keys = 'keys_qsdp6zt'
            arith = 2
            index =[3]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            napfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            napfoo = pd.concat([napfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_napfoo = rep.f_append_dfs(stacked_napfoo, napfoo)

        if report_run.loc['run_grncon', 'Run']:
            #returns consumption per ha per day
            # to get total consumption, change arith to 2 and remove den_weights
            type = 'pas'
            prod = 'cons_grnha_t_gop6lzt'
            na_prod = [0,1]  # q,s
            prod_weights = 1000 #to convert to kg/ha/day
            weights = 'greenpas_ha_qsfgop6lzt'
            den_weights = 'days_p6z'
            na_denweights = [1,3]
            keys = 'keys_qsfgop6lzt'
            arith = 1
            index =[5]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            grncon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, prod_weights=prod_weights,
                                    type=type, weights=weights, den_weights=den_weights, na_denweights=na_denweights,
                                    keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grncon = pd.concat([grncon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grncon = rep.f_append_dfs(stacked_grncon, grncon)

        if report_run.loc['run_drycon', 'Run']:
            #returns total consumption per day in each FP
            #todo once this is change to per ha variable then change to report consumption per ha per day (same as grn pas)
            type = 'pas'
            prod = np.array([1])
            weights = 'drypas_consumed_qsfdp6zlt'
            keys = 'keys_qsfdp6zlt'
            arith = 2
            index =[4]
            cols =[3,7] #d,t
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            drycon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drycon = pd.concat([drycon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drycon = rep.f_append_dfs(stacked_drycon, drycon)

        if report_run.loc['run_grnnv', 'Run']:
            #returns NV during each FP regardless of whether selected or not
            type = 'pas'
            prod = 'nv_grnha_fgop6lzt'
            weights = None
            keys = 'keys_fgop6lzt'
            arith = 5
            index = [3]
            cols = [2, 1]
            axis_slice = {}
            grnnv = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnnv = pd.concat([grnnv],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grnnv = rep.f_append_dfs(stacked_grnnv, grnnv)

        if report_run.loc['run_grndmd', 'Run']:
            #returns DMD during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dmd_diet_grnha_gop6lzt'
            weights = None
            keys = 'keys_gop6lzt'
            arith = 5
            index = [2]
            cols = [1, 0]
            axis_slice = {}
            grndmd = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grndmd = pd.concat([grndmd],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_grndmd = rep.f_append_dfs(stacked_grndmd, grndmd)

        if report_run.loc['run_avegrnfoo', 'Run']:
            #returns average FOO during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'foo_ave_grnha_gop6lzt'
            weights = None
            keys = 'keys_gop6lzt'
            arith = 5
            index = [2]
            cols = [1, 0]
            axis_slice = {}
            grnfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            grnfoo = pd.concat([grnfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_avegrnfoo = rep.f_append_dfs(stacked_avegrnfoo, grnfoo)

        if report_run.loc['run_drynv', 'Run']:
            #returns NV during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'nv_dry_fdp6zt'
            weights = None
            keys = 'keys_fdp6zt'
            arith = 5
            index = [2]
            cols = [1]
            axis_slice = {}
            drynv = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drynv = pd.concat([drynv],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drynv = rep.f_append_dfs(stacked_drynv, drynv)

        if report_run.loc['run_drydmd', 'Run']:
            #returns DMD during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dry_dmd_dp6zt'
            weights = None
            keys = 'keys_dp6zt'
            arith = 5
            index = [1]
            cols = [0]
            axis_slice = {}
            drydmd = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            drydmd = pd.concat([drydmd],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_drydmd = rep.f_append_dfs(stacked_drydmd, drydmd)

        if report_run.loc['run_dryfoo', 'Run']:
            #returns average FOO during each FP (regardless of whether selected or not)
            type = 'pas'
            prod = 'dry_foo_dp6zt'
            weights = None
            keys = 'keys_dp6zt'
            arith = 5
            index = [1]
            cols = [0]
            axis_slice = {}
            dryfoo = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            dryfoo = pd.concat([dryfoo],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_avedryfoo = rep.f_append_dfs(stacked_avedryfoo, dryfoo)

        if report_run.loc['run_napcon', 'Run']:
            #returns consumption in each FP
            prod = np.array([1])
            type = 'pas'
            weights = 'nap_consumed_qsfdp6zt'
            keys = 'keys_qsfdp6zt'
            arith = 2
            index =[4]
            cols =[]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            napcon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            napcon = pd.concat([napcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_napcon = rep.f_append_dfs(stacked_napcon, napcon)

        if report_run.loc['run_poccon', 'Run']:
            #returns consumption in each FP
            prod = np.array([1])
            type = 'pas'
            weights = 'poc_consumed_qsfp6lz'
            keys = 'keys_qsfp6lz'
            arith = 2
            index =[3]
            cols =[5]
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            poccon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            poccon = pd.concat([poccon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_poccon = rep.f_append_dfs(stacked_poccon, poccon)

        if report_run.loc['run_supcon', 'Run']:
            #returns consumption in each FP
            option = 1
            supcon = rep.f_grain_sup_summary(lp_vars, r_vals, option=option)
            supcon = pd.concat([supcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_supcon = rep.f_append_dfs(stacked_supcon, supcon)

        if report_run.loc['run_stubcon', 'Run']:
            #returns consumption in each FP
            prod = np.array([1])
            type = 'stub'
            weights = 'stub_qszp6fks1s2'
            keys = 'keys_qszp6fks1s2'
            arith = 2
            index =[0,1,2,3] #q,s,z,p6
            cols =[6] #s1 (stub cat)
            axis_slice = {}
            # axis_slice[0] = [0, 2, 1]
            stubcon = rep.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                   keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
            stubcon = pd.concat([stubcon],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_stubcon = rep.f_append_dfs(stacked_stubcon, stubcon)

        if report_run.loc['run_mvf', 'Run']:
            #returns consumption in each FP
            mvf = rep.f_mvf_summary(lp_vars)
            mvf = pd.concat([mvf],keys=[trial_name],names=['Trial'])  # add trial name as index level
            stacked_mvf = rep.f_append_dfs(stacked_mvf, mvf)

    ####################################
    #run between trial reports and save#
    ####################################
    print("Writing to Excel")
    ##first check that excel is not open (microsoft puts a lock on files so they can't be updated from elsewhere while open)
    if os.path.isfile("Output/Report{0}.xlsx".format(processor)): #to check if report.xl exists
        while True:   # repeat until the try statement succeeds
            try:
                myfile = open("Output/Report{0}.xlsx".format(processor),"w") # chucks an error if excel file is open
                break                             # exit the loop
            except IOError:
                input("Could not open file! Please close Excel. Press Enter to retry.")
                # restart the loop

    ## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
    writer = pd.ExcelWriter('Output/Report{0}.xlsx'.format(processor),engine='xlsxwriter')

    ##make empty df to store row and col index settings. Used when combining multiple report.xl
    df_settings = pd.DataFrame(columns=['index', 'cols'])

    ##write to excel
    ###determine the method of reporting rows and columns that are all zeros
    ### mode 0: df straight into excel
    ### mode 1: df into excel - collapsing rows/cols that contain only 0's.
    ### mode 2: df into excel - removing rows/cols that contain only 0's. This method make the writing process faster.
    try:
        xl_display_mode = int(sys.argv[4])  # reads in as string so need to convert to int, the script path is the first value.
    except IndexError:  # in case no arg passed to python
        xl_display_mode = 1 #default is to collapse rows/cols that are all 0's (ie they exist in excel but are hidden)

    df_settings = rep.f_df2xl(writer, stacked_infeasible, 'infeasible', df_settings, option=xl_display_mode)
    df_settings = rep.f_df2xl(writer, stacked_non_exist,'Non-exist',df_settings,option=0,colstart=0)
    if report_run.loc['run_summary', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_summary, 'summary', df_settings, option=xl_display_mode)
    if report_run.loc['run_areasum', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_areasum, 'areasum', df_settings, option=xl_display_mode)
    if report_run.loc['run_pnl', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_pnl, 'pnl', df_settings, option=xl_display_mode)
    if report_run.loc['run_wc', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_wc, 'wc', df_settings, option=xl_display_mode)
    if report_run.loc['run_profitarea', 'Run']:
        plot = rep.f_xy_graph(stacked_profitarea)
        plot.savefig('Output/profitarea_curve.png')
    if report_run.loc['run_feedbudget', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_feed, 'feed budget', df_settings, option=xl_display_mode)
    if report_run.loc['run_period_dates', 'Run']:
        fp_start_col = len(stacked_season_nodes.columns) + stacked_season_nodes.index.nlevels + 1
        dam_dvp_start_col = fp_start_col + len(stacked_feed_periods.columns) + stacked_feed_periods.index.nlevels + 1
        repro_start_col = dam_dvp_start_col + len(stacked_dam_dvp_dates.columns) + stacked_dam_dvp_dates.index.nlevels + 1
        offs_start_col = repro_start_col + len(stacked_repro_dates.columns) + stacked_repro_dates.index.nlevels + 1
        df_settings = rep.f_df2xl(writer, stacked_season_nodes, 'period_dates', df_settings, option=0, colstart=0)
        df_settings = rep.f_df2xl(writer, stacked_feed_periods, 'period_dates', df_settings, option=0, colstart=fp_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dam_dvp_dates, 'period_dates', df_settings, option=0, colstart=dam_dvp_start_col)
        df_settings = rep.f_df2xl(writer, stacked_repro_dates, 'period_dates', df_settings, option=0, colstart=repro_start_col)
        df_settings = rep.f_df2xl(writer, stacked_offs_dvp_dates, 'period_dates', df_settings, option=0, colstart=offs_start_col)
    if report_run.loc['run_saleprice', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_saleprice, 'saleprice', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salegrid_dams, 'salegrid_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_yatf', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salegrid_yatf, 'salegrid_yatf', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salegrid_offs, 'salegrid_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salevalue_offs, 'salevalue_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salevalue_dams, 'salevalue_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_prog', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_salevalue_prog, 'salevalue_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_woolvalue_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_woolvalue_offs, 'woolvalue_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_woolvalue_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_woolvalue_dams, 'woolvalue_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_saledate_offs', 'Run']:
        stacked_saledate_offs = stacked_saledate_offs.astype(object)
        stacked_saledate_offs[stacked_saledate_offs==np.datetime64('1970-01-01')] = 0
        df_settings = rep.f_df2xl(writer, stacked_saledate_offs, 'saledate_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_saledateEL_offs', 'Run']:
        stacked_saledateEL_offs = stacked_saledateEL_offs.astype(object)
        stacked_saledateEL_offs[stacked_saledateEL_offs==np.datetime64('1970-01-01')] = 0
        df_settings = rep.f_df2xl(writer, stacked_saledateEL_offs, 'saledateEL_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_cfw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_cfw_dams, 'cfw_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_fd_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_fd_dams, 'fd_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_cfw_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_cfw_offs, 'cfw_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_fd_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_fd_offs, 'fd_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_wbe_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_wbe_dams, 'wbe_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_wbe_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_wbe_offs, 'wbe_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_lw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_lw_dams, 'lw_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ffcfw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_dams, 'ffcfw_dams', df_settings, option=xl_display_mode)
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_dams, 'ffcfw_dams', df_settings, option=xl_display_mode)
    if True:#report_run.loc['run_ffcfw_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_cut_dams, 'ffcfw_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ffcfw_yatf', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_yatf, 'ffcfw_yatf', df_settings, option=xl_display_mode)
    if report_run.loc['run_nv_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_nv_dams, 'nv_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ffcfw_prog', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_prog, 'ffcfw_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_ffcfw_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_ffcfw_offs, 'ffcfw_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_nv_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_nv_offs, 'nv_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_lamb_survival', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_lamb_survival, 'lamb_survival', df_settings, option=xl_display_mode)
    if report_run.loc['run_weanper', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_weanper, 'wean_per', df_settings, option=xl_display_mode)
    if report_run.loc['run_scanper', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_scanper, 'scan_per', df_settings, option=xl_display_mode)
    if report_run.loc['run_dry_propn', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_dry_propn, 'dry_propn', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_mei_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_mei_dams, 'daily_mei_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_pi_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_pi_dams, 'daily_pi_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_mei_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_mei_offs, 'daily_mei_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_pi_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_daily_pi_offs, 'daily_pi_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_dams, 'numbers_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_dams_p', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_dams_p, 'numbers_dams_p', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_prog', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_prog, 'numbers_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_offs, 'numbers_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_offs_p', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_numbers_offs_p, 'numbers_offs_p', df_settings, option=xl_display_mode)
    if report_run.loc['run_mort_dams', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_mort_dams, 'mort_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_mort_offs', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_mort_offs, 'mort_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_dse', 'Run']:
        dams_start_col = len(stacked_dse_sire.columns) + stacked_dse_sire.index.nlevels + 1
        offs_start_col = dams_start_col + len(stacked_dse_dams.columns) + stacked_dse_dams.index.nlevels + 1
        df_settings = rep.f_df2xl(writer, stacked_dse_sire, 'dse_wt', df_settings, option=0, colstart=0)
        df_settings = rep.f_df2xl(writer, stacked_dse_dams, 'dse_wt', df_settings, option=0, colstart=dams_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse_offs, 'dse_wt', df_settings, option=0, colstart=offs_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse1_sire, 'dse_mei', df_settings, option=0, colstart=0)
        df_settings = rep.f_df2xl(writer, stacked_dse1_dams, 'dse_mei', df_settings, option=0, colstart=dams_start_col)
        df_settings = rep.f_df2xl(writer, stacked_dse1_offs, 'dse_mei', df_settings, option=0, colstart=offs_start_col)
    if report_run.loc['run_pgr', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_pgr, 'Total pg', df_settings, option=xl_display_mode)
    if report_run.loc['run_grnfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grnfoo, 'grnfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_dryfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_dryfoo, 'dryfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_napfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_napfoo, 'napfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_grnnv', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grnnv, 'grnnv', df_settings, option=xl_display_mode)
    if report_run.loc['run_grndmd', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grndmd, 'grndmd', df_settings, option=xl_display_mode)
    if report_run.loc['run_avegrnfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_avegrnfoo, 'avegrnfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_drynv', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drynv, 'drynv', df_settings, option=xl_display_mode)
    if report_run.loc['run_drydmd', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drydmd, 'drydmd', df_settings, option=xl_display_mode)
    if report_run.loc['run_avedryfoo', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_avedryfoo, 'avedryfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_grncon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_grncon, 'grncon', df_settings, option=xl_display_mode)
    if report_run.loc['run_drycon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_drycon, 'drycon', df_settings, option=xl_display_mode)
    if report_run.loc['run_napcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_napcon, 'napcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_poccon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_poccon, 'poccon', df_settings, option=xl_display_mode)
    if report_run.loc['run_supcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_supcon, 'supcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_stubcon', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_stubcon, 'stubcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_mvf', 'Run']:
        df_settings = rep.f_df2xl(writer, stacked_mvf, 'mvf', df_settings, option=xl_display_mode)


    df_settings.to_excel(writer, 'df_settings')
    writer.save()

    print("Report complete. Processor: {0}".format(processor))


if __name__ == '__main__':
    ##read in exp log
    exp_data, experiment_trials, trial_pinp = fun.f_read_exp()

    ##check if trial results are up to date. Out-dated if:
    ##  1. exp.xls has changed
    ##  2. any python module has been updated
    ##  3. the trial needed to be run last time but the user opted not to run that trial
    exp_data = fun.f_run_required(exp_data)
    exp_data = fun.f_group_exp(exp_data, experiment_trials)  # cut exp_data based on the experiment group
    trial_outdated = exp_data['run_req']  # returns true if trial is out of date

    ## enter the trials to summarise and the reports to include
    trials = np.array(exp_data.index.get_level_values(3))[
        pd.Series(exp_data.index.get_level_values(2)).fillna(0).astype(
            bool)]  # this is slightly complicated because blank rows in exp.xl result in nan, so nan must be converted to 0.

    ##check the trials you want to run exist and are up to date - if trial doesn't exist it is removed from trials to
    # report array so that the others can still be run. A list of trials that don't exist is the 'non_exist' sheet in report excel.
    trials, non_exist_trials = rep.f_errors(trial_outdated,trials)

    ##clear the old report.xlsx
    for f in glob.glob("Output/Report*.xlsx"):
        os.remove(f)


    ##print out the reports being run and number of trials
    print('Number of trials to run: ', len(trials))
    print("The following reports will be run: \n", report_run.index[report_run.loc[:,'Run']])

    ##determine the processor for each report
    ## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
    try:
        maximum_processes = int(sys.argv[2])  # reads in as string so need to convert to int, the script path is the first value hence take the second.
    except IndexError:  # in case no arg passed to python
        maximum_processes = 1

    ##start multiprocessing
    ### number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
    agents = min(multiprocessing.cpu_count(), len(trials), maximum_processes)
    ###set up dataset for f_report
    args = []
    for agent in list(range(agents)):
        start_trial = int(len(trials)/agents * agent)
        end_trial = int(len(trials)/agents * (agent+1))
        process_trials = trials[start_trial:end_trial]
        arg = [agent,process_trials, non_exist_trials]
        args.append(arg)
    with multiprocessing.Pool(processes=agents) as pool:
        pool.starmap(f_report, args)

    end = time.time()
    # print("Reports successfully completed")
    print(f'Reporting successfully completed at: {time.ctime()}, total time taken: {end - start:.2f}')
