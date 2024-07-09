"""

Background and usage
---------------------
Reports are generated in a two-step process. Firstly, each trial undergoes some ‘within trial’ calculations
(e.g. calc the sale price for a trial), the results are stacked together and stored in a table
(each report has its own table). The resulting table can then undergo some ‘between trial’ calculations
(e.g. graphing profit by sale price).

Tweaking the output of a given report is done in the ReportControl.py. Here the user specifies the report
properties. For example, they can specify which axes to report in the table (as either rows or columns in the table)
and which to average or another example. The user can also specify report options. For example, what type of profit
(e.g. is asset opportunity cost included or not) they want to use in the profit by area curve.

To run: execute the RunReportsRaw module with optional args <exp group> <processor number> <report number> <excel display mode>.
If no arguments are passed, all exp groups are reported, 1 processor is used, the 'default' report group is run
and rows/cols that contain only 0's are collapsed when written to excel.

The trials to report are controlled in exp.xl.
The reports to run for a given report number are controlled in exp.xl.

The default values are set for the web app. Users can adjust this by creating a Python file called ReportControlsUser.py.

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

.. tip:: When creating r_vals values try and do it in obvious spots, so it is easier to understand later.

.. note:: For livestock: livestock is slightly more complicated. If you add an r_val or lp_vars you
    will also need to add it to f_stock_reshape the allows you to get the shape correct
    (remove singleton axis and converts lp_vars from dict to numpy).


author: Young
"""


import numpy as np
import pandas as pd
import os

from . import ReportFunctions as rfun
from . import Functions as fun
from . import relativeFile




#todo Reports to add:
# 1. todo add a second mortality report that the weighted average mortality for the animals selected (to complement the current report that is mortality for each w axis)

def f_run_report(lp_vars, r_vals, report_run, trial_name, infeasible = None, user_controls={}):
    '''Function to wrap ReportControl.py so that multiprocessing can be used.'''
    # print('Start processor: {0}'.format(processor))
    # print('Start trials: {0}'.format(trials))

    if user_controls=={}:
        lp_vars_inc = True #default (if user controls module doesn't exist) is to include lp vars weighting
    else:
        lp_vars_inc = user_controls['lp_vars_inc']

    reports ={}
    ##handle infeasible trials
    if infeasible is None:
        infeasible_path = relativeFile.find(__file__, "../../Output/infeasible", f"{trial_name}.txt")
        if os.path.isfile(infeasible_path):
            infeasible = True
        else:
            infeasible = False
    if infeasible == True:
        reports["infeasible"] = pd.DataFrame([trial_name]).rename_axis('Trial')
        lp_vars = fun.f_clean_dict(lp_vars)  # if a trial is infeasible or doesn't solve all the lp values are None. This function converts them to 0 so the report can still run.
    else:
        reports["infeasible"] = pd.DataFrame()

    ##initilise lp vars
    rfun.f_var_reshape(lp_vars,r_vals)

    ##run report functions
    if report_run.loc['run_summary', 'Run']:
        reports["summary"] = rfun.f_summary(lp_vars,r_vals,"Summary")
    if report_run.loc['run_areasum', 'Run']:
        option = f_update_default_controls(user_controls, 'areasum', 'option', 10) #default is all rotations by lmu in p7[-1] with disagregate landuse index.
        active_z = f_update_default_controls(user_controls, 'areasum', 'active_z', True) #default is to show the q, s and z axes.
        reports["areasum"] = rfun.f_area_summary(lp_vars, r_vals, option=option, active_z=active_z)
    if report_run.loc['run_cropsum', 'Run']:
        option = f_update_default_controls(user_controls, 'cropsum', 'option', 2) #default is all rotations by lmu in p7[-1] with disagregate landuse index.
        reports["cropsum"] = rfun.f_crop_summary(lp_vars,r_vals, option=option)
    if report_run.loc['run_profit', 'Run']:
        option = f_update_default_controls(user_controls, 'profit', 'option', 4) #profit by zqs
        reports["profit"] = rfun.f_profit(lp_vars, r_vals, option=option)
    if report_run.loc['run_numbers_qsz', 'Run']:
        method = f_update_default_controls(user_controls, 'numbers_qsz', 'method', 0) #dse based on NW
        reports["numbers_qsz"] = rfun.f_dse(lp_vars, r_vals, method, per_ha=False, summary1=False, summary2=True)
    if report_run.loc['run_croparea_qsz', 'Run']:
        option = f_update_default_controls(user_controls, 'croparea_qsz', 'option', 2) #total crop area
        reports["croparea_qsz"] = rfun.f_area_summary(lp_vars, r_vals, option)
    if report_run.loc['run_pnl', 'Run']:
        option = f_update_default_controls(user_controls, 'pnl', 'option', 2) #1 = report q, s, & z. 2 = weighted average of q, s, & z
        reports["pnl"] = rfun.f_profitloss_table(lp_vars, r_vals, option=option)
    if report_run.loc['run_wc', 'Run']:
        reports["wc"] = rfun.f_wc_summary(lp_vars, r_vals)
    if report_run.loc['run_biomass_penalty', 'Run']:
        reports["penalty"] = rfun.f_biomass_penalty(lp_vars, r_vals)
    if report_run.loc['run_biomass_penalty', 'Run']:
        reports["sowing_date"] = rfun.f_mach_summary(lp_vars, r_vals, option=2)
    if report_run.loc['run_profitarea', 'Run']:
        area_option = f_update_default_controls(user_controls, 'profitarea', 'area_option', 2)     # 2 total crop area each season in p7[-1]
        profit_option = f_update_default_controls(user_controls, 'profitarea', 'profit_option', 0)   # 0 Profit, 1 Risk neutral Obj, 2 Utility, 3 range and std dev of profit.
        profitarea = pd.DataFrame(index=[trial_name], columns=['area','profit'])
        profitarea.loc[trial_name, 'area'] = rfun.f_area_summary(lp_vars,r_vals,area_option).squeeze()
        profitarea.loc[trial_name,'profit'] = rfun.f_profit(lp_vars,r_vals,profit_option)
        reports["profitarea"] = profitarea
    if report_run.loc['run_feedbudget', 'Run']:
        option = f_update_default_controls(user_controls, 'feed', 'option', 0)  #0 mei/hd/day & propn from each source, 1 total mei/d
        nv_option = f_update_default_controls(user_controls, 'feed', 'nv_option', 0)   #0 Separate NV pool, NV pool summed.
        residue_cols = f_update_default_controls(user_controls, 'feed', 'residue_cols', [])
        dams_cols = f_update_default_controls(user_controls, 'feed', 'dams_cols', [2]) #k
        offs_cols = f_update_default_controls(user_controls, 'feed', 'offs_cols', []) #shear opp
        reports["feed"] = rfun.f_feed_budget(lp_vars, r_vals, option=option, nv_option=nv_option, dams_cols=dams_cols, offs_cols=offs_cols, residue_cols=residue_cols)
    if report_run.loc['run_feedbudget', 'Run']:
        option = f_update_default_controls(user_controls, 'feed_total', 'option', 1)  #0 mei/hd/day & propn from each source, 1 total mei/d
        nv_option = f_update_default_controls(user_controls, 'feed_total', 'nv_option', 0)   #0 Separate NV pool, NV pool summed.
        residue_cols = f_update_default_controls(user_controls, 'feed_total', 'residue_cols', [])
        dams_cols = f_update_default_controls(user_controls, 'feed_total', 'dams_cols', []) #birth opp
        offs_cols = f_update_default_controls(user_controls, 'feed_total', 'offs_cols', []) #shear opp
        reports["feed_total"] = rfun.f_feed_budget(lp_vars, r_vals, option=option, nv_option=nv_option, dams_cols=dams_cols, offs_cols=offs_cols, residue_cols=residue_cols)
    if report_run.loc['run_feedbudget', 'Run']:
        reports["grazing"] = rfun.f_grazing_summary(lp_vars, r_vals)
    if report_run.loc['run_numbers_summary', 'Run']:
        ewe_numbers_summary, wethers_n_crossys_numbers_summary = rfun.f_stock_numbers_summary(r_vals)
        reports["ewe_numbers_summary"] = ewe_numbers_summary
        reports["wethers_n_crossys_numbers_summary"] = wethers_n_crossys_numbers_summary
    if report_run.loc['run_emissions', 'Run']:
        option = f_update_default_controls(user_controls, 'emissions', 'option', 1)
        reports["emissions"] = rfun.f_emission_summary(lp_vars, r_vals, option)
    if report_run.loc['run_period_dates', 'Run']:
        ###season nodes (p7)
        type = 'zgen'
        prod = 'date_season_node_p7z'
        keys = 'keys_p7z'
        arith = f_update_default_controls(user_controls, 'season_nodes', 'arith', 0)
        index = f_update_default_controls(user_controls, 'season_nodes', 'index', [0]) #p7
        cols = f_update_default_controls(user_controls, 'season_nodes', 'cols', [1]) #z
        axis_slice = f_update_default_controls(user_controls, 'season_nodes', 'axis_slice', {})
        reports["season_nodes"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, keys=keys,
                                                   arith=arith, index=index, cols=cols, axis_slice=axis_slice)
        ###feed periods (p6)
        type = 'pas'
        prod = 'fp_date_start_p6z'
        keys = 'keys_p6z'
        arith = f_update_default_controls(user_controls, 'feed_periods', 'arith', 0)
        index = f_update_default_controls(user_controls, 'feed_periods', 'index', [0]) #p7
        cols = f_update_default_controls(user_controls, 'feed_periods', 'cols', [1]) #z
        axis_slice = f_update_default_controls(user_controls, 'feed_periods', 'axis_slice', {})
        reports["feed_periods"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###dams dvp
        type = 'stock'
        prod = 'dvp_start_vezg1'
        keys = 'dams_keys_vezg1'
        arith = f_update_default_controls(user_controls, 'dam_dvp_dates', 'arith', 0)
        index = f_update_default_controls(user_controls, 'dam_dvp_dates', 'index', [0]) #v
        cols = f_update_default_controls(user_controls, 'dam_dvp_dates', 'cols', [1,3,2]) #e, g, z
        axis_slice = f_update_default_controls(user_controls, 'dam_dvp_dates', 'axis_slice', {})
        reports["dam_dvp_dates"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###dams repro dates
        type = 'stock'
        prod = 'r_repro_dates_roe1g1'
        keys = 'dams_keys_roeg1'
        arith = f_update_default_controls(user_controls, 'repro_dates', 'arith', 0)
        index = f_update_default_controls(user_controls, 'repro_dates', 'index', [1]) #e
        cols = f_update_default_controls(user_controls, 'repro_dates', 'cols', [0,2,3]) #r, e, g
        axis_slice = f_update_default_controls(user_controls, 'repro_dates', 'axis_slice', {})
        reports["repro_dates"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###offs dvp
        type = 'stock'
        prod = 'dvp_start_vzdxg3'
        keys = 'offs_keys_vzdxg3'
        arith = f_update_default_controls(user_controls, 'offs_dvp_dates', 'arith', 0)
        index = f_update_default_controls(user_controls, 'offs_dvp_dates', 'index', [0]) #v
        cols = f_update_default_controls(user_controls, 'offs_dvp_dates', 'cols', [4,3,2,1]) #g, x, d, z
        axis_slice = f_update_default_controls(user_controls, 'offs_dvp_dates', 'axis_slice', {})
        reports["offs_dvp_dates"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_saleprice', 'Run']:
        option = f_update_default_controls(user_controls, 'saleprice', 'option', 2)
        grid = f_update_default_controls(user_controls, 'saleprice', 'grid', [0,5,6])
        weight = f_update_default_controls(user_controls, 'saleprice', 'weight', [22,40,25])
        score = f_update_default_controls(user_controls, 'saleprice', 'score', [2,3,2])
        reports["saleprice"] = rfun.f_price_summary(lp_vars, r_vals, option=option, grid=grid, weight=weight, score=score)
    if report_run.loc['run_salegrid_dams', 'Run']:
        type = 'stock'
        prod = 'salegrid_tva1e1b1nwziyg1'
        keys = 'dams_keys_tva1e1b1nwziyg1'
        arith = f_update_default_controls(user_controls, 'salegrid_dams', 'arith', 0)
        index = f_update_default_controls(user_controls, 'salegrid_dams', 'index', [1])
        cols = f_update_default_controls(user_controls, 'salegrid_dams', 'cols', [10,0,4,6]) #g,t,b,w
        axis_slice = f_update_default_controls(user_controls, 'salegrid_dams', 'axis_slice', {})
        reports["salegrid_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_salegrid_yatf', 'Run']:
        type = 'stock'
        prod = 'salegrid_Tva1e1b1nwzixyg2'
        keys = 'yatf_keys_Tvaebnwzixy1g2'
        arith = f_update_default_controls(user_controls, 'salegrid_yatf', 'arith', 0)
        index = f_update_default_controls(user_controls, 'salegrid_yatf', 'index', [1])
        cols = f_update_default_controls(user_controls, 'salegrid_yatf', 'cols', [11,9,0,4,6]) #g,x,t,b,w
        axis_slice = f_update_default_controls(user_controls, 'salegrid_yatf', 'axis_slice', {})
        reports["salegrid_yatf"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_salegrid_offs', 'Run']:
        type = 'stock'
        prod = 'salegrid_tvnwzida0e0b0xyg3'
        keys = 'offs_keys_tvnwzida0e0b0xyg3'
        arith = f_update_default_controls(user_controls, 'salegrid_offs', 'arith', 0)
        index = f_update_default_controls(user_controls, 'salegrid_offs', 'index', [1])
        cols = f_update_default_controls(user_controls, 'salegrid_offs', 'cols', [0,3,8,9,10,12]) #t,w,e,b,x,g
        axis_slice = f_update_default_controls(user_controls, 'salegrid_offs', 'axis_slice', {})
        reports["salegrid_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_saleage_offs', 'Run']:
        type = 'stock'
        prod = 'saleage_k3k5tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'saleage_offs', 'arith', 0)
        index = f_update_default_controls(user_controls, 'saleage_offs', 'index', [5]) #v
        cols = f_update_default_controls(user_controls, 'saleage_offs', 'cols', [13, 2, 3, 4, 11]) #g3, dam age, BTRT, t, gender
        axis_slice = f_update_default_controls(user_controls, 'saleage_offs', 'axis_slice', {})
        reports["saleage_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_saledate_offs', 'Run']:
        type = 'stock'
        prod = 'saledate_k3k5tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'saledate_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'saledate_offs', 'index', [5, 7])              #DVP, w
        cols = f_update_default_controls(user_controls, 'saledate_offs', 'cols', [13, 2, 3, 4, 11])     #g3, dam age, BTRT, t, gender
        axis_slice = f_update_default_controls(user_controls, 'saledate_offs', 'axis_slice', {})
        reports["saledate_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_salevalue_dams', 'Run']:
        type = 'stock'
        prod = 'salevalue_k2p7tva1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [3]  #p7
        keys = 'dams_keys_qsk2p7tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'salevalue_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'salevalue_dams', 'index', [5])
        cols = f_update_default_controls(user_controls, 'salevalue_dams', 'cols', [2, 3, 4])  # k2, p7, t
        axis_slice = f_update_default_controls(user_controls, 'salevalue_dams', 'axis_slice', {})
        reports["salevalue_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_salevalue_offs', 'Run']:
        type = 'stock'
        prod = 'salevalue_k3k5p7tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [4]  #p7
        keys = 'offs_keys_qsk3k5p7tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'salevalue_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'salevalue_offs', 'index', [5])
        cols = f_update_default_controls(user_controls, 'salevalue_offs', 'cols', [2, 3, 4])  # k2, p7, t
        axis_slice = f_update_default_controls(user_controls, 'salevalue_offs', 'axis_slice', {})
        reports["salevalue_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_salevalue_prog', 'Run']:
        type = 'stock'
        prod = 'salevalue_k3k5p7twzia0xg2'
        na_prod = [0,1]  # q,s
        weights = 'prog_numbers_qsk3k5twzia0xg2'
        na_weights = [4]  #p7
        keys = 'prog_keys_qsk3k5p7twzia0xg2'
        arith = f_update_default_controls(user_controls, 'salevalue_prog', 'arith', 1)
        index = f_update_default_controls(user_controls, 'salevalue_prog', 'index', [6]) #w
        cols = f_update_default_controls(user_controls, 'salevalue_prog', 'cols', [4, 11, 5])    #cashflow period, g2, t
        axis_slice = f_update_default_controls(user_controls, 'salevalue_prog', 'axis_slice', {})
        reports["salevalue_prog"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_woolvalue_dams', 'Run']:
        type = 'stock'
        prod = 'woolvalue_k2p7tva1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [3]  #p7
        keys = 'dams_keys_qsk2p7tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'woolvalue_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'woolvalue_dams', 'index', [5])
        cols = f_update_default_controls(user_controls, 'woolvalue_dams', 'cols', [2, 3, 4])  # k2, p7, t
        axis_slice = f_update_default_controls(user_controls, 'woolvalue_dams', 'axis_slice', {})
        reports["woolvalue_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_woolvalue_offs', 'Run']:
        type = 'stock'
        prod = 'woolvalue_k3k5p7tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [4] #p7
        keys = 'offs_keys_qsk3k5p7tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'woolvalue_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'woolvalue_offs', 'index', [6, 12])     #DVP, gender
        cols = f_update_default_controls(user_controls, 'woolvalue_offs', 'cols', [4, 14, 5])   #cashflow period, g3, t
        axis_slice = f_update_default_controls(user_controls, 'woolvalue_offs', 'axis_slice', {})
        reports["woolvalue_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cfw_dams', 'Run']:
        type = 'stock'
        prod = 'cfw_hdmob_k2tva1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'cfw_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'cfw_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'cfw_dams', 'cols', [])
        axis_slice = f_update_default_controls(user_controls, 'cfw_dams', 'axis_slice', {4: [9, 10, 1]}) #dvp9 - 3.5yo
        reports["cfw_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fd_dams', 'Run']:
        type = 'stock'
        prod = 'fd_hdmob_k2tva1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'fd_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fd_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'fd_dams', 'cols', [])
        axis_slice = f_update_default_controls(user_controls, 'fd_dams', 'axis_slice', {4: [9, 10, 1]}) #dvp9 - 3.5yo
        reports["fd_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cfw_offs', 'Run']:
        type = 'stock'
        prod = 'cfw_hdmob_k3k5tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'cfw_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'cfw_offs', 'index', [5])      #DVP
        cols = f_update_default_controls(user_controls, 'cfw_offs', 'cols', [13, 2, 3, 4, 11])  #g3, dam age, BTRT, t, gender
        axis_slice = f_update_default_controls(user_controls, 'cfw_offs', 'axis_slice', {})
        reports["cfw_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fd_offs', 'Run']:
        type = 'stock'
        prod = 'fd_hdmob_k3k5tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'fd_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fd_offs', 'index', [5])      #DVP
        cols = f_update_default_controls(user_controls, 'fd_offs', 'cols', [13, 2, 3, 4, 11])  #g3, dam age, BTRT, t, gender
        axis_slice = f_update_default_controls(user_controls, 'fd_offs', 'axis_slice', {})
        reports["fd_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_wbe_dams', 'Run']:
        type = 'stock'
        prod = 'wbe_k2tva1nwziyg1'
        na_prod = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'wbe_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'wbe_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'wbe_dams', 'cols', [2]) #k2
        axis_slice = f_update_default_controls(user_controls, 'wbe_dams', 'axis_slice', {})
        reports["wbe_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_wbe_cut_dams', 'Run']:
        type = 'stock'
        prod = 'wbe_dams_k2tvPa1e1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        prod_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7,8]  #p,e,b
        den_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2tvPaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'wbe_cut_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'wbe_cut_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'wbe_cut_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'wbe_dams', 'axis_slice', {})
        # axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
        # axis_slice[4] = [0, 7, 1]  #All DVPs for Triplets
        reports["wbe_cut_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                , weights=weights, na_weights=na_weights
                                                , prod_weights=prod_weights, na_prodweights=na_prodweights
                                                , den_weights=den_weights, na_denweights=na_denweights
                                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fat_dams', 'Run']:
        type = 'stock'
        prod = 'fat_k2tva1nwziyg1'
        na_prod = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'fat_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fat_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'fat_dams', 'cols', [2]) #k2
        axis_slice = f_update_default_controls(user_controls, 'fat_dams', 'axis_slice', {})
        reports["fat_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fat_cut_dams', 'Run']:
        type = 'stock'
        prod = 'fat_dams_k2tvPa1e1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        prod_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7,8]  #p,e,b
        den_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2tvPaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'fat_cut_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fat_cut_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'fat_cut_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'fat_cut_dams', 'axis_slice', {})
        # axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
        # axis_slice[4] = [0, 7, 1]  #All DVPs for Triplets
        reports["fat_cut_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                , weights=weights, na_weights=na_weights
                                                , prod_weights=prod_weights, na_prodweights=na_prodweights
                                                , den_weights=den_weights, na_denweights=na_denweights
                                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_muscle_dams', 'Run']:
        type = 'stock'
        prod = 'muscle_k2tva1nwziyg1'
        na_prod = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'muscle_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'muscle_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'muscle_dams', 'cols', [2]) #k2
        axis_slice = f_update_default_controls(user_controls, 'muscle_dams', 'axis_slice', {})
        reports["muscle_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_muscle_cut_dams', 'Run']:
        type = 'stock'
        prod = 'muscle_dams_k2tvPa1e1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        prod_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7,8]  #p,e,b
        den_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2tvPaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'muscle_cut_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'muscle_cut_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'muscle_cut_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'muscle_cut_dams', 'axis_slice', {})
        # axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
        # axis_slice[4] = [0, 7, 1]  #All DVPs for Triplets
        reports["muscle_cut_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                , weights=weights, na_weights=na_weights
                                                , prod_weights=prod_weights, na_prodweights=na_prodweights
                                                , den_weights=den_weights, na_denweights=na_denweights
                                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_viscera_dams', 'Run']:
        type = 'stock'
        prod = 'viscera_k2tva1nwziyg1'
        na_prod = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'viscera_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'viscera_dams', 'index', [4])      #DVP
        cols = f_update_default_controls(user_controls, 'viscera_dams', 'cols', [2]) #k2
        axis_slice = f_update_default_controls(user_controls, 'viscera_dams', 'axis_slice', {})
        reports["viscera_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_viscera_cut_dams', 'Run']:
        type = 'stock'
        prod = 'viscera_dams_k2tvPa1e1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        prod_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7,8]  #p,e,b
        den_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2tvPaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'viscera_cut_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'viscera_cut_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'viscera_cut_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'viscera_cut_dams', 'axis_slice', {})
        # axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
        # axis_slice[4] = [0, 7, 1]  #All DVPs for Triplets
        reports["viscera_cut_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                , weights=weights, na_weights=na_weights
                                                , prod_weights=prod_weights, na_prodweights=na_prodweights
                                                , den_weights=den_weights, na_denweights=na_denweights
                                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_wbe_offs', 'Run']:
        type = 'stock'
        prod = 'wbe_k3k5tvnwziaxyg3'
        na_prod = [0,1] #q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'wbe_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'wbe_offs', 'index', [5])      #DVP
        cols = f_update_default_controls(user_controls, 'wbe_offs', 'cols', [13, 2, 3, 4])  #g3, dam age, BTRT, t
        axis_slice = f_update_default_controls(user_controls, 'wbe_offs', 'axis_slice', {})
        reports["wbe_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_lw_dams', 'Run']:
        ##Average dam lw with p, e & b axis. Lw is adjusted for animals that are sold but not adjusted by mortality (Ie if the light ones all die during a dvp then the weighting of animals that are averaged should change relatively across p)
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
        arith = f_update_default_controls(user_controls, 'lw_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'lw_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'lw_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'lw_dams', 'axis_slice', {})
        reports["lw_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                 , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights, den_weights=den_weights, na_denweights=na_denweights
                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_ebw_dams', 'Run']:
        ##Average dam ebw with p, e & b axis. ebw is adjusted for animals that are sold but not adjusted by mortality (Ie if the light ones all die then the weighting of ebw by p should change)
        ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'ebw_dams_k2Tvpa1e1b1nw8ziyg1'
        na_prod = [0,1] #q,s
        prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5, 7, 8]
        den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2tvpaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'ebw_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'ebw_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'ebw_dams', 'cols', []) #b1
        axis_slice = f_update_default_controls(user_controls, 'ebw_dams', 'axis_slice', {})
        reports['ebw_dams'] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                 , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith
                                 , index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_ebw_cut_dams', 'Run']:
        ##ebw for a select number of p periods
        type = 'stock'
        prod = 'ebw_dams_k2tvPa1e1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        prod_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7,8]  #p,e,b
        den_weights = 'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1'  # weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0, 1]  # q,s
        keys = 'dams_keys_qsk2tvPaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'ebw_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'ebw_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'ebw_dams', 'cols', [8]) #b1
        axis_slice = f_update_default_controls(user_controls, 'ebw_dams', 'axis_slice', {})
        reports["ebw_cut_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                , weights=weights, na_weights=na_weights
                                                , prod_weights=prod_weights, na_prodweights=na_prodweights
                                                , den_weights=den_weights, na_denweights=na_denweights
                                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cs_dams', 'Run']:
        ##Average dam condition score with p, e & b axis. 
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'cs_dams_k2Tvpa1e1b1nw8ziyg1'
        na_prod = [0,1] #q,s
        prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0, 1]  #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5, 7, 8]
        den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0, 1]  #q,s
        keys = 'dams_keys_qsk2tvpaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'cs_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'cs_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'cs_dams', 'cols', [14, 3, 8])  #g1, t & b1
        axis_slice = f_update_default_controls(user_controls, 'cs_dams', 'axis_slice', {})
        reports["cs_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod,
                               prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights, na_weights=na_weights,
                               den_weights=den_weights, na_denweights=na_denweights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fs_dams', 'Run']:
        ##Average dam fat score with p, e & b axis.
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'fs_dams_k2Tvpa1e1b1nw8ziyg1'
        na_prod = [0,1] #q,s
        prod_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0, 1]  #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5, 7, 8]
        den_weights = 'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1'  #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0, 1]  #q,s
        keys = 'dams_keys_qsk2tvpaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'fs_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fs_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'fs_dams', 'cols', [14, 3, 8])  #g1, t & b1
        axis_slice = f_update_default_controls(user_controls, 'fs_dams', 'axis_slice', {})
        reports["fs_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod,
                               prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights, na_weights=na_weights,
                               den_weights=den_weights, na_denweights=na_denweights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_nv_dams', 'Run']:
        ##Average dam NV with p, e & b axis. NV is adjusted for animals that are sold but not adjusted by mortality (Ie if the light ones all die then the weighting by p should change)
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
        arith = f_update_default_controls(user_controls, 'nv_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'nv_dams', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'nv_dams', 'cols', [14, 7, 3, 8])  #g1, e, t & b1
        axis_slice = f_update_default_controls(user_controls, 'nv_dams', 'axis_slice', {})
        reports["nv_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod,
                               prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights, na_weights=na_weights,
                               den_weights=den_weights, na_denweights=na_denweights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_ebw_yatf', 'Run']:
        ##Average yatf ebw with p, e & b axis. ebw is not adjusted by mortality (Ie if the light ones all die then the weighting by p should change)
        ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        ##For yatf the denom weight also includes a weighting for nyatf. The numerator also gets weighted by this.
        ##v_dam must be used because v_prog has a different w axis than yatf.
        ##note prog weight will be a bit higher than yatf weight because yatf weight is start of period and prog weight is end
        type = 'stock'
        prod = 'ebw_yatf_k2Tvpa1e1b1nw8zixyg1'
        na_prod = [0,1] #q,s
        prod_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5, 7, 8, 13]                  #p, e1, b1, x
        den_weights = 'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'yatf_keys_qsk2tvpaebnwzixy1g1'
        arith = f_update_default_controls(user_controls, 'ebw_yatf', 'arith', 1)
        index = f_update_default_controls(user_controls, 'ebw_yatf', 'index', [5])      #p
        cols = f_update_default_controls(user_controls, 'ebw_yatf', 'cols', [2,15, 10, 13])  #k2, g2, w8, x
        axis_slice = f_update_default_controls(user_controls, 'ebw_yatf', 'axis_slice', {8: [2,None,1]}) #b with yatf
        reports["ebw_yatf"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys
                                 , arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_ebw_prog', 'Run']:
        ##note prog weight will be a bit higher than yatf weight because yatf weight is start of period and prog weight is end
        type = 'stock'
        prod = 'ebw_prog_k3k5wzida0e0b0xyg2'
        na_prod = [0,1,4] #q,s,t
        prod_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight prod for propn of animals in e and b slice
        na_prodweights = [0,1] #q,s
        weights = 'prog_numbers_qsk3k5twzia0xg2'
        na_weights = [8,10,11,13] #d, e,b,y
        den_weights = 'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2' #weight numbers for propn of animals in e and b slice
        na_denweights = [0,1] #q,s
        keys = 'prog_keys_qsk3k5twzida0e0b0xyg2'
        arith = f_update_default_controls(user_controls, 'ebw_prog', 'arith', 1)
        index = f_update_default_controls(user_controls, 'ebw_prog', 'index', [5])      #w
        cols = f_update_default_controls(user_controls, 'ebw_prog', 'cols', [14,12,6,11])  #g2, gender, z, b0
        axis_slice = f_update_default_controls(user_controls, 'ebw_prog', 'axis_slice', {})
        reports["ebw_prog"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod
                                                 , prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights
                                                 , na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_ebw_offs', 'Run']:
        ##Average offs ebw with p, e & b axis. ebw is adjusted for animals that are sold but not adjusted by mortality
        ## because it adds an extra level of complexity for minimal gain (to include mort both the numerator and denominator need to be adjusted).
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'ebw_offs_k3k5Tvpnw8zida0e0b0xyg3'
        na_prod = [0,1] #q,s
        prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [6, 11, 13, 14]
        den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'offs_keys_qsk3k5tvpnwzidaebxyg3'
        arith = f_update_default_controls(user_controls, 'ebw_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'ebw_offs', 'index', [6])  #p
        cols = f_update_default_controls(user_controls, 'ebw_offs', 'cols', [])
        axis_slice = f_update_default_controls(user_controls, 'ebw_offs', 'axis_slice', {})
        reports["ebw_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cs_offs', 'Run']:
        ##Average offs cs with p, e & b axis. 
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'cs_offs_k3k5Tvpnw8zida0e0b0xyg3'
        na_prod = [0,1] #q,s
        prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [6, 11, 13, 14]
        den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'offs_keys_qsk3k5tvpnwzidaebxyg3'
        arith = f_update_default_controls(user_controls, 'cs_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'cs_offs', 'index', [6])  #p
        cols = f_update_default_controls(user_controls, 'cs_offs', 'cols', [17, 3, 15, 4])   #k5, g3, x, t
        axis_slice = f_update_default_controls(user_controls, 'cs_offs', 'axis_slice', {})
        reports["cs_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fs_offs', 'Run']:
        ##Average offs fs with p, e & b axis. 
        ##Denom (numbers) also needs to be weighted because of the new axis (p,e&b) being added and then summed in the weighted average.
        type = 'stock'
        prod = 'fs_offs_k3k5Tvpnw8zida0e0b0xyg3'
        na_prod = [0,1] #q,s
        prod_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight prod for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_prodweights = [0,1] #q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [6, 11, 13, 14]
        den_weights = 'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3' #weight numbers for propn of animals in e and b slice and on hand (prod will be equal to 0 if animal is off-hand)
        na_denweights = [0,1] #q,s
        keys = 'offs_keys_qsk3k5tvpnwzidaebxyg3'
        arith = f_update_default_controls(user_controls, 'fs_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'fs_offs', 'index', [6])  #p
        cols = f_update_default_controls(user_controls, 'fs_offs', 'cols', [17, 3, 15, 4])   #k5, g3, x, t
        axis_slice = f_update_default_controls(user_controls, 'fs_offs', 'axis_slice', {})
        reports["fs_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        arith = f_update_default_controls(user_controls, 'nv_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'nv_offs', 'index', [6])  #p
        cols = f_update_default_controls(user_controls, 'nv_offs', 'cols', [2,15,3,4,8])  #k3,x,k5,t,w
        axis_slice = f_update_default_controls(user_controls, 'nv_offs', 'axis_slice', {})
        reports["nv_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                               , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                               , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_lamb_survival', 'Run']:
        #axes are qsk2tvaeb9nwziy1g1      b9 axis is shorten b axis: [0,1,2,3]
        option = f_update_default_controls(user_controls, 'lamb_survival', 'option', 0)
        index = f_update_default_controls(user_controls, 'lamb_survival', 'index', [4])  #v
        cols = f_update_default_controls(user_controls, 'lamb_survival', 'cols', [13,11,0,1,10,7])    #g,i,q,s,z & b9 #report must include the b axis otherwise an error is caused because the axis added after the arith.
        axis_slice = f_update_default_controls(user_controls, 'lamb_survival', 'axis_slice', {})
        reports["lamb_survival"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                             , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
    if report_run.loc['run_weanper', 'Run']:
        #with the current structure w CANNOT be reported. 23Apr22 - seems to be working when not using lp_vars
        # problem could be that dams can change w slice between joining (nfoet) and lambing (nyatf)
        #axes are qsk2tvanwziy1g1
        option = f_update_default_controls(user_controls, 'weanper', 'option', 1)
        index = f_update_default_controls(user_controls, 'weanper', 'index', [])
        cols = f_update_default_controls(user_controls, 'weanper', 'cols', [])   #(needs t in report if no lp_vars)
        axis_slice = f_update_default_controls(user_controls, 'weanper', 'axis_slice', {})
        reports["weanper"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                       , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
    if report_run.loc['run_scanper', 'Run']:
        #axes are qsk2tvanwziy1g1
        option = f_update_default_controls(user_controls, 'scanper', 'option', 2)
        index = f_update_default_controls(user_controls, 'scanper', 'index', [])
        cols = f_update_default_controls(user_controls, 'scanper', 'cols', [])   #(needs t in report if no lp_vars)
        axis_slice = f_update_default_controls(user_controls, 'scanper', 'axis_slice', {})
        reports["scanper"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                       , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
    if report_run.loc['run_dry_propn', 'Run']:
        #axes are qsk2tvanwziy1g1
        option = f_update_default_controls(user_controls, 'dry_propn', 'option', 3)
        index = f_update_default_controls(user_controls, 'dry_propn', 'index', [4])  #v
        cols = f_update_default_controls(user_controls, 'dry_propn', 'cols', [11,9,0,1,8])   #g,i,q,s & z (needs t in report if no lp_vars)
        axis_slice = f_update_default_controls(user_controls, 'dry_propn', 'axis_slice', {})
        reports["dry_propn"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                         , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
    if report_run.loc['run_daily_mei_dams', 'Run']:
        type = 'stock'
        prod = 'mei_dams_k2p6ftva1nw8ziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [3, 4]
        den_weights = 'stock_days_k2p6ftva1nwziyg1'
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2p6ftvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'daily_mei_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'daily_mei_dams', 'index', [6, 3])  #v, p6
        cols = f_update_default_controls(user_controls, 'daily_mei_dams', 'cols', [2,4])  #k2,nv
        axis_slice = f_update_default_controls(user_controls, 'daily_mei_dams', 'axis_slice', {})
        reports["daily_mei_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                               index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_daily_pi_dams', 'Run']:
        type = 'stock'
        prod = 'pi_dams_k2p6ftva1nw8ziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [3, 4]
        den_weights = 'stock_days_k2p6ftva1nwziyg1'
        na_denweights = [0,1] #q,s
        keys = 'dams_keys_qsk2p6ftvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'daily_pi_dams', 'arith', 1)
        index = f_update_default_controls(user_controls, 'daily_pi_dams', 'index', [6, 3])  #v, p6
        cols = f_update_default_controls(user_controls, 'daily_pi_dams', 'cols', [2,4])  #k2,nv
        axis_slice = f_update_default_controls(user_controls, 'daily_pi_dams', 'axis_slice', {})
        reports["daily_pi_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_daily_mei_offs', 'Run']:
        type = 'stock'
        prod = 'mei_offs_k3k5p6ftvnw8ziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [4, 5]
        den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
        na_denweights = [0,1] #q,s
        keys = 'offs_keys_qsk3k5p6ftvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'daily_mei_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'daily_mei_offs', 'index', [7, 4])       #DVP, fp
        cols = f_update_default_controls(user_controls, 'daily_mei_offs', 'cols', [15, 3, 5])    #g3, BTRT, nv
        axis_slice = f_update_default_controls(user_controls, 'daily_mei_offs', 'axis_slice', {})
        reports["daily_mei_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                               index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_daily_pi_offs', 'Run']:
        type = 'stock'
        prod = 'pi_offs_k3k5p6ftvnw8ziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [4, 5]
        den_weights = 'stock_days_k3k5p6ftvnwziaxyg3'
        na_denweights = [0,1] #q,s
        keys = 'offs_keys_qsk3k5p6ftvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'daily_pi_offs', 'arith', 1)
        index = f_update_default_controls(user_controls, 'daily_pi_offs', 'index', [7, 4])       #DVP, fp
        cols = f_update_default_controls(user_controls, 'daily_pi_offs', 'cols', [15, 3, 5])    #g3, BTRT, nv
        axis_slice = f_update_default_controls(user_controls, 'daily_pi_offs', 'axis_slice', {})
        reports["daily_pi_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_dams', 'Run']:
        type = 'stock'
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = f_update_default_controls(user_controls, 'numbers_dams', 'arith', 2)
        index = f_update_default_controls(user_controls, 'numbers_dams', 'index', [4])       #DVP
        cols = f_update_default_controls(user_controls, 'numbers_dams', 'cols', [0,1,8, 2, 3]) #q, s, z, k2, t
        axis_slice = f_update_default_controls(user_controls, 'numbers_dams', 'axis_slice', {})
        reports["numbers_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_dams_p', 'Run']:
        type = 'stock'
        prod = 'on_hand_mort_k2tvpa1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5]
        keys = 'dams_keys_qsk2tvpanwziy1g1'
        arith = f_update_default_controls(user_controls, 'numbers_dams_p', 'arith', 2)
        index = f_update_default_controls(user_controls, 'numbers_dams_p', 'index', [4,5])       #DVP, p
        cols = f_update_default_controls(user_controls, 'numbers_dams_p', 'cols', [2, 3, 8]) #k2, t, w
        axis_slice = f_update_default_controls(user_controls, 'numbers_dams_p', 'axis_slice', {})
        reports["numbers_dams_p"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_prog', 'Run']:
        type = 'stock'
        weights = 'prog_numbers_qsk3k5twzia0xg2'
        keys = 'prog_keys_qsk3k5twzia0xg2'
        arith = f_update_default_controls(user_controls, 'numbers_prog', 'arith', 2)
        index = f_update_default_controls(user_controls, 'numbers_prog', 'index', [0,1,5])   #q, s, w
        cols = f_update_default_controls(user_controls, 'numbers_prog', 'cols', [6, 2, 3, 4, 9]) #z, dam age, birth type, t slice, gender
        axis_slice = f_update_default_controls(user_controls, 'numbers_prog', 'axis_slice', {})
        reports["numbers_prog"] = rfun.f_stock_pasture_summary(r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_offs', 'Run']:
        type = 'stock'
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'numbers_offs', 'arith', 2)
        index = f_update_default_controls(user_controls, 'numbers_offs', 'index', [5])   #v
        cols = f_update_default_controls(user_controls, 'numbers_offs', 'cols', [8, 13, 11, 2, 3, 4, 7])   #z, g3, Gender, dam age, BTRT, t, w
        axis_slice = f_update_default_controls(user_controls, 'numbers_offs', 'axis_slice', {})
        reports["numbers_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_offs_p', 'Run']:
        type = 'stock'
        prod = 'on_hand_mort_k3k5tvpnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        na_weights = [6]
        keys = 'offs_keys_qsk3k5tvpnwziaxyg3'
        arith = f_update_default_controls(user_controls, 'numbers_offs_p', 'arith', 2)
        index = f_update_default_controls(user_controls, 'numbers_offs_p', 'index', [6])   #p
        cols = f_update_default_controls(user_controls, 'numbers_offs_p', 'cols', [14, 2,12,3,4,8])  #genotype, dam age, gender, BTRT, t, w
        axis_slice = f_update_default_controls(user_controls, 'numbers_offs_p', 'axis_slice', {})
        reports["numbers_offs_p"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
    if report_run.loc['run_mort_dams', 'Run']:
        ##note t axis will be singleton unless stock generator was run with active t axis.
        type = 'stock'
        prod = 'mort_Tpa1e1b1nwziyg1' #uses b axis instead of k for extra detail when scan=0
        weights = None
        na_weights = []
        keys = 'dams_keys_Tpaebnwziy1g1'
        arith = f_update_default_controls(user_controls, 'mort_dams', 'arith', 4)
        index = f_update_default_controls(user_controls, 'mort_dams', 'index', [1])   #p
        cols = f_update_default_controls(user_controls, 'mort_dams', 'cols', [10, 4, 6])     #genotype, LSLN, w
        axis_slice = f_update_default_controls(user_controls, 'mort_dams', 'axis_slice', {})
        reports["mort_dams"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
    if report_run.loc['run_mort_offs', 'Run']:
        ##note t axis will be singleton unless stock generator was run with active t axis.
        type = 'stock'
        prod = 'mort_Tpnwzida0e0b0xyg3' #uses b axis instead of k for extra detail when scan=0
        weights = None
        na_weights = []
        keys = 'offs_keys_Tpnwzidaebxyg3'
        arith = f_update_default_controls(user_controls, 'mort_offs', 'arith', 4)
        index = f_update_default_controls(user_controls, 'mort_offs', 'index', [1])   #p
        cols = f_update_default_controls(user_controls, 'mort_offs', 'cols', [12, 9, 3, 10])     #g3, BTRT, w, gender
        axis_slice = f_update_default_controls(user_controls, 'mort_offs', 'axis_slice', {})
        reports["mort_offs"] = rfun.f_stock_pasture_summary(r_vals, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
    if report_run.loc['run_dse', 'Run']:
        ##you can go into f_dse to change the axis being reported.
        per_ha = f_update_default_controls(user_controls, 'dse_sire', 'per_ha', True)
        method1 = f_update_default_controls(user_controls, 'dse_sire', 'method1', 0)
        reports["dse_sire"], reports["dse_dams"], reports["dse_offs"] = rfun.f_dse(lp_vars, r_vals, method = method1, per_ha = per_ha)
        method2 = f_update_default_controls(user_controls, 'dse_sire', 'method2', 1)
        reports["dse1_sire"], reports["dse1_dams"], reports["dse1_offs"] = rfun.f_dse(lp_vars, r_vals, method = method2, per_ha = per_ha)
    if report_run.loc['run_grnfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = 'foo_end_grnha_gop6lzt'
        na_prod = [0,1,2] #q,s,f
        weights = 'greenpas_ha_qsfgop6lzt'
        keys = 'keys_qsfgop6lzt'
        arith = f_update_default_controls(user_controls, 'grnfoo', 'arith', 2)
        index = f_update_default_controls(user_controls, 'grnfoo', 'index', [7,5])   #p6 z
        cols = f_update_default_controls(user_controls, 'grnfoo', 'cols', [6])     #lmu
        axis_slice = f_update_default_controls(user_controls, 'grnfoo', 'axis_slice', {})
        reports["grnfoo"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_pgr', 'Run']:
        #returns average pgr per day per ha.
        #to get total PG change arith to 2 (den_weights won't be used)
        #todo would be good if this could include germination but doesnt work atm because germ has r axis.
        type = 'pas'
        prod = 'pgr_grnha_gop6lzt'
        na_prod = [0,1,2] #q,s,f
        weights = 'greenpas_ha_qsfgop6lzt'
        den_weights = 'days_p6z'
        na_denweights = [1,3]
        keys = 'keys_qsfgop6lzt'
        arith = f_update_default_controls(user_controls, 'pgr', 'arith', 2)
        index = f_update_default_controls(user_controls, 'pgr', 'index', [7,5])   #p6 z
        cols = f_update_default_controls(user_controls, 'pgr', 'cols', [6])     #lmu
        axis_slice = f_update_default_controls(user_controls, 'pgr', 'axis_slice', {})
        reports["pgr"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights, den_weights=den_weights,
                               na_denweights=na_denweights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_dryfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = np.array([1000])
        weights = 'drypas_transfer_qsdp6zlt'
        keys = 'keys_qsdp6zlt'
        arith = f_update_default_controls(user_controls, 'dryfoo', 'arith', 2)
        index = f_update_default_controls(user_controls, 'dryfoo', 'index', [4,3])   #p6 z
        cols = f_update_default_controls(user_controls, 'dryfoo', 'cols', [2])     #dry pools
        axis_slice = f_update_default_controls(user_controls, 'dryfoo', 'axis_slice', {})
        reports["dryfoo"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_napfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = np.array([1000])
        weights = 'nap_transfer_qsdp6zt'
        keys = 'keys_qsdp6zt'
        arith = f_update_default_controls(user_controls, 'napfoo', 'arith', 2)
        index = f_update_default_controls(user_controls, 'napfoo', 'index', [4,3])   #p6 z
        cols = f_update_default_controls(user_controls, 'napfoo', 'cols', [])
        axis_slice = f_update_default_controls(user_controls, 'napfoo', 'axis_slice', {})
        reports["napfoo"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        arith = f_update_default_controls(user_controls, 'grncon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'grncon', 'index', [7,5])   #p6 z
        cols = f_update_default_controls(user_controls, 'grncon', 'cols', [8,6])     #t,l
        axis_slice = f_update_default_controls(user_controls, 'grncon', 'axis_slice', {})
        reports["grncon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, prod_weights=prod_weights,
                                type=type, weights=weights, den_weights=den_weights, na_denweights=na_denweights,
                                keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drycon', 'Run']:
        #returns total consumption per day in each FP
        #todo once this is change to per ha variable then change to report consumption per ha per day (same as grn pas)
        type = 'pas'
        prod = np.array([1])
        weights = 'drypas_consumed_qsfdp6zlt'
        keys = 'keys_qsfdp6zlt'
        arith = f_update_default_controls(user_controls, 'drycon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'drycon', 'index', [5,4])   #z,p6
        cols = f_update_default_controls(user_controls, 'drycon', 'cols', [3,7])     #d,t
        axis_slice = f_update_default_controls(user_controls, 'drycon', 'axis_slice', {})
        reports["drycon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_grnnv', 'Run']:
        #returns NV during each FP regardless of whether selected or not
        type = 'pas'
        prod = 'nv_grnha_fgop6lzt'
        weights = None
        keys = 'keys_fgop6lzt'
        arith = f_update_default_controls(user_controls, 'grnnv', 'arith', 5)
        index = f_update_default_controls(user_controls, 'grnnv', 'index', [5,3])   #p6 z
        cols = f_update_default_controls(user_controls, 'grnnv', 'cols', [2,1])     #lmu
        axis_slice = f_update_default_controls(user_controls, 'grnnv', 'axis_slice', {})
        reports["grnnv"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_grndmd', 'Run']:
        #returns DMD during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dmd_diet_grnha_gop6lzt'
        weights = None
        keys = 'keys_gop6lzt'
        arith = f_update_default_controls(user_controls, 'grndmd', 'arith', 5)
        index = f_update_default_controls(user_controls, 'grndmd', 'index', [4,2])   #z,p6
        cols = f_update_default_controls(user_controls, 'grndmd', 'cols', [1,0])     #g, o
        axis_slice = f_update_default_controls(user_controls, 'grndmd', 'axis_slice', {})
        reports["grndmd"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_avegrnfoo', 'Run']:
        #returns average FOO during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'foo_ave_grnha_gop6lzt'
        weights = None
        keys = 'keys_gop6lzt'
        arith = f_update_default_controls(user_controls, 'avegrnfoo', 'arith', 5)
        index = f_update_default_controls(user_controls, 'avegrnfoo', 'index', [4,2])   #z,p6
        cols = f_update_default_controls(user_controls, 'avegrnfoo', 'cols', [1,0])     #g, o
        axis_slice = f_update_default_controls(user_controls, 'avegrnfoo', 'axis_slice', {})
        reports["avegrnfoo"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drynv', 'Run']:
        #returns NV during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'nv_dry_fdp6zt'
        weights = None
        keys = 'keys_fdp6zt'
        arith = f_update_default_controls(user_controls, 'drynv', 'arith', 5)
        index = f_update_default_controls(user_controls, 'drynv', 'index', [3,2])   #z,p6
        cols = f_update_default_controls(user_controls, 'drynv', 'cols', [1])     #d
        axis_slice = f_update_default_controls(user_controls, 'drynv', 'axis_slice', {})
        reports["drynv"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drydmd', 'Run']:
        #returns DMD during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dry_dmd_dp6zt'
        weights = None
        keys = 'keys_dp6zt'
        arith = f_update_default_controls(user_controls, 'drydmd', 'arith', 5)
        index = f_update_default_controls(user_controls, 'drydmd', 'index', [2,1])   #z,p6
        cols = f_update_default_controls(user_controls, 'drydmd', 'cols', [0])     #d
        axis_slice = f_update_default_controls(user_controls, 'drydmd', 'axis_slice', {})
        reports["drydmd"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_avedryfoo', 'Run']:
        #returns average FOO during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dry_foo_dp6zt'
        weights = None
        keys = 'keys_dp6zt'
        arith = f_update_default_controls(user_controls, 'avedryfoo', 'arith', 5)
        index = f_update_default_controls(user_controls, 'avedryfoo', 'index', [2,1])   #z,p6
        cols = f_update_default_controls(user_controls, 'avedryfoo', 'cols', [0])     #d
        axis_slice = f_update_default_controls(user_controls, 'avedryfoo', 'axis_slice', {})
        reports["avedryfoo"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_napcon', 'Run']:
        #returns consumption in each FP
        prod = np.array([1])
        type = 'pas'
        weights = 'nap_consumed_qsfdp6zt'
        keys = 'keys_qsfdp6zt'
        arith = f_update_default_controls(user_controls, 'napcon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'napcon', 'index', [5,4])   #p6 z
        cols = f_update_default_controls(user_controls, 'napcon', 'cols', [])
        axis_slice = f_update_default_controls(user_controls, 'napcon', 'axis_slice', {})
        reports["napcon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_poccon', 'Run']:
        #returns consumption in each FP
        prod = np.array([1])
        type = 'pas'
        weights = 'poc_consumed_qsfp6lz'
        keys = 'keys_qsfp6lz'
        arith = f_update_default_controls(user_controls, 'poccon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'poccon', 'index', [3])   #p6
        cols = f_update_default_controls(user_controls, 'poccon', 'cols', [5])     #z
        axis_slice = f_update_default_controls(user_controls, 'poccon', 'axis_slice', {})
        reports["poccon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_supcon', 'Run']:
        #returns consumption in each FP
        option = f_update_default_controls(user_controls, 'supcon', 'option', 1)
        reports["supcon"] = rfun.f_grain_sup_summary(lp_vars, r_vals, option=option)
    if report_run.loc['run_stubcon', 'Run']:
        #returns consumption in each FP
        prod = np.array([1])
        type = 'stub'
        weights = 'stub_qszp6fks1s2'
        keys = 'keys_qszp6fks1s2'
        arith = f_update_default_controls(user_controls, 'stubcon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'stubcon', 'index', [0,1,2,3]) #q,s,z,p6
        cols = f_update_default_controls(user_controls, 'stubcon', 'cols', [6])     #s1 (stub cat)
        axis_slice = f_update_default_controls(user_controls, 'stubcon', 'axis_slice', {})
        reports["stubcon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cropcon', 'Run']:
        ##crop grazing
        prod = np.array([1])
        type = 'crpgrz'
        weights = 'crop_consumed_qsfkp6p5zl'
        keys = 'keys_qsfkp6p5zl'
        arith = f_update_default_controls(user_controls, 'cropcon', 'arith', 2)
        index = f_update_default_controls(user_controls, 'cropcon', 'index', [0, 1, 6])  # q,s,z
        cols = f_update_default_controls(user_controls, 'cropcon', 'cols', [4,3]) #p6, k
        axis_slice = f_update_default_controls(user_controls, 'cropcon', 'axis_slice', {})
        reports["cropcon"] = rfun.f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                                                            keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cropcon', 'Run']:
        #returns consumption in each FP
        reports["cropcon_available"] = rfun.f_available_cropgrazing(r_vals)
    if report_run.loc['run_mvf', 'Run']:
        #returns consumption in each FP
        reports["mvf"] = rfun.f_mvf_summary(lp_vars)

    ##special reports for web app
    if report_run.loc['run_pasture_area', 'Run']:
        reports["pasture_area"] = rfun.f_pasture_area_analysis(lp_vars,r_vals,"Summary")
    if report_run.loc['run_stocking_rate', 'Run']:
        reports["stocking_rate"] = rfun.f_stocking_rate_analysis(lp_vars,r_vals,"Summary")
    if report_run.loc['run_legume', 'Run']:
        reports["legume"] = rfun.f_lupin_analysis(lp_vars,r_vals,"Summary")
    if report_run.loc['run_cropgraze', 'Run']:
        reports["cropgrazing"] = rfun.f_cropgrazing_analysis(lp_vars,r_vals,"Summary")
    return reports


def f_update_default_controls(user_controls, report_name, control_name, default_control):
    '''
    update the default report controls with specific user controls if available.

    :param user_controls: dict with all the user controls
    :param report_name: name of the report of interest
    :param control_name: name of the report control
    :param default_control: default control value
    :return: updated control value
    '''

    ##see if the customised values exist for this report and update accordingly
    try:
        control_value = user_controls[report_name][control_name]
    except KeyError:
        control_value = default_control

    return control_value