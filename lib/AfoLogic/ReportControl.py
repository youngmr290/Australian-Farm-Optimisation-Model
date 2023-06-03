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

def f_run_report(lp_vars, r_vals, report_run, trial_name, infeasible = None):
    '''Function to wrap ReportControl.py so that multiprocessing can be used.'''
    # print('Start processor: {0}'.format(processor))
    # print('Start trials: {0}'.format(trials))

    ## A control to switch between reporting the optimised production level (True) and the production assumptions (False)
    ### Note this is only active for some reports. It also changes the axes that are reported, often adding a w axis
    lp_vars_inc = False

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

    ##run report functions
    if report_run.loc['run_summary', 'Run']:
        reports["summary"] = rfun.f_summary(lp_vars,r_vals,"Summary")
    if report_run.loc['run_areasum', 'Run']:
        option = 0
        reports["areasum"] = rfun.f_area_summary(lp_vars, r_vals, option=option)
    if report_run.loc['run_profit', 'Run']:
        option = 4 #profit by zqs
        reports["profit"] = rfun.f_profit(lp_vars, r_vals, option=option)
    if report_run.loc['run_numbers_qsz', 'Run']:
        method = 0 #dse based on NW
        reports["numbers_qsz"] = rfun.f_dse(lp_vars, r_vals, method, per_ha=False, summary1=False, summary2=True)
    if report_run.loc['run_croparea_qsz', 'Run']:
        area_option = 2 #total crop area
        reports["croparea_qsz"] = rfun.f_area_summary(lp_vars,r_vals,area_option)
    if report_run.loc['run_pnl', 'Run']:
        option = 2 #1 = report q, s, & z. 2 = weighted average of q, s, & z
        reports["pnl"] = rfun.f_profitloss_table(lp_vars, r_vals, option=option)
    if report_run.loc['run_wc', 'Run']:
        reports["wc"] = rfun.f_wc_summary(lp_vars, r_vals)
    if report_run.loc['run_biomass_penalty', 'Run']:
        reports["penalty"] = rfun.f_biomass_penalty(lp_vars, r_vals)
    if report_run.loc['run_profitarea', 'Run']:
        area_option = 2     # 2 total crop area each season in p7[-1]
        profit_option = 0   # 0 Profit, 1 Risk neutral Obj, 2 Utility, 3 range and std dev of profit.
        profitarea = pd.DataFrame(index=[trial_name], columns=['area','profit'])
        profitarea.loc[trial_name, 'area'] = rfun.f_area_summary(lp_vars,r_vals,area_option).squeeze()
        profitarea.loc[trial_name,'profit'] = rfun.f_profit(lp_vars,r_vals,profit_option)
        reports["profitarea"] = profitarea
    if report_run.loc['run_feedbudget', 'Run']:
        option = 0      #0 mei/hd/day & propn from each source, 1 total mei
        nv_option = 0   #0 Separate NV pool, NV pool summed.
        dams_cols = [6] #birth opp
        offs_cols = [7] #shear opp
        reports["feed"] = rfun.f_feed_budget(lp_vars, r_vals, option=option, nv_option=nv_option, dams_cols=dams_cols, offs_cols=offs_cols)
    if report_run.loc['run_feedbudget', 'Run']:
        option = 1
        nv_option = 0
        dams_cols = [6] #birth opp
        offs_cols = [7] #shear opp
        reports["feed_total"] = rfun.f_feed_budget(lp_vars, r_vals, option=option, nv_option=nv_option, dams_cols=dams_cols, offs_cols=offs_cols)
    if report_run.loc['run_feedbudget', 'Run']:
        reports["grazing"] = rfun.f_grazing_summary(lp_vars, r_vals)
    if report_run.loc['run_period_dates', 'Run']:
        ###season nodes (p7)
        type = 'zgen'
        prod = 'date_season_node_p7z'
        keys = 'keys_p7z'
        arith = 0
        index =[0] #p7
        cols = [1] #z
        reports["season_nodes"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###feed periods (p6)
        type = 'pas'
        prod = 'fp_date_start_p6z'
        keys = 'keys_p6z'
        arith = 0
        index =[0] #p6
        cols = [1] #z
        reports["feed_periods"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###dams dvp
        type = 'stock'
        prod = 'dvp_start_vezg1'
        keys = 'dams_keys_vezg1'
        arith = 0
        index =[0] #v
        cols = [1,3,2] #e, g, z
        reports["dam_dvp_dates"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###dams repro dates
        type = 'stock'
        prod = 'r_repro_dates_roe1g1'
        keys = 'dams_keys_roeg1'
        arith = 0
        index =[1] #o
        cols = [0,2,3] #r, e, g
        reports["repro_dates"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
        ###offs dvp
        type = 'stock'
        prod = 'dvp_start_vzdxg3'
        keys = 'offs_keys_vzdxg3'
        arith = 0
        index =[0] #v
        cols = [4,3,2,1] #g, x, d, z
        reports["offs_dvp_dates"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                                                   keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_saleprice', 'Run']:
        option = 2
        grid = [0,5,6]
        weight = [22,40,25]
        fs = [2,3,2]
        reports["saleprice"] = rfun.f_price_summary(lp_vars, r_vals, option=option, grid=grid, weight=weight, fs=fs)
    if report_run.loc['run_salegrid_dams', 'Run']:
        type = 'stock'
        prod = 'salegrid_tva1e1b1nwziyg1'
        keys = 'dams_keys_tva1e1b1nwziyg1'
        arith = 0
        index =[1]
        cols = [10,0,4,6] #g,t,b,w
        reports["salegrid_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_salegrid_yatf', 'Run']:
        type = 'stock'
        prod = 'salegrid_Tva1e1b1nwzixyg2'
        keys = 'yatf_keys_Tvaebnwzixy1g2'
        arith = 0
        index =[1]
        cols = [11,9,0,4,6] #g,x,t,b,w
        reports["salegrid_yatf"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_salegrid_offs', 'Run']:
        type = 'stock'
        prod = 'salegrid_tvnwzida0e0b0xyg3'
        keys = 'offs_keys_tvnwzida0e0b0xyg3'
        arith = 0
        index =[1]
        cols = [0,3,8,9,10,12] #t,w,e,b,x,g
        reports["salegrid_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_saleage_offs', 'Run']:
        type = 'stock'
        prod = 'saleage_tvnwzida0e0b0xyg3'
        keys = 'offs_keys_tvnwzida0e0b0xyg3'
        arith = 0
        index =[1]
        cols = [0,6,9,10,12,3] #t,d,b,x,g,w  Note: need to exclude the w axis to report a N33 model
        reports["saleage_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod,
                               keys=keys, arith=arith, index=index, cols=cols)
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
        reports["salevalue_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
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
        reports["salevalue_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
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
        reports["salevalue_prog"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
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
        reports["woolvalue_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
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
        reports["woolvalue_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    if report_run.loc['run_saledate_offs', 'Run']:
        type = 'stock'
        prod = 'saledate_k3k5tvnwziaxyg3'
        na_prod = [0,1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = 1
        index = [5, 7]              #DVP, w
        cols = [13, 2, 3, 4, 11]     #g3, dam age, BTRT, t, gender
        reports["saledate_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols)
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
        reports["cfw_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_fd_dams', 'Run']:
        type = 'stock'
        prod = 'fd_hdmob_k2tva1nwziyg1'
        na_prod = [0,1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        arith = 1
        index =[4] #v
        cols =[11,3,2,8] #g,t,k2,w
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["fd_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["cfw_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["fd_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["wbe_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["wbe_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["lw_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                 , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights, den_weights=den_weights, na_denweights=na_denweights
                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["ffcfw_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights
                                 , na_weights=na_weights, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith
                                 , index=index, cols=cols, axis_slice=axis_slice)
    #todo remove after Triplets analysis
    if report_run.loc['run_ffcfw_cut_dams', 'Run']:
        type = 'stock'
        prod = 'ffcfw_dams_k2tvPa1b1nw8ziyg1'
        na_prod = [0, 1]  #q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        na_weights = [5,7]  #p
        keys = 'dams_keys_qsk2tvPabnwziy1g1'
        arith = 4   #average across slices if ffcfw>0
        index = [5]  #p
        cols = [2, 7]  #b1
        axis_slice = {}
        # axis_slice[2] = [2, 3, 1]     #the 11 slice  (in EL analysis only scanning for Preg Status)
        # axis_slice[4] = [0, 7, 1]  #All DVPs for Triplets
        reports["ffcfw_cut_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type
                                    , prod=prod, na_prod=na_prod
                                    , weights=weights, na_weights=na_weights
                                    # , prod_weights=prod_weights, na_prodweights=na_prodweights
                                    # , den_weights=den_weights, na_denweights=na_denweights
                                    , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["nv_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod,
                               prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights, na_weights=na_weights,
                               den_weights=den_weights, na_denweights=na_denweights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["ffcfw_yatf"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                 , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys
                                 , arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["ffcfw_prog"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod
                                                 , prod_weights=prod_weights, na_prodweights=na_prodweights, weights=weights
                                                 , na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                                                 , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["ffcfw_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
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
        arith = 1
        index =[6] #p
        cols =[2,15,3,4,8]  #k3,x,k5,t,w
        axis_slice = {}
        axis_slice[13] = [0,1,1] #first cycle
        axis_slice[11] = [2,-1,1] #Adult
        axis_slice[17] = [0,1,1] #BBB
        reports["nv_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                               , weights=weights, na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights
                               , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_lamb_survival', 'Run']:
        #axes are qsk2tvaeb9nwziy1g1      b9 axis is shorten b axis: [0,1,2,3]
        option = 0
        if lp_vars_inc:
            index =[4]      #v
            cols =[13,11,0,1,10,7]    #g,i,q,s,z & b9 #report must include the b axis otherwise an error is caused because the axis added after the arith.
        else:
            index = [4]     #v
            cols =[13,11,0,1,10,7,9]  #g,i,q,s,z,b & w
        axis_slice = {}
        reports["lamb_survival"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                             , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
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
        reports["weanper"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                       , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
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
        reports["scanper"] = rfun.f_lambing_status(lp_vars, r_vals, option=option, index=index, cols=cols
                                       , axis_slice=axis_slice, lp_vars_inc=lp_vars_inc)
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
        arith = 1               # for FP only
        index = [6, 3]          # [1]
        cols = [2, 4]           # [0]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["daily_mei_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
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
        arith = 1
        index =[6, 3] #v,p6
        cols =[2, 4] #k2, f
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["daily_pi_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
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
        arith = 1
        index =[7, 4]       #DVP, fp
        cols =[15, 3, 5]    #g3, BTRT, nv
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["daily_mei_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
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
        arith = 1
        index =[7, 4]       #DVP, fp
        cols =[15, 3, 5]    #g3, BTRT, nv
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["daily_pi_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                   na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights, keys=keys, arith=arith,
                                   index=index, cols=cols, axis_slice=axis_slice)
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
        reports["numbers_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["numbers_dams_p"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_prog', 'Run']:
        type = 'stock'
        weights = 'prog_numbers_qsk3k5twzia0xg2'
        keys = 'prog_keys_qsk3k5twzia0xg2'
        arith = 2
        index =[0,1,5]   #q, s, w
        cols =[6, 2, 3, 4, 9] #z, dam age, birth type, t slice, gender
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["numbers_prog"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_numbers_offs', 'Run']:
        type = 'stock'
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        arith = 2
        index =[5]                  #DVP
        cols =[8, 13, 11, 2, 3, 4, 7]   #z, g3, Gender, dam age, BTRT, t, w
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["numbers_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["numbers_offs_p"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
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
        reports["mort_dams"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
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
        reports["mort_offs"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols,
                               axis_slice=axis_slice)
    if report_run.loc['run_dse', 'Run']:
        ##you can go into f_dse to change the axis being reported.
        per_ha = True
        method = 0
        reports["dse_sire"], reports["dse_dams"], reports["dse_offs"] = rfun.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
        method = 1
        reports["dse1_sire"], reports["dse1_dams"], reports["dse1_offs"] = rfun.f_dse(lp_vars, r_vals, method = method, per_ha = per_ha)
    if report_run.loc['run_grnfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = 'foo_end_grnha_gop6lzt'
        na_prod = [0,1,2] #q,s,f
        weights = 'greenpas_ha_qsfgop6lzt'
        keys = 'keys_qsfgop6lzt'
        arith = 2
        index =[7,5]
        cols =[6]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["grnfoo"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        index =[7,5]
        cols =[6]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["pgr"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights, den_weights=den_weights,
                               na_denweights=na_denweights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_dryfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = np.array([1000])
        weights = 'drypas_transfer_qsdp6zlt'
        keys = 'keys_qsdp6zlt'
        arith = 2
        index =[4,3]
        cols =[2]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["dryfoo"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_napfoo', 'Run']:
        #returns foo at end of each FP
        type = 'pas'
        prod = np.array([1000])
        weights = 'nap_transfer_qsdp6zt'
        keys = 'keys_qsdp6zt'
        arith = 2
        index =[4,3]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["napfoo"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
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
        arith = 1
        index =[7,5]
        cols =[8]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["grncon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, prod_weights=prod_weights,
                                type=type, weights=weights, den_weights=den_weights, na_denweights=na_denweights,
                                keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drycon', 'Run']:
        #returns total consumption per day in each FP
        #todo once this is change to per ha variable then change to report consumption per ha per day (same as grn pas)
        type = 'pas'
        prod = np.array([1])
        weights = 'drypas_consumed_qsfdp6zlt'
        keys = 'keys_qsfdp6zlt'
        arith = 2
        index =[5,4]
        cols =[3,7] #d,t
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["drycon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_grnnv', 'Run']:
        #returns NV during each FP regardless of whether selected or not
        type = 'pas'
        prod = 'nv_grnha_fgop6lzt'
        weights = None
        keys = 'keys_fgop6lzt'
        arith = 5
        index = [5,3]
        cols = [2, 1]
        axis_slice = {}
        reports["grnnv"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_grndmd', 'Run']:
        #returns DMD during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dmd_diet_grnha_gop6lzt'
        weights = None
        keys = 'keys_gop6lzt'
        arith = 5
        index = [4,2]
        cols = [1, 0]
        axis_slice = {}
        reports["grndmd"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_avegrnfoo', 'Run']:
        #returns average FOO during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'foo_ave_grnha_gop6lzt'
        weights = None
        keys = 'keys_gop6lzt'
        arith = 5
        index = [4,2]
        cols = [1, 0]
        axis_slice = {}
        reports["grnfoo"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drynv', 'Run']:
        #returns NV during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'nv_dry_fdp6zt'
        weights = None
        keys = 'keys_fdp6zt'
        arith = 5
        index = [3,2]
        cols = [1]
        axis_slice = {}
        reports["drynv"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_drydmd', 'Run']:
        #returns DMD during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dry_dmd_dp6zt'
        weights = None
        keys = 'keys_dp6zt'
        arith = 5
        index = [2,1]
        cols = [0]
        axis_slice = {}
        reports["drydmd"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_avedryfoo', 'Run']:
        #returns average FOO during each FP (regardless of whether selected or not)
        type = 'pas'
        prod = 'dry_foo_dp6zt'
        weights = None
        keys = 'keys_dp6zt'
        arith = 5
        index = [2,1]
        cols = [0]
        axis_slice = {}
        reports["dryfoo"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_napcon', 'Run']:
        #returns consumption in each FP
        prod = np.array([1])
        type = 'pas'
        weights = 'nap_consumed_qsfdp6zt'
        keys = 'keys_qsfdp6zt'
        arith = 2
        index =[5,4]
        cols =[]
        axis_slice = {}
        # axis_slice[0] = [0, 2, 1]
        reports["napcon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
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
        reports["poccon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_supcon', 'Run']:
        #returns consumption in each FP
        option = 1
        reports["supcon"] = rfun.f_grain_sup_summary(lp_vars, r_vals, option=option)
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
        reports["stubcon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                               keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cropcon', 'Run']:
        ##crop grazing
        prod = np.array([1])
        type = 'crpgrz'
        weights = 'crop_consumed_qsfkp6p5zl'
        keys = 'keys_qsfkp6p5zl'
        arith = 2
        index = [0, 1, 6]  # q,s,z
        cols = [4,3] #p6
        axis_slice = {}
        reports["cropcon"] = rfun.f_stock_pasture_summary(lp_vars, r_vals, prod=prod, type=type, weights=weights,
                                                            keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    if report_run.loc['run_cropcon', 'Run']:
        #returns consumption in each FP
        reports["cropcon_available"] = rfun.f_available_cropgrazing(lp_vars, r_vals)
    if report_run.loc['run_mvf', 'Run']:
        #returns consumption in each FP
        reports["mvf"] = rfun.f_mvf_summary(lp_vars)

    return reports
