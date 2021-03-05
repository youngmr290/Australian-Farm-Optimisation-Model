"""
Created on Thu Dec 2 09:35:26 2020

@author: Young
"""

import pickle as pkl
import pandas as pd
import xlsxwriter

import ReportFunctions as rep
import Functions as fun

## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('Output/Report.xlsx',engine='xlsxwriter')

##read in exp log
exp_data_nosort = fun.f_read_exp()
exp_data_index = exp_data_nosort.index #need to use this so user can specify the trial number as per exp.xlsx
exp_data = exp_data_nosort.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx




# ##load pickle
# with open('pkl_lp_vars.pkl', "rb") as f:
#     lp_vars = pkl.load(f)
# with open('pkl_r_vals.pkl', "rb") as f:
#     r_vals = pkl.load(f)
# ###prev exp - used to determine if the report is using up to date data.
# with open('pkl_exp.pkl', "rb") as f:
#     prev_exp = pkl.load(f)

##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xlsx has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial
exp_data = fun.f_run_required(exp_data, check_pyomo=False)
trial_outdated = exp_data['run'] #returns true if trial is out of date

run_areasum = True #area summary
run_pnl = True #table of profit and loss
run_profitarea = True #graph profit by crop area
run_saleprice = True #table of gross saleprices for specified grids, weights & fat scores
run_cfw_dams = True #table of cfw
run_fec_dams = True #fec for the dams in each generator period
run_fec_offs = True #fec for the offspring in each generator period
run_weanper = True #table of weaning percent
run_scanper = True #table of scan percent
run_lamb_survival = True #table of lamb survival
run_daily_mei_dams = True #table of mei
run_daily_pi_dams = True #table of mei
run_numbers_dams = True #table of numbers
run_numbers_offs = True #table of numbers
run_dse = True #table of dse
run_grnfoo = True #table of green foo at end of fp
run_dryfoo = True #table of dry foo at end of fp
run_napfoo = True #table of nap foo at end of fp
run_grncon = True #table of green con during fp
run_drycon = True #table of dry con during fp
run_napcon = True #table of nap con during fp
run_poccon = True #table of poc con during fp
run_supcon = True #table of sup con during fp
run_stubcon = True #table of sup con during fp




def f_df2xl(writer, df, sheet, rowstart=0, colstart=0, option=0):
    '''
    Pandas to excel. https://xlsxwriter.readthedocs.io/working_with_pandas.html
        - You can simply stick a dataframe from pandas into excel using df.to_excel() function.
          for this you can specify the workbook the sheet and the start row or col (so you can put
          multiple dfs in one sheet)
        - The next level involves interacting with xlsxwriter. This allows you to do custom things like
          creating graphs, hiding rows/cols, filtering or grouping.

    :param writer: writer used. controls the workbook being writen to.
    :param df: dataframe going to excel
    :param sheet: str: sheet name.
    :param rowstart: start row in excel
    :param colstart: start col in excel
    :param option: int: specifying the writing option
                    0: df straight into excel
                    1: df into excel colapsing empty rows and cols
    '''
    ## simple write df to xl
    df.to_excel(writer, sheet, startrow=rowstart, startcol=colstart)

    ##set up xlsxwriter stuff needed for advanced options
    workbook = writer.book
    worksheet = writer.sheets[sheet]

    ## collapse rows and cols with all 0's
    if option==1:
        df = df.round(5)  # round so that very small numbers are dropped out in the next step
        for row in range(len(df)):
            if (df.iloc[row]==0).all():
                offset = df.columns.nlevels #number of columns used for names
                if offset>1:
                    offset += 1 #for some reason if the cols are multiindex the an extra row gets added when writing to excel
                worksheet.set_row(row+offset,None,None,{'level': 1, 'hidden': True}) #set hidden to true to collapse the level initially

        for col in range(len(df.columns)):
            if (df.iloc[:,col]==0).all():
                offset = df.index.nlevels
                col = xlsxwriter.utility.xl_col_to_name(col+offset) + ':' + xlsxwriter.utility.xl_col_to_name(col+offset) #convert col number to excel col reference eg 'A:B'
                worksheet.set_column(col,None,None,{'level': 1, 'hidden': True})
        return

    ##apply filter
    if option==2:
        # Activate autofilter
        worksheet.autofilter(f'B1:B{len(df)}')
        worksheet.filter_column('B', 'x < 5') # todo this will need to become function argument

        # Hide the rows that don't match the filter criteria.
        for idx,row_data in df.iterrows():
            region = row_data['Data']
            if not (region < 5):
                # We need to hide rows that don't match the filter.
                worksheet.set_row(idx + 1,options={'hidden': True})

    ##create chart
    if option==3:
        # Create a chart object.
        chart = workbook.add_chart({'type': 'column'}) # todo this will need to become function argument
        # Configure the series of the chart from the dataframe data.
        chart.add_series({'values': '=areasum!$B$2:$B$8'}) # todo this will need to become function argument
        # Insert the chart into the worksheet.
        worksheet.insert_chart('D2',chart)


##run report functions
if run_areasum:
    func = rep.f_area_summary
    trials = [34]
    option = 2
    areasum = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option)
    f_df2xl(writer, areasum, 'areasum', option=1)

if run_pnl:
    func = rep.f_profitloss_table
    trials = [34]
    pnl = rep.f_stack(func, trial_outdated, exp_data_index, trials)
    f_df2xl(writer, pnl, 'pnl', option=1)

if run_profitarea:
    func0 = rep.f_area_summary
    func1 = rep.f_profit
    func0_option = 4
    func1_option = 0
    trials = [34]
    plot = rep.f_xy_graph(func0, func1, trial_outdated, exp_data_index, trials, func0_option, func1_option)
    plot.savefig('Output/profitarea_curve.png')

if run_saleprice:
    func = rep.f_price_summary
    trials = [34]
    option = 2
    grid = [0,5,6]
    weight = [22,40,25]
    fs = [2,3,2]
    saleprice = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option, grid=grid, weight=weight, fs=fs)
    f_df2xl(writer, saleprice, 'saleprice', option=1)

if run_cfw_dams:
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    cfw_dams = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, cfw_dams, 'cfw_dams', option=1)

if run_fec_dams:
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    fec_dams = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           den_weights=den_weights, na_prod=na_prod, na_weights=na_weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, fec_dams, 'fec_dams', option=1)

if run_fec_offs:
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    fec_offs = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           den_weights=den_weights, na_prod=na_prod, na_weights=na_weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, fec_offs, 'fec_offs', option=1)

if run_lamb_survival:
    func = rep.f_survival_wean_scan
    trials = [0]
    option = 0
    arith_axis = [0,1,3,4,6,7,8,9,10]
    index =[2]
    cols =[5]
    axis_slice = {}
    lamb_survival = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option, arith_axis=arith_axis,
                                index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, lamb_survival, 'lamb_survival', option=1)

if run_weanper:
    func = rep.f_survival_wean_scan
    trials = [0]
    option = 1
    arith_axis = [0,2,3,4,5,6,7,8]
    index =[1]
    cols =[]
    axis_slice = {}
    lamb_survival = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option, arith_axis=arith_axis,
                                index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, lamb_survival, 'wean_per', option=1)

if run_scanper:
    func = rep.f_survival_wean_scan
    trials = [0]
    option = 2
    arith_axis = [0,2,3,4,5,6,7,8]
    index =[1]
    cols =[]
    axis_slice = {}
    lamb_survival = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option, arith_axis=arith_axis,
                                index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, lamb_survival, 'scan_per', option=1)


if run_daily_mei_dams:
    func = rep.f_stock_pasture_summary
    trials = [34]
    type = 'stock'
    prod = 'mei_dams_k2p6ftva1nw8ziyg1'
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
    daily_mei_dams = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                           na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                           arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, daily_mei_dams, 'daily_mei_dams', option=1)

if run_daily_pi_dams:
    func = rep.f_stock_pasture_summary
    trials = [34]
    type = 'stock'
    prod = 'pi_dams_k2p6ftva1nw8ziyg1'
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
    daily_pi_dams = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, prod=prod, weights=weights,
                               na_weights=na_weights, den_weights=den_weights, keys=keys, arith=arith,
                               arith_axis=arith_axis, index=index, cols=cols,
                               axis_slice=axis_slice)
    f_df2xl(writer, daily_pi_dams, 'daily_pi_dams', option=1)

if run_numbers_dams:
    func = rep.f_stock_pasture_summary
    trials = [34]
    type = 'stock'
    weights = 'dams_numbers_k2tvanwziy1g1'
    keys = 'dams_keys_k2tvanwziy1g1'
    arith = 2
    arith_axis = [3,4,5,6,7,8,9]
    index =[2]
    cols =[0,1]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    numbers_dams = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                           axis_slice=axis_slice)
    f_df2xl(writer, numbers_dams, 'numbers_dams', option=1)

if run_numbers_offs:
    func = rep.f_stock_pasture_summary
    trials = [34]
    type = 'stock'
    weights = 'offs_numbers_k3k5tvnwziaxyg3'
    keys = 'offs_keys_k3k5tvnwziaxyg3'
    arith = 2
    arith_axis = [4,5,6,7,8,9,10,11]
    index =[3]
    cols =[0,1,2]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    numbers_offs = rep.f_stack(func, trial_outdated, exp_data_index, trials, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols,
                           axis_slice=axis_slice)
    f_df2xl(writer, numbers_offs, 'numbers_offs', option=1)

if run_dse:
    func = rep.f_dse
    trials = [34]
    method = 0
    per_ha = True
    dse = rep.f_stack(func, trial_outdated, exp_data_index, trials, method = method, per_ha = per_ha)
    f_df2xl(writer, dse, 'dse', option=1)

if run_grnfoo:
    #returns foo at end of each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    grnfoo = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, grnfoo, 'grnfoo', option=1)

if run_dryfoo:
    #returns foo at end of each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    dryfoo = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, dryfoo, 'dryfoo', option=1)

if run_napfoo:
    #returns foo at end of each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    napfoo = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, napfoo, 'napfoo', option=1)

if run_grncon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    grncon = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, grncon, 'grncon', option=1)

if run_drycon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    drycon = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, drycon, 'drycon', option=1)

if run_napcon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
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
    napcon = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, napcon, 'napcon', option=1)

if run_poccon:
    #returns consumption in each fp
    func = rep.f_stock_pasture_summary
    trials = [34]
    prod = 1000
    type = 'pas'
    weights = 'poc_consumed_vflz'
    keys = 'keys_vflz'
    arith = 2
    arith_axis = [0,2]
    index =[1]
    cols =[]
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    poccon = rep.f_stack(func, trial_outdated, exp_data_index, trials, prod=prod, type=type, weights=weights,
                           keys=keys, arith=arith, arith_axis=arith_axis, index=index, cols=cols, axis_slice=axis_slice)
    f_df2xl(writer, poccon, 'poccon', option=1)

if run_supcon:
    #returns consumption in each fp
    func = rep.f_grain_sup_summary
    trials = [34]
    option = 1
    supcon = rep.f_stack(func, trial_outdated, exp_data_index, trials, option=option)
    f_df2xl(writer, supcon, 'supcon', option=1)

if run_stubcon:
    #returns consumption in each fp
    func = rep.f_stubble_summary
    trials = [34]
    stubcon = rep.f_stack(func, trial_outdated, exp_data_index, trials)
    f_df2xl(writer, stubcon, 'stubcon', option=1)




writer.save()