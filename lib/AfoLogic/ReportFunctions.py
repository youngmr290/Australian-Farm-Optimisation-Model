"""

This module contains function which are used for three main jobs:

#.  General jobs including error checking and reading and writing output files.
#.  'within trial' calculations. These functions manipulate the output variables from the LP and the
    values from the precalcs. These functions all return a data array.
#.  'between trial' calculations. These functions summarise the data array from the 'within trial' functions
    for all the trials in an experiment. These functions can return table or figures.

.. note:: This module should not import inputs (in case the inputs are adjusted during the exp
    so they will not be correct for r_vals)

.. tip:: There are cases where the report requires more detail than the decision variable returned from
    the lp contains. For example some livestock reports (e.g. NV) need an e b and p axis but the
    clustering has removed this level of detail in the lp variables.
    To handle this we make the corresponding r_vals array with both the detailed and clustered axis.

author: young
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import os.path
import sys
import xlsxwriter
import datetime as dt

from . import Functions as fun
from . import Exceptions as exc

na = np.newaxis

###################
#general functions#
###################
def f_df2xl(writer, df, sheet, df_settings=None, rowstart=0, colstart=0, option=0):
    '''
    Pandas to excel. https://xlsxwriter.readthedocs.io/working_with_pandas.html

        - You can simply stick a dataframe from pandas into Excel using df.to_excel() function.
          for this you can specify the workbook the sheet and the start row or col (so you can put
          multiple dfs in one sheet)
        - The next level involves interacting with xlsxwriter. This allows you to do custom things like
          creating graphs, hiding rows/cols, filtering or grouping.

    :param writer: writer used. controls the workbook being writen to.
    :param df: dataframe going to excel
    :param sheet: str: sheet name.
    :param df_settings: df: df to store number of row and col indexes.
    :param rowstart: start row in Excel
    :param colstart: start col in Excel
    :param option: int: specifying the writing option
                    0: df straight into Excel
                    1: df into Excel collapsing empty rows and cols (using the group function is xl - ie the rows/cols are still there they are just minimised)
                    2: df into Excel removing empty rows and cols (the rows/cols are completely removed)
    '''
    ##store df settings
    if df_settings is not None:
        df_settings.loc[sheet] = [df.index.nlevels, df.columns.nlevels]
    
    ## round to tidy and so that very small numbers are dropped out in the next step
    df = df.round(3)  
    
    ## Remove rows and cols with all 0's
    if option==2:
        ###rows are removed completely to reduce writing time - if all rows are 0 then make the last row True because cant write an empty df to xl.
        row_mask = (df != 0).any(axis=1)
        col_mask = (df != 0).any(axis=0)
        if (row_mask==False).all() and len(row_mask)>0:
            row_mask[-1] = True
        if (col_mask==False).all() and len(col_mask)>0:
            col_mask[-1] = True
        df = df.loc[row_mask, col_mask]
    
    ## simple write df to xl
    ###if df is empty just create a simple empty df. Empty df with multiindex causes error.
    if df.empty:
        df=pd.DataFrame()
    df.to_excel(writer, sheet, startrow=rowstart, startcol=colstart)

    ##set up xlsxwriter stuff needed for advanced options
    workbook = writer.book
    worksheet = writer.sheets[sheet]

    ## collapse cols with all 0's (rows are removed
    if option==1:
        for row in range(len(df)-1):   #range(len(df)) hides the last blank row but causes a blank line in some of report.xl
            if (df.iloc[row]==0).all():
                offset = df.columns.nlevels #number of columns used for names
                if offset>1:
                    offset += 1 #for some reason if the cols are multiindex the an extra row gets added when writing to excel
                worksheet.set_row(row+offset,None,None,{'level': 1, 'hidden': True}) #set hidden to true to collapse the level initially

        for col in range(len(df.columns)):
            if (df.iloc[:,col]==0).all():
                offset = df.index.nlevels
                col = xlsxwriter.utility.xl_col_to_name(col+offset) + ':' + xlsxwriter.utility.xl_col_to_name(col+offset) #convert col number to excel col reference e.g. 'A:B'
                worksheet.set_column(col,None,None,{'level': 1, 'hidden': True})

    ##apply filter
    if option==3:
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
    if option==4:
        # Create a chart object.
        chart = workbook.add_chart({'type': 'column'}) # todo this will need to become function argument
        # Configure the series of the chart from the dataframe data.
        chart.add_series({'values': '=areasum!$B$2:$B$8'}) # todo this will need to become function argument
        # Insert the chart into the worksheet.
        worksheet.insert_chart('D2',chart)

    return df_settings

def f_errors(trial_outdated, trials):
    '''
    The report module conducts three error checks before commencing:

    #. Have you run the main model for each trial you are trying to report. If not a warning will be printed and
       the given trial will be removed from the list of trials to report (allowing the remaining trials to still
       be reported).
    #. Are any trials out of date E.g. have you run the main model since updating the inputs or the code.
       If trials are out of date a warning message will be printed but the report code will continue to execute.
    #. Did all the trials you are reporting solve optimally. Infeasible trials are still reported the reports are just
       filled with 0's. A list of infeasible trials will be printed to the console and reported in the final Report.xlsx
       document.

    :param trial_outdated: boolean list of all trials stating if trial is out of date.
    :param trials: list of trials being run
    :return: None.
    '''
    ##first check if data exists for each desired trial
    infeasible_trials=[]
    for trial_name in trials:
        if os.path.isfile('Output/infeasible/{0}.txt'.format(trial_name)):
            infeasible_trials.append(trial_name)
        else:
            pass
    if infeasible_trials:
        print("Infeasible trials being reported:\n", infeasible_trials)

    ##second check if data exists for each desired trial
    non_exist_trials = []
    for trial_name in trials:
        if os.path.isfile('pkl/pkl_r_vals_{0}.pkl'.format(trial_name)):
            pass
        else:
            print('''WARNING: Trials for reporting don't all exist''')
            trials = trials[trials!=trial_name] #remove trials that don't exist from the list of trials to run.
            non_exist_trials.append(trial_name)

    ##third check if generating results using out of date data.
    outdatedbool = trial_outdated.loc[(slice(None), slice(None), slice(None), trials)].values  # have to use the trial name because the order is different
    if any(outdatedbool):  # have to use the trial name because the order is different
        print('''

              Generating reports from out dated data: Trial %s
                
              ''' %np.array(trials)[outdatedbool])
    return trials, non_exist_trials

def load_pkl(trial_name):
    '''load in lp_vars and r_vals output file.
    '''
    with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"rb") as f:
        lp_vars = pkl.load(f)
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"rb") as f:
        r_vals = pkl.load(f)
    return lp_vars, r_vals

def f_vars2np(lp_vars, var_key, axis_keys=None, maskz8=None, z_pos=-1):
    '''
    Converts lp_vars to numpy.

    :param lp_vars: dict of lp variables
    :param var_key: string - name of variable to convert to numpy
    :param axis_keys: list of lists/arrays, one per axis, giving the labels (e.g. [keys_q, keys_s, keys_f, ...])
    :param maskz8: z8 mask. Must be broadcastable to lp_vars
    :param z_pos: position of z axis
    :return: numpy array with un-clustered season axis.
    '''

    var_comp = lp_vars[var_key]
    shape = tuple(len(k) for k in axis_keys)
    arr = np.zeros(shape, dtype=float)

    axis_maps = [{label: i for i, label in enumerate(k)} for k in axis_keys]

    for idx, var in var_comp.items():
        if not isinstance(idx, tuple):
            idx = (idx,)
        pos = tuple(axis_maps[a][label] for a, label in enumerate(idx))
        arr[pos] = var#float(var.value or 0.0)

    arr[np.isnan(arr)] = 0  # replace nan with 0

    ##uncluster z so that each season gets complete information
    if maskz8 is not None:
        index_z = fun.f_expand(np.arange(maskz8.shape[z_pos]), z_pos)
        a_zcluster = np.maximum.accumulate(index_z * maskz8,axis=z_pos)
        a_zcluster = np.broadcast_to(a_zcluster, arr.shape)
        arr = np.take_along_axis(arr, a_zcluster, axis=z_pos)

    return arr


def f_vars2df(lp_vars, var_key, maskz8=None, z_pos=-1):
    '''
    converts lp_vars to pandas series.
    :param lp_vars: dict of variables.
    :param var_key: string - name of variable to convert to series.
    :param maskz8: z8 mask. Must be broadcastable to lp_vars once lp_vars if numpy
    :param z_pos: position (level) of z axis

    :return: series with season as index level 0
    '''

    vars = pd.Series(lp_vars[var_key])
    vars = vars.sort_index()

    ##uncluster z so that each season gets complete information
    if maskz8 is not None:
        ###store index before convert to np
        index = vars.index
        ###reshape array to be numpy
        reshape_size = vars.index.remove_unused_levels().levshape  # create a tuple with the rights dimensions
        vars = np.reshape(vars.values,reshape_size)
        ###uncluster numpy
        index_z = fun.f_expand(np.arange(maskz8.shape[z_pos]),z_pos)
        a_zcluster = np.maximum.accumulate(index_z * maskz8,axis=z_pos)
        a_zcluster = np.broadcast_to(a_zcluster,vars.shape)
        vars = np.take_along_axis(vars,a_zcluster,axis=z_pos)
        ###convert back to pd
        vars = pd.Series(vars.ravel(),index=index)

    return vars




def f_append_dfs(stacked_df, additional_df):
    new_stacked_df = pd.concat([stacked_df,additional_df], axis=0)
    ##reset index order. If two dfs are appended with different columns the pandas append function sorts the index.
    cols = stacked_df.columns.union(additional_df.columns,sort=False)
    new_stacked_df = new_stacked_df.reindex(cols,axis=1)
    return new_stacked_df.fillna(0) #fill na with 0 so that the function that writes to xl can hide the rows/cols (na gets entered if the two dfs being appended don't have all the same cols)

########################
# across trial reports #
########################
def f_xy_graph(data):
    '''Generic x-y line graphing function.

    :param data: df with data to plot. First col contains x values and second col contains y values
    :return: x-y plot

    '''
    ##loop through trials and generate pnl table
    x_vals = data.iloc[:,0]  # create list to append pnl table from each trial
    y_vals = data.iloc[:,1]  # create list to append pnl table from each trial
    plt.plot(x_vals, y_vals)
    return plt


###################
# variable setup  #
###################
def f_var_reshape(lp_vars, r_vals):
    '''
    Reshapes variables and creates a qsz weighted version.

    This function gets called once per trial at the start of the reporting process (doesnt get called for each report).
    The resulting dictionary is stored as a global variable so it can be access by all reports.

    Use the base version when z is being reported.
    Use the qsz weighted version when z is being averaged (i.e. not reported).

    Returns a dictionary with params.
    '''
    ##keys
    keys_a = r_vals['stock']['keys_a']
    keys_d = r_vals['stock']['keys_d']
    keys_dry_pool = r_vals['pas']['keys_d']
    keys_g = r_vals['pas']['keys_g']
    keys_g0 = r_vals['stock']['keys_g0']
    keys_g1 = r_vals['stock']['keys_g1']
    keys_g2 = r_vals['stock']['keys_g2']
    keys_g3 = r_vals['stock']['keys_g3']
    keys_f = r_vals['stock']['keys_f']
    keys_h1 = r_vals['stock']['keys_h1']
    keys_i = r_vals['stock']['keys_i']
    keys_k = r_vals['pas']['keys_k']
    keys_k1 = r_vals['stub']['keys_k1']
    keys_k2 = r_vals['stock']['keys_k2']
    keys_k3 = r_vals['stock']['keys_k3']
    keys_k5 = r_vals['stock']['keys_k5']
    keys_l = r_vals['pas']['keys_l']
    keys_lw1 = r_vals['stock']['keys_lw1']
    keys_lw3 = r_vals['stock']['keys_lw3']
    keys_lw_prog = r_vals['stock']['keys_lw_prog']
    keys_n1 = r_vals['stock']['keys_n1']
    keys_n3 = r_vals['stock']['keys_n3']
    keys_o = r_vals['pas']['keys_o']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_p7 = r_vals['fin']['keys_p7']
    keys_p8 = r_vals['stock']['keys_p8']
    keys_q = r_vals['zgen']['keys_q']
    keys_r = r_vals['pas']['keys_r']
    keys_s = r_vals['zgen']['keys_s']
    keys_s1 = r_vals['stub']['keys_s1']
    keys_s2 = r_vals['stub']['keys_s2']
    keys_t = r_vals['pas']['keys_t']
    keys_t1 = r_vals['stock']['keys_t1']
    keys_t2 = r_vals['stock']['keys_t2']
    keys_t3 = r_vals['stock']['keys_t3']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']
    keys_y0 = r_vals['stock']['keys_y0']
    keys_y1 = r_vals['stock']['keys_y1']
    keys_y3 = r_vals['stock']['keys_y3']
    keys_x = r_vals['stock']['keys_x']
    keys_z = r_vals['zgen']['keys_z']
    keys_pastures = r_vals['pas']['keys_pastures']

    ##create dict for reshaped variables
    global d_keys
    d_keys = {}
    global d_vars
    d_vars = {}
    d_vars['base'] = {}
    d_vars['qsz_weighted'] = {}
    

    ########
    ##keys #
    ########
    #note stock keys are built in sgen.
    ###pasture
    d_keys['keys_qsfgop6lzt'] = [keys_q, keys_s, keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    d_keys['keys_qfgop6lzt'] = [keys_q, keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    d_keys['keys_qgop6lzt'] = [keys_q, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    d_keys['keys_qsfdp6zt'] = [keys_q, keys_s, keys_f, keys_dry_pool, keys_p6, keys_z, keys_t]
    d_keys['keys_qsfdp6zlt'] = [keys_q, keys_s, keys_f, keys_dry_pool, keys_p6, keys_z, keys_l, keys_t]
    d_keys['keys_fdp6zt'] = [keys_f, keys_dry_pool, keys_p6, keys_z, keys_t]
    d_keys['keys_qsdp6zt'] = [keys_q, keys_s, keys_dry_pool, keys_p6, keys_z, keys_t]
    d_keys['keys_qsdp6zlt'] = [keys_q, keys_s, keys_dry_pool, keys_p6, keys_z, keys_l, keys_t]
    d_keys['keys_dp6zt'] = [keys_dry_pool, keys_p6, keys_z, keys_t]
    d_keys['keys_qsfp6lz'] = [keys_q, keys_s, keys_f, keys_p6, keys_l, keys_z]
    ###crop residue
    d_keys['keys_qszp6fks1s2'] = [keys_q, keys_s, keys_z, keys_p6, keys_f, keys_k1, keys_s1, keys_s2]
    ###crop grazing
    d_keys['keys_qsfkp6p5zl'] = [keys_q, keys_s, keys_f, keys_k1, keys_p6, keys_p5, keys_z, keys_l]
    ###saltbush
    d_keys['keys_qszp6fl'] = [keys_q, keys_s, keys_z, keys_p6, keys_f, keys_l]
    d_keys['keys_qsp7zl'] = [keys_q, keys_s, keys_p7, keys_z, keys_l]
    d_keys['keys_qszl'] = [keys_q, keys_s, keys_z, keys_l]
    ###trees
    d_keys['keys_p7zl'] = [keys_p7, keys_z, keys_l]
    ###periods
    d_keys['keys_p7z'] = [keys_p7, keys_z]
    d_keys['keys_p6z'] = [keys_p6, keys_z]
    ##machine
    d_keys['keys_qszp5kl'] = [keys_q, keys_s, keys_z, keys_p5, keys_k, keys_l]
    d_keys['keys_qszp5k'] = [keys_q, keys_s, keys_z, keys_p5, keys_k1]
    ##biomass
    d_keys['keys_qsp7zkls2'] = [keys_q, keys_s, keys_p7, keys_z, keys_k1, keys_l, keys_s2]
    ##v_phase residue
    d_keys['keys_qsp7p6zrlt'] = [keys_q, keys_s, keys_p7, keys_p6, keys_z, keys_r, keys_l, keys_t]
    d_keys['keys_qsp7zrl'] = [keys_q, keys_s, keys_p7, keys_z, keys_r, keys_l]
    ##trees
    d_keys['keys_l'] = [keys_l]


    ##############
    ##variables  #
    ##############
    prob_qsz = r_vals['zgen']['z_prob_qsz']
    ##stock
    ###sire
    sire_numbers_qsg0 = f_vars2np(lp_vars, 'v_sire', [keys_q, keys_s, keys_g0]).astype(float)
    sire_numbers_qszg0 = sire_numbers_qsg0[:,:,na,:] #give sire a singleton z axis (same numbers of sires in all z)
    d_vars['base']['sire_numbers_qszg0'] = sire_numbers_qszg0
    d_vars['qsz_weighted']['sire_numbers_qszg0'] = sire_numbers_qszg0 * prob_qsz[...,na]
    ###dams
    maskz8_k2tvanwziy1g1 = r_vals['stock']['maskz8_k2tvanwziy1g1']
    dams_numbers_qsk2tvanwziy1g1 = f_vars2np(lp_vars, 'v_dams', r_vals['stock']['dams_keys_qsk2tvanwziy1g1'], maskz8_k2tvanwziy1g1, z_pos=-4).astype(float)
    d_vars['base']['dams_numbers_qsk2tvanwziy1g1'] = dams_numbers_qsk2tvanwziy1g1
    d_vars['qsz_weighted']['dams_numbers_qsk2tvanwziy1g1'] = dams_numbers_qsk2tvanwziy1g1 * prob_qsz[...,na,na,na,na,na,na,:,na,na,na]
    ###prog
    prog_numbers_qsk3k5twzia0xg2 = f_vars2np(lp_vars, 'v_prog', r_vals['stock']['prog_keys_qsk3k5twzia0xg2']).astype(float)
    d_vars['base']['prog_numbers_qsk3k5twzia0xg2'] = prog_numbers_qsk3k5twzia0xg2
    d_vars['qsz_weighted']['prog_numbers_qsk3k5twzia0xg2'] = prog_numbers_qsk3k5twzia0xg2 * prob_qsz[...,na,na,na,na,:,na,na,na,na]
    ###offs
    maskz8_k3k5tvnwziaxyg3 = r_vals['stock']['maskz8_k3k5tvnwziaxyg3']
    offs_numbers_qsk3k5tvnwziaxyg3 = f_vars2np(lp_vars, 'v_offs', r_vals['stock']['offs_keys_qsk3k5tvnwziaxyg3'], maskz8_k3k5tvnwziaxyg3, z_pos=-6).astype(float)
    d_vars['base']['offs_numbers_qsk3k5tvnwziaxyg3'] = offs_numbers_qsk3k5tvnwziaxyg3
    d_vars['qsz_weighted']['offs_numbers_qsk3k5tvnwziaxyg3'] = offs_numbers_qsk3k5tvnwziaxyg3 * prob_qsz[...,na,na,na,na,na,na,:,na,na,na,na,na]

    ##biomass
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    v_use_biomass_qsp7zkls2 = f_vars2np(lp_vars, 'v_use_biomass', d_keys['keys_qsp7zkls2'], mask_season_p7z[:, :, na, na, na], z_pos=-4)
    d_vars['base']['v_use_biomass_qsp7zkls2'] = v_use_biomass_qsp7zkls2
    d_vars['qsz_weighted']['v_use_biomass_qsp7zkls2'] = v_use_biomass_qsp7zkls2 * prob_qsz[:,:,na,:,na,na,na]

    
    ##feedsupply
    ###reshape z8 mask to uncluster
    maskz8_p6z = r_vals['pas']['mask_fp_z8var_p6z']
    maskz8_zp6 = maskz8_p6z.T
    maskz8_p6zna = maskz8_p6z[:, :, na]
    maskz8_p6znana = maskz8_p6z[:, :, na, na]
    maskz8_p6naz = maskz8_p6z[:, na, :]
    maskz8_p6nazna = maskz8_p6z[:, na, :, na]
    ###green pasture hectare variable
    greenpas_ha_qsfgop6lzt = f_vars2np(lp_vars, 'v_greenpas_ha', d_keys['keys_qsfgop6lzt'], maskz8_p6nazna, z_pos=-2)
    d_vars['base']['greenpas_ha_qsfgop6lzt'] = greenpas_ha_qsfgop6lzt
    d_vars['qsz_weighted']['greenpas_ha_qsfgop6lzt'] = greenpas_ha_qsfgop6lzt * prob_qsz[...,na,na,na,na,na,:,na]
    ###dry end period
    drypas_transfer_qsdp6zlt = f_vars2np(lp_vars, 'v_drypas_transfer', d_keys['keys_qsdp6zlt'], maskz8_p6znana, z_pos=-3)
    d_vars['base']['drypas_transfer_qsdp6zlt'] = drypas_transfer_qsdp6zlt
    d_vars['qsz_weighted']['drypas_transfer_qsdp6zlt'] = drypas_transfer_qsdp6zlt * prob_qsz[...,na,na,:,na,na]
    ###nap end period
    nap_transfer_qsdp6zt = f_vars2np(lp_vars, 'v_nap_transfer', d_keys['keys_qsdp6zt'], maskz8_p6zna, z_pos=-2)
    d_vars['base']['nap_transfer_qsdp6zt'] = nap_transfer_qsdp6zt
    d_vars['qsz_weighted']['nap_transfer_qsdp6zt'] = nap_transfer_qsdp6zt * prob_qsz[...,na,na,:,na]
    ###dry consumed
    drypas_consumed_qsfdp6zlt = f_vars2np(lp_vars, 'v_drypas_consumed', d_keys['keys_qsfdp6zlt'], maskz8_p6znana, z_pos=-3)
    d_vars['base']['drypas_consumed_qsfdp6zlt'] = drypas_consumed_qsfdp6zlt
    d_vars['qsz_weighted']['drypas_consumed_qsfdp6zlt'] = drypas_consumed_qsfdp6zlt * prob_qsz[...,na,na,na,:,na,na]
    ###nap consumed
    nap_consumed_qsfdp6zt = f_vars2np(lp_vars, 'v_nap_consumed', d_keys['keys_qsfdp6zt'], maskz8_p6zna, z_pos=-2)
    d_vars['base']['nap_consumed_qsfdp6zt'] = nap_consumed_qsfdp6zt
    d_vars['qsz_weighted']['nap_consumed_qsfdp6zt'] = nap_consumed_qsfdp6zt * prob_qsz[...,na,na,na,:,na]
    ###poc consumed
    poc_consumed_qsfp6lz = f_vars2np(lp_vars, 'v_poc', d_keys['keys_qsfp6lz'], maskz8_p6naz, z_pos=-1)
    d_vars['base']['poc_consumed_qsfp6lz'] = poc_consumed_qsfp6lz
    d_vars['qsz_weighted']['poc_consumed_qsfp6lz'] = poc_consumed_qsfp6lz * prob_qsz[...,na,na,na,:]
    ###stubble consumed
    stub_qszp6fks1s2 = f_vars2np(lp_vars, 'v_stub_con', d_keys['keys_qszp6fks1s2'], maskz8_zp6[:, :, na, na, na, na], z_pos=-6)
    d_vars['base']['stub_qszp6fks1s2'] = stub_qszp6fks1s2
    d_vars['qsz_weighted']['stub_qszp6fks1s2'] = stub_qszp6fks1s2 * prob_qsz[...,na,na,na,na,na]
    ###crop consumed
    crop_consumed_qsfkp6p5zl = f_vars2np(lp_vars, 'v_tonnes_crop_consumed', d_keys['keys_qsfkp6p5zl'], maskz8_p6nazna, z_pos=-2)
    d_vars['base']['crop_consumed_qsfkp6p5zl'] = crop_consumed_qsfkp6p5zl
    d_vars['qsz_weighted']['crop_consumed_qsfkp6p5zl'] = crop_consumed_qsfkp6p5zl * prob_qsz[...,na,na,na,na,:,na]
    ###saltbush consumed
    v_tonnes_sb_consumed_qszp6fl = f_vars2np(lp_vars, 'v_tonnes_sb_consumed', d_keys['keys_qszp6fl'], maskz8_zp6[:, :, na, na], z_pos=-4)
    d_vars['base']['v_tonnes_sb_consumed_qszp6fl'] = v_tonnes_sb_consumed_qszp6fl
    d_vars['qsz_weighted']['v_tonnes_sb_consumed_qszp6fl'] = v_tonnes_sb_consumed_qszp6fl * prob_qsz[...,na,na,na]
    ###area saltbush
    v_slp_ha_qszl = f_vars2np(lp_vars, 'v_slp_ha', d_keys['keys_qszl'], z_pos=-2)
    d_vars['base']['v_slp_ha_qszl'] = v_slp_ha_qszl
    d_vars['qsz_weighted']['v_slp_ha_qszl'] = v_slp_ha_qszl * prob_qsz[...,na]

    ##Machine
    ###reshape z8 mask, used to uncluster
    maskz8_p5z = r_vals['lab']['maskz8_p5z']
    maskz8_zp5 = maskz8_p5z.T
    maskz8_zp5nana = maskz8_zp5[:, :, na, na]
    maskz8_zp5na = maskz8_zp5[:, :, na]
    ###mach variable
    v_contractseeding_ha = f_vars2np(lp_vars, 'v_contractseeding_ha', d_keys['keys_qszp5kl'], maskz8_zp5nana, z_pos=-4)
    d_vars['base']['v_contractseeding_ha'] = v_contractseeding_ha
    d_vars['base']['v_contractseeding_ha'] = v_contractseeding_ha * prob_qsz[...,na,na,na]
    v_seeding_machdays = f_vars2np(lp_vars, 'v_seeding_machdays', d_keys['keys_qszp5kl'], maskz8_zp5nana, z_pos=-4)
    d_vars['base']['v_seeding_machdays'] = v_seeding_machdays
    d_vars['qsz_weighted']['v_seeding_machdays'] = v_seeding_machdays * prob_qsz[...,na,na,na]
    v_harv_hours = f_vars2np(lp_vars, 'v_harv_hours', d_keys['keys_qszp5k'], maskz8_zp5na, z_pos=-3)
    d_vars['base']['v_harv_hours'] = v_harv_hours
    d_vars['qsz_weighted']['v_harv_hours'] = v_harv_hours * prob_qsz[...,na,na]
    v_contractharv_hours = f_vars2np(lp_vars, 'v_contractharv_hours', d_keys['keys_qszp5k'], maskz8_zp5na, z_pos=-3)
    d_vars['base']['v_contractharv_hours'] = v_contractharv_hours
    d_vars['qsz_weighted']['v_contractharv_hours'] = v_contractharv_hours * prob_qsz[...,na,na]

    ##v_phase
    v_phase_area_qsp7zrl = f_vars2np(lp_vars, 'v_phase_area', d_keys['keys_qsp7zrl'], mask_season_p7z[:, :, na, na], z_pos=-3)
    d_vars['base']['v_phase_area_qsp7zrl'] = v_phase_area_qsp7zrl
    d_vars['qsz_weighted']['v_phase_area_qsp7zrl'] = v_phase_area_qsp7zrl * prob_qsz[:,:,na,:,na,na]
    v_phase_change_increase_qsp7zrl = f_vars2np(lp_vars, 'v_phase_change_increase', d_keys['keys_qsp7zrl'], mask_season_p7z[:, :, na, na], z_pos=-3)
    d_vars['base']['v_phase_change_increase_qsp7zrl'] = v_phase_change_increase_qsp7zrl
    d_vars['qsz_weighted']['v_phase_change_increase_qsp7zrl'] = v_phase_change_increase_qsp7zrl * prob_qsz[:,:,na,:,na,na]

    ##trees
    v_tree_area_l = f_vars2np(lp_vars, 'v_tree_area_l', [keys_l])
    d_vars['base']['v_tree_area_l'] = v_tree_area_l
    d_vars['qsz_weighted']['v_tree_area_l'] = v_tree_area_l #doesnt need to be weight by qsz because doesnt vary



#########################################
# intermediate report building functions#
#########################################
def f_price_summary(lp_vars, r_vals, option, grid, weight, score):
    '''Returns price summaries. Prices are before subtracting selling costs.

    :param r_vals:
    :param option:

            #. farmgate grain price
            #. wool price STB price for FNF (free or nearly free of fault)
            #. sale price for specified grid at given weight and fat score (sale yard price)

    :param grid: list - sale grids to report. Has to be int between 0 and 7 inclusive.
    :param weight: float/int - stock weight to report price for.
    :param score: int - fat score to report price for. Has to be number between 1-5 inclusive.
    :return: price summary df
    '''

    ##grain price - farmgate (price received by farmer)
    if option == 0:
        return r_vals['crop']['farmgate_price']

    ##wool price - grid price
    if option == 1:
        return pd.Series(r_vals['stock']['woolp_mpg_w4'], index=r_vals['fd_range'])

    ##sale price - grid price
    if option == 2:
        ###create dataframe
        sale_index = pd.MultiIndex(levels=[[], [], []],
                                   codes=[[], [], []],
                                   names=['Grid', 'Weight', 'Fat Score'])
        saleprice = pd.DataFrame(index=sale_index, columns=['Price $/kg',
                                                            'Price $/hd'])  # need to initialise df with multiindex so rows can be added

        grid_price_s7s5s6 = r_vals['stock']['grid_price_s7s5s6']
        weight_range_s7s5 = r_vals['stock']['weight_range_s7s5']
        grid_keys = r_vals['stock']['salegrid_keys']
        for t_grid, t_weight, t_score in zip(grid, weight, score):
            ##grid name - used in table index
            grid_name = grid_keys[t_grid]
            ##index grid and score
            price_s5 = grid_price_s7s5s6[t_grid, :, t_score]
            ##interpolate to get price for specified weight
            lookup_weights = weight_range_s7s5[t_grid, :]
            price = np.interp(t_weight, lookup_weights, price_s5)
            ##attach to df
            ###if price is less than 10 it is assumed to be $/kg else $/hd
            if price < 10:
                col = 'Price $/kg'
            else:
                col = 'Price $/hd'
            saleprice.loc[(grid_name, t_weight, t_score), col] = price
        return saleprice



def f_summary(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['profit', 'profit max', 'profit min', 'profit stdev', 'risk neutral obj', 'utility',
                                                      'SR', 'SR max', 'SR min', 'SR stdev', 'Pas %', 'Pas % max', 'Pas % min', 'Pas % stdev',
                                                      'Ewes mated',
                                                      'Cereal %', 'Cereal % max', 'Cereal % min', 'Cereal % stdev',
                                                      'Canola %', 'Canola % max', 'Canola % min', 'Canola % stdev',
                                                      'Pulse %', 'Pulse % max', 'Pulse % min', 'Pulse % stdev',
                                                      'Fodder %', 'Fodder % max', 'Fodder % min', 'Fodder % stdev',
                                                      'Sup', 'Sup max', 'Sup min', 'Sup stdev'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    profit_max = round(f_profit(lp_vars, r_vals, option=3)[0],0)
    profit_min = round(f_profit(lp_vars, r_vals, option=3)[1],0)
    summary_df.loc[trial, 'profit max'] = profit_max * np.logical_not(profit_min==profit_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'profit min'] = profit_min * np.logical_not(profit_min==profit_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'profit stdev'] = f_profit(lp_vars, r_vals, option=3)[2]
    ##obj
    summary_df.loc[trial, 'risk neutral obj'] = f_profit(lp_vars, r_vals, option=1)
    ##utility
    summary_df.loc[trial, 'utility'] = f_profit(lp_vars, r_vals, option=2)
    ##total dse/ha in fp0
    summary_df.loc[trial, 'SR'] = round(f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0],1)
    SR_max = round(f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[1],1)
    SR_min = round(f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[2],1)
    summary_df.loc[trial, 'SR max'] = SR_max * np.logical_not(SR_min==SR_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'SR min'] = SR_min * np.logical_not(SR_min==SR_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'SR stdev'] = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[3]
    ##total dams mated
    type = 'stock'
    prod = 'dvp_is_mating_vzig1'
    na_prod = [0,1,2,3,5,6,7,10]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    arith = 2
    index = []
    cols = []
    axis_slice = {2:[1,None,1], 3:[2,None,1]} #slice off the not mate k1 slice (we only want mated dams) and slice off the sold animals so we dont count dams that are sold at prejoining (there is a sale opp at the start of dvp).
    dams_mated = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc[trial, 'Ewes mated'] = round(dams_mated.squeeze(),0)
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = round(f_area_summary(lp_vars, r_vals, option=5)[0],0)
    Pas_max = round(f_area_summary(lp_vars, r_vals, option=5)[1],0)
    Pas_min = round(f_area_summary(lp_vars, r_vals, option=5)[2],0)
    summary_df.loc[trial, 'Pas % max'] = Pas_max * np.logical_not(Pas_min==Pas_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pas % min'] = Pas_min * np.logical_not(Pas_min==Pas_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pas % stdev'] = f_area_summary(lp_vars, r_vals, option=5)[3]
    ##cereal %
    summary_df.loc[trial, 'Cereal %'] = round(f_area_summary(lp_vars, r_vals, option=6)[0],0)
    Cereal_max = round(f_area_summary(lp_vars, r_vals, option=6)[1],0)
    Cereal_min = round(f_area_summary(lp_vars, r_vals, option=6)[2],0)
    summary_df.loc[trial, 'Cereal % max'] = Cereal_max * np.logical_not(Cereal_min==Cereal_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Cereal % min'] = Cereal_min * np.logical_not(Cereal_min==Cereal_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Cereal % stdev'] = f_area_summary(lp_vars, r_vals, option=6)[3]
    ##canola %
    summary_df.loc[trial, 'Canola %'] = round(f_area_summary(lp_vars, r_vals, option=7)[0],0)
    Canola_max = round(f_area_summary(lp_vars, r_vals, option=7)[1],0)
    Canola_min = round(f_area_summary(lp_vars, r_vals, option=7)[2],0)
    summary_df.loc[trial, 'Canola % max'] = Canola_max * np.logical_not(Canola_min==Canola_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Canola % min'] = Canola_min * np.logical_not(Canola_min==Canola_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Canola % stdev'] = f_area_summary(lp_vars, r_vals, option=7)[3]
    ##pulse %
    summary_df.loc[trial, 'Pulse %'] = round(f_area_summary(lp_vars, r_vals, option=8)[0],0)
    pulse_max = round(f_area_summary(lp_vars, r_vals, option=8)[1],0)
    pulse_min = round(f_area_summary(lp_vars, r_vals, option=8)[2],0)
    summary_df.loc[trial, 'Pulse % max'] = pulse_max * np.logical_not(pulse_min==pulse_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pulse % min'] = pulse_min * np.logical_not(pulse_min==pulse_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pulse % stdev'] = f_area_summary(lp_vars, r_vals, option=8)[3]
    ##fodder %
    summary_df.loc[trial, 'Fodder %'] = round(f_area_summary(lp_vars, r_vals, option=9)[0],0)
    fodder_max = round(f_area_summary(lp_vars, r_vals, option=9)[1],0)
    fodder_min = round(f_area_summary(lp_vars, r_vals, option=9)[2],0)
    summary_df.loc[trial, 'Fodder % max'] = fodder_max * np.logical_not(fodder_min==fodder_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Fodder % min'] = fodder_min * np.logical_not(fodder_min==fodder_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Fodder % stdev'] = f_area_summary(lp_vars, r_vals, option=9)[3]
    ##supplement
    summary_df.loc[trial, 'Sup'] = round(f_grain_sup_summary(lp_vars,r_vals,option=4)[0],0)
    Sup_max = round(f_grain_sup_summary(lp_vars, r_vals, option=4)[1],0)
    Sup_min = round(f_grain_sup_summary(lp_vars, r_vals, option=4)[2],0)
    summary_df.loc[trial, 'Sup max'] = Sup_max * np.logical_not(Sup_min==Sup_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Sup min'] = Sup_min * np.logical_not(Sup_min==Sup_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Sup stdev'] = f_grain_sup_summary(lp_vars, r_vals, option=4)[3]
    return summary_df




def f_rotation(lp_vars, r_vals):
    '''
    manipulates the rotation solution into usable format. This is used in many function.
    '''
    ##rotation
    phases_df = r_vals['rot']['phases']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    phases_rk = phases_df.set_index(phases_df.columns[-1], append=True)  # add landuse as index level
    phases_rk.index.rename(['rot','landuse'],inplace=True) #rename index
    v_phase_area_qsp7zrl = f_vars2df(lp_vars, 'v_phase_area', mask_season_p7z[:,:,na,na], z_pos=-3)
    v_phase_change_increase_area_qsp7zrl = f_vars2df(lp_vars, 'v_phase_change_increase', mask_season_p7z[:,:,na,na], z_pos=-3)
    ##add landuse to the axis & remove level names
    v_phase_area_qsp7zlrk = v_phase_area_qsp7zrl.unstack(4).reindex(phases_rk.index, axis=1, level=0).stack([0,1])  # add landuse to the axis
    v_phase_area_qsp7zlrk.index.names = [None] * v_phase_area_qsp7zlrk.index.nlevels
    ##unstack p7 - p7 is generally a col where this is used so this saves time later
    v_phase_area_qszlrk_p7 = v_phase_area_qsp7zlrk.unstack(2)
    v_phase_area_qszrl_p7 = v_phase_area_qsp7zrl.unstack(2)
    v_phase_change_increase_area_qszrl_p7 = v_phase_change_increase_area_qsp7zrl.unstack(2)
    return phases_rk, v_phase_change_increase_area_qszrl_p7, v_phase_area_qszrl_p7, v_phase_area_qszlrk_p7


def f_area_summary(lp_vars, r_vals, option, active_z=True):
    '''
    Rotation & landuse area summary. With multiple output levels.
    return options:

    :param lp_vars: dict
    :param r_vals: dict
    :key option:
    :key active_z: Bool stating if z is active.

        #. table all rotations by lmu
        #. total pasture area each season in p7[-1]
        #. total crop area each season in p7[-1]
        #. table crop and pasture area by lmu and season
        #. landuse area in p7[-1]
        #. float pasture %, max, min & stdev in p7[-1]
        #. float cereal %, max, min & stdev in p7[-1]
        #. float canola %, max, min & stdev in p7[-1]
        #. float pulse %, max, min & stdev in p7[-1]
        #. float fodder %, max, min & stdev in p7[-1]
        #. table all rotations by lmu with disagregated land uses as index
        #. landuse area in p7[-1] with lmu axis

    '''

    ##read from other functions
    phases_rk, v_phase_change_increase_area_qszrl_p7, rot_area_qszrl_p7, rot_area_qszlrk_p7 = f_rotation(lp_vars, r_vals)
    landuse_area_k_p7qszl = rot_area_qszlrk_p7.groupby(axis=0, level=(0,1,2,3,5)).sum().unstack([0,1,2,3])  # area of each landuse (sum lmu and rotation)

    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)

    ##all rotations by lmu and p7
    rot_area_qszr_lp7 = rot_area_qszrl_p7.stack().unstack([-2,-1])
    if option == 0:
        ###weight z if required
        if active_z == False:
            rot_area_r_lp7 = rot_area_qszr_lp7.unstack(-1).mul(z_prob_qsz, axis=0).sum(axis=0).unstack(-1).T
            return rot_area_r_lp7.round(2)
        else:
            return rot_area_qszr_lp7.round(2)

    ##all rotations by lmu - with expanded landuse as index
    if option == 10:
        ### slice p7[-1] and unstack lmu
        rot_area_qszr_l = rot_area_qszrl_p7.iloc[:,-1].unstack(-1)
        ###remove current land use from index and add to df
        phases_r = phases_rk.droplevel(1)
        phases_r.insert(loc=len(phases_r.columns), column=len(phases_r.columns), value=phases_rk.index.get_level_values(1))
        ###add disagregated landuse as index.
        rot_area_qszr_l = rot_area_qszr_l.reset_index([0,1,2]).join(phases_r).set_index(['level_0','level_1','level_2']+list(range(len(phases_r.columns))))
        ###weight z if required
        if active_z == False:
            rot_area_r_l = rot_area_qszr_l.unstack(['level_0','level_1','level_2']).stack(0).mul(z_prob_qsz, axis=1).sum(axis=1).unstack(-1)
            return rot_area_r_l.round(2)
        else:
            return rot_area_qszr_l.round(2)

    ###pasture area
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    pasture_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_pas)].sum()  # sum landuse
    pasture_area_qszl = pasture_area_p7qszl.loc[pasture_area_p7qszl.index.levels[0][-1].tolist()] #slice for p7[-1]
    if option == 1:
        pasture_area_qsz = pasture_area_qszl.groupby(level=(0, 1, 2)).sum()  # sum lmu
        ###weight z if required
        if active_z == False:
            pasture_area = pasture_area_qsz.mul(z_prob_qsz, axis=0).sum(axis=0)
            return pd.DataFrame([pasture_area]).round(0)
        else:
            return pasture_area_qsz.round(0)

    ###crop area
    crop_area_p7qszl = landuse_area_k_p7qszl[~landuse_area_k_p7qszl.index.isin(all_pas)].sum()  # sum landuse
    crop_area_qszl = crop_area_p7qszl.loc[crop_area_p7qszl.index.levels[0][-1].tolist()]
    if option == 2:
        crop_area_qsz = crop_area_qszl.groupby(level=(0,1,2)).sum().round(0) #sum lmu
        ###weight z if required
        if active_z == False:
            crop_area = crop_area_qsz.mul(z_prob_qsz, axis=0).sum(axis=0)
            return pd.DataFrame([crop_area]).round(0)
        else:
            return crop_area_qsz.unstack(-1)

    ##crop & pasture area by lmu
    if option == 3:
        croppas_area_qszl_k = pd.DataFrame()
        croppas_area_qszl_k['pasture'] = pasture_area_qszl
        croppas_area_qszl_k['crop'] = crop_area_qszl
        ###weight z if required
        if active_z == False:
            croppas_area_l_k = croppas_area_qszl_k.unstack(-1).mul(z_prob_qsz, axis=0).sum(axis=0).unstack(-1).T
            return croppas_area_l_k.round(0)
        else:
            return croppas_area_qszl_k.round(0)

    if option==4: #landuse area in p7[-1] (lmu summed)
        landuse_area_k_qszl = landuse_area_k_p7qszl.loc[:,landuse_area_k_p7qszl.columns.levels[0][-1].tolist()]
        landuse_area_k_qsz = landuse_area_k_qszl.groupby(axis=1, level=(0,1,2)).sum()
        landuse_area_qsz_k = landuse_area_k_qsz.T.round(2)
        landuse_area_qsz_k = landuse_area_qsz_k.reindex(r_vals['pas']['keys_k'], axis=1).fillna(0) #expand to full k (incase landuses were masked out) and unused landuses get set to 0
        ###weight z if required
        if active_z == False:
            landuse_area_k = landuse_area_qsz_k.mul(z_prob_qsz, axis=0).sum(axis=0)
            return pd.DataFrame(landuse_area_k).round(0)
        else:
            return landuse_area_qsz_k.round(0)

    if option==11: #landuse area in p7[-1] (lmu active)
        landuse_area_k_qszl = landuse_area_k_p7qszl.loc[:,landuse_area_k_p7qszl.columns.levels[0][-1].tolist()]
        landuse_area_qszl_k = landuse_area_k_qszl.T.round(2)
        landuse_area_qszl_k = landuse_area_qszl_k.reindex(r_vals['pas']['keys_k'], axis=1).fillna(0) #expand to full k (incase landuses were masked out) and unused landuses get set to 0
        ###weight z if required
        if active_z == False:
            landuse_area_l_k = landuse_area_qszl_k.unstack(-1).mul(z_prob_qsz, axis=0).sum(axis=0).unstack(-1).T
            return landuse_area_l_k.round(0)
        else:
            return landuse_area_qszl_k.round(0)

    if option==5 or option==6 or option==7 or option==8 or option==9: #average % of pasture/cereal/canola in p7[-1]
        rot_area_qsz = r_vals['rot']['total_farm_area']
        if option == 5:
            pasture_area_p7qsz = pasture_area_p7qszl.groupby(level=(0,1,2,3)).sum() #sum l
            pasture_area_qsz = pasture_area_p7qsz.unstack(0).iloc[:,-1]  # slice for p7[-1]
            pas_area_qsz = fun.f_divide(pasture_area_qsz, rot_area_qsz) * 100
            ###stdev and range
            pas_area_mean = np.sum(pas_area_qsz * z_prob_qsz)
            ma_pas_area_qsz = np.ma.masked_array(pas_area_qsz, z_prob_qsz == 0)
            pas_area_max = np.max(ma_pas_area_qsz)
            pas_area_min = np.min(ma_pas_area_qsz)
            pas_area_stdev = np.sum((pas_area_qsz - pas_area_mean) ** 2 * z_prob_qsz)**0.5
            return round(pas_area_mean, 1), round(pas_area_max, 1), round(pas_area_min, 1), round(pas_area_stdev, 1)
        if option == 6:
            all_cereals = r_vals['rot']['all_cereals']  # landuse sets
            cereal_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_cereals)].sum()  # sum landuse
            cereal_area_p7qsz = cereal_area_p7qszl.groupby(level=(0, 1, 2, 3)).sum()  # sum l
            cereal_area_qsz = cereal_area_p7qsz.unstack(0).iloc[:, -1]  # slice for p7[-1]
            cereal_area_qsz = fun.f_divide(cereal_area_qsz, rot_area_qsz) * 100
            ###stdev and range
            cereal_area_mean = np.sum(cereal_area_qsz * z_prob_qsz)
            ma_cereal_area_qsz = np.ma.masked_array(cereal_area_qsz, z_prob_qsz == 0)
            cereal_area_max = np.max(ma_cereal_area_qsz)
            cereal_area_min = np.min(ma_cereal_area_qsz)
            cereal_area_stdev = np.sum((cereal_area_qsz - cereal_area_mean) ** 2 * z_prob_qsz) ** 0.5
            return round(cereal_area_mean, 1), round(cereal_area_max, 1), round(cereal_area_min, 1), round(cereal_area_stdev, 1)
        if option == 7:
            all_canolas = r_vals['rot']['all_canolas']  # landuse sets
            canola_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_canolas)].sum()  # sum landuse
            canola_area_p7qsz = canola_area_p7qszl.groupby(level=(0, 1, 2, 3)).sum()  # sum l
            canola_area_qsz = canola_area_p7qsz.unstack(0).iloc[:, -1]  # slice for p7[-1]
            canola_area_qsz = fun.f_divide(canola_area_qsz, rot_area_qsz) * 100
            ###stdev and range
            canola_area_mean = np.sum(canola_area_qsz * z_prob_qsz)
            ma_canola_area_qsz = np.ma.masked_array(canola_area_qsz, z_prob_qsz == 0)
            canola_area_max = np.max(ma_canola_area_qsz)
            canola_area_min = np.min(ma_canola_area_qsz)
            canola_area_stdev = np.sum((canola_area_qsz - canola_area_mean) ** 2 * z_prob_qsz) ** 0.5
            return round(canola_area_mean, 1), round(canola_area_max, 1), round(canola_area_min, 1), round(canola_area_stdev, 1)
        if option == 8:
            all_pulses = r_vals['rot']['all_pulses']  # landuse sets
            pulse_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_pulses)].sum()  # sum landuse
            pulse_area_p7qsz = pulse_area_p7qszl.groupby(level=(0, 1, 2, 3)).sum()  # sum l
            pulse_area_qsz = pulse_area_p7qsz.unstack(0).iloc[:, -1]  # slice for p7[-1]
            pulse_area_qsz = fun.f_divide(pulse_area_qsz, rot_area_qsz) * 100
            ###stdev and range
            pulse_area_mean = np.sum(pulse_area_qsz * z_prob_qsz)
            ma_pulse_area_qsz = np.ma.masked_array(pulse_area_qsz, z_prob_qsz == 0)
            pulse_area_max = np.max(ma_pulse_area_qsz)
            pulse_area_min = np.min(ma_pulse_area_qsz)
            pulse_area_stdev = np.sum((pulse_area_qsz - pulse_area_mean) ** 2 * z_prob_qsz) ** 0.5
            return round(pulse_area_mean, 1), round(pulse_area_max, 1), round(pulse_area_min, 1), round(pulse_area_stdev, 1)
        if option == 9:
            all_fodders = r_vals['rot']['fodder']  # landuse sets
            fodder_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_fodders)].sum()  # sum landuse
            fodder_area_p7qsz = fodder_area_p7qszl.groupby(level=(0, 1, 2, 3)).sum()  # sum l
            fodder_area_qsz = fodder_area_p7qsz.unstack(0).iloc[:, -1]  # slice for p7[-1]
            fodder_area_qsz = fun.f_divide(fodder_area_qsz, rot_area_qsz) * 100
            ###stdev and range
            fodder_area_mean = np.sum(fodder_area_qsz * z_prob_qsz)
            ma_fodder_area_qsz = np.ma.masked_array(fodder_area_qsz, z_prob_qsz == 0)
            fodder_area_max = np.max(ma_fodder_area_qsz)
            fodder_area_min = np.min(ma_fodder_area_qsz)
            fodder_area_stdev = np.sum((fodder_area_qsz - fodder_area_mean) ** 2 * z_prob_qsz) ** 0.5
            return round(fodder_area_mean, 1), round(fodder_area_max, 1), round(fodder_area_min, 1), round(fodder_area_stdev, 1)


def f_mach_summary(lp_vars, r_vals, option=0):
    '''
    Machine summary.
    :param option:

        #. table: total machine cost for each crop in each cash period
        #. table: total seeding biomass penalty for untimely sowing.
        #. table: average sowing date.
        #. table: full summary.

    '''
    ##call rotation function to get rotation info
    phases_rk, v_phase_change_increase_area_qszrl_p7, rot_area_qszrl_p7 = f_rotation(lp_vars, r_vals)[0:3]
    rot_area_zrlqs_p7 = rot_area_qszrl_p7.reorder_levels([2,3,4,0,1], axis=0).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)
    v_phase_change_increase_area_zrlqs_p7 = v_phase_change_increase_area_qszrl_p7.reorder_levels([2, 3, 4, 0, 1], axis=0).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)

    ##masks to uncluster z axis
    maskz8_zp5 = r_vals['lab']['maskz8_p5z'].T

    ##variables used in multiple places.
    seeding_days_qszp5_kl = f_vars2df(lp_vars, 'v_seeding_machdays', maskz8_zp5[:,:,na,na], z_pos=-4).unstack([4,5])
    seeding_rate_kl = r_vals['mach']['seeding_rate'].stack()
    contractseeding_ha_qszp5k_l = f_vars2df(lp_vars, 'v_contractseeding_ha', maskz8_zp5[:,:,na,na], z_pos=-4).unstack(-1)
    ###ha sown by farmer
    seeding_ha_qszp5_kl = seeding_days_qszp5_kl.mul(seeding_rate_kl.reindex(seeding_days_qszp5_kl.columns), axis=1)  # note seeding ha won't equal the rotation area because arable area is included in seed_ha.
    seeding_ha_qszp5k_l = seeding_ha_qszp5_kl.stack(0)

    ##return mach costs
    if option == 0 or option == 4:
        ##harv
        contractharv_hours_qszp5k = f_vars2df(lp_vars, 'v_contractharv_hours', maskz8_zp5[:,:,na], z_pos=-3)
        contractharv_hours_zp5kqs = contractharv_hours_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)
        harv_hours_qszp5k = f_vars2df(lp_vars, 'v_harv_hours', maskz8_zp5[:,:,na], z_pos=-3)
        harv_hours_zp5kqs = harv_hours_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)
        contract_harvest_cost_zp5k_p7 = r_vals['mach']['contract_harvest_cost'].unstack(0)
        contract_harvest_cost_zp5kqs_p7 = contract_harvest_cost_zp5k_p7.reindex(contractharv_hours_zp5kqs.index, axis=0)
        contract_harvest_cost_zp5kqs_p7 = contract_harvest_cost_zp5kqs_p7.mul(contractharv_hours_zp5kqs, axis=0)
        own_harvest_cost_zp5k_p7 = r_vals['mach']['harvest_cost'].unstack(0)
        own_harvest_cost_zp5kqs_p7 = own_harvest_cost_zp5k_p7.reindex(harv_hours_zp5kqs.index, axis=0)
        own_harvest_cost_zp5kqs_p7 = own_harvest_cost_zp5kqs_p7.mul(harv_hours_zp5kqs, axis=0)
        harvest_cost_zp5kqs_p7 = contract_harvest_cost_zp5kqs_p7 + own_harvest_cost_zp5kqs_p7
        harvest_cost_zkqs_p7 = harvest_cost_zp5kqs_p7.groupby(axis=0, level=(0,2,3,4)).sum() #sum p5

        ##seeding
        seeding_ha_zp5lkqs = seeding_ha_qszp5_kl.stack([0,1]).reorder_levels([2,3,5,4,0,1]).sort_index()
        seeding_cost_zp5l_p7 = r_vals['mach']['seeding_cost'].unstack(0)
        seeding_cost_zp5lkqs_p7 = seeding_cost_zp5l_p7.reindex(seeding_ha_zp5lkqs.index, axis=0)
        seeding_cost_own_zkqs_p7 = seeding_cost_zp5lkqs_p7.mul(seeding_ha_zp5lkqs, axis=0).groupby(axis=0, level=(0,3,4,5)).sum()  # sum lmu axis and p5

        contractseeding_ha_qszp5k = contractseeding_ha_qszp5k_l.sum(axis=1)  # sum lmu axis (cost doesn't vary by lmu for contract)
        contractseeding_ha_zp5kqs = contractseeding_ha_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()
        contractseed_cost_ha_zp5_p7 = r_vals['mach']['contractseed_cost'].unstack(0)
        contractseed_cost_ha_zp5kqs_p7 = contractseed_cost_ha_zp5_p7.reindex(contractseeding_ha_zp5kqs.index, axis=0)
        seeding_cost_contract_zkqs_p7 =  contractseed_cost_ha_zp5kqs_p7.mul(contractseeding_ha_zp5kqs, axis=0).groupby(axis=0, level=(0,2,3,4)).sum()  # sum p5
        seeding_cost_zkqs_p7 = seeding_cost_contract_zkqs_p7 + seeding_cost_own_zkqs_p7
        # seeding_cost_c0p7z_k = seeding_cost_c0p7_zk.stack(0)

        ##fert & chem mach cost
        ###v_phase
        fert_app_cost_rl_p7z = r_vals['crop']['fert_app_cost']
        chem_app_cost_ha_rl_p7z = r_vals['crop']['chem_app_cost_ha']
        fertchem_cost_rl_p7z = pd.concat([fert_app_cost_rl_p7z, chem_app_cost_ha_rl_p7z], axis=1).groupby(axis=1, level=(0,1)).sum()  # cost per ha
        fertchem_cost_zrl_p7 = fertchem_cost_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
        fertchem_cost_zrlqs_p7 = fertchem_cost_zrl_p7.reindex(rot_area_zrlqs_p7.index, axis=0)
        fertchem_cost_zrqs_p7 = fertchem_cost_zrlqs_p7.mul(rot_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
        fertchem_cost_k_p7zqs = fertchem_cost_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,level=1).sum()  # reindex to include landuse and sum rot
        fertchem_cost_zkqs_p7 = fertchem_cost_k_p7zqs.stack([1,2,3]).swaplevel(0,1)
        ###v_phase increment
        fert_app_cost_increment_rl_p7z = r_vals['crop']['fert_app_cost_increment'].unstack([-2,-1])
        chem_app_cost_increment_rl_p7z = r_vals['crop']['chem_cost_increment'].unstack([-2,-1])
        fertchem_cost_increment_rl_p7z = pd.concat([fert_app_cost_increment_rl_p7z, chem_app_cost_increment_rl_p7z], axis=1).groupby(axis=1, level=(0,1)).sum()  # cost per ha
        fertchem_cost_increment_zrl_p7 = fertchem_cost_increment_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
        fertchem_cost_increment_zrlqs_p7 = fertchem_cost_increment_zrl_p7.reindex(v_phase_change_increase_area_zrlqs_p7.index, axis=0)
        fertchem_cost_increment_zrqs_p7 = fertchem_cost_increment_zrlqs_p7.mul(v_phase_change_increase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
        fertchem_cost_increment_k_p7zqs = fertchem_cost_increment_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,level=1).sum()  # reindex to include landuse and sum rot
        fertchem_cost_increment_zkqs_p7 = fertchem_cost_increment_k_p7zqs.stack([1,2,3]).swaplevel(0,1)
        ###total
        fertchem_cost_zkqs_p7 = fertchem_cost_zkqs_p7 + fertchem_cost_increment_zkqs_p7

        ##combine all costs
        exp_mach_zkqs_p7 = pd.concat([fertchem_cost_zkqs_p7, seeding_cost_zkqs_p7, harvest_cost_zkqs_p7
                                   ], axis=0).groupby(axis=0, level=(0,1,2,3)).sum()
        exp_mach_k_p7zqs = exp_mach_zkqs_p7.unstack([0,2,3])
        ##insurance
        mach_insurance_p7z = r_vals['mach']['mach_insurance']
        if option == 0:
            return exp_mach_k_p7zqs, mach_insurance_p7z

    ##yield penalty
    if option == 1:
        ##yield penalty from untimely sowing
        sowing_yield_penalty_p7p5zkl = r_vals['mach']['sowing_yield_penalty_p7p5zkl']
        sowing_yield_penalty_p5zkl = sowing_yield_penalty_p7p5zkl.groupby(level=(1,2,3,4)).sum() #sum p7
        sowing_yield_penalty_zp5kl = sowing_yield_penalty_p5zkl.reorder_levels((1,0,2,3))
        ###reindex penalty param
        ####add q & s axis
        sowing_yield_penalty_qsz_lkp5 = sowing_yield_penalty_zp5kl.unstack((-1,-2,-3)).reindex(seeding_ha_qszp5_kl.unstack(-1).index, axis=0, level=-1)
        sowing_yield_penalty_qszp5k_l = sowing_yield_penalty_qsz_lkp5.stack((2,1))
        ####expand k to include pastures
        sowing_yield_penalty_qszp5k_l = sowing_yield_penalty_qszp5k_l.reindex(seeding_ha_qszp5k_l.index)
        ###calc penalty
        farmer_penalty_qszp5k_l = seeding_ha_qszp5k_l.mul(sowing_yield_penalty_qszp5k_l)
        farmer_penalty_qszp5k = farmer_penalty_qszp5k_l.sum(axis=1)
        contract_penalty_qszp5k_l = contractseeding_ha_qszp5k_l.mul(sowing_yield_penalty_qszp5k_l)
        contract_penalty_qszp5k = contract_penalty_qszp5k_l.sum(axis=1)
        total_penalty_qszk = farmer_penalty_qszp5k.add(contract_penalty_qszp5k).unstack(3).sum(axis=1)
        return total_penalty_qszk/1000 #convert to tonnes of penalty

    ##sowing date
    if option==2:
        labour_period_start_p5z = r_vals['lab']['lp_start_p5z']
        labour_period_end_p5z = r_vals['lab']['lp_end_p5z']
        labour_period_ave_p5z = (labour_period_start_p5z + labour_period_end_p5z)/2
        labour_period_ave_zp5 = labour_period_ave_p5z.T
        labour_period_ave_zp5 = pd.DataFrame(labour_period_ave_zp5, index=r_vals['zgen']['keys_z'], columns=r_vals['lab']['keys_p5']).stack()
        ha_sown_qszp5k_l = contractseeding_ha_qszp5k_l + seeding_ha_qszp5k_l
        ha_sown_qszp5k = ha_sown_qszp5k_l.sum(axis=1) #sum l
        ha_sown_qsk_zp5 = ha_sown_qszp5k.unstack([-3,-2])
        ave_sow_date_qskz = ha_sown_qsk_zp5.mul(labour_period_ave_zp5,axis=1).stack(0).sum(axis=1).div(ha_sown_qsk_zp5.stack(0).sum(axis=1))
        ave_sow_date_qsz_k = ave_sow_date_qskz.unstack(-2)
        contractseeding_ha_qsz = contractseeding_ha_qszp5k_l.unstack((-1,-2)).sum(axis=1)
        contractseeding_ha_qsz = pd.DataFrame(contractseeding_ha_qsz, columns=["Contract seeded (ha)"])
        return pd.concat([contractseeding_ha_qsz, ave_sow_date_qsz_k], axis=1)

    if option == 4:
        ##create p/l dataframe
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        idx = pd.IndexSlice
        subtype = ['Total harvest costs', 'Owner harvest costs', 'Owner harvest hours', 'Contract harvest costs', 'Contract harvest hours',
                   'Total seeding costs', 'Owner seeding costs', 'Owner seeding days', 'Contract seeding costs', 'Contract seeded hectares',
                   'Total spreading and spraying costs', 'Spreading hours', 'Spraying hours',
                   'Variable depreciation', 'Fixed depreciation',
                   'Insurance']
        mach_index = pd.MultiIndex.from_product([keys_q, keys_s, keys_z, subtype], names=['Sequence_year', 'Sequence', 'Season', 'Subtype'])
        mach = pd.DataFrame(index=mach_index, columns=["item"])  # need to initialise df with multiindex so rows can be added

        ##spraying and spreading time - doesnt require phase_increment. Just calculate based on the existing rotations in p7[-1]. Will need updating if dual cropping.
        rot_area_qs_rzl = rot_area_qszrl_p7.iloc[:,-1].unstack([3,2,4]) # slice p7[-1]
        chem_time_rzl = r_vals['crop']['chem_time_ha_rzl_n'].sum(axis=1)
        fert_time_rzl = r_vals['crop']['fert_time_rzl_n'].sum(axis=1)
        spraying_time_qs_z = rot_area_qs_rzl.mul(chem_time_rzl, axis=1).groupby(axis=1, level=1).sum()  # mul area and sum lmu and rot
        spreading_time_qs_z = rot_area_qs_rzl.mul(fert_time_rzl, axis=1).groupby(axis=1, level=1).sum()  # mul area and sum lmu and rot

        ###dep
        harv_hours_qsz = harv_hours_zp5kqs.groupby(axis=0, level=(0, 3, 4)).sum().reorder_levels([1, 2, 0])  # sum p5, k and reorder
        spray_dep_hourly = r_vals['crop']['spray_dep_hourly']
        spread_dep_hourly = r_vals['crop']['spread_dep_hourly']
        seeding_dep_ha_kl = r_vals['mach']['seeding_dep_ha_kl']
        harv_dep_hourly = r_vals['mach']['harv_dep_hourly']
        spray_dep_qsz = spray_dep_hourly * spraying_time_qs_z.stack()
        spread_dep_qsz = spread_dep_hourly * spreading_time_qs_z.stack()
        harv_dep_qsz = harv_dep_hourly * harv_hours_qsz / r_vals['mach']['number_seeding_gear']
        seeding_ha_qszk_l = seeding_ha_qszp5k_l.groupby(axis=0,level=(0,1,2,4)).sum() #sum p5
        seeding_ha_qsz_kl = seeding_ha_qszk_l.unstack(-1).reorder_levels([1, 0], axis=1).reindex(seeding_dep_ha_kl.index, axis=1)
        seeder_dep_qsz = seeding_ha_qsz_kl.mul(seeding_dep_ha_kl, axis=1).sum(axis=1)  / r_vals['mach']['number_of_harvesters']

        ###insurance
        mach_insurance_z = mach_insurance_p7z.unstack(0).sum(axis=1)
        mach_insurance_qsz = mach_insurance_z.reindex(pd.MultiIndex.from_product([keys_q, keys_s, keys_z], names=["q", "s", "z"]),level=-1)

        ###harv summary
        mach.loc[idx[:, :, :, 'Total harvest costs'],:] = harvest_cost_zkqs_p7.sum(axis=1).groupby(axis=0,level=(0,2,3)).sum().reorder_levels([1, 2, 0]) #sum p7, k and reorder
        mach.loc[idx[:, :, :, 'Owner harvest costs'],:] = own_harvest_cost_zp5kqs_p7.sum(axis=1).groupby(axis=0,level=(0,3,4)).sum().reorder_levels([1, 2, 0]) #sum p7, p5, k and reorder
        mach.loc[idx[:, :, :, 'Owner harvest hours'],:] = harv_hours_qsz
        mach.loc[idx[:, :, :, 'Contract harvest costs'],:] = contract_harvest_cost_zp5kqs_p7.sum(axis=1).groupby(axis=0,level=(0,3,4)).sum().reorder_levels([1, 2, 0]) #sum p7, p5, k and reorder
        mach.loc[idx[:, :, :, 'Contract harvest hours'],:] = contractharv_hours_zp5kqs.groupby(axis=0,level=(0,3,4)).sum().reorder_levels([1, 2, 0]) #sum p5, k and reorder
        ###seeding summary
        mach.loc[idx[:, :, :, 'Total seeding costs'],:] = seeding_cost_zkqs_p7.sum(axis=1).groupby(axis=0,level=(0,2,3)).sum().reorder_levels([1, 2, 0]) #sum p7, k and reorder
        mach.loc[idx[:, :, :, 'Owner seeding costs'],:] = seeding_cost_own_zkqs_p7.sum(axis=1).groupby(axis=0,level=(0,2,3)).sum().reorder_levels([1, 2, 0]) #sum p7, k and reorder
        mach.loc[idx[:, :, :, 'Owner seeding days'],:] = seeding_days_qszp5_kl.sum(axis=1).groupby(axis=0,level=(0,1,2)).sum() #sum k, l, p5
        mach.loc[idx[:, :, :, 'Contract seeding costs'],:] = seeding_cost_contract_zkqs_p7.sum(axis=1).groupby(axis=0,level=(0,2,3)).sum().reorder_levels([1, 2, 0]) #sum p7, k and reorder
        mach.loc[idx[:, :, :, 'Contract seeded hectares'],:] = contractseeding_ha_zp5kqs.groupby(axis=0,level=(0,3,4)).sum().reorder_levels([1, 2, 0]) #sum p5, k and reorder
        ###spreading and sprarying
        mach.loc[idx[:, :, :, 'Total spreading and spraying costs'],:] = fertchem_cost_zkqs_p7.sum(axis=1).groupby(axis=0,level=(0,2,3)).sum().reorder_levels([1, 2, 0]) #sum p7, k and reorder
        mach.loc[idx[:, :, :, 'Spreading hours'],:] = spreading_time_qs_z.stack()
        mach.loc[idx[:, :, :, 'Spraying hours'],:] = spraying_time_qs_z.stack()
        ###dep
        mach.loc[idx[:, :, :, 'Variable depreciation'],:] = spray_dep_qsz + spread_dep_qsz + harv_dep_qsz + seeder_dep_qsz
        mach.loc[idx[:, :, :, 'Fixed depreciation'],:] = r_vals['mach']['fixed_dep_p7'].sum()
        ###insurance
        mach.loc[idx[:, :, :, 'Insurance'],:] = mach_insurance_qsz
        # ###total
        # total_cost_items = [
        #     'Total harvest costs',
        #     'Total seeding costs',
        #     'Total spreading and spraying costs',
        #     'Variable depreciation',
        #     'Fixed depreciation',
        #     'Insurance'
        # ]
        # # Filter rows where the last level matches any of the cost items
        # filtered = mach.loc[idx[:, :, :, total_cost_items], :]
        # mach.loc[idx[:, :, :, 'Total'], :] = filtered.groupby(level=[0, 1, 2]).sum()

        return mach.round(0).astype(int)

def f_available_cropgrazing(r_vals):
    '''
    Calculates the total crop that CAN be grazed based on seeding timing (the actual amount consumed is optimised).
    '''
    v_contractseeding_ha_qszp5kl = d_vars['base']['v_contractseeding_ha'] #use base vars because z is being reported
    v_seeding_machdays_qszp5kl = d_vars['base']['v_seeding_machdays'] #use base vars because z is being reported

    ##calc ha sown by farmer
    seeding_rate_kl = r_vals['mach']['seeding_rate']
    farmerseeding_ha_qszp5kl = v_seeding_machdays_qszp5kl * np.array(seeding_rate_kl)

    ##total ha sown
    ha_sown_qszp5kl = v_contractseeding_ha_qszp5kl + farmerseeding_ha_qszp5kl
    ###cut the k axis to show just the crops
    keys_k = r_vals['pas']['keys_k']
    keys_k1 = r_vals['stub']['keys_k1']
    mask_k = np.any(keys_k1[:,na] == keys_k, axis=0)
    ha_sown_qszp5kl = np.compress(mask_k, ha_sown_qszp5kl, axis=-2)

    ##total crop DM available for grazing
    crop_DM_provided_z8p5p6klz9 = np.moveaxis(r_vals['crpgrz']['crop_DM_provided_kp6p5z8lz9'], [0,1,2,3],[3,2,1,0])
    crop_DM_required_zp5p6k = np.moveaxis(r_vals['crpgrz']['crop_DM_required_kp6p5z'], [0,1,2,3],[3,2,1,0])
    ###adjust for trampling/wastage to calc total available for consumption
    crop_DM_available_z8p5p6klz9 = fun.f_divide(crop_DM_provided_z8p5p6klz9, crop_DM_required_zp5p6k[...,na,na])
    ###convert from per ha to total
    total_crop_DM_qsp6z9 = np.sum(ha_sown_qszp5kl[...,na,:,:,na] * crop_DM_available_z8p5p6klz9, axis=(2,3,5,6)) #sum z8, p5, k, l

    ##convert to df
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_p6 = r_vals['pas']['keys_p6']
    keys_qsp6z = [keys_q, keys_s, keys_p6, keys_z]
    df_crop_available_qsz_p6 = f_numpy2df(total_crop_DM_qsp6z9, keys_qsp6z, [0,1,3], [2])
    return df_crop_available_qsz_p6

def f_biomass_penalty(lp_vars, r_vals):
    ##seeding
    seeding_penalty_qszp5k = f_mach_summary(lp_vars, r_vals, option=1)

    ##crop grazing
    prod = 'crop_grazing_biomass_penalty_kp6z'
    na_prod = [0, 1, 2, 5, 7]  # q,s,f,p5,l
    type = 'crpgrz'
    weights = 'crop_consumed_qsfkp6p5zl'
    keys = 'keys_qsfkp6p5zl'
    arith = 2
    index = [0, 1, 6, 3]  # q,s,z,k
    cols = []
    crop_grazing_penalty_qszk = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)

    penalty = pd.concat([seeding_penalty_qszp5k, crop_grazing_penalty_qszk], axis=1)
    penalty.columns=['seeding (t)', 'crop_grazing (t)']
    return penalty

def f_grain_sup_summary(lp_vars, r_vals, option=0):
    '''
    Summary of grain, supplement and their costs

    :param option: int:

            #. return dict with sup cost and revenue from grain sales
            #. return total supplement fed in each feed period
            #. return total of each grain supplement fed in each feed period in each season
            #. return total of each grain supplement fed in each feed period for each feed pool in each season
            #. return total sup fed (weighted by season prob)
            #. return total grain/hay produced on farm.

    '''
    ##z masks to uncluster lp_vars
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    maskz8_zp6 = r_vals['pas']['mask_fp_z8var_p6z'].T

    ##grain fed
    grain_fed_qszk3gvp6 = f_vars2df(lp_vars, 'v_sup_con', maskz8_zp6[:,na,na,na,:], z_pos=-5)

    if option == 1:
        grain_fed_qszp6 = grain_fed_qszk3gvp6.groupby(level=(0, 1, 2, 6)).sum()  # sum feed pool, landuse and grain pool
        return grain_fed_qszp6.to_frame()

    if option == 2:
        grain_fed_qszk3p6 = grain_fed_qszk3gvp6.groupby(level=(0, 1, 2, 3, 6)).sum()  # sum feed pool and grain pool
        return grain_fed_qszk3p6

    if option == 3:
        grain_fed_qszk3fp6 = grain_fed_qszk3gvp6.groupby(level=(0, 1, 2, 3, 5, 6)).sum()  # sum grain pool
        return grain_fed_qszk3fp6

    if option == 4:
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        grain_fed_qsz = grain_fed_qszk3gvp6.groupby(level=(0,1,2)).sum() #sum all axis except season ones (q,s,z)
        ###stdev and range
        grain_fed_mean = grain_fed_qsz.mul(z_prob_qsz).sum()
        ma_grain_fed_qsz = np.ma.masked_array(grain_fed_qsz, z_prob_qsz == 0)
        grain_fed_max = np.max(ma_grain_fed_qsz)
        grain_fed_min = np.min(ma_grain_fed_qsz)
        grain_fed_stdev = (((grain_fed_qsz - grain_fed_mean) ** 2).mul(z_prob_qsz).sum())**0.5
        return round(grain_fed_mean, 1), round(grain_fed_max, 1), round(grain_fed_min, 1), round(grain_fed_stdev, 1),

    ##NOTE: this only works if there is one time of grain purchase/sale
    if option == 0 or option == 5:
        ##create dict to store grain variables
        grain = {}
        ##prices
        grains_sale_price_zk1s2gq_p7 = r_vals['crop']['grain_price'].stack([0,2]).reorder_levels([4,0,1,2,3]).sort_index()
        grains_buy_price_zk3s2gq_p7 = r_vals['sup']['buy_grain_price'].stack([0,2]).reorder_levels([4,0,1,2,3]).sort_index()

        ##grain purchased
        grain_purchased_qsp7zk3s2g = f_vars2df(lp_vars,'v_buy_product', mask_season_p7z[:,:,na,na,na], z_pos=-4)
        grain_purchased_qszk3s2g = grain_purchased_qsp7zk3s2g.groupby(level=(0,1,3,4,5,6)).sum()  # sum p7
        grain_purchased_zk3s2gqs = grain_purchased_qszk3s2g.reorder_levels([2,3,4,5,0,1]) .sort_index()#change the order so that reindexing works (new levels being added must be at the end)

        ##grain sold
        grain_sold_qsp7zk1s2g = f_vars2df(lp_vars,'v_sell_product', mask_season_p7z[:,:,na,na,na], z_pos=-4)
        grain_sold_qszk1s2g = grain_sold_qsp7zk1s2g.groupby(level=(0,1,3,4,5,6)).sum()  # sum p7
        grain_sold_zk1s2gqs = grain_sold_qszk1s2g.reorder_levels([2,3,4,5,0,1]).sort_index() #change the order so that reindexing works (new levels being added must be at the end)

        ##grain fed - s2 axis added because sup feed is allocated to a given s2 slice and therefore the variable doesn't have an active s2 axis
        sup_s2_ks2 = r_vals['sup']['sup_s2_k_s2'].stack()
        grain_fed_qszk3g = grain_fed_qszk3gvp6.groupby(level=(0, 1, 2, 3, 4)).sum()  # sum feed pool and feed period
        grain_fed_qszg_k3s2 = grain_fed_qszk3g.unstack(3).mul(sup_s2_ks2, axis=1, level=0)
        grain_fed_zk3s2gqs = grain_fed_qszg_k3s2.stack([0,1]).reorder_levels([2,4,5,3,0,1]).sort_index() #change the order so that reindexing works (new levels being added must be at the end)

        ##total grain produced by crop enterprise
        grain_transferred_crop_to_sheep_zk3s2gqs = grain_fed_zk3s2gqs - grain_purchased_zk3s2gqs
        grain_transferred_crop_to_sheep_zk1s2gqs = grain_transferred_crop_to_sheep_zk3s2gqs.reindex(grain_sold_zk1s2gqs.index).fillna(0)
        total_grain_produced_zk1s2gqs = grain_sold_zk1s2gqs + grain_transferred_crop_to_sheep_zk1s2gqs  # total grain produced by crop enterprise
        if option==5:
            return total_grain_produced_zk1s2gqs
        grains_sale_price_zk1s2gqs_p7 = grains_sale_price_zk1s2gq_p7.reindex(total_grain_produced_zk1s2gqs.index, axis=0)
        rev_grain_k1_p7zqs = grains_sale_price_zk1s2gqs_p7.mul(total_grain_produced_zk1s2gqs, axis=0).unstack([0,4,5]).groupby(axis=0, level=0).sum()  # sum grain pool and s2
        grain['rev_grain_k1_p7zqs'] = rev_grain_k1_p7zqs

        ##supplementary cost: cost = sale_price * (grain_fed - grain_purchased) + buy_price * grain_purchased
        ###cost of grain transferred from cropping enterprise (this is only the k3 crops that exist in k1 set hence use k1 axis)
        sup_transferred_exp_zqs_p7 = grains_sale_price_zk1s2gqs_p7.mul(grain_transferred_crop_to_sheep_zk1s2gqs, axis=0).groupby(axis=0, level=(0, 4, 5)).sum()  # sum grain pool & landuse & s2
        ###cost of purchases
        grains_buy_price_zk3s2gqs_p7 = grains_buy_price_zk3s2gq_p7.reindex(grain_purchased_zk3s2gqs.index, axis=0)
        buy_sup_exp_zqs_p7 = grains_buy_price_zk3s2gqs_p7.mul(grain_purchased_zk3s2gqs, axis=0).groupby(axis=0,level=(0,4,5)).sum()  # sum grain pool & landuse & s2
        total_sup_exp_zqs_p7 = sup_transferred_exp_zqs_p7 + buy_sup_exp_zqs_p7
        grain['sup_exp_p7zqs'] = total_sup_exp_zqs_p7.unstack([0,1,2])
        return grain


def f_mvf_summary(lp_vars):
    mvf_qsp6vq = f_vars2df(lp_vars, 'mvf')
    return mvf_qsp6vq.unstack([-1,-2])


def f_crop_cash_summary(lp_vars, r_vals, option=0):
    '''
    Crop summary. Includes pasture inputs.
    :param option:

        #. Return - tuple: fert cost, chem cost, miscellaneous costs and grain revenue for each landuse

    '''
    ##call rotation function to get rotation info
    phases_rk, v_phase_change_increase_area_qszrl_p7, rot_area_qszrl_p7 = f_rotation(lp_vars, r_vals)[0:3]
    v_phase_area_zrlqs_p7 = rot_area_qszrl_p7.reorder_levels([2,3,4,0,1], axis=0).sort_index() #change the order so that reindexing works (new levels being added must be at the end)
    v_phase_change_increase_area_zrlqs_p7 = v_phase_change_increase_area_qszrl_p7.reorder_levels([2,3,4,0,1], axis=0).sort_index() #change the order so that reindexing works (new levels being added must be at the end)
    ##expenses
    ###fert
    ####v_phase
    phase_fert_cost_rl_p7z = r_vals['crop']['phase_fert_cost']
    exp_fert_ha_zrl_p7 = phase_fert_cost_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    exp_fert_ha_zrlqs_p7 = exp_fert_ha_zrl_p7.reindex(v_phase_area_zrlqs_p7.index, axis=0)
    exp_fert_zrqs_p7 = exp_fert_ha_zrlqs_p7.mul(v_phase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
    ####v_phase_increment
    phase_fert_cost_increment_rl_p7z = r_vals['crop']['phase_fert_cost_increment'].unstack([-2,-1])
    exp_fert_ha_increment_zrl_p7 = phase_fert_cost_increment_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    exp_fert_ha_increment_zrlqs_p7 = exp_fert_ha_increment_zrl_p7.reindex(v_phase_change_increase_area_zrlqs_p7.index, axis=0)
    exp_fert_increment_zrqs_p7 = exp_fert_ha_increment_zrlqs_p7.mul(v_phase_change_increase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
    ####combine v_phase and v_phase_increase
    exp_fert_zrqs_p7 = pd.concat([exp_fert_zrqs_p7, exp_fert_increment_zrqs_p7], axis=1).groupby(axis=1, level=(0)).sum()
    exp_fert_k_p7zqs = exp_fert_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,
                                                                            level=1).sum()  # reindex to include landuse and sum rot

    ###chem
    ####v_phase
    chem_cost_rl_p7z = r_vals['crop']['chem_cost']
    chem_cost_zrl_p7 = chem_cost_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    chem_cost_zrlqs_p7 = chem_cost_zrl_p7.reindex(v_phase_area_zrlqs_p7.index, axis=0)
    exp_chem_zrqs_p7 = chem_cost_zrlqs_p7.mul(v_phase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
    ####v_phase_increment
    chem_cost_increment_rl_p7z = r_vals['crop']['chem_cost_increment'].unstack([-2,-1])
    chem_cost_increment_zrl_p7 = chem_cost_increment_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    chem_cost_increment_zrlqs_p7 = chem_cost_increment_zrl_p7.reindex(v_phase_change_increase_area_zrlqs_p7.index, axis=0)
    exp_chem_increment_zrqs_p7 = chem_cost_increment_zrlqs_p7.mul(v_phase_change_increase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
    ####combine v_phase and v_phase_increase
    exp_chem_zrqs_p7 = pd.concat([exp_chem_zrqs_p7, exp_chem_increment_zrqs_p7], axis=1).groupby(axis=1, level=(0)).sum()
    exp_chem_k_p7zqs = exp_chem_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,
                                                                            level=1).sum()  # reindex to include landuse and sum rot

    ###misc
    ####v_phase
    stub_cost_rl_p7z = r_vals['crop']['stub_cost']
    insurance_cost_rl_p7z = r_vals['crop']['insurance_cost']
    seedcost_rl_p7z = r_vals['crop']['seedcost']
    misc_exp_ha_rl_p7z = pd.concat([stub_cost_rl_p7z, insurance_cost_rl_p7z, seedcost_rl_p7z], axis=1).groupby(axis=1, level=(0,1)).sum()  # stubble, seed & insurance
    misc_exp_ha_zrl_p7 = misc_exp_ha_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    misc_exp_ha_zrlqs_p7 = misc_exp_ha_zrl_p7.reindex(v_phase_area_zrlqs_p7.index, axis=0)
    misc_exp_ha_zrqs_p7 = misc_exp_ha_zrlqs_p7.mul(v_phase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu, need to reindex because some rotations have been dropped
    # misc_exp_ha_zr_c0p7 = misc_exp_ha_zrl_c0p7.reindex(rot_area_zrl.index).mul(rot_area_zrl, axis=0).sum(axis=0, level=(0,1))  # mul area and sum lmu, need to reindex because some rotations have been dropped
    ####v_phase_increment
    stub_cost_increment_rl_p7z = r_vals['crop']['stub_cost_increment'].unstack([-2,-1])
    insurance_cost_increment_rl_p7z = r_vals['crop']['insurance_cost_increment'].unstack([-2,-1])
    seedcost_increment_rl_p7z = r_vals['crop']['seedcost_increment'].unstack([-2,-1])
    misc_exp_ha_increment_rl_p7z = pd.concat([stub_cost_increment_rl_p7z, insurance_cost_increment_rl_p7z, seedcost_increment_rl_p7z], axis=1).groupby(axis=1, level=(0,1)).sum()  # stubble, seed & insurance
    misc_exp_ha_increment_zrl_p7 = misc_exp_ha_increment_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    misc_exp_ha_increment_zrlqs_p7 = misc_exp_ha_increment_zrl_p7.reindex(v_phase_change_increase_area_zrlqs_p7.index, axis=0)
    misc_exp_ha_increment_zrqs_p7 = misc_exp_ha_increment_zrlqs_p7.mul(v_phase_change_increase_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu, need to reindex because some rotations have been dropped
    ####combine v_phase and v_phase_increase
    misc_exp_ha_zrqs_p7 = pd.concat([misc_exp_ha_zrqs_p7, misc_exp_ha_increment_zrqs_p7], axis=1).groupby(axis=1, level=(0)).sum()
    misc_exp_k_p7zqs = misc_exp_ha_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,
                                                                            level=1).sum()  # reindex to include landuse and sum rot

    ##revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    rev_grain_k1_p7zqs = grain_summary['rev_grain_k1_p7zqs']

    ##return all if option==0
    if option == 0:
        return exp_fert_k_p7zqs, exp_chem_k_p7zqs, misc_exp_k_p7zqs, rev_grain_k1_p7zqs


def f_stock_cash_summary(lp_vars, r_vals):
    '''
    Returns:

        #. expense and revenue items

    '''
    ##get reshaped variable
    stock_vars = d_vars['base'] #use vars base because z is active (in the pnl table it gets weighted by qsz later if required)

    ##numbers
    sire_numbers_qszg0 = stock_vars['sire_numbers_qszg0']
    dams_numbers_qsk2tvanwziy1g1 = stock_vars['dams_numbers_qsk2tvanwziy1g1']
    prog_numbers_qsk3k5twzia0xg2 = stock_vars['prog_numbers_qsk3k5twzia0xg2']
    offs_numbers_qsk3k5tvnwziaxyg3 = stock_vars['offs_numbers_qsk3k5tvnwziaxyg3']

    ##husb cost
    sire_cost_p7qszg0 = r_vals['stock']['sire_cost_p7zg0'][:,na,na,...] * sire_numbers_qszg0
    dams_cost_p7qsk2tva1nwziyg1 = r_vals['stock']['dams_cost_p7k2tva1nwziyg1'][:,na,na,...] * dams_numbers_qsk2tvanwziy1g1
    offs_cost_p7qsk3k5tvnwziaxyg3 = r_vals['stock']['offs_cost_p7k3k5tvnwziaxyg3'][:,na,na,...] * offs_numbers_qsk3k5tvnwziaxyg3

    ##purchase cost
    sire_purchcost_p7qszg0 = r_vals['stock']['purchcost_sire_p7zg0'][:,na,na,...] * sire_numbers_qszg0

    ##sale income
    salevalue_p7qszg0 = r_vals['stock']['salevalue_p7qzg0'][:,:,na,...] * sire_numbers_qszg0
    salevalue_p7qsk2tva1nwziyg1 = r_vals['stock']['salevalue_p7qk2tva1nwziyg1'][:,:,na,...] * dams_numbers_qsk2tvanwziy1g1
    salevalue_p7qsk3k5twzia0xg2 = r_vals['stock']['salevalue_p7qk3k5twzia0xg2'][:,:,na,...] * prog_numbers_qsk3k5twzia0xg2
    salevalue_p7qsk3k5tvnwziaxyg3 = r_vals['stock']['salevalue_p7qk3k5tvnwziaxyg3'][:,:,na,...] * offs_numbers_qsk3k5tvnwziaxyg3

    ##asset income - used to calculate the change in asset value at the start of season. This is required in the pnl report because sale management can vary across the z axis and then at the start of the season all z get averaged e.g. z0 might retain sheep and z4 might sell. If there is no asset value then z0 will look worse.
    assetvalue_startseason_p7qszg0 = r_vals['stock']['assetvalue_startseason_p7qzg0'][:,:,na,...] * sire_numbers_qszg0
    assetvalue_endseason_p7qszg0 = r_vals['stock']['assetvalue_endseason_p7qzg0'][:,:,na,...] * sire_numbers_qszg0
    assetvalue_startseason_p7qsk2tva1nwziyg1 = r_vals['stock']['assetvalue_startseason_p7qk2tva1nwziyg1'][:,:,na,...] * dams_numbers_qsk2tvanwziy1g1
    assetvalue_endseason_p7qsk2tva1nwziyg1 = r_vals['stock']['assetvalue_endseason_p7qk2tva1nwziyg1'][:,:,na,...] * dams_numbers_qsk2tvanwziy1g1
    assetvalue_startseason_p7qsk3k5tvnwziaxyg3 = r_vals['stock']['assetvalue_startseason_p7qk3k5tvnwziaxyg3'][:,:,na,...] * offs_numbers_qsk3k5tvnwziaxyg3
    assetvalue_endseason_p7qsk3k5tvnwziaxyg3 = r_vals['stock']['assetvalue_endseason_p7qk3k5tvnwziaxyg3'][:,:,na,...] * offs_numbers_qsk3k5tvnwziaxyg3

    ##wool income
    woolvalue_p7qszg0 = r_vals['stock']['woolvalue_p7qzg0'][:,:,na,...] * sire_numbers_qszg0
    woolvalue_p7qsk2tva1nwziyg1 = r_vals['stock']['woolvalue_p7qk2tva1nwziyg1'][:,:,na,...] * dams_numbers_qsk2tvanwziy1g1
    woolvalue_p7qsk3k5tvnwziaxyg3 = r_vals['stock']['woolvalue_p7qk3k5tvnwziaxyg3'][:,:,na,...] * offs_numbers_qsk3k5tvnwziaxyg3

    ###sum axis to return total income in each cash period
    siresale_p7qsz = fun.f_reduce_skipfew(np.sum, salevalue_p7qszg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7
    damssale_p7qsz = fun.f_reduce_skipfew(np.sum, salevalue_p7qsk2tva1nwziyg1, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z
    progsale_p7qsz = fun.f_reduce_skipfew(np.sum, salevalue_p7qsk3k5twzia0xg2, preserveAxis=(0,1,2,7))  # sum all axis except q,s,p7,z
    offssale_p7qsz = fun.f_reduce_skipfew(np.sum, salevalue_p7qsk3k5tvnwziaxyg3, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z
    sirewool_p7qsz = fun.f_reduce_skipfew(np.sum, woolvalue_p7qszg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7,z
    damswool_p7qsz = fun.f_reduce_skipfew(np.sum, woolvalue_p7qsk2tva1nwziyg1, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z
    offswool_p7qsz = fun.f_reduce_skipfew(np.sum, woolvalue_p7qsk3k5tvnwziaxyg3, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z
    stocksale_p7qsz = siresale_p7qsz + damssale_p7qsz + progsale_p7qsz + offssale_p7qsz
    wool_p7qsz = sirewool_p7qsz + damswool_p7qsz + offswool_p7qsz

    sirecost_p7qsz = fun.f_reduce_skipfew(np.sum, sire_cost_p7qszg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7,z
    damscost_p7qsz = fun.f_reduce_skipfew(np.sum, dams_cost_p7qsk2tva1nwziyg1, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z
    offscost_p7qsz = fun.f_reduce_skipfew(np.sum, offs_cost_p7qsk3k5tvnwziaxyg3, preserveAxis=(0,1,2,9))  # sum all axis except q,s,p7,z

    sire_purchcost_p7qsz = fun.f_reduce_skipfew(np.sum, sire_purchcost_p7qszg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7

    ###change in asset value at the season start. This needs to be reported in pnl because at the season start stock numbers get averaged.
    ### Meaning that if z0 retains animals and z1 sells animals z0 will have a lower apparent profit which is not correct.
    ### Adding this essentially means that at the end of the season animals are purchased and sold to get back to the starting point (e.g. a good season with more animals sells some animals to the poor season that has less animals so that all seasons have the same starting point).
    ### Note if weaning occurs in the period before season start there will be an error (value of prog that are sold will get double counted). Easiest solution is to change weaning date.
    trade_value_sire_p7qsz = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_p7qszg0, preserveAxis=(0,1,2,3))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_p7qszg0, preserveAxis=(0,1,2,3)), shift=-1, axis=0))  # sum all axis except q,s,p7. Roll so that the result is in the end p7
    trade_value_dams_p7qsz = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_p7qsk2tva1nwziyg1, preserveAxis=(0,1,2,9))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_p7qsk2tva1nwziyg1, preserveAxis=(0,1,2,9)), shift=-1, axis=0))  # sum all axis except q,s,p7,z. Roll so that the result is in the end p7
    trade_value_offs_p7qsz = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_p7qsk3k5tvnwziaxyg3, preserveAxis=(0,1,2,9))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_p7qsk3k5tvnwziaxyg3, preserveAxis=(0,1,2,9)), shift=-1, axis=0))  # sum all axis except q,s,p7,z. Roll so that the result is in the end p7
    trade_value_p7qsz = trade_value_sire_p7qsz + trade_value_dams_p7qsz + trade_value_offs_p7qsz

    ##expenses sup feeding
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    sup_grain_cost_p7zqs = grain_summary['sup_exp_p7zqs']
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)

    grain_fed_zkfp6qs = grain_fed_qszkfp6.reorder_levels([2,3,4,5,0,1]).sort_index() # change the order so that reindexing works (new levels being added must be at the end)
    supp_feedstorage_cost_p7zp6k3f = r_vals['sup']['total_sup_cost_p7zp6k3f']
    supp_feedstorage_cost_p7_zk3fp6qs = supp_feedstorage_cost_p7zp6k3f.unstack([1,3,4,2]).reindex(grain_fed_zkfp6qs.index, axis=1)
    supp_feedstorage_cost_p7_zk3fp6qs = supp_feedstorage_cost_p7_zk3fp6qs.mul(grain_fed_zkfp6qs, axis=1)
    supp_feedstorage_cost_p7zqs = supp_feedstorage_cost_p7_zk3fp6qs.groupby(axis=1, level=(0,4,5)).sum().stack([0,1,2]) #sum k & p6 & f

    ##infrastructure
    fixed_infra_cost_p7qsz = r_vals['stock']['rm_stockinfra_fix_p7z'][:,na,na,:] * r_vals['zgen']['mask_qs'][:,:,na]

    ##total costs
    husbcost_p7qsz = sirecost_p7qsz + damscost_p7qsz + offscost_p7qsz + fixed_infra_cost_p7qsz
    supcost_p7zqs = sup_grain_cost_p7zqs + supp_feedstorage_cost_p7zqs
    purchasecost_qsp7z = sire_purchcost_p7qsz

    ##get axis in correct order for pnl table
    stocksale_qszp7 = np.moveaxis(stocksale_p7qsz, source=0, destination=-1)
    trade_value_qszp7 = np.moveaxis(trade_value_p7qsz, source=0, destination=-1)
    wool_qszp7 = np.moveaxis(wool_p7qsz, source=0, destination=-1)
    husbcost_qszp7 = np.moveaxis(husbcost_p7qsz, source=0, destination=-1)
    purchasecost_qszp7 = np.moveaxis(purchasecost_qsp7z, source=0, destination=-1)
    supcost_qsz_p7 = supcost_p7zqs.unstack([2,3,1]).sort_index(axis=1).T
    return stocksale_qszp7, wool_qszp7, husbcost_qszp7, supcost_qsz_p7, purchasecost_qszp7, trade_value_qszp7


def f_labour_summary(lp_vars, r_vals, option=0):
    '''
    :param option:

        #. return total labour cost
        #. return amount for each enterprise

    '''
    ##mask to uncluster lp_vars
    maskz8_p5z = r_vals['lab']['maskz8_p5z']

    ##shapes
    keys_p5 = r_vals['lab']['keys_p5']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']

    qsp5z = keys_q, keys_s, keys_p5, keys_z
    qs = keys_q, keys_s

    ##total labour cost
    if option == 0:
        ###casual
        casual_cost_p7zp5 = r_vals['lab']['casual_cost_p7zp5']
        quantity_casual_qsp5z = f_vars2np(lp_vars, 'v_quantity_casual', qsp5z, maskz8_p5z, z_pos=-1)
        quantity_casual_qszp5 = np.swapaxes(quantity_casual_qsp5z, -1, -2)
        cas_cost_p7qsz = np.sum(casual_cost_p7zp5[:,na,na,:,:] * quantity_casual_qszp5, axis=-1)
        ###perm
        quantity_perm_qs = f_vars2np(lp_vars, 'v_quantity_perm', qs)  #no mask because no z axis
        perm_cost_p7z = r_vals['lab']['perm_cost_p7z']
        perm_cost_p7qsz = perm_cost_p7z[:,na,na,:] * quantity_perm_qs[:,:,na] * r_vals['zgen']['mask_qs'][:,:,na]
        ###manager
        quantity_manager_qs = f_vars2np(lp_vars, 'v_quantity_manager', qs) #no mask because no z axis
        manager_cost_p7z = r_vals['lab']['manager_cost_p7z']
        manager_cost_p7qsz = manager_cost_p7z[:,na,na,:] * quantity_manager_qs[:,:,na] * r_vals['zgen']['mask_qs'][:,:,na]
        ###total
        total_lab_cost_p7qsz = cas_cost_p7qsz + perm_cost_p7qsz + manager_cost_p7qsz
        return total_lab_cost_p7qsz

    ##labour breakdown for each worker level (table: labour period by worker level)
    if option == 1:
        ###sheep
        manager_sheep_qsp5z_w = f_vars2df(lp_vars, 'v_sheep_labour_manager', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        prem_sheep_qsp5z_w = f_vars2df(lp_vars, 'v_sheep_labour_permanent', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        casual_sheep_qsp5z_w = f_vars2df(lp_vars, 'v_sheep_labour_casual', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        sheep_labour = pd.concat([manager_sheep_qsp5z_w, prem_sheep_qsp5z_w, casual_sheep_qsp5z_w], axis=1).sum(axis=1, level=0)
        ###crop
        manager_crop_qsp5z_w = f_vars2df(lp_vars, 'v_phase_labour_manager', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        prem_crop_qsp5z_w = f_vars2df(lp_vars, 'v_phase_labour_permanent', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        casual_crop_qsp5z_w = f_vars2df(lp_vars, 'v_phase_labour_casual', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        crop_labour = pd.concat([manager_crop_qsp5z_w, prem_crop_qsp5z_w, casual_crop_qsp5z_w], axis=1).sum(axis=1, level=0)
        ###fixed
        manager_fixed_qsp5z_w = f_vars2df(lp_vars, 'v_fixed_labour_manager', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        prem_fixed_qsp5z_w = f_vars2df(lp_vars, 'v_fixed_labour_permanent', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        casual_fixed_qsp5z_w = f_vars2df(lp_vars, 'v_fixed_labour_casual', maskz8_p5z[:,na,:], z_pos=-1).unstack(-2)
        fixed_labour = pd.concat([manager_fixed_qsp5z_w, prem_fixed_qsp5z_w, casual_fixed_qsp5z_w], axis=1).sum(axis=1, level=0)
        return sheep_labour, crop_labour, fixed_labour


def f_dep_summary(lp_vars, r_vals):
    ##depreciation total
    keys_p7 = r_vals['fin']['keys_p7']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    qsp7z = keys_q, keys_s, keys_p7, keys_z
    dep_qsp7z = f_vars2np(lp_vars, 'v_dep', qsp7z, mask_season_p7z, z_pos=-1)
    return dep_qsp7z

def f_minroe_summary(lp_vars, r_vals):
    ##min return on expense cost
    keys_p7 = r_vals['fin']['keys_p7']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    qsp7z = keys_q, keys_s, keys_p7, keys_z

    minroe_qsp7z = f_vars2np(lp_vars, 'v_minroe', qsp7z, mask_season_p7z, z_pos=-1)
    return minroe_qsp7z

def f_asset_cost_summary(lp_vars, r_vals):
    ##asset opportunity cost
    keys_p7 = r_vals['fin']['keys_p7']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    qsp7z = keys_q, keys_s, keys_p7, keys_z
    asset_cost_qsp7z = f_vars2np(lp_vars, 'v_asset_cost', qsp7z, mask_season_p7z, z_pos=-1)
    return asset_cost_qsp7z

def f_wc_summary(lp_vars, r_vals):
    ##returns the maximum capital
    keys_p7 = r_vals['fin']['keys_p7']
    keys_c0 = r_vals['fin']['keys_c0']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    qsc0p7z = keys_q, keys_s, keys_c0, keys_p7, keys_z
    capital_qsc0p7z = f_vars2np(lp_vars, 'v_wc_debit', qsc0p7z, mask_season_p7z, z_pos=-1)
    capital_qsz = np.max(capital_qsc0p7z, axis=(2,3)) #max c0 and p7 axis
    keys_qsz = [keys_q, keys_s, keys_z]
    df_capital_qsz = f_numpy2df(capital_qsz, keys_qsz, [0,1], [2])
    return df_capital_qsz

def f_overhead_summary(r_vals):
    ##overheads/fixed expenses
    exp_fix_c = r_vals['fin']['overheads']
    return exp_fix_c

def f_deepflow_summary(r_vals):
    '''average recharge across the whole farm'''
    ##deepflow phases
    v_phase_area_qsp7zrl = d_vars['base']['v_phase_area_qsp7zrl']
    v_phase_area_qszrl = v_phase_area_qsp7zrl[:,:,-1,...] #slice p7[-1] - will need to be more complex if dual season.
    recharge_rl = r_vals['crop']['recharge_rl']
    total_rot_recharge_qsz = np.sum(v_phase_area_qszrl * recharge_rl, axis=(3,4))

    ##deepflow trees
    v_tree_area_l = d_vars['base']['v_tree_area_l']
    tree_recharge_l = r_vals['tree']['recharge_l']
    total_tree_recharge = np.sum(v_tree_area_l * tree_recharge_l)

    ##average deepflow across farm
    total_area_qsz = np.sum(v_tree_area_l) + np.sum(v_phase_area_qszrl, axis=(3,4))
    total_recharge_qsz = total_rot_recharge_qsz + total_tree_recharge
    ave_recharge_qsz = total_recharge_qsz/total_area_qsz

    ##make df
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_qsz = [keys_q, keys_s, keys_z]
    ave_recharge_qsz = f_numpy2df(ave_recharge_qsz, keys_qsz, [0,1], [2])

    return ave_recharge_qsz

def f_tree_summary(r_vals):
    ##fixed costs
    keys_p7 = r_vals['zgen']['keys_p7']
    keys_z = r_vals['zgen']['keys_z']
    sequestration_fixed_cost_p7z = r_vals['tree']['tree_sequestration_fixed_cost_p7z']
    biodiversity_fixed_cost_p7z = r_vals['tree']['tree_biodiversity_fixed_cost_p7z']
    total_fixed_costs_z_p7 = f_numpy2df(sequestration_fixed_cost_p7z + biodiversity_fixed_cost_p7z, [keys_p7, keys_z], [1], [0])

    ##variable costs
    tree_sequestration_variable_cost_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_sequestration_variable_cost_p7z', na_prod=[2]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])
    tree_biodiversity_variable_cost_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_biodiversity_variable_cost_p7z', na_prod=[2]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])
    tree_biomass_cost_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_biomass_cost_p7zl', na_prod=[]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])
    tree_estab_cost_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_estab_cost_p7zl', na_prod=[]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])

    ##total costs
    total_cost_z_p7 = (tree_sequestration_variable_cost_z_p7 + tree_biodiversity_variable_cost_z_p7 +
                       tree_biomass_cost_z_p7 + tree_estab_cost_z_p7 + total_fixed_costs_z_p7)

    ##income
    tree_sequestration_income_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_sequestration_income_p7zl', na_prod=[]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])
    tree_biodiversity_income_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_biodiversity_income_p7z', na_prod=[2]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])
    tree_biomass_income_z_p7 = f_stock_pasture_summary(r_vals, type='tree', prod='tree_biomass_income_p7zl', na_prod=[]
                                                    , weights='v_tree_area_l', na_weights=[0,1]
                                                    , keys='keys_p7zl', arith=2, index=[1], cols=[0])

    total_income_z_p7 = tree_sequestration_income_z_p7 + tree_biodiversity_income_z_p7 + tree_biomass_income_z_p7

    return total_cost_z_p7, total_income_z_p7

def f_dse(lp_vars, r_vals, method, per_ha, summary1=False, summary2=False, summary3=False):
    '''
    DSE calculation.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param method: int

            0. dse by normal weight
            1. dse by mei

    :param per_ha: Bool
        if true it returns DSE/ha else it returns total dse
    :param summary1: Bool
        if true it returns the total expected DSE/ha in winter. Used in the summary report
    :param summary2: Bool
        if true it returns the total numbers at the start and end of the season with qsz axis. Used in numbers summary report.
    :param summary3: Bool
        if true it returns the total DSE/ha in winter for each season.
    :return: DSE per pasture hectare for each sheep group.

    '''
    ##keys for table that is reported
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']
    shape_qsz = len(keys_q), len(keys_s), len(keys_z)
    ##user can change this if they want to report different axis. Keys must be a list and axis must be tuple. Check names below to get the axis positions.
    sire_preserve_ax = (0, 1, 2 ,3)
    sire_key = [keys_q, keys_s, keys_p6, keys_z]
    dams_preserve_ax = (0, 1, 3, 9)
    dams_key = [keys_q, keys_s, keys_p6, keys_z]
    offs_preserve_ax = (0, 1, 4, 9)
    offs_key = [keys_q, keys_s, keys_p6, keys_z]

    if summary1: #for summary DSE needs to be calculated with p6 and z axis (q,s,z axis is weighted and summed below)
        sire_preserve_ax = (0, 1, 2 ,3)
        dams_preserve_ax = (0, 1, 3, 9)
        offs_preserve_ax = (0, 1, 4, 9)

    if summary2: #for summary2 DSE needs to be calculated with q,s,z axis
        sire_preserve_ax = (1, 2, 3)
        dams_preserve_ax = (1, 2, 9)
        offs_preserve_ax = (1, 2, 9)

    if summary3: #for summary3 DSE needs to be calculated with q, s, p6 & z
        sire_preserve_ax = (0, 1, 2 ,3)
        dams_preserve_ax = (0, 1, 3, 9)
        offs_preserve_ax = (0, 1, 4, 9)

    stock_vars = d_vars['base'] #change this if z is not preserved

    if method == 0:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_qszg0'][:, :, na, :, :]
                                        * r_vals['stock']['dsenw_p6zg0'][na,na,...], preserveAxis=sire_preserve_ax)  # sum all axis except preserveAxis
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_qsk2tvanwziy1g1'][:, :, :, na, ...]
                                        * r_vals['stock']['dsenw_k2p6tva1nwziyg1'][na,na,...], preserveAxis=dams_preserve_ax)  # sum all axis except preserveAxis
        ##offs
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'][:, :, :, :, na, ...]
                                        * r_vals['stock']['dsenw_k3k5p6tvnwziaxyg3'][na,na,...], preserveAxis=offs_preserve_ax)  # sum all axis except preserveAxis
    else:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_qszg0'][:, :, na, :, :]
                                        * r_vals['stock']['dsemj_p6zg0'][na,na,...], preserveAxis=sire_preserve_ax)  # sum all axis except preserveAxis
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_qsk2tvanwziy1g1'][:, :, :, na, ...]
                                        * r_vals['stock']['dsemj_k2p6tva1nwziyg1'][na,na,...], preserveAxis=dams_preserve_ax)  # sum all axis except preserveAxis
        ##offs
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'][:, :, :, :, na, ...]
                                        * r_vals['stock']['dsemj_k3k5p6tvnwziaxyg3'][na,na,...], preserveAxis=offs_preserve_ax)  # sum all axis except preserveAxis

    ##dse per ha if user opts for this level of detail
    if per_ha:
        df_pasture_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
        pasture_area_qsp6z = df_pasture_area_qsz.values.reshape(shape_qsz)[:,:,na,:]
        dse_sire = fun.f_divide(dse_sire, pasture_area_qsp6z) #this only works if z is the last axis
        dse_dams = fun.f_divide(dse_dams, pasture_area_qsp6z)
        dse_offs = fun.f_divide(dse_offs, pasture_area_qsp6z)

    if summary1:
        prob_qsz = r_vals['zgen']['z_prob_qsz'][:,:,:]
        sr_qsz = np.sum(r_vals['stock']['wg_propn_p6z'] * (dse_sire + dse_dams + dse_offs), axis=-2).round(2)  #sum SR for all sheep groups in winter grazed fp (to return winter sr)
        ###stdev and range
        sr_mean = np.sum(sr_qsz * prob_qsz)
        ma_sr_qsz = np.ma.masked_array(sr_qsz, prob_qsz==0)
        sr_max = np.max(ma_sr_qsz)
        sr_min = np.min(ma_sr_qsz)
        sr_stdev = np.sum((sr_qsz - sr_mean) ** 2 * prob_qsz)**0.5
        return sr_mean, sr_max, sr_min, sr_stdev
    elif summary2:
        sire_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_p7qzg0'][:, :, na, ...]>0)
                                                           * stock_vars['sire_numbers_qszg0'], preserveAxis=sire_preserve_ax)
        sire_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_p7qzg0'][:, :, na, ...]>0)
                                                         * stock_vars['sire_numbers_qszg0'], preserveAxis=sire_preserve_ax)
        dams_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_p7qk2tva1nwziyg1'][:, :, na, ...]>0)
                                                                     * stock_vars['dams_numbers_qsk2tvanwziy1g1'], preserveAxis=dams_preserve_ax)
        dams_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_p7qk2tva1nwziyg1'][:, :, na, ...]>0)
                                                                   * stock_vars['dams_numbers_qsk2tvanwziy1g1'], preserveAxis=dams_preserve_ax)
        offs_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_p7qk3k5tvnwziaxyg3'][:, :, na, ...]>0)
                                                                       * stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'], preserveAxis=offs_preserve_ax)
        offs_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_p7qk3k5tvnwziaxyg3'][:, :, na, ...]>0)
                                                                     * stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'], preserveAxis=offs_preserve_ax)

        keys_qszS = [keys_q, keys_s, keys_z, ['start', 'end']]
        numbers_start_qsz = (sire_numbers_startseason_qsz + dams_numbers_startseason_qsz + offs_numbers_startseason_qsz).round(2)
        numbers_end_qsz = (sire_numbers_endseason_qsz + dams_numbers_endseason_qsz + offs_numbers_endseason_qsz).round(2)
        numbers_qsz = np.stack([numbers_start_qsz, numbers_end_qsz], axis=-1)
        numbers_qsz = f_numpy2df(numbers_qsz, keys_qszS, [0, 1], [2, 3])
        return numbers_qsz

    elif summary3:
        sr_qsz = np.sum(r_vals['stock']['wg_propn_p6z'] * (dse_sire + dse_dams + dse_offs), axis=-2).round(2)  #sum SR for all sheep groups in winter grazed fp (to return winter sr)
        keys_qsz = [keys_q, keys_s, keys_z]
        return f_numpy2df(sr_qsz, keys_qsz, [0, 1, 2], [])

    ##turn to table - rows and cols need to be a list of lists/arrays
    dse_sire = fun.f_produce_df(dse_sire.ravel(), rows=sire_key, columns=[['Sire DSE']])
    dse_dams = fun.f_produce_df(dse_dams.ravel(), rows=dams_key, columns=[['Dams DSE']])
    dse_offs = fun.f_produce_df(dse_offs.ravel(), rows=offs_key, columns=[['Offs DSE']])

    return dse_sire, dse_dams, dse_offs


def f_profitloss_table(lp_vars, r_vals, option=1):
    '''
    Returns profit and loss statement for selected trials. Multiple trials result in a stacked pnl table.

    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :param option: int - controls how q, s and z are reported. Default is to report them. Option 2 does a weighted average.
    :return: dataframe

    '''
    ##read stuff from other functions that is used in rev and cost section
    exp_fert_k_p7zqs, exp_chem_k_p7zqs, misc_exp_k_p7zqs, rev_grain_k1_p7zqs = f_crop_cash_summary(lp_vars, r_vals, option=0)
    exp_mach_k_p7zqs, mach_insurance_p7z = f_mach_summary(lp_vars, r_vals)
    stocksale_qszp7, wool_qszp7, husbcost_qszp7, supcost_qsz_p7, purchasecost_qszp7, trade_value_qszp7 = f_stock_cash_summary(lp_vars, r_vals)
    slp_estab_cost_qsz_p7 = f_stock_pasture_summary(r_vals, type='slp', prod='slp_estab_cost_p7z', na_prod=[0,1,4]
                                             , weights='v_slp_ha_qszl', na_weights=[2]
                                             , keys='keys_qsp7zl', arith=2, index=[0,1,3], cols=[2])
    tree_cost_z_p7, tree_income_z_p7 = f_tree_summary(r_vals)
    labour_p7qsz = f_labour_summary(lp_vars, r_vals, option=0)
    exp_fix_p7_z = f_overhead_summary(r_vals).unstack()
    dep_qsp7z = f_dep_summary(lp_vars, r_vals)
    minroe_qsp7z = f_minroe_summary(lp_vars,r_vals)
    asset_cost_qsp7z = f_asset_cost_summary(lp_vars,r_vals)

    ##manipulate arrays into correct shape
    slp_landuses = ['sp']
    all_pas = np.setxor1d(r_vals['rot']['all_pastures'], slp_landuses)  # landuse sets - excludes slp
    all_crops = r_vals['rot']['all_crops']
    keys_p7 = r_vals['fin']['keys_p7']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    len_p7 = len(keys_p7)
    len_z = len(keys_z)
    ###rev
    rev_grain_p7_qsz = rev_grain_k1_p7zqs.sum(axis=0).unstack([2,3,1]).sort_index(axis=1)  # sum landuse axis
    ###exp
    ####machinery
    df_mask_qs = pd.DataFrame(r_vals['zgen']['mask_qs'],keys_q,keys_s).stack()
    mach_p7zqs = exp_mach_k_p7zqs.sum(axis=0)  # sum landuse
    mach_p7_qsz = mach_p7zqs.unstack([2,3]).add(mach_insurance_p7z, axis=0).mul(df_mask_qs,axis=1).unstack()
    ####crop & pasture
    pasfert_p7_qsz = exp_fert_k_p7zqs[exp_fert_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    slpfert_p7_qsz = exp_fert_k_p7zqs[exp_fert_k_p7zqs.index.isin(slp_landuses)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropfert_p7_qsz = exp_fert_k_p7zqs[exp_fert_k_p7zqs.index.isin(all_crops)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    paschem_p7_qsz = exp_chem_k_p7zqs[exp_chem_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    slpchem_p7_qsz = exp_chem_k_p7zqs[exp_chem_k_p7zqs.index.isin(slp_landuses)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropchem_p7_qsz = exp_chem_k_p7zqs[exp_chem_k_p7zqs.index.isin(all_crops)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    pasmisc_p7_qsz = misc_exp_k_p7zqs[misc_exp_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    slpmisc_p7_qsz = misc_exp_k_p7zqs[misc_exp_k_p7zqs.index.isin(slp_landuses)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropmisc_p7_qsz = misc_exp_k_p7zqs[misc_exp_k_p7zqs.index.isin(all_crops)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    pas_p7_qsz = pd.concat([pasfert_p7_qsz, paschem_p7_qsz, pasmisc_p7_qsz], axis=0).groupby(axis=0, level=0).sum()
    slp_p7_qsz = pd.concat([slpfert_p7_qsz, slpchem_p7_qsz, slpmisc_p7_qsz], axis=0).groupby(axis=0, level=0).sum()
    crop_p7_qsz = pd.concat([cropfert_p7_qsz, cropchem_p7_qsz, cropmisc_p7_qsz], axis=0).groupby(axis=0, level=0).sum()
    ####fixed overhead expenses
    index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
    exp_fix_p7_qsz = exp_fix_p7_z.reindex(index_qsz, axis=1, level=-1).stack().mul(df_mask_qs,axis=1).unstack()
    ###depreciation
    dep_qsz = dep_qsp7z[:,:,-1,:].ravel() #take end slice of season stages
    ###minroe
    minroe_qsz = minroe_qsp7z[:,:,-1,:].ravel() #take end slice of season stages
    ###asset opportunity cost
    asset_cost_qsz = asset_cost_qsp7z[:,:,-1,:].ravel() #take end slice of season stages

    ##create p/l dataframe
    idx = pd.IndexSlice
    subtype_rev = ['grain', 'sheep sales', 'wool', 'trees', 'Total Revenue (net of selling costs and freight)']
    subtype_exp = ['crop', 'pasture', 'salt land pasture', 'trees', 'stock husb and infra', 'stock sup', 'stock purchase', 'machinery', 'labour', 'fixed', 'Total expenses']
    subtype_tot = ['1 EBITDA', '2 depreciation', '3 asset value change', '4 profit', '5 opportunity_cost', '6 minRoe', '7 obj'] #numbered to keep them in the correct order
    pnl_rev_index = pd.MultiIndex.from_product([keys_q, keys_s, keys_z, ['Revenue'], subtype_rev], names=['Sequence_year', 'Sequence', 'Season', 'Type', 'Subtype'])
    pnl_exp_index = pd.MultiIndex.from_product([keys_q, keys_s, keys_z, ['Expense'], subtype_exp], names=['Sequence_year', 'Sequence', 'Season', 'Type', 'Subtype'])
    pnl_tot_index = pd.MultiIndex.from_product([keys_q, keys_s, keys_z, ['Total'], subtype_tot], names=['Sequence_year', 'Sequence', 'Season', 'Type', 'Subtype'])
    pnl_dsp_index = pd.MultiIndex.from_product([['Weighted obj - AFO'], [''], [''], [''], ['']], names=['Sequence_year', 'Sequence', 'Season', 'Type', 'Subtype'])
    pnl_dsp2_index = pd.MultiIndex.from_product([['Weighted obj - PNL'], [''], [''], [''], ['']], names=['Sequence_year', 'Sequence', 'Season', 'Type', 'Subtype'])
    pnl_index = pnl_rev_index.append(pnl_exp_index).append(pnl_tot_index).append(pnl_dsp_index).append(pnl_dsp2_index)
    # pnl_cols = pd.MultiIndex.from_product([keys_c0, keys_p7])
    pnl_cols = keys_p7
    pnl = pd.DataFrame(index=pnl_index, columns=pnl_cols)  # need to initialise df with multiindex so rows can be added
    pnl = pnl.sort_index() #have to sort to stop performance warning

    ##income - add to p/l table each as a new row
    ### Note: season start trade - at the start of each season stock numbers are averaged across the z axis. This item essentially accounts for a season with more animals selling some of its animals to seasons with less animals.
    pnl.loc[idx[:, :, :,'Revenue','grain'],:] = rev_grain_p7_qsz.T.reindex(pnl_cols, axis=1).values #reindex because  has been sorted alphabetically
    pnl.loc[idx[:, :, :, 'Revenue', 'sheep sales'], :] = stocksale_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Revenue', 'wool'], :] = wool_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Revenue', 'trees'], :] = tree_income_z_p7.values
    pnl.loc[idx[:, :, :, 'Revenue', 'Total Revenue (net of selling costs and freight)'], :] = pnl.loc[pnl.index.get_level_values(3) == 'Revenue'].groupby(axis=0,level=(0,1,2)).sum().values

    ##expenses - add to p/l table each as a new row
    pnl.loc[idx[:, :, :, 'Expense', 'crop'], :] = crop_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'pasture'], :] = pas_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'salt land pasture'], :] = slp_estab_cost_qsz_p7.add(slp_p7_qsz.T).values
    pnl.loc[idx[:, :, :, 'Expense', 'trees'], :] = tree_cost_z_p7.values
    pnl.loc[idx[:, :, :, 'Expense', 'stock husb and infra'], :] = husbcost_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Expense', 'stock sup'], :] = supcost_qsz_p7.values
    pnl.loc[idx[:, :, :, 'Expense', 'stock purchase'], :] = purchasecost_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Expense', 'machinery'], :] = mach_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'labour'], :] = labour_p7qsz.reshape(len_p7, -1).T
    pnl.loc[idx[:, :, :, 'Expense', 'fixed'], :] = exp_fix_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'Total expenses'], :] = pnl.loc[pnl.index.get_level_values(3) == 'Expense'].groupby(axis=0,level=(0,1,2)).sum().values

    ##EBITDA
    ebtd = pnl.loc[idx[:, :, :, 'Revenue', 'Total Revenue (net of selling costs and freight)']].values - pnl.loc[idx[:, :, :, 'Expense', 'Total expenses']].values
    pnl.loc[idx[:, :, :, 'Total', '1 EBITDA'], :] = ebtd #interest is counted in the cashflow of each item - it is hard to separate so it is not reported separately

    ##Full year - add a column which is total of all cashflow period
    pnl['Full year'] = pnl.sum(axis=1)


    ##farm profit - cash transaction minus depreciation and change in asset value
    ###depreciation
    pnl.loc[idx[:, :, :, 'Total', '2 depreciation'], 'Full year'] = dep_qsz
    ###change in asset value - this is to reflect the change in assets if a poor season sell part of the core flock
    pnl.loc[idx[:, :, :, 'Total', '3 asset value change'], 'Full year'] = trade_value_qszp7.reshape(-1, len_p7).sum(axis=1)
    ###calc farm profit - cash transaction minus depreciation and change in asset value
    profit_qsz = ebtd.sum(axis=1) - dep_qsz + trade_value_qszp7.reshape(-1, len_p7).sum(axis=1)
    pnl.loc[idx[:, :, :, 'Total', '4 profit'], 'Full year'] = profit_qsz

    ##obj - profit minus asset opp and minroe
    ##add the assets & minroe & depreciation
    pnl.loc[idx[:, :, :, 'Total', '5 opportunity_cost'], 'Full year'] = asset_cost_qsz
    pnl.loc[idx[:, :, :, 'Total', '6 minRoe'], 'Full year'] = minroe_qsz
    ###add the estimated obj for each season (calced from info above) profit minus opportunity cost and minroe
    season_obj_qsz = profit_qsz - asset_cost_qsz - minroe_qsz
    pnl.loc[idx[:, :, :, 'Total', '7 obj'], 'Full year'] = season_obj_qsz


    ##add the objective of all seasons - these should both be the same if the pnl is being calculated correctly - these are the only parts that have been discounted (time value of money)
    ###have to calc here after the step above so it doesnt turn to nan
    pnl.loc[idx['Weighted obj - AFO', '', '', '', ''], 'Full year'] = f_profit(lp_vars, r_vals, option=1)
    pnl.loc[idx['Weighted obj - PNL', '', '', '', ''], 'Full year'] = np.sum(season_obj_qsz * (r_vals['zgen']['z_prob_qsz'] * r_vals['zgen']['discount_factor_q'][:,na,na]).ravel())

    ##weight the qsz axis if option 2
    if option==2:
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        z_prob_qsz = z_prob_qsz.reindex(pnl.index, axis=0)
        pnl = pnl.mul(z_prob_qsz, axis=0).groupby(level=(-2,-1), axis=0).sum()
        ###add the objective of all seasons - need to do again because it becomes nan in the step above - these are the only parts that have been discounted (time value of money)
        pnl.loc[idx['Weighted obj - AFO', ''], 'Full year'] = f_profit(lp_vars, r_vals, option=1)
        pnl.loc[idx['Weighted obj - PNL', ''], 'Full year'] = np.sum(season_obj_qsz * (r_vals['zgen']['z_prob_qsz'] * r_vals['zgen']['discount_factor_q'][:,na,na]).ravel())

    ##round numbers in df
    pnl = pnl.astype(float).round(1).fillna(0)  # have to go to float so rounding works

    return pnl


def f_crop_summary(lp_vars, r_vals, option):
    '''
    Returns a production summary for each crop land use. Similar to that found in a farmers budget report.

    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :return: dataframe

    '''
    ##crop area
    keys_k = r_vals['pas']['keys_k']
    keys_k1 = r_vals['stub']['keys_k1']
    is_crop = np.any(keys_k==keys_k1[:,na], axis=0)
    landuse_area_qsz_k = f_area_summary(lp_vars, r_vals, option=4)
    landuse_area_qsz_k1 = landuse_area_qsz_k.loc[:,is_crop]

    ##propn fodder
    v_use_biomass_qsp7zkls2 = d_vars['base']['v_use_biomass_qsp7zkls2']  # use base vars because z is being reported
    v_use_biomass_qszks2 = v_use_biomass_qsp7zkls2.sum(axis=(2,5))
    total_biomass_qszk = v_use_biomass_qszks2.sum(axis=-1)
    graz_idx = list(r_vals['stub']['keys_s2']).index("Graz")
    biomass_fodder_qszk = v_use_biomass_qszks2[:,:,:,:,graz_idx]
    fodder_percent_qszk = fun.f_divide(biomass_fodder_qszk, total_biomass_qszk) * 100

    ##grain harvested
    total_grain_and_hay_produced_zk1s2gqs = f_grain_sup_summary(lp_vars, r_vals, option=5)
    total_grain_and_hay_produced_zk1s2qs = total_grain_and_hay_produced_zk1s2gqs.groupby(level=(0,1,2,4,5)).sum() #sum grain pools (note price has already been adjusted for propn of seconds)
    total_grain_produced_zk1qs = total_grain_and_hay_produced_zk1s2qs.unstack(2).loc[:,'Harv']
    total_grain_produced_qsz_k1 = total_grain_produced_zk1qs.unstack(1).reorder_levels([1,2,0], axis=0)

    ##grain price
    grain_price_k = r_vals['crop']['farmgate_price'].loc[:,'Harv']

    ##hay made
    total_hay_made_zk1qs = total_grain_and_hay_produced_zk1s2qs.unstack(2).loc[:,'Bale']
    total_hay_made_qsz_k1 = total_hay_made_zk1qs.unstack(1).reorder_levels([1,2,0], axis=0)

    ##hay price
    hay_price_k = r_vals['crop']['farmgate_price'].loc[:,'Bale']

    ##create p/l dataframe
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    len_k1 = len(keys_k1)


    idx = pd.IndexSlice
    type = ['Area', 'Fodder %', 'Grain Harvested (t)', 'Hay Made (t)', 'Grain Price', 'Hay Price']
    index = pd.MultiIndex.from_product([keys_q, keys_s, keys_z, type], names=['Sequence_year', 'Sequence', 'Season', 'Type'])
    cropsum = pd.DataFrame(index=index, columns=keys_k1)  # need to initialise df with multiindex so rows can be added
    cropsum = cropsum.sort_index() #have to sort to stop performance warning

    ##income - add to p/l table each as a new row
    ### Note: season start trade - at the start of each season stock numbers are averaged across the z axis. This item essentially accounts for a season with more animals selling some of its animals to seasons with less animals.
    cropsum.loc[idx[:, :, :,'Area'],:] = landuse_area_qsz_k1.values #reindex because  has been sorted alphabetically
    cropsum.loc[idx[:, :, :, 'Fodder %'], :] = fodder_percent_qszk.reshape(-1, len_k1)
    cropsum.loc[idx[:, :, :, 'Grain Harvested (t)'], :] = total_grain_produced_qsz_k1.values
    cropsum.loc[idx[:, :, :, 'Hay Made (t)'], :] = total_hay_made_qsz_k1.values
    cropsum.loc[idx[:, :, :, 'Grain Price'], :] = grain_price_k.values
    cropsum.loc[idx[:, :, :, 'Hay Price'], :] = hay_price_k.values

    ##weight the qsz axis if option 2
    if option==2:
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        z_prob_qsz = z_prob_qsz.reindex(cropsum.index, axis=0)
        cropsum = cropsum.mul(z_prob_qsz, axis=0).groupby(level=(-1), axis=0).sum()

    ##round numbers in df
    cropsum = cropsum.astype(float).round(1).fillna(0)  # have to go to float so rounding works

    return cropsum


def f_profit(lp_vars, r_vals, option=0):
    '''returns profit
    0- Profit = (rev - (exp + dep) * discount_factor)
    1- Risk neutral objective = (rev - (exp + minroe + asset_cost +dep) * discount_factor).
    2- Utility - this is the same as risk neutral obj if risk aversion is not included
    3- range and stdev of profit
    4- profit by zqs
    '''
    prob_qsz =r_vals['zgen']['z_prob_qsz']
    prob_c1 =r_vals['fin']['prob_c1'].values
    # obj_profit = f_vars2df(lp_vars, 'profit', keys_z)#.droplevel(1) #drop level 1 because no sets therefore nan
    minroe_qsp7z = f_minroe_summary(lp_vars, r_vals)
    asset_cost_qsp7z = f_asset_cost_summary(lp_vars, r_vals)
    if option == 0:
        return lp_vars['profit']
    elif option==1:
        minroe = np.sum(minroe_qsp7z[:,:,-1,:] * prob_qsz * r_vals['zgen']['discount_factor_q'][:,na,na])  #take end slice of season stages
        asset_cost = np.sum(asset_cost_qsp7z[:,:,-1,:] * prob_qsz* r_vals['zgen']['discount_factor_q'][:,na,na]) #take end slice of season stages
        return lp_vars['profit'] - minroe - asset_cost
    elif option == 2:
        return lp_vars['utility']
    else:
        keys_p7 = r_vals['fin']['keys_p7']
        keys_c1 = r_vals['fin']['keys_c1']
        mask_season_p7z = r_vals['zgen']['mask_season_p7z']
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        qsc1p7z = keys_q, keys_s, keys_c1, keys_p7, keys_z
        ###credit/debit (profit/loss before depreciation and tradevalue)
        credit_qsc1p7z = f_vars2np(lp_vars, 'v_credit', qsc1p7z, mask_season_p7z, z_pos=-1)
        debit_qsc1p7z = f_vars2np(lp_vars, 'v_debit', qsc1p7z, mask_season_p7z, z_pos=-1)
        credit_qsc1z = credit_qsc1p7z[...,-1,:]
        debit_qsc1z = debit_qsc1p7z[...,-1,:]
        ###dep
        dep_qsp7z = f_dep_summary(lp_vars, r_vals)
        dep_qsz = dep_qsp7z[:,:,-1,:]
        ###tradevalue
        trade_value_qszp7 = f_stock_cash_summary(lp_vars, r_vals)[-1]
        trade_value_qsz = np.sum(trade_value_qszp7, axis=-1)
        ###profit for each scenario
        profit_qsc1z = credit_qsc1z - debit_qsc1z + trade_value_qsz[:,:,na,:] - dep_qsz[:,:,na,:] #dep & tradevalue doesnt vary by price scenario
        ###stdev and range
        t_prob_qsz = np.broadcast_to(prob_qsz[:,:,na,:], profit_qsc1z.shape) #broadcast so the next step works
        ma_profit_qsc1z = np.ma.masked_array(profit_qsc1z, t_prob_qsz[:,:,na,:] == 0)
        profit_max = np.max(ma_profit_qsc1z)
        profit_min = np.min(ma_profit_qsc1z)
        profit_mean = np.sum(profit_qsc1z * prob_qsz[:,:,na,:] * prob_c1[:,na])
        profit_stdev = np.sum((profit_qsc1z - profit_mean)**2 * prob_qsz[:,:,na,:] * prob_c1[:,na])**0.5
        if option == 3:
            return profit_max, profit_min, profit_stdev
        elif option == 4:
            profit_qsz = np.sum(profit_qsc1z * prob_c1[:,na], axis=2) #average c1
            index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
            return pd.Series(profit_qsz.ravel(), index=index_qsz).unstack(-1)


def f_stock_pasture_summary(r_vals, build_df=True, keys=None, type=None, index=[], cols=[], arith=0,
                            prod=np.array([1]), na_prod=[], weights=None, na_weights=[], axis_slice={},
                            na_denweights=[], den_weights=1, na_prodweights=[], prod_weights=1,
                            den_assoc=None, na_den_assoc=[], assoc_axis=0):
    '''

    ..Note::

        #. prod and weights must be broadcastable.
        #. Specify axes to broadcasted/expanded version.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param build_df: bool: return df
    :key type: str: either 'stock', 'pas' or 'stub' to indicate which r_vals
    :key key: str: dict key for the axis keys
    :key index (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :key arith (optional, default = 0): int: arithmetic operation used.

                - option 0: return production param averaged across all axis that are not reported.
                - option 1: return weighted average of production param (using denominator weight returns production per day the animal is on hand)
                - option 2: weighted total production summed across all axis that are not reported.
                - option 3: weighted total production for each  (axis not reported are disregarded)
                - option 4: return weighted average of production param using prod>0 as the weights
                - option 5: return the maximum value across all axis that are not reported.

    :key prod (optional, default = 1): str/int/float: if it is a string then it is used as a key for stock_vars, if it is a number that number is used as the prod value
    :key na_prod (optional, default = []): list: position to add new axis
    :key weights (optional, default = None): str: weights to be used in arith (typically a lp variable e.g. numbers). Only required when arith>0
    :key na_weights (optional, default = []): list: position to add new axis
    :key den_weights (optional, default = 1): str: key to variable used to weight the denominator in the weighted average (required p6 reporting)
    :key na_denweights (optional, default = []): list: position to add new axis
    :key den_assoc (optional, default = None): str: key to variable used as an association (np.take_along) for the weight of the denominator in the weighted average (required for wean % report)
    :key prod_weights (optional, default = 1): str: keys to r_vals referencing array used to weight production.
    :key na_prodweights (optional, default = []): list: position to add new axis
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: summary of a numpy array in a pandas table.
    '''
    keys_key = keys
    keys_z = r_vals['zgen']['keys_z'] #before r_vals get sliced
    keys_q = r_vals['zgen']['keys_q'] #before r_vals get sliced

    ##read from stock reshape function
    if type == 'stock':
        r_vals = r_vals['stock']
        ###keys that will become the index and cols for table
        keys = r_vals[keys_key]
    else:
        r_vals = r_vals[type]
        ###keys that will become the index and cols for table
        keys = d_keys[keys_key]

    ##determine if use weighted or base version of vars (use base if z is in the index or cols). If z is singleton the q axis controls if weighting happens (this is required for MP model).
    ## this is not a perfect system it doesnt handle the sq model if z was summed and q was reported but that is not very likely.
    if len(keys_z) > 1 and any(list(key) == list(keys_z) for key in [keys[i] for i in index+cols]):
        vars = d_vars['base']
    elif len(keys_z) > 1:
        vars = d_vars['qsz_weighted']
    elif len(keys_q) > 1 and any(list(key) == list(keys_q) for key in [keys[i] for i in index+cols]):
        vars = d_vars['base']
    elif len(keys_q) > 1:
        vars = d_vars['qsz_weighted']
    else:
        vars = d_vars['base']

    ##An error here means the key provided for weights does not exist in lp_vars
    ###using None as the default for weights so that an error is generated later if an Arith option is selected that requires weights
    if weights is not None:
        try:
            weights = vars[weights]
        except KeyError: #sometimes when lp_vars are not included we want to use r_val as the weights
            weights = r_vals[weights]

        ###set weights to 0 if very small number (otherwise it can show up in report when it shouldn't)
        weights[np.isclose(weights, 0)] = 0

    ##initialise prod array from either r_vals or default value (this means you can perform arith with any number - mainly used for pasture when there is no production param)
    if isinstance(prod, str):
        prod = r_vals[prod]
    # else:
    #     prod = np.array([prod])     #this was adding another axis if an array was passed in
    ###set prod and weights to 0 if very small number (otherwise it can show up in report when it shouldn't)
    prod = np.where(np.isclose(prod, 0), 0, prod) #handles cases where prod is not an array

    ##initialise prod_weight array from either r_vals or default value
    if isinstance(prod_weights, str):
        prod_weights = r_vals[prod_weights]
    else:
        prod_weights = np.array([prod_weights])

    ##den weight - used in weighted average calc (default is 1)
    if isinstance(den_weights, str):
        den_weights = r_vals[den_weights]

    ##den accosiation - only use for weaning % report
    if isinstance(den_assoc, str):
        den_assoc = r_vals[den_assoc]

    ##other manipulation
    prod, weights, den_weights, prod_weights, den_assoc = f_add_axis(prod, na_prod, prod_weights, na_prodweights, weights, na_weights, den_weights, na_denweights, den_assoc, na_den_assoc)
    prod, prod_weights, weights, den_weights, keys = f_slice(prod, prod_weights, weights, den_weights, keys, arith, axis_slice)
    ##perform arith. if an axis is not reported it is included in the arith and the axis disappears
    report_idx = index + cols
    arith_axis = list(set(range(len(prod.shape))) - set(report_idx))
    prod = f_arith(prod, prod_weights, weights, den_weights, arith, arith_axis, den_assoc, assoc_axis)
    ##check for errors
    f_numpy2df_error(prod, weights, arith_axis, index, cols)
    if build_df:
        prod = f_numpy2df(prod, keys, index, cols)
        return prod
    else:
        return prod, keys


def f_lambing_status(lp_vars, r_vals, option=0, keys=None, index=[], cols=[], axis_slice={}, lp_vars_inc=True):
    '''
    Depending on the option selected this function can calc:
        Lamb survival (per ewe at start of dvp when lambing occurs - e.g. mort is included)
        Weaning %  (per dam at the start of the dvp when mating occurs - e.g. mort is included)
        Scanning %
        Proportion of dry ewes

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param option: (optional, default = 0): int:
            option 0: survival %
            option 1: wean %
            option 2: scan %
            option 3: Proportion of dry ewes
    :param keys:
    :param index: (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :param cols: (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :param axis_slice: (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :param lp_vars_inc: weight the report using the results stored in lp_vars. If false, the result is as if single animal in all slices
    :return: pandas df
    '''

    ##params for all options
    type = 'stock'

    ###lamb survival
    if option == 0:
        prod = 'nyatf_birth_k2tva1e1b1nw8ziyg1'
        na_prod = [0,1]
        prod2 = 'nfoet_birth_k2tva1e1b1nw8ziyg1'
        na_prod2 = [0,1]
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = [6,7]
            arith = 1
        else:
            weights = None
            na_weights = []
            arith = 4
        keys = 'dams_keys_qsk2tvaeb9nwziy1g1'

    ###wean percent
    elif option == 1:
        prod = 'nyatf_wean_k2tva1nw8ziyg1'
        na_prod = [0,1]
        den_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_denweights = [0,1]
        den_assoc = 'a_prev_matingv_wean_va1ziyg1' #this is used to roll numbers from mating to weaning
        na_den_assoc = [0,1,2,3,6,7]
        assoc_axis = 4 #v
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = 'r_numbers_start_k2tva1nwziyg1'
            na_weights = [0,1]
            arith = 1
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ###scan percent
    elif option == 2:
        prod = 'nfoet_scan_k2tva1nw8ziyg1'
        na_prod = [0,1]
        den_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_denweights = [0,1]
        den_assoc = 'a_prev_matingv_scan_va1ziyg1' #this is used to roll numbers from mating to scan
        na_den_assoc = [0,1,2,3,6,7]
        assoc_axis = 4 #v
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = 'r_numbers_start_k2tva1nwziyg1'
            na_weights = [0,1]
            arith = 1
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ###dry propn
    elif option == 3:
        prod = 'n_drys_k2tva1nw8ziyg1'
        na_prod = [0,1]
        den_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_denweights = [0,1]
        den_assoc = 'a_prev_matingv_scan_va1ziyg1' #this is used to roll numbers from mating to scan
        na_den_assoc = [0,1,2,3,6,7]
        assoc_axis = 4 #v
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = 'r_numbers_start_k2tva1nwziyg1'
            na_weights = [0,1]
            arith = 1
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ##calcs for survival
    if option == 0:
        ##colate the lp and report vals using f_stock_pasture_summary
        numerator, keys_sliced = f_stock_pasture_summary(r_vals, build_df=False, type=type
                                , prod=prod, na_prod=na_prod, weights=weights, na_weights=na_weights
                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
        denominator, keys_sliced = f_stock_pasture_summary(r_vals, build_df=False, type=type
                                , prod=prod2, na_prod=na_prod2, weights=weights, na_weights=na_weights
                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
        prog_alive_qsk2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(numerator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7) #b9 axis is shorten b axis: [0,1,2,3]
        prog_born_qsk2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(denominator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7)
        percentage = fun.f_divide(prog_alive_qsk2tvpa1e1b1nw8ziyg1, prog_born_qsk2tvpa1e1b1nw8ziyg1)
        ##make table
        percentage = f_numpy2df(percentage, keys_sliced, index, cols)

    ##calc for wean % or scan % or dry %
    else:
        intermediate, keys_sliced  = f_stock_pasture_summary(r_vals, build_df=False, type=type
                                    , prod=prod, na_prod=na_prod, weights=weights, na_weights=na_weights
                                    , den_weights=den_weights, na_denweights=na_denweights
                                    , den_assoc=den_assoc, na_den_assoc=na_den_assoc, assoc_axis=assoc_axis
                                    , keys=keys, arith=arith, index=index
                                    , cols=cols, axis_slice=axis_slice)
        percentage = f_numpy2df(intermediate, keys_sliced, index, cols)

    return percentage

def f_feed_budget(lp_vars, r_vals, option=0, nv_option=0, dams_cols=[], offs_cols=[], residue_cols=[]):
    '''
    Feed budget: stock mei requirement and feed mei supply.

    Reported axes for stock can vary.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :param option: (optional, default = 0): int:
            option 0: mei/hd/day & propn of mei from each feed source
            option 1: total mei / day
    :param nv_option: (optional, default = 0): int:
            option 0: Active NV pool
            option 1: NV pool summed (not active)
    :param dams_cols: (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :param offs_cols: (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :param residue_cols: (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :return: pandas df
    '''
    ##mei supply
    ###grn pasture
    type = 'pas'
    prod = 'me_cons_grnha_qfgop6lzt'
    na_prod = [1]  # s
    weights = 'greenpas_ha_qsfgop6lzt'
    keys = 'keys_qsfgop6lzt'
    arith = 2
    if nv_option==0:
        index = [0,1,7,5,2] #[q,s,z,p6,nv]
    else:
        index = [0,1,7,5] #[q,s,z,p6]
    cols = [8] #t
    grn_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    grn_mei = pd.concat([grn_mei], keys=['Green Pas'], axis=1)  # add feed type as header

    ###poc pasture
    type = 'pas'
    prod = 'poc_md_fp6z'
    na_prod = [0, 1, 4]  # q,s,l
    weights = 'poc_consumed_qsfp6lz'
    keys = 'keys_qsfp6lz'
    arith = 2
    if nv_option == 0:
        index = [0,1,5,3,2] #[q,s,z,p6,nv]
    else:
        index = [0,1,5,3] #[q,s,z,p6]
    cols = []
    poc_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    poc_mei.columns = ['Green Crop Paddock Pas'] # add feed type as header

    ###dry pasture
    type = 'pas'
    prod = 'dry_mecons_t_fdp6zt'
    na_prod = [0, 1, 6]  # q,s,l
    weights = 'drypas_consumed_qsfdp6zlt'
    keys = 'keys_qsfdp6zlt'
    arith = 2
    if nv_option==0:
        index = [0,1,5,4,2] #[q,s,z,p6,nv]
    else:
        index = [0,1,5,4] #[q,s,z,p6]
    cols = [7] #t
    dry_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    dry_mei = pd.concat([dry_mei], keys=['Dry Pas'], axis=1)  # add feed type as header

    ###nap pasture
    type = 'pas'
    prod = 'dry_mecons_t_fdp6zt' #nap is same md as dry pasture
    na_prod = [0, 1]  # q,s
    weights = 'nap_consumed_qsfdp6zt'
    keys = 'keys_qsfdp6zt'
    arith = 2
    if nv_option==0:
        index = [0,1,5,4,2] #[q,s,z,p6,nv]
    else:
        index = [0,1,5,4] #[q,s,z,p6]
    cols = []
    nap_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    nap_mei.columns = ['Dry Crop Paddock Pas'] # add feed type as header

    ###residue
    prod = 'md_zp6fks1'
    na_prod = [0, 1, 7]  # q,s, s2
    type = 'stub'
    weights = 'stub_qszp6fks1s2'
    keys = 'keys_qszp6fks1s2'
    arith = 2
    if nv_option==0:
        index = [0, 1, 2, 3, 4]  # q,s,z,p6,nv
    else:
        index = [0, 1, 2, 3]  # q,s,z,p6
    cols = residue_cols
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    res_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    res_mei = pd.concat([res_mei.squeeze()], keys=['Crop Residue'], axis=1)  # add feed type as header. Squeeze gets rid of 0 level from header if no columns are reported.

    ###crop graze
    prod = 'crop_md_fkp6p5zl'
    na_prod = [0, 1]  # q,s
    type = 'crpgrz'
    weights = 'crop_consumed_qsfkp6p5zl'
    keys = 'keys_qsfkp6p5zl'
    arith = 2
    if nv_option==0:
        index = [0, 1, 6, 4, 2]  # q,s,z,p6,nv
    else:
        index = [0, 1, 6, 4]  # q,s,z,p6
    cols = []
    crop_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    crop_mei.columns = ['Crop Graze'] # add feed type as header

    ###saltbush (just the saltbush not the understory)
    prod = 'sb_me_zp6f'
    na_prod = [0, 1, 5]  # q,s
    type = 'slp'
    weights = 'v_tonnes_sb_consumed_qszp6fl'
    keys = 'keys_qszp6fl'
    arith = 2
    if nv_option==0:
        index = [0, 1, 2, 3, 4]  # q,s,z,p6,nv
    else:
        index = [0, 1, 2, 3]  # q,s,z,p6
    cols = []
    sb_mei = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    sb_mei.columns = ['Saltbush'] # add feed type as header

    ###sup
    sup_md_tonne_fk3p6z = r_vals['sup']['md_tonne_fk3p6z']
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)
    sup_mei_qs_fk3p6z = grain_fed_qszkfp6.unstack([4,3,5,2]).sort_index(axis=1).mul(sup_md_tonne_fk3p6z, axis=1)
    sup_mei_qszp6f = sup_mei_qs_fk3p6z.stack([3,2,0]).sort_index(axis=1).sum(axis=1)
    if nv_option==1:
        sup_mei_qszp6f = sup_mei_qszp6f.unstack().sum(axis=1)
    sup_mei = pd.DataFrame(sup_mei_qszp6f, columns=['Supplement']) # add feed type as header

    ##stock mei requirement
    if option==0:
        arith = 1
    else:
        arith = 2

    ###sires
    type = 'stock'
    prod = 'mei_sire_p6fzg0'
    na_prod = [0, 1]  # q,s
    weights = 'sire_numbers_qszg0'
    na_weights = [2, 3] #p6, f
    den_weights = 'stock_days_p6fzg0'
    na_denweights = [0, 1]  # q,s
    keys = 'sire_keys_qsp6fzg0'
    if nv_option==0:
        index = [0, 1, 4, 2, 3]  # [q,s,z,p6,nv]
    else:
        index = [0, 1, 4, 2]  # [q,s,z,p6]
    cols = []
    mei_sire = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_sire = pd.concat([mei_sire.squeeze()], keys=['Sire'], axis=1) # add stock type as header. Squeeze gets rid of 0 level from header if no columns are reported.

    ###dams
    type = 'stock'
    prod = 'mei_dams_k2p6ftova1nw8ziyg1'
    na_prod = [0, 1]  # q,s
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [3, 4, 6] #p6, f, o
    den_weights = 'stock_days_k2p6ftova1nwziyg1'
    na_denweights = [0, 1]  # q,s, o
    keys = 'dams_keys_qsk2p6ftovanwziy1g1'
    if nv_option==0:
        index = [0, 1, 11, 3, 4]  # [q,s,z,p6,nv]
    else:
        index = [0, 1, 11, 3]  # [q,s,z,p6]
    cols = dams_cols
    mei_dams = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_dams = pd.concat([mei_dams.squeeze()], keys=['Dams'], axis=1) # add stock type as header. Squeeze gets rid of 0 level from header if no columns are reported.

    ###offs
    type = 'stock'
    prod = 'mei_offs_k3k5p6ftsvnw8ziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [4, 5, 7] #p6, f, shear
    den_weights = 'stock_days_k3k5p6ftsvnwziaxyg3'
    na_denweights = [0, 1]  # q,s
    keys = 'offs_keys_qsk3k5p6ftsvnwziaxyg3'
    if nv_option==0:
        index = [0, 1, 11, 4, 5]  # [q,s,z,p6,nv]
    else:
        index = [0, 1, 11, 4]  # [q,s,z,p6]
    cols = offs_cols
    mei_offs = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_offs = pd.concat([mei_offs.squeeze()], keys=['Offs'], axis=1) # add stock type as header. Squeeze gets rid of 0 level from header if no columns are reported.

    ##stick feed stuff together
    ###first make everything have the same number of col levels - not the neatest but couldn't find a better way
    arrays = [grn_mei, dry_mei, poc_mei, nap_mei, res_mei, crop_mei, sb_mei, sup_mei, mei_sire, mei_dams, mei_offs]
    ####determine the max number of column levels
    max_levels=1
    for array in arrays:
        max_levels = max(max_levels, array.columns.nlevels)
    for array in range(len(arrays)):
        extra_levels = max_levels - arrays[array].columns.nlevels
        for extra_lev in range(extra_levels):
            old_idx = arrays[array].columns.to_frame() # Convert index to dataframe
            old_idx.insert(len(old_idx.columns), len(old_idx.columns), '') # Insert new level at the end
            arrays[array].columns = pd.MultiIndex.from_frame(old_idx) # Convert back to MultiIndex
    feed_budget_supply = pd.concat(arrays[0:8], axis=1).round(1) #round so that little numbers don't cause issues
    feed_budget_req = pd.concat(arrays[8:], axis=1).round(1) #round so that little numbers don't cause issues

    ###if option 0 - calc propn of mei from each feed source (has to be done after summing nv axis)
    if option==0:
        feed_budget_supply = feed_budget_supply.div(feed_budget_supply.sum(axis=1), axis=0).fillna(0)
    else:##calc total MEI per day
        days_zp6 = pd.DataFrame(r_vals['pas']['days_p6z'], index=r_vals['pas']['keys_p6'], columns=r_vals['zgen']['keys_z']).T.stack()
        if nv_option==0:
            feed_budget_supply = feed_budget_supply.unstack([0,1,-1]).div(days_zp6, axis=0).stack([-3,-2,-1]).reorder_levels([2,3,0,1,4])
            feed_budget_req = feed_budget_req.unstack([0,1,-1]).div(days_zp6, axis=0).stack([-3,-2,-1]).reorder_levels([2,3,0,1,4])
        else:
            feed_budget_supply = feed_budget_supply.unstack([0,1]).div(days_zp6, axis=0).stack([-2,-1]).reorder_levels([2,3,0,1])
            feed_budget_req = feed_budget_req.unstack([0,1]).div(days_zp6, axis=0).stack([-2,-1]).reorder_levels([2,3,0,1])

    ###add stock mei requirement
    feed_budget = pd.concat([feed_budget_supply, feed_budget_req], axis=1)

    ##add fp date to index
    keys_p6 = r_vals['pas']['keys_p6']
    keys_z = r_vals['zgen']['keys_z']
    fp_dates = r_vals['pas']['fp_date_start_p6z']
    fp_dates = pd.DataFrame(np.maximum(1, fp_dates), index=keys_p6, columns=keys_z).T.stack() #min 1 because to_datetime cant handle 0.
    fp_dates = pd.to_datetime(fp_dates, format='%j').dt.strftime('%d-%b')
    if nv_option == 0:
        fp_idx = feed_budget.index.droplevel([0,1,-1]).tolist()
    else:
        fp_idx = feed_budget.index.droplevel([0,1]).tolist()
    new_level_values = fp_dates.loc[fp_idx].values
    idx = feed_budget.index.to_frame() # Convert index to dataframe
    idx.insert(4, 'fp_date', new_level_values) # Insert new level at specified location
    feed_budget.index = pd.MultiIndex.from_frame(idx) # Convert back to MultiIndex

    return feed_budget.astype(float).round(2)


def f_sup_per_dse(lp_vars, r_vals):
    '''
    Total supplement fed per dse per day.
    '''
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)
    grain_fed_qsp6z = grain_fed_qszkfp6.unstack([3,4]).sum(axis=1).reorder_levels([0,1,3,2])
    dse_sire_qsp6z, dse_dams_qsp6z, dse_offs_qsp6z = f_dse(lp_vars, r_vals, 0, False)
    total_dse_qsp6z = dse_dams_qsp6z.add(dse_offs_qsp6z, fill_value=0).add(dse_sire_qsp6z, fill_value=0).sum(axis=1)
    days_p6z = pd.DataFrame(r_vals['pas']['days_p6z'], index=r_vals['pas']['keys_p6'], columns=r_vals['zgen']['keys_z']).stack()
    sup_per_dse_qsp6z = grain_fed_qsp6z/total_dse_qsp6z
    sup_per_dse_per_day_qs_p6z = sup_per_dse_qsp6z.unstack([-2,-1]).div(days_p6z, axis=1)
    sup_per_dse_per_day_qsp6_z = sup_per_dse_per_day_qs_p6z.stack(0)

    return sup_per_dse_per_day_qsp6_z * 1000


def f_emission_summary(lp_vars, r_vals, option=0):
    '''
    Summary of whole farm emissions. The report summarises the methane, nitrous oxide, carbon dioxide and
    carbon dioxide equivalents.

    :param lp_vars:
    :param r_vals:
    :param option: 0 = active z, 1 = weighted average z
    :return:
    '''
    ##inputs
    ch4_gwp_factor = r_vals['stock']['ch4_gwp_factor']
    n2o_gwp_factor = r_vals['stock']['n2o_gwp_factor']
    arith = 2

    ##calculate n2o and ch4 emissions from livestock (emissions are linked to animal and feed activities)
    n2o = {}
    ch4 = {}
    for e in ['ch4', 'n2o']:
        d = eval(e)
        ###sires
        type = 'stock'
        prod = '{0}_animal_zg0'.format(e)
        na_prod = [0, 1]  # q,s
        weights = 'sire_numbers_qszg0'
        keys = 'sire_keys_qszg0'
        index = [0,1,2] #q,s,z
        cols = []
        d['sire'] = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     keys=keys, arith=arith, index=index, cols=cols)

        ###dams
        type = 'stock'
        prod = '{0}_animal_k2tva1nwziyg1'.format(e)
        na_prod = [0, 1]  # q,s
        weights = 'dams_numbers_qsk2tvanwziy1g1'
        keys = 'dams_keys_qsk2tvanwziy1g1'
        index = [0,1,8] #q,s,z
        cols = []
        d['dams'] = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     keys=keys, arith=arith, index=index, cols=cols)

        ###offs
        type = 'stock'
        prod = '{0}_animal_k3k5tvnwziaxyg3'.format(e)
        na_prod = [0, 1]  # q,s
        weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
        keys = 'offs_keys_qsk3k5tvnwziaxyg3'
        index = [0,1,8] #q,s,z
        cols = []
        d['offs'] = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     keys=keys, arith=arith, index=index, cols=cols)

        ###grn pasture
        type = 'pas'
        prod = 'stock_{0}_grnpas_qgop6lzt'.format(e)
        na_prod = [1,2]  # s,f
        weights = 'greenpas_ha_qsfgop6lzt'
        keys = 'keys_qsfgop6lzt'
        index = [0,1,7] #q,s,z
        cols = []
        d['grnpas'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols)

        ###poc pasture
        type = 'pas'
        prod = 'stock_{0}_poc_p6z'.format(e)
        na_prod = [0, 1, 2, 4]  # q,s,f,l
        weights = 'poc_consumed_qsfp6lz'
        keys = 'keys_qsfp6lz'
        index = [0,1,5] #[q,s,z]
        cols = []
        d['poc'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols)

        ###dry pasture
        type = 'pas'
        prod = 'stock_{0}_drypas_dp6zt'.format(e)
        na_prod = [0, 1, 2, 6]  # q,s,f,l
        weights = 'drypas_consumed_qsfdp6zlt'
        keys = 'keys_qsfdp6zlt'
        index = [0,1,5] #[q,s,z]
        cols = []
        d['drypas'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols)

        ###nap pasture
        type = 'pas'
        prod = 'stock_{0}_drypas_dp6zt'.format(e) #nap is same emissions as dry pasture
        na_prod = [0, 1, 2]  # q,s,f
        weights = 'nap_consumed_qsfdp6zt'
        keys = 'keys_qsfdp6zt'
        index = [0,1,5] #[q,s,z]
        cols = []
        d['nappas'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols)

        ###residue
        type = 'stub'
        prod = 'stock_{0}_stub_zp6ks1'.format(e)
        na_prod = [0, 1, 4, 7]  # q,s,p6,f,s2
        weights = 'stub_qszp6fks1s2'
        keys = 'keys_qszp6fks1s2'
        index = [0, 1, 2]  # q,s,z
        cols = []
        d['res'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                              keys=keys, arith=arith, index=index, cols=cols)

        ###crop graze
        type = 'crpgrz'
        prod = 'stock_{0}_cropgraze_kp6z'.format(e)
        na_prod = [0, 1,2,5,7]  # q,s,f,p5,l
        weights = 'crop_consumed_qsfkp6p5zl'
        keys = 'keys_qsfkp6p5zl'
        index = [0, 1, 6]  # q,s,z
        cols = []
        d['grncrop'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                              keys=keys, arith=arith, index=index, cols=cols)

        ###saltbush (just the saltbush not the understory)
        type = 'slp'
        prod = 'stock_{0}_sb_zp6'.format(e)
        na_prod = [0, 1, 4, 5]  # q,s,f,l
        weights = 'v_tonnes_sb_consumed_qszp6fl'
        keys = 'keys_qszp6fl'
        index = [0, 1, 2]  # q,s,z
        cols = []
        d['sb'] = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                              keys=keys, arith=arith, index=index, cols=cols)

        ###sup
        sup_emissions_fk3 = r_vals['sup']['stock_{0}_sup_fk3'.format(e)]
        grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)
        sup_emissions_qszp6_fk3 = grain_fed_qszkfp6.unstack([4,3]).sort_index(axis=1).mul(sup_emissions_fk3, axis=1)
        d['sup_emissions_qsz'] = pd.DataFrame(sup_emissions_qszp6_fk3.unstack([3]).sum(axis=1))

    ###total livestock emissions
    for i, emission_cat in enumerate(ch4):
        if i==0:
            livestock_ch4_qsz = ch4[emission_cat]
            livestock_n2o_qsz = n2o[emission_cat]
        else:
            livestock_ch4_qsz = livestock_ch4_qsz.add(ch4[emission_cat])
            livestock_n2o_qsz = livestock_n2o_qsz.add(n2o[emission_cat])
    ch4_liveco2e_qsz = livestock_ch4_qsz * ch4_gwp_factor / 1000 #convert to tonnes
    n2o_liveco2e_qsz = livestock_n2o_qsz * n2o_gwp_factor / 1000 #convert to tonnes
    total_liveco2e_qsz = ch4_liveco2e_qsz + n2o_liveco2e_qsz

    ##Emissions from crop residues
    ###n2o - residue production at harvest
    type = 'stub'
    prod = 'residue_harv_n2o_zk'
    na_prod = [0, 1, 2,5,6]  # q,s,p7,l,s2
    prod_weights = 'biomass2residue_ks2' #needed to convert v_biomass to residue
    na_prod_weights = [0, 1, 2, 3,5]  # q,s,p7,z,l
    weights = 'v_use_biomass_qsp7zkls2'
    keys = 'keys_qsp7zkls2'
    index = [0, 1, 3]  # q,s,z
    cols = []
    residue_harv_n2o_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, prod_weights=prod_weights, 
                                                   na_prodweights=na_prod_weights, type=type, weights=weights,
                                                   keys=keys, arith=arith, index=index, cols=cols)
    ###ch4 - residue production at harvest
    type = 'stub'
    prod = 'residue_harv_ch4_zk'
    na_prod = [0, 1, 2,5,6]  # q,s,p7,l,s2
    prod_weights = 'biomass2residue_ks2' #needed to convert v_biomass to residue
    na_prod_weights = [0, 1, 2, 3,5]  # q,s,p7,z,l
    weights = 'v_use_biomass_qsp7zkls2'
    keys = 'keys_qsp7zkls2'
    index = [0, 1, 3]  # q,s,z
    cols = []
    residue_harv_ch4_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, prod_weights=prod_weights, 
                                                   na_prodweights=na_prod_weights, type=type, weights=weights,
                                                   keys=keys, arith=arith, index=index, cols=cols)
    ###n2o - residue consumption - reduces emissions from residues
    type = 'stub'
    prod = 'residue_cons_n2o_zp6k'
    na_prod = [0, 1, 4, 6,7]  # q,s,f,s1,s2
    weights = 'stub_qszp6fks1s2'
    keys = 'keys_qszp6fks1s2'
    index = [0, 1, 2]  # q,s,z
    cols = []
    residue_cons_n2o_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                       keys=keys, arith=arith, index=index, cols=cols)
    ###ch4 - residue consumption - reduces emissions from residues
    type = 'stub'
    prod = 'residue_cons_ch4_zp6k'
    na_prod = [0, 1, 4, 6,7]  # q,s,f,s1,s2
    weights = 'stub_qszp6fks1s2'
    keys = 'keys_qszp6fks1s2'
    index = [0, 1, 2]  # q,s,z
    cols = []
    residue_cons_ch4_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                       keys=keys, arith=arith, index=index, cols=cols)

    ###total residue emissions
    ch4_residue_co2e_qsz = (residue_harv_ch4_qsz + residue_cons_ch4_qsz) * ch4_gwp_factor / 1000 #convert to tonnes
    n2o_residue_co2e_qsz = (residue_harv_n2o_qsz + residue_cons_n2o_qsz) * n2o_gwp_factor / 1000 #convert to tonnes
    total_residue_co2e_qsz = ch4_residue_co2e_qsz + n2o_residue_co2e_qsz

    ##Emissions from pasture residues (POC not included atm - see note in EmissionFunctions)
    ###germination and nap
    type = 'pas'
    prod = 'n2o_pas_residue_v_phase_growth_qp6zrlt'
    na_prod = [1, 2]  # q,s,p7
    weights = 'v_phase_change_increase_qsp7zrl'
    na_weights = [3,7] #p6,t
    keys = 'keys_qsp7p6zrlt'
    index = [0, 1, 4]  # q,s,z
    cols = []
    residue_n2o_vphase_growth = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###grn pasture
    type = 'pas'
    prod = 'grnpas_n2o_residue_qgop6lzt'
    na_prod = [1, 2]  # s,f
    weights = 'greenpas_ha_qsfgop6lzt'
    keys = 'keys_qsfgop6lzt'
    index = [0, 1, 7]  # q,s,z
    cols = []
    residue_n2o_grnpas = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###dry pasture
    type = 'pas'
    prod = 'pas_n2o_residue_cons_t'
    na_prod = [0,1,2,3,4,5] #q,s,f,d,p6,z
    weights = 'drypas_consumed_qsfdp6zlt'
    keys = 'keys_qsfdp6zlt'
    index = [0, 1, 5]  # [q,s,z]
    cols = []
    residue_n2o_drypas_cons = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###nap pasture
    type = 'pas'
    prod = 'pas_n2o_residue_cons_t'  # nap is same emissions as dry pasture
    na_prod = [0,1,2,3,4,5] #q,s,f,d,p6,z
    weights = 'nap_consumed_qsfdp6zt'
    keys = 'keys_qsfdp6zt'
    index = [0, 1, 5]  # [q,s,z]
    cols = []
    residue_n2o_nap_cons = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###total residue emissions
    n2o_pas_residue_co2e_qsz = (residue_n2o_vphase_growth + residue_n2o_grnpas + residue_n2o_drypas_cons + residue_n2o_nap_cons) * n2o_gwp_factor / 1000 #convert to tonnes

    ##Emissions from fuel use
    ###seeding
    type = 'mach'
    prod = 'co2e_seeding_fuel_l'
    na_prod = [0,1,2,3,4]  # q,s,z,p5,k
    prod_weights = 'seeding_rate'
    na_prodweights = [0,1,2,3] #q,s,z,p5
    weights = 'v_seeding_machdays'
    keys = 'keys_qszp5kl'
    index = [0, 1, 2]  # q,s,z
    cols = []
    fuel_co2e_seeding_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          prod_weights=prod_weights, na_prodweights=na_prodweights, keys=keys,
                                          arith=arith, index=index, cols=cols)
    ###contract seeding
    type = 'mach'
    prod = 'co2e_seeding_fuel_l'
    na_prod = [0,1,2,3,4]  # q,s,z,p5,k
    weights = 'v_contractseeding_ha'
    keys = 'keys_qszp5kl'
    index = [0, 1, 2]  # q,s,z
    cols = []
    fuel_co2e_contract_seeding_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###harvesting
    type = 'mach'
    prod = 'co2e_harv_fuel'
    na_prod = [0,1,2,3]  # q,s,z,p5 - prod already has a singlton k axis
    weights = 'v_harv_hours'
    keys = 'keys_qszp5k'
    index = [0, 1, 2]  # q,s,z
    cols = []
    fuel_co2e_harv_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###contract harvesting
    type = 'mach'
    prod = 'co2e_harv_fuel'
    na_prod = [0,1,2,3]  # q,s,z,p5 - prod already has a singlton k axis
    weights = 'v_contractharv_hours'
    keys = 'keys_qszp5k'
    index = [0, 1, 2]  # q,s,z
    cols = []
    fuel_co2e_contract_harv_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###spreading. spraying & stubble handling
    type = 'crop'
    prod = 'co2e_phase_fuel_zrl'
    na_prod = [0,1,2]  # q,s,p7
    weights = 'v_phase_change_increase_qsp7zrl'
    keys = 'keys_qsp7zrl'
    index = [0, 1, 3]  # q,s,z
    cols = []
    fuel_co2e_phase_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    ###sup
    fuel_sup_emissions_fk3 = r_vals['sup']['co2e_sup_fuel_fk3']
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)
    fuel_sup_emissions_qszp6_fk3 = grain_fed_qszkfp6.unstack([4, 3]).sort_index(axis=1).mul(fuel_sup_emissions_fk3, axis=1)
    fuel_co2e_sup_emissions_qsz = pd.DataFrame(fuel_sup_emissions_qszp6_fk3.unstack([3]).sum(axis=1))

    ##planting and maintaining trees
    type = 'tree'
    prod = 'tree_co2e_fuel'
    na_prod = [0]  #l
    weights = 'v_tree_area_l'
    keys = 'keys_l'
    index = []  #none
    cols = []
    fuel_co2e_tree_emissions = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols).squeeze()

    total_fuel_co2e_qsz = (fuel_co2e_seeding_qsz + fuel_co2e_contract_seeding_qsz + fuel_co2e_harv_qsz +
                           fuel_co2e_contract_harv_qsz + fuel_co2e_phase_qsz + fuel_co2e_sup_emissions_qsz +
                           fuel_co2e_tree_emissions)/ 1000 #convert to tonnes. Note it has already been converted to co2e.

    ##Fertiliser
    type = 'crop'
    prod = 'co2e_fert_zrl'.format(e)
    na_prod = [0,1,2]  # q,s,p7
    weights = 'v_phase_change_increase_qsp7zrl'
    keys = 'keys_qsp7zrl'
    index = [0, 1, 3]  # q,s,z
    cols = [4]
    fert_co2e_qsz_r = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    fert_co2e_qsz_r = fert_co2e_qsz_r/1000 #convert to tonnes
    ###convert r to k
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(phases_df.columns[-1], append=True)  # add landuse as index level
    fert_co2e_qsz_rk = fert_co2e_qsz_r.reindex(phases_rk.index, axis=1, level=0)
    fert_co2e_qsz_k = fert_co2e_qsz_rk.groupby(axis=1,level=1).sum()
    fert_co2e_qsz = fert_co2e_qsz_k.sum(axis=1).to_frame()
    fert_cols_k = [str("Fertiliser CO2e ")+i+str(" (t)") for i in fert_co2e_qsz_k.columns]

    ##trees
    type = 'tree'
    prod = 'tree_co2_sequestration_l'
    weights = 'v_tree_area_l'
    keys = 'keys_l'
    index = []  #none
    cols = []
    annual_sequestration = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols).squeeze()
    annual_sequestration = annual_sequestration/ 1000  # convert to tonnes

    ##carbon sold
    type = 'tree'
    prod = 'tree_co2e_sold_l'
    weights = 'v_tree_area_l'
    keys = 'keys_l'
    index = []  #none
    cols = []
    tree_co2e_sold = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols).squeeze()
    tree_co2e_sold = tree_co2e_sold/ 1000  # convert to tonnes
    ###add qsz axis
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_qsz = [keys_q, keys_s, keys_z]
    idx = pd.MultiIndex.from_product(keys_qsz)
    annual_sequestration_qsz = pd.Series(annual_sequestration, index=idx)
    tree_co2e_sold_qsz = pd.Series(tree_co2e_sold, index=idx)

    ##calc info for intensity calcs
    ###wool production
    ####sires
    type = 'stock'
    prod = 'cfw_hdmob_zg0'
    na_prod = [0, 1]  # q,s
    weights = 'sire_numbers_qszg0'
    keys = 'sire_keys_qszg0'
    index = [0, 1, 2]  # q,s,z
    cols = []
    cfw_sire = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                        keys=keys, arith=arith, index=index, cols=cols)
    ####dams
    type = 'stock'
    prod = 'cfw_hdmob_k2tva1nwziyg1'
    na_prod = [0, 1]  # q,s
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    index = [0, 1, 8]  # q,s,z
    cols = []
    cfw_dams = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                        keys=keys, arith=arith, index=index, cols=cols)
    ####offs
    type = 'stock'
    prod = 'cfw_hd_k3k5tvnwziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    index = [0, 1, 8]  # q,s,z
    cols = []
    cfw_offs = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                        keys=keys, arith=arith, index=index, cols=cols)
    ###total wool
    total_clean_wool_qsz = cfw_sire + cfw_dams + cfw_offs

    ##animal production
    ####dams
    type = 'stock'
    prod = 'sale_ffcfw_s7k2tva1nwziyg1'
    na_prod = [0, 1]  # q,s
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [2] #s7
    keys = 'dams_keys_qss7k2tvanwziy1g1'
    index = [0, 1, 9]  # q,s,z
    cols = [2] #s7
    ffcfw_sold_dams = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                        na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ####prog
    type = 'stock'
    prod = 'sale_ffcfw_s7k3k5twziaxyg2'
    na_prod = [0, 1]  # q,s
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    na_weights = [2] #s7
    keys = 'prog_keys_qss7k3k5twzia0xg2'
    index = [0, 1, 7]  # q,s,z
    cols = [2] #s7
    ffcfw_sold_prog = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                        na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ####offs
    type = 'stock'
    prod = 'sale_ffcfw_s7k3k5tvnwziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [2] #s7
    keys = 'offs_keys_qss7k3k5tvnwziaxyg3'
    index = [0, 1, 9]  # q,s,z
    cols = [2] #s7
    ffcfw_sold_offs = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###total meat sold
    total_meat_qsz_s7 = ffcfw_sold_prog + ffcfw_sold_dams + ffcfw_sold_offs
    meat_cols = [str("FFCFW Sold ")+i[:-3]+str("(kg)") for i in total_meat_qsz_s7.columns]

    ###crop (grain/hay) production
    type = 'crop'
    prod = 'biomass2product_ks2'
    na_prod = [0, 1, 2,3,5]  # q,s,p7,z,l
    weights = 'v_use_biomass_qsp7zkls2'
    keys = 'keys_qsp7zkls2'
    index = [0, 1, 3]  # q,s,z
    cols = []
    crop_production_qsz = f_stock_pasture_summary(r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                                   keys=keys, arith=arith, index=index, cols=cols)

    ##tally farm emissions
    total_farm_co2e_qsz = (total_liveco2e_qsz + total_residue_co2e_qsz + n2o_pas_residue_co2e_qsz + total_fuel_co2e_qsz + fert_co2e_qsz
                           - annual_sequestration + tree_co2e_sold) #if sequestered carbon is sold it can't offset a farms footprint



    ##make final df
    emissions_qsz = pd.concat([total_farm_co2e_qsz,
                               total_liveco2e_qsz, ch4_liveco2e_qsz, n2o_liveco2e_qsz,
                               total_residue_co2e_qsz, ch4_residue_co2e_qsz, n2o_residue_co2e_qsz,
                               n2o_pas_residue_co2e_qsz, n2o_pas_residue_co2e_qsz,
                               total_fuel_co2e_qsz,
                               fert_co2e_qsz, fert_co2e_qsz_k,
                               annual_sequestration_qsz,
                               tree_co2e_sold_qsz,
                               crop_production_qsz,
                               total_clean_wool_qsz, total_meat_qsz_s7], axis=1)
    emissions_qsz.columns = ['Total Farm CO2e (t)',
                             'Total Livestock CO2e (t)', 'Livestock Methane co2e (t)', 'Livestock Nitrous Oxide co2e (t)',
                             'Total Crop Residue CO2e (t)', 'Crop Residue Methane co2e (t)', 'Crop Residue Nitrous Oxide co2e (t)',
                             'Total Pas Residue CO2e (t)', 'Pasture Residue Nitrous Oxide co2e (t)',
                             'Total Fuel CO2e (t)',
                             'Total Fertiliser CO2e (t)'] + fert_cols_k + [
                             'Total Sequestered CO2e (t)',
                             'Total Sold CO2e (t)',
                             'Crop yield (t)',
                             'Clean Wool Sold (kg)']+meat_cols
    ##weighted average of z axis if required.
    if option==1:
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        emissions_qsz = pd.DataFrame(emissions_qsz.mul(z_prob_qsz, axis=0).sum(axis=0)).T
    return emissions_qsz.round(1)


def f_grazing_summary(lp_vars, r_vals):
    '''
    Green pasture grazing summary
    :return: data table with zp6lo as index and g and column. Containing the hectares of each activity, average foo and starting foo.
    '''
    feed_vars = d_vars['base'] #z is reported so use base version
    greenpas_ha_qsfgop6lzt = feed_vars['greenpas_ha_qsfgop6lzt']
    foo_ave_grnha_qgop6lzt = r_vals['pas']['foo_ave_grnha_qgop6lzt']
    foo_start_grnha_qop6lzt = r_vals['pas']['foo_start_grnha_qop6lzt']
    foo_gi_gt = r_vals['pas']['i_foo_graze_propn_gt']

    ##calc average foo for f (nv pool), o (start foo) and g (grazing int)
    foo_ave_grnha_qsp6lzt = fun.f_weighted_average(foo_ave_grnha_qgop6lzt[:,na,na,...], greenpas_ha_qsfgop6lzt, axis=(2,3,4))

    ##sum f axis to return total ha of each pasture activity
    greenpas_ha_qsgop6lzt = np.sum(greenpas_ha_qsfgop6lzt, axis=2)

    ##combine everything
    graze_info_iqsgop6lzt = np.stack(np.broadcast_arrays(greenpas_ha_qsgop6lzt, foo_start_grnha_qop6lzt[:,na,na,...], foo_ave_grnha_qsp6lzt[:,:,na,na,...]), axis=0)

    ##make df
    keys_g = r_vals['pas']['keys_g']
    keys_l = r_vals['pas']['keys_l']
    keys_o = r_vals['pas']['keys_o']
    keys_p6 = r_vals['pas']['keys_p6']
    keys_t = r_vals['pas']['keys_t']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_i = ["area", "foo start", "ave_foo"]
    keys_iqsgop6lzt = [keys_i, keys_q, keys_s, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]

    graze_info_qsztlp6o_ig = f_numpy2df(graze_info_iqsgop6lzt, keys_iqsgop6lzt, [1,2,7,8,6,5,4], [0,3])
    return graze_info_qsztlp6o_ig

############################
# reports for web app      #
############################
def f_stock_numbers_summary(r_vals):
    '''Returns one reconciliation table per trial (season results are averaged)'''
    ##prog numbers sold
    type = 'stock'
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 2
    index = [10]  # g
    cols = [9]  # gender
    axis_slice = {4:[0,1,1]} #sale suckers
    numbers_prog_g_x = f_stock_pasture_summary(r_vals, type=type, weights=weights, keys=keys, arith=arith, index=index, cols=cols,
                                                           axis_slice=axis_slice)
    ###female prog sold
    try:
        female_prog_sold = numbers_prog_g_x.loc[['BBB','BBM'],'F'] #wrapped in try incase BBM are not included in the trial. Note BBT are added with wethers.
    except KeyError:
        female_prog_sold = numbers_prog_g_x.loc['BBB', 'F']
    ###wether & crossy prog sold
    wether_prog_sold = numbers_prog_g_x.values.sum() - female_prog_sold

    ##dam numbers open
    type = 'stock'
    prod = 'dvp_is_mating_or_weaning_yvzig1'
    na_prod = [0,1,2,3,6,7,8,11]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [4] #y (year)
    keys = 'dams_keys_qsk2tyvanwziy1g1'
    arith = 2
    index = [4] #y
    cols = [3,5] #v,t
    open_numbers_dams_y_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###dam open numbers at prejoining (sum v & t to get total start numbers for each year)
    open_numbers_dams_y = open_numbers_dams_y_tv.sum(axis=1)
    #count wethers and crossy prog that were born (this get used in its own column), then set open numbers to 0 since there is no open lambs.
    females_born = open_numbers_dams_y.iloc[0] + female_prog_sold #need to include any suckers that are sold because not captured in offs numbers.
    open_numbers_dams_y.iloc[0] = 0 #

    ##dam numbers sale
    type = 'stock'
    prod = 'dvp_is_sale_tyvzig1'
    na_prod = [0,1,2,6,7,8,11]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [4] #y (year)
    keys = 'dams_keys_qsk2tyvanwziy1g1'
    arith = 2
    index = [4] #y
    cols = [3,5] #v,t
    sale_numbers_dams_y_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###dams sold each year
    sale_numbers_dams_y = sale_numbers_dams_y_tv.sum(axis=1)
    #add female prog that were sold
    sale_numbers_dams_y.iloc[0] = female_prog_sold
    
    ###concat open, birth and sale
    numbers_dams = pd.DataFrame()
    numbers_dams.insert(0, "Open Numbers", open_numbers_dams_y)
    numbers_dams.insert(1, "Births", 0) #add empty col
    numbers_dams.iloc[0,1] = females_born
    numbers_dams.insert(2, "Sales", sale_numbers_dams_y)

    ##open offs numbers
    type = 'stock'
    prod = 'dvp_is_shear_or_weaning_yvzixg3'
    na_prod = [0,1,2,3,4,7,8,11,13]
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [5] #y (year)
    keys = 'offs_keys_qsk3k5tyvnwziaxyg3'
    arith = 2
    index = [5] #y
    cols = [4,6] #v,t
    numbers_offs_y_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###offs open numbers at prejoining (sum v & t to get total start numbers for each year)
    open_numbers_offs_y = numbers_offs_y_tv.sum(axis=1)
    #count wethers and crossy prog that were born (this get used in its own column), then set open numbers to 0 since there is no open lambs.
    wethers_born = open_numbers_offs_y.iloc[0] + wether_prog_sold #need to include any suckers that are sold becuase not capture in offs numbers.
    open_numbers_offs_y.iloc[0] = 0 

    ##sale offs numbers
    type = 'stock'
    prod = 'dvp_is_sale_tyvzixg3'
    na_prod = [0,1,2,3,7,8,11,13]
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [5] #y (year)
    keys = 'offs_keys_qsk3k5tyvnwziaxyg3'
    arith = 2
    index = [5] #y
    cols = [4,6] #v,t
    sale_numbers_offs_y_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ###age at sale
    type = 'stock'
    prod = 'saleage_k3k5tvnwziaxyg3'
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = []
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 1
    index = [4,5] #tv
    cols = []  #
    saleage_offs_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, weights=weights,na_weights=na_weights,
                                              keys=keys, arith=arith, index=index, cols=cols)
    ###add sale age as headers
    sale_numbers_offs_y_tv.columns = np.round(saleage_offs_tv.values.squeeze() / 30, 0) #div 30 to convert to months
    sale_numbers_offs_y_tv = sale_numbers_offs_y_tv.sort_index(axis=1)
    ####add wether and crossy prog that were sold (they need to be included in the number of lambs born)
    sale_numbers_offs_y_tv.insert(0, "Weaning",0)
    sale_numbers_offs_y_tv.iloc[0,0] = wether_prog_sold
    ###sum cols with same sale age (to stop error when concat the report with other trials because of duplicate col names)
    sale_numbers_offs_y_tv = sale_numbers_offs_y_tv.groupby(sale_numbers_offs_y_tv.columns, axis=1).sum()
    
    ###concat open, birth and sale
    numbers_offs = pd.concat([sale_numbers_offs_y_tv], keys=['Sales (months of age)'], axis=1)
    numbers_offs.insert(0, "Open Numbers", open_numbers_offs_y)
    numbers_offs.insert(1, "Births", 0) #add empty col
    numbers_offs.iloc[0,1] = wethers_born

    return numbers_dams.round(0), numbers_offs.round(0)

def f_pasture_area_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'Pas area', 'Ewes mated', 'Pas %', 'Cereal %', 'Canola %', 'Pulse %', 'Fodder %', 'Sup'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##pasture area
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    summary_df.loc[trial, 'Pas area'] = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    ##total dams mated
    type = 'stock'
    prod = 'dvp_is_mating_vzig1'
    na_prod = [0,1,2,3,5,6,7,10]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    arith = 2
    index = []
    cols = []
    axis_slice = {2: [1, None, 1], 3: [2, None, 1]}  # slice off the not mate k1 slice (we only want mated dams) and slice off the sold animals so we dont count dams that are sold at prejoining (there is a sale opp at the start of dvp).
    dams_mated = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc[trial, 'Ewes mated'] = dams_mated.squeeze()
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    ##cereal %
    summary_df.loc[trial, 'Cereal %'] = f_area_summary(lp_vars, r_vals, option=6)[0]
    ##canola %
    summary_df.loc[trial, 'Canola %'] = f_area_summary(lp_vars, r_vals, option=7)[0]
    ##pulse %
    summary_df.loc[trial, 'Pulse %'] = f_area_summary(lp_vars, r_vals, option=8)[0]
    ##fodder %
    summary_df.loc[trial, 'Fodder %'] = f_area_summary(lp_vars, r_vals, option=9)[0]
    ##supplement
    summary_df.loc[trial, 'Sup'] = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    return summary_df

def f_stocking_rate_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'SR', 'Ewes mated', 'Pas area', 'Sup/DSE'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##total dams mated
    type = 'stock'
    prod = 'dvp_is_mating_vzig1'
    na_prod = [0,1,2,3,5,6,7,10]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    arith = 2
    index = []
    cols = []
    axis_slice = {2: [1, None, 1], 3: [2, None, 1]}  # slice off the not mate k1 slice (we only want mated dams) and slice off the sold animals so we dont count dams that are sold at prejoining (there is a sale opp at the start of dvp).
    dams_mated = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc[trial, 'Ewes mated'] = round(dams_mated.squeeze(),0)
    ##pasture area
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    summary_df.loc[trial, 'Pas area'] = total_pas_are
    ##supplement
    total_sup = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    summary_df.loc[trial, 'Sup/DSE'] = round(fun.f_divide_float(total_sup * 1000, (total_pas_are * sr)))
    return summary_df

def f_lupin_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'Legume Area', 'Cereal %', 'Canola %', 'Pas %'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##pulse %
    total_legume_area = f_area_summary(lp_vars, r_vals, option=8)[0]/100 * r_vals['rot']['total_farm_area']
    summary_df.loc[trial, 'Legume Area'] = total_legume_area
    ##cereal %
    summary_df.loc[trial, 'Cereal %'] = f_area_summary(lp_vars, r_vals, option=6)[0]
    ##canola %
    summary_df.loc[trial, 'Canola %'] = f_area_summary(lp_vars, r_vals, option=7)[0]
    ##pasture %
    summary_df.loc[trial, 'Pasture %'] = f_area_summary(lp_vars, r_vals, option=5)[0]

    legume_names = ["Faba Bean", "Lentils", "Chickpea", "Lupins", "Vetch"]
    legume_keys = ['f','i', 'k', 'l', 'v']
    for legume_name, legume_key in zip(legume_names, legume_keys):
        if legume_key in r_vals['rot']['all_crops']:
            ##legume area
            landuse_area_qsz_k = f_area_summary(lp_vars, r_vals, option=4)
            legume_area_qsz = landuse_area_qsz_k.loc[:,legume_key]
            z_prob_qsz = r_vals['zgen']['z_prob_qsz']
            legume_area = np.sum(legume_area_qsz * z_prob_qsz.ravel())
            summary_df.loc[trial, '{0} Area'.format(legume_name)] = fun.f_divide(legume_area, total_legume_area)
            ##expected legume income - uses average legume yield in all rotations on the base lmu (this matches the yield graph on web app)
            legume_price_qp7z = r_vals['crop']['grain_price'].loc[(legume_key,"Harv","firsts"),:]
            legume_price_qz = legume_price_qp7z.groupby(level=(0,2)).sum() #sum p7 - price should only exist in one p7 period
            legume_price = np.sum(legume_price_qz.unstack().values[:,na,:] * z_prob_qsz) #avevrage price across q and z
            expected_yields_k_z = r_vals['crop']['base_yields_k_z']
            expected_yields_k_z = expected_yields_k_z.reindex(r_vals['pas']['keys_k'], axis=0).fillna(0)  # expand to full k (incase landuses were masked out) and unused landuses get set to 0
            expected_legume_yield_z = expected_yields_k_z.loc[legume_key,:]
            expected_legume_yield = np.sum(expected_legume_yield_z.values * z_prob_qsz) #avevrage yield across z.
            summary_df.loc[trial, 'Expected {0} Income'.format(legume_name)] = round(legume_price * expected_legume_yield/1000, 0)
        else: #if crop is not included set to None
            summary_df.loc[trial, '{0} Area'.format(legume_name)] = ""
            summary_df.loc[trial, 'Expected {0} Income'.format(legume_name)] = ""

    return summary_df

def f_cropgrazing_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'SR', 'Pas area (%)', 'Crop area (%)', 'Sup/DSE', 'Crop grazing intensity (kg/ha)'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0))
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##pasture %
    pas_area_percent = f_area_summary(lp_vars, r_vals, option=5)[0]
    summary_df.loc[trial, 'Pas area (%)'] = round(pas_area_percent)
    ##crop %
    summary_df.loc[trial, 'Crop area (%)'] = round(100 - pas_area_percent)
    ##supplement
    total_sup = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    summary_df.loc[trial, 'Sup/DSE'] = round(fun.f_divide_float(total_sup * 1000, (total_pas_are * sr)))
    ##crop grazing intensity
    keys_k = r_vals['pas']['keys_k']
    keys_k1 = r_vals['stub']['keys_k1']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_l = r_vals['pas']['keys_l']
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)
    len_l = len(keys_l)
    len_k1 = len(keys_k1)
    ###get landuse area with crop landuse (k1) axis
    landuse_area_qszl_k = f_area_summary(lp_vars, r_vals, option=11)
    is_crop = np.any(keys_k==keys_k1[:,na], axis=0)
    landuse_area_qszl_k1 = landuse_area_qszl_k.loc[:,is_crop]
    landuse_area_qszlk1 = landuse_area_qszl_k1.stack().values.reshape(len_q, len_s, len_z, len_l, len_k1)
    ###total crop consumed (kgs)
    crop_consumed_qsz = np.sum(d_vars['base']['crop_consumed_qsfkp6p5zl'], axis=(2,3,4,5,7)) * 1000
    ###kilograms of crop consumed per hectare of crop that could have been grazed.
    GI_qsz = fun.f_divide(crop_consumed_qsz, np.sum(landuse_area_qszlk1 * r_vals['crpgrz']['propn_area_grazable_k1l'].T, axis=(-1,-2)))
    ###weight z
    GI = np.sum(GI_qsz * r_vals['zgen']['z_prob_qsz'])
    summary_df.loc[trial, 'Crop grazing intensity (kg/ha)'] = round(GI)

    return summary_df

def f_saleage_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'SR', 'Ewes mated', 'Sale weight', 'Sale value', 'Sheep sales ($/WgHa)', 'Wool sales ($/WgHa)', 'Feed cost ($/WgHa)'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0))
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##total dams mated
    type = 'stock'
    prod = 'dvp_is_mating_vzig1'
    na_prod = [0,1,2,3,5,6,7,10]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    arith = 2
    index = []
    cols = []
    axis_slice = {2: [1, None, 1], 3: [2, None, 1]}  # slice off the not mate k1 slice (we only want mated dams) and slice off the sold animals so we dont count dams that are sold at prejoining (there is a sale opp at the start of dvp).
    dams_mated = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc[trial, 'Ewes mated'] = round(dams_mated.squeeze(),0)
    ##ave wether sale price
    ###offs
    type = 'stock'
    prod = 'salevalue_p7qk3k5tvnwziaxyg3'
    na_prod = [2]  # s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [0]  # p7
    den_weights = 'alloc_p7k3vzixg3' #this is required to add p7 axis to numbers (otherwise there are numbers in all p7 for a given v)
    na_denweights = [1,2,4,5,7,8,11,13]  # q,s,k5,t,n.w,a,y
    keys = 'offs_keys_p7qsk3k5tvnwziaxyg3'
    arith = 1
    index = []
    cols = []
    axis_slice = {5:[1,None,1]} #only sale slices
    ave_salevalue_offs = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights,
                                                 keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ###prog
    type = 'stock'
    prod = 'salevalue_p7qk3k5twzia0xg2'
    na_prod = [2]  # s
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    na_weights = [0]  # p7
    den_weights = 'wean_alloc_p7k3zg2'  # this is required to add p7 axis to numbers (otherwise there are numbers in all p7 for a given v)
    na_denweights = [1, 2, 4, 5, 6, 8, 9, 10]  # q,s,k5,t,w,z,i,a0,x
    keys = 'prog_keys_p7qsk3k5twzia0xg2'
    arith = 1
    index = []
    cols = []
    axis_slice = {5:[0,1,1]} #only sale slices
    ave_salevalue_prog = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights,
                                                 keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ##ave wether sale weight
    ###offs
    type = 'stock'
    prod = 'sale_ffcfw_k3k5tvnwziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = []
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 1
    index = []
    cols = []
    axis_slice = {4:[1,None,1]} #only sale slices
    ave_saleweight_offs = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ###prog
    type = 'stock'
    prod = 'sale_ffcfw_k3k5twziaxyg2'
    na_prod = [0, 1]  # q,s
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    na_weights = []
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 1
    index = []
    cols = []
    axis_slice = {4:[0,1,1]} #only sale slices
    ave_saleweight_prog = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ###calc average sale value and weight of offs and prog
    ####get offs and prog sale numbers for weighted average
    type = 'stock'
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 2
    index = []
    cols = []
    axis_slice = {4:[1,None,1]} #only sale slices
    salenumber_offs = f_stock_pasture_summary(r_vals, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    type = 'stock'
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 2
    index = []
    cols = []
    axis_slice = {4:[0,1,1]} #only sale slices
    salenumber_prog = f_stock_pasture_summary(r_vals, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc[trial, 'Sale value'] = np.round(fun.f_divide(ave_salevalue_offs*salenumber_offs + ave_salevalue_prog*salenumber_prog, salenumber_offs + salenumber_prog),0)[0,0]
    summary_df.loc[trial, 'Sale weight'] = np.round(fun.f_divide(ave_saleweight_offs*salenumber_offs + ave_saleweight_prog*salenumber_prog, salenumber_offs + salenumber_prog),0)[0,0]

    ##pasture area
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    ##sheep $$
    stocksale_qszp7, wool_qszp7, husbcost_qszp7, supcost_qsz_p7, purchasecost_qszp7, trade_value_qszp7 = f_stock_cash_summary(lp_vars, r_vals)
    ##sale income
    stocksale = np.sum(stocksale_qszp7 * z_prob_qsz[...,na])
    stocksale_per_wgha = stocksale / total_pas_are if total_pas_are != 0 else 0
    summary_df.loc[trial, 'Sheep sales ($/WgHa)'] = round(stocksale_per_wgha, 0)
    ##wool income
    woolsale = np.sum(wool_qszp7 * z_prob_qsz[...,na])
    woolsale_per_wgha = woolsale / total_pas_are if total_pas_are != 0 else 0
    summary_df.loc[trial, 'Wool sales ($/WgHa)'] = round(woolsale_per_wgha, 0)
    ##feed cost
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
    z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
    feedcost_qsz = supcost_qsz_p7.sum(axis=1)
    feedcost = feedcost_qsz.mul(z_prob_qsz, axis=0).sum(axis=0)
    feedcost_per_wgha = feedcost / total_pas_are if total_pas_are != 0 else 0
    summary_df.loc[trial, 'Feed cost ($/WgHa)'] = round(feedcost_per_wgha, 0)

    return summary_df

def f_slp_area_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'SLP area', 'SR', 'Pas %', 'Sup/DSE'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##slp area
    slp_area = np.sum(d_vars['qsz_weighted']['v_slp_ha_qszl'])
    summary_df.loc[trial, 'SLP area'] = round(slp_area,0)
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    ##supplement
    total_sup = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    summary_df.loc[trial, 'Sup/DSE'] = round(fun.f_divide_float(total_sup * 1000, (total_pas_are * sr)))
    return summary_df


def f_fodder_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'Cereal fodder area', 'Legume fodder area', 'SR', 'Pas %', 'Sup/DSE'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##fodder area
    landuse_area_k = f_area_summary(lp_vars, r_vals, option=4, active_z=False).squeeze()
    summary_df.loc[trial, 'Cereal fodder area'] = round(fun.f1_get_value(landuse_area_k, "of"),0)
    summary_df.loc[trial, 'Legume fodder area'] = round(fun.f1_get_value(landuse_area_k, "lf"),0)
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    ##supplement
    total_sup = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    summary_df.loc[trial, 'Sup/DSE'] = round(fun.f_divide_float(total_sup * 1000, (total_pas_are * sr)))
    return summary_df


def f_perennial_analysis(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['Profit', 'Perennial pas area', 'Total Pas %', 'SR', 'Sup/DSE'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'Profit'] = round(f_profit(lp_vars, r_vals, option=0),0)
    ##perennial area
    perennial_pas_landuses = r_vals['rot']['perennial_pas']  # landuse sets
    landuse_area_k = f_area_summary(lp_vars, r_vals, option=4, active_z=False).squeeze()
    summary_df.loc[trial, 'Perennial pas area'] = round(fun.f1_get_value(landuse_area_k, perennial_pas_landuses), 0)
    ##pasture %
    summary_df.loc[trial, 'Total Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    ##stocking rate
    sr = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    summary_df.loc[trial, 'SR'] = round(sr, 1)
    ##supplement
    total_sup = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    z_prob_qsz = r_vals['zgen']['z_prob_qsz']
    total_pas_are = np.sum(pas_area_qsz * z_prob_qsz.ravel())
    summary_df.loc[trial, 'Sup/DSE'] = round(fun.f_divide_float(total_sup * 1000, (total_pas_are * sr)))
    return summary_df


def mp_report(lp_vars, r_vals, option=1):
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
    summary_df = pd.DataFrame(index=[], columns=index_qsz)

    ##ewe sale info - this is done first because the summary table uses some info from these calcs
    ###prog numbers sold
    type = 'stock'
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 2
    index = [4,10,9]  # g, gender
    cols = [0,1,6]  # q,s,z
    numbers_prog_tgx_qsz = f_stock_pasture_summary(r_vals, type=type, weights=weights, keys=keys, arith=arith, index=index, cols=cols)

    ####total prog weaned
    numbers_prog_weaned_qsz = numbers_prog_tgx_qsz.sum(axis=0)
    ####total prog sold
    numbers_prog_sold_qsz = numbers_prog_tgx_qsz.loc['t0', :].sum(axis=0)

    ####female prog weaned
    try:#wrapped in try incase BBM are not included in the trial. Note BBT are added with wethers.
        female_prog_t_qsz = numbers_prog_tgx_qsz.loc[(slice(None),['BBB','BBM'],'F'),:].groupby(axis=0, level=0).sum()
    except KeyError:
        female_prog_t_qsz = numbers_prog_tgx_qsz.loc[(slice(None),['BBB'],'F'),:].groupby(axis=0, level=0).sum()

    ####prog sold
    female_prog_sold_qsz = female_prog_t_qsz.loc['t0', :]
    wether_prog_sold_qsz = numbers_prog_sold_qsz - female_prog_sold_qsz

    ###dam numbers sale
    type = 'stock'
    prod = 'dvp_is_sale_tyvzig1'
    na_prod = [0,1,2,6,7,8,11]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [4] #y (year)
    keys = 'dams_keys_qsk2tyvanwziy1g1'
    arith = 2
    index = [0,1,9,4] #q,s,z,y
    cols = [3,5] #v,t
    sale_numbers_dams_qszy_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                               na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols)
    ####dams sold each year
    sale_numbers_dams_qszy = sale_numbers_dams_qszy_tv.sum(axis=1)
    sale_numbers_dams_y_qsz = sale_numbers_dams_qszy.unstack().T
    sale_numbers_dams_y_qsz = sale_numbers_dams_y_qsz.reindex(sale_numbers_dams_qszy.index.unique(-1)) #put "lambs" back at the top of the y axis.
    ####add female prog that were sold
    sale_numbers_dams_y_qsz.iloc[0] = female_prog_sold_qsz
    sale_numbers_dams_y_qsz = round(sale_numbers_dams_y_qsz, 0)

    ##sale offs numbers
    type = 'stock'
    prod = 'dvp_is_sale_tyvzixg3'
    na_prod = [0, 1, 2, 3, 7, 8, 11, 13]
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [5]  # y (year)
    keys = 'offs_keys_qsk3k5tyvnwziaxyg3'
    arith = 2
    index = [0,1,9]  # q,s,z
    cols = [4, 6]  # v,t
    sale_numbers_offs_qsz_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     na_weights=na_weights, keys=keys, arith=arith, index=index,
                                                     cols=cols)
    ###age at sale
    type = 'stock'
    prod = 'saleage_k3k5tvnwziaxyg3'
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = []
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 1
    index = [4, 5]  # tv
    cols = []  #
    saleage_offs_tv = f_stock_pasture_summary(r_vals, type=type, prod=prod, weights=weights, na_weights=na_weights,
                                              keys=keys, arith=arith, index=index, cols=cols)
    ###add sale age as headers
    sale_numbers_offs_qsz_tv.columns = np.round(saleage_offs_tv.values.squeeze() / 30, 0).astype(int)  # div 30 to convert to months
    ###sum cols with same sale age (to stop error when concat the report with other trials because of duplicate col names)
    sale_numbers_offs_qsz_tv = sale_numbers_offs_qsz_tv.groupby(sale_numbers_offs_qsz_tv.columns, axis=1).sum()
    sale_numbers_offs_qsz_tv.columns = ['%s mo old' %i for i in sale_numbers_offs_qsz_tv.columns] #add extra info to header name
    ####add wether and crossy prog that were sold (they need to be included in the number of lambs born)
    sale_numbers_offs_qsz_tv.rename(columns={'0 mo old': 'Weaning'}, inplace=True)
    sale_numbers_offs_qsz_tv.iloc[:, 0] = wether_prog_sold_qsz
    sale_numbers_offs_tv_qsz = round(sale_numbers_offs_qsz_tv).T

    ##summary info by q
    ###profit - no minroe and asset
    profit_qsz = round(f_profit(lp_vars, r_vals, option=4)).stack()
    summary_df.loc['Profit',:] = profit_qsz
    ###pasture %
    pas_area_qsz = f_area_summary(lp_vars, r_vals, option=1)
    pas_percent_qsz = fun.f_divide(pas_area_qsz, r_vals['rot']['total_farm_area']) * 100
    summary_df.loc['Pas area (%)',:] = np.round(pas_percent_qsz, 0)
    ###crop %
    summary_df.loc['Crop area (%)',:] = np.round(100 - pas_percent_qsz)
    ###stocking rate
    sr_qsz = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary3=True)
    summary_df.loc['Stocking rate (DSE/WgHa)',:] = round(sr_qsz.squeeze(), 1)
    ###supplement
    total_sup_qsz = f_grain_sup_summary(lp_vars, r_vals, option=1).groupby(level=(0, 1, 2)).sum()
    summary_df.loc['Total supplement (t)',:] = round(total_sup_qsz.squeeze(), 1)
    ###sup/dse
    Sup_DSE_qsz = np.round(fun.f_divide(total_sup_qsz.squeeze() * 1000, (pas_area_qsz * sr_qsz.squeeze())))
    summary_df.loc['Supplement (kg/DSE)',:] = Sup_DSE_qsz
    ##propn fodder
    v_use_biomass_qsp7zkls2 = d_vars['base']['v_use_biomass_qsp7zkls2']  # use base vars because z is being reported
    v_use_biomass_qszs2 = v_use_biomass_qsp7zkls2.sum(axis=(2,4,5))
    total_biomass_qsz = v_use_biomass_qszs2.sum(axis=-1)
    graz_idx = list(r_vals['stub']['keys_s2']).index("Graz")
    biomass_fodder_qsz = v_use_biomass_qszs2[:,:,:,graz_idx]
    fodder_percent_qsz = fun.f_divide(biomass_fodder_qsz, total_biomass_qsz) * 100
    summary_df.loc['Fodder (%)',:] = fodder_percent_qsz.ravel().round()
    ##crop grazing
    prod = np.array([1])
    type = 'crpgrz'
    weights = 'crop_consumed_qsfkp6p5zl'
    keys = 'keys_qsfkp6p5zl'
    arith = 2
    index = []
    cols = [0, 1, 6]  # q,s,z
    cropgrazed_qsz = f_stock_pasture_summary(r_vals, prod=prod, type=type, weights=weights,
                                                      keys=keys, arith=arith, index=index, cols=cols)
    summary_df.loc['Grn Crop (t)', :] = round(cropgrazed_qsz.squeeze(),0)
    ###total dams mated
    type = 'stock'
    prod = 'dvp_is_mating_vzig1'
    na_prod = [0,1,2,3,5,6,7,10]
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    keys = 'dams_keys_qsk2tvanwziy1g1'
    arith = 2
    index = []
    cols = [0,1,8] #q,s,z
    axis_slice = {2: [1, None, 1], 3: [2, None, 1]}  # slice off the not mate k1 slice (we only want mated dams) and slice off the sold animals so we dont count dams that are sold at prejoining (there is a sale opp at the start of dvp).
    dams_mated_qsz = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc['Ewes mated',:] = round(dams_mated_qsz.squeeze(),0)
    ###prog weaned
    summary_df.loc['Lambs weaned',:] = round(numbers_prog_weaned_qsz,0)
    ###total ewe sales
    summary_df.loc['Ewe sales',:] = sale_numbers_dams_y_qsz.sum(axis=0)
    ###total wether sales
    summary_df.loc['Wether and crossy sales',:] = round(sale_numbers_offs_qsz_tv.sum(axis=1), 0)
    ###ave wether sale price
    ####get offs and prog sale numbers for weighted average
    type = 'stock'
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 2
    index = []
    cols = [0,1,8]  #q,s,z
    axis_slice = {4:[1,None,1]} #only sale slices
    salenumber_offs_qsz = f_stock_pasture_summary(r_vals, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    type = 'stock'
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 2
    index = []
    cols = [0,1,6]  #q,s,z
    axis_slice = {4:[0,1,1]} #only sale slices
    salenumber_prog_qsz = f_stock_pasture_summary(r_vals, type=type, weights=weights,
                                             keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ####offs
    type = 'stock'
    prod = 'salevalue_p7qk3k5tvnwziaxyg3'
    na_prod = [2]  # s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [0]  # p7
    den_weights = 'alloc_p7k3vzixg3' #this is required to add p7 axis to numbers (otherwise there are numbers in all p7 for a given v)
    na_denweights = [1,2,4,5,7,8,11,13]  # q,s,k5,t,n.w,a,y
    keys = 'offs_keys_p7qsk3k5tvnwziaxyg3'
    arith = 1
    index = []
    cols = [1,2,9]  #q,s,z
    axis_slice = {5:[1,None,1]} #only sale slices
    ave_salevalue_offs_qsz = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights,
                                                     keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ####prog
    type = 'stock'
    prod = 'salevalue_p7qk3k5twzia0xg2'
    na_prod = [2]  # s
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    na_weights = [0]  # p7
    den_weights = 'wean_alloc_p7k3zg2'  # this is required to add p7 axis to numbers (otherwise there are numbers in all p7 for a given v)
    na_denweights = [1, 2, 4, 5, 6, 8, 9, 10]  # q,s,k5,t,w,i,a0,x
    keys = 'prog_keys_p7qsk3k5twzia0xg2'
    arith = 1
    index = []
    cols = [1,2,7] #q,s,z
    axis_slice = {5:[0,1,1]} #only sale slices
    ave_salevalue_prog_qsz = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                     na_weights=na_weights, den_weights=den_weights, na_denweights=na_denweights,
                                                     keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc['Ave sale value'] = np.round(fun.f_divide(ave_salevalue_offs_qsz*salenumber_offs_qsz + ave_salevalue_prog_qsz*salenumber_prog_qsz, salenumber_offs_qsz + salenumber_prog_qsz),0).squeeze()
    ###ave wether sale weight
    ####offs
    type = 'stock'
    prod = 'sale_ffcfw_k3k5tvnwziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = []
    keys = 'offs_keys_qsk3k5tvnwziaxyg3'
    arith = 1
    index = []
    cols = [0,1,8] #qsz
    axis_slice = {4:[1,None,1]} #only sale slices
    ave_saleweight_offs_qsz = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    ####prog
    type = 'stock'
    prod = 'sale_ffcfw_k3k5twziaxyg2'
    na_prod = [0, 1]  # q,s
    weights = 'prog_numbers_qsk3k5twzia0xg2'
    na_weights = []
    keys = 'prog_keys_qsk3k5twzia0xg2'
    arith = 1
    index = []
    cols = [0,1,6] #qsz
    axis_slice = {4:[0,1,1]} #only sale slices
    ave_saleweight_prog_qsz = f_stock_pasture_summary(r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                       na_weights=na_weights, keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    summary_df.loc['Ave sale weight'] = np.round(fun.f_divide(ave_saleweight_offs_qsz*salenumber_offs_qsz + ave_saleweight_prog_qsz*salenumber_prog_qsz, salenumber_offs_qsz + salenumber_prog_qsz),0).squeeze()

    ##land use area
    landuse_area_k_qsz = f_area_summary(lp_vars, r_vals, option=4, active_z=True).T

    ##weight qsz if required (this is used for testing AFO)
    if option == 2:
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        summary_df = pd.DataFrame(summary_df.mul(z_prob_qsz, axis=1).sum(axis=1))
        landuse_area_k_qsz = pd.DataFrame(landuse_area_k_qsz.mul(z_prob_qsz, axis=1).sum(axis=1))
        sale_numbers_offs_tv_qsz = pd.DataFrame(sale_numbers_offs_tv_qsz.mul(z_prob_qsz, axis=1).sum(axis=1))
        sale_numbers_dams_y_qsz = pd.DataFrame(sale_numbers_dams_y_qsz.mul(z_prob_qsz, axis=1).sum(axis=1))

    return summary_df, landuse_area_k_qsz, sale_numbers_dams_y_qsz, sale_numbers_offs_tv_qsz

############################
# functions for numpy arrays#
############################

def f_numpy2df_error(prod, weights, arith_axis, index, cols):
    ##error handle 1: can't perform arithmetic along an axis and also report that axis and the index or col
    arith_occur = len(arith_axis) >= 1
    arith_error = any(item in index for item in arith_axis) or any(item in cols for item in arith_axis)
    if arith_occur and arith_error:  # if arith is happening and there is an error in selected axis
        raise exc.ArithError('''Arith error: can't perform operation along an axis that is going to be reported as the index or col''')

    ##error handle 2: can't report an axis as index and col
    axis_error = any(col in index for col in cols)
    if axis_error:  # if cols and index have any overlapping axis.
        raise exc.ArithError('''Arith error: can't have the same axis in index and cols''')

    ##error handle 3: once arith has been completed all axis that are not singleton must be used in either the index or cols
    if arith_occur:
        nonzero_idx = arith_axis + index + cols  # join lists
    else:
        nonzero_idx = index + cols  # join lists
    error = [prod.shape.index(size) not in nonzero_idx for size in prod.shape if size > 1]
    if any(error):
        raise exc.AxisError('''Axis error: active axes exist that are not used in arith or being reported as index or columns''')

    # This error flag is not required because doing Arith with weights==None throws a python error. Note: some Arith options don't require weights
    # ##error 4: performing arith with no weights
    # if arith_occur and weights is None:
    #     raise exc.ArithError('''Arith error: weights are not included''')
    return


def f_add_axis(prod, na_prod, prod_weights, na_prodweights, weights, na_weights, den_weights, na_denweights, den_assoc, na_den_assoc):
    '''
    Adds new axis if required.

    :param weights: array
    :param na_weights: list: position to add new axis
    :param prod: array
    :param na_prod: list: position to add new axis
    :return: expanded array
    '''
    weights = np.expand_dims(weights, na_weights)
    den_weights = np.expand_dims(den_weights, na_denweights)
    prod = np.expand_dims(prod, na_prod)
    prod_weights = np.expand_dims(prod_weights, na_prodweights)
    if den_assoc is not None:
        den_assoc = np.expand_dims(den_assoc, na_den_assoc)
    return prod, weights, den_weights, prod_weights, den_assoc


def f_slice(prod, prod_weights, weights, den_weights, keys, arith, axis_slice):
    '''
    Slices the prod, weights and key arrays

    :param prod: array: production param
    :param prod_weights: array: production param weights
    :param weights: array: weights (typically the variable associated with the prod param)
    :param keys: list: keys for axes
    :param axis_slice: dict: containing list of with slice params (start, stop, step)
    :return: prod array
    '''
    ## if arith is being conducted these arrays need to be the same size so slicing can work
    prod, prod_weights, weights, den_weights = np.broadcast_arrays(prod, prod_weights, weights, den_weights)

    ##slice axis - slice the keys and the array - if user hasn't specified slice the whole axis will be included
    sl = [slice(None)] * prod.ndim
    keys = keys.copy()  # need to copy so that it doesn't change the underlying array (because assigning in a loop)
    for axis, slc in axis_slice.items():
        start = slc[0]
        stop = slc[1]
        step = slc[2]
        sl[axis] = slice(start, stop, step)
        keys[axis] = keys[axis][start:stop:step]
    ###apply slice to np array
    prod = prod[tuple(sl)]
    prod_weights = prod_weights[tuple(sl)]
    weights = weights[tuple(sl)]
    den_weights = den_weights[tuple(sl)]
    return prod, prod_weights, weights, den_weights, keys


def f_arith(prod, prod_weights, weight, den_weights, arith, axis, den_assoc=None, assoc_axis=0):
    '''
    option 0: return production param averaged across all axis that are not reported.
    option 1: return weighted average of production param (using denominator weight returns production per day the animal is on hand)
    option 2: weighted total production summed across all axis that are not reported.
    option 3: weighted total production for each axis  (axis not reported are disregarded)
    option 4: return weighted average of production param using prod>0 as the weights
    option 5: return the maximum value across all axis that are not reported.

    :param prod: array: production param
    :param prod_weight: array: weights the production param
    :param weight: array: weights (typically the variable associated with the prod param)
    :param den_weight: array: weights the denominator in the weighted average calculation
    :param arith: int: arith option
    :param axis: list: axes to perform arith along
    :param den_assoc: array: pointer array to preform association (used only on denomiator)
    :return: array
    '''
    ##adjust prod by prod_weights
    prod = prod * prod_weights
    ##calc if keep dims
    keepdims = len(axis) != len(prod.shape)
    ##option 0
    if arith == 0:
        prod = np.mean(prod, tuple(axis), keepdims=keepdims)
    ##option 1
    if arith == 1:
        prod = fun.f_weighted_average(prod, weight, tuple(axis), keepdims=keepdims, den_weights=den_weights, den_assoc=den_assoc, assoc_axis=assoc_axis)
    ##option 2
    if arith == 2:
        prod = np.sum(prod * weight, tuple(axis), keepdims=keepdims)
    ##option 3
    if arith == 3:
        prod = prod * weight
    ##option 4
    if arith == 4:
        prod = fun.f_divide(np.sum(prod * (prod>0), tuple(axis), keepdims=keepdims), np.sum(prod>0, tuple(axis), keepdims=keepdims))
    ##option 5
    if arith == 5:
        prod = np.max(prod, tuple(axis), keepdims=keepdims)

    return prod


def f_numpy2df(prod, keys, index, cols):
    if prod.size <= 1 and prod.ndim <= 1:
        return pd.DataFrame([prod])  # don't need to reshape etc. if everything is summed and prod is just one number
    ##move x axis to front
    dest = list(range(len(index)))
    prod = np.moveaxis(prod, index, dest)

    ##move y axis to front behind x axis (note if an axis is not an index or col then it should be singleton)
    np_cols_y = np.array(cols)
    np_cols_xy = np_cols_y[na]
    np_index_xy = np.array(index)[:, na]
    cols_adj = np.sum(np_index_xy > np_cols_xy, axis=0)
    np_cols_y = np_cols_y + cols_adj
    dest = list(range(len(index), len(index) + len(cols)))
    prod = np.moveaxis(prod, np_cols_y, dest)

    ##select keys
    x_keys = []
    x_len = 1
    for axis in index:
        x_len *= len(keys[axis])
        x_keys.append(keys[axis])
    y_keys = []
    y_len = 1
    for axis in cols:
        y_len *= len(keys[axis])
        y_keys.append(keys[axis])

    ##reshape
    prod = prod.reshape(x_len, y_len)

    ##make df
    prod = fun.f_produce_df(prod, x_keys, y_keys)

    return prod
