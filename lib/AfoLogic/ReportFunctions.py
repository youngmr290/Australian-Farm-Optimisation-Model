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

def f_vars2np(lp_vars, var_key, shape, maskz8=None, z_pos=-1):
    '''
    Converts lp_vars to numpy.

    :param lp_vars: dict of lp variables
    :param var_key: string - name of variable to convert to numpy
    :param shape: shape of desired numpy array
    :param maskz8: z8 mask. Must be broadcastable to lp_vars
    :param z_pos: position of z axis
    :return: numpy array with un-clustered season axis.
    '''

    vars = np.array(list(lp_vars[var_key].values())).astype(float)
    vars = vars.reshape(shape)
    vars[np.isnan(vars)] = 0  # replace nan with 0

    ##uncluster z so that each season gets complete information
    if maskz8 is not None:
        index_z = fun.f_expand(np.arange(maskz8.shape[z_pos]),z_pos)
        a_zcluster = np.maximum.accumulate(index_z * maskz8,axis=z_pos)
        a_zcluster = np.broadcast_to(a_zcluster,vars.shape)
        vars = np.take_along_axis(vars,a_zcluster,axis=z_pos)

    return vars


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
def f_stock_reshape(lp_vars, r_vals):
    '''
    Stock reshape. Gets everything into the correct shape.
    Returns a dictionary with stock params.
    '''
    ##keys
    keys_p7 = r_vals['fin']['keys_p7']
    keys_a = r_vals['stock']['keys_a']
    keys_d = r_vals['stock']['keys_d']
    keys_g0 = r_vals['stock']['keys_g0']
    keys_g1 = r_vals['stock']['keys_g1']
    keys_g2 = r_vals['stock']['keys_g2']
    keys_g3 = r_vals['stock']['keys_g3']
    keys_f = r_vals['stock']['keys_f']
    keys_h1 = r_vals['stock']['keys_h1']
    keys_i = r_vals['stock']['keys_i']
    keys_k2 = r_vals['stock']['keys_k2']
    keys_k3 = r_vals['stock']['keys_k3']
    keys_k5 = r_vals['stock']['keys_k5']
    keys_lw1 = r_vals['stock']['keys_lw1']
    keys_lw3 = r_vals['stock']['keys_lw3']
    keys_lw_prog = r_vals['stock']['keys_lw_prog']
    keys_n1 = r_vals['stock']['keys_n1']
    keys_n3 = r_vals['stock']['keys_n3']
    keys_p8 = r_vals['stock']['keys_p8']
    keys_t1 = r_vals['stock']['keys_t1']
    keys_t2 = r_vals['stock']['keys_t2']
    keys_t3 = r_vals['stock']['keys_t3']
    keys_v1 = r_vals['stock']['keys_v1']
    keys_v3 = r_vals['stock']['keys_v3']
    keys_y0 = r_vals['stock']['keys_y0']
    keys_y1 = r_vals['stock']['keys_y1']
    keys_y3 = r_vals['stock']['keys_y3']
    keys_x = r_vals['stock']['keys_x']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_pastures = r_vals['pas']['keys_pastures']

    ##axis len
    len_a = len(keys_a)
    len_p7 = len(keys_p7)
    len_d = len(keys_d)
    len_g0 = len(keys_g0)
    len_g1 = len(keys_g1)
    len_g2 = len(keys_g2)
    len_g3 = len(keys_g3)
    len_f = len(keys_f)
    len_h1 = len(keys_h1)
    len_i = len(keys_i)
    len_k2 = len(keys_k2)
    len_k3 = len(keys_k3)
    len_k5 = len(keys_k5)
    len_lw1 = len(keys_lw1)
    len_lw3 = len(keys_lw3)
    len_lw_prog = len(keys_lw_prog)
    len_n1 = len(keys_n1)
    len_n3 = len(keys_n3)
    len_p8 = len(keys_p8)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_t1 = len(keys_t1)
    len_t2 = len(keys_t2)
    len_t3 = len(keys_t3)
    len_v1 = len(keys_v1)
    len_v3 = len(keys_v3)
    len_y0 = len(keys_y0)
    len_y1 = len(keys_y1)
    len_y3 = len(keys_y3)
    len_x = len(keys_x)
    len_z = len(keys_z)
    len_p6 = len(keys_p6)
    len_p5 = len(keys_p5)

    ##create dict for reshaped variables
    stock_vars = {}

    ##animal numbers
    ###shapes
    sire_shape = len_q, len_s, len_g0
    dams_shape = len_q, len_s, len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    prog_shape = len_q, len_s, len_k3, len_k5, len_t2, len_lw_prog, len_z, len_i, len_a, len_x, len_g2
    offs_shape = len_q, len_s, len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    infra_shape = len_q, len_s, len_h1, len_z
    ###sire
    stock_vars['sire_numbers_qsg0'] = f_vars2np(lp_vars, 'v_sire', sire_shape).astype(float)
    ###dams
    maskz8_k2tvanwziy1g1 = r_vals['stock']['maskz8_k2tvanwziy1g1']
    stock_vars['dams_numbers_qsk2tvanwziy1g1'] = f_vars2np(lp_vars, 'v_dams', dams_shape, maskz8_k2tvanwziy1g1,
                                                           z_pos=-4).astype(float)
    ###prog
    stock_vars['prog_numbers_qsk3k5twzia0xg2'] = f_vars2np(lp_vars, 'v_prog', prog_shape).astype(float)
    ###offs
    maskz8_k3k5tvnwziaxyg3 = r_vals['stock']['maskz8_k3k5tvnwziaxyg3']
    stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'] = f_vars2np(lp_vars, 'v_offs', offs_shape, maskz8_k3k5tvnwziaxyg3,
                                                             z_pos=-6).astype(float)
    ###infrastructure
    stock_vars['infrastructure_qsh1z'] = f_vars2np(lp_vars, 'v_infrastructure', infra_shape).astype(float)

    return stock_vars


def f_feed_reshape(lp_vars, r_vals):
    '''
    Reshape feed (pasture, residue & crop grazing) lp variables into numpy array.

    This is separate to the stock function above to save processing time (and the feed stuff overlaps a lot i.e.
    uses same keys).

    :param lp_vars: lp variables
    :return: dict
    '''
    keys_d = r_vals['pas']['keys_d']
    keys_f = r_vals['pas']['keys_f']
    keys_g = r_vals['pas']['keys_g']
    keys_k = r_vals['pas']['keys_k']
    keys_k1 = r_vals['stub']['keys_k1']
    keys_l = r_vals['pas']['keys_l']
    keys_o = r_vals['pas']['keys_o']
    keys_p5 = r_vals['pas']['keys_p5']
    keys_p6 = r_vals['pas']['keys_p6']
    keys_p7 = r_vals['zgen']['keys_p7']
    keys_r = r_vals['pas']['keys_r']
    keys_t = r_vals['pas']['keys_t']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_s1 = r_vals['stub']['keys_s1']
    keys_s2 = r_vals['stub']['keys_s2']
    keys_z = r_vals['zgen']['keys_z']

    len_d = len(keys_d)
    len_f = len(keys_f)
    len_p6 = len(keys_p6)
    len_g = len(keys_g)
    len_k = len(keys_k)
    len_k1 = len(keys_k1)
    len_l = len(keys_l)
    len_o = len(keys_o)
    len_p5 = len(keys_p5)
    len_q = len(keys_q)
    len_r = len(keys_r)
    len_s = len(keys_s)
    len_s1 = len(keys_s1)
    len_s2 = len(keys_s2)
    len_t = len(keys_t)
    len_z = len(keys_z)

    ##dict to store reshaped pasture stuff in
    feed_vars = {}

    ##store keys - must be in axis order
    ###pasture
    feed_vars['keys_qsfgop6lzt'] = [keys_q, keys_s, keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    feed_vars['keys_fgop6lzt'] = [keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    feed_vars['keys_gop6lzt'] = [keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]
    feed_vars['keys_qsfdp6zt'] = [keys_q, keys_s, keys_f, keys_d, keys_p6, keys_z, keys_t]
    feed_vars['keys_qsfdp6zlt'] = [keys_q, keys_s, keys_f, keys_d, keys_p6, keys_z, keys_l, keys_t]
    feed_vars['keys_fdp6zt'] = [keys_f, keys_d, keys_p6, keys_z, keys_t]
    feed_vars['keys_qsdp6zt'] = [keys_q, keys_s, keys_d, keys_p6, keys_z, keys_t]
    feed_vars['keys_qsdp6zlt'] = [keys_q, keys_s, keys_d, keys_p6, keys_z, keys_l, keys_t]
    feed_vars['keys_dp6zt'] = [keys_d, keys_p6, keys_z, keys_t]
    feed_vars['keys_qsfp6lz'] = [keys_q, keys_s, keys_f, keys_p6, keys_l, keys_z]
    ###crop residue
    feed_vars['keys_qszp6fks1s2'] = [keys_q, keys_s, keys_z, keys_p6, keys_f, keys_k1, keys_s1, keys_s2]
    ###crop grazing
    feed_vars['keys_qsfkp6p5zl'] = [keys_q, keys_s, keys_f, keys_k1, keys_p6, keys_p5, keys_z, keys_l]
    ###saltbush
    feed_vars['keys_qszp6fl'] = [keys_q, keys_s, keys_z, keys_p6, keys_f, keys_l]
    feed_vars['keys_qsp7zl'] = [keys_q, keys_s, keys_p7, keys_z, keys_l]
    ###periods
    feed_vars['keys_p7z'] = [keys_p7, keys_z]
    feed_vars['keys_p6z'] = [keys_p6, keys_z]

    ##shapes
    ###pasture
    qsfgop6lzt = len_q, len_s, len_f, len_g, len_o, len_p6, len_l, len_z, len_t
    qsfdp6zt = len_q, len_s, len_f, len_d, len_p6, len_z, len_t
    qsfdp6zlt = len_q, len_s, len_f, len_d, len_p6, len_z, len_l, len_t
    qsdp6zt = len_q, len_s, len_d, len_p6, len_z, len_t
    qsdp6zlt = len_q, len_s, len_d, len_p6, len_z, len_l, len_t
    qsfp6lz = len_q, len_s, len_f, len_p6, len_l, len_z
    ###residue
    qszp6fks1s2 = len_q, len_s, len_z, len_p6, len_f, len_k1, len_s1, len_s2
    ###crop graze
    qsfkp6p5zl = len_q, len_s, len_f, len_k1, len_p6, len_p5, len_z, len_l
    ###saltbush
    qszp6fl = len_q, len_s, len_z, len_p6, len_f, len_l
    qszl = len_q, len_s, len_z, len_l

    ##reshape z8 mask to uncluster
    maskz8_p6z = r_vals['pas']['mask_fp_z8var_p6z']
    maskz8_zp6 = maskz8_p6z.T
    maskz8_p6zna = maskz8_p6z[:, :, na]
    maskz8_p6znana = maskz8_p6z[:, :, na, na]
    maskz8_p6naz = maskz8_p6z[:, na, :]
    maskz8_p6nazna = maskz8_p6z[:, na, :, na]

    ##pasture
    ###green pasture hectare variable
    feed_vars['greenpas_ha_qsfgop6lzt'] = f_vars2np(lp_vars, 'v_greenpas_ha', qsfgop6lzt, maskz8_p6nazna, z_pos=-2)
    ###dry end period
    feed_vars['drypas_transfer_qsdp6zlt'] = f_vars2np(lp_vars, 'v_drypas_transfer', qsdp6zlt, maskz8_p6znana, z_pos=-3)
    ###nap end period
    feed_vars['nap_transfer_qsdp6zt'] = f_vars2np(lp_vars, 'v_nap_transfer', qsdp6zt, maskz8_p6zna, z_pos=-2)
    ###dry consumed
    feed_vars['drypas_consumed_qsfdp6zlt'] = f_vars2np(lp_vars, 'v_drypas_consumed', qsfdp6zlt, maskz8_p6znana,
                                                       z_pos=-3)
    ###nap consumed
    feed_vars['nap_consumed_qsfdp6zt'] = f_vars2np(lp_vars, 'v_nap_consumed', qsfdp6zt, maskz8_p6zna, z_pos=-2)
    ###poc consumed
    feed_vars['poc_consumed_qsfp6lz'] = f_vars2np(lp_vars, 'v_poc', qsfp6lz, maskz8_p6naz, z_pos=-1)

    ##crop residue
    ###stubble consumed
    feed_vars['stub_qszp6fks1s2'] = f_vars2np(lp_vars, 'v_stub_con', qszp6fks1s2, maskz8_zp6[:, :, na, na, na, na],
                                              z_pos=-6)

    ##crop grazing
    ###crop consumed
    feed_vars['crop_consumed_qsfkp6p5zl'] = f_vars2np(lp_vars, 'v_tonnes_crop_consumed', qsfkp6p5zl, maskz8_p6nazna,
                                                      z_pos=-2)

    ##saltbush
    ###saltbush consumed
    feed_vars['v_tonnes_sb_consumed_qszp6fl'] = f_vars2np(lp_vars, 'v_tonnes_sb_consumed', qszp6fl,
                                                          maskz8_zp6[:, :, na, na], z_pos=-4)
    feed_vars['v_slp_ha_qszl'] = f_vars2np(lp_vars, 'v_slp_ha', qszl, z_pos=-2)

    return feed_vars


def f_mach_reshape(lp_vars, r_vals):
    '''
    Reshape mach lp variables into numpy array.


    :param lp_vars: lp variables
    :return: dict
    '''
    keys_k = r_vals['pas']['keys_k']
    keys_l = r_vals['pas']['keys_l']
    keys_p5 = r_vals['pas']['keys_p5']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']

    len_k = len(keys_k)
    len_l = len(keys_l)
    len_p5 = len(keys_p5)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)

    ##dict to store reshaped pasture stuff in
    mach_vars = {}

    ##store keys - must be in axis order
    mach_vars['keys_qszp5kl'] = [keys_q, keys_s, keys_z, keys_p5, keys_k, keys_l]

    ##shapes
    qszp5kl = len_q, len_s, len_z, len_p5, len_k, len_l

    ##reshape z8 mask to uncluster
    maskz8_p5z = r_vals['lab']['maskz8_p5z']
    maskz8_zp5 = maskz8_p5z.T
    maskz8_zp5nana = maskz8_zp5[:, :, na, na]

    ##pasture
    ###green pasture hectare variable
    mach_vars['v_contractseeding_ha'] = f_vars2np(lp_vars, 'v_contractseeding_ha', qszp5kl, maskz8_zp5nana, z_pos=-4)
    ###dry end period
    mach_vars['v_seeding_machdays'] = f_vars2np(lp_vars, 'v_seeding_machdays', qszp5kl, maskz8_zp5nana, z_pos=-4)

    return mach_vars


#########################################
# intermediate report building functions#
#########################################
def f_price_summary(lp_vars, r_vals, option, grid, weight, fs):
    '''Returns price summaries

    :param r_vals:
    :param option:

            #. farmgate grain price
            #. wool price STB price for FNF (free or nearly free of fault)
            #. sale price for specified grid at given weight and fat score (sale yard price)

    :param grid: list - sale grids to report. Has to be int between 0 and 7 inclusive.
    :param weight: float/int - stock weight to report price for.
    :param fs: int - fat score to report price for. Has to be number between 1-5 inclusive.
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
        for t_grid, t_weight, t_fs in zip(grid, weight, fs):
            ##grid name - used in table index
            grid_name = grid_keys[t_grid]
            ##index grid and fs
            price_s5 = grid_price_s7s5s6[t_grid, :, t_fs]
            ##interpolate to get price for specified weight
            lookup_weights = weight_range_s7s5[t_grid, :]
            price = np.interp(t_weight, lookup_weights, price_s5)
            ##attach to df
            ###if price is less than 10 it is assumed to be $/kg else $/hd
            if price < 10:
                col = 'Price $/kg'
            else:
                col = 'Price $/hd'
            saleprice.loc[(grid_name, t_weight, t_fs), col] = price
        return saleprice



def f_summary(lp_vars, r_vals, trial):
    '''Returns a simple 1 row summary of the trial (season results are averaged)'''
    summary_df = pd.DataFrame(index=[trial], columns=['profit', 'profit max', 'profit min', 'profit stdev', 'risk neutral obj', 'utility',
                                                      'SR', 'SR max', 'SR min', 'SR stdev', 'Pas %', 'Pas % max', 'Pas % min', 'Pas % stdev',
                                                      'Cereal %', 'Cereal % max', 'Cereal % min', 'Cereal % stdev',
                                                      'Canola %', 'Canola % max', 'Canola % min', 'Canola % stdev',
                                                      'Sup', 'Sup max', 'Sup min', 'Sup stdev'])
    ##profit - no minroe and asset
    summary_df.loc[trial, 'profit'] = f_profit(lp_vars, r_vals, option=0)
    profit_max = f_profit(lp_vars, r_vals, option=3)[0]
    profit_min = f_profit(lp_vars, r_vals, option=3)[1]
    summary_df.loc[trial, 'profit max'] = profit_max * np.logical_not(profit_min==profit_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'profit min'] = profit_min * np.logical_not(profit_min==profit_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'profit stdev'] = f_profit(lp_vars, r_vals, option=3)[2]
    ##obj
    summary_df.loc[trial, 'risk neutral obj'] = f_profit(lp_vars, r_vals, option=1)
    ##utility
    summary_df.loc[trial, 'utility'] = f_profit(lp_vars, r_vals, option=2)
    ##total dse/ha in fp0
    summary_df.loc[trial, 'SR'] = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[0]
    SR_max = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[1]
    SR_min = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[2]
    summary_df.loc[trial, 'SR max'] = SR_max * np.logical_not(SR_min==SR_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'SR min'] = SR_min * np.logical_not(SR_min==SR_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'SR stdev'] = f_dse(lp_vars, r_vals, method=r_vals['stock']['dse_type'], per_ha=True, summary1=True)[3]
    ##pasture %
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    summary_df.loc[trial, 'Pas %'] = f_area_summary(lp_vars, r_vals, option=5)[0]
    Pas_max = f_area_summary(lp_vars, r_vals, option=5)[1]
    Pas_min = f_area_summary(lp_vars, r_vals, option=5)[2]
    summary_df.loc[trial, 'Pas % max'] = Pas_max * np.logical_not(Pas_min==Pas_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pas % min'] = Pas_min * np.logical_not(Pas_min==Pas_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Pas % stdev'] = f_area_summary(lp_vars, r_vals, option=5)[3]
    ##cereal %
    summary_df.loc[trial, 'Cereal %'] = f_area_summary(lp_vars, r_vals, option=6)[0]
    Cereal_max = f_area_summary(lp_vars, r_vals, option=6)[1]
    Cereal_min = f_area_summary(lp_vars, r_vals, option=6)[2]
    summary_df.loc[trial, 'Cereal % max'] = Cereal_max * np.logical_not(Cereal_min==Cereal_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Cereal % min'] = Cereal_min * np.logical_not(Cereal_min==Cereal_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Cereal % stdev'] = f_area_summary(lp_vars, r_vals, option=6)[3]
    ##canola %
    summary_df.loc[trial, 'Canola %'] = f_area_summary(lp_vars, r_vals, option=7)[0]
    Canola_max = f_area_summary(lp_vars, r_vals, option=7)[1]
    Canola_min = f_area_summary(lp_vars, r_vals, option=7)[2]
    summary_df.loc[trial, 'Canola % max'] = Canola_max * np.logical_not(Canola_min==Canola_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Canola % min'] = Canola_min * np.logical_not(Canola_min==Canola_max) #sets min/max to 0 if range is 0 so the cols get hidden
    summary_df.loc[trial, 'Canola % stdev'] = f_area_summary(lp_vars, r_vals, option=7)[3]
    ##supplement
    summary_df.loc[trial, 'Sup'] = f_grain_sup_summary(lp_vars,r_vals,option=4)[0]
    Sup_max = f_grain_sup_summary(lp_vars, r_vals, option=4)[1]
    Sup_min = f_grain_sup_summary(lp_vars, r_vals, option=4)[2]
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


def f_area_summary(lp_vars, r_vals, option):
    '''
    Rotation & landuse area summary. With multiple output levels.
    return options:

    :param lp_vars: dict
    :param r_vals: dict
    :key option:

        #. table all rotations by lmu
        #. total pasture area each season in p7[-1]
        #. total crop area each season in p7[-1]
        #. table crop and pasture area by lmu and season
        #. landuse area in p7[-1]
        #. float pasture %, max, min & stdev in p7[-1]
        #. float cereal %, max, min & stdev in p7[-1]
        #. float canola %, max, min & stdev in p7[-1]

    '''

    ##read from other functions
    rot_area_qszrl_p7, rot_area_qszlrk_p7 = f_rotation(lp_vars, r_vals)[2:4]
    landuse_area_k_p7qszl = rot_area_qszlrk_p7.groupby(axis=0, level=(0,1,2,3,5)).sum().unstack([0,1,2,3])  # area of each landuse (sum lmu and rotation)

    ##all rotations by lmu and p7
    rot_area_qszr_lp7 = rot_area_qszrl_p7.stack().unstack([-2,-1])
    if option == 0:
        return rot_area_qszr_lp7.round(2)

    ###pasture area
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    pasture_area_p7qszl = landuse_area_k_p7qszl[landuse_area_k_p7qszl.index.isin(all_pas)].sum()  # sum landuse
    if option == 1:
        pasture_area_qszl = pasture_area_p7qszl.loc[pasture_area_p7qszl.index.levels[0][-1].tolist()] #slice for p7[-1]
        return pasture_area_qszl.groupby(level=(0,1,2)).sum().round(0) #sum lmu

    ###crop area
    crop_area_p7qszl = landuse_area_k_p7qszl[~landuse_area_k_p7qszl.index.isin(all_pas)].sum()  # sum landuse
    if option == 2:
        crop_area_qszl = crop_area_p7qszl.loc[crop_area_p7qszl.index.levels[0][-1].tolist()]
        crop_area_qsz = crop_area_qszl.groupby(level=(0,1,2)).sum().round(0) #sum lmu
        return crop_area_qsz.unstack(-1)

    ##crop & pasture area by lmu
    if option == 3:
        croppas_area_qszl = pd.DataFrame()
        croppas_area_qszl['pasture'] = pasture_area_p7qszl
        croppas_area_qszl['crop'] = crop_area_p7qszl
        return croppas_area_qszl.round(0)

    if option==4: #landuse area in p7[-1] (lmu summed)
        landuse_area_k_qszl = landuse_area_k_p7qszl.loc[:,landuse_area_k_p7qszl.columns.levels[0][-1].tolist()]
        landuse_area_k_qsz = landuse_area_k_qszl.groupby(axis=1, level=(0,1,2)).sum()
        landuse_area_qsz_k = landuse_area_k_qsz.T.round(2)
        return landuse_area_qsz_k

    if option==5 or option==6 or option==7: #average % of pasture/cereal/canola in p7[-1]
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        rot_area_qsz_p7 = rot_area_qszrl_p7.groupby(level=(0,1,2)).sum() #sum r & l
        rot_area_qsz = rot_area_qsz_p7.iloc[:, -1]  # slice for p7[-1]
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


def f_mach_summary(lp_vars, r_vals, option=0):
    '''
    Machine summary.
    :param option:

        #. table: total machine cost for each crop in each cash period
        #. table: total seeding biomass penalty for untimely sowing.

    '''
    ##call rotation function to get rotation info
    phases_rk, v_phase_change_increase_area_qszrl_p7, rot_area_qszrl_p7 = f_rotation(lp_vars, r_vals)[0:3]
    rot_area_zrlqs_p7 = rot_area_qszrl_p7.reorder_levels([2,3,4,0,1], axis=0).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)

    ##masks to uncluster z axis
    maskz8_zp5 = r_vals['lab']['maskz8_p5z'].T

    ##harv
    contractharv_hours_qszp5k = f_vars2df(lp_vars, 'v_contractharv_hours', maskz8_zp5[:,:,na], z_pos=-3)
    contractharv_hours_zp5kqs = contractharv_hours_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)
    harv_hours_qszp5k = f_vars2df(lp_vars, 'v_harv_hours', maskz8_zp5[:,:,na], z_pos=-3)
    harv_hours_zp5kqs = harv_hours_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()  # change the order so that reindexing works (new levels being added must be at the end)
    contract_harvest_cost_zp5k_p7 = r_vals['mach']['contract_harvest_cost'].unstack(0)
    contract_harvest_cost_zp5kqs_p7 = contract_harvest_cost_zp5k_p7.reindex(contractharv_hours_zp5kqs.index, axis=0)
    own_harvest_cost_zp5k_p7 = r_vals['mach']['harvest_cost'].unstack(0)
    own_harvest_cost_zp5kqs_p7 = own_harvest_cost_zp5k_p7.reindex(harv_hours_zp5kqs.index, axis=0)
    harvest_cost_zp5kqs_p7 = contract_harvest_cost_zp5kqs_p7.mul(contractharv_hours_zp5kqs, axis=0) + own_harvest_cost_zp5kqs_p7.mul(harv_hours_zp5kqs, axis=0)
    harvest_cost_zkqs_p7 = harvest_cost_zp5kqs_p7.groupby(axis=0, level=(0,2,3,4)).sum() #sum p5

    ##seeding
    seeding_days_qszp5_kl = f_vars2df(lp_vars, 'v_seeding_machdays', maskz8_zp5[:,:,na,na], z_pos=-4).unstack([4,5])
    seeding_rate_kl = r_vals['mach']['seeding_rate'].stack()
    seeding_ha_qszp5_kl = seeding_days_qszp5_kl.mul(seeding_rate_kl.reindex(seeding_days_qszp5_kl.columns), axis=1) # note seeding ha won't equal the rotation area because arable area is included in seed_ha.
    seeding_ha_zp5lkqs = seeding_ha_qszp5_kl.stack([0,1]).reorder_levels([2,3,5,4,0,1]).sort_index()
    seeding_cost_zp5l_p7 = r_vals['mach']['seeding_cost'].unstack(0)
    seeding_cost_zp5lkqs_p7 = seeding_cost_zp5l_p7.reindex(seeding_ha_zp5lkqs.index, axis=0)
    seeding_cost_own_zkqs_p7 = seeding_cost_zp5lkqs_p7.mul(seeding_ha_zp5lkqs, axis=0).groupby(axis=0, level=(0,3,4,5)).sum()  # sum lmu axis and p5

    contractseeding_ha_qszp5k = f_vars2df(lp_vars, 'v_contractseeding_ha', maskz8_zp5[:,:,na,na], z_pos=-4).groupby(level=(0,1,2,3,4)).sum()  # sum lmu axis (cost doesn't vary by lmu for contract)
    contractseeding_ha_zp5kqs = contractseeding_ha_qszp5k.reorder_levels([2,3,4,0,1]).sort_index()
    contractseed_cost_ha_zp5_p7 = r_vals['mach']['contractseed_cost'].unstack(0)
    contractseed_cost_ha_zp5kqs_p7 = contractseed_cost_ha_zp5_p7.reindex(contractseeding_ha_zp5kqs.index, axis=0)
    seeding_cost_contract_zkqs_p7 =  contractseed_cost_ha_zp5kqs_p7.mul(contractseeding_ha_zp5kqs, axis=0).groupby(axis=0, level=(0,2,3,4)).sum()  # sum p5
    seeding_cost_zkqs_p7 = seeding_cost_contract_zkqs_p7 + seeding_cost_own_zkqs_p7
    # seeding_cost_c0p7z_k = seeding_cost_c0p7_zk.stack(0)

    ##fert & chem mach cost
    fert_app_cost_rl_p7z = r_vals['crop']['fert_app_cost']
    chem_app_cost_ha_rl_p7z = r_vals['crop']['chem_app_cost_ha']
    fertchem_cost_rl_p7z = pd.concat([fert_app_cost_rl_p7z, chem_app_cost_ha_rl_p7z], axis=1).groupby(axis=1, level=(0,1)).sum()  # cost per ha

    fertchem_cost_zrl_p7 = fertchem_cost_rl_p7z.stack().reorder_levels([2,0,1], axis=0).sort_index()
    fertchem_cost_zrlqs_p7 = fertchem_cost_zrl_p7.reindex(rot_area_zrlqs_p7.index, axis=0)
    fertchem_cost_zrqs_p7 = fertchem_cost_zrlqs_p7.mul(rot_area_zrlqs_p7, axis=0).groupby(axis=0, level=(0,1,3,4)).sum()  # mul area and sum lmu
    fertchem_cost_k_p7zqs = fertchem_cost_zrqs_p7.unstack([0,2,3]).reindex(phases_rk.index, axis=0, level=0).groupby(axis=0,level=1).sum()  # reindex to include landuse and sum rot
    fertchem_cost_zkqs_p7 = fertchem_cost_k_p7zqs.stack([1,2,3]).swaplevel(0,1)

    ##combine all costs
    exp_mach_zkqs_p7 = pd.concat([fertchem_cost_zkqs_p7, seeding_cost_zkqs_p7, harvest_cost_zkqs_p7
                               ], axis=0).groupby(axis=0, level=(0,1,2,3)).sum()
    exp_mach_k_p7zqs = exp_mach_zkqs_p7.unstack([0,2,3])
    ##insurance
    mach_insurance_p7z = r_vals['mach']['mach_insurance']

    ##yield penalty from untimely sowing
    sowing_yield_penalty_p7p5zk = r_vals['mach']['sowing_yield_penalty_p7p5zk']
    sowing_yield_penalty_p5zk = sowing_yield_penalty_p7p5zk.groupby(level=(1,2,3)).sum() #sum p7
    sowing_yield_penalty_zp5k = sowing_yield_penalty_p5zk.reorder_levels((1,0,2))
    ###ha sown by farmer
    seeding_ha_qszp5_kl = seeding_days_qszp5_kl.mul(seeding_rate_kl.reindex(seeding_days_qszp5_kl.columns), axis=1) # note seeding ha won't equal the rotation area because arable area is included in seed_ha.
    seeding_ha_qszp5k_l = seeding_ha_qszp5_kl.stack(0)
    ###reindex penalty param
    ####add q & s axis
    sowing_yield_penalty_qsz_kp5 = sowing_yield_penalty_zp5k.unstack((-1,-2)).reindex(seeding_ha_qszp5_kl.unstack(-1).index, axis=0, level=-1)
    sowing_yield_penalty_qszp5k = sowing_yield_penalty_qsz_kp5.stack((1,0))
    ####expand k to include pastures
    sowing_yield_penalty_qszp5k = sowing_yield_penalty_qszp5k.reindex(seeding_ha_qszp5k_l.index)
    ###calc penalty
    farmer_penalty_qszp5k_l = seeding_ha_qszp5k_l.mul(sowing_yield_penalty_qszp5k, axis=0)
    farmer_penalty_qszp5k = farmer_penalty_qszp5k_l.sum(axis=1)
    contract_penalty_qszp5k = contractseeding_ha_qszp5k.mul(sowing_yield_penalty_qszp5k, axis=0)
    total_penalty_qszk = farmer_penalty_qszp5k.add(contract_penalty_qszp5k).unstack(3).sum(axis=1)
    if option == 1:
        return total_penalty_qszk
    ##return all if option==0
    if option == 0:
        return exp_mach_k_p7zqs, mach_insurance_p7z

def f_available_cropgrazing(lp_vars, r_vals):
    '''
    Calculates the total crop that CAN be grazed based on seeding timing (the actual amount consumed is optimised).
    '''
    mach_vars = f_mach_reshape(lp_vars, r_vals)
    v_contractseeding_ha_qszp5kl = mach_vars['v_contractseeding_ha']
    v_seeding_machdays_qszp5kl = mach_vars['v_seeding_machdays']

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
    crop_grazing_penalty_qszk = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)

    penalty = pd.concat([seeding_penalty_qszp5k, crop_grazing_penalty_qszk], axis=1)
    penalty.columns=['seeding', 'crop_grazing']
    return penalty

def f_grain_sup_summary(lp_vars, r_vals, option=0):
    '''
    Summary of grain, supplement and their costs

    :param option: int:

            #. return dict with sup cost
            #. return total supplement fed in each feed period
            #. return total of each grain supplement fed in each feed period in each season
            #. return total of each grain supplement fed in each feed period for each feed pool in each season
            #. return total sup fed (weighted by season prob)

    '''
    ##z masks to uncluster lp_vars
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    maskz8_zp6 = r_vals['pas']['mask_fp_z8var_p6z'].T

    ##grain fed
    grain_fed_qszkgvp6 = f_vars2df(lp_vars, 'v_sup_con', maskz8_zp6[:,na,na,na,:], z_pos=-5)

    if option == 1:
        grain_fed_qszp6 = grain_fed_qszkgvp6.groupby(level=(0, 1, 2, 6)).sum()  # sum feed pool, landuse and grain pool
        return grain_fed_qszp6.to_frame()

    if option == 2:
        grain_fed_qszkp6 = grain_fed_qszkgvp6.groupby(level=(0, 1, 2, 3, 6)).sum()  # sum feed pool and grain pool
        return grain_fed_qszkp6

    if option == 3:
        grain_fed_qszkfp6 = grain_fed_qszkgvp6.groupby(level=(0, 1, 2, 3, 5, 6)).sum()  # sum grain pool
        return grain_fed_qszkfp6

    if option == 4:
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        grain_fed_qsz = grain_fed_qszkgvp6.groupby(level=(0,1,2)).sum() #sum all axis except season ones (q,s,z)
        ###stdev and range
        grain_fed_mean = grain_fed_qsz.mul(z_prob_qsz).sum()
        ma_grain_fed_qsz = np.ma.masked_array(grain_fed_qsz, z_prob_qsz == 0)
        grain_fed_max = np.max(ma_grain_fed_qsz)
        grain_fed_min = np.min(ma_grain_fed_qsz)
        grain_fed_stdev = (((grain_fed_qsz - grain_fed_mean) ** 2).mul(z_prob_qsz).sum())**0.5
        return round(grain_fed_mean, 1), round(grain_fed_max, 1), round(grain_fed_min, 1), round(grain_fed_stdev, 1),

    ##NOTE: this only works if there is one time of grain purchase/sale
    if option == 0:
        ##create dict to store grain variables
        grain = {}
        ##prices
        grains_sale_price_zks2g_p7 = r_vals['crop']['grain_price'].stack().reorder_levels([3,0,1,2]).sort_index()
        grains_buy_price_zks2g_p7 = r_vals['sup']['buy_grain_price'].stack().reorder_levels([3,0,1,2]).sort_index()

        ##grain purchased
        grain_purchased_qsp7zks2g = f_vars2df(lp_vars,'v_buy_product', mask_season_p7z[:,:,na,na,na], z_pos=-4)
        grain_purchased_qszks2g = grain_purchased_qsp7zks2g.groupby(level=(0,1,3,4,5,6)).sum()  # sum p7
        grain_purchased_zks2gqs = grain_purchased_qszks2g.reorder_levels([2,3,4,5,0,1]) .sort_index()#change the order so that reindexing works (new levels being added must be at the end)

        ##grain sold
        grain_sold_qsp7zks2g = f_vars2df(lp_vars,'v_sell_product', mask_season_p7z[:,:,na,na,na], z_pos=-4)
        grain_sold_qszks2g = grain_sold_qsp7zks2g.groupby(level=(0,1,3,4,5,6)).sum()  # sum p7
        grain_sold_zks2gqs = grain_sold_qszks2g.reorder_levels([2,3,4,5,0,1]).sort_index() #change the order so that reindexing works (new levels being added must be at the end)

        ##grain fed - s2 axis added because sup feed is allocated to a given s2 slice and therefore the variable doesn't have an active s2 axis
        sup_s2_ks2 = r_vals['sup']['sup_s2_k_s2'].stack()
        grain_fed_qszkg = grain_fed_qszkgvp6.groupby(level=(0, 1, 2, 3, 4)).sum()  # sum feed pool and feed period
        grain_fed_qszg_ks2 = grain_fed_qszkg.unstack(3).mul(sup_s2_ks2, axis=1, level=0)
        grain_fed_zks2gqs = grain_fed_qszg_ks2.stack([0,1]).reorder_levels([2,4,5,3,0,1]).sort_index() #change the order so that reindexing works (new levels being added must be at the end)

        ##total grain produced by crop enterprise
        total_grain_produced_zks2gqs = grain_sold_zks2gqs + grain_fed_zks2gqs - grain_purchased_zks2gqs  # total grain produced by crop enterprise
        grains_sale_price_zks2gqs_p7 = grains_sale_price_zks2g_p7.reindex(total_grain_produced_zks2gqs.index, axis=0)
        grains_buy_price_zks2gqs_p7 = grains_buy_price_zks2g_p7.reindex(total_grain_produced_zks2gqs.index, axis=0)
        rev_grain_k_p7zqs = grains_sale_price_zks2gqs_p7.mul(total_grain_produced_zks2gqs, axis=0).unstack([0,4,5]).groupby(axis=0, level=0).sum()  # sum grain pool and s2
        grain['rev_grain_k_p7zqs'] = rev_grain_k_p7zqs

        ##supplementary cost: cost = sale_price * (grain_fed - grain_purchased) + buy_price * grain_purchased
        sup_exp_zqs_p7 = (grains_sale_price_zks2gqs_p7.mul(grain_fed_zks2gqs - grain_purchased_zks2gqs, axis=0)
                     + grains_buy_price_zks2gqs_p7.mul(grain_purchased_zks2gqs, axis=0)).groupby(axis=0,level=(0,4,5)).sum()  # sum grain pool & landuse & s2
        grain['sup_exp_p7zqs'] = sup_exp_zqs_p7.unstack([0,1,2])
        return grain


def f_mvf_summary(lp_vars):
    mvf_qsp6vq = f_vars2df(lp_vars, 'mvf')
    return mvf_qsp6vq.unstack([-1,-2])


def f_crop_summary(lp_vars, r_vals, option=0):
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
    rev_grain_k_p7zqs = grain_summary['rev_grain_k_p7zqs']

    ##return all if option==0
    if option == 0:
        return exp_fert_k_p7zqs, exp_chem_k_p7zqs, misc_exp_k_p7zqs, rev_grain_k_p7zqs


def f_stock_cash_summary(lp_vars, r_vals):
    '''
    Returns:

        #. expense and revenue items

    '''
    ##get reshaped variable
    stock_vars = f_stock_reshape(lp_vars, r_vals)

    ##numbers
    sire_numbers_qsg0 = stock_vars['sire_numbers_qsg0']
    dams_numbers_qsk2tvanwziy1g1 = stock_vars['dams_numbers_qsk2tvanwziy1g1']
    prog_numbers_qsk3k5twzia0xg2 = stock_vars['prog_numbers_qsk3k5twzia0xg2']
    offs_numbers_qsk3k5tvnwziaxyg3 = stock_vars['offs_numbers_qsk3k5tvnwziaxyg3']

    ##husb cost
    sire_cost_qsp7zg0 = r_vals['stock']['sire_cost_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]
    dams_cost_qsk2p7tva1nwziyg1 = r_vals['stock']['dams_cost_k2p7tva1nwziyg1'] * dams_numbers_qsk2tvanwziy1g1[:, :, :, na, ...]
    offs_cost_qsk3k5p7tvnwziaxyg3 = r_vals['stock']['offs_cost_k3k5p7tvnwziaxyg3'] * offs_numbers_qsk3k5tvnwziaxyg3[:, :, :, :, na, ...]

    ##purchase cost
    sire_purchcost_qsp7zg0 = r_vals['stock']['purchcost_sire_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]

    ##sale income
    salevalue_qsp7zg0 = r_vals['stock']['salevalue_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]
    salevalue_qsk2p7tva1nwziyg1 = r_vals['stock']['salevalue_k2p7tva1nwziyg1'] * dams_numbers_qsk2tvanwziy1g1[:, :, :, na, ...]
    salevalue_qsk3k5p7twzia0xg2 = r_vals['stock']['salevalue_k3k5p7twzia0xg2'] * prog_numbers_qsk3k5twzia0xg2[:, :, :, :, na, ...]
    salevalue_qsk3k5p7tvnwziaxyg3 = r_vals['stock']['salevalue_k3k5p7tvnwziaxyg3'] * offs_numbers_qsk3k5tvnwziaxyg3[:, :, :, :, na, ...]

    ##asset income - used to calculate the change in asset value at the start of season. This is required in the pnl report because sale management can vary across the z axis and then at the start of the season all z get averaged e.g. z0 might retain sheep and z4 might sell. If there is no asset value then z0 will look worse.
    assetvalue_startseason_qsp7zg0 = r_vals['stock']['assetvalue_startseason_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]
    assetvalue_endseason_qsp7zg0 = r_vals['stock']['assetvalue_endseason_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]
    assetvalue_startseason_qsk2p7tva1nwziyg1 = r_vals['stock']['assetvalue_startseason_k2p7tva1nwziyg1'] * dams_numbers_qsk2tvanwziy1g1[:, :, :, na, ...]
    assetvalue_endseason_qsk2p7tva1nwziyg1 = r_vals['stock']['assetvalue_endseason_k2p7tva1nwziyg1'] * dams_numbers_qsk2tvanwziy1g1[:, :, :, na, ...]
    assetvalue_startseason_qsk3k5p7tvnwziaxyg3 = r_vals['stock']['assetvalue_startseason_k3k5p7tvnwziaxyg3'] * offs_numbers_qsk3k5tvnwziaxyg3[:, :, :, :, na, ...]
    assetvalue_endseason_qsk3k5p7tvnwziaxyg3 = r_vals['stock']['assetvalue_endseason_k3k5p7tvnwziaxyg3'] * offs_numbers_qsk3k5tvnwziaxyg3[:, :, :, :, na, ...]

    ##wool income
    woolvalue_qsp7zg0 = r_vals['stock']['woolvalue_p7zg0'] * sire_numbers_qsg0[:, :, na, na, :]
    woolvalue_qsk2p7tva1nwziyg1 = r_vals['stock']['woolvalue_k2p7tva1nwziyg1'] * dams_numbers_qsk2tvanwziy1g1[:, :, :, na, ...]
    woolvalue_qsk3k5p7tvnwziaxyg3 = r_vals['stock']['woolvalue_k3k5p7tvnwziaxyg3'] * offs_numbers_qsk3k5tvnwziaxyg3[:, :, :, :, na, ...]

    ###sum axis to return total income in each cash period
    siresale_qsp7z = fun.f_reduce_skipfew(np.sum, salevalue_qsp7zg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7
    damssale_qsp7z = fun.f_reduce_skipfew(np.sum, salevalue_qsk2p7tva1nwziyg1, preserveAxis=(0,1,3,9))  # sum all axis except q,s,p7,z
    progsale_qsp7z = fun.f_reduce_skipfew(np.sum, salevalue_qsk3k5p7twzia0xg2, preserveAxis=(0,1,4,7))  # sum all axis except q,s,p7,z
    offssale_qsp7z = fun.f_reduce_skipfew(np.sum, salevalue_qsk3k5p7tvnwziaxyg3, preserveAxis=(0,1,4,9))  # sum all axis except q,s,p7,z
    sirewool_qsp7z = fun.f_reduce_skipfew(np.sum, woolvalue_qsp7zg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7,z
    damswool_qsp7z = fun.f_reduce_skipfew(np.sum, woolvalue_qsk2p7tva1nwziyg1, preserveAxis=(0,1,3,9))  # sum all axis except q,s,p7,z
    offswool_qsp7z = fun.f_reduce_skipfew(np.sum, woolvalue_qsk3k5p7tvnwziaxyg3, preserveAxis=(0,1,4,9))  # sum all axis except q,s,p7,z
    stocksale_qsp7z = siresale_qsp7z + damssale_qsp7z + progsale_qsp7z + offssale_qsp7z
    wool_qsp7z = sirewool_qsp7z + damswool_qsp7z + offswool_qsp7z

    sirecost_qsp7z = fun.f_reduce_skipfew(np.sum, sire_cost_qsp7zg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7,z
    damscost_qsp7z = fun.f_reduce_skipfew(np.sum, dams_cost_qsk2p7tva1nwziyg1, preserveAxis=(0,1,3,9))  # sum all axis except q,s,p7,z
    offscost_qsp7z = fun.f_reduce_skipfew(np.sum, offs_cost_qsk3k5p7tvnwziaxyg3, preserveAxis=(0,1,4,9))  # sum all axis except q,s,p7,z

    sire_purchcost_qsp7z = fun.f_reduce_skipfew(np.sum, sire_purchcost_qsp7zg0, preserveAxis=(0,1,2,3))  # sum all axis except q,s,p7

    ###change in asset value at the season start. This needs to be reported in pnl because at the season start stock numbers get averaged.
    ### Meaning that if z0 retains animals and z1 sells animals z0 will have a lower apparent profit which is not correct.
    ### Adding this essentially means that at the end of the season animals are purchased and sold to get back to the starting point (e.g. a good season with more animals sells some animals to the poor season that has less animals so that all seasons have the same starting point).
    ### Note if weaning occurs in the period before season start there will be an error (value of prog that are sold will get double counted). Easiest solution is to change weaning date.
    trade_value_sire_qsp7z = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_qsp7zg0, preserveAxis=(0,1,2,3))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_qsp7zg0, preserveAxis=(0,1,2,3)), shift=-1, axis=2))  # sum all axis except q,s,p7. Roll so that the result is in the end p7
    trade_value_dams_qsp7z = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_qsk2p7tva1nwziyg1, preserveAxis=(0,1,3,9))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_qsk2p7tva1nwziyg1, preserveAxis=(0,1,3,9)), shift=-1, axis=2))  # sum all axis except q,s,p7,z. Roll so that the result is in the end p7
    trade_value_offs_qsp7z = (fun.f_reduce_skipfew(np.sum, assetvalue_endseason_qsk3k5p7tvnwziaxyg3, preserveAxis=(0,1,4,9))
                                 - np.roll(fun.f_reduce_skipfew(np.sum, assetvalue_startseason_qsk3k5p7tvnwziaxyg3, preserveAxis=(0,1,4,9)), shift=-1, axis=2))  # sum all axis except q,s,p7,z. Roll so that the result is in the end p7
    trade_value_qsp7z = trade_value_sire_qsp7z + trade_value_dams_qsp7z + trade_value_offs_qsp7z

    ##expenses sup feeding
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    sup_grain_cost_p7zqs = grain_summary['sup_exp_p7zqs']
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)

    grain_fed_zkfp6qs = grain_fed_qszkfp6.reorder_levels([2,3,4,5,0,1]).sort_index() # change the order so that reindexing works (new levels being added must be at the end)
    supp_feedstorage_cost_p7zp6kf = r_vals['sup']['total_sup_cost_p7zp6kf']
    supp_feedstorage_cost_p7_zkfp6qs = supp_feedstorage_cost_p7zp6kf.unstack([1,3,4,2]).reindex(grain_fed_zkfp6qs.index, axis=1)
    supp_feedstorage_cost_p7_zkfp6qs = supp_feedstorage_cost_p7_zkfp6qs.mul(grain_fed_zkfp6qs, axis=1)
    supp_feedstorage_cost_p7zqs = supp_feedstorage_cost_p7_zkfp6qs.groupby(axis=1, level=(0,4,5)).sum().stack([0,1,2]) #sum k & p6 & f

    ##infrastructure
    fixed_infra_cost_qsp7z = np.sum(r_vals['stock']['rm_stockinfra_fix_h1p7z'], axis=0) * r_vals['zgen']['mask_qs'][:,:,na,na]
    var_infra_cost_qsp7z = np.sum(r_vals['stock']['rm_stockinfra_var_h1p7z'] * stock_vars['infrastructure_qsh1z'][:,:,:,na,:], axis=2)
    total_infra_cost_qsp7z = fixed_infra_cost_qsp7z + var_infra_cost_qsp7z

    ##total costs
    husbcost_qsp7z = sirecost_qsp7z + damscost_qsp7z + offscost_qsp7z + total_infra_cost_qsp7z
    supcost_p7zqs = sup_grain_cost_p7zqs + supp_feedstorage_cost_p7zqs
    purchasecost_qsp7z = sire_purchcost_qsp7z

    ##get axis in correct order for pnl table
    stocksale_qszp7 = np.moveaxis(stocksale_qsp7z, source=-1, destination=2)
    trade_value_qszp7 = np.moveaxis(trade_value_qsp7z, source=-1, destination=2)
    wool_qszp7 = np.moveaxis(wool_qsp7z, source=-1, destination=2)
    husbcost_qszp7 = np.moveaxis(husbcost_qsp7z, source=-1, destination=2)
    purchasecost_qszp7 = np.moveaxis(purchasecost_qsp7z, source=-1, destination=2)
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
    len_p5 = len(keys_p5)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)

    qsp5z = len_q, len_s, len_p5, len_z
    qsz = len_q, len_s, len_z

    ##total labour cost
    if option == 0:
        ###casual
        casual_cost_p7zp5 = r_vals['lab']['casual_cost_p7zp5']
        quantity_casual_qsp5z = f_vars2np(lp_vars, 'v_quantity_casual', qsp5z, maskz8_p5z, z_pos=-1)
        quantity_casual_qszp5 = np.swapaxes(quantity_casual_qsp5z, -1, -2)
        cas_cost_p7qsz = np.sum(casual_cost_p7zp5[:,na,na,:,:] * quantity_casual_qszp5, axis=-1)
        ###perm
        quantity_perm = f_vars2np(lp_vars, 'v_quantity_perm', 1)  #1 because not sets
        perm_cost_p7z = r_vals['lab']['perm_cost_p7z']
        perm_cost_p7qsz = perm_cost_p7z[:,na,na,:] * quantity_perm * r_vals['zgen']['mask_qs'][:,:,na]
        ###manager
        quantity_manager = f_vars2np(lp_vars, 'v_quantity_manager', 1) #1 because not sets
        manager_cost_p7z = r_vals['lab']['manager_cost_p7z']
        manager_cost_p7qsz = manager_cost_p7z[:,na,na,:] * quantity_manager * r_vals['zgen']['mask_qs'][:,:,na]
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
    len_p7 = len(keys_p7)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)
    qsp7z = len_q, len_s, len_p7, len_z
    dep_qsp7z = f_vars2np(lp_vars, 'v_dep', qsp7z, mask_season_p7z, z_pos=-1)
    return dep_qsp7z

def f_minroe_summary(lp_vars, r_vals):
    ##min return on expense cost
    keys_p7 = r_vals['fin']['keys_p7']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    len_p7 = len(keys_p7)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)
    qsp7z = len_q, len_s, len_p7, len_z

    minroe_qsp7z = f_vars2np(lp_vars, 'v_minroe', qsp7z, mask_season_p7z, z_pos=-1)
    return minroe_qsp7z

def f_asset_cost_summary(lp_vars, r_vals):
    ##asset opportunity cost
    keys_p7 = r_vals['fin']['keys_p7']
    mask_season_p7z = r_vals['zgen']['mask_season_p7z']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    len_p7 = len(keys_p7)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)
    qsp7z = len_q, len_s, len_p7, len_z
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
    len_p7 = len(keys_p7)
    len_c0 = len(keys_c0)
    len_q = len(keys_q)
    len_s = len(keys_s)
    len_z = len(keys_z)
    qsc0p7z = len_q, len_s, len_c0, len_p7, len_z
    capital_qsc0p7z = f_vars2np(lp_vars, 'v_wc_debit', qsc0p7z, mask_season_p7z, z_pos=-1)
    capital_qsz = np.max(capital_qsc0p7z, axis=(2,3)) #max c0 and p7 axis
    keys_qsz = [keys_q, keys_s, keys_z]
    df_capital_qsz = f_numpy2df(capital_qsz, keys_qsz, [0,1], [2])
    return df_capital_qsz

def f_overhead_summary(r_vals):
    ##overheads/fixed expenses
    exp_fix_c = r_vals['fin']['overheads']
    return exp_fix_c

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
    :param summary1: Bool
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
        sire_preserve_ax = (0, 1, 3)
        dams_preserve_ax = (0, 1, 9)
        offs_preserve_ax = (0, 1, 9)

    stock_vars = f_stock_reshape(lp_vars, r_vals)

    if method == 0:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_qsg0'][:, :, na, na, :]
                                        * r_vals['stock']['dsenw_p6zg0'][na,na,...], preserveAxis=sire_preserve_ax)  # sum all axis except preserveAxis
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_qsk2tvanwziy1g1'][:, :, :, na, ...]
                                        * r_vals['stock']['dsenw_k2p6tva1nwziyg1'][na,na,...], preserveAxis=dams_preserve_ax)  # sum all axis except preserveAxis
        ##offs
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'][:, :, :, :, na, ...]
                                        * r_vals['stock']['dsenw_k3k5p6tvnwziaxyg3'][na,na,...], preserveAxis=offs_preserve_ax)  # sum all axis except preserveAxis
    else:
        ##sire
        dse_sire = fun.f_reduce_skipfew(np.sum, stock_vars['sire_numbers_qsg0'][:, :, na, na, :]
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
        sire_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_p7zg0']>0)
                                                           * stock_vars['sire_numbers_qsg0'][:, :, na, na, :], preserveAxis=sire_preserve_ax)
        sire_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_p7zg0']>0)
                                                         * stock_vars['sire_numbers_qsg0'][:, :, na, na, :], preserveAxis=sire_preserve_ax)
        dams_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_k2p7tva1nwziyg1']>0)
                                                                     * stock_vars['dams_numbers_qsk2tvanwziy1g1'][:, :, :, na, ...], preserveAxis=dams_preserve_ax)
        dams_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_k2p7tva1nwziyg1']>0)
                                                                   * stock_vars['dams_numbers_qsk2tvanwziy1g1'][:, :, :, na, ...], preserveAxis=dams_preserve_ax)
        offs_numbers_startseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_startseason_k3k5p7tvnwziaxyg3']>0)
                                                                       * stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'][:, :, :, :, na, ...], preserveAxis=offs_preserve_ax)
        offs_numbers_endseason_qsz = fun.f_reduce_skipfew(np.sum, (r_vals['stock']['assetvalue_endseason_k3k5p7tvnwziaxyg3']>0)
                                                                     * stock_vars['offs_numbers_qsk3k5tvnwziaxyg3'][:, :, :, :, na, ...], preserveAxis=offs_preserve_ax)

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
    exp_fert_k_p7zqs, exp_chem_k_p7zqs, misc_exp_k_p7zqs, rev_grain_k_p7zqs = f_crop_summary(lp_vars, r_vals, option=0)
    exp_mach_k_p7zqs, mach_insurance_p7z = f_mach_summary(lp_vars, r_vals)
    stocksale_qszp7, wool_qszp7, husbcost_qszp7, supcost_qsz_p7, purchasecost_qszp7, trade_value_qszp7 = f_stock_cash_summary(lp_vars, r_vals)
    slp_estab_cost_qsz_p7 = f_stock_pasture_summary(lp_vars, r_vals, type='slp', prod='slp_estab_cost_p7z', na_prod=[0,1,4]
                                             , weights='v_slp_ha_qszl', na_weights=[2]
                                             , keys='keys_qsp7zl', arith=2, index=[0,1,3], cols=[2])
    labour_p7qsz = f_labour_summary(lp_vars, r_vals, option=0)
    exp_fix_p7_z = f_overhead_summary(r_vals).unstack()
    dep_qsp7z = f_dep_summary(lp_vars, r_vals)
    minroe_qsp7z = f_minroe_summary(lp_vars,r_vals)
    asset_cost_qsp7z = f_asset_cost_summary(lp_vars,r_vals)



    ##manipulate arrays into correct shape
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    keys_p7 = r_vals['fin']['keys_p7']
    keys_q = r_vals['zgen']['keys_q']
    keys_s = r_vals['zgen']['keys_s']
    keys_z = r_vals['zgen']['keys_z']
    len_p7 = len(keys_p7)
    len_z = len(keys_z)
    ###rev
    rev_grain_p7_qsz = rev_grain_k_p7zqs.sum(axis=0).unstack([2,3,1]).sort_index(axis=1)  # sum landuse axis
    ###exp
    ####machinery
    df_mask_qs = pd.DataFrame(r_vals['zgen']['mask_qs'],keys_q,keys_s).stack()
    mach_p7zqs = exp_mach_k_p7zqs.sum(axis=0)  # sum landuse
    mach_p7_qsz = mach_p7zqs.unstack([2,3]).add(mach_insurance_p7z, axis=0).mul(df_mask_qs,axis=1).unstack()
    ####crop & pasture
    pasfert_p7_qsz = exp_fert_k_p7zqs[exp_fert_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropfert_p7_qsz = exp_fert_k_p7zqs[~exp_fert_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    paschem_p7_qsz = exp_chem_k_p7zqs[exp_chem_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropchem_p7_qsz = exp_chem_k_p7zqs[~exp_chem_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    pasmisc_p7_qsz = misc_exp_k_p7zqs[misc_exp_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    cropmisc_p7_qsz = misc_exp_k_p7zqs[~misc_exp_k_p7zqs.index.isin(all_pas)].sum(axis=0).unstack([2,3,1]).sort_index(axis=1)
    pas_p7_qsz = pd.concat([pasfert_p7_qsz, paschem_p7_qsz, pasmisc_p7_qsz], axis=0).groupby(axis=0, level=0).sum()
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
    subtype_rev = ['grain', 'sheep sales', 'wool', 'Total Revenue']
    subtype_exp = ['crop', 'pasture', 'slp', 'stock husb', 'stock sup', 'stock purchase', 'machinery', 'labour', 'fixed', 'Total expenses']
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
    pnl.loc[idx[:, :, :, 'Revenue', 'Total Revenue'], :] = pnl.loc[pnl.index.get_level_values(3) == 'Revenue'].groupby(axis=0,level=(0,1,2)).sum().values

    ##expenses - add to p/l table each as a new row
    pnl.loc[idx[:, :, :, 'Expense', 'crop'], :] = crop_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'pasture'], :] = pas_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'slp'], :] = slp_estab_cost_qsz_p7.values
    pnl.loc[idx[:, :, :, 'Expense', 'stock husb'], :] = husbcost_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Expense', 'stock sup'], :] = supcost_qsz_p7.values
    pnl.loc[idx[:, :, :, 'Expense', 'stock purchase'], :] = purchasecost_qszp7.reshape(-1, len_p7)
    pnl.loc[idx[:, :, :, 'Expense', 'machinery'], :] = mach_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'labour'], :] = labour_p7qsz.reshape(len_p7, -1).T
    pnl.loc[idx[:, :, :, 'Expense', 'fixed'], :] = exp_fix_p7_qsz.T.values
    pnl.loc[idx[:, :, :, 'Expense', 'Total expenses'], :] = pnl.loc[pnl.index.get_level_values(3) == 'Expense'].groupby(axis=0,level=(0,1,2)).sum().values

    ##EBITDA
    ebtd = pnl.loc[idx[:, :, :, 'Revenue', 'Total Revenue']].values - pnl.loc[idx[:, :, :, 'Expense', 'Total expenses']].values
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


    ##add the objective of all seasons - these should both be the same if the pnl is being calculated correctly
    ###have to calc here after the step above so it doesnt turn to nan
    pnl.loc[idx['Weighted obj - AFO', '', '', '', ''], 'Full year'] = f_profit(lp_vars, r_vals, option=1)
    pnl.loc[idx['Weighted obj - PNL', '', '', '', ''], 'Full year'] = np.sum(season_obj_qsz * r_vals['zgen']['z_prob_qsz'].ravel())

    ##weight the qsz axis if option 2
    if option==2:
        keys_q = r_vals['zgen']['keys_q']
        keys_s = r_vals['zgen']['keys_s']
        keys_z = r_vals['zgen']['keys_z']
        index_qsz = pd.MultiIndex.from_product([keys_q, keys_s, keys_z])
        z_prob_qsz = r_vals['zgen']['z_prob_qsz']
        z_prob_qsz = pd.Series(z_prob_qsz.ravel(), index=index_qsz)
        z_prob_qsz = z_prob_qsz.reindex(pnl.index, axis=0)
        pnl = pnl.mul(z_prob_qsz, axis=0).groupby(level=(-2,-1), axis=0).sum()
        ###add the objective of all seasons - need to do again because it becomes nan in the step above
        pnl.loc[idx['Weighted obj - AFO', ''], 'Full year'] = f_profit(lp_vars, r_vals, option=1)
        pnl.loc[idx['Weighted obj - PNL', ''], 'Full year'] = np.sum(season_obj_qsz * r_vals['zgen']['z_prob_qsz'].ravel())

    ##round numbers in df
    pnl = pnl.astype(float).round(1).fillna(0)  # have to go to float so rounding works

    return pnl


def f_profit(lp_vars, r_vals, option=0):
    '''returns profit
    0- Profit = rev - (exp + dep)
    1- Risk neutral objective = rev - (exp + minroe + asset_cost +dep).
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
        minroe = np.sum(minroe_qsp7z[:,:,-1,:] * prob_qsz)  #take end slice of season stages
        asset_cost = np.sum(asset_cost_qsp7z[:,:,-1,:] * prob_qsz) #take end slice of season stages
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
        len_p7 = len(keys_p7)
        len_c1 = len(keys_c1)
        len_q = len(keys_q)
        len_s = len(keys_s)
        len_z = len(keys_z)
        qsc1p7z = len_q, len_s, len_c1, len_p7, len_z
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
        trade_value_qsz = trade_value_qszp7[...,-1]
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


def f_stock_pasture_summary(lp_vars, r_vals, build_df=True, keys=None, type=None, index=[], cols=[], arith=0,
                            prod=np.array([1]), na_prod=[], weights=None, na_weights=[], axis_slice={},
                            na_denweights=[], den_weights=1, na_prodweights=[], prod_weights=1):
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
    :key prod_weights (optional, default = 1): str: keys to r_vals referencing array used to weight production.
    :key na_prodweights (optional, default = []): list: position to add new axis
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: summary of a numpy array in a pandas table.
    '''
    keys_key = keys

    ##read from stock reshape function
    if type == 'stock':
        vars = f_stock_reshape(lp_vars, r_vals)
        r_vals = r_vals['stock']
        ###keys that will become the index and cols for table
        keys = r_vals[keys_key]
    else:
        vars = f_feed_reshape(lp_vars, r_vals)
        r_vals = r_vals[type]
        ###keys that will become the index and cols for table
        keys = vars[keys_key]

    ##An error here means the key provided for weights does not exist in lp_vars
    ###using None as the default for weights so that an error is generated later if an Arith option is selected that requires weights
    if weights is not None:
        weights = vars[weights]
        ###set weights to 0 if very small number (otherwise it can show up in report when it shouldn't)
        weights[np.isclose(weights, 0)] = 0

    ##initialise prod array from either r_vals or default value (this means you can perform arith with any number - mainly used for pasture when there is no production param)
    if isinstance(prod, str):
        prod = r_vals[prod]
    # else:
    #     prod = np.array([prod])     #this was adding another axis if an array was passed in
    ###set prod and weights to 0 if very small number (otherwise it can show up in report when it shouldn't)
    prod[np.isclose(prod, 0)] = 0

    ##initialise prod_weight array from either r_vals or default value
    if isinstance(prod_weights, str):
        prod_weights = r_vals[prod_weights]
    else:
        prod_weights = np.array([prod_weights])

    ##den weight - used in weighted average calc (default is 1)
    if isinstance(den_weights, str):
        den_weights = r_vals[den_weights]

    ##other manipulation
    prod, weights, den_weights, prod_weights = f_add_axis(prod, na_prod, prod_weights, na_prodweights, weights, na_weights, den_weights, na_denweights)
    prod, prod_weights, weights, den_weights, keys = f_slice(prod, prod_weights, weights, den_weights, keys, arith, axis_slice)
    ##perform arith. if an axis is not reported it is included in the arith and the axis disappears
    report_idx = index + cols
    arith_axis = list(set(range(len(prod.shape))) - set(report_idx))
    prod = f_arith(prod, prod_weights, weights, den_weights, arith, arith_axis)
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
        prod_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_prodweights = [0,1]
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = None
            na_weights = []
            arith = 4
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ###scan percent
    elif option == 2:
        prod = 'nfoet_scan_k2tva1nw8ziyg1'
        na_prod = [0,1]
        prod_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_prodweights = [0,1]
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = None
            na_weights = []
            arith = 4
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ###dry propn
    elif option == 3:
        prod = 'n_drys_k2tva1nw8ziyg1'
        na_prod = [0,1]
        prod_weights = 'n_mated_k2Tva1nw8ziyg1'
        na_prodweights = [0,1]
        if lp_vars_inc:
            weights = 'dams_numbers_qsk2tvanwziy1g1'
            na_weights = []
            arith = 1
        else:
            weights = None
            na_weights = []
            arith = 4
        keys = 'dams_keys_qsk2tvanwziy1g1'

    ##calcs for survival
    if option == 0:
        ##colate the lp and report vals using f_stock_pasture_summary
        numerator, keys_sliced = f_stock_pasture_summary(lp_vars, r_vals, build_df=False, type=type
                                , prod=prod, na_prod=na_prod, weights=weights, na_weights=na_weights
                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
        denominator, keys_sliced = f_stock_pasture_summary(lp_vars, r_vals, build_df=False, type=type
                                , prod=prod2, na_prod=na_prod2, weights=weights, na_weights=na_weights
                                , keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
        prog_alive_qsk2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(numerator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7) #b9 axis is shorten b axis: [0,1,2,3]
        prog_born_qsk2tvpa1e1b1nw8ziyg1 = np.moveaxis(np.sum(denominator[...,na] * r_vals['stock']['mask_b1b9_preg_b1nwziygb9'], axis=-8), -1, -7)
        percentage = fun.f_divide(prog_alive_qsk2tvpa1e1b1nw8ziyg1, prog_born_qsk2tvpa1e1b1nw8ziyg1)
        ##make table
        percentage = f_numpy2df(percentage, keys_sliced, index, cols)

    ##calc for wean % or scan % or dry %
    else:
        ### The columns that describe an individual (initial) animal (a, y & k2- which was e&b) need to be summed
        ### after the calculation, because the calculation (prod * prod_weights) returns the n_yatf, n_foet or # dry
        ### per dam mated. These axes comprise a single dam mated and therefore must be summed rather than averaged.
        ### These axes are excluded from the calculation by adding them to the columns being reported
        cols_report = cols.copy()
        k2_pos=2
        a1_pos=5
        y_pos=10
        cols_summed = [k2_pos, a1_pos, y_pos]
        cols = list(set(cols_report) | set(cols_summed))
        intermediate, keys_sliced  = f_stock_pasture_summary(lp_vars, r_vals, build_df=False, type=type
                                    , prod=prod, na_prod=na_prod, prod_weights=prod_weights, na_prodweights=na_prodweights
                                    , weights=weights, na_weights=na_weights, keys=keys, arith=arith, index=index
                                    , cols=cols, axis_slice=axis_slice)
        ###sum k2, a & y if not reported and make table
        cols_summed = list(set(cols_summed) - set(cols_report))
        if cols_summed:    #test that this at least one element in the columns to be summed
            intermediate = np.sum(intermediate, axis=tuple(cols_summed), keepdims=True)
        percentage = f_numpy2df(intermediate, keys_sliced, index, cols_report)

    return percentage

def f_feed_budget(lp_vars, r_vals, option=0, nv_option=0, dams_cols=[], offs_cols=[]):
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
    :return: pandas df
    '''
    ##mei supply
    ###grn pasture
    type = 'pas'
    prod = 'me_cons_grnha_fgop6lzt'
    na_prod = [0, 1]  # q,s
    weights = 'greenpas_ha_qsfgop6lzt'
    keys = 'keys_qsfgop6lzt'
    arith = 2
    index = [0,1,7,5,2] #[q,s,z,p6,nv]
    cols = [8] #t
    grn_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    grn_mei = pd.concat([grn_mei], keys=['Green Pas'], axis=1)  # add feed type as header

    ###poc pasture
    type = 'pas'
    prod = 'poc_md_fp6z'
    na_prod = [0, 1, 4]  # q,s,l
    weights = 'poc_consumed_qsfp6lz'
    keys = 'keys_qsfp6lz'
    arith = 2
    index = [0,1,5,3,2] #[q,s,z,p6,nv]
    cols = []
    poc_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    poc_mei.columns = ['Green Crop Paddock Pas'] # add feed type as header

    ###dry pasture
    type = 'pas'
    prod = 'dry_mecons_t_fdp6zt'
    na_prod = [0, 1, 6]  # q,s,l
    weights = 'drypas_consumed_qsfdp6zlt'
    keys = 'keys_qsfdp6zlt'
    arith = 2
    index = [0,1,5,4,2] #[q,s,z,p6,nv]
    cols = [7] #t
    dry_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    dry_mei = pd.concat([dry_mei], keys=['Dry Pas'], axis=1)  # add feed type as header

    ###nap pasture
    type = 'pas'
    prod = 'dry_mecons_t_fdp6zt' #nap is same md as dry pasture
    na_prod = [0, 1]  # q,s
    weights = 'nap_consumed_qsfdp6zt'
    keys = 'keys_qsfdp6zt'
    arith = 2
    index = [0,1,5,4,2] #[q,s,z,p6,nv]
    cols = []
    nap_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                         keys=keys, arith=arith, index=index, cols=cols)
    nap_mei.columns = ['Dry Crop Paddock Pas'] # add feed type as header

    ###residue
    prod = 'md_zp6fks1'
    na_prod = [0, 1, 7]  # q,s, s2
    type = 'stub'
    weights = 'stub_qszp6fks1s2'
    keys = 'keys_qszp6fks1s2'
    arith = 2
    index = [0, 1, 2, 3, 4]  # q,s,z,p6,nv
    cols = []
    axis_slice = {}
    # axis_slice[0] = [0, 2, 1]
    res_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols, axis_slice=axis_slice)
    res_mei.columns = ['Crop Residue'] # add feed type as header

    ###crop graze
    prod = 'crop_md_fkp6p5zl'
    na_prod = [0, 1]  # q,s
    type = 'crpgrz'
    weights = 'crop_consumed_qsfkp6p5zl'
    keys = 'keys_qsfkp6p5zl'
    arith = 2
    index = [0, 1, 6, 4, 2]  # q,s,z,p6,nv
    cols = []
    crop_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    crop_mei.columns = ['Crop Graze'] # add feed type as header

    ###saltbush (just the saltbush not the understory)
    prod = 'sb_me_zp6f'
    na_prod = [0, 1, 5]  # q,s
    type = 'slp'
    weights = 'v_tonnes_sb_consumed_qszp6fl'
    keys = 'keys_qszp6fl'
    arith = 2
    index = [0, 1, 2, 3, 4]  # q,s,z,p6,nv
    cols = []
    sb_mei = f_stock_pasture_summary(lp_vars, r_vals, prod=prod, na_prod=na_prod, type=type, weights=weights,
                                          keys=keys, arith=arith, index=index, cols=cols)
    sb_mei.columns = ['Saltbush'] # add feed type as header

    ###sup
    sup_md_tonne_fkp6z = r_vals['sup']['md_tonne_fkp6z']
    grain_fed_qszkfp6 = f_grain_sup_summary(lp_vars, r_vals, option=3)
    sup_mei_qs_fkp6z = grain_fed_qszkfp6.unstack([4,3,5,2]).sort_index(axis=1).mul(sup_md_tonne_fkp6z, axis=1)
    sup_mei_qszp6f = sup_mei_qs_fkp6z.stack([3,2,0]).sort_index(axis=1).sum(axis=1)
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
    weights = 'sire_numbers_qsg0'
    na_weights = [2, 3, 4] #p6, f, z
    den_weights = 'stock_days_p6fzg0'
    na_denweights = [0, 1]  # q,s
    keys = 'sire_keys_qsp6fzg0'
    index = [0, 1, 4, 2, 3]  # [q,s,z,p6,nv]
    cols = []
    mei_sire = f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_sire = pd.concat([mei_sire], keys=['Sire'], axis=1) # add stock type as header

    ###dams
    type = 'stock'
    prod = 'mei_dams_k2p6ftova1nw8ziyg1'
    na_prod = [0, 1]  # q,s
    weights = 'dams_numbers_qsk2tvanwziy1g1'
    na_weights = [3, 4, 6] #p6, f, o
    den_weights = 'stock_days_k2p6ftova1nwziyg1'
    na_denweights = [0, 1]  # q,s, o
    keys = 'dams_keys_qsk2p6ftovanwziy1g1'
    index = [0, 1, 11, 3, 4]  # [q,s,z,p6,nv]
    cols = dams_cols
    mei_dams = f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_dams = pd.concat([mei_dams], keys=['Dams'], axis=1) # add stock type as header

    ###offs
    type = 'stock'
    prod = 'mei_offs_k3k5p6ftsvnw8ziaxyg3'
    na_prod = [0, 1]  # q,s
    weights = 'offs_numbers_qsk3k5tvnwziaxyg3'
    na_weights = [4, 5, 7] #p6, f, shear
    den_weights = 'stock_days_k3k5p6ftsvnwziaxyg3'
    na_denweights = [0, 1]  # q,s
    keys = 'offs_keys_qsk3k5p6ftsvnwziaxyg3'
    index = [0, 1, 11, 4, 5]  # [q,s,z,p6,nv]
    cols = offs_cols
    mei_offs = f_stock_pasture_summary(lp_vars, r_vals, type=type, prod=prod, na_prod=na_prod, weights=weights,
                                                 na_weights=na_weights, den_weights=den_weights,
                                                 na_denweights=na_denweights, keys=keys, arith=arith,
                                                 index=index, cols=cols)
    mei_offs = pd.concat([mei_offs], keys=['Offs'], axis=1)

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
            arrays[array].columns = pd.MultiIndex.from_product([arrays[array].columns, ['']])
            # arrays[array] = pd.concat([arrays[array]], keys=[''], axis=1)
    feed_budget_supply = pd.concat(arrays[0:8], axis=1).round(1) #round so that little numbers don't cause issues
    feed_budget_req = pd.concat(arrays[8:], axis=1).round(1) #round so that little numbers don't cause issues

    ##sum nv axis if nv_option is 1
    if nv_option==1:
        feed_budget_supply = feed_budget_supply.groupby(axis=0, level=(0,1,2,3)).sum()
        feed_budget_req = feed_budget_req.groupby(axis=0, level=(0,1,2,3)).sum()

    ###if option 1 - calc propn of mei from each feed source
    if option==0:
        feed_budget_supply = feed_budget_supply.div(feed_budget_supply.sum(axis=1), axis=0).fillna(0)
    else:
        days_zp6 = pd.DataFrame(r_vals['pas']['days_p6z'], index=r_vals['pas']['keys_p6'], columns=r_vals['zgen']['keys_z']).T.stack()
        feed_budget_supply = feed_budget_supply.unstack([0,1,-1]).div(days_zp6, axis=0).stack([-3,-2,-1]).reorder_levels([2,3,0,1,4])
        feed_budget_req = feed_budget_req.unstack([0,1,-1]).div(days_zp6, axis=0).stack([-3,-2,-1]).reorder_levels([2,3,0,1,4])

    ###add stock mei requirement
    feed_budget = pd.concat([feed_budget_supply, feed_budget_req], axis=1)

    ##add fp date to index
    keys_p6 = r_vals['pas']['keys_p6']
    keys_z = r_vals['zgen']['keys_z']
    fp_dates = r_vals['pas']['fp_date_start_p6z']
    fp_dates = pd.DataFrame(fp_dates, index=keys_p6, columns=keys_z).T.stack()
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


def f_grazing_summary(lp_vars, r_vals):
    '''
    Green pasture grazing summary
    :return: data table with zp6lo as index and g and column. Containing the hectares of each activity, average foo and starting foo.
    '''
    feed_vars = f_feed_reshape(lp_vars, r_vals)
    greenpas_ha_qsfgop6lzt = feed_vars['greenpas_ha_qsfgop6lzt']
    foo_ave_grnha_gop6lzt = r_vals['pas']['foo_ave_grnha_gop6lzt']
    foo_start_grnha_op6lzt = r_vals['pas']['foo_start_grnha_op6lzt']
    foo_gi_gt = r_vals['pas']['i_foo_graze_propn_gt']

    ##calc average foo for f (nv pool), o (start foo) and g (grazing int)
    foo_ave_grnha_qsp6lzt = fun.f_weighted_average(foo_ave_grnha_gop6lzt, greenpas_ha_qsfgop6lzt, axis=(2,3,4))

    ##sum f axis to return total ha of each pasture activity
    greenpas_ha_qsgop6lzt = np.sum(greenpas_ha_qsfgop6lzt, axis=2)

    ##combine everything
    graze_info_iqsgop6lzt = np.stack(np.broadcast_arrays(greenpas_ha_qsgop6lzt, foo_start_grnha_op6lzt[na,na,na], foo_ave_grnha_qsp6lzt[:,:,na,na,...]), axis=0)

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


def f_add_axis(prod, na_prod, prod_weights, na_prodweights, weights, na_weights, den_weights, na_denweights):
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
    return prod, weights, den_weights, prod_weights


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
    ### if arith is being conducted these arrays need to be the same size so slicing can work
    prod, prod_weights, weights, den_weights = np.broadcast_arrays(prod, prod_weights, weights, den_weights)
    prod = prod[tuple(sl)]
    prod_weights = prod_weights[tuple(sl)]
    weights = weights[tuple(sl)]
    den_weights = den_weights[tuple(sl)]
    return prod, prod_weights, weights, den_weights, keys


def f_arith(prod, prod_weights, weight, den_weights, arith, axis):
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
        prod = fun.f_weighted_average(prod, weight, tuple(axis), keepdims=keepdims, den_weights=den_weights)
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
