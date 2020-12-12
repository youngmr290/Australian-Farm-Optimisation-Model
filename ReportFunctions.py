# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (incase the inputs are adjusted during the exp so they will not be correct for r_valsing)
When creating r_vals values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import Functions as fun
import Exceptions as exc
na=np.newaxis


def f_errors(r_vals, exp_data_index, trial_outdated, trials):
    ##first check if data exists for each desired trial
    try:
        for row in trials:
            r_vals[exp_data_index[row][2]]
    except KeyError:
        raise exc.TrialException
    ##second check if generating results using out of date data.
    if any(trial_outdated.loc[exp_data_index[trials]]): #have to use the trial name because the order is different
        print('''

              Generating reports from out dated data

              ''')
    return

def f_make_table(data, index, header):
    '''function to return table
    ^currently just returns a df but there are python packages which make nice tables'''
    return pd.DataFrame(data, index=index, columns=header)

def produce_df(data, rows, columns, row_names=None, column_names=None):
    """rows is a list of lists that will be used to build a MultiIndex
    columns is a list of lists that will be used to build a MultiIndex"""
    row_index = pd.MultiIndex.from_product(rows, names=row_names)
    col_index = pd.MultiIndex.from_product(columns, names=column_names)
    return pd.DataFrame(data, index=row_index, columns=col_index)

#################
# Final reports #
#################

def f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, **kwargs):
    '''
    Returns dataframe for sepecified function. Multiple trials result in a stacked table with trial name as index level.

    :param func: report function whose return value is to be stacked
    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :param trial_outdated: series indicating which trials are outdated
    :param exp_data_index: trial names - in the same order as exp.xlsx
    :param trials: trials to return info for
    :param kwargs: args for specified function. This is optional.
    '''
    ##check for errors
    try:
        f_errors(r_vals, exp_data_index, trial_outdated, trials)
    except exc.TrialError:
        print('''Trials for reporting dont all exist''')
        return
    ##loop through trials and generate pnl table
    try:
        result_stacked = pd.DataFrame() #create df to append pnl table from each trial
        for row in trials:
            result = func(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], **kwargs)
            result = pd.concat([result], keys=[exp_data_index[row][2]], names=['Trial']) #add trial name as index level
            result_stacked = result_stacked.append(result)
    except exc.ArithError:
        print('''Arith error: can't preform operation along an axis that is going to be reported as the index or col''')
        return
    except exc.AxisError:
        print('''Axis error: active axes exist that are not used in arith or being reported as index or columns''')
        return
    return result_stacked

def f_croparea_profit(lp_vars, r_vals, trial_outdated, exp_data_index, trials, area_option, profit_option):
    '''returns graph of crop area (x - axis) by profit (y - axis)
    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :param trial_outdated: series indicating which trials are outdated
    :param exp_data_index: trial names - in the same order as exp.xlsx
    :param trials: trials to return info for
    :param area_option:
            3: total pasture area
            4: total crop area
    :param profit_option:
            0: profit = rev - (exp + minroe + asset_opp +dep)
            1: profit = rev - (exp + dep)
    '''
    ##check for errors
    try:
        f_errors(r_vals, exp_data_index, trial_outdated, trials)
    except exc.TrialError:
        print('''Trials for reporting dont all exist''')
        return
    ##loop through trials and generate pnl table
    profit = [] #create list to append pnl table from each trial
    area = [] #create list to append pnl table from each trial
    for row in trials:
        profit.append(f_profit(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], option=profit_option))
        area.append(f_area_summary(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], option=area_option))
    plt.plot(area, profit)
    plt.show()

# def f_saleprice(r_vals, trial_outdated, exp_data_index, trials, option, grid, weight, fs):
#     '''Returns price summaries
#     :param r_vals: dict - report variable
#     :param trial_outdated: series indicating which trials are outdated
#     :param exp_data_index: trial names - in the same order as exp.xlsx
#     :param trials: trials to return info for
#     :param option:
#             0: farmgate grain price
#             1: wool price STB price for FNF (free or nearly free of fault)
#             2: sale price for specified grid at given weight and fat score
#     :param grid: list - sale prices grids you want to view the price for
#     :param weight: float/int - weight you want to view the price for
#     :param fs: int - fat score you want to view the price for
#     :return: df
#     '''
#
#     ##check for errors
#     try:
#         f_errors(r_vals, exp_data_index, trial_outdated, trials)
#     except exc.TrialError:
#         print('''Trials for reporting dont all exist''')
#         return
#     ##loop through trials and generate pnl table
#     price_stacked = pd.DataFrame() #create df to append pnl table from each trial
#     for row in trials:
#         price = f_price_summary(r_vals[exp_data_index[row][2]], **kwargs)
#         price = pd.concat([price], keys=[exp_data_index[row][2]], names=['Trial']) #add trial name as index level
#         price_stacked.append(price)



###################
# input summaries #
###################

def f_price_summary(lp_vars, r_vals, **kwargs):
    '''Returns price summaries
    :param r_vals:
    :key option:
            0- farmgate grain price
            1- wool price STB price for FNF (free or nearly free of fault)
            2- sale price for specified grid at given weight and fat score
    :key grid: list - sale grids to report. Has to be int between 0 and 7 inclusive.
    :key weight: float/int - stock weight to report price for.
    :key fs: int - fat score to report price for. Has to be number between 1-5 inclusinve.
    :return: df
    '''
    ##unpack kwargs
    option = kwargs['option']
    grid = kwargs['grid']
    weight = kwargs['weight']
    fs = kwargs['fs']

    ##grain price - farmgate (price recieved by farmer)
    if option==0:
        return r_vals['crop']['farmgate_price']

    ##wool price - grid price
    if option==1:
        return pd.Series(r_vals['woolp_mpg_w4'], index= r_vals['fd_range'])

    ##sale price - grid price
    if option==2:
        ###create dataframe
        sale_index = pd.MultiIndex(levels=[[], [], []],
                                 codes=[[], [], []],
                                 names=['Grid', 'Weight', 'Fat Score'])
        saleprice = pd.DataFrame(index=sale_index, columns=['Price $/kg', 'Price $/hd']) #need to initilise df with multiindex so rows can be added

        grid_price_s7s5s6 = r_vals['grid_price_s7s5s6']
        weight_range_s7s5 = r_vals['weight_range_s7s5']
        grid_keys = r_vals['salegrid_keys']
        for t_grid, t_weight, t_fs in zip(grid, weight, fs):
            ##grid name - used in table index
            grid_name = grid_keys[t_grid]
            ##index grid and fs
            price_s5 = grid_price_s7s5s6[t_grid, :, t_fs]
            ##interpolate to get price for specified weight
            lookup_weights = weight_range_s7s5[t_grid,:]
            price = np.interp(t_weight, lookup_weights, price_s5)
            ##attach to df
            ###if price is less than 10 it is assumed to be $/kg else $/hd
            if price < 10:
                col = 'Price $/kg'
            else:
                col = 'Price $/hd'
            saleprice[(grid_name, t_weight, t_fs), col] = price
        return saleprice


#########################################
# intermidiate report building functions#
#########################################


def f_rotation(lp_vars, r_vals):
    '''
    manipulates the rotation solution into usable format. This is used in many function.
    '''
    ##rotation
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(5, append=True)  # add landuse as index level
    rot_area_rl = pd.Series(lp_vars['v_phase_area']).sort_index() #create a series of all the phase areas, need to sort the index because it was chuck error for some calculations
    rot_area_rkl = rot_area_rl.unstack().reindex(phases_rk.index, axis=0, level=0).stack() #add landuse to the axis
    return phases_rk, rot_area_rl, rot_area_rkl

def f_area_summary(lp_vars, r_vals, option=0):
    '''
    Rotation & landuse area summary. With multiple output levels.
    return options:
    0- tuple: all results wraped in tuple
    1- table: all rotations by lmu
    2- table: selected rotations by lmu
    3- float: total pasture area
    4- float: total crop area
    5- table: crop and pasture area by lmu

    '''
    rot_area_rl, rot_area_rkl = f_rotation(lp_vars, r_vals)[1:3]
    landuse_area_kl = rot_area_rkl.sum(axis=0,level=(1,2)).unstack() #area of each landuse (sum lmu and rotation)
    ##all rotations by lmu
    rot_area_rl = rot_area_rl.unstack()
    if option==1:
        return rot_area_rl
    ##selected rotations by lmu
    rot_area_selected_rl = rot_area_rl[rot_area_rl.any(axis=1)]
    if option==2:
        return rot_area_selected_rl
    ###crop & pasture area
    ####you can now use isin pasture or crop sets to calc the area of crop or pasture
    all_pas = r_vals['rot']['all_pastures'] #landuse sets
    pasture_area_l = landuse_area_kl[landuse_area_kl.index.isin(all_pas)].sum() #sum landuse
    if option==3:
        return pasture_area_l.sum()
    crop_area_l = landuse_area_kl[~landuse_area_kl.index.isin(all_pas)].sum() #sum landuse
    if option==4:
        return crop_area_l.sum()
    ##crop & pasture area by lmu
    croppas_area_l = pd.DataFrame()
    croppas_area_l.loc['pasture'] = pasture_area_l
    croppas_area_l.loc['crop'] = crop_area_l
    if option==5:
        return croppas_area_l
    ##return all if option==0
    if option==0:
        return rot_area_rl, rot_area_selected_rl, pasture_area_l, crop_area_l, croppas_area_l

def f_mach_summary(lp_vars, r_vals, option=0):
    '''
    Machine summary.
    return options:
    0- table: total machine cost for each crop in each cash period

    '''
    ##call rotation functin to get rotation info
    phases_rk, rot_area_rl = f_rotation(lp_vars, r_vals)[0:2]
    ##harv
    contractharv_hours = pd.Series(lp_vars['v_contractharv_hours'])
    harv_hours = pd.Series(lp_vars['v_harv_hours']).sum(level=1) #sum p5 axis
    harvest_cost = r_vals['mach']['contract_harvest_cost'].mul(contractharv_hours, axis=1)  + r_vals['mach']['harvest_cost'].mul(harv_hours, axis=1)
    ##seeding
    seeding_days = pd.Series(lp_vars['v_seeding_machdays']).sum(level=(1,2)) #sum labour period axis
    contractseeding_ha = pd.Series(lp_vars['v_contractseeding_ha']).sum(level=1) #sum labour period and lmu axis
    seeding_ha = r_vals['mach']['seeding_rate'].mul(seeding_days.unstack()).stack() #note seeding ha wont equal the rotation area because arable area is included in seed_ha.
    seeding_cost_own = r_vals['mach']['seeding_cost'].reindex(seeding_ha.index, axis=1,level=1).mul(seeding_ha,axis=1).sum(axis=1, level=0) #sum lmu axis
    contractseed_cost_ha = r_vals['mach']['contractseed_cost']
    idx = pd.MultiIndex.from_product([contractseed_cost_ha.index, contractseeding_ha.index])
    seeding_cost_contract = contractseed_cost_ha.reindex(idx, level=0).mul(contractseeding_ha, level=1).unstack()
    ##fert & chem mach cost
    fertchem_cost_rlc = pd.concat([r_vals['crop']['fert_app_cost'], r_vals['crop']['nap_fert_app_cost'], r_vals['crop']['chem_app_cost_ha']],axis=1).sum(axis=1,level=0) #cost per ha
    fertchem_cost_rc = fertchem_cost_rlc.mul(rot_area_rl,axis=0).sum(axis=0,level=0) #mul area and sum lmu
    fertchem_cost_kc = fertchem_cost_rc.reindex(phases_rk.index,axis=0,level=0).sum(axis=0,level=1) #reindex to include landuse and sum rot
    ##conbime all costs
    exp_mach_kc = pd.concat([fertchem_cost_kc.T, seeding_cost_own, seeding_cost_contract, harvest_cost],axis=0).sum(axis=0,level=0).T
    ##return all if option==0
    if option==0:
        return exp_mach_kc

def f_grain_sup_summary(lp_vars, r_vals):
    ##create dict to store grain variables
    grain = {}

    ##prices
    grains_sale_price_kgc = r_vals['crop']['grain_price'].T.stack()
    grains_buy_price_kgc = r_vals['sup']['buy_grain_price'].T.stack()

    ##grain purchased
    grain_purchased_kg = pd.Series(lp_vars['v_buy_grain'])

    ##grain sold
    grain_sold_kg = pd.Series(lp_vars['v_sell_grain'])

    ##grain fed
    grain_fed_kg = pd.Series(lp_vars['v_sup_con']).sum(level=(0,1)) #sum feed pool and feed period
    grain_fed_kp5 = pd.Series(lp_vars['v_sup_con']).sum(level=(0,3)).swaplevel() #sum feed pool and grain pool

    ##total grain produced by crop enterprise
    total_grain_produced = grain_sold_kg + grain_fed_kg - grain_purchased_kg #total grain produced by crop enterprise
    rev_grain_c = grains_sale_price_kgc.mul(total_grain_produced.reindex(grains_sale_price_kgc.index), axis=0).sum(axis=0,level=0) #sum grain pool, have to reindex (not really sure why since it is the same index - maybe one has been condensed ie index with nan removed)
    grain['rev_grain_c'] = rev_grain_c

    ##supplementary cost: cost = sale_price * (grain_fed - grain_purchased) + buy_price * grain_purchased
    sup_exp_c = (grains_sale_price_kgc.mul((grain_fed_kg - grain_purchased_kg).reindex(grains_sale_price_kgc.index), axis=0)
                + grains_buy_price_kgc.mul(grain_purchased_kg.reindex(grains_buy_price_kgc.index), axis=0)).sum(axis=0,level=0) #sum grain pool
    grain['sup_exp_c'] = sup_exp_c
    return grain

def f_crop_summary(lp_vars, r_vals, option=0):
    '''
    Crop summary. Includes pasture inputs.
    return options:
    0- tuple: fert cost, chem cost, miscilaneous costs and grain revenue for each landuse

    '''
    ##call rotation functin to get rotation info
    phases_rk, rot_area_rl = f_rotation(lp_vars, r_vals)[0:2]
    ##expenses
    ###fert
    exp_fert_ha_rlc = pd.concat([r_vals['crop']['phase_fert_cost'], r_vals['crop']['nap_phase_fert_cost']],axis=1).sum(axis=1,level=0)
    exp_fert_rc = exp_fert_ha_rlc.mul(rot_area_rl,axis=0).sum(axis=0,level=0) #mul area and sum lmu
    exp_fert_kc = exp_fert_rc.reindex(phases_rk.index,axis=0,level=0).sum(axis=0,level=1) #reindex to include landuse and sum rot
    ###chem
    exp_chem_rc = r_vals['crop']['chem_cost'].mul(rot_area_rl,axis=0).sum(axis=0,level=0) #mul area and sum lmu
    exp_chem_kc = exp_chem_rc.reindex(phases_rk.index,axis=0,level=0).sum(axis=0,level=1) #reindex to include landuse and sum rot
    ###misc
    misc_exp_ha_rlc = pd.concat([r_vals['crop']['stub_cost'], r_vals['crop']['insurance_cost'], r_vals['crop']['seedcost']],axis=1).sum(axis=1,level=0) #stubble, seed & insurance
    misc_exp_rc = misc_exp_ha_rlc.reindex(rot_area_rl.index,axis=0).mul(rot_area_rl,axis=0).sum(axis=0,level=0) #mul area and sum lmu
    misc_exp_kc = misc_exp_rc.reindex(phases_rk.index,axis=0,level=0).sum(axis=0,level=1) #reindex to include landuse and sum rot
    ##revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    rev_grain_c = grain_summary['rev_grain_c']
    ##return all if option==0
    if option==0:
        return exp_fert_kc, exp_chem_kc, misc_exp_kc, rev_grain_c

def f_stock_reshape(lp_vars, r_vals):
    '''
    Stock reshape. Gets everything into the correct shape.
    Returns a dictionary with stock params.
    '''
    ##keys
    keys_c = r_vals['fin']['keys_c']
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
    keys_z = r_vals['stock']['keys_z']
    keys_p6 = r_vals['stock']['keys_p6']
    keys_p5 = r_vals['lab']['keys_p5']
    keys_pastures = r_vals['pas']['keys_pastures']

    ##axis len
    len_c = len(keys_c)
    len_a = len(keys_a)
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

    ##create dict to stick reshaped variable is
    stock_vars = {}

    #store keys - must be in axis order
    stock_vars['sire_keys_g0'] = [keys_g0]
    stock_vars['dams_keys_k2tvanwziy1g1'] = [keys_k2, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    stock_vars['offs_keys_k3k5tvnwziaxy1g3'] = [keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ##animal numbers
    ###shapes
    sire_shape = len_g0
    dams_shape = len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offs_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    ###sire
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    sire_numbers_g0 = sire_numbers.reshape(sire_shape)
    sire_numbers_g0[sire_numbers_g0==None] = 0 #replace None with 0
    stock_vars['sire_numbers_g0'] = sire_numbers_g0
    ###dams
    dam_numbers = np.array(list(lp_vars['v_dams'].values()))
    dam_numbers_k2tvanwziy1g1 = dam_numbers.reshape(dams_shape)
    dam_numbers_k2tvanwziy1g1[dam_numbers_k2tvanwziy1g1==None] = 0 #replace None with 0
    stock_vars['dam_numbers_k2tvanwziy1g1'] = dam_numbers_k2tvanwziy1g1
    ###offs
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    offs_numbers_k3k5tvnwziaxyg3 = offs_numbers.reshape(offs_shape)
    offs_numbers_k3k5tvnwziaxyg3[offs_numbers_k3k5tvnwziaxyg3==None] = 0 #replace None with 0
    stock_vars['offs_numbers_k3k5tvnwziaxyg3'] = offs_numbers_k3k5tvnwziaxyg3

    ##dse
    ###shape
    siredse_shape = len_p6, len_g0
    damsdse_shape = len_k2, len_p6, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offsdse_shape = len_k3, len_k5, len_p6, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    ###nw dse
    stock_vars['dsenw_p6g0'] = r_vals['stock']['dsenw_p6g0'].reshape(siredse_shape)
    stock_vars['dsenw_k2p6tva1nwziyg1'] = r_vals['stock']['dsenw_k2p6tva1nwziyg1'].reshape(damsdse_shape)
    stock_vars['dsenw_k3k5p6tvnwzixyg3'] = r_vals['stock']['dsenw_k3k5p6tvnwzixyg3'].reshape(offsdse_shape)
    ###mj dse
    stock_vars['dsemj_p6g0'] = r_vals['stock']['dsemj_p6g0'].reshape(siredse_shape)
    stock_vars['dsemj_k2p6tva1mjziyg1'] = r_vals['stock']['dsemj_k2p6tva1nwziyg1'].reshape(damsdse_shape)
    stock_vars['dsemj_k3k5p6tvmjzixyg3'] = r_vals['stock']['dsemj_k3k5p6tvnwzixyg3'].reshape(offsdse_shape)

    ##cfw
    ###cfw per head average for the mob - includes the mortality factor
    stock_vars['cfw_hdmob_g0'] = r_vals['stock']['r_cfw_hdmob_g0'].reshape(sire_shape)
    stock_vars['cfw_hdmob_k2tva1nwziyg1'] = r_vals['stock']['r_cfw_hdmob_k2ctva1nwziyg1'].reshape(dams_shape)
    stock_vars['cfw_hdmob_k3k5tvnwzixyg3'] = r_vals['stock']['r_cfw_hdmob_k3k5ctvnwzixyg3'].reshape(offs_shape)
    ###cfw per head - wool cut for 1 whole animal, no account for mortality (numbers)
    stock_vars['cfw_hd_g0'] = r_vals['stock']['r_cfw_hd_g0'].reshape(sire_shape)
    stock_vars['cfw_hd_k2tva1nwziyg1'] = r_vals['stock']['r_cfw_hd_k2ctva1nwziyg1'].reshape(dams_shape)
    stock_vars['cfw_hd_k3k5tvnwzixyg3'] = r_vals['stock']['r_cfw_hd_k3k5ctvnwzixyg3'].reshape(offs_shape)

    ##husbandry expense
    sirecost_shape = len_c, len_g0
    damscost_shape = len_k2, len_c, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    offscost_shape = len_k3, len_k5, len_c, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3

    stock_vars['sire_cost_cg0'] = r_vals['stock']['cost_cg0'].reshape(sirecost_shape)
    stock_vars['dams_cost_k2ctva1nwziyg1'] = r_vals['stock']['cost_k2ctva1nwziyg1'].reshape(damscost_shape)
    stock_vars['offs_cost_k3k5ctvnwzixyg3'] = r_vals['stock']['cost_k3k5ctvnwzixyg3'].reshape(offscost_shape)
    ###sale income
    stock_vars['salevalue_cg0'] = r_vals['stock']['salevalue_cg0'].reshape(sirecost_shape)
    stock_vars['salevalue_k2ctva1nwziyg1'] = r_vals['stock']['salevalue_k2ctva1nwziyg1'].reshape(damscost_shape)
    stock_vars['salevalue_k3k5ctvnwzixyg3'] = r_vals['stock']['salevalue_k3k5ctvnwzixyg3'].reshape(offscost_shape)
    ###wool income
    stock_vars['woolvalue_cg0'] = r_vals['stock']['woolvalue_cg0'].reshape(sirecost_shape)
    stock_vars['woolvalue_k2ctva1nwziyg1'] = r_vals['stock']['woolvalue_k2ctva1nwziyg1'].reshape(damscost_shape)
    stock_vars['woolvalue_k3k5ctvnwzixyg3'] = r_vals['stock']['woolvalue_k3k5ctvnwzixyg3'].reshape(offscost_shape)
    return stock_vars

def f_stock_cash_summary(lp_vars, r_vals):
    '''
    Returns:
    0- expesnse and revenue items

    '''
    ##get reshaped variable
    stock_vars = f_stock_reshape(lp_vars, r_vals)

    ##numbers
    sire_numbers_g0 = stock_vars['sire_numbers_g0']
    dam_numbers_k2tvanwziy1g1 = stock_vars['dam_numbers_k2tvanwziy1g1']
    offs_numbers_k3k5tvnwziaxyg3 = stock_vars['offs_numbers_k3k5tvnwziaxyg3']

    ##husb cost
    sire_cost_cg0 = stock_vars['sire_cost_cg0'] * sire_numbers_g0
    dams_cost_k2ctva1nwziyg1 = stock_vars['dams_cost_k2ctva1nwziyg1']  * dam_numbers_k2tvanwziy1g1[:,na,...]
    offs_cost_k3k5ctvnwzixyg3 = stock_vars['offs_cost_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxyg3[:,:,na,...]

    ##sale income
    salevalue_cg0 = stock_vars['salevalue_cg0'] * sire_numbers_g0
    salevalue_k2ctva1nwziyg1 = stock_vars['salevalue_k2ctva1nwziyg1'] * dam_numbers_k2tvanwziy1g1[:,na,...]
    salevalue_k3k5ctvnwzixyg3 = stock_vars['salevalue_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxyg3[:,:,na,...]

    ##wool income
    woolvalue_cg0 = stock_vars['woolvalue_cg0'] * sire_numbers_g0
    woolvalue_k2ctva1nwziyg1 = stock_vars['woolvalue_k2ctva1nwziyg1'] * dam_numbers_k2tvanwziy1g1[:,na,...]
    woolvalue_k3k5ctvnwzixyg3 = stock_vars['woolvalue_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxyg3[:,:,na,...]

    ###sum axis to return total income in each cash peirod
    siresale_c = fun.f_reduce_skipfew(np.sum, salevalue_cg0, preserveAxis=0) #sum all axis except c
    damssale_c = fun.f_reduce_skipfew(np.sum, salevalue_k2ctva1nwziyg1, preserveAxis=1) #sum all axis except c
    offssale_c = fun.f_reduce_skipfew(np.sum, salevalue_k3k5ctvnwzixyg3, preserveAxis=2) #sum all axis except c
    sirewool_c = fun.f_reduce_skipfew(np.sum, woolvalue_cg0, preserveAxis=0) #sum all axis except c
    damswool_c = fun.f_reduce_skipfew(np.sum, woolvalue_k2ctva1nwziyg1, preserveAxis=1) #sum all axis except c
    offswool_c = fun.f_reduce_skipfew(np.sum, woolvalue_k3k5ctvnwzixyg3, preserveAxis=2) #sum all axis except c
    stocksale_c = siresale_c + damssale_c + offssale_c
    wool_c = sirewool_c + damswool_c + offswool_c

    sirecost_c = fun.f_reduce_skipfew(np.sum, sire_cost_cg0, preserveAxis=0) #sum all axis except c
    damscost_c = fun.f_reduce_skipfew(np.sum, dams_cost_k2ctva1nwziyg1, preserveAxis=1) #sum all axis except c
    offscost_c = fun.f_reduce_skipfew(np.sum, offs_cost_k3k5ctvnwzixyg3, preserveAxis=2) #sum all axis except c
    stockcost_c = sirecost_c + damscost_c + offscost_c

    ##expenses sup feeding
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    sup_cost_c = grain_summary['sup_exp_c']

    ##infrastructure


    return stocksale_c, wool_c, sirecost_c, stockcost_c, sup_cost_c

def f_pasture_summary():
    ''''''

def f_labour_summary(lp_vars, r_vals, option=0):
    '''
    :return:
    0- total labour cost
    1- amount for each enterprise
    '''
    ##total labour cost
    if option==0:
        cas_cost_pc = r_vals['lab']['casual_cost'].mul(pd.Series(lp_vars['v_quantity_casual']),level=0)
        perm_cost_c = r_vals['lab']['perm_cost'] * pd.Series(lp_vars['v_quantity_perm']).values
        manager_cost_c = r_vals['lab']['manager_cost'] * pd.Series(lp_vars['v_quantity_manager']).values
        total_lab_cost = cas_cost_pc.sum(level=1) + perm_cost_c + manager_cost_c
        return total_lab_cost
    ##labour breakdown for each worker level (table: labour period by worker level)
    if option==1:
        ###sheep
        manager_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_manager']).unstack()
        prem_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_permanent']).unstack()
        casual_sheep_p5w = pd.Series(lp_vars['v_sheep_labour_casual']).unstack()
        sheep_labour = pd.concat([manager_sheep_p5w, prem_sheep_p5w, casual_sheep_p5w], axis=1).sum(axis=1, level=0)
        ###crop
        manager_crop_p5w = pd.Series(lp_vars['v_crop_labour_manager']).unstack()
        prem_crop_p5w = pd.Series(lp_vars['v_crop_labour_permanent']).unstack()
        casual_crop_p5w = pd.Series(lp_vars['v_crop_labour_casual']).unstack()
        crop_labour = pd.concat([manager_crop_p5w, prem_crop_p5w, casual_crop_p5w], axis=1).sum(axis=1, level=0)
        ###fixed
        manager_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_manager']).unstack()
        prem_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_permanent']).unstack()
        casual_fixed_p5w = pd.Series(lp_vars['v_fixed_labour_casual']).unstack()
        fixed_labour = pd.concat([manager_fixed_p5w, prem_fixed_p5w, casual_fixed_p5w], axis=1).sum(axis=1, level=0)
        return sheep_labour, crop_labour, fixed_labour

def f_dep_summary(lp_vars, r_vals):
    keys_c = r_vals['fin']['keys_c']
    len_c = len(keys_c)
    ##dep - depreciation is yearly but for the profit and loss it is equally divided into each cash period
    dep = lp_vars['v_dep'][None]/len_c #convert to dep per cashflow period
    dep_c = pd.Series([dep]*len_c, index=keys_c)  #convert to df with cashflow period as index
    return dep_c

def f_overhead_summary(r_vals):
    ##overheads/fixed expenses
    exp_fix_c = r_vals['fin']['overheads']
    return exp_fix_c




    # df_rot = df_rot.rename_axis(['rot','lmu'])
    # phase_area = pd.merge(r_vals['rot']['phases'], df_rot, how='left', left_index=True, right_on=['rot']) #merge full phase array with area array
    # phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    # inter['pasture_area'] = df_rot[phase_is_pasture].sum()
    # pasture_area_rt = pd.DataFrame(r_vals['pas']['pasture_area_rt'], index=phases_df.index, columns=keys_pastures)
    # inter['pasture_area'] = pasture_area_rt.mul(rot_area,axis=0,level=0).sum(axis=0) #return the area of each pasture type
    # inter['crop_area'] = df_rot[~phase_is_pasture].sum() #^do i have something like pasture already? or do i need to do option 1? how can i get area for each crop set?

def f_dse(inter,method=0,per_ha=False):
    '''

    :param
    inter: dict
    method: int
            0 - dse by normal weight
            1 - dse by mei
    per_ha: Bool
        if true it returns DSE/ha else it returns total dse
    :return DSE per pasture hectare for each sheep group:
    '''
    if method==0:
        ##sire
        dse_sire = inter['sire_numbers_g0'] * inter['dsenw_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'][:,na,...] * inter['dsenw_k2p6tva1nwziyg1'], preserveAxis=1) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxyg3'][:,:,na,...] * inter['dsenw_k3k5p6tvnwzixyg3'], preserveAxis=2) #sum all axis except p6
    else:
        ##sire
        dse_sire = inter['sire_numbers_g0'] * inter['dsemj_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'][:,na,...] * inter['dsemj_k2p6tva1nwziyg1'], preserveAxis=1) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxyg3'][:,:,na,...] * inter['dsemj_k3k5p6tvnwzixyg3'], preserveAxis=2) #sum all axis except p6

    ##dse per ha if user opts for this level of detail
    if per_ha:
        dse_sire = dse_sire/inter['pasture_area']
        dse_dams = dse_dams/inter['pasture_area']
        dse_offs = dse_offs/inter['pasture_area']

    ##turn to table
    dse_sire = f_make_table(dse_sire, inter['keys_p6'], ['Sire DSE'])
    dse_dams = f_make_table(dse_dams, inter['keys_p6'], ['Dams DSE'])
    dse_offs = f_make_table(dse_offs, inter['keys_p6'], ['Offs DSE'])
    return dse_sire, dse_dams, dse_offs

def f_profitloss_table(lp_vars, r_vals):
    '''
    Returns profit and loss statement for selected trials. Multiple trials result in a stacked pnl table.

    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :return: dataframe

    '''
    ##read stuff from other functions that is used in rev and cost section
    exp_fert_kc, exp_chem_kc, misc_exp_kc, rev_grain_kc = f_crop_summary(lp_vars, r_vals, option=0)
    exp_mach_kc = f_mach_summary(lp_vars, r_vals)
    stocksale_c, wool_c, sirecost_c, stockcost_c, sup_cost_c = f_stock_cash_summary(lp_vars, r_vals)
    ##other info required below
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    keys_c = r_vals['fin']['keys_c']

    ##create p/l dataframe
    pnl_index = pd.MultiIndex(levels=[[], []],
                             codes=[[], []],
                             names=['Type', 'Subtype'])
    pnl = pd.DataFrame(index=pnl_index, columns=keys_c) #need to initilise df with multiindex so rows can be added

    ##income
    rev_grain_c = rev_grain_kc.sum(axis=0) #sum landuse axis
    ###add to p/l table each as a new row
    pnl.loc[('Revenue', 'grain'),:] = rev_grain_c
    pnl.loc[('Revenue', 'sheep sales'),:] = stocksale_c
    pnl.loc[('Revenue', 'wool'),:] = wool_c
    pnl.loc[('Revenue', 'Total Revenue'),:] = pnl.loc[pnl.index.get_level_values(0) == 'Revenue'].sum(axis=0)

    ##expenses
    ####machinery
    mach_c = exp_mach_kc.sum(axis=0) #sum landuse
    ####crop & pasture
    pasfert_c = exp_fert_kc[exp_fert_kc.index.isin(all_pas)].sum(axis=0)
    cropfert_c = exp_fert_kc[~exp_fert_kc.index.isin(all_pas)].sum(axis=0)
    paschem_c = exp_chem_kc[exp_chem_kc.index.isin(all_pas)].sum(axis=0)
    cropchem_c = exp_chem_kc[~exp_chem_kc.index.isin(all_pas)].sum(axis=0)
    pasmisc_c = misc_exp_kc[misc_exp_kc.index.isin(all_pas)].sum(axis=0)
    cropmisc_c = misc_exp_kc[~misc_exp_kc.index.isin(all_pas)].sum(axis=0)
    pas_c = pasfert_c + paschem_c + pasmisc_c
    crop_c = cropfert_c + cropchem_c + cropmisc_c
    ####labour
    labour_c = f_labour_summary(lp_vars, r_vals, option=0)
    ####depreciation
    dep_c = f_dep_summary(lp_vars, r_vals)
    ####fixed overhead expenses
    exp_fix_c = f_overhead_summary(r_vals)
    ###add to p/l table each as a new row
    pnl.loc[('Expense', 'Crop'),:] = crop_c
    pnl.loc[('Expense', 'pasture'),:] = pas_c
    pnl.loc[('Expense', 'stock'),:] = stockcost_c
    pnl.loc[('Expense', 'machinery'),:] = mach_c
    pnl.loc[('Expense', 'labour'),:] = labour_c
    pnl.loc[('Expense', 'fixed'),:] = exp_fix_c
    pnl.loc[('Expense', 'depreciation'),:] = dep_c
    pnl.loc[('Expense', 'Total expenses'),:] = pnl.loc[pnl.index.get_level_values(0) == 'Expense'].sum(axis=0)

    ##EBIT
    pnl.loc[('', 'EBIT'),:] = pnl.loc[('Revenue', 'Total Revenue')] - pnl.loc[('Expense', 'Total expenses')]

    ##add a column which is total of all casflow period
    pnl['Full year'] = pnl.sum(axis=1)

    ##round numbers in df
    pnl = pnl.round(1)
    return pnl

def f_profit(lp_vars, r_vals, option=0):
    '''returns profit
    0- rev - (exp + minroe + asset_opp +dep)
    1- rev - (exp + dep)
    '''
    obj_profit = r_vals['profit']
    minroe = pd.Series(lp_vars['v_minroe'])
    asset_opportunity_cost = pd.Series(lp_vars['v_asset'])
    if option==0:
        return obj_profit
    else:
        return obj_profit + minroe - (asset_opportunity_cost * r_vals['opportunity_cost_capital'])


def f_stock_summary(lp_vars, r_vals, **kwargs):
    '''
    Returns summary of a numpy array in a pandas table.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :key prod: str: with key for stock_vars
    :key group: string: sheep group to be reported
    :key arith: int: arithmetic operation used.
            0: production param (nothing)
            1: production multipled by numbers and summed
            2: production weighted average with numbers
    :key arith_axis: list: axis to preform arithmetic operation along.
    :key index: list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols: list: axis you want as the cols of pandas df (order of list is the col level order).
    :key axis_slice: dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: pandas df
    '''
    ##unpack dict
    arith = kwargs['arith']
    arith_axis = kwargs['arith_axis']
    index = kwargs['index']
    cols = kwargs['cols']
    prod_key = kwargs['prod']
    group = kwargs['group']
    axis_slice = kwargs['axis_slice']

    ##read from stock reshape function
    stock_vars = f_stock_reshape(lp_vars, r_vals)
    prod = stock_vars[prod_key]

    ##error handle 1: cant preform arithmetic along an axis and also report that axis and the index or col
    arith_occur = arith == 1 or arith == 2
    arith_error = any(item in index for item in arith_axis) or any(item in cols for item in arith_axis)
    if arith_occur and arith_error:  # if arith is happening and there is an error in selected axis
        raise exc.ArithError

    ##error handle 2: once arith has been completed all axis that are not singleton must be used in either the index or cols
    if arith_occur:
        nonzero_idx = arith_axis + index + cols  # join lists
    else:
        nonzero_idx = index + cols  # join lists
    error = [prod.shape.index(size) not in nonzero_idx for size in prod.shape if size > 1]
    if any(error):
        raise exc.AxisError

    ##numbers
    if group == 'sire':
        numbers = stock_vars['sire_numbers_g0']
        keys = stock_vars['sire_keys_g0']
    if group == 'dams':
        numbers = stock_vars['dam_numbers_k2tvanwziy1g1']
        keys = stock_vars['dams_keys_k2tvanwziy1g1']
    if group == 'offs':
        numbers = stock_vars['offs_numbers_k3k5tvnwziaxyg3']
        keys = stock_vars['offs_keys_k3k5tvnwziaxy1g3']

    ##slice axis - slice the keys and the array - if user hasnt specified slice the whole axis will be included
    sl = [slice(None)] * prod.ndim
    for axis, slc in axis_slice.items():
        start = slc[0]
        stop = slc[1]
        step = slc[2]
        sl[axis] = slice(start, stop, step)
        keys[axis] = keys[axis][start:stop:step]
    ###apply slice to np array
    numbers = numbers[tuple(sl)]
    prod = prod[tuple(sl)]

    ##option 1
    if arith == 1:
        prod = fun.f_weighted_average(prod, numbers, tuple(arith_axis), keepdims=True)
    ##option 2
    if arith == 2:
        prod = np.sum(prod * numbers, tuple(arith_axis), keepdims=True)

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
    prod = produce_df(prod, x_keys, y_keys)
    return prod

