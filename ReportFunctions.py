# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (in case the inputs are adjusted during the exp so they will not be correct for r_vals)
When creating r_vals values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import Functions as fun
import Exceptions as exc

na = np.newaxis


def f_errors(r_vals, exp_data_index, trial_outdated, trials):
    ##first check if data exists for each desired trial
    try:
        for row in trials:
            r_vals[exp_data_index[row][2]]
    except KeyError:
        raise exc.TrialError('''Trials for reporting don't all exist''')
    ##second check if generating results using out of date data.
    if any(trial_outdated.loc[exp_data_index[trials]]):  # have to use the trial name because the order is different
        print('''

              Generating reports from out dated data

              ''')
    return


#################
# Final reports #
#################

def f_stack(func, lp_vars, r_vals, trial_outdated, exp_data_index, trials, **kwargs):
    '''
    Returns dataframe for specified function. Multiple trials result in a stacked table with trial name as index level.

    :param func: report function whose return value is to be stacked
    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :param trial_outdated: series indicating which trials are outdated
    :param exp_data_index: trial names - in the same order as exp.xlsx
    :param trials: trials to return info for
    :param kwargs: args for specified function. This is optional.
    '''
    ##check for errors
    f_errors(r_vals, exp_data_index, trial_outdated, trials)
    ##loop through trials and generate pnl table
    result_stacked = pd.DataFrame()  # create df to append table from each trial
    for row in trials:
        result = func(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], **kwargs)
        result = pd.concat([result], keys=[exp_data_index[row][2]], names=['Trial'])  # add trial name as index level
        result_stacked = result_stacked.append(result)

    return result_stacked


def f_xy_graph(func0, func1, lp_vars, r_vals, trial_outdated, exp_data_index, trials, func0_options, func1_options):
    '''returns graph of crop area (x - axis) by profit (y - axis)

    :param func0: func to generate x values
    :param func1:func to generate y values
    :param lp_vars: dict - results from pyomo
    :param r_vals: dict - report variable
    :param trial_outdated: series indicating which trials are outdated
    :param exp_data_index: trial names - in the same order as exp.xlsx
    :param trials: trials to return info for
    :param func0_options:
            3: total pasture area
            4: total crop area
    :param func1_options:
            0: profit = rev - (exp + minroe + asset_opp +dep)
            1: profit = rev - (exp + dep)
    '''
    ##check for errors
    f_errors(r_vals, exp_data_index, trial_outdated, trials)
    ##loop through trials and generate pnl table
    y_vals = []  # create list to append pnl table from each trial
    x_vals = []  # create list to append pnl table from each trial
    for row in trials:
        x_vals.append(func0(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], option=func0_options))
        y_vals.append(func1(lp_vars[exp_data_index[row][2]], r_vals[exp_data_index[row][2]], option=func1_options))
    plt.plot(x_vals, y_vals)
    return plt


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
    :key fs: int - fat score to report price for. Has to be number between 1-5 inclusive.
    :return: df
    '''
    ##unpack kwargs
    option = kwargs['option']
    grid = kwargs['grid']
    weight = kwargs['weight']
    fs = kwargs['fs']

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


#########################################
# intermediate report building functions#
#########################################


def f_rotation(lp_vars, r_vals):
    '''
    manipulates the rotation solution into usable format. This is used in many function.
    '''
    ##rotation
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(5, append=True)  # add landuse as index level
    rot_area_rl = pd.Series(lp_vars[
                                'v_phase_area']).sort_index()  # create a series of all the phase areas, need to sort the index because it was chuck error for some calculations
    rot_area_rkl = rot_area_rl.unstack().reindex(phases_rk.index, axis=0, level=0).stack()  # add landuse to the axis
    return phases_rk, rot_area_rl, rot_area_rkl


def f_area_summary(lp_vars, r_vals, **kwargs):
    '''
    Rotation & landuse area summary. With multiple output levels.
    return options:

    :param lp_vars: dict
    :param r_vals: dict
    :key option:
        0: tuple all results wrapped in tuple
        1: table all rotations by lmu
        2: table selected rotations by lmu
        3: float total pasture area
        4: float total crop area
        5: table crop and pasture area by lmu
    '''
    ##unpack kwargs
    option = kwargs['option']

    ##read from other functions
    rot_area_rl, rot_area_rkl = f_rotation(lp_vars, r_vals)[1:3]
    landuse_area_kl = rot_area_rkl.sum(axis=0, level=(1, 2)).unstack()  # area of each landuse (sum lmu and rotation)

    ##all rotations by lmu
    rot_area_rl = rot_area_rl.unstack()
    if option == 1:
        return rot_area_rl.round(0)

    ##selected rotations by lmu
    rot_area_selected_rl = rot_area_rl[rot_area_rl.any(axis=1)]
    if option == 2:
        return rot_area_selected_rl.round(0)
    ###crop & pasture area
    ####you can now use isin pasture or crop sets to calc the area of crop or pasture
    all_pas = r_vals['rot']['all_pastures']  # landuse sets
    pasture_area_l = landuse_area_kl[landuse_area_kl.index.isin(all_pas)].sum()  # sum landuse
    if option == 3:
        return pasture_area_l.sum().round(0)
    crop_area_l = landuse_area_kl[~landuse_area_kl.index.isin(all_pas)].sum()  # sum landuse
    if option == 4:
        return crop_area_l.sum().round(0)

    ##crop & pasture area by lmu
    croppas_area_l = pd.DataFrame()
    croppas_area_l.loc['pasture'] = pasture_area_l
    croppas_area_l.loc['crop'] = crop_area_l
    if option == 5:
        return croppas_area_l.round(0)

    ##return all if option==0
    if option == 0:
        return rot_area_rl, rot_area_selected_rl, pasture_area_l, crop_area_l, croppas_area_l


def f_mach_summary(lp_vars, r_vals, option=0):
    '''
    Machine summary.
    return options:
    0- table: total machine cost for each crop in each cash period

    '''
    ##call rotation function to get rotation info
    phases_rk, rot_area_rl = f_rotation(lp_vars, r_vals)[0:2]
    ##harv
    contractharv_hours = pd.Series(lp_vars['v_contractharv_hours'])
    harv_hours = pd.Series(lp_vars['v_harv_hours']).sum(level=1)  # sum p5 axis
    harvest_cost = r_vals['mach']['contract_harvest_cost'].mul(contractharv_hours, axis=1) + r_vals['mach'][
        'harvest_cost'].mul(harv_hours, axis=1)
    ##seeding
    seeding_days = pd.Series(lp_vars['v_seeding_machdays']).sum(level=(1, 2))  # sum labour period axis
    contractseeding_ha = pd.Series(lp_vars['v_contractseeding_ha']).sum(level=1)  # sum labour period and lmu axis
    seeding_ha = r_vals['mach']['seeding_rate'].mul(
        seeding_days.unstack()).stack()  # note seeding ha wont equal the rotation area because arable area is included in seed_ha.
    seeding_cost_own = r_vals['mach']['seeding_cost'].reindex(seeding_ha.index, axis=1, level=1).mul(seeding_ha,
                                                                                                     axis=1).sum(axis=1,
                                                                                                                 level=0)  # sum lmu axis
    contractseed_cost_ha = r_vals['mach']['contractseed_cost']
    idx = pd.MultiIndex.from_product([contractseed_cost_ha.index, contractseeding_ha.index])
    seeding_cost_contract = contractseed_cost_ha.reindex(idx, level=0).mul(contractseeding_ha, level=1).unstack()
    ##fert & chem mach cost
    fertchem_cost_rlc = pd.concat(
        [r_vals['crop']['fert_app_cost'], r_vals['crop']['nap_fert_app_cost'], r_vals['crop']['chem_app_cost_ha']],
        axis=1).sum(axis=1, level=0)  # cost per ha
    fertchem_cost_rc = fertchem_cost_rlc.mul(rot_area_rl, axis=0).sum(axis=0, level=0)  # mul area and sum lmu
    fertchem_cost_kc = fertchem_cost_rc.reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                                      level=1)  # reindex to include landuse and sum rot
    ##conbime all costs
    exp_mach_kc = pd.concat([fertchem_cost_kc.T, seeding_cost_own, seeding_cost_contract, harvest_cost], axis=0).sum(
        axis=0, level=0).T
    ##return all if option==0
    if option == 0:
        return exp_mach_kc


def f_grain_sup_summary(lp_vars, r_vals, option=0):
    '''
    Summary of grain, supplement and their costs

    :param option: int:
            0: return dict with various elements
            1: return total supplement fed in each feed period

    '''
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
    grain_fed_kg = pd.Series(lp_vars['v_sup_con']).sum(level=(0, 1))  # sum feed pool and feed period
    grain_fed_kp6 = pd.Series(lp_vars['v_sup_con']).sum(level=(0, 3)).swaplevel()  # sum feed pool and grain pool
    grain_fed_p6 = pd.Series(lp_vars['v_sup_con']).sum(level=(3))  # sum feed pool, landuse and grain pool
    if option == 1:
        return grain_fed_p6.to_frame()
    ##total grain produced by crop enterprise
    total_grain_produced = grain_sold_kg + grain_fed_kg - grain_purchased_kg  # total grain produced by crop enterprise
    rev_grain_c = grains_sale_price_kgc.mul(total_grain_produced.reindex(grains_sale_price_kgc.index), axis=0).sum(
        axis=0,
        level=0)  # sum grain pool, have to reindex (not really sure why since it is the same index - maybe one has been condensed ie index with nan removed)
    grain['rev_grain_c'] = rev_grain_c

    ##supplementary cost: cost = sale_price * (grain_fed - grain_purchased) + buy_price * grain_purchased
    sup_exp_c = (grains_sale_price_kgc.mul((grain_fed_kg - grain_purchased_kg).reindex(grains_sale_price_kgc.index),
                                           axis=0)
                 + grains_buy_price_kgc.mul(grain_purchased_kg.reindex(grains_buy_price_kgc.index), axis=0)).sum(axis=0,
                                                                                                                 level=0)  # sum grain pool
    grain['sup_exp_c'] = sup_exp_c
    return grain


def f_stubble_summary(lp_vars, r_vals):
    stub_fp6ks = pd.Series(lp_vars['v_stub_con'])
    return stub_fp6ks.sum(level=(1, 3)).unstack()


def f_crop_summary(lp_vars, r_vals, option=0):
    '''
    Crop summary. Includes pasture inputs.
    return options:
    0- tuple: fert cost, chem cost, miscellaneous costs and grain revenue for each landuse

    '''
    ##call rotation function to get rotation info
    phases_rk, rot_area_rl = f_rotation(lp_vars, r_vals)[0:2]
    ##expenses
    ###fert
    exp_fert_ha_rlc = pd.concat([r_vals['crop']['phase_fert_cost'], r_vals['crop']['nap_phase_fert_cost']], axis=1).sum(
        axis=1, level=0)
    exp_fert_rc = exp_fert_ha_rlc.mul(rot_area_rl, axis=0).sum(axis=0, level=0)  # mul area and sum lmu
    exp_fert_kc = exp_fert_rc.reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ###chem
    exp_chem_rc = r_vals['crop']['chem_cost'].mul(rot_area_rl, axis=0).sum(axis=0, level=0)  # mul area and sum lmu
    exp_chem_kc = exp_chem_rc.reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ###misc
    misc_exp_ha_rlc = pd.concat(
        [r_vals['crop']['stub_cost'], r_vals['crop']['insurance_cost'], r_vals['crop']['seedcost']], axis=1).sum(axis=1,
                                                                                                                 level=0)  # stubble, seed & insurance
    misc_exp_rc = misc_exp_ha_rlc.reindex(rot_area_rl.index, axis=0).mul(rot_area_rl, axis=0).sum(axis=0,
                                                                                                  level=0)  # mul area and sum lmu
    misc_exp_kc = misc_exp_rc.reindex(phases_rk.index, axis=0, level=0).sum(axis=0,
                                                                            level=1)  # reindex to include landuse and sum rot
    ##revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    rev_grain_c = grain_summary['rev_grain_c']
    ##return all if option==0
    if option == 0:
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
    len_a = len(keys_a)
    len_c = len(keys_c)
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


    ##animal numbers
    ###shapes
    sire_shape = len_g0
    dams_shape = len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1
    prog_shape = len_k5, len_t2, len_lw_prog, len_z, len_i, len_d, len_a, len_x, len_g2
    offs_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3
    ###sire
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    sire_numbers_g0 = sire_numbers.reshape(sire_shape)
    sire_numbers_g0[sire_numbers_g0 == None] = 0  # replace None with 0
    stock_vars['sire_numbers_g0'] = sire_numbers_g0.astype(float)
    ###dams
    dams_numbers = np.array(list(lp_vars['v_dams'].values()))
    dams_numbers_k2tvanwziy1g1 = dams_numbers.reshape(dams_shape)
    dams_numbers_k2tvanwziy1g1[dams_numbers_k2tvanwziy1g1 == None] = 0  # replace None with 0
    stock_vars['dams_numbers_k2tvanwziy1g1'] = dams_numbers_k2tvanwziy1g1.astype(float)
    stock_vars['dams_numbers_tvanwziy1g1'] = np.sum(dams_numbers_k2tvanwziy1g1.astype(float), axis=0)
    ###prog
    prog_numbers = np.array(list(lp_vars['v_prog'].values()))
    prog_numbers_k5twzida0xg2 = prog_numbers.reshape(prog_shape)
    prog_numbers_k5twzida0xg2[prog_numbers_k5twzida0xg2 == None] = 0  # replace None with 0
    stock_vars['prog_numbers_k5twzida0xg2'] = prog_numbers_k5twzida0xg2.astype(float)
    ###offs
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    offs_numbers_k3k5tvnwziaxyg3 = offs_numbers.reshape(offs_shape)
    offs_numbers_k3k5tvnwziaxyg3[offs_numbers_k3k5tvnwziaxyg3 == None] = 0  # replace None with 0
    stock_vars['offs_numbers_k3k5tvnwziaxyg3'] = offs_numbers_k3k5tvnwziaxyg3.astype(float)

    return stock_vars


def f_pasture_reshape(lp_vars, r_vals):
    '''
    Reshape pasture lp variables into numpy array

    :param lp_vars: lp variables
    :return: dict
    '''
    keys_d = r_vals['pas']['keys_d']
    keys_v = r_vals['pas']['keys_v']
    keys_f = r_vals['pas']['keys_f']
    keys_g = r_vals['pas']['keys_g']
    keys_l = r_vals['pas']['keys_l']
    keys_o = r_vals['pas']['keys_o']
    keys_p = r_vals['pas']['keys_p']
    keys_r = r_vals['pas']['keys_r']
    keys_t = r_vals['pas']['keys_t']
    keys_k = r_vals['pas']['keys_k']

    len_d = len(keys_d)
    len_v = len(keys_v)
    len_f = len(keys_f)
    len_g = len(keys_g)
    len_l = len(keys_l)
    len_o = len(keys_o)
    len_p = len(keys_p)
    len_r = len(keys_r)
    len_t = len(keys_t)
    len_k = len(keys_k)

    ##dict to store reshaped pasture stuff in
    pas_vars = {}

    # store keys - must be in axis order
    pas_vars['keys_vgoflt'] = [keys_v, keys_g, keys_o, keys_f, keys_l, keys_t]
    pas_vars['keys_vdft'] = [keys_v, keys_d, keys_f, keys_t]
    pas_vars['keys_dft'] = [keys_d, keys_f, keys_t]
    pas_vars['keys_vfl'] = [keys_v, keys_f, keys_l]

    ##shapes
    vgoflt = len_v, len_g, len_o, len_f, len_l, len_t
    vdft = len_v, len_d, len_f, len_t
    dft = len_d, len_f, len_t
    vfl = len_v, len_f, len_l

    ##reshape green pasture hectare variable
    greenpas_ha = np.array(list(lp_vars['v_greenpas_ha'].values()))
    greenpas_ha_vgoflt = greenpas_ha.reshape(vgoflt)
    greenpas_ha_vgoflt[greenpas_ha_vgoflt == None] = 0  # replace None with 0
    pas_vars['greenpas_ha_vgoflt'] = greenpas_ha_vgoflt

    ##dry end period
    drypas_transfer = np.array(list(lp_vars['v_drypas_transfer'].values()))
    drypas_transfer_dft = drypas_transfer.reshape(dft)
    drypas_transfer_dft[drypas_transfer_dft == None] = 0  # replace None with 0
    pas_vars['drypas_transfer_dft'] = drypas_transfer_dft

    ##nap end period
    nap_transfer = np.array(list(lp_vars['v_nap_transfer'].values()))
    nap_transfer_dft = nap_transfer.reshape(dft)
    nap_transfer_dft[nap_transfer_dft == None] = 0  # replace None with 0
    pas_vars['nap_transfer_dft'] = nap_transfer_dft

    ##dry consumed
    drypas_consumed = np.array(list(lp_vars['v_drypas_consumed'].values()))
    drypas_consumed_vdft = drypas_consumed.reshape(vdft)
    drypas_consumed_vdft[drypas_consumed_vdft == None] = 0  # replace None with 0
    pas_vars['drypas_consumed_vdft'] = drypas_consumed_vdft

    ##nap consumed
    nap_consumed = np.array(list(lp_vars['v_nap_consumed'].values()))
    nap_consumed_vdft = nap_consumed.reshape(vdft)
    nap_consumed_vdft[nap_consumed_vdft == None] = 0  # replace None with 0
    pas_vars['nap_consumed_vdft'] = nap_consumed_vdft

    ##poc consumed
    poc_consumed = np.array(list(lp_vars['v_poc'].values()))
    poc_consumed_vfl = poc_consumed.reshape(vfl)
    poc_consumed_vfl[poc_consumed_vfl == None] = 0  # replace None with 0
    pas_vars['poc_consumed_vfl'] = poc_consumed_vfl

    return pas_vars


def f_stock_cash_summary(lp_vars, r_vals):
    '''
    Returns:
    0- expense and revenue items

    '''
    ##get reshaped variable
    stock_vars = f_stock_reshape(lp_vars, r_vals)

    ##numbers
    sire_numbers_g0 = stock_vars['sire_numbers_g0']
    dams_numbers_k2tvanwziy1g1 = stock_vars['dams_numbers_k2tvanwziy1g1']
    prog_numbers_k5twzida0xg2 = stock_vars['prog_numbers_k5twzida0xg2']
    offs_numbers_k3k5tvnwziaxyg3 = stock_vars['offs_numbers_k3k5tvnwziaxyg3']

    ##husb cost
    sire_cost_cg0 = r_vals['stock']['sire_cost_cg0'] * sire_numbers_g0
    dams_cost_k2ctva1nwziyg1 = r_vals['stock']['dams_cost_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    offs_cost_k3k5ctvnwziaxyg3 = r_vals['stock']['offs_cost_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ##sale income
    salevalue_cg0 = r_vals['stock']['salevalue_cg0'] * sire_numbers_g0
    salevalue_k2ctva1nwziyg1 = r_vals['stock']['salevalue_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    salevalue_k5ctwzida0xg2 = r_vals['stock']['salevalue_ctwzia0xg2'][..., na, :, :, :] * prog_numbers_k5twzida0xg2[:, na,
                                                                                     ...]
    salevalue_k3k5ctvnwziaxyg3 = r_vals['stock']['salevalue_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ##wool income
    woolvalue_cg0 = r_vals['stock']['woolvalue_cg0'] * sire_numbers_g0
    woolvalue_k2ctva1nwziyg1 = r_vals['stock']['woolvalue_k2ctva1nwziyg1'] * dams_numbers_k2tvanwziy1g1[:, na, ...]
    woolvalue_k3k5ctvnwziaxyg3 = r_vals['stock']['woolvalue_k3k5ctvnwziaxyg3'] * offs_numbers_k3k5tvnwziaxyg3[:, :, na, ...]

    ###sum axis to return total income in each cash period
    siresale_c = fun.f_reduce_skipfew(np.sum, salevalue_cg0, preserveAxis=0)  # sum all axis except c
    damssale_c = fun.f_reduce_skipfew(np.sum, salevalue_k2ctva1nwziyg1, preserveAxis=1)  # sum all axis except c
    progsale_c = fun.f_reduce_skipfew(np.sum, salevalue_k5ctwzida0xg2, preserveAxis=1)  # sum all axis except c
    offssale_c = fun.f_reduce_skipfew(np.sum, salevalue_k3k5ctvnwziaxyg3, preserveAxis=2)  # sum all axis except c
    sirewool_c = fun.f_reduce_skipfew(np.sum, woolvalue_cg0, preserveAxis=0)  # sum all axis except c
    damswool_c = fun.f_reduce_skipfew(np.sum, woolvalue_k2ctva1nwziyg1, preserveAxis=1)  # sum all axis except c
    offswool_c = fun.f_reduce_skipfew(np.sum, woolvalue_k3k5ctvnwziaxyg3, preserveAxis=2)  # sum all axis except c
    stocksale_c = siresale_c + damssale_c + progsale_c + offssale_c
    wool_c = sirewool_c + damswool_c + offswool_c

    sirecost_c = fun.f_reduce_skipfew(np.sum, sire_cost_cg0, preserveAxis=0)  # sum all axis except c
    damscost_c = fun.f_reduce_skipfew(np.sum, dams_cost_k2ctva1nwziyg1, preserveAxis=1)  # sum all axis except c
    offscost_c = fun.f_reduce_skipfew(np.sum, offs_cost_k3k5ctvnwziaxyg3, preserveAxis=2)  # sum all axis except c
    stockcost_c = sirecost_c + damscost_c + offscost_c

    ##expenses sup feeding
    ###read in dict from grain summary
    grain_summary = f_grain_sup_summary(lp_vars, r_vals)
    sup_cost_c = grain_summary['sup_exp_c']

    ##infrastructure

    return stocksale_c, wool_c, sirecost_c, stockcost_c, sup_cost_c


def f_labour_summary(lp_vars, r_vals, option=0):
    '''
    :return:
    0- total labour cost
    1- amount for each enterprise
    '''
    ##total labour cost
    if option == 0:
        cas_cost_pc = r_vals['lab']['casual_cost'].mul(pd.Series(lp_vars['v_quantity_casual']), level=0)
        perm_cost_c = r_vals['lab']['perm_cost'] * pd.Series(lp_vars['v_quantity_perm']).values
        manager_cost_c = r_vals['lab']['manager_cost'] * pd.Series(lp_vars['v_quantity_manager']).values
        total_lab_cost = cas_cost_pc.sum(level=1) + perm_cost_c + manager_cost_c
        return total_lab_cost
    ##labour breakdown for each worker level (table: labour period by worker level)
    if option == 1:
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
    dep = lp_vars['v_dep'][None] / len_c  # convert to dep per cashflow period
    dep_c = pd.Series([dep] * len_c, index=keys_c)  # convert to df with cashflow period as index
    return dep_c


def f_overhead_summary(r_vals):
    ##overheads/fixed expenses
    exp_fix_c = r_vals['fin']['overheads']
    return exp_fix_c


def f_dse(lp_vars, r_vals, **kwargs):
    '''
    DSE calculation

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :key method: int
            0 - dse by normal weight
            1 - dse by mei
    :key per_ha: Bool
        if true it returns DSE/ha else it returns total dse
    :return DSE per pasture hectare for each sheep group:
    '''
    method = kwargs['method']
    per_ha = kwargs['per_ha']
    stock_vars = f_stock_reshape(lp_vars, r_vals)

    if method == 0:
        ##sire
        dse_sire = stock_vars['sire_numbers_g0'] * r_vals['stock']['dsenw_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_k2tvanwziy1g1'][:, na, ...] * r_vals['stock'][
            'dsenw_k2p6tva1nwziyg1'], preserveAxis=1)  # sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_k3k5tvnwziaxyg3'][:, :, na, ...] * r_vals['stock'][
            'dsenw_k3k5p6tvnwziaxyg3'], preserveAxis=2)  # sum all axis except p6
    else:
        ##sire
        dse_sire = stock_vars['sire_numbers_g0'] * r_vals['stock']['dsemj_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, stock_vars['dams_numbers_k2tvanwziy1g1'][:, na, ...] * r_vals['stock'][
            'dsemj_k2p6tva1nwziyg1'], preserveAxis=1)  # sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, stock_vars['offs_numbers_k3k5tvnwziaxyg3'][:, :, na, ...] * r_vals['stock'][
            'dsemj_k3k5p6tvnwziaxyg3'], preserveAxis=2)  # sum all axis except p6

    ##dse per ha if user opts for this level of detail
    if per_ha:
        pasture_area = f_area_summary(lp_vars, r_vals, option=3)
        dse_sire = dse_sire / pasture_area
        dse_dams = dse_dams / pasture_area
        dse_offs = dse_offs / pasture_area

    ##turn to table
    key_p6 = r_vals['stock']['keys_p6']
    dse_sire = fun.f_make_table(dse_sire, key_p6, ['Sire DSE'])
    dse_dams = fun.f_make_table(dse_dams, key_p6, ['Dams DSE'])
    dse_offs = fun.f_make_table(dse_offs, key_p6, ['Offs DSE'])

    ##concat
    dse = pd.concat([dse_sire, dse_dams, dse_offs], axis=1)
    return dse


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
    pnl = pd.DataFrame(index=pnl_index, columns=keys_c)  # need to initialise df with multiindex so rows can be added

    ##income
    rev_grain_c = rev_grain_kc.sum(axis=0)  # sum landuse axis
    ###add to p/l table each as a new row
    pnl.loc[('Revenue', 'grain'), :] = rev_grain_c
    pnl.loc[('Revenue', 'sheep sales'), :] = stocksale_c
    pnl.loc[('Revenue', 'wool'), :] = wool_c
    pnl.loc[('Revenue', 'Total Revenue'), :] = pnl.loc[pnl.index.get_level_values(0) == 'Revenue'].sum(axis=0)

    ##expenses
    ####machinery
    mach_c = exp_mach_kc.sum(axis=0)  # sum landuse
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
    pnl.loc[('Expense', 'Crop'), :] = crop_c
    pnl.loc[('Expense', 'pasture'), :] = pas_c
    pnl.loc[('Expense', 'stock'), :] = stockcost_c
    pnl.loc[('Expense', 'machinery'), :] = mach_c
    pnl.loc[('Expense', 'labour'), :] = labour_c
    pnl.loc[('Expense', 'fixed'), :] = exp_fix_c
    pnl.loc[('Expense', 'depreciation'), :] = dep_c
    pnl.loc[('Expense', 'Total expenses'), :] = pnl.loc[pnl.index.get_level_values(0) == 'Expense'].sum(axis=0)

    ##EBIT
    pnl.loc[('', 'EBIT'), :] = pnl.loc[('Revenue', 'Total Revenue')] - pnl.loc[('Expense', 'Total expenses')]

    ##add a column which is total of all cashflow period
    pnl['Full year'] = pnl.sum(axis=1)

    ##round numbers in df
    pnl = pnl.astype(float).round(1)  # have to go to float so rounding works
    return pnl


def f_profit(lp_vars, r_vals, option=0):
    '''returns profit
    0- rev - (exp + minroe + asset_opp +dep)
    1- rev - (exp + dep)
    '''
    obj_profit = r_vals['profit']
    minroe = pd.Series(lp_vars['v_minroe'])
    asset_opportunity_cost = pd.Series(lp_vars['v_asset'])
    if option == 0:
        return obj_profit
    else:
        return obj_profit + minroe - (asset_opportunity_cost * r_vals['opportunity_cost_capital'])


def f_stock_pasture_summary(lp_vars, r_vals, **kwargs):
    '''
    Returns summary of a numpy array in a pandas table.
    Note: 1. prod and weights must be broadcastable.
          2. Specify axes the broadcasted/expanded version.

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :key type: str: either 'stock' or 'pas' to indicate calc type
    :key key: str: dict key for the axis keys
    :key index (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :key arith (optional, default = 0): int: arithmetic operation used.
            option 0: return production param, averaged on given axis
            option 1: return weighted average of production param (using denominator weight return production per day the animal is on hand)
            option 2: total production for a given axis np.sum(prod * weight, axis)
            option 3: total production for each activity
            option 4: return weighted average of production param (using denominator weight returns the average production for period eg less if animal is sold part way through)
    :key arith_axis (optional, default = []): list: axis to preform arithmetic operation along.
    :key prod (optional, default = 1): str/int/float: if it is a string then it is used as a key for stock_vars, if it is an number that number is used as the prod value
    :key na_prod (optional, default = []): list: position to add new axis
    :key weights (optional, default = None): str: weights to be used in arith (typically a lp variable eg numbers). Only required when arith>0
    :key na_weights (optional, default = []): list: position to add new axis
    :key den_weights (optional, default = 1): str: key to variable used to weight the denominator in the weighted average (required p6 reporting)
    :key na_denweights (optional, default = []): list: position to add new axis
    :key denom (optional, default = 1): str: keys to r_vals indicating denominator to divide production by (after other operations have been applied).
    :key na_denom (optional, default = []): list: position to add new axis
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: pandas df
    '''
    ##unpack dict adding default values
    ###no default value (must exist)
    keys_key = kwargs['keys']
    type = kwargs['type']
    ###default values exist
    try:
        na_weights = kwargs['na_weights']
    except KeyError:
        na_weights = []

    try:
        na_prod = kwargs['na_prod']
    except KeyError:
        na_prod = []

    try:
        na_denweights = kwargs['na_denweights']
    except KeyError:
        na_denweights = []

    try:
        den_weights = kwargs['den_weights']
    except KeyError:
        den_weights = 1

    try:
        arith = kwargs['arith']
    except KeyError:
        arith = 0

    try:
        arith_axis = kwargs['arith_axis']
    except KeyError:
        arith_axis = []

    try:
        denom = r_vals[kwargs['denom'][0]][kwargs['denom'][1]]
    except KeyError:
        denom = 1

    try:
        na_denom = kwargs['denom_pos']
    except KeyError:
        na_denom = []

    try:
        index = kwargs['index']
    except KeyError:
        index = []

    try:
        cols = kwargs['cols']
    except KeyError:
        cols = []

    try:
        prod_key = kwargs['prod']
    except KeyError:
        prod_key = 1

    try:
        axis_slice = kwargs['axis_slice']
    except KeyError:
        axis_slice = {}

    ##read from stock reshape function
    if type == 'stock':
        vars = f_stock_reshape(lp_vars, r_vals)
        ###if production doesnt exist eg it is 1 or some other number (this means you can preform arith with any number - mainly used for pasture when there is no production param)
        if isinstance(prod_key, str):
            prod = r_vals['stock'][prod_key]
        else:
            prod = np.array([prod_key])
        ###den weight - used in weighted average calc (default is 1)
        if isinstance(den_weights, str):
            den_weights = r_vals['stock'][den_weights]
    else:
        vars = f_pasture_reshape(lp_vars, r_vals)
        ###if production doesnt exist eg it is 1 or some other number (this means you can preform arith with any number - mainly used for pasture when there is no production param)
        if isinstance(prod_key, str):
            prod = r_vals['pas'][prod_key]
        else:
            prod = np.array([prod_key])
        ###den weight - used in weighted average calc (default is 1)
        if isinstance(den_weights, str):
            den_weights = r_vals['pas'][den_weights]  # pasture params don't need to go through reshape function

    ##keys that will become the index and cols for table
    keys = vars[keys_key]

    ##if no weights then make None
    try:
        weights = vars[kwargs['weights']]
    except KeyError:
        weights = None

    ##other manipulation
    f_numpy2df_error(prod, weights, arith_axis, index, cols)
    prod, weights, den_weights, denom = f_add_axis(prod, weights, den_weights, denom, na_weights, na_prod, na_denweights, na_denom)
    prod, weights, den_weights = f_slice(prod, weights, den_weights, keys, arith, axis_slice)
    prod = f_arith(prod, weights, den_weights, arith, arith_axis)
    prod = fun.f_divide(prod, denom)
    prod = f_numpy2df(prod, keys, index, cols)
    return prod

def f_survival(lp_vars, r_vals, **kwargs):
    '''

    :param lp_vars: dict: results from pyomo
    :param r_vals: dict: report variable
    :key key: str: dict key for the axis keys
    :key index (optional, default = []): list: axis you want as the index of pandas df (order of list is the index level order).
    :key cols (optional, default = []): list: axis you want as the cols of pandas df (order of list is the col level order).
    :key arith (optional, default = 0): int: arithmetic operation used.
            option 0: return production param, averaged on given axis
            option 1: return weighted average of production param (using denominator weight return production per day the animal is on hand)
            option 2: total production for a given axis np.sum(prod * weight, axis)
            option 3: total production for each activity
            option 4: return weighted average of production param (using denominator weight returns the average production for period eg less if animal is sold part way through)
    :key arith_axis (optional, default = []): list: axis to preform arithmetic operation along.
    :key prod (optional, default = 1): str/int/float: if it is a string then it is used as a key for stock_vars, if it is an number that number is used as the prod value
    :key na_prod (optional, default = []): list: position to add new axis
    :key weights (optional, default = None): str: weights to be used in arith (typically a lp variable eg numbers). Only required when arith>0
    :key na_weights (optional, default = []): list: position to add new axis
    :key den_weights (optional, default = 1): str: key to variable used to weight the denominator in the weighted average (required p6 reporting)
    :key na_denweights (optional, default = []): list: position to add new axis
    :key denom (optional, default = 1): str: keys to r_vals indicating denominator to divide production by (after other operations have been applied).
    :key na_denom (optional, default = []): list: position to add new axis
    :key axis_slice (optional, default = {}): dict: keys (int) is the axis. value (list) is the start, stop and step of the slice
    :return: pandas df
    '''
    ##unpack dict adding default values
    ###default values exist
    # try:
    #     na_weights = kwargs['na_weights']
    # except KeyError:
    #     na_weights = []
    #
    # try:
    #     na_prod = kwargs['na_prod']
    # except KeyError:
    #     na_prod = []
    #
    # try:
    #     na_denweights = kwargs['na_denweights']
    # except KeyError:
    #     na_denweights = []
    #
    # try:
    #     den_weights = kwargs['den_weights']
    # except KeyError:
    #     den_weights = 1

    try:
        arith = kwargs['arith']
    except KeyError:
        arith = 0

    try:
        arith_axis = kwargs['arith_axis']
    except KeyError:
        arith_axis = []


    try:
        index = kwargs['index']
    except KeyError:
        index = []

    try:
        cols = kwargs['cols']
    except KeyError:
        cols = []

    try:
        axis_slice = kwargs['axis_slice']
    except KeyError:
        axis_slice = {}

    ##read from stock reshape function


    vars = f_stock_reshape(lp_vars, r_vals)
    v_dams_k2tvaebnwziy1g1 = vars['dams_numbers_k2tvanwziy1g1'][:,:,:,:,na,na,...] #add na for e and b

    ##number of prog in each e/b slice for each cluster.
    prog_born = np.sum(r_vals['stock']['prog_born_k2tva1e1b1nw8ziyg1'] * v_dams_k2tvaebnwziy1g1, axis= tuple([0,1,3,6,7,8,9,10,11]),keepdims=True)
    prog_alive = np.sum(r_vals['stock']['prog_alive_k2tva1e1b1nw8ziyg1'] * v_dams_k2tvaebnwziy1g1, axis= tuple([0,1,3,6,7,8,9,10,11]),keepdims=True)
    # index_b9
    survival=prog_born/prog_alive
    # ##keys that will become the index and cols for table
    # keys = vars[keys_key]
    #
    # ##other manipulation
    # f_numpy2df_error(numbers_start_k2tva1e1b1nw8ziyg1, v_dams, arith_axis, index, cols)
    #
    # prod, weights, den_weights = f_slice(prod, weights, den_weights, keys, arith, axis_slice)
    # prod = f_arith(prod, weights, den_weights, arith, arith_axis)
    # prod = fun.f_divide(prod, denom)
    # prod = f_numpy2df(prod, keys, index, cols)
    return survival


############################
# functions for numpy arrays#
############################

def f_numpy2df_error(prod, weights, arith_axis, index, cols):
    ##error handle 1: cant preform arithmetic along an axis and also report that axis and the index or col
    arith_occur = len(arith_axis) >= 1
    arith_error = any(item in index for item in arith_axis) or any(item in cols for item in arith_axis)
    if arith_occur and arith_error:  # if arith is happening and there is an error in selected axis
        raise exc.ArithError('''Arith error: can't preform operation along an axis that is going to be reported as the index or col''')

    ##error handle 2: once arith has been completed all axis that are not singleton must be used in either the index or cols
    if arith_occur:
        nonzero_idx = arith_axis + index + cols  # join lists
    else:
        nonzero_idx = index + cols  # join lists
    error = [prod.shape.index(size) not in nonzero_idx for size in prod.shape if size > 1]
    if any(error):
        raise exc.AxisError('''Axis error: active axes exist that are not used in arith or being reported as index or columns''')

    ##error 3: preforming arith with no weights
    if arith_occur and weights is None:
        raise exc.ArithError('''Arith error: weights are not included''')
    return


def f_add_axis(prod, weights, den_weights, denom, na_weights, na_prod, na_denweights, na_denom):
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
    denom = np.expand_dims(denom, na_denom)
    return prod, weights, den_weights, denom


def f_slice(prod, weights, den_weights, keys, arith, axis_slice):
    '''
    Slices the prod, weights and key arrays

    :param prod: array: production param
    :param weights: array: weights (typically the variable associated with the prod param)
    :param keys: list: keys for axes
    :param axis_slice: dict: containing list of with slice params (start, stop, step)
    :return: prod array
    '''
    ##slice axis - slice the keys and the array - if user hasn't specified slice the whole axis will be included
    sl = [slice(None)] * prod.ndim
    for axis, slc in axis_slice.items():
        start = slc[0]
        stop = slc[1]
        step = slc[2]
        sl[axis] = slice(start, stop, step)
        keys[axis] = keys[axis][start:stop:step]
    ###apply slice to np array
    if arith == 0:
        prod = prod[tuple(sl)]
    else:
        prod, weights, den_weights = np.broadcast_arrays(prod, weights,
                                                         den_weights)  # if arith is being conducted these arrays need to be the same size so slicing can work
        prod = prod[tuple(sl)]
        weights = weights[tuple(sl)]
        den_weights = den_weights[tuple(sl)]
    return prod, weights, den_weights


def f_arith(prod, weight, den_weights, arith, axis):
    '''
    option 0: return production param averaged on specified axis
    option 1: return weighted average of production param (using denominator weight return production per day the animal is on hand)
    option 2: total production for a given axis
    option 3: total production for each activity
    option 4: return weighted average of production param (using denominator weight returns the average production for period eg less if animal is sold part way through)

    :param prod: array: production param
    :param weight: array: weights (typically the variable associated with the prod param)
    :param arith: int: arith option
    :param axis: list: axes to preform arith along
    :return: array
    '''
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
        prod = fun.f_weighted_average(prod, weight, tuple(axis), keepdims=keepdims,
                                      den_weights=(den_weights / den_weights))
    return prod


def f_numpy2df(prod, keys, index, cols):
    if prod.size <= 1 and prod.ndim <= 1:
        return pd.DataFrame([prod])  # don't need to reshape etc if everything is summed and prod is just one number
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
