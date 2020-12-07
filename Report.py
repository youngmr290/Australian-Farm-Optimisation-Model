# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (incase the inputs are adjusted during the exp so they will not be correct for r_valsing)
When creating r_vals values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd
import numpy as np

import Functions as fun
na=np.newaxis

def intermediates(inter, r_vals, lp_vars):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in a dict to store intermidiate values.
    r_vals : Dict
        Pass in dict with all r_vals values from precalcs.
    lp_vals : Dict
        Pass in dict with lp variable results from pyomo.

    Returns
    -------
    Here we manipulate r_vals variables and lp result variables.
    Everything is stored in the 'inter' dict which is passed into each
    function which builds a table or figure.

    There are two options
        1.convert the dict to pandas dataframe: see crop and pasture area
        2.convert dict to numpy: see dse

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

    ##landuse sets
    all_pas = r_vals['rot']['all_pastures']

    ##rotation
    phases_df = r_vals['rot']['phases']
    phases_rk = phases_df.set_index(5, append=True)  # add landuse as index level
    rot_area = pd.Series(lp_vars['v_phase_area']) #create a series of all the phase areas
    rot_area_rkl = rot_area.unstack().reindex(phases_rk.index, axis=0, level=0).stack() #add landuse to the axis
    landuse_area_k = rot_area_rkl.sum(axis=0,level=1) #area of each landuse (sum lmu and rotation)
    ###crop & pasture area
    ####you can now use isin pasture or crop sets to calc the area of crop or pasture
    total_pasture_area = landuse_area_k[landuse_area_k.index.isin(all_pas)].sum()
    total_crop_area = landuse_area_k[~landuse_area_k.index.isin(all_pas)].sum()
    inter['pasture_area'] = total_pasture_area
    inter['crop_area'] = total_crop_area

    ##mach
    contractharv_hours = pd.Series(lp_vars['v_contractharv_hours'])
    harv_hours = pd.Series(lp_vars['v_harv_hours']).sum(level=1) #sum labour axis
    harvest_cost = r_vals['mach']['contract_harvest_cost'].mul(contractharv_hours, axis=1)  + r_vals['mach']['harvest_cost'].mul(harv_hours, axis=1)
    seeding_days = pd.Series(lp_vars['v_seeding_machdays']).sum(level=(1,2)) #sum labour period axis
    contractseeding_ha = pd.Series(lp_vars['v_contractseeding_ha']).sum(level=1) #sum labour period and lmu axis
    seeding_ha = r_vals['mach']['seeding_rate'].stack().mul(seeding_days, level=0)
    seeding_cost_own = r_vals['mach']['seeding_cost'].reindex(seeding_ha.index, axis=1,level=1).mul(seeding_ha,axis=1).sum(axis=1, level=0) #sum lmu axis
    contractseed_cost_ha = r_vals['mach']['contractseed_cost']
    idx = pd.MultiIndex.from_product([contractseed_cost_ha.index, contractseeding_ha.index])
    seeding_cost_contract = contractseed_cost_ha.reindex(idx, level=0).mul(contractseeding_ha, level=1)
    exp_mach_rkc = (r_vals['crop']['fert_app_cost'] + r_vals['crop']['nap_fert_app_cost'] + r_vals['crop']['chem_app_cost_ha']) * rot_area_rkl +seeding_cost_own + seeding_cost_contract + harvest_cost


    ##cropping
    ###expenses
    exp_crop_fert = (r_vals['phase_fert_cost'] + r_vals['nap_phase_fert_cost']) * rot_area_rkl
    exp_crop_chem = r_vals['crop']['chem_cost']* rot_area_rkl
    misc_cropping_exp = (r_vals['stub_cost'] + r_vals['insurance_cost'] + r_vals['seed_cost']) * rot_area_rkl  #stubble, seed & insurance
    ###revenue. rev = (grain_sold + grain_fed - grain_purchased) * sell_price
    grain_purchased = pd.Series(lp_vars['v_buy_grain'])
    grain_sold = pd.Series(lp_vars['v_buy_grain'])
    grain_fed = pd.Series(lp_vars['v_sup_con'])
    grains_sale_price = r_vals['crop']['grain_price']


    ##stock
    ###animal numbers
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    sire_numbers_g0 = sire_numbers.reshape(len_g0)
    inter['sire_numbers_g0'] = sire_numbers_g0
    dam_numbers = np.array(list(lp_vars['v_dams'].values()))
    dam_numbers_k2tvanwziy1g1 = dam_numbers.reshape(len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1)
    inter['dam_numbers_k2tvanwziy1g1'] = dam_numbers_k2tvanwziy1g1
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    offs_numbers_k3k5tvnwziaxy1g1 = offs_numbers.reshape(len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3)
    inter['offs_numbers_k3k5tvnwziaxy1g1'] = offs_numbers_k3k5tvnwziaxy1g1
    ###expenses sup feeding
    grains_buy_price = r_vals['sup']['buy_grain_price']
    grain_exp= (grain_fed - grain_purchased) * grains_sale_price + grain_purchased * grains_buy_price
    feeding_exp = grain_fed * r_vals['sup']['total_sup_cost'] #feeding and storage cost related to sup
    ###husbandry expense
    sire_cost = r_vals['stock']['cost_cg0 '] * sire_numbers_g0
    dams_cost = r_vals['stock']['cost_k2ctva1nwziyg1'] * dam_numbers_k2tvanwziy1g1
    offs_cost = r_vals['stock']['cost_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxy1g1
    ###sale income
    sire_sale = r_vals['stock']['sale_cg0 '] * sire_numbers_g0
    dams_sale = r_vals['stock']['sale_k2ctva1nwziyg1'] * dam_numbers_k2tvanwziy1g1
    offs_sale = r_vals['stock']['sale_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxy1g1
    ###wool income
    sire_wool = r_vals['stock']['wool_cg0 '] * sire_numbers_g0
    dams_wool = r_vals['stock']['wool_k2ctva1nwziyg1'] * dam_numbers_k2tvanwziy1g1
    offs_wool = r_vals['stock']['wool_k3k5ctvnwzixyg3'] * offs_numbers_k3k5tvnwziaxy1g1


    ##labour
    r_vals['lab']['casual_cost']

    # df_rot = df_rot.rename_axis(['rot','lmu'])
    # phase_area = pd.merge(r_vals['rot']['phases'], df_rot, how='left', left_index=True, right_on=['rot']) #merge full phase array with area array
    # phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    # inter['pasture_area'] = df_rot[phase_is_pasture].sum()
    # pasture_area_rt = pd.DataFrame(r_vals['pas']['pasture_area_rt'], index=phases_df.index, columns=keys_pastures)
    # inter['pasture_area'] = pasture_area_rt.mul(rot_area,axis=0,level=0).sum(axis=0) #return the area of each pasture type
    # inter['crop_area'] = df_rot[~phase_is_pasture].sum() #^do i have something like pasture already? or do i need to do option 1? how can i get area for each crop set?



    ##dep - depreciation is yearly but for the profit and loss it is equally divided into each cash period
    dep = lp_vars['v_dep'][None]/len_c #convert to dep per cashflow period
    inter['dep_c'] = pd.DataFrame([dep]*len_c, index=[keys_c])  #convert to df with cashflow period as index
    ##overheads/fixed expenses
    exp_fix = r_vals['overheads']

    # exp_crop_labour =
    # exp_stock_fert =
    # exp_stock_chem =
    # exp_stock_mach =
    # exp_stock_labour =

    ##rev
    # rev_grain =
    # rev_wool =
    # rev_sale =



def f_make_table(data, index, header):
    '''function to return table
    ^currently just returns a df but there are python packages which make nice tables'''
    return pd.DataFrame(data, index=index, columns=header)

def f_dse(inter,r_vals,method=0,per_ha=False):
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
        dse_sire = inter['sire_numbers_g0'] * r_vals['stock']['dsenw_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'] * np.moveaxis(r_vals['stock']['dsenw_k2p6tva1nwziyg1'],1,0), preserveAxis=0) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxy1g1'] * np.moveaxis(r_vals['stock']['dsenw_k3k5p6tvnwzixyg3'],2,0), preserveAxis=0) #sum all axis except p6
    else:
        ##sire
        dse_sire = inter['sire_numbers_g0'] * r_vals['stock']['dsemj_p6g0']
        ##dams
        dse_dams = fun.f_reduce_skipfew(np.sum, inter['dam_numbers_k2tvanwziy1g1'] * np.moveaxis(r_vals['stock']['dsemj_k2p6tva1nwziyg1'],1,0), preserveAxis=0) #sum all axis except p6
        ##dams
        dse_offs = fun.f_reduce_skipfew(np.sum, inter['offs_numbers_k3k5tvnwziaxy1g1'] * np.moveaxis(r_vals['stock']['dsemj_k3k5p6tvnwzixyg3'],2,0), preserveAxis=0) #sum all axis except p6

    ##dse per ha if user opts for this level of detail
    if per_ha:
        dse_sire = dse_sire/inter['pasture_area']
        dse_dams = dse_dams/inter['pasture_area']
        dse_offs = dse_offs/inter['pasture_area']

    ##turn to table
    dse_sire = f_make_table(dse_sire, 'Sire DSE', r_vals['stock']['keys_p6'])
    dse_dams = f_make_table(dse_dams, 'Dams DSE', r_vals['stock']['keys_p6'])
    dse_offs = f_make_table(dse_offs, 'Offs DSE', r_vals['stock']['keys_p6'])
    return dse_sire, dse_dams, dse_offs

def f_profitloss_table(inter):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in dict with all intermidiate values required to calculate r_valss.

    Returns
    -------
    r_vals (table or figure etc).

    '''
    # exp_dep =
