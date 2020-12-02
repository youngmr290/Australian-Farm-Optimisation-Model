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
    len_p6 = len(keys_p5)

    ##crop & pasture area
    df_rot = pd.DataFrame(lp_vars['v_phase_area'], index=['v_phase_area']).T #create a df of all the phase areas
    df_rot = df_rot.rename_axis(['rot','lmu'])
    phase_area = pd.merge(r_vals['rot']['phases'], df_rot, how='left', left_index=True, right_on=['rot']) #merge full phase array with area array
    phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    inter['pasture_area'] = df_rot[phase_is_pasture].sum()
    inter['crop_area'] = df_rot[~phase_is_pasture].sum()


    ##animal numbers
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    inter['sire_numbers_g0'] = sire_numbers.reshape(len_g0)
    dam_numbers = np.array(list(lp_vars['v_dams'].values()))
    inter['dam_numbers_k2tvanwziy1g1'] = dam_numbers.reshape(len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1)
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    inter['offs_numbers_k3k5tvnwziaxy1g1'] = offs_numbers.reshape(len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3)

    ##dep - depreciation is yearly but for the profit and loss it is equally divided into each cash period
    dep = lp_vars['v_dep'][None]/len_c #convert to dep per cashflow period
    inter['dep_c'] = pd.DataFrame([dep]*len_c, index=[keys_c])  #convert to df with cashflow period as index
    ##exp
    exp_fix = r_vals['overheads']
    # exp_crop_fert =
    # exp_crop_chem =
    # exp_crop_mach =
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
