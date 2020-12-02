# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (incase the inputs are adjusted during the exp so they will not be correct for r_valsing)
When creating r_vals values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd
import numpy as np


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
    ##axis len
    len_c = len(r_vals['keys_c'])
    len_a = r_vals['keys_a']
    len_d = r_vals['keys_d']
    len_g0 = r_vals['keys_g0']
    len_g1 = r_vals['keys_g1']
    len_g2 = r_vals['keys_g2']
    len_g3 = r_vals['keys_g3']
    len_f = r_vals['keys_f']
    len_h1 = r_vals['keys_h1']
    len_i = r_vals['keys_i']
    len_k2 = r_vals['keys_k2']
    len_k3 = r_vals['keys_k3']
    len_k5 = r_vals['keys_k5']
    len_lw1 = r_vals['keys_lw1']
    len_lw3 = r_vals['keys_lw3']
    len_lw_prog = r_vals['keys_lw_prog']
    len_n1 = r_vals['keys_n1']
    len_n3 = r_vals['keys_n3']
    len_p8 = r_vals['keys_p8']
    len_t1 = r_vals['keys_t1']
    len_t2 = r_vals['keys_t2']
    len_t3 = r_vals['keys_t3']
    len_v1 = r_vals['keys_v1']
    len_v3 = r_vals['keys_v3']
    len_y0 = r_vals['keys_y0']
    len_y1 = r_vals['keys_y1']
    len_y3 = r_vals['keys_y3']
    len_x = r_vals['keys_x']
    len_z = r_vals['keys_z']
    len_p6 = r_vals['keys_p6']

    ##crop & pasture area
    df_rot = pd.DataFrame(lp_vars['v_phase_area'], index=['v_phase_area']).T #create a df of all the phase areas
    t_df_rot = df_rot.droplevel(1)
    phase_area = pd.merge(r_vals['rot']['phases'], t_df_rot, how='left', left_index=True, right_index=True) #merge full phase array with area array
    phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    inter['pasture_area'] = sum(df_rot[phase_is_pasture])
    inter['crop_area'] = sum(df_rot[~phase_is_pasture])


    ##animal numbers
    ##animal numbers
    sire_numbers = np.array(list(lp_vars['v_sire'].values()))
    sire_numbers_g0 = sire_numbers.reshape(len_g0)
    dam_numbers = np.array(list(lp_vars['v_dams'].values()))
    dam_numbers_k2tvanwziy1g1 = dam_numbers.reshape(len_k2, len_t1, len_v1, len_a, len_n1, len_lw1, len_z, len_i, len_y1, len_g1)
    offs_numbers = np.array(list(lp_vars['v_offs'].values()))
    offs_numbers_k3k5tvnwziaxy1g1 = offs_numbers.reshape(len_k3, len_k5, len_t3, len_v3, len_n3, len_lw3, len_z, len_i, len_a, len_x, len_y3, len_g3)


    ##dep - depreciation is yearly but for the profit and loss it is equally divided into each cash period
    dep = lp_vars['v_dep']/len_c #convert to dep per cashflow period
    inter['dep_c'] = pd.DataFrame(dep, index=[r_vals['keys_c']])  #convert to df with cashflow period as index
    ##exp
    # exp_fix =
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





def dse(inter,r_vals):
    '''

    :param inter:
    :return DSE per pasture hectare for each sheep group:
    '''




def profitloss_table(inter):
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
