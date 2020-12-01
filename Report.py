# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

This module should not import inputs (incase the inputs are adjusted during the exp so they will not be correct for reporting)
When creating report values try and do it in obvious spots even if you need to go out of the way to do it eg phases in rotation.py
"""

import pandas as pd


def intermediates(inter, r_vals, lp_vars):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in a dict to store intermidiate values.
    r_vals : Dict
        Pass in dict with all report values from precalcs.
    lp_vals : Dict
        Pass in dict with lp variable results from pyomo.

    Returns
    -------
    Here we manipulate report variables and lp result variables.
    Everything is stored in the 'inter' dict which is passed into each
    function which builds a table or figure.

    There are two options
        1.convert the dict to pandas dataframe: see crop and pasture area
        2.convert dict to numpy: see dse

    '''
    ##crop & pasture area
    df_rot = pd.DataFrame(lp_vars['v_phase_area'], index=['v_phase_area']).T #create a df of all the phase areas
    t_df_rot = df_rot.droplevel(1)
    phase_area = pd.merge(r_vals['rot']['phases'], t_df_rot, how='left', left_index=True, right_index=True) #merge full phase array with area array
    phase_is_pasture = phase_area.iloc[:,-2].isin(r_vals['rot']['all_pastures'])
    pasture_area = sum(df_rot[phase_is_pasture])


    ##dse
    dam_numbers = 5
    pasture_area = 9






def dse(inter):
    '''

    :param inter:
    :return DSE per pasture hectare for each sheep group:
    '''
    # dse = pd.DataFrame(data=inter['dse'], index=, columns='DSE')



def profitloss_table(inter):
    '''

    Parameters
    ----------
    inter : Dict
        Pass in dict with all intermidiate values required to calculate reports.

    Returns
    -------
    Report (table or figure etc).

    '''