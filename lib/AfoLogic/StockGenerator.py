# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John


Code structure
1. when reshaping input arrays use the i_len_? values but in the rest of the code use the len_? values.
2. sim params use their own pos and len variables.

Notes
1. Inputs need to be put together. One sheet in property.xl and one in universal.xl (and parameter sheet in universal so basically three sheets total)
2. Find and replace all the sheep input dicts so only the final ones used.


to do:
    check the inputs in structure. and update them



"""

"""
import functions from other modules
"""
# import datetime as dt
# import pandas as pd
import numpy as np
import pickle as pkl
#import matplotlib.pyplot as plt
import time
import collections
# from numba import jit
import sys,traceback
import os.path

from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import Finance as fin
from . import FeedSupplyStock as fsstk
from . import FeedsupplyFunctions as fsfun
from . import Sensitivity as sen
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import StockFunctions as sfun
from . import EmissionFunctions as efun
from . import Periods as per
from . import PlotViewer as pv
from . import Exceptions as exc


# np.seterr(all='raise')











# from memory_profiler import profile
# @profile
def generator(params={},r_vals={},nv={},pkl_fs_info={}, pkl_fs={}, stubble=None, plots = False):
    """
    A function to wrap the generator and post-processing that can be called by SheepPyomo.

    Called after the sensitivity variables have been updated.
    It populates the arrays by looping through the time periods
    Globally define arrays are used to transfer results to sheep_parameters()

    Returns
    -------
    None.
    """

    print("starting generator")
    generator_start = time.time()

    ######################
    ##background vars    #
    ######################
    na=np.newaxis
    a0_pos = sinp.stock['i_a0_pos']
    a1_pos = sinp.stock['i_a1_pos']
    b0_pos = sinp.stock['i_b0_pos']
    b1_pos = sinp.stock['i_b1_pos']
    d_pos = sinp.stock['i_d_pos']
    e0_pos = sinp.stock['i_e0_pos']
    e1_pos = sinp.stock['i_e1_pos']
    g_pos = -1
    i_pos = sinp.stock['i_i_pos']
    k2_pos = sinp.stock['i_k2_pos']
    k3_pos = sinp.stock['i_k3_pos']
    k5_pos = sinp.stock['i_k5_pos']
    n_pos = sinp.stock['i_n_pos']
    p_pos = sinp.stock['i_p_pos']
    w_pos = sinp.stock['i_w_pos']
    x_pos = sinp.stock['i_x_pos']
    y_pos = sinp.stock['i_y_pos']
    z_pos = sinp.stock['i_z_pos']

    ######################
    ##date               #
    ######################
    #todo The length of the year is hardwired to 364days which implies that n_periods is 52 and step is 7 so why bother calculating
    # if we want flexibility to adjust periods per year (to save model size) then this needs attention.
    ## define the periods - default (dams and sires)
    sim_years = sinp.stock['i_age_max']
    # sim_years = 4
    sim_years_offs = min(sinp.stock['i_age_max_offs'], sim_years)
    n_sim_periods, date_start_p, date_start_P, date_end_p, date_end_P, p_index_p, step \
        = sfun.f1_sim_periods(sinp.stock['i_sim_periods_year'], sim_years, pinp.sheep['i_o_len'])
    date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(p_pos+1, 0)))
    date_end_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_end_p, axis = tuple(range(p_pos+1, 0)))
    p_index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(p_index_p, axis = tuple(range(p_pos+1, 0)))
    ## define the periods - offs - these make the p axis customisable for offs which means they can be smaller
    n_sim_periods_offs, offs_date_start_p, offs_date_start_P, offs_date_end_p, offs_date_end_P, p_index_offs_p, step \
        = sfun.f1_sim_periods(sinp.stock['i_sim_periods_year'], sim_years_offs, pinp.sheep['i_o_len'])
    date_start_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_start_p, axis = tuple(range(p_pos+1, 0)))
    date_end_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_end_p, axis = tuple(range(p_pos+1, 0)))
    p_index_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(p_index_offs_p, axis = tuple(range(p_pos+1, 0)))
    mask_p_offs_p = p_index_p<=(n_sim_periods_offs-1)
    ##day of the year (mid-point day of each period)
    doy_pa1e1b1nwzida0e0b0xyg = date_start_pa1e1b1nwzida0e0b0xyg % 364 + step / 2
    ##day length
    dl_pa1e1b1nwzida0e0b0xyg = fun.f_daylength(doy_pa1e1b1nwzida0e0b0xyg, pinp.sheep['i_latitude'])
    ##days in each period
    days_period_pa1e1b1nwzida0e0b0xyg = date_end_pa1e1b1nwzida0e0b0xyg - date_start_pa1e1b1nwzida0e0b0xyg

    ###################################
    ## calculate masks                #
    ###################################
    ##masks required for initialising arrays
    mask_sire_inc_g0 = np.any(sinp.stock['i_mask_g0g3'] * pinp.sheep['i_g3_inc'], axis =1)
    mask_dams_inc_g1 = np.any(sinp.stock['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis =1)
    mask_yatf_inc_g2 = np.any(sinp.stock['i_mask_g2g3'] * pinp.sheep['i_g3_inc'], axis =1)
    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis =1)
    ##o/d mask - if dob is after the end of the sim then it is masked out -  the mask is created before the date of birth is adjusted to the start of a period however it is adjusted to the start of the next period so the mask won't cut out a birth event that actually would occur, additionally this is the birth of the first however the matrix sees the birth of average animal which is also later therefore if anything the mask will leave in unnecessary o slices
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = fun.f_expand(pinp.sheep['i_date_born1st_iog2'], i_pos, right_pos=g_pos, swap=True,
                                                      left_pos2=p_pos,right_pos2=i_pos, condition=mask_yatf_inc_g2, axis=g_pos,
                                                      condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    mask_o_dams = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2-uinp.sheep['prejoin_to_lamb_offset_approx']<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp). Offset is so that the o mask is controlled based on prejoining which has to happen so that the prejoining dvp exists even if the generator finishes before birth.
    mask_d_offs = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2-uinp.sheep['prejoin_to_lamb_offset_approx']<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp). Offset is so that the o mask is controlled based on prejoining which has to happen so that the prejoining dvp exists even if the generator finishes before birth.
    mask_x = pinp.sheep['i_gender_propn_x']>0
    bool_steady_state = pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1
    mask_node_is_fvp = pinp.general['i_node_is_fvp'] * (pinp.general['i_inc_node_periods']
                                                        or np.logical_not(bool_steady_state)) #node fvp/dvp are not included if it is steadystate.
    fvp_mask_dams = np.concatenate([mask_node_is_fvp[0:1], sinp.stock['i_fixed_fvp_mask_dams'], sinp.structuralsa['i_fvp_mask_dams'], mask_node_is_fvp[1:]]) #season start is at the front. because ss has to be first in the fvp/dvp
    fvp_mask_offs = np.concatenate([mask_node_is_fvp[0:1], sinp.structuralsa['i_fvp_mask_offs'], mask_node_is_fvp[1:]]) #season start is at the front. because ss has to be first in the fvp/dvp
    ##t1 mask
    mask_t1 = np.concatenate([np.full(pinp.sheep['i_n_dam_sales'], True), mask_sire_inc_g0])  # sale t slices plus transfer t slices

    ###################################
    ### axis len                      #
    ###################################
    ## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
    len_a0 = np.count_nonzero(pinp.sheep['i_mask_a'])
    len_a1 = np.count_nonzero(pinp.sheep['i_mask_a'])
    len_b0 = np.count_nonzero(sinp.stock['i_mask_b0_b1'])
    len_b1 = len(sinp.stock['i_mask_b0_b1'])
    len_c0 = len(sinp.general['i_enterprises_c0'])
    len_c1 = uinp.price_variation['len_c1']
    len_d = np.count_nonzero(mask_d_offs)
    len_e0 = np.max(pinp.sheep['i_join_cycles_ig1'])
    len_e1 = np.max(pinp.sheep['i_join_cycles_ig1'])
    len_g0 = np.count_nonzero(mask_sire_inc_g0)
    len_g1 = np.count_nonzero(mask_dams_inc_g1)
    len_g2 = np.count_nonzero(mask_dams_inc_g1) #same as dams
    len_g3 = np.count_nonzero(mask_offs_inc_g3)
    len_i = np.count_nonzero(pinp.sheep['i_mask_i'])
    lensire_i = np.count_nonzero(pinp.sheep['i_masksire_i'])
    len_k3 = len(pinp.sheep['i_k3_idx_offs'])
    len_n0 = sinp.structuralsa['i_n0_matrix_len']
    len_n1 = sinp.structuralsa['i_n1_matrix_len']
    len_n2 = sinp.structuralsa['i_n1_matrix_len'] #same as dams
    len_n3 = sinp.structuralsa['i_n3_matrix_len']
    len_p0 = int(step)
    len_p2 = sinp.stock['i_lag_wool']
    len_p3 = sinp.stock['i_lag_organs']
    len_o = np.count_nonzero(mask_o_dams)
    len_p = len(date_start_p)
    lenoffs_p = len(offs_date_start_p)
    len_p6 = len(per.f_feed_periods()) - 1 #-1 because the end feed period date is included
    len_p7 = len(per.f_season_periods(keys=True))
    len_p8 = np.count_nonzero(pinp.sheep['i_mask_p8'])
    len_q0	 = uinp.sheep['i_eqn_exists_q0q1'].shape[1]
    len_q1	 = len(uinp.sheep['i_eqn_reportvars_q1'])
    len_q2	 = np.max(uinp.sheep['i_eqn_reportvars_q1'])
    len_t0 = 1 #alway just one t slice for sires
    len_t1 = pinp.sheep['i_n_dam_sales'] + len_g0
    len_t2 = pinp.sheep['i_t2_len']
    len_t3 = pinp.sheep['i_t3_len']
    len_w0 = sinp.structuralsa['i_w0_len']
    len_w_prog = sinp.structuralsa['i_progeny_w2_len']
    len_x = np.count_nonzero(mask_x)
    len_y1 = np.count_nonzero(uinp.parameters['i_mask_y'])
    len_y2 = np.count_nonzero(uinp.parameters['i_mask_y'])
    len_y3 = np.count_nonzero(uinp.parameters['i_mask_y'])
    if bool_steady_state:
        len_z = 1
    else:
        len_z = np.count_nonzero(pinp.general['i_mask_z'])
    len_q = pinp.general['i_len_q'] #length of season sequence
    len_s = np.power(len_z,len_q - 1)

    ###length t used in generator due to pkl feedsupply (user can specify to generate with t axis - default is active t axis)
    ### t is always singleton for sires
    ### t is same len for dams and yatf in gen because yatf use dams fs
    if sinp.structuralsa['i_fs_use_pkl'] and sinp.structuralsa['i_generate_with_t']:
        len_gen_t1 = len_t1
        len_gen_t3 = len_t3
    else:
        len_gen_t1 = 1
        len_gen_t3 = 1


    ########################
    #dvp/fvp related inputs #
    ########################
    ##sire
    n_fs_sire = sinp.structuralsa['i_n0_len']
    n_fvp_periods_sire = sinp.structuralsa['i_n_fvp_period0']

    ##dams & yatf
    w_start_len1 = sinp.structuralsa['i_w_start_len1']
    n_fs_dams = sinp.structuralsa['i_n1_len']
    n_fvps_percondense_dams = np.count_nonzero(fvp_mask_dams)
    len_w1 = w_start_len1 * n_fs_dams ** n_fvps_percondense_dams
    len_w2 = len_w1 #yatf and dams are same
    len_nut_dams = (n_fs_dams ** n_fvps_percondense_dams)

    ###if generating for stubble then w axis is controlled by dmd levels rather than fvps and nut
    if stubble:
        w_start_len1 = 1
        n_fs_dams = 1
        len_w1 = stubble['dmd_pw'].shape[1]
        len_w2 = len_w1 #yatf and dams are same

    ##offspring
    w_start_len3 = sinp.structuralsa['i_w_start_len3']
    n_fs_offs = sinp.structuralsa['i_n3_len']
    n_fvps_percondense_offs= np.count_nonzero(fvp_mask_offs)
    len_w3 = w_start_len3 * n_fs_offs ** n_fvps_percondense_offs
    len_nut_offs = (n_fs_offs ** n_fvps_percondense_offs)
    fvp0_offset_ida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_fvp0_offset_ig3'], i_pos, right_pos=g_pos, condition=mask_offs_inc_g3, axis=g_pos,
                                           condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    fvp1_offset_ida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_fvp1_offset_ig3'], i_pos, right_pos=g_pos, condition=mask_offs_inc_g3, axis=g_pos,
                                           condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    fvp2_offset_ida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_fvp2_offset_ig3'], i_pos, right_pos=g_pos, condition=mask_offs_inc_g3, axis=g_pos,
                                           condition2=pinp.sheep['i_mask_i'], axis2=i_pos)

    ###if generating for stubble then w axis is controlled by dmd levels rather than fvps and nut
    if stubble:
        w_start_len3 = 1
        n_fs_offs = 1
        len_w3 = stubble['dmd_pw'].shape[1]

    ##season nodes - dvp must be added for each node (fvp is optional)
    date_node_zm = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'],numpy=True,axis=0) #have to use this rather than season periods because season periods get cut down in SE model, but we still might want all the season nodes as dvp/fvp.
    date_node_zidaebxygm = fun.f_expand(date_node_zm, z_pos-1, right_pos=-1)
    len_m = date_node_zidaebxygm.shape[-1]
    ###calc node date for each yr in generator
    date_node_ya1e1b1nwzidaebxygm = date_node_zidaebxygm + fun.f_expand(np.arange(np.ceil(sim_years)) * 364, p_pos - 1)
    ###set the node fvp to start at the next generator period following the node (needs to be next so that clustering works).
    idx_ya1e1b1nwzida0e0b0xygm = np.searchsorted(date_start_P, date_node_ya1e1b1nwzidaebxygm,
                                                 'left')  # use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_node_ya1e1b1nwzidaebxygm = date_start_P[idx_ya1e1b1nwzida0e0b0xygm]

    ###################################
    ### index arrays                  #
    ###################################
    # index_p = np.arange(300)#asarray(300)
    index_a0e0b0xyg = fun.f_expand(np.arange(len_a1), a0_pos)
    index_a1e1b1nwzida0e0b0xyg = fun.f_expand(np.arange(len_a1), a1_pos)
    index_b0xyg = fun.f_expand(np.arange(len_b0), b0_pos)
    index_b1nwzida0e0b0xyg = fun.f_expand(np.arange(len_b1), b1_pos)
    index_l = np.arange(sinp.stock['i_len_l']) #gbal
    index_sc = np.arange(sinp.stock['i_len_s']) #scan
    index_d = np.arange(len_d)
    index_da0e0b0xyg = fun.f_expand(index_d, d_pos)
    index_e = np.arange(np.max(pinp.sheep['i_join_cycles_ig1']))
    index_e1b1nwzida0e0b0xyg = fun.f_expand(index_e, e1_pos)
    index_e0b0xyg = fun.f_expand(index_e, e0_pos)
    index_g0 = np.arange(len_g0)
    index_g1 = np.arange(len_g1)
    index_g9 = index_g1
    index_g1g = index_g1[:,na]
    index_i = np.arange(len_i)
    index_ida0e0b0xyg = fun.f_expand(index_i, i_pos)
    index_i9 = index_i
    index_k3k5tva1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(len_k3),k3_pos)
    index_p0 = np.arange(len_p0)
    index_m0 = np.arange(12)*2  #2hourly steps for chill calculations with actual time
    index_z = np.arange(len_z)
    index_w0 = np.arange(len_w0)
    index_wzida0e0b0xyg0 = fun.f_expand(index_w0, w_pos)
    index_w1 = np.arange(len_w1)
    index_wzida0e0b0xyg1 = fun.f_expand(index_w1, w_pos)
    index_w2 = np.arange(len_w_prog)
    index_w3 = np.arange(len_w3)
    index_wzida0e0b0xyg3 = fun.f_expand(index_w3, w_pos)
    index_tva1e1b1nw8zida0e0b0xyg1 = fun.f_expand(np.arange(len_t1), p_pos-1)
    index_tva1e1b1nw8zida0e0b0xyg1w9 = index_tva1e1b1nw8zida0e0b0xyg1[...,na]
    index_t2 = np.arange(len_t2)
    index_tva1e1b1nwzida0e0b0xyg2w9 = fun.f_expand(index_t2, p_pos-2)
    index_tva1e1b1nw8zida0e0b0xyg3 = fun.f_expand(np.arange(len_t3), p_pos-1)
    index_tva1e1b1nw8zida0e0b0xyg3w9 = index_tva1e1b1nw8zida0e0b0xyg3[...,na]
    index_xyg = fun.f_expand(np.arange(len_x), x_pos)

    prejoin_tup = (a1_pos, b1_pos, e1_pos)
    season_tup = (z_pos)


    ############################
    ### initialise arrays      #
    ############################
    '''only if assigned with a slice'''
    ##unique array shapes required to initialise arrays
    qg0 = (len_q0, len_q1, len_q2, len_t0, len_p, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
    qg1 = (len_q0, len_q1, len_q2, len_t1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    qg2 = (len_q0, len_q1, len_q2, len_t2, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g1)
    qg3 = (len_q0, len_q1, len_q2, len_t3, lenoffs_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    tpg0 = (1, len_p, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
    pg1 = (len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    pg3 = (lenoffs_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    tpg1 = (len_gen_t1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    tpg2 = (len_gen_t1, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g1)
    tpg3 = (len_gen_t3, lenoffs_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    c1tpg0 = (len_c1, 1, len_p, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
    c1tpg1 = (len_c1, len_gen_t1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    c1tpg2 = (len_c1, len_gen_t1, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g1)
    c1tpg3 = (len_c1, len_gen_t3, lenoffs_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    p2g0 = (len_p2, 1, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
    p2g1 = (len_p2, len_gen_t1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    p2g2 = (len_p2, len_gen_t1, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g2)
    p2g3 = (len_p2, len_gen_t3, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    p3g0 = (len_p3, 1, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0) #t is always singleton for sires
    p3g1 = (len_p3, len_gen_t1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    p3g2 = (len_p3, len_gen_t1, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g2)
    p3g3 = (len_p3, len_gen_t3, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)

    ###special one to initialise variables for the generator that are only used for some functions (but are required to have correct shape for condensing)
    tag1 = (len_gen_t1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)

    ##output variables for postprocessing & reporting
    dtype='float32' #using 64 was getting slow
    dtypeint='int32' #using 64 was getting slow

    ##sire
    ###array for generator
    omer_history_start_p3g0 = np.zeros(p3g0, dtype = 'float64')
    d_cfw_history_start_p2g0 = np.zeros(p2g0, dtype = 'float64')
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 = np.zeros(c1tpg0, dtype =dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg0 = np.zeros(c1tpg0, dtype =dtype)
    w_w_yatf = np.array([0])
    ###arrays for postprocessing
    o_numbers_start_tpsire = np.zeros(tpg0, dtype =dtype)
    o_numbers_end_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ffcfw_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ffcfw_condensed_tpsire = np.zeros(tpg0, dtype =dtype)
    o_nw_start_tpsire = np.zeros(tpg0, dtype=dtype)
    o_lw_tpsire = np.zeros(tpg0, dtype =dtype)
    o_pi_tpsire = np.zeros(tpg0, dtype =dtype)
    o_mei_solid_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ch4_animal_tpsire = np.zeros(tpg0, dtype =dtype)
    o_n2o_animal_tpsire = np.zeros(tpg0, dtype =dtype)
    o_cfw_tpsire = np.zeros(tpg0, dtype =dtype)
    o_sl_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ss_tpsire = np.zeros(tpg0, dtype =dtype)
    o_fd_tpsire = np.zeros(tpg0, dtype =dtype)
    o_fd_min_tpsire = np.zeros(tpg0, dtype =dtype)
    o_rc_start_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ebg_tpsire = np.zeros(tpg0, dtype =dtype)
    ###arrays for report variables
    r_compare_q0q1q2tpsire = np.zeros(qg0, dtype = dtype) #empty arrays to store different return values from the equation systems in the p loop.
    r_salegrid_c1tpa1e1b1nwzida0e0b0xyg0 = np.zeros(c1tpg0, dtype =dtype)

    ##dams
    ###array for generator
    omer_history_start_p3g1 = np.zeros(p3g1, dtype = 'float64')
    d_cfw_history_start_p2g1 = np.zeros(p2g1, dtype = 'float64')
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 = np.zeros((len_c1,)+(len_t1,)+tpg1[1:], dtype =dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg1 = np.zeros(c1tpg1, dtype =dtype)
    ###arrays for postprocessing
    o_numbers_start_tpdams = np.zeros(tpg1, dtype =dtype) #default 1 so that dvp0 (p0) has start numbers
    o_numbers_join_tpdams = np.zeros(tpg1, dtype =dtype)
    o_numbers_end_tpdams = np.zeros(tpg1, dtype =dtype) #default 1 so that transfer can exist for dvps before weaning
    o_ffcfw_tpdams = np.zeros(tpg1, dtype =dtype)
    o_ffcfw_season_tpdams = np.zeros(tpg1, dtype =dtype)
    o_ffcfw_condensed_tpdams = np.zeros(tpg1, dtype =dtype)
    o_nw_start_tpdams = np.zeros(tpg1, dtype = dtype)
    o_mortality_dams = np.zeros(tpg1, dtype =dtype)
    o_lw_tpdams = np.zeros(tpg1, dtype =dtype)
    o_pi_tpdams = np.zeros(tpg1, dtype =dtype)
    o_mei_solid_tpdams = np.zeros(tpg1, dtype =dtype)
    o_ch4_animal_tpdams = np.zeros(tpg1, dtype =dtype)
    o_n2o_animal_tpdams = np.zeros(tpg1, dtype =dtype)
    o_cfw_tpdams = np.zeros(tpg1, dtype =dtype)
    # o_gfw_tpdams = np.zeros(tpg1, dtype =dtype)
    o_sl_tpdams = np.zeros(tpg1, dtype =dtype)
    o_ss_tpdams = np.zeros(tpg1, dtype =dtype)
    o_fd_tpdams = np.zeros(tpg1, dtype =dtype)
    o_fd_min_tpdams = np.zeros(tpg1, dtype =dtype)
    o_rc_start_tpdams = np.zeros(tpg1, dtype =dtype)
    o_ebg_tpdams = np.zeros(tpg1, dtype =dtype)
    o_cfw_ltwadj_tpdams = np.zeros(tpg1, dtype =dtype)
    o_fd_ltwadj_tpdams = np.zeros(tpg1, dtype =dtype)
    o_n_sire_tpa1e1b1nwzida0e0b0xyg1g0p8 = np.zeros((len_gen_t1, len_p, 1, 1, 1, 1, 1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1,len_g0,len_p8), dtype =dtype)
    ###arrays for report variables
    r_compare_q0q1q2tpdams = np.zeros(qg1, dtype = dtype) #empty arrays to store different return values from the equation systems in the p loop.
    r_foo_tpdams = np.zeros(tpg1, dtype = dtype)
    r_dmd_tpdams = np.zeros(tpg1, dtype = dtype)
    r_evg_tpdams = np.zeros(tpg1, dtype = dtype)
    r_intake_f_tpdams = np.zeros(tpg1, dtype = dtype)
    r_md_solid_tpdams = np.zeros(tpg1, dtype = dtype)
    r_mp2_tpdams = np.zeros(tpg1, dtype = dtype)
    r_d_cfw_tpdams =  np.zeros(tpg1, dtype = dtype)
    r_wbe_tpdams = np.zeros(tpg1, dtype = dtype)
    r_salegrid_c1tpa1e1b1nwzida0e0b0xyg1 = np.zeros(c1tpg1, dtype =dtype)

    ##yatf
    ###array for generator
    omer_history_start_p3g2 = np.zeros(p3g2, dtype = 'float64')
    d_cfw_history_start_p2g2 = np.zeros(p2g2, dtype = 'float64')
    ###array for postprocessing
    o_numbers_start_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_numbers_end_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_ffcfw_start_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_wean_w_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_ffcfw_condensed_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_pi_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_mei_solid_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_ch4_animal_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_n2o_animal_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_cfw_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_gfw_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_sl_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_ss_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_fd_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_fd_min_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_rc_start_tpyatf = np.zeros(tpg2, dtype =dtype)
    ###arrays for report variables
    r_compare_q0q1q2tpyatf = np.zeros(qg2, dtype = dtype) #empty arrays to store different return values from the equation systems in the p loop.
    r_ffcfw_start_tpyatf = np.zeros(tpg2, dtype =dtype)   # requires a variable separate from o_ffcfw_start_tpyatf so that it is only stored when days_period > 0
    r_ebg_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_evg_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_mem_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_mei_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_mei_solid_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_propn_solid_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_pi_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_kg_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_mp2_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_intake_f_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_nw_start_tpyatf = np.zeros(tpg2, dtype = dtype)
    r_salegrid_c1tpa1e1b1nwzida0e0b0xyg2 = np.zeros(c1tpg2, dtype = dtype)
    salevalue_perdam_tpa1e1b1nwzida0e0b0xyg2 = np.zeros(tpg2, dtype = dtype)


    ##offs
    ###array for generator
    omer_history_start_p3g3 = np.zeros(p3g3, dtype = 'float64')
    d_cfw_history_start_p2g3 = np.zeros(p2g3, dtype = 'float64')
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg3 = np.zeros((len_c1,)+(len_t3,)+tpg3[1:], dtype =dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg3 = np.zeros(c1tpg3, dtype =dtype)
    ###array for postprocessing
    o_numbers_start_tpoffs = np.zeros(tpg3, dtype =dtype) # filled with the initial numbers later, so that dvp0 (p0) has start numbers.
    o_numbers_end_tpoffs = np.zeros(tpg3, dtype =dtype) # filled with the initial numbers later, so that transfer can exist for dvps before weaning
    o_ffcfw_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ffcfw_season_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ffcfw_condensed_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_nw_start_tpoffs = np.zeros(tpg3, dtype=dtype)
    o_mortality_offs = np.zeros(tpg3, dtype =dtype)
    o_lw_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_pi_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_mei_solid_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ch4_animal_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_n2o_animal_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_cfw_tpoffs = np.zeros(tpg3, dtype =dtype)
    # o_gfw_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_sl_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ss_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_fd_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_fd_min_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_rc_start_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ebg_tpoffs = np.zeros(tpg3, dtype =dtype)
    ###arrays for report variables
    r_compare_q0q1q2tpoffs = np.zeros(qg3, dtype = dtype) #empty arrays to store different return values from the equation systems in the p loop.
    r_wbe_tpoffs = np.zeros(tpg3, dtype =dtype)
    r_salegrid_c1tpa1e1b1nwzida0e0b0xyg3 = np.zeros(c1tpg3, dtype =dtype)



    #########################################
    #  management, age, date, timing inputs #
    #########################################
    ##gender propn yatf
    gender_propn_xyg = fun.f_expand(pinp.sheep['i_gender_propn_x'], x_pos, condition=mask_x, axis=0).astype(dtype)

    ##drys management - two versions: first one controls the bound and the second ones (est) are estimates used in the generator.
    ## the bound version is not used in the generator otherwise randomness will be introduced. Because changing if drys are retained or not
    ## alters the numbers in the generator but doesn't necessarily alter the selected flock structure.
    dry_retained_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_dry_retained_forced_o'], p_pos
                                                       , condition=mask_o_dams, axis=p_pos)
    est_drys_retained_scan_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_drys_retained_scan_est_o'], p_pos
                                                                 , condition=mask_o_dams, axis=p_pos)
    est_drys_retained_birth_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_drys_retained_birth_est_o'], p_pos
                                                                  , condition=mask_o_dams, axis=p_pos)

    ##join
    join_cycles_ida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_join_cycles_ig1'], i_pos, right_pos=g_pos,
                                           condition=mask_dams_inc_g1, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)

    ##lamb and lost
    gbal_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_gbal_og1'], p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                              axis=g_pos, condition2=mask_o_dams, axis2=p_pos) #need axis up to p so that p association can be applied
    gbal_da0e0b0xyg3 = fun.f_expand(pinp.sheep['i_gbal_og1'], d_pos, right_pos=g_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                   condition2=mask_d_offs, axis2=d_pos) #need axis up to p so that p association can be applied

    ##scanning
    scan_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_scan_og1'], p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                              axis=g_pos, condition2=mask_o_dams, axis2=p_pos) #need axis up to p so that p association can be applied
    scan_da0e0b0xyg3 = fun.f_expand(pinp.sheep['i_scan_og1'], d_pos, right_pos=g_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                   condition2=mask_d_offs, axis2=d_pos) #need axis up to p so that p association can be applied

    ##Chill adjustment based on litter size and scanning. Note: adjusted later so only active if scanning
    #todo the scaling across the b1 axis could be improved by including scan_std for the flock & std DSE/hd (replicating the calculations in the PregScanning exp.xl)
    #This could account for the number of dams re-allocated based on min(DSE of multiples in exposed, DSE of singles in sheltered)
    # The current calculation is all multiples allocated to sheltered paddocks and all singles to exposed paddocks.
    chill_adj_b1nwzida0e0b0xyg1 = pinp.sheep['i_chill_adj'] * fun.f_expand(sinp.stock['i_chill_adj_b1'], b1_pos)

    ##post weaning management
    wean_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_wean_og1'], p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                              axis=g_pos, condition2=mask_o_dams, axis2=p_pos) #need axis up to p so that p association can be applied

    ##association between offspring and sire/dam (used to determine wean age of sire and dams based on the inputted wean age of offs)
    a_g0_g1 = pinp.sheep['ia_g0_g1'][mask_dams_inc_g1]
    a_g3_g0 = pinp.sheep['ia_g3_g0'][mask_sire_inc_g0]   # the sire association (purebred B, M & T) are all based on purebred B because there are no purebred M & T inputs
    a_g3_g1 = pinp.sheep['ia_g3_g1'][mask_dams_inc_g1]   # if BMT exist then BBM exist, and they will be in slice 1, therefore the association value doesn't need to be adjusted for "prior exclusions"

    ##age weaning- used to calc wean date and also to calc p1 stuff, sire and dams have no active a0 slice therefore just take the first slice
    ###note: if age_wean_g3 gets a d axis it needs to be the same for all animals that get clustered (see date born below)
    age_wean1st_a0e0b0xyg3 = fun.f_expand(pinp.sheep['i_age_wean_a0g3'], a0_pos, right_pos=g_pos, condition=mask_offs_inc_g3,
                                         axis=g_pos, condition2=pinp.sheep['i_mask_a'], axis2=a0_pos)
    age_wean1st_e0b0xyg0 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g0],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
    age_wean1st_e0b0xyg1 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g1],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end

    ##date first lamb is born - need to apply 'i' mask to these inputs - make sure animals are born at beginning of gen period
    date_born1st_ida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_date_born1st_ig0'], i_pos, right_pos=g_pos, condition=mask_sire_inc_g0,
                                            axis=g_pos, condition2=pinp.sheep['i_masksire_i'], axis2=i_pos)
    date_born1st_ida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_date_born1st_ig1'], i_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                            axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos)
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2[mask_o_dams,...] #input read in in the mask section
    date_born1st_ida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_date_born1st_idg3'], d_pos, right_pos=g_pos,
                                            condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i']
                                           , axis2=i_pos, condition3=mask_d_offs, axis3=d_pos)
    date_born1st_ida0e0b0xyg3[:,len_k3-1:,...] = date_born1st_ida0e0b0xyg3[:,len_k3-1:len_k3,...] #for animals in the same d cluster date born must be the same (so that the dvp and fvp dates are the same for all animals that get clustered)

    ##mating
    sire_propn_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_sire_propn_oig1'], i_pos, right_pos=g_pos, swap=True,
                                                   left_pos2=p_pos,right_pos2=i_pos, condition=mask_dams_inc_g1, axis=g_pos,
                                                    condition2=pinp.sheep['i_mask_i'], axis2=i_pos, condition3=mask_o_dams, axis3=p_pos)
    sire_periods_p8g0 = fun.f_expand(pinp.sheep['i_sire_periods_p8g0'], condition=mask_sire_inc_g0, axis=g_pos, condition2=pinp.sheep['i_mask_p8'], axis2=0)
    sire_periods_g0p8 = np.swapaxes(sire_periods_p8g0, 0, 1) #can't swap in function above because g needs to be in pos-1

    ##propn of dams mated (bound) - default is inf which gets skipped in the bound constraint hence the model can optimise the propn mated.
    ##used for bound - the bound version is not used in the generator otherwise randomness could be introduced. Because changing est propn mated
    ## alters the numbers in the generator but doesn't necessarily alter the selected flock structure.
    prop_dams_mated_og1 = fun.f_sa(np.array([999],dtype=float), sen.sav['bnd_propn_dams_mated_og1'], 5) #999 just an arbitrary value used then converted to np.inf because np.inf causes errors in the f_update which is called by f_sa
    prop_dams_mated_og1[prop_dams_mated_og1==999] = np.inf
    prop_dams_mated_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(prop_dams_mated_og1, left_pos=p_pos, right_pos=-1
                                                         , condition=mask_o_dams, axis=p_pos, condition2=mask_dams_inc_g1, axis2=-1)
    ##estimated propn of dams mated (generator)
    est_prop_dams_mated_og1 = fun.f_sa(np.array([1],dtype=float), sen.sav['est_propn_dams_mated_og1'], 5) #if an estimate is not specified then use 100% is mated.
    est_prop_dams_mated_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(est_prop_dams_mated_og1, left_pos=p_pos, right_pos=-1
                                                         , condition=mask_o_dams, axis=p_pos, condition2=mask_dams_inc_g1, axis2=-1)

    ##Shearing date - set to be on the last day of a generator period
    ###sire
    date_shear_sida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_date_shear_sixg0'], x_pos, right_pos=g_pos, swap=True
                                          ,left_pos2=i_pos,right_pos2=x_pos, condition=mask_sire_inc_g0, axis=g_pos,
                                           condition2=pinp.sheep['i_masksire_i'], axis2=i_pos
                                          )[...,0:1,:,:] #slice x-axis for only male
    mask_shear_g0 = np.max(date_shear_sida0e0b0xyg0<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg0 = date_shear_sida0e0b0xyg0[mask_shear_g0]
    ###dam
    date_shear_sida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_date_shear_sixg1'], x_pos, right_pos=g_pos, swap=True,left_pos2=i_pos,right_pos2=x_pos,
                                           condition=mask_dams_inc_g1, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos
                                          )[...,1:2,:,:] #slice x-axis for only female
    mask_shear_g1 = np.max(date_shear_sida0e0b0xyg1<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg1 = date_shear_sida0e0b0xyg1[mask_shear_g1]
    ###off - the first shearing must occur as offspring because if yatf were shorn then all lambs would have to be shorn (ie no scope to not shear the lambs that are going to be fed up and sold)
    #### the offspring decision variables are not linked to the yatf (which are in the dam decision variables) and it would require doubling the dam DVs to have shorn and unshorn yatf
    ####note: if age_wean_g3 gets a d axis it needs to be the same for all animals that get clustered (see date born below)
    date_shear_sida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_date_shear_sixg3'], x_pos, right_pos=g_pos, swap=True,left_pos2=i_pos,right_pos2=x_pos,
                                           condition=mask_offs_inc_g3, axis=g_pos, condition2=pinp.sheep['i_mask_i'], axis2=i_pos,
                                           condition3=mask_x, axis3=x_pos)
    mask_shear_g3 = np.max(date_shear_sida0e0b0xyg3<=offs_date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg3 = date_shear_sida0e0b0xyg3[mask_shear_g3]
    len_s3 = np.count_nonzero(mask_shear_g3)

    ##if generating for stubble then overwrite some of these inputs to match the stubble trial
    if stubble:
        ##shearing
        date_shear_sida0e0b0xyg1[...] = pinp.stubble['shear_date']
        date_shear_sida0e0b0xyg3[...] = pinp.stubble['shear_date']
        ###birth control
        date_born1st_oa1e1b1nwzida0e0b0xyg2[...] = pinp.stubble['lambing_date']

    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##select the genotype
    a_c2_c0 = pinp.sheep['a_c2_c0']
    i_g3_inc = pinp.sheep['i_g3_inc']
    ##if generating for stubble then overwrite genotype selection
    if stubble:
        a_c2_c0 = pinp.stubble['a_c2_c0']
        i_g3_inc = pinp.stubble['i_g3_inc']

    ##association for the retained t of each g slice
    a_t_g1 = np.arange(pinp.sheep['i_n_dam_sales'], pinp.sheep['i_n_dam_sales']+len_g1)
    a_t_tpg1 = fun.f_expand(a_t_g1, p_pos - 2, right_pos=-1)
    a_t_g3 = np.array([0]) #g axis doesn't affect t for offs
    a_t_tpg3 = fun.f_expand(a_t_g3, p_pos - 2, right_pos=-1)

    ##convert input params from c to g
    ###production params
    agedam_propn_da0e0b0xyg0, agedam_propn_da0e0b0xyg1, agedam_propn_da0e0b0xyg2, agedam_propn_da0e0b0xyg3 = \
        sfun.f1_c2g(uinp.parameters['i_agedam_propn_std_dc2'], uinp.parameters['i_agedam_propn_y'], a_c2_c0, i_g3_inc,
                    uinp.parameters['i_agedam_propn_pos'], condition=mask_o_dams, axis=d_pos) #yatf and off never used
    agedam_propn_da0e0b0xyg0 = agedam_propn_da0e0b0xyg0 / np.sum(agedam_propn_da0e0b0xyg0, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg1 = agedam_propn_da0e0b0xyg1 / np.sum(agedam_propn_da0e0b0xyg1, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg2 = agedam_propn_da0e0b0xyg2 / np.sum(agedam_propn_da0e0b0xyg2, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg3 = agedam_propn_da0e0b0xyg3 / np.sum(agedam_propn_da0e0b0xyg3, axis=d_pos) #scale unmasked slices to a total of 1
    aw_propn_wean_yg0, aw_propn_wean_yg1, aw_propn_wean_yg2, aw_propn_wean_yg3 = sfun.f1_c2g(uinp.parameters['i_aw_propn_wean_c2'], uinp.parameters['i_aw_wean_y'], a_c2_c0, i_g3_inc)
    aw_propn_birth_yg0, aw_propn_birth_yg1, aw_propn_birth_yg2, aw_propn_birth_yg3 = sfun.f1_c2g(uinp.parameters['i_aw_propn_birth_c2'], uinp.parameters['i_aw_birth_y'], a_c2_c0, i_g3_inc) #only for yatf
    bw_propn_wean_yg0, bw_propn_wean_yg1, bw_propn_wean_yg2, bw_propn_wean_yg3 = sfun.f1_c2g(uinp.parameters['i_bw_propn_wean_c2'], uinp.parameters['i_bw_wean_y'], a_c2_c0, i_g3_inc)
    bw_propn_birth_yg0, bw_propn_birth_yg1, bw_propn_birth_yg2, bw_propn_birth_yg3 = sfun.f1_c2g(uinp.parameters['i_bw_propn_birth_c2'], uinp.parameters['i_bw_birth_y'], a_c2_c0, i_g3_inc) #only for yatf
    cfw_propn_yg0, cfw_propn_yg1, cfw_propn_yg2, cfw_propn_yg3 = sfun.f1_c2g(uinp.parameters['i_cfw_propn_c2'], uinp.parameters['i_cfw_propn_y'], a_c2_c0, i_g3_inc)
    fl_birth_yg0, fl_birth_yg1, fl_birth_yg2, fl_birth_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_birth_c2'], uinp.parameters['i_fl_birth_y'], a_c2_c0, i_g3_inc)
    fl_shear_yg0, fl_shear_yg1, fl_shear_yg2, fl_shear_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_shear_c2'], uinp.parameters['i_fl_shear_y'], a_c2_c0, i_g3_inc)
    mw_propn_wean_yg0, mw_propn_wean_yg1, mw_propn_wean_yg2, mw_propn_wean_yg3 = sfun.f1_c2g(uinp.parameters['i_mw_propn_wean_c2'], uinp.parameters['i_mw_wean_y'], a_c2_c0, i_g3_inc)
    mw_propn_birth_yg0, mw_propn_birth_yg1, mw_propn_birth_yg2, mw_propn_birth_yg3 = sfun.f1_c2g(uinp.parameters['i_mw_propn_birth_c2'], uinp.parameters['i_mw_birth_y'], a_c2_c0, i_g3_inc) #only for yatf
    pss_std_yg0, pss_std_yg1, pss_std_yg2, pss_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lss_std_c2'], uinp.parameters['i_lss_std_y'], a_c2_c0, i_g3_inc)
    pstr_std_yg0, pstr_std_yg1, pstr_std_yg2, pstr_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lstr_std_c2'], uinp.parameters['i_lstr_std_y'], a_c2_c0, i_g3_inc)
    pstw_std_yg0, pstw_std_yg1, pstw_std_yg2, pstw_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lstw_std_c2'], uinp.parameters['i_lstw_std_y'], a_c2_c0, i_g3_inc)
    scan_std_yg0, scan_std_yg1, scan_std_yg2, scan_std_yg3 = sfun.f1_c2g(uinp.parameters['i_scan_std_c2'], uinp.parameters['i_scan_std_y'], a_c2_c0, i_g3_inc) #scan_std_yg2/3 not used
    scan_std_doj_yg0, scan_std_doj_yg1, scan_std_doj_yg2, scan_std_doj_yg3 = sfun.f1_c2g(uinp.parameters['i_scan_std_doj_c2'], uinp.parameters['i_scan_std_doj_y'], a_c2_c0, i_g3_inc)
    scan_dams_std_yg3 = scan_std_yg1 #offs needs to be the same as dams because scan_std is used to calc starting propn of BTRT which is dependent on dams scanning
    nlb_yg0, nlb_yg1, nlb_yg2, nlb_yg3 = sfun.f1_c2g(uinp.parameters['i_nlb_c2'], uinp.parameters['i_nlb_y'], a_c2_c0, i_g3_inc)
    sfd_yg0, sfd_yg1, sfd_yg2, sfd_yg3 = sfun.f1_c2g(uinp.parameters['i_sfd_c2'], uinp.parameters['i_sfd_y'], a_c2_c0, i_g3_inc)
    sfw_yg0, sfw_yg1, sfw_yg2, sfw_yg3 = sfun.f1_c2g(uinp.parameters['i_sfw_c2'], uinp.parameters['i_sfw_y'], a_c2_c0, i_g3_inc)
    srw_female_yg0, srw_female_yg1, srw_female_yg2, srw_female_yg3 = sfun.f1_c2g(uinp.parameters['i_srw_c2'], uinp.parameters['i_srw_y'], a_c2_c0, i_g3_inc) #srw of a female of the given genotype (this is the definition of the inputs)

    ###p1 variation params (used for mort)
    cv_weight_sire, cv_weight_dams, cv_weight_yatf, cv_weight_offs = sfun.f1_c2g(uinp.parameters['i_cv_weight_c2'], uinp.parameters['i_cv_weight_y'], a_c2_c0, i_g3_inc)
    cv_bw_sire, cv_bw_dams, cv_bw_yatf, cv_bw_offs = sfun.f1_c2g(uinp.parameters['i_cv_bw_c2'], uinp.parameters['i_cv_bw_y'], a_c2_c0, i_g3_inc)
    cv_cs_sire, cv_cs_dams, cv_cs_yatf, cv_cs_offs = sfun.f1_c2g(uinp.parameters['i_cv_cs_c2'], uinp.parameters['i_cv_cs_y'], a_c2_c0, i_g3_inc)
    sd_ebg_sire, sd_ebg_dams, sd_ebg_yatf, sd_ebg_offs = sfun.f1_c2g(uinp.parameters['i_sd_ebg_c2'], uinp.parameters['i_sd_ebg_y'], a_c2_c0, i_g3_inc)

    ###sim params
    ca_sire, ca_dams, ca_yatf, ca_offs = sfun.f1_c2g(uinp.parameters['i_ca_c2'], uinp.parameters['i_ca_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ca_pos'])
    cb0_sire, cb0_dams, cb0_yatf, cb0_offs = sfun.f1_c2g(uinp.parameters['i_cb0_c2'], uinp.parameters['i_cb0_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cb0_pos'])
    #### cc_yatf needs an active p axis to represent parameter change when age < 30 days.
    cc_sire, cc_dams, cc_yatf, cc_offs = sfun.f1_c2g(uinp.parameters['i_cc_c2'], uinp.parameters['i_cc_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cc_pos'])
    cd_sire, cd_dams, cd_yatf, cd_offs = sfun.f1_c2g(uinp.parameters['i_cd_c2'], uinp.parameters['i_cd_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cd_pos'])
    ce_sire, ce_dams, ce_yatf, ce_offs = sfun.f1_c2g(uinp.parameters['i_ce_c2'], uinp.parameters['i_ce_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ce_pos'], condition=mask_o_dams, axis=d_pos)
    ce_offs = sfun.f1_c2g(uinp.parameters['i_ce_c2'], uinp.parameters['i_ce_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ce_pos'], condition=mask_d_offs, axis=d_pos)[3]  #re calc off using off d mask
    cf_sire, cf_dams, cf_yatf, cf_offs = sfun.f1_c2g(uinp.parameters['i_cf_c2'], uinp.parameters['i_cf_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cf_pos'])
    cg_sire, cg_dams, cg_yatf, cg_offs = sfun.f1_c2g(uinp.parameters['i_cg_c2'], uinp.parameters['i_cg_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cg_pos'])
    ch_sire, ch_dams, ch_yatf, ch_offs = sfun.f1_c2g(uinp.parameters['i_ch_c2'], uinp.parameters['i_ch_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ch_pos'])
    ci_sire, ci_dams, ci_yatf, ci_offs = sfun.f1_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ci_pos'])
    ck_sire, ck_dams, ck_yatf, ck_offs = sfun.f1_c2g(uinp.parameters['i_ck_c2'], uinp.parameters['i_ck_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_ck_pos'])
    cl0_sire, cl0_dams, cl0_yatf, cl0_offs = sfun.f1_c2g(uinp.parameters['i_cl0_c2'], uinp.parameters['i_cl0_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cl0_pos'])
    cl1_sire, cl1_dams, cl1_yatf, cl1_offs = sfun.f1_c2g(uinp.parameters['i_cl1_c2'], uinp.parameters['i_cl1_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cl1_pos'])
    cl_sire, cl_dams, cl_yatf, cl_offs = sfun.f1_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cl_pos'])
    cm_sire, cm_dams, cm_yatf, cm_offs = sfun.f1_c2g(uinp.parameters['i_cm_c2'], uinp.parameters['i_cm_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cm_pos'])
    cn_sire, cn_dams, cn_yatf, cn_offs = sfun.f1_c2g(uinp.parameters['i_cn_c2'], uinp.parameters['i_cn_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cn_pos'])
    cp_sire, cp_dams, cp_yatf, cp_offs = sfun.f1_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cp_pos'])
    cr_sire, cr_dams, cr_yatf, cr_offs = sfun.f1_c2g(uinp.parameters['i_cr_c2'], uinp.parameters['i_cr_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cr_pos'])
    crd_sire, crd_dams, crd_yatf, crd_offs = sfun.f1_c2g(uinp.parameters['i_crd_c2'], uinp.parameters['i_crd_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_crd_pos'])
    cu0_sire, cu0_dams, cu0_yatf, cu0_offs = sfun.f1_c2g(uinp.parameters['i_cu0_c2'], uinp.parameters['i_cu0_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cu0_pos'])
    cu1_sire, cu1_dams, cu1_yatf, cu1_offs = sfun.f1_c2g(uinp.parameters['i_cu1_c2'], uinp.parameters['i_cu1_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cu1_pos'])
    cu2_sire, cu2_dams, cu2_yatf, cu2_offs = sfun.f1_c2g(uinp.parameters['i_cu2_c2'], uinp.parameters['i_cu2_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cu2_pos'])
    cw_sire, cw_dams, cw_yatf, cw_offs = sfun.f1_c2g(uinp.parameters['i_cw_c2'], uinp.parameters['i_cw_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cw_pos'])
    cx_sire, cx_dams, cx_yatf, cx_offs = sfun.f1_c2g(uinp.parameters['i_cx_c2'], uinp.parameters['i_cx_y'], a_c2_c0, i_g3_inc, uinp.parameters['i_cx_pos'])
    ##Convert the cl0 & cl1 to cb1 (dams and yatf only need cb1, sires and offs don't have b1 axis)
    cb1_dams = cl0_dams[:,sinp.stock['a_nfoet_b1']] + cl1_dams[:,sinp.stock['a_nyatf_b1']]
    cb1_yatf = cl0_yatf[:,sinp.stock['a_nfoet_b1']] + cl1_yatf[:,sinp.stock['a_nyatf_b1']]
    ###Alter select slices only for yatf (yatf don't have cb0 axis - instead they use cb1 so it aligns with dams)
    ###The b1 parameters that are relevant to the dams relate to either number of foetus (entered as cl0) or number of yatf (entered as cl1).
    ### For the yatf the b1 axis also represents their BTRT, therefore some parameters that change due to
    ### birth type and rear type (BTRT - b0) are needed in the b1 axis for the yatf to estimate values for the yatf
    ### This can be done because the slices of b0 & b1 (b1 is l0 & l1) do not overlap.
    cb1_yatf[11, ...] = np.expand_dims(cb0_yatf[11, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[12, ...] = np.expand_dims(cb0_yatf[12, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[13, ...] = np.expand_dims(cb0_yatf[13, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[17, ...] = np.expand_dims(cb0_yatf[17, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[18, ...] = np.expand_dims(cb0_yatf[18, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1

    ########################################
    ###create vars with extra period axes  #
    #######################################
    ## p0 axis - the individual days of a generator period i.e. days of each step of the simulation
    ### most other p0 variables are created later but these are required early
    doy_pa1e1b1nwzida0e0b0xygp0 = doy_pa1e1b1nwzida0e0b0xyg[...,na] - step / 2 + index_p0  #calculate the p1 axis from the start day rather than mid-point day

    ## p1 axis - each day of the oestrus cycle for reproduction which can vary with g (e.g. if sheep and cattle in model)
    ### The average day of oestrus is day 0 of the generator period with a spread in the mob on either side.
    ### If beyond the end of oestrus then set to nan (& use nanmean when calculating the average value in later code)
    len_p1_ygp1 = cf_dams[4, ..., na]  # length of the oestrus cycle
    index_p1 = np.arange(np.max(len_p1_ygp1))
    index_ygp1 = np.where(index_p1 < len_p1_ygp1, index_p1, np.nan) - (len_p1_ygp1 - 1) / 2 #index with g axis and nan
    doy_pa1e1b1nwzida0e0b0xygp1 = doy_pa1e1b1nwzida0e0b0xyg[...,na] - step / 2 + index_ygp1  #calculate the p1 axis from the start day rather than mid-point day


    ###################################
    ###group independent / feed info  #
    ###################################
    ##association between b0 and b1
    a_b0_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['ia_b0_b1'],b1_pos)
    ##nfoet expanded
    nfoet_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['a_nfoet_b1'],b1_pos)
    ##nyatf expanded to b1 & b0
    nyatf_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['a_nyatf_b1'],b1_pos)
    ##legume proportion in each period
    legume_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_legume_p6z'], z_pos, move=True, source=0, dest=-1,
                                                 left_pos2=p_pos, right_pos2=z_pos)
    legume_p6a1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(legume_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos)
    ##season type probability
    i_season_propn_z = zfun.f_z_prob()
    season_propn_zida0e0b0xyg = fun.f_expand(i_season_propn_z, z_pos)

    ##wind speed (with animal height adjustment)
    #todo add a distribution to the windspeed (after checking the importance for chill_index)
    # might need to do this with a longer axis length so that it is not the distribution in the week but in the month
    # enter a number of days above a threshold (along with the threshold values maybe 1) and then average values for the windiest days in the month.
    ws_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_ws_p4'],p_pos)
    ### adjust to windspeed at animal height using Thornley & Johnson equation (as used in Ag360)
    ###Reference height is 10m as per BOM, canopy height = 0.3 (from Ag360 which seems too high, but it was used in stats analysis so used here)
    ###Note: Horton etal 2019 used a scalar of 0.58 for open paddocks and down to 0.05 for sheltered paddocks.
    canopy = 0.3
    ref_height = 10
    animal_height = 0.4
    roughness = 0.13 * canopy
    height_adjust = np.log(animal_height / roughness) / np.log(ref_height / roughness)
    ws_p4a1e1b1nwzida0e0b0xyg = ws_p4a1e1b1nwzida0e0b0xyg * height_adjust

    ##expected stocking density
    density_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_density_p6z'], z_pos, move=True, source=0, dest=-1,
                                                  left_pos2=p_pos, right_pos2=z_pos)  # p6 axis converted to p axis later (association section)
    density_p6a1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(density_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos).astype(
        int)
    ##nutrition adjustment for expected stocking density
    density_nwzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_density_n1'][0:n_fs_dams],n_pos) # cut to the correct length based on number of nutrition options (i_len_n structural input)
    density_nwzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_density_n3'][0:n_fs_offs],n_pos) # cut to the correct length based on number of nutrition options (i_len_n structural input)
    ##Mob size. mob_size_dams used for lamb survival, all used in husbandry
    mobsize_p6a1e1b1nwzida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_mobsize_sire_zp6i'], i_pos, swap=True, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg0 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg0,numpy=True,axis=z_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_mobsize_dams_zp6i'], i_pos, swap=True, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg1 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg1,numpy=True,axis=z_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_mobsize_offs_zp6i'], i_pos, swap=True, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg3 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg3,numpy=True,axis=z_pos)
    ##Calculation of rainfall distribution across the week - i_rain_distribution_p4p1 = how much rain falls on each day of the week sorted in order of quantity of rain. SO the most rain falls on the day with the highest rainfall.
    rain_p4a1e1b1nwzida0e0b0xygp0 = fun.f_expand(
        pinp.sheep['i_rain_p4'][...,na] * pinp.sheep['i_rain_distribution_p4p1'] * (7 / 30.4),p_pos - 1,
        right_pos=-1)  # -1 because p is -16 when p1 axis is included
    ##Mean daily temperature
    #todo examine importance of temperature variation on chill_index with view to adding
    temp_ave_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_ave_p4'],p_pos)
    ##Mean daily maximum temperature
    temp_max_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_max_p4'],p_pos)
    ##Mean daily minimum temperature
    temp_min_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_temp_min_p4'],p_pos)
    ##latitude
    lat_deg = pinp.sheep['i_latitude']
    lat_rad = np.radians(pinp.sheep['i_latitude'])

    '''if running the gen for stubble generation then the weather info above gets overwritten with stubble trial info
    '''
    if stubble:
        legume_p6a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['clover_propn_in_sward_stubble']
        density_p6a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['i_sr_s2'][0] #take the harv slice of sr given that it is not important enough to keep the s2 axis
        ws_p4a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['i_ws']
        rain_p4a1e1b1nwzida0e0b0xygp0[...] = pinp.stubble['i_rain']
        temp_ave_p4a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['i_temp_ave']
        temp_max_p4a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['i_temp_max']
        temp_min_p4a1e1b1nwzida0e0b0xyg[...] = pinp.stubble['i_temp_min']


    ########################################
    #adjust input to align with gen periods#
    ########################################
    # 1. Adjust date born (average) = start period  (only yatf date born needs to be adjusted to start of generator period because it is used for fvp)
    # 2. Calc date born1st from slice 0 subtract 8 days
    # 3. Calc wean date
    # 4 adjust wean date to occur at start period
    # 5 calc adjusted wean age

    ##calc and adjust date born average of group - convert from date of first lamb born to average date born of lambs in the first cycle
    ###sire
    date_born_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + (0.5 * cf_sire[4, 0:1,:])	 #times by 0.5 to get the average birthdate for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_P, date_born_ida0e0b0xyg0, 0, 'left') #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_born_ida0e0b0xyg0 = date_start_P[date_born_idx_ida0e0b0xyg0]
    ###dams
    date_born_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + (0.5 * cf_dams[4, 0:1,:])	 #times by 0.5 to get the average birthdate for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg1 = fun.f_next_prev_association(date_start_P,date_born_ida0e0b0xyg1,0,'left') #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_born_ida0e0b0xyg1 = date_start_P[date_born_idx_ida0e0b0xyg1]
    ###yatf - needs to be rounded to start of gen period because this controls the start of a dvp
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + ((index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:])	 #times by 0.5 to get the average birthdate for all lambs because ewes can be conceived anytime within joining cycle. e_index is to account for ewe cycles.
    date_born_idx_oa1e1b1nwzida0e0b0xyg2 =fun.f_next_prev_association(date_start_P, date_born_oa1e1b1nwzida0e0b0xyg2, 0, 'left')   #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_start_P[date_born_idx_oa1e1b1nwzida0e0b0xyg2]
    ###offs
    date_born_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + ((index_e0b0xyg + 0.5) * cf_offs[4, 0:1,:])	 #times by 0.5 to get the average birthdate for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_P, date_born_ida0e0b0xyg3, 0, 'left') #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_born_ida0e0b0xyg3 = date_start_P[date_born_idx_ida0e0b0xyg3]
    ##recalc date_born1st from the adjusted average birthdate
    date_born1st_ida0e0b0xyg0 = date_born_ida0e0b0xyg0 - (0.5 * cf_sire[4, 0:1,:]).astype(int)
    date_born1st_ida0e0b0xyg1 = date_born_ida0e0b0xyg1 - (0.5 * cf_dams[4, 0:1,:]).astype(int)
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born_oa1e1b1nwzida0e0b0xyg2[:,:,0:1,...] - (0.5 * cf_yatf[4, 0:1,:]).astype(int) #take slice 0 of e axis
    date_born1st_ida0e0b0xyg3 = date_born_ida0e0b0xyg3[:,:,:,0:1,...] - (0.5 * cf_offs[4, 0:1,:]).astype(int) #take slice 0 of e axis because date born first is from first estrus cycle

    ##calc wean date (weaning input is counting from the date of the first lamb)
    date_weaned_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + age_wean1st_e0b0xyg0
    date_weaned_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + age_wean1st_e0b0xyg1
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + age_wean1st_a0e0b0xyg3 #use off wean age
    date_weaned_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + age_wean1st_a0e0b0xyg3
    ###adjust weaning to occur at the beginning of generator period and recalc wean age
    ####sire
    date_weaned_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_P, date_weaned_ida0e0b0xyg0, 0, 'left')  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_weaned_ida0e0b0xyg0 = date_start_P[date_weaned_idx_ida0e0b0xyg0]
    age_wean1st_ida0e0b0xyg0 = date_weaned_ida0e0b0xyg0 - date_born1st_ida0e0b0xyg0
    ####dams
    date_weaned_idx_ida0e0b0xyg1=fun.f_next_prev_association(date_start_P, date_weaned_ida0e0b0xyg1, 0, 'left') #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_weaned_ida0e0b0xyg1 = date_start_P[date_weaned_idx_ida0e0b0xyg1]
    age_wean1st_ida0e0b0xyg1 = date_weaned_ida0e0b0xyg1 -date_born1st_ida0e0b0xyg1
    ####yatf - this is the same as offs except without the offs periods (offs potentially have fewer periods)
    date_weaned_idx_oa1e1b1nwzida0e0b0xyg2=fun.f_next_prev_association(date_start_P, date_weaned_oa1e1b1nwzida0e0b0xyg2, 0, 'left')  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_start_P[date_weaned_idx_oa1e1b1nwzida0e0b0xyg2]
    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = date_weaned_oa1e1b1nwzida0e0b0xyg2 - date_born1st_oa1e1b1nwzida0e0b0xyg2
    ####offs
    date_weaned_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_P, date_weaned_ida0e0b0xyg3, 0, 'left')  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    date_weaned_ida0e0b0xyg3 = offs_date_start_P[date_weaned_idx_ida0e0b0xyg3]
    age_wean1st_ida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 - date_born1st_ida0e0b0xyg3

    ##Shearing date - set to be on the last day of a sim period
    ###sire
    idx_sida0e0b0xyg0 = fun.f_next_prev_association(date_end_P, date_shear_sida0e0b0xyg0,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg0 = fun.f_expand(date_end_P[idx_sida0e0b0xyg0], p_pos, right_pos=i_pos)  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes)
    ###dam
    idx_sida0e0b0xyg1 = fun.f_next_prev_association(date_end_P, date_shear_sida0e0b0xyg1,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg1 = fun.f_expand(date_end_P[idx_sida0e0b0xyg1], p_pos, right_pos=i_pos)  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes)
    ###off - shearing can't occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = np.maximum(date_born1st_ida0e0b0xyg3 + age_wean1st_ida0e0b0xyg3, date_shear_sida0e0b0xyg3) #shearing must be after weaning. This makes the d axis active because weaning has an active d.
    idx_sida0e0b0xyg3 = fun.f_next_prev_association(offs_date_end_P, date_shear_sida0e0b0xyg3,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg3 = fun.f_expand(offs_date_end_P[idx_sida0e0b0xyg3], p_pos, right_pos=i_pos) #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)


    ############################
    ## calc for associations   #
    ############################
    ##date joined (when the rams go in)
    date_joined_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 - cp_dams[1,...,0:1,:] #take slice 0 from y-axis because cp1 is not affected by genetic merit
    ##expand feed periods over all the years of the sim so that an association between sim period can be made.
    ##set fp to start at the next generator period following the node (needs to be next so that clustering works). Lp are adjusted so that they get clustered the same as dvps
    feedperiods_p6z = per.f_feed_periods()[:-1] #remove last date because that is the end date of the last period (not required)
    feedperiods_p6z = feedperiods_p6z - 364 * (date_start_p[0] < feedperiods_p6z[0]) #this is to make sure the first sim period date is greater than the first feed period date.
    feedperiods_p6z = (feedperiods_p6z  + (np.arange(np.ceil(sim_years +1)) * 364)[...,na,na]).reshape((-1, len_z)) #expand then ravel to return 1d array of the feed period dates expanded the length of the sim. +1 because feed periods start and finish mid-yr so add one to ensure they go to the end of the sim.
    feedperiods_idx_p6z = np.minimum(len(date_start_P) - 1, np.searchsorted(date_start_P,feedperiods_p6z,'left'))  # maximum idx is the number of generator periods
    feedperiods_p6z = date_start_P[feedperiods_idx_p6z]


    ##################
    # FVP background #
    ##################
    '''
    FVPs have gotten a little bit complex. To ensure that weights, masks, transfers and distributions occur correctly there are some rules that need to be met.
    The DVP rules are documented below in the dvp section.
    
    FVPs before weaning (ie while numbers are all 0) are removed if they occur across all axis. If they don't occur
    across all axis then they are set to the date of weaning. If multiple fvps occur at weaning they get off set by 1 
    period. Type is set to extra so nothing is triggered e.g. if season start is before weaning and it gets moved to weaning
    we don't want to trigger a distribution.
    
    FVPs that occur after the end of the generator are handled the same as above. I.e they are removed if the same across 
    all axis otherwise they are set to the last period of the generator. If multiple fvps occur on the last period they are offset 
    by 1 period (this doesn't need to happen it just does so that the fvp no clash test is passed but fvps clashing on the
    last gen period would actually be fine).
    
    No FVPs can clash. If an fvp clashes with another fvp then it doesn't trigger a fvp_start therefore a 0 day fvp 
    does not trigger liveweight pattern expansion. This would result in different active w for different axes which
    could get confusing and may not work. So to handle this no fvp can clash. If they clash an error will be thrown and 
    the user will need to adjust the inputs.
    
    FVPs are set to occur at the start of a gen period.
    
    '''
    ###################################
    # Feed variation period calcs dams#
    ###################################
    ##fvp/dvp types
    core_dvp_types_f1 = sinp.stock['i_core_dvp_types_f1'] #core dvps/fvps need a certain type because repro dvps are linked to them
    prejoin_vtype1 = core_dvp_types_f1[0]
    condense_vtype1 = prejoin_vtype1 #currently for dams, condensing must occur at prejoining, most of the code is flexible to handle different timing except the lw_distribution section.
    scan_vtype1 = core_dvp_types_f1[1]
    birth_vtype1 = core_dvp_types_f1[2]
    season_vtype1 = max(core_dvp_types_f1) + 1
    other_vtype1 = max(core_dvp_types_f1) + 2

    ##if stubble we don't want it to condense so change condense type to something that is not triggered
    if stubble:
        condense_vtype1 = other_vtype1+1

    ##beginning - first day of generator
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg[0:1]

    ##early pregnancy fvp start - The pre-joining accumulation of the dams from the previous reproduction cycle - this date must correspond to the start date of period
    prejoining_approx_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 - sinp.stock['i_prejoin_offset'] #approx date of prejoining - in the next line of code prejoin date is adjusted to be the start of a sim period in which the approx date falls
    idx = np.searchsorted(date_start_p, prejoining_approx_oa1e1b1nwzida0e0b0xyg1, 'right') - 1 #gets the sim period index for the period that prejoining occurs (e.g. prejoining fvp starts at the beginning of the sim period when prejoining approx occurs), side=right so that if the date is already the start of a period it remains in that period.
    prejoining_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx]
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = prejoining_oa1e1b1nwzida0e0b0xyg1

    ##late pregnancy fvp start - Scanning if carried out, day 90 from joining (ram in) if not scanned.
    late_preg_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :] + pinp.sheep['i_scan_day'][scan_oa1e1b1nwzida0e0b0xyg1]
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_P, late_preg_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when dams in late preg (e.g. late preg fvp starts at the beginning of the sim period when late preg occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = date_start_P[idx_oa1e1b1nwzida0e0b0xyg]

    ## lactation fvp start - average date of lambing (with e axis if scanning/managing e differentially) (already adjusted to start of gen period)
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = date_born_oa1e1b1nwzida0e0b0xyg2.copy()+step #birth dvp needs to start the period after birth because numbers start needs to reflect the birth status (numbers start doesn't update until the period after birth)
    ### birth fvp/dvp must be the same when e axis is clustered (otherwise something goes wrong in the pp/matrix).
    t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.copy()
    t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[...] = fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[:,:,-1:,...] #needs to be the final e slice so that all e slices have lambed when new dvp starts.
    e_fvp_mask = (scan_oa1e1b1nwzida0e0b0xyg1 < 4)  #mask with true when fvp/dvp date should be the same along the e axis
    e_fvp_mask = np.broadcast_to(e_fvp_mask, fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape)
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[e_fvp_mask] = t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[e_fvp_mask]

    ##weaning (already adjusted to start of gen period)
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = date_weaned_oa1e1b1nwzida0e0b0xyg2

    ##user defined fvp - rounded to the nearest sim period
    fvp_other_iu = sinp.structuralsa['i_dams_user_fvp_date_iu']
    fvp_other_iu = fun.f_sa(fvp_other_iu, sen.sav['user_fvp_date_dams_iu'], 5)
    n_user_fvp = fvp_other_iu.shape[-1]
    user_fvp_u = np.zeros(n_user_fvp, dtype=object)
    fvp_other_yiu = fvp_other_iu + np.arange(np.ceil(sim_years))[:,na,na] * 364
    fvp_other_yiu = fun.f_sa(fvp_other_yiu, sen.sav['user_fvp_date_dams_yiu'], 5)
    for u in range(n_user_fvp):
        fvp_other_ya1e1b1nwzida0e0b0xyg = fun.f_expand(fvp_other_yiu[...,u], left_pos=i_pos, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
        idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_P, fvp_other_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (e.g. break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
        fvp_other_start_ya1e1b1nwzida0e0b0xyg = date_start_P[idx_ya1e1b1nwzida0e0b0xyg]  #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
        user_fvp_u[u] = fvp_other_start_ya1e1b1nwzida0e0b0xyg

    ##season nodes - these get masked out if steady state.
    node_fvp_m = np.zeros(len_m, dtype=object)
    for m in range(len_m):
        node_fvp_m[m] = date_node_ya1e1b1nwzidaebxygm[...,m]
    ###store season start date - used to determine period_is_season_start
    seasonstart_ya1e1b1nwzida0e0b0xyg = node_fvp_m[0]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[1:],
                               seasonstart_ya1e1b1nwzida0e0b0xyg.shape[1:]]) #create shape which has the max size, this is used for o array

    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    for u in range(n_user_fvp):
        user_fvp_u[u] = np.broadcast_to(user_fvp_u[u],(user_fvp_u[u].shape[0],)+tuple(shape))
    for m in range(len_m):
        node_fvp_m[m] = np.broadcast_to(node_fvp_m[m],(node_fvp_m[m].shape[0],)+tuple(shape))

    ##create fvp type arrays. these are the same shape as the start arrays and are filled with the number corresponding to the fvp number
    fvp_begin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape, condense_vtype1)
    fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape,prejoin_vtype1)
    fvp_scan_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape, scan_vtype1)
    fvp_birth_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape, birth_vtype1)
    fvp_wean_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape, other_vtype1)
    user_fvp_type_u = np.zeros_like(user_fvp_u)
    for u in range(n_user_fvp):
        user_fvp_type_u[u] = np.full(user_fvp_u[u].shape,other_vtype1)
    node_fvp_type_m = np.zeros_like(node_fvp_m)
    for m in range(len_m):
        ##season start dvp/fvp needs its own type because it needs to be distinguishable so that dvp can get distributed (only when season nodes included).
        if m==0:
            node_fvp_type_m[m] = np.full(node_fvp_m[m].shape, season_vtype1)
        else:
            node_fvp_type_m[m] = np.full(node_fvp_m[m].shape, other_vtype1)

    ##stack & mask which dvps are included - this must be in the order as per the input mask
    fvp_date_all_f1 = np.array([fvp_begin_start_ba1e1b1nwzida0e0b0xyg1, fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,
                                fvp_scan_start_oa1e1b1nwzida0e0b0xyg1, fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,
                                fvp_wean_start_oa1e1b1nwzida0e0b0xyg1], dtype=object)
    fvp_date_all_f1 = np.concatenate([node_fvp_m[0:1], fvp_date_all_f1, user_fvp_u, node_fvp_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp_type_all_f1 = np.array([fvp_begin_type_va1e1b1nwzida0e0b0xyg1, fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1,
                                fvp_scan_type_va1e1b1nwzida0e0b0xyg1, fvp_birth_type_va1e1b1nwzida0e0b0xyg1,
                                fvp_wean_type_va1e1b1nwzida0e0b0xyg1], dtype=object)
    fvp_type_all_f1 = np.concatenate([node_fvp_type_m[0:1], fvp_type_all_f1, user_fvp_type_u, node_fvp_type_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp1_inc = np.concatenate([fvp_mask_dams[0:1], np.array([True]), fvp_mask_dams[1:]]) #True in the middle is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesn't occur each year)
    fvp_date_inc_f1 = fvp_date_all_f1[fvp1_inc]
    fvp_type_inc_f1 = fvp_type_all_f1[fvp1_inc]
    fvp_start_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_date_inc_f1,axis=0)
    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_type_inc_f1,axis=0)

    ##handle pre weaning fvps or post gen
    fvp_start_fa1e1b1nwzida0e0b0xyg1,fvp_type_fa1e1b1nwzida0e0b0xyg1 = \
        sfun.f1_fvpdvp_adj(fvp_start_fa1e1b1nwzida0e0b0xyg1,fvp_type_fa1e1b1nwzida0e0b0xyg1,date_weaned_ida0e0b0xyg1,
                      date_start_p,other_vtype1,condense_vtype1,step)

    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg1, axis=0)
    fvp_start_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)
    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)

    ##error check
    #can't be any clashes
    if not stubble:  # when generating for stubble fvp don't matter
        for f in range(fvp_start_fa1e1b1nwzida0e0b0xyg1.shape[0]): #maybe there is a way to do this without a loop.
            if np.any(fvp_start_fa1e1b1nwzida0e0b0xyg1[f,...] == fvp_start_fa1e1b1nwzida0e0b0xyg1[0:f,...]):
                raise exc.FVPError('''Multiple dams FVP on the same date. Use inputs to change.''')



    # ##if all axis have fvps on the last gen period that fvp can be sliced off
    # duplicate_fvp_mask_f = np.logical_not(fun.f_reduce_skipfew(np.all, fvp_start_fa1e1b1nwzida0e0b0xyg1==date_start_p[-1], preserveAxis=0))
    # fvp_start_fa1e1b1nwzida0e0b0xyg1 = fvp_start_fa1e1b1nwzida0e0b0xyg1[duplicate_fvp_mask_f, ...]
    # fvp_type_fa1e1b1nwzida0e0b0xyg1 = fvp_type_fa1e1b1nwzida0e0b0xyg1[duplicate_fvp_mask_f, ...]


    ##condensing - currently this is fixed to be the same as prejoining for dams (has a full fvp axis but fvp that are not condensing just have the same date as previous condensing)
    condense_bool_fa1e1b1nwzida0e0b0xyg1 = fvp_type_fa1e1b1nwzida0e0b0xyg1!=condense_vtype1
    condensing_date_oa1e1b1nwzida0e0b0xyg1 = fvp_start_fa1e1b1nwzida0e0b0xyg1.copy()
    condensing_date_oa1e1b1nwzida0e0b0xyg1[condense_bool_fa1e1b1nwzida0e0b0xyg1] = 0
    condensing_date_oa1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(condensing_date_oa1e1b1nwzida0e0b0xyg1, axis=0)


    ####################################
    # Feed variation period calcs offs #
    ####################################
    ##Animals which are clustered must have the same fvp/dvps. To handle this inputs (dateborn) with a d axis that effect
    # dvp/fvps are set to be the same within a cluster.

    ##fvp/dvp types
    condense_vtype3 = 0 #for offs, condensing can occur at any dvp. Currently it occurs at shearing.
    other_vtype3 = condense_vtype3 + 1
    season_vtype3 = other_vtype3 + 1

    ##if stubble we don't want it to condense so change condense type
    if stubble:
        condense_vtype3 = other_vtype3 + 1

    ##scale factor for fvps between shearing. This is required because if shearing occurs more frequently than once a yr the fvps that are determined from shearing date must be closer together.
    shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3 = (np.roll(date_shear_sa1e1b1nwzida0e0b0xyg3, -1, axis=0) - date_shear_sa1e1b1nwzida0e0b0xyg3).astype(float) / 364
    shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3[-1] = 1

    ##fvp's between weaning and first shearing - there will be 3 fvp's equally spaced between wean and first shearing (unless shearing occurs within 3 periods from weaning - if weaning and shearing are close the extra fvp are masked out in the stacking process below)
    ###b0
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg3[0:1]
    ###b1
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_P, fvp_b1_start_ba1e1b1nwzida0e0b0xyg3, side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period). use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_P[idx_ba1e1b1nwzida0e0b0xyg]
    ###b2
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + 2 * (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_P, fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period). use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_P[idx_ba1e1b1nwzida0e0b0xyg]

    ##fvp0 - date shearing plus 1 day because shearing is the last day of period
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + np.maximum(1, fvp0_offset_ida0e0b0xyg3) #plus 1 at least 1 because shearing is the last day of the period and the fvp should start after shearing
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_P, fvp_0_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period. side=right so that if the date is already the start of a period it remains in that period.
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_P[idx_sa1e1b1nwzida0e0b0xyg] #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)

    ##fvp1 - date shearing plus offset1 (this is the first day of sim period)
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + (fvp1_offset_ida0e0b0xyg3 * shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3).astype(int)
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_P, fvp_1_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_P[idx_sa1e1b1nwzida0e0b0xyg] #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)

    ##fvp2 - date shearing plus offset2 (this is the first day of sim period)
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + (fvp2_offset_ida0e0b0xyg3 * shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3).astype(int)
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_P, fvp_2_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_P[idx_sa1e1b1nwzida0e0b0xyg] #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)

    ##user defined fvp - rounded to the nearest sim period
    fvp_other_iu = sinp.structuralsa['i_offs_user_fvp_date_iu']
    fvp_other_iu = fun.f_sa(fvp_other_iu, sen.sav['user_fvp_date_offs_iu'], 5)
    n_user_fvp = fvp_other_iu.shape[-1]
    user_fvp_u = np.zeros(n_user_fvp, dtype=object)
    fvp_other_yiu = fvp_other_iu + np.arange(np.ceil(sim_years))[:,na,na] * 364
    fvp_other_yiu = fun.f_sa(fvp_other_yiu, sen.sav['user_fvp_date_offs_yiu'], 5)
    for u in range(n_user_fvp):
        fvp_other_ya1e1b1nwzida0e0b0xyg = fun.f_expand(fvp_other_yiu[u], left_pos=i_pos, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
        idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_P, fvp_other_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (e.g. break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
        fvp_other_start_ya1e1b1nwzida0e0b0xyg = date_start_P[idx_ya1e1b1nwzida0e0b0xyg] #use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
        user_fvp_u[u] = fvp_other_start_ya1e1b1nwzida0e0b0xyg


    ##season nodes - these get masked out if steady state.
    node_fvp_m = np.zeros(len_m, dtype=object)
    for m in range(len_m):
        node_fvp_m[m] = date_node_ya1e1b1nwzidaebxygm[...,m]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[1:], fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[1:], seasonstart_ya1e1b1nwzida0e0b0xyg.shape[1:]]) #create shape which has the max size, this is used for o array
    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_0_start_sa1e1b1nwzida0e0b0xyg3,(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_1_start_sa1e1b1nwzida0e0b0xyg3,(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_2_start_sa1e1b1nwzida0e0b0xyg3,(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    for u in range(n_user_fvp):
        user_fvp_u[u] = np.broadcast_to(user_fvp_u[u],(user_fvp_u[u].shape[0],)+tuple(shape))
    for m in range(len_m):
        node_fvp_m[m] = np.broadcast_to(node_fvp_m[m],(node_fvp_m[m].shape[0],)+tuple(shape))

    fvp_b0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape, condense_vtype3)
    fvp_b1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_b2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape, condense_vtype3)
    fvp_1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    user_fvp_type_u = np.zeros_like(user_fvp_u)
    for u in range(n_user_fvp):
        user_fvp_type_u[u] = np.full(user_fvp_u[u].shape,other_vtype3)
    for m in range(len_m):
        ##season start dvp/fvp needs its own type because it needs to be distinguishable so that dvp can get distributed (only when season nodes included).
        if m==0:
            node_fvp_type_m[m] = np.full(node_fvp_m[m].shape, season_vtype3)
        else:
            node_fvp_type_m[m] = np.full(node_fvp_m[m].shape, other_vtype3)

    ##stack & mask which dvps are included - this must be in the order as per the input mask
    fvp_date_all_f3 = np.array([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,
                                fvp_b2_start_ba1e1b1nwzida0e0b0xyg3, fvp_0_start_sa1e1b1nwzida0e0b0xyg3,
                                fvp_1_start_sa1e1b1nwzida0e0b0xyg3, fvp_2_start_sa1e1b1nwzida0e0b0xyg3], dtype=object)
    fvp_date_all_f3 = np.concatenate([node_fvp_m[0:1], fvp_date_all_f3, user_fvp_u, node_fvp_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp_type_all_f3 = np.array([fvp_b0_type_va1e1b1nwzida0e0b0xyg3, fvp_b1_type_va1e1b1nwzida0e0b0xyg3,
                                fvp_b2_type_va1e1b1nwzida0e0b0xyg3, fvp_0_type_va1e1b1nwzida0e0b0xyg3,
                                fvp_1_type_va1e1b1nwzida0e0b0xyg3, fvp_2_type_va1e1b1nwzida0e0b0xyg3], dtype=object)
    fvp_type_all_f3 = np.concatenate([node_fvp_type_m[0:1], fvp_type_all_f3, user_fvp_type_u, node_fvp_type_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    ###if shearing is less than 3 sim periods after weaning then set the break fvp dates to the first date of the sim (so they aren't used)
    mask_initial_fvp = np.all((date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3) > ((step+1)*3)) #true if not enough gap between weaning and shearing for extra dvps.
    ###create the fvp mask. fvps are masked out depending on what the user has specified (the extra fvps at the start are removed if weaning is within 3weeks of shearing).
    fvp3_inc = np.concatenate([fvp_mask_offs[0:1], np.array([True, True and mask_initial_fvp, True and mask_initial_fvp]), fvp_mask_offs[1:]]) #Trues in middle are to count for the extra fvp at the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesn't occur each year)
    fvp_date_inc_f3 = fvp_date_all_f3[fvp3_inc]
    fvp_type_inc_f3 = fvp_type_all_f3[fvp3_inc]
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(fvp_date_inc_f3,axis=0)
    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(fvp_type_inc_f3,axis=0)

    ##handle pre weaning fvps or post gen
    fvp_start_fa1e1b1nwzida0e0b0xyg3,fvp_type_fa1e1b1nwzida0e0b0xyg3 = \
        sfun.f1_fvpdvp_adj(fvp_start_fa1e1b1nwzida0e0b0xyg3,fvp_type_fa1e1b1nwzida0e0b0xyg3,date_weaned_ida0e0b0xyg3
                      ,offs_date_start_p,other_vtype3,condense_vtype3,step)

    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg3, axis=0)
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)
    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)

    ##error check - can't be any clashes
    if not stubble: # when generating for stubble fvp don't matter
        for f in range(fvp_start_fa1e1b1nwzida0e0b0xyg3.shape[0]):  # maybe there is a way to do this without a loop.
            if np.any(fvp_start_fa1e1b1nwzida0e0b0xyg3[f,...] == fvp_start_fa1e1b1nwzida0e0b0xyg3[0:f,...]):
                raise exc.FVPError('''Multiple offs FVP on the same date. Use inputs to change.''')

    ##condensing dates (has a full fvp axis but fvp that are not condensing just have the same date as previous condensing)
    condense_bool_fa1e1b1nwzida0e0b0xyg3 = fvp_type_fa1e1b1nwzida0e0b0xyg3!=condense_vtype3
    condensing_date_oa1e1b1nwzida0e0b0xyg3 = fvp_start_fa1e1b1nwzida0e0b0xyg3.copy()
    condensing_date_oa1e1b1nwzida0e0b0xyg3[condense_bool_fa1e1b1nwzida0e0b0xyg3] = 0
    condensing_date_oa1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(condensing_date_oa1e1b1nwzida0e0b0xyg3, axis=0)


    ##########
    # DVP    #
    ##########
    '''
    Unlike FVPs dvps can have clashes but there are some conditions.
    1. if a clash occurs that contains season start or condense dvp then they must be first. This ensures the distribution still works
       this is why season start is at the start before the other fvps. 
    2. can remove dvp if it clashes for all axis and type ==other. Because repro dvps are used for clustering and season and condense are used for distributing.
    3. prejoin must be in the same v slice across g axis because g activities can transfer to other g activities
    4. season start must be in the same v slice across all z axis because the weighted average needs all season starts to be in same v.
    5. condense and season start cant clash (unless they have the same vtype).
    
    Nothing guarantees that the order of other dvps are the same e.g. for z1 weaning dvp could be before summer node dvp
    but for z2 weaning dvp could be after summer node dvp. However, it is probably not that likely that
    dvps will be in a different order because most dvp dates are the same or similar across axes.
    '''

    if not stubble:  # when generating for stubble dvps aren't required

        ##dams
        ###build dvps from fvps
        mask_node_is_dvp = np.full(len_m, True) * (pinp.general['i_inc_node_periods'] or np.logical_not(bool_steady_state)) #node fvp/dvp are not included if it is steadystate.
        dvp_mask_f1 = np.concatenate([mask_node_is_dvp[0:1], sinp.stock['i_fixed_dvp_mask_f1'], sinp.structuralsa['i_dvp_mask_f1'], mask_node_is_dvp[1:]]) #season start is first
        dvp1_inc = np.concatenate([dvp_mask_f1[0:1], np.array([True]), dvp_mask_f1[1:]]) #True at start is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesn't occur each year)
        dvp_date_inc_v1 = fvp_date_all_f1[dvp1_inc]
        dvp_type_inc_v1 = fvp_type_all_f1[dvp1_inc]
        dvp_start_va1e1b1nwzida0e0b0xyg1 = np.concatenate(dvp_date_inc_v1,axis=0)
        dvp_type_va1e1b1nwzida0e0b0xyg1 = np.concatenate(dvp_type_inc_v1,axis=0) #note dvp type doesn't have to start at 0 or be consecutive.

        ##handle pre weaning fvps or post gen
        dvp_start_va1e1b1nwzida0e0b0xyg1,dvp_type_va1e1b1nwzida0e0b0xyg1 = \
            sfun.f1_fvpdvp_adj(dvp_start_va1e1b1nwzida0e0b0xyg1,dvp_type_va1e1b1nwzida0e0b0xyg1,date_weaned_ida0e0b0xyg1,
                          date_start_p,other_vtype1,condense_vtype1,step)

        ##check season start is same v slice across z axis
        if np.any(np.logical_and(np.any(dvp_type_va1e1b1nwzida0e0b0xyg1==season_vtype1, axis=z_pos), np.logical_not(np.all(dvp_type_va1e1b1nwzida0e0b0xyg1==season_vtype1, axis=z_pos)))):
            raise exc.FVPError('''Dams - Season start is not in the same v slice across all z.''')

        ##check prejoining is same v slice across g and e
        if np.any(np.logical_and(np.any(dvp_type_va1e1b1nwzida0e0b0xyg1==prejoin_vtype1, axis=(e1_pos,-1)),
                                 np.logical_not(np.all(dvp_type_va1e1b1nwzida0e0b0xyg1==prejoin_vtype1, axis=(e1_pos,-1))))):
            raise exc.FVPError('''Dams - Prejoining start is not in the same v slice across all g or e.''')

        ##remove clashes (note can only remove type==other)
        duplicate_mask_v = []
        for v in range(dvp_start_va1e1b1nwzida0e0b0xyg1.shape[0]): #maybe there is a way to do this without a loop.
            can_remove_type  = np.all(dvp_type_va1e1b1nwzida0e0b0xyg1[v,...]==other_vtype1) #can only remove dvp if it is type=other.
            can_remove_date = np.all(np.any(dvp_start_va1e1b1nwzida0e0b0xyg1[v,...] == dvp_start_va1e1b1nwzida0e0b0xyg1[0:v,...], axis=0, keepdims=True))
            duplicate_mask_v.append(np.logical_not(np.logical_and(can_remove_type, can_remove_date)))
            ###check that condense and season start don't clash - note this doesn't throw an error if condense_type==season_type (this is correct).
            clash_type = dvp_type_va1e1b1nwzida0e0b0xyg1[0:v][dvp_start_va1e1b1nwzida0e0b0xyg1[v,...] == dvp_start_va1e1b1nwzida0e0b0xyg1[0:v,...]]
            current_type = dvp_type_va1e1b1nwzida0e0b0xyg1[v]
            season_condense_clash = np.logical_and(np.any(np.logical_or(clash_type==season_vtype1, clash_type==condense_vtype1)),
                                                   np.any(np.logical_or(clash_type == season_vtype1, clash_type == condense_vtype1)))
            if season_condense_clash:
                raise exc.FVPError('''Dams - Condense and season start dvps cant clash otherwise error with distribution.''')
        dvp_start_va1e1b1nwzida0e0b0xyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1[duplicate_mask_v]
        dvp_type_va1e1b1nwzida0e0b0xyg1 = dvp_type_va1e1b1nwzida0e0b0xyg1[duplicate_mask_v]

        ###sort into order
        ind=np.argsort(dvp_start_va1e1b1nwzida0e0b0xyg1, axis=0)
        dvp_start_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg1, ind, axis=0)
        dvp_type_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dvp_type_va1e1b1nwzida0e0b0xyg1, ind, axis=0)


        ##offs
        ###mask which dvps are included
        dvp_mask_f3 = np.concatenate([mask_node_is_dvp[0:1], sinp.structuralsa['i_dvp_mask_f3'], mask_node_is_dvp[1:]]) #season start is first
        dvp3_inc = np.concatenate([dvp_mask_f3[0:1], np.array([True, False, False]), dvp_mask_f3[1:]]) #True in middle is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesn't occur each year)
        ###build dvps from fvps
        dvp_date_inc_v3 = fvp_date_all_f3[dvp3_inc]
        dvp_type_inc_v3 = fvp_type_all_f3[dvp3_inc]
        dvp_type_va1e1b1nwzida0e0b0xyg3 = np.concatenate(dvp_type_inc_v3,axis=0)
        dvp_start_va1e1b1nwzida0e0b0xyg3 = np.concatenate(dvp_date_inc_v3,axis=0)

        ##handle pre weaning dvps or post gen
        dvp_start_va1e1b1nwzida0e0b0xyg3,dvp_type_va1e1b1nwzida0e0b0xyg3 = \
            sfun.f1_fvpdvp_adj(dvp_start_va1e1b1nwzida0e0b0xyg3,dvp_type_va1e1b1nwzida0e0b0xyg3,date_weaned_ida0e0b0xyg3
                          ,offs_date_start_p,other_vtype3,condense_vtype3,step)

        ##check season start is same v slice across z axis
        if np.any(np.logical_and(np.any(dvp_type_va1e1b1nwzida0e0b0xyg3==season_vtype1, axis=z_pos), np.logical_not(np.all(dvp_type_va1e1b1nwzida0e0b0xyg3==season_vtype1, axis=z_pos)))):
            raise exc.FVPError('''Offs - Season start is not in the same v slice across all z.''')

        ##check prejoining is same v slice across g and e
        if np.any(np.logical_and(np.any(dvp_type_va1e1b1nwzida0e0b0xyg3==prejoin_vtype1, axis=(e1_pos,-1)),
                                 np.logical_not(np.all(dvp_type_va1e1b1nwzida0e0b0xyg3==prejoin_vtype1, axis=(e1_pos,-1))))):
            raise exc.FVPError('''Offs - Prejoining start is not in the same v slice across all g or e.''')

        ##remove clashes (note can only remove type==other)
        duplicate_mask_v = []
        for v in range(dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0]): #maybe there is a way to do this without a loop.
            can_remove_type  = np.all(dvp_type_va1e1b1nwzida0e0b0xyg3[v,...]==other_vtype3) #can only remove dvp if it is type=other.
            can_remove_date = np.all(np.any(dvp_start_va1e1b1nwzida0e0b0xyg3[v,...] == dvp_start_va1e1b1nwzida0e0b0xyg3[0:v,...], axis=0, keepdims=True))
            duplicate_mask_v.append(np.logical_not(np.logical_and(can_remove_type, can_remove_date)))
            ###check that condense and season start don't clash - note this doesn't throw an error if condense_type==season_type (this is correct).
            clash_type = dvp_type_va1e1b1nwzida0e0b0xyg3[0:v][dvp_start_va1e1b1nwzida0e0b0xyg3[v,...] == dvp_start_va1e1b1nwzida0e0b0xyg3[0:v,...]]
            current_type = dvp_type_va1e1b1nwzida0e0b0xyg3[v]
            season_condense_clash = np.logical_and(np.any(np.logical_or(clash_type==season_vtype3, clash_type==condense_vtype3)),
                                                   np.any(np.logical_or(clash_type == season_vtype3, clash_type == condense_vtype3)))
            if season_condense_clash:
                raise exc.FVPError('''Offs - Condense and season start dvps cant clash otherwise error with distribution.''')
        dvp_start_va1e1b1nwzida0e0b0xyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3[duplicate_mask_v]
        dvp_type_va1e1b1nwzida0e0b0xyg3 = dvp_type_va1e1b1nwzida0e0b0xyg3[duplicate_mask_v]

        ###sort into order
        ind=np.argsort(dvp_start_va1e1b1nwzida0e0b0xyg3, axis=0)
        dvp_start_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg3, ind, axis=0)
        dvp_type_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(dvp_type_va1e1b1nwzida0e0b0xyg3, ind, axis=0)


    ############################
    ### associations           #
    ############################
    ### Aim is to create an association array that points from period to opportunity
    ## The date_p array will be the start date for each period.
    ## A pointer (association) is required that points from the period to the previous or next lambing opportunity
    ## The Lambing opportunity is defined by the joining_date array (which has an 'o' axis)
    ## The previous lambing opportunity is the latest joining date that is less than the date at the end of the period (ie previous/current opportunity)
    ## # ('end of the period' so that if joining occurs during the period it is the previous
    ## The next lambing opportunity is the earliest joining date that is greater than or equal to the end date at the end of the period
    ##prev is just next - 1

    ##joining opportunity association
    ##a_nextprejoining_o returns the final prejoining in the sim if the next prejoining is beyond the end of the sim.
    ## It doesn't cause errors for the current uses, however, important to check any new uses
    a_nextprejoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, prejoining_oa1e1b1nwzida0e0b0xyg1, date_start_p, 0,'left')
    a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, prejoining_oa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, date_joined_oa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2 = np.apply_along_axis(fun.f_next_prev_association, 0, date_born_oa1e1b1nwzida0e0b0xyg2, date_end_p, 1,'right')
    ##dam age association, note this is the same as birth opp (just using a new variable name to avoid confusion in the rest of the code)
    a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2 = a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2
    ##start of season
    a_seasonstart_pa1e1b1nwzida0e0b0xyg = np.apply_along_axis(fun.f_next_prev_association, 0, seasonstart_ya1e1b1nwzida0e0b0xyg, date_end_p, 1,'right')
    ##condensing
    a_condensing_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, condensing_date_oa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_condensing_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, condensing_date_oa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')

    ##Feed period for each generator period
    a_p6_pz = np.apply_along_axis(fun.f_next_prev_association, 0, feedperiods_p6z, date_end_p, 1,'right') % len_p6 #% 10 required to convert association back to only the number of feed periods
    a_p6_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_p6_pz,z_pos,left_pos2=p_pos,right_pos2=z_pos)

    ##shearing opp (previous/current)
    a_prev_s_pa1e1b1nwzida0e0b0xyg0 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg0, date_end_p, 1,'right')
    a_prev_s_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_prev_s_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')
    ##shearing opp (next/current) - points at the next shearing period unless shearing is the current period in which case it points at the current period
    ##a_next_s returns the final shearing occurrence in the sim if the next shearing is beyond the end of the sim.
    ## this would cause an error for the use of a_next_s in the calculation of the lo_bound_offs and is corrected in the bound section
    a_next_s_pa1e1b1nwzida0e0b0xyg0 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg0, date_start_p, 0,'left')
    a_next_s_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg1, date_start_p, 0,'left')
    a_next_s_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg3, offs_date_start_p, 0,'left')
    ##p7 to p association - used for equation systems
    a_g0_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g0_p7'], date_end_p, 1,'right')
    a_g1_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g1_p7'], date_end_p, 1,'right')
    a_g2_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g2_p7'], date_end_p, 1,'right')
    a_g3_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g3_p7'], date_end_p, 1,'right')
    ##month of each period (0 - 11 not 1 -12 because this is association array)
    a_p4_p = ((date_start_p % 364)/(364/12)).astype(int)
    a_p4dp_pg = fun.f_expand(((date_start_p % 364)/(364/12)) - a_p4_p, p_pos) #decimal point version (basically this is the proportion of the way through the month)- used to smooth out the step function
    ##feed variation period
    a_fvp_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_start_fa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_fvp_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_start_fa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')


    ############################
    ### apply associations     #
    ############################
    '''
    The association applied determines when the increment to the next opportunity will occur:
        e.g. if you use a_prev_joining the date in the p slice will increment at joining each time.
    
    '''
    ##shearing
    date_shear_pa1e1b1nwzida0e0b0xyg0 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg0, a_prev_s_pa1e1b1nwzida0e0b0xyg0,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    date_shear_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg1, a_prev_s_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    date_shear_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg3, a_prev_s_pa1e1b1nwzida0e0b0xyg3,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array

    ##management for weaning, gbal and scan options - adjusted further down to represent time of the repro cycle and management
    wean_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(wean_oa1e1b1nwzida0e0b0xyg1, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    gbal_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(gbal_oa1e1b1nwzida0e0b0xyg1, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    scan_option_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(scan_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array

    ##adjust the chill to represent differential paddock allocation if scanned for multiples or greater
    chill_adj_pa1e1b1nwzida0e0b0xyg1 = chill_adj_b1nwzida0e0b0xyg1 * (scan_option_pa1e1b1nwzida0e0b0xyg1 >= 2)

    ##drys management, actual value for the Bounds and an estimate for the generator (don't use bound to control generator otherwise introduce randomness)
    dry_retained_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dry_retained_oa1e1b1nwzida0e0b0xyg1
                                                             , a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(est_drys_retained_scan_oa1e1b1nwzida0e0b0xyg1
                                                                      , a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
    est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(est_drys_retained_birth_oa1e1b1nwzida0e0b0xyg1
                                                                       , a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)

    ##date, age, timing
    date_born1st_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)
    date_born_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born_oa1e1b1nwzida0e0b0xyg2, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)
    date_born1st2_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
    date_born2_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born_oa1e1b1nwzida0e0b0xyg2, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
    ##date_prejoin_next returns the date of the final prejoining in the sim if the next prejoining is beyond the end of the sim.
    ## this will not cause an error for both of the uses of date_prejoin_next (date_between & calculation of ltw_adj)
    date_prejoin_next_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prejoining_oa1e1b1nwzida0e0b0xyg1, a_nextprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
    date_prejoin_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prejoining_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
    date_joined_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(date_joined_oa1e1b1nwzida0e0b0xyg1, a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1,0)
    date_joined2_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(date_joined_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining

    ##condensing
    date_condensing_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(condensing_date_oa1e1b1nwzida0e0b0xyg1, a_condensing_pa1e1b1nwzida0e0b0xyg1,0)
    date_condensing_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(condensing_date_oa1e1b1nwzida0e0b0xyg3, a_condensing_pa1e1b1nwzida0e0b0xyg3,0)

    ###create age & date weaned yatf
    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = np.swapaxes(age_wean1st_oa1e1b1nwzida0e0b0xyg2, a0_pos, a1_pos) #swap a0 and a1 because yatf have to be same shape as dams
    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = np.swapaxes(age_wean1st_oa1e1b1nwzida0e0b0xyg2, e0_pos, e1_pos) #swap e0 and e1 because yatf have to be same shape as dams
    age_wean1st_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(age_wean1st_oa1e1b1nwzida0e0b0xyg2, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)#use off wean age - may vary by d axis therefore have to convert to a p array
    age_wean1st2_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(age_wean1st_oa1e1b1nwzida0e0b0xyg2, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
    date_weaned2_pa1e1b1nwzida0e0b0xyg2 = date_born1st2_pa1e1b1nwzida0e0b0xyg2 + age_wean1st2_pa1e1b1nwzida0e0b0xyg2 #this needs to increment at prejoining for period between weaning and prejoining, so that it is false after prejoining and before weaning.

    ##sim params - turn d to p axis based on pre-joining (change d slice at birth)
    t_ce_dams = np.expand_dims(ce_dams, axis = tuple(range(p_pos,d_pos)))
    ce_pdams = np.take_along_axis(t_ce_dams,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1[na,...],d_pos)
    t_ce_yatf = np.expand_dims(ce_yatf, axis = tuple(range(p_pos,d_pos)))
    ce_pyatf = np.take_along_axis(t_ce_yatf,a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2[na,...],d_pos)

    ##feed period
    legume_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(legume_p6a1e1b1nwzida0e0b0xyg, a_p6_pa1e1b1nwzida0e0b0xyg, 0)

    ##expected stocking density
    density_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(density_p6a1e1b1nwzida0e0b0xyg, a_p6_pa1e1b1nwzida0e0b0xyg, 0)

    ##mob size
    mobsize_pa1e1b1nwzida0e0b0xyg0 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg0, a_p6_pa1e1b1nwzida0e0b0xyg,0)
    mobsize_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg1,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    mobsize_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg3, a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], 0)

    ##select which equation is used for the sheep sim functions for each period
    eqn_used_g0_q1p = uinp.sheep['i_eqn_used_g0_q1p7'][:, a_g0_p7_p]
    eqn_used_g1_q1p = uinp.sheep['i_eqn_used_g1_q1p7'][:, a_g1_p7_p]
    eqn_used_g2_q1p = uinp.sheep['i_eqn_used_g2_q1p7'][:, a_g2_p7_p]
    eqn_used_g3_q1p = uinp.sheep['i_eqn_used_g3_q1p7'][:, a_g3_p7_p]

    ##mating
    sire_propn_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(sire_propn_oa1e1b1nwzida0e0b0xyg1,a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.take_along uses the number in the second array as the index for the first array. and returns a same shaped array
    sire_include_idx = np.arange(len(mask_sire_inc_g0))[mask_sire_inc_g0]
    sire_propn_pa1e1b1nwzida0e0b0xyg1g0=sire_propn_pa1e1b1nwzida0e0b0xyg1[..., na] * (a_g0_g1[:,na] == sire_include_idx) #add g0 axis

    ##weather
    ws_pa1e1b1nwzida0e0b0xyg = ws_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    rain_pa1e1b1nwzida0e0b0xygp0 = rain_p4a1e1b1nwzida0e0b0xygp0[a_p4_p]
    temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_p4a1e1b1nwzida0e0b0xyg[a_p4_p]


    ws_pa1e1b1nwzida0e0b0xyg = ws_p4a1e1b1nwzida0e0b0xyg[a_p4_p] * (1-a_p4dp_pg) \
                               + ws_p4a1e1b1nwzida0e0b0xyg[(a_p4_p + 1) % 12] * a_p4dp_pg
    rain_pa1e1b1nwzida0e0b0xygp0 = rain_p4a1e1b1nwzida0e0b0xygp0[a_p4_p] * (1-a_p4dp_pg[...,na]) \
                               + rain_p4a1e1b1nwzida0e0b0xygp0[(a_p4_p + 1) % 12] * a_p4dp_pg[...,na]
    temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_p4a1e1b1nwzida0e0b0xyg[a_p4_p] * (1-a_p4dp_pg) \
                               + temp_ave_p4a1e1b1nwzida0e0b0xyg[(a_p4_p + 1) % 12] * a_p4dp_pg
    temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_p4a1e1b1nwzida0e0b0xyg[a_p4_p] * (1-a_p4dp_pg) \
                               + temp_max_p4a1e1b1nwzida0e0b0xyg[(a_p4_p + 1) % 12] * a_p4dp_pg
    temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_p4a1e1b1nwzida0e0b0xyg[a_p4_p] * (1-a_p4dp_pg) \
                               + temp_min_p4a1e1b1nwzida0e0b0xyg[(a_p4_p + 1) % 12] * a_p4dp_pg

    ##feed variation
    # fvp_type_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)
    # fvp_type_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg3,a_fvp_pa1e1b1nwzida0e0b0xyg3,0)
    fvp_date_start_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)
    fvp_date_start_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg3,a_fvp_pa1e1b1nwzida0e0b0xyg3,0)

    ##propn of dams mated, actual value for the Bounds and an estimate for the generator (don't use bound to control generator otherwise introduce randomness)
    prop_dams_mated_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_oa1e1b1nwzida0e0b0xyg1,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
    est_prop_dams_mated_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(est_prop_dams_mated_oa1e1b1nwzida0e0b0xyg1,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining

    ##break of season
    date_prev_seasonstart_pa1e1b1nwzida0e0b0xyg=np.take_along_axis(seasonstart_ya1e1b1nwzida0e0b0xyg,a_seasonstart_pa1e1b1nwzida0e0b0xyg,0)

    ##create association between n and w for each generator period i.e. what nutrition level is being offered to this LW profile in this period
    ###sire ^not required because sires only have 1 fvp and 1 n slice
    # a_n_pa1e1b1nwzida0e0b0xyg0 = (np.trunc(index_wzida0e0b0xyg0 / (n_fs_sire ** ((n_fvp_periods_g0-1) - fvp_type_pa1e1b1nwzida0e0b0xyg0))) % n_fs_sire).astype(int) #needs to be int so it can be an indice
    ###dams
    ####for dams period_is_condense is the same as period_is_prejoin atm, but it is designed so it can be different (the lw_distribution will just need to be updated)
    period_is_condense_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_condensing_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_startfvp_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', fvp_date_start_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 = np.cumsum(period_is_startfvp_pa1e1b1nwzida0e0b0xyg1, axis=0)
    n_prior_fvps_pa1e1b1nwzida0e0b0xyg1 = (n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 -
                                           np.maximum.accumulate(n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 * np.logical_or(period_is_condense_pa1e1b1nwzida0e0b0xyg1
                                                                , p_index_pa1e1b1nwzida0e0b0xyg==0), axis=0)) #there is no fvps prior to p0 hence index==0
    a_n_pa1e1b1nwzida0e0b0xyg1 = (np.trunc(index_wzida0e0b0xyg1 / (n_fs_dams ** ((n_fvps_percondense_dams-1) - n_prior_fvps_pa1e1b1nwzida0e0b0xyg1))) % n_fs_dams).astype(int) #needs to be int so it can be an indice
    ###offs
    period_is_condense_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_condensing_pa1e1b1nwzida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3) #g2 date born is the equivalent of date lambed g1
    period_is_startfvp_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', fvp_date_start_pa1e1b1nwzida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 = np.cumsum(period_is_startfvp_pa1e1b1nwzida0e0b0xyg3, axis=0)
    n_prior_fvps_pa1e1b1nwzida0e0b0xyg3 = (n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 -
                                          np.maximum.accumulate(n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 * np.logical_or(period_is_condense_pa1e1b1nwzida0e0b0xyg3
                                                                , p_index_pa1e1b1nwzida0e0b0xyg3==0), axis=0))
    a_n_pa1e1b1nwzida0e0b0xyg3 = (np.trunc(index_wzida0e0b0xyg3 / (n_fs_offs ** ((n_fvps_percondense_offs-1) - n_prior_fvps_pa1e1b1nwzida0e0b0xyg3))) % n_fs_offs).astype(int) #needs to be int so it can be an indice


    #########################################################
    #adjust sensitivities used in intermediate calculations #
    #########################################################
    ##mort prog
    saa_mortalityx_oa1e1b1nwzida0e0b0xyg = fun.f_expand(sen.saa['mortalityx_ol0g1'][:,sinp.stock['a_nfoet_b1'],:]
                                                        , b1_pos, right_pos=g_pos, left_pos2=p_pos, right_pos2=b1_pos
                                                        , condition=mask_dams_inc_g1, axis=g_pos)#add axes between g & b1, and b1 & p
    saa_mortalityx_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(saa_mortalityx_oa1e1b1nwzida0e0b0xyg,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ##mort birth
    saa_mortalitye_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(sen.saa['mortalitye_ol0g1'][:,sinp.stock['a_nfoet_b1'],:]
                                                        , b1_pos, right_pos=g_pos, left_pos2=p_pos, right_pos2=b1_pos
                                                        , condition=mask_dams_inc_g1, axis=g_pos)#add axes between g & b1, and b1 & p
    saa_mortalitye_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(saa_mortalitye_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ##mort of progeny at birth
    sap_mortalityp_oa1e1b1nwzida0e0b0xyg2 = fun.f_expand(sen.sap['mortalityp_ol0g2'][:,sinp.stock['a_nfoet_b1'],:]
                                                        , b1_pos, right_pos=g_pos, left_pos2=p_pos, right_pos2=b1_pos
                                                        , condition=mask_dams_inc_g1, axis=g_pos)#add axes between g & b1, and b1 & p
    sap_mortalityp_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(sap_mortalityp_oa1e1b1nwzida0e0b0xyg2,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ## sum saa[rr] and saa[rr_age] so there is only one saa to handle in f_conception_cs & f_conception_ltw
    ## Note: the proportions of the BTRT doesn't include rr_age_og1 because the BTRT calculations can't vary by age of the dam
    rr_age_og1 = sen.saa['rr_age_og1']
    saa_rr_age_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(rr_age_og1, p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                                    axis=g_pos, condition2=mask_o_dams, axis2=p_pos)
    saa_rr_age_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(saa_rr_age_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    #todo work out the best connection for sam[rr_og1] - currently it is just used for SA associated with extra ewes so don't want linked effects other than numbers
    ##There has to be sam['rr'] separate from sam['rr_og1'] because 'rr' is used to scale scan_std_c2 which doesn't have an o axis.
    rr_og1 = sen.sam['rr_og1'] * sen.sam['rr']
    sam_rr_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(rr_og1, p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                                    axis=g_pos, condition2=mask_o_dams, axis2=p_pos)
    sam_rr_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(sam_rr_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ## Alter the standard scanning rate for f_conception_ltw to include saa['rr_age'] (scan_std_yg0 has already been adjusted by saa['rr'] in UniversalInputs.py
    scan_std_pa1e1b1nwzida0e0b0xyg1 = scan_std_yg1 + saa_rr_age_pa1e1b1nwzida0e0b0xyg1
    ## Combine saa['rr'] and saa['rr_age'] for f_conception_cs
    saa_rr_age_pa1e1b1nwzida0e0b0xyg1 = saa_rr_age_pa1e1b1nwzida0e0b0xyg1 + sen.saa['rr']
    ##littesize
    saa_littersize_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(sen.saa['littersize_og1'], p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                                    axis=g_pos, condition2=mask_o_dams, axis2=p_pos)
    saa_littersize_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(saa_littersize_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ##conception
    saa_conception_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(sen.saa['conception_og1'], p_pos, right_pos=g_pos, condition=mask_dams_inc_g1,
                                                    axis=g_pos, condition2=mask_o_dams, axis2=p_pos)
    saa_conception_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(saa_conception_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ##preg_increment
    saa_preg_increment_oa1e1b1nwzida0e0b0xyg = fun.f_expand(sen.saa['preg_increment_ol0g1'][:,sinp.stock['a_nfoet_b1'],:]
                                                        , b1_pos, right_pos=g_pos, left_pos2=p_pos, right_pos2=b1_pos
                                                        , condition=mask_dams_inc_g1, axis=g_pos)#add axes between g & b1, and b1 & p
    saa_preg_increment_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(saa_preg_increment_oa1e1b1nwzida0e0b0xyg,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ##wean_redn
    sam_wean_redn_oa1e1b1nwzida0e0b0xyg2 = fun.f_expand(sen.sam['wean_redn_ol0g2'][:,sinp.stock['a_nfoet_b1'],:]
                                                        , b1_pos, right_pos=g_pos, left_pos2=p_pos, right_pos2=b1_pos
                                                        , condition=mask_dams_inc_g1, axis=g_pos)#add axes between g & b1, and b1 & p
    sam_wean_redn_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(sam_wean_redn_oa1e1b1nwzida0e0b0xyg2,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array


    ###########################
    ##genotype calculations   #
    ###########################
    ##calc proportion of dry, singles, twin and triplets based on the genotype as born.
    ###e.g. BBM dams are based on BBB scanning and BBB survival. BBM offspring are based on BBB scanning and BBM survival
    ###calculated without saa['rr_age']. These calculations do not include a 'p' axis because it is one value for all the initial animals
    dstwtr_l0yg0 = np.moveaxis(sfun.f1_DSTw(scan_std_yg0, cycles=2), -1, 0) #todo add cpg_doy_ltw scalar to alter scan_std by TOL
    dstwtr_l0yg1 = np.moveaxis(sfun.f1_DSTw(scan_std_yg1, cycles=2), -1, 0)
    dstwtr_l0yg3 = np.moveaxis(sfun.f1_DSTw(scan_dams_std_yg3, cycles=2), -1, 0)

    ##Std scanning & survival: propn of progeny in each BTRT b0 category - 11, 22, 33, 21, 32, 31 and lambs surviving birth per ewe joined (standard weaning percentage)
    btrt_propn_b0xyg0, npw_std_xyg0 = sfun.f1_btrt0(dstwtr_l0yg0,pss_std_yg0,pstw_std_yg0,pstr_std_yg0)
    btrt_propn_b0xyg1, npw_std_xyg1 = sfun.f1_btrt0(dstwtr_l0yg1,pss_std_yg1,pstw_std_yg1,pstr_std_yg1)
    btrt_propn_b0xyg3, npw_std_xyg3 = sfun.f1_btrt0(dstwtr_l0yg3,pss_std_yg3,pstw_std_yg3,pstr_std_yg3)

    ##Std scanning & survival: proportion of progeny reared in each BTRT b1 category - NM, 00, 11, 22, 33, 21, 32, 31, 10, 20, 30
    # btrt_propn_b1nwzida0e0b0xyg0 = sfun.f1_btrt1(dstwtr_l0yg0,pss_std_yg0,pstw_std_yg0,pstr_std_yg0)
    btrt_propn_b1nwzida0e0b0xyg1 = sfun.f1_btrt1(dstwtr_l0yg1,pss_std_yg1,pstw_std_yg1,pstr_std_yg1)
    # btrt_propn_b1nwzida0e0b0xyg2 = sfun.f1_btrt1(dstwtr_l0yg3,pss_std_yg2,pstw_std_yg2,pstr_std_yg2)
    # btrt_propn_b1nwzida0e0b0xyg3 = sfun.f1_btrt1(dstwtr_l0yg3,pss_std_yg3,pstw_std_yg3,pstr_std_yg3)

    ##Std scanning & survival: proportion of dams in each LSLN b1 category - NM, 00, 11, 22, 33, 21, 32, 31, 10, 20, 30
    lsln_propn_b1nwzida0e0b0xyg1 = sfun.f1_lsln(dstwtr_l0yg1,pss_std_yg1,pstw_std_yg1,pstr_std_yg1)


    ###calc adjustments sfw
    adja_sfw_d_a0e0b0xyg0 = np.sum(ce_sire[12, ...] * agedam_propn_da0e0b0xyg0, axis = 0)
    adja_sfw_d_a0e0b0xyg1 = np.sum(ce_dams[12, ...] * agedam_propn_da0e0b0xyg1, axis = 0)
    adja_sfw_d_pa1e1b1nwzida0e0b0xyg2 = ce_pyatf[12,...]
    adja_sfw_d_da0e0b0xyg3 = ce_offs[12, ...]
    adja_sfw_b0_xyg0 = np.sum(cb0_sire[12, ...] * btrt_propn_b0xyg0, axis = 0)
    adja_sfw_b0_xyg1 = np.sum(cb0_dams[12, ...] * btrt_propn_b0xyg1, axis = 0)
    adja_sfw_b0_b1nwzida0e0b0xyg2 = cb1_yatf[12, ...]
    adja_sfw_b0_b0xyg3 = cb0_offs[12, ...]
    ###apply adjustments sfw
    sfw_a0e0b0xyg0 = sfw_yg0 + adja_sfw_d_a0e0b0xyg0 + adja_sfw_b0_xyg0
    sfw_a0e0b0xyg1 = sfw_yg1 + adja_sfw_d_a0e0b0xyg1 + adja_sfw_b0_xyg1
    sfw_pa1e1b1nwzida0e0b0xyg2 = sfw_yg2 + adja_sfw_d_pa1e1b1nwzida0e0b0xyg2 + adja_sfw_b0_b1nwzida0e0b0xyg2
    sfw_da0e0b0xyg3 = sfw_yg3 + adja_sfw_d_da0e0b0xyg3 + adja_sfw_b0_b0xyg3
    ###calc adjustments sfd
    adja_sfd_d_a0e0b0xyg0 = np.sum(ce_sire[13, ...] * agedam_propn_da0e0b0xyg0, axis = 0)
    adja_sfd_d_a0e0b0xyg1 = np.sum(ce_dams[13, ...] * agedam_propn_da0e0b0xyg1, axis = 0)
    adja_sfd_d_pa1e1b1nwzida0e0b0xyg2 = ce_pyatf[13, ...]
    adja_sfd_d_da0e0b0xyg3 = ce_offs[13, ...]
    adja_sfd_b0_xyg0 = np.sum(cb0_sire[13, ...] * btrt_propn_b0xyg0, axis = 0)
    adja_sfd_b0_xyg1 = np.sum(cb0_dams[13, ...] * btrt_propn_b0xyg1, axis = 0)
    adja_sfd_b0_b0xyg2 = cb1_yatf[13, ...]
    adja_sfd_b0_b0xyg3 = cb0_offs[13, ...]
    ###apply adjustments sfd
    sfd_a0e0b0xyg0 = sfd_yg0 + adja_sfd_d_a0e0b0xyg0 + adja_sfd_b0_xyg0
    sfd_a0e0b0xyg1 = sfd_yg1 + adja_sfd_d_a0e0b0xyg1 + adja_sfd_b0_xyg1
    sfd_pa1e1b1nwzida0e0b0xyg2 = sfd_yg2 + adja_sfd_d_pa1e1b1nwzida0e0b0xyg2 + adja_sfd_b0_b0xyg2
    sfd_da0e0b0xyg3 = sfd_yg3 + adja_sfd_d_da0e0b0xyg3 + adja_sfd_b0_b0xyg3

    ###gender adjustment for srw
    srw_xyg0 = srw_female_yg0 * cx_sire[11, 0:1, ...]  #11 is the srw parameter, 0:1 is the sire gender slice (retaining the axis).
    srw_xyg1 = srw_female_yg1 * cx_dams[11, 1:2, ...]
    srw_xyg2 = srw_female_yg2 * cx_yatf[11, mask_x,...] #all gender slices
    srw_xyg3 = srw_female_yg3 * cx_offs[11, mask_x,...] #all gender slices

    ###BTRT adjustment for srw
    srw_b0xyg0 = srw_xyg0 * np.sum(cb0_sire[11, ...] * btrt_propn_b0xyg0, axis = 0)
    srw_b0xyg1 = srw_xyg1 * np.sum(cb0_dams[11, ...] * btrt_propn_b0xyg1, axis = 0)
    srw_b1xyg2 = srw_xyg2 * cb1_yatf[11, ...]
    srw_b0xyg3 = srw_xyg3 * cb0_offs[11, ...]

    ##Standard birth weight.
    ### Does not include the gender scalar on SRW because std BW is related to the female SRW of the genotype
    ### Does not include the BTRT scalar of the dam i.e. assuming that the BTRT adjustment of dam SRW doesn't affect her progeny
    w_b_std_b0xyg0 = srw_female_yg0 * np.sum(cb0_sire[15, ...] * btrt_propn_b0xyg0, axis = b0_pos, keepdims=True) * cx_sire[15, 0:1, ...]
    w_b_std_b0xyg1 = srw_female_yg1 * np.sum(cb0_dams[15, ...] * btrt_propn_b0xyg1, axis = b0_pos, keepdims=True) * cx_dams[15, 1:2, ...]
    w_b_std_b0xyg3 = srw_female_yg3 * cb0_offs[15, ...] * cx_offs[15, mask_x,...]
    ##fetal param - normal birthweight young - used as target birthweight during pregnancy if sheep fed well. Therefore, average gender effect.
    w_b_std_y_b1nwzida0e0b0xyg1 = srw_female_yg2 * cb1_yatf[15, ...] #gender not considered until actual birth therefore no cx
    ##wool growth efficiency
    ###wge is sfw divided by srw of a ewe of the given genotype. Scales the growth per unit intake to allow for the expected change in intake due to SRW
    ###Use SRW of the ewe so that males have same efficiency as females and hence grow more wool due to higher intake.
    wge_a0e0b0xyg0 = sfw_a0e0b0xyg0 / srw_female_yg0
    wge_a0e0b0xyg1 = sfw_a0e0b0xyg1 / srw_female_yg1
    wge_pa1e1b1nwzida0e0b0xyg2 = sfw_pa1e1b1nwzida0e0b0xyg2 / srw_female_yg2
    wge_da0e0b0xyg3 = sfw_da0e0b0xyg3 / srw_female_yg3

    ##Legume impact on efficiency
    lgf_eff_pa1e1b1nwzida0e0b0xyg0 = 1 + ck_sire[14,...] * legume_pa1e1b1nwzida0e0b0xyg
    lgf_eff_pa1e1b1nwzida0e0b0xyg1 = 1 + ck_dams[14,...] * legume_pa1e1b1nwzida0e0b0xyg
    lgf_eff_pa1e1b1nwzida0e0b0xyg2 = 1 + ck_yatf[14,...] * legume_pa1e1b1nwzida0e0b0xyg
    lgf_eff_pa1e1b1nwzida0e0b0xyg3 = 1 + ck_offs[14,...] * legume_pa1e1b1nwzida0e0b0xyg

    ##Day length factor on wool
    dlf_wool_pa1e1b1nwzida0e0b0xyg0 = 1 + cw_sire[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
    dlf_wool_pa1e1b1nwzida0e0b0xyg1 = 1 + cw_dams[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
    dlf_wool_pa1e1b1nwzida0e0b0xyg2 = 1 + cw_yatf[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
    dlf_wool_pa1e1b1nwzida0e0b0xyg3 = 1 + cw_offs[6,...] * (dl_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p] - 12)

    ##Efficiency for wool
    kw_yg0 = ck_sire[17,...]
    kw_yg1 = ck_dams[17,...]
    kw_yg2 = ck_yatf[17,...]
    kw_yg3 = ck_offs[17,...]

    ##Efficiency for conceptus
    kc_yg1 = ck_dams[8,...]


    ####################
    #initial conditions#
    ####################
    ##step - used to expand start lw adjustment to all lw patterns
    step_w1 = len_w1/w_start_len1
    step_w3 = len_w3/w_start_len3

    ##convert i_adjp to adjp - add necessary axes for 'a' and 'w'
    adjp_lw_initial_a0e0b0xyg = fun.f_expand(pinp.sheep['i_adjp_lw_initial_a'], a0_pos, condition=pinp.sheep['i_mask_a'], axis=a0_pos)
    adjp_lw_initial_wzida0e0b0xyg0 = fun.f_expand(sinp.structuralsa['i_adjp_lw_initial_w0'], w_pos)
    adjp_lw_initial_wzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_adjp_lw_initial_w1'][np.trunc(index_w1/step_w1).astype(int)], w_pos)
    adjp_lw_initial_wzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_adjp_lw_initial_w3'][np.trunc(index_w3/step_w3).astype(int)], w_pos)
    adjp_cfw_initial_a0e0b0xyg = fun.f_expand(pinp.sheep['i_adjp_cfw_initial_a'], a0_pos, condition=pinp.sheep['i_mask_a'], axis=a0_pos)
    adjp_cfw_initial_wzida0e0b0xyg0 = fun.f_expand(sinp.structuralsa['i_adjp_cfw_initial_w0'], w_pos)
    adjp_cfw_initial_wzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_adjp_cfw_initial_w1'][np.trunc(index_w1/step_w1).astype(int)], w_pos)
    adjp_cfw_initial_wzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_adjp_cfw_initial_w3'][np.trunc(index_w3/step_w3).astype(int)], w_pos)
    adjp_fd_initial_a0e0b0xyg = fun.f_expand(pinp.sheep['i_adjp_fd_initial_a'], a0_pos, condition=pinp.sheep['i_mask_a'], axis=a0_pos)
    adjp_fd_initial_wzida0e0b0xyg0 = fun.f_expand(sinp.structuralsa['i_adjp_fd_initial_w0'], w_pos)
    adjp_fd_initial_wzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_adjp_fd_initial_w1'][np.trunc(index_w1/step_w1).astype(int)], w_pos)
    adjp_fd_initial_wzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_adjp_fd_initial_w3'][np.trunc(index_w3/step_w3).astype(int)], w_pos)
    adjp_fl_initial_a0e0b0xyg = fun.f_expand(pinp.sheep['i_adjp_fl_initial_a'], a0_pos, condition=pinp.sheep['i_mask_a'], axis=a0_pos)
    adjp_fl_initial_wzida0e0b0xyg0 = fun.f_expand(sinp.structuralsa['i_adjp_fl_initial_w0'], w_pos)
    adjp_fl_initial_wzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_adjp_fl_initial_w1'][np.trunc(index_w1/step_w1).astype(int)], w_pos)
    adjp_fl_initial_wzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_adjp_fl_initial_w3'][np.trunc(index_w3/step_w3).astype(int)], w_pos)


    ##convert variable from c2 to g (yatf is not used, only here because it is return from the function) then adjust by initial lw pattern
    lw_initial_yg0, lw_initial_yg1, lw_initial_yatf, lw_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_lw_initial_c2'], uinp.parameters['i_lw_initial_y'], a_c2_c0, i_g3_inc)
    ###the initial lw input is a proportion of srw
    ### Uses srw_female to remove the randomness that would occur with srw_b0 when changing RR.
    lw_initial_wzida0e0b0xyg0 = (lw_initial_yg0 * (1 + adjp_lw_initial_wzida0e0b0xyg0)) * srw_female_yg0
    lw_initial_wzida0e0b0xyg1 = (lw_initial_yg1 * (1 + adjp_lw_initial_wzida0e0b0xyg1)) * srw_female_yg1
    lw_initial_wzida0e0b0xyg3 = (lw_initial_yg3 * (1 + adjp_lw_initial_wzida0e0b0xyg3)) * srw_female_yg3
    cfw_initial_yg0, cfw_initial_yg1, cfw_initial_yatf, cfw_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_cfw_initial_c2'], uinp.parameters['i_cfw_initial_y'], a_c2_c0, i_g3_inc)
    ###the initial cfw input is a proportion of sfw
    cfw_initial_wzida0e0b0xyg0 = (cfw_initial_yg0 * (1 + adjp_cfw_initial_wzida0e0b0xyg0)) * sfw_yg0
    cfw_initial_wzida0e0b0xyg1 = (cfw_initial_yg1 * (1 + adjp_cfw_initial_wzida0e0b0xyg1)) * sfw_yg1
    cfw_initial_wzida0e0b0xyg3 = (cfw_initial_yg3 * (1 + adjp_cfw_initial_wzida0e0b0xyg3)) * sfw_yg3
    fd_initial_yg0, fd_initial_yg1, fd_initial_yatf, fd_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_fd_initial_c2'], uinp.parameters['i_fd_initial_y'], a_c2_c0, i_g3_inc)
    fd_initial_wzida0e0b0xyg0 = fd_initial_yg0 * (1 + adjp_fd_initial_wzida0e0b0xyg0)
    fd_initial_wzida0e0b0xyg1 = fd_initial_yg1 * (1 + adjp_fd_initial_wzida0e0b0xyg1)
    fd_initial_wzida0e0b0xyg3 = fd_initial_yg3 * (1 + adjp_fd_initial_wzida0e0b0xyg3)
    fl_initial_yg0, fl_initial_yg1, fl_initial_yatf, fl_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_initial_c2'], uinp.parameters['i_fl_initial_y'], a_c2_c0, i_g3_inc)
    fl_initial_wzida0e0b0xyg0 = fl_initial_yg0 * (1 + adjp_fl_initial_wzida0e0b0xyg0)
    fl_initial_wzida0e0b0xyg1 = fl_initial_yg1 * (1 + adjp_fl_initial_wzida0e0b0xyg1)
    fl_initial_wzida0e0b0xyg3 = fl_initial_yg3 * (1 + adjp_fl_initial_wzida0e0b0xyg3)

    ##adjustment for weaning age
    adjp_lw_initial_a_a0e0b0xyg0 = adjp_lw_initial_a0e0b0xyg[0:1,...]
    adjp_lw_initial_a_a0e0b0xyg1 = adjp_lw_initial_a0e0b0xyg[0:1,...]
    adjp_lw_initial_a_a0e0b0xyg3 = adjp_lw_initial_a0e0b0xyg
    adjp_cfw_initial_a_a0e0b0xyg0 = adjp_cfw_initial_a0e0b0xyg[0:1,...]
    adjp_cfw_initial_a_a0e0b0xyg1 = adjp_cfw_initial_a0e0b0xyg[0:1,...]
    adjp_cfw_initial_a_a0e0b0xyg3 = adjp_cfw_initial_a0e0b0xyg
    adjp_fd_initial_a_a0e0b0xyg0 = adjp_fd_initial_a0e0b0xyg[0:1,...]
    adjp_fd_initial_a_a0e0b0xyg1 = adjp_fd_initial_a0e0b0xyg[0:1,...]
    adjp_fd_initial_a_a0e0b0xyg3 = adjp_fd_initial_a0e0b0xyg
    adjp_fl_initial_a_a0e0b0xyg0 = adjp_fl_initial_a0e0b0xyg[0:1,...]
    adjp_fl_initial_a_a0e0b0xyg1 = adjp_fl_initial_a0e0b0xyg[0:1,...]
    adjp_fl_initial_a_a0e0b0xyg3 = adjp_fl_initial_a0e0b0xyg
    ##adjustment for gender. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl)
    ### e.g. the impact of gender on cfw will be much less if only a short growth period (the parameter is a yearly factor e.g. male sheep have 0.02 kg more wool each yr)
    adja_lw_initial_x_xyg0 = cx_sire[17, 0:1, ...] #17 is the weaning wt parameter, 0:1 is the sire gender slice (retaining the axis).
    adja_lw_initial_x_xyg1 = cx_dams[17, 1:2, ...]
    adja_lw_initial_x_xyg3 = cx_offs[17, mask_x, ...]
    adja_cfw_initial_x_wzida0e0b0xyg0 = cx_sire[12, 0:1, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0
    adja_cfw_initial_x_wzida0e0b0xyg1 = cx_dams[12, 1:2, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
    adja_cfw_initial_x_wzida0e0b0xyg3 = cx_offs[12, mask_x, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    adja_fd_initial_x_xyg0 = cx_sire[13, 0:1, ...]
    adja_fd_initial_x_xyg1 = cx_dams[13, 1:2, ...]
    adja_fd_initial_x_xyg3 = cx_offs[13, mask_x, ...]
    adja_fl_initial_x_wzida0e0b0xyg0 = cx_sire[12, 0:1, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 # more understandable to think of the eqn as being fl_initial * cx[12] (cfw adj due to gender) / sfw
    adja_fl_initial_x_wzida0e0b0xyg1 = cx_dams[12, 1:2, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
    adja_fl_initial_x_wzida0e0b0xyg3 = cx_offs[12, mask_x, ...] * fl_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    ##adjust for dam age. Note cfw & fl accumulate during the year therefore the adjustment factor is divided by std_fw because the full effect is only realised after a full wool growth cycle (whereas a fibre diameter difference is expressed every day)
    adja_lw_initial_d_a0e0b0xyg0 = np.sum(ce_sire[17, ...] * agedam_propn_da0e0b0xyg0, axis=0) #d axis lost when summing
    adja_lw_initial_d_a0e0b0xyg1 = np.sum(ce_dams[17, ...] * agedam_propn_da0e0b0xyg1, axis=0)
    adja_lw_initial_d_da0e0b0xyg3 = ce_offs[17, ...]
    adja_cfw_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[12, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_da0e0b0xyg0, axis=d_pos, keepdims=True)
    adja_cfw_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[12, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_da0e0b0xyg1, axis=d_pos, keepdims=True)
    adja_cfw_initial_d_wzida0e0b0xyg3 = ce_offs[12, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    adja_fd_initial_d_a0e0b0xyg0 = np.sum(ce_sire[13, ...] * agedam_propn_da0e0b0xyg0, axis=0) #d axis lost when summing
    adja_fd_initial_d_a0e0b0xyg1 = np.sum(ce_dams[13, ...] * agedam_propn_da0e0b0xyg1, axis=0)
    adja_fd_initial_d_da0e0b0xyg3 = ce_offs[13, ...]
    adja_fl_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[12, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_da0e0b0xyg0, axis=d_pos, keepdims=True) #Should be fl_initial / sfw  So more understandable to think of the eqn as being fl_initial * cx[0] (cfw adj due to gender) / sfw
    adja_fl_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[12, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_da0e0b0xyg1, axis=d_pos, keepdims=True)
    adja_fl_initial_d_wzida0e0b0xyg3 = ce_offs[12, ...] * fl_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    ##adjust for btrt. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) e.g. the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor e.g. male sheep have 0.02 kg more wool each yr)
    adja_lw_initial_b0_xyg0 = np.sum(cb0_sire[17, ...] * btrt_propn_b0xyg0, axis=0) #d axis lost when summing
    adja_lw_initial_b0_xyg1 = np.sum(cb0_dams[17, ...] * btrt_propn_b0xyg1, axis=0)
    adja_lw_initial_b0_b0xyg3 = cb0_offs[17, ...]
    adja_cfw_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[12, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_propn_b0xyg0, axis=b0_pos, keepdims=True)
    adja_cfw_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[12, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_propn_b0xyg1, axis=b0_pos, keepdims=True)
    adja_cfw_initial_b0_wzida0e0b0xyg3 = cb0_offs[12, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    adja_fd_initial_b0_xyg0 = np.sum(cb0_sire[13, ...] * btrt_propn_b0xyg0, axis=0) #d axis lost when summing
    adja_fd_initial_b0_xyg1 = np.sum(cb0_dams[13, ...] * btrt_propn_b0xyg1, axis=0)
    adja_fd_initial_b0_b0xyg3 = cb0_offs[13, ...]
    adja_fl_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[12, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_propn_b0xyg0, axis=b0_pos, keepdims=True) #Should be fl_initial / sfw  So more understandable to think of the eqn as being fl_initial * cx[0] (cfw adj due to gender) / sfw
    adja_fl_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[12, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_propn_b0xyg1, axis=b0_pos, keepdims=True)
    adja_fl_initial_b0_wzida0e0b0xyg3 = cb0_offs[12, ...] * fl_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
    ##apply adjustments to initial variables
    lw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * (1 + adjp_lw_initial_a_a0e0b0xyg0) + adja_lw_initial_x_xyg0 + adja_lw_initial_d_a0e0b0xyg0 + adja_lw_initial_b0_xyg0
    lw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * (1 + adjp_lw_initial_a_a0e0b0xyg1) + adja_lw_initial_x_xyg1 + adja_lw_initial_d_a0e0b0xyg1 + adja_lw_initial_b0_xyg1
    lw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * (1 + adjp_lw_initial_a_a0e0b0xyg3) + adja_lw_initial_x_xyg3 + adja_lw_initial_d_da0e0b0xyg3 + adja_lw_initial_b0_b0xyg3
    cfw_initial_wzida0e0b0xyg0 = cfw_initial_wzida0e0b0xyg0 * (1 + adjp_cfw_initial_a_a0e0b0xyg0) + adja_cfw_initial_x_wzida0e0b0xyg0 + adja_cfw_initial_d_wzida0e0b0xyg0 + adja_cfw_initial_b0_wzida0e0b0xyg0
    cfw_initial_wzida0e0b0xyg1 = cfw_initial_wzida0e0b0xyg1 * (1 + adjp_cfw_initial_a_a0e0b0xyg1) + adja_cfw_initial_x_wzida0e0b0xyg1 + adja_cfw_initial_d_wzida0e0b0xyg1 + adja_cfw_initial_b0_wzida0e0b0xyg1
    cfw_initial_wzida0e0b0xyg3 = cfw_initial_wzida0e0b0xyg3 * (1 + adjp_cfw_initial_a_a0e0b0xyg3) + adja_cfw_initial_x_wzida0e0b0xyg3 + adja_cfw_initial_d_wzida0e0b0xyg3 + adja_cfw_initial_b0_wzida0e0b0xyg3
    fd_initial_wzida0e0b0xyg0 = fd_initial_wzida0e0b0xyg0 * (1 + adjp_fd_initial_a_a0e0b0xyg0) + adja_fd_initial_x_xyg0 + adja_fd_initial_d_a0e0b0xyg0 + adja_fd_initial_b0_xyg0
    fd_initial_wzida0e0b0xyg1 = fd_initial_wzida0e0b0xyg1 * (1 + adjp_fd_initial_a_a0e0b0xyg1) + adja_fd_initial_x_xyg1 + adja_fd_initial_d_a0e0b0xyg1 + adja_fd_initial_b0_xyg1
    fd_initial_wzida0e0b0xyg3 = fd_initial_wzida0e0b0xyg3 * (1 + adjp_fd_initial_a_a0e0b0xyg3) + adja_fd_initial_x_xyg3 + adja_fd_initial_d_da0e0b0xyg3 + adja_fd_initial_b0_b0xyg3
    fl_initial_wzida0e0b0xyg0 = fl_initial_wzida0e0b0xyg0 * (1 + adjp_fl_initial_a_a0e0b0xyg0) + adja_fl_initial_x_wzida0e0b0xyg0 + adja_fl_initial_d_wzida0e0b0xyg0 + adja_fl_initial_b0_wzida0e0b0xyg0
    fl_initial_wzida0e0b0xyg1 = fl_initial_wzida0e0b0xyg1 * (1 + adjp_fl_initial_a_a0e0b0xyg1) + adja_fl_initial_x_wzida0e0b0xyg1 + adja_fl_initial_d_wzida0e0b0xyg1 + adja_fl_initial_b0_wzida0e0b0xyg1
    fl_initial_wzida0e0b0xyg3 = fl_initial_wzida0e0b0xyg3 * (1 + adjp_fl_initial_a_a0e0b0xyg3) + adja_fl_initial_x_wzida0e0b0xyg3 + adja_fl_initial_d_wzida0e0b0xyg3 + adja_fl_initial_b0_wzida0e0b0xyg3

    ##if generating for stubble update initial params to reflect paddock trial
    if stubble:
        lw_initial_wzida0e0b0xyg1[...] = stubble['lw'][stubble['p_start']]
        lw_initial_wzida0e0b0xyg3[...] = stubble['lw'][stubble['p_start']]
        cfw_initial_wzida0e0b0xyg1[...] = pinp.stubble['i_gfw'] * cw_dams[3, ...]
        cfw_initial_wzida0e0b0xyg3[...] = pinp.stubble['i_gfw'] * cw_dams[3, ...]
        fd_initial_wzida0e0b0xyg1[...] = pinp.stubble['i_fd']
        fd_initial_wzida0e0b0xyg3[...] = pinp.stubble['i_fd']
        fl_initial_wzida0e0b0xyg1[...] = pinp.stubble['i_fl']
        fl_initial_wzida0e0b0xyg3[...] = pinp.stubble['i_fl']

    ##calc initial ffcfw
    ffcfw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 - cfw_initial_wzida0e0b0xyg0 / cw_sire[3, ...]
    ffcfw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 - cfw_initial_wzida0e0b0xyg1 / cw_dams[3, ...]
    ffcfw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 - cfw_initial_wzida0e0b0xyg3 / cw_offs[3, ...]

    ##calc aw, bw and mw (adipose, bone and muscle weight)
    aw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * aw_propn_wean_yg0
    aw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * aw_propn_wean_yg1
    aw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * aw_propn_wean_yg3
    bw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * bw_propn_wean_yg0
    bw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * bw_propn_wean_yg1
    bw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * bw_propn_wean_yg3
    mw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * mw_propn_wean_yg0
    mw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * mw_propn_wean_yg1
    mw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * mw_propn_wean_yg3

    ##if stubble update aw, bw & mw.
    if stubble:
        aw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * pinp.stubble['i_aw']
        aw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * pinp.stubble['i_aw']
        aw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * pinp.stubble['i_aw']
        bw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * pinp.stubble['i_bw']
        bw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * pinp.stubble['i_bw']
        bw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * pinp.stubble['i_bw']
        mw_initial_wzida0e0b0xyg0 = ffcfw_initial_wzida0e0b0xyg0 * pinp.stubble['i_mw']
        mw_initial_wzida0e0b0xyg1 = ffcfw_initial_wzida0e0b0xyg1 * pinp.stubble['i_mw']
        mw_initial_wzida0e0b0xyg3 = ffcfw_initial_wzida0e0b0xyg3 * pinp.stubble['i_mw']

    ##numbers
    ###Distribution of initial numbers across the a1 axis
    initial_a1 = pinp.sheep['i_initial_a1'][pinp.sheep['i_mask_a']] / np.sum(pinp.sheep['i_initial_a1'][pinp.sheep['i_mask_a']])
    initial_a1e1b1nwzida0e0b0xyg = fun.f_expand(initial_a1, a1_pos)
    ###Distribution of initial numbers across the b1 axis
    initial_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['i_initial_b1'], b1_pos)
    ###Distribution of initial numbers across the y-axis
    initial_yg0 = fun.f_expand(uinp.parameters['i_initial_y0'], y_pos, condition = uinp.parameters['i_mask_y0'], axis = y_pos)
    initial_yg1 = fun.f_expand(uinp.parameters['i_initial_y1'], y_pos, condition = uinp.parameters['i_mask_y1'], axis = y_pos)
    initial_yg3 = fun.f_expand(uinp.parameters['i_initial_y3'], y_pos, condition = uinp.parameters['i_mask_y3'], axis = y_pos)
    ###Distribution of initial numbers across the e1 axis
    initial_e1 = np.zeros(len_e1)
    initial_e1[0]=1
    initial_e1b1nwzida0e0b0xyg = fun.f_expand(initial_e1, e1_pos)
    ###distribution of numbers on the e0 axis
    propn_e_ida0e0b0xyg = fun.f_expand(pinp.sheep['i_propn_e_i'], i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    ###If 70% of ewes that are not pregnant get pregnant in a cycle then
    ###first cycle 70% pregnant (30% not pregnant)
    ###2nd cycle 70% of the 30% get pregnant = 21% (9 % not pregnant)
    ###3rd cycle 70% of the 9% = 6.3% (2.7% not pregnant)
    ###Proportion conceiving in each cycle from the view of the proportion of dams conceiving in each cycle
    e0_propn_ida0e0b0xyg = (1 - propn_e_ida0e0b0xyg) ** index_e0b0xyg * propn_e_ida0e0b0xyg
    #propn of the off born in each cycle (propn of dams conceived in each cycle / total number conceiving during the mating period)
    e0_propn_ida0e0b0xyg = e0_propn_ida0e0b0xyg / np.sum(e0_propn_ida0e0b0xyg, axis = e0_pos, keepdims=True)
    ###sire
    numbers_initial_zida0e0b0xyg0 = initial_yg0
    ###dams
    numbers_initial_propn_repro_a1e1b1nwzida0e0b0xyg1 = initial_a1e1b1nwzida0e0b0xyg * initial_e1b1nwzida0e0b0xyg * initial_b1nwzida0e0b0xyg
    numbers_initial_a1e1b1nwzida0e0b0xyg1 = numbers_initial_propn_repro_a1e1b1nwzida0e0b0xyg1 * initial_yg1
    ###offs
    ####Initial proportion of offspring if clustered.
    ####These proportions are used if the offspring are clustered
    numbers_initial_cluster_ida0e0b0xyg3 = btrt_propn_b0xyg3 * e0_propn_ida0e0b0xyg
    ####initial offs numbers
    numbers_initial_ida0e0b0xyg3 = initial_yg3 * numbers_initial_cluster_ida0e0b0xyg3

    ##set default numbers to initial numbers. So that p0 (dvp0 has numbers_start)
    o_numbers_start_tpsire[...] =  numbers_initial_zida0e0b0xyg0  # default 1 so that dvp0 (p0) has start numbers
    o_numbers_start_tpdams[...] =  numbers_initial_a1e1b1nwzida0e0b0xyg1  # default 1 so that dvp0 (p0) has start numbers
    o_numbers_start_tpoffs[...] = numbers_initial_ida0e0b0xyg3 # ones so that dvp0 (p0) has start numbers.

    #######################
    ##Age, date, timing 1 #
    #######################

    ## date mated (average date that the dams conceive in this cycle)
    date_mated_pa1e1b1nwzida0e0b0xyg1 = date_born2_pa1e1b1nwzida0e0b0xyg2 - cp_dams[1,0:1,:] #use dateborn2 so it increments at prejoining
    ##day90 after mating (for use in the LTW calculations)
    date_d90_pa1e1b1nwzida0e0b0xyg1 = date_mated_pa1e1b1nwzida0e0b0xyg1 + np.array([90])
    ##pre-lambing assessment in MU trials (135 days after mating (for use in the LTW calculations))
    date_prebirth_pa1e1b1nwzida0e0b0xyg1 = date_mated_pa1e1b1nwzida0e0b0xyg1 + np.array([135])
    ##Age of dam when first lamb is born
    agedam_lamb1st_a1e1b1nwzida0e0b0xyg3 = np.swapaxes(date_born1st_oa1e1b1nwzida0e0b0xyg2 - date_born1st_ida0e0b0xyg1,0,d_pos)[0,...] #replace the d axis with the o axis then remove the d axis by taking slice 0 (note the d axis was not active)
    if np.count_nonzero(pinp.sheep['i_mask_i']) > 1: #complicated by the fact that sire tol is not necessarily the same as dams and off
        agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 = np.compress(pinp.sheep['i_masksire_i'], agedam_lamb1st_a1e1b1nwzida0e0b0xyg3[...,a_g3_g0], i_pos) #don't mask if both tol are included
    else:
        agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 = np.compress(pinp.sheep['i_masksire_i'][pinp.sheep['i_masksire_i']], agedam_lamb1st_a1e1b1nwzida0e0b0xyg3[...,a_g3_g0], i_pos) #have to mask masksire_i  because it needs to be the same length as i
    agedam_lamb1st_a1e1b1nwzida0e0b0xyg1 = agedam_lamb1st_a1e1b1nwzida0e0b0xyg3[...,a_g3_g1]
    agedam_lamb1st_a1e1b1nwzida0e0b0xyg3 = np.compress(mask_d_offs, agedam_lamb1st_a1e1b1nwzida0e0b0xyg3, d_pos) #mask d axis (compress function masks a specific axis)

    ##wean date (weaning input is counting from the date of the first lamb (not the date of the average lamb in the first cycle))
    date_weaned_pa1e1b1nwzida0e0b0xyg2 = date_born1st_pa1e1b1nwzida0e0b0xyg2 + age_wean1st_pa1e1b1nwzida0e0b0xyg2 #use offs wean age input and has the same birth day (offset by a yr) therefore it will automatically align with a period start

    ##age start open (not capped at weaning or before birth) used to calc p1 stuff
    age_start_open_pa1e1b1nwzida0e0b0xyg0 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_ida0e0b0xyg0
    age_start_open_pa1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_ida0e0b0xyg1
    age_start_open_pa1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg3 - date_born_ida0e0b0xyg3
    age_start_open_pa1e1b1nwzida0e0b0xyg2 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_pa1e1b1nwzida0e0b0xyg2
    ##age start
    age_start_pa1e1b1nwzida0e0b0xyg0 = (np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg0) - date_born_ida0e0b0xyg0).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_start_pa1e1b1nwzida0e0b0xyg1 = (np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg1) - date_born_ida0e0b0xyg1).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_start_pa1e1b1nwzida0e0b0xyg3 = (np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg3) - date_born_ida0e0b0xyg3).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_start_pa1e1b1nwzida0e0b0xyg2 = (np.maximum(np.array([0]),np.minimum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2) - date_born_pa1e1b1nwzida0e0b0xyg2)).astype(int) #use min and max so that the min age is 0 and the max age is the age at weaning
    ##Age_end: age at the beginning of the last day of the given period
    ##age end, minus one to allow the plus one in the next step when period date is less than weaning date (the minus one ensures that when the p_date is less than weaning the animal gets 0 days in the given period)
    age_end_pa1e1b1nwzida0e0b0xyg0 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg0 -1) - date_born_ida0e0b0xyg0).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg1 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg1 -1) - date_born_ida0e0b0xyg1).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg3 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg3 -1) - date_born_ida0e0b0xyg3).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg2 = (np.maximum(np.array([-1]),np.minimum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2 -1) - date_born_pa1e1b1nwzida0e0b0xyg2)).astype(int)  #use min and max so that the min age is 0 and the max age is the age at weaning

    ##age mid-period , plus one to get the age at the end of the last day of the period ie needed to get the full len of period.
    age_pa1e1b1nwzida0e0b0xyg0 = (age_start_pa1e1b1nwzida0e0b0xyg0 + age_end_pa1e1b1nwzida0e0b0xyg0 +1) /2
    age_pa1e1b1nwzida0e0b0xyg1 = (age_start_pa1e1b1nwzida0e0b0xyg1 + age_end_pa1e1b1nwzida0e0b0xyg1 +1) /2
    age_pa1e1b1nwzida0e0b0xyg2 = (age_start_pa1e1b1nwzida0e0b0xyg2 + age_end_pa1e1b1nwzida0e0b0xyg2 +1) /2
    age_pa1e1b1nwzida0e0b0xyg3 = (age_start_pa1e1b1nwzida0e0b0xyg3 + age_end_pa1e1b1nwzida0e0b0xyg3 +1) /2

    ##days in each period for each animal - can't mask the offs p axis because need full axis so it can be used in the generator (if days_period[p] > 0)
    days_period_pa1e1b1nwzida0e0b0xyg0 = age_end_pa1e1b1nwzida0e0b0xyg0 +1 - age_start_pa1e1b1nwzida0e0b0xyg0
    days_period_pa1e1b1nwzida0e0b0xyg1 = age_end_pa1e1b1nwzida0e0b0xyg1 +1 - age_start_pa1e1b1nwzida0e0b0xyg1
    days_period_pa1e1b1nwzida0e0b0xyg2 = age_end_pa1e1b1nwzida0e0b0xyg2 +1 - age_start_pa1e1b1nwzida0e0b0xyg2
    days_period_pa1e1b1nwzida0e0b0xyg3 = (age_end_pa1e1b1nwzida0e0b0xyg3 +1 - age_start_pa1e1b1nwzida0e0b0xyg3
                                          ) * (p_index_pa1e1b1nwzida0e0b0xyg<np.count_nonzero(mask_p_offs_p)-1) #make days per period zero if period is not required for offs. -1 because we want days per period to be 0 in the period before the mask turns to false because there are some spots where p+1 is used as the index
    days_period_cut_pa1e1b1nwzida0e0b0xyg3 = days_period_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] #masked version of p axis

    ##Age of foetus (start of period, end of period and mid-period - days)
    #todo will need fixing when adding cattle
    age_f_start_open_pa1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1
    age_f_start_pa1e1b1nwzida0e0b0xyg1 = np.maximum(np.array([0])
                                            , np.minimum(cp_dams[1, 0:1, :]
                                                , date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1))
    age_f_end_pa1e1b1nwzida0e0b0xyg1 = np.minimum(cp_dams[1, 0:1, :] - 1
                                                  , date_end_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1) #open at bottom capped at top, cp -1 so that the period_days formula below is correct when p_date - date_mated is greater than cp (because plus 1)


    ############################
    ### Daily steps            #
    ############################
    ##definition for this is that the action e.g. weaning, occurs at 12am on the given date.
    ## Therefore, if weaning occurs on day 150 the lambs are counted as weaned lambs on that day.
    ##This info determines the side with > or >=


    ##add p0 axis
    date_start_pa1e1b1nwzida0e0b0xygp0 = date_start_pa1e1b1nwzida0e0b0xyg[...,na] + index_p0
    ##age open ie not capped at weaning
    age_p0_pa1e1b1nwzida0e0b0xyg0p0 = (age_start_open_pa1e1b1nwzida0e0b0xyg0[..., na] + index_p0)
    age_p0_pa1e1b1nwzida0e0b0xyg1p0 = (age_start_open_pa1e1b1nwzida0e0b0xyg1[..., na] + index_p0)
    age_p0_pa1e1b1nwzida0e0b0xyg2p0 = (age_start_open_pa1e1b1nwzida0e0b0xyg2[..., na] + index_p0)
    age_p0_pa1e1b1nwzida0e0b0xyg3p0 = (age_start_open_pa1e1b1nwzida0e0b0xyg3[..., na] + index_p0)
    ##calc p1 weights - if age<=weaning it has 0 weighting (ie false) else weighting = 1 (ie true) - need so that the values are not included in the mean calculations below which determine the average production for a given p1 period.
    age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0 = age_p0_pa1e1b1nwzida0e0b0xyg0p0>=(date_weaned_ida0e0b0xyg0 - date_born_ida0e0b0xyg0)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0 = age_p0_pa1e1b1nwzida0e0b0xyg1p0>=(date_weaned_ida0e0b0xyg1 - date_born_ida0e0b0xyg1)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0 = age_p0_pa1e1b1nwzida0e0b0xyg3p0>=(date_weaned_ida0e0b0xyg3 - date_born_ida0e0b0xyg3)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    ##calc yatf p1 weighting - if age is greater than weaning or less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0 = np.logical_and(age_p0_pa1e1b1nwzida0e0b0xyg2p0>=0, age_p0_pa1e1b1nwzida0e0b0xyg2p0<(date_weaned_pa1e1b1nwzida0e0b0xyg2 - date_born_pa1e1b1nwzida0e0b0xyg2)[..., na].astype(int)) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    ##Age of foetus with minor axis (days)
    age_f_p0_pa1e1b1nwzida0e0b0xyg1p0 = (age_f_start_open_pa1e1b1nwzida0e0b0xyg1[...,na] + index_p0)
    ##calc foetus p1 weighting - if age is greater than birth or less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    #todo will need fixing when adding cattle
    age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0 = np.logical_and(age_f_p0_pa1e1b1nwzida0e0b0xyg1p0>=0, age_f_p0_pa1e1b1nwzida0e0b0xyg1p0<cp_dams[1, 0, :, na])
    #age_f_p0_pa1e1b1nwzida0e0b0xyg1p0[age_f_p0_pa1e1b1nwzida0e0b0xyg1p0 <= 0] = np.nan
    #age_f_p0_pa1e1b1nwzida0e0b0xyg1p0[age_f_p0_pa1e1b1nwzida0e0b0xyg1p0 > cp_dams[1, 0, :, na]] = np.nan
    ##adjusted age of young (adjusted by intake factor - basically the factor of how age of young effect dam intake, the adjustment factor basically alters the age of the young to influence intake.)
    age_y_adj_pa1e1b1nwzida0e0b0xyg1p0 = age_p0_pa1e1b1nwzida0e0b0xyg2p0 + np.maximum(0, (date_start_pa1e1b1nwzida0e0b0xygp0 - date_weaned_pa1e1b1nwzida0e0b0xyg2[..., na])) * (ci_dams[21, ..., na] - 1) #minus 1 because the ci factor is applied to the age post weaning but using the open date means it has already been included once ie we want x + y *ci but using date open gives  x  + y + y*ci, x = age to weaning, y = age between period and weaning, therefore minus 1 x  + y + y*(ci-1)
    ##calc young p1 weighting - if age is less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    age_y_adj_weights_pa1e1b1nwzida0e0b0xyg1p0 = age_y_adj_pa1e1b1nwzida0e0b0xyg1p0 > 0  #no max cap (ie represents age young would be if never weaned off mum)
    ##Foetal age relative to parturition (based on gestation length of the first genotype) with p0 axis
    #todo this will need fixing when cattle added that have different gestation length
    relage_f_pa1e1b1nwzida0e0b0xyg1p0 = np.maximum(0,age_f_p0_pa1e1b1nwzida0e0b0xyg1p0 / cp_dams[1, 0, :, na])
    ##Age of lamb relative to peak intake-with minor function
    pimi_pa1e1b1nwzida0e0b0xyg1p0 = age_y_adj_pa1e1b1nwzida0e0b0xyg1p0 / ci_dams[8, ..., na]
    ##Age of lamb relative to peak lactation-with minor axis
    lmm_pa1e1b1nwzida0e0b0xyg1p0 = (age_p0_pa1e1b1nwzida0e0b0xyg2p0 + cl_dams[1, ..., na]) / cl_dams[2, ..., na]
    ##Chill index for lamb survival
    #todo consider adding p1p2p3 axes for chill for rain, ws & temp_ave.
    chill_index_pa1e1b1nwzida0e0b0xygp0 = (481 + (11.7 + 3.1 * ws_pa1e1b1nwzida0e0b0xyg[..., na] ** 0.5)
                                           * (40 - temp_ave_pa1e1b1nwzida0e0b0xyg[..., na])
                                           + 418 * (1-np.exp(-0.04 * rain_pa1e1b1nwzida0e0b0xygp0))
                                           + chill_adj_pa1e1b1nwzida0e0b0xyg1[..., na])
    chill_index_pa1e1b1nwzida0e0b0xygp0 = fun.f_sa(chill_index_pa1e1b1nwzida0e0b0xygp0, sen.sam['chill'])

    ##Proportion of SRW with age
    srw_age_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.exp(-cn_sire[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg0p0
                            / srw_b0xyg0[..., na] ** cn_sire[2, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(-cn_dams[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg1p0
                            / srw_b0xyg1[..., na] ** cn_dams[2, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.exp(-cn_yatf[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg2p0
                            / srw_b1xyg2[..., na] ** cn_yatf[2, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.exp(-cn_offs[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg3p0
                            / srw_b0xyg3[..., na] ** cn_offs[2, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0, axis = -1)

    #srw_age_pa1e1b1nwzida0e0b0xyg0 = np.nanmean(np.exp(-cn_sire[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg0p0 / srw_b0xyg0[..., na] ** cn_sire[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(-cn_dams[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg1p0 / srw_b0xyg1[..., na] ** cn_dams[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(np.exp(-cn_yatf[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg2p0 / srw_b1xyg2[..., na] ** cn_yatf[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg3 = np.nanmean(np.exp(-cn_offs[1, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg3p0 / srw_b0xyg3[..., na] ** cn_offs[2, ..., na]), axis = -1)

    ##age factor wool part 1- reduces fleece growth early in life based on Lyne 1961 & follicle maturation
    af1_wool_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(cw_sire[5, ..., na] + (1 - cw_sire[5, ..., na])
                                                * (1-np.exp(-cw_sire[12, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg0p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cw_dams[5, ..., na] + (1 - cw_dams[5, ..., na])
                                                * (1-np.exp(-cw_dams[12, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg1p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(cw_yatf[5, ..., na] + (1 - cw_yatf[5, ..., na])
                                                * (1-np.exp(-cw_yatf[12, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg2p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(cw_offs[5, ..., na] + (1 - cw_offs[5, ..., na])
                                                * (1-np.exp(-cw_offs[12, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg3p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0, axis = -1)
    ##age factor wool part 2 - reduces fleece growth later in life (data used to create equations from Lifetime Productivity by Richards&Atkins)
    af2_wool_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(2 - np.exp(cw_sire[17, ..., na]
                                                * np.maximum(0,age_p0_pa1e1b1nwzida0e0b0xyg0p0 - cw_sire[18, ..., na])
                                                **cw_sire[19, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(2 - np.exp(cw_dams[17, ..., na]
                                                * np.maximum(0,age_p0_pa1e1b1nwzida0e0b0xyg1p0 - cw_dams[18, ..., na])
                                                **cw_dams[19, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(2 - np.exp(cw_yatf[17, ..., na]
                                                * np.maximum(0,age_p0_pa1e1b1nwzida0e0b0xyg2p0 - cw_yatf[18, ..., na])
                                                **cw_yatf[19, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(2 - np.exp(cw_offs[17, ..., na]
                                                * np.maximum(0,age_p0_pa1e1b1nwzida0e0b0xyg3p0 - cw_offs[18, ..., na])
                                                **cw_offs[19, ..., na]), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0, axis = -1)
    ##overall age factor - reduction for young animals and older animals
    af_wool_pa1e1b1nwzida0e0b0xyg0 = af1_wool_pa1e1b1nwzida0e0b0xyg0 * af2_wool_pa1e1b1nwzida0e0b0xyg0
    af_wool_pa1e1b1nwzida0e0b0xyg1 = af1_wool_pa1e1b1nwzida0e0b0xyg1 * af2_wool_pa1e1b1nwzida0e0b0xyg1
    af_wool_pa1e1b1nwzida0e0b0xyg2 = af1_wool_pa1e1b1nwzida0e0b0xyg2 * af2_wool_pa1e1b1nwzida0e0b0xyg2
    af_wool_pa1e1b1nwzida0e0b0xyg3 = af1_wool_pa1e1b1nwzida0e0b0xyg3 * af2_wool_pa1e1b1nwzida0e0b0xyg3

    ##Day length factor on efficiency
    dlf_eff_pa1e1b1nwzida0e0b0xyg = np.average(lat_deg / 40 * np.sin(2 * np.pi * doy_pa1e1b1nwzida0e0b0xygp0 / 364), axis = -1)
    ##Pattern of maintenance with age
    mr_age_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.maximum(cm_sire[4, ..., na], np.exp(-cm_sire[3, ..., na]
                                        * age_p0_pa1e1b1nwzida0e0b0xyg0p0)), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(cm_dams[4, ..., na], np.exp(-cm_dams[3, ..., na]
                                        * age_p0_pa1e1b1nwzida0e0b0xyg1p0)), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.maximum(cm_offs[4, ..., na], np.exp(-cm_offs[3, ..., na]
                                        * age_p0_pa1e1b1nwzida0e0b0xyg2p0)), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.maximum(cm_yatf[4, ..., na], np.exp(-cm_yatf[3, ..., na]
                                        * age_p0_pa1e1b1nwzida0e0b0xyg3p0)), weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0, axis = -1)
    ##Impact of rainfall on 'cold' intake increment
    rain_intake_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp0 / ci_sire[18, ..., na]),  weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg0p0, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp0 / ci_dams[18, ..., na]),  weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp0 / ci_offs[18, ..., na]),  weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp0[mask_p_offs_p] / ci_yatf[18, ..., na]),  weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg3p0, axis = -1)
    ##Proportion of peak intake due to time from birth
    pi_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cb1_dams[19, ..., na] * np.maximum(0,pimi_pa1e1b1nwzida0e0b0xyg1p0) ** ci_dams[9, ..., na] * np.exp(ci_dams[9, ..., na] * (1 - pimi_pa1e1b1nwzida0e0b0xyg1p0)), weights=age_y_adj_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1) #maximum to stop error in power (not sure why the negatives were causing a problem)
    ##Peak milk production pattern (time from birth). Includes scalar for milk yield (cl[0]). Average for the days that the dam is lactating
    mp_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cl_dams[0, ..., na] * cb1_dams[0, ..., na]
                                        * lmm_pa1e1b1nwzida0e0b0xyg1p0 ** cl_dams[3, ..., na]
                                        * np.exp(cl_dams[3, ..., na] * (1 - lmm_pa1e1b1nwzida0e0b0xyg1p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    ##Suckling volume pattern. Includes scalar for milk yield (cl[0]) and SA for potential intake of the young at foot.
    ## Average for the days that the dam is lactating
    mp2_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cl_dams[0, ..., na] * nyatf_b1nwzida0e0b0xyg[...,na]
                                        * cl_dams[6, ..., na] * ( cl_dams[12, ..., na] + cl_dams[13, ..., na]
                                        * np.exp(-cl_dams[14, ..., na] * age_p0_pa1e1b1nwzida0e0b0xyg2p0))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1) * sen.sam['pi_yatf']
    ##Pattern of conception efficiency (doy). Different methods are used to represent seasonality in the 3 conception functions
    ### cpg_doy_cs is for the GrazPlan equations to predict the seasonal effect on proportion greater than conception rate - active b1 axis
    cpg_doy_cs_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.maximum(0,1 - cb1_dams[1, ..., na]
                                                * (1 - np.sin(2 * np.pi * (doy_pa1e1b1nwzida0e0b0xygp1 + 10) / 364))
                                                * np.sin(lat_rad) / -0.57), axis = -1)
    ### rr_doy_ltw scales the LTW equations to predict the seasonal effect on reproductive rate - singleton b1 axis
    ### value is scaled so at the doy of scan_std the value is 1 and doesn't alter scan_std if mating on that day
    rr_doy_ltw_pa1e1b1nwzida0e0b0xyg1 = fun.f_divide(np.nanmean(np.maximum(0,1 - cf_dams[1, ..., na]
                                                    * (1 - np.sin(2 * np.pi * (doy_pa1e1b1nwzida0e0b0xygp1 + 10) / 364))
                                                    * np.sin(lat_rad) / -0.57), axis = -1)
                                                , np.maximum(0, 1 - cf_dams[1, ...]
                                                    * (1 - np.sin(2 * np.pi * (scan_std_doj_yg1[...] + 10) / 364))
                                                    * np.sin(lat_rad) / -0.57))
    # cpl no longer used in the LMAT equations. Replaced by DOJ, DOJ**2 & Latitude in the transformed equation
    #### doj & doj2 are multiplied by a coefficient in the linear equation prior to logit back transformation.
    #### The proportions of empty, single, twin & triplet change in the ratio determined by the cut-off coefficients.
    ### doj as fitted by Gav was days from 1 Sep (day 244 of the year) and counting up until the following Sep
    ### The coefficients were converted to day of the year see Calibration1 pg14 with 1 Jan as D0 and negative if earlier.
    doj_pa1e1b1nwzida0e0b0xygp1 = doy_pa1e1b1nwzida0e0b0xygp1
    doj_pa1e1b1nwzida0e0b0xygp1[doj_pa1e1b1nwzida0e0b0xygp1 >= 244] -= 364
    doj_pa1e1b1nwzida0e0b0xygp1 = np.clip(doj_pa1e1b1nwzida0e0b0xygp1, -52, 98)
    doj_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(doj_pa1e1b1nwzida0e0b0xygp1, axis = -1)
    doj2_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(doj_pa1e1b1nwzida0e0b0xygp1 ** 2, axis=-1)

    ##Rumen development factor on PI - yatf
    piyf_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(fun.f_back_transform(ci_yatf[3, ..., na]
                                        * (age_p0_pa1e1b1nwzida0e0b0xyg2p0 - ci_yatf[4, ..., na]))
                                                , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    # piyf_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(1/(1 + np.exp(-ci_yatf[3, ..., na]
    #                                     * (age_p0_pa1e1b1nwzida0e0b0xyg2p0 - ci_yatf[4, ..., na])))
    #                                           , weights=age_p0_weights_pa1e1b1nwzida0e0b0xyg2p0, axis = -1)
    piyf_pa1e1b1nwzida0e0b0xyg2 = piyf_pa1e1b1nwzida0e0b0xyg2 * (nyatf_b1nwzida0e0b0xyg > 0) #set pi to 0 if no yatf.
    ##Foetal normal weight pattern (mid-period)
    nwf_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[2, ..., na] * (1 - np.exp(cp_dams[3, ..., na]
                                            * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p0))))
                                                    , weights=age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    ##Conceptus weight pattern (mid-period)
    guw_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[6, ..., na] * (1 - np.exp(cp_dams[7, ..., na]
                                            * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p0))))
                                                    , weights=age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    ##Conceptus energy pattern (end of period)
    # ce_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[9, ..., na] * (1 - np.exp(cp_dams[10, ..., na]
    #                                       * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p0))))
    #                                               , weights=age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    ##Conceptus energy pattern (d_nec). Average for the days that the dam is gestating
    dce_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cp_dams[9, ..., na] * cp_dams[10, ..., na] / cp_dams[1, 0, ..., na]
                                            * np.exp(cp_dams[10, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p0)
                                            + cp_dams[9, ..., na] * (1 - np.exp(cp_dams[10, ..., na]
                                            * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p0))))
                                                    , weights=age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    ##Conceptus energy pattern (dcdt) for New Feeding Standards. Average for the days that the dam is gestating
    dcdt_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cp_dams[17, ..., na] * cp_dams[18, ..., na]
                                            * np.exp(-cp_dams[18, ..., na] * age_f_p0_pa1e1b1nwzida0e0b0xyg1p0)
                                                    , weights=age_f_p0_weights_pa1e1b1nwzida0e0b0xyg1p0, axis = -1)
    ##Conceptus energy pattern (c_start) on day 1
    ce_day1_f_dams = np.exp(cp_dams[16, ..., na] - cp_dams[17, ..., na] * np.exp(-cp_dams[18, 0, ..., na] * 1)) / 4

    ##genotype calc that requires af_wool. ME for minimum wool growth (with no intake, relsize = 1)
    mew_min_pa1e1b1nwzida0e0b0xyg0 =cw_sire[14, ...] * sfw_a0e0b0xyg0[0, ...] / cw_sire[3,...] / 364 * af_wool_pa1e1b1nwzida0e0b0xyg0 * dlf_wool_pa1e1b1nwzida0e0b0xyg0 * cw_sire[1, ...] / kw_yg0
    mew_min_pa1e1b1nwzida0e0b0xyg1 =cw_dams[14, ...] * sfw_a0e0b0xyg1[0, ...] / cw_dams[3,...] / 364 * af_wool_pa1e1b1nwzida0e0b0xyg1 * dlf_wool_pa1e1b1nwzida0e0b0xyg1 * cw_dams[1, ...] / kw_yg1
    mew_min_pa1e1b1nwzida0e0b0xyg2 =cw_yatf[14, ...] * sfw_pa1e1b1nwzida0e0b0xyg2[0, ...] / cw_yatf[3,...] / 364 * af_wool_pa1e1b1nwzida0e0b0xyg2 * dlf_wool_pa1e1b1nwzida0e0b0xyg2 * cw_yatf[1, ...] / kw_yg2
    mew_min_pa1e1b1nwzida0e0b0xyg3 =cw_offs[14, ...] * sfw_da0e0b0xyg3[0, ...] / cw_offs[3,...] / 364 * af_wool_pa1e1b1nwzida0e0b0xyg3 * dlf_wool_pa1e1b1nwzida0e0b0xyg3 * cw_offs[1, ...] / kw_yg3

    ##plot above x-axis is p
    # array = srw_age_pa1e1b1nwzida0e0b0xyg2
    # array = dce_age_f_pa1e1b1nwzida0e0b0xyg1
    # array = guw_age_f_pa1e1b1nwzida0e0b0xyg1
    # array = nwf_age_f_pa1e1b1nwzida0e0b0xyg1
    # array = mp2_age_y_pa1e1b1nwzida0e0b0xyg1
    # array = mp_age_y_pa1e1b1nwzida0e0b0xyg1
    # array = piyf_pa1e1b1nwzida0e0b0xyg2
    # array = af_wool_pa1e1b1nwzida0e0b0xyg2
    # array = mr_age_pa1e1b1nwzida0e0b0xyg2
    # shape=array.shape
    # for a1 in range(shape[-14]):
    #     for e1 in range(shape[-13]):
    #         for b1 in range(shape[-12]):
    #             for n in range(shape[-11]):
    #                 for w in range(shape[-10]):
    #                     for z in range(shape[-9]):
    #                         for i in range(shape[-8]):
    #                             for d in range(shape[-7]):
    #                                 for a0 in range(shape[-6]):
    #                                     for e0 in range(shape[-5]):
    #                                         for b0 in range(shape[-4]):
    #                                             for x in range(shape[-3]):
    #                                                 for y in range(shape[-2]):
    #                                                     for g in range(shape[-1]):
    #                                                         plt.plot(array[:100, a1-1, e1-1, b1-1, n-1, w-1, z-1, i-1, d-1, a0-1, e0-1, b0-1, x-1, y-1, g-1])
    # plt.show()



    #######################
    ##Age, date, timing 2 #
    #######################
    ##Days per period in the simulation - foetus
    days_period_f_pa1e1b1nwzida0e0b0xyg1 = np.maximum(0,(age_f_end_pa1e1b1nwzida0e0b0xyg1 +1 - age_f_start_pa1e1b1nwzida0e0b0xyg1).astype(int))

    ##proportion of days gestating during the period
    gest_propn_pa1e1b1nwzida0e0b0xyg1 = days_period_f_pa1e1b1nwzida0e0b0xyg1 / np.maximum(1, days_period_pa1e1b1nwzida0e0b0xyg1) #use max to stop div/0 error (when dams days per period =0 then days_f will also be zero)

    ##proportion of days of lactating during the period
    lact_propn_pa1e1b1nwzida0e0b0xyg1 = days_period_pa1e1b1nwzida0e0b0xyg2 / np.maximum(1, days_period_pa1e1b1nwzida0e0b0xyg1) #use max to stop div/0 error (when dams days per period =0 then days_f will also be zero)

    ##Is nutrition effecting lactation (no effect until about day 15 = cl16 * cl2)
    lact_nut_effect_pa1e1b1nwzida0e0b0xyg1 = (age_pa1e1b1nwzida0e0b0xyg2  > (cl_dams[16, ...] * cl_dams[2, ...]))

    ##Average daily CFW
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg0 = sfw_a0e0b0xyg0 * af_wool_pa1e1b1nwzida0e0b0xyg0 / 364
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg1 = sfw_a0e0b0xyg1 * af_wool_pa1e1b1nwzida0e0b0xyg1 / 364
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg2 = sfw_pa1e1b1nwzida0e0b0xyg2 * af_wool_pa1e1b1nwzida0e0b0xyg2 / 364
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg3 = sfw_da0e0b0xyg3 * af_wool_pa1e1b1nwzida0e0b0xyg3 / 364

    ##Expected relative size
    relsize_exp_a1e1b1nwzida0e0b0xyg0  = (srw_b0xyg0 - (srw_b0xyg0 - w_b_std_b0xyg0) * np.exp(-cn_sire[1, ...] * agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 / (srw_b0xyg0**cn_sire[2, ...]))) / srw_b0xyg0
    relsize_exp_a1e1b1nwzida0e0b0xyg1  = (srw_b0xyg1 - (srw_b0xyg1 - w_b_std_b0xyg1) * np.exp(-cn_dams[1, ...] * agedam_lamb1st_a1e1b1nwzida0e0b0xyg1 / (srw_b0xyg1**cn_dams[2, ...]))) / srw_b0xyg1
    relsize_exp_a1e1b1nwzida0e0b0xyg3  = (srw_b0xyg3 - (srw_b0xyg3 - w_b_std_b0xyg3) * np.exp(-cn_offs[1, ...] * agedam_lamb1st_a1e1b1nwzida0e0b0xyg3 / (srw_b0xyg3**cn_offs[2, ...]))) / srw_b0xyg3

    ##Adjust the tissue insulation parameter (cc[3]) for yatf 30 days or younger.
    shape = (cc_yatf.shape[0],) + age_pa1e1b1nwzida0e0b0xyg2.shape
    cc_cpa1e1b1nwzida0e0b0xyg2 = np.zeros(shape)  #make a new array - similar to age_yatf but with c axis
    cc_cpa1e1b1nwzida0e0b0xyg2[...] = fun.f_expand(cc_yatf, p_pos-1, right_pos=uinp.parameters['i_y_pos'])
    cc_cpa1e1b1nwzida0e0b0xyg2[3:4, ...] *= np.minimum(1, 0.4 + 0.02 * age_pa1e1b1nwzida0e0b0xyg2)
    cc_pyatf = cc_cpa1e1b1nwzida0e0b0xyg2 #rename to keep consistent

    ##adjust ce sim param (^ ce12 &13 should be scaled by relsize (similar to ce15)) -  (#todo instead of setting ce with relsize adjustment then adjusting birth weight could just adjust birthweight directly with relsize factor - to avoid doing this code below)
    shape = (ce_sire.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg0.shape #get shape of the new ce array
    ce_ca1e1b1nwzida0e0b0xyg0 = np.zeros(shape) #make a new array - same as ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg0[...] = fun.f_expand(ce_sire, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg0[15, ...] = 1 - cp_sire[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg0) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_sire = ce_ca1e1b1nwzida0e0b0xyg0 #rename to keep consistent

    shape = (ce_dams.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg1.shape #get shape of the new ce array - required because assigning relsize which is diff size
    ce_ca1e1b1nwzida0e0b0xyg1 = np.zeros(shape) #make a new array - same as ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg1[...] = fun.f_expand(ce_dams, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg1[15, ...] = 1 - cp_dams[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg1) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_dams = ce_ca1e1b1nwzida0e0b0xyg1 #rename to keep consistent

    shape = (ce_offs.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg3.shape #get shape of the new ce array - required because assigning relsize which is diff size
    ce_ca1e1b1nwzida0e0b0xyg3 = np.zeros(shape) #make a new array - same as ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg3[...] = fun.f_expand(ce_offs, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg3[15, ...] = 1 - cp_offs[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg3) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_offs = ce_ca1e1b1nwzida0e0b0xyg3 #rename to keep consistent

    ##birth weight expected - includes relsize factor
    w_b_exp_a1e1b1nwzida0e0b0xyg0 = w_b_std_b0xyg0 * np.sum(ce_sire[15, ...] * agedam_propn_da0e0b0xyg0, axis = d_pos, keepdims = True)
    w_b_exp_a1e1b1nwzida0e0b0xyg1 = w_b_std_b0xyg1 * np.sum(ce_dams[15, ...] * agedam_propn_da0e0b0xyg1, axis = d_pos, keepdims = True)
    w_b_exp_a1e1b1nwzida0e0b0xyg3 = w_b_std_b0xyg3 * ce_offs[15, ...]

    ##Normal weight max (if animal is well-fed)
    nw_max_pa1e1b1nwzida0e0b0xyg0 = srw_b0xyg0 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg0) + w_b_exp_a1e1b1nwzida0e0b0xyg0 * srw_age_pa1e1b1nwzida0e0b0xyg0
    nw_max_pa1e1b1nwzida0e0b0xyg1 = srw_b0xyg1 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg1) + w_b_exp_a1e1b1nwzida0e0b0xyg1 * srw_age_pa1e1b1nwzida0e0b0xyg1
    nw_max_pa1e1b1nwzida0e0b0xyg3 = srw_b0xyg3 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg3) + w_b_exp_a1e1b1nwzida0e0b0xyg3 * srw_age_pa1e1b1nwzida0e0b0xyg3

    ##Change in normal weight max - the last period will be 0 by default but this is okay because nw hits an asymptote so change in will be 0 in the last period.
    d_nw_max_pa1e1b1nwzida0e0b0xyg0 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg0)
    d_nw_max_pa1e1b1nwzida0e0b0xyg0[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg0[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg0[0:-1, ...]) / np.maximum(1,days_period_pa1e1b1nwzida0e0b0xyg0[0:-1, ...]) #np max to stop div/0
    d_nw_max_pa1e1b1nwzida0e0b0xyg1 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg1)
    d_nw_max_pa1e1b1nwzida0e0b0xyg1[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg1[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg1[0:-1, ...]) / np.maximum(1,days_period_pa1e1b1nwzida0e0b0xyg1[0:-1, ...]) #np max to stop div/0
    d_nw_max_pa1e1b1nwzida0e0b0xyg3 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg3)
    d_nw_max_pa1e1b1nwzida0e0b0xyg3[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg3[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg3[0:-1, ...]) / np.maximum(1,days_period_cut_pa1e1b1nwzida0e0b0xyg3[0:-1, ...]) #np max to stop div/0


    #########################
    # management calc       #
    #########################
    ##scan (the specified number of days after sires are removed)
    date_scan_pa1e1b1nwzida0e0b0xyg1 = date_joined2_pa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :] \
                                       + pinp.sheep['i_scan_day'][scan_option_pa1e1b1nwzida0e0b0xyg1]
    # retain_scan_b1nwzida0e0b0xyg = np.max(pinp.sheep['i_drysretained_scan'], np.min(1, nfoet_b1nwzida0e0b0xyg))
    # retain_birth_b1nwzida0e0b0xyg = np.max(pinp.sheep['i_drysretained_birth'], np.min(1, nyatf_b1nwzida0e0b0xyg))
    ##Expected stocking density
    density_pa1e1b1nwzida0e0b0xyg0 = density_pa1e1b1nwzida0e0b0xyg
    density_pa1e1b1nwzida0e0b0xyg1 = density_pa1e1b1nwzida0e0b0xyg * density_nwzida0e0b0xyg1
    density_pa1e1b1nwzida0e0b0xyg3 = density_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p] * density_nwzida0e0b0xyg3
    ###convert density from active n to active w axis
    densityw_pa1e1b1nwzida0e0b0xyg0 = density_pa1e1b1nwzida0e0b0xyg0#^don't need following association until multiple n levels sire. np.take_along_axis(density_pa1e1b1nwzida0e0b0xyg0,a_n_pa1e1b1nwzida0e0b0xyg0,axis=n_pos)
    densityw_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(density_pa1e1b1nwzida0e0b0xyg1,a_n_pa1e1b1nwzida0e0b0xyg1,axis=n_pos)
    densityw_pa1e1b1nwzida0e0b0xyg2 = densityw_pa1e1b1nwzida0e0b0xyg1  #yes this is meant to be the same as dams
    densityw_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(density_pa1e1b1nwzida0e0b0xyg3,a_n_pa1e1b1nwzida0e0b0xyg3,axis=n_pos)



    #########################
    # period is ...         #
    #########################
    period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_prejoin_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_scan_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
    period_between_mated90_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_mated_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_d90_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
    period_between_scanprebirth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_scan_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_prebirth_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_scan_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_d90birth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_d90_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_prebirthbirth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_prebirth_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_birthwean_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_born_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg)
    period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_weaned2_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_prejoin_next_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_scan_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_scan_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_prebirth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_prebirth_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_birth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_born_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_wean_pa1e1b1nwzida0e0b0xyg2 = sfun.f1_period_is_('period_is', date_weaned_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date wean is the equivalent of date weaned g1
    previousperiod_is_wean_pa1e1b1nwzida0e0b0xyg2 = np.roll(period_is_wean_pa1e1b1nwzida0e0b0xyg2, 1, axis=0)
    period_is_wean_pa1e1b1nwzida0e0b0xyg1 = period_is_wean_pa1e1b1nwzida0e0b0xyg2
    # prev_period_is_birth_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_birth_pa1e1b1nwzida0e0b0xyg1,1,axis=p_pos)
    period_is_mating_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_mated_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_born2_pa1e1b1nwzida0e0b0xyg2-np.array([(6*7)])
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #This is within 6 weeks prior to Birth period (use dateborn2 because it increments at prejoining)
    ###shearing
    period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg0
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg3
                        , date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    period_is_startseason_pa1e1b1nwzida0e0b0xyg = sfun.f1_period_is_('period_is', date_prev_seasonstart_pa1e1b1nwzida0e0b0xyg
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg = np.roll(period_is_startseason_pa1e1b1nwzida0e0b0xyg,-1,axis=0)
    nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_startseason_pa1e1b1nwzida0e0b0xyg,-1,axis=0)[mask_p_offs_p]
    period_is_prejoin_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_prejoin_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_join_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_joined_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1

    ##This is the end of the Mating period. (no active e axis - end of mating inclusive of all e slices)
    period_is_matingend_pa1e1b1nwzida0e0b0xyg1 = np.any(np.logical_and(period_is_mating_pa1e1b1nwzida0e0b0xyg1
                                                , index_e1b1nwzida0e0b0xyg == np.max(pinp.sheep['i_join_cycles_ig1']) - 1)
                                                , axis=e1_pos,keepdims=True)
    period_isbetween_prejoinmatingend_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_prejoin_pa1e1b1nwzida0e0b0xyg1,
                                                                                   date_start_pa1e1b1nwzida0e0b0xyg,
                                                                                   np.max(date_mated_pa1e1b1nwzida0e0b0xyg1, axis=e1_pos,keepdims=True),
                                                                                   date_end_pa1e1b1nwzida0e0b0xyg)

    ##dvp
    if stubble: #if generating for stubble then dvps dont matter
            period_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = np.full_like(date_start_pa1e1b1nwzida0e0b0xyg, False)
            period_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = np.full_like(date_start_pa1e1b1nwzida0e0b0xyg, False)[mask_p_offs_p]
    else:
        period_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = np.any(sfun.f1_period_is_('period_is', dvp_start_va1e1b1nwzida0e0b0xyg1[:,na,...]
                                            , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg), axis=0)
        period_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = np.any(sfun.f1_period_is_('period_is', dvp_start_va1e1b1nwzida0e0b0xyg3[:,na,...]
                                            , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg), axis=0)[mask_p_offs_p]
        # dvp_date_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1,0)
        # nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)
        # nextperiod_is_prejoin_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_prejoin_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)

    ##################################################
    #adjust lsln management for timing of repro cycle#
    ##################################################
    ##calc lsln management association based on sheep identification options (scanning vs no scanning), management practise (differential management once identifying different groups) & time of the year (e.g. even if you scan you still need to manage sheep the same before scanning)
    ###have to create a_t array that is maximum size of the arrays that are used to mask it.
    ###t = 0 is prescan, 1 is postscan, 2 is lactation, 3 not used in V1 but would be is post wean
    shape = np.maximum.reduce([period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1.shape, period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1.shape
                                  , period_between_birthwean_pa1e1b1nwzida0e0b0xyg1.shape]) #create shape which has the max size
    a_mgt_pa1e1b1nwzida0e0b0xyg1 = np.zeros(shape)
    period_between_prejoinscan_mask = np.broadcast_arrays(a_mgt_pa1e1b1nwzida0e0b0xyg1, period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1)[1] #mask must be manually broadcasted then applied - for some reason numpy doesn't automatically broadcast them.
    period_between_scanbirth_mask = np.broadcast_arrays(a_mgt_pa1e1b1nwzida0e0b0xyg1, period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1)[1]
    period_between_birthwean_mask = np.broadcast_arrays(a_mgt_pa1e1b1nwzida0e0b0xyg1, period_between_birthwean_pa1e1b1nwzida0e0b0xyg1)[1]
    ###order matters because post wean does not have a cap ie it is overwritten by others
    a_mgt_pa1e1b1nwzida0e0b0xyg1[...] = 3 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 not used in V1 but would be is post wean
    a_mgt_pa1e1b1nwzida0e0b0xyg1[period_between_prejoinscan_mask] = 0 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean
    a_mgt_pa1e1b1nwzida0e0b0xyg1[period_between_scanbirth_mask] = 1 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean
    a_mgt_pa1e1b1nwzida0e0b0xyg1[period_between_birthwean_mask] = 2 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean

    ###dams management in each period based on scanning detail, time of the year (even if you are scanning you can't manage sheep differently before scanning) and management (you can scan and then not differentially manage)
    scan_management_pa1e1b1nwzida0e0b0xyg1 = (scan_option_pa1e1b1nwzida0e0b0xyg1) * (a_mgt_pa1e1b1nwzida0e0b0xyg1 >= 1) * pinp.sheep['i_dam_lsln_diffman_t'][1]
    gbal_management_pa1e1b1nwzida0e0b0xyg1 = (gbal_pa1e1b1nwzida0e0b0xyg1 -1) * (a_mgt_pa1e1b1nwzida0e0b0xyg1 >= 2) * pinp.sheep['i_dam_lsln_diffman_t'][2] + 1  #minus 1 then plus 1 ensures that the wean option before lactation is 1
    wean_management_pa1e1b1nwzida0e0b0xyg1 = (wean_pa1e1b1nwzida0e0b0xyg1 -1) * (a_mgt_pa1e1b1nwzida0e0b0xyg1 >= 3) * pinp.sheep['i_dam_lsln_diffman_t'][3] + 1  #minus 1 then plus 1 ensures that the wean option before weaning is 1

    ############################
    ### ewe mob size calcs     #
    ############################
    ### scale the mob size along the b axis with allowance for scanning to identify the litter size
    ### association between axis for input (based on litter size) and scanning during period from birth to weaning when mob size is adjusted
    a_l_pa1e1b1nwzida0e0b0xyg1 = np.minimum(nfoet_b1nwzida0e0b0xyg, scan_management_pa1e1b1nwzida0e0b0xyg1) * period_between_birthwean_pa1e1b1nwzida0e0b0xyg1
    ### select the mob size scalar on the b1 axis
    mobsize_scalar_b1 = fun.f_expand(uinp.sheep['i_mobsize_scalar_l0'], b1_pos, right_pos=0, left_pos2=p_pos-1, right_pos2=b1_pos)
    mobsize_scalar_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(mobsize_scalar_b1, a_l_pa1e1b1nwzida0e0b0xyg1, b1_pos)
    ### adjust scalar for the expected proportion of each litter size so that the resulting average mob size == the input
    #### This calculation means that this input can't be used to fudge the average mobsize.
    #### A constant average mob size means that the mob based husbandry cost does not change.
    mobsize_scalar_pa1e1b1nwzida0e0b0xyg1 = mobsize_scalar_pa1e1b1nwzida0e0b0xyg1 * np.sum(lsln_propn_b1nwzida0e0b0xyg1
                                                    / mobsize_scalar_pa1e1b1nwzida0e0b0xyg1, axis=b1_pos, keepdims=True)
    ### scale the dam mob size
    mobsize_pa1e1b1nwzida0e0b0xyg1 = mobsize_pa1e1b1nwzida0e0b0xyg1 * mobsize_scalar_pa1e1b1nwzida0e0b0xyg1

    ############################
    ### feed supply calcs      #
    ############################
    legume_p6a1e1b1nwzida0e0b0xyg,bool_confinement_g0_n,bool_confinement_g1_n,bool_confinement_g3_n, \
    nv_p6a1e1b1j1wzida0e0b0xyg0,foo_p6a1e1b1j1wzida0e0b0xyg0,dmd_p6a1e1b1j1wzida0e0b0xyg0,supp_p6a1e1b1j1wzida0e0b0xyg0, \
    nv_p6a1e1b1j1wzida0e0b0xyg1,foo_p6a1e1b1j1wzida0e0b0xyg1,dmd_p6a1e1b1j1wzida0e0b0xyg1,supp_p6a1e1b1j1wzida0e0b0xyg1, \
    nv_p6a1e1b1j1wzida0e0b0xyg3,foo_p6a1e1b1j1wzida0e0b0xyg3,dmd_p6a1e1b1j1wzida0e0b0xyg3,supp_p6a1e1b1j1wzida0e0b0xyg3, \
    feedsupplyw_tpa1e1b1nwzida0e0b0xyg0,feedsupplyw_tpa1e1b1nwzida0e0b0xyg1,feedsupplyw_tpa1e1b1nwzida0e0b0xyg3, \
    confinementw_tpa1e1b1nwzida0e0b0xyg0,confinementw_tpa1e1b1nwzida0e0b0xyg1,confinementw_tpa1e1b1nwzida0e0b0xyg3\
      = fsstk.f1_stock_fs(cr_sire,cr_dams,cr_offs,cu0_sire,cu0_dams,cu0_offs,a_p6_pa1e1b1nwzida0e0b0xyg
                          , period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1, scan_management_pa1e1b1nwzida0e0b0xyg1
                          , gbal_management_pa1e1b1nwzida0e0b0xyg1, wean_management_pa1e1b1nwzida0e0b0xyg1
                          , a_n_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg3, a_t_tpg1, mask_p_offs_p, len_p, pkl_fs_info, pkl_fs)

    '''if running the gen for stubble generation then the feed supply info above gets overwritten with
    the stubble feed from the trial.'''
    if stubble:
        foo_dams = pinp.stubble['i_foo']
        foo_yatf = pinp.stubble['i_foo']
        foo_offs = pinp.stubble['i_foo']
        dmd_pwg = fun.f_expand(stubble['dmd_pw'],w_pos, left_pos2=p_pos, right_pos2=w_pos)
        intake_s_dams = pinp.stubble['i_intake_s']
        intake_s_yatf = pinp.stubble['i_intake_s']
        intake_s_offs = pinp.stubble['i_intake_s']
        confinementw_tpa1e1b1nwzida0e0b0xyg1[...] = False
        confinementw_tpa1e1b1nwzida0e0b0xyg3[...] = False

    #######################
    #start generator loops#
    #######################
    '''
    See google doc for more LTW info. 
    '''
    ##Start the LTW loop here so that the arrays are reinitialised from the inputs
    ### set the LTW adjustments to zero for the first loop. Sires do not have a LTW adjust because they are born off farm
    sfw_ltwadj_g0 = 1
    sfd_ltwadj_g0 = 0
    sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 = np.zeros(pg1)
    sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 = np.zeros(pg1)
    sfw_ltwadj_g2 = 1
    sfd_ltwadj_g2 = 0
    sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 = np.zeros(pg3)[0:1, ...]  # slice the p axis to convert to singleton
    sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 = np.zeros(pg3)[0:1, ...]  # slice the p axis to convert to singleton

    ## 2 LTW loops unless either:
    ###     a. the feedsupply comes from pickle and pkl_ltwadj can be broadcast (loop_ltw_len=1).
    ###     b. the LTW adjustment for dams & offs are both set to 0 (loop_ltw_len=1)
    ### Note: The resulting number determined from the above steps can be increased by SAV if extra precision is required.
    loop_ltw_len = 2

    ##If using feedsupply from pkl, read in LTW adjustment from pkl.
    fs_use_number = sinp.structuralsa['i_fs_use_number']
    if sinp.structuralsa['i_fs_use_pkl']:
        ###update ltwadj with ltwadj from pkl
        pkl_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 = pkl_fs['ltw_adj']['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1']
        pkl_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 = pkl_fs['ltw_adj']['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1']
        pkl_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 = pkl_fs['ltw_adj']['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3']
        pkl_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 = pkl_fs['ltw_adj']['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3']
        try:   #broadcast the ltwadj from pkl to the current feedsupply shape
            sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[...] = pkl_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1
            sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[...] = pkl_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1
            sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3[...] = pkl_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3
            sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3[...] = pkl_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3
            loop_ltw_len = 1   #set number of loops to 1 if the feedsupply comes from pickle and the ltwadj could be broadcast
        except ValueError: #could not broadcast the ltwadj array from shape x into shape y so carry out default ltw loops
            pass
            # loop_ltw_len = max(loop_ltw_len, 2)

    ##Turn off ltw loop if:
        ## If both dams & offs are not used (i.e. LTW_? == 0) then don't loop.
        ## If generating for stubble then don't loop because only running a few periods so no lifetime information is known.
    if (sen.sam['LTW_dams'] == 0 and sen.sam['LTW_offs'] == 0) or stubble:
        loop_ltw_len = 1

    ##increment the number of loops. This may be specified in the SAV to finetune the LTW effect.
    loop_ltw_len = loop_ltw_len + fun.f_sa(0, sen.sav['LTW_loops_increment'], 5)

    for loop_ltw in range(loop_ltw_len):
        #todo The double loop could be replaced by separating the offspring into their own loop
        # it doesn't remove the requirement to loop for the dams because they need to have the first loop to generate the inputs for the second loop
        # but it would reduce the number of offspring calculations, allow offspring wean wt to be based on ffcfw_yatf at weaning and allow loop length to be customised
        # the drawback of a separate loop is that the structure of the function calls would need to be repeated.
        # an alternative would be to replace "if days_period_g3[p] > 0" with another variable that is defined at the start, like 'calculate_this_period_pg3'
        # calculate_this_period_pg3 = np.logical_and(np.any(days_period_g3[p]>0), loop_ltw = sen.sav['LTW_loops_increment'] # only calculate the progeny & sires in the final LTW loop

        ####################################
        ### initialise arrays for sim loop  # axis names not always track from now on because they change between p=0 and p=1
        ####################################
        ##apply the LTW adjustment sensitivity - only plus 1 for sfw
        ## apply at the top of the loop so that the values that get pickled don't include the sam (incase sam changes between trials)
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 = 1 + sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 * sen.sam['LTW_dams']
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 * sen.sam['LTW_dams']
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 = 1 + sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 * sen.sam['LTW_offs']
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 * sen.sam['LTW_offs']

        ##all groups
        eqn_compare = uinp.sheep['i_eqn_compare']
        ##sire
        ffcfw_start_sire = ffcfw_initial_wzida0e0b0xyg0
        ffcfw_max_start_sire = ffcfw_start_sire
        omer_history_start_p3g0[...] = np.nan
        d_cfw_history_start_p2g0[...] = np.nan
        cfw_start_sire = cfw_initial_wzida0e0b0xyg0
        fd_start_sire = fd_initial_wzida0e0b0xyg0
        fl_start_sire = fl_initial_wzida0e0b0xyg0
        fd_min_start_sire = fd_initial_wzida0e0b0xyg0
        aw_start_sire = aw_initial_wzida0e0b0xyg0
        mw_start_sire = mw_initial_wzida0e0b0xyg0
        bw_start_sire = bw_initial_wzida0e0b0xyg0
        nw_start_sire = 0 #no dimensions to start
        temp_lc_sire = np.array([0]) #this is calculated in the chill function, but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_sire = numbers_initial_zida0e0b0xyg0
        numbers_start_condense_sire = numbers_initial_zida0e0b0xyg0 #just need a default because this is processed using update function.
        # ebg_start_sire=0

        ##dams
        numbers_join_dams = 0
        ldr_start_dams = np.array([1.0])
        lb_start_dams = np.array([1.0])
        w_f_start_dams = np.array([0.0])
        nw_f_start_dams = np.array([0.0])
        nec_cum_start_dams = np.array([0.0])
        cfw_ltwadj_start_dams = np.array([0.0])
        fd_ltwadj_start_dams = np.array([0.0])
        cf_cfwltw_start_dams = np.array([0.0])
        cf_cfwltw_dams = np.zeros(tag1, dtype =dtype) #this is required as default when mu fleece function is not being called (it is required in the start production function)
        cf_fdltw_start_dams = np.array([0.0])
        cf_fdltw_dams = np.zeros(tag1, dtype =dtype) #this is required as default when mu fleece function is not being called (it is required in the start production function)
        cf_w_b_start_dams = np.array([0.0])
        cf_w_b_dams = np.zeros(tag1, dtype =dtype) #this is required as default when mu birth weight function is not being called (it is required in the start production function)
        cf_w_w_start_dams = np.array([0.0])
        cf_w_w_dams = np.zeros(tag1, dtype =dtype) #this is required as default when mu wean function is not being called (it is required in the start production function)
        cf_csc_start_dams = np.array([0.0])
        cf_csc_dams = np.zeros(tag1, dtype =dtype) #this is required as default when mu2 peri-natal mortality function is not being called (it is required in the start production function)
        # cf_conception_start_dams = np.array([0.0])
        # cf_conception_dams = np.zeros(tag1, dtype =dtype) #not currently used. Will be used if profile prior to joining (i.e. previous year) is included in the repro functions.
        guw_start_dams = np.array([0.0])
        rc_birth_start_dams = np.array([1.0])
        ffcfw_start_dams = fun.f_expand(ffcfw_initial_wzida0e0b0xyg1, p_pos, right_pos=w_pos) #add axis w to a1 because e and b axis are sliced before they are added via calculation
        ffcfw_max_start_dams = ffcfw_start_dams
        ffcfw_mating_dams = 0.0
        lwc_mating_dams = 0.0
        omer_history_start_p3g1[...] = np.nan
        d_cfw_history_start_p2g1[...] = np.nan
        cfw_start_dams = cfw_initial_wzida0e0b0xyg1
        fd_start_dams = fd_initial_wzida0e0b0xyg1
        fl_start_dams = fl_initial_wzida0e0b0xyg1
        fd_min_start_dams = fd_initial_wzida0e0b0xyg1
        aw_start_dams = aw_initial_wzida0e0b0xyg1
        mw_start_dams = mw_initial_wzida0e0b0xyg1
        bw_start_dams = bw_initial_wzida0e0b0xyg1
        nw_start_dams = np.array([0.0])
        temp_lc_dams = np.array([0.0]) #this is calculated in the chill function, but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_dams = numbers_initial_a1e1b1nwzida0e0b0xyg1
        numbers_start_condense_dams = numbers_initial_a1e1b1nwzida0e0b0xyg1 #just need a default because this is processed using update function.
        scanning = 0 #variable is used only for reporting
        # ebg_start_dams=0
        o_mortality_dams[...] = 0 #have to reset when doing the ltw loop because it is used to back date numbers
        dm_dams = np.array([0.0]) #passed as an argument to f_foetus_nfs() so needs to be defined prior to first assignment
        m_start_dams = mw_start_dams * cg_dams[27, ...] * cg_dams[21, ...]
        c_start_dams = np.array([0.0]) #passed as an argument to f_foetus_nfs() so needs to be defined prior to first assignment

        ##yatf
        omer_history_start_p3g2[...] = np.nan
        d_cfw_history_start_p2g2[...] = np.nan
        nw_start_yatf = 0.0
        rc_start_yatf = 0.0
        ffcfw_start_yatf = w_b_std_y_b1nwzida0e0b0xyg1 #this is just an estimate, it is updated with the real weight at birth - needed to calc milk production in birth period because milk prod is calculated before yatf weight is updated
        ffcfw_max_start_yatf = ffcfw_start_yatf
        mortality_birth_yatf=0.0 #required for dam numbers before progeny born
        cfw_start_yatf = 0.0
        temp_lc_yatf = np.array([0.0]) #this is calculated in the chill function, but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_yatf = nyatf_b1nwzida0e0b0xyg * gender_propn_xyg   # nyatf is accounting for peri-natal mortality. But doesn't include the differential mortality of female and male offspring at birth
        numbers_start_condense_yatf = numbers_start_yatf #just need a default because this is processed using update function.
        numbers_end_yatf = 0.0 #need a default because this is required in f_start[p+1] prior to being assigned.
        # ebg_start_yatf=0
        ebg_yatf = 0.0 #need a default because used in call to WWt of yatf
        fl_start_yatf=fl_birth_yg2 #can't use fl_initial because that is at weaning
        fd_start_yatf=0.0
        fd_min_start_yatf = 1000.0
        w_b_start_yatf = 0.0
        w_b_ltw_std_yatf = 0.0
        w_w_start_yatf = 0.0
        foo_lact_ave_start = 0.0
        foo_lact_ave = np.zeros(tag1, dtype =dtype) #required because only calculated if using mu function
        aw_start_yatf = 0.0
        bw_start_yatf = 0.0
        mw_start_yatf = 0.0

        ##offs
        ffcfw_start_offs = ffcfw_initial_wzida0e0b0xyg3
        ffcfw_max_start_offs = ffcfw_start_offs
        omer_history_start_p3g3[...] = np.nan
        d_cfw_history_start_p2g3[...] = np.nan
        cfw_start_offs = cfw_initial_wzida0e0b0xyg3
        fd_start_offs = fd_initial_wzida0e0b0xyg3
        fl_start_offs = fl_initial_wzida0e0b0xyg3
        fd_min_start_offs = fd_initial_wzida0e0b0xyg3
        aw_start_offs = aw_initial_wzida0e0b0xyg3
        mw_start_offs = mw_initial_wzida0e0b0xyg3
        bw_start_offs = bw_initial_wzida0e0b0xyg3
        nw_start_offs = 0.0
        temp_lc_offs = np.array([0.0]) #this is calculated in the chill function, but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_offs = numbers_initial_ida0e0b0xyg3
        numbers_start_condense_offs = numbers_initial_ida0e0b0xyg3 #just need a default because this is processed using update function.

        '''if generating for stubble then overwrite some initial params to align with paddock trial.
           Ffcfw and other initial values are overwritten above'''
        if stubble:
            ##dams
            w_f_start_dams = pinp.stubble['w_foetus_start']
            nw_f_start_dams = w_f_start_dams

            ##yatf
            ffcfw_start_yatf = np.array([pinp.stubble['i_lw_yatf'] - pinp.stubble['i_gfw_yatf']]) #have to make it an array so it can handle new axis.
            ffcfw_max_start_yatf = ffcfw_start_yatf
            nw_start_yatf = ffcfw_start_yatf
            rc_start_yatf = 1
            cfw_start_yatf = pinp.stubble['i_gfw_yatf'] * cw_yatf[3, ...]
            fl_start_yatf = pinp.stubble['i_fl_yatf']
            fd_start_yatf = pinp.stubble['i_fd_yatf'] #not used for anything so just use the same one as adult
            foo_lact_ave_start = pinp.stubble['i_foo']
            aw_start_yatf = ffcfw_start_yatf * pinp.stubble['i_aw_yatf']
            bw_start_yatf = ffcfw_start_yatf * pinp.stubble['i_bw_yatf']
            mw_start_yatf = ffcfw_start_yatf * pinp.stubble['i_mw_yatf']

            ##offs
            nw_start_offs = 0.0
            temp_lc_offs = np.array([0.0])  # this is calculated in the chill function, but it is required for the intake function so it is set to 0 for the first period.

        ######################
        ### sim engine       #
        ######################
        ##load in or create REV dict if doing a relative economic value analysis
        rev_number = sinp.structuralsa['i_rev_number']
        if sinp.structuralsa['i_rev_create'] or not np.any(sinp.structuralsa['i_rev_trait_inc']): #if rev is not being used an empty dict is still required.
            rev_trait_values = collections.defaultdict(dict)
            for p in range(n_sim_periods - 1):
                rev_trait_values['sire'][p] = {}
                rev_trait_values['dams'][p] = {}
                rev_trait_values['yatf'][p] = {}
                rev_trait_values['offs'][p] = {}
        elif np.any(sinp.structuralsa['i_rev_trait_inc']):
            print('REV values being used.')
            with open('pkl/pkl_rev_trait{0}.pkl'.format(rev_number),"rb") as f:
                rev_trait_values = pkl.load(f)

        ##load in and create condensed start dict - used to standardise the starting animal at condensing time.
        ###load condensed start info from previous trial if being used in this trial.
        if sinp.structuralsa['i_use_pkl_condensed_start_condition']:
            pkl_condensed_values = pkl_fs['pkl_condensed_values']
        ###create empty to store condensed start info for current trial - this is only stored at the end if fs is stored.
        else:
            pkl_condensed_values = collections.defaultdict(dict)
            for p in range(n_sim_periods - 1):
                pkl_condensed_values['sire'][p] = {}
                pkl_condensed_values['dams'][p] = {}
                pkl_condensed_values['yatf'][p] = {}
                pkl_condensed_values['offs'][p] = {}

        ## Loop through each week of the simulation (p)
        ###if generating for stubble then only some periods need to be run.
        if stubble:
            p_start = stubble['p_start']
            p_end = stubble['p_end']
        else:
            p_start = 0
            p_end = n_sim_periods-1   #-1 because assigns to [p+1] for start values
        for p in range(p_start, p_end):   #-1 because assigns to [p+1] for start values
            # print(p)
            # if np.any(period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]):
            #     print("period is lactation: ", period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
            # if np.any(period_is_wean_pa1e1b1nwzida0e0b0xyg1[p]):
            #     print("period is weaning: ", period_is_wean_pa1e1b1nwzida0e0b0xyg1[p])
            # if np.any(period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]):
            #     print("period is gest: ", period_is_mating_pa1e1b1nwzida0e0b0xyg1[p])
            # if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg1[p]):
            #     print("period is fvp0 dams: ", period_is_condense_pa1e1b1nwzida0e0b0xyg1[p])
            # if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg1[p]):
            #     print("period is fvp0 offs: ", period_is_condense_pa1e1b1nwzida0e0b0xyg1[p])



            ###################################################################################
            #reset variables if period is beginning of a reproduction cycle or shearing cycle #
            ###################################################################################
            ##sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###cfw
                cfw_start_sire = fun.f_update(cfw_start_sire, 0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_sire = fun.f_update(fl_start_sire, fl_shear_yg0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_sire = fun.f_update(fd_min_start_sire, 1000, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)

            ##dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
            ##Reproduction
                ###Lagged DR (lactation deficit)
                ldr_start_dams = fun.f_update(ldr_start_dams, 1, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Loss of potential milk due to consistent under production
                lb_start_dams = fun.f_update(lb_start_dams, 1, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Weight of foetus (start)
                w_f_start_dams = fun.f_update(w_f_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Weight of gravid uterus (start)
                guw_start_dams = fun.f_update(guw_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Normal weight of foetus (start)
                nw_f_start_dams = fun.f_update(nw_f_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Birth weight carryover (running tally of foetal weight diff)
                cf_w_b_start_dams = fun.f_update(cf_w_b_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###Weaning weight carryover (running tally of foetal weight diff)
                cf_w_w_start_dams = fun.f_update(cf_w_w_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###CS change carryover (running tally of dam CS change in late pregnancy)
                cf_csc_start_dams = fun.f_update(cf_csc_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                # ###Carry forward conception
                # cf_conception_start_dams = fun.f_update(cf_conception_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###LTW CFW adjustment carryover (running tally of LTW progeny CFW)
                cfw_ltwadj_start_dams = fun.f_update(cfw_ltwadj_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                cf_cfwltw_start_dams = fun.f_update(cf_cfwltw_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###LTW FD adjustment carryover (running tally of LTW progeny FD)
                fd_ltwadj_start_dams = fun.f_update(fd_ltwadj_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                cf_fdltw_start_dams = fun.f_update(cf_fdltw_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])

            ##Wool Production
                ###cfw
                cfw_start_dams = fun.f_update(cfw_start_dams, 0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_dams = fun.f_update(fl_start_dams, fl_shear_yg1, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_dams = fun.f_update(fd_min_start_dams, 1000, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)

            ##offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###cfw
                cfw_start_offs = fun.f_update(cfw_start_offs, 0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_offs = fun.f_update(fl_start_offs, fl_shear_yg3, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_offs = fun.f_update(fd_min_start_offs, 1000, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)



            ###################
            ##dependent start #
            ###################
            ##sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p, ...] > 0):
                ###GFW (start)
                gfw_start_sire = cfw_start_sire / cw_sire[3, ...]
                ###LW (start -with fleece & conceptus)
                lw_start_sire = ffcfw_start_sire + gfw_start_sire
                ###Normal weight (start)
                nw_start_sire = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg0[p], np.maximum(nw_start_sire, ffcfw_start_sire + cn_sire[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg0[p]  - ffcfw_start_sire)))
                ###Relative condition (start)
                rc_start_sire = ffcfw_start_sire / nw_start_sire
                ##Condition score at  start of p
                cs_start_sire = sfun.f1_condition_score(rc_start_sire, cn_sire)
                ###staple length
                sl_start_sire = fl_start_sire * cw_sire[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_sire = np.minimum(1, nw_start_sire / srw_b0xyg0)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_sire = np.minimum(ffcfw_max_start_sire, nw_max_pa1e1b1nwzida0e0b0xyg0[p]) / srw_b0xyg0
                ###PI Size factor (for cattle)
                zf_sire = np.maximum(1, 1 + cr_sire[7, ...] - relsize_start_sire)
                ###EVG Size factor (decreases as z increases)
                ####Note: This equation purposefully has the opposite sign for cg[4] to Freer et al 2012
                ####There is an error in the documentation and this representation is consistent with Sheep Explorer.
                zf1_sire = fun.f_back_transform(-cg_sire[4, ...] * (relsize1_start_sire - cg_sire[5, ...]))
                ###EVG Size factor (increases at maturity)
                zf2_sire = np.clip((relsize1_start_sire - cg_sire[6, ...]) / (cg_sire[7, ...] - cg_sire[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_kg_sire = fun.f_update(1, sen.sam['kg_adult'], zf2_sire == 1)
                sam_mr_sire = fun.f_update(1, sen.sam['mr_adult'], zf2_sire == 1)
                sam_pi_sire = fun.f_update(1, sen.sam['pi_adult'], zf2_sire == 1)

            ##dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                ###GFW (start)
                gfw_start_dams = cfw_start_dams / cw_dams[3, ...]
                ###LW (start -with fleece & conceptus)
                lw_start_dams = ffcfw_start_dams + guw_start_dams + gfw_start_dams
                ###Normal weight (start)
                nw_start_dams = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg1[p], np.maximum(nw_start_dams, ffcfw_start_dams + cn_dams[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg1[p]  - ffcfw_start_dams)))
                ###Relative condition (start)
                rc_start_dams = ffcfw_start_dams / nw_start_dams
                ##Condition score of the dam at  start of p
                cs_start_dams = sfun.f1_condition_score(rc_start_dams, cn_dams)
                ###Relative condition of dam at parturition - needs to be remembered between loops (milk production) - Loss of potential milk due to consistent under production
                rc_birth_dams = fun.f_update(rc_birth_start_dams, rc_start_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                ###staple length
                sl_start_dams = fl_start_dams * cw_dams[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_dams = np.minimum(1, nw_start_dams / srw_b0xyg1)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_dams = np.minimum(ffcfw_max_start_dams, nw_max_pa1e1b1nwzida0e0b0xyg1[p]) / srw_b0xyg1
                ###PI Size factor (for cattle)
                zf_dams = np.maximum(1, 1 + cr_dams[7, ...] - relsize_start_dams)
                ###EVG Size factor (decreases as z increases)
                ####Note: This equation purposefully has the opposite sign for cg[4] to Freer et al 2012
                ####There is an error in the documentation and this representation is consistent with Sheep Explorer.
                zf1_dams = fun.f_back_transform(-cg_dams[4, ...] * (relsize1_start_dams - cg_dams[5, ...]))
                ###EVG Size factor (increases at maturity)
                zf2_dams = np.clip((relsize1_start_dams - cg_dams[6, ...]) / (cg_dams[7, ...] - cg_dams[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_kg_dams = fun.f_update(1, sen.sam['kg_adult'], zf2_dams == 1)
                sam_mr_dams = fun.f_update(1, sen.sam['mr_adult'], zf2_dams == 1)
                sam_pi_dams = fun.f_update(1, sen.sam['pi_adult'], zf2_dams == 1)
                ##sires for mating
                n_sire_a1e1b1nwzida0e0b0xyg1g0p8 = sfun.f_sire_req(sire_propn_pa1e1b1nwzida0e0b0xyg1g0[p], sire_periods_g0p8, pinp.sheep['i_sire_recovery']
                                                                   , date_end_pa1e1b1nwzida0e0b0xyg[p], period_is_join_pa1e1b1nwzida0e0b0xyg1[p])

            ##yatf
            ##note: most yatf calculated later in the code (except for ffcfw from bw)
            ###Set FFCFW to the expected birth weight if period is birth
            ### Required because bw is not calculated until after milk production is calculated
            if np.any(period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                ffcfw_start_yatf = fun.f_update(ffcfw_start_yatf, w_b_std_y_b1nwzida0e0b0xyg1
                                                , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ffcfw_max_start_yatf = fun.f_update(ffcfw_max_start_yatf, w_b_std_y_b1nwzida0e0b0xyg1
                                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])

            ##offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] > 0):
                ###GFW (start)
                gfw_start_offs = cfw_start_offs / cw_offs[3, ...]
                ###LW (start -with fleece & conceptus)
                lw_start_offs = ffcfw_start_offs + gfw_start_offs
                ###Normal weight (start)
                nw_start_offs = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg3[p], np.maximum(nw_start_offs, ffcfw_start_offs + cn_offs[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg3[p]  - ffcfw_start_offs)))
                ###Relative condition (start)
                rc_start_offs = ffcfw_start_offs / nw_start_offs
                ##Condition score at  start of p
                cs_start_offs = sfun.f1_condition_score(rc_start_offs, cn_offs)
                ###staple length
                sl_start_offs = fl_start_offs * cw_offs[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_offs = np.minimum(1, nw_start_offs / srw_b0xyg3)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_offs = np.minimum(ffcfw_max_start_offs, nw_max_pa1e1b1nwzida0e0b0xyg3[p]) / srw_b0xyg3
                ###PI Size factor (for cattle)
                zf_offs = np.maximum(1, 1 + cr_offs[7, ...] - relsize_start_offs)
                ###EVG Size factor (decreases as z increases)
                ####Note: This equation purposefully has the opposite sign for cg[4] to Freer et al 2012
                ####There is an error in the documentation and this representation is consistent with Sheep Explorer.
                zf1_offs = fun.f_back_transform(-cg_offs[4, ...] * (relsize1_start_offs - cg_offs[5, ...]))
                ###EVG Size factor (increases at maturity)
                zf2_offs = np.clip((relsize1_start_offs - cg_offs[6, ...]) / (cg_offs[7, ...] - cg_offs[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_kg_offs = fun.f_update(1, sen.sam['kg_adult'], zf2_offs == 1)
                sam_mr_offs = fun.f_update(1, sen.sam['mr_adult'], zf2_offs == 1)
                sam_pi_offs = fun.f_update(1, sen.sam['pi_adult'], zf2_offs == 1)



            ##feed supply loop
            ##this loop is only required if a LW target is specified for the animals
            ##if there is a target then the loop needs to continue until
            ##the feed supply has converged on a value that generates a liveweight
            ##change close to the target
            ##The loop needs to execute at least once, then repeat if there
            ##is a target and the result is not close enough to the target

            ###initial info
            ####set lw target - None is default which means lw target is not used.
            if not 'target_lwc_dams' in locals(): #only required once.
                target_lwc_dams = fun.f_sa(np.full(len_p, 9999.), sen.sav['target_lwc_dams_P'][0:len_p], 5)
                target_lwc_offs = fun.f_sa(np.full(len_p, 9999.), sen.sav['target_lwc_offs_P'][0:len_p], 5)
                epsilon = 0.050 #within 50g of target
                n_max_itn = sinp.stock['i_feedsupply_itn_max']
            shape = feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:, 0].shape + (n_max_itn,) + (2,) #slice 0 to remove the p axis
            attempts_dams = np.full(shape, np.nan) #initialise
            shape = feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:, 0].shape + (n_max_itn,) + (2,) #slice 0 to remove the p axis
            attempts_offs = np.full(shape, np.nan) #initialise

            for itn in range(n_max_itn):
                ##potential intake
                eqn_group = 4
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0 = sfun.f_potential_intake_cs(ci_sire, cl_sire, srw_b0xyg0, relsize_start_sire, rc_start_sire, temp_lc_sire
                                                           , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                           , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg0[p]
                                                           , sam_pi = sam_pi_sire)
                        if eqn_used:
                            pi_sire = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0 = sfun.f_potential_intake_cs(ci_dams, cl_dams, srw_b0xyg1, relsize_start_dams, rc_start_dams, temp_lc_dams
                                                           , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                           , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg1[p]
                                                           , rc_birth_start = rc_birth_dams, pi_age_y = pi_age_y_pa1e1b1nwzida0e0b0xyg1[p]
                                                           , lb_start = lb_start_dams, sam_pi = sam_pi_dams)
                        if eqn_used:
                            pi_dams = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0 = sfun.f_potential_intake_cs(ci_offs, cl_offs, srw_b0xyg3, relsize_start_offs, rc_start_offs, temp_lc_offs
                                                           , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                           , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg3[p]
                                                           , sam_pi = sam_pi_offs)
                        if eqn_used:
                            pi_offs = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ###murdoch #todo function doesn't exist yet, add args when it is built
                eqn_system = 1 # mu = 1
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_b0xyg0)
                        if eqn_used:
                            pi_sire = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_b0xyg1)
                        if eqn_used:
                            pi_dams = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_b0xyg3)
                        if eqn_used:
                            pi_offs = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ##feedsupply - calculated after pi because pi required for intake_s
                ## feedsupply is calculated a bit different when generating for stubble
                a_p6_cut_pa1e1b1nwzida0e0b0xyg = a_p6_pa1e1b1nwzida0e0b0xyg[p:p+1]  # the slice of p6 for the current generator period. Has active z axis so need to use expanded version.
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    nv_a1e1b1j1wzida0e0b0xyg0 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg0, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    foo_a1e1b1j1wzida0e0b0xyg0 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg0, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    dmd_a1e1b1j1wzida0e0b0xyg0 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg0, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    supp_a1e1b1j1wzida0e0b0xyg0 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg0, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    mei_sire, foo_sire, dmd_sire, mei_solid_sire, md_solid_sire, md_herb_sire, intake_f_sire, intake_s_sire, mei_propn_milk_sire, mei_propn_supp_sire, mei_propn_herb_sire   \
                        = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg0[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg0[:,p]
                                            , nv_a1e1b1j1wzida0e0b0xyg0, foo_a1e1b1j1wzida0e0b0xyg0
                                            , dmd_a1e1b1j1wzida0e0b0xyg0, supp_a1e1b1j1wzida0e0b0xyg0, pi_sire)
                    
                if not stubble:
                    if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        nv_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        foo_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        dmd_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        supp_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        mei_dams, foo_dams, dmd_dams, mei_solid_dams, md_solid_dams, md_herb_dams, intake_f_dams, intake_s_dams, mei_propn_milk_dams, mei_propn_supp_dams, mei_propn_herb_dams   \
                            = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                                , nv_a1e1b1j1wzida0e0b0xyg1, foo_a1e1b1j1wzida0e0b0xyg1
                                                , dmd_a1e1b1j1wzida0e0b0xyg1, supp_a1e1b1j1wzida0e0b0xyg1, pi_dams)
                ###if generating for stubble then nv doesn't exist so need to calc a bit differently.
                else:
                    ###calc dmd and md_herb - done within if statement because dmd & md_herb are calculated differently for stubble sim.
                    dmd_dams = dmd_pwg[p]
                    md_herb_dams = fsfun.f1_dmd_to_md(dmd_dams)
                    
                    ###relative ingestibility (quality) - dams
                    eqn_group = 6
                    eqn_system = 0  # CSIRO = 0
                    if uinp.sheep['i_eqn_exists_q0q1'][
                        eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                        eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                            temp0 = fsfun.f_rq_cs(dmd_dams, legume_pa1e1b1nwzida0e0b0xyg[p], cr_dams, pinp.sheep['i_sf'])
                            if eqn_used:
                                rq_dams = temp0
                            if eqn_compare:
                                r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    ###intake - dams
                    if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                        ri_dams = fsfun.f_rel_intake(1, rq_dams, legume_pa1e1b1nwzida0e0b0xyg[p], cr_dams)  # use ra=1 for stubble
                        mei_dams, mei_solid_dams, intake_f_dams, md_solid_dams, mei_propn_milk_dams, mei_propn_herb_dams, mei_propn_supp_dams \
                            = sfun.f_intake(pi_dams, ri_dams, md_herb_dams, False, intake_s_dams, pinp.sheep['i_md_supp'])
                        
                if not stubble:
                    if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        nv_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        foo_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        dmd_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        supp_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                        mei_offs, foo_offs, dmd_offs, mei_solid_offs, md_solid_offs, md_herb_offs, intake_f_offs, intake_s_offs, mei_propn_milk_offs, mei_propn_supp_offs, mei_propn_herb_offs   \
                            = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg3[:,p]
                                                , nv_a1e1b1j1wzida0e0b0xyg3, foo_a1e1b1j1wzida0e0b0xyg3
                                                , dmd_a1e1b1j1wzida0e0b0xyg3, supp_a1e1b1j1wzida0e0b0xyg3, pi_offs)
                ###if generating for stubble then nv doesn't exist so need to calc a bit differently.
                else:
                    ###calc dmd and md_herb - done within if statement because dmd & md_herb are calculated differently for stubble sim.
                    dmd_offs = dmd_pwg[p]
                    md_herb_offs = fsfun.f1_dmd_to_md(dmd_offs)
                    ###relative ingestibility (quality) - offs
                    eqn_group = 6
                    eqn_system = 0  # CSIRO = 0
                    if uinp.sheep['i_eqn_exists_q0q1'][
                        eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                        eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] > 0):
                            temp0 = fsfun.f_rq_cs(dmd_offs, legume_pa1e1b1nwzida0e0b0xyg[p], cr_offs, pinp.sheep['i_sf'])
                            if eqn_used:
                                rq_offs = temp0
                            if eqn_compare:
                                r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    ###intake - offs
                    if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] > 0):
                        ri_offs = fsfun.f_rel_intake(1, rq_offs, legume_pa1e1b1nwzida0e0b0xyg[p], cr_offs)  # use ra=1 for stubble
                        mei_offs, mei_solid_offs, intake_f_offs, md_solid_offs, mei_propn_milk_offs, mei_propn_herb_offs, mei_propn_supp_offs \
                            = sfun.f_intake(pi_offs, ri_offs, md_herb_offs, False, intake_s_offs, pinp.sheep['i_md_supp'])

                ##energy
                eqn_group = 7
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3 = sfun.f1_efficiency(ck_sire, md_solid_sire, pinp.sheep['i_md_supp']
                                                                    , md_herb_sire, lgf_eff_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                                                    , dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], sam_kg = sam_kg_sire)
                        if eqn_used:
                            km_sire = temp0
                            kg_fodd_sire = temp1
                            kg_supp_sire = temp2  #temp3 is not used for sires
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        temp0, temp1 = sfun.f_energy_cs(cx_sire[:,0:1,...], cm_sire, lw_start_sire, ffcfw_start_sire
                                                        , mr_age_pa1e1b1nwzida0e0b0xyg0[p], mei_sire, omer_history_start_p3g0
                                                        , days_period_pa1e1b1nwzida0e0b0xyg0[p], km_sire, pinp.sheep['i_steepness']
                                                        , densityw_pa1e1b1nwzida0e0b0xyg0[p], foo_sire, confinementw_tpa1e1b1nwzida0e0b0xyg0[:,p]
                                                        , intake_f_sire, dmd_sire, sam_mr = sam_mr_sire)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_sire = temp1
                        if eqn_used:
                            meme_sire = temp0
                            hp_maint_sire = meme_sire
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3 = sfun.f1_efficiency(ck_dams, md_solid_dams, pinp.sheep['i_md_supp']
                                                                      , md_herb_dams, lgf_eff_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                                                      , dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...], sam_kg=sam_kg_dams)
                        if eqn_used:
                            km_dams = temp0
                            kg_fodd_dams = temp1
                            kg_supp_dams = temp2
                            kl_dams = temp3
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        temp0, temp1 = sfun.f_energy_cs(cx_dams[:,1:2,...], cm_dams, lw_start_dams, ffcfw_start_dams
                                                        , mr_age_pa1e1b1nwzida0e0b0xyg1[p], mei_dams, omer_history_start_p3g1
                                                        , days_period_pa1e1b1nwzida0e0b0xyg1[p], km_dams, pinp.sheep['i_steepness']
                                                        , densityw_pa1e1b1nwzida0e0b0xyg1[p], foo_dams, confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                                        , intake_f_dams, dmd_dams, sam_mr = sam_mr_dams)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_dams = temp1
                        if eqn_used:
                            meme_dams = temp0
                            hp_maint_dams = meme_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3 = sfun.f1_efficiency(ck_offs, md_solid_offs, pinp.sheep['i_md_supp']
                                                                        , md_herb_offs, lgf_eff_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                                                        , dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...], sam_kg=sam_kg_offs)
                        if eqn_used:
                            km_offs = temp0
                            kg_fodd_offs = temp1
                            kg_supp_offs = temp2  # temp3 is not used for offspring
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        temp0, temp1 = sfun.f_energy_cs(cx_offs[:,mask_x,...], cm_offs, lw_start_offs, ffcfw_start_offs
                                                        , mr_age_pa1e1b1nwzida0e0b0xyg3[p], mei_offs, omer_history_start_p3g3
                                                        , days_period_pa1e1b1nwzida0e0b0xyg3[p], km_offs, pinp.sheep['i_steepness']
                                                        , densityw_pa1e1b1nwzida0e0b0xyg3[p], foo_offs, confinementw_tpa1e1b1nwzida0e0b0xyg3[:,p]
                                                        , intake_f_offs, dmd_offs, sam_mr = sam_mr_offs)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_offs = temp1
                        if eqn_used:
                            meme_offs = temp0
                            hp_maint_offs = meme_offs
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                eqn_system = 2 # New feeding standards = 2
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p, ...] >0):
                        temp0, temp1 = sfun.f_energy_nfs(ck_sire, cm_sire, lw_start_sire, ffcfw_start_sire, f_start_sire
                                                         , v_start_sire, m_start_sire, mei_sire, md_solid_sire
                                                         , pinp.sheep['i_steepness'], densityw_pa1e1b1nwzida0e0b0xyg0[p]
                                                         , foo_sire, confinementw_tpa1e1b1nwzida0e0b0xyg0[:, p]
                                                         , intake_f_sire, dmd_sire, sam_mr = sam_mr_sire)
                        if eqn_used:
                            hp_maint_sire = temp0
                            meme_sire - hp_maint_sire
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] >0):
                        temp0, temp1 = sfun.f_energy_nfs(ck_dams, cm_dams, lw_start_dams, ffcfw_start_dams, f_start_dams
                                                         , v_start_dams, m_start_dams, mei_dams, md_solid_dams
                                                         , pinp.sheep['i_steepness'], densityw_pa1e1b1nwzida0e0b0xyg1[p]
                                                         , foo_dams, confinementw_tpa1e1b1nwzida0e0b0xyg1[:, p]
                                                         , intake_f_dams, dmd_dams, sam_mr = sam_mr_dams)
                        if eqn_used:
                            hp_maint_dams = temp0
                            kl_dams = temp1
                            meme_dams = hp_maint_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0  # kl could be retained
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] >0):
                        temp0, temp1 = sfun.f_energy_nfs(ck_offs, cm_offs, lw_start_offs, ffcfw_start_offs, f_start_offs
                                                         , v_start_offs, m_start_offs, mei_offs, md_solid_offs
                                                         , pinp.sheep['i_steepness'], densityw_pa1e1b1nwzida0e0b0xyg3[p]
                                                         , foo_offs, confinementw_tpa1e1b1nwzida0e0b0xyg3[:, p]
                                                         , intake_f_offs, dmd_offs, sam_mr = sam_mr_offs)
                        if eqn_used:
                            hp_maint_offs = temp0
                            meme_offs = hp_maint_offs
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0


                ##foetal growth - dams
                eqn_group = 9
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        ##first method is using the nec_cum method
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_foetus_cs(cp_dams, cb1_dams, kc_yg1, nfoet_b1nwzida0e0b0xyg, relsize_start_dams
                                        , rc_start_dams, w_b_std_y_b1nwzida0e0b0xyg1, w_f_start_dams, nw_f_start_dams, nwf_age_f_pa1e1b1nwzida0e0b0xyg1[p]
                                        , guw_age_f_pa1e1b1nwzida0e0b0xyg1[p], dce_age_f_pa1e1b1nwzida0e0b0xyg1[p])
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        w_f_dams = temp0
                        nw_f_dams = temp4
                        if eqn_used:
                            mec_dams = temp1
                            nec_dams = temp2
                            w_b_exp_y_dams = temp3
                            guw_dams = temp5
                            dc_dams = nec_dams
                            hp_dc_dams = mec_dams - nec_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1

                eqn_system = 2  # New Feeding Standards = 2
                if uinp.sheep['i_eqn_exists_q0q1'][
                    eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    eqn_used = (eqn_used_g2_q1p[
                                    eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                        ##first method is using the nec_cum method
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_foetus_nfs(cp_dams, ck_dams, step
                                        , c_start_dams, m_start_dams, dm_dams, nfoet_b1nwzida0e0b0xyg, relsize_start_dams
                                        , w_b_std_y_b1nwzida0e0b0xyg1, w_f_start_dams
                                        , nwf_age_f_pa1e1b1nwzida0e0b0xyg1[p], guw_age_f_pa1e1b1nwzida0e0b0xyg1[p]
                                        , ce_day1_f_dams, dcdt_age_f_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gest_propn = gest_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        w_f_dams = temp0
                        nw_f_dams = temp4
                        if eqn_used:
                            dc_dams = temp1
                            hp_dc_dams = temp2
                            w_b_exp_y_dams = temp3
                            guw_dams = temp5
                            nec_dams = dc_dams
                            mec_dams = dc_dams + hp_dc_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1

                ##milk production
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    ###Expected ffcfw of yatf with p1 axis - each period
                    #### The test on index_p is to test for the end of lactation. Start of lactation (birth) is always the start of a period
                    ffcfw_exp_a1e1b1nwzida0e0b0xyg2p0 = (ffcfw_start_yatf[..., na] + (index_p0 * cn_yatf[7, ...][...,na])) * (
                                index_p0 < days_period_pa1e1b1nwzida0e0b0xyg2[...,na][p])
                    ###Expected average metabolic LW of yatf during period
                    ffcfw75_exp_yatf = np.sum(ffcfw_exp_a1e1b1nwzida0e0b0xyg2p0 ** 0.75, axis=-1) / np.maximum(1, days_period_pa1e1b1nwzida0e0b0xyg2[p, ...])

                    mp2_dams, mel_dams, nel_dams, ldr_dams, lb_dams \
                        = sfun.f_milk_cs(cl_dams, srw_b0xyg1, relsize_start_dams, rc_birth_dams, mei_dams, meme_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p]
                            , rc_start_dams, ffcfw75_exp_yatf, lb_start_dams, ldr_start_dams, age_pa1e1b1nwzida0e0b0xyg2[p]
                            , mp_age_y_pa1e1b1nwzida0e0b0xyg1[p], mp2_age_y_pa1e1b1nwzida0e0b0xyg1[p], x_pos
                            , days_period_pa1e1b1nwzida0e0b0xyg2[p], kl_dams, lact_nut_effect_pa1e1b1nwzida0e0b0xyg1[p])
                    mp2_yatf = fun.f_divide(mp2_dams, nyatf_b1nwzida0e0b0xyg) # 0 if given slice of b1 axis has no yatf
                    dl_dams = nel_dams
                    hp_dl_dams = mel_dams - nel_dams

                ##wool production
                eqn_group = 17
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_cs(cw_sire, cc_sire, ffcfw_start_sire
                                           , relsize_start_sire, d_cfw_history_start_p2g0
                                           , mei_sire, mew_min_pa1e1b1nwzida0e0b0xyg0[p]
                                           , d_cfw_ave_pa1e1b1nwzida0e0b0xyg0[p, ...],  sfd_a0e0b0xyg0, wge_a0e0b0xyg0
                                           , af_wool_pa1e1b1nwzida0e0b0xyg0[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                           , kw_yg0, days_period_pa1e1b1nwzida0e0b0xyg0[p], sfw_ltwadj_g0, sfd_ltwadj_g0
                                           , rev_trait_values['sire'][p])
                        if eqn_used:
                            d_cfw_sire = temp0
                            d_fd_sire = temp1
                            d_fl_sire = temp2
                            d_cfw_history_sire_p2 = temp3
                            mew_sire = temp4
                            new_sire = temp5
                            dw_sire = new_sire
                            hp_dw_sire = mew_sire - new_sire
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_cs(cw_dams, cc_dams, ffcfw_start_dams
                                           , relsize_start_dams, d_cfw_history_start_p2g1
                                           , mei_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p]
                                           , d_cfw_ave_pa1e1b1nwzida0e0b0xyg1[p, ...], sfd_a0e0b0xyg1, wge_a0e0b0xyg1
                                           , af_wool_pa1e1b1nwzida0e0b0xyg1[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                           , kw_yg1, days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                           , sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[p, ...], sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                           , rev_trait_values['dams'][p]
                                           , mec_dams, mel_dams, gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                           , lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            d_cfw_dams = temp0
                            d_fd_dams = temp1
                            d_fl_dams = temp2
                            d_cfw_history_dams_p2 = temp3
                            mew_dams = temp4
                            new_dams = temp5
                            dw_dams = new_dams
                            hp_dw_dams = mew_dams - new_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] > 0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_cs(cw_offs, cc_offs, ffcfw_start_offs
                                           , relsize_start_offs, d_cfw_history_start_p2g3
                                           , mei_offs, mew_min_pa1e1b1nwzida0e0b0xyg3[p]
                                           , d_cfw_ave_pa1e1b1nwzida0e0b0xyg3[p, ...], sfd_da0e0b0xyg3, wge_da0e0b0xyg3
                                           , af_wool_pa1e1b1nwzida0e0b0xyg3[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                           , kw_yg3, days_period_pa1e1b1nwzida0e0b0xyg3[p]
                                           , sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3, sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3
                                           , rev_trait_values['offs'][p])
                        if eqn_used:
                            d_cfw_offs = temp0
                            d_fd_offs = temp1
                            d_fl_offs = temp2
                            d_cfw_history_offs_p2 = temp3
                            mew_offs = temp4
                            new_offs = temp5
                            dw_offs = new_offs
                            hp_dw_offs = mew_offs - new_offs
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                eqn_system = 2 # New Feeding Standards = 2
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_nfs(cw_sire, cc_sire, cg_sire, ck_sire
                                            , ffcfw_start_sire, relsize_start_sire, d_cfw_history_start_p2g0, mei_sire
                                            , mew_min_pa1e1b1nwzida0e0b0xyg0[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                            , sfd_a0e0b0xyg0, wge_a0e0b0xyg0, af_wool_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                            , dlf_wool_pa1e1b1nwzida0e0b0xyg0[p, ...], days_period_pa1e1b1nwzida0e0b0xyg0[p]
                                            , sfw_ltwadj_g0, sfd_ltwadj_g0, rev_trait_values['sire'][p])
                        if eqn_used:
                            d_cfw_sire = temp0
                            d_fd_sire = temp1
                            d_fl_sire = temp2
                            d_cfw_history_sire_p2 = temp3
                            dw_sire = temp4
                            hp_dw_sire = temp5
                            new_sire = dw_sire
                            mew_sire = dw_sire + hp_dw_sire
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p, ...] > 0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_nfs(cw_dams, cc_dams, cg_dams, ck_dams
                                            , ffcfw_start_dams, relsize_start_dams, d_cfw_history_start_p2g1, mei_dams
                                            , mew_min_pa1e1b1nwzida0e0b0xyg1[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                            , sfd_a0e0b0xyg1, wge_a0e0b0xyg1, af_wool_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                            , dlf_wool_pa1e1b1nwzida0e0b0xyg1[p, ...], days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                            , sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[p, ...], sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                            , rev_trait_values['dams'][p], mec_dams, mel_dams
                                            , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            d_cfw_dams = temp0
                            d_fd_dams = temp1
                            d_fl_dams = temp2
                            d_cfw_history_dams_p2 = temp3
                            dw_dams = temp4
                            hp_dw_dams = temp5
                            new_dams = dw_dams
                            mew_dams = dw_dams + hp_dw_dams
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p, ...] > 0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_nfs(cw_offs, cc_offs, cg_offs, ck_offs
                                            , ffcfw_start_offs, relsize_start_offs, d_cfw_history_start_p2g3, mei_offs
                                            , mew_min_pa1e1b1nwzida0e0b0xyg3[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                            , sfd_da0e0b0xyg3, wge_da0e0b0xyg3, af_wool_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                            , dlf_wool_pa1e1b1nwzida0e0b0xyg3[p, ...], days_period_pa1e1b1nwzida0e0b0xyg3[p]
                                            , sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3, sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3
                                           , rev_trait_values['offs'][p])
                        if eqn_used:
                            d_cfw_offs = temp0
                            d_fd_offs = temp1
                            d_fl_offs = temp2
                            d_cfw_history_offs_p2 = temp3
                            dw_offs = temp4
                            hp_dw_offs = temp5
                            new_offs = dw_offs
                            mew_offs = dw_offs + hp_dw_offs
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ##total heat production (excluding chill) & energy to offset chilling
                eqn_group = 7
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1 = sfun.f_heat_cs(cc_sire, ck_sire, mei_sire, meme_sire, mew_sire, new_sire, km_sire
                                               , kg_supp_sire, kg_fodd_sire, mei_propn_supp_sire, mei_propn_herb_sire)
                        if eqn_used:
                            hp_total_sire = temp0
                            level_sire = temp1
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0, temp1, temp2 = sfun.f_chill_cs(cc_sire, ck_sire, ffcfw_start_sire, rc_start_sire, sl_start_sire, mei_sire
                                                          , hp_total_sire, meme_sire, mew_sire, km_sire, kg_supp_sire, kg_fodd_sire, mei_propn_supp_sire
                                                          , mei_propn_herb_sire, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                          , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp0[p]
                                                          , index_m0)
                        if eqn_used:
                            mem_sire = temp0
                            temp_lc_sire = temp1
                            kg_sire = temp2
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1 = sfun.f_heat_cs(cc_dams, ck_dams, mei_dams, meme_dams, mew_dams, new_dams, km_dams
                                               , kg_supp_dams, kg_fodd_dams, mei_propn_supp_dams, mei_propn_herb_dams
                                               , guw = guw_dams, kl = kl_dams, mei_propn_milk = mei_propn_milk_dams
                                               , mec = mec_dams, mel = mel_dams, nec = nec_dams, nel = nel_dams
                                               , gest_propn = gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                               , lact_propn = lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            hp_total_dams = temp0
                            level_dams = temp1
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0, temp1, temp2 = sfun.f_chill_cs(cc_dams, ck_dams, ffcfw_start_dams, rc_start_dams, sl_start_dams, mei_dams
                                                              , hp_total_dams, meme_dams, mew_dams, km_dams, kg_supp_dams, kg_fodd_dams, mei_propn_supp_dams
                                                              , mei_propn_herb_dams, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                              , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp0[p]
                                                              , index_m0, kl=kl_dams, mei_propn_milk=mei_propn_milk_dams, mec=mec_dams
                                                              , mel=mel_dams, gest_propn=gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                                              , lact_propn=lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            mem_dams = temp0
                            temp_lc_dams = temp1
                            kg_dams = temp2
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1 = sfun.f_heat_cs(cc_offs, ck_offs, mei_offs, meme_offs, mew_offs, new_offs, km_offs
                                               , kg_supp_offs, kg_fodd_offs, mei_propn_supp_offs, mei_propn_herb_offs)
                        if eqn_used:
                            hp_total_offs = temp0
                            level_offs = temp1
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0, temp1, temp2 = sfun.f_chill_cs(cc_offs, ck_offs, ffcfw_start_offs, rc_start_offs, sl_start_offs, mei_offs
                                                              , hp_total_offs, meme_offs, mew_offs, km_offs, kg_supp_offs, kg_fodd_offs, mei_propn_supp_offs
                                                              , mei_propn_herb_offs, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                              , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp0[p]
                                                              , index_m0)
                        if eqn_used:
                            mem_offs = temp0
                            temp_lc_offs = temp1
                            kg_offs = temp2
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

                eqn_system = 2 # New Feeding Standards = 2
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        # temp0, temp1 = sfun.f_heat_nfs(cc_sire, hp_maint_sire, hp_dw_sire) #hp_dv, hp_dm & hp_df not available at this point in the code
                        # if eqn_used:
                        #     hp_total_sire = temp0
                        #     level_sire = temp1
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0 = sfun.f_heatloss_nfs(cc_sire, ffcfw_start_sire, rc_start_sire, sl_start_sire
                                                 , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                 , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                 , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            heat_loss_sire = temp0
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        # temp0, temp1 = sfun.f_heat_nfs(cc_dams, hp_maint_dams, hp_dw_dams    #hp_dv, hp_dm & hp_df not available at this point in the code
                        #                         , hp_dc = hp_dc_dams, hp_dl = hp_dl_dams, guw = guw_dams
                        #                         , gest_propn = gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                        #                         , lact_propn = lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        # if eqn_used:
                        #     hp_total_dams = temp0
                        #     level_dams = temp1
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0 = sfun.f_heatloss_nfs(cc_dams, ffcfw_start_dams, rc_start_dams, sl_start_dams
                                                 , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                 , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                 , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            heat_loss_dams = temp0
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        # temp0, temp1 = sfun.f_heat_nfs(cc_offs, hp_maint_offs, hp_dw_offs) #hp_dv, hp_dm & hp_df not available at this point in the code
                        # if eqn_used:
                        #     hp_total_offs = temp0
                        #     level_offs = temp1
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                        temp0 = sfun.f_heatloss_nfs(cc_offs, ck_offs, ffcfw_start_offs, rc_start_offs, sl_start_offs
                                                 , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                 , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                 , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            heat_loss_offs = temp0
                        # if eqn_compare:
                        #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

                ##calc lwc
                eqn_group = 7
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_sire, rc_start_sire, mei_sire, mem_sire
                                                                , mew_sire, zf1_sire, zf2_sire, kg_sire, rev_trait_values['sire'][p])
                        if eqn_used:
                            ebg_sire = temp0
                            evg_sire = temp1
                            pg_sire = temp2
                            fg_sire = temp3
                            surplus_energy_sire = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_dams, rc_start_dams, mei_dams, mem_dams
                                                , mew_dams, zf1_dams, zf2_dams, kg_dams, rev_trait_values['dams'][p], mec_dams, mel_dams
                                                , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            ebg_dams = temp0
                            evg_dams = temp1
                            pg_dams = temp2
                            fg_dams = temp3
                            surplus_energy_dams = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_offs, rc_start_offs, mei_offs, mem_offs
                                                                , mew_offs, zf1_offs, zf2_offs, kg_offs, rev_trait_values['offs'][p])
                        if eqn_used:
                            ebg_offs = temp0
                            evg_offs = temp1
                            pg_offs = temp2
                            fg_offs = temp3
                            surplus_energy_offs = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp1

                eqn_system = 1 # Murdoch = 1
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_mu(cg_sire, rc_start_sire, mei_sire, mem_sire
                                                                , mew_sire, zf1_sire, zf2_sire, kg_sire, rev_trait_values['sire'][p])
                        if eqn_used:
                            ebg_sire = temp0
                            evg_sire = temp1
                            pg_sire = temp2
                            fg_sire = temp3
                            surplus_energy_sire = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_mu(cg_dams, rc_start_dams, mei_dams, mem_dams
                                                , mew_dams, zf1_dams, zf2_dams, kg_dams, rev_trait_values['dams'][p], mec_dams, mel_dams
                                                , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            ebg_dams = temp0
                            evg_dams = temp1
                            pg_dams = temp2
                            fg_dams = temp3
                            surplus_energy_dams = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_mu(cg_offs, rc_start_offs, mei_offs, mem_offs
                                                                , mew_offs, zf1_offs, zf2_offs, kg_offs, rev_trait_values['offs'][p])
                        if eqn_used:
                            ebg_offs = temp0
                            evg_offs = temp1
                            pg_offs = temp2
                            fg_offs = temp3
                            surplus_energy_offs = temp4
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp1


                eqn_system = 2 # New Feeding Standards = 2
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_nfs(cm_sire, cg_sire, rc_start_sire
                                                , mei_sire , md_solid_sire, mew_sire, zf1_sire, zf2_sire, kg_sire, step
                                                , rev_trait_values['sire'][p])
                        if eqn_used:
                            ebg_sire = temp0
                            evg_sire = temp1
                            df_sire = temp2
                            dm_sire = temp3
                            dv_sire = temp4
                            surplus_energy_sire = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp1
                        temp0 = sfun.level_nfs(mei_sire, hp_maint_sire)
                        if eqn_used:
                            level_sire = temp0
                        temp0, temp1 = sfun.templc(cc_sire, ffcfw_start_sire, rc_start_sire, sl_start_sire, hp_total_sire
                                                   , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                   , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                   , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            temp_lc_sire = temp0  #temp1 not required here
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_nfs(cm_dams, cg_dams, rc_start_dams
                                                , mei_dams , md_solid_dams, mew_dams, zf1_dams, zf2_dams, kg_dams, step
                                                , rev_trait_values['dams'][p], mec_dams, mel_dams
                                                , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            ebg_dams = temp0
                            evg_dams = temp1
                            df_dams = temp2
                            dm_dams = temp3
                            dv_dams = temp4
                            surplus_energy_dams = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1
                        temp0 = sfun.level_nfs(mei_dams, hp_maint_dams)
                        if eqn_used:
                            level_dams = temp0
                        temp0, temp1 = sfun.templc(cc_dams, ffcfw_start_dams, rc_start_dams, sl_start_dams, hp_total_dams
                                                   , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                   , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                   , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            temp_lc_dams = temp0  #temp1 not required here
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_nfs(cm_offs, cg_offs, rc_start_offs
                                                , mei_offs , md_solid_offs, mew_offs, zf1_offs, zf2_offs, kg_offs, step
                                                , rev_trait_values['offs'][p])
                        if eqn_used:
                            ebg_offs = temp0
                            evg_offs = temp1
                            df_offs = temp2
                            dm_offs = temp3
                            dv_offs = temp4
                            surplus_energy_offs = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp1
                        temp0 = sfun.level_nfs(mei_offs, hp_maint_offs)
                        if eqn_used:
                            level_offs = temp0
                        temp0, temp1 = sfun.templc(cc_offs, ffcfw_start_offs, rc_start_offs, sl_start_offs, hp_total_offs
                                                   , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                   , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                                   , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                        if eqn_used:
                            temp_lc_offs = temp0  #temp1 not required here


                ###if there is a target then adjust feedsupply, if not break out of feedsupply loop
                if target_lwc_dams[p] == 9999 and target_lwc_offs[p] == 9999:
                    break
                print('fs target iteration: ', itn)
                if target_lwc_dams[p] != 9999 and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p]>0):
                    ###calc error
                    error = (ebg_dams * cg_dams[18, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p]) - target_lwc_dams[p] * (days_period_pa1e1b1nwzida0e0b0xyg1[p]>0) #if 0 days in period then target is 0
                    ###store in attempts array - build new array assign old array and then add current itn results - done like this to handle the shape changing and because we don't know what shape feedsupply and error are before this loop starts
                    attempts_dams[...,itn,0] = feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                    attempts_dams[...,itn,1] = error
                    ###is error within tolerance
                    if np.all(np.abs(error) <= epsilon):
                        break
                    ###max attempts reached
                    elif itn == n_max_itn-1: #minus 1 because range() and hence itn starts from 0
                        ####select best feed supply option
                        best_idx = np.argmin(np.abs(attempts_dams[...,1]),axis=-1)
                        feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p] = np.take_along_axis(attempts_dams[...,0], best_idx[...,na], axis=-1)[...,0] #get rid of singleton itn axis
                        break
                    feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p:p+1] = sfun.f1_feedsupply_adjust(attempts_dams,feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p],itn)
                if target_lwc_offs[p] != 9999 and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p]>0):
                    ###calc error
                    error = (ebg_offs * cg_offs[18, ...] * days_period_pa1e1b1nwzida0e0b0xyg3[p]) - target_lwc_offs[p] * (days_period_pa1e1b1nwzida0e0b0xyg3[p]>0) #if 0 days in period then target is 0
                    ###store in attempts array - build new array assign old array and then add current itn results - done like this to handle the shape changing and because we don't know what shape feedsupply and error are before this loop starts
                    attempts_offs[...,itn,0] = feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p]
                    attempts_offs[...,itn,1] = error
                    ###is error within tolerance
                    if np.all(np.abs(error) <= epsilon):
                        break
                    ###max attempts reached
                    elif itn == n_max_itn-1: #minus 1 because range() and hence itn starts from 0
                        ####select best feed supply option
                        best_idx = np.argmin(np.abs(attempts_offs[...,1]),axis=-1)
                        feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p] = np.take_along_axis(attempts_offs[...,0], best_idx[...,na], axis=-1)[...,0] #get rid of singleton itn axis
                        break
                    feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p] = sfun.f1_feedsupply_adjust(attempts_offs,feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p],itn)
                itn+=1

            ##dam weight at a given time during period - used for special events like mating, birth & weaning.
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ##Dam weight at mating - to estimate the weight at mating we are wanting to use the growth rate of the dams that are not yet pregnant
                ## because mating doesn't happen at the start of the period.
                ##relative size and relative condition of the dams at mating are the determinants of conception
                ## use the condition of dams in the 11 slice because mated animals can have a different feed supply
                ## use dams in e[-1] because want the condition of the animal before it conceives. Note all e slices will have the same condition until conceived because they have the same feedsupply until scanning.
                ffcfw_e1b1sliced = fun.f_dynamic_slice(ffcfw_start_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                relsize_start_dams_e1b1sliced = fun.f_dynamic_slice(relsize_start_dams, e1_pos, -1, None, b1_pos, 2, 3)  #slice e1 & b1 axis
                ebg_e1b1sliced = fun.f_dynamic_slice(ebg_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                nw_start_dams_e1b1sliced = fun.f_dynamic_slice(nw_start_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                gest_propn_b1sliced = fun.f_dynamic_slice(gest_propn_pa1e1b1nwzida0e0b0xyg1[p], b1_pos, 2, 3) #slice b1 axis
                days_period_b1sliced = fun.f_dynamic_slice(days_period_pa1e1b1nwzida0e0b0xyg1[p], b1_pos, 2, 3) #slice b1 axis

                t_w_mating = np.sum((ffcfw_e1b1sliced + ebg_e1b1sliced * cg_dams[18, ...] * days_period_b1sliced
                                     * (1 - gest_propn_b1sliced)) * period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                    , axis=e1_pos, keepdims=True) #Temporary variable for mating weight
                ffcfw_mating_dams = fun.f_update(ffcfw_mating_dams, t_w_mating, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p])
                maternallw_mating_dams = ffcfw_mating_dams
                ##LW change during joining is required for the LMAT conception equation. Using LWC in the single generator period
                lwc_mating_dams = fun.f_update(lwc_mating_dams, ebg_e1b1sliced, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]) * cg_dams[18, ...]
                ##Relative condition of the dam at mating - required to determine milk production
                rc_mating_dams = ffcfw_mating_dams / nw_start_dams_e1b1sliced
                ##Condition score of the dams at mating
                cs_mating_dams = sfun.f1_condition_score(rc_mating_dams, cn_dams)
                ##Relative size of the dams at mating
                relsize_mating_dams = relsize_start_dams_e1b1sliced

                ##Dam weight at birth ^probs don't need this since birth is first day of period - just need to pass in ffcfw_start to function
                # t_w_birth = ffcfw_start_dams + ebg_dams * cg_dams[18, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p] * gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                # ffcfw_birth_dams = fun.f_update(ffcfw_birth_dams, t_w_birth, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                ##Relative condition of the dam at most recent birth
                # rc_birth_dams = ffcfw_birth_dams / nw_start_dams

                ##Dam weight at weaning
                # t_w_weaning = ffcfw_start_dams + ebg_dams * cg_dams[18, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p] * lact_propn_pa1e1b1nwzida0e0b0xyg1[p]
                # ffcfw_weaning_dams = fun.f_update(ffcfw_weaning_dams, t_w_weaning, period_is_wean_pa1e1b1nwzida0e0b0xyg1[p])


            ##birth weight yatf - calculated when dams days per period > 0  - calced for start of period (ebg is mul by gest propn so not included for period when birth happens)
            eqn_group = 10
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_birthweight_cs(cx_yatf[:,mask_x,...], w_b_start_yatf, w_f_start_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]) #pass in wf_start because animal is born on first day of period
                    if eqn_used:
                        w_b_yatf = temp0 * (nfoet_b1nwzida0e0b0xyg>0) #so that only b slices with nfoet have a weight (need to leave a weight in 30, 20, 10 because used in prog mort)
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # MU = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0, temp1 = sfun.f_birthweight_mu(cu1_yatf, cb1_yatf, cg_yatf, cx_yatf[:,mask_x,...], ce_pyatf[:,p,...]
                                        , w_b_start_yatf, cf_w_b_start_dams, ffcfw_start_dams, ebg_dams
                                        , days_period_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                        , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                        , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                        , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]) #have to use yatf days per period if using prejoinng to scanning
                    ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                    cf_w_b_dams = temp1
                    if eqn_used:
                        w_b_yatf = temp0 * (nfoet_b1nwzida0e0b0xyg>0) #so that only b slices with nfoet have a weight (need to leave a weight in 30, 20, 10 because used in prog mort)
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##progeny fleece prodn adjustment due to dam profile (LTW adjustment)
            eqn_group = 14
            eqn_system = 0 # CSIRO = 0 - doesn't exist for LTW impacts on progeny
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                # if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                #     temp0 = sfun.f_progenycfw_cs(cx_yatf[:,mask_x,...], w_b_start_yatf, w_f_start_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]) #pass in wf_start because animal is born on first day of period
                #     if eqn_used:
                #         cfw_ltwadj_dams = temp0
                #         fdltw = temp1
                #     if eqn_compare:
                #         r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                #         r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1
            eqn_system = 1 # MU = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0, temp1 = sfun.f_progenycfw_mu(cu1_yatf, cg_yatf, cfw_ltwadj_start_dams, cf_cfwltw_start_dams
                                            , ffcfw_start_dams, nw_start_dams, ebg_dams
                                            , days_period_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                    temp2, temp3 = sfun.f_progenyfd_mu(cu1_yatf, cg_yatf, fd_ltwadj_start_dams, cf_fdltw_start_dams
                                            , ffcfw_start_dams, nw_start_dams, ebg_dams
                                            , days_period_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                    #if eqn_used:
                    ## these variables all need to be stored even if the equation system is not used so that the equations can be compared
                    cfw_ltwadj_dams = temp0
                    cf_cfwltw_dams = temp1
                    fd_ltwadj_dams = temp2
                    cf_fdltw_dams = temp3
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp2

            ##yatf resets
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ##reset start variables if period is birth
                ###ffcf weight of yatf
                ffcfw_start_yatf = fun.f_update(ffcfw_start_yatf, w_b_yatf, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ffcfw_max_start_yatf = fun.f_update(ffcfw_max_start_yatf, w_b_yatf, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###normal weight of yatf
                nw_start_yatf	= fun.f_update(nw_start_yatf, w_b_yatf, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###adipose weight of yatf
                aw_start_yatf	= fun.f_update(aw_start_yatf, w_b_yatf * aw_propn_birth_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###muscle weight of yatf
                mw_start_yatf	= fun.f_update(mw_start_yatf, w_b_yatf * mw_propn_birth_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###bone weight of the yatf
                bw_start_yatf	= fun.f_update(bw_start_yatf, w_b_yatf * bw_propn_birth_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###clean fleece weight of yatf
                cfw_start_yatf	= fun.f_update(cfw_start_yatf, 0, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###fibre diameter of yatf
                fd_start_yatf	= fun.f_update(fd_start_yatf, 0, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###fibre length of yatf
                fl_start_yatf	= fun.f_update(fl_start_yatf, fl_birth_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###minimum fibre diameter of yatf
                fd_min_start_yatf	= fun.f_update(fd_min_start_yatf, 1000, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])

                ##Yatf dependent start values
                ###Normal weight max (if animal is well-fed) - yatf
                nw_max_yatf	= srw_b1xyg2 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg2[p]) + w_b_yatf * srw_age_pa1e1b1nwzida0e0b0xyg2[p]
                ##Dependent start: Change in normal weight max - yatf
                ###nw_max = srw - (srw - bw) * srw_age[p] so d_nw_max = (srw - (srw-bw) * srw_age[p]) - (srw - (srw - bw) * srw_age[p-1]) and that simplifies to d_nw_max = (srw_age[p-1] - srw_age[p]) * (srw-bw)
                d_nw_max_yatf = fun.f_divide((srw_age_pa1e1b1nwzida0e0b0xyg2[p-1, ...] - srw_age_pa1e1b1nwzida0e0b0xyg2[p, ...]) * (srw_b1xyg2 - w_b_yatf)
                                             , days_period_pa1e1b1nwzida0e0b0xyg2[p])
                ###GFW (start)
                gfw_start_yatf = cfw_start_yatf / cw_yatf[3, ...]
                ###LW (start -with fleece & conceptus)
                lw_start_yatf = ffcfw_start_yatf + gfw_start_yatf
                ###Normal weight (start)
                nw_start_yatf = np.minimum(nw_max_yatf, np.maximum(nw_start_yatf, ffcfw_start_yatf + cn_yatf[3, ...] * (nw_max_yatf  - ffcfw_start_yatf)))
                ###Relative condition (start) - use update function so that when 0 days/period we keep the rc of the last period because it is used to calc sale value which is period_is_weaning which has 0 days because sold at beginning.
                temp_rc_start_yatf = ffcfw_start_yatf / nw_start_yatf
                rc_start_yatf = fun.f_update(rc_start_yatf, temp_rc_start_yatf, days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0)
                ##Condition score of the dam at  start of p
                cs_start_yatf = sfun.f1_condition_score(rc_start_yatf, cn_yatf)
                ###staple length
                sl_start_yatf = fl_start_yatf * cw_yatf[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_yatf = np.minimum(1, nw_start_yatf / srw_b1xyg2)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_yatf = np.minimum(ffcfw_max_start_yatf, nw_max_yatf) / srw_b1xyg2
                ###PI Size factor (for cattle)
                zf_yatf = np.maximum(1, 1 + cr_yatf[7, ...] - relsize_start_yatf)
                ###EVG Size factor (decreases as z increases)
                ####Note: This equation purposefully has the opposite sign for cg[4] to Freer et al 2012
                ####There is an error in the documentation and this representation is consistent with Sheep Explorer.
                zf1_yatf = fun.f_back_transform(-cg_yatf[4, ...] * (relsize1_start_yatf - cg_yatf[5, ...]))
                ###EVG Size factor (increases at maturity)
                zf2_yatf = np.clip((relsize1_start_yatf - cg_yatf[6, ...]) / (cg_yatf[7, ...] - cg_yatf[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) for yatf.
                ### Creation of a new variable is to be consistent with adults
                sam_kg_yatf = sen.sam['kg_yatf']    # unlike adults this is not dependent on zf2.
                sam_mr_yatf = sen.sam['mr_yatf']
                sam_pi_yatf = sen.sam['pi_yatf']


            ##potential intake - yatf
            eqn_group = 4
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = sfun.f_potential_intake_cs(ci_yatf, cl_yatf, srw_b1xyg2, relsize_start_yatf, rc_start_yatf, temp_lc_yatf
                                                       , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                       , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg2[p]
                                                       , mp2 = mp2_yatf, piyf = piyf_pa1e1b1nwzida0e0b0xyg2[p]
                                                       , period_between_birthwean = period_between_birthwean_pa1e1b1nwzida0e0b0xyg1[p]
                                                       , sam_pi = sam_pi_yatf)
                    if eqn_used:
                        pi_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##feedsupply
            if not stubble:
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    mei_yatf,foo_yatf,dmd_yatf,mei_solid_yatf, md_solid_yatf, md_herb_yatf, intake_f_yatf, intake_s_yatf\
                        , mei_propn_milk_yatf, mei_propn_supp_yatf, mei_propn_herb_yatf \
                        = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                             , nv_a1e1b1j1wzida0e0b0xyg1, foo_a1e1b1j1wzida0e0b0xyg1
                                             , dmd_a1e1b1j1wzida0e0b0xyg1, supp_a1e1b1j1wzida0e0b0xyg1, pi_yatf, mp2_yatf)
            ###if generating for stubble then nv doesn't exist so need to calc a bit differently.
            else:
                ##use ra=1 for stubble
                # ##relative availability - yatf
                # eqn_group = 5
                # eqn_system = 0 # CSIRO = 0
                # if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                #     eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                #     if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                #         temp0 = fsfun.f_ra_cs(foo_yatf, hf_yatf, cr_yatf, zf_yatf)
                #         if eqn_used:
                #             ra_yatf = temp0
                #         if eqn_compare:
                #             r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                # eqn_system = 1 # Mu = 1
                # if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                #     eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                #     if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                #         temp0 = fsfun.f_ra_mu(foo_yatf, hf_yatf, zf_yatf, cu0_yatf)
                #         if eqn_used:
                #             ra_yatf = temp0
                #         if eqn_compare:
                #             r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ###calc dmd and md_herb - done within if statement because dmd & md_herb are calculated differently for stubble sim.
                dmd_yatf = dmd_pwg[p]
                md_herb_yatf = fsfun.f1_dmd_to_md(dmd_yatf)

                ##relative ingestibility (quality) - yatf
                eqn_group = 6
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                        temp0 = fsfun.f_rq_cs(dmd_yatf, legume_pa1e1b1nwzida0e0b0xyg[p], cr_yatf, pinp.sheep['i_sf'])
                        if eqn_used:
                            rq_yatf = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ##intake - yatf - use RA=1 for stubble
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    ri_yatf = fsfun.f_rel_intake(1, rq_yatf, legume_pa1e1b1nwzida0e0b0xyg[p], cr_yatf) #use ra=1 for stubble
                    mei_yatf, mei_solid_yatf, intake_f_yatf, md_solid_yatf, mei_propn_milk_yatf, mei_propn_herb_yatf, mei_propn_supp_yatf \
                                = sfun.f_intake(pi_yatf, ri_yatf, md_herb_yatf, feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[p]
                                                , intake_s_yatf, pinp.sheep['i_md_supp'], mp2_yatf)   #same feedsupply as dams

            ##energy - yatf
            eqn_group = 7
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3 = sfun.f1_efficiency(ck_yatf, md_solid_yatf, pinp.sheep['i_md_supp']
                                                                    , md_herb_yatf, lgf_eff_pa1e1b1nwzida0e0b0xyg2[p]
                                                                    , dlf_eff_pa1e1b1nwzida0e0b0xyg[p], mei_propn_milk_yatf
                                                                    , sam_kg=sam_kg_yatf)  #same feedsupply as dams
                    if eqn_used:
                        km_yatf = temp0
                        kg_fodd_yatf = temp1
                        kg_supp_yatf = temp2  # temp3 is not used for yatf
                    # if eqn_compare:
                    #     r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained
                    temp0, temp1 = sfun.f_energy_cs(cx_yatf[:,mask_x,...], cm_yatf, lw_start_yatf, ffcfw_start_yatf
                                                    , mr_age_pa1e1b1nwzida0e0b0xyg2[p], mei_yatf, omer_history_start_p3g2
                                                    , days_period_pa1e1b1nwzida0e0b0xyg2[p], km_yatf, pinp.sheep['i_steepness']
                                                    , densityw_pa1e1b1nwzida0e0b0xyg2[p], foo_yatf, confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                                    , intake_f_yatf, dmd_yatf, mei_propn_milk_yatf, sam_mr = sam_mr_yatf)  #same feedsupply as dams
                    ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                    omer_history_yatf = temp1
                    if eqn_used:
                        meme_yatf = temp0
                        hp_maint_yatf = meme_yatf
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained

            eqn_system = 2  # New Feeding Standards = 2
            if uinp.sheep['i_eqn_exists_q0q1'][
                eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    temp0, temp1 = sfun.f_energy_nfs(ck_yatf, cm_yatf, lw_start_yatf, ffcfw_start_yatf, f_start_yatf
                                                     , v_start_yatf, m_start_yatf, mei_yatf, md_solid_yatf
                                                     , pinp.sheep['i_steepness'], densityw_pa1e1b1nwzida0e0b0xyg2[p]
                                                     , foo_yatf, confinementw_tpa1e1b1nwzida0e0b0xyg1[:, p]
                                                     , intake_f_yatf, dmd_yatf, sam_mr = sam_mr_yatf)  #same feedsupply as dams
                    if eqn_used:
                        hp_maint_yatf = temp0
                        meme_yatf = hp_maint_yatf
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p,...] = temp0

            ##wool production - yatf
            eqn_group = 17
            eqn_system = 0  # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][
                eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_cs(cw_yatf, cc_yatf, ffcfw_start_yatf
                                   , relsize_start_yatf, d_cfw_history_start_p2g2
                                   , mei_yatf, mew_min_pa1e1b1nwzida0e0b0xyg2[p]
                                   , d_cfw_ave_pa1e1b1nwzida0e0b0xyg2[p, ...], sfd_pa1e1b1nwzida0e0b0xyg2[p]
                                   , wge_pa1e1b1nwzida0e0b0xyg2[p], af_wool_pa1e1b1nwzida0e0b0xyg2[p, ...]
                                   , dlf_wool_pa1e1b1nwzida0e0b0xyg2[p, ...], kw_yg2
                                   , days_period_pa1e1b1nwzida0e0b0xyg2[p], sfw_ltwadj_g2, sfd_ltwadj_g2
                                   , rev_trait_values['yatf'][p])

                    if eqn_used:
                        d_cfw_yatf = temp0
                        d_fd_yatf = temp1
                        d_fl_yatf = temp2
                        d_cfw_history_yatf_p2 = temp3
                        mew_yatf = temp4
                        new_yatf = temp5
                        dw_yatf = new_yatf
                        hp_dw_yatf = mew_yatf - new_yatf
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0

            eqn_system = 2  # New Feeding Standards = 2
            if uinp.sheep['i_eqn_exists_q0q1'][
                eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_fibre_nfs(cw_yatf, cc_yatf, cg_yatf, ck_yatf
                                    ,ffcfw_start_yatf, relsize_start_yatf, d_cfw_history_start_p2g2, mei_yatf
                                    , mew_min_pa1e1b1nwzida0e0b0xyg2[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg2[p, ...]
                                    , sfd_pa1e1b1nwzida0e0b0xyg2[p], wge_pa1e1b1nwzida0e0b0xyg2[p]
                                    , af_wool_pa1e1b1nwzida0e0b0xyg2[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg2[p, ...]
                                    , days_period_pa1e1b1nwzida0e0b0xyg2[p], sfw_ltwadj_g2, sfd_ltwadj_g2
                                    , rev_trait_values['yatf'][p])

                    if eqn_used:
                        d_cfw_yatf = temp0
                        d_fd_yatf = temp1
                        d_fl_yatf = temp2
                        d_cfw_history_yatf_p2 = temp3
                        dw_yatf = temp4
                        hp_dw_yatfd = temp5
                        new_yatf = dw_yatf
                        mew_yatf = dw_yatf + hp_dw_yatf
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0


            ##total heat production (excluding chill) & energy to offset chilling - yatf
            eqn_group = 7
            eqn_system = 0  # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    temp0, temp1 = sfun.f_heat_cs(cc_yatf, ck_yatf, mei_yatf, meme_yatf, mew_yatf, new_yatf, km_yatf
                                           , kg_supp_yatf, kg_fodd_yatf, mei_propn_supp_yatf, mei_propn_herb_yatf
                                           ,  mei_propn_milk=mei_propn_milk_yatf)
                    if eqn_used:
                        hp_total_yatf = temp0
                        level_yatf = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                    temp0, temp1, temp2 = sfun.f_chill_cs(cc_pyatf[:, p, ...], ck_yatf, ffcfw_start_yatf, rc_start_yatf, sl_start_yatf, mei_yatf
                                                          , hp_total_yatf, meme_yatf, mew_yatf, km_yatf, kg_supp_yatf, kg_fodd_yatf, mei_propn_supp_yatf
                                                          , mei_propn_herb_yatf, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                          , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp0[p]
                                                          , index_m0, mei_propn_milk=mei_propn_milk_yatf)
                    if eqn_used:
                        mem_yatf = temp0
                        temp_lc_yatf = temp1
                        kg_yatf = temp2
                    # if eqn_compare:
                    #     r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 3, :, p, ...] = temp0  # storing as the third variable

            eqn_system = 2  # New Feeding Standards = 2
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    # temp0, temp1 = sfun.f_heat_nfs(hp_maint_yatf, hp_dw_yatf, mei_yatf, meme_yatf, mew_yatf, new_yatf, km_yatf
                    #                        , kg_supp_yatf, kg_fodd_yatf, mei_propn_supp_yatf, mei_propn_herb_yatf
                    #                        , mei_propn_milk=mei_propn_milk_yatf)
                    # if eqn_used:
                    #     hp_total_yatf = temp0
                    #     level_yatf = temp1
                    # if eqn_compare:
                    #     r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp0  # storing as the second variable

                    temp0 = sfun.f_heatloss_nfs(cc_yatf, ffcfw_start_yatf, rc_start_yatf, sl_start_yatf
                                             , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                             , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                             , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                    if eqn_used:
                        heat_loss_yatf = temp0
                    # if eqn_compare:
                    #     r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp0

            ##calc lwc - yatf
            eqn_group = 7
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_yatf, rc_start_yatf, mei_yatf, mem_yatf
                                                            , mew_yatf, zf1_yatf, zf2_yatf, kg_yatf, rev_trait_values['yatf'][p])
                    if eqn_used:
                        ebg_yatf = temp0
                        evg_yatf = temp1
                        pg_yatf = temp2
                        fg_yatf = temp3
                        surplus_energy_yatf = temp4
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1

            eqn_system = 1 # Mu = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_mu(cg_yatf, rc_start_yatf, mei_yatf, mem_yatf
                                                                             , mew_yatf, zf1_yatf, zf2_yatf, kg_yatf, rev_trait_values['yatf'][p])
                    if eqn_used:
                        ebg_yatf = temp0
                        evg_yatf = temp1
                        pg_yatf = temp2
                        fg_yatf = temp3
                        surplus_energy_yatf = temp4
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1

            eqn_system = 2 # New Feeding Standards = 2
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_nfs(cm_yatf, cg_yatf, rc_start_yatf
                                            , mei_yatf , md_solid_yatf, mew_yatf, zf1_yatf, zf2_yatf, kg_yatf, step
                                            , rev_trait_values['yatf'][p])
                    if eqn_used:
                        ebg_yatf = temp0
                        evg_yatf = temp1
                        df_yatf = temp2
                        dm_yatf = temp3
                        dv_yatf = temp4
                        surplus_energy_yatf = temp5
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    temp0 = sfun.level_nfs(mei_yatf, hp_maint_yatf)
                    if eqn_used:
                        level_yatf = temp0
                    temp0, temp1 = sfun.templc(cc_yatf, ffcfw_start_yatf, rc_start_yatf, sl_start_yatf, hp_total_yatf
                                               , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                               , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p]
                                               , rain_pa1e1b1nwzida0e0b0xygp0[p], index_m0)
                    if eqn_used:
                        temp_lc_yatf = temp0  #temp1 not required here


            ##weaning weight yatf - called when dams days per period greater than 0 - calculates the weight at the start of the period
            eqn_group = 11
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                ##based on days_period_dams because weaning occurs at start of period so days_period_yatf==0
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_weanweight_cs(cg_yatf, w_w_start_yatf, ffcfw_start_yatf, ebg_yatf, days_period_pa1e1b1nwzida0e0b0xyg2[p]
                                                 , period_is_wean_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        w_w_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # Mu = 1   #it is okay to use ebg of current period because it is mul by lact propn
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                ##based on days_period_dams because weaning occurs at start of period so days_period_yatf==0
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0, temp1, temp2 = sfun.f_weanweight_mu(cu1_yatf, cb1_yatf, cg_yatf, cx_yatf[:,mask_x,...]
                                            , ce_pyatf[:,p,...], nyatf_b1nwzida0e0b0xyg, w_w_start_yatf, cf_w_w_start_dams
                                            , ffcfw_start_dams, ebg_dams, foo_dams, foo_lact_ave_start
                                            , days_period_pa1e1b1nwzida0e0b0xyg2[p], age_start_pa1e1b1nwzida0e0b0xyg2[p]
                                            , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_between_birthwean_pa1e1b1nwzida0e0b0xyg1[p]
                                            , period_is_wean_pa1e1b1nwzida0e0b0xyg1[p]) #have to use yatf days per period if using prejoining to scanning
                    ## these variables need to be available if being compared (but not used) so they can be condensed
                    cf_w_w_dams = temp1
                    foo_lact_ave = temp2
                    if eqn_used:
                        w_w_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0



            ##methane emissions
            eqn_group = 12
            eqn_system = 0 # National Greenhouse Gas Inventory Report
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ###sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = efun.f_ch4_animal_nir()
                    if eqn_used:
                        ch4_animal_sire = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = efun.f_ch4_animal_nir()
                    if eqn_used:
                        ch4_animal_dams = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = efun.f_ch4_animal_nir(mp2_yatf)
                    if eqn_used:
                        ch4_animal_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = efun.f_ch4_animal_nir()
                    if eqn_used:
                        ch4_animal_offs = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

            eqn_system = 1 # Baxter and Clapperton
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ###sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = efun.f_ch4_animal_bc(ch_sire, intake_f_sire, intake_s_sire, md_solid_sire, level_sire)
                    if eqn_used:
                        ch4_animal_sire = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = efun.f_ch4_animal_bc(ch_dams, intake_f_dams, intake_s_dams, md_solid_dams, level_dams)
                    if eqn_used:
                        ch4_animal_dams = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = efun.f_ch4_animal_bc(ch_yatf, intake_f_yatf, intake_s_yatf, md_solid_yatf, level_yatf)
                    if eqn_used:
                        ch4_animal_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = efun.f_ch4_animal_bc(ch_offs, intake_f_offs, intake_s_offs, md_solid_offs, level_offs)
                    if eqn_used:
                        ch4_animal_offs = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##Nitrous oxide emissions
            eqn_group = 13 #Nitrous oxide
            eqn_system = 0 #National Greenhouse Gas Inventory Report
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ###sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = efun.f_n2o_animal_nir(cl_sire, d_cfw_sire, relsize_start_sire, srw_b0xyg0, ebg_sire, mp=0, mc=0)
                    if eqn_used:
                        n2o_animal_sire = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = efun.f_n2o_animal_nir(cl_dams, d_cfw_dams, relsize_start_dams, srw_b0xyg1, ebg_dams, mp=mp2_dams)
                    if eqn_used:
                        n2o_animal_dams = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = efun.f_n2o_animal_nir(cl_yatf, d_cfw_yatf, relsize_start_yatf, srw_b1xyg2, ebg_yatf, mc=mp2_yatf)
                    if eqn_used:
                        n2o_animal_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = efun.f_n2o_animal_nir(cl_offs, d_cfw_offs, relsize_start_offs, srw_b0xyg3, ebg_offs)
                    if eqn_used:
                        n2o_animal_offs = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0


            ##conception Dams
            eqn_group = 0
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_conception_cs(cf_dams, cb1_dams, relsize_mating_dams, rc_mating_dams
                                                 , cpg_doy_cs_pa1e1b1nwzida0e0b0xyg1[p], nfoet_b1nwzida0e0b0xyg
                                                 , nyatf_b1nwzida0e0b0xyg, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                                 , index_e1b1nwzida0e0b0xyg, rev_trait_values['dams'][p]
                                                 , saa_rr_age_pa1e1b1nwzida0e0b0xyg1[p], sam_rr_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        conception_dams =  temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # MU LTW = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_conception_ltw(cf_dams, cu0_dams, relsize_mating_dams, cs_mating_dams
                                                  , scan_std_pa1e1b1nwzida0e0b0xyg1[p], doy_pa1e1b1nwzida0e0b0xyg[p]
                                                  , rr_doy_ltw_pa1e1b1nwzida0e0b0xyg1[p], nfoet_b1nwzida0e0b0xyg
                                                  , nyatf_b1nwzida0e0b0xyg, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                                  , index_e1b1nwzida0e0b0xyg, rev_trait_values['dams'][p])
                    if eqn_used:
                        conception_dams = temp0
                    ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 2 # MU 2 = 2
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_conception_mu2(cf_dams, cb1_dams, cu2_dams, srw_female_yg1, maternallw_mating_dams
                                                   , lwc_mating_dams * 1000, age_pa1e1b1nwzida0e0b0xyg1[p], nlb_yg3 * 100
                                                   , doj_pa1e1b1nwzida0e0b0xyg1[p:p+1], doj2_pa1e1b1nwzida0e0b0xyg1[p:p+1]
                                                   , lat_deg, nfoet_b1nwzida0e0b0xyg
                                                   , nyatf_b1nwzida0e0b0xyg, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                                   , rev_trait_values['dams'][p]
                                                   , saa_rr_age_pa1e1b1nwzida0e0b0xyg1[p], sam_rr_pa1e1b1nwzida0e0b0xyg1[p]
                                                   , saa_littersize_pa1e1b1nwzida0e0b0xyg1[p], saa_conception_pa1e1b1nwzida0e0b0xyg1[p]
                                                   , saa_preg_increment_pa1e1b1nwzida0e0b0xyg[p])
                    if eqn_used:
                        conception_dams = temp0
                    ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##Scanning percentage per ewe scanned (if scanning) -  report variable only
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                t_scanning = np.sum(numbers_start_dams * nfoet_b1nwzida0e0b0xyg, axis=prejoin_tup, keepdims=True) / np.sum(numbers_start_dams, axis=prejoin_tup, keepdims=True) * period_is_scan_pa1e1b1nwzida0e0b0xyg1[p]
                ##Scanning percentage per ewe scanned (if scanning)
                scanning = fun.f_update(scanning, t_scanning, period_is_scan_pa1e1b1nwzida0e0b0xyg1[p])


            ## base mortality - comments about mortality functions can be found in sfun.
            eqn_group = 15
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ####sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = sfun.f_mortality_base_cs(cd_sire, cg_sire, rc_start_sire, cv_weight_sire, ebg_sire, sd_ebg_sire, d_nw_max_pa1e1b1nwzida0e0b0xyg0[p]
                                                     , days_period_pa1e1b1nwzida0e0b0xyg0[p]
                                                     , rev_trait_values['sire'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_sire = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_base_cs(cd_dams, cg_dams, rc_start_dams, cv_weight_dams, ebg_dams, sd_ebg_dams, d_nw_max_pa1e1b1nwzida0e0b0xyg1[p]
                                                     , days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                     , rev_trait_values['dams'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_dams = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = sfun.f_mortality_base_cs(cd_yatf, cg_yatf, rc_start_yatf, cv_weight_yatf, ebg_yatf, sd_ebg_yatf, d_nw_max_yatf
                                                     , days_period_pa1e1b1nwzida0e0b0xyg2[p]
                                                     , rev_trait_values['yatf'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = sfun.f_mortality_base_cs(cd_offs, cg_offs, rc_start_offs, cv_weight_offs, ebg_offs, sd_ebg_offs, d_nw_max_pa1e1b1nwzida0e0b0xyg3[p]
                                                     , days_period_pa1e1b1nwzida0e0b0xyg3[p]
                                                     , rev_trait_values['offs'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_offs = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # MU/LTW = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ####sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = sfun.f_mortality_base_mu(cd_sire, cg_sire, rc_start_sire, cv_weight_sire, ebg_sire, sd_ebg_sire
                                                     , d_nw_max_pa1e1b1nwzida0e0b0xyg0[p], days_period_pa1e1b1nwzida0e0b0xyg0[p]
                                                     , rev_trait_values['sire'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_sire = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_base_mu(cd_dams, cg_dams, rc_start_dams, cv_weight_dams, ebg_dams, sd_ebg_dams
                                                     , d_nw_max_pa1e1b1nwzida0e0b0xyg1[p], days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                     , rev_trait_values['dams'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_dams = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = sfun.f_mortality_base_mu(cd_yatf, cg_yatf, rc_start_yatf, cv_weight_yatf, ebg_yatf, sd_ebg_yatf
                                                     , d_nw_max_yatf, days_period_pa1e1b1nwzida0e0b0xyg2[p]
                                                     , rev_trait_values['yatf'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_yatf = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ####offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = sfun.f_mortality_base_mu(cd_offs, cg_offs, rc_start_offs, cv_weight_offs, ebg_offs, sd_ebg_offs
                                                     , d_nw_max_pa1e1b1nwzida0e0b0xyg3[p], days_period_pa1e1b1nwzida0e0b0xyg3[p]
                                                     , rev_trait_values['offs'][p], sen.sap['mortalityb'])
                    if eqn_used:
                        mortality_offs = temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##weaner mortality - comments about mortality functions can be found in sfun.
            eqn_group = 2
            eqn_system = 0 # CSIRO = 0
            ####sire
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_cs(cd_sire, cg_sire, age_pa1e1b1nwzida0e0b0xyg0[p], ebg_sire, sd_ebg_sire
                                        , d_nw_max_pa1e1b1nwzida0e0b0xyg0[p], days_period_pa1e1b1nwzida0e0b0xyg0[p])
                    if eqn_used:
                        mortality_sire += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####dams
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_cs(cd_dams, cg_dams, age_pa1e1b1nwzida0e0b0xyg1[p], ebg_dams, sd_ebg_dams
                                        , d_nw_max_pa1e1b1nwzida0e0b0xyg1[p], days_period_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####offs
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_cs(cd_offs, cg_offs, age_pa1e1b1nwzida0e0b0xyg3[p], ebg_offs, sd_ebg_offs
                                        , d_nw_max_pa1e1b1nwzida0e0b0xyg3[p], days_period_pa1e1b1nwzida0e0b0xyg3[p])
                    if eqn_used:
                        mortality_offs += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # MU = 1
            ####sire
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_mu(cu2_sire)
                    if eqn_used:
                        mortality_sire += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####dams
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_mu(cu2_dams)  #no ce_dams because dam weaners don't have a d axis
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####offs
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_mu(cu2_offs, ce_dams)
                    if eqn_used:
                        mortality_offs += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##Peri-natal (around birth) Dam mortality - comments about mortality functions can be found in sfun.
            eqn_group = 3
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_dam_cs()
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # mu = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_dam_mu(cu2_dams, ce_pdams[:,p,...], cb1_dams, cs_start_dams, cv_cs_dams
                                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , nfoet_b1nwzida0e0b0xyg, saa_mortalitye_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        mortality_dams += temp0 #dam mort at birth due to low CS
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 2 # mu2 = 2
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    ## calculate CS change of dams (to reduce the arguments required)
                    csc_dams = ebg_dams * cg_dams[18, ...] / (cn_dams[5, ...] * nw_start_dams)
                    temp0, temp1 = sfun.f_mortality_dam_mu2(cu2_dams, ce_pdams[:,p,...], cb1_dams, cf_csc_start_dams
                                        , csc_dams, cs_start_dams, cv_cs_dams, period_between_scanprebirth_pa1e1b1nwzida0e0b0xyg1[p]
                                        , period_is_prebirth_pa1e1b1nwzida0e0b0xyg1[p], nfoet_b1nwzida0e0b0xyg
                                        , days_period_pa1e1b1nwzida0e0b0xyg1[p], saa_mortalitye_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        mortality_dams += temp0 #dam mort pre-birth due to low CS and loss of condition to pre-birth.
                        cf_csc_dams = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##preg tox Dam mortality - comments about mortality functions can be found in sfun.
            eqn_group = 16
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_pregtox_cs(cb1_dams, cg_dams, nw_start_dams, ebg_dams, sd_ebg_dams
                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], saa_mortalitye_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # mu = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_pregtox_mu(cb1_dams, cg_dams, nw_start_dams, ebg_dams, sd_ebg_dams
                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_prebirthbirth_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , gest_propn_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ### Peri-natal progeny mortality (progeny survival) - comments about mortality functions can be found in sfun.
            eqn_group = 1
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)   # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2 = sfun.f_mortality_progeny_cs(cd_yatf, cb1_yatf, w_b_yatf, rc_start_dams, cv_bw_yatf
                                    , w_b_exp_y_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]
                                    , chill_index_pa1e1b1nwzida0e0b0xygp0[p], nfoet_b1nwzida0e0b0xyg
                                    , rev_trait_values['yatf'][p], sap_mortalityp_pa1e1b1nwzida0e0b0xyg2[p]
                                    , saa_mortalityx_pa1e1b1nwzida0e0b0xyg[p])
                    if eqn_used:
                        mortality_birth_yatf = temp1 #mortalityd, assign first because it has x-axis
                        mortality_birth_yatf += temp0 #mortalityx
                        mortality_dams += temp2 #mortality due to dystocia
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1
            eqn_system = 1 # MU = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)   # equation used is based on the yatf system
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    ##calculate the standard BW which is used in the paddock level scaling
                    w_b_ltw_std_yatf, t_cf = sfun.f_birthweight_mu(cu1_yatf, cb1_yatf, cg_yatf, cx_yatf[:,mask_x,...]
                                                    , ce_pyatf[:,p,...], w_b_ltw_std_yatf, 0, nw_start_dams, 0
                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                    temp0 = sfun.f_mortality_progeny_mu(cu2_yatf, cb1_yatf, cx_yatf[:,mask_x,...], ce_pyatf[:,p,...]
                                    , w_b_yatf, w_b_ltw_std_yatf, cv_bw_yatf
                                    , foo_yatf, chill_index_pa1e1b1nwzida0e0b0xygp0[p], mobsize_pa1e1b1nwzida0e0b0xyg1[p]
                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p], rev_trait_values['yatf'][p]
                                    , sap_mortalityp_pa1e1b1nwzida0e0b0xyg2[p], saa_mortalityx_pa1e1b1nwzida0e0b0xyg[p])  ##code for absolute BW
                    # temp0 = sfun.f_mortality_progeny_mu(cu2_yatf, cb1_yatf, cx_yatf[:,mask_x,...], ce_pyatf[:,p,...]
                    #                 , w_b_yatf / srw_female_yg2, w_b_ltw_std_yatf / srw_female_yg2, cv_bw_yatf
                    #                 , foo_yatf, chill_index_pa1e1b1nwzida0e0b0xygp0[p], mobsize_pa1e1b1nwzida0e0b0xyg1[p]
                    #                 , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p], rev_trait_values['yatf'][p]
                    #                 , sap_mortalityp_pa1e1b1nwzida0e0b0xyg2[p], saa_mortalityx_pa1e1b1nwzida0e0b0xyg[p])   ##code for BW/SRW
                    if eqn_used:
                        mortality_birth_yatf = temp0 #mortality
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0


            ##end numbers - accounts for mortality and other activity during the period - this is the number in the different classes as at the end of the period
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                numbers_end_sire = sfun.f1_period_end_nums(numbers_start_sire, mortality_sire, group=0)
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                numbers_end_dams = sfun.f1_period_end_nums(numbers_start_dams, mortality_dams, mortality_yatf=mortality_birth_yatf,
                             nfoet_b1=nfoet_b1nwzida0e0b0xyg, nyatf_b1=nyatf_b1nwzida0e0b0xyg, group=1, conception=conception_dams,
                             gender_propn_x=gender_propn_xyg, period_is_mating = period_is_mating_pa1e1b1nwzida0e0b0xyg1[p],
                             period_is_matingend=period_is_matingend_pa1e1b1nwzida0e0b0xyg1[p], period_is_birth = period_is_birth_pa1e1b1nwzida0e0b0xyg1[p],
                             period_isbetween_prejoinmatingend=period_isbetween_prejoinmatingend_pa1e1b1nwzida0e0b0xyg1[p],
                             propn_dams_mated=est_prop_dams_mated_pa1e1b1nwzida0e0b0xyg1[p])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                numbers_end_yatf = sfun.f1_period_end_nums(numbers_start_yatf, mortality_yatf, group=2)
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                numbers_end_offs = sfun.f1_period_end_nums(numbers_start_offs, mortality_offs, group=3)

            ##################################################
            #post calculation sensitivity for intake & energy#
            ##################################################
            ##These sensitivities alter potential intake and me intake required without altering the liveweight profile.
            ##The aim is to have the same effect as using the within function SA and then altering the feed supply to
            ##  generate the same LW profile, they just require less work to test the effect of not altering LW profile (as per old MIDAS)
            ##A change to pi_post doesn't alter the total MEI (just the quality of the diet required) therefore there
            ## is no effect on wool production and no change to wool growth efficiency.
            ##However, adjustments are required for mr_post and kg_post because these both reduce the MEI required in this section.
            ## To do this requires altering d_cfw and d_fl. These would both change in f_fibre if using the within
            ## function sa while still achieving the same FFCFW profile.
            ## If cfw and fl were not scaled it would imply an increased wool growth efficiency (and this would
            ## be inconsistent with the in function SA)

            #todo these SA will need equation groups to be implemented with the new feeding standards
            #the new feeding standards are not conducive to doing a SA on efficiency of gain. Therefore retain the kg approach

            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post_adult'], zf2_sire == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post_adult'], zf2_sire == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post_adult'], zf2_sire == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_sire = fun.f_sa(pi_sire, sam_pi)
                #### alter mei
                mei_solid_sire = mei_solid_sire + (mem_sire * sap_mr
                                                   - surplus_energy_sire * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_sire, mei_solid_sire)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_sire, mei_solid_sire)
                d_cfw_sire = d_cfw_sire / scalar_mr
                d_fl_sire = d_fl_sire / scalar_mr
                d_cfw_sire = d_cfw_sire / scalar_kg
                d_fl_sire = d_fl_sire / scalar_kg

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post_adult'], zf2_dams == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post_adult'], zf2_dams == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post_adult'], zf2_dams == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_dams = fun.f_sa(pi_dams, sam_pi)
                #### alter mei
                mei_solid_dams = mei_solid_dams + (mem_dams * sap_mr
                                                   - surplus_energy_dams * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_dams, mei_solid_dams)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_dams, mei_solid_dams)
                d_cfw_dams = d_cfw_dams / scalar_mr
                d_fl_dams = d_fl_dams / scalar_mr
                d_cfw_dams = d_cfw_dams / scalar_kg
                d_fl_dams = d_fl_dams / scalar_kg

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_pi = sen.sam['pi_post_yatf']   #potential intake
                sap_mr = sen.sap['mr_post_yatf']   #maintenance energy (MEm)
                sap_kg = sen.sap['kg_post_yatf']   #efficiency of gain (kg)
                #### alter potential intake
                pi_yatf = fun.f_sa(pi_yatf, sam_pi)
                #### alter mei (only calculate impact on mei_solid because that passes to the mei parameter in the matrix)
                #### this is an error in this application because all the change in energy is related to pasture and none to milk
                mei_solid_yatf = mei_solid_yatf + (mem_yatf * sap_mr
                                                   - surplus_energy_yatf * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change (use mei rather than mei_solid so it is change as a proportion of total mei)
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_yatf, mei_yatf)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_yatf, mei_yatf)
                d_cfw_yatf = d_cfw_yatf / scalar_mr
                d_fl_yatf = d_fl_yatf / scalar_mr
                d_cfw_yatf = d_cfw_yatf / scalar_kg
                d_fl_yatf = d_fl_yatf / scalar_kg

            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on zf2 - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post_adult'], zf2_offs == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post_adult'], zf2_offs == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post_adult'], zf2_offs == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_offs = fun.f_sa(pi_offs, sam_pi)
                #### alter mei
                mei_solid_offs = mei_solid_offs + (mem_offs * sap_mr
                                                   - surplus_energy_offs * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_offs, mei_solid_offs)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_offs, mei_solid_offs)
                d_cfw_offs = d_cfw_offs / scalar_mr
                d_fl_offs = d_fl_offs / scalar_mr
                d_cfw_offs = d_cfw_offs / scalar_kg
                d_fl_offs = d_fl_offs / scalar_kg

            ##############
            ##end values #
            ##############
            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_sire = np.maximum(0,ffcfw_start_sire + cg_sire[18, ...] * ebg_sire * days_period_pa1e1b1nwzida0e0b0xyg0[p])
                ##FFCFW maximum to date
                ffcfw_max_sire = np.maximum(ffcfw_sire, ffcfw_max_start_sire)
                ##Energy in fat, muscle, viscera, wool & conceptus
                # f_xxxx = f_start_xxxx + df_xxxx
                # m_xxxx = m_start_xxxx + dm_xxxx
                # v_xxxx = v_start_xxxx + dv_xxxx
                # w_xxxx = w_start_xxxx + dw_xxxx
                # c_xxxx = c_start_xxxx + dc_xxxx
                ##Weight of fat adipose (end)
                aw_sire = aw_start_sire + fg_sire / cg_sire[26, ...] * days_period_pa1e1b1nwzida0e0b0xyg0[p]
                ##Weight of muscle (end)
                mw_sire = mw_start_sire + pg_sire / cg_sire[27, ...] * days_period_pa1e1b1nwzida0e0b0xyg0[p]
                ##Weight of bone (end)	bw #todo formula needs finishing
                bw_sire = bw_start_sire
                ##Weight of water (end)
                ww_sire = mw_sire * (1 - cg_sire[27, ...]) + aw_sire * (1 - cg_sire[26, ...])
                ##Weight of gutfill (end)
                gw_sire = ffcfw_sire * (1 - 1 / cg_sire[18, ...])
                ##Clean fleece weight (end)
                cfw_sire = cfw_start_sire + d_cfw_sire * days_period_pa1e1b1nwzida0e0b0xyg0[p] * cfw_propn_yg0
                ##Greasy fleece weight (end)
                gfw_sire = cfw_sire / cw_sire[3, ...]
                ##LW with conceptus and fleece (end)
                lw_sire = ffcfw_sire + gfw_sire
                ##Fibre length since shearing (end)
                fl_sire = fl_start_sire + d_fl_sire * days_period_pa1e1b1nwzida0e0b0xyg0[p]
                ##Average FD since shearing (end)
                fd_sire = (fl_start_sire * fd_start_sire + d_fl_sire * days_period_pa1e1b1nwzida0e0b0xyg0[p] * d_fd_sire) / fl_sire
                ##Minimum FD since shearing (end)
                fd_min_sire = np.minimum(fd_min_start_sire, d_fd_sire)
                ##Staple length if shorn(end)
                sl_sire = (fl_sire - fl_shear_yg0) * cw_sire[15, ...]
                ##Staple strength if shorn(end)
                ss_sire = fd_min_sire**2 / fd_sire **2 * cw_sire[16, ...]



            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_dams = np.maximum(0,ffcfw_start_dams + cg_dams[18, ...] * ebg_dams * days_period_pa1e1b1nwzida0e0b0xyg1[p])
                ##FFCFW maximum to date
                ffcfw_max_dams = np.maximum(ffcfw_dams, ffcfw_max_start_dams)
                ##Energy in fat, muscle, viscera, wool & conceptus
                # f_xxxx = f_start_xxxx + df_xxxx
                # m_xxxx = m_start_xxxx + dm_xxxx
                # v_xxxx = v_start_xxxx + dv_xxxx
                # w_xxxx = w_start_xxxx + dw_xxxx
                # c_xxxx = c_start_xxxx + dc_xxxx
                ##Weight of fat adipose (end)
                aw_dams = aw_start_dams + fg_dams / cg_dams[26, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p]
                ##Weight of muscle (end)
                mw_dams = mw_start_dams + pg_dams / cg_dams[27, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p]
                ##Weight of bone (end)	bw #todo formula needs finishing
                bw_dams = bw_start_dams
                ##Weight of water (end)
                ww_dams = mw_dams * (1 - cg_dams[27, ...]) + aw_dams * (1 - cg_dams[26, ...])
                ##Weight of gutfill (end)
                gw_dams = ffcfw_dams* (1 - 1 / cg_dams[18, ...])
                ##Whole body energy (calculated from muscle and adipose weight)
                wbe_dams = sfun.f_wbe_mu(cg_dams, aw_dams, mw_dams)
                ##Clean fleece weight (end)
                cfw_dams = cfw_start_dams + d_cfw_dams * days_period_pa1e1b1nwzida0e0b0xyg1[p] * cfw_propn_yg1
                ##Greasy fleece weight (end)
                gfw_dams = cfw_dams / cw_dams[3, ...]
                ##LW with conceptus and fleece (end)
                lw_dams = ffcfw_dams + guw_dams + gfw_dams
                ##Fibre length since shearing (end)
                fl_dams = fl_start_dams + d_fl_dams * days_period_pa1e1b1nwzida0e0b0xyg1[p]
                ##Average FD since shearing (end)
                fd_dams = (fl_start_dams * fd_start_dams + d_fl_dams * days_period_pa1e1b1nwzida0e0b0xyg1[p] * d_fd_dams) / fl_dams
                ##Minimum FD since shearing (end)
                fd_min_dams = np.minimum(fd_min_start_dams, d_fd_dams)
                ##Staple length if shorn(end)
                sl_dams = (fl_dams - fl_shear_yg1) * cw_dams[15, ...]
                ##Staple strength if shorn(end)
                ss_dams = fd_min_dams ** 2 / fd_dams ** 2 * cw_dams[16, ...]


            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_yatf = np.maximum(0, ffcfw_start_yatf + cg_yatf[18, ...] * ebg_yatf * days_period_pa1e1b1nwzida0e0b0xyg2[p])
                ##FFCFW maximum to date
                ffcfw_max_yatf = np.maximum(ffcfw_yatf, ffcfw_max_start_yatf)
                ##Energy in fat, muscle, viscera, wool & conceptus
                # f_xxxx = f_start_xxxx + df_xxxx
                # m_xxxx = m_start_xxxx + dm_xxxx
                # v_xxxx = v_start_xxxx + dv_xxxx
                # w_xxxx = w_start_xxxx + dw_xxxx
                # c_xxxx = c_start_xxxx + dc_xxxx
                ##Weight of fat adipose (end)
                aw_yatf = aw_start_yatf + fg_yatf / cg_yatf[26, ...] * days_period_pa1e1b1nwzida0e0b0xyg2[p]
                ##Weight of muscle (end)
                mw_yatf = mw_start_yatf + pg_yatf / cg_yatf[27, ...] * days_period_pa1e1b1nwzida0e0b0xyg2[p]
                ##Weight of bone (end)	bw #todo formula needs finishing
                bw_yatf = bw_start_yatf
                ##Weight of water (end)
                ww_yatf = mw_yatf * (1 - cg_yatf[27, ...]) + aw_yatf * (1 - cg_yatf[26, ...])
                ##Weight of gutfill (end)
                gw_yatf = ffcfw_yatf * (1 - 1 / cg_yatf[18, ...])
                ##Clean fleece weight (end)
                cfw_yatf = cfw_start_yatf + d_cfw_yatf * days_period_pa1e1b1nwzida0e0b0xyg2[p] * cfw_propn_yg2
                ##Greasy fleece weight (end)
                gfw_yatf = cfw_yatf / cw_yatf[3, ...]
                ##LW with conceptus and fleece (end)
                lw_yatf = ffcfw_yatf + gfw_yatf
                ##Fibre length since shearing (end)
                fl_yatf = fl_start_yatf + d_fl_yatf * days_period_pa1e1b1nwzida0e0b0xyg2[p]
                ##Average FD since shearing (end)
                fd_yatf = (fl_start_yatf * fd_start_yatf + d_fl_yatf * days_period_pa1e1b1nwzida0e0b0xyg2[p] * d_fd_yatf) / fl_yatf #d_fd is actually the fd of the weeks growth. Not the change in fd.
                ##Minimum FD since shearing (end)
                fd_min_yatf = np.minimum(fd_min_start_yatf, d_fd_yatf) #d_fd is actually the fd of the weeks growth. Not the change in fd.
                ##Staple length if shorn(end)
                sl_yatf = (fl_yatf - fl_shear_yg2) * cw_yatf[15, ...]
                ##Staple strength if shorn(end)
                ss_yatf = fun.f_divide(fd_min_yatf ** 2 , fd_yatf ** 2 * cw_yatf[16, ...])

            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_offs = np.maximum(0,ffcfw_start_offs + cg_offs[18, ...] * ebg_offs * days_period_pa1e1b1nwzida0e0b0xyg3[p])
                ##FFCFW maximum to date
                ffcfw_max_offs = np.maximum(ffcfw_offs, ffcfw_max_start_offs)
                ##Energy in fat, muscle, viscera, wool & conceptus
                # f_xxxx = f_start_xxxx + df_xxxx
                # m_xxxx = m_start_xxxx + dm_xxxx
                # v_xxxx = v_start_xxxx + dv_xxxx
                # w_xxxx = w_start_xxxx + dw_xxxx
                # c_xxxx = c_start_xxxx + dc_xxxx
                ##Weight of fat adipose (end)
                aw_offs = aw_start_offs + fg_offs / cg_offs[26, ...] * days_period_pa1e1b1nwzida0e0b0xyg3[p]
                ##Weight of muscle (end)
                mw_offs = mw_start_offs + pg_offs / cg_offs[27, ...] * days_period_pa1e1b1nwzida0e0b0xyg3[p]
                ##Weight of bone (end)	bw #todo formula needs finishing
                bw_offs = bw_start_offs
                ##Weight of water (end)
                ww_offs = mw_offs * (1 - cg_offs[27, ...]) + aw_offs * (1 - cg_offs[26, ...])
                ##Weight of gutfill (end)
                gw_offs = ffcfw_offs* (1 - 1 / cg_offs[18, ...])
                ##Whole body energy (end - calculated from muscle and adipose weight)
                wbe_offs = sfun.f_wbe_mu(cg_offs, aw_offs, mw_offs)
                ##Clean fleece weight (end)
                cfw_offs = cfw_start_offs + d_cfw_offs * days_period_pa1e1b1nwzida0e0b0xyg3[p] * cfw_propn_yg3
                ##Greasy fleece weight (end)
                gfw_offs = cfw_offs / cw_offs[3, ...]
                ##LW with conceptus and fleece (end)
                lw_offs = ffcfw_offs + gfw_offs
                ##Fibre length since shearing (end)
                fl_offs = fl_start_offs + d_fl_offs * days_period_pa1e1b1nwzida0e0b0xyg3[p]
                ##Average FD since shearing (end)
                fd_offs = (fl_start_offs * fd_start_offs + d_fl_offs * days_period_pa1e1b1nwzida0e0b0xyg3[p] * d_fd_offs) / fl_offs
                ##Minimum FD since shearing (end)
                fd_min_offs = np.minimum(fd_min_start_offs, d_fd_offs)
                ##Staple length if shorn(end)
                sl_offs = (fl_offs - fl_shear_yg3) * cw_offs[15, ...]
                ##Staple strength if shorn(end)
                ss_offs = fd_min_offs ** 2 / fd_offs ** 2 * cw_offs[16, ...]

            ######################################
            #store postprocessing and report vars#
            ######################################
            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p, ...] > 0):
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort (no feedlot mask for sires (only for offs) because feedlotting sires doesn't indicate they are being sold).
                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_sire = np.broadcast_to(numbers_start_condense_sire, numbers_end_sire.shape) #required for the first condensing because condense numbers start doesn't have all the axis.
                surv_sire = fun.f_divide(np.sum(numbers_end_sire,axis=prejoin_tup + (season_tup,), keepdims=True)
                                         , np.sum(numbers_start_condense_sire, axis=prejoin_tup + (season_tup,), keepdims=True))  # sum e,b,z axis because numbers are distributed along those axis so need to sum to determine if w has mortality > 10%
                threshold = np.minimum(0.9, fun.f_divide(surv_sire.sum(w_pos,keepdims=True),(surv_sire!=0).sum(w_pos,keepdims=True)))  # threshold is the lower of average survival and 90% (animals with 0 survival are not included)
                mort_mask_sire = surv_sire >= threshold

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_sire = mort_mask_sire

                ###sorted index of w. used for condensing and used below.
                idx_sorted_w_sire = np.argsort(ffcfw_sire, axis=w_pos)

                ###mask with a true for the z and w slices with the lightest animal
                mask_min_lw_wz_sire = np.isclose(ffcfw_sire, np.min(ffcfw_sire, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the lightest animal after taking the weighted average across z
                t_ffcfw_sire = fun.f_weighted_average(ffcfw_sire, numbers_end_sire, season_tup, keepdims=True, non_zero=True)
                mask_min_wa_lw_w_sire = np.isclose(t_ffcfw_sire, np.min(t_ffcfw_sire, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw

                ###mask with a true for the z and w slices with the heaviest animal
                mask_max_lw_wz_sire = np.isclose(ffcfw_sire, np.max(ffcfw_sire, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the heaviest animal after taking the weighted average across z
                t_ffcfw_sire = fun.f_weighted_average(ffcfw_sire, numbers_end_sire, season_tup, keepdims=True, non_zero=True)
                mask_max_wa_lw_w_sire = np.isclose(t_ffcfw_sire, np.max(t_ffcfw_sire, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post-processing
                o_numbers_start_tpsire[:,p] = numbers_start_sire
                o_numbers_end_tpsire[:,p] = numbers_end_sire
                o_ffcfw_tpsire[:,p] = ffcfw_sire
                o_lw_tpsire[:,p] = lw_sire
                o_pi_tpsire[:,p] = pi_sire
                o_mei_solid_tpsire[:,p] = mei_solid_sire
                o_ch4_animal_tpsire[:,p] = ch4_animal_sire
                o_n2o_animal_tpsire[:,p] = n2o_animal_sire
                o_cfw_tpsire[:,p] = cfw_sire
                o_sl_tpsire[:,p] = sl_sire
                o_fd_tpsire[:,p] = fd_sire
                o_fd_min_tpsire[:,p] = fd_min_sire
                o_ss_tpsire[:,p] = ss_sire
                o_rc_start_tpsire[:,p] = rc_start_sire
                o_ebg_tpsire[:,p] = ebg_sire

                ###store report variables for dams - individual variables can be deleted if not needed - store in report dictionary in the report section at end of this module
                o_nw_start_tpsire[:,p] = nw_start_sire

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort  (no feedlot mask for dams (only for offs) because feedlotting dams doesn't indicate they are being sold).
                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_dams = np.broadcast_to(numbers_start_condense_dams, numbers_end_dams.shape) #required for the first condensing because condense numbers start doesn't have all the axis.
                surv_dams = fun.f_divide(np.sum(numbers_end_dams,axis=prejoin_tup + (season_tup,), keepdims=True)
                                         , np.sum(numbers_start_condense_dams, axis=prejoin_tup + (season_tup,), keepdims=True))  # sum e,b,z axis because numbers are distributed along those axes so need to sum to determine if w has mortality > 10%
                threshold = np.minimum(0.9, fun.f_divide(surv_dams.sum(w_pos,keepdims=True),(surv_dams!=0).sum(w_pos,keepdims=True)))  # threshold is the lower of average survival and 90% (animals with 0 survival are not included)
                mort_mask_dams = surv_dams >= threshold

                ###print warning if min mort is greater than 10% since the previous condense
                ###this is to ensure we are condensing to an animal that the lp will select (ie not point having an animal that has more than 10% mort)
                ### Note 1: if ewe lambs FS is not set up for mating and it is estimated that ewe lambs are mated then a warning is likely to be triggered. No warning will be triggered if estimate mating propn is 0 because only a very small number of animals will be in the mated b slices and therefore because the b axis is summed to build surv_dams mort won't be significantly effected by them.
                if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]):
                    min_mort = 1 - np.max(surv_dams, axis=w_pos)
                    ####only use the retained t slice (animals that have multiple fvps per dvp and are sold in the first fvp only get medium fs in following fvps due to lw clustering e.g. w9 is high-medium-medium, so this can trigger unwanted mort warning)
                    if len_gen_t1 > 1:
                        min_mort = min_mort[a_t_g1]
                    if np.any(min_mort > 0.1):
                        print('WARNING: HIGH MORTALITY DAMS: period ', p)

                ###combine mort and feedlot mask  - True means the w slice is included in condensing. Currently dams
                ### that go into the feedlot are used to create condensed animal because it is common to feedlot/confine retained dams at the start of the season.
                condense_w_mask_dams = mort_mask_dams

                ###sorted index of w. used for condensing.
                idx_sorted_w_dams = np.argsort(ffcfw_dams * condense_w_mask_dams, axis=w_pos)

                ###When using the pkl condensed values there may be cases when they do not have enough spread (e.g.
                ### the generated condensed animal is heavier than the pkl condensed animal this would result in weight vanishing in the distribution)
                ### in these cases the pkl condense values are overwritten by the generated condensed values.
                #### controls if the generated condensed values are used. False means pkl_condensed_values are used.
                mask_gen_condensed_used_dams = sfun.f1_gen_condensed_used(ffcfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                                                  , n_fs_dams, len_w1, len_t1, n_fvps_percondense_dams
                                                                  , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                                                  , adjp_lw_initial_wzida0e0b0xyg1
                                                                  , pkl_condensed_values['dams'][p], 'o_ffcfw_dams')
                    
                ###mask with a true for the z and w slices with the lightest animal
                mask_min_lw_wz_dams = np.isclose(ffcfw_dams, np.min(ffcfw_dams, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the lightest animal after taking the weighted average across z
                t_ffcfw_dams = fun.f_weighted_average(ffcfw_dams, numbers_end_dams, season_tup, keepdims=True, non_zero=True)
                mask_min_wa_lw_w_dams = np.isclose(t_ffcfw_dams, np.min(t_ffcfw_dams, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw

                ###mask with a true for the z and w slices with the heaviest animal
                mask_max_lw_wz_dams = np.isclose(ffcfw_dams, np.max(ffcfw_dams, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the heaviest animal after taking the weighted average across z
                t_ffcfw_dams = fun.f_weighted_average(ffcfw_dams, numbers_end_dams, season_tup, keepdims=True, non_zero=True)
                mask_max_wa_lw_w_dams = np.isclose(t_ffcfw_dams, np.max(t_ffcfw_dams, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post-processing
                o_mortality_dams[:,p] = mortality_dams #has to be stored before back dating numbers
                o_numbers_start_tpdams[:,p] = numbers_start_dams
                o_numbers_end_tpdams[:,p] = numbers_end_dams

                ###back date numbers from end of mating to prejoining
                if np.any(period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]): #need to back date the numbers from conception to prejoining because otherwise in the matrix there is not a dvp between prejoining and mating therefore this is required so that other slices have energy etc. requirement
                    ###period is between prejoining and the end of current period (testing this for each p slice)
                    between_prejoinnow = sfun.f1_period_is_('period_is_between', date_prejoin_pa1e1b1nwzida0e0b0xyg1[p], date_start_pa1e1b1nwzida0e0b0xyg, date_end_pa1e1b1nwzida0e0b0xyg[p], date_end_pa1e1b1nwzida0e0b0xyg)

                    ###scale numbers at the end of mating (to account for mortality) to each period (note only the periods between prejoining and end of mating get used)
                    cum_mortality_dams = np.flip(np.cumprod(np.flip(1-o_mortality_dams, axis=p_pos), axis=p_pos), axis=p_pos)
                    t_scaled_start_numbers = numbers_end_dams[:, na] / cum_mortality_dams
                    t_scaled_end_numbers = numbers_end_dams[:, na] / np.roll(cum_mortality_dams, shift=-1, axis=p_pos)

                    ###if period is mating back date the end number after mating to all the periods since prejoining
                    o_numbers_end_tpdams = fun.f_update(o_numbers_end_tpdams, t_scaled_end_numbers.astype(dtype)
                                                        , (period_is_matingend_pa1e1b1nwzida0e0b0xyg1[p] * between_prejoinnow))
                    o_numbers_start_tpdams = fun.f_update(o_numbers_start_tpdams, t_scaled_start_numbers.astype(dtype)
                                                         , (period_is_matingend_pa1e1b1nwzida0e0b0xyg1[p] * between_prejoinnow))
                o_ffcfw_tpdams[:,p] = ffcfw_dams
                o_ffcfw_season_tpdams[:,p] = sfun.f1_season_wa(numbers_end_dams, ffcfw_dams, season_tup, mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1])
                o_ffcfw_condensed_tpdams[:,p] = sfun.f1_condensed(ffcfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                                              , n_fs_dams, len_w1, n_fvps_percondense_dams
                                                              , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                                              , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'o_ffcfw_dams')  #condensed lw at the end of the period
                o_nw_start_tpdams[:,p] = nw_start_dams
                numbers_join_dams = fun.f_update(numbers_join_dams, numbers_start_dams, period_is_join_pa1e1b1nwzida0e0b0xyg1[p])
                o_numbers_join_tpdams[:,p] = numbers_join_dams #store the numbers at joining until next
                o_lw_tpdams[:,p] = lw_dams
                o_pi_tpdams[:,p] = pi_dams
                o_mei_solid_tpdams[:,p] = mei_solid_dams
                o_ch4_animal_tpdams[:,p] = ch4_animal_dams
                o_n2o_animal_tpdams[:,p] = n2o_animal_dams
                o_cfw_tpdams[:,p] = cfw_dams
                # o_gfw_tpdams[:,p] = gfw_dams
                o_sl_tpdams[:,p] = sl_dams
                o_fd_tpdams[:,p] = fd_dams
                o_fd_min_tpdams[:,p] = fd_min_dams
                o_ss_tpdams[:,p] = ss_dams
                o_n_sire_tpa1e1b1nwzida0e0b0xyg1g0p8[:,p] = n_sire_a1e1b1nwzida0e0b0xyg1g0p8
                o_rc_start_tpdams[:,p] = rc_start_dams
                o_ebg_tpdams[:,p] = ebg_dams
                o_cfw_ltwadj_tpdams[:,p] = cfw_ltwadj_dams
                o_fd_ltwadj_tpdams[:,p] = fd_ltwadj_dams

                ###store report variables for dams - individual variables can be deleted if not needed - store in report dictionary in the report section at end of this module
                r_intake_f_tpdams[:,p] = intake_f_dams
                r_md_solid_tpdams[:,p] = md_solid_dams
                r_foo_tpdams[:,p] = foo_dams
                r_dmd_tpdams[:,p] = dmd_dams
                r_evg_tpdams[:,p] = evg_dams
                r_mp2_tpdams[:,p] = mp2_dams
                r_d_cfw_tpdams[:,p] = d_cfw_dams
                r_wbe_tpdams[:,p] = wbe_dams


            ###yatf
            o_ffcfw_start_tpyatf[:,p] = ffcfw_start_yatf #use ffcfw_start because weaning start of period, has to be outside of the 'if' because days per period = 0 when weaning occurs (because once they are weaned they are not yatf therefore 0 days per period) because weaning is first day of period. But we need to know the start ffcfw.
            o_numbers_start_tpyatf[:,p] = numbers_start_yatf #used for prog calculations - use numbers start because weaning is start of period - has to be out of the 'if' because there is 0 days in the period when weaning occurs but we still want to store the start numbers (because once they are weaned they are not yatf therefore 0 days per period)
            o_rc_start_tpyatf[:,p] = rc_start_yatf #outside because used for sale value which is weaning which has 0 days per period because weaning is first day (this means the rc at weaning is actually the rc at the start of the previous period because it doesn't recalculate once days per period goes to 0) (because once they are weaned they are not yatf therefore 0 days per period)
            o_wean_w_tpyatf[:, p] = w_w_yatf #outside the if statement because the days_period_yatf are 0 in the weaning period because weaning is at the start of the period
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort or have been in the feedlot.
                ### The logic behind this is that the model will not want to select animals with greater than 10% mort so not point using them to determine condensed weights
                ### and animals that have been in the feed lot will have been sold therefore it is not useful to include these animal in the condensing because
                ### this will increase the condensing weight but all the heavy animals were sold so the high condense weight becomes too high for many animals to distribute into and hence is a waste.
                no_confinement_yatf = np.all(np.logical_not(confinementw_tpa1e1b1nwzida0e0b0xyg1[:,0:p]), axis=p_pos)  # True if animal has never been in feedlot
                ###if all nut spread options include confinement then confinement w must be included in condensing.
                ### ie if all n is confinement then overwrite no_confinement to True
                no_confinement_yatf = np.logical_or(np.all(bool_confinement_g1_n), no_confinement_yatf)

                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_yatf = np.broadcast_to(numbers_start_condense_yatf, numbers_end_yatf.shape) #required for the first condensing because condense numbers start doesn't have all the axis.
                surv_yatf = fun.f_divide(np.sum(numbers_end_yatf,axis=season_tup, keepdims=True)
                                        , np.sum(numbers_start_condense_yatf, axis=season_tup, keepdims=True))  # sum z axis because numbers are distributed along z axis so need to sum to determine if w has mortality > 10% (don't sum e&b because yatf stay in the same slice)
                threshold = np.minimum(0.9, fun.f_divide(surv_yatf.sum(w_pos,keepdims=True),(surv_yatf!=0).sum(w_pos,keepdims=True)))  # threshold is the lower of average survival and 90% (animals with 0 survival are not included)
                mort_mask_yatf = surv_yatf >= threshold

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_yatf = np.logical_and(no_confinement_yatf, mort_mask_yatf)

                ###sorted index of w. used for condensing.
                idx_sorted_w_yatf = np.argsort(ffcfw_yatf * condense_w_mask_yatf, axis=w_pos)
                
                ###When using the pkl condensed values there may be cases when they do not have enough spread (e.g.
                ### the generated condensed animal is heavier than the pkl condensed animal this would result in weight vanishing in the distribution)
                ### in these cases the pkl condense values are overwritten by the generated condensed values.
                #### controls if the generated condensed values are used. False means pkl_condensed_values are used.
                mask_gen_condensed_used_yatf = sfun.f1_gen_condensed_used(ffcfw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                                                  , n_fs_dams, len_w2, len_t2, n_fvps_percondense_dams
                                                                  , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                                                  , adjp_lw_initial_wzida0e0b0xyg1 #use dams because yatf are initial dams
                                                                  , pkl_condensed_values['yatf'][p], 'o_ffcfw_yatf')

                ###mask with a true for the z and w slices with the lightest animal
                mask_min_lw_wz_yatf = np.isclose(ffcfw_yatf, np.min(ffcfw_yatf, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the lightest animal after taking the weighted average across z
                t_ffcfw_yatf = fun.f_weighted_average(ffcfw_yatf, numbers_end_yatf, season_tup, keepdims=True, non_zero=True)
                mask_min_wa_lw_w_yatf = np.isclose(ffcfw_yatf, np.min(t_ffcfw_yatf, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw

                ###mask with a true for the z and w slices with the heaviest animal
                mask_max_lw_wz_yatf = np.isclose(ffcfw_yatf, np.max(ffcfw_yatf, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the heaviest animal after taking the weighted average across z
                t_ffcfw_yatf = fun.f_weighted_average(ffcfw_yatf, numbers_end_yatf, season_tup, keepdims=True, non_zero=True)
                mask_max_wa_lw_w_yatf = np.isclose(t_ffcfw_yatf, np.max(t_ffcfw_yatf, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw


                ###store output variables for the post-processing
                o_pi_tpyatf[:,p] = pi_yatf
                o_mei_solid_tpyatf[:,p] = mei_solid_yatf
                o_ch4_animal_tpyatf[:,p] = ch4_animal_yatf
                o_n2o_animal_tpyatf[:,p] = n2o_animal_yatf
                # o_cfw_tpyatf[:,p] = cfw_yatf
                # o_gfw_tpyatf[:,p] = gfw_yatf
                # o_sl_tpyatf[:,p] = sl_yatf
                # o_fd_tpyatf[:,p] = fd_yatf
                # o_fd_min_tpyatf[:,p] = fd_min_yatf
                # o_ss_tpyatf[:,p] = ss_yatf

                ###store report variables - individual variables can be deleted if not needed - store in report dictionary in the report section at end of this module
                #### store a report version of ffcfw_yatf
                ####use ffcfw_start to get birth weight which is at the start of the period.
                #### Need a separate variable to o_ffcfw_start_tpyatf because that variable is non-zero all year and this affects the reported birth weight when averaging across the e & b axes
                #### Store a zero value if yatf don't exist for this slice (e1 or i)
                r_ffcfw_start_tpyatf[:,p] = ffcfw_start_yatf * (days_period_pa1e1b1nwzida0e0b0xyg2[p,...] > 0)
                r_ebg_tpyatf[:,p] = ebg_yatf
                r_evg_tpyatf[:,p] = evg_yatf
                r_mp2_tpyatf[:,p] = mp2_yatf
                r_mem_tpyatf[:,p] = mem_yatf
                r_mei_tpyatf[:,p] = mei_yatf
                r_mei_solid_tpyatf[:,p] = mei_solid_yatf
                r_propn_solid_tpyatf[:,p] = mei_propn_herb_yatf
                r_pi_tpyatf[:,p] = pi_yatf
                r_kg_tpyatf[:,p] = kg_yatf
                r_intake_f_tpyatf[:,p] = intake_f_yatf
                r_nw_start_tpyatf[:,p] = nw_start_yatf



            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort or have been in the feedlot.
                ### The logic behind this is that the model will not want to select animals with greater than 10% mort so not point using them to determine condensed weights
                ### and animals that have been in the feed lot will have been sold therefore it is not useful to include these animal in the condensing because
                ### this will increase the condensing weight but all the heavy animals were sold so the high condense weight becomes too high for many animals to distribute into and hence is a waste.
                no_confinement_offs = np.all(np.logical_not(confinementw_tpa1e1b1nwzida0e0b0xyg3[:,0:p]), axis=p_pos)  # True if animal has never been in feedlot
                ###if all nut spread options include confinement then confinement w must be included in condensing.
                ### ie if all n is confinement then overwrite no_confinement to True
                no_confinement_offs = np.logical_or(np.all(bool_confinement_g3_n), no_confinement_offs)

                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_offs = np.broadcast_to(numbers_start_condense_offs, numbers_end_offs.shape) #required for the first condensing because condense numbers start doesn't have all the axis.
                surv_offs = fun.f_divide(np.sum(numbers_end_offs,axis=season_tup, keepdims=True)
                                         , np.sum(numbers_start_condense_offs, axis=season_tup, keepdims=True))  # sum z axis because numbers are distributed along those axis so need to sum to determine if w has mortality > 10% (don't sum e&b because offs don't change slice)
                threshold = np.minimum(0.9, fun.f_divide(surv_offs.sum(w_pos,keepdims=True),(surv_offs!=0).sum(w_pos,keepdims=True))) #threshold is the lower of average survival and 90% (animals with 0 survival are not included)
                mort_mask_offs = surv_offs >= threshold

                ###print warning if min mort is greater than 10% since the previous condense
                if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]):
                    min_mort = 1 - np.max(surv_offs, axis=w_pos)
                    ####only use the retained t slice because if there is a dvp that spans two fvp and the animal is sold in
                    #### the first fvp then the fs may not be good in the second fvp (because the w are clustered e.g. w9 is high fs in the first fvp followed by medium)
                    if len_gen_t3 > 1:
                        min_mort = min_mort[0]
                    if np.any(min_mort > 0.1):
                        print('WARNING: HIGH MORTALITY OFFS: period ', p)

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_offs = np.logical_and(no_confinement_offs, mort_mask_offs)

                ###sorted index of w. used for condensing.
                idx_sorted_w_offs = np.argsort(ffcfw_offs * condense_w_mask_offs, axis=w_pos) #set animal

                ###When using the pkl condensed values there may be cases when they do not have enough spread (e.g.
                ### the generated condensed animal is heavier than the pkl condensed animal this would result in weight vanishing in the distribution)
                ### in these cases the pkl condense values are overwritten by the generated condensed values.
                #### controls if the generated condensed values are used. False means pkl_condensed_values are used.
                mask_gen_condensed_used_offs = sfun.f1_gen_condensed_used(ffcfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                                                  , n_fs_offs, len_w3, len_t3, n_fvps_percondense_offs
                                                                  , period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                                                  , adjp_lw_initial_wzida0e0b0xyg3
                                                                  , pkl_condensed_values['offs'][p], 'o_ffcfw_offs')

                ###mask with a true for f1_season_wa and f_start_prod to identify w & z slices with the lightest animal
                mask_min_lw_wz_offs = np.isclose(ffcfw_offs, np.min(ffcfw_offs, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the lightest animal after taking the weighted average across z
                t_ffcfw_offs = fun.f_weighted_average(ffcfw_offs, numbers_end_offs, season_tup, keepdims=True, non_zero=True)
                mask_min_wa_lw_w_offs = np.isclose(t_ffcfw_offs, np.min(t_ffcfw_offs, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw
                #handle cases where multiple w slices have the same weight (we only want one true per w axis)

                ###mask with a true for f1_season_wa and f_start_prod to identify w & z slices with the heaviest animal
                mask_max_lw_wz_offs = np.isclose(ffcfw_offs, np.max(ffcfw_offs, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw
                ###mask with a true for the w slice with the heaviest animal after taking the weighted average across z
                t_ffcfw_offs = fun.f_weighted_average(ffcfw_offs, numbers_end_offs, season_tup, keepdims=True, non_zero=True)
                mask_max_wa_lw_w_offs = np.isclose(t_ffcfw_offs, np.max(t_ffcfw_offs, axis=w_pos, keepdims=True)) #use isclose in case small rounding error in lw


                ###store output variables for the post-processing
                o_numbers_start_tpoffs[:,p] = numbers_start_offs
                o_numbers_end_tpoffs[:,p] = numbers_end_offs
                o_ffcfw_tpoffs[:,p] = ffcfw_offs
                o_ffcfw_season_tpoffs[:,p] = sfun.f1_season_wa(numbers_end_offs, ffcfw_offs, season_tup, mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                                               , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1])
                o_ffcfw_condensed_tpoffs[:,p] = sfun.f1_condensed(ffcfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                                              , n_fs_offs, len_w3, n_fvps_percondense_offs
                                                              , period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                                              , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'o_ffcfw_offs')  #condensed lw at the end of the period before fvp0
                o_nw_start_tpoffs[:,p] = nw_start_offs
                o_mortality_offs[:,p] = mortality_offs
                o_lw_tpoffs[:,p] = lw_offs
                o_pi_tpoffs[:,p] = pi_offs
                o_mei_solid_tpoffs[:,p] = mei_solid_offs
                o_ch4_animal_tpoffs[:,p] = ch4_animal_offs
                o_n2o_animal_tpoffs[:,p] = n2o_animal_offs
                o_cfw_tpoffs[:,p] = cfw_offs
                # o_gfw_tpoffs[:,p] = gfw_offs
                o_sl_tpoffs[:,p] = sl_offs
                o_fd_tpoffs[:,p] = fd_offs
                o_fd_min_tpoffs[:,p] = fd_min_offs
                o_ss_tpoffs[:,p] = ss_offs
                o_rc_start_tpoffs[:,p] = rc_start_offs
                o_ebg_tpoffs[:,p] = ebg_offs

                ###store report variables - individual variables can be deleted if not needed - store in report dictionary in the report section at end of this module
                r_wbe_tpoffs[:,p] = wbe_offs

            ################
            #stubble resets#
            ################
            ##if generating for stubble the starting animal needs to be selected each loop. The start animal is the
            ## animal that has the closest weight to the animals in the paddock trial.
            ## This is because the fs is fixed in here but not in the trial e.g. in here a sheep gets the same DMD
            ## for the whole time but in the paddock it starts high and gets lower. They get the same DMD in here
            ## because we are simulating sheep production at a range of dmd but we want the same starting animal each period
            if stubble:
                trial_lw = stubble['lw'][p+1]
                ###get the w slice which has the closest lw to the trial - this is used to determine the production of the starting animal next period.
                stub_lw_idx_dams = np.expand_dims(np.abs(lw_dams - trial_lw).argmin(axis=w_pos),w_pos)
                stub_lw_idx_offs = np.expand_dims(np.abs(lw_offs - trial_lw).argmin(axis=w_pos),w_pos)
            else:
                stub_lw_idx_dams = np.array(np.nan)
                stub_lw_idx_offs = np.array(np.nan)

            ###########################
            #stuff for next period    #
            ###########################
            '''
            What is done in following section (not done every period):
                1. W axis can be condensed
                2. the e and b axis can be combined
                3. the z axis can be combined
            The order of the following stuff DOES matter.
                1. production needs to be condensed (w handling) using non condensed end numbers.
                2. production for start of next period (z, e & b axis handling) needs to be calculated using condensed end numbers 
            '''

            ##condensing - this requires end number (that have NOT been condensed)
            ###sire - currently not condensed because only one dvp but code exists in case we add the detail later.
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_sire = sfun.f1_condensed(ffcfw_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###nw (condense - normal weight)	- yes this is meant to be updated from nw_start
                nw_start_condensed_sire = sfun.f1_condensed(nw_start_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###FFCFW maximum to date
                ffcfw_max_condensed_sire = sfun.f1_condensed(ffcfw_max_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Weight of adipose (condense)
                aw_condensed_sire = sfun.f1_condensed(aw_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Weight of muscle (condense)
                mw_condensed_sire = sfun.f1_condensed(mw_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Weight of bone (condense)
                bw_condensed_sire = sfun.f1_condensed(bw_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g0 = sfun.f1_condensed(omer_history_sire, idx_sorted_w_sire[na,...], condense_w_mask_sire[na,...]
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)  #increment the p slice, note this doesn't impact the p loop - this is required for the next section because we are calculating the production and numbers for the start of the next period.
                ###Clean fleece weight (condense)
                cfw_condensed_sire = sfun.f1_condensed(cfw_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g0 = sfun.f1_condensed(d_cfw_history_sire_p2, idx_sorted_w_sire[na,...], condense_w_mask_sire[na,...]
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Fibre length since shearing (condense)
                fl_condensed_sire = sfun.f1_condensed(fl_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Average FD since shearing (condense)
                fd_condensed_sire = sfun.f1_condensed(fd_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)
                ###Minimum FD since shearing (condense)
                fd_min_condensed_sire = sfun.f1_condensed(fd_min_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_dams = sfun.f1_condensed(ffcfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'ffcfw_dams')
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_dams = sfun.f1_condensed(nw_start_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'nw_start_dams')
                ###FFCFW maximum to date
                ffcfw_max_condensed_dams = sfun.f1_condensed(ffcfw_max_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'ffcfw_max_dams')
                ###Weight of adipose (condense)
                aw_condensed_dams = sfun.f1_condensed(aw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'aw_dams')
                ###Weight of muscle (condense)
                mw_condensed_dams = sfun.f1_condensed(mw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'mw_dams')
                ###Weight of bone (condense)
                bw_condensed_dams = sfun.f1_condensed(bw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'bw_dams')
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g1 = sfun.f1_condensed(omer_history_dams, idx_sorted_w_dams[na,...]
                                        , condense_w_mask_dams[na,...], n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'omer_history_dams')
                ###Clean fleece weight (condense)
                cfw_condensed_dams = sfun.f1_condensed(cfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cfw_dams')
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g1 = sfun.f1_condensed(d_cfw_history_dams_p2, idx_sorted_w_dams[na,...]
                                        , condense_w_mask_dams[na,...], n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'d_cfw_history_dams_p2')
                ###Fibre length since shearing (condense)
                fl_condensed_dams = sfun.f1_condensed(fl_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'fl_dams')
                ###Average FD since shearing (condense)
                fd_condensed_dams = sfun.f1_condensed(fd_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'fd_dams')
                ###Minimum FD since shearing (condense)
                fd_min_condensed_dams = sfun.f1_condensed(fd_min_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'fd_min_dams')
                ###Lagged DR (lactation deficit)
                ldr_condensed_dams = sfun.f1_condensed(ldr_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'ldr_dams')
                ###Loss of potential milk due to consistent under production
                lb_condensed_dams = sfun.f1_condensed(lb_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'lb_dams')
                ###Loss of potential milk due to consistent under production
                rc_birth_condensed_dams = sfun.f1_condensed(rc_birth_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'rc_birth_dams')
                ###Weight of foetus (condense)
                w_f_condensed_dams = sfun.f1_condensed(w_f_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'w_f_dams')
                ###Weight of gravid uterus (condense)
                guw_condensed_dams = sfun.f1_condensed(guw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'guw_dams')
                ###Normal weight of foetus (condense)
                nw_f_condensed_dams = sfun.f1_condensed(nw_f_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'nw_f_dams')
                ###Birth weight carryover (running tally of foetal weight diff)
                cf_w_b_condensed_dams = sfun.f1_condensed(cf_w_b_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cf_w_b_dams')
                ###LTW CFW carryover (running tally of CFW diff)
                cf_cfwltw_condensed_dams = sfun.f1_condensed(cf_cfwltw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cf_cfwltw_dams')
                ###LTW FD carryover (running tally of FD diff)
                cf_fdltw_condensed_dams = sfun.f1_condensed(cf_fdltw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cf_fdltw_dams')
                ##dams LTW CFW (total adjustment, calculated at birth)
                cfw_ltwadj_condensed_dams = sfun.f1_condensed(cfw_ltwadj_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cfw_ltwadj_dams')
                ##dams LTW FD (total adjustment, calculated at birth)
                fd_ltwadj_condensed_dams = sfun.f1_condensed(fd_ltwadj_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p], 'fd_ltwadj_dams')
                # ###Carry forward conception
                # cf_conception_condensed_dams = sfun.f1_condensed(cf_conception_dams
                #                         , idx_sorted_w_dams, condense_w_mask_dams
                #                         , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                #                         , pkl_condensed_values['dams'][p],'cf_conception_dams'))
                ###Weaning weight carryover (running tally of weaning weight diff)
                cf_w_w_condensed_dams = sfun.f1_condensed(cf_w_w_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cf_w_w_dams')
                ###Condition score change carryover (running tally of dam CS change in late pregnancy)
                cf_csc_condensed_dams = sfun.f1_condensed(cf_csc_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'cf_csc_dams')
                ###Average FOO during lactation (for weaning weight calculation)
                foo_lact_ave_condensed = sfun.f1_condensed(foo_lact_ave, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_dams, pkl_condensed_values['dams'][p],'foo_lact_ave')

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_yatf = sfun.f1_condensed(ffcfw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'ffcfw_yatf')
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_yatf = sfun.f1_condensed(nw_start_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'nw_start_yatf')
                ###FFCFW maximum to date
                ffcfw_max_condensed_yatf = sfun.f1_condensed(ffcfw_max_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'ffcfw_max_yatf')
                ###Weight of adipose (condense)
                aw_condensed_yatf = sfun.f1_condensed(aw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'aw_yatf')
                ###Weight of muscle (condense)
                mw_condensed_yatf = sfun.f1_condensed(mw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'mw_yatf')
                ###Weight of bone (condense)
                bw_condensed_yatf = sfun.f1_condensed(bw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'bw_yatf')
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g2 = sfun.f1_condensed(omer_history_yatf, idx_sorted_w_yatf[na,...], condense_w_mask_yatf[na,...]
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'omer_history_yatf')
                ###Clean fleece weight (condense)
                cfw_condensed_yatf = sfun.f1_condensed(cfw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'cfw_yatf')
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g2 = sfun.f1_condensed(d_cfw_history_yatf_p2, idx_sorted_w_yatf[na,...], condense_w_mask_yatf[na,...]
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'d_cfw_history_yatf_p2')
                ###Fibre length since shearing (condense)
                fl_condensed_yatf = sfun.f1_condensed(fl_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'fl_yatf')
                ###Average FD since shearing (condense)
                fd_condensed_yatf = sfun.f1_condensed(fd_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'fd_yatf')
                ###Minimum FD since shearing (condense)
                fd_min_condensed_yatf = sfun.f1_condensed(fd_min_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'fd_min_yatf')
                ##yatf birth weight
                w_b_condensed_yatf = sfun.f1_condensed(w_b_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'w_b_yatf')
                ##yatf wean weight
                w_w_condensed_yatf = sfun.f1_condensed(w_w_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , mask_gen_condensed_used_yatf, pkl_condensed_values['yatf'][p],'w_w_yatf')
            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_offs = sfun.f1_condensed(ffcfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'ffcfw_offs')
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_offs = sfun.f1_condensed(nw_start_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'nw_start_offs')
                ###FFCFW maximum to date
                ffcfw_max_condensed_offs = sfun.f1_condensed(ffcfw_max_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'ffcfw_max_offs')
                ###Weight of adipose (condense)
                aw_condensed_offs = sfun.f1_condensed(aw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'aw_offs')
                ###Weight of muscle (condense)
                mw_condensed_offs = sfun.f1_condensed(mw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'mw_offs')
                ###Weight of bone (condense)
                bw_condensed_offs = sfun.f1_condensed(bw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'bw_offs')
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g3 = sfun.f1_condensed(omer_history_offs, idx_sorted_w_offs[na,...], condense_w_mask_offs[na,...]
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'omer_history_offs')
                ###Clean fleece weight (condense)
                cfw_condensed_offs = sfun.f1_condensed(cfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'cfw_offs')
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g3 = sfun.f1_condensed(d_cfw_history_offs_p2, idx_sorted_w_offs[na,...], condense_w_mask_offs[na,...]
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'d_cfw_history_offs_p2')
                ###Fibre length since shearing (condense)
                fl_condensed_offs = sfun.f1_condensed(fl_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'fl_offs')
                ###Average FD since shearing (condense)
                fd_condensed_offs = sfun.f1_condensed(fd_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'fd_offs')
                ###Minimum FD since shearing (condense)
                fd_min_condensed_offs = sfun.f1_condensed(fd_min_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , mask_gen_condensed_used_offs, pkl_condensed_values['offs'][p],'fd_min_offs')

            ##condense end numbers - have to condense the numbers before calc start production, but need to condense production using non-condensed end numbers
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                numbers_end_condensed_sire = sfun.f1_condensed(numbers_end_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                numbers_end_condensed_dams = sfun.f1_condensed(numbers_end_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , pkl_condensed_values['dams'][p],'numbers_end_dams')

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                numbers_end_condensed_yatf = sfun.f1_condensed(numbers_end_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvps_percondense_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , pkl_condensed_values['yatf'][p],'numbers_end_yatf')

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] > 0):
                numbers_end_condensed_offs = sfun.f1_condensed(numbers_end_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvps_percondense_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]
                                        , pkl_condensed_values['offs'][p],'numbers_end_offs')

            ##start production - this requires condensed end numbers
            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, ffcfw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###nw (start - normal weight)	- yes this is meant to be updated from nw_start
                nw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, nw_start_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###FFCFW maximum to date
                ffcfw_max_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, ffcfw_max_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Weight of adipose (start)
                aw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, aw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Weight of muscle (start)
                mw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, mw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Weight of bone (start)
                bw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, bw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Organ energy requirement (start)
                omer_history_start_p3g0 = sfun.f1_period_start_prod(numbers_end_condensed_sire, omer_history_condensed_p3g0
                                        , prejoin_tup, season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire[na,...]
                                        , mask_min_wa_lw_w_sire[na,...], mask_max_lw_wz_sire[na,...], mask_max_wa_lw_w_sire[na,...])
                ###Clean fleece weight (start)
                cfw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, cfw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g0 = sfun.f1_period_start_prod(numbers_end_condensed_sire, d_cfw_history_condensed_p2g0, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire[na,...]
                                        , mask_min_wa_lw_w_sire[na,...], mask_max_lw_wz_sire[na,...], mask_max_wa_lw_w_sire[na,...])
                ###Fibre length since shearing (start)
                fl_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fl_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Average FD since shearing (start)
                fd_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fd_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)
                ###Minimum FD since shearing (start)
                fd_min_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fd_min_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_sire
                                        , mask_min_wa_lw_w_sire, mask_max_lw_wz_sire, mask_max_wa_lw_w_sire)

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ffcfw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p], stub_lw_idx=stub_lw_idx_dams #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1, period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, nw_start_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###FFCFW maximum to date
                ffcfw_max_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ffcfw_max_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of adipose (start)
                aw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, aw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of muscle (start)
                mw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, mw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of bone (start)
                bw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, bw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Organ energy requirement (start)
                omer_history_start_p3g1 = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , omer_history_condensed_p3g1, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams[na,...], mask_min_wa_lw_w_dams[na,...]
                                        , mask_max_lw_wz_dams[na,...], mask_max_wa_lw_w_dams[na,...], period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams[na,...], len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (start)
                cfw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cfw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g1 = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , d_cfw_history_condensed_p2g1, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams[na,...], mask_min_wa_lw_w_dams[na,...]
                                        , mask_max_lw_wz_dams[na,...], mask_max_wa_lw_w_dams[na,...], period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams[na,...], len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Fibre length since shearing (start)
                fl_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fl_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FD since shearing (start)
                fd_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Minimum FD since shearing (start)
                fd_min_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_min_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Lagged DR (lactation deficit)
                ldr_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ldr_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Loss of potential milk due to consistent under production
                lb_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, lb_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Loss of potential milk due to consistent under production
                rc_birth_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, rc_birth_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of foetus (start)
                w_f_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, w_f_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of gravid uterus (start)
                guw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, guw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Normal weight of foetus (start)
                nw_f_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, nw_f_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Birth weight carryover (running tally of foetal weight diff)
                cf_w_b_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_w_b_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###LTW CFW carryover (running tally of CFW diff)
                cf_cfwltw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_cfwltw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###LTW FD carryover (running tally of FD diff)
                cf_fdltw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_fdltw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##dams LTW CFW (total adjustment, calculated at birth)
                cfw_ltwadj_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , cfw_ltwadj_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##dams LTW FD (total adjustment, calculated at birth)
                fd_ltwadj_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_ltwadj_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                # ###Carry forward conception
                # cf_conception_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams
                #                         , cf_conception_condensed_dams, prejoin_tup
                #                         , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                #                         , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                #                         , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                #                         , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                #                         , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                #                         , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                #                         , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                #                         , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weaning weight carryover (running tally of foetal weight diff)
                cf_w_w_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_w_w_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###CS change carryover (running tally of dam CS change in late pregnancy)
                cf_csc_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_csc_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FOO during lactation (for weaning weight calculation)
                foo_lact_ave_start = sfun.f1_period_start_prod(numbers_end_condensed_dams, foo_lact_ave_condensed, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_dams, mask_min_wa_lw_w_dams
                                        , mask_max_lw_wz_dams, mask_max_wa_lw_w_dams, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal=gbal_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_scan=est_drys_retained_scan_pa1e1b1nwzida0e0b0xyg1[p]
                                        , drysretained_birth=est_drys_retained_birth_pa1e1b1nwzida0e0b0xyg1[p] #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                                        , stub_lw_idx=stub_lw_idx_dams, len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, ffcfw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, nw_start_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###FFCFW maximum to date
                ffcfw_max_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, ffcfw_max_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of adipose (start)
                aw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, aw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of muscle (start)
                mw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, mw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of bone (start)
                bw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, bw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Organ energy requirement (start)
                omer_history_start_p3g2 = sfun.f1_period_start_prod(numbers_end_condensed_yatf, omer_history_condensed_p3g2, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf[na,...]
                                        , mask_min_wa_lw_w_yatf[na,...], mask_max_lw_wz_yatf[na,...], mask_max_wa_lw_w_yatf[na,...]
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (start)
                cfw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, cfw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g2 = sfun.f1_period_start_prod(numbers_end_condensed_yatf
                                        , d_cfw_history_condensed_p2g2, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf[na,...]
                                        , mask_min_wa_lw_w_yatf[na,...], mask_max_lw_wz_yatf[na,...], mask_max_wa_lw_w_yatf[na,...]
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Fibre length since shearing (start)
                fl_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fl_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FD since shearing (start)
                fd_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fd_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Minimum FD since shearing (start)
                fd_min_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fd_min_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##yatf birth weight
                w_b_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, w_b_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##yatf wean weight
                w_w_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, w_w_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_yatf
                                        , mask_min_wa_lw_w_yatf, mask_max_lw_wz_yatf, mask_max_wa_lw_w_yatf
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1 , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, ffcfw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, nw_start_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###FFCFW maximum to date
                ffcfw_max_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, ffcfw_max_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of adipose (start)
                aw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, aw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of muscle (start)
                mw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, mw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of bone (start)
                bw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, bw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Organ energy requirement (start)
                omer_history_start_p3g3 = sfun.f1_period_start_prod(numbers_end_condensed_offs, omer_history_condensed_p3g3, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs[na,...], mask_min_wa_lw_w_offs[na,...]
                                        , mask_max_lw_wz_offs[na,...], mask_max_wa_lw_w_offs[na,...], stub_lw_idx=stub_lw_idx_offs[na,...], len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Clean fleece weight (start)
                cfw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, cfw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g3 = sfun.f1_period_start_prod(numbers_end_condensed_offs, d_cfw_history_condensed_p2g3
                                        , prejoin_tup, season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1]
                                        , mask_min_lw_wz_offs[na,...], mask_min_wa_lw_w_offs[na,...], mask_max_lw_wz_offs[na,...], mask_max_wa_lw_w_offs[na,...], stub_lw_idx=stub_lw_idx_offs[na,...]
                                        , len_gen_t=len_gen_t3, a_t_g=a_t_g3, period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Fibre length since shearing (start)
                fl_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fl_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Average FD since shearing (start)
                fd_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fd_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Minimum FD since shearing (start)
                fd_min_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fd_min_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_wz_offs, mask_min_wa_lw_w_offs
                                        , mask_max_lw_wz_offs, mask_max_wa_lw_w_offs, stub_lw_idx=stub_lw_idx_offs, len_gen_t=len_gen_t3, a_t_g=a_t_g3
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p+1])


            ##start numbers - has to be after production because the numbers are being calced for the current period and are used in the start production function
            ## doesn't have to use condensed numbers because we are only interested in the start vs end numbers of a dvp (using condensed numbers would still work).
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                numbers_start_sire = sfun.f1_period_start_nums(numbers_end_sire, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=0)
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_sire = fun.f_update(numbers_start_condense_sire, numbers_start_sire, False) #currently sire don't have any fvp

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                numbers_start_dams = sfun.f1_period_start_nums(numbers_end_dams, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=1
                                        , numbers_initial_repro=numbers_initial_propn_repro_a1e1b1nwzida0e0b0xyg1
                                        , period_is_prejoin=period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1
                                        , period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_dams = fun.f_update(numbers_start_condense_dams, numbers_start_dams
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p-1:p+2,...] >0): #use p+2 so that initial numbers get set when birth is next period, p-1 so that numbers get set to 0 after weaning.
                numbers_start_yatf = sfun.f1_period_start_nums(numbers_end_yatf, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg
                                        , nyatf_b1=nyatf_b1nwzida0e0b0xyg, gender_propn_x=gender_propn_xyg
                                        , period_is_birth=period_is_birth_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , prevperiod_is_wean=previousperiod_is_wean_pa1e1b1nwzida0e0b0xyg2[p+1], group=2
                                        , len_gen_t=len_gen_t1, a_t_g=a_t_g1, period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1[p + 1])
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    numbers_start_condense_yatf = fun.f_update(numbers_start_condense_yatf, numbers_start_yatf
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                numbers_start_offs = sfun.f1_period_start_nums(numbers_end_offs, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=3
                                        , len_gen_t=len_gen_t3, a_t_g=a_t_g3, period_is_startdvp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3[p + 1])
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_offs = fun.f_update(numbers_start_condense_offs, numbers_start_offs
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
            ##This is the end of the p loop

        ## Calculate LTW sfw multiplier & sfd addition then repeat the generator loops with updated LTW adjuster
        ### sires don't have an adjuster calculated because they are born off-farm & unrelated to the dam nutrition profile
        ### yatf don't have an adjuster because the LTW project did not show a consistent effect of dam profile on the wool shorn at the lamb shearing.
        ### CFW is a scalar adjustment so the LTW effect as a proportion of sfw, which can be applied across genotypes
        ### FD is an absolute change, it not scaled by sfd.
        ### Note: the ltw adjustment is 0 for dams with no yatf (the LW profile of ewe with no yatf does not affect the next generation)

        ## The LTW adjuster from lambing is distributed across the periods from pre-joining to next_period_is_prejoining
        ### The LTW adjuster is retained in the variable through until period_is_prejoining, so it can be accessed when next_period_is_prejoining

        ### Create the association between nextperiod_is_prejoin and the current period
        a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1 = fun.f_next_prev_association(date_end_p, date_prejoin_next_pa1e1b1nwzida0e0b0xyg1, 1, 'right').astype(dtypeint)[na] #p indx of period before prejoining - when nextperiod is prejoining this returns the current period
        ### populate ltwadj with the value from the period before prejoining. That value is the final value that has been carried forward from the whole profile change
        o_cfw_ltwadj_tpdams = np.take_along_axis(o_cfw_ltwadj_tpdams, a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1, axis=p_pos)
        o_fd_ltwadj_tpdams = np.take_along_axis(o_fd_ltwadj_tpdams, a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1, axis=p_pos)

        ## There are 2 scenarios (n_fs_dams==1 & n_fs_dams>1) for which ltwadj needs to be calculated.
        ## 1. n_fs_dams==1 (t1_sfw_ltwadj_tpdams with singleton p). With only 1 dam nutrition profile (per starting
        ### weight) all progeny will be from dams with this LW profile. LTWadj for dams and offspring can be estimated
        ### using a standard structure (age & btrt) for dams and a weighted average across all active axes except i,
        ### y & g1 (because these represent classes that are not combined at prejoining)
        ### This makes the assumption that the dams are equally spread across the starting weights.
        ## 2. n_fs_dams>1 (t2_sfw_ltwadj_tpdams with active p). With multiple feedsupply for dams the ltwadj can only be
        ### approximated. LTWadj is varied for each class of dams based on the LW profile of the dams themselves and
        ### scaled by the number of progeny they rear as a proportion of the total number weaned. This ltwadj factor is
        ### then applied only to that class of dams (rather than averaged and then applied across all classes).
        ### In this scenario the ltwadj for progeny is unknowable because the optimum dam LW profile is not known.

        ### In both scenarios the offspring ltwadj is calculated the same, based on the w[0] dam profile.

        ### Note: The accuracy of the ltwadj that is saved in the feedsupply pickle could be improved by using the
        ### actual dam numbers from lp_vars, however, this is not done because this could cause some 'randomness'
        ### within an experiment with ltwadj varying between trials due to the optimum dam numbers in the 'creating' trial.

        t1_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(o_cfw_ltwadj_tpdams, o_numbers_start_tpdams
                                                            * season_propn_zida0e0b0xyg
                                                            * btrt_propn_b1nwzida0e0b0xyg1
                                                            * period_is_birth_pa1e1b1nwzida0e0b0xyg1
                                                            , axis=(p_pos, a1_pos, e1_pos, b1_pos, n_pos, w_pos, z_pos)   #presuming all offspring axes are singleton and don't need to be included
                                                            , keepdims=True) / sfw_a0e0b0xyg1

        t1_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(o_fd_ltwadj_tpdams, o_numbers_start_tpdams
                                                            * season_propn_zida0e0b0xyg
                                                            * btrt_propn_b1nwzida0e0b0xyg1
                                                            * period_is_birth_pa1e1b1nwzida0e0b0xyg1
                                                            , axis=(p_pos, a1_pos, e1_pos, b1_pos, n_pos, w_pos, z_pos)   #presuming all offspring axes are singleton and don't need to be included
                                                            , keepdims=True)

        t2_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = (0.5 * o_cfw_ltwadj_tpdams * nyatf_b1nwzida0e0b0xyg / npw_std_xyg1**2
                                                / sfw_a0e0b0xyg1)
        t2_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = 0.5 * o_fd_ltwadj_tpdams * nyatf_b1nwzida0e0b0xyg / npw_std_xyg1**2

        if n_fs_dams == 1:
            #use t1
            t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = t1_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1
            t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = t1_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1
        else:
            #use t2
            t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = t2_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1
            t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = t2_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1

        #### If generating with a t axis then take the t slice that corresponds with the animals being retained.
        if len_gen_t1 > 1:
            t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1, a_t_tpg1, axis=0)[0]
            t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1, a_t_tpg1, axis=0)[0]
        ### Index the now singleton t axis to remove
        t_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1 = t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[0]
        t_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1 = t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[0]

        ## allocate the LTW adjustment to the slices of g1 based on the female parent of each g1 slice.
        ## nutrition of BBB dams [0:1] affects BB-B, BB-M & BB-T during their lifetime.
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[...] = t_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., 0:1]
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[...] = t_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., 0:1]
        ## nutrition of BBM dams [1] affects BM-T [-1] during their lifetime. (Needs to be [-1] to handle if BBT have been masked)
        if mask_dams_inc_g1[3:4]:
            sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., -1] = t_sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., 1]
            sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., -1] = t_sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1[..., 1]

        ## the offspring lifetime adjustment is based on dam LW pattern 0. An estimated dam pattern is required
        ### because there is not a link in the matrix between dam profile and the offspring DVs.
        ### Note: The offspring LTW adjustment works correctly if dams N==1, because all the progeny are from pattern 0.
        ### The offspring CFW effect is a multiplier based on the dam LTW effect as a proportion of the dam sfw,
        ### this allows for the offspring to be a different genotype than the dam and get a proportional adjustment.
        ### For offspring the weighted average is not required because the offspring relate to specific slices of the dams
        ### For each offspring d slice select the p slice from o_cfw_ltwadj based on a_prevjoining_o_p when period_is_join
        ###         e1 axis in the position of e0
        ###         b1 axis in the position of b0 and simplified using a_b0_b1
        ###         w axis to only have slice 0
        ###         z axis is the weighted average across season types
        temporary = np.sum(fun.f_dynamic_slice(o_cfw_ltwadj_tpdams,w_pos,0,1) / sfw_a0e0b0xyg1
                           * (a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 == index_da0e0b0xyg)
                           * period_is_birth_pa1e1b1nwzida0e0b0xyg1, axis=p_pos, keepdims = True)
        ##dams have an e1 axis, whereas offspring have an e0 axis, swap the e1 into position of e0
        temporary = np.swapaxes(temporary, e1_pos, e0_pos)
        ##the b1 axis needs to be transformed into b0 because they are different lengths.
        temporary = np.sum(temporary * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                           , axis=b1_pos, keepdims=True)  #0 for dams with no yatf because for those b1 slices there is no corresponding slice in b0
        t_season_propn_pg = np.broadcast_to(season_propn_zida0e0b0xyg, temporary.shape)
        t3_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(temporary, t_season_propn_pg, axis=z_pos, keepdims=True)

        ## repeat for FD
        temporary = np.sum(fun.f_dynamic_slice(o_fd_ltwadj_tpdams,w_pos,0,1)
                           * (a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 == index_da0e0b0xyg)
                           * period_is_birth_pa1e1b1nwzida0e0b0xyg1, axis=p_pos, keepdims = True)
        temporary = np.swapaxes(temporary, e1_pos, e0_pos)
        temporary = np.sum(temporary * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                           , axis=b1_pos, keepdims=True)  #0 for dams with no yatf because for those b1 slices there is no corresponding slice in b0
        t_season_propn_pg = np.broadcast_to(season_propn_zida0e0b0xyg, temporary.shape)
        t3_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(temporary, t_season_propn_pg, axis=z_pos, keepdims=True)

        #### If generating with a t axis then take the t slice that corresponds with the animals being retained.
        if len_gen_t1 > 1:
            t3_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(t3_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg3, a_t_tpg1, axis=0)
            t3_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(t3_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg3, a_t_tpg1, axis=0)
        ### Index the now singleton t axis to remove
        sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3 = t3_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg3[0]
        sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3 = t3_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg3[0]

        ##store ltw adjustments so they can be pickled
        ## store on the second last ltw loop to remove randomness when pkl (so that the ltw adj that is pkl is the same as the ltw adj used in final iteration)
        if loop_ltw == loop_ltw_len-2 or loop_ltw_len==1:
            pkl_fs_info['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1'] = sfw_ltwadj_pa1e1b1nwzida0e0b0xyg1
            pkl_fs_info['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1'] = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg1
            pkl_fs_info['sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3'] = sfw_ltwadj_pa1e1b1nwzida0e0b0xyg3
            pkl_fs_info['sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3'] = sfd_ltwadj_pa1e1b1nwzida0e0b0xyg3

        ##This is the end of the LTW loop

    postp_start=time.time()
    print(f'completed generator loops: {postp_start - generator_start}')


    ## Call Steve graphing routine here if Generator is throwing an error in the post-processing.
    ### Change scan-spreadsheet to True to activate
    scan_spreadsheet = False
    while scan_spreadsheet:
        try:
            yvar1, yvar2, xlabels, wvar, xvar, axes, dimensions, verticals = pv.read_spreadsheet()
            loc = locals()
            yvar1 = loc[yvar1]
            yvar2 = loc[yvar2]
            xlabels = loc[xlabels]
            wvar = loc[wvar]
            xvar = loc[xvar]
            pv.create_plots(yvar1, yvar2, xlabels, wvar, xvar, axes, dimensions, verticals)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            scan_spreadsheet = input("Enter 1 to rescan spreadsheet for new plots: ")



    if stubble:
        return o_pi_tpdams, o_pi_tpoffs, o_ebg_tpdams, o_ebg_tpoffs

    ###########################
    #post processing inputs  #
    ###########################
    ##general
    ###feed period
    index_p6 = np.arange(len_p6)
    index_p6tpa1e1b1nwzida0e0b0xyg = fun.f_expand(index_p6, p_pos-2).astype(dtypeint)
    ###cash period allocation
    ###call allocation/interest function - assumption is that cashflow happens on the first day of the generator period.
    cash_allocation_p7tpa1e1b1nwzida0e0b0xyg, wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg = fin.f_cashflow_allocation(
        date_start_pa1e1b1nwzida0e0b0xyg[na], enterprise='stk', z_pos=z_pos) #add t axis
    ###season period allocation - required for asset value
    alloc_p7tpa1e1b1nwzida0e0b0xyg = zfun.f1_z_period_alloc(date_start_pa1e1b1nwzida0e0b0xyg[na,na,...],z_pos=z_pos, mask_z=False) #don't want to mask z at this point, z gets masked when clustered

    ###labour period
    labour_periods = per.f_p_date2_df().to_numpy() #convert from df to numpy
    labour_periods_yp5z = labour_periods + (np.arange(np.ceil(sim_years)) * 364)[:,na,na] #expand from single yr to all length of generator
    labour_periods_p5z = labour_periods_yp5z.reshape((-1, len_z))
    ####set lp to start at the next generator period following the node (needs to be next so that clustering works). Lp are adjusted so that they get clustered the same as dvps
    idx_p5z = np.minimum(len(date_start_P) - 1, np.searchsorted(date_start_P,labour_periods_p5z,'left'))  # maximum idx is the number of generator periods. use P so that it handles cases where look-up date is after the end of the generator (only really an issue for arrays with o, s & d axes but done to all for consistency)
    labour_periods_p5z = date_start_P[idx_p5z]
    a_p5_pz = np.apply_along_axis(fun.f_next_prev_association, 0, labour_periods_p5z, date_end_p, 1, 'right') % len(labour_periods)
    a_p5_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_p5_pz,z_pos,left_pos2=p_pos,right_pos2=z_pos).astype(dtype)
    index_p5tpa1e1b1nwzida0e0b0xyg = fun.f_expand(np.arange(len(labour_periods)), p_pos-2)
    ###asset value timing - the date when the asset value is tallied
    assetvalue_timing = pinp.sheep['i_date_cashflow_stock_i'][pinp.sheep['i_mask_i']]
    assetvalue_timing = assetvalue_timing.mean(keepdims=True).astype(int) #take mean in case multiple tol included
    assetvalue_timing_y = assetvalue_timing + (np.arange(np.ceil(sim_years)) * 364) #timing of asset value calculation each yr
    a_assetvalue_p = fun.f_next_prev_association(assetvalue_timing_y, date_end_p, 1,'right')
    a_assetvalue_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_assetvalue_p, p_pos)
    assetvalue_timing_pa1e1b1nwzida0e0b0xyg = assetvalue_timing_y[a_assetvalue_pa1e1b1nwzida0e0b0xyg]
    period_is_assetvalue_pa1e1b1nwzida0e0b0xyg = sfun.f1_period_is_('period_is', assetvalue_timing_pa1e1b1nwzida0e0b0xyg, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    ###add a5 axis - add start of season and end of season to asset value so the asset value can be stored at the end and start of season. This is used to trade livestock trading.
    period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg = np.stack(np.broadcast_arrays(period_is_assetvalue_pa1e1b1nwzida0e0b0xyg,
                                                                                period_is_startseason_pa1e1b1nwzida0e0b0xyg,
                                                                                nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg), axis=0)
    ###feed pool - sheep are grouped based on energy volume ratio
    ##wool
    vm_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_vm_p4'], p_pos).astype(dtype)
    pmb_p4s4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_pmb_p4s'], p_pos).astype(dtype)
    ##sale
    score_range_s8s6tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_score_range_s8s6'], p_pos - 2).astype(dtype)
    price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_months_priceadj_s7s9p4'][:,:,na], p_pos).astype(dtype)
    dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_dressp_adj_s6'], p_pos-2).astype(dtype)
    dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_dressp_adj_s7'], p_pos-2).astype(dtype)
    score_pricescalar_s7s5s6 = uinp.sheep['i_salep_score_scalar_s7s5s6'].astype(dtype)
    weight_pricescalar_s7s5s6 = uinp.sheep['i_salep_weight_scalar_s7s5s6'].astype(dtype)
    grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_weight_range_s7s5'], p_pos-2).astype(dtype)
    price_type_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_salep_price_type_s7'], p_pos-2)
    # a_s8_s7pa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['ia_s8_s7'], p_pos-1)
    cvlw_s7s5tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_cvlw_s7'], p_pos-3).astype(dtype)
    cvscore_s7s6tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_cvscore_s7'], p_pos-3).astype(dtype)
    discount_age_s7tpa1e1b1nwzida0e0b0xyg = fun.f_convert_to_inf(fun.f_expand(uinp.sheep['i_salep_discount_age_s7'],
                                                                                     p_pos-2)).astype(dtype)  # convert -- and ++ to inf
    sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_sale_cost_pc_s7'], p_pos-2).astype(dtype)
    sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_sale_cost_hd_s7'], p_pos-2).astype(dtype)
    mask_s7x_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_mask_s7x'], x_pos, left_pos2=p_pos-2, right_pos2=x_pos) #don't mask x-axis
    mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3 = fun.f_expand(uinp.sheep['i_mask_s7x'], x_pos, left_pos2=p_pos-2, right_pos2=x_pos, condition=mask_x, axis=x_pos)
    mask_s7g_s7tpa1e1b1nwzida0e0b0xyg0 = fun.f_expand(uinp.sheep['i_mask_s7g'], g_pos, left_pos2=p_pos-2, right_pos2=g_pos, condition=mask_sire_inc_g0, axis=g_pos)
    mask_s7g_s7tpa1e1b1nwzida0e0b0xyg1 = fun.f_expand(uinp.sheep['i_mask_s7g'], g_pos, left_pos2=p_pos-2, right_pos2=g_pos, condition=mask_dams_inc_g1, axis=g_pos)
    mask_s7g_s7tpa1e1b1nwzida0e0b0xyg2 = fun.f_expand(uinp.sheep['i_mask_s7g'], g_pos, left_pos2=p_pos-2, right_pos2=g_pos, condition=mask_yatf_inc_g2, axis=g_pos)
    mask_s7g_s7tpa1e1b1nwzida0e0b0xyg3 = fun.f_expand(uinp.sheep['i_mask_s7g'], g_pos, left_pos2=p_pos-2, right_pos2=g_pos, condition=mask_offs_inc_g3, axis=g_pos)
    sale_agemax_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg2  \
        , sale_agemax_s7tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_c2g(uinp.parameters['i_agemax_s7c2'],uinp.parameters['i_agemax_s7y'], a_c2_c0, i_g3_inc,p_pos-2, dtype=dtype)
    sale_agemin_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg2  \
        , sale_agemin_s7tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_c2g(uinp.parameters['i_agemin_s7c2'],uinp.parameters['i_agemin_s7y'], a_c2_c0, i_g3_inc,p_pos-2, dtype=dtype)
    sale_ffcfw_max_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_sale_ffcfw_max'], p_pos-2)
    sale_ffcfw_min_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_sale_ffcfw_min'], p_pos-2)
    dresspercent_adj_yg0, dresspercent_adj_yg1, dresspercent_adj_yg2, dresspercent_adj_yg3 = sfun.f1_c2g(uinp.parameters['i_dressp_adj_c2'],uinp.parameters['i_dressp_adj_y'], a_c2_c0, i_g3_inc, dtype=dtype)
    ##husbandry
    wool_genes_yg0, wool_genes_yg1, wool_genes_yg2, wool_genes_yg3 = sfun.f1_c2g(uinp.parameters['i_wool_genes_c2'],uinp.parameters['i_wool_genes_y'], a_c2_c0, i_g3_inc, dtype=dtype)
    animal_mated_b1g1 = fun.f_expand(sinp.stock['i_mated_b1'], b1_pos)
    operations_triggerlevels_h5h7h2tpg = fun.f_convert_to_inf(fun.f_expand(pinp.sheep['i_husb_operations_triggerlevels_h5h7h2'], p_pos-2,
                                                                                  swap=True, swap2=True)).astype(dtype)  # convert -- and ++ to inf
    husb_operations_muster_propn_h2tpg = fun.f_expand(uinp.sheep['i_husb_operations_muster_propn_h2'], p_pos-2).astype(dtype)
    husb_requisite_cost_h6tpg = fun.f_expand(uinp.sheep['i_husb_requisite_cost_h6'], p_pos-2).astype(dtype)
    husb_operations_requisites_prob_h6h2tpg = fun.f_expand(uinp.sheep['i_husb_operations_requisites_prob_h6h2'], p_pos-2,swap=True).astype(dtype)
    operations_per_hour_l2h2tpg = fun.f_expand(uinp.sheep['i_husb_operations_labourreq_l2h2'], p_pos-2,swap=True).astype(dtype)
    husb_operations_infrastructurereq_h1h2tpg = fun.f_expand(uinp.sheep['i_husb_operations_infrastructurereq_h1h2'], p_pos-2,swap=True).astype(dtype)
    husb_operations_contract_cost_h2tpg = fun.f_expand(uinp.sheep['i_husb_operations_contract_cost_h2'], p_pos-2).astype(dtype)
    husb_muster_requisites_prob_h6h4tpg = fun.f_expand(uinp.sheep['i_husb_muster_requisites_prob_h6h4'], p_pos-2,swap=True).astype(dtype)
    musters_per_hour_l2h4tpg = fun.f_expand(uinp.sheep['i_husb_muster_labourreq_l2h4'], p_pos-2,swap=True)
    husb_muster_infrastructurereq_h1h4tpg = fun.f_expand(uinp.sheep['i_husb_muster_infrastructurereq_h1h4'], p_pos-2,swap=True).astype(dtype)
    period_is_wean_pa1e1b1nwzida0e0b0xyg0 = sfun.f1_period_is_('period_is', date_weaned_ida0e0b0xyg0, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_wean_husb_pa1e1b1nwzida0e0b0xyg1 = np.logical_or(period_is_wean_pa1e1b1nwzida0e0b0xyg1, sfun.f1_period_is_('period_is', date_weaned_ida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)) #includes the weaning of the dam itself and the yatf because there is husbandry for the ewe when yatf are weaned e.g. the dams have to be mustered
    period_is_wean_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_weaned_ida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    gender_xyg = fun.f_expand(np.arange(len(mask_x)), x_pos)
    ##sire
    purchcost_ida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_purchcost_sire_ig0'], i_pos, right_pos=g_pos, condition=mask_sire_inc_g0, axis=g_pos,
                               condition2=pinp.sheep['i_masksire_i'], axis2=i_pos) #Not divided by number of years onhand because the number of years of use is reflected in the number of dams that are serviced (because one sire can service multiple dam ages)
    date_purch_oa1e1b1nwzida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_date_purch_ig0'], i_pos, right_pos=g_pos, left_pos2=p_pos-1,
                                                    right_pos2=i_pos, condition=mask_sire_inc_g0, axis=g_pos,
                                                    condition2=pinp.sheep['i_masksire_i'], axis2=i_pos)
    date_sale_oa1e1b1nwzida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_date_sale_ig0'], i_pos, right_pos=g_pos, left_pos2=p_pos-1,
                                                   right_pos2=i_pos, condition=mask_sire_inc_g0, axis=g_pos,
                                                   condition2=pinp.sheep['i_masksire_i'], axis2=i_pos)
    sire_periods_g0p8y = sire_periods_g0p8[..., na] + (np.arange(np.ceil(sim_years)) * 364)
    period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8y = sfun.f1_period_is_('period_is', sire_periods_g0p8y, date_start_pa1e1b1nwzida0e0b0xyg[...,na,na], date_end_p = date_end_pa1e1b1nwzida0e0b0xyg[...,na,na])
    period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8 = np.any(period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8y, axis=-1) #condense the y-axis - it is now accounted for by p axis
    ##dams
    ###mask for nutrition profiles. this doesn't have a full w axis because it only has the nutrition options it is expanded to w further down.
    sav_mask_nut_dams_oWi = sen.sav['nut_mask_dams_oWi'][:,0:len_nut_dams,:] #This controls if a nutrition pattern is included. (if error here adjust len_max_W1 in sen.py)
    mask_nut_dams_oWi = fun.f_sa(np.array(True), sav_mask_nut_dams_oWi,5) #all nut options included unless SAV is false
    mask_nut_oa1e1b1nWzida0e0b0xyg1 = fun.f_expand(mask_nut_dams_oWi,i_pos, left_pos2=w_pos, left_pos3=p_pos,
                                                   right_pos2=i_pos, right_pos3=w_pos, condition=pinp.sheep['i_mask_i'],
                                                   axis=i_pos, condition2=mask_o_dams, axis2=p_pos)
    ##yatf
    ###apply the sam to weaning numbers of yatf. This is like mortality at weaning
    ###When used in combination with increased rr the outcome is the cost of extra ewes lactating.
    o_numbers_start_tpyatf = fun.f_update(o_numbers_start_tpyatf, o_numbers_start_tpyatf
                                          * sam_wean_redn_pa1e1b1nwzida0e0b0xyg2, period_is_wean_pa1e1b1nwzida0e0b0xyg2)

    ##offs
    ###dvp mask - basically the shearing mask plus a true for the first dvp which is weaning
    sale_mask_g3 = np.concatenate([np.array([True]), mask_shear_g3]) #need to add true to the start of the shear mask because the first dvp is weaning
    ###age for sale opp
    sale_age_tsa1e1b1nwzida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_sales_age_tsg3'], p_pos, right_pos=g_pos,
                                                       condition=mask_offs_inc_g3, axis=g_pos, condition2=sale_mask_g3, axis2=p_pos)
    ###target weight in a dvp where sale occurs
    target_weight_tsa1e1b1nwzida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_target_weight_tsg3'], p_pos, right_pos=g_pos,
                                                        condition=mask_offs_inc_g3, axis=g_pos, condition2=sale_mask_g3, axis2=p_pos) #plus 1 because it is shearing opp and weaning (ie the dvp for offs)
    ###mask for nutrition profiles. this doesn't have a full w axis because it only has the nutrition options it is expanded to w further down.
    sav_mask_nut_offs_sWix = sen.sav['nut_mask_offs_sWix'][:,0:len_nut_offs,...] #This controls if a nutrition pattern is included. (if error here adjust len_max_W3 in sen.py)
    mask_nut_offs_sWix = fun.f_sa(np.array(True), sav_mask_nut_offs_sWix,5) #all nut options included unless SAV is false
    mask_nut_sa1e1b1nWzida0e0b0xyg3 = fun.f_expand(mask_nut_offs_sWix, x_pos, left_pos2=i_pos, left_pos3=w_pos,left_pos4=p_pos,
                                                   right_pos2=x_pos,right_pos3=i_pos,right_pos4=w_pos,
                                                   condition=pinp.sheep['i_mask_i'], axis=i_pos,
                                                   condition2=mask_shear_g3, axis2=p_pos, condition3=mask_x, axis3=x_pos)

    #################################
    ##post processing associations  #
    #################################
    ##yatf
    ###association between the birth time of yatf and the birth time of dams
    a_i_ida0e0b0xyg2 = fun.f_expand(pinp.sheep['ia_i_idg2'], d_pos, right_pos=g_pos, condition=mask_yatf_inc_g2, axis=g_pos,
                                   condition2=pinp.sheep['i_mask_i'], axis2=i_pos, condition3=mask_d_offs, axis3=d_pos)
    a_g1_g2 = fun.f_expand(pinp.sheep['ia_g1_g2'], condition=mask_yatf_inc_g2, axis=g_pos)

    ##dams
    ###transfer
    a_g1_tpa1e1b1nwzida0e0b0xyg1 = fun.f_expand(sinp.stock['ia_g1_tg1'], p_pos-1, right_pos=g_pos,
                                               condition=mask_dams_inc_g1, axis=g_pos)
    transfer_exists_tpa1e1b1nwzida0e0b0xyg1 = fun.f_expand(sinp.stock['i_transfer_exists_tg1'], p_pos-1, right_pos=g_pos,
                                                          condition=mask_dams_inc_g1, axis=g_pos)
    #### adjust the pointers for excluded sires (t axis starts as just the sires e.g. dams transfer to different sire type)
    prior_sire_excluded_tpa1e1b1nwzida0e0b0xyg0 = fun.f_expand(np.cumsum(~mask_sire_inc_g0), p_pos-1) #put the g0 axis in the t position
    a_g1_tpa1e1b1nwzida0e0b0xyg1 = a_g1_tpa1e1b1nwzida0e0b0xyg1 - prior_sire_excluded_tpa1e1b1nwzida0e0b0xyg0
    ####mask inputs for g0
    transfer_exists_tpa1e1b1nwzida0e0b0xyg1 = transfer_exists_tpa1e1b1nwzida0e0b0xyg1[mask_sire_inc_g0]
    a_g1_tpa1e1b1nwzida0e0b0xyg1 = a_g1_tpa1e1b1nwzida0e0b0xyg1[mask_sire_inc_g0]
    #### add the sale slices for the t axis
    slices_to_add = np.full((pinp.sheep['i_n_dam_sales'],)+transfer_exists_tpa1e1b1nwzida0e0b0xyg1.shape[1:], False)
    transfer_exists_tpa1e1b1nwzida0e0b0xyg1 = np.concatenate([slices_to_add, transfer_exists_tpa1e1b1nwzida0e0b0xyg1],0)
    slices_to_add = ~slices_to_add * np.arange(len_g1)
    a_g1_tpa1e1b1nwzida0e0b0xyg1 = np.concatenate([slices_to_add, a_g1_tpa1e1b1nwzida0e0b0xyg1],0)

    ###dvp pointer and index
    a_v_pa1e1b1nwzida0e0b0xyg1 =  np.apply_along_axis(fun.f_next_prev_association, 0, dvp_start_va1e1b1nwzida0e0b0xyg1
                                                      , date_end_p, 1,'right')
    index_va1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg1)+1), p_pos)
    index_vpa1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg1)+1), p_pos-1)

    ###calculate period pointer here because it needed a_v_p association
    dvp_is_mating = sfun.f1_p2v(period_is_mating_pa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1).astype(dtypeint)
    dvp_is_mating = fun.f_dynamic_slice(dvp_is_mating, e1_pos, 0, 1) #slice e axis because e axis doesn't alter the mating DVP.

    ###other dvp associations and masks
    a_p_va1e1b1nwzida0e0b0xyg1 = fun.f_next_prev_association(date_start_p, dvp_start_va1e1b1nwzida0e0b0xyg1
                                                             , 1, 'right').astype(dtypeint) #returns the period index for the start of each dvp
    # dvp_date_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1,0)
    dvp_type_next_va1e1b1nwzida0e0b0xyg1 = np.roll(dvp_type_va1e1b1nwzida0e0b0xyg1, -1, axis=p_pos)
    # period_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', dvp_date_pa1e1b1nwzida0e0b0xyg1
    #                                     , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)
    nextperiod_is_prejoin_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_prejoin_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)
    #### the transfer to a dvp_type other than type==0 (prejoin) only occurs when transferring to and from the same genotype
    #### this occurs when (index_g1 == a_g1_tg1) and the transfer exists
    mask_dvp_type_next_tg1 = transfer_exists_tpa1e1b1nwzida0e0b0xyg1 * (index_g1 == a_g1_tpa1e1b1nwzida0e0b0xyg1)
    ####dvp type next is a little more complex for animals transferring.
    #### However, the destination for transfer is always dvp type next ==0 (either it is going from dvp 2 to 0 or from 0 to 0).
    dvp_type_next_tva1e1b1nwzida0e0b0xyg1 = dvp_type_next_va1e1b1nwzida0e0b0xyg1 * mask_dvp_type_next_tg1
    ####association between dvp and lambing opp
    index_v1 = np.arange(index_vpa1e1b1nwzida0e0b0xyg1.shape[0])
    a_prev_o_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1,0)

    ###cluster
    a_ppk2g1_slra1e1b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['ia_ppk2g1_rlsb1'], b1_pos, swap=True, ax1=0, ax2=2
                                                    , left_pos2=p_pos, right_pos2=b1_pos)
    a_ppk2g1_ra1e1b1nwzida0e0b0xygsl = np.moveaxis(np.moveaxis(a_ppk2g1_slra1e1b1nwzida0e0b0xyg, 0,-1),0,-1) #move the axes 's' (scanning) and 'l' (gave birth and lost) to the end as are summed away shortly
    ####association between reproduction dvp and full dvp list. Returns array which is v long and points at the repro type that each dvp falls in.
    #### rtype is 0 to len(r) and this is the index for the reproduction cycle - this is the desired result
    #### rdvp is the dvp type for each of the reproduction dvps
    rdvp_type = sinp.stock['rdvp_type_r']  #the fvp/dvp type of each reproduction dvp
    dvp_is_repro_va1e1b1nwzida0e0b0xyg1 = np.isin(dvp_type_va1e1b1nwzida0e0b0xyg1, rdvp_type) #which of the full list of dvps are reproduction dvps.
    a_rdvp_va1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(dvp_is_repro_va1e1b1nwzida0e0b0xyg1 * index_va1e1b1nwzida0e0b0xyg1, axis=0) #return the index for each rdvp or index for previous rdvp if the dvp is not an rdvp
    rdvp_type_va1e1b1nwzida0e0b0xyg = np.take_along_axis(dvp_type_va1e1b1nwzida0e0b0xyg1,a_rdvp_va1e1b1nwzida0e0b0xyg1, axis=0) #the dvp type of the previous repro dvp
    post_prejoining_mask = np.maximum.accumulate(dvp_type_va1e1b1nwzida0e0b0xyg1 == prejoin_vtype1, axis=0) #which dvps occur prior to the first prejoining. These must be set to the prejoining r type because no special clustering happens between weaning and first prejoining.
    rdvp_type_va1e1b1nwzida0e0b0xyg[~post_prejoining_mask] = prejoin_vtype1
    #####convert rdvp to rtype - to handle when repro dvps are not dvp0,1,2.
    for r in range(len(rdvp_type)):
        rdvp = rdvp_type[r]
        rdvp_type_va1e1b1nwzida0e0b0xyg[rdvp_type_va1e1b1nwzida0e0b0xyg==rdvp] = r
    a_r_va1e1b1nwzida0e0b0xyg1 = rdvp_type_va1e1b1nwzida0e0b0xyg
    ####expand cluster input from rtype to v
    a_ppk2g1_va1e1b1nwzida0e0b0xygsl = np.take_along_axis(a_ppk2g1_ra1e1b1nwzida0e0b0xygsl,a_r_va1e1b1nwzida0e0b0xyg1[...,na,na], axis=0)

    ##offs
    ###build array of shearing dates including the initial weaning - weaning is used for sale stuff because inputs are based on weaning date.
    date_weaned_a1e1b1nwzida0e0b0xyg3 = np.broadcast_to(date_weaned_ida0e0b0xyg3,fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[1:]) #need wean date rather than first day of yr because selling inputs are days from weaning.
    date_wean_shearing_sa1e1b1nwzida0e0b0xyg3 = np.concatenate([date_weaned_a1e1b1nwzida0e0b0xyg3[na,...]
                                                                , fvp_0_start_sa1e1b1nwzida0e0b0xyg3], axis=0)

    ###dvp pointer and index
    a_v_pa1e1b1nwzida0e0b0xyg3 =  np.apply_along_axis(fun.f_next_prev_association, 0, dvp_start_va1e1b1nwzida0e0b0xyg3
                                                      , offs_date_end_p, 1,'right')
    a_p_va1e1b1nwzida0e0b0xyg3 = fun.f_next_prev_association(date_start_p, dvp_start_va1e1b1nwzida0e0b0xyg3
                                                             , 1, 'right').astype(dtypeint) #returns the period index for the start of each dvp
    index_va1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg3)+1), p_pos)
    index_vpa1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg3)+1), p_pos-1)
    dvp_type_next_va1e1b1nwzida0e0b0xyg3 = np.roll(dvp_type_va1e1b1nwzida0e0b0xyg3, -1, axis=p_pos)
    # dvp_date_pa1e1b1nwzida0e0b0xyg3=np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg3,a_v_pa1e1b1nwzida0e0b0xyg3,0)
    # period_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', dvp_date_pa1e1b1nwzida0e0b0xyg3
    #                                 , date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg3,-1,axis=0)
    nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_condense_pa1e1b1nwzida0e0b0xyg3,-1,axis=0)

    ####association between dvp and shearing - this is required because in the last dvp that the animal exist (ie when the generator ends) the sheep did not exist at shearing.
    ##the main practical difference between the prev & next association is for the DVP prior to the first shearing opportunity
    a_prev_s_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(a_prev_s_pa1e1b1nwzida0e0b0xyg3, a_p_va1e1b1nwzida0e0b0xyg3, axis=0)   #used for masking
    a_next_s_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(a_next_s_pa1e1b1nwzida0e0b0xyg3, a_p_va1e1b1nwzida0e0b0xyg3, axis=0)   # used for bounds #todo error in the final DVP which points to the previous opportunity
    a_sw_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association,0, date_wean_shearing_sa1e1b1nwzida0e0b0xyg3
                                                      , offs_date_end_p, 1, 'right')  #shearing opp with weaning included.
    ###cluster
    a_k5cluster_lsb0xyg = fun.f_expand(sinp.stock['ia_ppk5_lsb0'], b0_pos)
    a_k5cluster_b0xygls = np.moveaxis(np.moveaxis(a_k5cluster_lsb0xyg, 0,-1),0,-1) #put s and l at the end they are summed away shortly


    ##############################
    #on hand / sale / shear mask #
    ##############################
    '''
    All animals onhand at main shearing are shorn. This may be a slight limitation for lambs that are destined for sale
    a few months after main shearing because in reality farmers would wait and shear them before sale. This is tricky
    to handle in AFO because shearing in the generator is not differentiated with a t axis and if there is a dvp
    between main shearing and selling, whether the animal was shorn isn't remembered.
    
    Animals that are sold are also shorn if cfw at sale is above an inputted threshold. For these animals, sale and shearing
    occur in the same gen period. This is because including an offset was getting complex and error prone (particularly
    if main shearing falls between shearing due to selling and sale because this meant the animals got double wool income).
    In reality farmers tend to wait a bit after shearing before selling because animals are off water and feed for up to
    48hrs and because animals tend to gain weight at a faster rate directly after shearing. In AFO we don't represent 
    either of these things thus shearing and selling in the same period is not a big limitation (the two factors are 
    likely to cancel each other out so likely not a big error).  
    
    There are 3 aspects to the problem of being able to retain an animal at shearing and then selling soon after in the 
    new season year and/or cashflow year using a tactical sale option.
    1. The working capital constraint (animals can be retained and sold at the beginning of next financial yr to 
       reduce wc constraint). 
       This is not a problem for MP (because final sheep numbers aren't carried to the initial, and the end cashflow 
       balance carries forward each year).
       For the SQ model the end cashflow balance carries forward each year, however, there is still a problem that
       the final sheep numbers are carried to the initial so animals can be retained in the final year and sold 
       in the initial year to reduce wc requirement.
       It is difficult to solve for DSP and therefore the capacity of the wc constraint is compromised.
       A conceptual fix is to remove tactical sale options from the 'better' seasons at the beginning of the year,
       to force selection of the 'strategic' sale times at the end of the previous year. However, the outcome 
       could just be to alter the management of the livestock in the "better" years. 
       Note: there aren't any tactical sale options in the SE model.
    2. Gaining utility in the DSP by selling sheep in the low income year (technically this is reducing risk but not 
       in a very sensible way - it is the same as withdrawing cash from the bank in a poor year). This problem 
       has been solved by adding the 'Livestock Trading Profit'.
    3. Extra cashflow interest achieved by moving income from the end of the previous year to the start of the next year. 
       This is handled by the asset value on animals at the start of the year which adds an interest cost for animals
       retained and this will offset the interest earned.
       Note: Asset cost is not required in the MP model for the years that cashflow is carried forward, 
       but is required in the final 'equilibrium' year.  
    '''

    onhandshear_start=time.time()

    ##sire - purchased and sold on given date and shorn at main shearing - sires are simulated from weaning but for the pp we only look at a subset
    ### shearing - determined by the main shearing date - no t axis so just use the period is shearing from generator
    ###round purchase and sale date of sire to the nearest period
    date_purch_oa1e1b1nwzida0e0b0xyg0 = fun.f_next_prev_association(date_start_p, date_purch_oa1e1b1nwzida0e0b0xyg0, 0, 'left') #move input date to the beginning of the next generator period
    date_purch_oa1e1b1nwzida0e0b0xyg0 = date_start_p[date_purch_oa1e1b1nwzida0e0b0xyg0]
    date_sale_oa1e1b1nwzida0e0b0xyg0 = fun.f_next_prev_association(date_start_p, date_sale_oa1e1b1nwzida0e0b0xyg0, 0, 'left') #move input date to the beginning of the next generator period
    date_sale_oa1e1b1nwzida0e0b0xyg0 = date_start_p[date_sale_oa1e1b1nwzida0e0b0xyg0]
    on_hand_pa1e1b1nwzida0e0b0xyg0 = sfun.f1_period_is_('period_is_between', date_purch_oa1e1b1nwzida0e0b0xyg0, date_start_pa1e1b1nwzida0e0b0xyg, date_sale_oa1e1b1nwzida0e0b0xyg0, date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_sale_pa1e1b1nwzida0e0b0xyg0 = np.logical_and(on_hand_pa1e1b1nwzida0e0b0xyg0==True, np.roll(on_hand_pa1e1b1nwzida0e0b0xyg0,-1,axis=0)==False)
    ###the dvp start for the sires is essentially the date they are purchased
    period_is_startdvp_purchase_pa1e1b1nwzida0e0b0xyg0 = sfun.f1_period_is_('period_is', date_purch_oa1e1b1nwzida0e0b0xyg0, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)


    ##offs
    ###t0 - retained
    ###t1 - For dsp (and when nodes are included) sold when seasons first identified (if dvp is not a node then sale is as per SE). For SE sold target age or target weight or sold on the last day of dvp (not much value selling at start of dvp for SE model because there is only 1 dvp)
    ###t2 - sold target age or target weight or sold on the last day of dvp

    ###calc sale date then determine shearing date
    ###sale - at age
    sale_age_tpa1e1b1nwzida0e0b0xyg3=np.take_along_axis(sale_age_tsa1e1b1nwzida0e0b0xyg3,a_sw_pa1e1b1nwzida0e0b0xyg3[na],1)
    ###sale - weight target
    ####convert from shearing/dvp to p array. Increments at dvp ie point to previous sale opp until new dvp then point at next dvp.
    target_weight_tpa1e1b1nwzida0e0b0xyg3=np.take_along_axis(target_weight_tsa1e1b1nwzida0e0b0xyg3, a_sw_pa1e1b1nwzida0e0b0xyg3[na],1) #gets the target weight for each gen period
    ####adjust generator lw to reflect the cumulative max per period
    #####lw could go above target then drop back below, but it is already sold so the on hand bool shouldn't change. therefore need to use a cumulative max and reset each dvp
    weight_tpa1e1b1nwzida0e0b0xyg3= sfun.f1_cum_dvp(o_ffcfw_tpoffs,a_v_pa1e1b1nwzida0e0b0xyg3, axis=p_pos)
    ###period is sale
    #### t0 slice = True - this is handled by the inputs ie weight and date are high therefore not reached therefore on hand == true
    #### t1 & t2 slice date_p<sale_date or weight<target weight (these slices are also sold on the last period of a dvp if there is no other sale opp, see code below)
    sale_opp_tpa1e1b1nwzida0e0b0xyg3 = np.logical_or(np.logical_and(age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] <= sale_age_tpa1e1b1nwzida0e0b0xyg3,
                                                                    sale_age_tpa1e1b1nwzida0e0b0xyg3 <= age_end_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p]),
                                                     weight_tpa1e1b1nwzida0e0b0xyg3>target_weight_tpa1e1b1nwzida0e0b0xyg3)
    ###if dsp then t1 gets a sale opportunity at the start of dvp when seasons are identified (this will be first period of dvp so any other sale opportunities in that dvp will be disregarded).
    if not bool_steady_state or pinp.general['i_inc_node_periods']:
        period_is_startseasondvp_ypa1e1b1nwzida0e0b0xyg3m: object = sfun.f1_period_is_('period_is', date_node_ya1e1b1nwzidaebxygm[:,na,...], date_start_pa1e1b1nwzida0e0b0xyg3[...,na], date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3[...,na])
        period_is_startseasondvp_pa1e1b1nwzida0e0b0xyg3 = np.any(period_is_startseasondvp_ypa1e1b1nwzida0e0b0xyg3m, axis=(0,-1))
        period_is_startseasondvp_pa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_startseasondvp_pa1e1b1nwzida0e0b0xyg3, days_period_cut_pa1e1b1nwzida0e0b0xyg3>0) #only have sale opp if animal exists.
        sale_opp_tpa1e1b1nwzida0e0b0xyg3[1,...] = np.logical_or(sale_opp_tpa1e1b1nwzida0e0b0xyg3[1,...], period_is_startseasondvp_pa1e1b1nwzida0e0b0xyg3)
    ###on hand - combine period_is_sale & period_is_transfer then use cumulative max to convert to on_hand
    ### note: animals are on hand in the period they are sold ie sale takes place on the last minute of the period.
    off_hand_tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_cum_dvp(sale_opp_tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, axis=1,
        shift=1)  # this ensures that once they are sold they remain off hand for the rest of the dvp, shift =1 so that sheep are on-hand in the period they are sold because sale is end of period
    on_hand_tpa1e1b1nwzida0e0b0xyg3 = np.logical_not(off_hand_tpa1e1b1nwzida0e0b0xyg3)
    ###period is sale - one true per dvp when sale actually occurs - sale occurs in the period where sheep were on hand at the beginning and not on hand at the beginning of the next period
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(on_hand_tpa1e1b1nwzida0e0b0xyg3==True, np.roll(on_hand_tpa1e1b1nwzida0e0b0xyg3,-1,axis=1)==False)
    ###make the last period in a dvp sale if no sale has occurred previously in the dvp (this doesn't affect on_hand because animals are on hand in the period they are sold and the next period is a new dvp)
    period_is_enddvp_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg3, shift=-1, axis=0)
    period_is_sale_enddvp_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_enddvp_pa1e1b1nwzida0e0b0xyg3, on_hand_tpa1e1b1nwzida0e0b0xyg3)
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, period_is_sale_enddvp_tpa1e1b1nwzida0e0b0xyg3)
    ####set t0 sale_opp to false because no sale in t[0]
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3[0] = False

    ###bound wether sale age - default is to allow all ages to be sold. User can change this using wether sale SAV.
    min_age_wether_sale_g3 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_wether_g3'][mask_offs_inc_g3], 5)
    max_age_wether_sale_g3 = fun.f_sa(np.array([sim_years*364]), sen.sav['bnd_max_sale_age_wether_g3'][mask_offs_inc_g3], 5)
    wether_sale_mask_pa1e1b1nwzida0e0b0xyg3 = np.logical_or((gender_xyg[mask_x] != 2),
        np.logical_and(age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] > min_age_wether_sale_g3,
                       age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] < max_age_wether_sale_g3))
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, wether_sale_mask_pa1e1b1nwzida0e0b0xyg3)
    ###bound female sale age - this sets the minimum age a ewe offs can be sold. Default is no min age e.g. can be sold anytime.
    min_age_female_sale_dg3 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_dg3'], 5)
    min_age_female_sale_da0e0b0xyg3 = fun.f_expand(min_age_female_sale_dg3, left_pos=d_pos, right_pos=-1
                                           , condition=mask_d_offs, axis=d_pos, condition2=mask_offs_inc_g3, axis2=-1)
    max_age_female_sale_g3 = fun.f_sa(np.array([sim_years*365]), sen.sav['bnd_max_sale_age_female_g3'][mask_offs_inc_g3], 5)
    off_sale_mask_pa1e1b1nwzida0e0b0xyg3 = np.logical_or((gender_xyg[mask_x] != 1)
                , np.logical_and(age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] > min_age_female_sale_da0e0b0xyg3
                               , age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] < max_age_female_sale_g3))
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, off_sale_mask_pa1e1b1nwzida0e0b0xyg3)
    ###shearing - one true per dvp when shearing actually occurs
    ###shearing occurs at main shearing if the animal is on hand or at sale if cfw is above an inputted threshold.
    shearing_mincfw_g3 = pinp.sheep['i_shearing_mincfw_g3'][mask_offs_inc_g3]
    period_is_saleshear_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, o_cfw_tpoffs>=shearing_mincfw_g3)
    period_is_shearing_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(np.logical_or(period_is_saleshear_tpa1e1b1nwzida0e0b0xyg3,
                                                                              period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3),
                                                                on_hand_tpa1e1b1nwzida0e0b0xyg3)

    ##Dams
    ###t0 = sale after shearing or at the end of the dvp
    ###t1 = Tactical sale at start of dvp (scanning is at the start of dvp so this handles sale of drys. In the DSP it handles sale after finding out new season info)
    ###t>=2 = retain
    ###determine t0 sale slice - end of dvp part added below
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1 = period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1
    ###determine t1 slice - tactical sale at start of dvp (sale due to new season info and/or info from scanning)
    period_is_sale_t1_pa1e1b1nwzida0e0b0xyg1 = period_is_startdvp_pa1e1b1nwzida0e0b0xyg1
    ####if drys are retained set b1[drys] to false
    period_is_sale_t1_pa1e1b1nwzida0e0b0xyg1 = period_is_sale_t1_pa1e1b1nwzida0e0b0xyg1 * \
                                               np.logical_or(np.logical_or(nyatf_b1nwzida0e0b0xyg>0,index_b1nwzida0e0b0xyg==0),
                                                             np.logical_not(dry_retained_pa1e1b1nwzida0e0b0xyg1)) #not is required because variable is drys off hand ie sold. if forced to retain the variable wants to be false
    #todo  MRY 24/4/22 I think the code above will now work for gbal.
    # fix the syntax then include the following line with the previous line when gbal is activated
    #                                         or period_is_birth_pa1e1b1nwzida0e0b0xyg1 * (gbal_management_pa1e1b1nwzida0e0b0xyg1 >= 1) * np.logical_not(dry_retained_pa1e1b1nwzida0e0b0xyg1)

    ###combine sale t slices (t0 & t1) to produce period is sale
    shape =  tuple(np.maximum.reduce([period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1.shape, period_is_sale_t1_pa1e1b1nwzida0e0b0xyg1.shape]))
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1 = np.zeros((len_t1,)+shape, dtype=bool) #initialise on hand array with 3 t slices.
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[...]=False
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[0] = period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[1] = period_is_sale_t1_pa1e1b1nwzida0e0b0xyg1
    ###bound female sale age - this sets the minimum age dams can be sold. Default is no min age e.g. can be sold anytime.
    min_age_female_sale_g1 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_g1'][mask_dams_inc_g1], 5)
    ewe_sale_mask_pa1e1b1nwzida0e0b0xyg1 = age_start_pa1e1b1nwzida0e0b0xyg1 > min_age_female_sale_g1
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, ewe_sale_mask_pa1e1b1nwzida0e0b0xyg1)
    ####transfer - calculate period_is_finish when the dams are transferred from the current g slice to the destination g slice
    period_is_transfer_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(nextperiod_is_prejoin_pa1e1b1nwzida0e0b0xyg1[na, ...], a_g1_tpa1e1b1nwzida0e0b0xyg1, -1) * transfer_exists_tpa1e1b1nwzida0e0b0xyg1
    ###on hand - combine period_is_sale & period_is_transfer then use cumulative max to convert to on_hand
    ### note: animals are on hand in the period they are sold ie sale takes place on the last minute of the period.
    off_hand_tpa1e1b1nwzida0e0b0xyg1= sfun.f1_cum_dvp(np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg1,period_is_transfer_tpa1e1b1nwzida0e0b0xyg1),a_v_pa1e1b1nwzida0e0b0xyg1,axis=1, shift=1) #this ensures that once they are sold they remain off hand for the rest of the dvp, shift =1 so that sheep are on-hand in the period they are sold because sale is end of period
    on_hand_tpa1e1b1nwzida0e0b0xyg1 = np.logical_not(off_hand_tpa1e1b1nwzida0e0b0xyg1)
    ###make the last period in a dvp sale if no sale has occurred previously in the dvp (this doesn't affect on_hand because animals are on hand in the period they are sold and the next period is a new dvp)
    period_is_enddvp_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg1, shift=-1, axis=0)
    period_is_sale_enddvp_tpa1e1b1nwzida0e0b0xyg1 = np.logical_and(np.logical_and(period_is_enddvp_pa1e1b1nwzida0e0b0xyg1, ewe_sale_mask_pa1e1b1nwzida0e0b0xyg1),
                                                                  on_hand_tpa1e1b1nwzida0e0b0xyg1)
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[0] = np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, period_is_sale_enddvp_tpa1e1b1nwzida0e0b0xyg1)[0]
    ###calc period is shear. Shearing occurs before sale if cfw is above an inputted threshold. For t1 shearing occurs in the same period as sale.
    ###shearing - one true per dvp when shearing actually occurs
    ###retained animals are shorn at main shearing and sold animals are shorn a certain number of gen periods before sale.
    ###shearing can't occur in a different dvp to sale
    ###shearing occurs at main shearing if the animal is on hand or at sale if cfw is above an inputted threshold.
    shearing_mincfw_g1 = pinp.sheep['i_shearing_mincfw_g1'][mask_dams_inc_g1]
    period_is_saleshear_tpa1e1b1nwzida0e0b0xyg1 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, o_cfw_tpdams>=shearing_mincfw_g1)
    period_is_shearing_tpa1e1b1nwzida0e0b0xyg1 = np.logical_and(np.logical_or(period_is_saleshear_tpa1e1b1nwzida0e0b0xyg1,
                                                                              period_is_mainshearing_pa1e1b1nwzida0e0b0xyg1),
                                                                on_hand_tpa1e1b1nwzida0e0b0xyg1)

    ##Yatf
    ###t0 = sold at weaning as sucker, t1 & t2 = retained
    ###the other t slices are added further down in the code
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2 = period_is_wean_pa1e1b1nwzida0e0b0xyg2
    ###bound wether sale age - default is to allow all ages to be sold. User can change this using wether sale SAV.
    min_age_castrate_sale_g2 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_wether_g3'][mask_yatf_inc_g2], 5)
    yatf_castrate_sale_mask_pa1e1b1nwzida0e0b0xyg2 = np.logical_or((gender_xyg[mask_x] != 2)
                                            , age_start_pa1e1b1nwzida0e0b0xyg2 > min_age_castrate_sale_g2)
    ###bound female sale age - this sets the minimum age a female prog can be sold. Default is no min age e.g. can be sold anytime.
    min_age_female_sale_dg2 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_dg3'], 5)
    min_age_female_sale_oa1e1b1nwzida0e0b0xyg2 = fun.f_expand(min_age_female_sale_dg2, left_pos=p_pos, right_pos=-1
                                           , condition=mask_d_offs, axis=p_pos, condition2=mask_yatf_inc_g2, axis2=-1)
    min_age_female_sale_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(min_age_female_sale_oa1e1b1nwzida0e0b0xyg2
                                                        , a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
    yatf_female_sale_mask_pa1e1b1nwzida0e0b0xyg2 = np.logical_or((gender_xyg[mask_x] != 1)
                                                , age_start_pa1e1b1nwzida0e0b0xyg2 > min_age_female_sale_pa1e1b1nwzida0e0b0xyg2)
    ###combine the male and female yatf masks
    yatf_sale_mask_pa1e1b1nwzida0e0b0xyg2 = np.logical_and(yatf_castrate_sale_mask_pa1e1b1nwzida0e0b0xyg2, yatf_female_sale_mask_pa1e1b1nwzida0e0b0xyg2)
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2 = np.logical_and(period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2, yatf_sale_mask_pa1e1b1nwzida0e0b0xyg2)


    ######################
    #calc cost and income#
    ######################
    calc_cost_start = time.time()

    ##price variation scalars
    ###c1 prob
    prob_c1 = uinp.price_variation['prob_c1']
    prob_c1tpg = fun.f_expand(prob_c1, p_pos-2)
    ###c1z wool price scalar
    wool_price_scalar_c1z = zfun.f_seasonal_inp(uinp.price_variation['wool_price_scalar_c1z'], numpy=True, axis=-1)
    wool_price_scalar_c1tpg = fun.f_expand(wool_price_scalar_c1z, z_pos, left_pos2=p_pos-2, right_pos2=z_pos)
    ###c1z sale price scalar
    sale_price_scalar_c1z = zfun.f_seasonal_inp(uinp.price_variation['meat_price_scalar_c1z'], numpy=True, axis=-1)
    sale_price_scalar_c1s7tpg = fun.f_expand(sale_price_scalar_c1z, z_pos, left_pos2=p_pos-3, right_pos2=z_pos)


    ##purchase cost
    purchcost_p7tpa1e1b1nwzida0e0b0xyg0 = purchcost_ida0e0b0xyg0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    purchcost_wc_c0p7tpa1e1b1nwzida0e0b0xyg0 = purchcost_ida0e0b0xyg0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg

    ##calc wool value - To speed the calculation process the p array is condensed to only include periods where shearing occurs. Using a slightly different association it is then converted to a v array (this process usually used a p to v association, in this case we use s to v association).
    ###create mask which is the periods where shearing occurs
    shear_mask_p0 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0, period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos) #performs np.any across all axis except axis 1
    shear_mask_p1 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_shearing_tpa1e1b1nwzida0e0b0xyg1[:,na,...], period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos) #performs np.any across all axis except axis 1
    shear_mask_p3 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_shearing_tpa1e1b1nwzida0e0b0xyg3[:,na,...], period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg[:,mask_p_offs_p]), preserveAxis=p_pos) #performs np.any across all axis except axis 2
    ###create association between p and s
    a_p_p9a1e1b1nwzida0e0b0xyg0 = fun.f_expand(np.nonzero(shear_mask_p0)[0],p_pos)  #take [0] because nonzero function returns tuple
    a_p_p9a1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.nonzero(shear_mask_p1)[0],p_pos)  #take [0] because nonzero function returns tuple
    a_p_p9a1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.nonzero(shear_mask_p3)[0],p_pos)  #take [0] because nonzero function returns tuple
    index_p9a1e1b1nwzida0e0b0xyg0 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p0)),p_pos)
    index_p9a1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p1)),p_pos)
    index_p9a1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p3)),p_pos)
    ###convert period is shearing array to the condensed version
    period_is_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.compress(shear_mask_p0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0, p_pos)
    period_is_shearing_tp9a1e1b1nwzida0e0b0xyg1 = np.compress(shear_mask_p1, period_is_shearing_tpa1e1b1nwzida0e0b0xyg1, p_pos)
    period_is_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.compress(shear_mask_p3, period_is_shearing_tpa1e1b1nwzida0e0b0xyg3, p_pos)
    ###Vegatative Matter if shorn(end)
    vm_lower_step_pa1e1b1nwzida0e0b0xyg = np.take(vm_p4a1e1b1nwzida0e0b0xyg, a_p4_p, p_pos) #lower step is the vm in the current month
    vm_upper_step_pa1e1b1nwzida0e0b0xyg = np.take(vm_p4a1e1b1nwzida0e0b0xyg, (a_p4_p+1)%12, p_pos) #upper step is the vm in the following month
    vm_pa1e1b1nwzida0e0b0xyg = vm_lower_step_pa1e1b1nwzida0e0b0xyg * (1-a_p4dp_pg) + vm_upper_step_pa1e1b1nwzida0e0b0xyg * a_p4dp_pg #approx vm based on upper and lower step and the propn of the way through the month each gen period is.
    vm_p9a1e1b1nwzida0e0b0xyg0 = np.compress(shear_mask_p0, vm_pa1e1b1nwzida0e0b0xyg, p_pos) #expand p4 axis to p then mask to p9
    vm_p9a1e1b1nwzida0e0b0xyg1 = np.compress(shear_mask_p1, vm_pa1e1b1nwzida0e0b0xyg, p_pos) #expand p4 axis to p then mask to p9
    vm_p9a1e1b1nwzida0e0b0xyg3 = np.compress(shear_mask_p3, vm_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], p_pos)#expand p4 axis to p then mask to p9
    ###pmb - a little complex because it is dependent on time since previous shearing
    pmb_lower_step_ps4a1e1b1nwzida0e0b0xyg = np.take(pmb_p4s4a1e1b1nwzida0e0b0xyg, a_p4_p, p_pos-1) #lower step is the pmb in the current month
    pmb_upper_step_ps4a1e1b1nwzida0e0b0xyg = np.take(pmb_p4s4a1e1b1nwzida0e0b0xyg, (a_p4_p+1)%12, p_pos-1) #upper step is the pmb in the following month
    pmb_ps4a1e1b1nwzida0e0b0xyg = pmb_lower_step_ps4a1e1b1nwzida0e0b0xyg * (1-a_p4dp_pg[:,na,...]) + pmb_upper_step_ps4a1e1b1nwzida0e0b0xyg * a_p4dp_pg[:,na,...] #approx pmb based on upper and lower step and the propn of the way through the month each gen period is.
    pmb_p9s4a1e1b1nwzida0e0b0xyg0 = pmb_ps4a1e1b1nwzida0e0b0xyg[shear_mask_p0]
    pmb_p9s4a1e1b1nwzida0e0b0xyg1 = pmb_ps4a1e1b1nwzida0e0b0xyg[shear_mask_p1]
    pmb_p9s4a1e1b1nwzida0e0b0xyg3 = pmb_ps4a1e1b1nwzida0e0b0xyg[mask_p_offs_p][shear_mask_p3]
    period_current_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg0 * period_is_shearing_p9a1e1b1nwzida0e0b0xyg0, axis=p_pos) #returns the period number that the most recent shearing occurred
    period_current_shearing_tp9a1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg1 * period_is_shearing_tp9a1e1b1nwzida0e0b0xyg1, axis=p_pos) #returns the period number that the most recent shearing occurred
    period_current_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg3 * period_is_shearing_tp9a1e1b1nwzida0e0b0xyg3, axis=p_pos) #returns the period number that the most recent shearing occurred
    period_previous_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.roll(period_current_shearing_p9a1e1b1nwzida0e0b0xyg0, 1, axis=p_pos)*(index_p9a1e1b1nwzida0e0b0xyg0>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg1 = np.roll(period_current_shearing_tp9a1e1b1nwzida0e0b0xyg1, 1, axis=p_pos)*(index_p9a1e1b1nwzida0e0b0xyg1>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.roll(period_current_shearing_tp9a1e1b1nwzida0e0b0xyg3, 1, axis=p_pos)*(index_p9a1e1b1nwzida0e0b0xyg3>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    periods_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = a_p_p9a1e1b1nwzida0e0b0xyg0 - np.maximum(period_previous_shearing_p9a1e1b1nwzida0e0b0xyg0, date_born_idx_ida0e0b0xyg0)
    periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg1 = a_p_p9a1e1b1nwzida0e0b0xyg1 - np.maximum(period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg1, date_born_idx_ida0e0b0xyg1)
    periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = a_p_p9a1e1b1nwzida0e0b0xyg3 - np.maximum(period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg3, date_born_idx_ida0e0b0xyg3)
    months_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = periods_since_shearing_p9a1e1b1nwzida0e0b0xyg0 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesn't need to be perfect it's only an approximation)
    months_since_shearing_tp9a1e1b1nwzida0e0b0xyg1 = periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg1 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesn't need to be perfect it's only an approximation)
    months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesn't need to be perfect it's only an approximation)
    a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_p9a1e1b1nwzida0e0b0xyg0)#provides the index of the index which is closest to the actual months since shearing
    a_months_since_shearing_tp9a1e1b1nwzida0e0b0xyg1 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_tp9a1e1b1nwzida0e0b0xyg1)#provides the index of the index which is closest to the actual months since shearing
    a_months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3)#provides the index of the index which is closest to the actual months since shearing
    pmb_p9a1e1b1nwzida0e0b0xyg0 = np.squeeze(np.take_along_axis(pmb_p9s4a1e1b1nwzida0e0b0xyg0,a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg0[:,na,...],1),axis=p_pos) #select the relevant s4 (pmb interval) then squeeze that axis
    pmb_tp9a1e1b1nwzida0e0b0xyg1 = np.squeeze(np.take_along_axis(pmb_p9s4a1e1b1nwzida0e0b0xyg1[na,...],a_months_since_shearing_tp9a1e1b1nwzida0e0b0xyg1[:,:,na,...],2),axis=p_pos) #select the relevant s4 (pmb interval) then squeeze that axis
    pmb_tp9a1e1b1nwzida0e0b0xyg3 = np.squeeze(np.take_along_axis(pmb_p9s4a1e1b1nwzida0e0b0xyg3[na,...],a_months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3[:,:,na,...],2),axis=p_pos) #select the relevant s4 (pmb interval) then squeeze that axis
    ###apply period mask to condense p axis
    cfw_sire_p9 = np.compress(shear_mask_p0, o_cfw_tpsire, p_pos)
    fd_sire_p9 = np.compress(shear_mask_p0, o_fd_tpsire, p_pos)
    sl_sire_p9 = np.compress(shear_mask_p0, o_sl_tpsire, p_pos)
    ss_sire_p9 = np.compress(shear_mask_p0, o_ss_tpsire, p_pos)
    cfw_dams_p9 = np.compress(shear_mask_p1, o_cfw_tpdams, p_pos)
    fd_dams_p9 = np.compress(shear_mask_p1, o_fd_tpdams, p_pos)
    sl_dams_p9 = np.compress(shear_mask_p1, o_sl_tpdams, p_pos)
    ss_dams_p9 = np.compress(shear_mask_p1, o_ss_tpdams, p_pos)
    cfw_offs_p9 = np.compress(shear_mask_p3, o_cfw_tpoffs, p_pos)
    fd_offs_p9 = np.compress(shear_mask_p3, o_fd_tpoffs, p_pos)
    sl_offs_p9 = np.compress(shear_mask_p3, o_sl_tpoffs, p_pos)
    ss_offs_p9 = np.compress(shear_mask_p3, o_ss_tpoffs, p_pos)
    ###micron price guide
    stb_mpg_w4 = sfun.f1_woolprice().astype(dtype)/100
    r_vals['woolp_mpg_w4'] = stb_mpg_w4
    r_vals['fd_range'] = uinp.sheep['i_woolp_fd_range_w4']
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0[:,:,shear_mask_p0], woolp_stbnib_sire = (
                            sfun.f_wool_value(stb_mpg_w4, wool_price_scalar_c1tpg, cfw_sire_p9, fd_sire_p9, sl_sire_p9, ss_sire_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg0, pmb_p9a1e1b1nwzida0e0b0xyg0, dtype))
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1[:,:,shear_mask_p1], woolp_stbnib_dams = (
                            sfun.f_wool_value(stb_mpg_w4, wool_price_scalar_c1tpg, cfw_dams_p9, fd_dams_p9, sl_dams_p9, ss_dams_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg1, pmb_tp9a1e1b1nwzida0e0b0xyg1, dtype))
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg3[:,:,shear_mask_p3], woolp_stbnib_offs = (
                            sfun.f_wool_value(stb_mpg_w4, wool_price_scalar_c1tpg, cfw_offs_p9, fd_offs_p9, sl_offs_p9, ss_offs_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg3, pmb_tp9a1e1b1nwzida0e0b0xyg3, dtype))

    ###create woolvalue with average c1 - this is used for wc/minroe and reporting because we don't think c1 is needed for them
    woolvalue_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    woolvalue_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    woolvalue_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)
    wool_finish= time.time()


    ##Sale value - To speed the calculation process the p array is condensed to only include periods where shearing occurs. Using a slightly different association it is then converted to a v array (this process usually used a p to v association, in this case we use s to v association).
    ###create mask which is the periods where sale occurs
    sale_mask_p0 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_pa1e1b1nwzida0e0b0xyg0, period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos)
    sale_mask_p1 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg1[:,na,...], period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos)  #performs np.any on all axis except 1. only use the sale slices from the dam t axis
    sale_mask_p2 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2, period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos)  #performs np.any on all axis except 1. only use the sale slices from the dam t axis
    sale_mask_p3 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg3[:,na,...], period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg[:,mask_p_offs_p]), preserveAxis=p_pos)  #performs np.any on all axis except 1
    ###manipulate axis with associations
    grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg = score_range_s8s6tpa1e1b1nwzida0e0b0xyg[uinp.sheep['ia_s8_s7']] #s8 to s7
    month_scalar_s7tpa1e1b1nwzida0e0b0xyg = price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, a_p4_p][:,0] * (1-a_p4dp_pg) \
                                            + price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, (a_p4_p+1)%12][:,0] * a_p4dp_pg #month to p, then slice s9 (has to be separate because otherwise advanced indexing is triggered)
    month_discount_s7tpa1e1b1nwzida0e0b0xyg = price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, a_p4_p][:,1] * (1-a_p4dp_pg) \
                                              + price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, (a_p4_p+1)%12][:,1] * a_p4dp_pg#month to p, then slice s9 (has to be separate because otherwise advanced indexing is triggered)
    ###Sale price grids for selected price percentile and the scalars for LW & quality score
    grid_price_s7s5s6 = sfun.f1_saleprice(score_pricescalar_s7s5s6, weight_pricescalar_s7s5s6, dtype)
    r_vals['grid_price_s7s5s6'] = grid_price_s7s5s6
    r_vals['weight_range_s7s5'] = uinp.sheep['i_salep_weight_range_s7s5'].reshape(uinp.sheep['i_s7_len'], uinp.sheep['i_s5_len'])
    r_vals['salegrid_keys'] = uinp.sheep['i_salegrid_keys']
    grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg = fun.f_expand(grid_price_s7s5s6,p_pos-2)
    ###apply condensed periods mask
    month_scalar_s7tp9a1e1b1nwzida0e0b0xyg0 = month_scalar_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p0]
    month_scalar_s7tp9a1e1b1nwzida0e0b0xyg1 = month_scalar_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p1]
    month_scalar_s7tp9a1e1b1nwzida0e0b0xyg2 = month_scalar_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p2]
    month_scalar_s7tp9a1e1b1nwzida0e0b0xyg3 = month_scalar_s7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p][:,:,sale_mask_p3] #mask p axis with off p mask then mask p axis with sale mask
    month_discount_s7tp9a1e1b1nwzida0e0b0xyg0 = month_discount_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p0]
    month_discount_s7tp9a1e1b1nwzida0e0b0xyg1 = month_discount_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p1]
    month_discount_s7tp9a1e1b1nwzida0e0b0xyg2 = month_discount_s7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p2]
    month_discount_s7tp9a1e1b1nwzida0e0b0xyg3 = month_discount_s7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p][:,:,sale_mask_p3] #mask p axis with off p mask then mask p axis with sale mask
    rc_start_sire_tp9g = o_rc_start_tpsire[:,sale_mask_p0]
    rc_start_dams_tp9g = o_rc_start_tpdams[:,sale_mask_p1]
    rc_start_yatf_tp9g = o_rc_start_tpyatf[:,sale_mask_p2]
    rc_start_offs_tp9g = o_rc_start_tpoffs[:,sale_mask_p3]
    age_end_p9a1e1b1nwzida0e0b0xyg0 = age_end_pa1e1b1nwzida0e0b0xyg0[sale_mask_p0]
    age_end_p9a1e1b1nwzida0e0b0xyg1 = age_end_pa1e1b1nwzida0e0b0xyg1[sale_mask_p1]
    age_end_p9a1e1b1nwzida0e0b0xyg2 = age_end_pa1e1b1nwzida0e0b0xyg2[sale_mask_p2]
    age_end_p9a1e1b1nwzida0e0b0xyg3 = age_end_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p][sale_mask_p3]#mask p axis with off p mask then mask p axis with sale mask
    ffcfw_tp9a1e1b1nwzida0e0b0xyg0 = o_ffcfw_tpsire[:,sale_mask_p0]
    ffcfw_tp9a1e1b1nwzida0e0b0xyg1 = o_ffcfw_tpdams[:,sale_mask_p1]
    ffcfw_tp9a1e1b1nwzida0e0b0xyg2 = o_ffcfw_start_tpyatf[:,sale_mask_p2]
    ffcfw_tp9a1e1b1nwzida0e0b0xyg3 = o_ffcfw_tpoffs[:,sale_mask_p3]

    salevalue_c1tpa1e1b1nwzida0e0b0xyg0[:,:,sale_mask_p0], r_salegrid_c1tpa1e1b1nwzida0e0b0xyg0[:,:,sale_mask_p0] = sfun.f_sale_value(
        cn_sire.astype(dtype), cx_sire[:,0:1,...].astype(dtype), rc_start_sire_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg0
        , dresspercent_adj_yg0, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg, dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg0
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg0, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg0, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg[...,0:1,:,:], sale_agemax_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg0
        ,sale_ffcfw_min_s7tpa1e1b1nwzida0e0b0xyg, sale_ffcfw_max_s7tpa1e1b1nwzida0e0b0xyg, mask_s7g_s7tpa1e1b1nwzida0e0b0xyg0, dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg1[:,:,sale_mask_p1], r_salegrid_c1tpa1e1b1nwzida0e0b0xyg1[:,:,sale_mask_p1] = sfun.f_sale_value(
        cn_dams.astype(dtype), cx_dams[:,1:2,...].astype(dtype), rc_start_dams_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg1
        , dresspercent_adj_yg1, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg1
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg1, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg1, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg[...,1:2,:,:], sale_agemax_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg1
        ,sale_ffcfw_min_s7tpa1e1b1nwzida0e0b0xyg, sale_ffcfw_max_s7tpa1e1b1nwzida0e0b0xyg, mask_s7g_s7tpa1e1b1nwzida0e0b0xyg1, dtype)
    salevalue_c1tp9a1e1b1nwzida0e0b0xyg2, r_salegrid_c1tpa1e1b1nwzida0e0b0xyg2[:,:,sale_mask_p2] = sfun.f_sale_value(                                                #keep it as a condensed p axis
        cn_yatf.astype(dtype), cx_yatf[:,mask_x,...].astype(dtype), rc_start_yatf_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg2
        , dresspercent_adj_yg2, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg2
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg2, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg2, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg2, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg2
        ,sale_ffcfw_min_s7tpa1e1b1nwzida0e0b0xyg, sale_ffcfw_max_s7tpa1e1b1nwzida0e0b0xyg, mask_s7g_s7tpa1e1b1nwzida0e0b0xyg2, dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg3[:,:,sale_mask_p3], r_salegrid_c1tpa1e1b1nwzida0e0b0xyg3[:,:,sale_mask_p3] = sfun.f_sale_value(
        cn_offs, cx_offs[:,mask_x,...].astype(dtype), rc_start_offs_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg3
        , dresspercent_adj_yg3, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg3
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg3, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg3, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg3
        ,sale_ffcfw_min_s7tpa1e1b1nwzida0e0b0xyg, sale_ffcfw_max_s7tpa1e1b1nwzida0e0b0xyg, mask_s7g_s7tpa1e1b1nwzida0e0b0xyg3, dtype)

    ###create salevalue with average c1 - this is used for wc/minroe and reporting because we don't think c1 is needed for them
    salevalue_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    salevalue_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    salevalue_tp9a1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(salevalue_c1tp9a1e1b1nwzida0e0b0xyg2, prob_c1tpg, axis=0)
    salevalue_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg2, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)

    sale_finish= time.time()

    ##Husbandry - shearing costs apply to p[0] but they are dropped because no numbers in p[0]
    #todo add feedbudgeting and 'labour for maintenance of infrastructure' (it currently has a cost that is representing materials and labour)
    ###Sire: cost, labour and infrastructure requirements
    husbandry_cost_tpg0, husbandry_labour_l2tpg0, husbandry_infrastructure_h1tpg0 = sfun.f_husbandry(
        uinp.sheep['i_head_adjust_sire'], mobsize_pa1e1b1nwzida0e0b0xyg0, o_ffcfw_tpsire, o_cfw_tpsire, operations_triggerlevels_h5h7h2tpg,
        p_index_pa1e1b1nwzida0e0b0xyg, age_start_pa1e1b1nwzida0e0b0xyg0, period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0,
        period_is_wean_pa1e1b1nwzida0e0b0xyg0, gender_xyg[0], o_ebg_tpsire, wool_genes_yg0, husb_operations_muster_propn_h2tpg,
        husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg, operations_per_hour_l2h2tpg,
        husb_operations_infrastructurereq_h1h2tpg, husb_operations_contract_cost_h2tpg, husb_muster_requisites_prob_h6h4tpg,
        musters_per_hour_l2h4tpg, husb_muster_infrastructurereq_h1h4tpg, dtype=dtype)
    husbandry_cost_p7tpg0 = husbandry_cost_tpg0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    husbandry_cost_wc_c0p7tpg0 = husbandry_cost_tpg0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    ###Dams: cost, labour and infrastructure requirements - accounts for yatf costs as well
    husbandry_cost_tpg1, husbandry_labour_l2tpg1, husbandry_infrastructure_h1tpg1 = sfun.f_husbandry(
        uinp.sheep['i_head_adjust_dams'], mobsize_pa1e1b1nwzida0e0b0xyg1, o_ffcfw_tpdams, o_cfw_tpdams, operations_triggerlevels_h5h7h2tpg,
        p_index_pa1e1b1nwzida0e0b0xyg, age_start_pa1e1b1nwzida0e0b0xyg1, period_is_shearing_tpa1e1b1nwzida0e0b0xyg1,
        period_is_wean_husb_pa1e1b1nwzida0e0b0xyg1, gender_xyg[1], o_ebg_tpdams, wool_genes_yg1, husb_operations_muster_propn_h2tpg,
        husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg, operations_per_hour_l2h2tpg,
        husb_operations_infrastructurereq_h1h2tpg, husb_operations_contract_cost_h2tpg, husb_muster_requisites_prob_h6h4tpg,
        musters_per_hour_l2h4tpg, husb_muster_infrastructurereq_h1h4tpg, a_t_g1, nyatf_b1nwzida0e0b0xyg, period_is_join_pa1e1b1nwzida0e0b0xyg1,
        animal_mated_b1g1, scan_option_pa1e1b1nwzida0e0b0xyg1, period_is_matingend_pa1e1b1nwzida0e0b0xyg1, dtype=dtype)
    husbandry_cost_p7tpg1 = husbandry_cost_tpg1 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    husbandry_cost_wc_c0p7tpg1 = husbandry_cost_tpg1 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    ###offs: cost, labour and infrastructure requirements
    husbandry_cost_tpg3, husbandry_labour_l2tpg3, husbandry_infrastructure_h1tpg3 = sfun.f_husbandry(
        uinp.sheep['i_head_adjust_offs'], mobsize_pa1e1b1nwzida0e0b0xyg3, o_ffcfw_tpoffs, o_cfw_tpoffs, operations_triggerlevels_h5h7h2tpg,
        p_index_pa1e1b1nwzida0e0b0xyg3, age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p], period_is_shearing_tpa1e1b1nwzida0e0b0xyg3,
        period_is_wean_pa1e1b1nwzida0e0b0xyg3, gender_xyg[mask_x], o_ebg_tpoffs, wool_genes_yg3, husb_operations_muster_propn_h2tpg,
        husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg, operations_per_hour_l2h2tpg,
        husb_operations_infrastructurereq_h1h2tpg, husb_operations_contract_cost_h2tpg, husb_muster_requisites_prob_h6h4tpg,
        musters_per_hour_l2h4tpg, husb_muster_infrastructurereq_h1h4tpg, dtype=dtype)
    husbandry_cost_p7tpg3 = husbandry_cost_tpg3 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:, mask_p_offs_p]
    husbandry_cost_wc_c0p7tpg3 = husbandry_cost_tpg3 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg[:,:,:, mask_p_offs_p]

    husb_finish= time.time()

    ##asset value infra
    assetvalue_infra_h1 = uinp.sheep['i_infrastructure_asset_h1']

    ##infra r&m cost - #Overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even
    rm_start_c0 = per.f_cashflow_date() + 182 #Overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even
    ###call allocation/interest function
    rm_cash_allocation_p7z, rm_wc_allocation_c0p7z = fin.f_cashflow_allocation(rm_start_c0[:,na], enterprise='stk', z_pos=-1, c0_inc=True)

    ###cost - Overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even
    rm_stockinfra_var_h1 = uinp.sheep['i_infrastructure_costvariable_h1']
    rm_stockinfra_var_h1p7z = rm_stockinfra_var_h1[:,na,na] * rm_cash_allocation_p7z
    rm_stockinfra_var_wc_h1c0p7z = rm_stockinfra_var_h1[:,na,na,na] * rm_wc_allocation_c0p7z
    rm_stockinfra_fix_h1 = uinp.sheep['i_infrastructure_costfixed_h1']
    rm_stockinfra_fix_h1p7z = rm_stockinfra_fix_h1[:,na,na] * rm_cash_allocation_p7z
    rm_stockinfra_fix_wc_h1c0p7z = rm_stockinfra_fix_h1[:,na,na,na] * rm_wc_allocation_c0p7z

    ##combine income and cost from wool, sale and husb.
    ###sire
    ####asset value - calc asset value before adjusting by period is sale and shearing
    assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg0 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg0 + woolvalue_tpa1e1b1nwzida0e0b0xyg0)
                                            * period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg[:,na,na,...] * alloc_p7tpa1e1b1nwzida0e0b0xyg)
    ####adjust for period is sale/shear (needs to be done here rather than p2v so that cashflow can be combined).
    salevalue_tpa1e1b1nwzida0e0b0xyg0 = salevalue_tpa1e1b1nwzida0e0b0xyg0 * period_is_sale_pa1e1b1nwzida0e0b0xyg0
    salevalue_c1tpa1e1b1nwzida0e0b0xyg0 = salevalue_c1tpa1e1b1nwzida0e0b0xyg0 * period_is_sale_pa1e1b1nwzida0e0b0xyg0
    woolvalue_tpa1e1b1nwzida0e0b0xyg0 = woolvalue_tpa1e1b1nwzida0e0b0xyg0 * period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 * period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0
    ####cashflow and wc
    salevalue_c1p7tpa1e1b1nwzida0e0b0xyg0 = salevalue_c1tpa1e1b1nwzida0e0b0xyg0[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg0 = salevalue_tpa1e1b1nwzida0e0b0xyg0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg0 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg0[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg0 = woolvalue_tpa1e1b1nwzida0e0b0xyg0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    cashflow_c1p7tpa1e1b1nwzida0e0b0xyg0 =  (salevalue_c1p7tpa1e1b1nwzida0e0b0xyg0 + woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg0
                                         - husbandry_cost_p7tpg0)
    wc_c0p7tpa1e1b1nwzida0e0b0xyg0 =  (salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg0 + woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg0
                                         - husbandry_cost_wc_c0p7tpg0)
    ####report info
    r_salegrid_tpa1e1b1nwzida0e0b0xyg0 = r_salegrid_tpa1e1b1nwzida0e0b0xyg0 * period_is_sale_pa1e1b1nwzida0e0b0xyg0
    r_salevalue_p7tpa1e1b1nwzida0e0b0xyg0 = salevalue_tpa1e1b1nwzida0e0b0xyg0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg0 = woolvalue_tpa1e1b1nwzida0e0b0xyg0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg

    ###dams
    ####asset value - calc asset value before adjusting by period is sale and shearing. Dam asset value includes yatf
    salevalue_perdam_tpa1e1b1nwzida0e0b0xyg2[:,sale_mask_p2] = salevalue_tp9a1e1b1nwzida0e0b0xyg2 * o_numbers_start_tpyatf[:,sale_mask_p2]
    assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg1 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg1 + woolvalue_tpa1e1b1nwzida0e0b0xyg1
                                                + np.sum(salevalue_perdam_tpa1e1b1nwzida0e0b0xyg2, axis=x_pos, keepdims=True)) #sum x-axis for yatf
                                            * period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg[:,na,na,...] * alloc_p7tpa1e1b1nwzida0e0b0xyg)
    ####adjust for period is sale/shear (needs to be done here rather than p2v so that cashflow can be combined).
    salevalue_tpa1e1b1nwzida0e0b0xyg1 = salevalue_tpa1e1b1nwzida0e0b0xyg1 * period_is_sale_tpa1e1b1nwzida0e0b0xyg1
    salevalue_c1tpa1e1b1nwzida0e0b0xyg1 = salevalue_c1tpa1e1b1nwzida0e0b0xyg1 * period_is_sale_tpa1e1b1nwzida0e0b0xyg1
    woolvalue_tpa1e1b1nwzida0e0b0xyg1 = woolvalue_tpa1e1b1nwzida0e0b0xyg1 * period_is_shearing_tpa1e1b1nwzida0e0b0xyg1
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 * period_is_shearing_tpa1e1b1nwzida0e0b0xyg1
    ####cashflow and wc
    salevalue_c1p7tpa1e1b1nwzida0e0b0xyg1 = salevalue_c1tpa1e1b1nwzida0e0b0xyg1[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg1 = salevalue_tpa1e1b1nwzida0e0b0xyg1 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg1 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg1[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg1 = woolvalue_tpa1e1b1nwzida0e0b0xyg1 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    cashflow_c1p7tpa1e1b1nwzida0e0b0xyg1 =  (salevalue_c1p7tpa1e1b1nwzida0e0b0xyg1 + woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg1
                                         - husbandry_cost_p7tpg1)
    wc_c0p7tpa1e1b1nwzida0e0b0xyg1 =  (salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg1 + woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg1
                                         - husbandry_cost_wc_c0p7tpg1)
    ####report info
    r_salegrid_tpa1e1b1nwzida0e0b0xyg1 = r_salegrid_tpa1e1b1nwzida0e0b0xyg1 * period_is_sale_tpa1e1b1nwzida0e0b0xyg1
    r_salevalue_p7tpa1e1b1nwzida0e0b0xyg1 = salevalue_tpa1e1b1nwzida0e0b0xyg1 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg1 = woolvalue_tpa1e1b1nwzida0e0b0xyg1 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg

    ###yatf
    salevalue_c1p7tp9a1e1b1nwzida0e0b0xyg2 = salevalue_c1tp9a1e1b1nwzida0e0b0xyg2[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,sale_mask_p2,...]
    salevalue_wc_c0p7tp9a1e1b1nwzida0e0b0xyg2 = salevalue_tp9a1e1b1nwzida0e0b0xyg2 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg[:,:,:,sale_mask_p2,...]
    r_salegrid_tpa1e1b1nwzida0e0b0xyg2 = r_salegrid_tpa1e1b1nwzida0e0b0xyg2 * period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2

    ###offs
    ####asset value - calc asset value before adjusting by period is sale and shearing
    assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg3 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg3 + woolvalue_tpa1e1b1nwzida0e0b0xyg3)
                                            * period_is_assetvalue_a5pa1e1b1nwzida0e0b0xyg[:,na,na,mask_p_offs_p,...] * alloc_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p,...])
    ####adjust for period is sale/shear (needs to be done here rather than p2v so that cashflow can be combined).
    salevalue_tpa1e1b1nwzida0e0b0xyg3 = salevalue_tpa1e1b1nwzida0e0b0xyg3 * period_is_sale_tpa1e1b1nwzida0e0b0xyg3
    salevalue_c1tpa1e1b1nwzida0e0b0xyg3 = salevalue_c1tpa1e1b1nwzida0e0b0xyg3 * period_is_sale_tpa1e1b1nwzida0e0b0xyg3
    woolvalue_tpa1e1b1nwzida0e0b0xyg3 = woolvalue_tpa1e1b1nwzida0e0b0xyg3 * period_is_shearing_tpa1e1b1nwzida0e0b0xyg3
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg3 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg3 * period_is_shearing_tpa1e1b1nwzida0e0b0xyg3
    ####cashflow and wc
    salevalue_c1p7tpa1e1b1nwzida0e0b0xyg3 = salevalue_c1tpa1e1b1nwzida0e0b0xyg3[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]
    salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg3 = salevalue_tpa1e1b1nwzida0e0b0xyg3 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg[:,:,:,mask_p_offs_p]
    woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg3 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg3[:,na,...] * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]
    woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg3 = woolvalue_tpa1e1b1nwzida0e0b0xyg3 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg[:,:,:,mask_p_offs_p]
    cashflow_c1p7tpa1e1b1nwzida0e0b0xyg3 =  (salevalue_c1p7tpa1e1b1nwzida0e0b0xyg3 + woolvalue_c1p7tpa1e1b1nwzida0e0b0xyg3
                                         - husbandry_cost_p7tpg3)
    wc_c0p7tpa1e1b1nwzida0e0b0xyg3 =  (salevalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg3 + woolvalue_wc_c0p7tpa1e1b1nwzida0e0b0xyg3
                                         - husbandry_cost_wc_c0p7tpg3)
    ####report info
    r_saleage_tpa1e1b1nwzida0e0b0xyg3 = age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] * period_is_sale_tpa1e1b1nwzida0e0b0xyg3
    r_salegrid_tpa1e1b1nwzida0e0b0xyg3 = r_salegrid_tpa1e1b1nwzida0e0b0xyg3 * period_is_sale_tpa1e1b1nwzida0e0b0xyg3
    r_salevalue_p7tpa1e1b1nwzida0e0b0xyg3 = salevalue_tpa1e1b1nwzida0e0b0xyg3 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]
    r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg3 = woolvalue_tpa1e1b1nwzida0e0b0xyg3 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]


    ######################
    # add yatf to dams   #
    ######################
    ##me and pi
    o_pi_tpdams *= fun.f_divide(o_mei_solid_tpdams + np.sum(o_mei_solid_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True),
                              o_mei_solid_tpdams, dtype=dtype)  # done before adding yatf mei. This is instead of adding pi yatf with pi dams because some of the potential intake of the yatf is 'used' consuming milk. Doing it via mei keeps the ratio mei_dams/pi_dams the same before and after adding the yatf. This is what we want because it is saying that there is a given energy intake and it needs to be of a certain quality.
    o_mei_solid_tpdams = o_mei_solid_tpdams + np.sum(o_mei_solid_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True)
    ##emissions
    o_ch4_animal_tpdams = o_ch4_animal_tpdams + np.sum(o_ch4_animal_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True)
    o_n2o_animal_tpdams = o_n2o_animal_tpdams + np.sum(o_n2o_animal_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True)


    ############
    #feed pools#
    ############
    '''
    If you are in the season version and get warning because all slices are nan it is likely because some of the
     feed periods do not fall in any of the generator period because the feed period is too short. Thus need to 
     fix inputs.
    '''
    feedpools_start = time.time()
    ##nv masks and len
    confinement_inc = np.logical_or(np.any(confinementw_tpa1e1b1nwzida0e0b0xyg1),
                                 np.any(confinementw_tpa1e1b1nwzida0e0b0xyg3)) #if any fs is confinement then need to include confinement pool
    n_non_confinement_pools = sinp.structuralsa['i_len_f']
    len_f = n_non_confinement_pools + confinement_inc
    index_ftpa1e1b1nwzida0e0b0xyg = fun.f_expand(np.arange(len_f), p_pos-2)
    ###store info for pasture and stubble modules.
    nv['confinement_inc'] = confinement_inc
    nv['len_nv'] = len_f

    ##Calculate the feed pools (f) and allocate each intake period to a feed pool based on mei/volume (E/V). - this is done like this to handle the big arrays easier - also handles situations where offs and dams may have diff length p axis
    ###calculate ‘nv’ for each animal class.
    ###This needs to be calculated (rather than using feedsupplyw) because feedsupply could be updated in the lw target loop.
    nv_tpsire = fun.f_divide(o_mei_solid_tpsire, o_pi_tpsire, dtype=dtype)
    nv_tpdams = fun.f_divide(o_mei_solid_tpdams, o_pi_tpdams, dtype=dtype)
    nv_tpoffs = fun.f_divide(o_mei_solid_tpoffs, o_pi_tpoffs, dtype=dtype)
    ###store nv (feedsupply) so it can be used at the end of the trial to calc optimal fs
    pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg0'] = nv_tpsire
    pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg1'] = nv_tpdams
    pkl_fs_info['feedsupply_tpa1e1b1nwzida0e0b0xyg3'] = nv_tpoffs

    ##create the upper and lower cutoffs. If there is a confinement slice then it will be populated with values but they never get used.
    nv_upper_p6ftpzg = fun.f_expand(sinp.structuralsa['i_nv_upper_p6z'], left_pos=z_pos, left_pos2=p_pos-3, right_pos2=z_pos)
    nv_upper_p6ftpzg = zfun.f_seasonal_inp(nv_upper_p6ftpzg,numpy=True,axis=z_pos)
    nv_lower_p6ftpzg = fun.f_expand(sinp.structuralsa['i_nv_lower_p6z'], left_pos=z_pos, left_pos2=p_pos-3, right_pos2=z_pos)
    nv_lower_p6ftpzg = zfun.f_seasonal_inp(nv_lower_p6ftpzg,numpy=True,axis=z_pos)
    nv_cutoff_lower_p6ftpzg = nv_lower_p6ftpzg + (nv_upper_p6ftpzg - nv_lower_p6ftpzg)/n_non_confinement_pools * index_ftpa1e1b1nwzida0e0b0xyg
    nv_cutoff_upper_p6ftpzg = nv_lower_p6ftpzg + (nv_upper_p6ftpzg - nv_lower_p6ftpzg)/n_non_confinement_pools * (index_ftpa1e1b1nwzida0e0b0xyg + 1)
    ###Average these values to be passed to Pasture.py for efficiency of utilising ME and add to the dict
    nv_cutoff_ave_p6ftpzg = (nv_cutoff_lower_p6ftpzg + nv_cutoff_upper_p6ftpzg) / 2
    nv['nv_cutoff_ave_p6fz'] = np.squeeze(nv_cutoff_ave_p6ftpzg, axis=tuple(range(p_pos-1,z_pos))+tuple(range(z_pos+1,0)))

    ##Determining a std deviation for the distribution. This is an unknown but the value has been selected so that if
    ### an animal has a nv that is the mid-point of a feed pool then most of the mei & pi for that animal will occur
    ### in that feed pool. This is achieved by dividing the range of the feed pool by 6, because plus/minus 3 standard
    ### deviations from the mean is most of the range.
    nv_cutoffs_sd_p6ftpzg = (nv_upper_p6ftpzg - nv_lower_p6ftpzg) / n_non_confinement_pools / 6

    ##convert p6 to p - this reduces the final array size to save memory (otherwise final array would have p and p6 axis)
    nv_cutoff_upper_ftpzg = np.sum(nv_cutoff_upper_p6ftpzg * (a_p6_pa1e1b1nwzida0e0b0xyg==index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...]), axis=0)
    nv_cutoff_lower_ftpzg = np.sum(nv_cutoff_lower_p6ftpzg * (a_p6_pa1e1b1nwzida0e0b0xyg==index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...]), axis=0)
    nv_cutoffs_sd_ftpzg = np.sum(nv_cutoffs_sd_p6ftpzg * (a_p6_pa1e1b1nwzida0e0b0xyg==index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...]), axis=0)

    ##So that no animals are excluded the lowest cutoff[0] is set to -np.inf and the highest cutoff (excluding confinement pool) is set to np.inf
    nv_cutoff_lower_ftpzg[0, ...] = -np.inf
    nv_cutoff_upper_ftpzg[n_non_confinement_pools - 1, ...] = np.inf

    ##allocate each sheep class to a nv group
    ###Calculate a proportion of the mei & pi that goes in each pool
    nv_propn_ftpsire = fun.f_norm_cdf(nv_cutoff_upper_ftpzg, nv_tpsire, sd=nv_cutoffs_sd_ftpzg).astype(dtype) \
                       - fun.f_norm_cdf(nv_cutoff_lower_ftpzg, nv_tpsire, sd=nv_cutoffs_sd_ftpzg).astype(dtype)
    nv_propn_ftpdams = fun.f_norm_cdf(nv_cutoff_upper_ftpzg, nv_tpdams, sd=nv_cutoffs_sd_ftpzg).astype(dtype)  \
                       - fun.f_norm_cdf(nv_cutoff_lower_ftpzg, nv_tpdams, sd=nv_cutoffs_sd_ftpzg).astype(dtype)
    nv_propn_ftpoffs = fun.f_norm_cdf(nv_cutoff_upper_ftpzg[:,:,mask_p_offs_p,...], nv_tpoffs, sd=nv_cutoffs_sd_ftpzg[:,:,mask_p_offs_p,...]).astype(dtype)  \
                       - fun.f_norm_cdf(nv_cutoff_lower_ftpzg[:,:,mask_p_offs_p,...], nv_tpoffs, sd=nv_cutoffs_sd_ftpzg[:,:,mask_p_offs_p,...]).astype(dtype)
    ###adjust the calculated proportions for the confinement pool. If in confinement then:
    ####set all the slices to 0
    nv_propn_ftpsire = fun.f_update(nv_propn_ftpsire, 0.0, confinementw_tpa1e1b1nwzida0e0b0xyg0)
    nv_propn_ftpdams = fun.f_update(nv_propn_ftpdams, 0.0, confinementw_tpa1e1b1nwzida0e0b0xyg1)
    nv_propn_ftpoffs = fun.f_update(nv_propn_ftpoffs, 0.0, confinementw_tpa1e1b1nwzida0e0b0xyg3)
    ####set the confinement slice to 1.0
    nv_propn_ftpsire[-1, ...] = fun.f_update(nv_propn_ftpsire[-1, ...], 1.0, confinementw_tpa1e1b1nwzida0e0b0xyg0)
    nv_propn_ftpdams[-1, ...] = fun.f_update(nv_propn_ftpdams[-1, ...], 1.0, confinementw_tpa1e1b1nwzida0e0b0xyg1)
    nv_propn_ftpoffs[-1, ...] = fun.f_update(nv_propn_ftpoffs[-1, ...], 1.0, confinementw_tpa1e1b1nwzida0e0b0xyg3)

    ################################
    #convert variables from p to v #
    ################################
    p2v_start = time.time()
    ##every period - with base axes
    ###sire
    ch4_animal_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_ch4_animal_tpsire, numbers_p=o_numbers_end_tpsire
                                        , on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    n2o_animal_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_n2o_animal_tpsire, numbers_p=o_numbers_end_tpsire
                                        , on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    ###dams
    ch4_animal_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ch4_animal_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams
                                        , on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_pa1e1b1nwzida0e0b0xyg1)
    n2o_animal_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_n2o_animal_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams
                                        , on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_pa1e1b1nwzida0e0b0xyg1)
    ###offs
    ch4_animal_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ch4_animal_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs
                                        , on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_cut_pa1e1b1nwzida0e0b0xyg3)
    n2o_animal_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_n2o_animal_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs
                                        , on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_cut_pa1e1b1nwzida0e0b0xyg3)


    ##every period - with f & p6 axis
    ###sire - use p2v_std because there is not dvp so this version of the function may as well be used.
    mei_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_mei_solid_tpsire * nv_propn_ftpsire, numbers_p=o_numbers_end_tpsire
                                        , on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0
                                        , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tvp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])[:,:,:,na,...]#add singleton v
    pi_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_pi_tpsire * nv_propn_ftpsire, numbers_p=o_numbers_end_tpsire
                                        , on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0
                                        , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tvp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])[:,:,:,na,...]#add singleton v
    ###dams
    mei_p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_mei_solid_tpdams * nv_propn_ftpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams
                                       , on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_pa1e1b1nwzida0e0b0xyg1
                                       , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])

    pi_p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_pi_tpdams * nv_propn_ftpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams
                                           , on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_pa1e1b1nwzida0e0b0xyg1
                                           , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])

    ###offs
    mei_p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_mei_solid_tpoffs * nv_propn_ftpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs
                                       , on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_cut_pa1e1b1nwzida0e0b0xyg3
                                       , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])
    pi_p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_pi_tpoffs * nv_propn_ftpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs
                                           , on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_cut_pa1e1b1nwzida0e0b0xyg3
                                           , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])

    ##every period - with sire periods
    nsire_tva1e1b1nwzida0e0b0xyg1g0p8 = sfun.f1_p2v_std(o_n_sire_tpa1e1b1nwzida0e0b0xyg1g0p8[:,na,...], a_v_pa1e1b1nwzida0e0b0xyg1[...,na,na], index_vpa1e1b1nwzida0e0b0xyg1[...,na,na]
                                               , o_numbers_end_tpdams[:,na,...,na,na], on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...,na,na], sumadj=2)


    ##every period - with cost (p7) axis (when combining the cost the period_is arrays were already applied therefore converted from 'intermittent' to 'every period'
    ##cost requires a c axis for reporting - it is summed before converting to a param because MINROE doesn't need c axis
    ###sires
    purchcost_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(purchcost_p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                                       period_is_tvp=period_is_startdvp_purchase_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    purchcost_wc_c0p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(purchcost_wc_c0p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                                       period_is_tvp=period_is_startdvp_purchase_pa1e1b1nwzida0e0b0xyg0)[:,:,:,na,...]#add singleton v
    cashflow_c1p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(cashflow_c1p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    wc_c0p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(wc_c0p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,:,na,...]#add singleton v
    cost_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(husbandry_cost_p7tpg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    ###dams
    cashflow_c1p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(cashflow_c1p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    wc_c0p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(wc_c0p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    cost_p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(husbandry_cost_p7tpg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    ###yatf can be sold as sucker, not shorn therefore only include sale value. husbandry is accounted for with dams so don't need that here. use numbers start because weaning is beginning of period
    salevalue_d_c1p7ta1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v_std(salevalue_c1p7tp9a1e1b1nwzida0e0b0xyg2, period_is_tvp=period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2[sale_mask_p2]
                                                   , numbers_p=(o_numbers_start_tpyatf * o_numbers_start_tpdams)[:,sale_mask_p2]
                                                   , a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2[sale_mask_p2], index_any1tvp=index_da0e0b0xyg)
    salevalue_wc_d_c0p7ta1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v_std(salevalue_wc_c0p7tp9a1e1b1nwzida0e0b0xyg2, period_is_tvp=period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2[sale_mask_p2]
                                                   , numbers_p=(o_numbers_start_tpyatf * o_numbers_start_tpdams)[:,sale_mask_p2]
                                                   , a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2[sale_mask_p2], index_any1tvp=index_da0e0b0xyg)
    ###offs
    cashflow_c1p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(cashflow_c1p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)
    wc_c0p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(wc_c0p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)
    cost_p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(husbandry_cost_p7tpg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(assetvalue_a5p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)

    ##every period - with labour (p5) axis
    labour_l2p5tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(husbandry_labour_l2tpg0[:,na,...], numbers_p=o_numbers_end_tpsire,
                                             on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, a_any1_p=a_p5_pa1e1b1nwzida0e0b0xyg,index_any1tvp=index_p5tpa1e1b1nwzida0e0b0xyg)[:,:,:,na,...] #add v axis
    labour_l2p5tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(husbandry_labour_l2tpg1[:,na,...], a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1, a_any1_p=a_p5_pa1e1b1nwzida0e0b0xyg,index_any1tp=index_p5tpa1e1b1nwzida0e0b0xyg)
    labour_l2p5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(husbandry_labour_l2tpg3[:,na,...], a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3, a_any1_p=a_p5_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p],index_any1tp=index_p5tpa1e1b1nwzida0e0b0xyg)

    ##every period - with infrastructure (h1) axis
    infrastructure_h1tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(husbandry_infrastructure_h1tpg0, numbers_p=o_numbers_end_tpsire,
                                             on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    infrastructure_h1tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(husbandry_infrastructure_h1tpg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1)
    infrastructure_h1tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(husbandry_infrastructure_h1tpg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3)

    ##intermittent
    ###numbers
    ####sires - don't need any special treatment because they don't have dvps - sires only have one dvp which essentially starts when the activity is purchased
    numbers_start_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_numbers_start_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                          period_is_tvp=period_is_startdvp_purchase_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    numbers_startp8_tva1e1b1nwzida0e0b0xyg0p8 = sfun.f1_p2v_std(o_numbers_start_tpsire[...,na], on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0[...,na],
                                                          period_is_tvp=period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8, sumadj=1)[:,na,...]#add singleton v
    ####dams - need special dvp treatment for 0 day dvp. Essentially just set the start and end number to the same so that 1:1 transfer can occur.
    numbers_start_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_start_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1)
    numbers_start_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_start_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)
    numbers_end_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_end_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                                                     period_is_tp=np.logical_or(period_is_transfer_tpa1e1b1nwzida0e0b0xyg1, nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg1))  #on_hand included so that the early termination of a dvp is accounted for when transferring between ram groups
    numbers_end_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_end_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,
                                                          a_v_pa1e1b1nwzida0e0b0xyg1,numbers_start_tva1e1b1nwzida0e0b0xyg1) #intentionally numbers start - want numbers start and end to be the same for 0 day dvp so that 1:1 transfer happens.
    numbers_start_d_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_start_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1
                                                         , a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2, index_any1tp=index_da0e0b0xyg) #with a d axis.
    numbers_start_d_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_start_d_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)

    ####offs - need special dvp treatment for 0 day dvp. Essentially just set the start and end number to the same so that 1:1 transfer can occur.
    numbers_start_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_numbers_start_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3)
    numbers_start_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(numbers_start_tva1e1b1nwzida0e0b0xyg3,a_p_va1e1b1nwzida0e0b0xyg3,a_v_pa1e1b1nwzida0e0b0xyg3)
    numbers_end_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_numbers_end_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, period_is_tp=nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg3)
    numbers_end_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(numbers_end_tva1e1b1nwzida0e0b0xyg3,a_p_va1e1b1nwzida0e0b0xyg3,
                                                          a_v_pa1e1b1nwzida0e0b0xyg3,numbers_start_tva1e1b1nwzida0e0b0xyg3) #intentionally numbers start - want numbers start and end to be the same for 0 day dvp so that 1:1 transfer happens.

    ###yatf - no 0 day dvp adjustment because yatf have no dvps
    numbers_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v_std(o_numbers_start_tpyatf * o_numbers_start_tpdams
                                                                , period_is_tvp=period_is_wean_pa1e1b1nwzida0e0b0xyg2
                                                                , a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2
                                                                , index_any1tvp=index_da0e0b0xyg)  # Returns the total number of the yatf in the period in which they are weaned - with active d axis

    #### Return the weight of the yatf in the period in which they are weaned - with active dam v axis
    ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_wean_w_tpyatf, a_v_pa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_wean_pa1e1b1nwzida0e0b0xyg2)
    ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)

    #### Return the weight of the yatf in the period in which they are weaned - with active d axis
    ffcfw_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v_std(o_wean_w_tpyatf, period_is_tvp=period_is_wean_pa1e1b1nwzida0e0b0xyg2,
                                                           a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2, index_any1tvp=index_da0e0b0xyg)

    ###npw - Return the number of the yatf in the period in which they are weaned - with active d axis
    npw_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_start_tpyatf, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_start_tpdams,
                                        on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_wean_pa1e1b1nwzida0e0b0xyg2,
                                        a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2,index_any1tp=index_da0e0b0xyg) #use numbers start because weaning is beginning of period
    ###npw2 for wean % - without d axis and x summed
    t_npw = np.sum(o_numbers_start_tpyatf, axis=x_pos, keepdims=True)
    npw2_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(t_npw, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_start_tpdams,
                                        on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_wean_pa1e1b1nwzida0e0b0xyg2) #use numbers start because weaning is beginning of period


    # ##################################
    # #animal shifting between classes #
    # ##################################
    # axis b1 is the equivalent of b18 so when adding na add it after the b1 axis
    # a_prepost_b19nwzida0e0b0xyg1 = fun.f_expand(sinp.stock['a_prepost_b1'], b1_pos)
    # index_b19nwzida0e0b0xyg1 = fun.f_expand(np.arange(len_b1), b1_pos)
    # index_b18b19nwzida0e0b0xyg1 = index_b19nwzida0e0b0xyg1[:,na,...]
    # numbers_end_adj_va1e1b18nwzida0e0b0xyg1 = fun.f_update(numbers_end_va1e1b1nwzida0e0b0xyg1, np.sum(numbers_end_va1e1b1nwzida0e0b0xyg1, axis=repro_tup, keepdims=True), dvp_type_va1e1b1nwzida0e0b0xyg1 == 0)
    # numbers_end_adj_va1e1b18nwzida0e0b0xyg1 = fun.f_update(numbers_end_adj_va1e1b18nwzida0e0b0xyg1, numbers_end_va1e1b1nwzida0e0b0xyg1[:,:,:,sinp.stock['a_prepost_b1'],...], dvp_type_va1e1b1nwzida0e0b0xyg1 == 2)
    # bb_pointer_va1e1b18b19nwzida0e0b0xyg1 = fun.f_update(index_b19nwzida0e0b0xyg1, index_b18b19nwzida0e0b0xyg1, dvp_type_va1e1b1nwzida0e0b0xyg1[:,:,:,:,na,...] == 0)
    # bb_pointer_va1e1b18b19nwzida0e0b0xyg1 = sfun.f_update(bb_pointer_va1e1b18b19nwzida0e0b0xyg1, a_prepost_b19nwzida0e0b0xyg1, dvp_type_va1e1b1nwzida0e0b0xyg1[:,:,:,:,na,...] == 2)
    # dam_shift_vb18b19 = numbers_start_next_va1e1b1nwzida0e0b0xyg1[:,:,:,:,na,...] / numbers_end_adj_va1e1b18nwzida0e0b0xyg1[:,:,:,:,na,...] * (bb_pointer_va1e1b18b19nwzida0e0b0xyg1==index_b18b19nwzida0e0b0xyg1)

    ##############
    ##clustering #
    ##############
    cluster_start = time.time()
    ##dams
    ###create k2 association based on scanning and gbal
    gbal_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(gbal_management_pa1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,0)
    scan_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(scan_management_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1,0)
    a_k2cluster_va1e1b1nwzida0e0b0xyg1 = np.sum(a_ppk2g1_va1e1b1nwzida0e0b0xygsl * (gbal_va1e1b1nwzida0e0b0xyg1[...,na,na] == index_l)
                                                * (scan_va1e1b1nwzida0e0b0xyg1[...,na,na]==index_sc[:,na]), axis = (-1,-2))
    a_k2cluster_va1e1b1nwzida0e0b0xyg1 = a_k2cluster_va1e1b1nwzida0e0b0xyg1 + (len(sinp.stock['a_nfoet_b1'])
                                                                               * index_e1b1nwzida0e0b0xyg
                                                                               * (scan_va1e1b1nwzida0e0b0xyg1 == 4)
                                                                               * (nfoet_b1nwzida0e0b0xyg >= 1)) #If scanning for foetal age add 10 to the animals in the second & subsequent cycles that were scanned as pregnant (nfoet_b1 >= 1)
    ### Cluster with a t axis required for k29 which is associated with tvgg9. na to put result in g9 axis
    a_k2cluster_tva1e1b1nwzida0e0b0xyg1g9 = np.take_along_axis(a_k2cluster_va1e1b1nwzida0e0b0xyg1[na], a_g1_tpa1e1b1nwzida0e0b0xyg1, axis=-1)[..., na, :]
    ### a temporary array that is the cluster at prejoining with not mated (0) and mated (1) along the b1 axis
    temporary = np.ones_like(a_k2cluster_tva1e1b1nwzida0e0b0xyg1g9)
    temporary[:,:,:,:,0,...] = 0
    a_k2cluster_next_tva1e1b1nwzida0e0b0xyg1g9 = fun.f_update(np.roll(a_k2cluster_tva1e1b1nwzida0e0b0xyg1g9, -1, axis=1), temporary,
                                          (a_g1_tpa1e1b1nwzida0e0b0xyg1[..., na, :] != index_g1g) * transfer_exists_tpa1e1b1nwzida0e0b0xyg1[..., na])
    len_k2 = np.max(a_k2cluster_va1e1b1nwzida0e0b0xyg1)+1  #Added +1 because python starts at 0.
    index_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(len_k2), k2_pos)
    index_k29tva1e1b1nwzida0e0b0xyg1g9 = index_k2tva1e1b1nwzida0e0b0xyg1[...,na]
    index_k28k29tva1e1b1nwzida0e0b0xyg1 = index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...]

    ##offs
    ### d cluster - to change this cluster definition len_k3 needs to change (which can be done by changing k3_idx).
    a_k3cluster_da0e0b0xyg3 = fun.f_expand(np.minimum(len_k3-1, index_d), d_pos)
    ###b0 and e0 cluster
    a_k5cluster_da0e0b0xyg3 = np.sum(a_k5cluster_b0xygls * (gbal_da0e0b0xyg3[...,na,na]==index_l[:,na]) * (scan_da0e0b0xyg3[...,na,na]==index_sc), axis = (-1,-2))
    a_k5cluster_da0e0b0xyg3 = a_k5cluster_da0e0b0xyg3 + (len_b0 * index_e0b0xyg * (scan_da0e0b0xyg3 == 4)) #If scanning for foetal age add 6 to the animals in the second & subsequent cycles. 6 is the number of slices in the b0 axes
    len_k5 = np.max(a_k5cluster_da0e0b0xyg3)+1  #Added +1 because python starts at 0.
    index_k5tva1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(len_k5), k5_pos)

    ##store cluster associations for use in creating the optimal feedsupply at the end of the trial
    pkl_fs_info['a_v_pa1e1b1nwzida0e0b0xyg1'] = a_v_pa1e1b1nwzida0e0b0xyg1
    pkl_fs_info['a_k2cluster_va1e1b1nwzida0e0b0xyg1'] = a_k2cluster_va1e1b1nwzida0e0b0xyg1
    pkl_fs_info['a_v_pa1e1b1nwzida0e0b0xyg3'] = a_v_pa1e1b1nwzida0e0b0xyg3
    pkl_fs_info['a_k3cluster_da0e0b0xyg3'] = a_k3cluster_da0e0b0xyg3
    pkl_fs_info['a_k5cluster_da0e0b0xyg3'] = a_k5cluster_da0e0b0xyg3


    ######################################
    #Mask animals to nutrition profiles  #
    ######################################
    '''creates a mask that removes unnecessary values in the parameters. Because of the lw branching not every decision
        variable provides to a unique constraint and therefore some decision variables and constraints can be masked out.
        Note: FVPs don't have to be DVPs and DVPs don't have to be FVPs. 
        So there can be multiple FVPs in a DVP (because of this, the number of DVs at the start of the period is based on the FVP at the end of the DVP) 
        or multiple DVPs in a FVP (in this case the DVPs within the FVP pass 1:1 on the w8w9 axes between DVPs).
        
        The calculations require the following variables:
        n_fvps_v is the number of FVPs within the DVP
        n_prior_fvps_v is the number of FVPs prior to the start of this DVP since condensing. 
        So if the DVP dates are say 1 Feb, 1 May & 1 July 
        and the FVP dates are 1 Feb, 1 May, 1 June, 1 July, 1 Oct. Then 
        n_fvps_v = 1,2,2 because 1 fvp in the first DVP and 2 in each of the other two DVPs
        n_prior_fvps_v = 0, 1, 3 which is the cumulative sum of n_fvps_v in the previous DVPs (ie not including this DVP)
        Note: the values can change along the i axis (time of lambing) if the FVP date is not relative to a reproduction event.
        
        The inclusion of multiple FVPs in a DVP requires a variable step size based on the number of FVPs in a DVP (n_fvps_v)
        and the number of FVPs prior to this DVP (n_prior_fvps_v)
        step is equal to the total number of patterns i_w_start_len * (i_n_len ** i_n_fvp_period) divided by the number 
        of constraints : i_w_start_len * (i_n_len ** i_n_prior_fvps_v) 
        or decision variables: i_w_start_len * (i_n_len ** (i_n_prior_fvps_v + i_n_fvps_v)). 
        This can be simplified to a single line equation because the terms cancel out
        
        '''
    allocation_start = time.time()
    ##dams
    ###previous condense date - used to calc number of fvps since last condensing
    prev_condense_date_va1e1b1nwzida0e0b0xyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1.copy()
    prev_condense_date_va1e1b1nwzida0e0b0xyg1[~(dvp_type_va1e1b1nwzida0e0b0xyg1==condense_vtype1)] = 0
    prev_condense_date_va1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(prev_condense_date_va1e1b1nwzida0e0b0xyg1, axis=0)
    ###calc the number of fvps in each dvp and the number of fvps since condense
    n_fvps_va1e1b1nwzida0e0b0xyg1 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg1.shape)
    n_prior_fvps_va1e1b1nwzida0e0b0xyg1 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg1.shape)
    for v in range(dvp_start_va1e1b1nwzida0e0b0xyg1.shape[0]):
        dvp_start_v = dvp_start_va1e1b1nwzida0e0b0xyg1[v, ...]
        prev_condense_date_v = prev_condense_date_va1e1b1nwzida0e0b0xyg1[v, ...]
        n_fvp_since_condense_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= prev_condense_date_v)
                                & (fvp_start_fa1e1b1nwzida0e0b0xyg1 < dvp_start_v)).sum(axis=0)
        if v < dvp_start_va1e1b1nwzida0e0b0xyg1.shape[0] - 1:
            dvp_end_v = dvp_start_va1e1b1nwzida0e0b0xyg1[v + 1, ...]
            n_fvp_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= dvp_start_v)
                       & (fvp_start_fa1e1b1nwzida0e0b0xyg1 < dvp_end_v)).sum(axis=0)
        else:  # the final DVP gets special treatment because of wrap around
            n_fvp_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= dvp_start_v)).sum(axis=0)
        n_fvps_va1e1b1nwzida0e0b0xyg1[v,...] = n_fvp_v
        n_prior_fvps_va1e1b1nwzida0e0b0xyg1[v,...] = n_fvp_since_condense_v

    ###Steps for ‘Numbers Requires’ constraint is determined by the number of prior FVPs
    step_con_req_va1e1b1nw8zida0e0b0xyg1 = np.power(n_fs_dams, (n_fvps_percondense_dams
                                                              - n_prior_fvps_va1e1b1nwzida0e0b0xyg1))   #step is based on the FVPs at the start of the DVP
    ###Steps for the w8 decision variables is determined by the number of current & prior FVPs. w8 is the DV this period
    step_dv_va1e1b1nw8zida0e0b0xyg1 = np.power(n_fs_dams, (n_fvps_percondense_dams
                                                         - n_prior_fvps_va1e1b1nwzida0e0b0xyg1
                                                         - n_fvps_va1e1b1nwzida0e0b0xyg1))   #step is based on the FVPs at the end of the DVP
    ###Steps for ‘Numbers Provides’ is calculated with a t axis (because the t axis can alter the dvp type of the source relative to the destination)
    ###The 'provide' constraint is the same as the DV (because each DVP provides itself) except when the next DVP is condensing
    step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9 = fun.f_update(step_dv_va1e1b1nw8zida0e0b0xyg1
                                                            , np.power(n_fs_dams, n_fvps_percondense_dams)
                                                            , dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == condense_vtype1)[..., na]
    ###Mask the decision variables that are not active in this DVP in the matrix - because they share a common nutrition history (broadcast across t axis)
    mask_w8vars_va1e1b1nw8zida0e0b0xyg1 = index_wzida0e0b0xyg1 % step_dv_va1e1b1nw8zida0e0b0xyg1 == 0
    ###mask for the user defined nutrition profiles (this allows the user to exclude certain nutrition patterns e.g. high high high or low low low)
    ### this mask is combined with the main w8 & w9 masks below
    mask_nut_va1e1b1nWzida0e0b0xyg1 = np.take_along_axis(mask_nut_oa1e1b1nWzida0e0b0xyg1, a_prev_o_va1e1b1nwzida0e0b0xyg1, axis=0)
    ####association between the shortlist of nutrition profile inputs and the full range of LW patterns that include starting LW
    a_shortlist_w1 = index_w1 % len_nut_dams
    mask_nut_va1e1b1nwzida0e0b0xyg1 = mask_nut_va1e1b1nWzida0e0b0xyg1[:,:,:,:,:,a_shortlist_w1,...]  # expands the nutrition mask to all lw patterns.
    ####match the pattern requested with the pattern that is the 'history' for that pattern in previous DVPs
    mask_w8nut_va1e1b1nzida0e0b0xyg1w = np.sum(mask_nut_va1e1b1nwzida0e0b0xyg1[...,na] *
                                               (np.trunc(index_wzida0e0b0xyg1[...,na] / step_dv_va1e1b1nw8zida0e0b0xyg1[..., na])
                                                == index_w1 / step_dv_va1e1b1nw8zida0e0b0xyg1[...,na]), axis=w_pos-1) > 0 #don't keepdims so w8 axis is dropped, to allow move
    mask_w8nut_va1e1b1nw8zida0e0b0xyg1 = np.moveaxis(mask_w8nut_va1e1b1nzida0e0b0xyg1w,-1,w_pos) #move w axis to w8 position
    ###Combine the w8vars mask and the user nutrition mask
    mask_w8vars_va1e1b1nw8zida0e0b0xyg1 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_w8nut_va1e1b1nw8zida0e0b0xyg1
    ##Mask numbers provided based on the steps (with a t axis) and the next dvp type (with a t axis) (t0&1 are sold and never transfer so the mask doesn't mean anything for them. for t2 animals always transfer to themselves unless dvpnext is 'condense')
    dist_occurs_nextdvp_tva1e1b1nwzida0e0b0xyg1 = np.logical_or(dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == condense_vtype1
                                                                , dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == season_vtype1) #when distribution occurs any w8 can provide w9
    ###Mask the provide constraint (w9)
    mask_numbers_provw8w9_tva1e1b1nw8zida0e0b0xyg1w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1[...,na] \
                        * (np.trunc((index_wzida0e0b0xyg1[...,na] * np.logical_not(dist_occurs_nextdvp_tva1e1b1nwzida0e0b0xyg1[...,na])
                                     + index_w1 * dist_occurs_nextdvp_tva1e1b1nwzida0e0b0xyg1[...,na])
                                    / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9) == index_w1 / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9)
    ###Mask numbers required from the previous period (broadcast across t axis) - Note: req does not need a t axis because the destination decision variable don’t change for the transfer
    ###Mask for the require constraint (w9)
    mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1[...,na] \
                        * (np.trunc(index_wzida0e0b0xyg1 / step_con_req_va1e1b1nw8zida0e0b0xyg1)[...,na]
                           == index_w1 / step_con_req_va1e1b1nw8zida0e0b0xyg1[...,na])
    ###Create a mask for the constraint. This is used in the distribution of w8 to w9
    ### There are only Trues when next dvp is distribution
    mask_dest_tva1e1b1nwzida0e0b0xyg1 = (np.trunc(
        index_wzida0e0b0xyg1 * dist_occurs_nextdvp_tva1e1b1nwzida0e0b0xyg1 / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9[
            ..., 0]) == index_wzida0e0b0xyg1 / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9[..., 0])
    ####Add the nut mask so that distribution can't occur to a masked w slice.
    mask_dest_tva1e1b1nwzida0e0b0xyg1 = mask_dest_tva1e1b1nwzida0e0b0xyg1 * mask_w8nut_va1e1b1nw8zida0e0b0xyg1

    ###This code generate w8 mask for next period with a t axis (t axis is required for dams because of prejoining transfer)
    # ###Steps for the w9 decision variables is the same as the step for w8 in the following period except when
    # ####the next period is condensing (like step_con_prov).
    # ####The w9 axis is in the position of the w8 because the var is used to mask ffcfw_dest_wg (which has w9 in w8 pos)
    # #### The difference to step_con_prov is that step_w9 allows for multiple DV being provided by a single constraint.
    # step_w9_tva1e1b1nw8zida0e0b0xyg1 = fun.f_update(np.roll(step_dv_va1e1b1nw8zida0e0b0xyg1,-1, axis=p_pos)
    #                                                 , np.power(n_fs_dams, n_fvps_percondense_dams
    #                                                             - np.roll(n_fvps_va1e1b1nwzida0e0b0xyg1,-1,axis=p_pos))
    #                                                 , dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == condense_vtype1)
    # step_w9_tva1e1b1nw8zida0e0b0xyg1[:,-1,...] = 1  # set the last slice to 1 rather than use value rolled from v[0]
    # ###Mask w9 for the user defined profiles. This is required for the distribution at season start.
    # ### This requires a t axis for the dams because of the transfer between genotype (slices of the g axis).
    # mask_w9nut_va1e1b1nzida0e0b0xyg1w = np.sum(np.roll(mask_nut_va1e1b1nwzida0e0b0xyg1[..., na], -1, axis = 0) *
    #                                             (np.trunc(index_wzida0e0b0xyg1[..., na] / step_w9_tva1e1b1nw8zida0e0b0xyg1[..., na])
    #                                              == index_w1 / step_w9_tva1e1b1nw8zida0e0b0xyg1[..., na]),axis=w_pos-1) > 0 #keepdims so w8 is dropped, to allow move
    # mask_w9nut_va1e1b1nw9zida0e0b0xyg1 = np.moveaxis(mask_w9nut_va1e1b1nzida0e0b0xyg1w, -1, w_pos)  #move w axis to w8 position
    ###Create a mask for the distribution of w8 to w9, with w9 in the w position for the ffcfw_dest_wg
    ### This requires a t axis for the dams because of the transfer between genotype (slices of the g axis)
    # mask_w9vars_tva1e1b1nw9zida0e0b0xyg1 = (np.trunc(index_wzida0e0b0xyg1 / step_w9_tva1e1b1nw8zida0e0b0xyg1)
    #                                                  == index_wzida0e0b0xyg1 / step_w9_tva1e1b1nw8zida0e0b0xyg1)
    ###Combine the w9vars mask and the user nutrition mask
    # mask_w9vars_tva1e1b1nw9zida0e0b0xyg1 = mask_w9vars_tva1e1b1nw9zida0e0b0xyg1 * mask_w9nut_va1e1b1nw9zida0e0b0xyg1

    ##offs
    ###previous condense date - used to calc number of fvps since last condensing
    prev_condense_date_va1e1b1nwzida0e0b0xyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3.copy()
    prev_condense_date_va1e1b1nwzida0e0b0xyg3[~(dvp_type_va1e1b1nwzida0e0b0xyg3==condense_vtype3)] = 0
    prev_condense_date_va1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(prev_condense_date_va1e1b1nwzida0e0b0xyg3, axis=0)
    ###calc the number of fvps in each dvp and the number of fvps since condense
    n_fvps_va1e1b1nwzida0e0b0xyg3 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg3.shape)
    n_prior_fvps_va1e1b1nwzida0e0b0xyg3 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg3.shape)
    for v in range(dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0]):
        dvp_start_v = dvp_start_va1e1b1nwzida0e0b0xyg3[v, ...]
        prev_condense_date_v = prev_condense_date_va1e1b1nwzida0e0b0xyg3[v, ...]
        n_fvp_since_condense_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= prev_condense_date_v)
                                & (fvp_start_fa1e1b1nwzida0e0b0xyg3 < dvp_start_v)).sum(axis=0)
        if v < dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0] - 1:
            dvp_end_v = dvp_start_va1e1b1nwzida0e0b0xyg3[v + 1, ...]
            n_fvp_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= dvp_start_v)
                       & (fvp_start_fa1e1b1nwzida0e0b0xyg3 < dvp_end_v)).sum(axis=0)
        else:  # the final DVP gets special treatment because of wrap around
            n_fvp_v = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= dvp_start_v)).sum(axis=0)
        n_fvps_va1e1b1nwzida0e0b0xyg3[v,...] = n_fvp_v
        n_prior_fvps_va1e1b1nwzida0e0b0xyg3[v,...] = n_fvp_since_condense_v

    ###Steps for ‘Numbers Requires’ constraint is determined by the number of prior FVPs
    step_con_req_va1e1b1nw8zida0e0b0xyg3 = np.power(n_fs_offs, (n_fvps_percondense_offs
                                                              - n_prior_fvps_va1e1b1nwzida0e0b0xyg3))  #step is based on the FVPs at the start of the DVP
    ###Steps for the w8 decision variables is determined by the number of current & prior FVPs. w8 is the DV this period
    step_dv_va1e1b1nw8zida0e0b0xyg3 = np.power(n_fs_offs, (n_fvps_percondense_offs
                                                         - n_prior_fvps_va1e1b1nwzida0e0b0xyg3
                                                         - n_fvps_va1e1b1nwzida0e0b0xyg3))  #step is based on the FVPs at the start of the DVP
    ###Steps for ‘Numbers Provides’
    ###The 'provide' constraint is the same as the DV (because each DVP provides itself) except when the next DVP is condensing
    step_con_prov_va1e1b1nw8zida0e0b0xyg3w9 = fun.f_update(step_dv_va1e1b1nw8zida0e0b0xyg3,
                                                           np.power(n_fs_offs, n_fvps_percondense_offs)
                                                         , dvp_type_next_va1e1b1nwzida0e0b0xyg3 == condense_vtype3)[..., na]
    ###Mask the decision variables that are not active in this DVP in the matrix - because they share a common nutrition history (broadcast across t axis)
    mask_w8vars_va1e1b1nw8zida0e0b0xyg3 = (index_wzida0e0b0xyg3 % step_dv_va1e1b1nw8zida0e0b0xyg3) == 0
    ###mask for user defined nutrition profiles (this allows the user to exclude certain nutrition patterns e.g. high high high or low low low) 
    ### this mask is combined with the main w8 & w9 masks below
    mask_nut_va1e1b1nWzida0e0b0xyg3 = np.take_along_axis(mask_nut_sa1e1b1nWzida0e0b0xyg3, a_prev_s_va1e1b1nwzida0e0b0xyg3, axis=0)
    ####association between the shortlist of nutrition profile inputs and the full range of LW patterns that include starting LW
    a_shortlist_w3 = index_w3 % len_nut_offs
    mask_nut_va1e1b1nwzida0e0b0xyg3 = mask_nut_va1e1b1nWzida0e0b0xyg3[:,:,:,:,:,a_shortlist_w3,...]  # expands the nutrition mask to all lw patterns.
    ####match the pattern requested with the pattern that is the 'history' for that pattern in previous DVPs
    mask_w8nut_va1e1b1nzida0e0b0xyg3w = np.sum(mask_nut_va1e1b1nwzida0e0b0xyg3[...,na] *
                                               (np.trunc(index_wzida0e0b0xyg3[...,na] / step_dv_va1e1b1nw8zida0e0b0xyg3[..., na])
                                                == index_w3 / step_dv_va1e1b1nw8zida0e0b0xyg3[...,na]),
                                               axis=w_pos-1) > 0 #don't keepdims
    mask_w8nut_va1e1b1nwzida0e0b0xyg3 = np.moveaxis(mask_w8nut_va1e1b1nzida0e0b0xyg3w,-1,w_pos) #move w axis to correct w position
    ###Combine the w8vars mask and the user nutrition mask
    mask_w8vars_va1e1b1nw8zida0e0b0xyg3 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_w8nut_va1e1b1nwzida0e0b0xyg3
    ###Mask numbers provided based on the steps (with a t axis) and the next dvp type (with a t axis) (t0&1 are sold and never transfer so the mask doesn't mean anything for them. for t2 animals always transfer to themselves unless dvpnext is 'condense')
    dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3 = np.logical_or(dvp_type_next_va1e1b1nwzida0e0b0xyg3 == condense_vtype3
                                                               , dvp_type_next_va1e1b1nwzida0e0b0xyg3 == season_vtype3) #when distribution occurs any w8 can provide w9
    ###Mask the provide constraint (w9)
    mask_numbers_provw8w9_va1e1b1nw8zida0e0b0xyg3w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3[...,na] \
                        * (np.trunc((index_wzida0e0b0xyg3[...,na] * np.logical_not(dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3[...,na])
                                     + index_w3 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3[...,na])
                                    / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9) == index_w3 / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9)
    ###Mask numbers required from the previous period (broadcast across t axis) - Note: req does not need a t axis because the destination decision variable don’t change for the transfer
    mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg3w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3[...,na] \
                        * (np.trunc(index_wzida0e0b0xyg3 / step_con_req_va1e1b1nw8zida0e0b0xyg3)[...,na]
                           == index_w3 / step_con_req_va1e1b1nw8zida0e0b0xyg3[...,na])
    ###Create a mask for the constraint. This is used in the distribution of w8 to w9
    ### There are only Trues when next dvp is distribution
    mask_dest_va1e1b1nwzida0e0b0xyg3 = (np.trunc(index_wzida0e0b0xyg3 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3
                                                / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9[...,0])
                                       == index_wzida0e0b0xyg3 / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9[...,0])
    ####Add the nut mask so that distribution can't occur to a masked w slice.
    mask_dest_va1e1b1nwzida0e0b0xyg3 = mask_dest_va1e1b1nwzida0e0b0xyg3 * mask_w8nut_va1e1b1nwzida0e0b0xyg3

    ####################################################
    #Masking numbers transferred to other ram groups   #
    ####################################################
    ''' Mask numbers transferred - these mask remove unnecessary decision variables for:
               transfers to different sires between dvps that are not prejoining
               sales in dvps that are not shearing (t[0]) or not scanning (t[1])'''
    ##dams
    ###create a t mask for dam decision variables - an animal that is transferring between ram groups only has parameters in the dvp that is transfer
    ###dvp0 has no transfer even though it is dvp type 0
    ###animals that are sold t[0] & t[1] only exist if the period is sale. Note t[1] sale is already masked for scan>=1 & for dry ewes only
    period_is_transfer_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(period_is_transfer_tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)[:,:,:,0:1,...] #slice e[0] transfer not effected by e
    mask_tvars_tva1e1b1nw8zida0e0b0xyg1 = np.logical_or(np.logical_and(period_is_transfer_tva1e1b1nwzida0e0b0xyg1, index_va1e1b1nwzida0e0b0xyg1 != 0)
                                                        , (a_g1_tpa1e1b1nwzida0e0b0xyg1 == index_g1))
    period_is_sale_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
    period_is_sale_k2tva1e1b1nwzida0e0b0xyg1 = np.sum(period_is_sale_tva1e1b1nwzida0e0b0xyg1
                                                      * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1)
                                                      , axis=(e1_pos, b1_pos), keepdims=True)>0
    mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1 = np.logical_and(mask_tvars_tva1e1b1nw8zida0e0b0xyg1,
                                                           np.logical_or(period_is_sale_k2tva1e1b1nwzida0e0b0xyg1,
                                                                         index_tva1e1b1nw8zida0e0b0xyg1 >= 2))
    ####make k5 version of mask (used for npw), index_k5 + 2 is allowing for NM & 00 that are 1st 2 entries in the k2cluster that don't exist in the k5cluster
    period_is_sale_k5tva1e1b1nwzida0e0b0xyg1 = np.sum(period_is_sale_tva1e1b1nwzida0e0b0xyg1
                                                      * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k5tva1e1b1nwzida0e0b0xyg3 + 2)
                                                      , axis=(e1_pos, b1_pos), keepdims=True)>0
    mask_tvars_k5tva1e1b1nw8zida0e0b0xyg1 = np.logical_and(mask_tvars_tva1e1b1nw8zida0e0b0xyg1,
                                                           np.logical_or(period_is_sale_k5tva1e1b1nwzida0e0b0xyg1,
                                                                         index_tva1e1b1nw8zida0e0b0xyg1 >= 2))  # t>=2 is to
    #todo alternative is mask_tvars_k5 = mask_tvars_k2[2:8] (without calculating period_is_sale_k5). This is not any less flexible than '+ 2' in formula for period is sale
    #todo a 2nd alternative would be to create a_k2_k5tva1e1b1nwzida0e0b0xyg (which is essentially a_b1_b0xyg but with allowance for extra slices if scan == 4)
    # then create mask_tvars_k5 from mask_tvars_k2 replace line "*  (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == a_k2_k5tva1e1b1nwzida0e0b0xyg3)"

    ###numbers are provided by g1 to g9 (a_g1_tg1) in the dvps that are transfer periods (mask_tvars) for the t slices that are not sale (transfer exists)
    #### the association is being used as (a_g9_tg1 == index_g9)
    mask_numbers_provt_k2tva1e1b1nwzida0e0b0xyg1g9 = mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[...,na] * transfer_exists_tpa1e1b1nwzida0e0b0xyg1[..., na]  \
                                                      * (a_g1_tpa1e1b1nwzida0e0b0xyg1[..., na] == index_g1[..., na, :])

    ###numbers are required across the identity array between g1 & g9 in the periods that the transfer decision variable exists
    mask_numbers_reqt_k2tva1e1b1nwzida0e0b0xyg1g9 = mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[...,na] \
                                                    * (index_g9 == index_g1g)

    ##0ffs
    ##mask the t array so that only slice t0 provides numbers
    mask_numbers_provt_tva1e1b1nw8zida0e0b0xyg3w9 = (index_tva1e1b1nw8zida0e0b0xyg3w9 == 0)

    #########################################
    #Masking numbers for forced sale of drys#
    #########################################
    ''' Create a mask to remove retaining dry dams when sale of drys is forced
    The transfer is removed if all the following are true: they are in the dry cluster that is not a sale group, next DVP is prejoining, ewes are scanned, dry sales are forced.
    Dry dams must be sold before the next prejoining (e.g. they can be sold in any sale opp).'''
    #todo would be good to be able to specify if sale occurs at scanning, shearing or any. Tricky because shearing can be in different dvps and there is no drys identified in prejoining dvp.
    ##convert o to v.
    dry_sales_forced_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_dry_sales_forced_o'], p_pos)
    dry_sales_forced_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dry_sales_forced_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0) #increments at prejoining
    dry_sales_forced_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dry_sales_forced_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e is the same for prejoining
    ##make mask
    mask_numbers_provdry_k28k29tva1e1b1nwzida0e0b0xyg1 = np.logical_not((index_k28k29tva1e1b1nwzida0e0b0xyg1 == 1) * (index_tva1e1b1nw8zida0e0b0xyg1 >= 2)
                                                         * (scan_va1e1b1nwzida0e0b0xyg1 >= 1) #dvp1 because that's the scanning dvp
                                                         * (dvp_type_next_va1e1b1nwzida0e0b0xyg1 == prejoin_vtype1)
                                                         * dry_sales_forced_va1e1b1nwzida0e0b0xyg1)
    #todo add sale of drys at birth when gbal is activated with a np.logical_and to the following code
                    # = np.logical_not((index_k28k29tva1e1b1nwzida0e0b0xyg1 == b is 10, 20 or 30) * (index_tva1e1b1nw8zida0e0b0xyg1 >= 2)
                    #                  * (gbal_va1e1b1nwzida0e0b0xyg1 >= 1)  #dvp1 because that's the scanning dvp
                    #                  * (dvp_type_next_va1e1b1nwzida0e0b0xyg1 == prejoin_vtype1)
                    #                  * dry_sales_forced_va1e1b1nwzida0e0b0xyg1)
    ################################
    # Create Season transfer masks #
    ################################
    '''If a season is not identified then it does not transfer any parameters. Therefore to reduce size we can mask
    all parameters with a z8 axis. We also require a z8z9 mask which controls transfer params.
    z8z9 is required for parameters from the previous period that provide/require in the current period because 
    if a season is identified in a given dvp it provides to multiple z slices in the next dvp.'''

    ##inputs
    date_initiate_z = zfun.f_seasonal_inp(pinp.general['i_date_initiate_z'], numpy=True, axis=0)
    date_initiate_zidaebxyg = fun.f_expand(date_initiate_z, z_pos)
    index_zidaebxyg = fun.f_expand(index_z, z_pos)
    ##dams child parent transfer
    mask_provwithinz8z9_va1e1b1nwzida0e0b0xyg1z9, mask_provbetweenz8z9_va1e1b1nwzida0e0b0xyg1z9, \
    mask_childz_reqwithin_va1e1b1nwzida0e0b0xyg1, mask_childz_reqbetween_va1e1b1nwzida0e0b0xyg1 = zfun.f_season_transfer_mask(
        dvp_start_va1e1b1nwzida0e0b0xyg1, period_is_seasonstart_pz=dvp_type_va1e1b1nwzida0e0b0xyg1==season_vtype1, z_pos=z_pos)
    mask_z8var_va1e1b1nwzida0e0b0xyg1 = zfun.f_season_transfer_mask(dvp_start_va1e1b1nwzida0e0b0xyg1, z_pos=z_pos, mask=True)
    ###create z8z9 param that is index with v
    ####cluster e and b (e axis is active from the dvp dates)
    mask_childz_reqwithin_k2tva1e1b1nwzida0e0b0xyg1 = 1 * (np.sum(mask_childz_reqwithin_va1e1b1nwzida0e0b0xyg1
                                                                  * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k2tva1e1b1nwzida0e0b0xyg1),
                                                                  axis=(e1_pos,b1_pos), keepdims=True) > 0)
    mask_childz_reqbetween_k2tva1e1b1nwzida0e0b0xyg1 = 1 * (np.sum(mask_childz_reqbetween_va1e1b1nwzida0e0b0xyg1
                                                                  * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k2tva1e1b1nwzida0e0b0xyg1),
                                                                  axis=(e1_pos,b1_pos), keepdims=True) > 0)
    mask_provwithinz8z9_k2tva1e1b1nwzida0e0b0xyg1z9 = 1 * (np.sum(mask_provwithinz8z9_va1e1b1nwzida0e0b0xyg1z9
                                                                  * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k2tva1e1b1nwzida0e0b0xyg1)[...,na],
                                                                  axis=(e1_pos-1,b1_pos-1), keepdims=True) > 0)
    mask_provbetweenz8z9_k2tva1e1b1nwzida0e0b0xyg1z9 = 1 * (np.sum(mask_provbetweenz8z9_va1e1b1nwzida0e0b0xyg1z9
                                                                  * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k2tva1e1b1nwzida0e0b0xyg1)[...,na],
                                                                  axis=(e1_pos-1,b1_pos-1), keepdims=True) > 0)

    ##offs child parent transfer
    mask_provwithinz8z9_va1e1b1nwzida0e0b0xyg3z9, mask_provbetweenz8z9_va1e1b1nwzida0e0b0xyg3z9, \
    mask_childz_reqwithin_va1e1b1nwzida0e0b0xyg3, mask_childz_reqbetween_va1e1b1nwzida0e0b0xyg3 = zfun.f_season_transfer_mask(
        dvp_start_va1e1b1nwzida0e0b0xyg3, period_is_seasonstart_pz=dvp_type_va1e1b1nwzida0e0b0xyg3==season_vtype3, z_pos=z_pos)
    mask_z8var_va1e1b1nwzida0e0b0xyg3 = zfun.f_season_transfer_mask(dvp_start_va1e1b1nwzida0e0b0xyg3, z_pos=z_pos, mask=True)
    ###create z8z9 param that is index with v
    ####cluster d (d axis is active from the dvp dates)
    mask_childz_reqwithin_k3k5tva1e1b1nwzida0e0b0xyg3 = 1 * (np.sum(mask_childz_reqwithin_va1e1b1nwzida0e0b0xyg3
                                                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                               axis=d_pos, keepdims=True) > 0)
    mask_childz_reqbetween_k3k5tva1e1b1nwzida0e0b0xyg3 = 1 * (np.sum(mask_childz_reqbetween_va1e1b1nwzida0e0b0xyg3
                                                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                               axis=d_pos, keepdims=True) > 0)
    mask_provwithinz8z9_k3k5tva1e1b1nwzida0e0b0xyg3z9 = 1 * (np.sum(mask_provwithinz8z9_va1e1b1nwzida0e0b0xyg3z9
                                                                  * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na],
                                                                    axis=d_pos-1, keepdims=True) > 0)
    mask_provbetweenz8z9_k3k5tva1e1b1nwzida0e0b0xyg3z9 = 1 * (np.sum(mask_provbetweenz8z9_va1e1b1nwzida0e0b0xyg3z9
                                                                  * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na],
                                                                    axis=d_pos-1, keepdims=True) > 0)

    ##p6z mask - this is only for masking sire because they don't have a v axis
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(per.f_feed_periods()[:-1], z_pos=-1, mask=True)
    mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg = fun.f_expand(mask_fp_z8var_p6z, left_pos=z_pos, left_pos2=p_pos-2, right_pos2=z_pos)

    ##make p7z8 mask - used to mask sire $ stuff
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    mask_z8var_p7tva1e1b1nwzida0e0b0xyg = fun.f_expand(mask_season_p7z, left_pos=z_pos, left_pos2=p_pos-2, right_pos2=z_pos)

    ##p5z mask - used to mask sire labour
    lp_start_p5z = per.f_p_dates_df().iloc[:-1].values #slice off end date of last period
    maskz8_p5z = zfun.f_season_transfer_mask(lp_start_p5z,z_pos=-1,mask=True)
    mask_z8var_p5tva1e1b1nwzida0e0b0xyg = fun.f_expand(maskz8_p5z, left_pos=z_pos, left_pos2=p_pos-2, right_pos2=z_pos)

    ##################
    #lw distribution #
    ##################
    '''
    Distributing happens at the start of each season/season sequence when all the different seasons are combined back
    to a common season. It also happens when lw is condensed back to the starting number for the livestock year.
    For dams lw distributing is also required at prejoining when dams can be transferred to different sires and 
    dams with different LSLN in the previous reproduction cycle are combined for the next cycle. 

    Note: 1. For dams condensing for the livestock year is carried out at prejoining which coincides with when
          distribution is required. The generator can handle dam condensing and prejoining to be in different dvps 
          however the distribution below requires condensing to occur at prejoining (although it may be possible 
          to change the distribution code to handle condensing and prejoining in different dvps).

          2. Distribution of animals is controlled by LW and distributing maintains the current average LW. The
          errors associated with difference in wool production that are not 100% correlated with LW will be
          minimised if condensing and distributing are occurring soon after shearing.


    What this section does:
    Calc the ffcfw being distributed from and to.
    Calc the proportion of the source weight allocated to each destination weight.

    '''
    lwdist_start = time.time()

    ## calc the ‘source’ weight of the animal at the end of each period in which they can be transferred
    ###dams - the period is based on period_is_transfer which points at the nextperiod_is_prejoin for the destination g1 slice
    ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                                period_is_tp=period_is_transfer_tpa1e1b1nwzida0e0b0xyg1)  #numbers not required for ffcfw
    ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg1,
                                                                    a_p_va1e1b1nwzida0e0b0xyg1,
                                                                    a_v_pa1e1b1nwzida0e0b0xyg1)
    ffcfw_source_season_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                             period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg)  #numbers not required for ffcfw
    ffcfw_source_season_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_source_season_tva1e1b1nwzida0e0b0xyg1,
                                                                 a_p_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
    ###offs
    ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                               period_is_tp=nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg3,
                                                                   a_p_va1e1b1nwzida0e0b0xyg3,
                                                                   a_v_pa1e1b1nwzida0e0b0xyg3)
    ffcfw_source_season_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                             period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_source_season_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_source_season_tva1e1b1nwzida0e0b0xyg3,
                                                                 a_p_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)

    ## calc the ‘destination’ weight of each group of animal at the end of the period prior to the transfer (transfer is next period is prejoining for the destination animal)
    ### for dams select the ‘destination’ condensed weight for the ‘source’ slices using a_g1_tg1
    ffcfw_condensed_tdams = np.take_along_axis(o_ffcfw_condensed_tpdams, a_g1_tpa1e1b1nwzida0e0b0xyg1, -1)
    ### Convert from p to v.
    #### for dams the period is based on period_is_transfer which points at the nextperiod_is_prejoin for the destination g1 slice
    ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(ffcfw_condensed_tdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                              period_is_tp=period_is_transfer_tpa1e1b1nwzida0e0b0xyg1)  #numbers not required for ffcfw
    ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg1,
                                                                  a_p_va1e1b1nwzida0e0b0xyg1,
                                                                  a_v_pa1e1b1nwzida0e0b0xyg1)
    ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_season_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                           period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg)  #numbers not required for ffcfw
    ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg1,
                                                               a_p_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
    ###offs
    ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_condensed_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                             period_is_tp=nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg3,
                                                                 a_p_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)
    ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_season_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                           period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg3,
                                                               a_p_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)

    ##distributing at condensing - all lws back to starting number of LWs and dams to different sires at prejoining
    ###t0 and t1 are distributed however this is not used because t0 and t1 don't transfer to next dvp
    distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9 = sfun.f1_lw_distribution(
        ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg1, ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg1,
        mask_dest_tva1e1b1nwzida0e0b0xyg1,
        index_wzida0e0b0xyg1, dvp_type_next_tva1e1b1nwzida0e0b0xyg1[..., na], condense_vtype1)
    distribution_condense_tva1e1b1nw8zida0e0b0xyg3w9 = sfun.f1_lw_distribution(
        ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg3, ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg3,
        mask_dest_va1e1b1nwzida0e0b0xyg3[na],
        index_wzida0e0b0xyg3, dvp_type_next_va1e1b1nwzida0e0b0xyg3[..., na], condense_vtype3)

    ##redistribute at season start - all seasons back into a common season.
    distribution_season_tva1e1b1nw8zida0e0b0xyg1w9 = sfun.f1_lw_distribution(
        ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg1, ffcfw_source_season_tva1e1b1nwzida0e0b0xyg1,
        mask_dest_tva1e1b1nwzida0e0b0xyg1,
        index_wzida0e0b0xyg1, dvp_type_next_va1e1b1nwzida0e0b0xyg1[..., na], season_vtype1)
    distribution_season_tva1e1b1nw8zida0e0b0xyg3w9 = sfun.f1_lw_distribution(
        ffcfw_dest_season_tva1e1b1nwzida0e0b0xyg3, ffcfw_source_season_tva1e1b1nwzida0e0b0xyg3,
        mask_dest_va1e1b1nwzida0e0b0xyg3[na],
        index_wzida0e0b0xyg3, dvp_type_next_va1e1b1nwzida0e0b0xyg3[..., na], season_vtype3)

    ##combine distributions
    distribution_tva1e1b1nw8zida0e0b0xyg1w9 = distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9 * distribution_season_tva1e1b1nw8zida0e0b0xyg1w9
    distribution_tva1e1b1nw8zida0e0b0xyg3w9 = distribution_condense_tva1e1b1nw8zida0e0b0xyg3w9 * distribution_season_tva1e1b1nw8zida0e0b0xyg3w9

    # ##store cluster associations for use in creating the optimal feedsupply at the end of the trial
    # pkl_fs_info['distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9'] = distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9
    # a_prev_condense_va1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate((dvp_type_va1e1b1nwzida0e0b0xyg1==condense_vtype1) * index_va1e1b1nwzida0e0b0xyg1,
    #     axis=p_pos)  # return the index for each  condense dvp or index for previous condense dvp if the dvp is not condense
    # pkl_fs_info['a_prev_condense_va1e1b1nwzida0e0b0xyg1'] = a_prev_condense_va1e1b1nwzida0e0b0xyg1

    ###########################
    #create production params #
    ###########################
    '''some sire params don't go through here because no associations are required'''
    production_param_start = time.time()

    ##emissions
    ch4_animal_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', ch4_animal_tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    n2o_animal_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', n2o_animal_tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    ch4_animal_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', ch4_animal_tva1e1b1nwzida0e0b0xyg1
                                                                      , a_k2cluster_va1e1b1nwzida0e0b0xyg1
                                                                      , index_k2tva1e1b1nwzida0e0b0xyg1
                                                                      , numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1
                                                                      , mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1
                                                                                 *mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                                                 *mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1))
    n2o_animal_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', n2o_animal_tva1e1b1nwzida0e0b0xyg1
                                                                      , a_k2cluster_va1e1b1nwzida0e0b0xyg1
                                                                      , index_k2tva1e1b1nwzida0e0b0xyg1
                                                                      , numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1
                                                                      , mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1
                                                                                 *mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                                                 *mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1))
    ch4_animal_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', ch4_animal_tva1e1b1nwzida0e0b0xyg3
                                                                        , a_k3cluster_da0e0b0xyg3
                                                                        , index_k3k5tva1e1b1nwzida0e0b0xyg3
                                                                        , a_k5cluster_da0e0b0xyg3
                                                                        , index_k5tva1e1b1nwzida0e0b0xyg3
                                                                        , numbers_start_tva1e1b1nwzida0e0b0xyg3
                                                                        , mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)
    n2o_animal_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', n2o_animal_tva1e1b1nwzida0e0b0xyg3
                                                                        , a_k3cluster_da0e0b0xyg3
                                                                        , index_k3k5tva1e1b1nwzida0e0b0xyg3
                                                                        , a_k5cluster_da0e0b0xyg3
                                                                        , index_k5tva1e1b1nwzida0e0b0xyg3
                                                                        , numbers_start_tva1e1b1nwzida0e0b0xyg3
                                                                        , mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    co2e_animal_tva1e1b1nwzida0e0b0xyg0 = ch4_animal_tva1e1b1nwzida0e0b0xyg0 * uinp.emissions['i_ch4_gwp_factor'] + n2o_animal_tva1e1b1nwzida0e0b0xyg0 * uinp.emissions['i_n2o_gwp_factor']
    co2e_animal_k2tva1e1b1nwzida0e0b0xyg1 = ch4_animal_k2tva1e1b1nwzida0e0b0xyg1 * uinp.emissions['i_ch4_gwp_factor'] + n2o_animal_k2tva1e1b1nwzida0e0b0xyg1 * uinp.emissions['i_n2o_gwp_factor']
    co2e_animal_k3k5tva1e1b1nwzida0e0b0xyg3 = ch4_animal_k3k5tva1e1b1nwzida0e0b0xyg3 * uinp.emissions['i_ch4_gwp_factor'] + n2o_animal_k3k5tva1e1b1nwzida0e0b0xyg3 * uinp.emissions['i_n2o_gwp_factor']


    ##mei
    mei_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', mei_p6ftva1e1b1nwzida0e0b0xyg0,
                                                                   numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                   mask_vg=mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    mei_k2p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', mei_p6ftva1e1b1nwzida0e0b0xyg1
                                                                      , a_k2cluster_va1e1b1nwzida0e0b0xyg1
                                                                      , index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...]
                                                                      , numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1
                                                                      , mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1
                                                                                 *mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                                                 *mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,...]))
    mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', mei_p6ftva1e1b1nwzida0e0b0xyg3
                                                                        , a_k3cluster_da0e0b0xyg3
                                                                        , index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...]
                                                                        , a_k5cluster_da0e0b0xyg3
                                                                        , index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...]
                                                                        , numbers_start_tva1e1b1nwzida0e0b0xyg3
                                                                        , mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##pi
    pi_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', pi_p6ftva1e1b1nwzida0e0b0xyg0,
                                                                  numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                  mask_vg=mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    pi_k2p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', pi_p6ftva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,...]))
    pi_k3k5p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', pi_p6ftva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##cashflow & cost & working capital
    purchcost_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', purchcost_p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                           mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    cashflow_c1p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', cashflow_c1p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                          mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    cashflow_k2c1p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', cashflow_c1p7tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,...]))
    cashflow_k3k5c1p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', cashflow_c1p7tva1e1b1nwzida0e0b0xyg3,
                                                    a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)
    purchcost_wc_c0p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', purchcost_wc_c0p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                              mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    wc_c0p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', wc_c0p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                      mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    wc_k2c0p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', wc_c0p7tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,...]))
    wc_k3k5c0p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', wc_c0p7tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)
    cost_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', cost_p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                      mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    cost_k2p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', cost_p7tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                            index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                            mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    cost_k3k5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', cost_p7tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                            a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                            mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##asset value
    ###set start of season assetvalue for season trade to be the same across t & z axis (it has been calculated at the end of the first generator period but we want to reflect the start of the period - at the start all t & z are the same). All T & z need to be the same or the model can optimise trade vale.
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1[1, 0, ...] = np.take_along_axis(fun.f_dynamic_slice(assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1, z_pos, 0, 1), a_t_tpg1[na,na,...], axis=p_pos-1)[1, 0, ...]
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3[1, 0, ...] = np.take_along_axis(fun.f_dynamic_slice(assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3, z_pos, 0, 1), a_t_tpg3[na,na,...], axis=p_pos-1)[1, 0, ...]
    ###back calculate the end of season asset value at the end of the last dvp. Technically the asset value at the end
    #### of the season should equal the asset value at the start of the next season (because this is the
    ### same point in time). However, at the season start a new animal is formed (from the weighted average of all
    ### the seasons), so we can't use the same asset value parameter for the end and the start. We also can't simply calc
    ### the asset value for both periods because even just one generator period has a bit of effect on asset value
    ### due to mortality and LW change and this allowed the model to optimise tradevalue in a way that it shouldn't.
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0[2,-1,...] = assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0[1,0,...] #sires don't get distributed at season start so asset value end = start
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1[2,-1,...] = np.sum(np.swapaxes(np.roll(assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1, shift=-1, axis=p_pos)[1,0,...,na], axis1=w_pos-1, axis2=-1)
           * distribution_season_tva1e1b1nw8zida0e0b0xyg1w9, axis=-1)
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3[2,-1,...] = np.sum(np.swapaxes(np.roll(assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3, shift=-1, axis=p_pos)[1,0,...,na], axis1=w_pos-1, axis2=-1)
           * distribution_season_tva1e1b1nw8zida0e0b0xyg3w9, axis=-1)
    ###cluster
    assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                          mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    assetvalue_a5k2p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', assetvalue_a5p7tva1e1b1nwzida0e0b0xyg1[:,na,...], a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    assetvalue_a5k3k5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', assetvalue_a5p7tva1e1b1nwzida0e0b0xyg3[:,na,na,...], a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##labour - manager
    lab_manager_p5tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', labour_l2p5tva1e1b1nwzida0e0b0xyg0[0], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                            mask_vg=mask_z8var_p5tva1e1b1nwzida0e0b0xyg)
    lab_manager_k2p5tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', labour_l2p5tva1e1b1nwzida0e0b0xyg1[0], a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    lab_manager_k3k5p5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', labour_l2p5tva1e1b1nwzida0e0b0xyg3[0], a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##labour - permanent
    lab_perm_p5tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', labour_l2p5tva1e1b1nwzida0e0b0xyg0[1], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                         mask_vg=mask_z8var_p5tva1e1b1nwzida0e0b0xyg)
    lab_perm_k2p5tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', labour_l2p5tva1e1b1nwzida0e0b0xyg1[1], a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    lab_perm_k3k5p5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', labour_l2p5tva1e1b1nwzida0e0b0xyg3[1], a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##labour - anyone
    lab_anyone_p5tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', labour_l2p5tva1e1b1nwzida0e0b0xyg0[2], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                           mask_vg=mask_z8var_p5tva1e1b1nwzida0e0b0xyg)
    lab_anyone_k2p5tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', labour_l2p5tva1e1b1nwzida0e0b0xyg1[2], a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    lab_anyone_k3k5p5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', labour_l2p5tva1e1b1nwzida0e0b0xyg3[2], a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##infrastructure
    infrastructure_h1tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', infrastructure_h1tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    infrastructure_k2h1tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', infrastructure_h1tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    infrastructure_k3k5p5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', infrastructure_h1tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)



    ###########################
    #Livestock trade          #
    ###########################
    '''
    Similar to benchmarking we have a 'Livestock Trading Profit' concept. This is required because when using the DSP
    stock numbers are averaged at the beginning of the each season. Thus essentially the poor season (with less stock numbers)
    buys some stock from the good season. This is not reflected as a cash transaction and doesn't effect profit
    but when using risk aversion the model opts to sell more sheep in the poor year when the utility is higher. This gives the
    appearance that risk is being lowered but the bit that is not captured is that at the start of the season
    the numbers are averaged across z. Which essentially means the good season is selling more to the poor season.
    The 'Livestock Trading Profit' captures this transfer.

    LTP = livestock value at end of year - value at start of the year.
    The value of the stock at the end of the year minus the value at the start of the year should tally to 0 across all z.
    The concept is a little hard to visualise because the generator is over multiple years (best to think about each yr
    in the gen as a different age group animal). Because the model is in steady state the animals at the start equal
    the animals at the end.
    
    Potentially we could alter the cost a bit to include a transaction cost. To reflect the effort and genetic loss
    associated with selling sheep in the poor year and then having to buy them back at the market.

    There is a small amount of error in the calculation (about 8k) which will have a small impact on the risk included version (tradevalue is only included in the obj in the risk version).
    Unsure where the 8k comes from (MRY spent a lot of time improving the trade value calc which improved the error a lot but there is still 8k unexplained).
    '''
    ##trade value - end value minus start value (do it here to save time in pyomo)
    ## a5[1] start of season & a5[2] end of season.
    ###sire
    start_assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0[1]
    end_assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0[2]
    assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = end_assetvalue_p7tva1e1b1nwzida0e0b0xyg0 - start_assetvalue_p7tva1e1b1nwzida0e0b0xyg0
    ###dams
    start_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = assetvalue_a5k2p7tva1e1b1nwzida0e0b0xyg1[1]
    end_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = assetvalue_a5k2p7tva1e1b1nwzida0e0b0xyg1[2] * (index_tva1e1b1nw8zida0e0b0xyg1>=2) #only retained animals have an end value (animals that are sold don't have a value)
    assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = end_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 - start_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1
    ###offs
    start_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = assetvalue_a5k3k5p7tva1e1b1nwzida0e0b0xyg3[1]
    end_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = assetvalue_a5k3k5p7tva1e1b1nwzida0e0b0xyg3[2] * (index_tva1e1b1nw8zida0e0b0xyg3==0) #only retained animals have an end value (animals that are sold don't have a value)
    assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = end_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 - start_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3

    ##if steady state set everything to 0 - there is no "season start" in SE model so there is no opportunity for trade between seasons
    if bool_steady_state:
        ###sire
        start_assetvalue_p7tva1e1b1nwzida0e0b0xyg0[...] = 0
        end_assetvalue_p7tva1e1b1nwzida0e0b0xyg0[...] = 0
        assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = end_assetvalue_p7tva1e1b1nwzida0e0b0xyg0 - start_assetvalue_p7tva1e1b1nwzida0e0b0xyg0
        ###dams
        start_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1[...] = 0
        end_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1[...] = 0
        assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = end_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 - start_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1
        ###offs
        start_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3[...] = 0
        end_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3[...] = 0
        assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = end_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 - start_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3

    ##if risk is not included then tradevalue is not included in objective. For standard dsp the tradevalue is only required in the pnl report.
    ## if risk is included then need to include tradevalue in obj to stop the model selling extra sheep in poor year to increase utility.
    if not uinp.general['i_inc_risk']:
        assetvalue_p7tva1e1b1nwzida0e0b0xyg0[...] = 0
        assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1[...] = 0
        assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3[...] = 0

    ###########################
    #create numbers params    #
    ###########################
    number_param_start = time.time()
    ##number of sires available at mating - sire
    numbers_startp8_tva1e1b1nwzida0e0b0xyg0p8 = sfun.f1_create_production_param('sire', numbers_startp8_tva1e1b1nwzida0e0b0xyg0p8,
                                                                               numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0[...,na],
                                                                               pos_offset=1)
    ##number of sires for required for mating - dams
    ### mask the dams for w8 vars, t_vars. Also not mated if they are being transferred to another ram group. A transfer in the mating period (a_g1_tg1!=index_g1) indicates that the dam is going to be mated to another sire at a later date within the same DVP
    t_mask_k2tva1e1b1nw8zida0e0b0xyg1g0p8 = (mask_w8vars_va1e1b1nw8zida0e0b0xyg1* mask_z8var_va1e1b1nwzida0e0b0xyg1 * mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1
                                             * (a_g1_tpa1e1b1nwzida0e0b0xyg1 == index_g1))[...,na,na]
    nsire_k2tva1e1b1nwzida0e0b0xyg1g0p8 = sfun.f1_create_production_param('dams', nsire_tva1e1b1nwzida0e0b0xyg1g0p8,
                                                a_k2cluster_va1e1b1nwzida0e0b0xyg1[...,na,na], index_k2tva1e1b1nwzida0e0b0xyg1[...,na,na],
                                                numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1[...,na,na],
                                                mask_vg=t_mask_k2tva1e1b1nw8zida0e0b0xyg1g0p8, pos_offset=2)
    nsire_k2tva1e1b1nwzida0e0b0xyg1g0p8[0] = 0 #nm animals don't require sires

    ##numbers prov - numbers at the end of a dvp with the cluster of the next dvp divided by start numbers with cluster of current period
    ###dams total provided from this period
    numerator  = 0
    denominator = 0
    for b1 in range(len_b1): #loop on b1 to reduce memory
        numerator += (np.sum(numbers_end_tva1e1b1nwzida0e0b0xyg1[:,:,:,:,b1:b1+1,..., na,na]
                    * mask_numbers_provw8w9_tva1e1b1nw8zida0e0b0xyg1w9[..., na,:]
                    * mask_numbers_provt_k2tva1e1b1nwzida0e0b0xyg1g9[:,na,..., na]
                    * mask_numbers_provdry_k28k29tva1e1b1nwzida0e0b0xyg1[...,na,na]
                    * distribution_tva1e1b1nw8zida0e0b0xyg1w9[:,:,:,:,b1:b1+1,..., na,:]
                    * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,:,:,b1:b1+1,...] == index_k28k29tva1e1b1nwzida0e0b0xyg1)[..., na,na]                #The numerator has both k2 with g9 axis and without. One to reflect the decision variable (k28) and one for the constraint (k29). So I think this is all good
                    * (a_k2cluster_next_tva1e1b1nwzida0e0b0xyg1g9[:,:,:,:,b1:b1+1,...] == index_k29tva1e1b1nwzida0e0b0xyg1g9)[..., na],
                    axis=(e1_pos - 2), keepdims=True))
        denominator += (np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg1[:,:,:,:,b1:b1+1,...] * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,:,:,b1:b1+1,...] == index_k28k29tva1e1b1nwzida0e0b0xyg1),
                    axis=e1_pos, keepdims=True)[..., na,na]) #na for w9 and g9 (use standard cluster without t/g9 axis because the denominator is (the clustering for) the decision variable as at the start of the DVP)
    numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_divide(numerator,denominator, dtype=dtype)

    ####dams transferring between ram groups in the same DVP.
    #### This occurs if the transfer is to a different ram group (a_g1_tg1 != g1) and is occurring from a prejoining dvp which indicates that the transfer is from prejoining dvp to prejoining dvp because the destination ram group is joining after the source
    numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = (numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9
                                                                * ((dvp_type_va1e1b1nwzida0e0b0xyg1[:, :, 0:1, ...] == prejoin_vtype1)
                                                                   * (a_g1_tpa1e1b1nwzida0e0b0xyg1 != index_g1))[...,na,na])   #take slice 0 of e (for prejoining all e slices are the same)
    ####dams providing to the next period (the norm)
    ####Only different from the total because it excludes those providing to the same period
    numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = (numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9
                                                            - numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9)
    ###combine nm and 00 cluster for the numbers provided to the prejoining period (so matrix can optimise choice of joining or not)
    temporary = np.sum(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=1, keepdims=True) * (index_k29tva1e1b1nwzida0e0b0xyg1g9[...,na] == 0)  # put the sum of the k29 in slice 0
    numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary,
                                                                        dvp_type_next_tva1e1b1nwzida0e0b0xyg1[:, :, :, 0:1, ..., na,na] == prejoin_vtype1)  #take slice 0 of e (for prejoining all e slices are the same)
    ###combine nm and 00 cluster (so matrix can optimise choice of joining or not). DVP type of destination is always 0 for the "provide this period" so don't need to test
    numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = np.sum(numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=1, keepdims=True) * (index_k29tva1e1b1nwzida0e0b0xyg1g9[...,na] == 0)  # put the sum of the k29 in slice 0
    ###combine wean numbers at prejoining to allow the matrix to select a different weaning time for the coming yr.
    #todo can't just sum across the 'a' slice (decision variable), to allow a0 to provide a1 we will need another 'a' axis (see google doc) - fix this in version 2
    # temporary = np.sum(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=a1_pos-1, keepdims=True) * (index_a1e1b1nwzida0e0b0xyg[...,na] == 0)
    # numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary, dvp_type_next_tva1e1b1nwzida0e0b0xyg1[:,:,:,0:1,...,na] == 0) #take slice 0 of e (for prejoining all e slices are the same

    ###offs
    numerator  = 0
    denominator = 0
    for b0 in range(len_b0): #loop on b1 to reduce memory
        numerator += np.sum(numbers_end_tva1e1b1nwzida0e0b0xyg3[...,b0:b0+1,:,:,:,na]  * distribution_tva1e1b1nw8zida0e0b0xyg3w9[...,b0:b0+1,:,:,:,:]
                            * mask_numbers_provw8w9_va1e1b1nw8zida0e0b0xyg3w9
                            * (a_k3cluster_da0e0b0xyg3==index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                            * (a_k5cluster_da0e0b0xyg3[...,b0:b0+1,:,:,:]==index_k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                            , axis = (d_pos-1, e0_pos-1), keepdims=True)
        denominator += np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg3[...,b0:b0+1,:,:,:] * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                              * (a_k5cluster_da0e0b0xyg3[...,b0:b0+1,:,:,:]==index_k5tva1e1b1nwzida0e0b0xyg3)
                              , axis = (d_pos, e0_pos), keepdims=True)[...,na]
    numbers_prov_offs_k3k5tva1e1b1nw8zida0e0b0xygw9 = fun.f_divide(numerator,denominator, dtype=dtype) * mask_numbers_provt_tva1e1b1nw8zida0e0b0xyg3w9

    ##numbers required
    ###dams
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = 0
    for b1 in range(len_b1):  # loop on b1 to reduce memory
        numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 +=  np.sum(mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9[...,na,:] * mask_numbers_reqt_k2tva1e1b1nwzida0e0b0xyg1g9[:,na,...,na]
                                                                       * mask_z8var_va1e1b1nwzida0e0b0xyg1[...,na,na]
                                                                       * ((a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,:,:,b1:b1+1,...] == index_k28k29tva1e1b1nwzida0e0b0xyg1)
                                                                          * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,:,:,b1:b1+1,...] == index_k2tva1e1b1nwzida0e0b0xyg1))[...,na,na]
                                                                       , axis = (e1_pos-2), keepdims=True)
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = (numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9>0) *1 #*1 to change to float instead of bool
    ####combine nm and 00 cluster for prejoining to scanning
    temporary = np.sum(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=1, keepdims=True) * (index_k29tva1e1b1nwzida0e0b0xyg1g9[...,na] == 0)  # put the sum of the k29 in slice 0
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary,
                                                                        dvp_type_va1e1b1nwzida0e0b0xyg1[:, :, 0:1, ..., na,na] == prejoin_vtype1)  #take slice 0 of e (for prejoining all e slices are the same)
    ####combine wean numbers at prejoining to allow the matrix to select a different weaning time for the coming yr.
    #todo can't just sum across the 'a' slice (decision variable), to allow a0 to provide a1 we will need another 'a' axis (see google doc)
    # temporary = np.sum(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=a1_pos-1, keepdims=True) * (index_a1e1b1nwzida0e0b0xyg[...,na] == 0)
    # numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary, dvp_type_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...,na] == 0) #take slice 0 of e (for prejoining all e slices are the same


    ###offs
    numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9 =  1*(np.sum((a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                                *(a_k5cluster_da0e0b0xyg3==index_k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                                * mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg3w9
                                                                * mask_z8var_va1e1b1nwzida0e0b0xyg3[...,na]
                                                             , axis = (d_pos-1, b0_pos-1, e0_pos-1), keepdims=True)  >0) #add active v axis
    # numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9 =  1*(np.sum((a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)*(a_k5cluster_da0e0b0xyg3==index_k5tva1e1b1nwzida0e0b0xyg3)
    #                                                          , axis = (d_pos, b0_pos, e0_pos), keepdims=True)[...,na
    #                                                   ] * mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg3w9 * (index_vpa1e1b1nwzida0e0b0xyg3==index_vpa1e1b1nwzida0e0b0xyg3) >0) #add active v axis

    ##Setting the parameters at the end of the generator to 0 removes passing animals into the constraint that links the end of life with the beginning of life.
    numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9[:,:,:,-1,...] = 0
    numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9[:,:,:,-1,...] = 0
    numbers_prov_offs_k3k5tva1e1b1nw8zida0e0b0xygw9[:,:,:,-1,...] = 0
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9[:,:,:,0,...] = 0
    numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9[:,:,:,0,...] = 0


    ##mask dams activity (used in bounds)
    mask_dams_k2tva1e1b1nw8zida0e0b0xyg1 =  1 * (np.sum(mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                        * mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1
                                                        * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1)
                                                        * (a_g1_tpa1e1b1nwzida0e0b0xyg1 == index_g1)
                                                          , axis = (b1_pos, e1_pos), keepdims=True)>0)

    ##mask offs activity (used in bounds)
    mask_offs_k3k5tva1e1b1nw8zida0e0b0xyg3 =  1 * (np.sum(mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3
                                                        * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                                                          , axis = (d_pos), keepdims=True)>0)


    #################
    #progeny weaned #
    #################
    '''yatf are first transferred to progeny activity then they are either sold as sucker, transferred to dam or transferred to offs.
    yatf have a t axis due to the dams feedsupply. This t axis is removed and replaced with the prog t axis.
    
    Prog can be clustered without losing lw info because the distributions (prog2 and yatf2prog) are done with
    active e&b axes.
    Clustering means the model can only differentially manage based on LW. If the model scans then it can differentially
    manage based on BTRT.    
    '''
    ##condense yatf from total number of finishing lw to 10
    ### mask the ffcfw & salevalue for only those that have numbers > 0. Removes animals that have died or don't exist
    ffcfw_range_ta1e1b1nwzida0e0b0xyg2 = ffcfw_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 * (numbers_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 > 0)
    salevalue_range_c1p7ta1e1b1nwzida0e0b0xyg2 = salevalue_d_c1p7ta1e1b1nwzida0e0b0xyg2 * (numbers_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 > 0)
    salevalue_wc_range_c0p7ta1e1b1nwzida0e0b0xyg2 = salevalue_wc_d_c0p7ta1e1b1nwzida0e0b0xyg2 * (numbers_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 > 0)
    ###remove t axis by reshaping with w axis. Thus the t is reflected by more w slices.
    ffcfw_range_a1e1b1nwzida0e0b0xyg2 = fun.f_merge_axis(ffcfw_range_ta1e1b1nwzida0e0b0xyg2, source_axis=0, target_axis=w_pos)
    numbers_start_d_yatf_a1e1b1nwzida0e0b0xyg2 = fun.f_merge_axis(numbers_start_d_yatf_ta1e1b1nwzida0e0b0xyg2, source_axis=0, target_axis=w_pos)
    salevalue_range_c1p7a1e1b1nwzida0e0b0xyg2 = fun.f_merge_axis(salevalue_range_c1p7ta1e1b1nwzida0e0b0xyg2, source_axis=2, target_axis=w_pos)
    salevalue_wc_range_c0p7a1e1b1nwzida0e0b0xyg2 = fun.f_merge_axis(salevalue_wc_range_c0p7ta1e1b1nwzida0e0b0xyg2, source_axis=2, target_axis=w_pos)
    ### The index that sorts the weight array
    ind_sorted_a1e1b1nwzida0e0b0xyg2 = np.argsort(ffcfw_range_a1e1b1nwzida0e0b0xyg2, axis = w_pos)
    ### Select the values for the 10 equally spaced values spanning lowest to highest inclusive.
    start_a1e1b1nwzida0e0b0xyg2 = np.minimum(ffcfw_range_a1e1b1nwzida0e0b0xyg2.shape[w_pos] - np.count_nonzero(ffcfw_range_a1e1b1nwzida0e0b0xyg2, axis = w_pos), len_w1-1)
    ind_selected_a1e1b1nwzida0e0b0xyg2 = np.linspace(start_a1e1b1nwzida0e0b0xyg2, ffcfw_range_a1e1b1nwzida0e0b0xyg2.shape[w_pos] - 1, len_w_prog, dtype = int, axis = w_pos)
    ### The indices for the required values are the selected values from the sorted indices
    ind = np.take_along_axis(ind_sorted_a1e1b1nwzida0e0b0xyg2, ind_selected_a1e1b1nwzida0e0b0xyg2, axis = w_pos)
    ### Extract the condensed weights, the numbers and the sale_value of the condensed vars
    #### Later these variables are used with the 10 weights in the i_w_pos, so note whether w9 on end or not
    ffcfw_prog_a1e1b1_a1e1b1nwzida0e0b0xyg2 = np.take_along_axis(ffcfw_range_a1e1b1nwzida0e0b0xyg2, ind, axis = w_pos)
    salevalue_prog_a1e1b1_c1p7a1e1b1nwzida0e0b0xyg2 = np.take_along_axis(salevalue_range_c1p7a1e1b1nwzida0e0b0xyg2, ind[na,na,...], axis = w_pos)
    salevalue_wc_prog_a1e1b1_c0p7a1e1b1nwzida0e0b0xyg2 = np.take_along_axis(salevalue_wc_range_c0p7a1e1b1nwzida0e0b0xyg2, ind[na,na,...], axis = w_pos)
    t_numbers_start_d_prog_a1e1b1_a1e1b1nwzida0e0b0xyg2 = np.take_along_axis(numbers_start_d_yatf_a1e1b1nwzida0e0b0xyg2, ind, axis = w_pos)

    ##distribute the yatf to the intermediate progeny activity
    distribution_2prog_va1e1b1nw8zida0e0b0xyg1w9 = sfun.f1_lw_distribution(ffcfw_prog_a1e1b1_a1e1b1nwzida0e0b0xyg2[na,na]
                                                                          , ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1)


    ##convert a1, e1 & b1 to a0, e0 & b0 so prog can interact with offs
    t_ffcfw_prog_a1e1b0_a1e1b1nwzida0e0b0xyg2 = np.sum(ffcfw_prog_a1e1b1_a1e1b1nwzida0e0b0xyg2 * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                                              , axis=b1_pos, keepdims=True) #convert b1 to b0
    t_salevalue_prog_a1e1b0_c1p7a1e1b1nwzida0e0b0xyg2 = np.sum(salevalue_prog_a1e1b1_c1p7a1e1b1nwzida0e0b0xyg2 * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                                              , axis=b1_pos, keepdims=True) #convert b1 to b0
    t_salevalue_wc_prog_a1e1b0_c0p7a1e1b1nwzida0e0b0xyg2 = np.sum(salevalue_wc_prog_a1e1b1_c0p7a1e1b1nwzida0e0b0xyg2 * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                                              , axis=b1_pos, keepdims=True) #convert b1 to b0
    t_numbers_start_d_prog_a1e1b0_a1e1b1nwzida0e0b0xyg2 = np.sum(t_numbers_start_d_prog_a1e1b1_a1e1b1nwzida0e0b0xyg2 * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                                              , axis=b1_pos, keepdims=True) #convert b1 to b0
    t_ffcfw_prog_a0e1b0_a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_ffcfw_prog_a1e1b0_a1e1b1nwzida0e0b0xyg2, a1_pos, a0_pos) #swap a1 and a0
    ffcfw_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_ffcfw_prog_a0e1b0_a1e1b1nwzida0e0b0xyg2, e1_pos, e0_pos) #swap e1 and e0
    t_salevalue_prog_a0e1b0_c1p7a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_salevalue_prog_a1e1b0_c1p7a1e1b1nwzida0e0b0xyg2, a1_pos, a0_pos) #swap a1 and a0
    t_salevalue_wc_prog_a0e1b0_c0p7a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_salevalue_wc_prog_a1e1b0_c0p7a1e1b1nwzida0e0b0xyg2, a1_pos, a0_pos) #swap a1 and a0
    salevalue_prog_a0e0b0_c1p7a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_salevalue_prog_a0e1b0_c1p7a1e1b1nwzida0e0b0xyg2, e1_pos, e0_pos) #swap e1 and e0
    salevalue_wc_prog_a0e0b0_c0p7a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_salevalue_wc_prog_a0e1b0_c0p7a1e1b1nwzida0e0b0xyg2, e1_pos, e0_pos) #swap e1 and e0
    t_numbers_start_d_prog_a0e1b0_a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_numbers_start_d_prog_a1e1b0_a1e1b1nwzida0e0b0xyg2, a1_pos, a0_pos) #swap a1 and a0
    numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2 = np.swapaxes(t_numbers_start_d_prog_a0e1b0_a1e1b1nwzida0e0b0xyg2, e1_pos, e0_pos) #swap e1 and e0

    ##add t axis to progeny - slice 0 is sold as sucker, slice 1 (to dams) and 2 (to offs) are retained
    index_tpa1e1b1nwzida0e0b0xyg2 = fun.f_expand(index_t2, p_pos-1)
    salevalue_prog_c1p7tva1e1b1nwzida0e0b0xyg2 = salevalue_prog_a0e0b0_c1p7a1e1b1nwzida0e0b0xyg2[:,:,na,na,...] * (index_tpa1e1b1nwzida0e0b0xyg2==0)
    salevalue_wc_prog_c0p7tva1e1b1nwzida0e0b0xyg2 = salevalue_wc_prog_a0e0b0_c0p7a1e1b1nwzida0e0b0xyg2[:,:,na,na,...] * (index_tpa1e1b1nwzida0e0b0xyg2==0)

    #add c axis to prog - using period_is_wean so that correct c slice is activated
    # period_is_wean_d_pa1e1b1nwzida0e0b0xyg2 = period_is_wean_pa1e1b1nwzida0e0b0xyg2 *  (a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2==index_da0e0b0xyg)
    # salevalue_prog_cta1e1b1nwzida0e0b0xyg2 = (sfun.f1_p2v_std(salevalue_prog_tpa1e1b1nwzida0e0b0xyg2,
    #                                                           numbers_p=numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2,
    #                                                           period_is_tvp=period_is_wean_d_pa1e1b1nwzida0e0b0xyg2[:,:,0:1,...], #weaning is same for all e slices
    #                                  a_any1_p=a_c_pa1e1b1nwzida0e0b0xyg,index_any1tvp=index_ctpa1e1b1nwzida0e0b0xyg)).astype(dtype)

    ## cluster sale value - can use offs function because clustering is the same.
    salevalue_prog_k3k5c1p7tva1e1b1nwzida0e0b0xyg2 = sfun.f1_create_production_param('offs',
                                                                                  salevalue_prog_c1p7tva1e1b1nwzida0e0b0xyg2,
                                                                                  a_k3cluster_da0e0b0xyg3,
                                                                                  index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],
                                                                                  a_k5cluster_da0e0b0xyg3,
                                                                                  index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...],
                                                                                  numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2)
    salevalue_wc_prog_k3k5c0p7tva1e1b1nwzida0e0b0xyg2 = sfun.f1_create_production_param('offs',
                                                                                  salevalue_wc_prog_c0p7tva1e1b1nwzida0e0b0xyg2,
                                                                                  a_k3cluster_da0e0b0xyg3,
                                                                                  index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:, na,na,...],
                                                                                  a_k5cluster_da0e0b0xyg3,
                                                                                  index_k5tva1e1b1nwzida0e0b0xyg3[:,na, na,...],
                                                                                  numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2)

    ##mask w8 (prog) to w9 (dams)
    step_con_prog2dams = n_fs_dams ** n_fvps_percondense_dams
    mask_numbers_prog2damsw8w9_w9 = (index_w1 % step_con_prog2dams) == 0
    ##mask w8 (prog) to w9 (offs)
    step_con_prog2offs = n_fs_offs ** n_fvps_percondense_offs
    mask_numbers_prog2offsw8w9_w9 = (index_w3 % step_con_prog2offs) == 0

    ###The association between birth time of the progeny and the birth time and lambing opportunity/dam age
    prior_times_excluded_ida0e0b0xyg = fun.f_expand(np.cumsum(~pinp.sheep['i_mask_i'])[pinp.sheep['i_mask_i']], i_pos)
    a_i_ida0e0b0xyg2 = (a_i_ida0e0b0xyg2 - prior_times_excluded_ida0e0b0xyg)

    ###k2 and k5 associations - keys also used for pyomo sets in the section below
    keys_k2 = np.ravel(sinp.stock['i_k2_idx_dams'])[:len_k2]
    keys_k2ktva1e1b1nwzida0e0b0xyg = fun.f_expand(keys_k2, k2_pos-1)
    keys_k5 = np.ravel(sinp.stock['i_k5_idx_offs'])[:len_k5].astype('>U4') #not sure why it is not automatically going to >U4
    keys_k5tva1e1b1nwzida0e0b0xyg = fun.f_expand(keys_k5, k2_pos)


    ###number of progeny weaned
    #### compare a_k2cluster with index_k5 to only retain the values for the dams that have yatf.
    #### index_k5 + 2 is allowing for NM & 00 that are first 2 entries in the k2cluster that don't exist in the k5cluster
    npw_k3k5tva1e1b1nwzida0e0b0xyg1w9i9 = fun.f_divide(
          np.sum(npw_tva1e1b1nwzida0e0b0xyg1[...,na,na] * distribution_2prog_va1e1b1nw8zida0e0b0xyg1w9[...,na]
                 * mask_w8vars_va1e1b1nw8zida0e0b0xyg1[...,na,na] * mask_z8var_va1e1b1nwzida0e0b0xyg1[...,na,na] * mask_tvars_k5tva1e1b1nw8zida0e0b0xyg1[...,na,na]
                 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k5tva1e1b1nwzida0e0b0xyg3 + 2)[...,na,na] #convert e1 and b1 to k5 cluster - using a k5 cluster because progeny don't need all the k2 slices and the relevant ones align between k2 and k5 e.g. 11, 22 etc
                 * (a_i_ida0e0b0xyg2==index_ida0e0b0xyg)[...,na,na] * (index_ida0e0b0xyg[...,na,na] == index_i9)  #i9 (like w9 & g9) is the lambing time of the destination weaner. If lambing interval is not 12 months then a dam born in July may be giving birth in March and this weaner need to then become a dam replacement in the 'March' flock. Changing lambing time then requires a second i axis in teh parameter.
                 * (a_k3cluster_da0e0b0xyg3==index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                 , axis=(b1_pos - 2, e1_pos - 2, d_pos -2), keepdims=True)
        , np.sum(numbers_start_d_tva1e1b1nwzida0e0b0xyg1 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k5tva1e1b1nwzida0e0b0xyg3+2)
                 * (a_k3cluster_da0e0b0xyg3==index_k3k5tva1e1b1nwzida0e0b0xyg3)
                 , axis=(b1_pos, e1_pos, d_pos), keepdims=True)[...,na,na], dtype=dtype)
    #todo an alternative would be to create a_k2_k5tva1e1b1nwzida0e0b0xyg (which is essentially a_b1_b0xyg but with allowance for extra slices if scan == 4)
    # then use "*  (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == a_k2_k5tva1e1b1nwzida0e0b0xyg3)"

    ###npw required by prog activity
    ####mask numbers req (also used for prog2dams) - The progeny decision variable can be masked for gender and dam age for t[1] (t[1] are those that get transferred to dams). Gender only requires females, and the age of the dam only requires those that contribute to the initial age structure.
    mask_prog_tdx_tva1e1b1nwzida0e0b0xyg2w9 = np.logical_or((index_tva1e1b1nwzida0e0b0xyg2w9 != 1),
                                                            np.logical_and(np.logical_and((gender_xyg[mask_x] == 1)[...,na] ,
                                                                           (agedam_propn_da0e0b0xyg1 > 0)[...,na]),
                                                                           np.isin(index_g1, a_g1_g2)[...,na]))
    numbers_prog_req_k3k5tva1e1b1nwzida0e0b0xyg2w9 = 1 * np.any(mask_prog_tdx_tva1e1b1nwzida0e0b0xyg2w9
                                                            * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                            , axis = d_pos-1)


    ##transfer progeny to dam replacements
    ###liveweight distribution
    ffcfw_initial_wzida0e0b0xyg1 = (lw_initial_wzida0e0b0xyg1 - cfw_initial_wzida0e0b0xyg1 / cw_dams[3, ...]).astype(dtype)
    distribution_2dams_a1e1b1nwzida0e0b0xyg2w9 = sfun.f1_lw_distribution(ffcfw_initial_wzida0e0b0xyg1[na,na,na,na,...]
                                                                        , ffcfw_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2)

    ###numbers provided
    numbers_prog2dams_k3k5tva1e1b1nwzida0e0b0xyg2g9w9 = fun.f_weighted_average(distribution_2dams_a1e1b1nwzida0e0b0xyg2w9[...,na,:] * mask_numbers_prog2damsw8w9_w9
                                                                    * mask_prog_tdx_tva1e1b1nwzida0e0b0xyg2w9[...,na,:]
                                                                    * (index_a0e0b0xyg == 0) #only a[0] prog can provide dams
                                                                    * (a_g1_g2[...,na,:]==index_g1)[...,na] * (index_tva1e1b1nwzida0e0b0xyg2w9 == 1)[...,na,:]
                                                                    * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                                                                    * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na,na],
                                                               weights=numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2[...,na,na]
                                                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                                                               * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                                                               , axis=(d_pos -2, e0_pos-2, b0_pos-2), keepdims=True)

    ###numbers required - no d axis for Dam DVs
    ####the total number required for each dam is 1.0 when summed across the progeny k3 (progeny dam age) & k5 (progeny BTRT) axes
    ####collapse the e1 axis on the mask prior to np.sum because can't test for > 0 as per other numbers_req (because need proportions of age & BTRT)
    #### but don't want to increase the numbers if joining for multiple cycles
    #### Loop to reduce memory
    numbers_progreq_k2k3k5tva1e1b1nw8zida0e0b0xyg1g9w9 = 0
    for d in range(len_d): #loop on b1 to reduce memory
        numbers_progreq_k2k3k5tva1e1b1nw8zida0e0b0xyg1g9w9 += 1 * np.sum(np.any(mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9 * mask_z8var_va1e1b1nwzida0e0b0xyg1[...,na], axis=e1_pos-1, keepdims=True)[0, ...,na,:]
                                                                        * mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,:,0:1,...,na,na]  # mask based on the t axis for dvp0
                                                                        * (index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,..., na,na] == 0) #only NM slice requires prog
                                                                        * (index_g1[...,na]==index_g1)[...,na]
                                                                        * btrt_propn_b0xyg1[...,na,na].astype(dtype)   #todo this would be better if it had a d axis in this calculation so that propn of DST could vary by age of the dam e.g. if replacing flock with more prog from young ewes there would be more single prog making up the starting animal.
                                                                        * e0_propn_ida0e0b0xyg[...,na,na].astype(dtype)
                                                                        * agedam_propn_da0e0b0xyg1[d:d+1,...,na,na].astype(dtype)
                                                                        * (a_k3cluster_da0e0b0xyg3[d:d+1,...] == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                                                                        * (a_k5cluster_da0e0b0xyg3[d:d+1,...] == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na,na],
                                                                         axis=(e1_pos-2, d_pos-2, b0_pos-2, e0_pos-2),keepdims=True)

    ##transfer progeny to offs
    ###numbers provide
    ffcfw_initial_wzida0e0b0xyg3 = (lw_initial_wzida0e0b0xyg3 - cfw_initial_wzida0e0b0xyg3 / cw_offs[3, ...]).astype(dtype)

    distribution_a1e1b1nwzida0e0b0xyg2w9 = sfun.f1_lw_distribution(ffcfw_initial_wzida0e0b0xyg3[na,na,na,na,...]
                                                           , ffcfw_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2)
    numbers_prog2offs_k3k5tva1e1b1nwzida0e0b0xyg2w9 = fun.f_weighted_average(distribution_a1e1b1nwzida0e0b0xyg2w9
                                                             * mask_numbers_prog2offsw8w9_w9
                                                             * (index_tva1e1b1nwzida0e0b0xyg2w9 == 2)
                                                             * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                             * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                             , weights=numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2[...,na]
                                                             * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                             * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                             , axis=(d_pos-1, b0_pos-1, e0_pos-1),keepdims=True)

    ###numbers req
    numbers_progreq_k3k5tva1e1b1nw8zida0e0b0xyg3w9 = 1 * (np.sum(mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg3w9
                                                       * mask_z8var_va1e1b1nwzida0e0b0xyg3[...,na]
                                                       * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na],
                                                       axis=(d_pos-1),keepdims=True) > 0)





    ###########################
    #  report P2V             #
    ###########################
    ##cashflow stuff
    r_salevalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(r_salevalue_p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,:,na,...]#add singleton v
    r_salegrid_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(r_salegrid_tpa1e1b1nwzida0e0b0xyg0, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_woolvalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,:,na,...]#add singleton v
    r_salevalue_p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(r_salevalue_p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    r_salegrid_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(r_salegrid_tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                    on_hand_tpa1e1b1nwzida0e0b0xyg1)
    r_salegrid_tva1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v(r_salegrid_tpa1e1b1nwzida0e0b0xyg2, a_v_pa1e1b1nwzida0e0b0xyg1)
    r_woolvalue_p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    r_salevalue_p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_salevalue_p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)
    r_saleage_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_saleage_tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                     on_hand_tpa1e1b1nwzida0e0b0xyg3)
    r_salegrid_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_salegrid_tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                     on_hand_tpa1e1b1nwzida0e0b0xyg3)
    r_woolvalue_p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)

    ##sale time - no numbers needed because they don't affect sale date. Use date end since sale is end of period.
    r_saledate_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(date_end_pa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                    period_is_tp=period_is_sale_tpa1e1b1nwzida0e0b0xyg3)

    ##cfw per head average for the mob - includes the mortality factor
    r_cfw_hdmob_tvg0 = sfun.f1_p2v_std(o_cfw_tpsire, numbers_p=o_numbers_end_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_cfw_hdmob_tvg1 = sfun.f1_p2v(o_cfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg1)
    r_cfw_hdmob_tvg3 = sfun.f1_p2v(o_cfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)
    ##cfw per head - wool cut for 1 whole animal, no account for mortality
    r_cfw_hd_tvg0 = sfun.f1_p2v_std(o_cfw_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_cfw_hd_tvg1 = sfun.f1_p2v(o_cfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                               period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg1)
    r_cfw_hd_tvg3 = sfun.f1_p2v(o_cfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg3,
                               period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)

    ##fd per head average for the mob - includes the mortality factor
    r_fd_hdmob_tvg0 = sfun.f1_p2v_std(o_fd_tpsire, numbers_p=o_numbers_end_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_fd_hdmob_tvg1 = sfun.f1_p2v(o_fd_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg1)
    r_fd_hdmob_tvg3 = sfun.f1_p2v(o_fd_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)
    ##fd per head - wool cut for 1 whole animal, no account for mortality
    r_fd_hd_tvg0 = sfun.f1_p2v_std(o_fd_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_mainshearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_fd_hd_tvg1 = sfun.f1_p2v(o_fd_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                               period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg1)
    r_fd_hd_tvg3 = sfun.f1_p2v(o_fd_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg3,
                               period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)

    ##wbe at start of the DVP - not accounting for mortality
    r_wbe_tvg1 = sfun.f1_p2v(r_wbe_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1)
    r_wbe_tvg3 = sfun.f1_p2v(r_wbe_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg3)

    ##nfoet scanning
    r_nfoet_scan_tvg1 = sfun.f1_p2v(nfoet_b1nwzida0e0b0xyg, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_scan_pa1e1b1nwzida0e0b0xyg1)

    ##nfoet birth
    r_nfoet_birth_tvg1 = sfun.f1_p2v(nfoet_b1nwzida0e0b0xyg, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                   on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_birth_pa1e1b1nwzida0e0b0xyg1)

    ##nyatf birth
    r_nyatf_birth_tvg1 = sfun.f1_p2v(nyatf_b1nwzida0e0b0xyg, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                   on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_birth_pa1e1b1nwzida0e0b0xyg1)

    ##numbers dry
    n_drys_b1g1 = fun.f_expand(sinp.stock['i_is_dry_b1'],b1_pos)
    r_n_drys_tvg1 = sfun.f1_p2v(n_drys_b1g1*1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_scan_pa1e1b1nwzida0e0b0xyg1)

    ##number of mated animals as a proportion of the dams in each slice that has initial numbers == 1
    ### calculated from the number of ewes mated total (across a1, e1, b1 & y-axis) per ewe in the slice.
    ### This is designed to be the number mated equivalent of the number of foetuses per ewe
    n_mated_tpg1 = animal_mated_b1g1 * o_numbers_end_tpdams
    r_n_mated_tvg1 = sfun.f1_p2v(n_mated_tpg1, a_v_pa1e1b1nwzida0e0b0xyg1, 1,
                                on_hand_tp=True, period_is_tp=period_is_matingend_pa1e1b1nwzida0e0b0xyg1)
    r_n_mated_tvg1 = np.sum(r_n_mated_tvg1, axis=(a1_pos, e1_pos, b1_pos, y_pos), keepdims=True)
    ###update periods that are not mating with mating numbers
    a_matingv_tvg1 =  np.maximum.accumulate(np.any(r_n_mated_tvg1 != 0, axis=b1_pos, keepdims=True)
                                            * index_va1e1b1nwzida0e0b0xyg1, axis=p_pos) #create association pointing at previous/current mating dvp.
    r_n_mated_tvg1= np.take_along_axis(r_n_mated_tvg1, a_matingv_tvg1, axis=p_pos)
    # n_mated_tpg1 = fun.f_divide(np.sum(animal_mated_b1g1 * o_numbers_end_tpdams, axis=(a1_pos, e1_pos, b1_pos, y_pos), keepdims=True)
    #                             , o_numbers_end_tpdams) * (animal_mated_b1g1>0)
    # r_n_mated_tvg1 = sfun.f1_p2v(n_mated_tpg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
    #                             on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_matingend_pa1e1b1nwzida0e0b0xyg1)
    # ###update periods that are not mating with mating numbers
    # a_matingv_tvg1 =  np.maximum.accumulate(np.any(r_n_mated_tvg1 != 0, axis=b1_pos, keepdims=True)
    #                                         * index_va1e1b1nwzida0e0b0xyg1, axis=p_pos) #create association pointing at previous/current mating dvp.
    # r_n_mated_tvg1= np.take_along_axis(r_n_mated_tvg1, a_matingv_tvg1, axis=p_pos)


    ###########################
    # create report params    #
    ###########################
    ##sale value - needed for reporting
    r_salevalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_salevalue_p7tva1e1b1nwzida0e0b0xyg0,
                                                                          numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    r_salevalue_k2p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_salevalue_p7tva1e1b1nwzida0e0b0xyg1,
                                                                            a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                            index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                            numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                            mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_salevalue_prog_k3k5p7tva1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(salevalue_prog_k3k5c1p7tva1e1b1nwzida0e0b0xyg2, prob_c1tpg[:,na,...], axis=p_pos-3)
    r_salevalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',
                                                                              r_salevalue_p7tva1e1b1nwzida0e0b0xyg3,
                                                                              a_k3cluster_da0e0b0xyg3,
                                                                              index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                                              a_k5cluster_da0e0b0xyg3,
                                                                              index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...],
                                                                              numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                                              mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                      * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##sale date - no numbers needed because they don't affect sale date. need to mask the denominator so that only the d, e, b slices where the animal was sold is used in the divide.
    ###can't use production function because need to mask the denominator.
    r_saledate_k3k5tva1e1b1nwzida0e0b0xyg3 = fun.f_divide(np.sum(r_saledate_tva1e1b1nwzida0e0b0xyg3 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                                                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)
                                                                 , axis = (sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True),
                                                          np.sum((a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3) * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)*
                                                                 (r_saledate_tva1e1b1nwzida0e0b0xyg3!=0), #need to include this mask to make sure we are only averaging the sale date with e,b,d slice animals that were sold.
                                                                 axis=(sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True))

    ##wool value
    r_woolvalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_woolvalue_p7tva1e1b1nwzida0e0b0xyg0,
                                                                          numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    r_woolvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_woolvalue_p7tva1e1b1nwzida0e0b0xyg1,
                                                                            a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                            index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                            numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                            mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_woolvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',
                                                                              r_woolvalue_p7tva1e1b1nwzida0e0b0xyg3,
                                                                              a_k3cluster_da0e0b0xyg3,
                                                                              index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                                              a_k5cluster_da0e0b0xyg3,
                                                                              index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...],
                                                                              numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                                              mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                      * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##cfw per head average for the mob - includes the mortality factor
    r_cfw_hdmob_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_cfw_hdmob_tvg0,
                                                                         numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    r_cfw_hdmob_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_cfw_hdmob_tvg1,
                                                                           a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                           index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                           numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                           mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_cfw_hdmob_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_cfw_hdmob_tvg3,
                                                                             a_k3cluster_da0e0b0xyg3,
                                                                             index_k3k5tva1e1b1nwzida0e0b0xyg3,
                                                                             a_k5cluster_da0e0b0xyg3,
                                                                             index_k5tva1e1b1nwzida0e0b0xyg3,
                                                                             numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                                             mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                     * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##cfw per head - wool cut per animal at shearing, no account for mortality (numbers)
    r_cfw_hd_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_cfw_hd_tvg0)
    r_cfw_hd_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_cfw_hd_tvg1,
                                                                        a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                        index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                        mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_cfw_hd_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_cfw_hd_tvg3,a_k3cluster_da0e0b0xyg3,
                                                                          index_k3k5tva1e1b1nwzida0e0b0xyg3,
                                                                          a_k5cluster_da0e0b0xyg3,
                                                                          index_k5tva1e1b1nwzida0e0b0xyg3,
                                                                          mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                  * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##fd per head average for the mob - includes the mortality factor
    r_fd_hdmob_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_fd_hdmob_tvg0,
                                                                         numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    r_fd_hdmob_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_fd_hdmob_tvg1,
                                                                           a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                           index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                           numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                           mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_fd_hdmob_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_fd_hdmob_tvg3,
                                                                             a_k3cluster_da0e0b0xyg3,
                                                                             index_k3k5tva1e1b1nwzida0e0b0xyg3,
                                                                             a_k5cluster_da0e0b0xyg3,
                                                                             index_k5tva1e1b1nwzida0e0b0xyg3,
                                                                             numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                                             mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                     * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##fd per head - wool cut per animal at shearing, no account for mortality (numbers)
    r_fd_hd_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire',r_fd_hd_tvg0)
    r_fd_hd_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_fd_hd_tvg1,
                                                                        a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                        index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                        mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_fd_hd_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_fd_hd_tvg3,a_k3cluster_da0e0b0xyg3,
                                                                          index_k3k5tva1e1b1nwzida0e0b0xyg3,
                                                                          a_k5cluster_da0e0b0xyg3,
                                                                          index_k5tva1e1b1nwzida0e0b0xyg3,
                                                                          mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                  * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ##wbe - wholebody energy at start of DVP, no account for mortality
    r_wbe_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_wbe_tvg1,
                                                                        a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                        index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                        mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    r_wbe_k3k5tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_wbe_tvg3,a_k3cluster_da0e0b0xyg3,
                                                                          index_k3k5tva1e1b1nwzida0e0b0xyg3,
                                                                          a_k5cluster_da0e0b0xyg3,
                                                                          index_k5tva1e1b1nwzida0e0b0xyg3,
                                                                          mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3
                                                                                  * mask_z8var_va1e1b1nwzida0e0b0xyg3)


    #############################################
    #weaning %, scan % and lamb survival reports#
    #############################################
    ##The inverse (1/(r_n_mated/start) == start/r_n_mated) of the number of ewes mated as a proportion of the number of ewes at the start of the DVP (accounts for mortality)
    ### Note: the sum across the a1, e1, b1 & y axes is a single starting animal. Each slice of the other axes are single animals
    ### Can't call f1_create_production_param() because don't want to cluster r_n_mated_tvg1
    ### Inverse because number of ewes mated is the denominator of the reproduction calculations
    #todo at some point we might want repro reports that don't cluster the e & b axes so that repro of multiple in a clustered mob can be reported
    # This will require having another version of the following variables that are not clustered & different maths in ReportFunction.py
    mask_sliced = fun.f_dynamic_slice(mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
                                      , e1_pos, 0 ,1) #slice e[0] because don't want to add e axis to rnmated
    r_n_mated_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg1
                        * mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
                        * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1)
                        , axis=(a1_pos, b1_pos, e1_pos, y_pos), keepdims=True)
                    , np.sum(r_n_mated_tvg1 * mask_sliced, axis=(a1_pos, b1_pos, e1_pos, y_pos)
                        , keepdims=True), dtype=r_n_mated_tvg1.dtype)
    # r_n_mated_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(1, fun.f_divide(np.sum(r_n_mated_tvg1
    #                 * mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
    #                 , axis=(a1_pos, b1_pos, e1_pos, y_pos), keepdims=True)
    #             , np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg1
    #                 * mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
    #                 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1)
    #                 , axis=(a1_pos, b1_pos, e1_pos, y_pos), keepdims=True), dtype=r_n_mated_tvg1.dtype))

    ##proportion of drys - per ewe at start of dvp (ie accounting for dam mortality)
    r_n_drys_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_n_drys_tvg1,
                                                                          a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                          index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                       numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                       mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)  # no clustering required for scanning percent because it is a measure of all dams

    ##wean percent - weaning percent - npw per ewe at start of dvp (ie accounting for mortality)
    ###number of progeny weaned. Cluster and account for mortality between start of dvp and weaning so it is compatible with the dams variable
    r_nyatf_wean_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',npw2_tva1e1b1nwzida0e0b0xyg1,
                                                                          a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                          index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                       numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                       mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1) # no clustering required for wean percent because it is a measure of all dams

    ##scanning percent - foetuses scanned per ewe at start of dvp (ie accounting for dam mortality)
    r_nfoet_scan_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_nfoet_scan_tvg1,
                                                                          a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                          index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                       numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                       mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)  # no clustering required for scanning percent because it is a measure of all dams

    ##nfoet - nfoet per ewe at start of dvp (ie accounting for dam mortality - so it can be multiplied by the v_dams from lp solution)
    r_nfoet_birth_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(r_nfoet_birth_tvg1 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1) #can't use param function because we need to keep e and b axis
                                                           * mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                            , np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg1 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1),
                                                                     axis = (e1_pos, b1_pos), keepdims=True))


    ##nyatf - nyatf per ewe at start of dvp (ie accounting for dam mortality - so it can be multiplied by the v_dams from lp solution)
    r_nyatf_birth_k2tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(r_nyatf_birth_tvg1 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1) #can't use param function because we need to keep e and b axis
                                                           * mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                            , np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg1 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1),
                                                                     axis = (e1_pos, b1_pos), keepdims=True))

    ##ffcfw for select p - to keep the report small it doesn't have full p axis
    period_is_reportffcfw_p = fun.f_sa(np.array([False]), sen.sav['period_is_reportffcfw_p'], 5)
    period_is_reportffcfw_p = period_is_reportffcfw_p[0:len_p]

    ##ffcfw in select p slices to reduce size.
    r_ffcfw_dams_k2tvPdams = (o_ffcfw_tpdams[:, na, period_is_reportffcfw_p, ...]
                              * (a_v_pa1e1b1nwzida0e0b0xyg1[period_is_reportffcfw_p] == index_vpa1e1b1nwzida0e0b0xyg1)
                              * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:, na, ...]
                                 == index_k2tva1e1b1nwzida0e0b0xyg1[:, :,:, na, ...]))

    ##############
    #big reports #
    ##############
    ##mortality - at the start of each dvp mortality is 0. it accumulates over the dvp. This is its own report as well as used in on_hand_mort.
    if sinp.rep['i_store_on_hand_mort'] or sinp.rep['i_store_mort']:
        ###get the cumulative mort for periods in each dvp
        r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg1 = sfun.f1_cum_sum_dvp(o_mortality_dams, a_v_pa1e1b1nwzida0e0b0xyg1, axis=p_pos)
        r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_cum_sum_dvp(o_mortality_offs, a_v_pa1e1b1nwzida0e0b0xyg3, axis=p_pos)
        ###mask w & z slices
        mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg1 = np.take_along_axis(mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1
                                                                 , axis=0)
        mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg3 = np.take_along_axis(mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3
                                                                 , axis=0)
        r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg1 = r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg1 * mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg1
        r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg3 = r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg3 * mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg3

    ##on hand mort- this is used for numbers_p report so that the report can have a p axis to increase numbers detail.
    ##              accounts for mortality as well as on hand.
    if sinp.rep['i_store_on_hand_mort']:
        ###add v axis and adjust for onhand
        r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg1 = r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg1[:,na,...] * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...] * (
                                                  a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
        r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg3 = r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg3[:,na,...] * on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...] * (
                                                  a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
        ###cluster e,b
        r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg1,
                                                                              a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                              index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],
                                                                              numbers_start_vg = on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...])  #on_hand to handle the periods when e slices are in different dvps (e.g. can't just have default 1 otherwise it will divide by 2 because both e gets summed)
        r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg3,
                                                                                     a_k3cluster_da0e0b0xyg3,
                                                                                     index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],
                                                                                     a_k5cluster_da0e0b0xyg3,
                                                                                     index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...],
                                                                                     numbers_start_vg=on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...]) #on_hand to handle the periods when e slices are in different dvps (e.g. can't just have default 1 otherwise it will divide by 2 because both e gets summed)

        ###convert to on hand mort (1-mort)
        r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1 = 1 - r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1
        r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1[r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1==0] = 0 #if mort is 0 the animal is not on hand
        r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3 = 1 - r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3
        r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3[r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3==0] = 0 #if mort is 0 the animal is not on hand

    ###lw - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. Don't need it because it doesn't affect lw
    if sinp.rep['i_store_lw_rep']:
        r_lw_sire_tpsire = o_lw_tpsire
        r_lw_dams_k2Tvpdams = (o_lw_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_lw_offs_k3k5Tvpoffs = (o_lw_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...]))

    ##ffcfw - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. Don't need it because it doesn't affect ffcfw
    if sinp.rep['i_store_ffcfw_rep']:
        r_ffcfw_sire_tpsire = o_ffcfw_tpsire
        r_ffcfw_dams_k2Tvpdams = (o_ffcfw_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_ffcfw_yatf_k2Tvpyatf = (r_ffcfw_start_tpyatf[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_ffcfw_offs_k3k5Tvpoffs = (o_ffcfw_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...]))
    ####calculate ffcfw_prog for all trials. It is a small variable because it has singleton p axis
    r_ffcfw_prog_k3k5tva1e1b1nwzida0e0b0xyg2 = ffcfw_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2 \
                                             * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3) \
                                             * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)

    ##NV - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. t is not required because it doesn't affect NV
    if sinp.rep['i_store_nv_rep']:
        r_nv_sire_pg = nv_tpsire
        r_nv_dams_k2Tvpg = (nv_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_nv_offs_k3k5Tvpg = (nv_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                               * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...]))

    ##cs - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. t is not required because it doesn't affect NV
    if sinp.rep['i_store_cs_rep']:
        cs_tpg0 = sfun.f1_condition_score(o_rc_start_tpsire, cn_sire.astype(dtype))
        cs_tpg1 = sfun.f1_condition_score(o_rc_start_tpdams, cn_dams.astype(dtype))
        cs_tpg3 = sfun.f1_condition_score(o_rc_start_tpoffs, cn_offs.astype(dtype))
        r_cs_sire_pg = cs_tpg0
        r_cs_dams_k2Tvpg = (cs_tpg1[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_cs_offs_k3k5Tvpg = (cs_tpg3[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                               * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...]))

    ##fs - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. t is not required because it doesn't affect NV
    if sinp.rep['i_store_fs_rep']:
        fs_tpg0 = sfun.f1_fat_score(o_rc_start_tpsire, cn_sire.astype(dtype))
        fs_tpg1 = sfun.f1_fat_score(o_rc_start_tpdams, cn_dams.astype(dtype))
        fs_tpg3 = sfun.f1_fat_score(o_rc_start_tpoffs, cn_offs.astype(dtype))
        r_fs_sire_pg = fs_tpg0
        r_fs_dams_k2Tvpg = (fs_tpg1[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_fs_offs_k3k5Tvpg = (fs_tpg3[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                               * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...]))

    ###############
    ## report dse #
    ###############
    days_p6z = per.f_feed_periods(option=1)
    days_p6_p6tva1e1b1nwzida0e0b0xyg = fun.f_expand(days_p6z, z_pos,  left_pos2=p_pos-2, right_pos2=z_pos)
    ###DSE based on MJ/d
    ####returns the average mj/d for each animal for each feed period (mei accounts for if the animal is on hand - if the animal is sold the average mei/d will be lower in that dvp)
    mj_ave_p6ftva1e1b1nwzida0e0b0xyg0 = fun.f_divide(mei_p6ftva1e1b1nwzida0e0b0xyg0, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    mj_ave_k2p6ftva1e1b1nwzida0e0b0xyg1 = fun.f_divide(mei_k2p6ftva1e1b1nwzida0e0b0xyg1, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    mj_ave_k3k5p6ftva1e1b1nwzida0e0b0xyg3 = fun.f_divide(mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    ####returns the number of dse of each animal in each dvp - this is combined with the variable numbers when reporting, to get the total dse
    # note: sires have a single long DVP and are on-hand for multiple years. Therefore, mj_ave is higher (because the decision variable is representing multiple sires)
    dsemj_p6tva1e1b1nwzida0e0b0xyg0 = np.sum(mj_ave_p6ftva1e1b1nwzida0e0b0xyg0 / pinp.sheep['i_dse_mj'], axis = 1)
    dsemj_k2p6tva1e1b1nwzida0e0b0xyg1 = np.sum(mj_ave_k2p6ftva1e1b1nwzida0e0b0xyg1 / pinp.sheep['i_dse_mj'], axis = 2)
    dsemj_k3k5p6tva1e1b1nwzida0e0b0xyg3 = np.sum(mj_ave_k3k5p6ftva1e1b1nwzida0e0b0xyg3 / pinp.sheep['i_dse_mj'], axis = 3)

    ###DSE based on SRW or nw (select code below - to switch between SRW & NW requires changing the sire value in pinp)
    ####account for b1 axis effect on dse & select the dse group (note sire and offs don't have b1 axis so simple slice)
    dse_group_dp6tva1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_dse_group'], z_pos, left_pos2=p_pos - 2, right_pos2=z_pos)
    dse_group_dp6tva1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(dse_group_dp6tva1e1b1nwzida0e0b0xyg, numpy=True, axis=z_pos)
    a_dams_dsegroup_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['ia_dams_dsegroup_b1'], b1_pos)

    #### DSE bsed on NW - Using nw doesn't account for the extra MEI of the young growing animals
    #### cumulative total of metabolic nw (unw) over the periods that exist in each p6 (with p6 axis)
    unw_cum_p6tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_nw_start_tpsire**0.75, numbers_p=o_numbers_end_tpsire,
                                        on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tvp=index_p6tpa1e1b1nwzida0e0b0xyg)[:,:,na,...]#add singleton v
    unw_cum_p6tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_nw_start_tpdams**0.75, a_v_pa1e1b1nwzida0e0b0xyg1, numbers_p=o_numbers_end_tpdams,
                                        on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg1,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg)
    unw_cum_p6tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_nw_start_tpoffs**0.75, a_v_pa1e1b1nwzida0e0b0xyg3, numbers_p=o_numbers_end_tpoffs,
                                        on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_p=days_period_cut_pa1e1b1nwzida0e0b0xyg3,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg)
    ####returns the average nw for each animal for each feed period (cum nw accounts for if the animal is on hand - if the animal is sold the average nw will be lower in that feed period)
    # note: sires have a single long DVP and are on-hand for multiple years of the same p6. Therefore, nw_ave is higher (because the decision variable is representing multiple sires)
    unw_ave_p6tva1e1b1nwzida0e0b0xyg0 = fun.f_divide(unw_cum_p6tva1e1b1nwzida0e0b0xyg0, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    unw_ave_p6tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(unw_cum_p6tva1e1b1nwzida0e0b0xyg1, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    unw_ave_p6tva1e1b1nwzida0e0b0xyg3 = fun.f_divide(unw_cum_p6tva1e1b1nwzida0e0b0xyg3, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    ####convert nw to dse based on relative metabolic weight (srw**0.75).
    dsehd_p6tva1e1b1nwzida0e0b0xyg0 = unw_ave_p6tva1e1b1nwzida0e0b0xyg0 / pinp.sheep['i_dse_srw']**0.75
    dsehd_p6tva1e1b1nwzida0e0b0xyg1 = unw_ave_p6tva1e1b1nwzida0e0b0xyg1 / pinp.sheep['i_dse_srw']**0.75
    dsehd_p6tva1e1b1nwzida0e0b0xyg3 = unw_ave_p6tva1e1b1nwzida0e0b0xyg3 / pinp.sheep['i_dse_srw']**0.75
    dsenw_p6tva1e1b1nwzida0e0b0xyg0 = dsehd_p6tva1e1b1nwzida0e0b0xyg0 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_sire_dsegroup']]
    dsenw_p6tva1e1b1nwzida0e0b0xyg1 = dsehd_p6tva1e1b1nwzida0e0b0xyg1 * np.take_along_axis(dse_group_dp6tva1e1b1nwzida0e0b0xyg
                                                                        , a_dams_dsegroup_b1nwzida0e0b0xyg[na,na,na,na,na,na],0)[0,...] #take along the dse group axis and remove the d axis from the front
    dsenw_p6tva1e1b1nwzida0e0b0xyg3 = dsehd_p6tva1e1b1nwzida0e0b0xyg3 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_offs_dsegroup']]

    # ###DSE based on SRW. Currently not working because the axes don't align with the _nw method
    # ####convert SRW to dse based on metabolic weight (w**0.75). SRW of the genotype relative to the base SRW.
    # dsehd_female_yg0 = srw_female_yg0**0.75 / pinp.sheep['i_dse_srw']**0.75
    # dsehd_female_yg1 = srw_female_yg1**0.75 / pinp.sheep['i_dse_srw']**0.75
    # dsehd_female_yg3 = srw_female_yg3**0.75 / pinp.sheep['i_dse_srw']**0.75
    # dsenw_p6tva1e1b1nwzida0e0b0xyg0 = dsehd_female_yg0 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_sire_dsegroup']]
    # dsenw_p6tva1e1b1nwzida0e0b0xyg1 = dsehd_female_yg1 * np.take_along_axis(dse_group_dp6tva1e1b1nwzida0e0b0xyg
    #                                                , a_dams_dsegroup_b1nwzida0e0b0xyg[na,na,na,na,na,na],0)[0,...] #take along the dse group axis and remove the d axis from the front
    # dsenw_p6tva1e1b1nwzida0e0b0xyg3 = dsehd_female_yg3 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_offs_dsegroup']]

    ##cluster and account for numbers/mortality
    dsenw_p6tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', dsenw_p6tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    dsenw_k2p6tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', dsenw_p6tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                mask_vg = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1 * mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...])
    dsenw_k3k5p6tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', dsenw_p6tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ####################
    #report stock days #
    ####################
    ##this needs to be accounted for when reporting variables that have p6 and v axis because they are both periods that do not align
    ##and the number variable returned from pyomo does not have p6 axis. So need to account for the propn of the dvp that the feed period exists.
    stock_days_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(on_hand_pa1e1b1nwzida0e0b0xyg0 * nv_propn_ftpsire
                                        , numbers_p=o_numbers_end_tpsire, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0
                                        , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tvp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])[:,:,:,na,...]#add singleton v
    stock_days_p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(on_hand_tpa1e1b1nwzida0e0b0xyg1 * nv_propn_ftpdams, a_v_pa1e1b1nwzida0e0b0xyg1
                                            , numbers_p=o_numbers_end_tpdams, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg1
                                            , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])
    stock_days_p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(on_hand_tpa1e1b1nwzida0e0b0xyg3 * nv_propn_ftpoffs, a_v_pa1e1b1nwzida0e0b0xyg3
                                            , numbers_p=o_numbers_end_tpoffs, days_period_p=days_period_cut_pa1e1b1nwzida0e0b0xyg3,
                                            a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na,...])

    ##cluster and account for numbers/mortality
    stock_days_p6ftva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', stock_days_p6ftva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    stock_days_k2p6ftva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', stock_days_p6ftva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                mask_vg = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    stock_days_k3k5p6ftva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', stock_days_p6ftva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)





    # plt.plot(o_ffcfw_tpdams[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])         #compare e1 for singles
    # plt.plot(o_ffcfw_tpdams[:, 0, 1, 3, 0, 17:19, 0, 0, 0, 0, 0, 0, 0, 0, 0])         #compare w for singles and e1[1]
    # plt.plot(r_ffcfw_start_tpyatf[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])   #compare e1 for singles
    # plt.plot(o_ebg_tpdams[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])           #compare e1 for singles
    # plt.plot(o_pi_tpdams[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])            #compare e1 for singles
    # plt.plot(o_mei_solid_tpdams[:, 0, 0, :, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])           #compare b1 for first cycle
    # plt.plot(r_intake_f_tpdams[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])        #compare e1 for singles
    # plt.plot(r_md_solid_tpdams[:, 0, 0:2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])        #compare e1 for singles
    # plt.show()

    #############
    #params keys#
    #############
    keys_start=time.time()
    ##param keys - make numpy str to keep size small
    keys_a = pinp.sheep['i_a_idx'][pinp.sheep['i_mask_a']]
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_c1 = np.array(['c1_%d' % i for i in range(len_c1)])
    keys_d = pinp.sheep['i_d_idx'][mask_d_offs]
    keys_g0 = pinp.sheep['i_g_idx_sire'][mask_sire_inc_g0]
    keys_g1 = pinp.sheep['i_g_idx_dams'][mask_dams_inc_g1]
    keys_g2 = keys_g1
    keys_g3 = pinp.sheep['i_g_idx_offs'][mask_offs_inc_g3]
    keys_f = np.array(['nv{0}' .format(i) for i in range(len_f)])
    keys_h1 = np.asarray(uinp.sheep['i_h1_idx'])
    keys_i = pinp.sheep['i_i_idx'][pinp.sheep['i_mask_i']]
    keys_k3 = np.ravel(pinp.sheep['i_k3_idx_offs'])[:len_k3]
    # keys_lw0 = np.array(sinp.stock['i_w_idx_sire'])
    keys_lw1 = np.array(['w%03d'%i for i in range(len_w1)])
    keys_lw3 = np.array(['w%03d'%i for i in range(len_w3)])
    keys_lw_prog = np.array(['w%03d'%i for i in range(len_w_prog)])
    # keys_n0 = sinp.stock['i_n_idx_sire']
    keys_p7 = per.f_season_periods(keys=True)
    keys_n1 = np.array(['n%s'%i for i in range(sinp.structuralsa['i_n1_matrix_len'])])
    keys_n3 = np.array(['n%s'%i for i in range(sinp.structuralsa['i_n3_matrix_len'])])
    keys_p5 = np.array(per.f_p_date2_df().index).astype('str')
    keys_p6 = pinp.period['i_fp_idx']
    keys_p7 = per.f_season_periods(keys=True)
    keys_p8 = np.array(['g0p%s'%i for i in range(len_p8)])
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    keys_s = np.array(['s%s' % i for i in range(len_s)])
    keys_t1 = np.array(['t%s'%i for i in range(len_t1)])
    keys_T1 = np.array(['t%s'%i for i in range(len_gen_t1)]) #generator t keys
    keys_t2 = np.array(['t%s'%i for i in range(len_t2)])
    keys_t3 = np.array(['t%s'%i for i in range(len_t3)])
    keys_T3 = np.array(['t%s'%i for i in range(len_gen_t3)]) #generator t keys
    keys_v1 = np.array(['dv%02d'%i for i in range(dvp_type_va1e1b1nwzida0e0b0xyg1.shape[0])])
    keys_v3 = np.array(['dv%02d'%i for i in range(dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0])])
    keys_x = pinp.sheep['i_x_idx'][mask_x]
    keys_y0 = uinp.parameters['i_y_idx_sire'][uinp.parameters['i_mask_y']]
    keys_y1 = uinp.parameters['i_y_idx_dams'][uinp.parameters['i_mask_y']]
    keys_y3 = uinp.parameters['i_y_idx_offs'][uinp.parameters['i_mask_y']]
    keys_z = zfun.f_keys_z()
    ##save k2 set for pyomo - required because this can't easily be built without information in this module
    params['a_idx'] = keys_a
    params['d_idx'] = keys_d
    params['i_idx'] = keys_i
    params['p8_idx'] = keys_p8
    params['g_idx_sire'] = keys_g0
    params['y_idx_sire'] = keys_y0
    params['dvp_idx_dams'] = keys_v1
    params['w_idx_dams'] = keys_lw1
    params['w_idx_offs'] = keys_lw3
    params['g_idx_dams'] = keys_g1
    params['g_idx_yatf'] = keys_g2
    params['k2_idx_dams'] = keys_k2
    params['t_idx_dams'] = keys_t1
    params['y_idx_dams'] = keys_y1
    params['dvp_idx_offs'] = keys_v3
    params['g_idx_offs'] = keys_g3
    params['k3_idx_offs'] = keys_k3
    params['k5_idx_offs'] = keys_k5
    params['y_idx_offs'] = keys_y3
    params['x_idx_offs'] = keys_x

    ##infrastructure
    arrays_h1p7z = [keys_h1,keys_p7, keys_z]
    arrays_h1c0p7z = [keys_h1, keys_c0, keys_p7, keys_z]

    ##sire related
    ###base sire
    arrays_zg0 = [keys_z, keys_g0]
    ###nsire prov
    arrays_zg0p8 = [keys_z, keys_g0, keys_p8]
    ###nsire req
    arrays_k2tvanwziyg1g0p8 = [keys_k2, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1, keys_g0, keys_p8]

    ##prog related
    ###npw req
    arrays_k3txg = [keys_k3, keys_t2, keys_x, keys_g2]
    ###npw
    arrays_k3k5tva1nw8zixyg1w9i9 = [keys_k3,keys_k5,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_x,
                                    keys_y1,keys_g1,keys_lw_prog,keys_i]
    ###prog to dams req
    arrays_k2k3k5tw8ziyg1g9w9 = [keys_k2, keys_k3, keys_k5, keys_t1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1, keys_g1, keys_lw1]
    ###prog to dams prov
    arrays_k3k5tw8zia0xyg2g9w9 = [keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i, keys_a, keys_x, keys_y1, keys_g2, keys_g1, keys_lw1]
    ###prog to offs prov
    arrays_k3k5tw8ziaxyg2w9 = [keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i, keys_a, keys_x, keys_y1, keys_g3, keys_lw3]
    ###prog to offs req
    arrays_k3vw8zixg3w9 = [keys_k3, keys_v3, keys_lw3, keys_z, keys_i, keys_x, keys_g3, keys_lw3]

    ##dams numbers related
    ###base dams
    arrays_k2tva1nwziyg1 = [keys_k2,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_y1, keys_g1]
    ###numbers req dams
    arrays_k2k2tva1nw8ziyg1g9w9 = [keys_k2,keys_k2,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_y1,
                                   keys_g1,keys_g1,keys_lw1]
    ###numbers dams prov
    arrays_k2k2tvanwziyg1g9w9 = [keys_k2,keys_k2,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_y1,keys_g1,
                                keys_g1,keys_lw1]

    ##offs related
    ###base offs
    arrays_k3k5tvnwziaxyg3 = [keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]
    ###numbers prov offs
    arrays_k3k5tvnw8ziaxyg3w9 = [keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3, keys_lw3]
    ###k3k5wixg3w9 - numbers req offs (doesn't have many active axis)
    arrays_k3k5vw8zixg3w9 = [keys_k3, keys_k5, keys_v3, keys_lw3, keys_z, keys_i, keys_x, keys_g3, keys_lw3]

    ##mei and pi
    ###mei & pi sire
    arrays_p6fzg0 = [keys_p6, keys_f, keys_z, keys_g0]
    ###mei & pi dams
    arrays_k2p6ftva1nwziyg1 = [keys_k2, keys_p6, keys_f, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ###mei & pi offs
    arrays_k3k5p6ftvnwziaxyg3 = [keys_k3, keys_k5, keys_p6, keys_f, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ###p6z - winter grazed propn
    arrays_p6z = [keys_p6, keys_z]
    ###p6zg0 - dse sire
    arrays_p6zg0 = [keys_p6, keys_z, keys_g0]
    ###k2p6tva1nwziyg1 - dse dams
    arrays_k2p6tva1nwziyg1 = [keys_k2, keys_p6, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ###k3k5p6tvnwziaxyg3 - dse offs
    arrays_k3k5p6tvnwziaxyg3 = [keys_k3, keys_k5, keys_p6, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ###cashflow & wc sire
    arrays_c1p7zg0 = [keys_c1, keys_p7, keys_z, keys_g0]
    arrays_c0p7zg0 = [keys_c0, keys_p7, keys_z, keys_g0]
    ###cashflow & wc dams
    arrays_k2c1p7tvanwziyg1 = [keys_k2, keys_c1, keys_p7, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    arrays_k2c0p7tvanwziyg1 = [keys_k2, keys_c0, keys_p7, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ###cashflow & wc prog
    arrays_k3k5c1p7twzia0xg2 = [keys_k3, keys_k5, keys_c1, keys_p7, keys_t2, keys_lw_prog, keys_z, keys_i, keys_a, keys_x, keys_g2]
    arrays_k3k5c0p7twzia0xg2 = [keys_k3, keys_k5, keys_c0, keys_p7, keys_t2, keys_lw_prog, keys_z, keys_i, keys_a, keys_x, keys_g2]
    ###cashflow & wc offs
    arrays_k3k5c1p7tvnwziaxyg3 = [keys_k3, keys_k5, keys_c1, keys_p7, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]
    arrays_k3k5c0p7tvnwziaxyg3 = [keys_k3, keys_k5, keys_c0, keys_p7, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ### asset sire
    arrays_p7zg0 = [keys_p7, keys_z, keys_g0]
    ### asset dams
    arrays_k2p7tvanwziyg1 = [keys_k2, keys_p7, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ### asset offs
    arrays_k3k5p7tvnwziaxyg3 = [keys_k3, keys_k5, keys_p7, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ###p5zg0 - labour sire
    arrays_p5zg0 = [keys_p5, keys_z, keys_g0]
    ###k2p5tvanwziyg1 - labour dams
    arrays_k2p5tvanwziyg1 = [keys_k2, keys_p5, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ###k3k5p5tvnwziaxyg3 - labour offs
    arrays_k3k5p5tvnwziaxyg3 = [keys_k3, keys_k5, keys_p5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ###h1g0 - infrastructure dams
    arrays_h1zg0 = [keys_h1, keys_z, keys_g0]
    ###k2h1tvanwziyg1 - infrastructure dams
    arrays_k2h1tvanwziyg1 = [keys_k2, keys_h1, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    ###k3k5h1tvnwziaxyg3 - infrastructure offs
    arrays_k3k5h1tvnwziaxyg3 = [keys_k3, keys_k5, keys_h1, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3]

    ###z8z9 - season req numbers transfer
    arrays_z8z9 = [keys_z, keys_z]
    ###k2vz8g1z9 - season transfer dams
    arrays_k2vz8g1z9 = [keys_k2, keys_v1, keys_z, keys_g1, keys_z]
    ###k2vz8g1 - season transfer dams (req)
    arrays_k2vz8g1 = [keys_k2, keys_v1, keys_z, keys_g1]
    ###k3vz8xg3z9 - season transfer offs
    arrays_k3vz8xg3z9 = [keys_k3, keys_v3, keys_z, keys_x, keys_g3, keys_z]
    ###k3vz8xg3 - season transfer offs (req)
    arrays_k3vz8xg3 = [keys_k3, keys_v3, keys_z, keys_x, keys_g3]


    ################
    #create params #
    ################

    ##infra r&m cost
    params['p_rm_stockinfra_var'] = fun.f1_make_pyomo_dict(rm_stockinfra_var_h1p7z, arrays_h1p7z)
    params['p_rm_stockinfra_fix'] = fun.f1_make_pyomo_dict(rm_stockinfra_fix_h1p7z, arrays_h1p7z)
    params['p_rm_stockinfra_var_wc'] = fun.f1_make_pyomo_dict(rm_stockinfra_var_wc_h1c0p7z, arrays_h1c0p7z)
    params['p_rm_stockinfra_fix_wc'] = fun.f1_make_pyomo_dict(rm_stockinfra_fix_wc_h1c0p7z, arrays_h1c0p7z)

    ##asset value infra - all in the last season period (doesn't really matter where since it is transferred between each season period)
    keys_p7_end = keys_p7[-1:]
    arrays_p7h1 = [keys_p7_end,keys_h1]
    params['p_infra'] = fun.f1_make_pyomo_dict(assetvalue_infra_h1, arrays_p7h1)


    ##sire related
    ###sires provided
    params['p_nsire_prov_sire'] = fun.f1_make_pyomo_dict(numbers_startp8_tva1e1b1nwzida0e0b0xyg0p8, arrays_zg0p8)
    ###nsire_dams
    params['p_nsire_req_dams'] = fun.f1_make_pyomo_dict(nsire_k2tva1e1b1nwzida0e0b0xyg1g0p8, arrays_k2tvanwziyg1g0p8)

    ##prog related
    ###npw required by prog activity
    params['p_npw_req_prog'] = fun.f1_make_pyomo_dict(numbers_prog_req_k3k5tva1e1b1nwzida0e0b0xyg2w9, arrays_k3txg)
    ###number prog weaned
    params['p_npw_dams'] = fun.f1_make_pyomo_dict(npw_k3k5tva1e1b1nwzida0e0b0xyg1w9i9, arrays_k3k5tva1nw8zixyg1w9i9, loop_axis_pos=p_pos-2, index_loop_axis_pos=-11) #different because the w pos in the param is different to the keys due to singleton axis which are removed.
    ###number prog require by dams
    params['p_progreq_dams'] = fun.f1_make_pyomo_dict(numbers_progreq_k2k3k5tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k3k5tw8ziyg1g9w9, loop_axis_pos=0, index_loop_axis_pos=0) #loop on k2 axis
    ###number prog require by offs
    #todo add a y-axis to prog. Requires changing this parameter
    params['p_progreq_offs'] = fun.f1_make_pyomo_dict(numbers_progreq_k3k5tva1e1b1nw8zida0e0b0xyg3w9, arrays_k3vw8zixg3w9, loop_axis_pos=0, index_loop_axis_pos=0) #loop on k3 axis
    ###number prog provided to dams
    params['p_progprov_dams'] = fun.f1_make_pyomo_dict(numbers_prog2dams_k3k5tva1e1b1nwzida0e0b0xyg2g9w9, arrays_k3k5tw8zia0xyg2g9w9, loop_axis_pos=0, index_loop_axis_pos=0) #loop on k3 axis
    ###number prog provided to offs
    params['p_progprov_offs'] = fun.f1_make_pyomo_dict(numbers_prog2offs_k3k5tva1e1b1nwzida0e0b0xyg2w9, arrays_k3k5tw8ziaxyg2w9, loop_axis_pos=0, index_loop_axis_pos=0) #loop on k3 axis

    ##dams
    ###numbers_req_dams
    params['numbers_req_numpyversion_k2k2tva1nw8ziyg1g9w9'] = numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9[:,:,:,:,:,0,0,:,:,:,:,0,0,0,0,0,:,:,:,:]  #don't use squeeze here because keeping all relevant axis even if singleton speeds the pyomo constraint.
    params['p_numbers_req_dams'] = fun.f1_make_pyomo_dict(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tva1nw8ziyg1g9w9, loop_axis_pos=p_pos-2, index_loop_axis_pos=-10)
    ###numbers_prov_dams
    ####numbers provided into next period (the norm)
    params['p_numbers_prov_dams'] = fun.f1_make_pyomo_dict(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tvanwziyg1g9w9, loop_axis_pos=p_pos-2, index_loop_axis_pos=-10)
    #### provided into this period (when transferring from an earlier lambing ram group to a later lambing)
    params['p_numbers_provthis_dams'] = fun.f1_make_pyomo_dict(numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tvanwziyg1g9w9, loop_axis_pos=p_pos-2, index_loop_axis_pos=-10)

    ##offs related
    ###numbers_req_offs
    params['numbers_req_numpyversion_k3k5vw8zixg3w9'] = numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9[:,:,0,:,0,0,0,0,:,:,:,0,0,0,0,:,0,:,:]  #don't use squeeze here because keeping all relevant axis even if singleton speeds the pyomo constraint.
    params['p_numbers_req_offs'] = fun.f1_make_pyomo_dict(numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9, arrays_k3k5vw8zixg3w9, loop_axis_pos=p_pos-1, index_loop_axis_pos=-7)
    ###numbers_prov_offs
    params['p_numbers_prov_offs'] = fun.f1_make_pyomo_dict(numbers_prov_offs_k3k5tva1e1b1nw8zida0e0b0xygw9, arrays_k3k5tvnw8ziaxyg3w9, loop_axis_pos=p_pos-1, index_loop_axis_pos=-10)

    ##emissions
    params['p_co2e_zg0'] = fun.f1_make_pyomo_dict(co2e_animal_tva1e1b1nwzida0e0b0xyg0, arrays_zg0)
    params['p_co2e_k2tva1nwziyg1'] = fun.f1_make_pyomo_dict(co2e_animal_k2tva1e1b1nwzida0e0b0xyg1, arrays_k2tva1nwziyg1)
    params['p_co2e_k3k5tvnwziaxyg3'] = fun.f1_make_pyomo_dict(co2e_animal_k3k5tva1e1b1nwzida0e0b0xyg3, arrays_k3k5tvnwziaxyg3)


    ##mei
    ###mei - sire
    params['p_mei_sire'] = fun.f1_make_pyomo_dict(mei_p6ftva1e1b1nwzida0e0b0xyg0, arrays_p6fzg0)
    ###mei - dams
    params['p_mei_dams'] = fun.f1_make_pyomo_dict(mei_k2p6ftva1e1b1nwzida0e0b0xyg1, arrays_k2p6ftva1nwziyg1)
    ###mei - offs
    params['p_mei_offs'] = fun.f1_make_pyomo_dict(mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3, arrays_k3k5p6ftvnwziaxyg3, loop_axis_pos=p_pos, index_loop_axis_pos=-9)

    ##pi
    ###pi - sire
    params['p_pi_sire'] = fun.f1_make_pyomo_dict(pi_p6ftva1e1b1nwzida0e0b0xyg0, arrays_p6fzg0)
    ###pi - dams
    params['p_pi_dams'] = fun.f1_make_pyomo_dict(pi_k2p6ftva1e1b1nwzida0e0b0xyg1, arrays_k2p6ftva1nwziyg1)
    ###pi - offs
    params['p_pi_offs'] = fun.f1_make_pyomo_dict(pi_k3k5p6ftva1e1b1nwzida0e0b0xyg3, arrays_k3k5p6ftvnwziaxyg3)

    ##cashflow
    ###cashflow - sire
    params['p_cashflow_sire'] = fun.f1_make_pyomo_dict(cashflow_c1p7tva1e1b1nwzida0e0b0xyg0, arrays_c1p7zg0)
    ###cashflow - dams
    params['p_cashflow_dams'] = fun.f1_make_pyomo_dict(cashflow_k2c1p7tva1e1b1nwzida0e0b0xyg1, arrays_k2c1p7tvanwziyg1)
    ###cashflow - prog - only consists of sale value
    params['p_cashflow_prog'] = fun.f1_make_pyomo_dict(salevalue_prog_k3k5c1p7tva1e1b1nwzida0e0b0xyg2, arrays_k3k5c1p7twzia0xg2)
    ###cashflow - offs
    params['p_cashflow_offs'] = fun.f1_make_pyomo_dict(cashflow_k3k5c1p7tva1e1b1nwzida0e0b0xyg3, arrays_k3k5c1p7tvnwziaxyg3)

    ##wc
    ###wc - sire
    params['p_wc_sire'] = fun.f1_make_pyomo_dict(wc_c0p7tva1e1b1nwzida0e0b0xyg0, arrays_c0p7zg0)
    ###wc - dams
    params['p_wc_dams'] = fun.f1_make_pyomo_dict(wc_k2c0p7tva1e1b1nwzida0e0b0xyg1, arrays_k2c0p7tvanwziyg1)
    ###wc - prog - only consists of sale value
    params['p_wc_prog'] = fun.f1_make_pyomo_dict(salevalue_wc_prog_k3k5c0p7tva1e1b1nwzida0e0b0xyg2, arrays_k3k5c0p7twzia0xg2)
    ###wc - offs
    params['p_wc_offs'] = fun.f1_make_pyomo_dict(wc_k3k5c0p7tva1e1b1nwzida0e0b0xyg3, arrays_k3k5c0p7tvnwziaxyg3)

    ##cost (for minROE)
    ###cost - sire
    params['p_cost_sire'] = fun.f1_make_pyomo_dict(cost_p7tva1e1b1nwzida0e0b0xyg0, arrays_p7zg0)
    ###cost - dams
    params['p_cost_dams'] = fun.f1_make_pyomo_dict(cost_k2p7tva1e1b1nwzida0e0b0xyg1, arrays_k2p7tvanwziyg1)
    ###cost - offs
    params['p_cost_offs'] = fun.f1_make_pyomo_dict(cost_k3k5p7tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p7tvnwziaxyg3)

    ##purchase cost
    ###purchcost - sire
    params['p_purchcost_sire'] = fun.f1_make_pyomo_dict(purchcost_p7tva1e1b1nwzida0e0b0xyg0, arrays_p7zg0)

    ##purchase wc
    ###purchcost wc - sire
    params['p_purchcost_wc_sire'] = fun.f1_make_pyomo_dict(purchcost_wc_c0p7tva1e1b1nwzida0e0b0xyg0, arrays_c0p7zg0)

    ##asset value - take slice a5[0] to get the asset value at the cashflow date
    ###sire
    params['p_assetvalue_sire'] = fun.f1_make_pyomo_dict(assetvalue_a5p7tva1e1b1nwzida0e0b0xyg0[0], arrays_p7zg0)
    ###dams
    params['p_assetvalue_dams'] = fun.f1_make_pyomo_dict(assetvalue_a5k2p7tva1e1b1nwzida0e0b0xyg1[0], arrays_k2p7tvanwziyg1)
    ###offs
    params['p_assetvalue_offs'] = fun.f1_make_pyomo_dict(assetvalue_a5k3k5p7tva1e1b1nwzida0e0b0xyg3[0], arrays_k3k5p7tvnwziaxyg3)

    ##trade value
    ## a5[1] start of season & a5[2] end of season.
    ###sire
    params['p_tradevalue_p7zg0'] = fun.f1_make_pyomo_dict(assetvalue_p7tva1e1b1nwzida0e0b0xyg0, arrays_p7zg0)
    ###dams
    params['p_tradevalue_k2p7tva1nwziyg1'] = fun.f1_make_pyomo_dict(assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1, arrays_k2p7tvanwziyg1)
    ###offs
    params['p_tradevalue_k3k5p7tvnwziaxyg3'] = fun.f1_make_pyomo_dict(assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p7tvnwziaxyg3)

    ##labour
    ###anyone labour - sire
    params['p_labour_anyone_sire'] = fun.f1_make_pyomo_dict(lab_anyone_p5tva1e1b1nwzida0e0b0xyg0, arrays_p5zg0)
    ###perm labour - sire
    params['p_labour_perm_sire'] = fun.f1_make_pyomo_dict(lab_perm_p5tva1e1b1nwzida0e0b0xyg0, arrays_p5zg0)
    ###manager labour - sire
    params['p_labour_manager_sire'] = fun.f1_make_pyomo_dict(lab_manager_p5tva1e1b1nwzida0e0b0xyg0, arrays_p5zg0)
    ###anyone labour - dams
    params['p_labour_anyone_dams'] = fun.f1_make_pyomo_dict(lab_anyone_k2p5tva1e1b1nwzida0e0b0xyg1, arrays_k2p5tvanwziyg1)
    ###perm labour - dams
    params['p_labour_perm_dams'] = fun.f1_make_pyomo_dict(lab_perm_k2p5tva1e1b1nwzida0e0b0xyg1, arrays_k2p5tvanwziyg1)
    ###manager labour - dams
    params['p_labour_manager_dams'] = fun.f1_make_pyomo_dict(lab_manager_k2p5tva1e1b1nwzida0e0b0xyg1, arrays_k2p5tvanwziyg1)
    ###anyone labour - offs
    params['p_labour_anyone_offs'] = fun.f1_make_pyomo_dict(lab_anyone_k3k5p5tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p5tvnwziaxyg3)
    ###perm labour - offs
    params['p_labour_perm_offs'] = fun.f1_make_pyomo_dict(lab_perm_k3k5p5tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p5tvnwziaxyg3)
    ###manager labour - offs
    params['p_labour_manager_offs'] = fun.f1_make_pyomo_dict(lab_manager_k3k5p5tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p5tvnwziaxyg3)

    ###infrastructure - sire
    params['p_infrastructure_sire'] = fun.f1_make_pyomo_dict(infrastructure_h1tva1e1b1nwzida0e0b0xyg0, arrays_h1zg0)
    ###infrastructure - dams
    params['p_infrastructure_dams'] = fun.f1_make_pyomo_dict(infrastructure_k2h1tva1e1b1nwzida0e0b0xyg1, arrays_k2h1tvanwziyg1)
    ###infrastructure - offs
    params['p_infrastructure_offs'] = fun.f1_make_pyomo_dict(infrastructure_k3k5p5tva1e1b1nwzida0e0b0xyg3, arrays_k3k5h1tvnwziaxyg3)

    ##DSE - sire
    if pinp.sheep['i_dse_type'] == 0:
        params['p_dse_sire'] = fun.f1_make_pyomo_dict(dsenw_p6tva1e1b1nwzida0e0b0xyg0, arrays_p6zg0)
    else:
        params['p_dse_sire'] = fun.f1_make_pyomo_dict(dsemj_p6tva1e1b1nwzida0e0b0xyg0, arrays_p6zg0)
    ##DSE - dams
    if pinp.sheep['i_dse_type'] == 0:
        params['p_dse_dams'] = fun.f1_make_pyomo_dict(dsenw_k2p6tva1e1b1nwzida0e0b0xyg1, arrays_k2p6tva1nwziyg1)
    else:
        params['p_dse_dams'] = fun.f1_make_pyomo_dict(dsemj_k2p6tva1e1b1nwzida0e0b0xyg1, arrays_k2p6tva1nwziyg1)
    ##DSE - offs
    if pinp.sheep['i_dse_type'] == 0:
        params['p_dse_offs'] = fun.f1_make_pyomo_dict(dsenw_k3k5p6tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p6tvnwziaxyg3)
    else:
        params['p_dse_offs'] = fun.f1_make_pyomo_dict(dsemj_k3k5p6tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p6tvnwziaxyg3)

    ##winter grazed propn - indicates the propn of the DSE in each FP that is used to calculate total DSE for SR
    wg_propn_p6z = zfun.f_seasonal_inp(pinp.sheep['i_wg_propn_p6z'], numpy=True, axis=-1)
    params['p_wg_propn_p6z'] =  fun.f1_make_pyomo_dict(wg_propn_p6z, arrays_p6z)

    ##season transfer masks
    ###dams req within
    params['p_mask_childz_within_dams'] = fun.f1_make_pyomo_dict(mask_childz_reqwithin_k2tva1e1b1nwzida0e0b0xyg1, arrays_k2vz8g1)
    ###dams req between
    params['p_mask_childz_between_dams'] = fun.f1_make_pyomo_dict(mask_childz_reqbetween_k2tva1e1b1nwzida0e0b0xyg1, arrays_k2vz8g1)
    ###offs req within
    params['p_mask_childz_within_offs'] = fun.f1_make_pyomo_dict(mask_childz_reqwithin_k3k5tva1e1b1nwzida0e0b0xyg3, arrays_k3vz8xg3)
    ###offs req between
    params['p_mask_childz_between_offs'] = fun.f1_make_pyomo_dict(mask_childz_reqbetween_k3k5tva1e1b1nwzida0e0b0xyg3, arrays_k3vz8xg3)

    ###dams prov within
    params['p_parentz_provwithin_dams'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_k2tva1e1b1nwzida0e0b0xyg1z9, arrays_k2vz8g1z9)
    ###dams prov between
    params['p_parentz_provbetween_dams'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_k2tva1e1b1nwzida0e0b0xyg1z9, arrays_k2vz8g1z9)
    ###offs prov within
    params['p_parentz_provwithin_offs'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_k3k5tva1e1b1nwzida0e0b0xyg3z9, arrays_k3vz8xg3z9)
    ###offs prov between
    params['p_parentz_provbetween_offs'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_k3k5tva1e1b1nwzida0e0b0xyg3z9, arrays_k3vz8xg3z9)



    ###############
    # REV         #
    ###############
    ##store rev if trial is rev_create
    if sinp.structuralsa['i_rev_create']:
        with open('pkl/pkl_rev_trait{0}.pkl'.format(rev_number),"wb") as f:
            pkl.dump(rev_trait_values, f)

    ###############################
    # Save condensed start animal #
    ###############################
    ##store condensed start info. This is used in future trials to improve fs optimisation. See google doc (randomness section) for more info.
    ##this gets saved in the pkl file with other feed info in FeedSupplyStock.py
    pkl_fs_info['pkl_condensed_values'] = pkl_condensed_values

    ################
    # Bound params #
    ################
    '''store params used in BoundsPyomo.py'''
    ###shapes
    len_v1 = len(keys_v1)
    len_v3 = len(keys_v3)

    ##mask for dam activities
    arrays_k2tvwzg1 = [keys_k2, keys_t1, keys_v1, keys_lw1, keys_z, keys_g1]
    params['p_mask_dams'] = fun.f1_make_pyomo_dict(mask_dams_k2tva1e1b1nw8zida0e0b0xyg1, arrays_k2tvwzg1)
    ##mask for prog activities
    arrays_tdxg2 = [keys_t2, keys_d, keys_x, keys_g2]
    params['p_mask_prog'] = fun.f1_make_pyomo_dict(mask_prog_tdx_tva1e1b1nwzida0e0b0xyg2w9, arrays_tdxg2)
    ##mask for offs activities
    arrays_k3vwzxg3 = [keys_k3, keys_v3, keys_lw3, keys_z, keys_x, keys_g3]
    params['p_mask_offs'] = fun.f1_make_pyomo_dict(mask_offs_k3k5tva1e1b1nw8zida0e0b0xyg3, arrays_k3vwzxg3)

    ##lower bound dams
    ### this bound can be defined with either tog1 axes or tvg1 axes in exp.xl. Uncomment the relevant code to align with exp.xl
    ### Note: if using the V axis, be aware of changes that might add slices to the v axis (season nodes, or new DVPs) that will alter the outcome of the bound defined in exp.xl
    # bnd_lower_dams_tog1 = fun.f_sa(np.array([0],dtype=float), sen.sav['bnd_lo_dams_tog1'], 5)
    # bnd_lower_dams_toa1e1b1nwzida0e0b0xyg1 = fun.f_expand(bnd_lower_dams_tog1, left_pos=p_pos, right_pos=-1,
    #                                                 condition=mask_t1, axis=p_pos-1, condition2=mask_o_dams, axis2=p_pos, condition3=mask_dams_inc_g1, axis3=-1)
    # bnd_lower_dams_tva1e1b1nwzida0e0b0xyg1 = np.take_along_axis(bnd_lower_dams_toa1e1b1nwzida0e0b0xyg1,
    #                                                             a_prev_o_va1e1b1nwzida0e0b0xyg1[na,:,:,0:1,...], #take e[0] because o is not effected by e
    #                                                             axis=p_pos)  # increments at prejoining
    bnd_lower_dams_tVg1 = fun.f_sa(np.array([0],dtype=float), sen.sav['bnd_lo_dams_tVg1'], 5)
    bnd_lower_dams_tVa1e1b1nwzida0e0b0xyg1 = fun.f_expand(bnd_lower_dams_tVg1, left_pos=p_pos, right_pos=-1,
                                                    condition=mask_t1, axis=p_pos-1, condition2=mask_dams_inc_g1, axis2=-1)
    bnd_lower_dams_tVa1e1b1nwzida0e0b0xyg1 = bnd_lower_dams_tVa1e1b1nwzida0e0b0xyg1 * (index_zidaebxyg==index_zidaebxyg) #need to activate z axis because z is active if you use the lobound method above.
    ### slice the approximated V axis created in Sensitivity.py to the correct length
    bnd_lower_dams_tva1e1b1nwzida0e0b0xyg1 = bnd_lower_dams_tVa1e1b1nwzida0e0b0xyg1[:, 0:len_v1, ...]
    arrays_tvzg1 = [keys_t1, keys_v1, keys_z, keys_g1]
    params['p_dams_lobound'] = fun.f1_make_pyomo_dict(bnd_lower_dams_tva1e1b1nwzida0e0b0xyg1, arrays_tvzg1)

    ##upper bound dams
    ### this bound can be defined with either tog1 axes or tvg1 axes in exp.xl. Uncomment the relevant code to align with exp.xl
    ### Note: if using the V axis, be aware of changes that might add slices to the v axis (season nodes, or new DVPs) that will alter the outcome of the bound defined in exp.xl
    # bnd_upper_dams_tog1 = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_up_dams_tog1'], 5) #999999 just an arbitrary value used then converted to np.inf because np.inf causes errors in the f_update which is called by f_sa
    # bnd_upper_dams_tog1[bnd_upper_dams_tog1==999999] = np.inf
    # bnd_upper_dams_toa1e1b1nwzida0e0b0xyg1 = fun.f_expand(bnd_upper_dams_tog1, left_pos=p_pos, right_pos=-1,
    #                                                 condition=mask_t1, axis=p_pos-1, condition2=mask_o_dams, axis2=p_pos, condition3=mask_dams_inc_g1, axis3=-1)
    # bnd_upper_dams_tva1e1b1nwzida0e0b0xyg1 = np.take_along_axis(bnd_upper_dams_toa1e1b1nwzida0e0b0xyg1,
    #                                                             a_prev_o_va1e1b1nwzida0e0b0xyg1[na,:,:,0:1,...], #take e[0] because o is not effected by e
    #                                                             axis=p_pos)  # increments at prejoining
    bnd_upper_dams_tVg1 = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_up_dams_tVg1'], 5) #999999 just an arbitrary value used then converted to np.inf because np.inf causes errors in the f_update which is called by f_sa
    bnd_upper_dams_tVg1[bnd_upper_dams_tVg1==999999] = np.inf
    bnd_upper_dams_tVa1e1b1nwzida0e0b0xyg1 = fun.f_expand(bnd_upper_dams_tVg1, left_pos=p_pos, right_pos=-1,
                                                    condition=mask_t1, axis=p_pos-1, condition2=mask_dams_inc_g1, axis2=-1)
    bnd_upper_dams_tVa1e1b1nwzida0e0b0xyg1 = bnd_upper_dams_tVa1e1b1nwzida0e0b0xyg1 * (index_zidaebxyg==index_zidaebxyg) #need to activate z axis because z is active if you use the upbound method above.
    ### slice the approximated V axis created in Sensitivity.py to the correct length
    bnd_upper_dams_tva1e1b1nwzida0e0b0xyg1 = bnd_upper_dams_tVa1e1b1nwzida0e0b0xyg1[:, 0:len_v1, ...]
    arrays_tvzg1 = [keys_t1, keys_v1, keys_z, keys_g1]
    params['p_dams_upbound'] = fun.f1_make_pyomo_dict(bnd_upper_dams_tva1e1b1nwzida0e0b0xyg1, arrays_tvzg1)

    ##proportion of dams mated. inf means the model can optimise the proportion because inf is used to skip the constraint.
    prop_dams_mated_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesn't impact mating propn
    prop_dams_mated_va1e1b1nwzida0e0b0xyg1[np.logical_not(dvp_is_mating)] = np.inf
    #prop_dams_mated_va1e1b1nwzida0e0b0xyg1 = fun.f_update(prop_dams_mated_va1e1b1nwzida0e0b0xyg1, dvp_is_mating==0, np.inf)
    arrays_vzg1 = [keys_v1, keys_z, keys_g1]
    params['p_prop_dams_mated'] = fun.f1_make_pyomo_dict(prop_dams_mated_va1e1b1nwzida0e0b0xyg1, arrays_vzg1)

    ##proportion of dry dams as a propn of preg dams at shearing sale. This is different to the propn in the dry report because it is the propn at a given time rather than per animal at the beginning of mating.
    ## This is used to force retention of drys at the main (t[0]) sale time. You can only sell drys if you sell non-drys. This param indicates the propn of dry that can be sold per non-dry dam.
    propn_drys_tpg1 = fun.f_divide(np.sum(o_numbers_end_tpdams*n_drys_b1g1, axis=(e1_pos,b1_pos), keepdims=True),
                              np.sum(o_numbers_end_tpdams * (nyatf_b1nwzida0e0b0xyg>0),axis=(e1_pos,b1_pos), keepdims=True))
    propn_drys_t0_vg1 = sfun.f1_p2v(propn_drys_tpg1[0:1,...], a_v_pa1e1b1nwzida0e0b0xyg1[:,:,0:1,...], #only interested in the shearing sale t[0] (t axis will be active if generating with t
                                period_is_tp=period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1[:,:,0:1,...]) #take e[0] it is the same as e[1] so don't need it.
    arrays_vanwziyg1 = [keys_v1, keys_a, keys_n1, keys_lw1, keys_z, keys_i, keys_y1, keys_g1]
    params['p_prop_dry_t0_dams'] = fun.f1_make_pyomo_dict(propn_drys_t0_vg1, arrays_vanwziyg1)

    ##drys retained (bool used to control if bound constraint is built that limits the number of drys sold using p_prop_dry_t0_dams)
    ### can only sell drys only if pregnant dams are also being sold.
    dry_retained_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(dry_retained_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesn't impact o axis (o is the input axis)
    arrays_vzg1 = [keys_v1, keys_z, keys_g1]
    params['p_drys_retained'] = fun.f1_make_pyomo_dict(dry_retained_va1e1b1nwzida0e0b0xyg1, arrays_vzg1)
    #todo include the birth timing in this param when gbal is activated (currently it only forces retention in scanning dvp. Birth dvp could be activated is gbal used)

    ##proportion of drys that are twice dry
    ###expand for p axis
    prop_twice_dry_dams_oa1e1b1nwzia0e0b0xyg1 = np.moveaxis(ce_dams[2,...], source=d_pos, destination=0) #move d axis to p pos
    prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1 = np.expand_dims(prop_twice_dry_dams_oa1e1b1nwzia0e0b0xyg1, d_pos) #add singleton d axis
    prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1[0] = 0 #can't have any twice drys in the first mating opportunity. (this line is just here to stop error if user accidentally puts in a value for o[0]).
    prop_twice_dry_dams_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0) #increments at prejoining
    ###convert to v axis
    prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_twice_dry_dams_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesn't impact mating propn
    ###adjust 2-tooth twice drys for yearling mating (and other age groups if not 100% mated).
    ### the proportion of yearlings that were mated adjusts the proportion of the dry 2- tooths that are twice dry
    ### E.g. if no yearlings were mated then no 2-tooths are twice dry.
    ####calc propn of dams mated in previous opportunity uses the estimated proportion of dams mated
    prop_dams_mated_prev_oa1e1b1nwzida0e0b0xyg1 = np.roll(est_prop_dams_mated_oa1e1b1nwzida0e0b0xyg1, shift=1, axis=0)
    prop_dams_mated_prev_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_prev_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0) #increments at prejoining
    prop_dams_mated_prev_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_prev_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesn't impact mating propn
    prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 = prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 * np.minimum(1,prop_dams_mated_prev_va1e1b1nwzida0e0b0xyg1)
    ###create param
    arrays_vziyg1 = [keys_v1, keys_z, keys_i, keys_y1, keys_g1]
    params['p_prop_twice_dry_dams'] = fun.f1_make_pyomo_dict(prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1, arrays_vziyg1)
    params['p_prejoin_v_dams'] = keys_v1[dvp_type_va1e1b1nwzida0e0b0xyg1[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0]==prejoin_vtype1] #get the dvp keys which are prejoining (same for all animals hence take slice 0)
    params['p_scan_v_dams'] = keys_v1[dvp_type_va1e1b1nwzida0e0b0xyg1[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0]==scan_vtype1] #get the dvp keys which are scan (same for all animals hence take slice 0)

    ##lower bound offs
    ###build a mask which indicates if there is a future shearing
    t_period_is_shearing_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(period_is_mainshearing_pa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)
    future_shearing_exists_tva1e1b1nwzida0e0b0xyg3 = np.flip(np.maximum.accumulate(np.flip(t_period_is_shearing_va1e1b1nwzida0e0b0xyg3, axis=p_pos), axis=p_pos), axis=p_pos)
    ###build bnds
    bnd_lower_offs_tsdxg3 = fun.f_sa(np.array([0],dtype=float), sen.sav['bnd_lo_offs_tsdxg3'], 5)
    bnd_lower_offs_tsa1e1b1nwzida0e0b0xyg3 = fun.f_expand(bnd_lower_offs_tsdxg3, left_pos=x_pos, right_pos=-1,
                                                          left_pos2=d_pos, right_pos2=x_pos, left_pos3=p_pos, right_pos3=d_pos,
                                                          condition=mask_d_offs, axis=d_pos, condition2=mask_x, axis2=x_pos, condition3=mask_offs_inc_g3, axis3=-1)
    bnd_lower_offs_tva1e1b1nwzida0e0b0xyg3 = np.take_along_axis(bnd_lower_offs_tsa1e1b1nwzida0e0b0xyg3,
                                                                a_next_s_va1e1b1nwzida0e0b0xyg3[na,...],
                                                                axis=p_pos) * future_shearing_exists_tva1e1b1nwzida0e0b0xyg3 #mask is to set the low bnd to 0 after the last shearing (don't want a low bnd if there is not shearing opportunity because the animals can't be sold)
    ##Note: when next_s is beyond the end of the sim a_next_s points to the final shearing occurrence.
    ### this requires a one-off fix so that the bound does not incorrectly operate on the final DVP
    bnd_lower_offs_k3k5tva1e1b1nwzida0e0b0xyg3 = np.sum(bnd_lower_offs_tva1e1b1nwzida0e0b0xyg3
                                                         * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                         axis=d_pos, keepdims=True) #cluster d
    arrays_k3tvzxg3 = [keys_k3, keys_t3, keys_v3, keys_z, keys_x, keys_g3]
    params['p_offs_lobound'] = fun.f1_make_pyomo_dict(bnd_lower_offs_k3k5tva1e1b1nwzida0e0b0xyg3, arrays_k3tvzxg3)

    ##upper bound offs
    bnd_upper_offs_tsdxg3 = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_up_offs_tsdxg3'], 5) #999999 just an arbitrary high value (can't use np.inf because it becomes nan in the following calcs)
    # bnd_upper_offs_tsdxg3[bnd_upper_offs_tsdxg3==999999] = np.inf
    bnd_upper_offs_tsa1e1b1nwzida0e0b0xyg3 = fun.f_expand(bnd_upper_offs_tsdxg3, left_pos=x_pos, right_pos=-1,
                                                          left_pos2=d_pos, right_pos2=x_pos, left_pos3=p_pos, right_pos3=d_pos,
                                                          condition=mask_d_offs, axis=d_pos, condition2=mask_x, axis2=x_pos, condition3=mask_offs_inc_g3, axis3=-1)
    bnd_upper_offs_tva1e1b1nwzida0e0b0xyg3 = np.take_along_axis(bnd_upper_offs_tsa1e1b1nwzida0e0b0xyg3,
                                                                a_next_s_va1e1b1nwzida0e0b0xyg3[na,...],
                                                                axis=p_pos)
    bnd_upper_offs_k3k5tva1e1b1nwzida0e0b0xyg3 = np.sum(bnd_upper_offs_tva1e1b1nwzida0e0b0xyg3
                                                         * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                         axis=d_pos, keepdims=True) #cluster d
    arrays_k3tvzxg3 = [keys_k3, keys_t3, keys_v3, keys_z, keys_x, keys_g3]
    params['p_offs_upbound'] = fun.f1_make_pyomo_dict(bnd_upper_offs_k3k5tva1e1b1nwzida0e0b0xyg3, arrays_k3tvzxg3)

    ##upper bound prog
    bnd_upper_prog_tdxg2 = fun.f_sa(np.array([999999],dtype=float), sen.sav['bnd_up_prog_tdxg2'], 5) #999999 just an arbitrary high value
    # bnd_upper_prog_tdxg2[bnd_upper_prog_tdxg2==999999] = np.inf  # (can't use np.inf because it becomes nan in the following calcs)
    bnd_upper_prog_tva1e1b1nwzida0e0b0xyg2 = fun.f_expand(bnd_upper_prog_tdxg2, left_pos=x_pos, right_pos=-1,
                                                          left_pos2=d_pos, right_pos2=x_pos, left_pos3=p_pos-1, right_pos3=d_pos,
                                                          condition=mask_d_offs, axis=d_pos, condition2=mask_x, axis2=x_pos, condition3=mask_offs_inc_g3, axis3=-1)
    bnd_upper_prog_k3k5tva1e1b1nwzida0e0b0xyg2 = np.sum(bnd_upper_prog_tva1e1b1nwzida0e0b0xyg2
                                                         * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                         axis=d_pos, keepdims=True) #cluster d
    arrays_k3txg2 = [keys_k3, keys_t2, keys_x, keys_g2]
    params['p_prog_upbound'] = fun.f1_make_pyomo_dict(bnd_upper_prog_k3k5tva1e1b1nwzida0e0b0xyg2, arrays_k3txg2)




    ###############
    # report      #
    ###############
    '''add report values to report dict and do any additional calculations'''

    ##create z8 mask to uncluster report vars
    ###dams - cluster e and b (e axis is active from the dvp dates)
    mask_z8var_va1e1b1nwzida0e0b0xyg1 = zfun.f_season_transfer_mask(dvp_start_va1e1b1nwzida0e0b0xyg1, z_pos=z_pos, mask=True)
    mask_z8var_k2tva1e1b1nwzida0e0b0xyg1 = 1 * (np.sum(mask_z8var_va1e1b1nwzida0e0b0xyg1
                                                * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k2tva1e1b1nwzida0e0b0xyg1),
                                                axis=(e1_pos,b1_pos), keepdims=True) > 0)

    ###offs - cluster d (d axis is active from the dvp dates)
    mask_z8var_va1e1b1nwzida0e0b0xyg3 = zfun.f_season_transfer_mask(dvp_start_va1e1b1nwzida0e0b0xyg3, z_pos=z_pos, mask=True)
    mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3 = 1 * (np.sum(mask_z8var_va1e1b1nwzida0e0b0xyg3
                                                         * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3),
                                                         axis=d_pos, keepdims=True) > 0)
    ##store in report dict
    ###keys
    fun.f1_make_r_val(r_vals,keys_a,'keys_a')
    fun.f1_make_r_val(r_vals,keys_d,'keys_d')
    fun.f1_make_r_val(r_vals,keys_g0,'keys_g0')
    fun.f1_make_r_val(r_vals,keys_g1,'keys_g1')
    fun.f1_make_r_val(r_vals,keys_g2,'keys_g2')
    fun.f1_make_r_val(r_vals,keys_g3,'keys_g3')
    fun.f1_make_r_val(r_vals,keys_f,'keys_f')
    fun.f1_make_r_val(r_vals,keys_h1,'keys_h1')
    fun.f1_make_r_val(r_vals,keys_i,'keys_i')
    fun.f1_make_r_val(r_vals,keys_k2,'keys_k2')
    fun.f1_make_r_val(r_vals,keys_k3,'keys_k3')
    fun.f1_make_r_val(r_vals,keys_k5,'keys_k5')
    fun.f1_make_r_val(r_vals,keys_lw1,'keys_lw1')
    fun.f1_make_r_val(r_vals,keys_lw3,'keys_lw3')
    fun.f1_make_r_val(r_vals,keys_lw_prog,'keys_lw_prog')
    fun.f1_make_r_val(r_vals,keys_n1,'keys_n1')
    fun.f1_make_r_val(r_vals,keys_n3,'keys_n3')
    fun.f1_make_r_val(r_vals,keys_p6,'keys_p6')
    fun.f1_make_r_val(r_vals,keys_p8,'keys_p8')
    fun.f1_make_r_val(r_vals,keys_t1,'keys_t1')
    fun.f1_make_r_val(r_vals,keys_t2,'keys_t2')
    fun.f1_make_r_val(r_vals,keys_t3,'keys_t3')
    fun.f1_make_r_val(r_vals,keys_v1,'keys_v1')
    fun.f1_make_r_val(r_vals,keys_v3,'keys_v3')
    fun.f1_make_r_val(r_vals,keys_y0,'keys_y0')
    fun.f1_make_r_val(r_vals,keys_y1,'keys_y1')
    fun.f1_make_r_val(r_vals,keys_y3,'keys_y3')
    fun.f1_make_r_val(r_vals,keys_x,'keys_x')

    ##other random r_vals
    fun.f1_make_r_val(r_vals,pinp.sheep['i_dse_type'],'dse_type')

    ##key lists used to form table headers and indexs
    keys_e = ['e%s'%i for i in range(len_e1)]
    keys_b = sinp.stock['i_lsln_idx_dams']
    keys_e0 = ['e%s'%i for i in range(len_e0)]
    keys_b0 = sinp.stock['i_btrt_idx_offs']
    keys_b9 = sinp.stock['i_lsln_idx_dams'][1:5]
    keys_o = np.array(['o%s'%i for i in range(len_o)])
    keys_r = ['prejoin', 'scan', 'birth', 'wean'] #used for the repro date report
    keys_s3 = np.array(['s%s'%i for i in range(len_s3)])
    keys_p = np.array(['p%s'%i for i in range(len_p)])
    keys_p3 = keys_p[mask_p_offs_p]

    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_z, keys_g0],'sire_keys_qszg0')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_p6, keys_f, keys_z, keys_g0],'sire_keys_qsp6fzg0')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1
                                             , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_p7, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1
                                             , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2p7tvanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_a, keys_e, keys_b9, keys_n1, keys_lw1
                                             , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvaeb9nwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                             , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvaebnwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_t1, keys_v1, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                             , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_tva1e1b1nwziyg1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_p, keys_a, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvpanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_p, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvpaebnwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_p[period_is_reportffcfw_p], keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2tvPaebnwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_p6, keys_f, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2p6ftvanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_p6, keys_f, keys_t1, keys_o, keys_v1, keys_a, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2p6ftovanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_T1, keys_p, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_Tpaebnwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_p, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_x, keys_y1, keys_g2],'yatf_keys_qsk2tvpaebnwzixy1g1')
    fun.f1_make_r_val(r_vals,[keys_T1, keys_v1, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_x, keys_y1, keys_g2],'yatf_keys_Tvaebnwzixy1g2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i
                                            , keys_a, keys_x, keys_g2],'prog_keys_qsk3k5twzia0xg2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i, keys_d
                                            , keys_a, keys_e0, keys_b0, keys_x, keys_y3, keys_g2],'prog_keys_qsk3k5twzida0e0b0xyg2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_p7, keys_t2, keys_lw_prog, keys_z, keys_i
                                            , keys_a, keys_x, keys_g2],'prog_keys_qsk3k5p7twzia0xg2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i
                                            , keys_a, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5tvnwziaxyg3')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_p7, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i
                                            , keys_a, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5p7tvnwziaxyg3')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t3, keys_v3, keys_p3, keys_n3, keys_lw3, keys_z
                                            , keys_i, keys_a, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5tvpnwziaxyg3')
    fun.f1_make_r_val(r_vals,[keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_d, keys_a, keys_e0
                                            , keys_b0, keys_x, keys_y3, keys_g3],'offs_keys_tvnwzida0e0b0xyg3')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t3, keys_v3, keys_p3, keys_n3, keys_lw3, keys_z
                                            , keys_i, keys_d, keys_a, keys_e0, keys_b0, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5tvpnwzidaebxyg3')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_p6, keys_f, keys_t3, keys_v3, keys_n3
                                            , keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5p6ftvnwziaxyg3')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_p6, keys_f, keys_t3, keys_s3, keys_v3, keys_n3
                                            , keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3],'offs_keys_qsk3k5p6ftsvnwziaxyg3')
    fun.f1_make_r_val(r_vals,[keys_T3, keys_p3, keys_n3, keys_lw3, keys_z, keys_i, keys_d, keys_a, keys_e0, keys_b0
                                            , keys_x, keys_y3, keys_g3],'offs_keys_Tpnwzidaebxyg3')
    fun.f1_make_r_val(r_vals,[keys_v3, keys_z, keys_d, keys_x, keys_g3],'offs_keys_vzdxg3')
    fun.f1_make_r_val(r_vals,[keys_v1, keys_e, keys_z, keys_g1],'dams_keys_vezg1')
    fun.f1_make_r_val(r_vals,[keys_r, keys_o, keys_e, keys_g1],'dams_keys_roeg1')


    ####std
    zg0_shape = len_z, len_g0
    k2tva1nwziyg1_shape = len_k2, len_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k5twzidaxyg2_shape = len_k5, len_t2, len_w_prog, len_z, len_i, len_d, len_a1, len_x, len_g2
    k3k5tvnwziaxyg3_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####std
    tva1e1b1nwziyg1_shape = len_t1, len_v1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    Tva1e1b1nwzixyg2_shape = len_gen_t1, len_v1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_x, len_y1, len_g2 #only has t axis from generator
    tvnwzidaebxyg3_shape = len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3

    ####kv with generator t axis
    k2Tva1nwziyg1_shape = len_k2, len_gen_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5Tvnwziaxyg3_shape = len_k3, len_k5, len_gen_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####kveb
    k2tva1e1b1nwziyg1_shape = len_k2, len_t1, len_v1, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1

    ####ktvPaeb
    k2TvPa1e1b1nwziyg1_shape = len_k2, len_gen_t1, len_v1, np.count_nonzero(period_is_reportffcfw_p), len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k2tvPa1e1b1nwziyg1_shape = len_k2, len_t1, len_v1, np.count_nonzero(period_is_reportffcfw_p), len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1

    ####kvpeb
    pzg0_shape = len_p, len_z, len_g0
    k2Tvpa1e1b1nwziyg1_shape = len_k2, len_gen_t1, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k2Tvpa1e1b1nwzixyg1_shape = len_k2, len_gen_t1, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_x, len_y1, len_g1
    k3k5wzida0e0b0xyg2_shape = len_k3, len_k5, len_w_prog, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y2,  len_g2
    k3k5Tvpnwzidae0b0xyg3_shape = len_k3, len_k5, len_gen_t3, len_v3, lenoffs_p, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3

    ####ktvpeb
    k2tvpa1e1b1nwziyg1_shape = len_k2, len_t1, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k2tvpa1e1b1nwzixyg1_shape = len_k2, len_t1, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_x, len_y1, len_g1
    k3k5tvpnwzidae0b0xyg3_shape = len_k3, len_k5, len_t3, len_v3, lenoffs_p, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3

    ####ktvp
    pzg0_shape = len_p, len_z, len_g0
    k2tvpa1nwziyg1_shape = len_k2, len_t1, len_v1, len_p, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5tvpnwziaxyg3_shape = len_k3, len_k5, len_t3, len_v3, lenoffs_p, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####p6
    p6zg0_shape = len_p6, len_z, len_g0
    k2p6tva1nwziyg1_shape = len_k2, len_p6, len_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5p6tvnwziaxyg3_shape = len_k3, len_k5, len_p6, len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####p6f
    p6fzg0_shape = len_p6, len_f, len_z, len_g0
    k2p6ftva1nwziyg1_shape = len_k2, len_p6, len_f, len_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5p6ftvnwziaxyg3_shape = len_k3, len_k5, len_p6, len_f, len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####p6fv0
    k2p6ftova1nwziyg1_shape = len_k2, len_p6, len_f, len_t1, len_o, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5p6ftsvnwziaxyg3_shape = len_k3, len_k5, len_p6, len_f, len_t3, len_s3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####cg
    p7zg0_shape = len_p7, len_z, len_g0
    k2p7tva1nwziyg1_shape = len_k2, len_p7, len_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5p7twziaxyg2_shape = len_k3, len_k5, len_p7, len_t2, len_w_prog, len_z, len_i, len_a1, len_x, len_g2
    k3k5p7tvnwziaxyg3_shape = len_k3, len_k5, len_p7, len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

    ####period dates
    roe1g1_shape = 4, len_o, len_e1, len_g1 #4 is the number of repro dates stored
    ve1zg1_shape = len_v1, len_e1, len_z, len_g1
    vzdxg3_shape = len_v3, len_z, len_d, len_x, len_g3

    ###z8 masks for unclustering lp_vars
    fun.f1_make_r_val(r_vals,mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,:,0,0,:,:,:,:,0,0,0,0,0,:,:],'maskz8_k2tvanwziy1g1') #slice off unused axis
    fun.f1_make_r_val(r_vals,mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,0,0,0,:,:,:,:,0,:,0,0,:,:,:],'maskz8_k3k5tvnwziaxyg3') #slice off unused axis

    ###dse
    fun.f1_make_r_val(r_vals,dsenw_p6tva1e1b1nwzida0e0b0xyg0,'dsenw_p6zg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg,z_pos, p6zg0_shape)
    fun.f1_make_r_val(r_vals,dsemj_p6tva1e1b1nwzida0e0b0xyg0,'dsemj_p6zg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg,z_pos, p6zg0_shape)
    fun.f1_make_r_val(r_vals,dsenw_k2p6tva1e1b1nwzida0e0b0xyg1,'dsenw_k2p6tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p6tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,dsemj_k2p6tva1e1b1nwzida0e0b0xyg1,'dsemj_k2p6tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p6tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,dsenw_k3k5p6tva1e1b1nwzida0e0b0xyg3,'dsenw_k3k5p6tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p6tvnwziaxyg3_shape)
    fun.f1_make_r_val(r_vals,dsemj_k3k5p6tva1e1b1nwzida0e0b0xyg3,'dsemj_k3k5p6tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p6tvnwziaxyg3_shape)
    fun.f1_make_r_val(r_vals,wg_propn_p6z,'wg_propn_p6z',mask_fp_z8var_p6z,z_pos=-1)

    ###stock days
    fun.f1_make_r_val(r_vals,stock_days_p6ftva1e1b1nwzida0e0b0xyg0,'stock_days_p6fzg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...],z_pos, p6fzg0_shape)
    fun.f1_make_r_val(r_vals,stock_days_k2p6ftva1e1b1nwzida0e0b0xyg1,'stock_days_k2p6ftva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],z_pos, k2p6ftva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,stock_days_k3k5p6ftva1e1b1nwzida0e0b0xyg3,'stock_days_k3k5p6ftvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],z_pos, k3k5p6ftvnwziaxyg3_shape)

    ###cashflow
    fun.f1_make_r_val(r_vals,cost_p7tva1e1b1nwzida0e0b0xyg0,'sire_cost_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,cost_k2p7tva1e1b1nwzida0e0b0xyg1,'dams_cost_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,cost_k3k5p7tva1e1b1nwzida0e0b0xyg3,'offs_cost_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,r_salevalue_p7tva1e1b1nwzida0e0b0xyg0,'salevalue_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,r_salevalue_k2p7tva1e1b1nwzida0e0b0xyg1,'salevalue_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_salevalue_prog_k3k5p7tva1e1b1nwzida0e0b0xyg2,'salevalue_k3k5p7twzia0xg2',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, k3k5p7twziaxyg2_shape)
    fun.f1_make_r_val(r_vals,r_salevalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'salevalue_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg0,'salegrid_zg0', shape=zg0_shape)
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg1,'salegrid_tva1e1b1nwziyg1', shape=tva1e1b1nwziyg1_shape) #didn't worry about unclustering since not important report and wasn't masked by z8
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg2,'salegrid_Tva1e1b1nwzixyg2', shape=Tva1e1b1nwzixyg2_shape) #didn't worry about unclustering since not important report and wasn't masked by z8
    fun.f1_make_r_val(r_vals,r_saleage_tva1e1b1nwzida0e0b0xyg3,'saleage_tvnwzida0e0b0xyg3', shape=tvnwzidaebxyg3_shape) #didn't worry about unclustering since not important report and wasn't masked by z8
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg3,'salegrid_tvnwzida0e0b0xyg3', shape=tvnwzidaebxyg3_shape) #didn't worry about unclustering since not important report and wasn't masked by z8

    fun.f1_make_r_val(r_vals,r_woolvalue_p7tva1e1b1nwzida0e0b0xyg0,'woolvalue_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,r_woolvalue_k2p7tva1e1b1nwzida0e0b0xyg1,'woolvalue_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_woolvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'woolvalue_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,rm_stockinfra_var_h1p7z,'rm_stockinfra_var_h1p7z',mask_season_p7z,z_pos=-1)
    fun.f1_make_r_val(r_vals,rm_stockinfra_fix_h1p7z,'rm_stockinfra_fix_h1p7z',mask_season_p7z,z_pos=-1)

    ###asset value used in pnl report to track changes in stock on hand because different z could sell at different time. e.g. if z0 retains but z3 sells z3 will look more profitable even though it is not.
    fun.f1_make_r_val(r_vals,start_assetvalue_p7tva1e1b1nwzida0e0b0xyg0,'assetvalue_startseason_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,end_assetvalue_p7tva1e1b1nwzida0e0b0xyg0,'assetvalue_endseason_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,start_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1,'assetvalue_startseason_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,end_assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1,'assetvalue_endseason_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,start_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'assetvalue_startseason_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)
    fun.f1_make_r_val(r_vals,end_assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'assetvalue_endseason_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    ###purchase costs
    fun.f1_make_r_val(r_vals,purchcost_p7tva1e1b1nwzida0e0b0xyg0,'purchcost_sire_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)

    ###sale date
    fun.f1_make_r_val(r_vals,r_saledate_k3k5tva1e1b1nwzida0e0b0xyg3,'saledate_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    ###dvp date
    r_repro_dates_roe1zg1 = np.stack([fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1, fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,
                                     fvp_birth_start_oa1e1b1nwzida0e0b0xyg1, fvp_wean_start_oa1e1b1nwzida0e0b0xyg1], axis=0)
    r_repro_dates_roe1g1 = fun.f_dynamic_slice(r_repro_dates_roe1zg1, axis=z_pos, start=0, stop=1) #remove z axis since repro dates don't change along z
    fun.f1_make_r_val(r_vals,dvp_start_va1e1b1nwzida0e0b0xyg1 % 364,'dvp_start_vezg1', shape=ve1zg1_shape) #mod 364 so that all dates are from the start of the yr (makes it easier to compare in the report)
    fun.f1_make_r_val(r_vals,dvp_start_va1e1b1nwzida0e0b0xyg3 % 364,'dvp_start_vzdxg3', shape=vzdxg3_shape) #mod 364 so that all dates are from the start of the yr (makes it easier to compare in the report)
    fun.f1_make_r_val(r_vals,r_repro_dates_roe1g1 % 364,'r_repro_dates_roe1g1', shape=roe1g1_shape) #mod 364 so that all dates are from the start of the yr (makes it easier to compare in the report)

    ###wbe - this uses generator t axis (thus it can be singleton but it is always broadcastable with normal t)
    fun.f1_make_r_val(r_vals,r_wbe_k2tva1e1b1nwzida0e0b0xyg1,'wbe_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2Tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_wbe_k3k5tva1e1b1nwzida0e0b0xyg3,'wbe_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5Tvnwziaxyg3_shape)

    ###cfw
    fun.f1_make_r_val(r_vals,r_cfw_hdmob_tva1e1b1nwzida0e0b0xyg0,'cfw_hdmob_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals,r_cfw_hdmob_k2tva1e1b1nwzida0e0b0xyg1,'cfw_hdmob_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_cfw_hdmob_k3k5tva1e1b1nwzida0e0b0xyg3,'cfw_hdmob_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,r_cfw_hd_tva1e1b1nwzida0e0b0xyg0,'cfw_hd_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals,r_cfw_hd_k2tva1e1b1nwzida0e0b0xyg1,'cfw_hd_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_cfw_hd_k3k5tva1e1b1nwzida0e0b0xyg3,'cfw_hd_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    ###fd
    fun.f1_make_r_val(r_vals,r_fd_hdmob_tva1e1b1nwzida0e0b0xyg0,'fd_hdmob_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals,r_fd_hdmob_k2tva1e1b1nwzida0e0b0xyg1,'fd_hdmob_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_fd_hdmob_k3k5tva1e1b1nwzida0e0b0xyg3,'fd_hdmob_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,r_fd_hd_tva1e1b1nwzida0e0b0xyg0,'fd_hd_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals,r_fd_hd_k2tva1e1b1nwzida0e0b0xyg1,'fd_hd_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_fd_hd_k3k5tva1e1b1nwzida0e0b0xyg3,'fd_hd_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    ###emissions
    fun.f1_make_r_val(r_vals, uinp.emissions['i_ch4_gwp_factor'], 'ch4_gwp_factor')
    fun.f1_make_r_val(r_vals, uinp.emissions['i_n2o_gwp_factor'], 'n2o_gwp_factor')
    fun.f1_make_r_val(r_vals, ch4_animal_tva1e1b1nwzida0e0b0xyg0, 'ch4_animal_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals, n2o_animal_tva1e1b1nwzida0e0b0xyg0, 'n2o_animal_zg0', shape=zg0_shape) #no mask needed since no active period axis
    fun.f1_make_r_val(r_vals, ch4_animal_k2tva1e1b1nwzida0e0b0xyg1, 'ch4_animal_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals, n2o_animal_k2tva1e1b1nwzida0e0b0xyg1, 'n2o_animal_k2tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals, ch4_animal_k3k5tva1e1b1nwzida0e0b0xyg3, 'ch4_animal_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)
    fun.f1_make_r_val(r_vals, n2o_animal_k3k5tva1e1b1nwzida0e0b0xyg3, 'n2o_animal_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

    ###mei and pi
    fun.f1_make_r_val(r_vals,mei_p6ftva1e1b1nwzida0e0b0xyg0,'mei_sire_p6fzg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...],z_pos, p6fzg0_shape)
    fun.f1_make_r_val(r_vals,mei_k2p6ftva1e1b1nwzida0e0b0xyg1,'mei_dams_k2p6ftva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],z_pos, k2p6ftva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3,'mei_offs_k3k5p6ftvnw8ziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],z_pos, k3k5p6ftvnwziaxyg3_shape)

    if sinp.rep['i_store_feedbud']:
        index_ova1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(len_o), p_pos-1)
        index_sva1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(len_s3), p_pos-1)

        ###create new stock days with an o axis.
        stock_days_p6ftova1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(
            on_hand_tpa1e1b1nwzida0e0b0xyg1[:, na, ...] * nv_propn_ftpdams[:, :, na, ...], a_v_pa1e1b1nwzida0e0b0xyg1
            , numbers_p=o_numbers_end_tpdams[:, na, ...], days_period_p=days_period_pa1e1b1nwzida0e0b0xyg1
            , a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:, na, :, na, ...]
            , a_any2_p=a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, index_any2any1tp = index_ova1e1b1nwzida0e0b0xyg1)
        stock_days_k2p6ftova1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',
                stock_days_p6ftova1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                index_k2tva1e1b1nwzida0e0b0xyg1[:, na, na, :, na, ...], numbers_start_vg = numbers_start_tva1e1b1nwzida0e0b0xyg1[:, na, ...],
                mask_vg = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
        fun.f1_make_r_val(r_vals,stock_days_k2p6ftova1e1b1nwzida0e0b0xyg1,'stock_days_k2p6ftova1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,:,na,...],z_pos, k2p6ftova1nwziyg1_shape)

        stock_days_p6ftsva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(on_hand_tpa1e1b1nwzida0e0b0xyg3[:, na, ...] * nv_propn_ftpoffs[:, :, na, ...], a_v_pa1e1b1nwzida0e0b0xyg3
                                                , numbers_p=o_numbers_end_tpoffs[:, na, ...], days_period_p=days_period_cut_pa1e1b1nwzida0e0b0xyg3,
                                                a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg[:,na, :, na, ...],
                                                a_any2_p=a_prev_s_pa1e1b1nwzida0e0b0xyg3, index_any2any1tp=index_sva1e1b1nwzida0e0b0xyg3)
        stock_days_k3k5p6ftsva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', stock_days_p6ftsva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,:,na,...],
                                                        a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,na,:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg3[:, na, ...],
                                                        mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)
        fun.f1_make_r_val(r_vals,stock_days_k3k5p6ftsva1e1b1nwzida0e0b0xyg3,'stock_days_k3k5p6ftsvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,:,na,...],z_pos, k3k5p6ftsvnwziaxyg3_shape)


        a_s_v_k3k5tsva1e1b1nwzida0e0b0xyg3 = (np.sum((a_prev_s_va1e1b1nwzida0e0b0xyg3==index_sva1e1b1nwzida0e0b0xyg3)
                * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...]), axis=d_pos, keepdims=True) > 0)
        r_mei_k2p6ftova1e1b1nwzida0e0b0xyg1 = mei_k2p6ftva1e1b1nwzida0e0b0xyg1[:,:,:,:,na,...] * (a_prev_o_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...]==index_ova1e1b1nwzida0e0b0xyg1) #take e[o]
        r_mei_k3k5p6ftsva1e1b1nwzida0e0b0xyg3 = mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3[:,:,:,:,:,na,...] * a_s_v_k3k5tsva1e1b1nwzida0e0b0xyg3[:,:,na,na,...]
        fun.f1_make_r_val(r_vals,r_mei_k2p6ftova1e1b1nwzida0e0b0xyg1,'mei_dams_k2p6ftova1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,:,na,...],z_pos, k2p6ftova1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_mei_k3k5p6ftsva1e1b1nwzida0e0b0xyg3,'mei_offs_k3k5p6ftsvnw8ziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,:,na,...],z_pos, k3k5p6ftsvnwziaxyg3_shape)


    fun.f1_make_r_val(r_vals,pi_p6ftva1e1b1nwzida0e0b0xyg0,'pi_sire_p6fzg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...],z_pos, p6fzg0_shape)
    fun.f1_make_r_val(r_vals,pi_k2p6ftva1e1b1nwzida0e0b0xyg1,'pi_dams_k2p6ftva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],z_pos, k2p6ftva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,pi_k3k5p6ftva1e1b1nwzida0e0b0xyg3,'pi_offs_k3k5p6ftvnw8ziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],z_pos, k3k5p6ftvnwziaxyg3_shape)


    ###proportion mated per dam at beginning of the period (e.g. accounts for mortality)
    fun.f1_make_r_val(r_vals,r_n_mated_k2tva1e1b1nwzida0e0b0xyg1,'n_mated_k2Tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2Tva1nwziyg1_shape)

    ###proportion of drys per dam at beginning of the period (e.g. accounts for mortality)
    fun.f1_make_r_val(r_vals,r_n_drys_k2tva1e1b1nwzida0e0b0xyg1,'n_drys_k2tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)

    ###number of foetuses scanned per dam at beginning of the period (e.g. accounts for mortality)
    fun.f1_make_r_val(r_vals,r_nfoet_scan_k2tva1e1b1nwzida0e0b0xyg1,'nfoet_scan_k2tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)

    ###wean percent
    fun.f1_make_r_val(r_vals,r_nyatf_wean_k2tva1e1b1nwzida0e0b0xyg1,'nyatf_wean_k2tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)

    ###nfoet and nyatf used to calc lamb survival and mortality
    fun.f1_make_r_val(r_vals,r_nfoet_birth_k2tva1e1b1nwzida0e0b0xyg1,'nfoet_birth_k2tva1e1b1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1e1b1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_nyatf_birth_k2tva1e1b1nwzida0e0b0xyg1,'nyatf_birth_k2tva1e1b1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1e1b1nwziyg1_shape)
    index_b9 = [0,1,2,3]
    nfoet_b1nwzida0e0b0xygb9 = nfoet_b1nwzida0e0b0xyg[...,na] == index_b9
    nyatf_b1nwzida0e0b0xygb9 = nyatf_b1nwzida0e0b0xyg[...,na] == index_b9
    fun.f1_make_r_val(r_vals,nfoet_b1nwzida0e0b0xygb9.squeeze(axis=(d_pos-1, a0_pos-1, e0_pos-1, b0_pos-1, x_pos-1)),'mask_b1b9_preg_b1nwziygb9')

    ###ffcfw with only a few p slices
    fun.f1_make_r_val(r_vals, r_ffcfw_dams_k2tvPdams, 'ffcfw_dams_k2tvPa1e1b1nw8ziyg1',
                      mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:, :, :, na, ...], z_pos, k2TvPa1e1b1nwziyg1_shape)

    ###mort - uses b axis instead of k for extra detail when scan=0
    if sinp.rep['i_store_mort']:
        ####note t axis will be singleton if generator was not run with t axis.
        fun.f1_make_r_val(r_vals,r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg1.squeeze(axis=(d_pos, a0_pos, e0_pos, b0_pos, x_pos)),'mort_Tpa1e1b1nwziyg1') #no unclustering because this wasn't masked by z8
        fun.f1_make_r_val(r_vals,r_cum_dvp_mort_tpa1e1b1nwzida0e0b0xyg3.squeeze(axis=(a1_pos, e1_pos, b1_pos)),'mort_Tpnwzida0e0b0xyg3') #no unclustering because this wasn't masked by z8

    ###on hand mort - proportion of each sheep remaining in each period after accounting for mort
    if sinp.rep['i_store_on_hand_mort']:
        fun.f1_make_r_val(r_vals,r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1,'on_hand_mort_k2tvpa1nwziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3,'on_hand_mort_k3k5tvpnwziaxyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5tvpnwziaxyg3_shape)

    ###numbers weights for reports with arrays that keep axis that are not present in lp array.
    if sinp.rep['i_store_lw_rep'] or sinp.rep['i_store_ffcfw_rep'] or sinp.rep['i_store_nv_rep']:

        ###weights the denominator and numerator - required for reports when p, e and b are added and weighted average is taken (otherwise broadcasting the variable activity to the new axis causes error in result)
        ###If these arrays get too big might have to add a second denom weight in reporting.
        pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1 = ((a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                                                      * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...])
                                                      * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...]
                                                      * o_numbers_start_tpdams[:,na,...]) #mul by numbers start to uncluster k axis.
        fun.f1_make_r_val(r_vals,pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1,'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1',
                          mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1e1b1nwziyg1_shape)

        ###for yatf a b1 weighting must be given
        pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1 = ((a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                                                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...])
                                                             * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...]
                                                             * o_numbers_start_tpyatf[:,na,...]* o_numbers_start_tpdams[:,na,...])#mul by numbers start to uncluster k axis. Need dam numbers to account for conception across e and b axis.
        fun.f1_make_r_val(r_vals,pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1,'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1',
                          mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1e1b1nwzixyg1_shape)

        pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3 = ((a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                                            * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                                            * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...])
                                                            * on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...]
                                                            * o_numbers_start_tpoffs[:,na,...])#mul by numbers start to uncluster k axis.
        fun.f1_make_r_val(r_vals,pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3,'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3',
                          mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5tvpnwzidae0b0xyg3_shape)

    ####make prog r_val for all trials. It is a small variable because it has singleton p axis
    de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2 = ((a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)
                                                         * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                                                         * numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2
                                                        ).squeeze(axis=(p_pos, a1_pos, e1_pos, b1_pos, n_pos)) #mul by numbers start to uncluster k axes.
    fun.f1_make_r_val(r_vals,de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2,'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2') #no mask because p axis to mask

    ###short p version
    Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1 = ((a_v_pa1e1b1nwzida0e0b0xyg1[period_is_reportffcfw_p, ...] == index_vpa1e1b1nwzida0e0b0xyg1)
                                                  * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...])
                                                  * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,period_is_reportffcfw_p,...]
                                                  * o_numbers_start_tpdams[:,na,period_is_reportffcfw_p,...])
    fun.f1_make_r_val(r_vals,Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1,'Pe1b1_numbers_weights_k2tvPa1e1b1nw8ziyg1',
                      mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvPa1e1b1nwziyg1_shape)


    ###lw - with p, e, b
    if sinp.rep['i_store_lw_rep']:
        fun.f1_make_r_val(r_vals,r_lw_sire_tpsire,'lw_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_lw_dams_k2Tvpdams,'lw_dams_k2Tvpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_lw_offs_k3k5Tvpoffs,'lw_offs_k3k5vpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5Tvpnwzidae0b0xyg3_shape)

    ###ffcfw - with p, e, b
    if sinp.rep['i_store_ffcfw_rep']:
        fun.f1_make_r_val(r_vals,r_ffcfw_sire_tpsire,'ffcfw_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_ffcfw_dams_k2Tvpdams,'ffcfw_dams_k2Tvpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_ffcfw_yatf_k2Tvpyatf,'ffcfw_yatf_k2Tvpa1e1b1nw8zixyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwzixyg1_shape)
        # fun.f1_make_r_val(r_vals,r_ffcfw_prog_k3k5tva1e1b1nwzida0e0b0xyg2,'ffcfw_prog_k3k5wzida0e0b0xyg2', shape=k3k5wzida0e0b0xyg2_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_ffcfw_offs_k3k5Tvpoffs,'ffcfw_offs_k3k5Tvpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5Tvpnwzidae0b0xyg3_shape)
    ####make prog r_val for all trials. It is a small variable because it has singleton p axis
    fun.f1_make_r_val(r_vals,r_ffcfw_prog_k3k5tva1e1b1nwzida0e0b0xyg2,'ffcfw_prog_k3k5wzida0e0b0xyg2', shape=k3k5wzida0e0b0xyg2_shape) #no v axis to mask

    ###NV - with p, e, b
    if sinp.rep['i_store_nv_rep']:
        fun.f1_make_r_val(r_vals,r_nv_sire_pg,'nv_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_nv_dams_k2Tvpg,'nv_dams_k2Tvpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_nv_offs_k3k5Tvpg,'nv_offs_k3k5Tvpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5Tvpnwzidae0b0xyg3_shape)

    ###condition score - with p, e, b
    if sinp.rep['i_store_cs_rep']:
        fun.f1_make_r_val(r_vals,r_cs_sire_pg,'cs_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_cs_dams_k2Tvpg,'cs_dams_k2Tvpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_cs_offs_k3k5Tvpg,'cs_offs_k3k5Tvpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5Tvpnwzidae0b0xyg3_shape)

    ###fat score - with p, e, b
    if sinp.rep['i_store_fs_rep']:
        fun.f1_make_r_val(r_vals,r_fs_sire_pg,'fs_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_fs_dams_k2Tvpg,'fs_dams_k2Tvpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2Tvpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_fs_offs_k3k5Tvpg,'fs_offs_k3k5Tvpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5Tvpnwzidae0b0xyg3_shape)




    # ###############
    # # season      # todo this can be removed with the new season structure
    # ###############
    # '''
    # stuff needed to allocate variable to stages for dsp
    # stored in params to be accessed in corepyomo when allocating variables to stages
    # '''
    # k2tva1nwiyg1_shape = len_k2, len_t1, len_v1, len_a1, len_n1, len_w1, len_i, len_y1, len_g1
    # k5twidaxg2_shape = len_k5, len_t2, len_w_prog, len_i, len_d, len_a1, len_x, len_g2
    # k3k5tvnwiaxyg3_shape = len_k3, len_k5, len_t3, len_v3, len_n3, len_w3, len_i, len_a0, len_x, len_y3, len_g3
    #
    # ##k2tvanwiyg1 - v_dams
    # arrays = [keys_k2, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1, keys_i, keys_y1, keys_g1]
    # index_k2tvanwiyg1 = fun.cartesian_product_simple_transpose(arrays)
    # tup_k2tvanwiyg1 = list(map(tuple,index_k2tvanwiyg1))
    # array_k2tvanwiyg1 = np.zeros(len(tup_k2tvanwiyg1),dtype=object)
    # array_k2tvanwiyg1[...] = tup_k2tvanwiyg1
    # array_k2tvanwiyg1 = array_k2tvanwiyg1.reshape(k2tva1nwiyg1_shape)
    # params['keys_v_dams'] = array_k2tvanwiyg1
    #
    # ##k2tvanwiyg1 - v_prog
    # arrays = [keys_k5, keys_t2, keys_lw_prog, keys_i, keys_d, keys_a, keys_x, keys_g2]
    # index_k5twidaxg2 = fun.cartesian_product_simple_transpose(arrays)
    # tup_k5twidaxg2 = list(map(tuple,index_k5twidaxg2))
    # array_k5twidaxg2 = np.zeros(len(tup_k5twidaxg2),dtype=object)
    # array_k5twidaxg2[...] = tup_k5twidaxg2
    # array_k5twidaxg2 = array_k5twidaxg2.reshape(k5twidaxg2_shape)
    # params['keys_v_prog'] = array_k5twidaxg2
    #
    # ##k3k5tvnwiaxyg3 - v_offs
    # arrays = [keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_i, keys_a, keys_x, keys_y3, keys_g3]
    # index_k3k5tvnwiaxyg3 = fun.cartesian_product_simple_transpose(arrays)
    # tup_k3k5tvnwiaxyg3 = list(map(tuple,index_k3k5tvnwiaxyg3))
    # array_k3k5tvnwiaxyg3 = np.zeros(len(tup_k3k5tvnwiaxyg3),dtype=object)
    # array_k3k5tvnwiaxyg3[...] = tup_k3k5tvnwiaxyg3
    # array_k3k5tvnwiaxyg3 = array_k3k5tvnwiaxyg3.reshape(k3k5tvnwiaxyg3_shape)
    # params['keys_v_offs'] = array_k3k5tvnwiaxyg3
    #
    # ##convert date array into the shape of stock variable.
    # dvp_date_k2tva1nwiyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1[na,na,:,:,0,0,:,:,0,:,0,0,0,0,0,:,:]
    # params['dvp1'] = np.broadcast_to(dvp_date_k2tva1nwiyg1, k2tva1nwiyg1_shape)
    # date_born_k5twidaxg2 = date_born_ida0e0b0xyg3[na,na,na,:,:,:,0,0,:,0,:] #average lamb along e slice
    # params['date_born_prog'] = np.broadcast_to(date_born_k5twidaxg2, k5twidaxg2_shape)
    # dvp_date_k3k5tvnwiaxyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3[na,na,na,:,0,0,0,:,:,0,:,0,:,0,0,:,:,:]
    # params['dvp3'] = np.broadcast_to(dvp_date_k3k5tvnwiaxyg3, k3k5tvnwiaxyg3_shape)
    #



    ##times - uncomment to report times.
    finish = time.time()
    # print('onhand and shearing arrays: ',calc_cost_start - onhandshear_start)
    # print('wool value calcs :', wool_finish - calc_cost_start)
    # print('sale value calcs :', sale_finish - wool_finish)
    # print('husb cost calcs :', husb_finish - sale_finish)
    # print('calc cost and income: ',feedpools_start - calc_cost_start)
    # print('feed pools arrays: ',p2v_start - feedpools_start)
    # print('p2v: ',lwdist_start - p2v_start)
    # print('lw distribution: ',cluster_start - lwdist_start)
    # print('clustering: ',allocation_start - cluster_start)
    # print('allocation: ',production_param_start - allocation_start)
    # print('p2v and building masks: ',lwdist_start - p2v_start)
    # print('production params: ', number_param_start - production_param_start)
    # print('number params: ', keys_start - number_param_start)
    # print('convert numpy to pyomo dict and reporting: ',finish - keys_start)

    ## Call Steve's graph generator.
    ## Will be bypassed unless called from SheepTest.py or line below is uncommented
    if plots:
        print('Interact with the graph generator using the PlotViewer spreadsheet, kill each plot to continue')
    scan_spreadsheet = plots   # argument passed to the StockGen function. True if called from SheepTest
    #    scan_spreadsheet = True    #make line active to generate plots when called from exp.py
    while scan_spreadsheet:
        try:
            yvar1, yvar2, xlabels, wvar, xvar, axes, dimensions, verticals = pv.read_spreadsheet()
            loc = locals()
            yvar1 = loc[yvar1]
            yvar2 = loc[yvar2]
            xlabels = loc[xlabels]
            wvar = loc[wvar]
            xvar = loc[xvar]
            pv.create_plots(yvar1, yvar2, xlabels, wvar, xvar, axes, dimensions, verticals)
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            scan_spreadsheet = input("Enter 1 to rescan spreadsheet for new plots: ")

    print('end of post processing in the generator')   # a line that can be used to break at the end of the generator
