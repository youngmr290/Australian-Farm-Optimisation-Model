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

import Functions as fun
import SeasonalFunctions as zfun
import Finance as fin
import FeedSupplyStock as fsstk
import FeedsupplyFunctions as fsfun
import Sensitivity as sen
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import StockFunctions as sfun
import Periods as per
import PlotViewer as pv
import Exceptions as exc


# np.seterr(all='raise')











# from memory_profiler import profile
# @profile
def generator(params,r_vals,nv,pkl_fs_info, plots = False):
    """
    A function to wrap the generator and post processing that can be called by SheepPyomo.

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
    ## define the periods - default (dams and sires)
    sim_years = sinp.stock['i_age_max']
    # sim_years = 4
    sim_years_offs = min(sinp.stock['i_age_max_offs'], sim_years)
    n_sim_periods, date_start_p, date_end_p, p_index_p, step \
        = sfun.f1_sim_periods(pinp.sheep['i_startyear'], sinp.stock['i_sim_periods_year'], sim_years)
    date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(p_pos+1, 0)))
    date_end_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_end_p, axis = tuple(range(p_pos+1, 0)))
    p_index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(p_index_p, axis = tuple(range(p_pos+1, 0)))
    ## define the periods - offs - these make the p axis customisable for offs which means they can be smaller
    n_sim_periods_offs, offs_date_start_p, offs_date_end_p, p_index_offs_p, step \
        = sfun.f1_sim_periods(pinp.sheep['i_startyear'], sinp.stock['i_sim_periods_year'], sim_years_offs)
    date_start_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_start_p, axis = tuple(range(p_pos+1, 0)))
    date_end_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_end_p, axis = tuple(range(p_pos+1, 0)))
    p_index_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(p_index_offs_p, axis = tuple(range(p_pos+1, 0)))
    mask_p_offs_p = p_index_p<=(n_sim_periods_offs-1)
    ##day of the year
    doy_pa1e1b1nwzida0e0b0xyg = (date_start_pa1e1b1nwzida0e0b0xyg - date_start_pa1e1b1nwzida0e0b0xyg.astype('datetime64[Y]')).astype(int) + 1 #plus one to include current day eg 7th - 1st = 6 plus 1 = 7th day of year
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
    mask_offs_inc_g3 = np.any(sinp.stock['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis =1)
    ##o/d mask - if dob is after the end of the sim then it is masked out -  the mask is created before the date of birth is adjusted to the start of a period however it is adjusted to the start of the next period so the mask won't cut out a birth event that actually would occur, additionally this is the birth of the first however the matrix sees the birth of average animal which is also later therefore if anything the mask will leave in unnecessary o slices
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = sfun.f1_g2g(pinp.sheep['i_date_born1st_oig2'],'yatf', i_pos, swap=True,left_pos2=p_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]')
    mask_o_dams = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_d_offs = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_x = pinp.sheep['i_gender_propn_x']>0
    bool_steady_state = pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1
    mask_node_is_fvp = pinp.general['i_node_is_fvp'] * (pinp.general['i_inc_node_periods']
                                                        or np.logical_not(bool_steady_state)) #node fvp/dvp are not included if it is steadystate.
    fvp_mask_dams = np.concatenate([mask_node_is_fvp[0:1], sinp.stock['i_fixed_fvp_mask_dams'], sinp.structuralsa['i_fvp_mask_dams'], mask_node_is_fvp[1:]]) #season start is at the front. because ss has to be first in the fvp/dvp
    fvp_mask_offs = np.concatenate([mask_node_is_fvp[0:1], sinp.structuralsa['i_fvp_mask_offs'], mask_node_is_fvp[1:]]) #season start is at the front. because ss has to be first in the fvp/dvp

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
    len_p1 = int(step / np.timedelta64(1, 'D')) #convert timedelta to float by dividing by one day
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
    n_fvp_periods_dams = np.count_nonzero(fvp_mask_dams)
    len_w1 = w_start_len1 * n_fs_dams ** n_fvp_periods_dams
    n_lw1_total = w_start_len1 * n_fs_dams ** (len(fvp_mask_dams))  # total lw if all dvps included
    len_w2 = len_w1 #yatf and dams are same
    len_nut_dams = (n_fs_dams ** n_fvp_periods_dams)

    ##offspring
    w_start_len3 = sinp.structuralsa['i_w_start_len3']
    n_fs_offs = sinp.structuralsa['i_n3_len']
    n_fvp_periods_offs= np.count_nonzero(fvp_mask_offs)
    len_w3 = w_start_len3 * n_fs_offs ** n_fvp_periods_offs
    n_lw3_total = w_start_len3 * n_fs_offs ** (len(fvp_mask_offs))  # total lw if all dvps included
    len_nut_offs = (n_fs_offs ** n_fvp_periods_offs)
    fvp0_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp0_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp1_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp1_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp2_offset_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_fvp2_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)

    ##season nodes - dvp must be added for each node (fvp is optional)
    date_node_zm = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'],numpy=True,axis=0).astype('datetime64') #have to use this rather than season periods because season periods get cut down in SE model but we still might want all the season nodes as dvp/fvp.
    date_node_zidaebxygm = fun.f_expand(date_node_zm, z_pos-1, right_pos=-1)
    len_m = date_node_zidaebxygm.shape[-1]

    ###################################
    ### index arrays                  #
    ###################################
    # index_p = np.arange(300)#asarray(300)
    index_a0e0b0xyg = fun.f_expand(np.arange(len_a1), a0_pos)
    index_a1e1b1nwzida0e0b0xyg = fun.f_expand(np.arange(len_a1), a1_pos)
    index_b0xyg = fun.f_expand(np.arange(len_b0), b0_pos)
    index_b1nwzida0e0b0xyg = fun.f_expand(np.arange(len_b1), b1_pos)
    index_l = np.arange(sinp.stock['i_len_l']) #gbal
    index_s = np.arange(sinp.stock['i_len_s']) #scan
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
    index_p1 = np.arange(len_p1)
    index_m0 = np.arange(12)*2  #2hourly steps for chill calculations
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
    index_tva1e1b1nw8zida0e0b0xyg3w9 = fun.f_expand(np.arange(len_t3), p_pos-2)
    index_xyg = fun.f_expand(np.arange(len_x), x_pos)

    prejoin_tup = (a1_pos, b1_pos, e1_pos)
    season_tup = (z_pos)


    ############################
    ### initialise arrays      #
    ############################
    '''only if assigned with a slice'''
    ##unique array shapes required to initialise arrays
    qg0 = (len_q0, len_q1, len_q2, len_p, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
    qg1 = (len_q0, len_q1, len_q2, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y1, len_g1)
    qg2 = (len_q0, len_q1, len_q2, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y2, len_g1)
    qg3 = (len_q0, len_q1, len_q2, lenoffs_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3)
    tpg0 = (1, len_p, 1, 1, 1, 1, 1, len_z, lensire_i, 1, 1, 1, 1, 1, 1, len_g0)
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


    ##output variables for postprocessing & reporting
    dtype='float32' #using 64 was getting slow
    dtypeint='int32' #using 64 was getting slow

    ##sire
    ###array for generator
    omer_history_start_p3g0 = np.zeros(p3g0, dtype = 'float64')
    d_cfw_history_start_p2g0 = np.zeros(p2g0, dtype = 'float64')
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 = np.zeros(c1tpg0, dtype =dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg0 = np.zeros(c1tpg0, dtype =dtype)
    ###arrays for postprocessing
    o_numbers_start_tpsire = np.zeros(tpg0, dtype =dtype)
    o_numbers_end_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ffcfw_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ffcfw_condensed_tpsire = np.zeros(tpg0, dtype =dtype)
    o_nw_start_tpsire = np.zeros(tpg0, dtype=dtype)
    o_lw_tpsire = np.zeros(tpg0, dtype =dtype)
    o_pi_tpsire = np.zeros(tpg0, dtype =dtype)
    o_mei_solid_tpsire = np.zeros(tpg0, dtype =dtype)
    o_ch4_total_tpsire = np.zeros(tpg0, dtype =dtype)
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
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 = np.zeros(c1tpg1, dtype =dtype)
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
    o_ch4_total_tpdams = np.zeros(tpg1, dtype =dtype)
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
    # o_ffcfw_condensed_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_pi_tpyatf = np.zeros(tpg2, dtype =dtype)
    o_mei_solid_tpyatf = np.zeros(tpg2, dtype =dtype)
    # o_ch4_total_tpyatf = np.zeros(tpg2, dtype =dtype)
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


    ##offs
    ###array for generator
    omer_history_start_p3g3 = np.zeros(p3g3, dtype = 'float64')
    d_cfw_history_start_p2g3 = np.zeros(p2g3, dtype = 'float64')
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg3 = np.zeros((len_c1,)+(len_t3,)+tpg3[1:], dtype =dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg3 = np.zeros(c1tpg3, dtype =dtype)
    ###array for postprocessing
    o_numbers_start_tpoffs = np.zeros(tpg3, dtype =dtype) # ones so that dvp0 (p0) has start numbers.
    o_numbers_end_tpoffs = np.zeros(tpg3, dtype =dtype) #ones so that transfer can exist for dvps before weaning
    o_ffcfw_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ffcfw_season_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ffcfw_condensed_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_nw_start_tpoffs = np.zeros(tpg3, dtype=dtype)
    o_mortality_offs = np.zeros(tpg3, dtype =dtype)
    o_lw_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_pi_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_mei_solid_tpoffs = np.zeros(tpg3, dtype =dtype)
    o_ch4_total_tpoffs = np.zeros(tpg3, dtype =dtype)
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



    ################################################
    #  management, age, date, timing inputs inputs #
    ################################################
    ##gender propn yatf
    gender_propn_xyg = fun.f_expand(pinp.sheep['i_gender_propn_x'], x_pos, condition=mask_x, axis=0).astype(dtype)
    ##join
    join_cycles_ida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_join_cycles_ig1'],'dams',i_pos)[pinp.sheep['i_mask_i'],...]

    ##lamb and lost
    gbal_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_gbal_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
    gbal_da0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_gbal_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied

    ##scanning
    scan_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_scan_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
    scan_da0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_scan_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied

    ##post weaning management
    wean_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_wean_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied

    ##association between offspring and sire/dam (used to determine the wean age of sire and dams based on the inputted wean age of offs)
    a_g0_g1 = sfun.f1_g2g(pinp.sheep['ia_g0_g1'],'dams')
    a_g3_g0 = sfun.f1_g2g(pinp.sheep['ia_g3_g0'],'sire')  # the sire association (pure bred B, M & T) are all based on purebred B because there are no pure bred M & T inputs
    a_g3_g1 = sfun.f1_g2g(pinp.sheep['ia_g3_g1'],'dams')  # if BMT exist then BBM exist and they will be in slice 1, therefore the association value doesn't need to be adjusted for "prior exclusions"

    ##age weaning- used to calc wean date and also to calc p1 stuff, sire and dams have no active a0 slice therefore just take the first slice
    ###note: if age_wean_g3 gets a d axis it need to be the same for all animals that get clustered (see date born below)
    age_wean1st_a0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_age_wean_a0g3'],'offs',a0_pos).astype('timedelta64[D]')[pinp.sheep['i_mask_a']]
    age_wean1st_e0b0xyg0 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g0],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
    age_wean1st_e0b0xyg1 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g1],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end

    ##date first lamb is born - need to apply i mask to these inputs - make sure animals are born at beginning of gen period
    date_born1st_ida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_born1st_ig0'],'sire',i_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos).astype('datetime64[D]')
    date_born1st_ida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_date_born1st_ig1'],'dams',i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2[mask_o_dams,...] #input read in in the mask section
    date_born1st_ida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_date_born1st_idg3'],'offs',d_pos, condition=pinp.sheep['i_mask_i']
                                           , axis=i_pos, condition2=mask_d_offs, axis2=d_pos).astype('datetime64[D]')
    date_born1st_ida0e0b0xyg3[:,len_k3-1:,...] = date_born1st_ida0e0b0xyg3[:,len_k3-1,...] #for animals in the same d cluster date born must be the same (so that the dvp and fvp dates are the same for all animals that get clustered)

    ##mating
    sire_propn_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_sire_propn_oig1'],'dams', i_pos, swap=True,
                                                   left_pos2=p_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'],
                                                   axis=i_pos, condition2=mask_o_dams, axis2=p_pos)
    sire_periods_p8g0 = sfun.f1_g2g(pinp.sheep['i_sire_periods_p8g0'], 'sire', condition=pinp.sheep['i_mask_p8'], axis=0)
    sire_periods_g0p8 = np.swapaxes(sire_periods_p8g0, 0, 1) #can't swap in function above because g needs to be in pos-1

    ##propn of dams mated - default is inf which gets skipped in the bound constraint hence the model can optimise the propn mated.
    prop_dams_mated_og1 = fun.f_sa(np.array([999],dtype=float), sen.sav['bnd_propn_dams_mated_og1'], 5) #999 just an arbitrary value used then converted to np.inf because np.inf causes errors in the f_update which is called by f_sa
    prop_dams_mated_og1[prop_dams_mated_og1==999] = np.inf
    prop_dams_mated_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(prop_dams_mated_og1, left_pos=p_pos, right_pos=-1,
                                                          condition=mask_o_dams, axis=p_pos, condition2=mask_dams_inc_g1, axis2=-1)

    ##Shearing date - set to be on the last day of a sim period
    ###sire
    date_shear_sida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg0'], 'sire', x_pos, swap=True
                                          ,left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos
                                          )[...,0:1,:,:].astype('datetime64[D]') #slice x axis for only male
    mask_shear_g0 = np.max(date_shear_sida0e0b0xyg0<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg0 = date_shear_sida0e0b0xyg0[mask_shear_g0]
    ###dam
    date_shear_sida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg1'],'dams',x_pos ,swap=True,left_pos2=i_pos,right_pos2=x_pos,
                                          condition=pinp.sheep['i_mask_i'], axis=i_pos
                                          )[...,1:2,:,:].astype('datetime64[D]') #slice x axis for only female
    mask_shear_g1 = np.max(date_shear_sida0e0b0xyg1<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg1 = date_shear_sida0e0b0xyg1[mask_shear_g1]
    ###off - the first shearing must occur as offspring because if yatf were shorn then all lambs would have to be shorn (ie no scope to not shear the lambs that are going to be fed up and sold)
    #### the offspring decision variables are not linked to the yatf (which are in the dam decision variables) and it would require doubling the dam DVs to have shorn and unshorn yatf
    ####note: if age_wean_g3 gets a d axis it need to be the same for all animals that get clustered (see date born below)
    date_shear_sida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_date_shear_sixg3'],'offs',x_pos,swap=True,left_pos2=i_pos,right_pos2=x_pos,
                                          condition=pinp.sheep['i_mask_i'], axis=i_pos
                                          , condition2=mask_x, axis2=x_pos).astype('datetime64[D]')
    mask_shear_g3 = np.max(date_shear_sida0e0b0xyg3<=offs_date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg3 = date_shear_sida0e0b0xyg3[mask_shear_g3]


    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from c to g
    ###production params
    agedam_propn_da0e0b0xyg0, agedam_propn_da0e0b0xyg1, agedam_propn_da0e0b0xyg2, agedam_propn_da0e0b0xyg3 = sfun.f1_c2g(uinp.parameters['i_agedam_propn_std_dc2'], uinp.parameters['i_agedam_propn_y'], uinp.parameters['i_agedam_propn_pos'], condition=mask_o_dams, axis=d_pos) #yatf and off never used
    agedam_propn_da0e0b0xyg0 = agedam_propn_da0e0b0xyg0 / np.sum(agedam_propn_da0e0b0xyg0, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg1 = agedam_propn_da0e0b0xyg1 / np.sum(agedam_propn_da0e0b0xyg1, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg2 = agedam_propn_da0e0b0xyg2 / np.sum(agedam_propn_da0e0b0xyg2, axis=d_pos) #scale unmasked slices to a total of 1
    agedam_propn_da0e0b0xyg3 = agedam_propn_da0e0b0xyg3 / np.sum(agedam_propn_da0e0b0xyg3, axis=d_pos) #scale unmasked slices to a total of 1
    aw_propn_yg0, aw_propn_yg1, aw_propn_yg2, aw_propn_yg3 = sfun.f1_c2g(uinp.parameters['i_aw_propn_wean_c2'], uinp.parameters['i_aw_wean_y'])
    bw_propn_yg0, bw_propn_yg1, bw_propn_yg2, bw_propn_yg3 = sfun.f1_c2g(uinp.parameters['i_bw_propn_wean_c2'], uinp.parameters['i_bw_wean_y'])
    cfw_propn_yg0, cfw_propn_yg1, cfw_propn_yg2, cfw_propn_yg3 = sfun.f1_c2g(uinp.parameters['i_cfw_propn_c2'], uinp.parameters['i_cfw_propn_y'])
    fl_birth_yg0, fl_birth_yg1, fl_birth_yg2, fl_birth_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_birth_c2'], uinp.parameters['i_fl_birth_y'])
    fl_shear_yg0, fl_shear_yg1, fl_shear_yg2, fl_shear_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_shear_c2'], uinp.parameters['i_fl_shear_y'])
    mw_propn_yg0, mw_propn_yg1, mw_propn_yg2, mw_propn_yg3 = sfun.f1_c2g(uinp.parameters['i_mw_propn_wean_c2'], uinp.parameters['i_mw_wean_y'])
    pss_std_yg0, pss_std_yg1, pss_std_yg2, pss_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lss_std_c2'], uinp.parameters['i_lss_std_y'])
    pstr_std_yg0, pstr_std_yg1, pstr_std_yg2, pstr_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lstr_std_c2'], uinp.parameters['i_lstr_std_y'])
    pstw_std_yg0, pstw_std_yg1, pstw_std_yg2, pstw_std_yg3 = sfun.f1_c2g(uinp.parameters['i_lstw_std_c2'], uinp.parameters['i_lstw_std_y'])
    scan_std_yg0, scan_std_yg1, scan_std_yg2, scan_std_yg3 = sfun.f1_c2g(uinp.parameters['i_scan_std_c2'], uinp.parameters['i_scan_std_y']) #scan_std_yg2/3 not used
    ###scan_std could change across the i axis, however, there is a tradeoff between LW at joining and time in the breeding season so assume these cancel out rather than adjusting by crg_doy here
    scan_dams_std_yg3 = scan_std_yg1 #offs needs to be the same as dams because scan_std is used to calc starting propn of BTRT which is dependant on dams scanning
    sfd_yg0, sfd_yg1, sfd_yg2, sfd_yg3 = sfun.f1_c2g(uinp.parameters['i_sfd_c2'], uinp.parameters['i_sfd_y'])
    sfw_yg0, sfw_yg1, sfw_yg2, sfw_yg3 = sfun.f1_c2g(uinp.parameters['i_sfw_c2'], uinp.parameters['i_sfw_y'])
    srw_female_yg0, srw_female_yg1, srw_female_yg2, srw_female_yg3 = sfun.f1_c2g(uinp.parameters['i_srw_c2'], uinp.parameters['i_srw_y']) #srw of a female of the given genotype (this is the definition of the inputs)

    ###p1 variation params (used for mort)
    cv_weight_sire, cv_weight_dams, cv_weight_yatf, cv_weight_offs = sfun.f1_c2g(uinp.parameters['i_cv_weight_c2'], uinp.parameters['i_cv_weight_y'])
    cv_cs_sire, cv_cs_dams, cv_cs_yatf, cv_cs_offs = sfun.f1_c2g(uinp.parameters['i_cv_cs_c2'], uinp.parameters['i_cv_cs_y'])
    sd_ebg_sire, sd_ebg_dams, sd_ebg_yatf, sd_ebg_offs = sfun.f1_c2g(uinp.parameters['i_sd_ebg_c2'], uinp.parameters['i_sd_ebg_y'])

    ###sim params
    ca_sire, ca_dams, ca_yatf, ca_offs = sfun.f1_c2g(uinp.parameters['i_ca_c2'], uinp.parameters['i_ca_y'], uinp.parameters['i_ca_pos'])
    cb0_sire, cb0_dams, cb0_yatf, cb0_offs = sfun.f1_c2g(uinp.parameters['i_cb0_c2'], uinp.parameters['i_cb0_y'], uinp.parameters['i_cb0_pos'])
    cc_sire, cc_dams, cc_yatf, cc_offs = sfun.f1_c2g(uinp.parameters['i_cc_c2'], uinp.parameters['i_cc_y'], uinp.parameters['i_cc_pos'])
    cd_sire, cd_dams, cd_yatf, cd_offs = sfun.f1_c2g(uinp.parameters['i_cd_c2'], uinp.parameters['i_cd_y'], uinp.parameters['i_cd_pos'])
    ce_sire, ce_dams, ce_yatf, ce_offs = sfun.f1_c2g(uinp.parameters['i_ce_c2'], uinp.parameters['i_ce_y'], uinp.parameters['i_ce_pos'], condition=mask_o_dams, axis=d_pos)
    ce_offs = sfun.f1_c2g(uinp.parameters['i_ce_c2'], uinp.parameters['i_ce_y'], uinp.parameters['i_ce_pos'], condition=mask_d_offs, axis=d_pos)[3]  #re calc off using off d mask
    cf_sire, cf_dams, cf_yatf, cf_offs = sfun.f1_c2g(uinp.parameters['i_cf_c2'], uinp.parameters['i_cf_y'], uinp.parameters['i_cf_pos'])
    cg_sire, cg_dams, cg_yatf, cg_offs = sfun.f1_c2g(uinp.parameters['i_cg_c2'], uinp.parameters['i_cg_y'], uinp.parameters['i_cg_pos'])
    ch_sire, ch_dams, ch_yatf, ch_offs = sfun.f1_c2g(uinp.parameters['i_ch_c2'], uinp.parameters['i_ch_y'], uinp.parameters['i_ch_pos'])
    ci_sire, ci_dams, ci_yatf, ci_offs = sfun.f1_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'])
    ck_sire, ck_dams, ck_yatf, ck_offs = sfun.f1_c2g(uinp.parameters['i_ck_c2'], uinp.parameters['i_ck_y'], uinp.parameters['i_ck_pos'])
    cl0_sire, cl0_dams, cl0_yatf, cl0_offs = sfun.f1_c2g(uinp.parameters['i_cl0_c2'], uinp.parameters['i_cl0_y'], uinp.parameters['i_cl0_pos'])
    cl1_sire, cl1_dams, cl1_yatf, cl1_offs = sfun.f1_c2g(uinp.parameters['i_cl1_c2'], uinp.parameters['i_cl1_y'], uinp.parameters['i_cl1_pos'])
    cl_sire, cl_dams, cl_yatf, cl_offs = sfun.f1_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'])
    cm_sire, cm_dams, cm_yatf, cm_offs = sfun.f1_c2g(uinp.parameters['i_cm_c2'], uinp.parameters['i_cm_y'], uinp.parameters['i_cm_pos'])
    cn_sire, cn_dams, cn_yatf, cn_offs = sfun.f1_c2g(uinp.parameters['i_cn_c2'], uinp.parameters['i_cn_y'], uinp.parameters['i_cn_pos'])
    cp_sire, cp_dams, cp_yatf, cp_offs = sfun.f1_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'])
    cr_sire, cr_dams, cr_yatf, cr_offs = sfun.f1_c2g(uinp.parameters['i_cr_c2'], uinp.parameters['i_cr_y'], uinp.parameters['i_cr_pos'])
    crd_sire, crd_dams, crd_yatf, crd_offs = sfun.f1_c2g(uinp.parameters['i_crd_c2'], uinp.parameters['i_crd_y'], uinp.parameters['i_crd_pos'])
    cu0_sire, cu0_dams, cu0_yatf, cu0_offs = sfun.f1_c2g(uinp.parameters['i_cu0_c2'], uinp.parameters['i_cu0_y'], uinp.parameters['i_cu0_pos'])
    cu1_sire, cu1_dams, cu1_yatf, cu1_offs = sfun.f1_c2g(uinp.parameters['i_cu1_c2'], uinp.parameters['i_cu1_y'], uinp.parameters['i_cu1_pos'])
    cu2_sire, cu2_dams, cu2_yatf, cu2_offs = sfun.f1_c2g(uinp.parameters['i_cu2_c2'], uinp.parameters['i_cu2_y'], uinp.parameters['i_cu2_pos'])
    cw_sire, cw_dams, cw_yatf, cw_offs = sfun.f1_c2g(uinp.parameters['i_cw_c2'], uinp.parameters['i_cw_y'], uinp.parameters['i_cw_pos'])
    cx_sire, cx_dams, cx_yatf, cx_offs = sfun.f1_c2g(uinp.parameters['i_cx_c2'], uinp.parameters['i_cx_y'], uinp.parameters['i_cx_pos'])
    ##Convert the cl0 & cl1 to cb1 (dams and yatf only need cb1, sires and offs don't have b1 axis)
    cb1_dams = cl0_dams[:,sinp.stock['a_nfoet_b1']] + cl1_dams[:,sinp.stock['a_nyatf_b1']]
    cb1_yatf = cl0_yatf[:,sinp.stock['a_nfoet_b1']] + cl1_yatf[:,sinp.stock['a_nyatf_b1']]
    ###Alter select slices only for yatf (yatf don't have cb0 axis - instead they use cb1 so it aligns with dams)
    ###The b1 parameters that are relevant to the dams relate to either number of foetus (entered as cl0) or number of yatf (entered as cl1). However, because the yatf also use the b1 axis some parameters that change based on the combination of birth type and rear type (BTRT - b0) are needed in the b1 axis for the yatf.
    ###The only role for these parameters is for estimating values for the yatf
    cb1_yatf[12, ...] = np.expand_dims(cb0_yatf[12, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[13, ...] = np.expand_dims(cb0_yatf[13, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[17, ...] = np.expand_dims(cb0_yatf[17, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1
    cb1_yatf[18, ...] = np.expand_dims(cb0_yatf[18, sinp.stock['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array aligns with b1

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
    ##wind speed
    #todo add a distribution to the windspeed (after checking the importance for chill_index)
    #might need to do this with a longer axis length so that it is not the distribution in the week but in the month
    #enter a number of days above a threshold (along with the threshold values maybe 1) and then average values for the windiest days in the month.
    ws_p4a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_ws_p4'],p_pos)
    ##expected stocking density
    density_p6a1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_density_p6z'], z_pos, move=True, source=0, dest=-1,
                                                  left_pos2=p_pos, right_pos2=z_pos)  # p6 axis converted to p axis later (association section)
    density_p6a1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(density_p6a1e1b1nwzida0e0b0xyg,numpy=True,axis=z_pos).astype(
        int)
    ##nutrition adjustment for expected stocking density
    density_nwzida0e0b0xyg1 = fun.f_expand(sinp.structuralsa['i_density_n1'][0:n_fs_dams],n_pos) # cut to the correct length based on number of nutrition options (i_len_n structural input)
    density_nwzida0e0b0xyg3 = fun.f_expand(sinp.structuralsa['i_density_n3'][0:n_fs_offs],n_pos) # cut to the correct length based on number of nutrition options (i_len_n structural input)
    ##Calculation of rainfall distribution across the week - i_rain_distribution_p4p1 = how much rain falls on each day of the week sorted in order of quantity of rain. SO the most rain falls on the day with the highest rainfall.
    rain_p4a1e1b1nwzida0e0b0xygp1 = fun.f_expand(
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
    date_born_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_p, date_born_ida0e0b0xyg0, 0, 'left')
    ###dams
    date_born_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg1 = fun.f_next_prev_association(date_start_p,date_born_ida0e0b0xyg1,0,'left')
    ###yatf - needs to be rounded to start of gen period because this controls the start of a dvp
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle. e_index is to account for ewe cycles.
    date_born_idx_oa1e1b1nwzida0e0b0xyg2 =fun.f_next_prev_association(date_start_p, date_born_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_born_idx_oa1e1b1nwzida0e0b0xyg2]
    ###offs
    date_born_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + (index_e0b0xyg + 0.5) * cf_offs[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_p, date_born_ida0e0b0xyg3, 0, 'left')
    ##recalc date_born1st from the adjusted average birth date
    date_born1st_ida0e0b0xyg0 = date_born_ida0e0b0xyg0 - 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')
    date_born1st_ida0e0b0xyg1 = date_born_ida0e0b0xyg1 - 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born_oa1e1b1nwzida0e0b0xyg2[:,:,0:1,...] - 0.5 * cf_yatf[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis
    date_born1st_ida0e0b0xyg3 = date_born_ida0e0b0xyg3[:,:,:,0:1,...] - 0.5 * cf_offs[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis

    ##calc wean date (weaning input is counting from the date of the first lamb)
    date_weaned_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + age_wean1st_e0b0xyg0
    date_weaned_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + age_wean1st_e0b0xyg1
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + age_wean1st_a0e0b0xyg3 #use off wean age
    date_weaned_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + age_wean1st_a0e0b0xyg3
    ###adjust weaning to occur at the beginning of generator period and recalc wean age
    ####sire
    date_weaned_idx_ida0e0b0xyg0=fun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg0, 0, 'left')
    date_weaned_ida0e0b0xyg0 = date_start_p[date_weaned_idx_ida0e0b0xyg0]
    age_wean1st_ida0e0b0xyg0 = date_weaned_ida0e0b0xyg0 - date_born1st_ida0e0b0xyg0
    ####dams
    date_weaned_idx_ida0e0b0xyg1=fun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg1, 0, 'left')
    date_weaned_ida0e0b0xyg1 = date_start_p[date_weaned_idx_ida0e0b0xyg1]
    age_wean1st_ida0e0b0xyg1 = date_weaned_ida0e0b0xyg1 -date_born1st_ida0e0b0xyg1
    ####yatf - this is the same as offs except without the offs periods (offs potentially have less periods)
    date_weaned_idx_oa1e1b1nwzida0e0b0xyg2=fun.f_next_prev_association(date_start_p, date_weaned_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_weaned_idx_oa1e1b1nwzida0e0b0xyg2]
    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = date_weaned_oa1e1b1nwzida0e0b0xyg2 - date_born1st_oa1e1b1nwzida0e0b0xyg2
    ####offs
    date_weaned_idx_ida0e0b0xyg3=fun.f_next_prev_association(offs_date_start_p, date_weaned_ida0e0b0xyg3, 0, 'left')
    date_weaned_ida0e0b0xyg3 = offs_date_start_p[date_weaned_idx_ida0e0b0xyg3]
    age_wean1st_ida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 - date_born1st_ida0e0b0xyg3

    ##Shearing date - set to be on the last day of a sim period
    ###sire
    idx_sida0e0b0xyg0 = fun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg0,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg0 = fun.f_expand(date_end_p[idx_sida0e0b0xyg0], p_pos, right_pos=i_pos)
    ###dam
    idx_sida0e0b0xyg1 = fun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg1,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg1 = fun.f_expand(date_end_p[idx_sida0e0b0xyg1], p_pos, right_pos=i_pos)
    ###off - shearing can't occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = np.maximum(date_born1st_ida0e0b0xyg3 + age_wean1st_ida0e0b0xyg3, date_shear_sida0e0b0xyg3) #shearing must be after weaning. This makes the d axis active because weaning has an active d.
    idx_sida0e0b0xyg3 = fun.f_next_prev_association(offs_date_end_p, date_shear_sida0e0b0xyg3,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg3 = fun.f_expand(offs_date_end_p[idx_sida0e0b0xyg3], p_pos, right_pos=i_pos)


    ############################
    ## calc for associations   #
    ############################
    ##date joined (when the rams go in)
    date_joined_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 - cp_dams[1,...,0:1,:].astype('timedelta64[D]') #take slice 0 from y axis because cp1 is not affected by genetic merit
    ##expand feed periods over all the years of the sim so that an association between sim period can be made.
    ##set fp to start at the next generator period following the node (needs to be next so that clustering works). Lp are adjusted so that they get clustered the same as dvps
    feedperiods_p6z = per.f_feed_periods().astype('datetime64[D]')[:-1] #remove last date because that is the end date of the last period (not required)
    feedperiods_p6z = feedperiods_p6z + np.timedelta64(365,'D') * ((date_start_p[0].astype('datetime64[Y]').astype(int) + 1970 -1) - (feedperiods_p6z[0].astype('datetime64[Y]').astype(int) + 1970)) #this is to make sure the first sim period date is greater than the first feed period date.
    feedperiods_p6z = (feedperiods_p6z  + (np.arange(np.ceil(sim_years +1)) * np.timedelta64(365,'D') )[...,na,na]).reshape((-1, len_z)) #expand then ravel to return 1d array of the feed period dates expanded the length of the sim. +1 because feed periods start and finish mid yr so add one to ensure they go to the end of the sim.
    feedperiods_idx_p6z = np.minimum(len(date_start_p) - 1,np.searchsorted(date_start_p,feedperiods_p6z,'left'))  # maximum idx is the number of generator periods
    feedperiods_p6z = date_start_p[feedperiods_idx_p6z]



    ##################
    # FVP background #
    ##################
    '''
    FVPs have gotten a little bit complex. To ensure that weights, masks, transfers and distributions occur correctly there are some rules that need to be met.
    The DVP rules are documented below in the dvp section.
    
    FVPs before weaning (ie while numbers are all 0) are removed if they occur across all axis. If they dont occur
    across all axis then they are set to the date of weaning. If multiple fvps occur at weaning they get off set by 1 
    period. Type is set to extra so nothing is triggered eg if season start is before weaning and it gets moved to weaing
    we dont want to trigger a distribution.
    
    FVPs that occur after the end of the generator are handled the same as above. I.e they are removed if the same across 
    all axis otherwise they are set to the last period of the generator. If multiple fvps occur on the last period they are offset 
    by 1 peirod (this doesnt need to happen it just does so that the fvp no clash test is passed but fvps clashing on the
    last gen peirod would actually be fine).
    
    No FVPs can clash. If an fvp clashes with another fvp then it doesnt trigger a fvp_start therefore a 0 day fvp 
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
    condense_vtype1 = prejoin_vtype1 #currently for dams condensing must occur at prejoining, most of the code is flexible to handle different timing except the lw_distribution section.
    scan_vtype1 = core_dvp_types_f1[1]
    birth_vtype1 = core_dvp_types_f1[2]
    season_vtype1 = max(core_dvp_types_f1) + 1
    other_vtype1 = max(core_dvp_types_f1) + 2

    ##beginning - first day of generator
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg[0:1]

    ##early pregnancy fvp start - The pre-joining accumulation of the dams from the previous reproduction cycle - this date must correspond to the start date of period
    prejoining_approx_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 - sinp.stock['i_prejoin_offset'] #approx date of prejoining - in the next line of code prejoin date is adjusted to be the start of a sim period in which the approx date falls
    idx = np.searchsorted(date_start_p, prejoining_approx_oa1e1b1nwzida0e0b0xyg1, 'right') - 1 #gets the sim period index for the period that prejoining occurs (eg prejoining fvp starts at the beginning of the sim period when prejoining approx occurs), side=right so that if the date is already the start of a period it remains in that period.
    prejoining_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx]
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = prejoining_oa1e1b1nwzida0e0b0xyg1

    ##late pregnancy fvp start - Scanning if carried out, day 90 from joining (ram in) if not scanned.
    late_preg_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') + pinp.sheep['i_scan_day'][scan_oa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]')
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, late_preg_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when dams in late preg (eg late preg fvp starts at the beginning of the sim period when late preg occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx_oa1e1b1nwzida0e0b0xyg]

    ## lactation fvp start - average date of lambing (with e axis if scanning/managing e differentially) (already adjusted to start of gen period)
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = date_born_oa1e1b1nwzida0e0b0xyg2.copy()
    ### birth fvp/dvp must be the same when e axis is clustered (otherwise something goes wrong in the pp/matrix).
    t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.copy()
    t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[...] = fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[:,:,-1:,...] #needs to be the final e slice so that all e slices have lambed when new dvp starts.
    e_fvp_mask = (scan_oa1e1b1nwzida0e0b0xyg1 < 4)  #mask with true when fvp/dvp date should be the same along the e axis
    e_fvp_mask = np.broadcast_to(e_fvp_mask, fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape)
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[e_fvp_mask] = t_fvp_birth_start_oa1e1b1nwzida0e0b0xyg1[e_fvp_mask]

    ##weaning (already adjusted to start of gen period)
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = date_weaned_oa1e1b1nwzida0e0b0xyg2

    ##user defined fvp - rounded to nearest sim period
    fvp_other_yi = sinp.structuralsa['i_fvp4_date_i'].astype(np.datetime64) + np.arange(np.ceil(sim_years))[:,na] * np.timedelta64(365,'D')
    fvp_other_ya1e1b1nwzida0e0b0xyg = fun.f_expand(fvp_other_yi, left_pos=i_pos, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, fvp_other_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (eg break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]
    ##season nodes - these get masked out if steady state.
    node_fvp_m = np.zeros(len_m, dtype=object)
    for m in range(len_m):
        date_node_zidaebxyg = date_node_zidaebxygm[...,m]
        date_node_ya1e1b1nwzidaebxyg = date_node_zidaebxyg.astype(np.datetime64) + fun.f_expand(np.arange(np.ceil(sim_years)) * np.timedelta64(365,'D'), p_pos)
        ###set the node fvp to start at the next generator period following the node (needs to be next so that clustering works).
        idx_ya1e1b1nwzida0e0b0xyg = np.minimum(len(date_start_p)-1, np.searchsorted(date_start_p, date_node_ya1e1b1nwzidaebxyg, 'left')) #maximum idx is the number of generator periods
        date_node_ya1e1b1nwzidaebxyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]
        node_fvp_m[m] = date_node_ya1e1b1nwzidaebxyg
    ###store season start date - used to determine period_is_season_start
    seasonstart_ya1e1b1nwzida0e0b0xyg = node_fvp_m[0]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[1:],
                               date_node_ya1e1b1nwzidaebxyg.shape[1:]]) #create shape which has the max size, this is used for o array

    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = np.broadcast_to(fvp_other_start_ya1e1b1nwzida0e0b0xyg,(fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape))
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    for m in range(len_m):
        node_fvp_m[m] = np.broadcast_to(node_fvp_m[m],(node_fvp_m[m].shape[0],)+tuple(shape))

    ##create fvp type arrays. these are the same shape as the start arrays and are filled with the number corresponding to the fvp number
    fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape,prejoin_vtype1)
    fvp_scan_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape, scan_vtype1)
    fvp_birth_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape, birth_vtype1)
    fvp_wean_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape, other_vtype1)
    fvp_other_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape, other_vtype1)
    fvp_begin_type_va1e1b1nwzida0e0b0xyg1 = np.full(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape, condense_vtype1)
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
                                fvp_wean_start_oa1e1b1nwzida0e0b0xyg1, fvp_other_start_ya1e1b1nwzida0e0b0xyg], dtype=object)
    fvp_date_all_f1 = np.concatenate([node_fvp_m[0:1], fvp_date_all_f1, node_fvp_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp_type_all_f1 = np.array([fvp_begin_type_va1e1b1nwzida0e0b0xyg1, fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1,
                                fvp_scan_type_va1e1b1nwzida0e0b0xyg1, fvp_birth_type_va1e1b1nwzida0e0b0xyg1,
                                fvp_wean_type_va1e1b1nwzida0e0b0xyg1, fvp_other_type_va1e1b1nwzida0e0b0xyg1], dtype=object)
    fvp_type_all_f1 = np.concatenate([node_fvp_type_m[0:1], fvp_type_all_f1, node_fvp_type_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp1_inc = np.concatenate([fvp_mask_dams[0:1], np.array([True]), fvp_mask_dams[1:]]) #True in the middle is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
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
    condense_vtype3 = 0 #for offs condensing can occur at any dvp. Currently it occurs at shearing.
    other_vtype3 = condense_vtype3 + 1
    season_vtype3 = other_vtype3 + 1

    ##scale factor for fvps between shearing. This is required because if shearing occurs more frequently than once a yr the fvps that are determined from shearing date must be closer together.
    shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3 = (np.roll(date_shear_sa1e1b1nwzida0e0b0xyg3, -1, axis=0) - date_shear_sa1e1b1nwzida0e0b0xyg3).astype(float) / 365
    shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3[-1] = 1

    ##fvp's between weaning and first shearing - there will be 3 fvp's equally spaced between wean and first shearing (unless shearing occurs within 3 periods from weaning - if weaning and shearing are close the extra fvp are masked out in the stacking process below)
    ###b0
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg3[0:1]
    ###b1
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b1_start_ba1e1b1nwzida0e0b0xyg3, side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_ba1e1b1nwzida0e0b0xyg]
    ###b2
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + 2 * (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_ba1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_ba1e1b1nwzida0e0b0xyg]
    ##fvp0 - date shearing plus 1 day because shearing is the last day of period
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + np.maximum(1, fvp0_offset_ida0e0b0xyg3) #plus 1 at least 1 because shearing is the last day of the period and the fvp should start after shearing
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_0_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period. side=right so that if the date is already the start of a period it remains in that period.
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]
    ##fvp1 - date shearing plus offset1 (this is the first day of sim period)
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + (fvp1_offset_ida0e0b0xyg3 * shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3).astype(int)
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_1_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]
    ##fvp2 - date shearing plus offset2 (this is the first day of sim period)
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + (fvp2_offset_ida0e0b0xyg3 * shear_offset_adj_factor_sa1e1b1nwzida0e0b0xyg3).astype(int)
    idx_sa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_2_start_sa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_sa1e1b1nwzida0e0b0xyg]
    ##season nodes - these get masked out if steady state.
    node_fvp_m = np.zeros(len_m, dtype=object)
    for m in range(len_m):
        date_node_zidaebxyg = date_node_zidaebxygm[...,m]
        date_node_ya1e1b1nwzidaebxyg = date_node_zidaebxyg.astype(np.datetime64) + fun.f_expand(np.arange(np.ceil(sim_years_offs)) * np.timedelta64(365,'D'), p_pos)
        ###set the node fvp to start at the next generator period following the node (needs to be next so that clustering works).
        idx_ya1e1b1nwzida0e0b0xyg = np.minimum(len(offs_date_start_p)-1, np.searchsorted(offs_date_start_p, date_node_ya1e1b1nwzidaebxyg, 'left')) #maximum idx is the number of generator periods
        date_node_ya1e1b1nwzidaebxyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]
        node_fvp_m[m] = date_node_ya1e1b1nwzidaebxyg

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[1:],
                               fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[1:], fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[1:],
                               date_node_ya1e1b1nwzidaebxyg.shape[1:]]) #create shape which has the max size, this is used for o array
    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_0_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_0_start_sa1e1b1nwzida0e0b0xyg3,(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_1_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_1_start_sa1e1b1nwzida0e0b0xyg3,(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_2_start_sa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_2_start_sa1e1b1nwzida0e0b0xyg3,(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    for m in range(len_m):
        node_fvp_m[m] = np.broadcast_to(node_fvp_m[m],(node_fvp_m[m].shape[0],)+tuple(shape))

    fvp_b0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape, condense_vtype3)
    fvp_b1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_b2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_0_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_0_start_sa1e1b1nwzida0e0b0xyg3.shape, condense_vtype3)
    fvp_1_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_1_start_sa1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
    fvp_2_type_va1e1b1nwzida0e0b0xyg3 = np.full(fvp_2_start_sa1e1b1nwzida0e0b0xyg3.shape, other_vtype3)
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
    fvp_date_all_f3 = np.concatenate([node_fvp_m[0:1], fvp_date_all_f3, node_fvp_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    fvp_type_all_f3 = np.array([fvp_b0_type_va1e1b1nwzida0e0b0xyg3, fvp_b1_type_va1e1b1nwzida0e0b0xyg3,
                                fvp_b2_type_va1e1b1nwzida0e0b0xyg3, fvp_0_type_va1e1b1nwzida0e0b0xyg3,
                                fvp_1_type_va1e1b1nwzida0e0b0xyg3, fvp_2_type_va1e1b1nwzida0e0b0xyg3], dtype=object)
    fvp_type_all_f3 = np.concatenate([node_fvp_type_m[0:1], fvp_type_all_f3, node_fvp_type_m[1:]]) #seasons start needs to be first because it needs to be the first dvp in situations where there is a clash. so that distributing can occur from v_prev.
    ###if shearing is less than 3 sim periods after weaning then set the break fvp dates to the first date of the sim (so they arent used)
    mask_initial_fvp = np.all((date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3) > ((step.astype('timedelta64[D]')+1)*3)) #true if not enough gap between weaning and shearing for extra dvps.
    ###create the fvp mask. fvps are masked out depending on what the user has specified (the extra fvps at the start are removed if weaning is within 3weeks of shearing).
    fvp3_inc = np.concatenate([fvp_mask_offs[0:1], np.array([True, True and mask_initial_fvp, True and mask_initial_fvp]), fvp_mask_offs[1:]]) #Trues in middle are to count for the extra fvp at the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
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

    ##error check
    # can't be any clashes
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
    4. season start must be in the same v slice across all z axis because of weighted average needs all season start to be in same v.
    5.condense and season start cant clash (unless they have the same vtype) (this doesnt ocur for offs because no condense).
    '''
    ##dams
    ###build dvps from fvps
    mask_node_is_dvp = np.full(len_m, True) * (pinp.general['i_inc_node_periods'] or np.logical_not(bool_steady_state)) #node fvp/dvp are not included if it is steadystate.
    dvp_mask_f1 = np.concatenate([mask_node_is_dvp[0:1], sinp.stock['i_fixed_dvp_mask_f1'], sinp.structuralsa['i_dvp_mask_f1'], mask_node_is_dvp[1:]]) #season start is first
    dvp1_inc = np.concatenate([dvp_mask_f1[0:1], np.array([True]), dvp_mask_f1[1:]]) #True at start is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
    dvp_date_inc_v1 = fvp_date_all_f1[dvp1_inc]
    dvp_type_inc_v1 = fvp_type_all_f1[dvp1_inc]
    dvp_start_va1e1b1nwzida0e0b0xyg1 = np.concatenate(dvp_date_inc_v1,axis=0)
    dvp_type_va1e1b1nwzida0e0b0xyg1 = np.concatenate(dvp_type_inc_v1,axis=0) #note dvp type doesnt have to start at 0 or be consecutive.

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
        ###check that condense and season start dont clash - note this doesnt throw an error if condense_type==season_type (this is correct).
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
    dvp3_inc = np.concatenate([dvp_mask_f3[0:1], np.array([True, False, False]), dvp_mask_f3[1:]]) #True at start is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
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
        ###check that condense and season start dont clash - note this doesnt throw an error if condense_type==season_type (this is correct).
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
    a_next_s_pa1e1b1nwzida0e0b0xyg0 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg0, date_start_p, 0,'left')
    a_next_s_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg1, date_start_p, 0,'left')
    a_next_s_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, date_shear_sa1e1b1nwzida0e0b0xyg3, offs_date_start_p, 0,'left')
    ##p7 to p association - used for equation systems
    a_g0_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g0_p7'].astype('datetime64[D]'), date_end_p, 1,'right')
    a_g1_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g1_p7'].astype('datetime64[D]'), date_end_p, 1,'right')
    a_g2_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g2_p7'].astype('datetime64[D]'), date_end_p, 1,'right')
    a_g3_p7_p = np.apply_along_axis(fun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g3_p7'].astype('datetime64[D]'), date_end_p, 1,'right')
    ##month of each period (0 - 11 not 1 -12 because this is association array)
    a_p4_p = date_start_p.astype('datetime64[M]').astype(int) % 12
    ##feed variation period
    a_fvp_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_start_fa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_fvp_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association, 0, fvp_start_fa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')


    ############################
    ### apply associations     #
    ############################
    '''
    The association applied determines when the increment to the next opportunity will occur:
        eg if you use a_prev_joining the date in the p slice will increment at joining each time.
    
    '''
    ##shearing
    date_shear_pa1e1b1nwzida0e0b0xyg0 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg0, a_prev_s_pa1e1b1nwzida0e0b0xyg0,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    date_shear_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg1, a_prev_s_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    date_shear_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(date_shear_sa1e1b1nwzida0e0b0xyg3, a_prev_s_pa1e1b1nwzida0e0b0xyg3,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array

    ##management for weaning, gbal and scan options - adjusted further down to represent time of the repro cycle and management
    wean_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(wean_oa1e1b1nwzida0e0b0xyg1, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    gbal_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(gbal_oa1e1b1nwzida0e0b0xyg1, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    scan_option_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(scan_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array

    ##date, age, timing
    date_born1st_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)
    date_born_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born_oa1e1b1nwzida0e0b0xyg2, a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)
    date_born1st2_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
    date_born2_pa1e1b1nwzida0e0b0xyg2 = np.take_along_axis(date_born_oa1e1b1nwzida0e0b0xyg2, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
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

    ##yatf sim params - turn d to p axis
    ce_yatf = np.expand_dims(ce_yatf, axis = tuple(range(p_pos,d_pos)))
    ce_yatf = np.take_along_axis(ce_yatf,a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2[na,...],d_pos)

    ##feed period
    legume_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(legume_p6a1e1b1nwzida0e0b0xyg, a_p6_pa1e1b1nwzida0e0b0xyg, 0)

    ##expected stocking density
    density_pa1e1b1nwzida0e0b0xyg = np.take_along_axis(density_p6a1e1b1nwzida0e0b0xyg, a_p6_pa1e1b1nwzida0e0b0xyg, 0)

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
    rain_pa1e1b1nwzida0e0b0xygp1 = rain_p4a1e1b1nwzida0e0b0xygp1[a_p4_p]
    temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_p4a1e1b1nwzida0e0b0xyg[a_p4_p]
    temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_p4a1e1b1nwzida0e0b0xyg[a_p4_p]

    ##feed variation
    # fvp_type_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)
    # fvp_type_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg3,a_fvp_pa1e1b1nwzida0e0b0xyg3,0)
    fvp_date_start_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)
    fvp_date_start_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg3,a_fvp_pa1e1b1nwzida0e0b0xyg3,0)

    ##propn of dams mated
    prop_dams_mated_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_oa1e1b1nwzida0e0b0xyg1,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining

    ##break of season
    date_prev_seasonstart_pa1e1b1nwzida0e0b0xyg=np.take_along_axis(seasonstart_ya1e1b1nwzida0e0b0xyg,a_seasonstart_pa1e1b1nwzida0e0b0xyg,0)

    ##create association between n and w for each generator period i.e. what nutrition level is being offered to this LW profile in this period
    ###sire ^not required because sires only have 1 fvp and 1 n slice
    # a_n_pa1e1b1nwzida0e0b0xyg0 = (np.trunc(index_wzida0e0b0xyg0 / (n_fs_sire ** ((n_fvp_periods_g0-1) - fvp_type_pa1e1b1nwzida0e0b0xyg0))) % n_fs_sire).astype(int) #needs to be int so it can be an indice
    ###dams
    ####for dams period is condense is the same as period is prejoin atm but it is designed so it can be different (the lw_distribution will just need to be updated)
    period_is_condense_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_condensing_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_startfvp_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', fvp_date_start_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 = np.cumsum(period_is_startfvp_pa1e1b1nwzida0e0b0xyg1, axis=0)
    n_prior_fvps_pa1e1b1nwzida0e0b0xyg1 = (n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 -
                                           np.maximum.accumulate(n_cum_fvps_pa1e1b1nwzida0e0b0xyg1 * np.logical_or(period_is_condense_pa1e1b1nwzida0e0b0xyg1
                                                                , p_index_pa1e1b1nwzida0e0b0xyg==0), axis=0)) #there is no fvps prior to p0 hence index==0
    a_n_pa1e1b1nwzida0e0b0xyg1 = (np.trunc(index_wzida0e0b0xyg1 / (n_fs_dams ** ((n_fvp_periods_dams-1) - n_prior_fvps_pa1e1b1nwzida0e0b0xyg1))) % n_fs_dams).astype(int) #needs to be int so it can be an indice
    ###offs
    period_is_condense_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_condensing_pa1e1b1nwzida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3) #g2 date born is the equivalent of date lambed g1
    period_is_startfvp_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', fvp_date_start_pa1e1b1nwzida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 = np.cumsum(period_is_startfvp_pa1e1b1nwzida0e0b0xyg3, axis=0)
    n_prior_fvps_pa1e1b1nwzida0e0b0xyg3 = (n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 -
                                          np.maximum.accumulate(n_cum_fvps_pa1e1b1nwzida0e0b0xyg3 * np.logical_or(period_is_condense_pa1e1b1nwzida0e0b0xyg3
                                                                , p_index_pa1e1b1nwzida0e0b0xyg3==0), axis=0))
    a_n_pa1e1b1nwzida0e0b0xyg3 = (np.trunc(index_wzida0e0b0xyg3 / (n_fs_offs ** ((n_fvp_periods_offs-1) - n_prior_fvps_pa1e1b1nwzida0e0b0xyg3))) % n_fs_offs).astype(int) #needs to be int so it can be an indice


    ######################
    #adjust sensitivities#
    ######################
    saa_mortalityx_b1nwzida0e0b0xyg = fun.f_expand(sen.saa['mortalityx'][sinp.stock['a_nfoet_b1']], b1_pos)
    ## sum saa[rr] and saa[rr_age] so there is only one saa to handle in f_conception_cs & f_conception_ltw
    ## Note: the proportions of the BTRT doesn't include rr_age_og1 because those calculations can't vary by age of the dam
    rr_age_og1 = sen.saa['rr_age_og1']
    saa_rr_age_oa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(rr_age_og1, 'dams', p_pos, condition=mask_o_dams, axis=p_pos)
    saa_rr_age_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(saa_rr_age_oa1e1b1nwzida0e0b0xyg1,
                                                     a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0)  #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
    ## Alter the standard scanning rate for f_conception_ltw to include saa['rr_age'] (scan_std_yg0 has already been adjusted by saa['rr'] in UniversalInputs.py
    scan_std_pa1e1b1nwzida0e0b0xyg1 = scan_std_yg1 + saa_rr_age_pa1e1b1nwzida0e0b0xyg1
    ## Combine saa['rr'] and saa['rr_age'] for f_conception_cs
    saa_rr_age_pa1e1b1nwzida0e0b0xyg1 = saa_rr_age_pa1e1b1nwzida0e0b0xyg1 + sen.saa['rr']


    ###########################
    ##genotype calculations   #
    ###########################
    ##calc proportion of dry, singles, twin and triplets
    ###calculated without saa['rr_age']. These calculations do not include a 'p' axis because it is one value for all the initial animals
    dstwtr_l0yg0 = np.moveaxis(sfun.f1_DSTw(scan_std_yg0), -1, 0)
    dstwtr_l0yg1 = np.moveaxis(sfun.f1_DSTw(scan_std_yg1), -1, 0)
    dstwtr_l0yg3 = np.moveaxis(sfun.f1_DSTw(scan_dams_std_yg3), -1, 0)

    ##Std scanning & survival: propn of progeny in each BTRT b0 category - 11, 22, 33, 21, 32, 31 and lambs surviving birth per ewe joined (standard weaning percentage)
    btrt_propn_b0xyg0, npw_std_xyg0 = sfun.f1_btrt0(dstwtr_l0yg0,pss_std_yg0,pstw_std_yg0,pstr_std_yg0)
    btrt_propn_b0xyg1, npw_std_xyg1 = sfun.f1_btrt0(dstwtr_l0yg1,pss_std_yg1,pstw_std_yg1,pstr_std_yg1)
    btrt_propn_b0xyg3, npw_std_xyg3 = sfun.f1_btrt0(dstwtr_l0yg3,pss_std_yg3,pstw_std_yg3,pstr_std_yg3)

    # ##Std scanning & survival: proportion of dams in each BTRT b1 category - NM, 00, 11, 22, 33, 21, 32, 31, 10, 20, 30
    # btrt_b1nwzida0e0b0xy0 = f_btrt1(dstwtr_l0yg0,pss_std_yg0,pstw_std_yg0,pstr_std_yg0)
    # btrt_b1nwzida0e0b0xy1 = f_btrt1(dstwtr_l0yg1,pss_std_yg1,pstw_std_yg1,pstr_std_yg1)
    # btrt_b1nwzida0e0b0xy2 = f_btrt1(dstwtr_l0yg2,pss_std_yg2,pstw_std_yg2,pstr_std_yg2)
    # btrt_b1nwzida0e0b0xy3 = f_btrt1(dstwtr_l0yg3,pss_std_yg3,pstw_std_yg3,pstr_std_yg3)

    ###calc adjustments sfw
    adja_sfw_d_a0e0b0xyg0 = np.sum(ce_sire[12, ...] * agedam_propn_da0e0b0xyg0, axis = 0)
    adja_sfw_d_a0e0b0xyg1 = np.sum(ce_dams[12, ...] * agedam_propn_da0e0b0xyg1, axis = 0)
    adja_sfw_d_pa1e1b1nwzida0e0b0xyg2 = ce_yatf[12,...]
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
    adja_sfd_d_pa1e1b1nwzida0e0b0xyg2 = ce_yatf[13, ...]
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

    ##Standard birth weight -
    w_b_std_b0xyg0 = srw_female_yg0 * np.sum(cb0_sire[15, ...] * btrt_propn_b0xyg0, axis = b0_pos, keepdims=True) * cx_sire[15, 0:1, ...]
    w_b_std_b0xyg1 = srw_female_yg1 * np.sum(cb0_dams[15, ...] * btrt_propn_b0xyg1, axis = b0_pos, keepdims=True) * cx_dams[15, 1:2, ...]
    w_b_std_b0xyg3 = srw_female_yg3 * cb0_offs[15, ...] * cx_offs[15, mask_x,...]
    ##fetal param - normal birthweight young - used as target birthweight during pregnancy if sheep fed well. Therefore average gender effect.
    w_b_std_y_b1nwzida0e0b0xyg1 = srw_female_yg2 * cb1_yatf[15, ...] #gender not considers until actual birth therefore no cx
    ##wool growth efficiency
    ###wge is sfw divided by srw of a ewe of the given genotype
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
    #mask adjustments along w axis so it can handle changing the number of dvps
    # n_lw1_total = n_fs_dams ** (len(fvp_mask_dams) + 1) #total lw if all dvps included
    # lw_adjp_mask_dams = np.full(n_lw1_total,False)
    # lw_adjp_mask_dams[0:len_w1] = True
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
    lw_initial_yg0, lw_initial_yg1, lw_initial_yatf, lw_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_lw_initial_c2'], uinp.parameters['i_lw_initial_y'])
    ###the initial lw input is a proportion of srw
    lw_initial_wzida0e0b0xyg0 = (lw_initial_yg0 * (1 + adjp_lw_initial_wzida0e0b0xyg0)) * srw_female_yg0
    lw_initial_wzida0e0b0xyg1 = (lw_initial_yg1 * (1 + adjp_lw_initial_wzida0e0b0xyg1)) * srw_female_yg1
    lw_initial_wzida0e0b0xyg3 = (lw_initial_yg3 * (1 + adjp_lw_initial_wzida0e0b0xyg3)) * srw_female_yg3
    cfw_initial_yg0, cfw_initial_yg1, cfw_initial_yatf, cfw_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_cfw_initial_c2'], uinp.parameters['i_cfw_initial_y'])
    ###the initial cfw input is a proportion of sfw
    cfw_initial_wzida0e0b0xyg0 = (cfw_initial_yg0 * (1 + adjp_cfw_initial_wzida0e0b0xyg0)) * sfw_yg0
    cfw_initial_wzida0e0b0xyg1 = (cfw_initial_yg1 * (1 + adjp_cfw_initial_wzida0e0b0xyg1)) * sfw_yg1
    cfw_initial_wzida0e0b0xyg3 = (cfw_initial_yg3 * (1 + adjp_cfw_initial_wzida0e0b0xyg3)) * sfw_yg3
    fd_initial_yg0, fd_initial_yg1, fd_initial_yatf, fd_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_fd_initial_c2'], uinp.parameters['i_fd_initial_y'])
    fd_initial_wzida0e0b0xyg0 = fd_initial_yg0 * (1 + adjp_fd_initial_wzida0e0b0xyg0)
    fd_initial_wzida0e0b0xyg1 = fd_initial_yg1 * (1 + adjp_fd_initial_wzida0e0b0xyg1)
    fd_initial_wzida0e0b0xyg3 = fd_initial_yg3 * (1 + adjp_fd_initial_wzida0e0b0xyg3)
    fl_initial_yg0, fl_initial_yg1, fl_initial_yatf, fl_initial_yg3 = sfun.f1_c2g(uinp.parameters['i_fl_initial_c2'], uinp.parameters['i_fl_initial_y'])
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
    ##adjustment for gender. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr)
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
    ##adjust for btrt. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr)
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
    ##calc aw, bw and mw (adipose, bone and muscle weight)
    aw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * aw_propn_yg0
    aw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * aw_propn_yg1
    aw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * aw_propn_yg3
    bw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * bw_propn_yg0
    bw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * bw_propn_yg1
    bw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * bw_propn_yg3
    mw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * mw_propn_yg0
    mw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * mw_propn_yg1
    mw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * mw_propn_yg3
    ##numbers
    ###Distribution of initial numbers across the a1 axis
    initial_a1 = pinp.sheep['i_initial_a1'][pinp.sheep['i_mask_a']] / np.sum(pinp.sheep['i_initial_a1'][pinp.sheep['i_mask_a']])
    initial_a1e1b1nwzida0e0b0xyg = fun.f_expand(initial_a1, a1_pos)
    ###Distribution of initial numbers across the b1 axis
    initial_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['i_initial_b1'], b1_pos)
    ###Distribution of initial numbers across the y axis
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
    date_mated_pa1e1b1nwzida0e0b0xyg1 = date_born2_pa1e1b1nwzida0e0b0xyg2 - cp_dams[1,0:1,:].astype('timedelta64[D]') #use dateborn2 so it increments at prejoining
    ##day90 after mating (for use in the LTW calculations)
    date_d90_pa1e1b1nwzida0e0b0xyg1 = date_mated_pa1e1b1nwzida0e0b0xyg1 + np.array([90]).astype('timedelta64[D]')
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
    age_start_pa1e1b1nwzida0e0b0xyg2 = (np.maximum(np.array([0]).astype('timedelta64[D]'),np.minimum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2) - date_born_pa1e1b1nwzida0e0b0xyg2)).astype(int) #use min and max so that the min age is 0 and the max age is the age at weaning
    ##Age_end: age at the beginning of the last day of the given period
    ##age end, minus one to allow the plus one in the next step when period date is less than weaning date (the minus one ensures that when the p_date is less than weaning the animal gets 0 days in the given period)
    age_end_pa1e1b1nwzida0e0b0xyg0 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg0 -1) - date_born_ida0e0b0xyg0).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg1 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg1 -1) - date_born_ida0e0b0xyg1).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg3 = (np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg3 -1) - date_born_ida0e0b0xyg3).astype(int) #use date weaned because the simulation for these animals is starting at weaning.
    age_end_pa1e1b1nwzida0e0b0xyg2 = (np.maximum(np.array([-1]).astype('timedelta64[D]'),np.minimum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2 -1) - date_born_pa1e1b1nwzida0e0b0xyg2)).astype(int)  #use min and max so that the min age is 0 and the max age is the age at weaning

    ##age mid period , plus one to get the age at the end of the last day of the period ie needed to get the full len of period.
    age_pa1e1b1nwzida0e0b0xyg0 = (age_start_pa1e1b1nwzida0e0b0xyg0 + age_end_pa1e1b1nwzida0e0b0xyg0 +1) /2
    age_pa1e1b1nwzida0e0b0xyg1 = (age_start_pa1e1b1nwzida0e0b0xyg1 + age_end_pa1e1b1nwzida0e0b0xyg1 +1) /2
    age_pa1e1b1nwzida0e0b0xyg2 = (age_start_pa1e1b1nwzida0e0b0xyg2 + age_end_pa1e1b1nwzida0e0b0xyg2 +1) /2
    age_pa1e1b1nwzida0e0b0xyg3 = (age_start_pa1e1b1nwzida0e0b0xyg3 + age_end_pa1e1b1nwzida0e0b0xyg3 +1) /2

    ##days in each period for each animal - cant mask the offs p axis because need full axis so it can be used in the generator (if days_period[p] > 0)
    days_period_pa1e1b1nwzida0e0b0xyg0 = age_end_pa1e1b1nwzida0e0b0xyg0 +1 - age_start_pa1e1b1nwzida0e0b0xyg0
    days_period_pa1e1b1nwzida0e0b0xyg1 = age_end_pa1e1b1nwzida0e0b0xyg1 +1 - age_start_pa1e1b1nwzida0e0b0xyg1
    days_period_pa1e1b1nwzida0e0b0xyg2 = age_end_pa1e1b1nwzida0e0b0xyg2 +1 - age_start_pa1e1b1nwzida0e0b0xyg2
    days_period_pa1e1b1nwzida0e0b0xyg3 = (age_end_pa1e1b1nwzida0e0b0xyg3 +1 - age_start_pa1e1b1nwzida0e0b0xyg3
                                          ) * (p_index_pa1e1b1nwzida0e0b0xyg<np.count_nonzero(mask_p_offs_p)-1) #make days per period zero if period is not required for offs. -1 because we want days per period to be 0 in the period before the mask turns to false because there are some spots where p+1 is used as the index
    days_period_cut_pa1e1b1nwzida0e0b0xyg3 = days_period_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] #masked version of p axis

    ##Age of foetus (start of period, end of period and mid period - days)
    age_f_start_open_pa1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1
    age_f_start_pa1e1b1nwzida0e0b0xyg1 = np.maximum(np.array([0]).astype('timedelta64[D]')
                                            , np.minimum(cp_dams[1, 0:1, :].astype('timedelta64[D]')
                                                , date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1))
    age_f_end_pa1e1b1nwzida0e0b0xyg1 = np.minimum(cp_dams[1, 0:1, :].astype('timedelta64[D]') - 1
                                                  , date_end_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1) #open at bottom capped at top, cp -1 so that the period_days formula below is correct when p_date - date_mated is greater than cp (because plus 1)


    ############################
    ### Daily steps            #
    ############################
    ##definition for this is that the action eg weaning occurs at 12am on the given date. therefore if weaning occurs on day 150 the lambs are counted as weaned lambs on that day.
    ##This info determines the side with > or >=


    ##add p1 axis
    date_start_pa1e1b1nwzida0e0b0xygp1 = date_start_pa1e1b1nwzida0e0b0xyg[...,na] + index_p1
    doy_pa1e1b1nwzida0e0b0xygp1= doy_pa1e1b1nwzida0e0b0xyg[...,na] + index_p1
    ##age open ie not capped at weaning
    age_p1_pa1e1b1nwzida0e0b0xyg0p1 = (age_start_open_pa1e1b1nwzida0e0b0xyg0[..., na] + index_p1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
    age_p1_pa1e1b1nwzida0e0b0xyg1p1 = (age_start_open_pa1e1b1nwzida0e0b0xyg1[..., na] + index_p1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
    age_p1_pa1e1b1nwzida0e0b0xyg2p1 = (age_start_open_pa1e1b1nwzida0e0b0xyg2[..., na] + index_p1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
    age_p1_pa1e1b1nwzida0e0b0xyg3p1 = (age_start_open_pa1e1b1nwzida0e0b0xyg3[..., na] + index_p1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
    ##calc p1 weights - if age<=weaning it has 0 weighting (ie false) else weighting = 1 (ie true) - need so that the values are not included in the mean calculations below which determine the average production for a given p1 period.
    age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1 = age_p1_pa1e1b1nwzida0e0b0xyg0p1>=(date_weaned_ida0e0b0xyg0 - date_born_ida0e0b0xyg0)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1 = age_p1_pa1e1b1nwzida0e0b0xyg1p1>=(date_weaned_ida0e0b0xyg1 - date_born_ida0e0b0xyg1)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1 = age_p1_pa1e1b1nwzida0e0b0xyg3p1>=(date_weaned_ida0e0b0xyg3 - date_born_ida0e0b0xyg3)[..., na].astype(int) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    ##calc yatf p1 weighting - if age is greater than weaning or less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1 = np.logical_and(age_p1_pa1e1b1nwzida0e0b0xyg2p1>=0, age_p1_pa1e1b1nwzida0e0b0xyg2p1<(date_weaned_pa1e1b1nwzida0e0b0xyg2 - date_born_pa1e1b1nwzida0e0b0xyg2)[..., na].astype(int)) #use date_wean - date born because that results in the average weaning age of all animal (age_weaned variable is just the age of the first animal)
    ##Age of foetus with minor axis (days)
    age_f_p1_pa1e1b1nwzida0e0b0xyg1p1 = (age_f_start_open_pa1e1b1nwzida0e0b0xyg1[...,na] + index_p1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
    ##calc foetus p1 weighting - if age is greater than birth or less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    age_f_p1_weights_pa1e1b1nwzida0e0b0xyg1p1 = np.logical_and(age_f_p1_pa1e1b1nwzida0e0b0xyg1p1>=0, age_f_p1_pa1e1b1nwzida0e0b0xyg1p1<cp_dams[1, 0, :, na])
    #age_f_p1_pa1e1b1nwzida0e0b0xyg1p1[age_f_p1_pa1e1b1nwzida0e0b0xyg1p1 <= 0] = np.nan
    #age_f_p1_pa1e1b1nwzida0e0b0xyg1p1[age_f_p1_pa1e1b1nwzida0e0b0xyg1p1 > cp_dams[1, 0, :, na]] = np.nan
    ##adjusted age of young (adjusted by intake factor - basically the factor of how age of young effect dam intake, the adjustment factor basically alters the age of the young to influence intake.)
    age_y_adj_pa1e1b1nwzida0e0b0xyg1p1 = age_p1_pa1e1b1nwzida0e0b0xyg2p1 + np.maximum(0, (date_start_pa1e1b1nwzida0e0b0xygp1 - date_weaned_pa1e1b1nwzida0e0b0xyg2[..., na]) /np.timedelta64(1, 'D')) * (ci_dams[21, ..., na] - 1) #minus 1 because the ci factor is applied to the age post weaning but using the open date means it has already been included once ie we want x + y *ci but using date open gives  x  + y + y*ci, x = age to weaning, y = age between period and weaning, therefore minus 1 x  + y + y*(ci-1)
    ##calc young p1 weighting - if age is less than 0 it will have 0 weighting in the p1 means calculated below else it will have weighting 1
    age_y_adj_weights_pa1e1b1nwzida0e0b0xyg1p1 = age_y_adj_pa1e1b1nwzida0e0b0xyg1p1 > 0  #no max cap (ie represents age young would be if never weaned off mum)
    ##Foetal age relative to parturition with minor axis
    relage_f_pa1e1b1nwzida0e0b0xyg1p1 = np.maximum(0,age_f_p1_pa1e1b1nwzida0e0b0xyg1p1 / cp_dams[1, 0, :, na])
    ##Age of lamb relative to peak intake-with minor function
    pimi_pa1e1b1nwzida0e0b0xyg1p1 = age_y_adj_pa1e1b1nwzida0e0b0xyg1p1 / ci_dams[8, ..., na]
    ##Age of lamb relative to peak lactation-with minor axis
    lmm_pa1e1b1nwzida0e0b0xyg1p1 = (age_p1_pa1e1b1nwzida0e0b0xyg2p1 + cl_dams[1, ..., na]) / cl_dams[2, ..., na]
    ##Chill index for lamb survival
    #todo consider adding p1p2p3 axes for chill for rain, ws & temp_ave.
    chill_index_pa1e1b1nwzida0e0b0xygp1 = (481 + (11.7 + 3.1 * ws_pa1e1b1nwzida0e0b0xyg[..., na] ** 0.5)
                                           * (40 - temp_ave_pa1e1b1nwzida0e0b0xyg[..., na])
                                           + 418 * (1-np.exp(-0.04 * rain_pa1e1b1nwzida0e0b0xygp1)))
    chill_index_pa1e1b1nwzida0e0b0xygp1 = fun.f_sa(chill_index_pa1e1b1nwzida0e0b0xygp1, sen.sam['chill'])

    ##Proportion of SRW with age
    srw_age_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.exp(-cn_sire[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg0p1 / srw_xyg0[..., na] ** cn_sire[2, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(-cn_dams[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg1p1 / srw_xyg1[..., na] ** cn_dams[2, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.exp(-cn_yatf[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg2p1 / srw_xyg2[..., na] ** cn_yatf[2, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    srw_age_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.exp(-cn_offs[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg3p1 / srw_xyg3[..., na] ** cn_offs[2, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1, axis = -1)

    #srw_age_pa1e1b1nwzida0e0b0xyg0 = np.nanmean(np.exp(-cn_sire[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg0p1 / srw_xyg0[..., na] ** cn_sire[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(-cn_dams[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg1p1 / srw_xyg1[..., na] ** cn_dams[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(np.exp(-cn_yatf[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg2p1 / srw_xyg2[..., na] ** cn_yatf[2, ..., na]), axis = -1)
    #srw_age_pa1e1b1nwzida0e0b0xyg3 = np.nanmean(np.exp(-cn_offs[1, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg3p1 / srw_xyg3[..., na] ** cn_offs[2, ..., na]), axis = -1)

    ##age factor wool part 1- reduces fleece growth early in life
    af1_wool_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(cw_sire[5, ..., na] + (1 - cw_sire[5, ..., na])*(1-np.exp(-cw_sire[12, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg0p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cw_dams[5, ..., na] + (1 - cw_dams[5, ..., na])*(1-np.exp(-cw_dams[12, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg1p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(cw_yatf[5, ..., na] + (1 - cw_yatf[5, ..., na])*(1-np.exp(-cw_yatf[12, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg2p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    af1_wool_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(cw_offs[5, ..., na] + (1 - cw_offs[5, ..., na])*(1-np.exp(-cw_offs[12, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg3p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1, axis = -1)
    ##age factor wool part 2 - reduces fleece growth later in life (data used to create equations from Lifetime Productivity by Richards&Atkins)
    af2_wool_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(2 - np.exp(cw_sire[17, ..., na] * np.maximum(0,age_p1_pa1e1b1nwzida0e0b0xyg0p1 - cw_sire[18, ..., na])**cw_sire[19, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(2 - np.exp(cw_dams[17, ..., na] * np.maximum(0,age_p1_pa1e1b1nwzida0e0b0xyg1p1 - cw_dams[18, ..., na])**cw_dams[19, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(2 - np.exp(cw_yatf[17, ..., na] * np.maximum(0,age_p1_pa1e1b1nwzida0e0b0xyg2p1 - cw_yatf[18, ..., na])**cw_yatf[19, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    af2_wool_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(2 - np.exp(cw_offs[17, ..., na] * np.maximum(0,age_p1_pa1e1b1nwzida0e0b0xyg3p1 - cw_offs[18, ..., na])**cw_offs[19, ..., na]), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1, axis = -1)
    ##overall age factor - reduction for young animals and older animals
    af_wool_pa1e1b1nwzida0e0b0xyg0 = af1_wool_pa1e1b1nwzida0e0b0xyg0 * af2_wool_pa1e1b1nwzida0e0b0xyg0
    af_wool_pa1e1b1nwzida0e0b0xyg1 = af1_wool_pa1e1b1nwzida0e0b0xyg1 * af2_wool_pa1e1b1nwzida0e0b0xyg1
    af_wool_pa1e1b1nwzida0e0b0xyg2 = af1_wool_pa1e1b1nwzida0e0b0xyg2 * af2_wool_pa1e1b1nwzida0e0b0xyg2
    af_wool_pa1e1b1nwzida0e0b0xyg3 = af1_wool_pa1e1b1nwzida0e0b0xyg3 * af2_wool_pa1e1b1nwzida0e0b0xyg3

    ##Day length factor on efficiency
    dlf_eff_pa1e1b1nwzida0e0b0xyg = np.average(lat_deg / 40 * np.sin(2 * np.pi * doy_pa1e1b1nwzida0e0b0xygp1 / 365), axis = -1)
    ##Pattern of maintenance with age
    mr_age_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.maximum(cm_sire[4, ..., na], np.exp(-cm_sire[3, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg0p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(cm_dams[4, ..., na], np.exp(-cm_dams[3, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg1p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.maximum(cm_offs[4, ..., na], np.exp(-cm_offs[3, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg2p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    mr_age_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.maximum(cm_yatf[4, ..., na], np.exp(-cm_yatf[3, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg3p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1, axis = -1)
    ##Impact of rainfall on 'cold' intake increment
    rain_intake_pa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp1 / ci_sire[18, ..., na]),  weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg0p1, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp1 / ci_dams[18, ..., na]),  weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp1 / ci_offs[18, ..., na]),  weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    rain_intake_pa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygp1[mask_p_offs_p] / ci_yatf[18, ..., na]),  weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg3p1, axis = -1)
    ##Proportion of peak intake due to time from birth
    pi_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cb1_dams[19, ..., na] * np.maximum(0,pimi_pa1e1b1nwzida0e0b0xyg1p1) ** ci_dams[9, ..., na] * np.exp(ci_dams[9, ..., na] * (1 - pimi_pa1e1b1nwzida0e0b0xyg1p1)), weights=age_y_adj_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1) #maximum to stop error in power (not sure why the negatives were causing a problem)
    ##Peak milk production pattern (time from birth)
    mp_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(cb1_dams[0, ..., na] * lmm_pa1e1b1nwzida0e0b0xyg1p1 ** cl_dams[3, ..., na] * np.exp(cl_dams[3, ..., na]* (1 - lmm_pa1e1b1nwzida0e0b0xyg1p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    ##Suckling volume pattern
    mp2_age_y_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(nyatf_b1nwzida0e0b0xyg[...,na] * cl_dams[6, ..., na] * ( cl_dams[12, ..., na] + cl_dams[13, ..., na] * np.exp(-cl_dams[14, ..., na] * age_p1_pa1e1b1nwzida0e0b0xyg2p1)), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    ##Pattern of conception efficiency (doy)
    crg_doy_pa1e1b1nwzida0e0b0xyg1 = np.average(np.maximum(0,1 - cb1_dams[1, ..., na] * (1 - np.sin(2 * np.pi * (doy_pa1e1b1nwzida0e0b0xygp1 + 10) / 365) * np.sin(lat_rad) / -0.57)), axis = -1)
    ##Rumen development factor on PI - yatf
    piyf_pa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(1/(1 + np.exp(-ci_yatf[3, ..., na] * (age_p1_pa1e1b1nwzida0e0b0xyg2p1 - ci_yatf[4, ..., na]))), weights=age_p1_weights_pa1e1b1nwzida0e0b0xyg2p1, axis = -1)
    piyf_pa1e1b1nwzida0e0b0xyg2 = piyf_pa1e1b1nwzida0e0b0xyg2 * (nyatf_b1nwzida0e0b0xyg > 0) #set pi to 0 if no yatf.
    ##Foetal normal weight pattern (mid period)
    nwf_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[2, ..., na] * (1 - np.exp(cp_dams[3, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p1)))), weights=age_f_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    ##Conceptus weight pattern (mid period)
    guw_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[6, ..., na] * (1 - np.exp(cp_dams[7, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p1)))), weights=age_f_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    ##Conceptus energy pattern (end of period)
    # ce_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.exp(cp_dams[9, ..., na] * (1 - np.exp(cp_dams[10, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p1)))), weights=age_f_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)
    ##Conceptus energy pattern (d_nec)
    dce_age_f_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average((cp_dams[9, ..., na] - cp_dams[10, ..., na]) / cp_dams[1, 0, ..., na] * np.exp(cp_dams[10, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p1) + cp_dams[9, ..., na] * (1 - np.exp(cp_dams[10, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1p1)))), weights=age_f_p1_weights_pa1e1b1nwzida0e0b0xyg1p1, axis = -1)

    ##genotype calc that requires af_wool. ME for minimum wool growth (with no intake, relsize = 1)
    mew_min_pa1e1b1nwzida0e0b0xyg0 =cw_sire[14, ...] * sfw_a0e0b0xyg0[0, ...] / cw_sire[3,...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg0 * dlf_wool_pa1e1b1nwzida0e0b0xyg0 * cw_sire[1, ...] / kw_yg0
    mew_min_pa1e1b1nwzida0e0b0xyg1 =cw_dams[14, ...] * sfw_a0e0b0xyg1[0, ...] / cw_dams[3,...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg1 * dlf_wool_pa1e1b1nwzida0e0b0xyg1 * cw_dams[1, ...] / kw_yg1
    mew_min_pa1e1b1nwzida0e0b0xyg2 =cw_yatf[14, ...] * sfw_pa1e1b1nwzida0e0b0xyg2[0, ...] / cw_yatf[3,...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg2 * dlf_wool_pa1e1b1nwzida0e0b0xyg2 * cw_yatf[1, ...] / kw_yg2
    mew_min_pa1e1b1nwzida0e0b0xyg3 =cw_offs[14, ...] * sfw_da0e0b0xyg3[0, ...] / cw_offs[3,...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg3 * dlf_wool_pa1e1b1nwzida0e0b0xyg3 * cw_offs[1, ...] / kw_yg3

    ##plot above x axis is p
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

    ##Is nutrition effecting lactation
    lact_nut_effect_pa1e1b1nwzida0e0b0xyg1 = (age_pa1e1b1nwzida0e0b0xyg2  > (cl_dams[16, ...] * cl_dams[2, ...]))

    ##Average daily CFW
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg0 = sfw_a0e0b0xyg0 * af_wool_pa1e1b1nwzida0e0b0xyg0 / 365
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg1 = sfw_a0e0b0xyg1 * af_wool_pa1e1b1nwzida0e0b0xyg1 / 365
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg2 = sfw_pa1e1b1nwzida0e0b0xyg2 * af_wool_pa1e1b1nwzida0e0b0xyg2 / 365
    d_cfw_ave_pa1e1b1nwzida0e0b0xyg3 = sfw_da0e0b0xyg3 * af_wool_pa1e1b1nwzida0e0b0xyg3 / 365

    ##Expected relative size
    relsize_exp_a1e1b1nwzida0e0b0xyg0  = (srw_xyg0 - (srw_xyg0 - w_b_std_b0xyg0) * np.exp(-cn_sire[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 /np.timedelta64(1, 'D')) / (srw_xyg0**cn_sire[2, ...]))) / srw_xyg0
    relsize_exp_a1e1b1nwzida0e0b0xyg1  = (srw_xyg1 - (srw_xyg1 - w_b_std_b0xyg1) * np.exp(-cn_dams[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg1/np.timedelta64(1, 'D')) / (srw_xyg1**cn_dams[2, ...]))) / srw_xyg1
    relsize_exp_a1e1b1nwzida0e0b0xyg3  = (srw_xyg3 - (srw_xyg3 - w_b_std_b0xyg3) * np.exp(-cn_offs[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg3/np.timedelta64(1, 'D')) / (srw_xyg3**cn_offs[2, ...]))) / srw_xyg3

    ##adjust ce sim param (^ ce12 &13 should be scaled by relsize (similar to ce15)) -  (^instead of setting ce with relsize adjustment then adjusting birth weight could just adjust birthweight directly with relsize factor - to avoid doing this code below)
    shape = (ce_sire.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg0.shape #get shape of the new ce array
    ce_ca1e1b1nwzida0e0b0xyg0 = np.zeros(shape) #make a new array - same a ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg0[...] = fun.f_expand(ce_sire, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg0[15, ...] = 1 - cp_sire[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg0) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_sire = ce_ca1e1b1nwzida0e0b0xyg0 #rename to keep consistent

    shape = (ce_dams.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg1.shape #get shape of the new ce array - required because assigning relsize which is diff size
    ce_ca1e1b1nwzida0e0b0xyg1 = np.zeros(shape) #make a new array - same a ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg1[...] = fun.f_expand(ce_dams, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg1[15, ...] = 1 - cp_dams[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg1) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_dams = ce_ca1e1b1nwzida0e0b0xyg1 #rename to keep consistent

    shape = (ce_offs.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg3.shape #get shape of the new ce array - required because assigning relsize which is diff size
    ce_ca1e1b1nwzida0e0b0xyg3 = np.zeros(shape) #make a new array - same a ce with an active i axis
    ce_ca1e1b1nwzida0e0b0xyg3[...] = fun.f_expand(ce_offs, p_pos, right_pos=uinp.parameters['i_ce_pos'])
    ce_ca1e1b1nwzida0e0b0xyg3[15, ...] = 1 - cp_offs[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg3) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
    ce_offs = ce_ca1e1b1nwzida0e0b0xyg3 #rename to keep consistent

    ##birth weight expected - includes relsize factor
    w_b_exp_a1e1b1nwzida0e0b0xyg0 = w_b_std_b0xyg0 * np.sum(ce_sire[15, ...] * agedam_propn_da0e0b0xyg0, axis = d_pos, keepdims = True)
    w_b_exp_a1e1b1nwzida0e0b0xyg1 = w_b_std_b0xyg1 * np.sum(ce_dams[15, ...] * agedam_propn_da0e0b0xyg1, axis = d_pos, keepdims = True)
    w_b_exp_a1e1b1nwzida0e0b0xyg3 = w_b_std_b0xyg3 * ce_offs[15, ...]

    ##Normal weight max (if animal is well fed)
    nw_max_pa1e1b1nwzida0e0b0xyg0 = srw_xyg0 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg0) + w_b_exp_a1e1b1nwzida0e0b0xyg0 * srw_age_pa1e1b1nwzida0e0b0xyg0
    nw_max_pa1e1b1nwzida0e0b0xyg1 = srw_xyg1 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg1) + w_b_exp_a1e1b1nwzida0e0b0xyg1 * srw_age_pa1e1b1nwzida0e0b0xyg1
    nw_max_pa1e1b1nwzida0e0b0xyg3 = srw_xyg3 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg3) + w_b_exp_a1e1b1nwzida0e0b0xyg3 * srw_age_pa1e1b1nwzida0e0b0xyg3

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
    date_scan_pa1e1b1nwzida0e0b0xyg1 = date_joined2_pa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') \
                                       + pinp.sheep['i_scan_day'][scan_option_pa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]')
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
    period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_scan_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_d90birth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_d90_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
    period_between_birthwean_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_born_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg)
    period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_weaned2_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_prejoin_next_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_scan_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_scan_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_birth_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_born_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_wean_pa1e1b1nwzida0e0b0xyg2 = sfun.f1_period_is_('period_is', date_weaned_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_is_wean_pa1e1b1nwzida0e0b0xyg1 = period_is_wean_pa1e1b1nwzida0e0b0xyg2
    # prev_period_is_birth_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_birth_pa1e1b1nwzida0e0b0xyg1,1,axis=p_pos)
    period_is_mating_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_mated_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivalent of date lambed g1
    period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is_between', date_born_pa1e1b1nwzida0e0b0xyg2
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_born_pa1e1b1nwzida0e0b0xyg2+np.array([(6*7)]).astype('timedelta64[D]'), date_end_pa1e1b1nwzida0e0b0xyg) #This is within 6 weeks of the Birth period
    ###shearing
    period_is_shearing_pa1e1b1nwzida0e0b0xyg0 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg0
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_shearing_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg1
                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    period_is_shearing_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_shear_pa1e1b1nwzida0e0b0xyg3
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

    ##################################################
    #adjust lsln management for timing of repro cycle#
    ##################################################
    ##calc lsln management association based on sheep identification options (scanning vs no scanning), management practise (differential management once identifying different groups) & time of the year (eg even if you scan you still need to manage sheep the same before scanning)
    ###have to create a_t array that is maximum size of the arrays that are used to mask it.
    ###t = 0 is prescan, 1 is postscan, 2 is lactation, 3 not used in V1 but would be is post wean
    shape = np.maximum.reduce([period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1.shape, period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1.shape
                                  , period_between_birthwean_pa1e1b1nwzida0e0b0xyg1.shape]) #create shape which has the max size
    a_t_pa1e1b1nwzida0e0b0xyg1 = np.zeros(shape)
    period_between_prejoinscan_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1)[1] #mask must be manually broadcasted then applied - for some reason numpy doesnt automatically broadcast them.
    period_between_scanbirth_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1)[1]
    period_between_birthwean_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_birthwean_pa1e1b1nwzida0e0b0xyg1)[1]
    ###order matters because post wean does not have a cap ie it is over written by others
    a_t_pa1e1b1nwzida0e0b0xyg1[...] = 3 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 not used in V1 but would be is post wean
    a_t_pa1e1b1nwzida0e0b0xyg1[period_between_prejoinscan_mask] = 0 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean
    a_t_pa1e1b1nwzida0e0b0xyg1[period_between_scanbirth_mask] = 1 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean
    a_t_pa1e1b1nwzida0e0b0xyg1[period_between_birthwean_mask] = 2 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is not used in V1 but would be post wean

    ###dams management in each period based on scanning detail, time of the year (even if you are scanning you cant manage sheep differently before scanning) and management (you can scan and then not differentially manage)
    scan_management_pa1e1b1nwzida0e0b0xyg1 = (scan_option_pa1e1b1nwzida0e0b0xyg1) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 1) * pinp.sheep['i_dam_lsln_diffman_t'][1]
    gbal_management_pa1e1b1nwzida0e0b0xyg1 = (gbal_pa1e1b1nwzida0e0b0xyg1 -1 ) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 2) * pinp.sheep['i_dam_lsln_diffman_t'][2] + 1  #minus 1 then plus 1 ensures that the wean option before lactation is 1
    wean_management_pa1e1b1nwzida0e0b0xyg1 = (wean_pa1e1b1nwzida0e0b0xyg1 -1 ) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 3) * pinp.sheep['i_dam_lsln_diffman_t'][3] + 1  #minus 1 then plus 1 ensures that the wean option before weaning is 1

    ############################
    ### feed supply calcs      #
    ############################
    legume_p6a1e1b1nwzida0e0b0xyg,bool_confinement_g0_n,bool_confinement_g1_n,bool_confinement_g3_n, \
    nv_p6a1e1b1j1wzida0e0b0xyg0,foo_p6a1e1b1j1wzida0e0b0xyg0,dmd_p6a1e1b1j1wzida0e0b0xyg0,supp_p6a1e1b1j1wzida0e0b0xyg0, \
    nv_p6a1e1b1j1wzida0e0b0xyg1,foo_p6a1e1b1j1wzida0e0b0xyg1,dmd_p6a1e1b1j1wzida0e0b0xyg1,supp_p6a1e1b1j1wzida0e0b0xyg1, \
    nv_p6a1e1b1j1wzida0e0b0xyg3,foo_p6a1e1b1j1wzida0e0b0xyg3,dmd_p6a1e1b1j1wzida0e0b0xyg3,supp_p6a1e1b1j1wzida0e0b0xyg3, \
    feedsupplyw_tpa1e1b1nwzida0e0b0xyg0,feedsupplyw_tpa1e1b1nwzida0e0b0xyg1,feedsupplyw_tpa1e1b1nwzida0e0b0xyg3, \
    confinementw_tpa1e1b1nwzida0e0b0xyg0,confinementw_tpa1e1b1nwzida0e0b0xyg1,confinementw_tpa1e1b1nwzida0e0b0xyg3 = \
    fsstk.f1_stock_fs(cr_sire,cr_dams,cr_offs,cu0_sire,cu0_dams,cu0_offs,a_p6_pa1e1b1nwzida0e0b0xyg,
                     period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1,
                     scan_management_pa1e1b1nwzida0e0b0xyg1, gbal_management_pa1e1b1nwzida0e0b0xyg1, wean_management_pa1e1b1nwzida0e0b0xyg1,
                     a_n_pa1e1b1nwzida0e0b0xyg1, a_n_pa1e1b1nwzida0e0b0xyg3, mask_p_offs_p, len_p, pkl_fs_info)


    #######################
    #start generator loops#
    #######################
    ##Start the LTW loop here so that the arrays are reinitialised from the inputs
    ### set the LTW adjustments to zero for the first loop. Sires do not have a LTW adjust because they are born off farm
    sfw_ltwadj_g0 = 1
    sfd_ltwadj_g0 = 0
    sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = np.ones(tpg1)
    sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = np.zeros(tpg1)
    sfw_ltwadj_g2 = 1
    sfd_ltwadj_g2 = 0
    sfw_ltwadj_ta1e1b1nwzida0e0b0xyg3 = np.ones(tpg3)[:,0, ...]  # slice the p axis to remove
    sfd_ltwadj_ta1e1b1nwzida0e0b0xyg3 = np.zeros(tpg3)[:,0, ...]  # slice the p axis to remove

    ## set whether it is necessary to loop for the LTW calculations. If both dams & offs are not used then don't loop
    if sen.sam['LTW_dams'] == 0 and sen.sam['LTW_offs'] == 0:
        loop_ltw_len = 1
    else:
        loop_ltw_len = 2

    for loop_ltw in range(loop_ltw_len):
        #todo The double loop could be replaced by separating the offspring into their own loop
        # it doesn't remove the requirement to loop for the dams because they need to have the first loop to generate the inputs for the second loop
        # but it would reduce the number of offspring calculations, allow offspring wean wt to be based on ffcfw_yat at weaning and allow loop length to be customised

        ####################################
        ### initialise arrays for sim loop  # axis names not always track from now on because they change between p=0 and p=1
        ####################################

        ##all groups
        eqn_compare = uinp.sheep['i_eqn_compare']
        ##sire
        ffcfw_start_sire = lw_initial_wzida0e0b0xyg0 - cfw_initial_wzida0e0b0xyg0 / cw_sire[3, ...]
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
        temp_lc_sire = np.array([0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.
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
        cf_cfwltw_dams = np.array([0.0]) #this is required as default when mu fleece function is not being called (it is required in the start production function)
        cf_fdltw_start_dams = np.array([0.0])
        cf_fdltw_dams = np.array([0.0]) #this is required as default when mu fleece function is not being called (it is required in the start production function)
        cf_w_b_start_dams = np.array([0.0])
        cf_w_b_dams = np.array([0.0]) #this is required as default when mu birth weight function is not being called (it is required in the start production function)
        cf_w_w_start_dams = np.array([0.0])
        cf_w_w_dams = np.array([0.0]) #this is required as default when mu wean function is not being called (it is required in the start production function)
        cf_conception_start_dams = np.array([0.0])
        cf_conception_dams = np.array([0.0]) #this is required as default when mu concep function is not being called (it is required in the start production function)
        conception_dams = 0.0 #initialise so it can be added to (conception += conception)
        guw_start_dams = np.array([0.0])
        rc_birth_start_dams = np.array([1.0])
        ffcfw_start_dams = fun.f_expand(lw_initial_wzida0e0b0xyg1 - cfw_initial_wzida0e0b0xyg1 / cw_dams[3, ...], p_pos, right_pos=w_pos) #add axis w to a1 because e and b axis are sliced before they are added via calculation
        ffcfw_max_start_dams = ffcfw_start_dams
        ffcfw_mating_dams = 0.0
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
        temp_lc_dams = np.array([0.0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_dams = numbers_initial_a1e1b1nwzida0e0b0xyg1
        numbers_start_condense_dams = numbers_initial_a1e1b1nwzida0e0b0xyg1 #just need a default because this is processed using update function.
        scanning = 0 #variable is used only for reporting
        # ebg_start_dams=0
        o_mortality_dams[...] = 0 #have to reset when doing the ltw loop because it is used to back date numbers

        ##yatf
        omer_history_start_p3g2[...] = np.nan
        d_cfw_history_start_p2g2[...] = np.nan
        nw_start_yatf = 0.0
        rc_start_yatf = 0.0
        ffcfw_start_yatf = w_b_std_y_b1nwzida0e0b0xyg1 #this is just an estimate, it is updated with the real weight at birth - needed to calc milk production in birth period because milk prod is calculated before yatf weight is updated)
        #todo will this cause an error for the second lambing because ffcfw_start_yatf will be last years weaning weight rather than this years expected birth weight - hard to see how the weight can be reset unless it is done the period after weaning
        ffcfw_max_start_yatf = ffcfw_start_yatf
        mortality_birth_yatf=0.0 #required for dam numbers before progeny born
        cfw_start_yatf = 0.0
        temp_lc_yatf = np.array([0.0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.
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
        w_w_yatf = 0.0
        foo_lact_ave_start = 0.0
        foo_lact_ave = 0.0
        aw_start_yatf = 0.0
        bw_start_yatf = 0.0
        mw_start_yatf = 0.0

        ##offs
        ffcfw_start_offs = lw_initial_wzida0e0b0xyg3 - cfw_initial_wzida0e0b0xyg3 / cw_offs[3, ...]
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
        temp_lc_offs = np.array([0.0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.
        numbers_start_offs = numbers_initial_ida0e0b0xyg3
        numbers_start_condense_offs = numbers_initial_ida0e0b0xyg3 #just need a default because this is processed using update function.
        # ebg_start_offs=0


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

        ## Loop through each week of the simulation (p) for ewes
        for p in range(n_sim_periods-1):   #-1 because assigns to [p+1] for start values
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
                cfw_start_sire = fun.f_update(cfw_start_sire, 0, period_is_shearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_sire = fun.f_update(fl_start_sire, fl_shear_yg0, period_is_shearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_sire = fun.f_update(fd_min_start_sire, 1000, period_is_shearing_pa1e1b1nwzida0e0b0xyg0[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)

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
                ###Carry forward conception
                cf_conception_start_dams = fun.f_update(cf_conception_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###LTW CFW adjustment carryover (running tally of LTW progeny CFW)
                cfw_ltwadj_start_dams = fun.f_update(cfw_ltwadj_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                cf_cfwltw_start_dams = fun.f_update(cf_cfwltw_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                ###LTW FD adjustment carryover (running tally of LTW progeny FD)
                fd_ltwadj_start_dams = fun.f_update(fd_ltwadj_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])
                cf_fdltw_start_dams = fun.f_update(cf_fdltw_start_dams, 0, period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])

            ##Wool Production
                ###cfw
                cfw_start_dams = fun.f_update(cfw_start_dams, 0, period_is_shearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_dams = fun.f_update(fl_start_dams, fl_shear_yg1, period_is_shearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_dams = fun.f_update(fd_min_start_dams, 1000, period_is_shearing_pa1e1b1nwzida0e0b0xyg1[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)

            ##offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###cfw
                cfw_start_offs = fun.f_update(cfw_start_offs, 0, period_is_shearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###fl
                fl_start_offs = fun.f_update(fl_start_offs, fl_shear_yg3, period_is_shearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)
                ###min fd
                fd_min_start_offs = fun.f_update(fd_min_start_offs, 1000, period_is_shearing_pa1e1b1nwzida0e0b0xyg3[p-1]) #reset if previous period is shearing (shearing occurs at the end of a period)



            ###################
            ##dependent start #
            ###################
            ##note: yatf calculated later in the code
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
                cs_start_sire = sfun.f1_condition_score(rc_start_sire, cu0_sire)
                ###staple length
                sl_start_sire = fl_start_sire * cw_sire[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_sire = np.minimum(1, nw_start_sire / srw_xyg0)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_sire = np.minimum(ffcfw_max_start_sire, nw_max_pa1e1b1nwzida0e0b0xyg0[p]) / srw_xyg0
                ###PI Size factor (for cattle)
                zf_sire = np.maximum(1, 1 + cr_sire[7, ...] - relsize_start_sire)
                ###EVG Size factor (decreases steadily - some uncertainty about the sign on cg[4])
                z1f_sire = 1 / (1 + np.exp(+cg_sire[4, ...] * (relsize1_start_sire - cg_sire[5, ...])))
                ###EVG Size factor (increases at maturity)
                z2f_sire = np.clip((relsize1_start_sire - cg_sire[6, ...]) / (cg_sire[7, ...] - cg_sire[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_kg_sire = fun.f_update(1, sen.sam['kg'], z2f_sire == 1)
                sam_mr_sire = fun.f_update(1, sen.sam['mr'], z2f_sire == 1)
                sam_pi_sire = fun.f_update(1, sen.sam['pi'], z2f_sire == 1)

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
                cs_start_dams = sfun.f1_condition_score(rc_start_dams, cu0_dams)
                ###Relative condition of dam at parturition - needs to be remembered between loops (milk production) - Loss of potential milk due to consistent under production
                rc_birth_dams = fun.f_update(rc_birth_start_dams, rc_start_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                ###staple length
                sl_start_dams = fl_start_dams * cw_dams[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_dams = np.minimum(1, nw_start_dams / srw_xyg1)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_dams = np.minimum(ffcfw_max_start_dams, nw_max_pa1e1b1nwzida0e0b0xyg1[p]) / srw_xyg1
                ###PI Size factor (for cattle)
                zf_dams = np.maximum(1, 1 + cr_dams[7, ...] - relsize_start_dams)
                ###EVG Size factor (decreases steadily - some uncertainty about the sign on cg[4])
                z1f_dams = 1 / (1 + np.exp(+cg_dams[4, ...] * (relsize1_start_dams - cg_dams[5, ...])))
                ###EVG Size factor (increases at maturity)
                z2f_dams = np.clip((relsize1_start_dams - cg_dams[6, ...]) / (cg_dams[7, ...] - cg_dams[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_kg_dams = fun.f_update(1, sen.sam['kg'], z2f_dams == 1)
                sam_mr_dams = fun.f_update(1, sen.sam['mr'], z2f_dams == 1)
                sam_pi_dams = fun.f_update(1, sen.sam['pi'], z2f_dams == 1)
                ##sires for mating
                n_sire_a1e1b1nwzida0e0b0xyg1g0p8 = sfun.f_sire_req(sire_propn_pa1e1b1nwzida0e0b0xyg1g0[p], sire_periods_g0p8, pinp.sheep['i_sire_recovery']
                                                                   , pinp.sheep['i_startyear'], date_end_pa1e1b1nwzida0e0b0xyg[p], period_is_join_pa1e1b1nwzida0e0b0xyg1[p])


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
                cs_start_offs = sfun.f1_condition_score(rc_start_offs, cu0_offs)
                ###staple length
                sl_start_offs = fl_start_offs * cw_offs[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_offs = np.minimum(1, nw_start_offs / srw_xyg3)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_offs = np.minimum(ffcfw_max_start_offs, nw_max_pa1e1b1nwzida0e0b0xyg3[p]) / srw_xyg3
                ###PI Size factor (for cattle)
                zf_offs = np.maximum(1, 1 + cr_offs[7, ...] - relsize_start_offs)
                ###EVG Size factor (decreases steadily - some uncertainty about the sign on cg[4])
                z1f_offs = 1 / (1 + np.exp(+cg_offs[4, ...] * (relsize1_start_offs - cg_offs[5, ...])))
                ###EVG Size factor (increases at maturity)
                z2f_offs = np.clip((relsize1_start_offs - cg_offs[6, ...]) / (cg_offs[7, ...] - cg_offs[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_kg_offs = fun.f_update(1, sen.sam['kg'], z2f_offs == 1)
                sam_mr_offs = fun.f_update(1, sen.sam['mr'], z2f_offs == 1)
                sam_pi_offs = fun.f_update(1, sen.sam['pi'], z2f_offs == 1)



            ##feed supply loop
            ##this loop is only required if a LW target is specified for the animals
            ##if there is a target then the loop needs to continue until
            ##the feed supply has converged on a value that generates a liveweight
            ##change close to the target
            ##The loop needs to execute at least once, then repeat if there
            ##is a target and the result is not close enough to the target

            ###initial info ^this will need to be hooked up with correct inputs, if they are the same for each period they don't need to be initialised below
            target_lwc = None
            epsilon = 0
            n_max_itn = sinp.stock['i_feedsupply_itn_max']
            attempts = 0 #initial

            for itn in range(n_max_itn):
                ##potential intake
                eqn_group = 4
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0 = sfun.f_potential_intake_cs(ci_sire, cl_sire, srw_xyg0, relsize_start_sire, rc_start_sire, temp_lc_sire
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
                        temp0 = sfun.f_potential_intake_cs(ci_dams, cl_dams, srw_xyg1, relsize_start_dams, rc_start_dams, temp_lc_dams
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
                        temp0 = sfun.f_potential_intake_cs(ci_offs, cl_offs, srw_xyg3, relsize_start_offs, rc_start_offs, temp_lc_offs
                                                           , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                           , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg3[p]
                                                           , sam_pi = sam_pi_offs)
                        if eqn_used:
                            pi_offs = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ###murdoch #todo function doesnt exist yet, add args when it is built
                eqn_system = 1 # mu = 1
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_xyg0)
                        if eqn_used:
                            pi_sire = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_xyg1)
                        if eqn_used:
                            pi_dams = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0 = sfun.f_potential_intake_mu(srw_xyg3)
                        if eqn_used:
                            pi_offs = temp0
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

                ##feedsupply - calculated after pi because pi required for intake_s
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
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    nv_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    foo_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    dmd_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    supp_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg1, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    mei_dams, foo_dams, dmd_dams, mei_solid_dams, md_solid_dams, md_herb_dams, intake_f_dams, intake_s_dams, mei_propn_milk_dams, mei_propn_supp_dams, mei_propn_herb_dams   \
                        = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                            , nv_a1e1b1j1wzida0e0b0xyg1, foo_a1e1b1j1wzida0e0b0xyg1
                                            , dmd_a1e1b1j1wzida0e0b0xyg1, supp_a1e1b1j1wzida0e0b0xyg1, pi_dams)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    nv_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    foo_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    dmd_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    supp_a1e1b1j1wzida0e0b0xyg3 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg3, a_p6_cut_pa1e1b1nwzida0e0b0xyg, axis=p_pos)[0] #[0] to remove singleton p axis
                    mei_offs, foo_offs, dmd_offs, mei_solid_offs, md_solid_offs, md_herb_offs, intake_f_offs, intake_s_offs, mei_propn_milk_offs, mei_propn_supp_offs, mei_propn_herb_offs   \
                        = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg3[:,p], confinementw_tpa1e1b1nwzida0e0b0xyg3[:,p]
                                            , nv_a1e1b1j1wzida0e0b0xyg3, foo_a1e1b1j1wzida0e0b0xyg3
                                            , dmd_a1e1b1j1wzida0e0b0xyg3, supp_a1e1b1j1wzida0e0b0xyg3, pi_offs)


                ##energy
                eqn_group = 7
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_sire, cx_sire[:,0:1,...], cm_sire, lw_start_sire, ffcfw_start_sire
                                                                    , mr_age_pa1e1b1nwzida0e0b0xyg0[p], mei_sire, omer_history_start_p3g0
                                                                    , days_period_pa1e1b1nwzida0e0b0xyg0[p], md_solid_sire, pinp.sheep['i_md_supp']
                                                                    , md_herb_sire, lgf_eff_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                                                    , dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness']
                                                                    , densityw_pa1e1b1nwzida0e0b0xyg0[p], foo_sire, confinementw_tpa1e1b1nwzida0e0b0xyg0[:,p]
                                                                    , intake_f_sire, dmd_sire, sam_kg = sam_kg_sire, sam_mr = sam_mr_sire)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_sire = temp1
                        if eqn_used:
                            meme_sire = temp0
                            km_sire = temp2
                            kg_fodd_sire = temp3
                            kg_supp_sire = temp4  # temp5 is not used for sires
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_dams, cx_dams[:,1:2,...], cm_dams, lw_start_dams, ffcfw_start_dams
                                                                    , mr_age_pa1e1b1nwzida0e0b0xyg1[p], mei_dams, omer_history_start_p3g1
                                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p], md_solid_dams, pinp.sheep['i_md_supp']
                                                                    , md_herb_dams, lgf_eff_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                                                    , dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness']
                                                                    , densityw_pa1e1b1nwzida0e0b0xyg1[p], foo_dams, confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                                                    , intake_f_dams, dmd_dams, sam_kg = sam_kg_dams, sam_mr = sam_mr_dams)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_dams = temp1
                        if eqn_used:
                            meme_dams = temp0
                            km_dams = temp2
                            kg_fodd_dams = temp3
                            kg_supp_dams = temp4
                            kl_dams = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_offs, cx_offs[:,mask_x,...], cm_offs, lw_start_offs, ffcfw_start_offs
                                                                    , mr_age_pa1e1b1nwzida0e0b0xyg3[p], mei_offs, omer_history_start_p3g3
                                                                    , days_period_pa1e1b1nwzida0e0b0xyg3[p], md_solid_offs, pinp.sheep['i_md_supp']
                                                                    , md_herb_offs, lgf_eff_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                                                    , dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness']
                                                                    , densityw_pa1e1b1nwzida0e0b0xyg3[p], foo_offs, confinementw_tpa1e1b1nwzida0e0b0xyg3[:,p]
                                                                    , intake_f_offs, dmd_offs, sam_kg = sam_kg_offs, sam_mr = sam_mr_offs)
                        ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                        omer_history_offs = temp1
                        if eqn_used:
                            meme_offs = temp0
                            km_offs = temp2
                            kg_fodd_offs = temp3
                            kg_supp_offs = temp4 # temp5 is not used for offspring
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained


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
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            # r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1



                ##milk production
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    ###Expected ffcfw of yatf with p1 axis - each period
                    ffcfw_exp_a1e1b1nwzida0e0b0xyg2p1 = (ffcfw_start_yatf[..., na] + (index_p1 * cn_yatf[7, ...][...,na])) * (
                                index_p1 < days_period_pa1e1b1nwzida0e0b0xyg2[...,na][p])
                    ###Expected average metabolic LW of yatf during period
                    ffcfw75_exp_yatf = np.sum(ffcfw_exp_a1e1b1nwzida0e0b0xyg2p1 ** 0.75, axis=-1) / np.maximum(1, days_period_pa1e1b1nwzida0e0b0xyg2[p, ...])

                    mp2_dams, mel_dams, nel_dams, ldr_dams, lb_dams \
                        = sfun.f_milk(cl_dams, srw_xyg1, relsize_start_dams, rc_birth_dams, mei_dams, meme_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p]
                            , rc_start_dams, ffcfw75_exp_yatf, lb_start_dams, ldr_start_dams, age_pa1e1b1nwzida0e0b0xyg2[p]
                            , mp_age_y_pa1e1b1nwzida0e0b0xyg1[p], mp2_age_y_pa1e1b1nwzida0e0b0xyg1[p], x_pos
                            , days_period_pa1e1b1nwzida0e0b0xyg2[p], kl_dams, lact_nut_effect_pa1e1b1nwzida0e0b0xyg1[p])
                    mp2_yatf = fun.f_divide(mp2_dams, nyatf_b1nwzida0e0b0xyg) # 0 if given slice of b1 axis has no yatf

                ##wool production
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    d_cfw_sire, d_fd_sire, d_fl_sire, d_cfw_history_sire_p2, mew_sire, new_sire  \
                        = sfun.f_fibre(cw_sire, cc_sire, ffcfw_start_sire, relsize_start_sire, d_cfw_history_start_p2g0
                                       , mei_sire, mew_min_pa1e1b1nwzida0e0b0xyg0[p]
                                       , d_cfw_ave_pa1e1b1nwzida0e0b0xyg0[p, ...],  sfd_a0e0b0xyg0, wge_a0e0b0xyg0
                                       , af_wool_pa1e1b1nwzida0e0b0xyg0[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg0[p, ...]
                                       , kw_yg0, days_period_pa1e1b1nwzida0e0b0xyg0[p], sfw_ltwadj_g0, sfd_ltwadj_g0
                                       , rev_trait_values['sire'][p], sam_pi = sam_pi_sire)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    d_cfw_dams, d_fd_dams, d_fl_dams, d_cfw_history_dams_p2, mew_dams, new_dams  \
                        = sfun.f_fibre(cw_dams, cc_dams, ffcfw_start_dams, relsize_start_dams, d_cfw_history_start_p2g1
                                       , mei_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p]
                                       , d_cfw_ave_pa1e1b1nwzida0e0b0xyg1[p, ...], sfd_a0e0b0xyg1, wge_a0e0b0xyg1
                                       , af_wool_pa1e1b1nwzida0e0b0xyg1[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg1[p, ...]
                                       , kw_yg1, days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                       , sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[:,p, ...], sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[:,p, ...]
                                       , rev_trait_values['dams'][p]
                                       , mec_dams, mel_dams, gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                       , lact_propn_pa1e1b1nwzida0e0b0xyg1[p], sam_pi = sam_pi_dams)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    d_cfw_offs, d_fd_offs, d_fl_offs, d_cfw_history_offs_p2, mew_offs, new_offs  \
                        = sfun.f_fibre(cw_offs, cc_offs, ffcfw_start_offs, relsize_start_offs, d_cfw_history_start_p2g3
                                       , mei_offs, mew_min_pa1e1b1nwzida0e0b0xyg3[p]
                                       , d_cfw_ave_pa1e1b1nwzida0e0b0xyg3[p, ...], sfd_da0e0b0xyg3, wge_da0e0b0xyg3
                                       , af_wool_pa1e1b1nwzida0e0b0xyg3[p, ...], dlf_wool_pa1e1b1nwzida0e0b0xyg3[p, ...]
                                       , kw_yg3, days_period_pa1e1b1nwzida0e0b0xyg3[p]
                                       , sfw_ltwadj_ta1e1b1nwzida0e0b0xyg3, sfd_ltwadj_ta1e1b1nwzida0e0b0xyg3
                                       , rev_trait_values['offs'][p], sam_pi = sam_pi_offs)

                ##energy to offset chilling
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    mem_sire, temp_lc_sire, kg_sire = sfun.f_chill_cs(cc_sire, ck_sire, ffcfw_start_sire, rc_start_sire, sl_start_sire, mei_sire
                                                            , meme_sire, mew_sire, new_sire, km_sire, kg_supp_sire, kg_fodd_sire, mei_propn_supp_sire
                                                            , mei_propn_herb_sire, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                            , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp1[p]
                                                            , index_m0)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    mem_dams, temp_lc_dams, kg_dams = sfun.f_chill_cs(cc_dams, ck_dams, ffcfw_start_dams, rc_start_dams, sl_start_dams, mei_dams
                                                            , meme_dams, mew_dams, new_dams, km_dams, kg_supp_dams, kg_fodd_dams, mei_propn_supp_dams
                                                            , mei_propn_herb_dams, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                            , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp1[p]
                                                            , index_m0, guw = guw_dams, kl = kl_dams, mei_propn_milk = mei_propn_milk_dams, mec = mec_dams
                                                            , mel = mel_dams, nec = nec_dams, nel = nel_dams, gest_propn = gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                                            , lact_propn = lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    mem_offs, temp_lc_offs, kg_offs = sfun.f_chill_cs(cc_offs, ck_offs, ffcfw_start_offs, rc_start_offs, sl_start_offs, mei_offs
                                                            , meme_offs, mew_offs, new_offs, km_offs, kg_supp_offs, kg_fodd_offs, mei_propn_supp_offs
                                                            , mei_propn_herb_offs, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                            , temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp1[p]
                                                            , index_m0)

                ##calc lwc
                eqn_group = 8
                eqn_system = 0 # CSIRO = 0
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_cs(cg_sire, rc_start_sire, mei_sire
                                                                , mem_sire, mew_sire, z1f_sire, z2f_sire, kg_sire, rev_trait_values['sire'][p])
                        if eqn_used:
                            ebg_sire = temp0
                            evg_sire = temp1
                            pg_sire = temp2
                            fg_sire = temp3
                            level_sire = temp4
                            surplus_energy_sire = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_cs(cg_dams, rc_start_dams, mei_dams
                                                , mem_dams, mew_dams, z1f_dams, z2f_dams, kg_dams, rev_trait_values['dams'][p], mec_dams, mel_dams
                                                , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            ebg_dams = temp0
                            evg_dams = temp1
                            pg_dams = temp2
                            fg_dams = temp3
                            level_dams = temp4
                            surplus_energy_dams = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_cs(cg_offs, rc_start_offs, mei_offs
                                                                , mem_offs, mew_offs, z1f_offs, z2f_offs, kg_offs, rev_trait_values['offs'][p])
                        if eqn_used:
                            ebg_offs = temp0
                            evg_offs = temp1
                            pg_offs = temp2
                            fg_offs = temp3
                            level_offs = temp4
                            surplus_energy_offs = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp1

                eqn_system = 1 # Murdoch = 1
                if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                    ###sire
                    eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_mu(cg_sire, rc_start_sire, mei_sire
                                                                , mem_sire, mew_sire, z1f_sire, z2f_sire, kg_sire, rev_trait_values['sire'][p])
                        if eqn_used:
                            ebg_sire = temp0
                            evg_sire = temp1
                            pg_sire = temp2
                            fg_sire = temp3
                            level_sire = temp4
                            surplus_energy_sire = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpsire[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###dams
                    eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_mu(cg_dams, rc_start_dams, mei_dams
                                                , mem_dams, mew_dams, z1f_dams, z2f_dams, kg_dams, rev_trait_values['dams'][p], mec_dams, mel_dams
                                                , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                        if eqn_used:
                            ebg_dams = temp0
                            evg_dams = temp1
                            pg_dams = temp2
                            fg_dams = temp3
                            level_dams = temp4
                            surplus_energy_dams = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpdams[eqn_system, eqn_group, 1, :, p, ...] = temp1
                    ###offs
                    eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                    if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                        temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_mu(cg_offs, rc_start_offs, mei_offs
                                                                , mem_offs, mew_offs, z1f_offs, z2f_offs, kg_offs, rev_trait_values['offs'][p])
                        if eqn_used:
                            ebg_offs = temp0
                            evg_offs = temp1
                            pg_offs = temp2
                            fg_offs = temp3
                            level_offs = temp4
                            surplus_energy_offs = temp5
                        if eqn_compare:
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0
                            r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 1, :, p, ...] = temp1



                ###if there is a target then adjust feedsupply, if not break out of feedsupply loop
                if target_lwc == None:
                    break
                ###calc error
                error = (ebg_dams * cg_dams) - target_lwc
                ###store in attempts array - build new array assign old array and then add current itn results - done like this to handle the shape changing and because we don't know what shape feedsupply and error are before this loop starts
                shape = tuple(np.maximum.reduce([feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p].shape, error.shape]))+(n_max_itn,)+(2,)
                attempts2= np.zeros(shape)
                attempts2[...] = attempts
                attempts2[...,itn,0] = feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                attempts2[...,itn,1] = error
                attempts = attempts2
                ###is error within tolerance
                if np.all(np.abs(error) <= epsilon):
                    break
                ###max attempts reached
                elif itn == n_max_itn-1: #minus 1 because range() and hence itn starts from 0
                    ####select best feed supply option
                    feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p] = attempts[...,attempts[...,1]==np.nanmin(np.abs(attempts[...,1]),axis=-1),0] #create boolean index using error array then index feedsupply array
                    break
                feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p:p+1] = sfun.f1_feedsupply_adjust(attempts,feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p],itn)
                itn+=1

            ##dam weight at a given time during period - used for special events like birth.
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ##Dam weight at mating - to estimate the weight at mating we are wanting to use the growth rate of the dams that are not yet pregnant
                ## because mating doesnt happen at the start of the period.
                ##relative size and relative condition of the dams at mating are the determinants of conception
                ## use the condition of dams in the 11 slice because mated animals can have a different feed supply
                ## use dams in e[-1] because want the condition of the animal before it concieves. Note all e slices will have the same condition until concieved because they have the same feedsupply until scanning.
                ffcfw_e1b1sliced = fun.f_dynamic_slice(ffcfw_start_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                ebg_e1b1sliced = fun.f_dynamic_slice(ebg_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                nw_start_dams_e1b1sliced = fun.f_dynamic_slice(nw_start_dams, e1_pos, -1, None, b1_pos, 2, 3) #slice e1 & b1 axis
                gest_propn_b1sliced = fun.f_dynamic_slice(gest_propn_pa1e1b1nwzida0e0b0xyg1[p], b1_pos, 2, 3) #slice b1 axis
                days_period_b1sliced = fun.f_dynamic_slice(days_period_pa1e1b1nwzida0e0b0xyg1[p], b1_pos, 2, 3) #slice b1 axis

                t_w_mating = np.sum((ffcfw_e1b1sliced + ebg_e1b1sliced * cg_dams[18, ...] * days_period_b1sliced * (1-gest_propn_b1sliced)) \
                             * period_is_mating_pa1e1b1nwzida0e0b0xyg1[p], axis=e1_pos, keepdims=True)#Temporary variable for mating weight
                ffcfw_mating_dams = fun.f_update(ffcfw_mating_dams, t_w_mating, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p])
                ##Relative condition of the dam at mating - required to determine milk production
                rc_mating_dams = ffcfw_mating_dams / nw_start_dams_e1b1sliced
                ##Condition score of the dams at mating
                cs_mating_dams = sfun.f1_condition_score(rc_mating_dams, cu0_dams)
                ##Relative size of the dams at mating
                relsize_mating_dams = relsize_start_dams

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
                    temp0, temp1 = sfun.f_birthweight_mu(cu1_yatf, cb1_yatf, cg_yatf, cx_yatf[:,mask_x,...], ce_yatf[:,p-1,...]
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
            eqn_group = 13
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
                ###normal weight of yatf
                nw_start_yatf	= fun.f_update(nw_start_yatf, w_b_yatf, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###adipose weight of yatf
                aw_start_yatf	= fun.f_update(aw_start_yatf, w_b_yatf * aw_propn_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###muscle weight of yatf
                mw_start_yatf	= fun.f_update(mw_start_yatf, w_b_yatf * mw_propn_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###bone weight of the yatf
                bw_start_yatf	= fun.f_update(bw_start_yatf, w_b_yatf * bw_propn_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###clean fleece weight of yatf
                cfw_start_yatf	= fun.f_update(cfw_start_yatf, 0, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###fibre diameter of yatf
                fd_start_yatf	= fun.f_update(fd_start_yatf, 0, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###fibre length of yatf
                fl_start_yatf	= fun.f_update(fl_start_yatf, fl_birth_yg2, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])
                ###minimum fibre diameter of yatf
                fd_min_start_yatf	= fun.f_update(fd_min_start_yatf, 1000, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...])

                ##Yatf dependent start values
                ###Normal weight max (if animal is well fed) - yatf
                nw_max_yatf	= srw_xyg2 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg2[p]) + w_b_yatf * srw_age_pa1e1b1nwzida0e0b0xyg2[p]
                ##Dependent start: Change in normal weight max - yatf
                d_nw_max_yatf = fun.f_divide(srw_age_pa1e1b1nwzida0e0b0xyg2[p-1, ...] - srw_age_pa1e1b1nwzida0e0b0xyg2[p, ...] * (srw_xyg2 - w_b_yatf)
                                             , days_period_pa1e1b1nwzida0e0b0xyg2[p]) #nw_max = srw - (srw - bw) * srw_age[p] so d_nw_max = (srw - (srw-bw) * srw_age[p]) - (srw - (srw - bw) * srw_age[p-1]) and that simplifies to d_nw_max = (srw_age[p-1] - srw_age[p]) * (srw-bw)
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
                cs_start_yatf = sfun.f1_condition_score(rc_start_yatf, cu0_yatf)
                ###staple length
                sl_start_yatf = fl_start_yatf * cw_yatf[15,...]
                ###Relative size (start) - dams & sires
                relsize_start_yatf = np.minimum(1, nw_start_yatf / srw_xyg2)
                ###Relative size for LWG (start). Capped by current LW
                relsize1_start_yatf = np.minimum(ffcfw_max_start_yatf, nw_max_yatf) / srw_xyg2
                ###PI Size factor (for cattle)
                zf_yatf = np.maximum(1, 1 + cr_yatf[7, ...] - relsize_start_yatf)
                ###EVG Size factor (decreases steadily - some uncertainty about the sign on cg[4])
                z1f_yatf = 1 / (1 + np.exp(+cg_yatf[4, ...] * (relsize1_start_yatf - cg_yatf[5, ...])))
                ###EVG Size factor (increases at maturity)
                z2f_yatf = np.clip((relsize1_start_yatf - cg_yatf[6, ...]) / (cg_yatf[7, ...] - cg_yatf[6, ...]), 0 ,1)
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults (only included here for consistency)
                sam_kg_yatf = fun.f_update(1, sen.sam['kg'], z2f_yatf == 1)
                sam_mr_yatf = fun.f_update(1, sen.sam['mr'], z2f_yatf == 1)
                sam_pi_yatf = fun.f_update(1, sen.sam['pi'], z2f_yatf == 1)


            ##potential intake - yatf
            eqn_group = 4
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0 = sfun.f_potential_intake_cs(ci_yatf, cl_yatf, srw_xyg2, relsize_start_yatf, rc_start_yatf, temp_lc_yatf
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
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                # nv_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(nv_p6a1e1b1j1wzida0e0b0xyg1,a_p6_cut_pa1e1b1nwzida0e0b0xyg,axis=p_pos)[0]  # [0] to remove singleton p axis
                # foo_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(foo_p6a1e1b1j1wzida0e0b0xyg1,a_p6_cut_pa1e1b1nwzida0e0b0xyg,axis=p_pos)[0]  # [0] to remove singleton p axis
                # dmd_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(dmd_p6a1e1b1j1wzida0e0b0xyg1,a_p6_cut_pa1e1b1nwzida0e0b0xyg,axis=p_pos)[0]  # [0] to remove singleton p axis
                # supp_a1e1b1j1wzida0e0b0xyg1 = np.take_along_axis(supp_p6a1e1b1j1wzida0e0b0xyg1,a_p6_cut_pa1e1b1nwzida0e0b0xyg,axis=p_pos)[0]  # [0] to remove singleton p axis
                mei_yatf,foo_yatf,dmd_yatf,mei_solid_yatf, md_solid_yatf,md_herb_yatf,intake_f_yatf,intake_s_yatf,mei_propn_milk_yatf,mei_propn_supp_yatf,mei_propn_herb_yatf \
                    = sfun.f1_feedsupply(feedsupplyw_tpa1e1b1nwzida0e0b0xyg1[:,p],confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p]
                                         ,nv_a1e1b1j1wzida0e0b0xyg1,foo_a1e1b1j1wzida0e0b0xyg1
                                         ,dmd_a1e1b1j1wzida0e0b0xyg1,supp_a1e1b1j1wzida0e0b0xyg1,pi_yatf, mp2_yatf)

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
            #
            #
            # ##relative ingestibility (quality) - yatf
            # eqn_group = 6
            # eqn_system = 0 # CSIRO = 0
            # if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            #     ###sire
            #     eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
            #     if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
            #         temp0 = fsfun.f_rq_cs(dmd_yatf, legume_pa1e1b1nwzida0e0b0xyg[p], cr_yatf, pinp.sheep['i_sf'])
            #         if eqn_used:
            #             rq_yatf = temp0
            #         if eqn_compare:
            #             r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
            #

            # ##intake - yatf
            # if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
            #     ri_yatf = fsfun.f_rel_intake(ra_yatf, rq_yatf, legume_pa1e1b1nwzida0e0b0xyg[p], cr_yatf)
            #     mei_yatf, mei_solid_yatf, intake_f_yatf, md_solid_yatf, mei_propn_milk_yatf, mei_propn_herb_yatf, mei_propn_supp_yatf \
            #                 = sfun.f_intake(pi_yatf, ri_yatf, md_herb_yatf, feedsupplyw_pa1e1b1nwzida0e0b0xyg1[p]
            #                                 , intake_s_yatf, pinp.sheep['i_md_supp'], mp2_yatf)   #same feedsupply as dams

            ##energy - yatf
            eqn_group = 7
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_yatf, cx_yatf[:,mask_x,...], cm_yatf, lw_start_yatf
                                                                , ffcfw_start_yatf, mr_age_pa1e1b1nwzida0e0b0xyg2[p], mei_yatf
                                                                , omer_history_start_p3g2, days_period_pa1e1b1nwzida0e0b0xyg2[p]
                                                                , md_solid_yatf, pinp.sheep['i_md_supp'], md_herb_yatf
                                                                , lgf_eff_pa1e1b1nwzida0e0b0xyg2[p], dlf_eff_pa1e1b1nwzida0e0b0xyg[p]
                                                                , pinp.sheep['i_steepness'], densityw_pa1e1b1nwzida0e0b0xyg2[p]
                                                                , foo_yatf, confinementw_tpa1e1b1nwzida0e0b0xyg1[:,p], intake_f_yatf
                                                                , dmd_yatf, mei_propn_milk_yatf, sam_kg_yatf, sam_mr = sam_mr_yatf)  #same feedsupply as dams
                    ## these variables need to be stored even if the equation system is not used so that the equations can be compared
                    omer_history_yatf = temp1
                    if eqn_used:
                        meme_yatf = temp0
                        km_yatf = temp2
                        kg_fodd_yatf = temp3
                        kg_supp_yatf = temp4  # temp5 is not used for yatf
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0  # more of the return variable could be retained


            ##wool production - yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                d_cfw_yatf, d_fd_yatf, d_fl_yatf, d_cfw_history_yatf_p2, mew_yatf, new_yatf  \
                    = sfun.f_fibre(cw_yatf, cc_yatf, ffcfw_start_yatf, relsize_start_yatf, d_cfw_history_start_p2g2
                                   , mei_yatf, mew_min_pa1e1b1nwzida0e0b0xyg2[p]
                                   , d_cfw_ave_pa1e1b1nwzida0e0b0xyg2[p, ...], sfd_pa1e1b1nwzida0e0b0xyg2[p]
                                   , wge_pa1e1b1nwzida0e0b0xyg2[p], af_wool_pa1e1b1nwzida0e0b0xyg2[p, ...]
                                   , dlf_wool_pa1e1b1nwzida0e0b0xyg2[p, ...], kw_yg2
                                   , days_period_pa1e1b1nwzida0e0b0xyg2[p], sfw_ltwadj_g2, sfd_ltwadj_g2
                                   , rev_trait_values['yatf'][p], sam_pi = sam_pi_yatf)


            ##energy to offset chilling - yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                mem_yatf, temp_lc_yatf, kg_yatf = sfun.f_chill_cs(cc_yatf, ck_yatf, ffcfw_start_yatf, rc_start_yatf, sl_start_yatf, mei_yatf,
                                                                  meme_yatf, mew_yatf, new_yatf, km_yatf, kg_supp_yatf, kg_fodd_yatf, mei_propn_supp_yatf,
                                                                  mei_propn_herb_yatf, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p],
                                                                  temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygp1[p],
                                                                  index_m0,  mei_propn_milk=mei_propn_milk_yatf)


            ##calc lwc - yatf
            eqn_group = 8
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_cs(cg_yatf, rc_start_yatf, mei_yatf, mem_yatf
                                                                             , mew_yatf, z1f_yatf, z2f_yatf, kg_yatf, rev_trait_values['yatf'][p])
                    if eqn_used:
                        ebg_yatf = temp0
                        evg_yatf = temp1
                        pg_yatf = temp2
                        fg_yatf = temp3
                        level_yatf = temp4
                        surplus_energy_yatf = temp5
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1

            eqn_system = 1 # Mu = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_lwc_mu(cg_yatf, rc_start_yatf, mei_yatf, mem_yatf
                                                                             , mew_yatf, z1f_yatf, z2f_yatf, kg_yatf, rev_trait_values['yatf'][p])
                    if eqn_used:
                        ebg_yatf = temp0
                        evg_yatf = temp1
                        pg_yatf = temp2
                        fg_yatf = temp3
                        level_yatf = temp4
                        surplus_energy_yatf = temp5
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 1, :, p, ...] = temp1

            ##weaning weight yatf - called when dams days per period greater than 0 - calculates the weight at the start of the period
            eqn_group = 11
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)  # equation used is based on the yatf system
                ##based on days_period_dams because weaning occurs at start of period so days_period_yatf==0
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_weanweight_cs(w_w_start_yatf, ffcfw_start_yatf, ebg_yatf, days_period_pa1e1b1nwzida0e0b0xyg2[p]
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
                                            , ce_yatf[:,p-1,...], nyatf_b1nwzida0e0b0xyg, w_w_start_yatf, cf_w_w_start_dams
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



            ##emissions
            eqn_group = 12
            eqn_system = 0 # Baxter and Clapperton = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                ###sire
                eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                    temp0, temp1 = sfun.f_emissions_bc(ch_sire, intake_f_sire, intake_s_sire, md_solid_sire, level_sire)
                    if eqn_used:
                        ch4_total_sire = temp0
                        ch4_animal_sire = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###dams
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0, temp1 = sfun.f_emissions_bc(ch_dams, intake_f_dams, intake_s_dams, md_solid_dams, level_dams)
                    if eqn_used:
                        ch4_total_dams = temp0
                        ch4_animal_dams = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###yatf
                eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                    temp0, temp1 = sfun.f_emissions_bc(ch_yatf, intake_f_yatf, intake_s_yatf, md_solid_yatf, level_yatf)
                    if eqn_used:
                        ch4_total_yatf = temp0
                        ch4_animal_yatf = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpyatf[eqn_system, eqn_group, 0, :, p, ...] = temp0
                ###offs
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0, temp1 = sfun.f_emissions_bc(ch_offs, intake_f_offs, intake_s_offs, md_solid_offs, level_offs)
                    if eqn_used:
                        ch4_total_offs = temp0
                        ch4_animal_offs = temp1
                    if eqn_compare:
                        r_compare_q0q1q2tpoffs[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##conception Dams
            eqn_group = 0
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_conception_cs(cf_dams, cb1_dams, relsize_mating_dams, rc_mating_dams
                                                 , crg_doy_pa1e1b1nwzida0e0b0xyg1[p], nfoet_b1nwzida0e0b0xyg
                                                 , nyatf_b1nwzida0e0b0xyg, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                                 , index_e1b1nwzida0e0b0xyg, rev_trait_values['dams'][p]
                                                 , saa_rr_age_pa1e1b1nwzida0e0b0xyg1[p])
                    if eqn_used:
                        conception_dams =  temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # MU LTW = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    #todo this need to be replaced by LMAT formula, if cf_conception_start is used in the LMAT formula cf_conception_dams = temp0 will need to be moved out of the if used statement.
                    temp0 = sfun.f_conception_ltw(cf_dams, cu0_dams, relsize_mating_dams, cs_mating_dams
                                                  , scan_std_pa1e1b1nwzida0e0b0xyg1, doy_pa1e1b1nwzida0e0b0xyg[p]
                                                  , crg_doy_pa1e1b1nwzida0e0b0xyg1[p], nfoet_b1nwzida0e0b0xyg
                                                  , nyatf_b1nwzida0e0b0xyg, period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]
                                                  , index_e1b1nwzida0e0b0xyg, rev_trait_values['dams'][p])
                    if eqn_used:
                        cf_conception_dams = temp0*0  #default set to 0 because required in start production function (only used in lmat conception function)
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
            eqn_group = 14
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
                    temp0 = sfun.f_mortality_weaner_mu()
                    if eqn_used:
                        mortality_sire += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpsire[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####dams
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_mu()
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            ####offs
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                    temp0 = sfun.f_mortality_weaner_mu()
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
                    temp0 = sfun.f_mortality_dam_mu(cu2_dams, cs_start_dams, cv_cs_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , nfoet_b1nwzida0e0b0xyg, sen.sap['mortalitye'])
                    if eqn_used:
                        mortality_dams += temp0 #dam mort at birth due to low CS
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0

            ##preg tox Dam mortality - comments about mortality functions can be found in sfun.
            eqn_group = 15
            eqn_system = 0 # CSIRO = 0
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_pregtox_cs(cb1_dams, cg_dams, nw_start_dams, ebg_dams, sd_ebg_dams, days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], sen.sap['mortalitye'])
                    if eqn_used:
                        mortality_dams += temp0
                    if eqn_compare:
                        r_compare_q0q1q2tpdams[eqn_system, eqn_group, 0, :, p, ...] = temp0
            eqn_system = 1 # mu = 1
            if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
                eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
                if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                    temp0 = sfun.f_mortality_pregtox_cs(cb1_dams, cg_dams, nw_start_dams, ebg_dams, sd_ebg_dams
                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , gest_propn_pa1e1b1nwzida0e0b0xyg1[p], sen.sap['mortalitye'])
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
                    temp0, temp1, temp2 = sfun.f_mortality_progeny_cs(cd_yatf, cb1_yatf, w_b_yatf, rc_start_dams, cv_weight_yatf
                                    , w_b_exp_y_dams, period_is_birth_pa1e1b1nwzida0e0b0xyg1[p]
                                    , chill_index_pa1e1b1nwzida0e0b0xygp1[p], nfoet_b1nwzida0e0b0xyg
                                    , rev_trait_values['yatf'][p], sen.sap['mortalityp'], saa_mortalityx_b1nwzida0e0b0xyg)
                    if eqn_used:
                        mortality_birth_yatf = temp1 #mortalityd, assign first because it has x axis
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
                                                    , ce_yatf[:,p-1,...], w_b_ltw_std_yatf, 0, nw_start_dams, 0
                                                    , days_period_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_mated90_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_between_d90birth_pa1e1b1nwzida0e0b0xyg1[p]
                                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                    temp0 = sfun.f_mortality_progeny_mu(cu2_yatf, cb1_yatf, cx_yatf[:,mask_x,...], ce_yatf[:,p,...]
                                    , w_b_yatf, w_b_ltw_std_yatf, cv_weight_yatf, foo_yatf, chill_index_pa1e1b1nwzida0e0b0xygp1[p]
                                    , period_is_birth_pa1e1b1nwzida0e0b0xyg1[p], rev_trait_values['yatf'][p]
                                    , sen.sap['mortalityp'], saa_mortalityx_b1nwzida0e0b0xyg)
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
                             propn_dams_mated=prop_dams_mated_pa1e1b1nwzida0e0b0xyg1[p])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                numbers_end_yatf = sfun.f1_period_end_nums(numbers_start_yatf, mortality_yatf, group=2)
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                numbers_end_offs = sfun.f1_period_end_nums(numbers_start_offs, mortality_offs, group=3)

            ##################################################
            #post calculation sensitivity for intake & energy#
            ##################################################
            ##These sensitivities alter potential intake and me intake required without altering the liveweight profile.
            ##The aim is to have the same effect as using the within function sa and then altering the feed supply to
            ##  generate the same LW profile, they just require less work to test the effect of not altering LW profile (as per old MIDAS)
            ##To do this requires altering d_cfw and d_fl. These would both change in f_fibre if using the within function sa
            ##Adjustments must be made for pi_post because wge is scaled by sam_pi in f_fibre (rather than scaling sfw in the main code)
            ##Adjustments are required for mr_post and kg_post because these both reduce the mei required in this section. If cfw and fl
            ## were not scaled it would imply an increased wool growth efficiency (and this would be inconsistent with the in function sa)

            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post'], z2f_sire == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post'], z2f_sire == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post'], z2f_sire == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_sire = fun.f_sa(pi_sire, sam_pi)
                #### alter mei
                mei_solid_sire = mei_solid_sire + (mem_sire * sap_mr
                                                   - surplus_energy_sire * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_sire, mei_solid_sire)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_sire, mei_solid_sire)
                d_cfw_sire = d_cfw_sire / sam_pi
                d_fl_sire = d_fl_sire / sam_pi
                d_cfw_sire = d_cfw_sire / scalar_mr
                d_fl_sire = d_fl_sire / scalar_mr
                d_cfw_sire = d_cfw_sire / scalar_kg
                d_fl_sire = d_fl_sire / scalar_kg

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post'], z2f_dams == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post'], z2f_dams == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post'], z2f_dams == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_dams = fun.f_sa(pi_dams, sam_pi)
                #### alter mei
                mei_solid_dams = mei_solid_dams + (mem_dams * sap_mr
                                                   - surplus_energy_dams * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_dams, mei_solid_dams)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_dams, mei_solid_dams)
                d_cfw_dams = d_cfw_dams / sam_pi
                d_fl_dams = d_fl_dams / sam_pi
                d_cfw_dams = d_cfw_dams / scalar_mr
                d_fl_dams = d_fl_dams / scalar_mr
                d_cfw_dams = d_cfw_dams / scalar_kg
                d_fl_dams = d_fl_dams / scalar_kg

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post'], z2f_yatf == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post'], z2f_yatf == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post'], z2f_yatf == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_yatf = fun.f_sa(pi_yatf, sam_pi)
                #### alter mei (only calculate impact on mei_solid because that passes to the mei parameter in the matrix)
                #### this is an error in this application because all the change in energy is related to pasture and none to milk)
                mei_solid_yatf = mei_solid_yatf + (mem_yatf * sap_mr
                                                   - surplus_energy_yatf * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change (use mei rather than mei_solid so it is change as a proportion of total mei)
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_yatf, mei_yatf)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_yatf, mei_yatf)
                d_cfw_yatf = d_cfw_yatf / sam_pi
                d_fl_yatf = d_fl_yatf / sam_pi
                d_cfw_yatf = d_cfw_yatf / scalar_mr
                d_fl_yatf = d_fl_yatf / scalar_mr
                d_cfw_yatf = d_cfw_yatf / scalar_kg
                d_fl_yatf = d_fl_yatf / scalar_kg

            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###sensitivity on kg (efficiency of gain), MR (maintenance req) and PI (Potential intake) based on z2f - the sensitivity is only for adults
                sam_pi = fun.f_update(1, sen.sam['pi_post'], z2f_offs == 1)   #potential intake
                sap_mr = fun.f_update(0, sen.sap['mr_post'], z2f_offs == 1)   #maintenance energy (MEm - doesn't include gestation and lactation requirements)
                sap_kg = fun.f_update(0, sen.sap['kg_post'], z2f_offs == 1)   #efficiency of gain (kg)
                #### alter potential intake
                pi_offs = fun.f_sa(pi_offs, sam_pi)
                #### alter mei
                mei_solid_offs = mei_solid_offs + (mem_offs * sap_mr
                                                   - surplus_energy_offs * sap_kg / (1 + sap_kg))
                ####alter wool production as energy params change
                scalar_mr = 1 + sap_mr * fun.f_divide(mem_offs, mei_solid_offs)
                scalar_kg = 1 - sap_kg / (1 + sap_kg) * fun.f_divide(surplus_energy_offs, mei_solid_offs)
                d_cfw_offs = d_cfw_offs / sam_pi
                d_fl_offs = d_fl_offs / sam_pi
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
                ##Weight of fat adipose (end)
                aw_sire = aw_start_sire + fg_sire / cg_sire[20, ...] * days_period_pa1e1b1nwzida0e0b0xyg0[p]
                ##Weight of muscle (end)
                mw_sire = mw_start_sire + pg_sire / cg_sire[19, ...] * days_period_pa1e1b1nwzida0e0b0xyg0[p]
                ##Weight of bone (end)	bw ^formula needs finishing
                bw_sire = bw_start_sire
                ##Weight of water (end)
                ww_sire = mw_sire * (1 - cg_sire[19, ...]) + aw_sire * (1 - cg_sire[20, ...])
                ##Weight of gutfill (end)
                gw_sire = ffcfw_sire* (1 - 1 / cg_sire[18, ...])
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
                sl_sire = (fl_sire - fl_shear_yg0) / cw_sire[15, ...]
                ##Staple strength if shorn(end)
                ss_sire = fd_min_sire**2 / fd_sire **2 * cw_sire[16, ...]



            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_dams = np.maximum(0,ffcfw_start_dams + cg_dams[18, ...] * ebg_dams * days_period_pa1e1b1nwzida0e0b0xyg1[p])
                ##FFCFW maximum to date
                ffcfw_max_dams = np.maximum(ffcfw_dams, ffcfw_max_start_dams)
                ##Weight of fat adipose (end)
                aw_dams = aw_start_dams + fg_dams / cg_dams[20, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p]
                ##Weight of muscle (end)
                mw_dams = mw_start_dams + pg_dams / cg_dams[19, ...] * days_period_pa1e1b1nwzida0e0b0xyg1[p]
                ##Weight of bone (end)	bw ^formula needs finishing
                bw_dams = bw_start_dams
                ##Weight of water (end)
                ww_dams = mw_dams * (1 - cg_dams[19, ...]) + aw_dams * (1 - cg_dams[20, ...])
                ##Weight of gutfill (end)
                gw_dams = ffcfw_dams* (1 - 1 / cg_dams[18, ...])
                ##Whole body energy (calculated from muscle and adipose weight)
                wbe_dams = sfun.f_wbe(aw_dams, mw_dams, cg_dams)
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
                sl_dams = (fl_dams - fl_shear_yg0) / cw_dams[15, ...]
                ##Staple strength if shorn(end)
                ss_dams = fd_min_dams ** 2 / fd_dams ** 2 * cw_dams[16, ...]


            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_yatf = np.maximum(0, ffcfw_start_yatf + cg_yatf[18, ...] * ebg_yatf * days_period_pa1e1b1nwzida0e0b0xyg2[p])
                ##FFCFW maximum to date
                ffcfw_max_yatf = np.maximum(ffcfw_yatf, ffcfw_max_start_yatf)
                ##Weight of fat adipose (end)
                aw_yatf = aw_start_yatf + fg_yatf / cg_yatf[20, ...] * days_period_pa1e1b1nwzida0e0b0xyg2[p]
                ##Weight of muscle (end)
                mw_yatf = mw_start_yatf + pg_yatf / cg_yatf[19, ...] * days_period_pa1e1b1nwzida0e0b0xyg2[p]
                ##Weight of bone (end)	bw ^formula needs finishing
                bw_yatf = bw_start_yatf
                ##Weight of water (end)
                ww_yatf = mw_yatf * (1 - cg_yatf[19, ...]) + aw_yatf * (1 - cg_yatf[20, ...])
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
                sl_yatf = (fl_yatf - fl_shear_yg0) / cw_yatf[15, ...]
                ##Staple strength if shorn(end)
                ss_yatf = fun.f_divide(fd_min_yatf ** 2 , fd_yatf ** 2 * cw_yatf[16, ...])

            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ##FFCFW (end - fleece free conceptus free)
                ffcfw_offs = np.maximum(0,ffcfw_start_offs + cg_offs[18, ...] * ebg_offs * days_period_pa1e1b1nwzida0e0b0xyg3[p])
                ##FFCFW maximum to date
                ffcfw_max_offs = np.maximum(ffcfw_offs, ffcfw_max_start_offs)
                ##Weight of fat adipose (end)
                aw_offs = aw_start_offs + fg_offs / cg_offs[20, ...] * days_period_pa1e1b1nwzida0e0b0xyg3[p]
                ##Weight of muscle (end)
                mw_offs = mw_start_offs + pg_offs / cg_offs[19, ...] * days_period_pa1e1b1nwzida0e0b0xyg3[p]
                ##Weight of bone (end)	bw ^formula needs finishing
                bw_offs = bw_start_offs
                ##Weight of water (end)
                ww_offs = mw_offs * (1 - cg_offs[19, ...]) + aw_offs * (1 - cg_offs[20, ...])
                ##Weight of gutfill (end)
                gw_offs = ffcfw_offs* (1 - 1 / cg_offs[18, ...])
                ##Whole body energy (end - calculated from muscle and adipose weight)
                wbe_offs = sfun.f_wbe(aw_offs, mw_offs, cg_offs)
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
                sl_offs = (fl_offs - fl_shear_yg0) / cw_offs[15, ...]
                ##Staple strength if shorn(end)
                ss_offs = fd_min_offs ** 2 / fd_offs ** 2 * cw_offs[16, ...]

            ######################################
            #store postprocessing and report vars#
            ######################################
            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p, ...] > 0):
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort (no feedlot mask for sires (only for offs) because feedlotting sires doesn indicat they are being sold).
                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_sire = np.broadcast_to(numbers_start_condense_sire, numbers_end_sire.shape) #required for the first condensing because condense numbers start doesnt have all the axis.
                surv_sire = (np.sum(numbers_end_sire,axis=prejoin_tup + (season_tup,), keepdims=True)
                             / np.sum(numbers_start_condense_sire, axis=prejoin_tup + (season_tup,), keepdims=True))  # sum e,b,z axis because numbers are distributed along those axis so need to sum to determine if w has mortality > 10%
                threshold = np.minimum(0.9, np.mean(surv_sire, axis=w_pos, keepdims=True)) #threshold is the lower of average survival and 90%
                mort_mask_sire = surv_sire > threshold

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_sire = mort_mask_sire

                ###sorted index of w. used for condensing and used below.
                idx_sorted_w_sire = np.argsort(ffcfw_sire, axis=w_pos)

                ###mask with a true for the season and w slices with the lightest animal
                mask_min_lw_z_sire = np.isclose(ffcfw_sire, np.min(ffcfw_sire, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post processing
                o_numbers_start_tpsire[:,p] = numbers_start_sire
                o_numbers_end_tpsire[:,p] = numbers_end_sire
                o_ffcfw_tpsire[:,p] = ffcfw_sire
                o_lw_tpsire[:,p] = lw_sire
                o_pi_tpsire[:,p] = pi_sire
                o_mei_solid_tpsire[:,p] = mei_solid_sire
                o_ch4_total_tpsire[:,p] = ch4_total_sire
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
                ###create a mask used to exclude w slices in the condensing func. exclude w slices that have greater than 10% mort  (no feedlot mask for dams (only for offs) because feedlotting dams doesn indicat they are being sold).
                ###mask for animals (slices of w) with mortality less than a threshold - True means mort is acceptable (below threshold)
                numbers_start_condense_dams = np.broadcast_to(numbers_start_condense_dams, numbers_end_dams.shape) #required for the first condensing because condense numbers start doesnt have all the axis.
                surv_dams = (np.sum(numbers_end_dams,axis=prejoin_tup + (season_tup,), keepdims=True)
                             / np.sum(numbers_start_condense_dams, axis=prejoin_tup + (season_tup,), keepdims=True))  # sum e,b,z axis because numbers are distributed along those axis so need to sum to determine if w has mortality > 10%
                threshold = np.minimum(0.9, np.mean(surv_dams, axis=w_pos, keepdims=True)) #threshold is the lower of average survival and 90%
                mort_mask_dams = surv_dams > threshold

                ###print warning if min mort is greater than 10% since the previous condense
                if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1]):
                    min_mort = 1- np.max(surv_dams, axis=w_pos)
                    if np.any(min_mort > 0.1):
                        print('WARNING: HIGH MORTALITY DAMS: period ', p)

                ###combine mort and feedlot mask  - True means the w slice is included in condensing.
                condense_w_mask_dams = mort_mask_dams

                ###sorted index of w. used for condensing.
                idx_sorted_w_dams = np.argsort(ffcfw_dams * condense_w_mask_dams, axis=w_pos)

                ###mask with a true for the season and w slices with the lightest animal
                mask_min_lw_z_dams = np.isclose(ffcfw_dams, np.min(ffcfw_dams, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post processing
                o_mortality_dams[:,p] = mortality_dams #has to be stored before back dating numbers
                o_numbers_start_tpdams[:,p] = numbers_start_dams
                o_numbers_end_tpdams[:,p] = numbers_end_dams

                ###back date numbers from end of mating to prejoining
                if np.any(period_is_mating_pa1e1b1nwzida0e0b0xyg1[p]): #need to back date the numbers from conception to prejoining because otherwise in the matrix there is not a dvp between prejoining and mating therefore this is require so that other slices have energy etc requirement
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
                o_ffcfw_season_tpdams[:,p] = sfun.f1_season_wa(numbers_end_dams, ffcfw_dams, season_tup, mask_min_lw_z_dams
                                                           , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1])
                o_ffcfw_condensed_tpdams[:,p] = sfun.f1_condensed(ffcfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                                              , n_fs_dams, len_w1, n_fvp_periods_dams
                                                              , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])  #condensed lw at the end of the period
                o_nw_start_tpdams[:,p] = nw_start_dams
                numbers_join_dams = fun.f_update(numbers_join_dams, numbers_start_dams, period_is_join_pa1e1b1nwzida0e0b0xyg1[p])
                o_numbers_join_tpdams[:,p] = numbers_join_dams #store the numbers at joining until next
                o_lw_tpdams[:,p] = lw_dams
                o_pi_tpdams[:,p] = pi_dams
                o_mei_solid_tpdams[:,p] = mei_solid_dams
                o_ch4_total_tpdams[:,p] = ch4_total_dams
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
            o_numbers_start_tpyatf[:,p] = numbers_start_yatf #used for npw calculation - use numbers start because weaning is start of period - has to be out of the 'if' because there is 0 days in the period when weaning occurs but we still want to store the start numbers (because once they are weaned they are not yatf therefore 0 days per period)
            o_rc_start_tpyatf[:,p] = rc_start_yatf #outside because used for sale value which is weaning which has 0 days per period because weaning is first day (this means the rc at weaning is actually the rc at the start of the previous period because it doesnt recalculate once days per period goes to 0) (because once they are weaned they are not yatf therefore 0 days per period)
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
                numbers_start_condense_yatf = np.broadcast_to(numbers_start_condense_yatf, numbers_end_yatf.shape) #required for the first condensing because condense numbers start doesnt have all the axis.
                surv_yatf = fun.f_divide(np.sum(numbers_end_yatf,axis=season_tup, keepdims=True)
                                        , np.sum(numbers_start_condense_yatf, axis=season_tup, keepdims=True))  # sum z axis because numbers are distributed along z axis so need to sum to determine if w has mortality > 10% (don't sum e&b because yatf stay in the same slice)
                threshold = np.minimum(0.9, np.mean(surv_yatf, axis=w_pos, keepdims=True)) #threshold is the lower of average survival and 90%
                mort_mask_yatf = surv_yatf > threshold

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_yatf = np.logical_and(no_confinement_yatf, mort_mask_yatf)

                ###sorted index of w. used for condensing.
                idx_sorted_w_yatf = np.argsort(ffcfw_yatf * condense_w_mask_yatf, axis=w_pos)

                ###mask with a true for the season and w slices with the lightest animal
                mask_min_lw_z_yatf = np.isclose(ffcfw_yatf, np.min(ffcfw_yatf, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post processing
                o_pi_tpyatf[:,p] = pi_yatf
                o_mei_solid_tpyatf[:,p] = mei_solid_yatf
                # o_ch4_total_tpyatf[:,p] = ch4_total_yatf
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
                numbers_start_condense_offs = np.broadcast_to(numbers_start_condense_offs, numbers_end_offs.shape) #required for the first condensing because condense numbers start doesnt have all the axis.
                surv_offs = (np.sum(numbers_end_offs,axis=season_tup, keepdims=True)
                             / np.sum(numbers_start_condense_offs, axis=season_tup, keepdims=True))  # sum z axis because numbers are distributed along those axis so need to sum to determine if w has mortality > 10% (don't sum e&b because offs don't change slice)
                threshold = np.minimum(0.9, np.mean(surv_offs, axis=w_pos, keepdims=True)) #threshold is the lower of average survival and 90%
                mort_mask_offs = surv_offs > threshold

                ###print warning if min mort is greater than 10% since the previous condense
                if np.any(period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1]):
                    min_mort = 1- np.max(surv_offs, axis=w_pos)
                    if np.any(min_mort > 0.1):
                        print('WARNING: HIGH MORTALITY OFFS: period ', p)

                ###combine mort and feedlot mask - True means the w slice is included in condensing.
                condense_w_mask_offs = np.logical_and(no_confinement_offs, mort_mask_offs)

                ###sorted index of w. used for condensing.
                idx_sorted_w_offs = np.argsort(ffcfw_offs * condense_w_mask_offs, axis=w_pos) #set animal

                ###mask with a true for f1_season_wa and f_start_prod to identify w slices with the lightest animal
                mask_min_lw_z_offs = np.isclose(ffcfw_offs, np.min(ffcfw_offs, axis=(w_pos, z_pos), keepdims=True)) #use isclose in case small rounding error in lw

                ###store output variables for the post processing
                o_numbers_start_tpoffs[:,p] = numbers_start_offs
                o_numbers_end_tpoffs[:,p] = numbers_end_offs
                o_ffcfw_tpoffs[:,p] = ffcfw_offs
                o_ffcfw_season_tpoffs[:,p] = sfun.f1_season_wa(numbers_end_offs, ffcfw_offs, season_tup, mask_min_lw_z_offs
                                                           , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1])
                o_ffcfw_condensed_tpoffs[:,p] = sfun.f1_condensed(ffcfw_offs, idx_sorted_w_offs, condense_w_mask_offs,
                                                              n_fs_offs, len_w3, n_fvp_periods_offs,
                                                              period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])  #condensed lw at the end of the period before fvp0
                o_nw_start_tpoffs[:,p] = nw_start_offs
                o_mortality_offs[:,p] = mortality_offs
                o_lw_tpoffs[:,p] = lw_offs
                o_pi_tpoffs[:,p] = pi_offs
                o_mei_solid_tpoffs[:,p] = mei_solid_offs
                o_ch4_total_tpoffs[:,p] = ch4_total_offs
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
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)  #increment the p slice, note this doesnt impact the p loop - this is required for the next section because we are calculating the production and numbers for the start of the next period.
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
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_dams = sfun.f1_condensed(nw_start_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###FFCFW maximum to date
                ffcfw_max_condensed_dams = sfun.f1_condensed(ffcfw_max_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of adipose (condense)
                aw_condensed_dams = sfun.f1_condensed(aw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of muscle (condense)
                mw_condensed_dams = sfun.f1_condensed(mw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of bone (condense)
                bw_condensed_dams = sfun.f1_condensed(bw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g1 = sfun.f1_condensed(omer_history_dams, idx_sorted_w_dams[na,...]
                                        , condense_w_mask_dams[na,...], n_fs_dams, len_w1, n_fvp_periods_dams
                                        , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (condense)
                cfw_condensed_dams = sfun.f1_condensed(cfw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g1 = sfun.f1_condensed(d_cfw_history_dams_p2, idx_sorted_w_dams[na,...]
                                        , condense_w_mask_dams[na,...], n_fs_dams, len_w1, n_fvp_periods_dams
                                        , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Fibre length since shearing (condense)
                fl_condensed_dams = sfun.f1_condensed(fl_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FD since shearing (condense)
                fd_condensed_dams = sfun.f1_condensed(fd_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Minimum FD since shearing (condense)
                fd_min_condensed_dams = sfun.f1_condensed(fd_min_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Lagged DR (lactation deficit)
                ldr_condensed_dams = sfun.f1_condensed(ldr_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Loss of potential milk due to consistent under production
                lb_condensed_dams = sfun.f1_condensed(lb_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Loss of potential milk due to consistent under production
                rc_birth_condensed_dams = sfun.f1_condensed(rc_birth_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of foetus (condense)
                w_f_condensed_dams = sfun.f1_condensed(w_f_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of gravid uterus (condense)
                guw_condensed_dams = sfun.f1_condensed(guw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Normal weight of foetus (condense)
                nw_f_condensed_dams = sfun.f1_condensed(nw_f_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Birth weight carryover (running tally of foetal weight diff)
                cf_w_b_condensed_dams = sfun.f1_condensed(cf_w_b_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###LTW CFW carryover (running tally of CFW diff)
                cf_cfwltw_condensed_dams = sfun.f1_condensed(cf_cfwltw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###LTW FD carryover (running tally of FD diff)
                cf_fdltw_condensed_dams = sfun.f1_condensed(cf_fdltw_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##dams LTW CFW (total adjustment, calculated at birth)
                cfw_ltwadj_condensed_dams = sfun.f1_condensed(cfw_ltwadj_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##dams LTW FD (total adjustment, calculated at birth)
                fd_ltwadj_condensed_dams = sfun.f1_condensed(fd_ltwadj_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Carry forward conception
                cf_conception_condensed_dams = sfun.f1_condensed(cf_conception_dams
                                        , idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weaning weight carryover (running tally of weaning weight diff)
                cf_w_w_condensed_dams = sfun.f1_condensed(cf_w_w_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FOO during lactation (for weaning weight calculation)
                foo_lact_ave_condensed = sfun.f1_condensed(foo_lact_ave, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_yatf = sfun.f1_condensed(ffcfw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_yatf = sfun.f1_condensed(nw_start_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###FFCFW maximum to date
                ffcfw_max_condensed_yatf = sfun.f1_condensed(ffcfw_max_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of adipose (condense)
                aw_condensed_yatf = sfun.f1_condensed(aw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of muscle (condense)
                mw_condensed_yatf = sfun.f1_condensed(mw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Weight of bone (condense)
                bw_condensed_yatf = sfun.f1_condensed(bw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g2 = sfun.f1_condensed(omer_history_yatf, idx_sorted_w_yatf[na,...], condense_w_mask_yatf[na,...]
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (condense)
                cfw_condensed_yatf = sfun.f1_condensed(cfw_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g2 = sfun.f1_condensed(d_cfw_history_yatf_p2, idx_sorted_w_yatf[na,...], condense_w_mask_yatf[na,...]
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Fibre length since shearing (condense)
                fl_condensed_yatf = sfun.f1_condensed(fl_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Average FD since shearing (condense)
                fd_condensed_yatf = sfun.f1_condensed(fd_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###Minimum FD since shearing (condense)
                fd_min_condensed_yatf = sfun.f1_condensed(fd_min_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##yatf birth weight
                w_b_condensed_yatf = sfun.f1_condensed(w_b_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
                ##yatf wean weight
                w_w_condensed_yatf = sfun.f1_condensed(w_w_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])
            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###FFCFW (condense - fleece free conceptus free)
                ffcfw_condensed_offs = sfun.f1_condensed(ffcfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_condensed_offs = sfun.f1_condensed(nw_start_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###FFCFW maximum to date
                ffcfw_max_condensed_offs = sfun.f1_condensed(ffcfw_max_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of adipose (condense)
                aw_condensed_offs = sfun.f1_condensed(aw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of muscle (condense)
                mw_condensed_offs = sfun.f1_condensed(mw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Weight of bone (condense)
                bw_condensed_offs = sfun.f1_condensed(bw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Organ energy requirement (condense)
                omer_history_condensed_p3g3 = sfun.f1_condensed(omer_history_offs, idx_sorted_w_offs[na,...], condense_w_mask_offs[na,...]
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Clean fleece weight (condense)
                cfw_condensed_offs = sfun.f1_condensed(cfw_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Clean fleece weight (condense)
                d_cfw_history_condensed_p2g3 = sfun.f1_condensed(d_cfw_history_offs_p2, idx_sorted_w_offs[na,...], condense_w_mask_offs[na,...]
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Fibre length since shearing (condense)
                fl_condensed_offs = sfun.f1_condensed(fl_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Average FD since shearing (condense)
                fd_condensed_offs = sfun.f1_condensed(fd_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])
                ###Minimum FD since shearing (condense)
                fd_min_condensed_offs = sfun.f1_condensed(fd_min_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])

            ##condense end numbers - have to condense the numbers before calc start production, but need to condense production using non condensed end numbers
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                numbers_end_condensed_sire = sfun.f1_condensed(numbers_end_sire, idx_sorted_w_sire, condense_w_mask_sire
                                        , n_fs_sire, len_w0, n_fvp_periods_sire, False)

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                numbers_end_condensed_dams = sfun.f1_condensed(numbers_end_dams, idx_sorted_w_dams, condense_w_mask_dams
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                numbers_end_condensed_yatf = sfun.f1_condensed(numbers_end_yatf, idx_sorted_w_yatf, condense_w_mask_yatf
                                        , n_fs_dams, len_w1, n_fvp_periods_dams, period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] > 0):
                numbers_end_condensed_offs = sfun.f1_condensed(numbers_end_offs, idx_sorted_w_offs, condense_w_mask_offs
                                        , n_fs_offs, len_w3, n_fvp_periods_offs, period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])

            ##start production - this requires condensed end number
            ###sire
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, ffcfw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###nw (start - normal weight)	- yes this is meant to be updated from nw_start
                nw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, nw_start_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###FFCFW maximum to date
                ffcfw_max_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, ffcfw_max_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Weight of adipose (start)
                aw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, aw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Weight of muscle (start)
                mw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, mw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Weight of bone (start)
                bw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, bw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Organ energy requirement (start)
                omer_history_start_p3g0 = sfun.f1_period_start_prod(numbers_end_condensed_sire, omer_history_condensed_p3g0
                                        , prejoin_tup, season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire[na,...])
                ###Clean fleece weight (start)
                cfw_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, cfw_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g0 = sfun.f1_period_start_prod(numbers_end_condensed_sire, d_cfw_history_condensed_p2g0, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire[na,...])
                ###Fibre length since shearing (start)
                fl_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fl_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Average FD since shearing (start)
                fd_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fd_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)
                ###Minimum FD since shearing (start)
                fd_min_start_sire = sfun.f1_period_start_prod(numbers_end_condensed_sire, fd_min_condensed_sire, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_sire)

            ###dams
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ffcfw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, nw_start_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###FFCFW maximum to date
                ffcfw_max_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ffcfw_max_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weight of adipose (start)
                aw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, aw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weight of muscle (start)
                mw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, mw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weight of bone (start)
                bw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, bw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Organ energy requirement (start)
                omer_history_start_p3g1 = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , omer_history_condensed_p3g1, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams[na,...]
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Clean fleece weight (start)
                cfw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cfw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g1 = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , d_cfw_history_condensed_p2g1, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams[na,...]
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Fibre length since shearing (start)
                fl_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fl_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Average FD since shearing (start)
                fd_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Minimum FD since shearing (start)
                fd_min_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_min_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Lagged DR (lactation deficit)
                ldr_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, ldr_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Loss of potential milk due to consistent under production
                lb_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, lb_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Loss of potential milk due to consistent under production
                rc_birth_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, rc_birth_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weight of foetus (start)
                w_f_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, w_f_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p+1]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weight of gravid uterus (start)
                guw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, guw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Normal weight of foetus (start)
                nw_f_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, nw_f_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Birth weight carryover (running tally of foetal weight diff)
                cf_w_b_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_w_b_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###LTW CFW carryover (running tally of CFW diff)
                cf_cfwltw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_cfwltw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###LTW FD carryover (running tally of FD diff)
                cf_fdltw_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_fdltw_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ##dams LTW CFW (total adjustment, calculated at birth)
                cfw_ltwadj_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , cfw_ltwadj_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ##dams LTW FD (total adjustment, calculated at birth)
                fd_ltwadj_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, fd_ltwadj_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Carry forward conception
                cf_conception_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams
                                        , cf_conception_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Weaning weight carryover (running tally of foetal weight diff)
                cf_w_w_start_dams = sfun.f1_period_start_prod(numbers_end_condensed_dams, cf_w_w_condensed_dams, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining
                ###Average FOO during lactation (for weaning weight calculation)
                foo_lact_ave_start = sfun.f1_period_start_prod(numbers_end_condensed_dams, foo_lact_ave_condensed, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_dams
                                        , period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1], group=1
                                        , scan_management=scan_management_pa1e1b1nwzida0e0b0xyg1[p]
                                        , gbal = gbal_management_pa1e1b1nwzida0e0b0xyg1[p]) #use p because we want to know scan management in the current repro cycle because that impacts if drys are included in the weighted average use to create the new animal at prejoining

            ###yatf
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, ffcfw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, nw_start_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###FFCFW maximum to date
                ffcfw_max_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, ffcfw_max_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Weight of adipose (start)
                aw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, aw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Weight of muscle (start)
                mw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, mw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Weight of bone (start)
                bw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, bw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Organ energy requirement (start)
                omer_history_start_p3g2 = sfun.f1_period_start_prod(numbers_end_condensed_yatf
                                        , omer_history_condensed_p3g2, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf[na,...])
                ###Clean fleece weight (start)
                cfw_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, cfw_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g2 = sfun.f1_period_start_prod(numbers_end_condensed_yatf
                                        , d_cfw_history_condensed_p2g2, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf[na,...])
                ###Fibre length since shearing (start)
                fl_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fl_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Average FD since shearing (start)
                fd_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fd_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ###Minimum FD since shearing (start)
                fd_min_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, fd_min_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ##yatf birth weight
                w_b_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, w_b_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
                ##yatf wean weight
                w_w_start_yatf = sfun.f1_period_start_prod(numbers_end_condensed_yatf, w_w_condensed_yatf, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_yatf)
            ###offs
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                ###FFCFW (start - fleece free conceptus free)
                ffcfw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, ffcfw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###normal weight	- yes this is meant to be updated from nw_start
                nw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, nw_start_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###FFCFW maximum to date
                ffcfw_max_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, ffcfw_max_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Weight of adipose (start)
                aw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, aw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Weight of muscle (start)
                mw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, mw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Weight of bone (start)
                bw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, bw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Organ energy requirement (start)
                omer_history_start_p3g3 = sfun.f1_period_start_prod(numbers_end_condensed_offs
                                        , omer_history_condensed_p3g3, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs[na,...])
                ###Clean fleece weight (start)
                cfw_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, cfw_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Clean fleece weight (start)
                d_cfw_history_start_p2g3 = sfun.f1_period_start_prod(numbers_end_condensed_offs
                                        , d_cfw_history_condensed_p2g3, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs[na,...])
                ###Fibre length since shearing (start)
                fl_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fl_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Average FD since shearing (start)
                fd_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fd_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)
                ###Minimum FD since shearing (start)
                fd_min_start_offs = sfun.f1_period_start_prod(numbers_end_condensed_offs, fd_min_condensed_offs, prejoin_tup
                                        , season_tup, period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], mask_min_lw_z_offs)


            ##start numbers - has to be after production because the numbers are being calced for the current period and are used in the start production function
            ## Doesnt have to use condensed numbers because we are only interested in the start vs end numbers of a dvp (using condensed numbers would still work).
            if np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                numbers_start_sire = sfun.f1_period_start_nums(numbers_end_sire, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=0)
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_sire = fun.f_update(numbers_start_condense_sire, numbers_start_sire, False) #currently sire don't have any fvp

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                numbers_start_dams = sfun.f1_period_start_nums(numbers_end_dams, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=1
                                        , numbers_initial_repro=numbers_initial_propn_repro_a1e1b1nwzida0e0b0xyg1
                                        , period_is_prejoin=period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p+1])
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_dams = fun.f_update(numbers_start_condense_dams, numbers_start_dams
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p+1,...] >0): #use p+1 so that initial numbers at birth can be set
                numbers_start_yatf = sfun.f1_period_start_nums(numbers_end_yatf, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg
                                        , nyatf_b1=nyatf_b1nwzida0e0b0xyg, gender_propn_x=gender_propn_xyg
                                        , period_is_birth=period_is_birth_pa1e1b1nwzida0e0b0xyg1[p+1], group=2)
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                if np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p, ...] > 0):
                    numbers_start_condense_yatf = fun.f_update(numbers_start_condense_yatf, numbers_start_yatf
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg1[p+1])

            if np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                numbers_start_offs = sfun.f1_period_start_nums(numbers_end_offs, prejoin_tup, season_tup
                                        , period_is_startseason_pa1e1b1nwzida0e0b0xyg[p+1], season_propn_zida0e0b0xyg, group=3)
                ###numbers at the beginning of fvp 0 (used to calc mort for the lw patterns to determine the lowest feasible level - used in the start prod func)
                numbers_start_condense_offs = fun.f_update(numbers_start_condense_offs, numbers_start_offs
                                                           , period_is_condense_pa1e1b1nwzida0e0b0xyg3[p+1])

        ## Calculate LTW sfw multiplier & sfd addition then repeat the generator loops with updated LTW adjuster
        ### sires don't have an adjuster calculated because they are born off-farm & unrelated to the dam nutrition profile
        ### yatf don't have an adjuster because the LTW project did not show a consistent effect of dam profile on the wool shorn at the lamb shearing.

        ## The LTW adjuster for a period within the range pre-joining to next_period_is_prejoining, is the value from that lambing
        ### The LTW adjuster is retained in the variable through until period_is_prejoining, so it can be accessed when next_period_is_prejoining
        ### Create the association between nextperiod_is_prejoin and the current period
        a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1 = fun.f_next_prev_association(date_end_p, date_prejoin_next_pa1e1b1nwzida0e0b0xyg1, 1, 'right').astype(dtypeint)[na] #p indx of period before prejoining - when nextperiod is prejoining this returns the current period

        ## the dam lifetime adjustment (for the p, e1, b1 & w axes) are based on the LW profile of the dams themselves and scaled by the number of progeny they rear as a proportion of the total number weaned.
        ##Thus ltw adjustment is 0 for dams with no yatf (the LW profile of ewe with no yatf does not effect the next generation)
        ### cfw is a scalar so it is the LTW effect as a proportion of sfw. FD is a change so it not scaled by sfd.
        ### populate ltwadj with the value from the period before prejoining. That value is the final value that has been carried forward from the whole profile change
        o_cfw_ltwadj_tpdams = np.take_along_axis(o_cfw_ltwadj_tpdams, a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1, axis=p_pos)
        o_fd_ltwadj_tpdams = np.take_along_axis(o_fd_ltwadj_tpdams, a_nextisprejoin_tpa1e1b1nwzida0e0b0xyg1, axis=p_pos)

        if n_fs_dams>1:
            #an approximation of the LTW effect of dam nutrition on the progeny that are the replacement dams
            sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = 1 + (o_cfw_ltwadj_tpdams * nyatf_b1nwzida0e0b0xyg
                                                     / npw_std_xyg1 / sfw_a0e0b0xyg1) * sen.sam['LTW_dams']
            sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = o_fd_ltwadj_tpdams * nyatf_b1nwzida0e0b0xyg / npw_std_xyg1 * sen.sam['LTW_dams']

        else:
            ## if n==1 the average LTW adjustment for the next generation is the average of the dam effects weighted by
            ##the number of progeny from each class of dams.
            ## Note: if the propn mated inputs are set to optimise (np.inf) then it is treated as 100% mated (included in numbers_start)
            ## Note: The approximation doesn't account for dam mortality during the year or Progeny mortality during lactation
            ## the weighted average is across all active axes except i, y & g1 (because these represent classes that are not combined at prejoining)
            ### CFW (as a proportion of sfw)
            t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = 1 + fun.f_weighted_average(o_cfw_ltwadj_tpdams, o_numbers_start_tpdams
                                                                * nyatf_b1nwzida0e0b0xyg * season_propn_zida0e0b0xyg
                                                                , axis=(p_pos, a1_pos, e1_pos, b1_pos, w_pos, z_pos)
                                                                , keepdims=True) / sfw_a0e0b0xyg1 * sen.sam['LTW_dams']

            ### FD (an absolute adjustment)
            t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(o_fd_ltwadj_tpdams, o_numbers_start_tpdams
                                                                * nyatf_b1nwzida0e0b0xyg * season_propn_zida0e0b0xyg
                                                                , axis=(p_pos, a1_pos, e1_pos, b1_pos, w_pos, z_pos)
                                                                , keepdims=True) * sen.sam['LTW_dams']

            ## allocate the LTW adjustment to the slices of g1 based on the female parent of each g1 slice.
            ## nutrition of BBB dams [0:1] affects BB-B, BB-M & BB-T during their lifetime.
            sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[...] = t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., 0:1]
            sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[...] = t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., 0:1]
            ## nutrition of BBM dams [1] affects BM-T [-1] during their lifetime. (Needs to be [-1] to handle if BBT have been masked)
            if mask_dams_inc_g1[3:4]:
                sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., -1] = t_sfw_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., 1]
                sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., -1] = t_sfd_ltwadj_tpa1e1b1nwzida0e0b0xyg1[..., 1]

        ## the offspring lifetime adjustment is based on dam LW pattern 0. The dam pattern must be specified/estimated
        ### because there is not a link in the matrix between dam profile and the offspring DVs.
        ### Note: The offspring LTW adjustment works as it should if N==1 for dams i.e. all the progeny are from pattern 0.
        ### The offspring CFW effect is a multiplier based on the dam LTW effect as a proportion of the dam sfw,
        ### this allows for the offspring to be a different genotype than the dam and get a proportional adjustment
        ### For each offspring d slice select the p slice from o_cfw_ltwadj based on a_prevjoining_o_p when period_is_join
        ###         e1 axis in the position of e0
        ###         b1 axis in the position of b0 and simplified using a_b0_b1
        ###         w axis to only have slice 0
        ###         z axis is the weighted average across season types
        temporary = np.sum(fun.f_dynamic_slice(o_cfw_ltwadj_tpdams,w_pos,0,1) / sfw_a0e0b0xyg1
                           * (a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 == index_da0e0b0xyg)
                           * period_is_join_pa1e1b1nwzida0e0b0xyg1, axis=p_pos)
        ##dams have a e1 axis, whereas offspring have an e0 axis, swap the e1 into position of e0
        temporary = np.swapaxes(temporary, e1_pos, e0_pos)
        ##the b1 axis needs to be transformed into b0 because they are different lengths.
        temporary = np.sum(temporary * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                           , axis=b1_pos, keepdims=True)  #0 for dams with no yatf because for those b1 slices there is no corresponding slice in b0
        t_season_propn_pg = np.broadcast_to(season_propn_zida0e0b0xyg, temporary.shape)
        temporary = fun.f_weighted_average(temporary, t_season_propn_pg, axis=z_pos, keepdims=True)
        sfw_ltwadj_ta1e1b1nwzida0e0b0xyg3 = 1 + temporary * sen.sam['LTW_offs']

        ## repeat for FD
        temporary = np.sum(fun.f_dynamic_slice(o_fd_ltwadj_tpdams,w_pos,0,1)
                           * (a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 == index_da0e0b0xyg)
                           * period_is_join_pa1e1b1nwzida0e0b0xyg1, axis=p_pos)
        temporary = np.swapaxes(temporary, e1_pos, e0_pos)
        temporary = np.sum(temporary * (a_b0_b1nwzida0e0b0xyg == index_b0xyg) * (nyatf_b1nwzida0e0b0xyg > 0)
                           , axis=b1_pos, keepdims=True)  #0 for dams with no yatf because for those b1 slices there is no corresponding slice in b0
        t_season_propn_pg = np.broadcast_to(season_propn_zida0e0b0xyg, temporary.shape)
        temporary = fun.f_weighted_average(temporary, t_season_propn_pg, axis=z_pos, keepdims=True)
        sfd_ltwadj_ta1e1b1nwzida0e0b0xyg3 = temporary * sen.sam['LTW_offs']

    postp_start=time.time()
    print(f'completed generator loops: {postp_start - generator_start}')


    ## Call Steve graphing routine here if Generator is throwing an error in the post processing.
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
    alloc_p7tpa1e1b1nwzida0e0b0xyg = zfun.f1_z_period_alloc(date_start_pa1e1b1nwzida0e0b0xyg[na,na,...],z_pos=z_pos)

    ###labour period
    labour_periods = per.f_p_date2_df().to_numpy().astype('datetime64[D]') #convert from df to numpy
    labour_periods_yp5z = labour_periods + (np.arange(np.ceil(sim_years)) * np.timedelta64(365,'D'))[:,na,na] #expand from single yr to all length of generator
    labour_periods_p5z = labour_periods_yp5z.reshape((-1, len_z))
    ####set lp to start at the next generator period following the node (needs to be next so that clustering works). Lp are adjusted so that they get clustered the same as dvps
    idx_p5z = np.minimum(len(date_start_p) - 1,np.searchsorted(date_start_p,labour_periods_p5z,'left'))  # maximum idx is the number of generator periods
    labour_periods_p5z = date_start_p[idx_p5z]
    a_p5_pz = np.apply_along_axis(fun.f_next_prev_association, 0, labour_periods_p5z, date_end_p, 1, 'right') % len(labour_periods)
    a_p5_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_p5_pz,z_pos,left_pos2=p_pos,right_pos2=z_pos).astype(dtype)
    index_p5tpa1e1b1nwzida0e0b0xyg = fun.f_expand(np.arange(len(labour_periods)), p_pos-2)
    ###asset value timing - the date when the asset value is tallied
    assetvalue_timing = pinp.sheep['i_date_cashflow_stock_i'][pinp.sheep['i_mask_i']].astype('datetime64')
    assetvalue_timing = assetvalue_timing.view('i8').mean(keepdims=True).astype(assetvalue_timing.dtype) #take mean in case multiple tol included
    assetvalue_timing_y = assetvalue_timing + (np.arange(np.ceil(sim_years)) * np.timedelta64(365,'D')) #timing of asset value calculation each yr
    a_assetvalue_p = fun.f_next_prev_association(assetvalue_timing_y, date_end_p, 1,'right')
    a_assetvalue_pa1e1b1nwzida0e0b0xyg = fun.f_expand(a_assetvalue_p, p_pos)
    assetvalue_timing_pa1e1b1nwzida0e0b0xyg = assetvalue_timing_y[a_assetvalue_pa1e1b1nwzida0e0b0xyg]
    period_is_assetvalue_pa1e1b1nwzida0e0b0xyg = sfun.f1_period_is_('period_is', assetvalue_timing_pa1e1b1nwzida0e0b0xyg, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
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
    mask_s7x_s7tpa1e1b1nwzida0e0b0xyg = fun.f_expand(uinp.sheep['i_mask_s7x'], x_pos, left_pos2=p_pos-2, right_pos2=x_pos) #don't mask x axis
    mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3 = fun.f_expand(uinp.sheep['i_mask_s7x'], x_pos, left_pos2=p_pos-2, right_pos2=x_pos, condition=mask_x, axis=x_pos)
    sale_agemax_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg2  \
        , sale_agemax_s7tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_c2g(uinp.parameters['i_agemax_s7c2'],uinp.parameters['i_agemax_s7y'],p_pos-2, dtype=dtype)
    sale_agemin_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg2  \
        , sale_agemin_s7tpa1e1b1nwzida0e0b0xyg3 = sfun.f1_c2g(uinp.parameters['i_agemin_s7c2'],uinp.parameters['i_agemin_s7y'],p_pos-2, dtype=dtype)
    dresspercent_adj_yg0, dresspercent_adj_yg1, dresspercent_adj_yg2, dresspercent_adj_yg3 = sfun.f1_c2g(uinp.parameters['i_dressp_adj_c2'],uinp.parameters['i_dressp_adj_y'], dtype=dtype)
    ##husbandry
    wool_genes_yg0, wool_genes_yg1, wool_genes_yg2, wool_genes_yg3 = sfun.f1_c2g(uinp.parameters['i_wool_genes_c2'],uinp.parameters['i_wool_genes_y'], dtype=dtype)
    mobsize_p6a1e1b1nwzida0e0b0xyg0 = fun.f_expand(pinp.sheep['i_mobsize_sire_p6zi'], i_pos, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg0 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg0,numpy=True,axis=z_pos)
    mobsize_pa1e1b1nwzida0e0b0xyg0 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg0, a_p6_pa1e1b1nwzida0e0b0xyg,0)
    mobsize_p6a1e1b1nwzida0e0b0xyg1 = fun.f_expand(pinp.sheep['i_mobsize_dams_p6zi'], i_pos, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg1 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg1,numpy=True,axis=z_pos)
    mobsize_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg1,a_p6_pa1e1b1nwzida0e0b0xyg,0)
    mobsize_p6a1e1b1nwzida0e0b0xyg3 = fun.f_expand(pinp.sheep['i_mobsize_offs_p6zi'], i_pos, left_pos2=p_pos, right_pos2=z_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    mobsize_p6a1e1b1nwzida0e0b0xyg3 = zfun.f_seasonal_inp(mobsize_p6a1e1b1nwzida0e0b0xyg3,numpy=True,axis=z_pos)
    mobsize_pa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(mobsize_p6a1e1b1nwzida0e0b0xyg3, a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], 0)
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
    period_is_wean_pa1e1b1nwzida0e0b0xyg1 = np.logical_or(period_is_wean_pa1e1b1nwzida0e0b0xyg1, sfun.f1_period_is_('period_is', date_weaned_ida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)) #includes the weaning of the dam itself and the yatf because there is husbandry for the ewe when yatf are weaned eg the dams have to be mustered
    period_is_wean_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', date_weaned_ida0e0b0xyg3, date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    gender_xyg = fun.f_expand(np.arange(len(mask_x)), x_pos)
    ##sire
    purchcost_g0 = sfun.f1_g2g(pinp.sheep['i_purchcost_sire_ig0'], 'sire', condition=pinp.sheep['i_masksire_i'], axis=0) #Not divided by number of years onhand because the number of years of use is reflected in the number of dams that are serviced (because one sire can service multiple dam ages)
    date_purch_oa1e1b1nwzida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_purch_ig0'], 'sire', i_pos, left_pos2=p_pos-1, right_pos2=i_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos).astype('datetime64[D]')
    date_sale_oa1e1b1nwzida0e0b0xyg0 = sfun.f1_g2g(pinp.sheep['i_date_sale_ig0'], 'sire', i_pos, left_pos2=p_pos-1, right_pos2=i_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos).astype('datetime64[D]')
    sire_periods_g0p8y = sire_periods_g0p8[..., na].astype('datetime64[D]') + (
                         np.arange(np.ceil(sim_years)) * np.timedelta64(365, 'D'))
    period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8y = sfun.f1_period_is_('period_is', sire_periods_g0p8y, date_start_pa1e1b1nwzida0e0b0xyg[...,na,na], date_end_p = date_end_pa1e1b1nwzida0e0b0xyg[...,na,na])
    period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8 = np.any(period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8y, axis=-1) #condense the y axis - it is now accounted for by p axis
    ##dams
    sale_delay_sa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(pinp.sheep['i_sales_delay_sg1'], 'dams', p_pos) #periods after shearing that sale occurs
    ###mask for nutrition profiles. this doesnt have a full w axis because it only has the nutrition options it is expanded to w further down.
    sav_mask_nut_dams_owi = sen.sav['nut_mask_dams_owi'][:,0:len_nut_dams,:] #This controls if a nutrition pattern is included.
    mask_nut_dams_owi = fun.f_sa(np.array(True), sav_mask_nut_dams_owi,5) #all nut options included unless SAV is false
    mask_nut_oa1e1b1nwzida0e0b0xyg1 = fun.f_expand(mask_nut_dams_owi,i_pos, left_pos2=w_pos, left_pos3=p_pos,
                                                   right_pos2=i_pos, right_pos3=w_pos, condition=pinp.sheep['i_mask_i'],
                                                   axis=i_pos, condition2=mask_o_dams, axis2=p_pos)
    ##offs
    ###dvp mask - basically the shearing mask plus a true for the first dvp which is weaning
    sale_mask_g3 = np.concatenate([np.array([True]), mask_shear_g3]) #need to add true to the start of the shear mask because the first dvp is weaning
    ###days from the start of the dvp when sale occurs
    sales_offset_tsa1e1b1nwzida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_sales_offset_tsg3'], 'offs', p_pos, condition=sale_mask_g3, axis=p_pos)
    ###target weight in a dvp where sale occurs
    target_weight_tsa1e1b1nwzida0e0b0xyg3 = sfun.f1_g2g(pinp.sheep['i_target_weight_tsg3'], 'offs', p_pos, condition=sale_mask_g3, axis=p_pos) #plus 1 because it is shearing opp and weaning (ie the dvp for offs)
    ###number of periods before sale that shearing occurs in each dvp
    shearing_offset_tsa1e1b1nwzida0e0b0xyg3= sfun.f1_g2g(pinp.sheep['i_shear_prior_tsg3'], 'offs', p_pos, condition=sale_mask_g3, axis=p_pos) #plus 1 because it is shearing opp and weaning (ie the dvp for offs)


    ###mask for nutrition profiles. this doesnt have a full w axis because it only has the nutrition options it is expanded to w further down.
    sav_mask_nut_offs_swix = sen.sav['nut_mask_offs_swix'][:,0:len_nut_offs,...] #This controls if a nutrition pattern is included.
    mask_nut_offs_swix = fun.f_sa(np.array(True), sav_mask_nut_offs_swix,5) #all nut options included unless SAV is false
    mask_nut_sa1e1b1nwzida0e0b0xyg3 = fun.f_expand(mask_nut_offs_swix, x_pos, left_pos2=i_pos, left_pos3=w_pos,left_pos4=p_pos,
                                                   right_pos2=x_pos,right_pos3=i_pos,right_pos4=w_pos,
                                                   condition=pinp.sheep['i_mask_i'], axis=i_pos,
                                                   condition2=mask_shear_g3, axis2=p_pos, condition3=mask_x, axis3=x_pos)

    #################################
    ##post processing associations  #
    #################################
    ##yatf
    ###association between the birth time of yatf and the birth time of dams
    a_i_ida0e0b0xyg2 = sfun.f1_g2g(pinp.sheep['ia_i_idg2'],'yatf',d_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos, condition2=mask_d_offs, axis2=d_pos)
    a_g1_g2 = sfun.f1_g2g(pinp.sheep['ia_g1_g2'],'yatf')

    ##dams
    ###transfer
    a_g1_tpa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(sinp.stock['ia_g1_tg1'], 'dams', p_pos-1)
    transfer_exists_tpa1e1b1nwzida0e0b0xyg1 = sfun.f1_g2g(sinp.stock['i_transfer_exists_tg1'], 'dams', p_pos-1)
    #### adjust the pointers for excluded sires (t axis starts as just the sires eg dams transfer to different sire type)
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

    ###retained t
    a_t_g1 = np.arange(pinp.sheep['i_n_dam_sales'], pinp.sheep['i_n_dam_sales']+len_g1)

    ###dvp pointer and index
    a_v_pa1e1b1nwzida0e0b0xyg1 =  np.apply_along_axis(fun.f_next_prev_association, 0, dvp_start_va1e1b1nwzida0e0b0xyg1
                                                      , date_end_p, 1,'right')
    index_va1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg1)+1), p_pos)
    index_vpa1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.max(a_v_pa1e1b1nwzida0e0b0xyg1)+1), p_pos-1)
    ###other dvp associations and masks
    a_p_va1e1b1nwzida0e0b0xyg1 = fun.f_next_prev_association(date_start_p, dvp_start_va1e1b1nwzida0e0b0xyg1
                                                             , 1, 'right').astype(dtypeint) #returns the period index for the start of each dvp
    dvp_date_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1,0)
    dvp_type_next_va1e1b1nwzida0e0b0xyg1 = np.roll(dvp_type_va1e1b1nwzida0e0b0xyg1, -1, axis=p_pos)
    period_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_period_is_('period_is', dvp_date_pa1e1b1nwzida0e0b0xyg1
                                        , date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg)
    nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)
    nextperiod_is_prejoin_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_prejoin_pa1e1b1nwzida0e0b0xyg1,-1,axis=0)
    #### the transfer to a dvp_type other than type==0 only occurs when transferring to and from the same genotype
    #### this occurs when (index_g1 == a_g1_tg1) and the transfer exists
    mask_dvp_type_next_tg1 = transfer_exists_tpa1e1b1nwzida0e0b0xyg1 * (index_g1 == a_g1_tpa1e1b1nwzida0e0b0xyg1) #dvp type next is a little more complex for animals transferring. However the destination for transfer is always dvp type next ==0 (either it is going from dvp 2 to 0 or from 0 to 0.
    dvp_type_next_tva1e1b1nwzida0e0b0xyg1 = dvp_type_next_va1e1b1nwzida0e0b0xyg1 * mask_dvp_type_next_tg1
    ####association between dvp and lambing opp
    index_v1 = np.arange(index_vpa1e1b1nwzida0e0b0xyg1.shape[0])
    a_o_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1,0)

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
    dvp_date_pa1e1b1nwzida0e0b0xyg3=np.take_along_axis(dvp_start_va1e1b1nwzida0e0b0xyg3,a_v_pa1e1b1nwzida0e0b0xyg3,0)
    period_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_period_is_('period_is', dvp_date_pa1e1b1nwzida0e0b0xyg3
                                    , date_start_pa1e1b1nwzida0e0b0xyg3, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg3)
    nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_startdvp_pa1e1b1nwzida0e0b0xyg3,-1,axis=0)
    nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3 = np.roll(period_is_condense_pa1e1b1nwzida0e0b0xyg3,-1,axis=0)

    ####association between dvp and shearing - this is required because in the last dvp that the animal exist (ie when the generator ends) the sheep did not exist at shearing.
    a_s_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(a_prev_s_pa1e1b1nwzida0e0b0xyg3, a_p_va1e1b1nwzida0e0b0xyg3, axis=0)
    a_sw_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(fun.f_next_prev_association,0, date_wean_shearing_sa1e1b1nwzida0e0b0xyg3
                                                      , offs_date_end_p, 1, 'right')  #shearing opp with weaning included.
    ###cluster
    a_k5cluster_lsb0xyg = fun.f_expand(sinp.stock['ia_ppk5_lsb0'], b0_pos)
    a_k5cluster_b0xygls = np.moveaxis(np.moveaxis(a_k5cluster_lsb0xyg, 0,-1),0,-1) #put s and l at the end they are summed away shortly


    #######################
    #on hand / shear mask #
    #######################
    onhandshear_start=time.time()

    ##sire - purchased and sold on given date and shorn at main shearing - sires are simulated from weaning but for the pp we only look at a subset
    ### shearing - determined by the main shearing date - no t axis so just use the period is shearing from generator
    ###round purchase and sale date of sire to nearest period
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
    ###t1&2 - sold slice date_p=sale_date or weight=target weight
    ###calc sale date then determine shearing date
    ###sale - on date
    sale_date_tsa1e1b1nwzida0e0b0xyg3 = (sales_offset_tsa1e1b1nwzida0e0b0xyg3 + date_wean_shearing_sa1e1b1nwzida0e0b0xyg3)  #date of dvp plus sale offset
    sale_date_tpa1e1b1nwzida0e0b0xyg3=np.take_along_axis(sale_date_tsa1e1b1nwzida0e0b0xyg3,a_sw_pa1e1b1nwzida0e0b0xyg3[na],1)
    ####adjust sale date to be last day of period
    sale_date_idx_tpa1e1b1nwzida0e0b0xyg3 = fun.f_next_prev_association(offs_date_end_p, sale_date_tpa1e1b1nwzida0e0b0xyg3,0, 'left')#sale occurs at the end of the current generator period therefore 0 offset
    sale_date_tpa1e1b1nwzida0e0b0xyg3 = offs_date_end_p[sale_date_idx_tpa1e1b1nwzida0e0b0xyg3]
    ###sale - weight target
    ####convert from shearing/dvp to p array. Increments at dvp ie point to previous sale opp until new dvp then point at next dvp.
    target_weight_tpa1e1b1nwzida0e0b0xyg3=np.take_along_axis(target_weight_tsa1e1b1nwzida0e0b0xyg3, a_sw_pa1e1b1nwzida0e0b0xyg3[na],1) #gets the target weight for each gen period
    ####adjust generator lw to reflect the cumulative max per period
    #####lw could go above target then drop back below but it is already sold so the on hand bool shouldn't change. therefore need to use accumulative max and reset each dvp
    weight_tpa1e1b1nwzida0e0b0xyg3= sfun.f1_cum_dvp(o_ffcfw_tpoffs,a_v_pa1e1b1nwzida0e0b0xyg3, axis=p_pos)
    ###on hand
    #### t0 slice = True - this is handled by the inputs ie weight and date are high therefore not reached therefore on hand == true
    #### t1 & t2 slice date_p<sale_date and weight<target weight
    on_hand_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(date_start_pa1e1b1nwzida0e0b0xyg3<sale_date_tpa1e1b1nwzida0e0b0xyg3,
                                                     weight_tpa1e1b1nwzida0e0b0xyg3<target_weight_tpa1e1b1nwzida0e0b0xyg3)
    ###period is sale - one true per dvp when sale actually occurs - sale occurs in the period where sheep were on hand at the beginning and not on hand at the beginning of the next period
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(on_hand_tpa1e1b1nwzida0e0b0xyg3==True, np.roll(on_hand_tpa1e1b1nwzida0e0b0xyg3,-1,axis=1)==False)
    ###bound wether sale age - default is to allow all ages to be sold. User can change this using wether sale SAV.
    min_age_wether_sale_g3 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_wether_g3'][mask_offs_inc_g3], 5)
    max_age_wether_sale_g3 = fun.f_sa(np.array([sim_years*365]), sen.sav['bnd_max_sale_age_wether_g3'][mask_offs_inc_g3], 5)
    wether_sale_mask_pa1e1b1nwzida0e0b0xyg3 = np.logical_or((gender_xyg[mask_x] != 2),
        np.logical_and(age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] > min_age_wether_sale_g3,
                       age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] < max_age_wether_sale_g3))
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, wether_sale_mask_pa1e1b1nwzida0e0b0xyg3)
    ###bound female sale age - this sets the minimum age a ewe offs can be sold. Default is no min age eg can be sold anytime.
    min_age_female_sale_g3 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_g3'][mask_offs_inc_g3], 5)
    ewe_sale_mask_pa1e1b1nwzida0e0b0xyg3 = np.logical_or((gender_xyg[mask_x] != 1), age_start_pa1e1b1nwzida0e0b0xyg3[mask_p_offs_p] > min_age_female_sale_g3)
    period_is_sale_tpa1e1b1nwzida0e0b0xyg3 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, ewe_sale_mask_pa1e1b1nwzida0e0b0xyg3)
    ###shearing - one true per dvp when shearing actually occurs
    ###in t0 shearing occurs on specified date, in t1 & t2 it happens a certain number of gen periods before sale.
    ####convert from s/dvp to p
    shearing_offset_tpa1e1b1nwzida0e0b0xyg3=np.take_along_axis(shearing_offset_tsa1e1b1nwzida0e0b0xyg3, a_sw_pa1e1b1nwzida0e0b0xyg3[na],1)
    ###shearing can't occur in a different dvp to sale therefore need to cap the offset for periods at the beginning of the dvp ie if sale occurs in p2 of dvp2 and offset is 3 the offset needs to be reduced because shearing must occur in dvp2
    ####get the period number where dvp changes
    prev_dvp_index = fun.f_next_prev_association(offs_date_start_p, dvp_date_pa1e1b1nwzida0e0b0xyg3, 1, 'right')
    periods_since_dvp = np.maximum(0,p_index_pa1e1b1nwzida0e0b0xyg3 - prev_dvp_index)  #first dvp starts at weaning so just put in the max 0 to stop negative results when the p date is less than weaning
    ####period when shearing will occur - this is the min of the shearing offset or the periods since dvp start
    shearing_idx_tpa1e1b1nwzida0e0b0xyg3 = p_index_pa1e1b1nwzida0e0b0xyg3 - np.minimum(shearing_offset_tpa1e1b1nwzida0e0b0xyg3, periods_since_dvp)
    ###period is shearing is the sale array - offset
    shearing_idx_tpa1e1b1nwzida0e0b0xyg3 = period_is_sale_tpa1e1b1nwzida0e0b0xyg3*shearing_idx_tpa1e1b1nwzida0e0b0xyg3.astype(dtype)
    shearing_idx_tpa1e1b1nwzida0e0b0xyg3[shearing_idx_tpa1e1b1nwzida0e0b0xyg3==0] = np.inf #don't want 0 effecting minimum in next line
    shearing_idx_tpa1e1b1nwzida0e0b0xyg3= np.flip(np.minimum.accumulate(np.flip(shearing_idx_tpa1e1b1nwzida0e0b0xyg3,1),axis=1),1)
    period_is_shearing_tpa1e1b1nwzida0e0b0xyg3 = p_index_pa1e1b1nwzida0e0b0xyg3 == shearing_idx_tpa1e1b1nwzida0e0b0xyg3
    ###make slice t0 the shear dates for retained offs
    period_is_shearing_retained_pa1e1b1nwzida0e0b0xyg3 = period_is_shearing_pa1e1b1nwzida0e0b0xyg3 #same as calculated in the generator.
    period_is_shearing_tpa1e1b1nwzida0e0b0xyg3[0,...] = period_is_shearing_retained_pa1e1b1nwzida0e0b0xyg3

    ##Dams
    ###t0 = sale after shearing
    ###t1 = sale drys
    ###t>=2 = retain
    ### calc shearing then determine sale - if this ever gets a t axis husbandry will need to be altered (f1_adjust_triggervalues_for_t)
    shear_period_pa1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(p_index_pa1e1b1nwzida0e0b0xyg * period_is_shearing_pa1e1b1nwzida0e0b0xyg1)
    ### all shearing in all t slices is determined by the main shearing date (shearing is the same for all t slices)
    ###determine t0 sale slice - note sale must occur in the same dvp as shearing so the offset is capped if shearing occurs near the end of a period
    sale_delay_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(sale_delay_sa1e1b1nwzida0e0b0xyg1, a_prev_s_pa1e1b1nwzida0e0b0xyg1,0)
    a_dvpnext_p_va1e1b1nwzida0e0b0xyg1 = np.roll(a_p_va1e1b1nwzida0e0b0xyg1,-1,0) #roll backwards to get the gen period index of the next dvp
    a_dvpnext_p_va1e1b1nwzida0e0b0xyg1[-1,...] = len_p #set the last element to the length of p (because the end period is the equivalent of the next dvp for the end dvp)
    next_dvp_index_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(a_dvpnext_p_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, 0) #points at the next dvp from the date at the start of the period
    periods_to_dvp = next_dvp_index_pa1e1b1nwzida0e0b0xyg1 - (p_index_pa1e1b1nwzida0e0b0xyg + 1) #periods to the next dvp. +1 because if the next period is the new dvp you must sell in the current period
    sale_period_pa1e1b1nwzida0e0b0xyg1 = np.minimum(sale_delay_pa1e1b1nwzida0e0b0xyg1, periods_to_dvp) + shear_period_pa1e1b1nwzida0e0b0xyg1
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1 = sale_period_pa1e1b1nwzida0e0b0xyg1 == p_index_pa1e1b1nwzida0e0b0xyg
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1[0] = False #don't want period 0 to be sale (but it will default to sale because the sale period association is 0 at the beginning which ==index in p[0])
    ###determine t1 slice - dry dams sold at scanning
    period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1 = period_is_scan_pa1e1b1nwzida0e0b0xyg1 * (scan_management_pa1e1b1nwzida0e0b0xyg1>=1) * (not pinp.sheep['i_dry_retained_forced']) #not is required because variable is drys off hand ie sold. if forced to retain the variable wants to be false
    period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1 = period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1 * (nfoet_b1nwzida0e0b0xyg==0) #make sure selling is not an option for animals with foet (have to do it this way so that b axis is added)
    period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1[:,:,:,0:1,...] = False #make sure selling is not an option for not mated  todo may turn on again in seasonality version
    ###combine sale t slices (t0 & t1) to produce period is sale
    shape =  tuple(np.maximum.reduce([period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1.shape, period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1.shape]))
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1 = np.zeros((len_t1,)+shape, dtype=bool) #initialise on hand array with 3 t slices.
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[...]=False
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[0] = period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1[1] = period_is_sale_drys_pa1e1b1nwzida0e0b0xyg1
    ###bound female sale age - this sets the minimum age dams can be sold. Default is no min age eg can be sold anytime.
    min_age_female_sale_g1 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_g1'][mask_dams_inc_g1], 5)
    ewe_sale_mask_pa1e1b1nwzida0e0b0xyg1 = age_start_pa1e1b1nwzida0e0b0xyg1 > min_age_female_sale_g1
    period_is_sale_tpa1e1b1nwzida0e0b0xyg1 = np.logical_and(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, ewe_sale_mask_pa1e1b1nwzida0e0b0xyg1)

    ####transfer - calculate period_is_finish when the dams are transferred from the current g slice to the destination g slice
    period_is_transfer_tpa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(nextperiod_is_prejoin_pa1e1b1nwzida0e0b0xyg1[na, ...], a_g1_tpa1e1b1nwzida0e0b0xyg1, -1) * transfer_exists_tpa1e1b1nwzida0e0b0xyg1
    ###on hand - combine period_is_sale & period_is_transfer then use cumulative max to convert to on_hand
    off_hand_tpa1e1b1nwzida0e0b0xyg1= sfun.f1_cum_dvp(np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg1,period_is_transfer_tpa1e1b1nwzida0e0b0xyg1),a_v_pa1e1b1nwzida0e0b0xyg1,axis=1, shift=1) #this ensures that once they are sold they remain off hand for the rest of the dvp, shift =1 so that sheep are on-hand in the period they are sold because sale is end of period
    on_hand_tpa1e1b1nwzida0e0b0xyg1 = np.logical_not(off_hand_tpa1e1b1nwzida0e0b0xyg1)

    ##Yatf
    ###t0 = sold at weaning as sucker, t1 & t2 = retained
    ###the other t slices are added further down in the code
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2 = period_is_wean_pa1e1b1nwzida0e0b0xyg2
    ###bound female sale age - this sets the minimum age a female prog can be sold. Default is no min age eg can be sold anytime.
    min_age_female_sale_g2 = fun.f_sa(np.array([0]), sen.sav['bnd_min_sale_age_female_g3'][mask_dams_inc_g1], 5)
    ewe_sale_mask_pa1e1b1nwzida0e0b0xyg2 = np.logical_or((gender_xyg[mask_x] != 1), age_start_pa1e1b1nwzida0e0b0xyg2 > min_age_female_sale_g2)
    period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2 = np.logical_and(period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2, ewe_sale_mask_pa1e1b1nwzida0e0b0xyg2)


    ######################
    #calc cost and income#
    ######################
    calc_cost_start = time.time()

    ##price variation scalars
    ###c1 prob
    prob_c1 = uinp.price_variation['prob_c1']
    prob_c1tpg = fun.f_expand(prob_c1, p_pos-3)
    ###c1z wool price scalar
    wool_price_scalar_c1z = zfun.f_seasonal_inp(uinp.price_variation_inp['wool_price_scalar_c1z'], numpy=True, axis=-1)
    wool_price_scalar_c1w4tpg = fun.f_expand(wool_price_scalar_c1z, z_pos, left_pos2=p_pos-3, right_pos2=z_pos)
    ###c1z sale price scalar
    sale_price_scalar_c1z = zfun.f_seasonal_inp(uinp.price_variation_inp['meat_price_scalar_c1z'], numpy=True, axis=-1)
    sale_price_scalar_c1s7tpg = fun.f_expand(sale_price_scalar_c1z, z_pos, left_pos2=p_pos-3, right_pos2=z_pos)


    ##purchase cost
    purchcost_p7tpa1e1b1nwzida0e0b0xyg0 = purchcost_g0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    purchcost_wc_c0p7tpa1e1b1nwzida0e0b0xyg0 = purchcost_g0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg

    ##calc wool value - To speed the calculation process the p array is condensed to only include periods where shearing occurs. Using a slightly different association it is then converted to a v array (this process usually used a p to v association, in this case we use s to v association).
    ###create mask which is the periods where shearing occurs
    shear_mask_p0 = np.any(np.logical_or(period_is_shearing_pa1e1b1nwzida0e0b0xyg0, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg), axis=tuple(range(p_pos+1,0)))
    shear_mask_p1 = np.any(np.logical_or(period_is_shearing_pa1e1b1nwzida0e0b0xyg1, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg), axis=tuple(range(p_pos+1,0)))
    shear_mask_p3 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_shearing_tpa1e1b1nwzida0e0b0xyg3, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p]), preserveAxis=1) #preforms np.any across all axis except axis 1
    ###create association between p and s
    a_p_p9a1e1b1nwzida0e0b0xyg0 = fun.f_expand(np.nonzero(shear_mask_p0)[0],p_pos)  #take [0] because nonzero function returns tuple
    a_p_p9a1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.nonzero(shear_mask_p1)[0],p_pos)  #take [0] because nonzero function returns tuple
    a_p_p9a1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.nonzero(shear_mask_p3)[0],p_pos)  #take [0] because nonzero function returns tuple
    index_p9a1e1b1nwzida0e0b0xyg0 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p0)),p_pos)
    index_p9a1e1b1nwzida0e0b0xyg1 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p1)),p_pos)
    index_p9a1e1b1nwzida0e0b0xyg3 = fun.f_expand(np.arange(np.count_nonzero(shear_mask_p3)),p_pos)
    ###convert period is shearing array to the condensed version
    period_is_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.compress(shear_mask_p0, period_is_shearing_pa1e1b1nwzida0e0b0xyg0, p_pos)
    period_is_shearing_p9a1e1b1nwzida0e0b0xyg1 = np.compress(shear_mask_p1, period_is_shearing_pa1e1b1nwzida0e0b0xyg1, p_pos)
    period_is_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.compress(shear_mask_p3, period_is_shearing_tpa1e1b1nwzida0e0b0xyg3, p_pos)
    ###Vegatative Matter if shorn(end)
    vm_p9a1e1b1nwzida0e0b0xyg0 = np.compress(shear_mask_p0, np.take(vm_p4a1e1b1nwzida0e0b0xyg,a_p4_p,p_pos), p_pos) #expand p4 axis to p then mask to p9
    vm_p9a1e1b1nwzida0e0b0xyg1 = np.compress(shear_mask_p1, np.take(vm_p4a1e1b1nwzida0e0b0xyg,a_p4_p,p_pos), p_pos) #expand p4 axis to p then mask to p9
    vm_p9a1e1b1nwzida0e0b0xyg3 = np.compress(shear_mask_p3, np.take(vm_p4a1e1b1nwzida0e0b0xyg,a_p4_p[mask_p_offs_p],p_pos), p_pos)#expand p4 axis to p then mask to p9
    ###pmb - a little complex because it is dependent on time since previous shearing
    pmb_p9s4a1e1b1nwzida0e0b0xyg0 = pmb_p4s4a1e1b1nwzida0e0b0xyg[a_p4_p,...][shear_mask_p0]
    pmb_p9s4a1e1b1nwzida0e0b0xyg1 = pmb_p4s4a1e1b1nwzida0e0b0xyg[a_p4_p,...][shear_mask_p1]
    pmb_p9s4a1e1b1nwzida0e0b0xyg3 = pmb_p4s4a1e1b1nwzida0e0b0xyg[a_p4_p[mask_p_offs_p],...][shear_mask_p3]
    period_current_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg0 * period_is_shearing_p9a1e1b1nwzida0e0b0xyg0, axis=0) #returns the period number that the most recent shearing occurred
    period_current_shearing_p9a1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg1 * period_is_shearing_p9a1e1b1nwzida0e0b0xyg1, axis=0) #returns the period number that the most recent shearing occurred
    period_current_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(a_p_p9a1e1b1nwzida0e0b0xyg3 * period_is_shearing_tp9a1e1b1nwzida0e0b0xyg3, axis=1) #returns the period number that the most recent shearing occurred
    period_previous_shearing_p9a1e1b1nwzida0e0b0xyg0 = np.roll(period_current_shearing_p9a1e1b1nwzida0e0b0xyg0, 1, axis=0)*(index_p9a1e1b1nwzida0e0b0xyg0>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    period_previous_shearing_p9a1e1b1nwzida0e0b0xyg1 = np.roll(period_current_shearing_p9a1e1b1nwzida0e0b0xyg1, 1, axis=0)*(index_p9a1e1b1nwzida0e0b0xyg1>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg3 = np.roll(period_current_shearing_tp9a1e1b1nwzida0e0b0xyg3, 1, axis=1)*(index_p9a1e1b1nwzida0e0b0xyg3>0)  #returns the period of the previous shearing and sets slice 0 to 0 (because there is no previous shearing for the first shearing)
    periods_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = a_p_p9a1e1b1nwzida0e0b0xyg0 - np.maximum(period_previous_shearing_p9a1e1b1nwzida0e0b0xyg0, date_born_idx_ida0e0b0xyg0)
    periods_since_shearing_p9a1e1b1nwzida0e0b0xyg1 = a_p_p9a1e1b1nwzida0e0b0xyg1 - np.maximum(period_previous_shearing_p9a1e1b1nwzida0e0b0xyg1, date_born_idx_ida0e0b0xyg1)
    periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = a_p_p9a1e1b1nwzida0e0b0xyg3 - np.maximum(period_previous_shearing_tp9a1e1b1nwzida0e0b0xyg3, date_born_idx_ida0e0b0xyg3)
    months_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = periods_since_shearing_p9a1e1b1nwzida0e0b0xyg0 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesnt need to be perfect its only an approximation)
    months_since_shearing_p9a1e1b1nwzida0e0b0xyg1 = periods_since_shearing_p9a1e1b1nwzida0e0b0xyg1 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesnt need to be perfect its only an approximation)
    months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = periods_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 * 7 / 30 #times 7 for day in period and div 30 to convert to months (this doesnt need to be perfect its only an approximation)
    a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg0 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_p9a1e1b1nwzida0e0b0xyg0)#provides the index of the index which is closest to the actual months since shearing
    a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg1 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_p9a1e1b1nwzida0e0b0xyg1)#provides the index of the index which is closest to the actual months since shearing
    a_months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3 = fun.f_find_closest(pinp.sheep['i_pmb_interval'], months_since_shearing_tp9a1e1b1nwzida0e0b0xyg3)#provides the index of the index which is closest to the actual months since shearing
    pmb_p9a1e1b1nwzida0e0b0xyg0 = np.squeeze(np.take_along_axis(pmb_p9s4a1e1b1nwzida0e0b0xyg0,a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg0[:,na,...],1),axis=p_pos) #select the relevant s4 (pmb interval) then squeeze that axis
    pmb_p9a1e1b1nwzida0e0b0xyg1 = np.squeeze(np.take_along_axis(pmb_p9s4a1e1b1nwzida0e0b0xyg1,a_months_since_shearing_p9a1e1b1nwzida0e0b0xyg1[:,na,...],1),axis=p_pos) #select the relevant s4 (pmb interval) then squeeze that axis
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
    woolp_mpg_w4 = sfun.f1_woolprice().astype(dtype)/100
    r_vals['woolp_mpg_w4'] = woolp_mpg_w4
    r_vals['fd_range'] = uinp.sheep['i_woolp_fd_range_w4']
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0[:,:,shear_mask_p0], woolp_stbnib_sire = (
                            sfun.f_wool_value(woolp_mpg_w4, wool_price_scalar_c1w4tpg, cfw_sire_p9, fd_sire_p9, sl_sire_p9, ss_sire_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg0, pmb_p9a1e1b1nwzida0e0b0xyg0, dtype))
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1[:,:,shear_mask_p1], woolp_stbnib_dams = (
                            sfun.f_wool_value(woolp_mpg_w4, wool_price_scalar_c1w4tpg, cfw_dams_p9, fd_dams_p9, sl_dams_p9, ss_dams_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg1, pmb_p9a1e1b1nwzida0e0b0xyg1, dtype))
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg3[:,:,shear_mask_p3], woolp_stbnib_offs = (
                            sfun.f_wool_value(woolp_mpg_w4, wool_price_scalar_c1w4tpg, cfw_offs_p9, fd_offs_p9, sl_offs_p9, ss_offs_p9
                                              , vm_p9a1e1b1nwzida0e0b0xyg3, pmb_tp9a1e1b1nwzida0e0b0xyg3, dtype))

    ###create woolvalue with average c1 - this is used for wc/minroe and reporting because we dont think c1 is needed for them
    woolvalue_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    woolvalue_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    woolvalue_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(woolvalue_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)
    wool_finish= time.time()


    ##Sale value - To speed the calculation process the p array is condensed to only include periods where shearing occurs. Using a slightly different association it is then converted to a v array (this process usually used a p to v association, in this case we use s to v association).
    ###create mask which is the periods where shearing occurs
    sale_mask_p0 = np.any(period_is_sale_pa1e1b1nwzida0e0b0xyg0, axis=tuple(range(p_pos+1,0)))
    sale_mask_p1 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg1, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos)  #preforms np.any on all axis except 1. only use the sale slices from the dam t axis
    sale_mask_p2 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_t0_pa1e1b1nwzida0e0b0xyg2, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg), preserveAxis=p_pos)  #preforms np.any on all axis except 1. only use the sale slices from the dam t axis
    sale_mask_p3 = fun.f_reduce_skipfew(np.any, np.logical_or(period_is_sale_tpa1e1b1nwzida0e0b0xyg3, period_is_assetvalue_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p]), preserveAxis=p_pos)  #preforms np.any on all axis except 1
    ###manipulate axis with associations
    grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg = score_range_s8s6tpa1e1b1nwzida0e0b0xyg[uinp.sheep['ia_s8_s7']] #s8 to s7
    month_scalar_s7tpa1e1b1nwzida0e0b0xyg = price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, a_p4_p][:,0] #month to p, then slice s9 (has to be seperate because otherwise advanced indexing is triggered)
    month_discount_s7tpa1e1b1nwzida0e0b0xyg = price_adj_months_s7s9tp4a1e1b1nwzida0e0b0xyg[:, :, :, a_p4_p][:,1] #month to p, then slice s9 (has to be seperate because otherwise advanced indexing is triggered)
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
        cu0_sire.astype(dtype), cx_sire[:,0:1,...].astype(dtype), rc_start_sire_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg0
        , dresspercent_adj_yg0, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg, dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg0
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg0, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg0, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg[...,0:1,:,:], sale_agemax_s7tpa1e1b1nwzida0e0b0xyg0, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg0, dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg1[:,:,sale_mask_p1], r_salegrid_c1tpa1e1b1nwzida0e0b0xyg1[:,:,sale_mask_p1] = sfun.f_sale_value(
        cu0_dams.astype(dtype), cx_dams[:,1:2,...].astype(dtype), rc_start_dams_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg1
        , dresspercent_adj_yg1, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg1
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg1, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg1, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg[...,1:2,:,:], sale_agemax_s7tpa1e1b1nwzida0e0b0xyg1, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg1, dtype)
    salevalue_c1tp9a1e1b1nwzida0e0b0xyg2, r_salegrid_c1tpa1e1b1nwzida0e0b0xyg2[:,:,sale_mask_p2] = sfun.f_sale_value(                                                #keep it as a condensed p axis
        cu0_yatf.astype(dtype), cx_yatf[:,mask_x,...].astype(dtype), rc_start_yatf_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg2
        , dresspercent_adj_yg2, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg2
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg2, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg2, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg2, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg2, dtype)
    salevalue_c1tpa1e1b1nwzida0e0b0xyg3[:,:,sale_mask_p3], r_salegrid_c1tpa1e1b1nwzida0e0b0xyg3[:,:,sale_mask_p3] = sfun.f_sale_value(
        cu0_offs, cx_offs[:,mask_x,...].astype(dtype), rc_start_offs_tp9g, ffcfw_tp9a1e1b1nwzida0e0b0xyg3
        , dresspercent_adj_yg3, dresspercent_adj_s6tpa1e1b1nwzida0e0b0xyg,dresspercent_adj_s7tpa1e1b1nwzida0e0b0xyg
        , grid_price_s7s5s6tpa1e1b1nwzida0e0b0xyg, sale_price_scalar_c1s7tpg, month_scalar_s7tp9a1e1b1nwzida0e0b0xyg3
        , month_discount_s7tp9a1e1b1nwzida0e0b0xyg3, price_type_s7tpa1e1b1nwzida0e0b0xyg, cvlw_s7s5tpa1e1b1nwzida0e0b0xyg
        , cvscore_s7s6tpa1e1b1nwzida0e0b0xyg, grid_weightrange_s7s5tpa1e1b1nwzida0e0b0xyg, grid_scorerange_s7s6tpa1e1b1nwzida0e0b0xyg
        , age_end_p9a1e1b1nwzida0e0b0xyg3, discount_age_s7tpa1e1b1nwzida0e0b0xyg
        , sale_cost_pc_s7tpa1e1b1nwzida0e0b0xyg, sale_cost_hd_s7tpa1e1b1nwzida0e0b0xyg
        , mask_s7x_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemax_s7tpa1e1b1nwzida0e0b0xyg3, sale_agemin_s7tpa1e1b1nwzida0e0b0xyg3, dtype)

    ###create woolvalue with average c1 - this is used for wc/minroe and reporting because we dont think c1 is needed for them
    salevalue_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    salevalue_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    salevalue_tp9a1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(salevalue_c1tp9a1e1b1nwzida0e0b0xyg2, prob_c1tpg, axis=0)
    salevalue_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(salevalue_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg0 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg0, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg1, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg2 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg2, prob_c1tpg, axis=0)
    r_salegrid_tpa1e1b1nwzida0e0b0xyg3 = fun.f_weighted_average(r_salegrid_c1tpa1e1b1nwzida0e0b0xyg3, prob_c1tpg, axis=0)

    sale_finish= time.time()

    ##Husbandry - shearing costs apply to p[0] but they are dropped because no numbers in p[0] #todo add feedbudgeting and labour for maintenance of infrastructure (it is currently has a cost that is representing materials and labour)
    ###Sire: cost, labour and infrastructure requirements
    husbandry_cost_tpg0, husbandry_labour_l2tpg0, husbandry_infrastructure_h1tpg0 = sfun.f_husbandry(
        uinp.sheep['i_head_adjust_sire'], mobsize_pa1e1b1nwzida0e0b0xyg0, o_ffcfw_tpsire, o_cfw_tpsire, operations_triggerlevels_h5h7h2tpg,
        p_index_pa1e1b1nwzida0e0b0xyg, age_start_pa1e1b1nwzida0e0b0xyg0, period_is_shearing_pa1e1b1nwzida0e0b0xyg0,
        period_is_wean_pa1e1b1nwzida0e0b0xyg0, gender_xyg[0], o_ebg_tpsire, wool_genes_yg0, husb_operations_muster_propn_h2tpg,
        husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg, operations_per_hour_l2h2tpg,
        husb_operations_infrastructurereq_h1h2tpg, husb_operations_contract_cost_h2tpg, husb_muster_requisites_prob_h6h4tpg,
        musters_per_hour_l2h4tpg, husb_muster_infrastructurereq_h1h4tpg, dtype=dtype)
    husbandry_cost_p7tpg0 = husbandry_cost_tpg0 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg
    husbandry_cost_wc_c0p7tpg0 = husbandry_cost_tpg0 * wc_allocation_c0p7tpa1e1b1nwzida0e0b0xyg
    ###Dams: cost, labour and infrastructure requirements - accounts for yatf costs as well
    ### for dams remove the generator t axis by selecting the retained t slice. This reduces the array sizes and doesnt lose much detail.
    if len_gen_t1==1:
        a_gen_t_g1 = np.array([0])
    else:
        a_gen_t_g1 = a_t_g1
    a_t_tpg1 = fun.f_expand(a_gen_t_g1, p_pos-2, right_pos=-1)
    t_o_ffcfw_tpdams = np.take_along_axis(o_ffcfw_tpdams, a_t_tpg1, p_pos-1)
    t_o_cfw_tpdams = np.take_along_axis(o_cfw_tpdams, a_t_tpg1, p_pos-1)
    t_o_ebg_tpdams = np.take_along_axis(o_ebg_tpdams, a_t_tpg1, p_pos-1)
    husbandry_cost_tpg1, husbandry_labour_l2tpg1, husbandry_infrastructure_h1tpg1 = sfun.f_husbandry(
        uinp.sheep['i_head_adjust_dams'], mobsize_pa1e1b1nwzida0e0b0xyg1, t_o_ffcfw_tpdams, t_o_cfw_tpdams, operations_triggerlevels_h5h7h2tpg,
        p_index_pa1e1b1nwzida0e0b0xyg, age_start_pa1e1b1nwzida0e0b0xyg1, period_is_shearing_pa1e1b1nwzida0e0b0xyg1,
        period_is_wean_pa1e1b1nwzida0e0b0xyg1, gender_xyg[1], t_o_ebg_tpdams, wool_genes_yg1, husb_operations_muster_propn_h2tpg,
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
    rm_start_c0 = per.f_cashflow_date() + np.timedelta64(182,'D')#Overheads are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even
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
    assetvalue_p7tpa1e1b1nwzida0e0b0xyg0 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg0 + woolvalue_tpa1e1b1nwzida0e0b0xyg0)
                                            * period_is_assetvalue_pa1e1b1nwzida0e0b0xyg * alloc_p7tpa1e1b1nwzida0e0b0xyg)
    ####adjust for period is sale/shear (needs to be done here rather than p2v so that cashflow can be combined).
    salevalue_tpa1e1b1nwzida0e0b0xyg0 = salevalue_tpa1e1b1nwzida0e0b0xyg0 * period_is_sale_pa1e1b1nwzida0e0b0xyg0
    salevalue_c1tpa1e1b1nwzida0e0b0xyg0 = salevalue_c1tpa1e1b1nwzida0e0b0xyg0 * period_is_sale_pa1e1b1nwzida0e0b0xyg0
    woolvalue_tpa1e1b1nwzida0e0b0xyg0 = woolvalue_tpa1e1b1nwzida0e0b0xyg0 * period_is_shearing_pa1e1b1nwzida0e0b0xyg0
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg0 * period_is_shearing_pa1e1b1nwzida0e0b0xyg0
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
    ####asset value - calc asset value before adjusting by period is sale and shearing
    assetvalue_p7tpa1e1b1nwzida0e0b0xyg1 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg1 + woolvalue_tpa1e1b1nwzida0e0b0xyg1)
                                            * period_is_assetvalue_pa1e1b1nwzida0e0b0xyg * alloc_p7tpa1e1b1nwzida0e0b0xyg)
    ####adjust for period is sale/shear (needs to be done here rather than p2v so that cashflow can be combined).
    salevalue_tpa1e1b1nwzida0e0b0xyg1 = salevalue_tpa1e1b1nwzida0e0b0xyg1 * period_is_sale_tpa1e1b1nwzida0e0b0xyg1
    salevalue_c1tpa1e1b1nwzida0e0b0xyg1 = salevalue_c1tpa1e1b1nwzida0e0b0xyg1 * period_is_sale_tpa1e1b1nwzida0e0b0xyg1
    woolvalue_tpa1e1b1nwzida0e0b0xyg1 = woolvalue_tpa1e1b1nwzida0e0b0xyg1 * period_is_shearing_pa1e1b1nwzida0e0b0xyg1
    woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 = woolvalue_c1tpa1e1b1nwzida0e0b0xyg1 * period_is_shearing_pa1e1b1nwzida0e0b0xyg1
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
    assetvalue_p7tpa1e1b1nwzida0e0b0xyg3 =  ((salevalue_tpa1e1b1nwzida0e0b0xyg3 + woolvalue_tpa1e1b1nwzida0e0b0xyg3)
                                            * period_is_assetvalue_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p] * alloc_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p,...])
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
    r_salegrid_tpa1e1b1nwzida0e0b0xyg3 = r_salegrid_tpa1e1b1nwzida0e0b0xyg3 * period_is_sale_tpa1e1b1nwzida0e0b0xyg3
    r_salevalue_p7tpa1e1b1nwzida0e0b0xyg3 = salevalue_tpa1e1b1nwzida0e0b0xyg3 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]
    r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg3 = woolvalue_tpa1e1b1nwzida0e0b0xyg3 * cash_allocation_p7tpa1e1b1nwzida0e0b0xyg[:,:,mask_p_offs_p]


    ######################
    # add yatf to dams   #
    ######################
    o_pi_tpdams *= fun.f_divide(o_mei_solid_tpdams + np.sum(o_mei_solid_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True),
                              o_mei_solid_tpdams, dtype=dtype)  # done before adding yatf mei. This is instead of adding pi yatf with pi dams because some of the potential intake of the yatf is 'used' consuming milk. Doing it via mei keeps the ratio mei_dams/pi_dams the same before and after adding the yatf. This is what we want because it is saying that there is a given energy intake and it needs to be of a certain quality.
    o_mei_solid_tpdams = o_mei_solid_tpdams + np.sum(o_mei_solid_tpyatf * gender_propn_xyg, axis=x_pos, keepdims=True)


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
    ###This needs to be calculated (rather than using feedsupplyw) because feedsupply could be update in the lw target loop.
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
    ### an animal has an nv that is the mid-point of a feed pool then most of the mei & pi for that animal will occur
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

    ##allocate each sheep class to an nv group
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
    ##cost requires a c axis for reporting - it is summed before converting to a param because MINROE doesnt need c axis
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
    assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(assetvalue_p7tpa1e1b1nwzida0e0b0xyg0, numbers_p=o_numbers_end_tpsire,
                                              on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0)[:,:,na,...]#add singleton v
    ###dams
    cashflow_c1p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(cashflow_c1p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    wc_c0p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(wc_c0p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    cost_p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(husbandry_cost_p7tpg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    assetvalue_p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(assetvalue_p7tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg1)
    ###yatf can be sold as sucker, not shorn therefore only include sale value. husbandry is accounted for with dams so don't need that here.
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
    assetvalue_p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(assetvalue_p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
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
    ####sires - dont need any special treatment because they dont have dvps - sires only have one dvp which essentially starts when the activity is purchased
    numbers_start_tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_numbers_start_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                          period_is_tvp=period_is_startdvp_purchase_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    numbers_startp8_tva1e1b1nwzida0e0b0xyg0p8 = sfun.f1_p2v_std(o_numbers_start_tpsire[...,na], on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0[...,na],
                                                          period_is_tvp=period_is_startp8_pa1e1b1nwzida0e0b0xyg0p8, sumadj=1)[:,na,...]#add singleton v
    ####dams - need special dvp treatment for 0 day dvp. Essentially just set the start and end number to the same so that 1:1 transfefr can occur.
    numbers_start_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_start_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1)
    numbers_start_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_start_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)
    numbers_end_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_end_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                                                     period_is_tp=np.logical_or(period_is_transfer_tpa1e1b1nwzida0e0b0xyg1, nextperiod_is_startdvp_pa1e1b1nwzida0e0b0xyg1))  #on_hand included so that the early termination of a dvp is accounted for when transferring between ram groups
    numbers_end_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_end_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,
                                                          a_v_pa1e1b1nwzida0e0b0xyg1,numbers_start_tva1e1b1nwzida0e0b0xyg1) #intentionally numbers start - want numbers start and end to be the same for 0 day dvp so that 1:1 transfer happens.
    numbers_start_d_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_numbers_start_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,period_is_tp=period_is_startdvp_pa1e1b1nwzida0e0b0xyg1
                                                         , a_any1_p=a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2, index_any1tp=index_da0e0b0xyg) #with a d axis.
    numbers_start_d_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(numbers_start_d_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)

    ####offs - need special dvp treatment for 0 day dvp. Essentially just set the start and end number to the same so that 1:1 transfefr can occur.
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
    ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_start_tpyatf, a_v_pa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_wean_pa1e1b1nwzida0e0b0xyg2)
    ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_start_v_yatf_tva1e1b1nwzida0e0b0xyg1,a_p_va1e1b1nwzida0e0b0xyg1,a_v_pa1e1b1nwzida0e0b0xyg1)

    #### Return the weight of the yatf in the period in which they are weaned - with active d axis
    ffcfw_start_d_yatf_ta1e1b1nwzida0e0b0xyg2 = sfun.f1_p2v_std(o_ffcfw_start_tpyatf, period_is_tvp=period_is_wean_pa1e1b1nwzida0e0b0xyg2,
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
                                                * (scan_va1e1b1nwzida0e0b0xyg1[...,na,na]==index_s[:,na]), axis = (-1,-2))
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
    a_k5cluster_da0e0b0xyg3 = np.sum(a_k5cluster_b0xygls * (gbal_da0e0b0xyg3[...,na,na]==index_l[:,na]) * (scan_da0e0b0xyg3[...,na,na]==index_s), axis = (-1,-2))
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
        variable provides to a unique constraint and therefore some decision variables and constraints can be masked out
        The inclusion of multiple FVPs in a DVP requires a variable step size based on the number of FVPs in a DVP (n_damfvps_v)
        and the number of FVPs prior to this DVP (n_prior_damfvps_v)
        step is equal to the total number of patterns i_w_start_len * (i_n_len ** i_n_fvp_period) divided by the number 
        of constraints : i_w_start_len * (i_n_len ** i_n_prior_damfvps_v) 
        or decision variables: i_w_start_len * (i_n_len ** (i_n_prior_damfvps_v + i_n_damfvps_v)). 
        This can be simplified to a single line equation because the terms cancel out
        
        n_fvps_v is the number of FVPs within the DVP
        n_prior_fvps_v is the number of FVPs prior to the start of this DVP since condensing. 
        So if the DVP dates are say 1 Feb, 1 May & 1 July 
        and the FVP dates are 1 Feb, 1 May, 1 June, 1 July, 1 Oct. Then 
        n_fvps_v = 1,2,2 because 1 fvp in the first DVP and 2 in each of the other two DVPs
        n_prior_fvps_v = 0, 1, 3 which is the cumulative sum of n_fvps_v in the previous DVPs (ie not including this DVP)
        Note: the values can change along the i axis if the FVP date is not relative to a reproduction event.
        '''
    allocation_start = time.time()
    ##dams
    n_fvps_va1e1b1nwzida0e0b0xyg1 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg1.shape)
    n_prior_fvps_va1e1b1nwzida0e0b0xyg1 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg1.shape)
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1.copy()
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1.copy()
    prev_condense_date_va1e1b1nwzida0e0b0xyg1 = dvp_start_va1e1b1nwzida0e0b0xyg1.copy()
    ###adjust dates to only include dvps which are also fvps (extra dvps just get the same values)
    dvp_is_fvp_va1e1b1nwzida0e0b0xyg1 = np.any(dvp_start_va1e1b1nwzida0e0b0xyg1==fvp_start_fa1e1b1nwzida0e0b0xyg1[:,na,...], axis=0)
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1[~dvp_is_fvp_va1e1b1nwzida0e0b0xyg1] = 0 #set dvp dates which are not fvps to the start dvp date, these get overwritten in the next step
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1) #start of prev dvp that is also an fvp
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg1[~dvp_is_fvp_va1e1b1nwzida0e0b0xyg1] = np.max(fvp_start_fa1e1b1nwzida0e0b0xyg1) #set dvp dates which are not fvps to the start dvp date, these get overwritten in the next step
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg1 = np.flip(np.minimum.accumulate(np.flip(prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg1, axis=0)), axis=0) #start of prev dvp that is also an fvp
    ###previous condense date - used to calc number of fvps since last condensing
    prev_condense_date_va1e1b1nwzida0e0b0xyg1[~(dvp_type_va1e1b1nwzida0e0b0xyg1==condense_vtype1)] = 0
    prev_condense_date_va1e1b1nwzida0e0b0xyg1 = np.maximum.accumulate(prev_condense_date_va1e1b1nwzida0e0b0xyg1, axis=0)
    for i in range(dvp_start_va1e1b1nwzida0e0b0xyg1.shape[0]):
        if i == dvp_start_va1e1b1nwzida0e0b0xyg1.shape[0] - 1:
            dvp_start = prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1[i,...]
            prev_condense_date = prev_condense_date_va1e1b1nwzida0e0b0xyg1[i,...]
            n_fvp = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= dvp_start)).sum(axis=0)
            n_fvp_since_condense = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= prev_condense_date) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg1 < dvp_start)).sum(axis=0)
        else:
            dvp_start = prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg1[i,...]
            prev_condense_date = prev_condense_date_va1e1b1nwzida0e0b0xyg1[i,...]
            dvp_end = prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg1[i + 1,...]
            n_fvp = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= dvp_start) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg1 < dvp_end)).sum(axis=0)
            n_fvp_since_condense = ((fvp_start_fa1e1b1nwzida0e0b0xyg1 >= prev_condense_date) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg1 < dvp_start)).sum(axis=0)
        n_fvps_va1e1b1nwzida0e0b0xyg1[i,...] = n_fvp
        n_prior_fvps_va1e1b1nwzida0e0b0xyg1[i,...] = n_fvp_since_condense

    ###Steps for ‘Numbers Requires’ constraint is determined by the number of prior FVPs
    step_con_req_va1e1b1nw8zida0e0b0xyg1 = np.power(n_fs_dams, (n_fvp_periods_dams
                                                              - n_prior_fvps_va1e1b1nwzida0e0b0xyg1))
    ###Steps for the decision variables is determined by the number of current & prior FVPs
    step_dv_va1e1b1nw8zida0e0b0xyg1 = np.power(n_fs_dams, (n_fvp_periods_dams
                                                         - n_prior_fvps_va1e1b1nwzida0e0b0xyg1
                                                         - n_fvps_va1e1b1nwzida0e0b0xyg1))
    ###Steps for ‘Numbers Provides’ is calculated with a t axis (because the t axis can alter the dvp type of the source relative to the destination)
    step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9 = fun.f_update(step_dv_va1e1b1nw8zida0e0b0xyg1, n_fs_dams ** n_fvp_periods_dams
                                                         , dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == condense_vtype1)[..., na]

    ##Mask the decision variables that are not active in this DVP in the matrix - because they share a common nutrition history (broadcast across t axis)
    mask_w8vars_va1e1b1nw8zida0e0b0xyg1 = index_wzida0e0b0xyg1 % step_dv_va1e1b1nw8zida0e0b0xyg1 == 0
    ##mask for nutrition profiles (this allows the user to examine certain nutrition patterns eg high high high vs low low low) - this mask is combined with the other w8 masks below
    mask_nut_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(mask_nut_oa1e1b1nwzida0e0b0xyg1, a_o_va1e1b1nwzida0e0b0xyg1, axis=0)
    ###association between the shortlist of nutrition profile inputs and the full range of LW patterns that include starting LW
    a_shortlist_w1 = index_w1 % len_nut_dams
    mask_nut_va1e1b1nwzida0e0b0xyg1 = mask_nut_va1e1b1nwzida0e0b0xyg1[:,:,:,:,:,a_shortlist_w1,...]  # expands the nutrition mask to all lw patterns.
    ### match the pattern requested with the pattern that is the 'history' for that pattern in previous DVPs
    mask_w8nut_va1e1b1nzida0e0b0xyg1w9 = np.sum(mask_nut_va1e1b1nwzida0e0b0xyg1[...,na] *
                                               (np.trunc(index_wzida0e0b0xyg1[...,na] / step_dv_va1e1b1nw8zida0e0b0xyg1[..., na])
                                                == index_w1 / step_dv_va1e1b1nw8zida0e0b0xyg1[...,na]),
                                               axis=w_pos-1) > 0 #don't keepdims
    mask_w8nut_va1e1b1nwzida0e0b0xyg1 = np.moveaxis(mask_w8nut_va1e1b1nzida0e0b0xyg1w9,-1,w_pos) #move w9 axis to w position
    ## Combine the w8vars mask and the user nutrition mask
    mask_w8vars_va1e1b1nw8zida0e0b0xyg1 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_w8nut_va1e1b1nwzida0e0b0xyg1
    ##Mask numbers provided based on the steps (with a t axis) and the next dvp type (with a t axis) (t0&1 are sold and never transfer so the mask doesnt mean anything for them. for t2 animals always transfer to themselves unless dvpnext is 'condense')
    dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg1 = np.logical_or(dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == condense_vtype1
                                                               , dvp_type_next_tva1e1b1nwzida0e0b0xyg1 == season_vtype1) #when distribution occurs any w8 can provide w9
    mask_numbers_provw8w9_tva1e1b1nw8zida0e0b0xyg1w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1[...,na] \
                        * (np.trunc((index_wzida0e0b0xyg1[...,na] * np.logical_not(dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg1[...,na])
                                     + index_w1 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg1[...,na])
                                    / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9) == index_w1 / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9)
    ##Create a mask for the distribution of w8 to w9, with w9 in the w position for the ffcfw_dest_wg
    mask_w9vars_va1e1b1nw9zida0e0b0xyg1 = (np.trunc(index_wzida0e0b0xyg1 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg1
                                                    / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9[...,0])
                                           == index_wzida0e0b0xyg1 / step_con_prov_tva1e1b1nw8zida0e0b0xyg1w9[...,0])
    ##Mask numbers required from the previous period (broadcast across t axis) - Note: req does not need a t axis because the destination decision variable don’t change for the transfer
    mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg1[...,na] \
                        * (np.trunc(index_wzida0e0b0xyg1 / step_con_req_va1e1b1nw8zida0e0b0xyg1)[...,na] == index_w1
                           / step_con_req_va1e1b1nw8zida0e0b0xyg1[...,na])

    ##offs
    ###create arrays
    n_fvps_va1e1b1nwzida0e0b0xyg3 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg3.shape)
    n_prior_fvps_va1e1b1nwzida0e0b0xyg3 = np.zeros(dvp_start_va1e1b1nwzida0e0b0xyg3.shape)
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3.copy()
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3.copy()
    prev_condense_date_va1e1b1nwzida0e0b0xyg3 = dvp_start_va1e1b1nwzida0e0b0xyg3.copy()
    ###adjust dates to only include dvps which are also fvps (extra dvps just get the same values)
    dvp_is_fvp_va1e1b1nwzida0e0b0xyg3 = np.any(dvp_start_va1e1b1nwzida0e0b0xyg3==fvp_start_fa1e1b1nwzida0e0b0xyg3[:,na,...], axis=0)
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3[~dvp_is_fvp_va1e1b1nwzida0e0b0xyg3] = 0 #set dvp dates which are not fvps to the start dvp date, these get overwritten in the next step
    prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3) #start of prev dvp that is also an fvp
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg3[~dvp_is_fvp_va1e1b1nwzida0e0b0xyg3] = np.max(fvp_start_fa1e1b1nwzida0e0b0xyg3) #set dvp dates which are not fvps to the start dvp date, these get overwritten in the next step
    prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg3 = np.flip(np.minimum.accumulate(np.flip(prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg3, axis=0)), axis=0) #start of prev dvp that is also an fvp
    ###previous condense date - used to calc number of fvps since last condensing
    prev_condense_date_va1e1b1nwzida0e0b0xyg3[~(dvp_type_va1e1b1nwzida0e0b0xyg3==condense_vtype3)] = 0
    prev_condense_date_va1e1b1nwzida0e0b0xyg3 = np.maximum.accumulate(prev_condense_date_va1e1b1nwzida0e0b0xyg3, axis=0)
    for i in range(dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0]):
        if i == dvp_start_va1e1b1nwzida0e0b0xyg3.shape[0] - 1:
            dvp_start = prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3[i,...]
            prev_condense_date = prev_condense_date_va1e1b1nwzida0e0b0xyg3[i,...]
            n_fvp = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= dvp_start)).sum(axis=0)
            n_fvp_since_condense = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= prev_condense_date) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg3 < dvp_start)).sum(axis=0)
        else:
            dvp_start = prev_dvp_is_fvp_start_va1e1b1nwzida0e0b0xyg3[i,...]
            prev_condense_date = prev_condense_date_va1e1b1nwzida0e0b0xyg3[i,...]
            dvp_end = prev_dvp_is_fvp_end_va1e1b1nwzida0e0b0xyg3[i + 1,...]
            n_fvp = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= dvp_start) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg3 < dvp_end)).sum(axis=0)
            n_fvp_since_condense = ((fvp_start_fa1e1b1nwzida0e0b0xyg3 >= prev_condense_date) & (
                        fvp_start_fa1e1b1nwzida0e0b0xyg3 < dvp_start)).sum(axis=0)
        n_fvps_va1e1b1nwzida0e0b0xyg3[i,...] = n_fvp
        n_prior_fvps_va1e1b1nwzida0e0b0xyg3[i,...] = n_fvp_since_condense

    ###Steps for ‘Numbers Requires’ constraint is determined by the number of prior FVPs
    step_con_req_va1e1b1nw8zida0e0b0xyg3 = np.power(n_fs_offs, (n_fvp_periods_offs
                                                              - n_prior_fvps_va1e1b1nwzida0e0b0xyg3))
    ###Steps for the decision variables is determined by the number of current & prior FVPs
    step_dv_va1e1b1nw8zida0e0b0xyg3 = np.power(n_fs_offs, (n_fvp_periods_offs
                                                         - n_prior_fvps_va1e1b1nwzida0e0b0xyg3
                                                         - n_fvps_va1e1b1nwzida0e0b0xyg3))
    ###Steps for ‘Numbers Provides’
    step_con_prov_va1e1b1nw8zida0e0b0xyg3w9 = fun.f_update(step_dv_va1e1b1nw8zida0e0b0xyg3, n_fs_offs ** n_fvp_periods_offs
                                                         , dvp_type_next_va1e1b1nwzida0e0b0xyg3 == condense_vtype3)[..., na]
    ##Mask the decision variables that are not active in this DVP in the matrix - because they share a common nutrition history (broadcast across t axis)
    mask_w8vars_va1e1b1nw8zida0e0b0xyg3 = (index_wzida0e0b0xyg3 % step_dv_va1e1b1nw8zida0e0b0xyg3) == 0
    ##mask for nutrition profiles (this allows the user to examine certain nutrition patterns eg high high high vs low low low) - this mask is renamed the w8 masks to be consistent with dams
    mask_nut_va1e1b1nwzida0e0b0xyg3 = np.take_along_axis(mask_nut_sa1e1b1nwzida0e0b0xyg3, a_s_va1e1b1nwzida0e0b0xyg3, axis=0)
    ###association between the shortlist of nutrition profile inputs and the full range of LW patterns that include starting LW
    a_shortlist_w3 = index_w3 % len_nut_offs
    mask_nut_va1e1b1nwzida0e0b0xyg3 = mask_nut_va1e1b1nwzida0e0b0xyg3[:,:,:,:,:,a_shortlist_w3,...]  # expands the nutrition mask to all lw patterns.
    ### match the pattern requested with the pattern that is the 'history' for that pattern in previous DVPs
    mask_w8nut_va1e1b1nzida0e0b0xyg3w9 = np.sum(mask_nut_va1e1b1nwzida0e0b0xyg3[...,na] *
                                               (np.trunc(index_wzida0e0b0xyg3[...,na] / step_dv_va1e1b1nw8zida0e0b0xyg3[..., na])
                                                == index_w3 / step_dv_va1e1b1nw8zida0e0b0xyg3[...,na]),
                                               axis=w_pos-1) > 0 #don't keepdims
    mask_w8nut_va1e1b1nwzida0e0b0xyg3 = np.moveaxis(mask_w8nut_va1e1b1nzida0e0b0xyg3w9,-1,w_pos) #move w9 axis to w position
    ## Combine the w8vars mask and the user nutrition mask
    mask_w8vars_va1e1b1nw8zida0e0b0xyg3 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_w8nut_va1e1b1nwzida0e0b0xyg3
    ##Mask numbers provided based on the steps (with a t axis) and the next dvp type (with a t axis) (t0&1 are sold and never transfer so the mask doesnt mean anything for them. for t2 animals always transfer to themselves unless dvpnext is 'condense')
    dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3 = np.logical_or(dvp_type_next_va1e1b1nwzida0e0b0xyg3 == condense_vtype3
                                                               , dvp_type_next_va1e1b1nwzida0e0b0xyg3 == season_vtype3) #when distribution occurs any w8 can provide w9
    mask_numbers_provw8w9_va1e1b1nw8zida0e0b0xyg3w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3[...,na] \
                        * (np.trunc((index_wzida0e0b0xyg3[...,na] * np.logical_not(dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3[...,na])
                                     + index_w3 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3[...,na])
                                    / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9) == index_w3 / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9)
    ##Create a mask for the distribution of w8 to w9, with w9 in the w position for the ffcfw_dest_wg
    mask_w9vars_va1e1b1nw9zida0e0b0xyg3 = (np.trunc(index_wzida0e0b0xyg3 * dist_occurs_nextdvp_va1e1b1nwzida0e0b0xyg3
                                                    / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9[...,0])
                                           == index_wzida0e0b0xyg3 / step_con_prov_va1e1b1nw8zida0e0b0xyg3w9[...,0])
    ##Mask numbers required from the previous period (broadcast across t axis) - Note: req does not need a t axis because the destination decision variable don’t change for the transfer
    mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg3w9 = mask_w8vars_va1e1b1nw8zida0e0b0xyg3[...,na] \
                        * (np.trunc(index_wzida0e0b0xyg3 / step_con_req_va1e1b1nw8zida0e0b0xyg3)[...,na] == index_w3
                           / step_con_req_va1e1b1nw8zida0e0b0xyg3[...,na])

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
    period_is_transfer_tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(period_is_transfer_tpa1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
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

    ###numbers are required across the identity array between g1 & g9 in the periods that the transfer decision variable exists exists
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
    Dry dams must be sold before the next prejoining (eg they can be sold in any sale opp).'''
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

    ################################
    # Create Season transfer masks #
    ################################
    '''If a season is not identified then it does not transfer any parameters. Therefore to reduce size we can mask
    all parameters with a z8 axis. We also require a z8z9 mask which controls transfer params.
    z8z9 is required for parameters from the previous period that provide/require in the current period because 
    if a season is identified in a given dvp it provides to multiple z slices in the next dvp.'''

    ##inputs
    date_initiate_z = zfun.f_seasonal_inp(pinp.general['i_date_initiate_z'], numpy=True, axis=0).astype('datetime64')
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
    ffcfw_source_season_va1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                             period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg)  #numbers not required for ffcfw
    ffcfw_source_season_va1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_source_season_va1e1b1nwzida0e0b0xyg1,
                                                                 a_p_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
    ###offs
    ffcfw_source_condense_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                               period_is_tp=nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_source_condense_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_source_condense_va1e1b1nwzida0e0b0xyg3,
                                                                   a_p_va1e1b1nwzida0e0b0xyg3,
                                                                   a_v_pa1e1b1nwzida0e0b0xyg3)
    ffcfw_source_season_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                             period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_source_season_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_source_season_va1e1b1nwzida0e0b0xyg3,
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
    ffcfw_dest_season_va1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_ffcfw_season_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1,
                                                           period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg)  #numbers not required for ffcfw
    ffcfw_dest_season_va1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v_adj(ffcfw_dest_season_va1e1b1nwzida0e0b0xyg1,
                                                               a_p_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1)
    ###offs
    ffcfw_dest_condense_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_condensed_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                             period_is_tp=nextperiod_is_condense_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_dest_condense_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_dest_condense_va1e1b1nwzida0e0b0xyg3,
                                                                 a_p_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)
    ffcfw_dest_season_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_ffcfw_season_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                           period_is_tp=nextperiod_is_startseason_pa1e1b1nwzida0e0b0xyg3)  #numbers not required for ffcfw
    ffcfw_dest_season_va1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v_adj(ffcfw_dest_season_va1e1b1nwzida0e0b0xyg3,
                                                               a_p_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3)

    ##distributing at condensing - all lws back to starting number of LWs and dams to different sires at prejoining
    ###t0 and t1 are distributed however this is not used because t0 and t1 don't transfer to next dvp
    distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9 = sfun.f1_lw_distribution(
        ffcfw_dest_condense_tva1e1b1nwzida0e0b0xyg1, ffcfw_source_condense_tva1e1b1nwzida0e0b0xyg1,
        mask_w9vars_va1e1b1nw9zida0e0b0xyg1,
        index_wzida0e0b0xyg1, dvp_type_next_tva1e1b1nwzida0e0b0xyg1[..., na], condense_vtype1)
    distribution_condense_va1e1b1nw8zida0e0b0xyg3w9 = sfun.f1_lw_distribution(
        ffcfw_dest_condense_va1e1b1nwzida0e0b0xyg3, ffcfw_source_condense_va1e1b1nwzida0e0b0xyg3,
        mask_w9vars_va1e1b1nw9zida0e0b0xyg3,
        index_wzida0e0b0xyg3, dvp_type_next_va1e1b1nwzida0e0b0xyg3[..., na], condense_vtype3)

    ##redistribute at season start - all seasons back into a common season.
    distribution_season_va1e1b1nw8zida0e0b0xyg1w9 = sfun.f1_lw_distribution(
        ffcfw_dest_season_va1e1b1nwzida0e0b0xyg1, ffcfw_source_season_va1e1b1nwzida0e0b0xyg1,
        mask_w9vars_va1e1b1nw9zida0e0b0xyg1,
        index_wzida0e0b0xyg1, dvp_type_next_va1e1b1nwzida0e0b0xyg1[..., na], season_vtype1)
    distribution_season_va1e1b1nw8zida0e0b0xyg3w9 = sfun.f1_lw_distribution(
        ffcfw_dest_season_va1e1b1nwzida0e0b0xyg3, ffcfw_source_season_va1e1b1nwzida0e0b0xyg3,
        mask_w9vars_va1e1b1nw9zida0e0b0xyg3,
        index_wzida0e0b0xyg3, dvp_type_next_va1e1b1nwzida0e0b0xyg3[..., na], season_vtype3)

    ##combine distributions
    distribution_tva1e1b1nw8zida0e0b0xyg1w9 = distribution_condense_tva1e1b1nw8zida0e0b0xyg1w9 * distribution_season_va1e1b1nw8zida0e0b0xyg1w9
    distribution_va1e1b1nw8zida0e0b0xyg3w9 = distribution_condense_va1e1b1nw8zida0e0b0xyg3w9 * distribution_season_va1e1b1nw8zida0e0b0xyg3w9

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
    assetvalue_p7tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', assetvalue_p7tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0,
                                                                          mask_vg=mask_z8var_p7tva1e1b1nwzida0e0b0xyg)
    assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', assetvalue_p7tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1, index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                 numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                 mask_vg=(mask_w8vars_va1e1b1nw8zida0e0b0xyg1*mask_z8var_va1e1b1nwzida0e0b0xyg1*mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,...]))
    assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', assetvalue_p7tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
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
    #^can't just sum across the a slice (decision variable) so allow a0 to provide a1 we will need another a axis (see google doc) - fix this in version 2
    # temporary = np.sum(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=a1_pos-1, keepdims=True) * (index_a1e1b1nwzida0e0b0xyg[...,na] == 0)
    # numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary, dvp_type_next_tva1e1b1nwzida0e0b0xyg1[:,:,:,0:1,...,na] == 0) #take slice 0 of e (for prejoining all e slices are the same

    ###offs
    numbers_prov_offs_k3k5tva1e1b1nw8zida0e0b0xygw9 = (fun.f_divide(np.sum(numbers_end_tva1e1b1nwzida0e0b0xyg3[...,na]  * distribution_va1e1b1nw8zida0e0b0xyg3w9
                                                                           * mask_numbers_provw8w9_va1e1b1nw8zida0e0b0xyg3w9
                                                                           * (a_k3cluster_da0e0b0xyg3==index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                                           * (a_k5cluster_da0e0b0xyg3==index_k5tva1e1b1nwzida0e0b0xyg3)[...,na]
                                                                           , axis = (d_pos-1, b0_pos-1, e0_pos-1), keepdims=True)
                                                                , np.sum(numbers_start_tva1e1b1nwzida0e0b0xyg3 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                                                                         * (a_k5cluster_da0e0b0xyg3==index_k5tva1e1b1nwzida0e0b0xyg3)
                                                                         , axis = (d_pos, b0_pos, e0_pos), keepdims=True)[...,na], dtype=dtype)
                                                       * mask_numbers_provt_tva1e1b1nw8zida0e0b0xyg3w9)

    ##numbers required
    ###dams
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 =  1 * (np.sum(mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9[...,na,:] * mask_numbers_reqt_k2tva1e1b1nwzida0e0b0xyg1g9[:,na,...,na]
                                                                       * mask_z8var_va1e1b1nwzida0e0b0xyg1[...,na,na]
                                                                       * ((a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k28k29tva1e1b1nwzida0e0b0xyg1) * (a_k2cluster_va1e1b1nwzida0e0b0xyg1 == index_k2tva1e1b1nwzida0e0b0xyg1))[...,na,na]
                                                                       , axis = (b1_pos-2, e1_pos-2), keepdims=True)>0)
    ####combine nm and 00 cluster for prejoining to scanning
    temporary = np.sum(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, axis=1, keepdims=True) * (index_k29tva1e1b1nwzida0e0b0xyg1g9[...,na] == 0)  # put the sum of the k29 in slice 0
    numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9 = fun.f_update(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, temporary,
                                                                        dvp_type_va1e1b1nwzida0e0b0xyg1[:, :, 0:1, ..., na,na] == prejoin_vtype1)  #take slice 0 of e (for prejoining all e slices are the same)
    ####combine wean numbers at prejoining to allow the matrix to select a different weaning time for the coming yr.
    #^can't just sum across the a slice (decision variable) so allow a0 to provide a1 we will need another a axis (see google doc)
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


    #################
    #progeny weaned #
    #################
    '''yatf are first transferred to progeny activity then they are either sold as sucker, transferred to dam or transferred to offs.
    yatf have a t axis due to the dams feedsupply. This t axis is removed and replaced with the prog t axis.'''
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
    step_con_prog2dams = n_fs_dams ** n_fvp_periods_dams
    mask_numbers_prog2damsw8w9_w9 = (index_w1 % step_con_prog2dams) == 0
    ##mask w8 (prog) to w9 (offs)
    step_con_prog2offs = n_fs_offs ** n_fvp_periods_offs
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
                 * (a_k2cluster_va1e1b1nwzida0e0b0xyg1==index_k5tva1e1b1nwzida0e0b0xyg3 + 2)[...,na,na] #convert e1 and b1 to k5 cluster - using a k5 cluster because progeny don't need all the k2 slices and the relevant ones align between k2 and k5 eg 11, 22 etc
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
    numbers_progreq_k2k3k5tva1e1b1nw8zida0e0b0xyg1g9w9 = 1 * np.sum(np.any(mask_numbers_reqw8w9_va1e1b1nw8zida0e0b0xyg1w9 * mask_z8var_va1e1b1nwzida0e0b0xyg1[...,na], axis=e1_pos-1, keepdims=True)[0, ...,na,:]
                                                                    * mask_tvars_k2tva1e1b1nw8zida0e0b0xyg1[:,na,na,:,0:1,...,na,na]  # mask based on the t axis for dvp0
                                                                    * (index_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,..., na,na] == 0) #only NM slice requires prog
                                                                    * (index_g1[...,na]==index_g1)[...,na]
                                                                    * btrt_propn_b0xyg1[...,na,na].astype(dtype)
                                                                    * e0_propn_ida0e0b0xyg[...,na,na].astype(dtype)
                                                                    * agedam_propn_da0e0b0xyg1[...,na,na].astype(dtype)
                                                                    * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)[...,na,na]
                                                                    * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)[...,na,na],
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
    r_salegrid_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_salegrid_tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3,
                                                     on_hand_tpa1e1b1nwzida0e0b0xyg3)
    r_woolvalue_p7tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(r_woolvalue_p7tpa1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                              on_hand_tpa1e1b1nwzida0e0b0xyg3)

    ##sale time - no numbers needed because they don't effect sale date
    r_saledate_tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(date_start_pa1e1b1nwzida0e0b0xyg3.astype('timedelta64[s]'), a_v_pa1e1b1nwzida0e0b0xyg3,
                                                    period_is_tp=period_is_sale_tpa1e1b1nwzida0e0b0xyg3) #have to convert to timedelta so can multiply (converted back to date after the report)

    ##cfw per head average for the mob - includes the mortality factor
    r_cfw_hdmob_tvg0 = sfun.f1_p2v_std(o_cfw_tpsire, numbers_p=o_numbers_end_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_shearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_cfw_hdmob_tvg1 = sfun.f1_p2v(o_cfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_shearing_pa1e1b1nwzida0e0b0xyg1)
    r_cfw_hdmob_tvg3 = sfun.f1_p2v(o_cfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)
    ##cfw per head - wool cut for 1 whole animal, no account for mortality
    r_cfw_hd_tvg0 = sfun.f1_p2v_std(o_cfw_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_shearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_cfw_hd_tvg1 = sfun.f1_p2v(o_cfw_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                               period_is_tp=period_is_shearing_pa1e1b1nwzida0e0b0xyg1)
    r_cfw_hd_tvg3 = sfun.f1_p2v(o_cfw_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg3,
                               period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)

    ##fd per head average for the mob - includes the mortality factor
    r_fd_hdmob_tvg0 = sfun.f1_p2v_std(o_fd_tpsire, numbers_p=o_numbers_end_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_shearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_fd_hdmob_tvg1 = sfun.f1_p2v(o_fd_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_shearing_pa1e1b1nwzida0e0b0xyg1)
    r_fd_hdmob_tvg3 = sfun.f1_p2v(o_fd_tpoffs, a_v_pa1e1b1nwzida0e0b0xyg3, o_numbers_end_tpoffs,
                                             on_hand_tpa1e1b1nwzida0e0b0xyg3, period_is_tp=period_is_shearing_tpa1e1b1nwzida0e0b0xyg3)
    ##fd per head - wool cut for 1 whole animal, no account for mortality
    r_fd_hd_tvg0 = sfun.f1_p2v_std(o_fd_tpsire, on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0,
                                                  period_is_tvp=period_is_shearing_pa1e1b1nwzida0e0b0xyg0)[:,na,...]#add singleton v
    r_fd_hd_tvg1 = sfun.f1_p2v(o_fd_tpdams, a_v_pa1e1b1nwzida0e0b0xyg1, on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1,
                               period_is_tp=period_is_shearing_pa1e1b1nwzida0e0b0xyg1)
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

    ##numbers mated per animal at the start of the dvp.
    r_n_mated_tvg1 = sfun.f1_p2v(animal_mated_b1g1*1, a_v_pa1e1b1nwzida0e0b0xyg1, o_numbers_end_tpdams,
                                on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, period_is_tp=period_is_mating_pa1e1b1nwzida0e0b0xyg1)
    ###update dvps that are not mating with mating numbers
    a_matingv_tvg1 =  np.maximum.accumulate(np.any(r_n_mated_tvg1 != 0, axis=b1_pos, keepdims=True) * index_va1e1b1nwzida0e0b0xyg1, axis=p_pos) #create association pointing at previous/current mating dvp.
    r_n_mated_tvg1= np.take_along_axis(r_n_mated_tvg1, a_matingv_tvg1, axis=p_pos)

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

    ##sale date - no numbers needed because they don't effect sale date. need to mask the denominator so that only the d, e, b slices where the animal was sold is used in the divide.
    ###cant use production function because need to mask the denominator.
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
    ##proportion mated - per ewe at start of dvp (ie accounting for dam mortality)
    r_n_mated_k2tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_n_mated_tvg1,
                                                                          a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                                          index_k2tva1e1b1nwzida0e0b0xyg1,
                                                                       numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                                       mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)  # no clustering required for scanning percent because it is a measure of all dams

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


    ##############
    #big reports #
    ##############
    ##mortality - at the start of each dvp mortality is 0. it accumulates over the dvp. This is its own report as well as used in on_hand_mort.
    if pinp.rep['i_store_on_hand_mort'] or pinp.rep['i_store_mort']:
        ###get the cumulative mort for periods in each dvp
        r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg1 = sfun.f1_cum_sum_dvp(o_mortality_dams, a_v_pa1e1b1nwzida0e0b0xyg1)
        r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg3 = sfun.f1_cum_sum_dvp(o_mortality_offs, a_v_pa1e1b1nwzida0e0b0xyg3)
        ###mask w & z slices
        mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg1 = np.take_along_axis(mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1, a_v_pa1e1b1nwzida0e0b0xyg1
                                                                 , axis=0)
        mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg3 = np.take_along_axis(mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3, a_v_pa1e1b1nwzida0e0b0xyg3
                                                                 , axis=0)
        r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg1 = r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg1 * mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg1
        r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg3 = r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg3 * mask_w8z8vars_pa1e1b1nw8zida0e0b0xyg3

    ##on hand mort- this is used for numbers_p report so that the report can have a p axis to increase numbers detail.
    ##              accounts for mortality as well as on hand.
    if pinp.rep['i_store_on_hand_mort']:
        ###add v axis and adjust for onhand
        r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg1 = r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg1 * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...] * (
                                                  a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
        r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg3 = r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg3 * on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...] * (
                                                  a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
        ###cluster e,b
        r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams',r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg1,
                                                                              a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...],
                                                                              index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],
                                                                              numbers_start_vg = on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...])  #on_hand to handle the periods when e slices are in different dvps (eg cant just have default 1 otherwise it will divide by 2 because both e gets summed)
        r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs',r_cum_dvp_mort_tvpa1e1b1nwzida0e0b0xyg3,
                                                                                     a_k3cluster_da0e0b0xyg3,
                                                                                     index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],
                                                                                     a_k5cluster_da0e0b0xyg3,
                                                                                     index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...],
                                                                                     numbers_start_vg=on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...]) #on_hand to handle the periods when e slices are in different dvps (eg cant just have default 1 otherwise it will divide by 2 because both e gets summed)

        ###convert to on hand mort (1-mort)
        r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1 = 1 - r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1
        r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1[r_cum_dvp_mort_k2tvpa1e1b1nwzida0e0b0xyg1==0] = 0 #if mort is 0 the animal is not on hand
        r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3 = 1 - r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3
        r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3[r_cum_dvp_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3==0] = 0 #if mort is 0 the animal is not on hand

    ###lw - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. Don't need it because it doesnt effect lw
    if pinp.rep['i_store_lw_rep']:
        r_lw_sire_tpsire = o_lw_tpsire
        r_lw_dams_k2tvpdams = (o_lw_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_lw_offs_k3k5tvpoffs = (o_lw_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...]))

    ##ffcfw - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. Don't need it because it doesnt effect ffcfw
    if pinp.rep['i_store_ffcfw_rep']:
        r_ffcfw_sire_tpsire = o_ffcfw_tpsire
        r_ffcfw_dams_k2tvpdams = (o_ffcfw_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_ffcfw_yatf_k2tvpyatf = (r_ffcfw_start_tpyatf * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                               * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_ffcfw_offs_k3k5tvpoffs = (o_ffcfw_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...]))
        r_ffcfw_prog_k3k5tva1e1b1nwzida0e0b0xyg2 = ffcfw_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2 \
                                                 * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3) \
                                                 * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3) #todo cluster d

    ##NV - need to add v and k2 axis but still keep p, e and b so that we can graph the desired patterns. This is a big array so only stored if user wants. t is not required because it doesnt effect NV
    if pinp.rep['i_store_nv_rep']:
        r_nv_sire_pg = nv_tpsire
        r_nv_dams_k2tvpg = (nv_tpdams[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...]))
        r_nv_offs_k3k5tvpg = (nv_tpoffs[:,na,...] * (a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                               * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                               * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...]))

    ###############
    ## report dse #
    ###############
    days_p6z = per.f_feed_periods(option=1)
    days_p6_p6tva1e1b1nwzida0e0b0xyg = fun.f_expand(days_p6z, z_pos,  left_pos2=p_pos-2, right_pos2=z_pos)
    ###DSE based on MJ/d
    ####returns the average mj/d for each animal for the each feed period (mei accounts for if the animal is on hand - if the animal is sold the average mei/d will be lower in that dvp)
    mj_ave_p6ftva1e1b1nwzida0e0b0xyg0 = fun.f_divide(mei_p6ftva1e1b1nwzida0e0b0xyg0, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    mj_ave_k2p6ftva1e1b1nwzida0e0b0xyg1 = fun.f_divide(mei_k2p6ftva1e1b1nwzida0e0b0xyg1, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    mj_ave_k3k5p6ftva1e1b1nwzida0e0b0xyg3 = fun.f_divide(mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3, days_p6_p6tva1e1b1nwzida0e0b0xyg[:,na,...])
    ####returns the number of dse of each animal in each dvp - this is combined with the variable numbers in reporting to get the total dse
    # note: sires have a single long DVP and are on-hand for multiple years. Therefore, mj_ave is higher (because the decision variable is representing multiple sires)
    dsemj_p6tva1e1b1nwzida0e0b0xyg0 = np.sum(mj_ave_p6ftva1e1b1nwzida0e0b0xyg0 / pinp.sheep['i_dse_mj'], axis = 1)
    dsemj_k2p6tva1e1b1nwzida0e0b0xyg1 = np.sum(mj_ave_k2p6ftva1e1b1nwzida0e0b0xyg1 / pinp.sheep['i_dse_mj'], axis = 2)
    dsemj_k3k5p6tva1e1b1nwzida0e0b0xyg3 = np.sum(mj_ave_k3k5p6ftva1e1b1nwzida0e0b0xyg3 / pinp.sheep['i_dse_mj'], axis = 3)

    ###DSE based on nw
    #### cumulative total of nw with p6 axis
    nw_cum_p6tva1e1b1nwzida0e0b0xyg0 = sfun.f1_p2v_std(o_nw_start_tpsire ** 0.75, numbers_p=o_numbers_end_tpsire,
                                        on_hand_tvp=on_hand_pa1e1b1nwzida0e0b0xyg0, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg0,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tvp=index_p6tpa1e1b1nwzida0e0b0xyg)[:,:,na,...]#add singleton v
    nw_cum_p6tva1e1b1nwzida0e0b0xyg1 = sfun.f1_p2v(o_nw_start_tpdams ** 0.75, a_v_pa1e1b1nwzida0e0b0xyg1, numbers_p=o_numbers_end_tpdams,
                                        on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg1, days_period_p=days_period_pa1e1b1nwzida0e0b0xyg1,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg, index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg)
    nw_cum_p6tva1e1b1nwzida0e0b0xyg3 = sfun.f1_p2v(o_nw_start_tpoffs**0.75, a_v_pa1e1b1nwzida0e0b0xyg3, numbers_p=o_numbers_end_tpoffs,
                                        on_hand_tp=on_hand_tpa1e1b1nwzida0e0b0xyg3, days_period_p=days_period_cut_pa1e1b1nwzida0e0b0xyg3,
                                        a_any1_p=a_p6_pa1e1b1nwzida0e0b0xyg[mask_p_offs_p], index_any1tp=index_p6tpa1e1b1nwzida0e0b0xyg)
    ####returns the average nw for each animal for the each feed period (cum nw accounts for if the animal is on hand - if the animal is sold the average nw will be lower in that feed period)
    # note: sires have a single long DVP and are on-hand for multiple years of the same p6. Therefore, nw_ave is higher (because the decision variable is representing multiple sires)
    nw_ave_p6tva1e1b1nwzida0e0b0xyg0 = fun.f_divide(nw_cum_p6tva1e1b1nwzida0e0b0xyg0, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    nw_ave_p6tva1e1b1nwzida0e0b0xyg1 = fun.f_divide(nw_cum_p6tva1e1b1nwzida0e0b0xyg1, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    nw_ave_p6tva1e1b1nwzida0e0b0xyg3 = fun.f_divide(nw_cum_p6tva1e1b1nwzida0e0b0xyg3, days_p6_p6tva1e1b1nwzida0e0b0xyg)
    ####convert nw to dse
    dsehd_p6tva1e1b1nwzida0e0b0xyg0 = nw_ave_p6tva1e1b1nwzida0e0b0xyg0 / pinp.sheep['i_dse_nw']**0.75
    dsehd_p6tva1e1b1nwzida0e0b0xyg1 = nw_ave_p6tva1e1b1nwzida0e0b0xyg1 / pinp.sheep['i_dse_nw']**0.75
    dsehd_p6tva1e1b1nwzida0e0b0xyg3 = nw_ave_p6tva1e1b1nwzida0e0b0xyg3 / pinp.sheep['i_dse_nw']**0.75
    ####account for b1 axis effect on dse & select the dse group (note sire and offs don't have b1 axis so simple slice)
    dse_group_dp6tva1e1b1nwzida0e0b0xyg = fun.f_expand(pinp.sheep['i_dse_group'], z_pos, left_pos2=p_pos - 2, right_pos2=z_pos)
    dse_group_dp6tva1e1b1nwzida0e0b0xyg = zfun.f_seasonal_inp(dse_group_dp6tva1e1b1nwzida0e0b0xyg, numpy=True, axis=z_pos)
    a_dams_dsegroup_b1nwzida0e0b0xyg = fun.f_expand(sinp.stock['ia_dams_dsegroup_b1'], b1_pos)
    dsenw_p6tva1e1b1nwzida0e0b0xyg0 = dsehd_p6tva1e1b1nwzida0e0b0xyg0 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_sire_dsegroup']]
    dsenw_p6tva1e1b1nwzida0e0b0xyg1 = dsehd_p6tva1e1b1nwzida0e0b0xyg1 * np.take_along_axis(dse_group_dp6tva1e1b1nwzida0e0b0xyg, a_dams_dsegroup_b1nwzida0e0b0xyg[na,na,na,na,na,na],0)[0,...] #take along the dse group axis and remove the d axis from the front
    dsenw_p6tva1e1b1nwzida0e0b0xyg3 = dsehd_p6tva1e1b1nwzida0e0b0xyg3 * dse_group_dp6tva1e1b1nwzida0e0b0xyg[sinp.stock['ia_offs_dsegroup']]
    ##cluster and account for numbers/mortality
    dsenw_p6tva1e1b1nwzida0e0b0xyg0 = sfun.f1_create_production_param('sire', dsenw_p6tva1e1b1nwzida0e0b0xyg0, numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg0)
    dsenw_k2p6tva1e1b1nwzida0e0b0xyg1 = sfun.f1_create_production_param('dams', dsenw_p6tva1e1b1nwzida0e0b0xyg1, a_k2cluster_va1e1b1nwzida0e0b0xyg1,
                                                index_k2tva1e1b1nwzida0e0b0xyg1[:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg1,
                                                mask_vg = mask_w8vars_va1e1b1nw8zida0e0b0xyg1 * mask_z8var_va1e1b1nwzida0e0b0xyg1)
    dsenw_k3k5p6tva1e1b1nwzida0e0b0xyg3 = sfun.f1_create_production_param('offs', dsenw_p6tva1e1b1nwzida0e0b0xyg3, a_k3cluster_da0e0b0xyg3, index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],
                                                    a_k5cluster_da0e0b0xyg3, index_k5tva1e1b1nwzida0e0b0xyg3[:,na,...], numbers_start_vg=numbers_start_tva1e1b1nwzida0e0b0xyg3,
                                                    mask_vg=mask_w8vars_va1e1b1nw8zida0e0b0xyg3 * mask_z8var_va1e1b1nwzida0e0b0xyg3)

    ####################
    #report stock days #
    ####################
    ##this needs to be accounted for when reporting variables that have p6 and v axis because they are both periods that do not align
    ##and the number variable returned from pyomo does not have p6 axis. So need to account for the propn of the dvp that the feed period exists.
    ##using a_p6_p is not perfect because a_p6_p is such that a generator period is only allocated to a single feed period
    ## eg if the feed period changed mid gen period the proportion will be slightly off (exaggerated for smaller feed periods).
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
    keys_g0 = sfun.f1_g2g(pinp.sheep['i_g_idx_sire'],'sire')
    keys_g1 = sfun.f1_g2g(pinp.sheep['i_g_idx_dams'],'dams')
    keys_g2 = keys_g1
    keys_g3 = sfun.f1_g2g(pinp.sheep['i_g_idx_offs'],'offs')
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
    keys_T1 = np.array(['t%s'%i for i in range(len_gen_t1)])
    keys_t2 = np.array(['t%s'%i for i in range(len_t2)])
    keys_t3 = np.array(['t%s'%i for i in range(len_t3)])
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
    ###numbers req dams
    arrays_k2k2tva1nw8ziyg1g9w9 = [keys_k2,keys_k2,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_y1,
                                   keys_g1,keys_g1,keys_lw1]
    ###numbers dams prov
    arrays_k2k2tvanwziyg1g9w9 = [keys_k2,keys_k2,keys_t1,keys_v1,keys_a,keys_n1,keys_lw1,keys_z,keys_i,keys_y1,keys_g1,
                                keys_g1,keys_lw1]

    ##offs related
    ###numbers prov offs
    arrays_k3k5tvnw8ziaxyg3w9 = [keys_k3, keys_k5, keys_t3, keys_v3, keys_n3, keys_lw3, keys_z, keys_i, keys_a, keys_x, keys_y3, keys_g3, keys_lw3]
    ###k3k5wixg3w9 - numbers req offs (doesnt have many active axis)
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

    ##asset value infra - all in the last season period (doesnt really matter where since it is transferred between each season period)
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
    params['p_npw_dams'] = fun.f1_make_pyomo_dict(npw_k3k5tva1e1b1nwzida0e0b0xyg1w9i9, arrays_k3k5tva1nw8zixyg1w9i9, loop_axis_pos=w_pos-2, index_loop_axis_pos=-8)
    ###number prog require by dams
    params['p_progreq_dams'] = fun.f1_make_pyomo_dict(numbers_progreq_k2k3k5tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k3k5tw8ziyg1g9w9, loop_axis_pos=-1, index_loop_axis_pos=-1)
    ###number prog require by offs
    params['p_progreq_offs'] = fun.f1_make_pyomo_dict(numbers_progreq_k3k5tva1e1b1nw8zida0e0b0xyg3w9, arrays_k3vw8zixg3w9)
    ###number prog provided to dams
    params['p_progprov_dams'] = fun.f1_make_pyomo_dict(numbers_prog2dams_k3k5tva1e1b1nwzida0e0b0xyg2g9w9, arrays_k3k5tw8zia0xyg2g9w9)
    ###number prog provided to offs
    params['p_progprov_offs'] = fun.f1_make_pyomo_dict(numbers_prog2offs_k3k5tva1e1b1nwzida0e0b0xyg2w9, arrays_k3k5tw8ziaxyg2w9)

    ##dams
    ###numbers_req_dams
    params['numbers_req_numpyversion_k2k2tva1nw8ziyg1g9w9'] = numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9[:,:,:,:,:,0,0,:,:,:,:,0,0,0,0,0,:,:,:,:]  #can't use squeeze here because i need to keep all relevant axis even if singleton. this is used to speed pyomo constraint.
    params['p_numbers_req_dams'] = fun.f1_make_pyomo_dict(numbers_req_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tva1nw8ziyg1g9w9, loop_axis_pos=-1, index_loop_axis_pos=-1)
    ###numbers_prov_dams
    ####numbers provided into next period (the norm)
    params['p_numbers_prov_dams'] = fun.f1_make_pyomo_dict(numbers_prov_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tvanwziyg1g9w9, loop_axis_pos=-1, index_loop_axis_pos=-1)
    #### provided into this period (when transferring from an earlier lambing ram group to a later lambing)
    params['p_numbers_provthis_dams'] = fun.f1_make_pyomo_dict(numbers_provthis_dams_k28k29tva1e1b1nw8zida0e0b0xyg1g9w9, arrays_k2k2tvanwziyg1g9w9, loop_axis_pos=-1, index_loop_axis_pos=-1)

    ##offs related
    ###numbers_req_offs
    params['numbers_req_numpyversion_k3k5vw8zixg3w9'] = numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9[:,:,0,:,0,0,0,0,:,:,:,0,0,0,0,:,0,:,:]  #can't use squeeze here because i need to keep all relevant axis even if singleton. this is used to speed pyomo constraint.
    params['p_numbers_req_offs'] = fun.f1_make_pyomo_dict(numbers_req_offs_k3k5tva1e1b1nw8zida0e0b0xygw9, arrays_k3k5vw8zixg3w9)
    ###numbers_prov_offs
    params['p_numbers_prov_offs'] = fun.f1_make_pyomo_dict(numbers_prov_offs_k3k5tva1e1b1nw8zida0e0b0xygw9, arrays_k3k5tvnw8ziaxyg3w9)

    ##mei
    ###mei - sire
    params['p_mei_sire'] = fun.f1_make_pyomo_dict(mei_p6ftva1e1b1nwzida0e0b0xyg0, arrays_p6fzg0)
    ###mei - dams
    params['p_mei_dams'] = fun.f1_make_pyomo_dict(mei_k2p6ftva1e1b1nwzida0e0b0xyg1, arrays_k2p6ftva1nwziyg1)
    ###mei - offs
    params['p_mei_offs'] = fun.f1_make_pyomo_dict(mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3, arrays_k3k5p6ftvnwziaxyg3)

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

    ##asset value
    ###assetvalue - sire
    params['p_assetvalue_sire'] = fun.f1_make_pyomo_dict(assetvalue_p7tva1e1b1nwzida0e0b0xyg0, arrays_p7zg0)
    ###assetvalue - dams
    params['p_assetvalue_dams'] = fun.f1_make_pyomo_dict(assetvalue_k2p7tva1e1b1nwzida0e0b0xyg1, arrays_k2p7tvanwziyg1)
    ###assetvalue - offs
    params['p_assetvalue_offs'] = fun.f1_make_pyomo_dict(assetvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3, arrays_k3k5p7tvnwziaxyg3)

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

    ################
    # Bound params #
    ################
    '''store params used in BoundsPyomo.py'''
    ###shapes
    len_v1 = len(keys_v1)
    len_v3 = len(keys_v3)

    ##mask for dam activities
    arrays = [keys_k2, keys_t1, keys_v1, keys_lw1, keys_g1]
    index_ktvwg1 = fun.cartesian_product_simple_transpose(arrays)
    tup_ktvwg1 = tuple(map(tuple,index_ktvwg1))
    mask_dams_ktvwg1 = mask_dams_k2tva1e1b1nw8zida0e0b0xyg1.ravel()
    params['p_mask_dams'] = dict(zip(tup_ktvwg1, mask_dams_ktvwg1))

    ##proportion of dams mated. inf means the model can optimise the proportion because inf is used to skip the constraint.
    arrays = [keys_v1, keys_g1]
    index_vg1 = fun.cartesian_product_simple_transpose(arrays)
    tup_vg1 = tuple(map(tuple,index_vg1))
    prop_dams_mated_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesnt impact mating propn
    prop_dams_mated_vg1 = prop_dams_mated_va1e1b1nwzida0e0b0xyg1.ravel()
    params['p_prop_dams_mated'] = dict(zip(tup_vg1, prop_dams_mated_vg1))

    ##proportion of dry dams as a propn of preg dams at shearing sale. This is different to the propn in the dry report because it is the propn at a given time rather than per animal at the beginning of mating.
    # This is used to force retention of drys at the main (t[0]) sale time. You can only sell drys if you sell non-drys. This param indicates the propn of dry that can be sold per non-dry dam.
    propn_drys_tpg1 = fun.f_divide(np.sum(o_numbers_end_tpdams*n_drys_b1g1, axis=(e1_pos,b1_pos), keepdims=True),
                              np.sum(o_numbers_end_tpdams * (nyatf_b1nwzida0e0b0xyg>0),axis=(e1_pos,b1_pos), keepdims=True))
    propn_drys_vg1 = sfun.f1_p2v(propn_drys_tpg1, a_v_pa1e1b1nwzida0e0b0xyg1,
                                period_is_tp=period_is_sale_t0_pa1e1b1nwzida0e0b0xyg1[:,:,0:1,...]) #only interested in the shearing sale, take e[0] it is the same as e[1] so don't need it.
    # propn_drys_vg1 = np.max(propn_drys_vg1, axis=) #get the max propn of drys along select axes to reduce size. Needs to be max so that all drys can be s
    arrays = [keys_v1, keys_a, keys_n1, keys_lw1, keys_i, keys_y1, keys_g1]
    index_vanwiyg1 = fun.cartesian_product_simple_transpose(arrays)
    tup_vanwiyg1 = tuple(map(tuple,index_vanwiyg1))
    propn_drys_vanwiyg1 = propn_drys_vg1.ravel()
    params['p_prop_dry_dams'] = dict(zip(tup_vanwiyg1, propn_drys_vanwiyg1))

    ##proportion of drys that are twice dry
    ###expand for p axis
    prop_twice_dry_dams_oa1e1b1nwzia0e0b0xyg1 = np.moveaxis(ce_dams[2,...], source=d_pos, destination=0) #move d axis to p pos
    prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1 = np.expand_dims(prop_twice_dry_dams_oa1e1b1nwzia0e0b0xyg1, d_pos) #add singleton d axis
    prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1[0] = 0 #cant have any twice drys in the first mating opportunity. (this line is just here to stop error if user accidentally puts in a value for o[0]).
    prop_twice_dry_dams_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_twice_dry_dams_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0) #increments at prejoining
    ###convert to v axis
    prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_twice_dry_dams_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesnt impact mating propn
    ###adjust maidens twice drys for yearling mating (if no yearlings are mated then there can not be any twice dry maidens)
    ####calc propn of dams mated in previous opportunity.
    prop_dams_mated_prev_oa1e1b1nwzida0e0b0xyg1 = np.roll(prop_dams_mated_oa1e1b1nwzida0e0b0xyg1, shift=1, axis=0)
    prop_dams_mated_prev_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_prev_oa1e1b1nwzida0e0b0xyg1, a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1, 0) #increments at prejoining
    prop_dams_mated_prev_va1e1b1nwzida0e0b0xyg1 = np.take_along_axis(prop_dams_mated_prev_pa1e1b1nwzida0e0b0xyg1, a_p_va1e1b1nwzida0e0b0xyg1[:,:,0:1,...], axis=0) #take e[0] because e doesnt impact mating propn
    prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 = prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1 * (prop_dams_mated_prev_va1e1b1nwzida0e0b0xyg1>1)
    ###create param
    arrays = [keys_v1, keys_i, keys_y1, keys_g1]
    index_viyg1 = fun.cartesian_product_simple_transpose(arrays)
    tup_viyg1 = tuple(map(tuple,index_viyg1))
    prop_twice_dry_dams_viyg1 = prop_twice_dry_dams_va1e1b1nwzida0e0b0xyg1.ravel()
    params['p_prop_twice_dry_dams'] = dict(zip(tup_viyg1, prop_twice_dry_dams_viyg1))
    params['p_prejoin_v_dams'] = keys_v1[dvp_type_va1e1b1nwzida0e0b0xyg1[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0]==prejoin_vtype1] #get the dvp keys which are prejoining (same for all animals hence take slice 0)
    params['p_scan_v_dams'] = keys_v1[dvp_type_va1e1b1nwzida0e0b0xyg1[:,0,0,0,0,0,0,0,0,0,0,0,0,0,0]==scan_vtype1] #get the dvp keys which are scan (same for all animals hence take slice 0)







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

    ##key lists used to form table headers and indexs
    keys_e = ['e%s'%i for i in range(len_e1)]
    keys_b = sinp.stock['i_lsln_idx_dams']
    keys_e0 = ['e%s'%i for i in range(len_e0)]
    keys_b0 = sinp.stock['i_btrt_idx_offs']
    keys_b9 = sinp.stock['i_lsln_idx_dams'][1:5]
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
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_p6, keys_f, keys_t1, keys_v1, keys_a, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_qsk2p6ftvanwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_p, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_y1, keys_g1],'dams_keys_paebnwziy1g1')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k2, keys_t1, keys_v1, keys_p, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_x, keys_y1, keys_g2],'yatf_keys_qsk2tvpaebnwzixy1g1')
    fun.f1_make_r_val(r_vals,[keys_T1, keys_v1, keys_a, keys_e, keys_b, keys_n1, keys_lw1
                                            , keys_z, keys_i, keys_x, keys_y1, keys_g2],'yatf_keys_Tvaebnwzixy1g2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i
                                            , keys_a, keys_x, keys_g2],'prog_keys_qsk3k5twzia0xg2')
    fun.f1_make_r_val(r_vals,[keys_q, keys_s, keys_k3, keys_k5, keys_t2, keys_lw_prog, keys_z, keys_i, keys_d
                                            , keys_a, keys_e0, keys_b0, keys_x, keys_y3, keys_g2],'prog_keys_qsk3k5twzida0e0b0xyg2')
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
    fun.f1_make_r_val(r_vals,[keys_p3, keys_n3, keys_lw3, keys_z, keys_i, keys_d, keys_a, keys_e0, keys_b0
                                            , keys_x, keys_y3, keys_g3],'offs_keys_pnwzidaebxyg3')

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

    ####kvpeb
    pzg0_shape = len_p, len_z, len_g0
    k2vpa1e1b1nwziyg1_shape = len_k2, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k2vpa1e1b1nwzixyg1_shape = len_k2, len_v1, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, len_x, len_y1, len_g1
    k3k5wzida0e0b0xyg2_shape = len_k3, len_k5, len_w_prog, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y2,  len_g2
    k3k5vpnwzidae0b0xyg3_shape = len_k3, len_k5, len_v3, lenoffs_p, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y3, len_g3

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

    ####cg
    p7zg0_shape = len_p7, len_z, len_g0
    k2p7tva1nwziyg1_shape = len_k2, len_p7, len_t1, len_v1, len_a1, len_n1, len_w1, len_z, len_i, len_y1, len_g1
    k3k5p7twziaxyg2_shape = len_k3, len_k5, len_p7, len_t2, len_w_prog, len_z, len_i, len_a1, len_x, len_g2
    k3k5p7tvnwziaxyg3_shape = len_k3, len_k5, len_p7, len_t3, len_v3, len_n3, len_w3, len_z, len_i, len_a0, len_x, len_y3, len_g3

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
    salevalue_prog_k3k5c1p7tva1e1b1nwzida0e0b0xyg2
    fun.f1_make_r_val(r_vals,r_salevalue_prog_k3k5p7tva1e1b1nwzida0e0b0xyg2,'salevalue_k3k5p7twzia0xg2',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, k3k5p7twziaxyg2_shape)
    fun.f1_make_r_val(r_vals,r_salevalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'salevalue_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg0,'salegrid_zg0', shape=zg0_shape)
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg1,'salegrid_tva1e1b1nwziyg1', shape=tva1e1b1nwziyg1_shape) #didnt worry about unclustering since not important report and wasnt masked by z8
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg2,'salegrid_Tva1e1b1nwzixyg2', shape=Tva1e1b1nwzixyg2_shape) #didnt worry about unclustering since not important report and wasnt masked by z8
    fun.f1_make_r_val(r_vals,r_salegrid_tva1e1b1nwzida0e0b0xyg3,'salegrid_tvnwzida0e0b0xyg3', shape=tvnwzidaebxyg3_shape) #didnt worry about unclustering since not important report and wasnt masked by z8

    fun.f1_make_r_val(r_vals,r_woolvalue_p7tva1e1b1nwzida0e0b0xyg0,'woolvalue_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)
    fun.f1_make_r_val(r_vals,r_woolvalue_k2p7tva1e1b1nwzida0e0b0xyg1,'woolvalue_k2p7tva1nwziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,...],z_pos, k2p7tva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,r_woolvalue_k3k5p7tva1e1b1nwzida0e0b0xyg3,'woolvalue_k3k5p7tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,...],z_pos, k3k5p7tvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,rm_stockinfra_var_h1p7z,'rm_stockinfra_var_h1p7z',mask_season_p7z,z_pos=-1)
    fun.f1_make_r_val(r_vals,rm_stockinfra_fix_h1p7z,'rm_stockinfra_fix_h1p7z',mask_season_p7z,z_pos=-1)

    ###purchase costs
    fun.f1_make_r_val(r_vals,purchcost_p7tva1e1b1nwzida0e0b0xyg0,'purchcost_sire_p7zg0',mask_z8var_p7tva1e1b1nwzida0e0b0xyg,z_pos, p7zg0_shape)

    ###sale date
    fun.f1_make_r_val(r_vals,r_saledate_k3k5tva1e1b1nwzida0e0b0xyg3,'saledate_k3k5tvnwziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3,z_pos, k3k5tvnwziaxyg3_shape)

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

    ###mei and pi and NV (nutritive value)
    fun.f1_make_r_val(r_vals,mei_p6ftva1e1b1nwzida0e0b0xyg0,'mei_sire_p6fzg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...],z_pos, p6fzg0_shape)
    fun.f1_make_r_val(r_vals,mei_k2p6ftva1e1b1nwzida0e0b0xyg1,'mei_dams_k2p6ftva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],z_pos, k2p6ftva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,mei_k3k5p6ftva1e1b1nwzida0e0b0xyg3,'mei_offs_k3k5p6ftvnw8ziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],z_pos, k3k5p6ftvnwziaxyg3_shape)

    fun.f1_make_r_val(r_vals,pi_p6ftva1e1b1nwzida0e0b0xyg0,'pi_sire_p6fzg0',mask_fp_z8var_p6tva1e1b1nwzida0e0b0xyg[:,na,...],z_pos, p6fzg0_shape)
    fun.f1_make_r_val(r_vals,pi_k2p6ftva1e1b1nwzida0e0b0xyg1,'pi_dams_k2p6ftva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,na,na,...],z_pos, k2p6ftva1nwziyg1_shape)
    fun.f1_make_r_val(r_vals,pi_k3k5p6ftva1e1b1nwzida0e0b0xyg3,'pi_offs_k3k5p6ftvnw8ziaxyg3',mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,na,na,...],z_pos, k3k5p6ftvnwziaxyg3_shape)


    ###proportion mated per dam at beginning of the period (eg accounts for mortality)
    fun.f1_make_r_val(r_vals,r_n_mated_k2tva1e1b1nwzida0e0b0xyg1,'n_mated_k2tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)

    ###proportion of drys per dam at beginning of the period (eg accounts for mortality)
    fun.f1_make_r_val(r_vals,r_n_drys_k2tva1e1b1nwzida0e0b0xyg1,'n_drys_k2tva1nw8ziyg1',mask_z8var_k2tva1e1b1nwzida0e0b0xyg1,z_pos, k2tva1nwziyg1_shape)

    ###number of foetuses scanned per dam at beginning of the period (eg accounts for mortality)
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

    ###mort - uses b axis instead of k for extra detail when scan=0
    if pinp.rep['i_store_mort']:
        fun.f1_make_r_val(r_vals,r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg1.squeeze(axis=(d_pos, a0_pos, e0_pos, b0_pos, x_pos)),'mort_pa1e1b1nwziyg1') #no unclustering because this wasnt masked by z8
        fun.f1_make_r_val(r_vals,r_cum_dvp_mort_pa1e1b1nwzida0e0b0xyg3.squeeze(axis=(a1_pos, e1_pos, b1_pos)),'mort_pnwzida0e0b0xyg3') #no unclustering because this wasnt masked by z8

    ###on hand mort - proportion of each sheep remaining in each period after accounting for mort
    if pinp.rep['i_store_on_hand_mort']:
        fun.f1_make_r_val(r_vals,r_on_hand_mort_k2tvpa1e1b1nwzida0e0b0xyg1,'on_hand_mort_k2tvpa1nwziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_on_hand_mort_k3k5tvpa1e1b1nwzida0e0b0xyg3,'on_hand_mort_k3k5tvpnwziaxyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5tvpnwziaxyg3_shape)

    ###numbers weights for reports with arrays that keep axis that are not present in lp array.
    if pinp.rep['i_store_lw_rep'] or pinp.rep['i_store_ffcfw_rep'] or pinp.rep['i_store_nv_rep']:

        ###weights the denominator and numerator - required for reports when p, e and b are added and weighted average is taken (otherwise broadcasting the variable activity to the new axis causes error in result)
        ###If these arrays get too big might have to add a second denom weight in reporting.
        pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1 = ((a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                                                      * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...])
                                                      * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...]
                                                      * o_numbers_start_tpdams)
        fun.f1_make_r_val(r_vals,pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1,'pe1b1_numbers_weights_k2tvpa1e1b1nw8ziyg1',
                          mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1e1b1nwziyg1_shape)

        ###for yatf a b1 weighting must be given
        pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1 = ((a_v_pa1e1b1nwzida0e0b0xyg1 == index_vpa1e1b1nwzida0e0b0xyg1)
                                                             * (a_k2cluster_va1e1b1nwzida0e0b0xyg1[:,na,...] == index_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...])
                                                             * on_hand_tpa1e1b1nwzida0e0b0xyg1[:,na,...]
                                                             * o_numbers_start_tpyatf)
        fun.f1_make_r_val(r_vals,pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1,'pe1b1_nyatf_numbers_weights_k2tvpa1e1b1nw8zixyg1',
                          mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2tvpa1e1b1nwzixyg1_shape)

        pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3 = ((a_v_pa1e1b1nwzida0e0b0xyg3 == index_vpa1e1b1nwzida0e0b0xyg3)
                                                            * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...])
                                                            * (a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3[:,:,:,na,...])
                                                            * on_hand_tpa1e1b1nwzida0e0b0xyg3[:,na,...]
                                                            * o_numbers_start_tpoffs)
        fun.f1_make_r_val(r_vals,pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3,'pde0b0_numbers_weights_k3k5tvpnw8zida0e0b0xyg3',
                          mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5tvpnwzidae0b0xyg3_shape)

        de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2 = ((a_k5cluster_da0e0b0xyg3 == index_k5tva1e1b1nwzida0e0b0xyg3)
                                                             * (a_k3cluster_da0e0b0xyg3 == index_k3k5tva1e1b1nwzida0e0b0xyg3)
                                                             * numbers_start_d_prog_a0e0b0_a1e1b1nwzida0e0b0xyg2
                                                            ).squeeze(axis=(p_pos, a1_pos, e1_pos, b1_pos, n_pos))
        fun.f1_make_r_val(r_vals,de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2,'de0b0_denom_weights_prog_k3k5tw8zida0e0b0xyg2') #no mask because p axis to mask

    ###lw - with p, e, b
    if pinp.rep['i_store_lw_rep']:
        fun.f1_make_r_val(r_vals,r_lw_sire_tpsire,'lw_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_lw_dams_k2tvpdams,'lw_dams_k2vpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2vpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_lw_offs_k3k5tvpoffs,'lw_offs_k3k5vpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5vpnwzidae0b0xyg3_shape)

    ###ffcfw - with p, e, b
    if pinp.rep['i_store_ffcfw_rep']:
        fun.f1_make_r_val(r_vals,r_ffcfw_sire_tpsire,'ffcfw_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_ffcfw_dams_k2tvpdams,'ffcfw_dams_k2vpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2vpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_ffcfw_yatf_k2tvpyatf,'ffcfw_yatf_k2vpa1e1b1nw8zixyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2vpa1e1b1nwzixyg1_shape)
        fun.f1_make_r_val(r_vals,r_ffcfw_prog_k3k5tva1e1b1nwzida0e0b0xyg2,'ffcfw_prog_k3k5wzida0e0b0xyg2', shape=k3k5wzida0e0b0xyg2_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_ffcfw_offs_k3k5tvpoffs,'ffcfw_offs_k3k5vpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5vpnwzidae0b0xyg3_shape)

    ###NV - with p, e, b
    if pinp.rep['i_store_nv_rep']:
        fun.f1_make_r_val(r_vals,r_nv_sire_pg,'nv_sire_pzg0',shape=pzg0_shape) #no v axis to mask
        fun.f1_make_r_val(r_vals,r_nv_dams_k2tvpg,'nv_dams_k2vpa1e1b1nw8ziyg1', mask_z8var_k2tva1e1b1nwzida0e0b0xyg1[:,:,:,na,...],z_pos,k2vpa1e1b1nwziyg1_shape)
        fun.f1_make_r_val(r_vals,r_nv_offs_k3k5tvpg,'nv_offs_k3k5vpnw8zida0e0b0xyg3', mask_z8var_k3k5tva1e1b1nwzida0e0b0xyg3[:,:,:,:,na,...],z_pos,k3k5vpnwzidae0b0xyg3_shape)




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
