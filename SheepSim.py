# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John
"""

"""
import functions from other modules
"""
import datetime as dt
import pandas as pd
import numpy as np

# from numba import jit

# import FeedBudget as fdb
# import Functions as fun
# import Periods as per
import PropertyInputs as pinp
import SheepSimRoutines as sfun
import UniversalInputs as uinp


############################
### _constants required    #
############################
# ## define some parameters required to size arrays.
# n_feed_pools        = uinp.n_feed_pools
# n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1

# ^ put the values as lists in universal.xlsx (SheepDefinitions!) or in the structure dict in universal.py (to keep consistent)
# then define n by length of the list
n_animal_types = 2      # a: wool, meat
n_btrt = 6              # b: 11, 22, 21, 33, 32, 31
n_genotype_options = 5  # c: number in spreadhseet
n_dam_ages = 3          # d: yearling, maiden, adult
n_max_ecycles = 2       # e: max number of estrus cycles they are joined
n_feed_var_periods = 10  # f:
n_genotypes = 6         # g: B, M, T, BM, BT, BMT
n_g0 = 3                # g0: B, M, T
n_g1 = 2                # g1: BB, BM
n_g2 = 4                # g2: BBB, BBM, BBT, BMT
n_husbandry_class = 1   # h
n_groups_rams = 3       # i & g0: genotypes of rams
n_groups_ewes = 4       # j: genotype groups of ewes
n_groups_offspring = 5  # k: genotype groups and growth profile
n_groups_lambing = 1    # l: lambing groups for the seed animals
# (1 unless doing a TOL analysis or 8 month joinings)
n_months = 12           # m: Jan to Dec
n_feed_periods = 10     # n:
n_lambing_opps = 15     # o:
# n_sim_periods         # p: see below
n_labour_periods = 16   # q:
# n_feed_variables        # r:
n_shearing_occs = 16    # s:
n_sale_times = 4        # t: weaner, backgrounded, finished, remainder
n_husbandry_options = 10  # u:
n_genders = 3           # w: ewe, wether, ram
n_litter_size = 5       # x: Dry, single, twin, triplet, not mated
n_lactation_number = 5  # y: dry, single, twin, triplet, in utero
n_sexes = 3             # w: ram, ewe, wether
# n_sim_periods see below
n_labour_periods = 16   # q
i_sim_periods_year = 52  # ^uinp.n_sim_periods_year  now in structure dict
i_oldest_animal = 6.5    # ^uinp.i_oldest_animal

birth_date_i = uinp.propertydata['ExcelName']   # Find the ExcelNames
birth_date_jl = uinp.propertydata['ExcelName']
birth_date_jel = uinp.propertydata['ExcelName']
birth_date_kel = uinp.propertydata['ExcelName']
# ## Some one time data manipulation for the inputs just read
start_year = np.min(birth_date_jl)
# ## might need to test and rebase the year for the other animal groups


### _define the periods
n_sim_periods, date_p, p_index_p, step \
        = sfun.sim_periods(start_year, i_sim_periods_year, i_oldest_animal)
### _array dimensions




###################################
### initialise global arrays      #
###################################
'''only create arrays that are used in sim and post processing.
'''
## Instantiate the globals arrays
## # these store the output of simulation and the parameters for pyomo
## # see documentation for a description of each variable




def simulation():
    """
    A function to wrap the simulation that can be called by SheepPyomo.

    Called after the sensitivty variables have been updated.
    It populates the arrays by looping through the time periods
    Globally define arrays are used to transfer results to sheep_paramters()

    Returns
    -------
    None.
    """
    ###################################
    ### index arrays                  # 
    ###################################
    index_p = np.arange(300)#asarray(300)

    ###################################
    ### reshape neccessary inputs     # 
    ###################################
    '''only >2 dim array'''


    ############################
    ### initialise arrays      #
    ############################
    '''only if assign with a slice'''
    ## Instantiate the arrays that are only required within this function
    ## mainly arrays that will store the input data that require pre-defining
    ## # see documentation for a description of each variable

    lact_nut_effect_piec1 #^ i think this array neds to be initilised with "False" as each value
    d_nw_max_pibgc1 #^so far the only code using this array involves assigning a slice
    
    ############################
    ### build arrays           #
    ############################
    '''this includes association arrays and other arrays that can be built outside loop'''
    ## the association arrays relate the slices of one array with the slices of another array
    ##needs to be within the loop because the genotype inputs can change in exp.xlsx

    ############################
    ### sim param arrays       #
    ############################
    #^check over k2g function... i dont understand it
    cc_gg	= sfun.f_k2g(i_cc_gk1, a_k1_g0, i_cc_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cd_gg	= sfun.f_k2g(i_cd_gk1, a_k1_g0, i_cd_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cf_gg	= sfun.f_k2g(i_cf_gk1, a_k1_g0, i_cf_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cg_gg	= sfun.f_k2g(i_cg_gk1, a_k1_g0, i_cg_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    ch_gg	= sfun.f_k2g(i_ch_gk1, a_k1_g0, i_ch_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    ci_gg	= sfun.f_k2g(i_ci_gk1, a_k1_g0, i_ci_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    ck_gg	= sfun.f_k2g(i_ck_gk1, a_k1_g0, i_ck_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cl_gg	= sfun.f_k2g(i_cl_gk1, a_k1_g0, i_cl_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cm_gg	= sfun.f_k2g(i_cm_gk1, a_k1_g0, i_cm_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cn_gg	= sfun.f_k2g(i_cn_gk1, a_k1_g0, i_cn_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cp_gg	= sfun.f_k2g(i_cp_gk1, a_k1_g0, i_cp_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cr_gg	= sfun.f_k2g(i_cr_gk1, a_k1_g0, i_cr_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cw_gg	= sfun.f_k2g(i_cw_gk1, a_k1_g0, i_cw_gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cb_bgg	= sfun.f_k2g(i_cb_bgk1, a_k1_g0, i_cb_bgk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    ce_dgg	= sfun.f_k2g(i_ce_dgk1, a_k1_g0, i_ce_dgk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cu1_u1gg	= sfun.f_k2g(i_cu1_u1gk1, a_k1_g0, i_cu1_u1gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    #^check line below. should it be cu2_u2gg?
    cu2_u1gg	= sfun.f_k2g(i_cu2_u1gk1, a_k1_g0, i_cu1_u1gk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cv_vgg	= sfun.f_k2g(i_cv_vgk1, a_k1_g0, i_cv_vgk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cx_xgg	= sfun.f_k2g(i_cx_xgk1, a_k1_g0, i_cx_xgk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cy_ygg	= sfun.f_k2g(i_cy_ygk1, a_k1_g0, i_cy_ygk2, a_k2_g0, a_maternal_g0_g1, a_paternal_g0_g1, a_maternal_g1_g2, a_paternal_g0_g2)			
    cc_gc0 = cc_gg[..., a_g_c0]		
    cc_gc1 = cc_gg[..., a_g_c1]		
    cc_gc2 = cc_gg[..., a_g_c2]			
    cd_gc0 = cd_gg[..., a_g_c0]			
    cd_gc1	= cd_gg[..., a_g_c1]			
    cd_gc2	= cd_gg[..., a_g_c2]			
    cf_gc0 = cf_gg[..., a_g_c0]			
    cf_gc1 = cf_gg[..., a_g_c1]			
    cf_gc2 = cf_gg[..., a_g_c2]			
    cg_gc0 = cg_gg[..., a_g_c0]			
    cg_gc1 = cg_gg[..., a_g_c1]			
    cg_gc2 = cg_gg[..., a_g_c2]			
    ch_gc0 = ch_gg[..., a_g_c0]			
    ch_gc1 = ch_gg[..., a_g_c1]			
    ch_gc2 = ch_gg[..., a_g_c2]			
    ci_gc0 = ci_gg[..., a_g_c0]			
    ci_gc1 = ci_gg[..., a_g_c1]			
    ci_gc2 = ci_gg[..., a_g_c2]			
    ck_gc0 = ck_gg[..., a_g_c0]			
    ck_gc1 = ck_gg[..., a_g_c1]			
    ck_gc2 = ck_gg[..., a_g_c2]			
    cl_gc0 = cl_gg[..., a_g_c0]			
    cl_gc1 = cl_gg[..., a_g_c1]			
    cl_gc2 = cl_gg[..., a_g_c2]			
    cm_gc0 = cm_gg[..., a_g_c0]			
    cm_gc1 = cm_gg[..., a_g_c1]			
    cm_gc2 = cm_gg[..., a_g_c2]			
    cn_gc0 = cn_gg[..., a_g_c0]			
    cn_gc1 = cn_gg[..., a_g_c1]			
    cn_gc2 = cn_gg[..., a_g_c2]			
    cp_gc0 = cp_gg[..., a_g_c0]			
    cp_gc1 = cp_gg[..., a_g_c1]			
    cp_gc2 = cp_gg[..., a_g_c2]			
    cr_gc0 = cr_gg[..., a_g_c0]			
    cr_gc1 = cr_gg[..., a_g_c1]			
    cr_gc2 = cr_gg[..., a_g_c2]			
    cw_gc0 = cw_gg[..., a_g_c0]			
    cw_gc1 = cw_gg[..., a_g_c1]			
    cw_gc2 = cw_gg[..., a_g_c2]			
    cb_bgc0 = cb_bgg[..., a_g_c0]			
    cb_bgc1 = cb_bgg[..., a_g_c1]			
    cb_bgc2 = cb_bgg[..., a_g_c2]			
    ce_bgc0 = ce_dgg[..., a_g_c0]			
    ce_bgc1 = ce_dgg[..., a_g_c1]			
    ce_bgc2 = ce_dgg[..., a_g_c2]			
    cu1_u1gc0 = cu1_u1gg[..., a_g_c0]			
    cu1_u1gc1 = cu1_u1gg[..., a_g_c1]			
    cu1_u1gc2 = cu1_u1gg[..., a_g_c2]			
    cu2_u2gc0 = cu2_u1gg[..., a_g_c0]	#^check cu2_u1gg? or cu2_u2gg?		
    cu2_u2gc1 = cu2_u1gg[..., a_g_c1]  #^check
    cu2_u2gc2 = cu2_u1gg[..., a_g_c2]	#^check	
    cv_vgc0 = cv_vgg[..., a_g_c0]			
    cv_vgc1 = cv_vgg[..., a_g_c1]			
    cv_vgc2 = cv_vgg[..., a_g_c2]			
    cx_xgc0 = cx_xgg[..., a_g_c0]			
    cx_xgc1 = cx_xgg[..., a_g_c1]			
    cx_xgc2 = cx_xgg[..., a_g_c2]			
    cy_ygc0 = cy_ygg[..., a_g_c0]			
    cy_ygc1 = cy_ygg[..., a_g_c1]			
    cy_ygc2 = cy_ygg[..., a_g_c2]			
    

    ############################
    ### management calculations#
    ############################
    date_birth_c0 =
    date_birth_ic1 =
    date_birth_diec2 =
    date_birth_y_oiec1 = 
    startdate = np.minimum(date_birth_c0 , date_birth_ic1 , date_birth_diec2)			
    step = sfun.sim_periods(start_year, i_sim_periods_year, i_oldest_animal)[3] #^need to populate function with actual args
    step_int = step.astype('timedelta64[D]').astype(int)			
    step_float = step.astype(float)/(24 * 60 * 60)			
    age_mated_oiec1 = pinp.sheep['i_age_join_oic1'][:, :, np.newaxis, ...] + cf_gc1[4, 0, :] * (index_e + 0.5)			
    age_scan_oiec1 = age_mated_oiec1 + i_scan_day_oic1[:, :, np.newaxis, ...]			
    age_day90_oiec1 = age_mated_oiec1 + 90			
    age_lamb_oiec1 = age_mated_oiec1 + cp_gc1[1, 1, ...]		#^gets info from paramerters	
    age_wean_oiaec1 = i_age_join_oic1[:, :, np.newaxis, np.newaxis, ...] + i_wean_day_oiac1[:, :, :, np.newaxis, ...]			
    age_wean_y_oiaec1 = age_wean_oiaec1 - age_lamb_oiec1[:, :, np.newaxis, ...]			
    age_6weekspp_oiec1 = age_lamb_oiec1 - 42			
    period_is_break
    period_is_mating
    period_is_scanning
    period_is_6wpp
    period_is_lambing
    period_is_weaning
    period_is_shearing
    period_is_fvp
    prev_is_mating
    prev_is_lambing
    period_is_lactation
    period_is_prejoiningfvp
    break_in_period
    mating_in_period
    scanning_in_period
    lambing_in_period
    weaning_in_period
    fvp_propn
    fvp_propn
    fvp_propn




    ############################
    ### Genotype calcs         #
    ############################

    c_srw_gg0	= i_srw_gk2[..., a_k2_g0]			
    c_srw_gg1	= (c_srw_gg0[..., a_paternal_g0_g1] + c_srw_gg0[..., a_maternal_g0_g1])/2			
    c_srw_gg2	= (c_srw_gg0[..., a_paternal_g0_g2] + c_srw_gg1[..., a_maternal_g1_g2])/2			
    c_srw_gg	Function required to determine which options exist			
    c_sfw_gg0	= i_gfw_gk2[..., a_k2_g0]			
    c_sfw_gg1	= (c_sfw_gg0[..., a_paternal_g0_g1] + c_sfw_gg0[..., a_maternal_g0_g1])/2			
    c_sfw_gg2	= (c_sfw_gg0[..., a_paternal_g0_g2] + c_sfw_gg1[..., a_maternal_g1_g2])/2						
    c_sfw_gg	Function required to determine which options exist			
    c_fd_gg0	= i_fd_gk2[..., a_k2_g0]			
    c_fd_gg1	= (c_fd_gg0[..., a_paternal_g0_g1] + c_fd_gg0[..., a_maternal_g0_g1])/2			
    c_fd_gg2	= (c_fd_gg0[..., a_paternal_g0_g2] + c_fd_gg1[..., a_maternal_g1_g2])/2			
    c_fd_gg	Function required to determine which options exist			
    c_scan_std_c1	= i_scan_std_k2:  convert k to g0, create g1 from g0, then associate g1 to c1			
    c_lss_std_c1	= i_lss_std_k2:  convert k to g0, create g1 from g0, then associate g1 to c1			
    c_lstw_std_c1	= i_lstw_std_k2:  convert k to g0, create g1 from g0, then associate g1 to c1			
    c_lstr_std_c1	= i_lstr_std_k2:  convert k to g0, create g1 from g0, then associate g1 to c1			
    agedam_propn_dc1	= i_agedam_propn_std_dk2:  convert k to g0, create g1 from g0, then associate g1 to c1			
    c_btrt_propn_std_bk2	= f_btrt(i_scan_std_k2& i_lss_std_k2& i_lstw_std_k2& i_lstr_std_k2)			
    btrt_propn_bc0 =c_btrt_propn_std_bk2: convert k to g0, create g1 from g0, create c1 from g1			
    btrt_propn_bc1	=c_btrt_propn_std_bk2: convert k to g0, create g1 from g0, create c1 from g1			
    bt_propn_xc0 = calculate from btrt: btrt_propn_bc0			
    bt_propn_xc1	= calculate from btrt: btrt_propn_bc1			
    srw_gc1	= c_srw_gg[..., a_g_c1] * cv_vgc1[1, 1, ...]   #(srw gender adjustment)			
    srw_gc0	= c_srw_gg[..., a_g_c0] * cv_vgc0[1, 0, ...]   #(srw gender adjustment)			
    srw_vgc2	= c_srw_gg[..., a_g_c2] * cv_vgc2[1, ...]   #(srw gender adjustment)			
    sfw_obgc1	= c_sfw_gg[..., a_g_c1] + cb_bgc1[0, ...] + ce_dgc1[0, :, np.newaxis, ...]			
    sfw_bgc1	= np.sum(sfw_obgc1 * agedam_propn_dc1[..., np.newaxis, np.newaxis, :], axis = 0)			
    sfw_bgc0 = c_sfw_gg[..., a_g_c0] + cb_bgc0[0, ...]			
    sfw_obgc2 = c_sfw_gg[..., a_g_c2] + cb_bgc2[0, ...] + ce_dgc2[0, :, np.newaxis, ...]			
    sfd_obgc1	= c_fd_gg[..., a_g_c1] + cb_bgc1[1, ...] + ce_dgc1[1, :, np.newaxis, ...]			
    sfd_bgc1	= np.sum(sfd_obgc1 * agedam_propn_dc1[..., np.newaxis, np.newaxis, :], axis = 0)			
    sfd_bgc0	= c_fd_gg[..., a_g_c0] + cb_bgc0[1, ...]			
    sfd_obgc2	= c_fd_gg[..., a_g_c2] + cb_bgc2[1, ...] + ce_dgc2[1, :, np.newaxis, ...]			
    wge_bgc0	= sfw_bgc0 / (srw_gc0 / cv_vgc0[1, 0, ...])			
    wge_gc1	= sfw_bgc1 / srw_gc1			
    wge_obgc2	= sfw_obgc2 / srw_vgc2[1, ...]			
    w_b_bgc0	= srw_gc0 * cx_xgc1[5, a_x_b, ...] * cv_vgc1[2, 0, ...]			
    w_b_bgc1	= srw_gc1 * cx_xgc1[5, a_x_b, ...]  * np.sum(ce_dgc1[2, ...] * agedam_propn_dc1[..., np.newaxis, :], axis = 0) * cv_vgc1[2, 1, ...]			
    nw_b_std_xgc2	= srw_vgc2[1, ...] * cx_xgc2[5, ...]			
    lgf_eff_pgc0 = 1 + ck_gc0[14,...] * i_legume_m[a_m_p, np.newaxis]			
    lgf_eff_pgc1 = 1 + ck_gc1[14,...] * i_legume_m[a_m_p, np.newaxis]			
    lgf_eff_pgc2 = 1 + ck_gc2[14,...] * i_legume_m[a_m_p, np.newaxis]			
    dlf_wool_pgc0 = 1 + cw_gc0[6,...] * (dl_p[:, np.newaxis, np.newaxis] - 12)			
    dlf_wool_pgc1 = 1 + cw_gc1[6,...] * (dl_p[:, np.newaxis, np.newaxis] - 12)			
    dlf_wool_pgc2 = 1 + cw_gc2[6,...] * (dl_p[:, np.newaxis, np.newaxis] - 12)			
    kw_gc0 = ck_gc0[17,...]			
    kw_gc1 = ck_gc1[17,...]			
    kw_gc2 = ck_gc2[17,...]			
    kc_gc1	= ck_gc1[8,...]	
#^needs af_wool from daily step section. 
    mew_min_pgc0 = cw_gc0[14, ...] * sfw_bgc0[0, ...] / 365 * af_wool_pgc0 * dlf_wool_pgc0[:, np.newaxis, ...] * cw_gc0[1, ...] / kw_gc0			
    mew_min_pigc1 = cw_gc1[14, ...] * sfw_bgc1[0, ...] / 365 * af_wool_pigc1 * dlf_wool_pgc1[:, np.newaxis, ...] * cw_gc1[1, ...] / kw_gc1			
    mew_min_pdiegc2 = cw_gc2[14, ...] * sfw_bgc2[0, ...] / 365 * af_wool_pdiegc2 * dlf_wool_pgc2[:, np.newaxis, ...] * cw_gc2[1, ...] / kw_gc2			
    mew_min_y_piegc1 = cw_gc1[14, ...] * sfw_bgc1[0, ...] / 365 * af_wool_y_piegc1 * dlf_wool_pgc1[:, np.newaxis, ...] * cw_gc1[1, ...] / kw_gc1			

    ############################
    ### Normal foetus          #
    ############################
ra_f	piec1m1	= age_f_piec1m1 / cp_gc1[1, 0, :, na]			
ra_f_end	piec2	= age_f_end_piec1 / cp_gc1[1, 0, :, na]			
nwf_age_f	piegc1	= np.nanmean(np.exp(cp_gc1[2, …, na] * (1 - np.exp(cp_gc1[3, …, na] * (1 - ra_f_piec1m1[…, na, :, :]))), axis = -1)			
nwf_age_f_end	piegc2	= np.exp(cp_gc1[2, …, na] * (1 - np.exp(cp_gc1[3, …, na] * (1 - ra_f_end_piec2[…, na, :])))			
d_nwf_age_f	piegc1	= np.nanmean(cp_gc1[2, ..., na] * cp_gc1[3, ..., na] / cp_gc1[1, 0, :, na] * np.exp(cp_gc1[3, ..., na] * (1 - ra_f_piec1m1[…, na, :, :]) + cp_gc1[2, ..., na] * (1 - np.exp(cp_gc1c[3, ..., na] * (1 - ra_f_piec1m1[…, na, :, :]))), axis = -1)			
cw_age_f	piegc1	= np.nanmean(np.exp(cp_gc1[6, ..., na] * (1 - np.exp(cp_gc1[7, ..., na] * (1 - ra_f_piec1m1[…, na, :, :]))), axis = -1)			
ce_age_f	piegc1	= np.exp(cp_gc1[9, ...] * (1 - np.exp(cp_gc1[10, ...] * (1 - ra_f_end_piec2[…, na, :])))			
nw_b	pixbgc1	= (1 - cp_gc1[4, ...] * (1 - relsize_max_pibgc1[:, na, ...])) * nw_b_std_xgc2[:, na, ...]			
nw_b	pvixbgc2	= (1 - cp_gc1[4, ...] * (1 - relsize_max_pibgc1[:, na, na, ...])) * cv_vgc1[2, :, na, na, ...] * nw_b_std_xgc2[:, na, ...]			
nw_f	piexbgc1	= nw_b_pixbgc1[:, :, na, ...] * nwf_age_f_piegc1[:, na, …]			
nw_f_end	pviexbgc2	= nw_b_pvixbgc2[:, :, :, na, ...] * nwf_age_f_end_piegc2[:, na, na, …]			
d_nw_f	piexbgc1	= nw_b_pixbgc1[:, :, na, ...] * d_nwf_age_f_piegc1[:, na, …]			
nwf_nwb	piexbgc1	= nw_f_piexbgc1 / nw_b_pixbgc1[:, :, na, ...]			
nw_c	piexbgc1	= cp_gc1[5, ...] * nw_b_pixbgc1[:, :, na, ...] * cw_age_f_piegc1[:, na, ...]			
normale_c	piexbgc1	= cp_gc1[8, ...] * cp_gc1[5, ...] * nw_b_pixbgc1[:, :, na, ...] * ce_age_f_piegc1[:, na, ...]			



    ############################
    ### Feed suply inputs      #
    ############################

    ######################
    ##group independent  #
    ######################
    date_start_p = (startdate + step * index_p).astype('datetime64[D]')			
    date_end_p = (startdate- 1 + step * (index_p + 1)).astype('datetime64[D]')			
    doy_p = (date_start_p - date_start_p.astype('datetime64[Y]')).astype(int) + 1			
    dl_p = sfun.daylength(dayOfYear, lat)			
    
    ######################
    ##Age, date, timing  #
    ######################
    hf_gc0 = 1 + cr_gc0[12,...] * (i_hr - 1)			
    hf_gc1 = 1 + cr_gc1[12,...] * (i_hr - 1)			
    hf_gc2 = 1 + cr_gc2[12,...] * (i_hr - 1)			
    age_start_pc0 = np.maximum(0, date_start_p - date_birth_c0)			
    age_start_pic1 = np.maximum(0, date_start_p - date_birth_ic1)			
    age_start_pdiec2 = np.maximum(0, date_start_p - date_birth_diec2)			
    age_end_pc0 = np.maximum(0, date_end_p - date_birth_c0)			
    age_end_pic1 = np.maximum(0, date_end_p - date_birth_ic1)			
    age_end_pdiec2 = np.maximum(0, date_end_p - date_birth_diec2)			
    age_pc0 = (age_start_pc0 + age_end_pc0 + 1) / 2			
    age_pic1	= (age_start_pic1 + age_end_pic1 + 1) / 2			
    age_pdiec2 = (age_start_pdiec2 + age_end_pdiec2 + 1) / 2			
    days_period_pc0 = age_end_pc0 - age_start_pc0 + 1			
    days_period_pic1 = age_end_pic1 - age_start_pic1 + 1			
    days_period_pdiec2 = age_end_pdiec2 - age_start_pdiec2 + 1	
#^below here needs the joining opportunity associations..(maybe just build it here because it requires age info above)		
    age_f_start_piec1	= np.maximum(0, np.minimum(cp_gc1[1, 0, :] + 1, age_start_pic1[:, np.newaxis, ...] - age_mated_oiec1[a_prevdam_o_pic1]))			
    age_f_end_piec1	= np.maximum(0, np.minimum(cp_gc1[1, 0, :], age_end_pic1[:, np.newaxis, ...] - age_mated_oiec1[a_prevdam_o_pic1]))			
    age_f_piec1	= (age_f_start_piec1 + age_f_end_piec1+ 1) / 2			
    days_period_f_piec1 = age_f_end_piec1 - age_f_start_piec1 + 1			
    age_y_start_piec1	= np.maximum(0, age_start_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevyaf_o_pic1[:, np.newaxis, ...]])			
    age_y_end_piec1	= np.maximum(0, age_end_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevyaf_o_pic1[:, np.newaxis, ...]])			
    age_y_piec1	=(age_y_start_piec1 + age_y_end_piec1+ 1) / 2
#^needs weaning assosiation array			
    age_y_adj_piaec1	= np.maximum(age_y_piec1[:, np.newaxis, ...] , age_wean_y_oiaec1[a_wean_o_piaec1] + (age_y_piec1[:, np.newaxis, ...] - age_wean_y_oiaec1[a_wean_o_piaec1]) * ci_gc1[21, ...]			
    lact_nut_effect_piec1[age_y_piec1 > cl_gc1[16, ...] * cl_gc1[2, ...]] = True			
    gest_propn_piec1	= np.maximum(0 , np.minimum(days_period_pic1, age_end_pic1[:, np.newaxis, ...] - age_mated_oiec1[a_prevdam_o_pic1], age_lamb_oiec1[a_prevdam_o_pic1] - age_start_pic1[:, np.newaxis, ...])) / days_period_pic1			
    lact_propn_piaec1	= np.maximum(0 , np.minimum(days_period_pic1, (age_end_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevdam_o_pic1])[:, np.newaxis, ...], (age_wean_oiaec1[a_prevyaf_o_pic1] - age_start_pic1[:, np.newaxis, ...])[:, np.newaxis, ...])) / days_period_pic1			
    weanage_e_oiaec1	
#^needs genotype calcs			
    d_cfw_ave_pbgc0 = cw_gc0[3, ...] * sfw_bgc0 * af_wool_pgc0[..., np.newaxis, :] * days_period_pc0[..., np.newaxis, :] / 365			
    d_cfw_ave_pibgc1 = cw_gc1[3, ...] * sfw_bgc1 * af_wool_pigc1[..., np.newaxis, :] * days_period_pic1[..., np.newaxis, :] / 365			
    d_cfw_ave_pdiebgc2 = cw_gc2[3, ...] * sfw_obgc2 * af_wool_pdiegc2[..., np.newaxis, :] * days_period_pdiec2[..., np.newaxis, :] / 365			
    nw_max_pbgc0 = srw_gc0 * (1 - srw_age_pgc0) + w_b_bgc0 * srw_age_pgc0			
    nw_max_pibgc1 = srw_gc1 * (1 - srw_age_pigc1) + w_b_bgc1 * srw_age_pigc1			
    d_nw_max_pibgc1[0:-1, ...] = (nw_max_pibgc1[1:, ...] - nw_max_pibgc1[0:-1]) / days_period_pic1[..., np.newaxis, :]			
    relsize_max_pibgc1 = nw_max_pibgc1 / srw_gc1			


    ############################
    ### Daily steps            #
    ############################
    n_minor_m1				
    axis_m1	= index_m1 - (n_minor - 1) / (2 * n_minor) * step			
    age_m_pc0m1= age_pc0[..., np.newaxis] + axis_m1
    age_m_pic1m1 = age_pic1[..., np.newaxis] + axis_m1
    age_m_pdic2m1 = age_pdiec2[..., np.newaxis] + axis_m1
    age_m_pic1m1[age_m_pic1m1<=0] = np.nan			
    doy_m_pm1 = doy_p[..., np.newaxis] + axis_m1			
    chill_pm1	= (481 + (11.7 + 3.1 * i_ws_p[..., np.newaxis] ** 0.5) * (40 - i_temp_ave_p[..., np.newaxis]) + 418 * (1-np.exp(-0.04 * c_rain_pm1))			
#^needs genotype calcs			
    srw_age_pgc0 = np.nanmean(np.exp(-cn_gc0[1, ..., np.newaxis] * age_m_pc0m1[..., np.newaxis, :, :] / srw_gc0[..., np.newaxis] ** cn_gc0[2, ..., np.newaxis]), axis = -1)			
    srw_age_pigc1	= np.nanmean(np.exp(-cn_gc1[1, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :] / srw_gc1[..., np.newaxis] ** cn_gc1[2, ..., np.newaxis]), axis = -1)			
    srw_age_y_pviegc1	= np.nanmean(np.exp(-cn_gc2[1, ..., np.newaxis] * age_y_m_piec1m1[:, np.newaxis, ..., np.newaxis, :, :] / srw_vgc2[:, np.newaxis, np.newaxis, ..., np.newaxis] ** cn_gc2[2, ..., np.newaxis]), axis = -1)			
    srw_age_pdvigc2	= np.nanmean(np.exp(-cn_gc2[1, ..., np.newaxis] * age_m_pdic2m1[:, :, np.newaxis, ..., np.newaxis, :, :] / srw_vgc2[:, np.newaxis, np.newaxis, ..., np.newaxis] ** cn_gc2[2, ..., np.newaxis]), axis = -1)			
    af_wool_pgc0 = np.nanmean(cw_gc0[5, ..., np.newaxis] + (1 - cw_gc0[5, ..., np.newaxis])*(1-np.exp(-cw_gc0[12, ..., np.newaxis] * age_m_pc0m1[..., np.newaxis, :, :]), axis = -1)			
    af_wool_pigc1 = np.nanmean(cw_gc1[5, ..., np.newaxis] + (1 - cw_gc1[5, ..., np.newaxis])*(1-np.exp(-cw_gc1[12, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :]), axis = -1)			
    af_wool_pdiegc2 = np.nanmean(cw_gc2[5, ..., np.newaxis] + (1 - cw_gc2[5, ..., np.newaxis])*(1-np.exp(-cw_gc2[12, ..., np.newaxis] * age_m_pdic2m1[..., np.newaxis, :, :]), axis = -1)			
#^what needs to change to calc yaf below (currently the same as dam)?
    af_wool_y_piegc1 = np.nanmean(cw_gc1[5, ..., np.newaxis] + (1 - cw_gc1[5, ..., np.newaxis])*(1-np.exp(-cw_gc1[12, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :]), axis = -1)			
    dlf_eff_p = np.average(i_latitude / 40 * sin(2π * doy_m_pm1 / 365), axis = -1)			
    mr_age_pgc0 = np.nanmean(np.maximum(cm_gc0[4, ..., np.newaxis], np.exp(-cm_gc0[3, ..., np.newaxis] * age_m_pc0m1[..., np.newaxis, :, :])), axis = -1)			
    mr_age_pigc1 = np.nanmean(np.maximum(cm_gc1[4, ..., np.newaxis], np.exp(-cm_gc1[3, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :])), axis = -1)			
    mr_age_pdiegc2	= np.nanmean(np.maximum(cm_gc2[4, ..., np.newaxis], np.exp(-cm_gc2[3, ..., np.newaxis] * age_m_pdic2m1[..., np.newaxis, :, :])), axis = -1)			
#^again what needs to change to calc yaf below (currently the same as dam)?
    mr_age_y_piegc1	= np.nanmean(np.maximum(cm_gc1[4, ..., np.newaxis], np.exp(-cm_gc1[3, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :])), axis = -1)			
    rain_intake_pgc0 = np.average(np.maximum(0, 1 - c_rain_pm1[:, np.newaxis, np.newaxis, :] / ci_gc0[8, ..., np.newaxis]), axis = -1)			
    rain_intake_pgc1 = np.average(np.maximum(0, 1 - c_rain_pm1[:, np.newaxis, np.newaxis, :] / ci_gc1[8, ..., np.newaxis]), axis = -1)			
    rain_intake_pgc2 = np.average(np.maximum(0, 1 - c_rain_pm1[:, np.newaxis, np.newaxis, :] / ci_gc2[8, ..., np.newaxis]), axis = -1)			


    ############################
    ### Daily steps Dams       #
    ############################
    age_f_piec1m1 = age_f_piec1[…, np.newaxis] + axis_m1
    age_f_piec1m1[age_f_piec1m1 <= 0] = np.nan
    age_f_piec1m1[age_f_piec1m1 > cp_gc1[1, 0, :]] = np.nan		
    age_y_m_piec1m1 = age_y_piec1[…, np.newaxis] + axis_m1
    age_y_m_piec1m1[age_y_m_piec1m1 <= 0] = np.nan
    age_y_m_piec1m1[age_y_m_piec1m1 > age_wean_y_oiaec1[a_wean_o_piaec1]] = np.nan			
    age_y_adj_m_piaec1m1	= age_y_adj_piaec1[…, np.newaxis] + axis_m1
    age_y_adj_m_piaec1m1[age_y_adj_m_piaec1m1 <= 0] = np.nan			
    pimi_piaegc1m1	= age_y_adj_m_piaec1m1[…, np.newaxis, :, :] / ci_gc1[8, …, np.newaxis]			
    lmm_piegc1m1	= (age_y_m_piec1m1[…, np.newaxis, :, :] + cl_gc1[1, …, np.newaxis]) / cl_gc1[2, …, np.newaxis]			
    pi_age_y_piaeygc1	= np.nanmean(cy_ygc1[1,:, np.newaxis, np.newaxis, np.newaxis, ..., np.newaxis] * (pimi_piaegc1m1[:, np.newaxis, ...]) ** ci_gc1[9, ..., np.newaxis] * np.exp(ci_gc1[9, ..., np.newaxis] * (1 - pimi_piaegc1m1[:, np.newaxis, ...])), aixs = -1)			
    mp_age_y_pieygc1	= np.nanmean(cy_ygc1[0, :, np.newaxis, np.newaxis, ...] * lmm_piegc1m1 ** cl_gc1[3, ...] * np.exp(cl_gc1[3, ...]* (1 - lmm_piegc1m1)), aixs = -1)			
    mp2_age_y_pieygc1	= np.nanmean(a_noffspring_x.reshape(-1, 1, 1, 1, 1) * cl_gc1[6, ..., np.newaxis] * ( cl_gc1[12, ..., np.newaxis] + cl_gc1[13, ..., np.newaxis] * np.exp(-cl_gc1[14, ..., np.newaxis] * age_y_m_piec1m1)), aixs = -1)			
    crg_doy_pxgc1	= np.average(1 - cx_xgc1[1, :, ..., np.newaxis] * (1 - sin(2π * (doy_m_pm1[:, np.newaxis, np.newaxis, np.newaxis,:] + 10) / 365) * sin(i_latitude) / -0.57), axis = -1)			
    yfi_piegc1	= np.nanmean(1/(1 + np.exp(-ci_gc1[3, ..., np.newaxis] * (age_y_m_piec1m1[…, np.newaxis, :, :] - ci_gc1[4, …, np.newaxis]))), axis = -1)			






    ############################
    ### association arrays     # ^some of these may have to be built in other sections due to data required and when they are used
    ############################
    
    ##genotype
    i_include_c2 = pinp.sheep['i_include_c2']
    a_g_c2 = pinp.sheep['a_g_g2'][i_include_c2]
    a_c0_c1 = pinp.sheep['a_paternal_g0_g2'][i_include_c2] #which sires are mated to which dams.
    
    ##joining age in each period ^need to check these are working as desired once i get some input data
    # a_prevdam_o_pic1 = np.apply_along_axis(sfun.f_next_prev_association, 0, pinp.sheep['i_age_join_oic1'], age_start_pic1[1:-1,:,:], 1)
    a_nextdam_o_pic1=sfun.f_next_prev_joining( pinp.sheep['i_age_join_oic1'], age_start_pic1, 0)
    a_prevdam_o_pic1=sfun.f_next_prev_joining(  pinp.sheep['i_age_join_oic1'], age_end_pic1, 1)
    a_nextyaf_o_pic1=sfun.f_next_prev_joining( pinp.sheep['i_age_join_oic1'], age_start_pic1, 0)
    a_prevyaf_o_pic1=sfun.f_next_prev_joining(  pinp.sheep['i_age_join_oic1'], age_end_pic1, 1)
    
    #^most of these will become inputs
    ##genotype option of the input genotypes
    a_k2_g0[0] = pinp.sheep['i_genotype_b0']
    a_k2_g0[1] = pinp.sheep['i_genotype_m0']
    a_k2_g0[2] = pinp.sheep['i_genotype_t0	']		
    ##Animal type option of the input genotypes
    a_k1_g0[0] = pinp.sheep['i_animaltype_b0']
    a_k1_g0[1] = pinp.sheep['i_animaltype_m0']
    a_k1_g0[2] = pinp.sheep['i_animaltype_t0	']		
    ##Maternal genotype of the dams
    a_maternal_g0_g1
    a_paternal_g0_g1
    a_maternal_g1_g2
    a_paternal_g0_g2


    # maybe this will have to be done by date rather than age
    a_next_o_plc1 = np.apply_along_axis(sfun.f_find_index, 0, i_age_join_olc1,
                                        age_start_plc1, 0)
    a_prev_o_plc1 = np.apply_along_axis(sfun.f_find_index, 0, i_age_join_olc1,
                                        age_end_plc1, -1)



    ###########################
    ### non-loop calculations #
    ###########################
    '''Calculations for which the inputs do not depend on previous periods
    See spreadsheet: Group independent and Age,Date,Timing'''

    doy_p =
    lgf_eff_p =
    dlf_eff_p =
    dlf_wool_p =
    chill_p =
    kw =
    kc =
    birth_date_i =
    birth_date_jl =
    birth_date_jel =
    birth_date_kel =
    age_pi =
    age_pjl =
    age_pjel =
    age_pkel =
    age_f_pjel =
    age_f_pjel =
    pimi_pjel =
    ra_pjel =
    age_y_adj_pjel =
    af_wool_pi =
    af_wool_pjl =
    af_wool_pjel =
    af_wool_pkel =
    mm_pjel =
    d_cfw_ave_pi =
    d_cfw_ave_pjl =
    d_cfw_ave_pjel =
    d_cfw_ave_pkel =
    nw_max_pi =
    nw_max_pjxyl =
    nw_max_pkdwebl =

    ### _feed inputs
    sfun.feed_inputs function


    ##########################################
    ### Calc standard feed supply for periods#
    ##########################################
    '''flow chart 5'''
    
    ##########################################
    ### Initialise then loop through periods #
    ##########################################
    ## initialise the arrays for the first period #
    lw_ffcf = i.weaning_wt
    mw = 0.7 * lw_ffcf
    aw = 0.2 * lw_ffcf
    bw = 0.1 * lw_ffcf
    cfw = 0.6 #cfw at weaning
    fd = 19 #fd at weaning
    fl = 10 #fl at weaning
    #set all arrays that are assigned using += to 0.

    ## Loop through each week of the simulation (p) for ewes
    ## # number of periods is a fixed value so I'm thinking a 'for' loop
    for p in range(n_sim_periods):
        if p != 0:  # only carry this out with p<>0
            ### _conception
            cr_ojexyl[mask] += sfun.conception(lw_ffcf[p,...], srw_j)[mask]
            # with a mask to a
            nlb_ojewbl += cr_ojexyl#convert conception in _xy format to _wb
            ### _mortality
            mr[p,...] = sfun.mortality(rc[p-1,...])
            tem[p,...], dmr[p,...], lmr[p,...] = sfun.ewe_mortality()
            nlw_ojewbl = nlb_ojewbl &
            ### _start numbers & weight
            number[p,...] = sfun.transfers(number[p-1,...], sales
                            , ewe_mortality, cr, lamb_mortality, ....)  #function call or in global
            number[p] = (number[p-1] - sales[p-1]) * (1 - mortality) ....
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        ### feed supply loop
        # this loop is only required if a LW target is specified for the animals
        # if there is a target then the loop needs to continue until
        # the feed supply has converged on a value that generates a liveweight
        # change close to the target
        # The loop needs to execute at least once, then repeat if there
        # is a target and the result is not close enough to the target
        if this period (p) is a new feed variation period (f) or a new MIDAS feed period (n):
            then feed_supply_jxyl = feed_supply_pjxyl[p,...]
            otherwise use feedsupply from last period (which was optimised for the target)
        Feed supply loop start
            # the loop will be a bit tricky because the target is for an array of values
            # and some parts of the array may be within the tolerance but other parts are not.
            # To further complicate it the target will often be associated with
            # the weighted average of a slice of the array rather than an individual
            # element.
            foo, dmd, supp = sfun.feed_supply(feed_supply_jxyl, foo_std, dmd_std)
            #'
            pi_jexyl = sfun.p_intake(rc, srw, rel_size)
            ri_jexyl = sfun.r_intake(foo, dmd, supp)
            mei_jexyl = pi_jexyl - np.newaxis(e, supp_jxyl) * ri_jexyl * nv_jexyl + newaxis(supp_jxyl) * supp_md
            p_mei_pjexyl[p,...] = mei_jexyl
            mem = sfun.energy(....)
            mep, cw = sfun.pregnancy(....)
            mel = sfun.lactation(....)
            dcfw, new = sfun.wool_growth(....)
            ebg, pg = sfun.lw_change(mei, mem, mep, mel, mew, mecold, wmax, zf1, zf2)
            lwc = ebg * (1)
            if there is a target and abs(lwc-target) > eps:
                update feed_supply
                #      feed supply is a number between 0 and 3. We could use a binary
                #      type process to converge on the feed supply. But given that
                #      the feed supply was calculated in the previous period and
                #      it should be close then maybe a step process might be quicker.
                #      The main advantage of the binary approach is that each element
                #      of the array should converge at a similar rate, whereas maybe
                #      not with the step approach
                #      Open to ideas here.
            loop if feed_supply was changed
        lw_ffcf_jexyl = lw_ffcf_start_jexyl + lwc_jexyl * step
        lw_ffcf_max_jexyl = np.maximum(lw_ffcf_jexyl, lw_ffcf_max_jexyl)
        aw_jexyl
        mw_jexyl
        bw_jexyl
        ww_jexyl
        gw_jexyl
        fw_end_jexyl
        cfw_jexyl = cfw_start_jexyl + dcfw * step
        fl_jexyl
        fd_min_jexyl
        fd_jexyl
        ldr_end_jexyl
        lb_end_jexyl
        lw_jexyl = lw_ffcf_jexyl + cw_jexyl + cfw_Jexyl
        r_lw_jexyl[p,...] = lw_jexyl


    # repeat loop for rams & then for offspring
    # these don't require conception, pregnancy, lactation and ewe mortality
    for p in range(n_sim_periods):
        if p <>0:  # only carry this out with p<>0
            ## or pass lw_cfff_end and nw_end & srw and calculate z and rc
            mr[p,...] = sfun.mortality(rc[p-1,...])   # offspring
            mr[p,...] = sfun.mortality(rc[p-1,...])   # rams
            .... = sfun.numbers(....)                 #offspring
            .... = sfun.numbers(....)                 #rams
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        Feed supply Loop for offspring
            #` mei and rc are not defined
            mei[p,...] = sfun.intake(rc, c_ci_gy, )
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values
        Feed supply Loop for rams #Probably will never need to loop this
            #because not specifying a target for the rams
            mei[p,...] = sfun.intake(....)
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values

def parameters():
    """Parameter generation for the pyomo variables


    Returns
    -------
    dictionaries for pyomo
    """
parameters = np.zeros((len(output_required),len(activities0)), dtype = 'float64')
    # Loop through the number of variables
    for a in activites:
        ### create array masks  for the pyomo variable
        ''' For each pyomo variable create a mask that represents the animals
        The arrays can then be summed across the axes for that mask '''
        mask = sfun.create_mask(i_activity_definition)

        ### apply each mask to each simulation output
        #output_required is a list of the arrays that are required as parameters
        for n, o in enumerate(output_required):
            parameters[n,a] = np.sum(o[mask])

return parameters

''' Or to allow one function call per constraint this function could
generate the array and then multiple functions that just return the
required row of the array.'''