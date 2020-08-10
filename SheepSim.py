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
from scipy import stats
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

##can use this when building arrays that require new axis
na = np.newaxis
two_na = (np.newaxis,np.newaxis)
birth_date_jl = uinp.propertydata['ExcelName']
birth_date_jel = uinp.propertydata['ExcelName']
birth_date_kel = uinp.propertydata['ExcelName']

#^abvoe this line is not currently used
birth_date_g1 = pinp.sheep['ExcelName']   # Find the ExcelNames
i_sim_periods_year = 52  # ^uinp.n_sim_periods_year  now in structure dict
i_oldest_animal = 6.5    # ^uinp.i_oldest_animal
# ## Some one time data manipulation for the inputs just read
start_year = dt.datetime(2019, 5, 15)#np.min(birth_date_g1)
start_date = dt.datetime(2019, 5, 15)#np.min(birth_date_g1)
# ## might need to test and rebase the year for the other animal groups


### _define the periods ^need to decide where to put this code
n_sim_periods, date_start_p, date_end_p, p_index_p, step \
        = sim_periods(start_year, i_sim_periods_year, i_oldest_animal)
### _array dimensions




###################################
### initialise global arrays      #
###################################
'''only create arrays that are used in sim and post processing.
'''
## Instantiate the globals arrays
## # these store the output of simulation and the parameters for pyomo
## # see documentation for a description of each variable

##^this function can possibly be moved to sheep routines once steve is done.
def f_k2g(params_k2, y=0, var_pos=0, len_ax1=0, len_ax2=0, g2g = False, group = 'dams'):
    '''
    Parameters
    ----------
    params_k2 : array
        parameter array - input from excel.
    y : array
        sensitivity array for genetic merit.
    var_pos : int
        position of last axis when inserted into all axis.
    len_ax1 : int
        length of axis 1 - used to reshape input array into multi dimension array.
    len_ax2 : int, optional
        length of axis 1 - used to reshape input array into multi dimension array. The default is 0.
    g2g: boolean, optional
        this determines if the user only wants to convert from g to g (ie select which genotype options need to be represented for the necesary offs) this only happens if the inputs dont have k axis.
    group: 
        this is used to specify the sheep group that the g2g mask is being applied
    Returns
    -------
    param array for each genotype. Grouped by sheep group ie sire, offs, dams, yatf.
    If g2g is selected then only the conversion from all g? to relevant g? is done using the g?g3 mask

    '''
    #^line can be deleted once working.
    # params_k2=parameters['i_gfw_k2']
    # y=parameters['i_gfw_y']
    # var_pos=parameters['i_cx_pos']
    # len_ax1=parameters['i_cx_len']
    # len_ax2=parameters['i_cx_len2']

    ##these inputs are used for each param so they don't need to be passed into the function.
    a_k2_k0 = pinp.sheep['a_k2_k0']
    i_g3_inc = pinp.sheep['i_g3_inc']
    i_mul_g0_k0 = uinp.structure['i_mul_g0k0']   
    i_mul_g1_k0 = uinp.structure['i_mul_g1k0']   
    i_mul_g2_k0 = uinp.structure['i_mul_g2k0']  
    i_mul_g3_k0 = uinp.structure['i_mul_g3k0']  
    i_mask_g0g3 = uinp.structure['i_mask_g0g3']   
    i_mask_g1g3 = uinp.structure['i_mask_g1g3']  
    i_mask_g2g3 = uinp.structure['i_mask_g2g3'] 
    i_mask_g3g3 = uinp.structure['i_mask_g3g3'] 
   
    if g2g == False:
        ##convert params from k2 to k0
        params_k0 = params_k2[...,a_k2_k0]
        ##add y axis
        na=np.newaxis
        ###if y is not numpy ie was read in as an int because it was a single cell, it needs to be converted
        if type(y) == int:
            y = np.asarray([y])
        ###y is a 2d array howvever currently it only has one slice so it is read in as a 1d array. so i need to add second array
        if y.ndim == 1 and params_k0.ndim != 1:
            y=y[...,na]
        params_k0 = np.multiply(params_k0[...,na,:],  y[...,na]) #na here is to account for k2 axis
        ##reshape parameter from 2d input to multi dim array
        len_y = y.shape[-1]
        ###make tuple of shape depending on the number of axis in input
        if len_ax2>0:
            shape=(len_ax1,len_ax2,len_y,3)
            params_k0 = params_k0.reshape(shape)
        elif len_ax1 > 0:
            shape=(len_ax1,len_y,3)
            params_k0 = params_k0.reshape(shape)
        else:
            pass#don't need to reshpae
        ##get axis into correct position
        if var_pos != None or var_pos != 0:
            extra_axes = tuple(range((var_pos + 1), -2))
        else: extra_axes = ()
        allaxis_params__k0 = np.expand_dims(params_k0, axis = extra_axes)
        ##create mask g?k0
        mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
        mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
        mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
        mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
        ##create array with the proportion of each pure genotype required to have each actual genotype included in the analysis
        mul_sire_genotypes_g0k0 = i_mul_g0_k0[mask_sire_inc_g0]
        mul_dams_genotypes_g0k0 = i_mul_g1_k0[mask_dams_inc_g1]
        mul_yatf_genotypes_g0k0 = i_mul_g2_k0[mask_yatf_inc_g2]
        mul_offs_genotypes_g0k0 = i_mul_g3_k0[mask_offs_inc_g3]
        ##convert params from k0 to g. nansum required when the selected k0 info is not filled out ^may be an issue if params are missing and mixed breed sheep is selected because it wont catch the error
        param_sire=np.nansum(allaxis_params__k0[..., na, :] * mul_sire_genotypes_g0k0, axis = -1) 
        param_dams=np.nansum(allaxis_params__k0[..., na, :] * mul_dams_genotypes_g0k0, axis = -1)
        param_yatf=np.nansum(allaxis_params__k0[..., na, :] * mul_yatf_genotypes_g0k0, axis = -1)
        param_offs=np.nansum(allaxis_params__k0[..., na, :] * mul_offs_genotypes_g0k0, axis = -1)
        return param_sire, param_dams, param_yatf, param_offs
    ##if user only wants to convert from g2g then can skip the k stuff
    elif group == 'sire':
        ##create mask g?g
        mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
        return params_k2[...,mask_sire_inc_g0] 
    elif group == 'dams':
        ##create mask g?g
        mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
        return params_k2[...,mask_dams_inc_g1] 
    elif group == 'offs':
        ##create mask g?g
        mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
        return params_k2[...,mask_offs_inc_g3] 
    
        


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
    na=np.newaxis
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
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from k to g
    ###production params
    agedam_propn_da0e0b0xyg0, agedam_propn_da0e0b0xyg1, agedam_propn_da0e0b0xyg2, agedam_propn_da0e0b0xyg03 = f_k2g(uinp.parameters['i_agedam_propn_std_dk2'], uinp.parameters['i_agedam_propn_y'], uinp.parameters['i_agedam_propn_pos']) #yatf and off never used	
    aw_propn_yg0, aw_propn_yg1, aw_propn_yg2, aw_propn_yg3 = f_k2g(uinp.parameters['i_aw_propn'], uinp.parameters['i_aw_y']) 
    bw_propn_yg0, bw_propn_yg1, bw_propn_yg2, bw_propn_yg3 = f_k2g(uinp.parameters['i_bw_propn'], uinp.parameters['i_bw_y']) 
    btrt_yg0, btrt_yg1, btrt_yg2, btrt_yg3 = f_k2g(uinp.parameters['i_scan_std_k2'], uinp.parameters['i_scan_std_y']) 
    cfw_propn_yg0, cfw_propn_yg1, cfw_propn_yg2, cfw_propn_yg3 = f_k2g(uinp.parameters['i_cfw_propn_k2'], uinp.parameters['i_cfw_propn_y'])			
    scan_std_yg0, scan_std_yg1, scan_std_yg2, scan_std_yg3 = f_k2g(uinp.parameters['i_scan_std_k2'], uinp.parameters['i_scan_std_y']) 
    lss_std_yg0, lss_std_yg1, lss_std_yg2, lss_std_yg3 = f_k2g(uinp.parameters['i_lss_std_k2'], uinp.parameters['i_lss_std_y']) 
    lstr_std_yg0, lstr_std_yg1, lstr_std_yg2, lstr_std_yg3 = f_k2g(uinp.parameters['i_lstr_std_k2'], uinp.parameters['i_lstr_std_y']) 
    lstw_std_yg0, lstw_std_yg1, lstw_std_yg2, lstw_std_yg3 = f_k2g(uinp.parameters['i_lstw_std_k2'], uinp.parameters['i_lstw_std_y']) 
    mw_propn_yg0, mw_propn_yg1, mw_propn_yg2, mw_propn_yg3 = f_k2g(uinp.parameters['i_mw_propn'], uinp.parameters['i_mw_y']) 
    sfd_yg0, sfd_yg1, sfd_yg2, sfd_yg3 = f_k2g(uinp.parameters['i_sfd_k2'], uinp.parameters['i_sfd_y'])			
    sfw_yg0, sfw_yg1, sfw_yg2, sfw_yg3 = f_k2g(uinp.parameters['i_sfw_k2'], uinp.parameters['i_sfw_y'])			
    srw_yg0, srw_yg1, srw_yg2, srw_yg3 = f_k2g(uinp.parameters['i_srw_k2'], uinp.parameters['i_srw_y'])			
    
    ###sim params
    ca_sire, ca_dams, ca_yatf, ca_offs = f_k2g(uinp.parameters['i_ca_k2'], uinp.parameters['i_ca_y'], uinp.parameters['i_ca_pos'], uinp.parameters['i_ca_len'])			
    cb0_sire, cb0_dams, cb0_yatf, cb0_offs = f_k2g(uinp.parameters['i_cb0_k2'], uinp.parameters['i_cb0_y'], uinp.parameters['i_cb0_pos'], uinp.parameters['i_cb0_len'], uinp.parameters['i_cb0_len2'])			
    cc_sire, cc_dams, cc_yatf, cc_offs = f_k2g(uinp.parameters['i_cc_k2'], uinp.parameters['i_cc_y'], uinp.parameters['i_cc_pos'], uinp.parameters['i_cc_len'])			
    cd_sire, cd_dams, cd_yatf, cd_offs = f_k2g(uinp.parameters['i_cd_k2'], uinp.parameters['i_cd_y'], uinp.parameters['i_cd_pos'], uinp.parameters['i_cd_len'])			
    ce_sire, ce_dams, ce_yatf, ce_offs = f_k2g(uinp.parameters['i_ce_k2'], uinp.parameters['i_ce_y'], uinp.parameters['i_ce_pos'], uinp.parameters['i_ce_len'], uinp.parameters['i_ce_len2'])			
    cf_sire, cf_dams, cf_yatf, cf_offs = f_k2g(uinp.parameters['i_cf_k2'], uinp.parameters['i_cf_y'], uinp.parameters['i_cf_pos'], uinp.parameters['i_cf_len'])			
    cg_sire, cg_dams, cg_yatf, cg_offs = f_k2g(uinp.parameters['i_cg_k2'], uinp.parameters['i_cg_y'], uinp.parameters['i_cg_pos'], uinp.parameters['i_cg_len'])			
    ch_sire, ch_dams, ch_yatf, ch_offs = f_k2g(uinp.parameters['i_ch_k2'], uinp.parameters['i_ch_y'], uinp.parameters['i_ch_pos'], uinp.parameters['i_ch_len'])			
    ci_sire, ci_dams, ci_yatf, ci_offs = f_k2g(uinp.parameters['i_ci_k2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'], uinp.parameters['i_ci_len'])			
    ck_sire, ck_dams, ck_yatf, ck_offs = f_k2g(uinp.parameters['i_ck_k2'], uinp.parameters['i_ck_y'], uinp.parameters['i_ck_pos'], uinp.parameters['i_ck_len'])			
    cl0_sire, cl0_dams, cl0_yatf, cl0_offs = f_k2g(uinp.parameters['i_cl0_k2'], uinp.parameters['i_cl0_y'], uinp.parameters['i_cl0_pos'], uinp.parameters['i_cl0_len'], uinp.parameters['i_cl0_len2'])			
    cl1_sire, cl1_dams, cl1_yatf, cl1_offs = f_k2g(uinp.parameters['i_cl1_k2'], uinp.parameters['i_cl1_y'], uinp.parameters['i_cl1_pos'], uinp.parameters['i_cl1_len'], uinp.parameters['i_cl1_len2'])			
    cl_sire, cl_dams, cl_yatf, cl_offs = f_k2g(uinp.parameters['i_cl_k2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'], uinp.parameters['i_cl_len'])			
    cm_sire, cm_dams, cm_yatf, cm_offs = f_k2g(uinp.parameters['i_cm_k2'], uinp.parameters['i_cm_y'], uinp.parameters['i_cm_pos'], uinp.parameters['i_cm_len'])			
    cn_sire, cn_dams, cn_yatf, cn_offs = f_k2g(uinp.parameters['i_cn_k2'], uinp.parameters['i_cn_y'], uinp.parameters['i_cn_pos'], uinp.parameters['i_cn_len'])			
    cp_sire, cp_dams, cp_yatf, cp_offs = f_k2g(uinp.parameters['i_cp_k2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'], uinp.parameters['i_cp_len'])			
    cr_sire, cr_dams, cr_yatf, cr_offs = f_k2g(uinp.parameters['i_cr_k2'], uinp.parameters['i_cr_y'], uinp.parameters['i_cr_pos'], uinp.parameters['i_cr_len'])			
    crd_sire, crd_dams, crd_yatf, crd_offs = f_k2g(uinp.parameters['i_crd_k2'], uinp.parameters['i_crd_y'], uinp.parameters['i_crd_pos'], uinp.parameters['i_crd_len'])			
    cu0_sire, cu0_dams, cu0_yatf, cu0_offs = f_k2g(uinp.parameters['i_cu0_k2'], uinp.parameters['i_cu0_y'], uinp.parameters['i_cu0_pos'], uinp.parameters['i_cu0_len'])			
    cu1_sire, cu1_dams, cu1_yatf, cu1_offs = f_k2g(uinp.parameters['i_cu1_k2'], uinp.parameters['i_cu1_y'], uinp.parameters['i_cu1_pos'], uinp.parameters['i_cu1_len'], uinp.parameters['i_cu1_len2'])			
    cu2_sire, cu2_dams, cu2_yatf, cu2_offs = f_k2g(uinp.parameters['i_cu2_k2'], uinp.parameters['i_cu2_y'], uinp.parameters['i_cu2_pos'], uinp.parameters['i_cu2_len'], uinp.parameters['i_cu2_len2'])			
    cu3_sire, cu3_dams, cu3_yatf, cu3_offs = f_k2g(uinp.parameters['i_cu3_k2'], uinp.parameters['i_cu3_y'], uinp.parameters['i_cu3_pos'], uinp.parameters['i_cu3_len'], uinp.parameters['i_cu3_len2'])			
    cu4_sire, cu4_dams, cu4_yatf, cu4_offs = f_k2g(uinp.parameters['i_cu4_k2'], uinp.parameters['i_cu4_y'], uinp.parameters['i_cu4_pos'], uinp.parameters['i_cu4_len'], uinp.parameters['i_cu4_len2'])			
    cw_sire, cw_dams, cw_yatf, cw_offs = f_k2g(uinp.parameters['i_cw_k2'], uinp.parameters['i_cw_y'], uinp.parameters['i_cw_pos'], uinp.parameters['i_cw_len'])			
    cx_sire, cx_dams, cx_yatf, cx_offs = f_k2g(uinp.parameters['i_cx_k2'], uinp.parameters['i_cx_y'], uinp.parameters['i_cx_pos'], uinp.parameters['i_cx_len'], uinp.parameters['i_cx_len2'])			
    
    ##Convert the cl0 & cl1 to cb1
    cb1_sire = cl0_sire[uinp.structure['a_nfoet_b1']] + cl1_sire[uinp.structure['a_nyatf_b1']] 
    cb1_dams = cl0_dams[uinp.structure['a_nfoet_b1']] + cl1_dams[uinp.structure['a_nyatf_b1']] 
    cb1_yatf = cl0_yatf[uinp.structure['a_nfoet_b1']] + cl1_yatf[uinp.structure['a_nyatf_b1']] 
    cb1_offs = cl0_offs[uinp.structure['a_nfoet_b1']] + cl1_offs[uinp.structure['a_nyatf_b1']] 
    
    ##sfw, sfd and srw adjustments
    ###calc adjustments srw
    adja_srw_x_xyg0 = cx_sire[11, 0:1, ...]  #11 is the swt parameter, 0:1 is the sire gender slice (retaining the axis).
    adja_srw_x_xyg1 = cx_dams[11, 1:2, ...]
    adja_srw_x_xyg2 = cx_yatf[11, ...] #all gender slices
    adja_srw_x_xyg3 = cx_offs[11, ...] #all gender slices
    ###calc adjustments sfw
    adja_sfw_d_a0e0b0xyg0 = np.sum(ce_sire[0, ...] * agedam_propn_sire, axis = 0)
    adja_sfw_d_a0e0b0xyg1 = np.sum(ce_dams[0, ...] * agedam_propn_dams, axis = 0)
    adja_sfw_d_da0e0b0xyg2 = ce_yatf[0, ...]
    adja_sfw_d_da0e0b0xyg3 = ce_offs[0, ...]
    adja_sfw_b0_xyg0 = np.sum(cb0_sire[0, ...] * btrt_propn_g0, axis = 0)
    adja_sfw_b0_xyg1 = np.sum(cb0_dams[0, ...] * btrt_propn_g1, axis = 0)
    adja_sfw_b0_b0xyg2 = cb0_yatf[0, ...]
    adja_sfw_b0_b0xyg3 = cb0_offs[0, ...]
    ###calc adjustments sfd
    adja_sfd_d_a0e0b0xyg0 = np.sum(ce_sire[1, ...] * agedam_propn_sire, axis = 0)
    adja_sfd_d_a0e0b0xyg1 = np.sum(ce_dams[1, ...] * agedam_propn_dams, axis = 0)
    adja_sfd_d_da0e0b0xyg2 = ce_yatf[1, ...]
    adja_sfd_d_da0e0b0xyg3 = ce_offs[1, ...]
    adja_sfd_b0_xyg0 = np.sum(cb0_sire[1, ...] * btrt_propn_g0, axis = 0)
    adja_sfd_b0_xyg1 = np.sum(cb0_dams[1, ...] * btrt_propn_g1, axis = 0)
    adja_sfd_b0_b0xyg2 = cb0_yatf[1, ...]
    adja_sfd_b0_b0xyg3 = cb0_offs[1, ...]
    ###apply adjustments srw
    srw_xyg0 = srw_yg0 + adja_srw_x_xyg0
    srw_xyg1 = srw_yg1 + adja_srw_x_xyg1
    srw_xyg2 = srw_yg2 + adja_srw_x_xyg2
    srw_xyg3 = srw_yg3 + adja_srw_x_xyg3
    
    ###apply adjustments sfw
    sfw_a0e0b0xyg0 = sfw_yg0 + adja_sfw_d_a0e0b0xyg0 + adja_sfw_b0_xyg0
    sfw_a0e0b0xyg1 = sfw_yg1 + adja_sfw_d_a0e0b0xyg1 + adja_sfw_b0_xyg1
    sfw_da0e0b0xyg2 = sfw_yg2 + adja_sfw_d_da0e0b0xyg2 + adja_sfw_b0_b0xyg2
    sfw_da0e0b0xyg3 = sfw_yg3 + adja_sfw_d_da0e0b0xyg3 + adja_sfw_b0_b0xyg3
    ###apply adjustments sfd
    sfd_a0e0b0xyg0 = sfd_yg0 + adja_sfd_d_a0e0b0xyg0 + adja_sfd_b0_xyg0
    sfd_a0e0b0xyg1 = sfd_yg1 + adja_sfd_d_a0e0b0xyg1 + adja_sfd_b0_xyg1
    sfd_da0e0b0xyg2 = sfd_yg2 + adja_sfd_d_da0e0b0xyg2 + adja_sfd_b0_b0xyg2
    sfd_da0e0b0xyg3 = sfd_yg3 + adja_sfd_d_da0e0b0xyg3 + adja_sfd_b0_b0xyg3
    
    ####################
    #initial conditions#
    ####################
    ##turn to numpy (currently just an int but if a second option is added this won't be needed)
    if type(pinp.sheep['i_adjp_lw_initial_a']) == int:
        i_adjp_lw_initial_a = np.array([pinp.sheep['i_adjp_lw_initial_a']])
        i_adjp_cfw_initial_a = np.array([pinp.sheep['i_adjp_cfw_initial_a']])
        i_adjp_fd_initial_a = np.array([pinp.sheep['i_adjp_fd_initial_a']])
        i_adjp_fl_initial_a = np.array([pinp.sheep['i_adjp_fl_initial_a']])
    ##convert i_adjp to adjp - add necessary axes for 'a' and 'w'
    adjp_lw_initial_a0e0b0xyg = np.expand_dims(i_adjp_lw_initial_a, axis = tuple(range(1,-pinp.sheep['i_a_pos'])))
    adjp_lw_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_lw_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_lw_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_cfw_initial_a0e0b0xyg = np.expand_dims(i_adjp_cfw_initial_a, axis = tuple(range(1,-pinp.sheep['i_a_pos'])))
    adjp_cfw_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_cfw_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_cfw_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fd_initial_a0e0b0xyg = np.expand_dims(i_adjp_fd_initial_a, axis = tuple(range(1,-pinp.sheep['i_a_pos'])))
    adjp_fd_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fd_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fd_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fl_initial_a0e0b0xyg = np.expand_dims(i_adjp_fl_initial_a, axis = tuple(range(1,-pinp.sheep['i_a_pos'])))
    adjp_fl_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fl_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
    adjp_fl_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))

    
    ##convert variable from k2 to g (yatf is not used, only here because it is return from the function) then addjust by initial lw pattern
    lw_initial_yg0, lw_initial_yg1, lw_initial_yatf, lw_initial_yg3 = f_k2g(uinp.parameters['i_lw_initial_k2'], uinp.parameters['i_lw_initial_y'])			
    lw_initial_wzida0e0b0xyg0 = lw_initial_yg0 * (1 + adjp_lw_initial_wzida0e0b0xyg0)
    lw_initial_wzida0e0b0xyg1 = lw_initial_yg1 * (1 + adjp_lw_initial_wzida0e0b0xyg1)
    lw_initial_wzida0e0b0xyg3 = lw_initial_yg3 * (1 + adjp_lw_initial_wzida0e0b0xyg3)
    cfw_initial_yg0, cfw_initial_yg1, cfw_initial_yatf, cfw_initial_yg3 = f_k2g(uinp.parameters['i_cfw_initial_k2'], uinp.parameters['i_cfw_initial_y'])			
    cfw_initial_wzida0e0b0xyg0 = cfw_initial_yg0 * (1 + adjp_cfw_initial_wzida0e0b0xyg0)
    cfw_initial_wzida0e0b0xyg1 = cfw_initial_yg1 * (1 + adjp_cfw_initial_wzida0e0b0xyg1)
    cfw_initial_wzida0e0b0xyg3 = cfw_initial_yg3 * (1 + adjp_cfw_initial_wzida0e0b0xyg3)
    fd_initial_yg0, fd_initial_yg1, fd_initial_yatf, fd_initial_yg3 = f_k2g(uinp.parameters['i_fd_initial_k2'], uinp.parameters['i_fd_initial_y'])			
    fd_initial_wzida0e0b0xyg0 = fd_initial_yg0 * (1 + adjp_fd_initial_wzida0e0b0xyg0)
    fd_initial_wzida0e0b0xyg1 = fd_initial_yg1 * (1 + adjp_fd_initial_wzida0e0b0xyg1)
    fd_initial_wzida0e0b0xyg3 = fd_initial_yg3 * (1 + adjp_fd_initial_wzida0e0b0xyg3)
    fl_initial_yg0, fl_initial_yg1, fl_initial_yatf, fl_initial_yg3 = f_k2g(uinp.parameters['i_fl_initial_k2'], uinp.parameters['i_fl_initial_y'])			
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
    adja_lw_initial_x_xyg0 = cx_sire[3, 0:1, ...] #3 is the weaning wt parameter, 0:1 is the sire gender slice (retaining the axis).
    adja_lw_initial_x_xyg1 = cx_dams[3, 1:2, ...] 
    adja_lw_initial_x_xyg3 = cx_offs[3, ...] 
    adja_cfw_initial_x_wzida0e0b0xyg0 = cx_sire[0, 0:1, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0
    adja_cfw_initial_x_wzida0e0b0xyg1 = cx_dams[0, 1:2, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
    adja_cfw_initial_x_wzida0e0b0xyg3 = cx_offs[0, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3
    adja_fd_initial_x_xyg0 = cx_sire[1, 0:1, ...] 
    adja_fd_initial_x_xyg1 = cx_dams[1, 1:2, ...] 
    adja_fd_initial_x_xyg3 = cx_offs[1, ...] 
    adja_fl_initial_x_wzida0e0b0xyg0 = cx_sire[0, 0:1, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 
    adja_fl_initial_x_wzida0e0b0xyg1 = cx_dams[0, 1:2, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
    adja_fl_initial_x_wzida0e0b0xyg3 = cx_offs[0, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3
    ##adjust for dam age. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr)
    adja_lw_initial_d_a0e0b0xyg0 = np.sum(ce_sire[3, ...] * agedam_propn_sire, axis=0) #d axis lost when summing
    adja_lw_initial_d_a0e0b0xyg1 = np.sum(ce_dams[3, ...] * agedam_propn_dams, axis=0) 
    adja_lw_initial_d_da0e0b0xyg3 = ce_offs[3, ...] 
    adja_cfw_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[0, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_sire, axis=uinp.parameters['i_agedam_propn_pos'], keepdims=True) #d axis lost when summing
    adja_cfw_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[0, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_dams, axis=uinp.parameters['i_agedam_propn_pos'], keepdims=True) 
    adja_cfw_initial_d_wzida0e0b0xyg3 = ce_offs[0, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3
    adja_fd_initial_d_a0e0b0xyg0 = np.sum(ce_sire[1, ...] * agedam_propn_sire, axis=0) #d axis lost when summing
    adja_fd_initial_d_a0e0b0xyg1 = np.sum(ce_dams[1, ...] * agedam_propn_dams, axis=0) 
    adja_fd_initial_d_da0e0b0xyg3 = ce_offs[1, ...]  
    adja_fl_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[0, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_sire, axis=uinp.parameters['i_agedam_propn_pos'], keepdims=True) #d axis lost when summing
    adja_fl_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[0, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_dams, axis=uinp.parameters['i_agedam_propn_pos'], keepdims=True) 
    adja_fl_initial_d_wzida0e0b0xyg3 = ce_offs[0, ...] * fl_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3 
    ##adjust for btrt. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr) 
    adja_lw_initial_b0_xyg0 = np.sum(cb0_sire[3, ...] * btrt_yg0, axis=0) #d axis lost when summing
    adja_lw_initial_b0_xyg1 = np.sum(cb0_dams[3, ...] * btrt_yg1, axis=0) 
    adja_lw_initial_b0_b0xyg3 = cb0_offs[3, ...] 
    adja_cfw_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[0, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_yg0, axis=uinp.parameters['i_cb0_pos'], keepdims=True) #d axis lost when summing
    adja_cfw_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[0, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_yg1, axis=uinp.parameters['i_cb0_pos'], keepdims=True) #d axis lost when summing
    adja_cfw_initial_b0_wzida0e0b0xyg3 = cb0_offs[0, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3
    adja_fd_initial_b0_xyg0 = np.sum(cb0_sire[1, ...] * btrt_yg0, axis=0) #d axis lost when summing
    adja_fd_initial_b0_xyg1 = np.sum(cb0_dams[1, ...] * btrt_yg1, axis=0) 
    adja_fd_initial_b0_b0xyg3 = cb0_offs[1, ...] 
    adja_fl_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[0, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_yg0, axis=uinp.parameters['i_cb0_pos'], keepdims=True) #d axis lost when summing
    adja_fl_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[0, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_yg1, axis=uinp.parameters['i_cb0_pos'], keepdims=True) #d axis lost when summing
    adja_fl_initial_b0_wzida0e0b0xyg3 = cb0_offs[0, ...] * fl_initial_wzida0e0b0xyg3 / sfw_a0e0b0xyg3
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
    ##calc aw, bw and mw (adipose, bone and muscel weight)
    aw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * aw_propn_sire
    aw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * aw_propn_dams
    aw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * aw_propn_offs
    bw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * bw_propn_sire
    bw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * bw_propn_dams
    bw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * bw_propn_offs
    mw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * mw_propn_sire
    mw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * mw_propn_dams
    mw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * mw_propn_offs

    ############################
    #  management inputs       # 
    ############################
    ##due to the different axis positions there is inconsistency with the adding of singelton axis therefore i have done it manually rather than in the function (it is a little too complex for the function without adding lots more detail) 
    ##the function call only applys the g masks ie determines which genotypes need to be represented for the given offs
    ##weaning age
    age_wean_ag1 = f_k2g(pinp.sheep['i_age_wean_ag1'], g2g=True, group='dams')
    age_wean_a0e0b0xyg1 = np.expand_dims(age_wean_ag1, axis = tuple(range(-(pinp.sheep['i_a_pos']+2)))) #^will need to alter this line slightly when a axis is included in inputs: tuple(range(1,-(pinp.sheep['i_a_pos']+1)))
    ##lambing date
    date_lamb_iog1 = pinp.sheep['i_date_lamb_oig1'].reshape(pinp.sheep['i_lamb_len'], pinp.sheep['i_lamb_len2'],  pinp.sheep['i_date_lamb_oig1'].shape[-1])
    date_lamb_oig1 = np.swapaxes(date_lamb_iog1,0,1)
    date_lamb_oig1 = f_k2g(date_lamb_oig1, g2g=True, group='dams')    
    date_lamb_oida0e0b0xyg1 = np.expand_dims(date_lamb_oig1, axis = tuple(range(2,-pinp.sheep['i_i_pos'])))
    ##Shearing date
    ###sire
    date_shear_isxg0 = pinp.sheep['i_date_shear_sixg0'].reshape(pinp.sheep['i_shear_len'], pinp.sheep['i_shear_len2'], pinp.sheep['i_shear_len3'],  pinp.sheep['i_date_shear_sixg0'].shape[-1])
    date_shear_sixg0 = np.swapaxes(date_shear_isxg0, 0, 1)
    date_shear_sixg0 = f_k2g(date_shear_sixg0, g2g=True, group='sire')    
    date_shear_sida0e0b0xyg0 = np.expand_dims(date_shear_sixg0[...,na,:], axis = tuple(range(2,-(pinp.sheep['i_i_pos']+2))))  #na it to add the y axis
    ###dam
    date_shear_isxg1 = pinp.sheep['i_date_shear_sixg1'].reshape(pinp.sheep['i_shear_len'], pinp.sheep['i_shear_len2'], pinp.sheep['i_shear_len3'],  pinp.sheep['i_date_shear_sixg1'].shape[-1])
    date_shear_sixg1 = np.swapaxes(date_shear_isxg1, 0, 1)
    date_shear_sixg1 = f_k2g(date_shear_sixg1, g2g=True, group='dams')    
    date_shear_sida0e0b0xyg1 = np.expand_dims(date_shear_sixg1[...,na,:], axis = tuple(range(2,-(pinp.sheep['i_i_pos']+2))))  #na it to add the y axis
    ###off
    date_shear_isxg3 = pinp.sheep['i_date_shear_sixg3'].reshape(pinp.sheep['i_shear_len'], pinp.sheep['i_shear_len2'], pinp.sheep['i_shear_len3'],  pinp.sheep['i_date_shear_sixg3'].shape[-1])
    date_shear_sixg3 = np.swapaxes(date_shear_isxg3, 0, 1)
    date_shear_sixg3 = f_k2g(date_shear_sixg3, g2g=True, group='offs')    
    date_shear_sida0e0b0xyg3 = np.expand_dims(date_shear_sixg3[...,na,:], axis = tuple(range(2,-(pinp.sheep['i_i_pos']+2))))  #na it to add the y axis
    ##join
    join_days_ig1 = f_k2g(pinp.sheep['i_join_days_ig1'], g2g=True, group='dams')    
    join_days_ida0e0b0xyg1 = np.expand_dims(join_days_ig1, axis = tuple(range(1,-(pinp.sheep['i_i_pos']+1)))) 
    ##lactation
    lal_og1 = f_k2g(pinp.sheep['i_lal_og1'], g2g=True, group='dams')    
    ##scanning
    scan_og1 = f_k2g(pinp.sheep['i_scan_og1'], g2g=True, group='dams')    


    ############################
    ### management calculations#
    ############################
    ##joining opp
    date_joined_oida0e0b0xyg1 = (date_lamb_oida0e0b0xyg1) - cp_dams[1,...,0:1,:]* dt.timedelta(days=1) #take slice 0 from y axis because cp1 is not affected by genetic merit
    date_joined_oida0e0b0xyg1.astype('datetime64[D]')
    
    ############################
    ### associations           #
    ############################
    a_prevdam_o_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_joined_oida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
    a_nextdam_o_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_joined_oida0e0b0xyg1.astype('datetime64[D]'), date_start_p, 0)
    a_prevyatf_o_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_lamb_oida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
    a_nextyatf_o_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_lamb_oida0e0b0xyg1.astype('datetime64[D]'), date_start_p, 0)
    ##shearing opp
    a_prev_s_pida0e0b0xyg0 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg0.astype('datetime64[D]'), date_end_p, 1)
    a_next_s_pida0e0b0xyg0 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg0.astype('datetime64[D]'), date_start_p, 0)
    a_prev_s_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
    a_next_s_pida0e0b0xyg1 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg1.astype('datetime64[D]'), date_start_p, 0)
    a_prev_s_pida0e0b0xyg3 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg3.astype('datetime64[D]'), date_end_p, 1)
    a_next_s_pida0e0b0xyg3 = np.apply_along_axis(f_next_prev_association, 0, date_shear_sida0e0b0xyg3.astype('datetime64[D]'), date_start_p, 0)


    
    
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
    age_y_start_piec1	= np.maximum(0, age_start_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevyatf_o_pic1[:, np.newaxis, ...]])			
    age_y_end_piec1	= np.maximum(0, age_end_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevyatf_o_pic1[:, np.newaxis, ...]])			
    age_y_piec1	=(age_y_start_piec1 + age_y_end_piec1+ 1) / 2
#^needs weaning assosiation array			
    age_y_adj_piaec1	= np.maximum(age_y_piec1[:, np.newaxis, ...] , age_wean_y_oiaec1[a_wean_o_piaec1] + (age_y_piec1[:, np.newaxis, ...] - age_wean_y_oiaec1[a_wean_o_piaec1]) * ci_gc1[21, ...]			
    lact_nut_effect_piec1[age_y_piec1 > cl_gc1[16, ...] * cl_gc1[2, ...]] = True			
    gest_propn_piec1	= np.maximum(0 , np.minimum(days_period_pic1, age_end_pic1[:, np.newaxis, ...] - age_mated_oiec1[a_prevdam_o_pic1], age_lamb_oiec1[a_prevdam_o_pic1] - age_start_pic1[:, np.newaxis, ...])) / days_period_pic1			
    lact_propn_piaec1	= np.maximum(0 , np.minimum(days_period_pic1, (age_end_pic1[:, np.newaxis, ...] - age_lamb_oiec1[a_prevdam_o_pic1])[:, np.newaxis, ...], (age_wean_oiaec1[a_prevyatf_o_pic1] - age_start_pic1[:, np.newaxis, ...])[:, np.newaxis, ...])) / days_period_pic1			
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
#^what needs to change to calc yatf below (currently the same as dam)?
    af_wool_y_piegc1 = np.nanmean(cw_gc1[5, ..., np.newaxis] + (1 - cw_gc1[5, ..., np.newaxis])*(1-np.exp(-cw_gc1[12, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :]), axis = -1)			
    dlf_eff_p = np.average(i_latitude / 40 * sin(2π * doy_m_pm1 / 365), axis = -1)			
    mr_age_pgc0 = np.nanmean(np.maximum(cm_gc0[4, ..., np.newaxis], np.exp(-cm_gc0[3, ..., np.newaxis] * age_m_pc0m1[..., np.newaxis, :, :])), axis = -1)			
    mr_age_pigc1 = np.nanmean(np.maximum(cm_gc1[4, ..., np.newaxis], np.exp(-cm_gc1[3, ..., np.newaxis] * age_m_pic1m1[..., np.newaxis, :, :])), axis = -1)			
    mr_age_pdiegc2	= np.nanmean(np.maximum(cm_gc2[4, ..., np.newaxis], np.exp(-cm_gc2[3, ..., np.newaxis] * age_m_pdic2m1[..., np.newaxis, :, :])), axis = -1)			
#^again what needs to change to calc yatf below (currently the same as dam)?
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
    a_nextyatf_o_pic1=sfun.f_next_prev_joining( pinp.sheep['i_age_join_oic1'], age_start_pic1, 0)
    a_prevyatf_o_pic1=sfun.f_next_prev_joining(  pinp.sheep['i_age_join_oic1'], age_end_pic1, 1)
    
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

sim = np.arange(dt.datetime(2020,1,1), dt.datetime(2028,1,1), dt.timedelta(days=7)).astype(dt.datetime)
    # maybe this will have to be done by date rather than age
    a_next_o_plc1 = np.apply_along_axis(sfun.f_find_index, 0, i_age_join_olc1,
                                        age_start_plc1, 0)
    a_prev_o_plc1 = np.apply_along_axis(sfun.f_find_index, 0, i_age_join_olc1,
                                        age_end_plc1, -1)



    ##set specific values related to reproduction and yat
    do some shit
    ##if offspring, create variables from yatf
    do offspring shit
    ##set other variables, including sim parameters
    
    ##calculate numpy arrays that are specific to sheep groulps but mot dependent on previous period values
    
    ##calc std feed supply for periods
    
    
    
    

##^this function can go in the function module at some point.
f_feedsupply_adjust(attempts,feedsupply,itn):
    ##create empty array to put new feedsuply into
    feedsupply = np.zeros_like(feedsupply)
    ##which feedsupplies can be calculated using binary method
    binary_mask = np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1) < 0 
    ##calc new feedsupply binary. Only adds the binary result to slices that have a negitive and a positive value (done using the mask created above)
    feedsupply[binary_mask] = (np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1))[binary_mask]
    ##calc feedsupply using interpolation
    ###first determine the slope, slope is always positive ie as feedsupply increases error increase because error = lwc - target and more feed means hihger lwc.
    if itn==0:
        slope=i_std_slope
    else:
        ####linregress only works on 1d array and cant use apply_over_axis because needs x and y. maybe there is a beter way but i looked for a while and found nothing
        slope=np.empty_like(feedsupply)
        feedsupply_all_itn = attempts[...,1]
        error_all_itn = attempts[...,1]
        for i in np.ndindex(error.shape[:-1]): #not exactly sure how this is working but it is creating tupple of each combo of slices in each axis.
            x= feedsupply_all_itn[i] #indexing with tupple works correctly if we are interested in the last axis otherwise it doesn't work properly for some reason.ie t[(0,0)] == t[0,0,:] but t[:,(0,0)] != t[:,0,0]
            y= error_all_itn[i]
            slope[i] = stats.linregress(x,y)
    ####new feedsupply = minerror / slope. It is assumed that the most recent itn has the most accurate feedsupply
    feedsupply[~binary_mask] = ((2 * attempts[...,-1,1]) / slope)[~binary_mask] # x2 to overshoot then switch to binary.
    return feedsupply
    


    ######################
    ### sim engine       #
    ######################

    ## initialise the arrays for the first period 
    lw_ffcf = i_weaning_wt
    mw = 0.7 * lw_ffcf
    aw = 0.2 * lw_ffcf
    bw = 0.1 * lw_ffcf
    cfw = 0.6 #cfw at weaning
    fd = 19 #fd at weaning
    fl = 10 #fl at weaning
    ##set all arrays that are assigned using += to 0.

    ## Loop through each week of the simulation (p) for ewes
    for p in range(n_sim_periods):
        if p != 0:  # only carry this out with p<>0
            ##set start values
            variable_start = variable_end
            ###check if the previous period was shearing for any of the sheep
            if np.any(prev period os shearing):
                ####reset all wool parameters ^i dont get this. what if not all groups were shorn?
            ###check if previous period was mating or lambing
            if np.any(previous period mating or lambing):
                ####calc weight transfers, calc n transfers
                
                ####update weights and numbers
            ###check if period is pre joining FVP
            if np.any(period is prejoining fvp):
                ####weights and production
                weights & prodn[not mated, in utero] = weighted average
                ####reset animal numbers
                NM,IU[-1,-1] = 0
                NM,IU[0:-1,0:-1] = 0
                ####reset birthweight
                bw = 0
                ####reset reproduction params
                ldr = 1
                lb = 1
            ###check if period is new FVP
            if np.any(period is new FVP): #^not sure why there is the np.any???
                ####set all numbers a weight values to the prime
                
            ###check if period is a new season
            if np.any(period is a new season):
                ####set patterns for each seaon type to the same starting point
                           
            ##update lw target
          
        ##calculate dependent start values
                
        ## conception, mortality and numbers
        ### base mortality
        mr[p,...] = sfun.mortality_csiro(rc[p-1,...])
        mr[p,...] = sfun.mortality_mu(rc[p-1,...])
        ### weaner mortality
        
        ###calc preg tox losses if less than 6wks to lambing.
        if date <= lambing - 42:
            mr[] += f_preg_tox_cs
            mr[] += f_preg_tox_mu
        ###if period is lambing calc dystocia losses
        if period_date == lambing:
            mr += f_dystocia_cs
            mr += f_dystocia_mu
        ###if previous period was lamning calc ewe mortality
        if period_date == lambing+7:
            mry += f_mortality_ewe_cs
            mry += f_mortality_ewe_mu
        ###if previous period was mating calc conception and transfers
        if period_date == mating+7:
            cr_ojexyl[mask] += sfun.conception(lw_ffcf[p,...], srw_j)[mask]
            # with a mask to a
            nlb_ojewbl += cr_ojexyl#convert conception in _xy format to _wb
        ###calc numbers after mortality and repro
        number[p,...] = sfun.transfers(number[p-1,...], sales
                        , ewe_mortality, cr, lamb_mortality, ....)  
        number[p] = (number[p-1] - sales[p-1]) * (1 - mortality) ....
        ###equation system loop ^dont know this enough to build it yet
        
        
        ##feed supply loop
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
        ##thought about making this a function but that is more difficult to debug so i just use a break if there is no target/need for a loop
        ##adjust feed supply
        ###initial info ^this will need to be hooked up with correct inputs, if they are the same for each period they donn't need to be initilised below
        target_lwc = 
        epsilon = 
        n_max_itn =
        feedsupply = 
        attempts = np.zeros(,n_max_itn,2) #^need to add the dimensions of lwc at the beginning
        for itn in range(n_max_itn):
            ###calc all functions 
            foo, dmd, supp = sfun.feed_supply(feed_supply_jxyl, foo_std, dmd_std)
            ####potential intake
            pi_jexyl = sfun.p_intake(rc, srw, rel_size)
            ####intake
            ri_jexyl = sfun.r_intake(foo, dmd, supp)
            ####energy
            mei_jexyl = pi_jexyl - np.newaxis(e, supp_jxyl) * ri_jexyl * nv_jexyl + newaxis(supp_jxyl) * supp_md
            p_mei_pjexyl[p,...] = mei_jexyl
            mem = sfun.energy(....)
            ####foetal growth
            mep, cw = sfun.pregnancy(....)
            ####energy milk production
            mel = sfun.lactation(....)
            ####energy wool production
            dcfw, new = sfun.wool_growth(....)
            ####energy to offset chilling
            ####calc lwc
            ebg, pg = sfun.lw_change(mei, mem, mep, mel, mew, mecold, wmax, zf1, zf2)
            lwc = ebg * (1)
            ###if there is a target then adjust feedsuply, if not break out of feedsuply loop
            if not target:
                break
            ###calc error
            error = lwc - target
            ###store in attempts array
            attempts[...,itn,0] = feedsupply
            attempts[...,itn,1] = error
            ###is error within tolerance
            if np.all(np.abs(error) <= epsilon):
                break
            ###max attempts reached
            elif itn == n_max_itn-1:
                ####select best feed supply option
                feedsupply = attempts[...,attempts[...,1]==np.nanmin(np.abs(attempts[...,1]),axis=-1),0] #create boolean index using error array then index feedsupply array
                break
            feedsupply = f_feedsupply_adjust(attempts,feedsupply,itn)
        ##emmisions
        
        ##end values
        

   
    















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