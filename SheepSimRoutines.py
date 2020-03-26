# -*- coding: utf-8 -*-
"""
All the calculation function for the simulation
Created on Sat Feb 29 08:05:09 2020

@author: John
"""

import numpy as np
import pandas as pd

def SIG(x,a,b):
    ''' Sig function CSIRO equation 124'''
    return

def RAMP(x,a,b):
    ''' RAMP function CSIRO equation 125a'''
    return


def genotype(excelinputs, a_c_g0, a_maternal_g0_g1, a_paternal_g0_g1
             , a_maternal_g1_g2, a_paternal_g0_g2):
    '''


    Returns
    -------
    outputs coloured green, yellow, orange and blue in the genotype function
    Arrays:
        c_srw_gw
        c_sfw_g
        c_cn_g
        c_ci_gy
        c_cr_g
        c_ck_g
        c_cm_g
        c_cp_gx
        c_cl_gy
        c_cw_g
        c_cc_g
        c_cg_g
        c_ch_g
        c_cf_gx
        c_cd_g

    '''
    ## take the genotype inputs and create the parameters that define the genotypes
    # Map genotype options to i_<arrays>_c: SRW, SFW, cw6 parameters
    #    where i_arrays_c: i_ means input arrays, <arrays> is the name & _c is the index
    i_srw_c = exceldata['<rangename>']
    i_gfw_c = exceldata['<rangename>']
    i_fd_c = exceldata['<rangename>']

    # Map parameter inputs to i_<arrays>_a: cn, cl, ck, cm, cp, cl, cw, cc, cg, ch, cf & cd
    i_cn_a = exceldata['<rangename>']
    i_ci_ay = exceldata['<rangename>']
    #inputs don't have an y dimension but the returned result does
    i_cr_a = exceldata['<rangename>']
    i_ck_a = exceldata['<rangename>']
    i_cm_a = exceldata['<rangename>']
    i_cp_ax = exceldata['<rangename>']
    #inputs don't have an x dimension but the returned result does
    i_cl_ay = exceldata['<rangename>']
    #inputs don't have an y dimension but the returned result does
    i_cw_a = exceldata['<rangename>']
    # i_cw_a[6] is a special case. With values being read into with  i_cw6_c
    # because it varies by genotype rather than just animal type
    i_cc_a = exceldata['<rangename>']
    i_cg_a = exceldata['<rangename>']
    i_ch_a = exceldata['<rangename>']
    i_cf_ax = exceldata['<rangename>']
    #inputs don't have an x dimension but the returned result does
    i_cd_a = exceldata['<rangename>']

    ### convert to _g arrays
    #from input data by csiro animal type and genotype option (c)
    # ram genotypes (B,M,T) are selected from the list of genotype options _c)
    g0 axis array = c axis Array[a_c_g0]
    # ewe genotype (BB,BM) are created by a combination of the 3 ram genotypes
    g1 axis array = ( g0 axis array[a_maternal_g0_g1]
                     +g0 axis array[a_paternal_g0_g1]) / 2
    # the offspring genotype (BBB, BBM, BBT, BMT) are created by a combination of the 2 ewe genotypes and the 3 ram genotypes
    g2 axis array = ( g1 axis array[a_maternal_g1_g2]
                     +g0 axis array[a_paternal_g0_g2]) / 2
    # create the full _g arrays (B, M, T, BM, BT, BMT)
    array_g[0:2] = array_g0  # the 3 ram genotypes
    array_g[3:4] = array_g2[1:3]  #the offspring genotypes excluding the first (B) because it was part of the ram genotypes

    return c_srw_gw, c_sfw_g, c_cn_g, c_ci_gy, c_cr_g, c_ck_g, c_cm_g, \
           c_cp_gx, c_cl_gy, c_cw_g, c_cc_g, c_cg_g, c_ch_g, c_cf_gx, c_cd_g

def sim_periods(start_year, periods_per_year, oldest_animal):
    '''Define the dates for the simulation periods.
    Starts on 1 Jan of the year with the earliest birthdate.

    Parameters:
    start_year = int: year to start simulation. Derived from the birth dates
    periods_per_year = int:
    oldest_animal = float: age of the oldest animal to be simulated (yrs)

    Returns:
    n_sim_periods
    array of period dates (1D periods)
    index of the periods (for pyomo)
    '''
    n_sim_periods = int(oldest_animal * periods_per_year)
    start_date = np.datetime64(start_year+'01-01','D')   #^ want this to return 1 Jan YYYY, but start year is a date itself
    step = pd.to_timedelta(365.25 / periods_per_year,'D')
    finish_date = start_date + step * n_sim_periods
    period_date_p = np.arange(start_date, finish_date, step, dtype='datetime64[D]')
    index_p = np.arange(n_sim_periods)
    return n_sim_periods, period_date_p, index_p, step

def condition_score(ffcflw, normal_weight, cs_propn = 0.19):
    ''' Estimate CS from LW. Works with scalars or arrays - provided they are broadcastable into ffcflw.

   ffcflw: (kg) Fleece free, conceptus free liveweight. normal_weight: (kg). cs_propn: (0.19) change in LW
   associated with 1 CS as a proportion of normal_weight.

   Returns: condition score - float
   '''
    return 3 + (ffcflw - normal_weight)/(cs_propn * normal_weight)

def feed_inputs():
    return feedsupply_pi, feedsupply_pjxyl, feedsupply_pkdwbl, foo_std_pr, dmd_std_pr


def conception(ffcflw, srw):
    #equations 122 to 124
    #nlm
    #increment number of rams required for the ewes that are being joined this period
#    n_rams_il = np.sum(n_pjexyl[this period is in the window for:
#                                   joining (in any lambing occurence)
#                                   and it is the first cycle (e=0)
#                                these ewes are mated to ram group i], axis(2,3,4))  \
#                * ram joining percentage_oj
    return nlm

def mortality_csiro(age_days, rc, ebg, nwg):
    #equation 125
    return mr

def ewe_mortality_csiro():
    #equations 126 to 129
    return mrt_ojexyl, mrd_ojexyl, mrl_ojewbl

def mortality_mu(age_days, rc, ebg, nwg):
    # these equations are not documented
    return mr

def ewe_mortality_mu():
    # these equations are not documented
    return mrt_ojexyl, mrd_ojexyl, mrl_ojewbl

def feed_supply(feedsupply, foo_std, dmd_std):
    # calculate the feed offered to the sheep from the feed supply value
    # this is not CSIRO equations. Related to inputs
    return foo, dmd, supp

def intake(feed_supply, srw, rel_size, rc, ):
    #do potential intake calculations
    #equations 2 to 10 & 72
    return mei

def energy(dmd, ):
    #equations 33 to 45
    km =
    return mem

def foetus():
    #equations 57 to 65
    return d_lw_f_jexl, cw_jexl, mec_jexl

def milk():
    # equations 66 to 75
    return ldr_jexyl, lb_jexyl, mel_jexyl

def fibre():
    #equations 77 to 86
    return d_cfw_wolag, d_cfw, mew, d_fd, d_fl

def chill():  #not for version 1
    # equations 88 to 100
    return me_cold, kg

def lwc():
    #equations 101 to 116
    # note not doing the protein component other than Eqn 105
    return ebg, pg

def wool_value(cfw, fl, fd, fd_min, wool_prices):
    # these equations are not documented
    return wool_value

def sale_value(lw, cs):
    #these equations are not documented
    return sale_value

def emissions_nggi():  #do this in v2
    return 0

def emissions_blaxter():  #do this in v2
    return 0

