# -*- coding: utf-8 -*-
"""
All the calculation function for the simulation
Created on Sat Feb 29 08:05:09 2020

@author: John
"""

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

def f_sig(x,a,b):
    ''' Sig function CSIRO equation 124 ^the equation below is the sig function from sheepexplorer'''
    return  1/(1+np.exp(-((2*(np.log(0.95) - np.log(0.05))/(b-a))*(x-(a+b)/2))))

def f_ramp(x,a,b):
    ''' RAMP function CSIRO equation 125a'''
    return  min(1,max(0,(a-x)/(a-b)))

def f_dim(x,y):
    '''a function that minimum value of zero otherwise differrrrence between the 2 inputs '''
    return max(0,x-y)	

def daylength(dayOfYear, lat):
    """Computes the length of the day (the time between sunrise and
    sunset) given the day of the year and latitude of the location.
    Function uses the Brock model for the computations.
    For more information see, for example,
    Forsythe et al., "A model comparison for daylength as a
    function of latitude and day of year", Ecological Modelling,
    1995.
    Parameters
    ----------
    dayOfYear : int
        The day of the year. 1 corresponds to 1st of January
        and 365 to 31st December (on a non-leap year).
    lat : float
        Latitude of the location in degrees. Positive values
        for north and negative for south.
    Returns
    -------
    d : float
        Daylength in hours.
    """
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    if -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) <= -1.0:
        return 24.0
    elif -np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth)) >= 1.0:
        return 0.0
    else:
        hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
        return 2.0*hourAngle/15.0		

#^this function can handle 2 multi D arrays. not required for P associations because P is 1D array
# def f_next_prev_joining(joining_date,age,offset):
#     '''
    

#     Parameters
#     ----------
#     Params must have identical axis 1 and 2
    
#     joining_date : Array
#         Joining date.
#     age : Array
#         age of animal at the begining of each period. 

#     Returns
#     -------
#     Array.

#     '''
#     a_next = np.zeros(len(age[0]) * len(joining_date[1]) * len(joining_date[2])).reshape(len(age[0]), len(joining_date[1]), len(joining_date[2])) 
#     for i in range(len(joining_date[1])):
#         for c in range(len(joining_date[2])):
#             ###next joining date
#             a_next[:,l,c1] = np.searchsorted(joining_date[:,l,c1], age[:,l,c1])
#     ###previous joining date
#     return np.minimum(a_next_o_plc1 - offset, len(age)-1)

def f_next_prev_association(datearray_sclice,*args):
    '''
    Depending on the inputs this function will return the next or previous assosiation.
    eg it can be used to determine the next lambing opportunity for each period.
    See john stuff.py for alternative methods.
    
    Parameters
    ----------
    datearray_sclice : Int
        This is the axis along which the array is being sliced (this must be sorted).
    *args : 1 - array, 2 - int
        Arg 1: the period array 1d that is the index is being found for, note the index is based off the start date therefore must do [1:-1] if you want idx based on end date.
        Arg 2: period offset (this may be needed if evaluating the end of the period.

    Returns
    -------
    Array
        The function finds the index value of the datearray which is either the next or previous date for a given input date.

    '''
    date=args[0]
    offset=args[1] #offset is used to get the previous datearray period
    idx_next = np.searchsorted(datearray_sclice, date)
    idx = np.clip(idx_next - offset, 0, len(datearray_sclice)-1) #makes the max value equal to the length of joining array, because if the period date is after the last lambing opportunity there is no 'next' 
    return idx

# def f_find_index(datearray_slice,*args):
#     '''
#     Parameters
#     ----------
#     datearray_slice : np.datetime64
#         The axis along which the array is being sliced (must be sorted).
#     *args : 1 - 1D array, 2 - integer
#         Arg 1: the 1D array that the index is being found for, note the index is based off the start date therefore must do [1:-1] if you want idx based on end date.
#         Arg 2: period offset. -1 if the previous period is required.

#     Returns
#     -------
#     Array
#         The function finds the index value of the datearray which is either the next or previous date for a given input date.

#     '''
#     date = args[0]
#     offset = args[1]  # offset is -1 to get the previous datearray period
#     idx_next = np.searchsorted(datearray_slice, date)
#     # ## clip the value between 0 and the length of joining array.
#     idx = np.clip(idx_next + offset, 0, datearray_slice.shape[0] - 1)
#     return idx

def convert_input_to_g(source, a_g0, a_paternal_g0_g1, a_maternal_g0_g1,
                                     a_paternal_g0_g2, a_maternal_g1_g2):
    """
    Convert numpy array source data to a selected genotype (g) array

    Parameters
    ----------
    source : array of data with last axis that is related to g by association.
    a_g0 : association between the source array and male genotypes (g0).
    a_paternal_g0_g1 : paternal association for the female (g1).
    a_maternal_g0_g1 : maternal association for the female (g1).
    a_paternal_g0_g2 : paternal association for the offspring (g2).
    a_maternal_g1_g2 : maternal association for the offspring (g2).

    Returns
    -------
    g : source data in a ...g array.
    """
    g0 = source[...,a_g0]
    g1 = (g0[a_paternal_g0_g1] + g0[a_maternal_g0_g1]) / 2
    g2 = (g0[a_paternal_g0_g2] + g1[a_maternal_g1_g2]) / 2
    g = g0
    ## requires a method to convert from g0, g1 & g2 to g
    ## difficulty is that the g slices change depending on the inputs
    return g


def f_k2g(input_k1, a_k1_g0, input_k2=None, a_k2_g0=None, *other_associations):
    """
    Create parameters by selected genotype from animal type and possible genotypes

    Parameters
    ----------
    input_k1 : A parameter array with animal type (u) as last axis.
    input_k2 : Optional parameter array with possible genotypes (k) as last axis.
    a_k1_g0 : An association array between animal type (u) and selected genotype (g).
    a_k2_g0 : Optional association array between input genotypes (k) and selected genotype.

    Returns
    -------
    Array of parameters with the last axis being selected genotype (g).

    """

    c_k1_g = convert_input_to_g(input_k1, a_k1_g0, other_associations)
    try:
        c_k2_g = convert_input_to_g(input_k2, a_k2_g0, other_associations)
    except:   #^enter exception type so the error catch is specific
        c_k2_g = c_k1_g
    c_g = c_k2_g
    c_g[c_k2_g==0] = c_k1_g[c_k2_g==0]
    return c_g

# def genotype(excelinputs, a_c_g0, a_maternal_g0_g1, a_paternal_g0_g1
#              , a_maternal_g1_g2, a_paternal_g0_g2):
#     '''


#     Returns
#     -------
#     outputs coloured green, yellow, orange and blue in the genotype function
#     Arrays:
#         c_srw_gw
#         c_sfw_g


#     '''
#     ## take the genotype inputs and create the parameters that define the genotypes
#     # Map genotype options to i_<arrays>_c: SRW, SFW, cw6 parameters
#     #    where i_arrays_c: i_ means input arrays, <arrays> is the name & _c is the index
#     i_srw_c = exceldata['<rangename>']
#     i_gfw_c = exceldata['<rangename>']
#     i_fd_c = exceldata['<rangename>']

#     # Map parameter inputs to i_<arrays>_a: cn, cl, ck, cm, cp, cl, cw, cc, cg, ch, cf & cd
#     i_cn_a = exceldata['<rangename>']
#     i_ci_ay = exceldata['<rangename>']
#     #inputs don't have an y dimension but the returned result does
#     i_cr_a = exceldata['<rangename>']
#     i_ck_a = exceldata['<rangename>']
#     i_cm_a = exceldata['<rangename>']
#     i_cp_ax = exceldata['<rangename>']
#     #inputs don't have an x dimension but the returned result does
#     i_cl_ay = exceldata['<rangename>']
#     #inputs don't have an y dimension but the returned result does
#     i_cw_a = exceldata['<rangename>']
#     # i_cw_a[6] is a special case. With values being read into with  i_cw6_c
#     # because it varies by genotype rather than just animal type
#     i_cc_a = exceldata['<rangename>']
#     i_cg_a = exceldata['<rangename>']
#     i_ch_a = exceldata['<rangename>']
#     i_cf_ax = exceldata['<rangename>']
#     #inputs don't have an x dimension but the returned result does
#     i_cd_a = exceldata['<rangename>']

#     ### convert to _g arrays
#     #from input data by csiro animal type and genotype option (c)
#     # ram genotypes (B,M,T) are selected from the list of genotype options _c)
#     g0 axis array = c axis Array[a_c_g0]
#     # ewe genotype (BB,BM) are created by a combination of the 3 ram genotypes
#     g1 axis array = ( g0 axis array[a_maternal_g0_g1]
#                      +g0 axis array[a_paternal_g0_g1]) / 2
#     # the offspring genotype (BBB, BBM, BBT, BMT) are created by a combination of the 2 ewe genotypes and the 3 ram genotypes
#     g2 axis array = ( g1 axis array[a_maternal_g1_g2]
#                      +g0 axis array[a_paternal_g0_g2]) / 2
#     # create the full _g arrays (B, M, T, BM, BT, BMT)
#     array_g[0:2] = array_g0  # the 3 ram genotypes
#     array_g[3:4] = array_g2[1:3]  #the offspring genotypes excluding the first (B) because it was part of the ram genotypes

#     return c_srw_gw, c_sfw_g

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
    step - seconds in each period
    '''
    n_sim_periods = int(oldest_animal * periods_per_year)
    start_date = start_year + relativedelta(month=1,day=1) 
    step = pd.to_timedelta(365.25 / periods_per_year,'D')
    step = step.to_numpy().astype('timedelta64[s]')
    index_p = np.arange(n_sim_periods + 1)
    date_start_p =  (np.datetime64(start_date) + (step * index_p)).astype('datetime64[D]') #astype day rounds the date to the nearest day
    date_end_p = (np.datetime64(start_date - dt.timedelta(days=1)) + (step * (index_p+1))).astype('datetime64[D]') #adding and then minusing 1 is to keep the slight offset (because step is 7.01 days) of the step length in the same place when the date is rounded to the nearest day.	
    return n_sim_periods, date_start_p, date_end_p, index_p, step

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

def r_intake(rc, c_ci_gy, feed_supply, srw, rel_size,  ):
    #equations 14 to 30 wihtout d
    

def p_intake(rc, c_ci_gy, feed_supply, srw, rel_size,  ):
    #do potential intake calculations
    #equations 2 to 10 & 72
    #`
    # 2 Imax = CI1 * SRWZ ( CI2 - Z ) * CF YF TF LF
    # 2 mei =
    # 3 CF = BC



    gy      = (n_genotypes
          ,n_lactation_number
          ,23 )    #` this dimension represents the subscript from GrazPLan

    a_g_j = (n_groups_ewes,
                n_genotypes)
    
    jexyl   = (n_groups_ewes
              ,n_max_ecycles
              ,n_litter_size
              ,n_lactation_number
              ,n_groups_lambing
              )

    c_ci_jy = [c_ci_gy[a_g_j[0],...], c_ci_gy[a_g_j[1],...], c_ci_gy[a_g_j[2],...], c_ci_gy[a_g_j[3],...]]
    c_ci_jy = c_ci_jy[:, np.newaxis, np.newaxis, :, np.newaxis]
    picf = rc
    picf[picf<=1] = 0
    picf = picf * (c_ci_jy[...,20] - rc) / (c_ci_jy[...,20] - 1)
    picf[picf==0] = 1

    yf = ( 1 - thetamilk ) / ( 1 + exp (c_ci_jy[...,13] * -1 * (a - c_ci_jy[...,14])))

    tf = 

    return mei

def energy(dmd, ):
    ## equations 33 to 45
    km =
    return mem

def foetus():
    ## equations 57 to 65
    return d_lw_f_jexl, cw_jexl, mec_jexl

def milk():
    ## equations 66 to 75
    return ldr_jexyl, lb_jexyl, mel_jexyl

def fibre():
    ## equations 77 to 86
    return d_cfw_wolag, d_cfw, mew, d_fd, d_fl

def chill():  # do this in v2
    ## equations 88 to 100
    ## remember to include the age adjustment for c_cc[3] if <30days old
    return me_cold, kg

def lwc():
    ## equations 101 to 116
    ## note not doing the protein component other than Eqn 105
    return ebg, pg

def wool_value(cfw, fl, fd, fd_min, wool_prices):
    ## these equations are not documented
    return wool_value

def sale_value(lw, cs):
    ## these equations are not documented
    return sale_value

def emissions_nggi():  # do this in v2
    return 0

def emissions_blaxter():  # do this in v2
    return 0

