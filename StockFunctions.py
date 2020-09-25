# -*- coding: utf-8 -*-
"""
All the calculation function for the simulation
Created on Sat Feb 29 08:05:09 2020

@author: John
"""
import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats
import math


# from dateutil.relativedelta import relativedelta

import Functions as fun
import PropertyInputs as pinp
import UniversalInputs as uinp

na=np.newaxis

def f_sig(x,a,b):
    ''' Sig function CSIRO equation 124 ^the equation below is the sig function from sheepexplorer'''
    return  1/(1+np.exp(-((2*(np.log(0.95) - np.log(0.05))/(b-a))*(x-(a+b)/2))))

def f_ramp(x,a,b):
    ''' RAMP function CSIRO equation 125a'''
    return  np.minimum(1,np.maximum(0,(a-x)/(a-b)))

def f_dim(x,y):
    '''a function that minimum value of zero otherwise differrrrence between the 2 inputs '''
    return np.maximum(0,x-y)

def f_daylength(dayOfYear, lat):
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
    dl=np.zeros_like(dayOfYear, dtype='float64')
    latInRad = np.deg2rad(lat)
    declinationOfEarth = 23.45*np.sin(np.deg2rad(360.0*(283.0+dayOfYear)/365.0))
    m1 = (-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))) <= -1.0
    m2 = (-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))) >= 1.0
    hourAngle = np.rad2deg(np.arccos(-np.tan(latInRad) * np.tan(np.deg2rad(declinationOfEarth))))
    daylen = 2.0*hourAngle/15.0
    dl[m1] = 24
    dl[m2] = 0
    dl[~np.logical_and(m2, m1)] = daylen[~np.logical_and(m2, m1)]
    return dl
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
        ## The previous opportunity is the latest opportunity date that is less than the date at the end of the period
        ## eg ('end of the period' so that if joining occurs during the period it is the previous
        ## The next opportunity is the earliest joining date that is greater than the date at the start of the period
        ## eg So it is the prev + 1 except if the joining is occurring within the period, in which case it points to this one.


    '''
    date=args[0]
    offset=args[1] #offset is used to get the previous datearray period
    idx_next = np.searchsorted(datearray_sclice, date)
    idx = np.clip(idx_next - offset, 0, len(datearray_sclice)-1) #makes the max value equal to the length of joining array, because if the period date is after the last lambing opportunity there is no 'next'
    return idx


def sim_periods(start_year, periods_per_year, oldest_animal):
    '''Define the dates for the simulation periods.
    Starts on 1 Jan of the year with the earliest birthdate.

    Parameters:
    start_year = int: year to start simulation.
    periods_per_year = int:
    oldest_animal = float: age of the oldest animal to be simulated (yrs)

    Returns:
    n_sim_periods
    array of period dates (1D periods)
    array of period end dates (1D periods) (date of the last day in the period)
    index of the periods (for pyomo)
    step - seconds in each period
    '''
    n_sim_periods = int(oldest_animal * periods_per_year)
    start_date = dt.date(year=start_year, month=1,day=1)
    step = pd.to_timedelta(365.25 / periods_per_year,'D')
    step = step.to_numpy().astype('timedelta64[s]')
    index_p = np.arange(n_sim_periods + 1)
    date_start_p =  (np.datetime64(start_date) + (step * index_p)).astype('datetime64[D]') #astype day rounds the date to the nearest day
    date_end_p = (np.datetime64(start_date - dt.timedelta(days=1)) + (step * (index_p+1))).astype('datetime64[D]') #minus one day to get the last day in the period not the first day of the next period.
    return n_sim_periods, date_start_p, date_end_p, index_p, step

def f_condition_score(ffcfw, normal_weight, cs_propn = 0.19):
    ''' Estimate CS from LW. Works with scalars or arrays - provided they are broadcastable into ffcflw.

   ffcfw: (kg) Fleece free, conceptus free liveweight. normal_weight: (kg). cs_propn: (0.19) change in LW
   associated with 1 CS as a proportion of normal_weight.

   Returns: condition score - float
   '''
    return 3 + (ffcfw - normal_weight)/(cs_propn * normal_weight)



###################################
#input and manipulation functions #
###################################

def f_c2g(params_c2, y=0, var_pos=0, len_ax1=0, len_ax2=0, group = 'dams'):
    '''
    Parameters
    ----------
    params_c2 : array
        parameter array - input from excel.
    y : array
        sensitivity array for genetic merit.
    var_pos : int
        position of last axis when inserted into all axis.
    len_ax1 : int
        length of axis 1 - used to reshape input array into multi dimension array (this should be i_len_?).
    len_ax2 : int, optional
        length of axis 1 - used to reshape input array into multi dimension array (this should be i_len_?). The default is 0.
    g2g: boolean, optional
        this determines if the user only wants to convert from g to g (ie select which genotype options need to be represented for the necesary offs) this only happens if the inputs dont have k axis.
    group:
        this is used to specify the sheep group that the g2g mask is being applied
    Returns
    -------
    param array for each genotype. Grouped by sheep group ie sire, offs, dams, yatf.
    If g2g is selected then only the conversion from all g? to relevant g? is done using the g?g3 mask

    '''

    ##these inputs are used for each param so they don't need to be passed into the function.
    a_c2_c0 = pinp.sheep['a_c2_c0']
    i_g3_inc = pinp.sheep['i_g3_inc']
    i_mul_g0_c0 = uinp.structure['i_mul_g0c0']
    i_mul_g1_c0 = uinp.structure['i_mul_g1c0']
    i_mul_g2_c0 = uinp.structure['i_mul_g2c0']
    i_mul_g3_c0 = uinp.structure['i_mul_g3c0']
    i_mask_g0g3 = uinp.structure['i_mask_g0g3']
    i_mask_g1g3 = uinp.structure['i_mask_g1g3']
    i_mask_g2g3 = uinp.structure['i_mask_g2g3']
    i_mask_g3g3 = uinp.structure['i_mask_g3g3']

    ##convert params from c2 to c0
    params_c0 = params_c2[...,a_c2_c0]
    ##add y axis
    na=np.newaxis
    ###if y is not numpy ie was read in as an int because it was a single cell, it needs to be converted
    if type(y) == int:
        y = np.asarray([y])
    ###y is a 2d array howvever currently it only has one slice so it is read in as a 1d array. so i need to add second array
    if y.ndim == 1 and params_c0.ndim != 1:
        y=y[...,na]
    params_c0 = np.multiply(params_c0[...,na,:],  y[...,na]) #na here is to account for c2 axis
    ##reshape parameter from 2d input to multi dim array
    len_y = y.shape[-1]
    ###make tuple of shape depending on the number of axis in input
    if len_ax2>0:
        shape=(len_ax1,len_ax2,len_y,3)
        params_c0 = params_c0.reshape(shape)
    elif len_ax1 > 0:
        shape=(len_ax1,len_y,3)
        params_c0 = params_c0.reshape(shape)
    else:
        pass#don't need to reshpae
    ##get axis into correct position
    if var_pos != None or var_pos != 0:
        extra_axes = tuple(range((var_pos + 1), -2))
    else: extra_axes = ()
    allaxis_params_c0 = np.expand_dims(params_c0, axis = extra_axes)
    ##create mask g?c0
    mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
    mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
    mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
    mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
    ##create array with the proportion of each pure genotype required to have each actual genotype included in the analysis
    mul_sire_genotypes_g0c0 = i_mul_g0_c0[mask_sire_inc_g0]
    mul_dams_genotypes_g0c0 = i_mul_g1_c0[mask_dams_inc_g1]
    mul_yatf_genotypes_g0c0 = i_mul_g2_c0[mask_yatf_inc_g2]
    mul_offs_genotypes_g0c0 = i_mul_g3_c0[mask_offs_inc_g3]
    ##convert params from c0 to g. nansum required when the selected c0 info is not filled out ^may be an issue if params are missing and mixed breed sheep is selected because it wont catch the error
    param_sire=np.nansum(allaxis_params_c0[..., na, :] * mul_sire_genotypes_g0c0, axis = -1)
    param_dams=np.nansum(allaxis_params_c0[..., na, :] * mul_dams_genotypes_g0c0, axis = -1)
    param_yatf=np.nansum(allaxis_params_c0[..., na, :] * mul_yatf_genotypes_g0c0, axis = -1)
    param_offs=np.nansum(allaxis_params_c0[..., na, :] * mul_offs_genotypes_g0c0, axis = -1)
    return param_sire, param_dams, param_yatf, param_offs


def f_g2g(array_g,group,left_pos=0,len_ax1=0,len_ax2=0,len_ax3=0,swap=False,right_pos=-1,left_pos2=0,right_pos2=-1, condition = None, axis = 0):
    '''
    Parameters
    ----------
    array_g : array
        parameter array - input from excel.
    group : TYPE
        DESCRIPTION.
    left_pos : int
        position of axis to the left of where the new axis will be added.
    len_ax1 : int
        length of axis 1 - used to reshape input array into multi dimension array (this should be i_len_?).
    len_ax2 : int, optional
        length of axis 3 - used to reshape input array into multi dimension array (this should be i_len_?). The default is 0.
    len_ax3 : int, optional
        length of axis 3 - used to reshape input array into multi dimension array (this should be i_len_?). The default is 0.
    swap : boolean, optional
        do you want to swap the first tow axis?. The default is False.
    right_pos : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    left_pos2 : int
        position of axis to the left of where the new axis will be added.
    right_pos2 : int, optional
        the position of the axis to the right of the singleton axis being added. The default is -1, for when the axis to the right is g?.
    condition: boolean, optional
        mask used to slice given axis.
    axis: int, optional
        axis to apply mask to.

    *note: if adding two sets of new axis add from right to left (then the pos variables allign)

    Returns
    -------
    Reshapes, swaps axis if required, expands and converts g array to the correct g slices.
    '''

    i_g3_inc = pinp.sheep['i_g3_inc']
    i_mask_g0g3 = uinp.structure['i_mask_g0g3']
    i_mask_g1g3 = uinp.structure['i_mask_g1g3']
    i_mask_g2g3 = uinp.structure['i_mask_g2g3']
    i_mask_g3g3 = uinp.structure['i_mask_g3g3']

    if len_ax3>0:
        shape=(len_ax1,len_ax2,len_ax3,array_g.shape[-1])
        array_g = array_g.reshape(shape)
    elif len_ax2>0:
        shape=(len_ax1,len_ax2,array_g.shape[-1])
        array_g = array_g.reshape(shape)
    elif len_ax1 > 0:
        shape=(len_ax1,array_g.shape[-1])
        array_g = array_g.reshape(shape)
    else:
        pass#don't need to reshpae
    ##swap axis if neccessary
    if swap:
        array_g = np.swapaxes(array_g, 0, 1)
    ##get axis into correct position 1
    if left_pos != None or left_pos != 0:
        extra_axes = tuple(range((left_pos + 1), right_pos))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)

    ##get axis into correct position 2 (some arrays need singleton axis added in multiple places ie seperated by a used axis)
    if left_pos2 != None or left_pos2 != 0:
        extra_axes = tuple(range((left_pos2 + 1), right_pos2))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)
    ##select the required genotypes based on the offspring the user wants to model.
    if group == 'sire':
        ##create mask g?g
        mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_sire_inc_g0]
    elif group == 'dams':
        ##create mask g?g
        mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_dams_inc_g1]
    elif group == 'offs':
        ##create mask g?g
        mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_offs_inc_g3]
    elif group == 'yatf':
        ##create mask g?g
        mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
        array = array_g[...,mask_yatf_inc_g2]
    ##apply mask if required
    if condition is not None: #see if condition exists
        if type(condition) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition= np.asarray([condition]) #convert to numpy if it is singular input
            array = np.compress(condition, array, axis)
        else:
            array = np.compress(condition, array, axis)
    return array


def f_DSTw(scan_std_yg):
    '''
    Parameters
    ----------
    scan_std : np array
        scanning percentage of genotypes.

    Returns
    -------
    Proportion of dry, single, twins & triplets. Numpy way of making this formula: y=int+ax+bx^2+cx^3+dx^4

    '''
    scan_powers_s = uinp.sheep['i_scan_powers']  #scan powers are the exponential powers used in the quadratic formula ie ^0, ^1, ^2, ^3, ^4
    scan_power_ygs = scan_std_yg[...,na] ** scan_powers_s #puts the variable to the powers ie x^0, x^1, x^2, x^3, x^4
    dstwtr_ygl0 = np.sum(uinp.sheep['i_scan_coeff_l0s'] * scan_power_ygs[...,na,:], axis = -1) #add the coefficients and sum all the elements of the equation ie int+ax+bx^2+cx^3+dx^4
    return dstwtr_ygl0

def f_btrt0(dstwtr,lss,lstw,lstr): #^this function is inflexible ie if you want to add qradruplets
    '''
    Parameters
    ----------
    dstwtr : np array
        proportion of dry, singles, twin and triplets.
    lss : np array
        single survival.
    lstw : np array
        twin survival.
    lstr : np array
        triplet survival.

    Returns
    -------
    btrt_b0xyg : np array
        proportion of lambs in each btrt category (eg 11, 22, 21 ...).

    '''
    ##lamb numbers is the number of lambs in each b0 category, based on survival of s, tw and tr after birth.
    lamb_numbers_b0yg = np.zeros((uinp.parameters['i_b0_len'],lss.shape[-2],lss.shape[-1]))
    lamb_numbers_b0yg[0,...] = lss
    lamb_numbers_b0yg[1,...] = 2 * lstw**2 #number of lambs when there are no deaths is 2, therefore 2p^2
    lamb_numbers_b0yg[2,...] = 3 * lstr**3 #number of lambs when there are no deaths is 2, therefore 3p^3
    lamb_numbers_b0yg[3,...] = 2 * lstw * (1 - lstw)  #the 2 is because it could be either lamb 1 that dies or lamb 2 that dies
    lamb_numbers_b0yg[4,...] = 2 * (3* lstr**2 * (1 - lstr))  #the 2x is because there are 2 lambs in the litter (so need to be accounted for to determine number of lambs) and the 3x because it could be either lamb 1, 2 or 3 that dies
    lamb_numbers_b0yg[5,...] = 3* lstr * (1 - lstr)**2  #the 3x because it could be either lamb 1, 2 or 3 that survives
    ##mul lamb numbers array with lambing percentage to get the number of lambs surviving per ewe.
    a_nfoet_b0 = uinp.structure['a_nfoet_b1'][uinp.structure['i_mask_b0_b1']] #create association between l0 and b0
    btrt_b0yg = lamb_numbers_b0yg * dstwtr[a_nfoet_b0] #multiply the lamb numbers by the proportion of single, twin, trip.
    ##add singleton x axis
    btrt_b0xyg = np.expand_dims(btrt_b0yg, axis = tuple(range((uinp.parameters['i_cb0_pos'] + 1), -2))) #note i_cb0_pos refers to b0 position
    ##finally convert to proportion of each category
    nlw = np.sum(btrt_b0xyg, axis=0) #this number is effectively number of lambs weaned per ewe joined
    btrt_propn_b0xyg = btrt_b0xyg / nlw
    return btrt_propn_b0xyg

def f_period_is_(period_is, date_array, date_start_p=0, date_array2 = 0, date_end_p=0):
    '''
    Parameters
    ----------
    period_is : string
        type of period is calc to return.
    date_start_p : datetime64[D]
        start date of each period (must have all axis).
    date_end_p : datetime64[D]
        end date of each period (must have all axis).
    date_array : datetime64[D]
        array of dates of interest eg mating dates.
    date_array2 : datetime64[D]
        array of end dates used to determine if period is between.

    Returns
    -------
    period_is: boolean array shaped like the date array with the addition of the p axis. This is is true if a given date from date array is within the date of a given period and false if not.

    period_is_any: 1D boolean array shape of the period dates array. True if any of the dates in the date array fall into a given period.

    period_is_between: return true if a the period is between two dates (it is inclusive ie if an activity occurs during the period that period will be treated as between the two dates)
    '''
    if period_is == 'period_is':
        period_is=np.logical_and((date_array>=date_start_p) , (date_array<=date_end_p))
        return period_is
    if period_is == 'period_is_any':
        period_is=np.logical_and((date_array>=date_start_p) , (date_array<=date_end_p))
        period_is_any = np.any(period_is,axis=tuple(range(1,period_is.ndim)))
        return period_is_any
    if period_is == 'period_is_pre':
        period_is_pre=(date_array>date_end_p)
        return period_is_pre
    if period_is == 'period_is_between':
        period_is_between= np.logical_and((date_array<=date_end_p) , (date_array2>=date_start_p))
        return period_is_between


################
#Sim functions #
################
def f_feedsupply_adjust(attempts,feedsupply,itn):
    ##create empty array to put new feedsuply into, this is done so it doesnt have the itn axis (probably could just create from attempts array shape without last axis)
    feedsupply = np.zeros_like(feedsupply)
    ##which feedsupplies can be calculated using binary method - must have a negitive and positive error
    binary_mask = np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1) < 0 #axis -1 is the itn axix ie take the min and max error from the previous itterations
    ##calc new feedsupply binary - take half of the two feedsupplys that have resulted in the error closest to 0. Only adds the binary result to slices that have a negitive and a positive value (done using the mask created above)
    ###feedsuply with negitive error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]>0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    neg_bool=np.ma.getdata(mask_attempts.max(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negitive but closest to 0
    neg_bool = neg_bool * binary_mask[...,na] #this just makes sure the neg mask only has a true in the same slice as the pos array (so it can be applied to the ffed supply array below)
    ###feedsuply with positive error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]<0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    pos_bool=np.ma.getdata(mask_attempts.min(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negitive but closest to 0
    pos_bool = pos_bool * binary_mask[...,na] #this just makes sure the pos mask only has a true in the same place as the neg mask. 
    ##calc feedsupply
    feedsupply[binary_mask] = (attempts[...,0][neg_bool] + attempts[...,0][pos_bool])/2    
    ##calc feedsupply using interpolation
    ###first determine the slope, slope is always positive ie as feedsupply increases error increase because error = lwc - target and more feed means hihger lwc.
    if itn==0:
        slope=pinp.sheep['i_feedsupply_slope_std']
    else:
        ####linregress only works on 1d array and cant use apply_over_axis because needs x and y. maybe there is a beter way but i looked for a while and found nothing
        slope=np.empty_like(feedsupply)
        feedsupply_all_itn = attempts[...,0]
        error_all_itn = attempts[...,1]
        for i in np.ndindex(error_all_itn.shape[:-1]): #not exactly sure how this is working but it is creating tupple of each combo of slices in each axis.
            x= feedsupply_all_itn[i] #indexing with tupple works correctly if we are interested in the last axis otherwise it doesn't work properly for some reason.ie t[(0,0)] == t[0,0,:] but t[:,(0,0)] != t[:,0,0]
            y= error_all_itn[i]
            slope[i] = stats.linregress(x,y)
    ####new feedsupply = minerror / slope. It is assumed that the most recent itn has the most accurate feedsupply
    feedsupply[~binary_mask] = ((2 * attempts[...,-1,1]) / slope)[~binary_mask] # x2 to overshoot then switch to binary.
    return feedsupply



def f_rq_cs(dmd,legume,cr=None, i_sf=0):
    ##To work for DMD as a % or a proportion
    try:
        if (dmd >= 1).any() : dmd /= 100
    except:
        if dmd >= 1:          dmd /= 100
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[1,…] using c2[0] (finewool merino)
        cr1 = uinp.parameters['i_cr_c2'][1,0]
        ###Scalar version of cr[3,…] using c2[0] (finewool merino)
        cr3 = uinp.parameters['i_cr_c2'][3,0]
    else:
        cr1=cr[1, ...]
        cr3=cr[3, ...]
    ##Scalar version of formula
    try:
        ###Relative ingestibility
        rq = max(0.01, min(1, 1 - cr3 * (cr1 - (dmd +  i_sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    ##Numpy version of formula
    except:
        ###Relative ingestibility
        rq = np.maximum(0.01, np.minimum(1, 1 - cr3 * (cr1 - (dmd +  i_sf * (1 - legume))))) #(1-legume) because sf is actually a factor related to the grass component of the sward
    return rq

def f_ra_cs(foo, hf, cr=None, zf=1):
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[1,…] using c2[0] (finewool merino)
        cr4 = uinp.parameters['i_cr_c2'][4,0]
        cr5 = uinp.parameters['i_cr_c2'][5,0]
        cr6 = uinp.parameters['i_cr_c2'][6,0]
        cr13 = uinp.parameters['i_cr_c2'][13,0]
    else:
        cr4=cr[4, ...]
        cr5=cr[5, ...]
        cr6=cr[6, ...]
        cr13=cr[13, ...]
    try: 
        ##Relative rate of eating	
        rr = 1 - math.exp(-(1+cr13 * 1) * cr4 * hf * zf * foo) #*1 is a reminder that this formula could be improved in a future version
        ##Relative time spent grazing	
        rt = 1 + cr5 * math.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    except: #numpy version
        ##Relative rate of eating	
        rr = 1 - np.exp(-(1+cr13 * 1) * cr4 * hf * zf * foo) #*1 is a reminder that this formula could be improved in a future version
        ##Relative time spent grazing	
        rt = 1 + cr5 * np.exp(-(1 + cr13 * 1) * (cr6 * hf * zf * foo)**2)
    ##Relative availability	
    ra = rr * rt
    return ra


def f_foo_convert(cu3, cu4, foo, i_hr_scalar, i_region, i_n_pasture_stage,i_hd_std, legume=0, pasture_stage=1, cr=None): 
    '''
    Parameters
    ----------
    cu3 : 
        this parameter should already be slice on the c4 axis.
    cu4 : 
        this parameter should already be slice on the c4 axis.
    '''
    ##create scalar cr if not passed in
    if cr is None:
        ###Scalar version of cr[1,…] using c2[0] (finewool merino)
        cr12 = uinp.parameters['i_cr_c2'][12,0]
    else:
        cr12=cr[12, ...]
    ##pasture conversion scenario
    conversion_scenario = i_region * i_n_pasture_stage + pasture_stage
    ##select cu3&4 params
    cu3=cu3[..., conversion_scenario]
    cu4=cu4[..., conversion_scenario]
    ##Convert FOO to hand shears measurement
    foo_shears = np.maximum(0, np.minimum(foo, cu3[2] + cu3[0] * foo + cu3[1] * legume))
    ##Estimate height of pasture
    height = np.maximum(0, np.exp(cu4[3] + cu4[0] * foo + cu4[1] * legume + cu4[2] * foo * legume) + cu4[5] + cu4[4] * foo)
    ##Height density (height per unit FOO)
    hd = height / foo_shears  
    ##height ratio                    
    hr = i_hr_scalar * hd / i_hd_std
    ##calc hf
    hf = 1 + cr12 * (hr -1)
    return foo_shears, hf

def f_dynamic_slice(arr, axis, start, stop, axis2=None, start2=None, stop2=None):
    ##check if arr is int - this is the case for the first loop because arr may be initilised as 0
    if type(arr)==int:
        return arr
    else:
        ##first axis slice
        sl = [slice(None)] * arr.ndim
        sl[axis] = slice( start, stop)
        arr = arr[tuple(sl)]
        if axis2 is not None:
            ##second axis slice if required
            sl = [slice(None)] * arr.ndim
            sl[axis2] = slice( start2, stop2)
            arr = arr[tuple(sl)]
        return arr



def f_history(history, new_value, days_in_period):
    '''
    The idea that the f_history is implementing is for traits that have a lag from increased nutrition to increased production. 
    The representation being that the production today is an average of the non-lagged estimated production from the last x days (where x is either len_m2 or len_m3, either of which can be 1 but are expected to be 25).
    The history function is keeping track of the last x days of estimated production (slice 0 is the most recent day, slice -1 is the oldest day of the history). 
    The process is:
        1. move the most recent history back (to make space for the production from this period: days_period) 
        2. make the 0:days_period = this period estimated production
    '''
    offset = np.max(np.minimum(days_in_period.astype(int), history.shape[0]))
    history = np.roll(history, offset, axis=0)
    ##make new value = nan when days per period==0
    days_in_period = np.broadcast_to(days_in_period, new_value.shape) #broadcast days array so it can be used to make mask
    new_value[days_in_period.astype(int)==0]=np.nan
    history[:offset, ...] = new_value
    lagged = np.nanmean(history, axis = 0)
    return lagged, history



def roll_slices(array, roll, roll_axis=0):
    '''
    The function rolls each slice for a given axis (the np roll function rolls each slice the same)
    you can roll different slices by different amounts
    :param array: array to be rolled
    :param roll: number of times the slice is to be rolled - this array should have one less dim than the main array
    :param axis: axis to roll down
    :return:
    '''
    #flattern array if multi dim
    a = array.reshape(array.shape[roll_axis], int(np.prod(array.shape)/array.shape[roll_axis]))
    r = roll.reshape(int(np.prod(roll.shape)))

    rows, column_indices = np.ogrid[:a.shape[0], :a.shape[1]]

    # Use always a negative shift, so that column_indices are valid.
    # (could also use module operation)
    r[r < 0] += a.shape[0]
    rows = rows - r
    result = a[rows, column_indices]
    return (result.reshape(array.shape))




def f_update(existing_value, new_value, mask_for_new):
    '''
    Parameters
    ----------
    existing_value : numpy array
        values you want when mask = false.
    new_value : numpy array
        values you want when mask = true.
    mask_for_new : boolean mask
        boolean mask for the final axis of the array (typically the g axis).

    Returns
    -------
    Numpy array
        returns a combination of the two input arrays determined by the mask. Note: multiplying by true return the origional number and multiplying by false results in 0.

    '''
    return existing_value * ~mask_for_new + new_value * mask_for_new







def f_potential_intake_cs(ci, srw, relsize_start, rc_start, temp_lc_dams, temp_ave, temp_max, temp_min, rain_intake, rc_birth_start = 1, pi_age_y = 0, lb_start = 0, mp2=0,piyf=1,period_between_birthwean=1):
    ##Condition factor on PI
    picf= np.minimum(1, rc_start * (ci[20, ...] - rc_start) / (ci[20, ...] - 1))
    ##Lactation adjustment (BC at parturition) - dam only because lb start = 0 for everything except lb
    la = 1 + ci[15, ...] * (rc_birth_start - 1)
    ##Lactation factor on PI - dam only
    pilf = 1 + pi_age_y * la * lb_start
    ##Temperature function
    piax = np.arccos(np.clip((temp_ave - temp_lc_dams) / (0.5 * (temp_max - temp_min)),-1,1))
    ##Temperature below the lower critical temp
    tlow = piax * (temp_lc_dams - temp_ave) + 0.5 * np.sin(piax) * (temp_max - temp_min) / np.pi
    ##Temperature factor on PI - high temperatures
    pitf_high = 1 - ci[5, ...] * (temp_ave - ci[6, ...])
    ##Temperature factor on PI - low temperatures
    pitf_low = 1 + ci[17,...] * tlow * rain_intake
    ##Temperature factor on PI
    pitf = np.minimum(1, pitf_high) * np.maximum(1, pitf_low)
    ##Potential intake
    pi = ci[1, ...] * srw * relsize_start * (ci[2, ...] - relsize_start) * picf * pitf * pilf
    ##Potential intake of pasture - young at foot only
    pi = (pi - mp2) * piyf
    ##Potential intake of pasture - young at foot only
    pi = pi * period_between_birthwean
    return np.maximum(0,pi)





def f_ra_mu(cu0, foo, hf, zf=1):
    return 1 - cu0[0, ...] ** (hf * zf * foo)



def f_intake(cr, pi, ra, rq, md_herb, feedsupply, intake_s, i_md_supp, legume, mp2=0):
    ##Relative Intake	
    ri = ra * rq * (1 + cr[2, ...] * ra **2 * legume)
    ##Pasture intake	
    intake_f = np.maximum(0, pi - intake_s) * ri * (feedsupply < 3)
    ##ME intake from forage	
    mei_forage = 0
    ##ME intake from supplement	
    mei_supp = intake_s * i_md_supp
    ##ME intake from herbage	
    mei_herb = intake_f * md_herb
    ##ME intake of solid food	
    mei_solid = mei_forage + mei_herb + mei_supp
    ##M/D of the diet (solids)	
    md_solid = mei_solid / (intake_f + intake_s)
    ##ME intake total	
    mei = mei_solid + mp2
    ##Proportion of ME as milk	
    mei_propn_milk = mp2 / mei
    ##Proportion of ME as herbage	
    mei_propn_herb = (mei_herb + mei_forage) / mei
    ##Proportion of ME as supp	
    mei_propn_supp = mei_supp / mei
    return mei, intake_f, md_solid, mei_propn_milk, mei_propn_herb, mei_propn_supp


def f_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb
         , kl = 0, mei_propn_milk = 0, lact_propn = 0):
    '''

    Parameters
    ----------
    ck : Numpy array
        sim parameters - efficiency of energy use.
    belowmaint : Numpy array of Boolean
        Is the animal class in energy deficit.
    km : Numpy array of float
        Efficiency of energy use for maintenance.
    kg_supp : Numpy array
        efficiency of supplement energy consumed used for LWG.
    kg_fodd : Numpy array
        efficiency of herbage energy consumed used for LWG.
    mei_propn_supp : Numpy array
        Proportion of energy consumed that was from supplement.
    mei_propn_herb : Numpy array
        Proportion of energy consumed that was from herbage.
    mei_propn_milk : Numpy array
        Proportion of energy consumed that was from milk.
    kl : Numpy array of float, Optional
        Efficiency of energy use for lactation. The default is 0.
    lact_propn : Numpy array, optional
        Proportion of the period that the dam is lactating. The default is 0.

    Returns
    -------
    kg - Efficiency of energy used for growth.

    '''
    # ##Set days_lact to numpy array if arg value is 0
    # lact_propn = np.asarray(lact_propn)
    ##Lactating, losing weight
    kg_lact_lose = kl / ck[10, ...]
    ##Lactation other
    kg_lact = kl * ck[9, ...]
    ##Non-lactating, losing weight
    kg_dry_lose = km / ck[11, ...]
    ##Non-lactating, other
    kg_dry = kg_supp * mei_propn_supp + kg_fodd * mei_propn_herb + ck[12, ...] * mei_propn_milk
    ##Compile lactation
    kg_lact = f_update(kg_lact , kg_lact_lose , belowmaint)
    ##Compile non-lactating
    kg_dry = f_update(kg_dry , kg_dry_lose , belowmaint)
    ##Compile whole formula
    kg = lact_propn * kg_lact + (1 - lact_propn) * kg_dry
    return kg



def f_energy_cs(ck, cx, cm, lw_start, mr_age, mei, omer_history_start, days_period, md_solid, i_md_supp, md_herb, lgf_eff, dlf_eff, i_steepness, density, foo, feedsupply, intake_f, dmd, mei_propn_milk=0):
    ##Efficiency for maintenance	
    km = (ck[1, ...] + ck[2, ...] * md_solid) * (1-mei_propn_milk) + ck[3, ...] * mei_propn_milk
    ##Efficiency for lactation - dam only	
    kl =  ck[5, ...] + ck[6, ...] * md_solid
    ##Efficiency for growth (supplement)	
    kg_supp = ck[16, ...] * i_md_supp
    ##Efficiency for growth (fodder)	
    kg_fodd = ck[13, ...] * lgf_eff * (1+ ck[15, ...] * dlf_eff) * md_herb
    ##Energy required at maint for metabolism	
    emetab = cx[10, ...] * cm[2, ...] * lw_start ** 0.75 * mr_age * (1 + cm[5, ...] * mei_propn_milk)
    ##Distance walked (horizontal equivalent)	
    distance = i_steepness * np.minimum(1, cm[17, ...] / density) / (cm[8, ...] * foo + cm[9, ...])
    ##Set Distance walked to 0 if in confinement	
    distance = distance * (feedsupply < 3)
    ##Energy required for movement	
    emove = cm[16, ...] * distance * lw_start
    ##Energy required for grazing	
    egraze = cm[6, ...] * lw_start * intake_f * (cm[7, ...] - dmd) + emove
    ##Energy associated with organ activity
    omer, omer_history = f_history(omer_history_start, cm[1, ...] * mei, days_period)
    ##ME requirement for maintenance (before ECold)
    meme = (emetab + egraze) / km + omer
    return meme, omer_history, km, kg_fodd, kg_supp, kl


def f_foetus_cs(cp, cb1, kc, nfoet, relsize_start, rc_start, nec_cum_start, w_b_std_y, w_f_start, nw_f_start, nwf_age_f, guw_age_f, ce_age_f, days_period_f):
    ##expected normal birth weight with dam age adj.	
    w_b_exp_y = (1 - cp[4, ...] * (1 - relsize_start)) * w_b_std_y
    ##Normal weight of foetus (mid period - dam calcs)	
    nw_f = w_b_exp_y * nwf_age_f
    ##change in normal weight of foetus	
    d_nw_f = nw_f - nw_f_start
    ##Proportion of normal foetal and birth weights	
    nwf_nwb = nw_f / w_b_std_y
    ##Normal weight of individual conceptus (mid period)	
    nw_gu = cp[5, ...] * w_b_exp_y * guw_age_f
    ##Normal energy of individual conceptus (end of period)	
    normale_gu = cp[8, ...] * cp[5, ...] * w_b_exp_y * ce_age_f
    ##Condition factor on BW	
    cfpreg = (rc_start - 1) * nwf_nwb
    ##change in foetus weight	
    d_w_f = d_nw_f *(1 + np.minimum(cfpreg, cfpreg * cb1[14, ...]))
    ##foetus weight (end of period)	
    w_f = w_f_start + d_w_f * days_period_f
    ##Weight of the gravid uterus (conceptus - mid period)	
    guw = nfoet * (nw_gu + (w_f - nw_f))
    ##Body condition of the foetus	
    rc_f = w_f / nw_f
    ##Cumulative ME required for conceptus	
    nec_cum = nfoet * rc_f * normale_gu
    ##NE required for conceptus	
    nec = (nec_cum - nec_cum_start) / days_period_f
    ##ME required for conceptus	
    mec = nec / kc
    return w_f, nec_cum, mec, nec, w_b_exp_y, nw_f, guw

def f_birthweight_cs(cx, w_b_yatf, w_f_dams, period_is_birth):
    ##set BW = foetal weight at end of period (if born)	
    t_w_b = w_f_dams * cx[15, ...] * period_is_birth
    ##update birth weight if it is birth period
    w_b_yatf = f_update(w_b_yatf, t_w_b, period_is_birth)
    return w_b_yatf

def f_birthweight_mu(cu1_yatf, cb1_yatf, cx_yatf, ce_yatf, w_b, cf_w_b_dams, ffcfw_birth_dams, ebg_dams, days_period, gest_propn, period_between_joinscan, period_between_scanbirth, period_is_birth):
    ##Carry forward BW increment	
    d_cf_w_b = f_carryforward_u1(cu1_yatf[16, ...], ebg_dams, False, period_between_joinscan, period_between_scanbirth, False, days_period, gest_propn)
    ##Carry forward BW increment	
    cf_w_b_dams = cf_w_b_dams + d_cf_w_b
    ##set BW = foetal weight at end of period (if born)	
    t_w_b_yatf = (cf_w_b_dams + cu1_yatf[16, -1, ...] + cu1_yatf[16, 0, ...] * ffcfw_birth_dams + cb1_yatf[16, ...] + cx_yatf[16, ...] + ce_yatf[16, ...])
    ##Update w_b if it is birth	
    w_b = f_update(w_b, t_w_b_yatf, period_is_birth)
    return w_b, cf_w_b_dams


def f_weanweight_cs(w_w_yatf, ffcfw_yatf, ebg_yatf, days_period, period_is_wean):
    ##set WWt = yatf weight at weaning	
    t_w_w = (ffcfw_yatf + ebg_yatf * days_period)
    ##update weaning weight if it is weaning period
    w_w_yatf = f_update(w_w_yatf, t_w_w, period_is_wean)
    return w_w_yatf

def f_weanweight_mu(cu1_yatf, cb1_yatf, cx_yatf, ce_yatf, w_w, cf_w_w_dams, ffcfw_wean_dams, ebg_dams, foo, days_period, lact_propn, period_between_joinscan, period_between_scanbirth, period_between_birthwean, period_is_wean):
    ##Carry forward WWt increment	
    d_cf_w_w = f_carryforward_u1(cu1_yatf[17, ...], ebg_dams, False, period_between_joinscan, period_between_scanbirth, period_between_birthwean, days_period, lact_propn)
    ##Carry forward WWt increment	
    cf_w_w_dams = cf_w_w_dams + d_cf_w_w
    ##set WWt = yatf weight at weaning	
    t_w_w = (cf_w_w_dams + cu1_yatf[17, -1, ...] + cu1_yatf[17, 0, ...] * ffcfw_wean_dams + cu1_yatf[17, 5, ...] * foo + cu1_yatf[17, 6, ...] * foo** 2 + cb1_yatf[17, ...] + cx_yatf[17, ...] + ce_yatf[17, ...])
    ##Update w_w if it is weaning	
    w_w = f_update(w_w, t_w_w, period_is_wean)
    return w_w, cf_w_w_dams

def f_milk(cl, srw, relsize_start, rc_birth_start, mei, meme, mew_min, rc_start, ffcfw75_exp_yatf, lb_start, ldr_start, age_yatf, mp_age_y,  mp2_age_y, i_x_pos, days_period_yatf, kl, lact_nut_effect):
    ##Max milk prodn based on dam CS birth	
    mpmax = srw** 0.75 * relsize_start * rc_birth_start * lb_start * mp_age_y
    ##Excess ME available for milk	
    mel_xs = (mei - (meme + mew_min * relsize_start)) * cl[5, ...] * kl
    ##Excess ME as a ratio of MPmax	
    milk_ratio = mel_xs / mpmax
    ##Age or energy factor	
    ad = np.maximum(age_yatf, milk_ratio / (2 * cl[22, ...]))
    ##Milk production based on energy available	
    mp1 = cl[7, ...] * mpmax / (1 + np.exp(-(-cl[19, ...] + cl[20, ...] * milk_ratio + cl[21, ...] * ad * (milk_ratio - cl[22, ...] * ad) - cl[23, ...] * rc_start * (milk_ratio - cl[24, ...] * rc_start))))
    ##Milk production (per animal) based on suckling volume	(milk production per day of lactation)
    mp2 = np.minimum(mp1, np.mean(f_dynamic_slice(ffcfw75_exp_yatf, i_x_pos, 1, None), axis = i_x_pos, keepdims=True) * mp2_age_y)   # averages female and castrates weight, ffcfw75 is metabolic weight
    ##ME for lactation (per day lactating)	
    mel = mp2 / (cl[5, ...] * kl)
    ##NE for lactation	
    nel = kl * mel
    ##ratio of actual to potential milk	
    dr = mp2 / mpmax
    ##Lagged DR (lactation deficit)	
    ldr = (ldr_start - dr) * (1 - cl[18, ...]) ** days_period_yatf + dr
    ##Loss of potential milk due to consistent under production	
    lb = lb_start - cl[17, ...] / cl[18, ...] * (1 - cl[18, ...]) * (1 - (1 - cl[18, ...]) ** days_period_yatf) * (ldr_start - dr)
    ##If early in lactation = 1	
    lb = lb * lact_nut_effect + ~lact_nut_effect
    return mp2, mel, nel, ldr, lb





def f_fibre(cw, cc, ffcfw_start, relsize_start, d_cfw_history_start_m2a1e1b1nwzida0e0b0xyg, mei, mew_min_a1e1b1nwzida0e0b0xyg, d_cfw_ave_a1e1b1nwzida0e0b0xyg, sfd_a0e0b0xyg, wge_a0e0b0xyg
            , af_wool_a1e1b1nwzida0e0b0xyg, dlf_eff_a1e1b1nwzida0e0b0xyg,  kw_yg, days_period_a1e1b1nwzida0e0b0xyg
            , mec=0, mel=0, gest_propn_a1e1b1nwzida0e0b0xyg=0, lact_propn_a1e1b1nwzida0e0b0xyg=0):
    ##ME available for wool growth
    mew_xs_a1e1b1nwzida0e0b0xyg = np.maximum(mew_min_a1e1b1nwzida0e0b0xyg * relsize_start, mei - (mec * gest_propn_a1e1b1nwzida0e0b0xyg + mel * lact_propn_a1e1b1nwzida0e0b0xyg))
    ##Wool growth (protein weight) wo (without) lag
    d_cfw_wolag_a1e1b1nwzida0e0b0xyg = cw[8, ...] * wge_a0e0b0xyg * af_wool_a1e1b1nwzida0e0b0xyg * dlf_eff_a1e1b1nwzida0e0b0xyg * mew_xs_a1e1b1nwzida0e0b0xyg
    ##Wool growth (protein weight) with and wo lag
    d_cfw_a1e1b1nwzida0e0b0xyg, d_cfw_history_m2a1e1b1nwzida0e0b0xyg = f_history(d_cfw_history_start_m2a1e1b1nwzida0e0b0xyg, d_cfw_wolag_a1e1b1nwzida0e0b0xyg, days_period_a1e1b1nwzida0e0b0xyg)
    ##Net energy required for wool
    new = cw[1, ...] * (d_cfw_a1e1b1nwzida0e0b0xyg - cw[2, ...] * relsize_start) / cw[3, ...]
    ##ME required for wool (above basal)
    mew = new / kw_yg
    ##Fibre diameter for the days growth
    d_fd_a1e1b1nwzida0e0b0xyg = sfd_a0e0b0xyg * (d_cfw_a1e1b1nwzida0e0b0xyg / d_cfw_ave_a1e1b1nwzida0e0b0xyg) ** cw[13, ...]
    ##Surface Area
    area = cc[1, ...] * ffcfw_start ** (2/3)
    ##Daily fibre length growth
    d_fl_a1e1b1nwzida0e0b0xyg = 400 * d_cfw_a1e1b1nwzida0e0b0xyg / (np.pi * cw[10, ...] * cw[11, ...] * area * (d_fd_a1e1b1nwzida0e0b0xyg / 10**6) ** 2)
    return d_cfw_a1e1b1nwzida0e0b0xyg, d_fd_a1e1b1nwzida0e0b0xyg, d_fl_a1e1b1nwzida0e0b0xyg, d_cfw_history_m2a1e1b1nwzida0e0b0xyg, mew, new



def f_chill_cs(cc, ck, ffcfw_start, rc_start, fl_start, mei, meme, mew, new, km, kg_supp, kg_fodd, mei_propn_supp, mei_propn_herb, temp_ave_a1e1b1nwzida0e0b0xyg, temp_max_a1e1b1nwzida0e0b0xyg, temp_min_a1e1b1nwzida0e0b0xyg, ws_a1e1b1nwzida0e0b0xyg, rain_a1e1b1nwzida0e0b0xygm1, index_m0, guw	= 0, kl = 0,	mei_propn_milk	= 0, mec = 0, mel = 0, nec = 0, nel = 0, gest_propn	= 0, lact_propn = 0):
    ##Animal is below maintenance
    belowmaint = mei < (meme + mec + mel + mew)
    ##Efficiency for growth (before ECold)
    kge = f_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk)
    ##Sinusoidal variation in temp & wind
    sin_var_m0 = np.sin(2 * np.pi / 12 *(index_m0 - 3))
    ##Ambient temp (2 hourly)
    temperature_a1e1b1nwzida0e0b0xygm0 = temp_ave_a1e1b1nwzida0e0b0xyg[..., na] + (temp_max_a1e1b1nwzida0e0b0xyg[..., na] - temp_min_a1e1b1nwzida0e0b0xyg[..., na]) / 2 * sin_var_m0
    ##Wind velocity (2 hourly)
    wind_a1e1b1nwzida0e0b0xygm0 = ws_a1e1b1nwzida0e0b0xyg[..., na] * (1 + 0.35 * sin_var_m0)
    ##Proportion of sky that is clear
    sky_clear_a1e1b1nwzida0e0b0xygm1 = 0.7 * np.exp(-0.25 * rain_a1e1b1nwzida0e0b0xygm1)
    ##radius of animal
    radius = cc[2, ...] * ffcfw_start ** (1/3)
    ##surface area of animal
    area = cc[1, ...] * ffcfw_start ** (2/3)
    ##Impact of wet fleece on insulation
    wetflc_a1e1b1nwzida0e0b0xygm1 = cc[5, ..., na] + (1 - cc[5, ..., na]) * np.exp(-cc[6, ..., na] * rain_a1e1b1nwzida0e0b0xygm1 / fl_start[..., na])
    ##Insulation of air (2 hourly)
    in_air_a1e1b1nwzida0e0b0xygm0 = radius[..., na] / (radius[..., na] + fl_start[..., na]) / (cc[7, ..., na] + cc[8, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of coat (2 hourly)
    in_coat_a1e1b1nwzida0e0b0xygm0 = radius[..., na] * np.log((radius[..., na] + fl_start[..., na]) / radius[..., na]) / (cc[9, ..., na] - cc[10, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of  tissue
    in_tissue = cc[3, ...] * (rc_start - cc[4, ...] * (rc_start - 1))
    ##Insulation of  air + coat (2 hourly)
    in_ext_a1e1b1nwzida0e0b0xygm0m1 = wetflc_a1e1b1nwzida0e0b0xygm1[..., na, :] * (in_air_a1e1b1nwzida0e0b0xygm0[..., na] + in_coat_a1e1b1nwzida0e0b0xygm0[..., na])
    ##Impact of clear night skies on ME loss
    sky_temp_a1e1b1nwzida0e0b0xygm0m1 = sky_clear_a1e1b1nwzida0e0b0xygm1[..., na, :] * cc[13,..., na, na] * np.exp(-cc[14, ..., na, na] * np.minimum(0, cc[15, ..., na, na] - temperature_a1e1b1nwzida0e0b0xygm0[..., na]) ** 2)
    ##Heat production per m2
    heat = ((mei - nec * gest_propn - nel * lact_propn - new - kge * (mei
            - (meme + mec * gest_propn + mel * lact_propn + mew))
            + cc[16, ...] * guw) / area)
    ##Lower critical temperature (2 hourly)
    temp_lc_a1e1b1nwzida0e0b0xygm0m1 = cc[11, ..., na, na]+ cc[12, ..., na, na] - heat[..., na, na] * (in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0m1) + sky_temp_a1e1b1nwzida0e0b0xygm0m1
    ##Lower critical temperature (period)
    temp_lc_a1e1b1nwzida0e0b0xyg = np.average(temp_lc_a1e1b1nwzida0e0b0xygm0m1, axis = (-1,-2))
    ##Extra ME required to keep warm
    mecold_a1e1b1nwzida0e0b0xyg = area * np.average(f_dim(temp_lc_a1e1b1nwzida0e0b0xygm0m1, temperature_a1e1b1nwzida0e0b0xygm0[..., na]) /(in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0m1), axis = (-1,-2))
    ##ME requirement for maintenance (inc ECold)
    mem = meme + mecold_a1e1b1nwzida0e0b0xyg
    ##Animal is below maintenance (incl ecold)
    belowmaint = mei < (mem + mec + mel + mew)
    ##Efficiency for growth (inc ECold) -different to the second line because belowmaint includes ecold
    kg = f_kg(ck, belowmaint, kl, km, kg_supp, kg_fodd, mei_propn_supp,
              mei_propn_herb, mei_propn_milk)
    return mem, temp_lc_a1e1b1nwzida0e0b0xyg, kg


def f_lwc_cs(cg, rc_start, mei, mem, mew, z1f, z2f, kg, mec = 0,
              mel = 0, gest_propn = 0, lact_propn = 0):
    ##Level of feeding (maint = 0)
    level = (mei /  (mem + mec * gest_propn + mel * lact_propn + mew)) - 1
    ##Net energy gain (based on ME)
    neg = kg * (mei - (mem + mec * gest_propn + mel * lact_propn + mew))
    ##Energy Value of gain
    evg = cg[8, ...] - z1f * (cg[9, ...] - cg[10, ...] * (level - 1)) + z2f * cg[11, ...] * (rc_start - 1)
    ##Protein content of gain
    pcg = cg[12, ...] - z1f * (cg[13, ...] - cg[14, ...] * (level - 1)) + z2f * cg[15, ...] * (rc_start - 1)
    ##Empty bodyweight gain
    ebg = neg / evg
    ##Protein gain
    pg = pcg * ebg
    ##fat gain
    fg = (neg - pg * cg[21, ...]) / cg[22, ...]
    return ebg, evg, pg, fg, level


def f_emissions_bc(ch, intake_f, intake_s, md_solid, level):
    ##Methane production total
    ch4_total = ch[1, ...] * (intake_f + intake_s)*((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))
    ##Methane production animal component
    ch4_animal = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)
    return ch4_total, ch4_animal





def f_feedsupply(cu3, cu4, cr, feedsupply_std_a1e1b1nwzida0e0b0xyg, paststd_foo_a1e1b1j0wzida0e0b0xyg, paststd_dmd_a1e1b1j0wzida0e0b0xyg, legume_a1e1b1nwzida0e0b0xyg, pi, pasture_stage_a1e1b1j0wzida0e0b0xyg, i_hr_scalar, i_region, i_n_pasture_stage, i_hd_std):
    ##level of pasture
    level_a1e1b1nwzida0e0b0xyg = np.trunc(np.minimum(2, feedsupply_std_a1e1b1nwzida0e0b0xyg)).astype('int') #note np.trunc rounds down to the nerest int (need to specify int type for the take along axis functin below)
    ##next level up of pasture
    next_level_a1e1b1nwzida0e0b0xyg = np.minimum(2, level_a1e1b1nwzida0e0b0xyg + 1)
    ##decimal component of feedsupply
    proportion_a1e1b1nwzida0e0b0xyg = feedsupply_std_a1e1b1nwzida0e0b0xyg % 1
    ##foo as measured
    paststd_foo_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    paststd_foo_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_a1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    foo_a1e1b1nwzida0e0b0xyg = paststd_foo_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_foo_next_a1e1b1nwzida0e0b0xyg - paststd_foo_a1e1b1nwzida0e0b0xyg)
    ##foo corrected to hand shears and estimated height
    foo, hf = f_foo_convert(cu3, cu4, foo_a1e1b1nwzida0e0b0xyg, i_hr_scalar,i_region, i_n_pasture_stage,i_hd_std, legume_a1e1b1nwzida0e0b0xyg, pasture_stage_a1e1b1j0wzida0e0b0xyg, cr)
    ##dmd
    paststd_dmd_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    paststd_dmd_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_a1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    dmd_a1e1b1nwzida0e0b0xyg = paststd_dmd_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_dmd_next_a1e1b1nwzida0e0b0xyg - paststd_dmd_a1e1b1nwzida0e0b0xyg)
    ##proportion of PI that is offered as supp
    supp_propn_a1e1b1nwzida0e0b0xyg = proportion_a1e1b1nwzida0e0b0xyg * (feedsupply_std_a1e1b1nwzida0e0b0xyg > 2) + (feedsupply_std_a1e1b1nwzida0e0b0xyg == 4)   # the proportion of diet if the value is above 2 and equal to 1.0 if fs==4
    intake_s = pi * supp_propn_a1e1b1nwzida0e0b0xyg
    ##calc herb md
    herb_md = fun.dmd_to_md(dmd_a1e1b1nwzida0e0b0xyg)
    return foo_a1e1b1nwzida0e0b0xyg, hf, dmd_a1e1b1nwzida0e0b0xyg, intake_s, herb_md




def f_conception_cs(cf, cb1, relsize_mating, rc_mating, crg_doy, period_is_mating):
    ##Conception greater than or equal to 1,2,3 foetus (what is chance you have more than x number of feotuses)
    relsize_mating_e1b1sliced = f_dynamic_slice(relsize_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], -1, None) #take slice from e1 & b1 axis
    rc_mating_e1b1sliced = f_dynamic_slice(rc_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], -1, None) #take slice from e1 & b1 axis
    crg = crg_doy * f_sig(relsize_mating_e1b1sliced * rc_mating_e1b1sliced, cb1[2, ...], cb1[3, ...])
    ##Define the temp array shape
    cr_temp = crg
    ##Conception equal to x (temporary array as if this period is joining)
    slc = [slice(None)] * len(cr_temp.shape)
    slc[uinp.parameters['i_b1_pos']] = slice(0,-1)
    cr_temp[tuple(slc)] = np.maximum(0, f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 0, -1) - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, None))    # (difference between '>x' and '>x+1')
    ##Dams that don't retain to 3rd trimester but do not return to service are added to 00 slice rather than staying in NM slice	
    slc[uinp.parameters['i_b1_pos']] = slice(0,1)
    cr_temp[tuple(slc)] = np.minimum(1 - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, 2), f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, 2)) * (cf[5, ...] / (1 - cf[5, ...]))
    ##Proportion of animals with conception equal to x (if this period is mating)	
    conception = cr_temp * period_is_mating
    ##Number remaining not-mated (cr[-1])
    slc = [slice(None)] * len(conception.shape)
    slc[pinp.sheep['i_e1_pos']] = slice(0,1)
    slc[uinp.parameters['i_b1_pos']] = slice(-1,None)
    conception[tuple(slc)] = -np.sum(f_dynamic_slice(conception, uinp.parameters['i_b1_pos'],0, 4), axis = (pinp.sheep['i_e1_pos'], uinp.parameters['i_b1_pos']), keepdims=True)  #slice 0 to 4 - dont want to include slices where progeny lost eg 21 (so we dont double count conception)
    return conception

def f_conception_ltw(cu0, cs_mating, scan_std, doy_p, period_is_mating):
    ##Slope of the RR vs CS relationship	
    slope = np.maximum(cu0[4, ...], cu0[2, ...] + np.sin(2 * np.pi * doy_p / 365) * cu0[3, ...])
    ##Reproduction rate
    cs_mating_e1b1sliced = f_dynamic_slice(cs_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], -1, None) #take slice from e1 & b1 axis (take slice from not mated to get cs because they are about to be mated)
    repro_rate = scan_std + (cs_mating_e1b1sliced - 3) * slope
    ###remove b1 axis by squeezing
    repro_rate = np.squeeze(repro_rate, axis=uinp.parameters['i_b1_pos'])
    ##Conception - propn dry/single/twin for given repro rate
    conception = f_DSTw(repro_rate) * period_is_mating
    conception = np.moveaxis(conception[...,uinp.structure['a_nfoet_b1']], -1, uinp.parameters['i_b1_pos']) #move the l0 axis into the b1 position. and expand to b1 size.
    ##Number remaining not-mated (cr[-1])	
    slc = [slice(None)] * len(conception.shape)
    slc[pinp.sheep['i_e1_pos']] = slice(0,1)
    slc[uinp.parameters['i_b1_pos']] = slice(-1,None)
    conception[slc] = -np.sum(f_dynamic_slice(conception, uinp.parameters['i_b1_pos'],0, 4), axis = (pinp.sheep['i_e1_pos'], uinp.parameters['i_b1_pos']), keepdims=True)
    return conception


def f_mortality_base(cd, cg, rc_start, ebg_start, d_nw_max):
    return cd[1, ...] + cd[2, ...] * np.maximum(0, cd[3, ...] - rc_start) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))
 

def f_sire_req(sire_propn_a1e1b1nwzida0e0b0xyg1, sire_periods_g0p8, i_sire_recovery, i_startyear, date_end_p, period_is_prejoin_a1e1b1nwzida0e0b0xyg1):
    ##Date at end of period adjusted to start year
    t_date_end_a1e1b1nwzida0e0b0xyg = date_end_p - (365 * (date_end_p.astype('datetime64[Y]').astype(int) + 1970 - i_startyear)).astype('timedelta64[D]')
    ##Date_end falls within the ram mating periods
    sire_required_a1e1b1nwzida0e0b0xyg0g1p8 = np.logical_and(t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] >= sire_periods_g0p8[:,na,:].astype('datetime64[D]') , t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] <= (sire_periods_g0p8[:,na,:].astype('datetime64[D]') + i_sire_recovery)) #add axis for p8 and g1
    ##Number of rams required per ewe (if this period is joining)
    n_sires = sire_required_a1e1b1nwzida0e0b0xyg0g1p8 * sire_propn_a1e1b1nwzida0e0b0xyg1[..., na,:,na] * period_is_prejoin_a1e1b1nwzida0e0b0xyg1[..., na,:,na] #add axis for g1 and p8
    return n_sires




def f_mortality_weaner_cs(cd, cg, age, ebg_start, d_nw_max):
    return cd[13, ...] * f_ramp(age, cd[15, ...], cd[14, ...]) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))


def f_mortality_dam_cs(cb1, cg, nw_start, ebg, days_period, period_between_birth6wks, gest_propn, sar_mortalitye):
    ##(Twin) Dam mortality in last 6 weeks (preg tox)
    t_mort = days_period * gest_propn /42 * f_sig(-42 * ebg * cg[18, ...] / nw_start, cb1[4, ...], cb1[5, ...])
    ##If not last 6 weeks then = 0
    mort = t_mort * period_between_birth6wks
    mort = f_sa(mort, sar_mortalitye, sa_type = 4)
    return mort

    
def f_mortality_dam_mu(cu2, cs_birth_dams, period_is_birth, sar_mortalitye):
    ##(Twin) Dam mortality in last 6 weeks (preg tox)	
    t_mortalitye_mu = cu2[22, 0, ...] * cs_birth_dams + cu2[22, 1, ...] * cs_birth_dams ** 2 + cu2[22, -1, ...]
    ##Non-multiple bearing ewes = 0	
    mortalitye_mu = np.exp(t_mortalitye_mu) / (1 + np.exp(t_mortalitye_mu)) * period_is_birth
    ##Dam (& progeny) losses at birth related to CSL	
    mortalitye_mu = f_sa(mortalitye_mu, sar_mortalitye, sa_type = 4)
    return mortalitye_mu


    
def f_mortality_progeny_cs(cd, cb1, w_b, rc_birth, w_b_exp_y, period_is_birth, chill_index_m1, nfoet_b1, sar_mortalityp):
    ##Progeny losses due to large progeny (dystocia)
    mortalityd_yatf = f_sig(w_b / w_b_exp_y * np.maximum(1, rc_birth), cb1[6, ...], cb1[7, ...]) * period_is_birth
    ##add sensitivity
    mortalityd_yatf = f_sa(mortalityd_yatf, sar_mortalityp, sa_type = 4)
    ##dam mort due to large progeny (dystocia)
    mortalityd_dams = np.mean(mortalityd_yatf, axis=uinp.parameters['i_x_pos'], keepdims=True) * cd[21,...] / nfoet_b1
    ##Progeny losses due to large progeny (dystocia) - so there is no double counting of progeny loses associated with dam mortality
    mortalityd_yatf = mortalityd_yatf * (1- cd[21,...])
    ##Exposure index
    xo = cd[8, ..., na] - cd[9, ..., na] * rc_birth[..., na] + cd[10, ..., na] * chill_index_m1 + cb1[11, ..., na]
    ##Progeny mortality at birth from exposure
    mortalityx = np.average(np.exp(xo) / (1 - np.exp(xo)) ,axis = -1) * period_is_birth #axis -1 is m1
    ##add sensitivity
    mortalityx = f_sa(mortalityx, sar_mortalityp, sa_type = 4)
    return mortalityx, mortalityd_yatf, mortalityd_dams


def f_mortality_progeny_mu(cu2, cb1, cx, ce, w_b, foo, chill_index_m1, period_is_birth, sar_mortalityp):
    ##transformed survival	
    t_mortalityp_mu = cu2[8, 0, ..., na] * w_b[..., na] + cu2[8, 1, ..., na] * w_b[..., na] ** 2 + cu2[8, 2, ..., na] * chill_index_m1 + cu2[8, 3, ..., na] * foo[..., na] + cu2[8, 4, ..., na] * foo[..., na] ** 2 + cu2[8, 5, ..., na] + cb1[8, ..., na] + cx[8, ..., na] + cx[9, ..., na] * chill_index_m1 + ce[8, ..., na]
    ##back tansformed	
    mortalityp_mu = np.average(1 / (1 + np.exp(-t_mortalityp_mu)),axis = -1) * period_is_birth #m1 axis averaged
    ##Progeny mortality at birth (LTW) with SA	
    mortalityp_mu = f_sa(mortalityp_mu, sar_mortalityp, sa_type = 4)
    return mortalityp_mu
        




def f_comb(n,k):
    ##Create an array of factorial values up to n	
    factorial = np.cumprod(np.arange(np.max(n))+1)
    ##Combination	
    combinations = factorial[n-1]/(factorial[k-1]*factorial[n-k-1])
    return combinations




def f_period_start_prod(numbers, var, prejoin_tup, season_tup, period_is_startfvp, period_is_break, period_is_prejoin=0, group=None):
    ##Set variable level = value at end of previous	
    var_start = var 
    ##make sure numbers and var are same shape - this is required for the np.average func below
    numbers, var_start = np.broadcast_arrays(numbers,var_start)
    ##a) Calculate temporary values as if start of FVP
    temporary = var_start #this is done to ensure that temp has the same size as var. In the next line np.diagonal removes the n axis so it is added back in using the expand function, but that is a singlton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
    temporary[...] = np.expand_dims(np.rollaxis(var_start.diagonal(axis1= uinp.structure['i_w_pos'], axis2= uinp.structure['i_n_pos']),-1,uinp.structure['i_w_pos']), uinp.structure['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
    ##Update if the period is start of a FVP	
    var_start = f_update(var_start, temporary, period_is_startfvp)
    ##b) Calculate temporary values as if period_is_break
    temporary = np.expand_dims(np.average(var_start, axis = season_tup, weights=numbers),season_tup) #gets the weighted average of production in the different seasons, have to add axis back because no keepdims arg
    ##Set values where it is beginning of FVP	
    var_start = f_update(var_start, temporary, period_is_break)
    if group==1:
        ##c) Calculate temporary values as if period_is_prejoin	
        temporary = np.expand_dims(np.average(var_start, axis = prejoin_tup, weights=numbers),prejoin_tup) #gets the weighted average of production in the different seasons, have to add axis back because no keepdims arg
        ##Set values where it is beginning of FVP	
        var_start = f_update(var_start, temporary, period_is_prejoin)
    return var_start




def f_period_start_nums(numbers, prejoin_tup, season_tup, period_is_startfvp, period_is_break, season_propn_z, group=None, numbers_initial_repro=0, period_is_prejoin=None):
    ##a) reallocate between w and n if the period is start of a FVP
    ###Calculate temporary values as if start of FVP - colapse n back to standard level (n axis is populated due to mortality)
    temporary = numbers #this is done to ensure that temp has the same size as var. In the next line np.diagonal removes the n axis so it is added back in using the expand function, but that is a singlton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
    temporary[...] = np.expand_dims(np.rollaxis(numbers.diagonal(axis1= uinp.structure['i_w_pos'], axis2= uinp.structure['i_n_pos']),-1,uinp.structure['i_w_pos']), uinp.structure['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
    numbers = f_update(numbers, temporary, period_is_startfvp)
    ##b) realocate for season type
    temporary = np.sum(numbers, axis = season_tup, keepdims=True)  * season_propn_z  #Calculate temporary values as if period_is_break
    numbers = f_update(numbers, temporary, period_is_break)  #Set values where it is beginning of FVP	
    ##things for dams - prejoining and moving between classes
    if group==1:    
        ##d) new repro cycle (prejoining)
        temporary = np.sum(numbers, axis = (prejoin_tup), keepdims=True) * numbers_initial_repro #Calculate temporary values as if period_is_prejoin
        numbers = f_update(numbers, temporary, period_is_prejoin)  #Set values where it is beginning of FVP	
    return numbers


def f_period_end_nums(numbers, mortality, mortality_yatf=0, nfoet_b1 = 0, nyatf_b1 = 0, group=None, conception = 0, scan=0, gbal=0, period_is_mating = False, period_is_birth=False, period_is_scan=False):
    '''
    This adjusts numbers for things like conception and mortality that happen during a given period
    '''
    ##a) mortality
    numbers = numbers * (1-mortality)
    ##things for dams - prejoining and moving between classes
    if group==1:    
        ###b) conception - conception is the change in numbers +ve for animals getting pregnancy and -ve in the NM slice
        temporary = numbers + conception * numbers[:, 0:1, -1:, ...]  # numbers_dams[..., 0,-1, ...] is the NM slice of cycle 0 ie the number of animals yet to be mated (conception will have negitive value in nm slice)
        numbers = f_update(numbers, temporary, period_is_mating) #needds to be previous period else conception is not calculated because numbers happens at begining of p loop
        ###c) scanning
        temp = np.maximum(pinp.sheep['i_drysretained_scan'],np.minimum(1, nfoet_b1)) * numbers # scale the level of drys by drys_retained, scale every other slice by 1 except drys if not retained
        numbers = f_update(numbers, temp, period_is_scan * (scan>=1))
        ###d) birth (account for birth status and if drys are retained)
        dam_propn_birth_b1 = f_comb(nfoet_b1, nyatf_b1) * (1 - mortality_yatf) ** nyatf_b1 * mortality_yatf ** (nfoet_b1 - nyatf_b1) # the proportion of dams of each LSLN based on (progeny) mortality
        temp = np.mean(dam_propn_birth_b1, axis=uinp.parameters['i_x_pos'], keepdims=True) * numbers[:,:,uinp.structure['a_nfoet_b1'],...] #have to average x axis so that it is not active for dams
        numbers = f_update(numbers, temp, period_is_birth)  # calculated in the period after birth when progeny mortality due to exposure is calculated
        temp = np.maximum(pinp.sheep['i_drysretained_birth'],np.minimum(1, nyatf_b1)) * numbers
        numbers = f_update(numbers, temp, period_is_birth * (gbal>=1)) # has to happen after the dams are moved due to progeny mortality so that gbal drys are also scaled by drys_retained
    ##things just for yatf
    if group==2:
        temp = nyatf_b1   # nyatf is accounting for peri-natal mortality
        f_update(numbers, temp, period_is_birth)
    return numbers


def f_sa(value, sa, sa_type=0, target=0):
    ##Type 0 is sam (sensitivity multiplier) - default
    if sa_type == 0:
        result = value * sa
    ##Type 1 is saa (sensitivity addition)
    elif sa_type == 1:
         result = value + sa
    ##Type 2 is sap (sensitivity proportion)
    elif sa_type == 2:
         result = value * (1 + sa)
    ##Type 3 is sat (sensitivity target)
    elif sa_type == 3:
         result = value + (target - value) * sa
    ##Type 4 is sar (sensitivity range)
    elif sa_type == 4:
         result = np.maximum(0, np.minimum(1, value * (1 - np.abs(value)) + np.maximum(0, sa)))
    ##Type 5 is value (return the SA value)
    elif sa_type == 5:
         result = sa
    return result


    
def f_carryforward_u1(cu1, ebg, period_between_joinstartend, period_between_joinscan, period_between_scanbirth, period_between_birthwean, days_period, period_propn):
    ##First 3 slices of the genotype axis = the sire genotypes	
    coeff_cf1 = f_update(0, cu1[1,...], period_between_joinstartend) #note cu1 has already had the first axis (animal) sliced when it was passed in 
    ##Loop over remaining slices	
    coeff_cf1 = f_update(coeff_cf1, cu1[2,...], period_between_joinscan)
    ##Loop over remaining slices	
    coeff_cf1 = f_update(coeff_cf1, cu1[3,...], period_between_scanbirth)
    ##Loop over remaining slices	
    coeff_cf1 = f_update(coeff_cf1, cu1[4,...], period_between_birthwean)
    ##Assign values based on maternal and paternal genotype	
    d_cf = coeff_cf1 * ebg * days_period * period_propn
    return d_cf






