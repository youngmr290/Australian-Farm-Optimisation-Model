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
import time


# from dateutil.relativedelta import relativedelta

import Functions as fun
import PropertyInputs as pinp
import UniversalInputs as uinp
import Sensitivity as sen

na=np.newaxis

def f_sig(x,a,b):
    ''' Sig function CSIRO equation 124 ^the equation below is the sig function from SheepExplorer'''
    return  1/(1+np.exp(-((2*(np.log(0.95) - np.log(0.05))/(b-a))*(x-(a+b)/2))))

def f_ramp(x,a,b):
    ''' RAMP function CSIRO equation 125a'''
    return  np.minimum(1,np.maximum(0,(a-x)/(a-b)))

def f_dim(x,y):
    '''a function that minimum value of zero otherwise difference between the 2 inputs '''
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
#         age of animal at the beginning of each period.

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

def f_next_prev_association(datearray_slice,*args):
    '''
    Depending on the inputs this function will return the next or previous association.
    eg it can be used to determine the next lambing opportunity for each period.
    See john stuff.py for alternative methods.

    Parameters
    ----------
    datearray_slice : any
        This is 1d array which is the second array is sorted into (this must be sorted).
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
    side=args[2]
    idx_next = np.searchsorted(datearray_slice, date,side)
    idx = np.clip(idx_next - offset, 0, len(datearray_slice)-1) #makes the max value equal to the length of joining array, because if the period date is after the last lambing opportunity there is no 'next'
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
    index_p = np.arange(n_sim_periods)
    date_start_p =  (np.datetime64(start_date) + (step * index_p)).astype('datetime64[D]') #astype day rounds the date to the nearest day
    date_end_p = (np.datetime64(start_date - dt.timedelta(days=1)) + (step * (index_p+1))).astype('datetime64[D]') #minus one day to get the last day in the period not the first day of the next period.
    return n_sim_periods, date_start_p, date_end_p, index_p, step



###################################
#input and manipulation functions #
###################################

def f_c2g(params_c2, y=0, var_pos=0, len_ax1=0, len_ax2=0, condition=None, axis=0, dtype=False):
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
    params_c2 = params_c2.astype(float) #this is so that blank cells are converted to nan not none type because none type cant be multiplied etc
    params_c0 = params_c2[...,a_c2_c0]
    ##add y axis
    na=np.newaxis
    ###if y is not numpy ie was read in as an int because it was a single cell, it needs to be converted
    if type(y) == int:
        y = np.asarray([y])
    ###y is a 2d array however currently it only has one slice so it is read in as a 1d array. so i need to add second array
    if y.ndim == 1 and params_c0.ndim != 1:
        y=y[...,na]
    ###apply y mask
    y=y[...,uinp.parameters['i_mask_y']]
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
        pass#don't need to reshape
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
    ##apply mask if required
    if condition is not None: #see if condition exists
        if type(condition) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition= np.asarray([condition]) #convert to numpy if it is singular input
        ###apply mask
        param_sire = np.compress(condition, param_sire, axis)
        param_dams = np.compress(condition, param_dams, axis)
        param_yatf = np.compress(condition, param_yatf, axis)
        param_offs = np.compress(condition, param_offs, axis)
    ##assign dtype
    if dtype:
        param_sire = param_sire.astype(dtype)
        param_dams = param_dams.astype(dtype)
        param_yatf = param_yatf.astype(dtype)
        param_offs = param_offs.astype(dtype)

    return param_sire, param_dams, param_yatf, param_offs


def f_g2g(array_g,group,left_pos=0,len_ax1=0,len_ax2=0,len_ax3=0,swap=False,right_pos=-1,left_pos2=0,right_pos2=-1, condition = None, axis = 0, condition2 = None, axis2 = 0):
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

    *note: if adding two sets of new axis add from right to left (then the pos variables align)

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
        pass#don't need to reshape
    ##swap axis if necessary
    if swap:
        array_g = np.swapaxes(array_g, 0, 1)
    ##get axis into correct position 1
    if left_pos != None or left_pos != 0:
        extra_axes = tuple(range((left_pos + 1), right_pos))
    else: extra_axes = ()
    array_g = np.expand_dims(array_g, axis = extra_axes)

    ##get axis into correct position 2 (some arrays need singleton axis added in multiple places ie separated by a used axis)
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
    if condition2 is not None: #see if condition exists
        if type(condition2) == bool: #check if array or single value - note array of T & F is not type bool (it is array)
            condition2= np.asarray([condition2]) #convert to numpy if it is singular input
            array = np.compress(condition2, array, axis2)
        else:
            array = np.compress(condition2, array, axis2)
    return array


def f_DSTw(scan_std_yg):
    '''
    Parameters
    ----------
    scan_std_yg : np array
        scanning percentage of genotypes.

    Returns
    -------
    Proportion of dry, single, twins & triplets. Numpy way of making this formula: y=int+ax+bx^2+cx^3+dx^4

    '''
    scan_powers_s = uinp.sheep['i_scan_powers']  #scan powers are the exponential powers used in the quadratic formula ie ^0, ^1, ^2, ^3, ^4
    scan_power_ygs = scan_std_yg[...,na] ** scan_powers_s #puts the variable to the powers ie x^0, x^1, x^2, x^3, x^4
    dstwtr_ygl0 = np.sum(uinp.sheep['i_scan_coeff_l0s'] * scan_power_ygs[...,na,:], axis = -1) #add the coefficients and sum all the elements of the equation ie int+ax+bx^2+cx^3+dx^4
    return dstwtr_ygl0

def f_btrt0(dstwtr,lss,lstw,lstr): #^this function is inflexible ie if you want to add quadruplets
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
        period_is_between= np.logical_and((date_array<=date_end_p) , (date_array2>date_start_p))
        return period_is_between


################
#Sim functions #
################
def f_feedsupply_adjust(attempts,feedsupply,itn):
    ##create empty array to put new feedsupply into, this is done so it doesnt have the itn axis (probably could just create from attempts array shape without last axis)
    feedsupply = np.zeros_like(feedsupply)
    ##which feedsupplies can be calculated using binary method - must have a negative and positive error
    binary_mask = np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1) < 0 #axis -1 is the itn axis ie take the min and max error from the previous iterations
    ##calc new feedsupply binary - take half of the two feedsupplies that have resulted in the error closest to 0. Only adds the binary result to slices that have a negative and a positive value (done using the mask created above)
    ###feedsupply with negative error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]>0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    neg_bool=np.ma.getdata(mask_attempts.max(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negative but closest to 0
    neg_bool = neg_bool * binary_mask[...,na] #this just makes sure the neg mask only has a true in the same slice as the pos array (so it can be applied to the feed supply array below)
    ###feedsupply with positive error that is closest to 0 - this is a little complex because applying a max function to a masked array
    mask_attempts= np.ma.masked_array(attempts[...,1],attempts[...,1]<0) #np.ma has a true and false the other way around (eg false means keep data) therefore the <> sign is opposite to what you want
    pos_bool=np.ma.getdata(mask_attempts.min(axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negative but closest to 0
    pos_bool = pos_bool * binary_mask[...,na] #this just makes sure the pos mask only has a true in the same place as the neg mask. 
    ##calc feedsupply
    feedsupply[binary_mask] = (attempts[...,0][neg_bool] + attempts[...,0][pos_bool])/2    
    ##calc feedsupply using interpolation
    ###first determine the slope, slope is always positive ie as feedsupply increases error increase because error = lwc - target and more feed means higher lwc.
    if itn==0:
        slope=pinp.sheep['i_feedsupply_slope_std']
    else:
        ####linregress only works on 1d array and cant use apply_over_axis because needs x and y. maybe there is a better way but i looked for a while and found nothing
        slope=np.empty_like(feedsupply)
        feedsupply_all_itn = attempts[...,0]
        error_all_itn = attempts[...,1]
        for i in np.ndindex(error_all_itn.shape[:-1]): #not exactly sure how this is working but it is creating tuple of each combo of slices in each axis.
            x= feedsupply_all_itn[i] #indexing with tuple works correctly if we are interested in the last axis otherwise it doesn't work properly for some reason.ie t[(0,0)] == t[0,0,:] but t[:,(0,0)] != t[:,0,0]
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
    hd = fun.f_divide(height, foo_shears) #handles div0 (eg if in feedlot with no pasture or adjusted foo is less than 0)
    ##height ratio                    
    hr = i_hr_scalar * hd / i_hd_std
    ##calc hf
    hf = 1 + cr12 * (hr -1)
    return foo_shears, hf

def f_dynamic_slice(arr, axis, start, stop, axis2=None, start2=None, stop2=None):
    ##check if arr is int - this is the case for the first loop because arr may be initialised as 0
    if type(arr)==int:
        return arr
    else:
        ##first axis slice if it is not singleton
        if arr.shape[axis]!=1:
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice( start, stop)
            arr = arr[tuple(sl)]
        if axis2 is not None:
            ##second axis slice if required and not singleton
            if arr.shape[axis2] != 1:
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
    weights = history!=np.nan
    t_history = np.nan_to_num(history) #convert nan to 0
    lagged = fun.f_weighted_average(t_history, weights=weights, axis = 0)
    return lagged, history



def roll_slices(array, roll, roll_axis=0):
    '''
    The function rolls each slice for a given axis (the np roll function rolls each slice the same)
    you can roll different slices by different amounts
    :param array: array to be rolled
    :param roll: number of times the slice is to be rolled - this array should have one less dim than the main array
    :param roll_axis: axis to roll down
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










def f_potential_intake_cs(ci, cl, srw, relsize_start, rc_start, temp_lc_dams, temp_ave, temp_max, temp_min, rain_intake, rc_birth_start = 1, pi_age_y = 0, lb_start = 0, mp2=0,piyf=1,period_between_birthwean=1):
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
    pi = (pi - mp2 / cl[6, ...] * cl[25, ...]) * piyf
    ##Potential intake of pasture - young at foot only
    pi = pi * period_between_birthwean
    return np.maximum(0,pi)


def f_potential_intake_mu():
    pi = 1.4
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
    md_solid = fun.f_divide(mei_solid, intake_f + intake_s) #func to stop div/0 error if fs  = 3 (eg zero feed)
    ##ME intake total	
    mei = mei_solid + mp2
    ##Proportion of ME as milk	
    mei_propn_milk = fun.f_divide(mp2, mei) #func to stop div/0 error - if mei is 0 then so is numerator
    ##Proportion of ME as herbage	
    mei_propn_herb = fun.f_divide((mei_herb + mei_forage), mei) #func to stop div/0 error - if mei is 0 then so is numerator
    ##Proportion of ME as supp	
    mei_propn_supp = fun.f_divide(mei_supp, mei) #func to stop div/0 error - if mei is 0 then so is numerator
    return mei, mei_solid, intake_f, md_solid, mei_propn_milk, mei_propn_herb, mei_propn_supp


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
    kg_lact = fun.f_update(kg_lact , kg_lact_lose , belowmaint)
    ##Compile non-lactating
    kg_dry = fun.f_update(kg_dry , kg_dry_lose , belowmaint)
    ##Compile whole formula
    kg = lact_propn * kg_lact + (1 - lact_propn) * kg_dry
    return kg



def f_energy_cs(ck, cx, cm, lw_start, ffcfw_start, mr_age, mei, omer_history_start, days_period, md_solid, i_md_supp, md_herb, lgf_eff, dlf_eff, i_steepness, density, foo, feedsupply, intake_f, dmd, mei_propn_milk=0):
    ##Efficiency for maintenance	
    km = (ck[1, ...] + ck[2, ...] * md_solid) * (1-mei_propn_milk) + ck[3, ...] * mei_propn_milk
    ##Efficiency for lactation - dam only	
    kl =  ck[5, ...] + ck[6, ...] * md_solid
    ##Efficiency for growth (supplement)	
    kg_supp = ck[16, ...] * i_md_supp
    ##Efficiency for growth (fodder)	
    kg_fodd = ck[13, ...] * lgf_eff * (1+ ck[15, ...] * dlf_eff) * md_herb
    ##Energy required at maint for metabolism	
    emetab = cx[10, ...] * cm[2, ...] * ffcfw_start ** 0.75 * mr_age * (1 + cm[5, ...] * mei_propn_milk)
    ##Distance walked (horizontal equivalent)	
    distance = (1 + np.tan(np.deg2rad(i_steepness))) * np.minimum(1, cm[17, ...] / density) / (cm[8, ...] * foo + cm[9, ...])
    ##Set Distance walked to 0 if in confinement	
    distance = distance * (feedsupply < 3)
    ##Energy required for movement	
    emove = cm[16, ...] * distance * lw_start
    ##Energy required for grazing (chewing and walking around)
    egraze = cm[6, ...] * ffcfw_start * intake_f * (cm[7, ...] - dmd) + emove
    ##Energy associated with organ activity
    omer, omer_history = f_history(omer_history_start, cm[1, ...] * mei, days_period)
    ##ME requirement for maintenance (before ECold)
    meme = (emetab + egraze) / km + omer
    return meme, omer_history, km, kg_fodd, kg_supp, kl


# def f_foetus_cs(cp, cb1, kc, nfoet, relsize_start, rc_start, nec_cum_start, w_b_std_y, w_f_start, nw_f_start, nwf_age_f, guw_age_f, dce_age_f, days_period_f):
def f_foetus_cs(cp, cb1, kc, nfoet, relsize_start, rc_start, w_b_std_y, w_f_start, nw_f_start, nwf_age_f, guw_age_f, dce_age_f):
    ##expected normal birth weight with dam age adj.
    w_b_exp_y = (1 - cp[4, ...] * (1 - relsize_start)) * w_b_std_y
    ##Normal weight of foetus (mid period - dam calcs)	
    nw_f = w_b_exp_y * nwf_age_f
    ##change in normal weight of foetus	
    d_nw_f = nw_f - nw_f_start
    ##Proportion of normal foetal and birth weights	
    nwf_nwb = fun.f_divide(nw_f, w_b_std_y)
    ##Normal weight of individual conceptus (mid period)	
    nw_gu = cp[5, ...] * w_b_exp_y * guw_age_f
    ##Normal energy of individual conceptus (end of period)	
    normale_dgu = cp[8, ...] * cp[5, ...] * w_b_exp_y * dce_age_f
    # normale_dgu = cp[8, ...] * cp[5, ...] * w_b_exp_y * ce_age_f
    ##Condition factor on BW
    cfpreg = (rc_start - 1) * nwf_nwb
    ##change in foetus weight	
    d_w_f = d_nw_f *(1 + np.minimum(cfpreg, cfpreg * cb1[14, ...]))
    ##foetus weight (end of period)	
    w_f = w_f_start + d_w_f
    ##Weight of the gravid uterus (conceptus - mid period)	
    guw = nfoet * (nw_gu + (w_f - nw_f))
    ##Body condition of the foetus	
    rc_f = fun.f_divide(w_f, nw_f) #func to handle div0 error
    ##Cumulative ME required for conceptus	
    # nec_cum = nfoet * rc_f * normale_dgu
    nec = nfoet * rc_f * normale_dgu
    ##NE required for conceptus
    # nec = np.maximum(0,fun.f_divide(nec_cum - nec_cum_start, days_period_f))
    ##ME required for conceptus	
    mec = nec / kc
    # return w_f, nec_cum, mec, nec, w_b_exp_y, nw_f, guw
    return w_f, mec, nec, w_b_exp_y, nw_f, guw

def f_birthweight_cs(cx, w_b_yatf, w_f_dams, period_is_birth):
    ##set BW = foetal weight at end of period (if born)	
    t_w_b = w_f_dams * cx[15, ...] * period_is_birth
    ##update birth weight if it is birth period
    w_b_yatf = fun.f_update(w_b_yatf, t_w_b, period_is_birth)
    return w_b_yatf

def f_birthweight_mu(cu1_yatf, cb1_yatf, cx_yatf, ce_yatf, w_b, cf_w_b_dams, ffcfw_birth_dams, ebg_dams, days_period, gest_propn, period_between_joinscan, period_between_scanbirth, period_is_birth):
    ##Carry forward BW increment	
    d_cf_w_b = f_carryforward_u1(cu1_yatf[16, ...], ebg_dams, False, period_between_joinscan, period_between_scanbirth, False, days_period, gest_propn)
    ##Carry forward BW increment	
    cf_w_b_dams = cf_w_b_dams + d_cf_w_b
    ##set BW = foetal weight at end of period (if born)	
    t_w_b_yatf = (cf_w_b_dams + cu1_yatf[16, -1, ...] + cu1_yatf[16, 0, ...] * ffcfw_birth_dams + cb1_yatf[16, ...] + cx_yatf[16, ...] + ce_yatf[16, ...])
    ##Update w_b if it is birth	
    w_b = fun.f_update(w_b, t_w_b_yatf, period_is_birth)
    return w_b, cf_w_b_dams


def f_weanweight_cs(w_w_yatf, ffcfw_start_yatf, ebg_yatf, days_period, lact_propn, period_is_wean):
    ##set WWt = yatf weight at weaning	
    t_w_w = (ffcfw_start_yatf + ebg_yatf * days_period * lact_propn)
    ##update weaning weight if it is weaning period
    w_w_yatf = fun.f_update(w_w_yatf, t_w_w, period_is_wean)
    return w_w_yatf

def f_weanweight_mu(cu1_yatf, cb1_yatf, cx_yatf, ce_yatf, w_w, cf_w_w_dams, ffcfw_wean_dams, ebg_dams, foo, days_period, lact_propn, period_between_joinscan, period_between_scanbirth, period_between_birthwean, period_is_wean):
    ##Carry forward WWt increment	
    d_cf_w_w = f_carryforward_u1(cu1_yatf[17, ...], ebg_dams, False, period_between_joinscan, period_between_scanbirth, period_between_birthwean, days_period, lact_propn)
    ##Carry forward WWt increment	
    cf_w_w_dams = cf_w_w_dams + d_cf_w_w
    ##set WWt = yatf weight at weaning	
    t_w_w = (cf_w_w_dams + cu1_yatf[17, -1, ...] + cu1_yatf[17, 0, ...] * ffcfw_wean_dams + cu1_yatf[17, 5, ...] * foo + cu1_yatf[17, 6, ...] * foo** 2 + cb1_yatf[17, ...] + cx_yatf[17, ...] + ce_yatf[17, ...])
    ##Update w_w if it is weaning	
    w_w = fun.f_update(w_w, t_w_w, period_is_wean)
    return w_w, cf_w_w_dams

def f_milk(cl, srw, relsize_start, rc_birth_start, mei, meme, mew_min, rc_start, ffcfw75_exp_yatf, lb_start, ldr_start, age_yatf, mp_age_y,  mp2_age_y, i_x_pos, days_period_yatf, kl, lact_nut_effect):
    ##Max milk prodn based on dam rc birth
    mpmax = srw** 0.75 * relsize_start * rc_birth_start * lb_start * mp_age_y
    ##Excess ME available for milk	
    mel_xs = np.maximum(0, (mei - (meme + mew_min * relsize_start))) * cl[5, ...] * kl
    ##Excess ME as a ratio of mpmax
    milk_ratio = fun.f_divide(mel_xs, mpmax) #func stops div0 error - and milk ratio is later discarded because days period f = 0
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
    dr = fun.f_divide(mp2, mpmax) #div func stops div0 error - and milk ratio is later discarded because days period f = 0
    ##Lagged DR (lactation deficit)
    ldr = (ldr_start - dr) * (1 - cl[18, ...]) ** days_period_yatf + dr
    ##Loss of potential milk due to consistent under production	
    lb = lb_start - cl[17, ...] / cl[18, ...] * (1 - cl[18, ...]) * (1 - (1 - cl[18, ...]) ** days_period_yatf) * (ldr_start - dr)
    ##If early in lactation = 1	
    lb = lb * lact_nut_effect + ~lact_nut_effect
    return mp2, mel, nel, ldr, lb





def f_fibre(cw, cc, ffcfw_start, relsize_start, d_cfw_history_start_m2a1e1b1nwzida0e0b0xyg, mei, mew_min_a1e1b1nwzida0e0b0xyg, d_cfw_ave_a1e1b1nwzida0e0b0xyg, sfd_a0e0b0xyg, wge_a0e0b0xyg
            , af_wool_a1e1b1nwzida0e0b0xyg, dlf_wool_a1e1b1nwzida0e0b0xyg,  kw_yg, days_period_a1e1b1nwzida0e0b0xyg
            , mec=0, mel=0, gest_propn_a1e1b1nwzida0e0b0xyg=0, lact_propn_a1e1b1nwzida0e0b0xyg=0):
    ##ME available for wool growth
    mew_xs_a1e1b1nwzida0e0b0xyg = np.maximum(mew_min_a1e1b1nwzida0e0b0xyg * relsize_start, mei - (mec * gest_propn_a1e1b1nwzida0e0b0xyg + mel * lact_propn_a1e1b1nwzida0e0b0xyg))
    ##Wool growth (protein weight-as shorn i.e. not DM) wo (without) lag
    d_cfw_wolag_a1e1b1nwzida0e0b0xyg = cw[8, ...] * wge_a0e0b0xyg * af_wool_a1e1b1nwzida0e0b0xyg * dlf_wool_a1e1b1nwzida0e0b0xyg * mew_xs_a1e1b1nwzida0e0b0xyg
    ##Wool growth (protein weight) with and wo lag
    d_cfw_a1e1b1nwzida0e0b0xyg, d_cfw_history_m2a1e1b1nwzida0e0b0xyg = f_history(d_cfw_history_start_m2a1e1b1nwzida0e0b0xyg, d_cfw_wolag_a1e1b1nwzida0e0b0xyg, days_period_a1e1b1nwzida0e0b0xyg)
    ##Net energy required for wool
    new = cw[1, ...] * (d_cfw_a1e1b1nwzida0e0b0xyg - cw[2, ...] * relsize_start) / cw[3, ...]
    ##ME required for wool (above basal)
    mew = new / kw_yg #can be negative because mem assumes 4g of wool is grown therefore if less energy is used mew essentially gives the energy back.
    ##Fibre diameter for the days growth
    d_fd_a1e1b1nwzida0e0b0xyg = sfd_a0e0b0xyg * fun.f_divide(d_cfw_a1e1b1nwzida0e0b0xyg, d_cfw_ave_a1e1b1nwzida0e0b0xyg) ** cw[13, ...]  #func to stop div/0 error when d_cfw_ave=0 so does d_cfw (only have a 0 when day period = 0)
    ##Surface Area
    area = cc[1, ...] * ffcfw_start ** (2/3)
    ##Daily fibre length growth
    d_fl_a1e1b1nwzida0e0b0xyg = 100 * fun.f_divide(d_cfw_a1e1b1nwzida0e0b0xyg, cw[10, ...] * cw[11, ...] * area * np.pi * (0.5 * d_fd_a1e1b1nwzida0e0b0xyg / 10**6) ** 2) #func to stop div/0 error when d_fd=0 so does d_cfw
    return d_cfw_a1e1b1nwzida0e0b0xyg, d_fd_a1e1b1nwzida0e0b0xyg, d_fl_a1e1b1nwzida0e0b0xyg, d_cfw_history_m2a1e1b1nwzida0e0b0xyg, mew, new



def f_chill_cs(cc, ck, ffcfw_start, rc_start, sl_start, mei, meme, mew, new, km, kg_supp, kg_fodd, mei_propn_supp, mei_propn_herb, temp_ave_a1e1b1nwzida0e0b0xyg, temp_max_a1e1b1nwzida0e0b0xyg, temp_min_a1e1b1nwzida0e0b0xyg, ws_a1e1b1nwzida0e0b0xyg, rain_a1e1b1nwzida0e0b0xygm1, index_m0, guw	= 0, kl = 0, mei_propn_milk	= 0, mec = 0, mel = 0, nec = 0, nel = 0, gest_propn	= 0, lact_propn = 0):
    ##Animal is below maintenance
    belowmaint = mei < (meme + mec + mel + mew)
    ##Efficiency for growth (before ECold)
    kge = f_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
    ##Sinusoidal variation in temp & wind
    sin_var_m0 = np.sin(2 * np.pi / 12 *(index_m0 - 3))
    ##Ambient temp (2 hourly)
    temperature_a1e1b1nwzida0e0b0xygm0 = temp_ave_a1e1b1nwzida0e0b0xyg[..., na] + (temp_max_a1e1b1nwzida0e0b0xyg[..., na] - temp_min_a1e1b1nwzida0e0b0xyg[..., na]) / 2 * sin_var_m0
    ##Wind velocity (2 hourly)
    wind_a1e1b1nwzida0e0b0xygm0 = ws_a1e1b1nwzida0e0b0xyg[..., na] * (1 + 0.35 * sin_var_m0)
    ##Proportion of sky that is clear
    sky_clear_a1e1b1nwzida0e0b0xygm1 = 0.7 * np.exp(-0.25 * rain_a1e1b1nwzida0e0b0xygm1)
    ##radius of animal
    radius = np.maximum(0.001,cc[2, ...] * ffcfw_start ** (1/3)) #max because realistic values of radius can be small for lambs - stops div0 error
    ##surface area of animal
    area = np.maximum(0.001,cc[1, ...] * ffcfw_start ** (2/3)) #max because area is in m2 so realistic values of area can be small for lambs
    ##Impact of wet fleece on insulation
    wetflc_a1e1b1nwzida0e0b0xygm1 = cc[5, ..., na] + (1 - cc[5, ..., na]) * np.exp(-cc[6, ..., na] * rain_a1e1b1nwzida0e0b0xygm1 / sl_start[..., na])
    ##Insulation of air (2 hourly)
    in_air_a1e1b1nwzida0e0b0xygm0 = radius[..., na] / (radius[..., na] + sl_start[..., na]) / (cc[7, ..., na] + cc[8, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of coat (2 hourly)
    in_coat_a1e1b1nwzida0e0b0xygm0 = radius[..., na] * np.log((radius[..., na] + sl_start[..., na]) / radius[..., na]) / (cc[9, ..., na] - cc[10, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
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
    kg = f_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
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
    level_a1e1b1nwzida0e0b0xyg = np.trunc(np.minimum(2, feedsupply_std_a1e1b1nwzida0e0b0xyg)).astype('int') #note np.trunc rounds down to the nearest int (need to specify int type for the take along axis function below)
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
    supp_propn_a1e1b1nwzida0e0b0xyg = proportion_a1e1b1nwzida0e0b0xyg * (feedsupply_std_a1e1b1nwzida0e0b0xyg > 2) + (feedsupply_std_a1e1b1nwzida0e0b0xyg == 4)   # the proportion of diet if the value is above 2 and equal to 1.0 if fs==4 (at fs 3 sheep have 0 sup and 0 fodder at fs4 sheep have 100% of pi is sup)
    intake_s = pi * supp_propn_a1e1b1nwzida0e0b0xyg
    ##calc herb md
    herb_md = fun.dmd_to_md(dmd_a1e1b1nwzida0e0b0xyg)
    return foo, hf, dmd_a1e1b1nwzida0e0b0xyg, intake_s, herb_md




def f_conception_cs(cf, cb1, relsize_mating, rc_mating, crg_doy, nfoet_b1any, nyatf_b1any, period_is_mating, index_e1):
    ##Conception greater than or equal to 1,2,3 foetus (what is chance you have more than x number of foetuses)
    relsize_mating_e1b1sliced = f_dynamic_slice(relsize_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], 0, 1) #take slice from e1 & b1 axis
    rc_mating_e1b1sliced = f_dynamic_slice(rc_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], 0, 1) #take slice from e1 & b1 axis
    crg = crg_doy * f_sig(relsize_mating_e1b1sliced * rc_mating_e1b1sliced, cb1[2, ...], cb1[3, ...])
    ##Define the temp array shape
    t_cr = crg.copy()
    ##Conception equal to x (temporary array as if this period is joining)
    slc = [slice(None)] * len(t_cr.shape)
    slc[uinp.parameters['i_b1_pos']] = slice(1,4)
    t_cr[tuple(slc)] = np.maximum(0, f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, 4) - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 2, 5))    # (difference between '>x' and '>x+1')
    ##Dams that don't retain to 3rd trimester but do not return to service (because they got pregnant) are added to 00 slice rather than staying in NM slice
    slc[uinp.parameters['i_b1_pos']] = slice(1,2)
    t_cr[tuple(slc)] = np.minimum(1 - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 2, 3), f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 2, 3) * (cf[5, ...] / (1 - cf[5, ...])))
    ##Proportion of animals with conception equal to x (if this period is mating)
    conception = t_cr * period_is_mating
    ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e = 0)
    slc = [slice(None)] * len(conception.shape)
    # slc[pinp.sheep['i_e1_pos']] = slice(0,1)
    slc[uinp.parameters['i_b1_pos']] = slice(0,1)
    conception[tuple(slc)] = -np.sum(f_dynamic_slice(conception, uinp.parameters['i_b1_pos'],1, 5), axis = (uinp.parameters['i_b1_pos']), keepdims=True)
    temporary = (index_e1 == 0) * np.sum(conception, axis=pinp.sheep['i_e1_pos'], keepdims=True) #sum across e axis into slice 0
    conception = fun.f_update(conception, temporary, (nyatf_b1any == 0)) #Put sum of e1 into slice 0 (e1) if nyatf == 0
    ##Set proportions for dams that gave birth and lost to 0 - this is required so that numbers in pp behave correctly
    conception *= (nfoet_b1any == nyatf_b1any)
    return conception

def f_conception_ltw(cu0, cs_mating, scan_std, doy_p, nfoet_b1any, nyatf_b1any, period_is_mating, index_e1):
    ##Slope of the RR vs CS relationship	
    slope = np.maximum(cu0[4, ...], cu0[2, ...] + np.sin(2 * np.pi * doy_p / 365) * cu0[3, ...])
    ##Reproduction rate
    cs_mating_e1b1sliced = f_dynamic_slice(cs_mating, pinp.sheep['i_e1_pos'], 0, 1, uinp.parameters['i_b1_pos'], 0, 1) #take slice from e1 & b1 axis (take slice from not mated to get cs because they are about to be mated)
    repro_rate = scan_std + (cs_mating_e1b1sliced - 3) * slope
    ###remove b1 axis by squeezing
    repro_rate = np.squeeze(repro_rate, axis=uinp.parameters['i_b1_pos'])
    ##Conception - propn dry/single/twin for given repro rate
    conception = np.moveaxis(f_DSTw(repro_rate)[...,uinp.structure['a_nfoet_b1']], -1, uinp.parameters['i_b1_pos']) * period_is_mating #move the l0 axis into the b1 position. and expand to b1 size.
    ##Number remaining not-mated (cr[-1])
    slc = [slice(None)] * len(conception.shape)
    slc[uinp.parameters['i_b1_pos']] = slice(0,1)
    conception[tuple(slc)] = -np.sum(f_dynamic_slice(conception, uinp.parameters['i_b1_pos'],1, 5), axis = (uinp.parameters['i_b1_pos']), keepdims=True)
    temporary = (index_e1 == 0) * np.sum(conception, axis=pinp.sheep['i_e1_pos'], keepdims=True)  # sum across e axis into slice 0
    conception = fun.f_update(conception, temporary, (nyatf_b1any == 0))  # Put sum of e1 into slice 0 (e1) if nyatf == 0
    ##Set proportions for dams that gave birth and lost to 0 - this is required so that numbers in pp behave correctly
    conception *= (nfoet_b1any == nyatf_b1any)
    return conception




def f_sire_req(sire_propn_a1e1b1nwzida0e0b0xyg1g0, sire_periods_g0p8, i_sire_recovery, i_startyear, date_end_p, period_is_prejoin_a1e1b1nwzida0e0b0xyg1):
    ##Date at end of period adjusted to start year
    t_date_end_a1e1b1nwzida0e0b0xyg = date_end_p - (365 * (date_end_p.astype('datetime64[Y]').astype(int) + 1970 - i_startyear)).astype('timedelta64[D]')
    ##Date_end falls within the ram mating periods
    sire_required_a1e1b1nwzida0e0b0xyg1g0p8 = np.logical_and(t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] >= sire_periods_g0p8.astype('datetime64[D]') , t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] <= (sire_periods_g0p8.astype('datetime64[D]') + i_sire_recovery)) #add axis for p8 and g1
    ##Number of rams required per ewe (if this period is joining)
    n_sires = sire_required_a1e1b1nwzida0e0b0xyg1g0p8 * sire_propn_a1e1b1nwzida0e0b0xyg1g0[..., na] * period_is_prejoin_a1e1b1nwzida0e0b0xyg1[..., na,na] #add axis for g1 and p8
    return n_sires


def f_mortality_base(cd, cg, rc_start, ebg_start, d_nw_max, days_period):
    return (cd[1, ...] + cd[2, ...] * np.maximum(0, cd[3, ...] - rc_start) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))) * days_period #mul be days period to convert from mort per day to per period


def f_mortality_weaner_cs(cd, cg, age, ebg_start, d_nw_max,days_period):
    return cd[13, ...] * f_ramp(age, cd[15, ...], cd[14, ...]) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))* days_period #mul be days period to convert from mort per day to per period


def f_mortality_dam_cs(cb1, cg, nw_start, ebg, days_period, period_between_birth6wks, gest_propn, sar_mortalitye):
    ##(Twin) Dam mortality in last 6 weeks (preg tox)
    t_mort = days_period * gest_propn /42 * f_sig(-42 * ebg * cg[18, ...] / nw_start, cb1[4, ...], cb1[5, ...]) #mul be days period to convert from mort per day to per period
    ##If not last 6 weeks then = 0
    mort = t_mort * period_between_birth6wks
    mort = fun.f_sa(mort, sar_mortalitye, sa_type = 4)
    return mort

    
def f_mortality_dam_mu(cu2, cs_birth_dams, period_is_birth, days_period, sar_mortalitye):
    ##(Twin) Dam mortality in last 6 weeks (preg tox)	
    t_mortalitye_mu = cu2[22, 0, ...] * cs_birth_dams + cu2[22, 1, ...] * cs_birth_dams ** 2 + cu2[22, -1, ...]
    ##Non-multiple bearing ewes = 0	
    mortalitye_mu = np.exp(t_mortalitye_mu) / (1 + np.exp(t_mortalitye_mu)) * period_is_birth * days_period #mul be days period to convert from mort per day to per period
    ##Dam (& progeny) losses at birth related to CSL	
    mortalitye_mu = fun.f_sa(mortalitye_mu, sar_mortalitye, sa_type = 4)
    return mortalitye_mu


    
def f_mortality_progeny_cs(cd, cb1, w_b, rc_birth, w_b_exp_y, period_is_birth, chill_index_m1, nfoet_b1, sar_mortalityp):
    ##Progeny losses due to large progeny (dystocia)
    mortalityd_yatf = f_sig(fun.f_divide(w_b, w_b_exp_y) * np.maximum(1, rc_birth), cb1[6, ...], cb1[7, ...]) * period_is_birth
    ##add sensitivity
    mortalityd_yatf = fun.f_sa(mortalityd_yatf, sar_mortalityp, sa_type = 4)
    ##dam mort due to large progeny (dystocia)
    mortalityd_dams = fun.f_divide(np.mean(mortalityd_yatf, axis=uinp.parameters['i_x_pos'], keepdims=True) * cd[21,...], nfoet_b1)  #returns 0 mort if there is 0 nfoet - this handles div0 error
    ##Progeny losses due to large progeny (dystocia) - so there is no double counting of progeny loses associated with dam mortality
    mortalityd_yatf = mortalityd_yatf * (1- cd[21,...])
    ##Exposure index
    xo = cd[8, ..., na] - cd[9, ..., na] * rc_birth[..., na] + cd[10, ..., na] * chill_index_m1 + cb1[11, ..., na]
    ##Progeny mortality at birth from exposure
    mortalityx = np.average(np.exp(xo) / (1 + np.exp(xo)) ,axis = -1) * period_is_birth #axis -1 is m1
    ##add sensitivity
    mortalityx = fun.f_sa(mortalityx, sar_mortalityp, sa_type = 4)
    return mortalityx, mortalityd_yatf, mortalityd_dams


def f_mortality_progeny_mu(cu2, cb1, cx, ce, w_b, foo, chill_index_m1, period_is_birth, sar_mortalityp):
    ##transformed survival	
    t_mortalityp_mu = cu2[8, 0, ..., na] * w_b[..., na] + cu2[8, 1, ..., na] * w_b[..., na] ** 2 + cu2[8, 2, ..., na] * chill_index_m1 + cu2[8, 3, ..., na] * foo[..., na] + cu2[8, 4, ..., na] * foo[..., na] ** 2 + cu2[8, 5, ..., na] + cb1[8, ..., na] + cx[8, ..., na] + cx[9, ..., na] * chill_index_m1 + ce[8, ..., na]
    ##back transformed
    mortalityp_mu = np.average(1 / (1 + np.exp(-t_mortalityp_mu)),axis = -1) * period_is_birth #m1 axis averaged
    ##Progeny mortality at birth (LTW) with SA	
    mortalityp_mu = fun.f_sa(mortalityp_mu, sar_mortalityp, sa_type = 4)
    return mortalityp_mu
        




def f_comb(n,k):
    # ##Create an array of factorial values up to n
    # factorial = np.cumprod(np.arange(np.max(n))+1)
    # ##Combination
    # combinations = factorial[n-1]/(factorial[k-1]*factorial[n-k-1])
    ##Create an array of factorial values up to n
    factorial_range = np.arange(np.max(n)+1)
    factorial_range[0] = 1
    factorial = np.cumprod(factorial_range)
    ##Combination
    combinations = factorial[n]/(factorial[k]*factorial[n-k])
    return combinations




def f_period_start_prod(numbers, var, prejoin_tup, season_tup, i_n_len, i_w_len, i_n_fvp_period, numbers_start_fvp0, period_is_startfvp0, period_is_startseason, period_is_prejoin=0, group=None):
    ##Set variable level = value at end of previous	
    var_start = var
    ##make sure numbers and var are same shape - this is required for the np.average func below
    numbers, var_start, numbers_start_fvp0 = np.broadcast_arrays(numbers,var_start,numbers_start_fvp0)
    ##a)update var if start of DVP
    var_start = f_condensed(numbers, var_start, prejoin_tup, season_tup, i_n_len, i_w_len, i_n_fvp_period, numbers_start_fvp0, period_is_startfvp0)
    ##b) Calculate temporary values as if period is start of season
    if np.any(period_is_startseason):
        temporary = fun.f_weighted_average(var_start, numbers, season_tup, keepdims=True, non_zero=True)#gets the weighted average of production in the different seasons
        ##Set values where it is beginning of FVP
        var_start = fun.f_update(var_start, temporary, period_is_startseason)
    if group==1 and np.any(period_is_prejoin):
        ##c) Calculate temporary values as if period_is_prejoin	
        temporary = fun.f_weighted_average(var_start, numbers, prejoin_tup, keepdims=True, non_zero=True) #gets the weighted average of production in the different seasons
        ##Set values where it is beginning of FVP
        var_start = fun.f_update(var_start, temporary, period_is_prejoin)
    return var_start

#^old version
# def f_period_start_prod(numbers, var, prejoin_tup, season_tup, period_is_startfvp, period_is_break, period_is_prejoin=0, group=None):
#     ##Set variable level = value at end of previous
#     var_start = var
#     ##make sure numbers and var are same shape - this is required for the np.average func below
#     numbers, var_start = np.broadcast_arrays(numbers,var_start)
#     ##a) Calculate temporary values as if start of FVP, only required if n axis is active
#     if uinp.structure['i_n1_len'] >= uinp.structure['i_w1_len']:
#         temporary = var_start #this is done to ensure that temp has the same size as var. In the next line np.diagonal removes the n axis so it is added back in using the expand function, but that is a singleton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
#         temporary[...] = np.expand_dims(np.rollaxis(var_start.diagonal(axis1= uinp.structure['i_w_pos'], axis2= uinp.structure['i_n_pos']),-1,uinp.structure['i_w_pos']), uinp.structure['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
#         ##Update if the period is start of a FVP
#         var_start = fun.f_update(var_start, temporary, period_is_startfvp)
#     ##b) Calculate temporary values as if period_is_break
#     temporary = fun.f_weighted_average(var_start, numbers, season_tup, keepdims=True)
#     #temporary = np.expand_dims(np.average(var_start, axis=season_tup, weights=numbers),season_tup) #gets the weighted average of production in the different seasons, have to add axis back because no keepdims arg
#     ##Set values where it is beginning of FVP
#     var_start = fun.f_update(var_start, temporary, period_is_break)
#     if group==1:
#         ##c) Calculate temporary values as if period_is_prejoin
#         temporary = fun.f_weighted_average(var_start, numbers, prejoin_tup, keepdims=True) #gets the weighted average of production in the different seasons
#         ##Set values where it is beginning of FVP
#         var_start = fun.f_update(var_start, temporary, period_is_prejoin)
#     return var_start

def f_condensed(numbers, var, prejoin_tup, season_tup, i_n_len, i_w_len, i_n_fvp_period, numbers_start_fvp0, period_is_startfvp0):
    '''condense variable to 3 common points along the w axis for the start of fvp0'''
    if np.any(period_is_startfvp0):
        temporary = var.copy()  #this is done to ensure that temp has the same size as var.
        ###test if array has diagonal and calc temp variables as if start of dvp - if there is not a diagonal use the alternative system for reallocating at the end of a DVP
        ### np.diagonal removes the n axis so it is added back in using the expand function, but that is a singleton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
        if i_n_len >= i_w_len:
            ####this method was the way we first tried - no longer used (might be used later if we add nutrient options back in)
            temporary[...] = np.expand_dims(np.rollaxis(temporary.diagonal(axis1= uinp.structure['i_w_pos'], axis2= uinp.structure['i_n_pos']),-1,uinp.structure['i_w_pos']), uinp.structure['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
        else:
            ###add high pattern
            temporary[...] = np.mean(
                f_dynamic_slice(var, uinp.structure['i_w_pos'], int(i_n_len ** i_n_fvp_period), int(i_n_len ** i_n_fvp_period) + int(i_w_len / 10)), uinp.structure['i_w_pos'],
                keepdims=True)  # average of the top lw patterns
            ###add mid pattern (w 0 - 27) - use slice method in case w axis changes position (cant use MRYs dynamic slice function because we are assigning)
            sl = [slice(None)] * temporary.ndim
            sl[uinp.structure['i_w_pos']] = slice(0, int(i_n_len ** i_n_fvp_period))
            temporary[tuple(sl)] = f_dynamic_slice(var, uinp.structure['i_w_pos'], 0, 1)  # the pattern that is feed supply 1 (median) for the entire year (the top w pattern)
            ###low pattern
            ind = np.argsort(var, axis=uinp.structure['i_w_pos'])  #sort into production order so we can select the lowest production with mort less than 10% - note sorts in ascending order
            var_sorted = np.take_along_axis(var, ind, axis=uinp.structure['i_w_pos'])
            numbers_start_sorted = np.take_along_axis(numbers_start_fvp0, ind, axis=uinp.structure['i_w_pos'])
            numbers_sorted = np.take_along_axis(numbers, ind, axis=uinp.structure['i_w_pos'])
            low_slice = i_w_len - np.sum(
                np.sum(numbers_start_sorted, axis=prejoin_tup + (season_tup,), keepdims=True) / np.sum(numbers_sorted,
                                                                                          axis=prejoin_tup + (season_tup,),
                                                                                          keepdims=True) > 0.9,
                uinp.structure['i_w_pos'],
                keepdims=True)  # returns bool if mort is less the 10% then sums the falses which give the index of the first w pattern that has mort less that 10%
            sl = [slice(None)] * temporary.ndim
            sl[uinp.structure['i_w_pos']] = slice(-int(i_n_len ** i_n_fvp_period), None)
            temporary[tuple(sl)] = np.take_along_axis(var_sorted, low_slice, uinp.structure[
                'i_w_pos'])  # production level of the lowest nutrition profile that has a mortality less than 10% for the year
        ###Update if the period is start of year (shearing for offs and prejoining for dams)
        var = fun.f_update(var, temporary, period_is_startfvp0)

    return var

def f_period_start_nums(numbers, prejoin_tup, season_tup, i_n_len, i_w_len, i_n_fvp_period, numbers_start_fvp0, period_is_startfvp0, period_is_startseason, season_propn_z, group=None, nyatf_b1 = 0, numbers_initial_repro=0, gender_propn_x=1, period_is_prejoin=0, period_is_birth=False):
    ##a)update numbers if start of DVP
    numbers = f_condensed(numbers, numbers, prejoin_tup, season_tup, i_n_len, i_w_len, i_n_fvp_period, numbers_start_fvp0, period_is_startfvp0)
    ##b) reallocate for season type
    if np.any(period_is_startseason):
        temporary = np.sum(numbers, axis = season_tup, keepdims=True)  * season_propn_z  #Calculate temporary values as if period_is_break
        numbers = fun.f_update(numbers, temporary, period_is_startseason)  #Set values where it is beginning of FVP
    ##things for dams - prejoining and moving between classes
    if group==1 and np.any(period_is_prejoin):
        ##d) new repro cycle (prejoining)
        temporary = np.sum(numbers, axis = (prejoin_tup), keepdims=True) * numbers_initial_repro #Calculate temporary values as if period_is_prejoin
        numbers = fun.f_update(numbers, temporary, period_is_prejoin)  #Set values where it is beginning of FVP
    ##things just for yatf
    if group==2:
        temp = nyatf_b1 * gender_propn_x   # nyatf is accounting for peri-natal mortality. But doesn't include the differential mortality of female and male offspring at birth
        numbers=fun.f_update(numbers, temp, period_is_birth)
    return numbers


# def f_period_start_nums(numbers, prejoin_tup, season_tup, period_is_startfvp, period_is_break, season_propn_z, group=None, numbers_initial_repro=0, period_is_prejoin=None):
#     ##a) reallocate between w and n if the period is start of a FVP
#     ###Calculate temporary values as if start of FVP - collapse n back to standard level (n axis is populated due to mortality)
#     if uinp.structure['i_n1_len'] >= uinp.structure['i_w1_len']:
#         temporary = numbers #this is done to ensure that temp has the same size as var. In the next line np.diagonal removes the n axis so it is added back in using the expand function, but that is a singleton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
#         temporary[...] = np.expand_dims(np.rollaxis(numbers.diagonal(axis1= uinp.structure['i_w_pos'], axis2= uinp.structure['i_n_pos']),-1,uinp.structure['i_w_pos']), uinp.structure['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
#         numbers = fun.f_update(numbers, temporary, period_is_startfvp)
#     ##b) reallocate for season type
#     temporary = np.sum(numbers, axis = season_tup, keepdims=True)  * season_propn_z  #Calculate temporary values as if period_is_break
#     numbers = fun.f_update(numbers, temporary, period_is_break)  #Set values where it is beginning of FVP
#     ##things for dams - prejoining and moving between classes
#     if group==1:
#         ##d) new repro cycle (prejoining)
#         temporary = np.sum(numbers, axis = (prejoin_tup), keepdims=True) * numbers_initial_repro #Calculate temporary values as if period_is_prejoin
#         numbers = fun.f_update(numbers, temporary, period_is_prejoin)  #Set values where it is beginning of FVP
#     return numbers


def f_period_end_nums(numbers, mortality, numbers_min_b1, mortality_yatf=0, nfoet_b1 = 0, nyatf_b1 = 0, group=None, conception = 0, scan=0, gbal=0, gender_propn_x=1, period_is_mating = False, period_is_matingend = False, period_is_birth=False, period_is_scan=False):
    '''
    This adjusts numbers for things like conception and mortality that happen during a given period
    '''
    ##a) mortality
    numbers = numbers * (1-mortality)
    ##numbers for post processing - don't include selling drys - assignment required here for when it is not group 1 or 2
    pp_numbers = numbers
    ##things for dams - prejoining and moving between classes
    if group==1:
        ###b) conception - conception is the change in numbers +ve for animals getting pregnancy and -ve in the NM e-0 slice (note the conception for e slice 1 and higher puts the negative numbers in the e-0 nm slice)
        if np.any(period_is_mating):
            temporary = numbers + conception * numbers[:, 0:1, 0:1, ...]  # numbers_dams[..., 0,0, ...] is the NM slice of cycle 0 ie the number of animals yet to be mated (conception will have negative value in nm slice)
            numbers = fun.f_update(numbers, temporary, np.any(period_is_mating, axis=pinp.sheep['i_e1_pos'])) #needs to be previous period else conception is not calculated because numbers happens at beginning of p loop
        ###at the end of mating move any remaining numbers from nm to 00 slice (note only the nm slice for e-0 has numbers - this is handled in the conception function)
        ###Set temporary to copy of current numbers
        if np.any(period_is_matingend):
            temporary  = np.copy(numbers)
            temporary[:, 0:1, 1:2, ...] += numbers[:, 0:1, 0:1, ...]
            temporary[:, 0, 0, ...] = 0.00001 #so nm can be an activity without nan. want a small number relative to mortality after allowing for multiple slices getting the small number
            numbers = fun.f_update(numbers, temporary, period_is_matingend)
        ###d) birth (account for birth status and if drys are retained)
        if np.any(period_is_birth):
            dam_propn_birth_b1 = f_comb(nfoet_b1, nyatf_b1) * (1 - mortality_yatf) ** nyatf_b1 * mortality_yatf ** (nfoet_b1 - nyatf_b1) # the proportion of dams of each LSLN based on (progeny) mortality
            temp = np.sum(dam_propn_birth_b1 * gender_propn_x, axis=uinp.parameters['i_x_pos'], keepdims=True) * numbers[:,:,uinp.structure['a_prepost_b1'],...] #have to average x axis so that it is not active for dams - times by gender propn to give approx weighting (ie because offs are not usually entire males so they will get low weighting)
            pp_numbers = fun.f_update(numbers, temp, period_is_birth)  # calculated in the period after birth when progeny mortality due to exposure is calculated
            temp = np.maximum(pinp.sheep['i_drysretained_birth'],np.minimum(1, nyatf_b1)) * pp_numbers
            numbers = fun.f_update(pp_numbers, temp, period_is_birth * (gbal>=2)) # has to happen after the dams are moved due to progeny mortality so that gbal drys are also scaled by drys_retained
        else:
            ##numbers for post processing - don't include selling drys - assignment required here in case it is not birth
            pp_numbers = numbers
        ###c) scanning
        if np.any(period_is_scan):
            temp = np.maximum(pinp.sheep['i_drysretained_scan'],np.minimum(1, nfoet_b1)) * numbers # scale the level of drys by drys_retained, scale every other slice by 1 except drys if not retained
            numbers = fun.f_update(numbers, temp, period_is_scan * (scan>=1))
        ###e)make the max of number 0.0001
        numbers=np.maximum(numbers_min_b1, numbers)
    return numbers,pp_numbers





    
def f_carryforward_u1(cu1, ebg, period_between_joinstartend, period_between_joinscan, period_between_scanbirth, period_between_birthwean, days_period, period_propn):
    ##First 3 slices of the genotype axis = the sire genotypes	
    coeff_cf1 = fun.f_update(0, cu1[1,...], period_between_joinstartend) #note cu1 has already had the first axis (animal) sliced when it was passed in
    ##Loop over remaining slices	
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[2,...], period_between_joinscan)
    ##Loop over remaining slices	
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[3,...], period_between_scanbirth)
    ##Loop over remaining slices	
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[4,...], period_between_birthwean)
    ##Assign values based on maternal and paternal genotype	
    d_cf = coeff_cf1 * ebg * days_period * period_propn
    return d_cf


def f_wool_additional(fd, sl, ss, vm,  pmb, cvfd=0.22, cvsl=0.18):
    cu5_u5c5=uinp.sheep['i_cu5_c5']
    i_eqn_ph=uinp.sheep['i_eqn_ph']
    i_eqn_cvh=uinp.sheep['i_eqn_cvh']
    i_eqn_romaine=uinp.sheep['i_eqn_romaine']
    ##adjusted pmb
    pmb = np.maximum(cu5_u5c5[4, i_eqn_ph], pmb)
    ##predicted hauteur price adj
    ph = cu5_u5c5[0, i_eqn_ph] * sl + cu5_u5c5[1, i_eqn_ph] * ss + cu5_u5c5[2, i_eqn_ph] * fd + cu5_u5c5[
        3, i_eqn_ph] * pmb + cu5_u5c5[5, i_eqn_ph] * vm + cu5_u5c5[6, i_eqn_ph] * cvfd + cu5_u5c5[
          7, i_eqn_ph] * cvsl + cu5_u5c5[8, i_eqn_ph]
    ##Back transform the ph if using CSIRO equation
    if i_eqn_ph == 0:
        ph = 1 / (1 + np.exp(-ph))
    ##predicted cv hauteur
    cvh = cu5_u5c5[0, i_eqn_cvh] * sl + cu5_u5c5[1, i_eqn_cvh] * ss + cu5_u5c5[2, i_eqn_cvh] * fd + cu5_u5c5[
        3, i_eqn_cvh] * pmb + cu5_u5c5[5, i_eqn_cvh] * vm + cu5_u5c5[6, i_eqn_cvh] * cvfd + cu5_u5c5[
          7, i_eqn_cvh] * cvsl + cu5_u5c5[8, i_eqn_cvh]
    ##predicted romaine
    romaine = cu5_u5c5[0, i_eqn_romaine] * sl + cu5_u5c5[1, i_eqn_romaine] * ss + cu5_u5c5[2, i_eqn_romaine] * fd + \
      cu5_u5c5[3, i_eqn_romaine] * pmb + cu5_u5c5[5, i_eqn_romaine] * vm + cu5_u5c5[6, i_eqn_romaine] * cvfd + \
      cu5_u5c5[7, i_eqn_romaine] * cvsl + cu5_u5c5[8, i_eqn_romaine]
    return ph, cvh, romaine

def f_woolprice():
    ##micron price guide percentile to use
    mpg_percentile = uinp.sheep['i_woolp_mpg_percentile']
    ##price for std fd at selected percentile
    mpg_stdfd = np.interp(mpg_percentile, uinp.sheep['i_woolp_mpg_range_w5'], uinp.sheep['i_woolp_mpg_w5'])
    ##Price at std FD (adjusted by sav)
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sav['woolp_mpg'], 5)
    ##Price at std FD (adjusted by sam)
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sam['woolp_mpg'])
    ##FD percentile to use (adjusted by sav)
    fd_percentile = uinp.sheep['i_woolp_fdprem_percentile']
    ##FD premium at selected percentile
    fdprem_w4 = np.array([np.interp(fd_percentile, uinp.sheep['i_woolp_fdprem_range_w5'], uinp.sheep['i_woolp_fdprem_w4w5'][i]) for i in range(uinp.sheep['i_woolp_fdprem_w4w5'].shape[0])])
    ##FD premium to use (adjusted by sav)
    fdprem_w4 = fun.f_sa(fdprem_w4, sen.sav['woolp_fdprem'], 5)
    ##Wool price for the analysis (note fdprem is the premium per micron - calculate like this because each step is not necessarily 1 micron)
    woolprice_w4 = mpg_stdfd * (1 + fdprem_w4) ** (uinp.sheep['i_woolp_fd_std'] - uinp.sheep['i_woolp_fd_range_w4'])
    return woolprice_w4


def f_wool_value(mpg_w4, cfw_pg, fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg,dtype=None):
    ##call function for ph cvh and romaine
    ph_pg, cvh_pg, romaine_pg = f_wool_additional(fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg)
    ##STB price for FNF (free or nearly free of fault)
    fnf_pg = np.interp(fd_pg, uinp.sheep['i_woolp_fd_range_w4'], mpg_w4 * uinp.sheep['i_stb_scalar_w4']).astype(dtype)
    ##vm price adj
    vm_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_vm_adj_w4w6'], uinp.sheep['i_woolp_vm_range_w6'], uinp.sheep['i_woolp_fd_range_w4'], vm_pg,fd_pg).astype(dtype)
    ##predicted hauteur price adj
    ph_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_ph_adj_w4w7'], uinp.sheep['i_woolp_ph_range_w7'], uinp.sheep['i_woolp_fd_range_w4'], ph_pg,fd_pg).astype(dtype)
    ##cv hauteur price adj
    cvh_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_cvh_adj_w4w8'], uinp.sheep['i_woolp_cvh_range_w8'], uinp.sheep['i_woolp_fd_range_w4'], cvh_pg,fd_pg).astype(dtype)
    ##romaine price adj
    romaine_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_romaine_adj_w4w9'], uinp.sheep['i_woolp_romaine_range_w9'], uinp.sheep['i_woolp_fd_range_w4'], romaine_pg,fd_pg).astype(dtype)
    ##wool price with adjustments
    woolp_stb_pg = fnf_pg * (1 + vm_adj_pg) * (1 + ph_adj_pg) * (1 - cvh_adj_pg) * (1 - romaine_adj_pg)
    ##stb net in the bank price
    woolp_stbnib_pg = woolp_stb_pg * (1 - uinp.sheep['i_wool_cost_pc']) - uinp.sheep['i_wool_cost_kg']
    ##wool value if shorn this period
    wool_value_pg = woolp_stbnib_pg * cfw_pg
    return wool_value_pg, woolp_stbnib_pg

def f_condition_score(rc, cu0):
    ''' Estimate CS from LW. Works with scalars or arrays - provided they are broadcastable into ffcfw.

       ffcfw: (kg) Fleece free, conceptus free liveweight. normal_weight: (kg). cs_propn: (0.19) change in LW
       associated with 1 CS as a proportion of normal_weight.

       long version of the formula (use rc instead of using to following): 3 + (ffcfw - normal_weight) / (cs_propn * normal_weight)
       Returns: condition score - float
       '''
    return np.maximum(1, 3 + (rc - 1) / cu0[1, ...]) #cs cant be below 1 because the animal would be dead

#todo needs updating - currently just a copy of the cs function
def f_fat_score(rc, cu0):
    return np.maximum(1, 3 + (rc - 1) / cu0[1, ...]) #fs cant be below 1


def f_norm_cdf(x, mu, cv):
    ##sd - standard deviation - maximum to stop div0 errors in next step.
    sd = np.maximum(1,mu) * cv
    ##standadise x
    std = (x - mu) / sd
    ##probability (<=x)
    prob = 1 / (np.exp(-358 / 23 * std + 111 * np.arctan(37 / 294 * std)) + 1)
    return prob

def f_saleprice(score_pricescalar_s7s5s6, weight_pricescalar_s7s5s6, dtype=None):
    ##Sale price percentile to use (adjusted by sav)
    salep_percentile = uinp.sheep['i_salep_percentile']
    ##Max price in grids at selected percentile - 1d interpolation along the s4 axis for each grid (s7 axis)
    grid_max_s7 = (np.array([np.interp(salep_percentile, uinp.sheep['i_salep_percentile_range_s4'], uinp.sheep['i_salep_percentile_scalar_s7s4'][i])
                            for i in range(uinp.sheep['i_salep_percentile_scalar_s7s4'].shape[0])]) * uinp.sheep['i_salep_price_max_s7']).astype(dtype)
    ##Max price in grids (adj sav)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sav['salep_max'], 5)
    ##Max price in grids (adj sam)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sam['salep_max'])
    ##Scalar for weight impact across the grid (sat adjusted)
    weight_scalar_s7s5s6 = weight_pricescalar_s7s5s6
    ##Scalar for score impact across the grid (sat adjusted)
    score_scalar_s7s5s6 = score_pricescalar_s7s5s6
    ##price for the analysis
    grid_s7s5s6 = grid_max_s7[:,na,na] * weight_scalar_s7s5s6 * score_scalar_s7s5s6
    return grid_s7s5s6




def f_salep_mob(weight_s7spg, scores_s7s6pg, cvlw_s7s5pg, cvscore_s7s6pg,
                grid_weightrange_s7s5pg, grid_scorerange_s7s6p5g, grid_priceslw_s7s5s6pg):
    '''A function to calculate the average price of the mob based on the average specifications in the mob.
    This is to represent that the distribution of weight & specification reduces the mob average price
    This representation allows valuing individual animal management and reducing the mob distribution.
    Note: if the distribution extends below the lower range of weight or score in the grid these animals have zero value (ncv)'''

    ## Probability for each lw step in grid based on the mob average weight and the coefficient of variation (CV) of weight
    ### probability of being less than the upper value of the step (roll) - probability of less than the lower value of the step
    prob_lw_s7s5pg = np.maximum(0, f_norm_cdf(np.roll(grid_weightrange_s7s5pg, -1, axis = 1), weight_s7spg, cvlw_s7s5pg)
                          - f_norm_cdf(grid_weightrange_s7s5pg, weight_s7spg, cvlw_s7s5pg))
    ## Probability for each score step in grid (fat score/CS) based on the mob average score and the CV of quality score
    prob_score_s7s6pg = np.maximum(0, f_norm_cdf(np.roll(grid_scorerange_s7s6p5g, -1, axis = 1), scores_s7s6pg, cvscore_s7s6pg)
                             - f_norm_cdf(grid_scorerange_s7s6p5g, scores_s7s6pg, cvscore_s7s6pg))
    ##Probability for each cell of grid (assuming that weight & score are independent allows multiplying weight and score probabilities)
    prob_grid_s7s5s6pg = prob_lw_s7s5pg[:,:,na, ...] * prob_score_s7s6pg[:,na,...]

    ##Average price for the mob is the sum of the probabilities in each cell of the grid and the price in that cell
    saleprice_mobaverage_s7pg = np.sum(prob_grid_s7s5s6pg * grid_priceslw_s7s5s6pg, axis = (1, 2))
    return saleprice_mobaverage_s7pg


def f_sale_value(cu0, cx, o_rc, o_ffcfw_pg, dressp_adj_yg, dresspercent_adj_s6pg,
                 dresspercent_adj_s7pg, grid_price_s7s5s6pg, month_scalar_s7pg,
                 month_discount_s7pg, price_type_s7pg, cvlw_s7s5pg, cvscore_s7s6pg,
                 grid_weightrange_s7s5pg, grid_scorerange_s7s6pg, age_end_p5g1, discount_age_s7pg,sale_cost_pc_s7pg,
                 sale_cost_hd_s7pg, mask_s7x_s7pg, sale_agemax_s7pg1, dtype=None):
    ##Calculate condition score from relative condition
    cs_pg = f_condition_score(o_rc, cu0)
    ##Calculate fat score from relative condition
    fs_pg = f_fat_score(o_rc, cu0)
    ##Combine the scores into single array
    scores_s8p = np.stack([fs_pg, cs_pg], axis=0)
    ##Select the quality scores (s8) for each price grid (s7)
    scores_s7s6pg = scores_s8p[uinp.sheep['ia_s8_s7']][:,na,...]
    ##Dressing percentage to adjust price grid from $/kg DW to $/kg LW
    ### It is easier to convert the price to $/kg LW than it is to convert a distribution of LW and fat score to a distribution of dressed weight and fat score
    ### because dressing percentage changes with fat score.
    dresspercent_for_price_s7s6pg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[23, ...] + dresspercent_adj_s6pg + dresspercent_adj_s7pg[:,na,...]
    ##Dressing percentage is set to 100% if price type is $/kg LW or $/hd
    dresspercent_for_price_s7s6pg = fun.f_update(dresspercent_for_price_s7s6pg, 1, price_type_s7pg[:,na,...] >= 1)
    ##Create the grid prices in $/kg LW
    grid_priceslw_s7s5s6pg = grid_price_s7s5s6pg * dresspercent_for_price_s7s6pg[:,na,...]

    ## Calculate the 'lookup' weight of the average animal in the units of each grid (some grids the weight is dressed weight other grids are LW)
    ## start with dressing percentage and set to 1 later if the grid is kg LW
    ###Interploate DP adjustment based on the average FS of the animals
    dressp_adj_fs_pg= np.interp(fs_pg, uinp.sheep['i_salep_score_range_s8s6'][0, ...], uinp.sheep['i_salep_dressp_adj_s6']).astype(dtype)
    ###Average Dressing percentage including effects of genotype, fat score and age (which varies with the grid).
    dresspercent_for_wt_s7pg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[23, ...] + dressp_adj_fs_pg + dresspercent_adj_s7pg
    ###Dressing percentage is 100% if price type is $/kg LW or $/hd
    dresspercent_wt_s7pg = fun.f_update(dresspercent_for_wt_s7pg, 1, price_type_s7pg >= 1)
    ###Scale ffcfw to the units in the grid
    weight_for_lookup_s7pg = o_ffcfw_pg * dresspercent_wt_s7pg

    ##Calculate mob average price in each grid from the mob average and the distribution of weight & score within the mob (this is just the price, not the total animal value)
    price_mobaverage_s7pg = f_salep_mob(weight_for_lookup_s7pg[:,na,...], scores_s7s6pg, cvlw_s7s5pg, cvscore_s7s6pg,
                                                      grid_weightrange_s7s5pg, grid_scorerange_s7s6pg, grid_priceslw_s7s5s6pg)

    ##Scale price received based on month of sale
    price_mobaverage_s7pg = price_mobaverage_s7pg * (1+month_scalar_s7pg)

    ## Apply the age based discount if the animal is greater than the threshold age
    ### Temporary value of the age based discount from the relevant month
    temporary_s7pg = price_mobaverage_s7pg * (1 + month_discount_s7pg)
    ###Apply discount if age is greater than threshold age
    price_mobaverage_s7pg = fun.f_update(price_mobaverage_s7pg, temporary_s7pg, age_end_p5g1/30 > discount_age_s7pg)  #divide 30 to convert to months

    ## Some grids are in $/hd. For these grids don't want to multiply grid value by weight (so set weight to 1)
    ### Convert weight to 1 if price is $/hd (price_type == 2)
    weight_for_value_s7pg = fun.f_update(o_ffcfw_pg, 1, price_type_s7pg == 2)

    ## Calculate the net value per head from the gross value minus the selling costs
    ### Calculate gross value per head
    sale_value_s7pg = price_mobaverage_s7pg * weight_for_value_s7pg
    ###Subtract the selling costs (some are percentage costs some are $/hd)
    sale_value_s7pg = sale_value_s7pg * (1 - sale_cost_pc_s7pg) - sale_cost_hd_s7pg

    ## Select the best net sale price from the relevant grids
    ###Mask the grids based on the maximum age and the gender for each grid
    sale_value_s7pg = sale_value_s7pg * mask_s7x_s7pg * (age_end_p5g1/30 <= sale_agemax_s7pg1) #divide 30 to convert to months
    ###Select the maximum value across the grids
    sale_value = np.max(sale_value_s7pg, axis=0) #take max on s6 axis as well to remove it (it is singleton so no effect)
    return sale_value



def f_animal_trigger_levels(index_pg, age_start, period_is_shearing_pg, a_next_s_pg, period_is_wean_pg, gender,
                            o_ebg_p, wool_genes, period_is_joining_pg, animal_mated, period_is_endmating_pg):
    ##Trigger value 1 - week of year
    trigger1_pg = index_pg % 52
    ##Trigger value 2 - age
    trigger2_pg = np.trunc(age_start / 7)
    ##Trigger value 3 - Weeks from previous shearing
    trigger3_pg = index_pg - np.maximum.accumulate(index_pg*period_is_shearing_pg)
    ##Trigger value 4 - weeks to next shearing - cant use period is array like in the other situations
    trigger4_pg = a_next_s_pg - index_pg #this will return 0 when the current period is shearing because the next association points at the current period when period is
    ##Trigger value 5 - weeks from previous joining
    trigger5_pg = index_pg - np.maximum.accumulate(index_pg*period_is_joining_pg)
    ##Trigger value 6 - weeks from end of mating
    trigger6_pg = index_pg - np.maximum.accumulate(index_pg*period_is_endmating_pg)
    ##Trigger value 7 - weeks from previous weaning
    trigger7_pg = index_pg - np.maximum.accumulate(index_pg*period_is_wean_pg)
    ##Trigger value 8 - whether animals was mated
    trigger8_pg = animal_mated
    ##Trigger value 9 - gender of the animal
    trigger9_pg = gender
    ##Trigger value 10 - rate of empty body gain
    trigger10_pg = o_ebg_p
    ##Trigger value 11 - the 'wooliness' of the genotype
    trigger11_pg = wool_genes
    ##Stack the triggers
    animal_triggervalues_h7pg = np.stack(np.broadcast_arrays(trigger1_pg, trigger2_pg, trigger3_pg, trigger4_pg, trigger5_pg, trigger6_pg, trigger7_pg, trigger8_pg, trigger9_pg, trigger10_pg, trigger11_pg), axis = 0)
    return animal_triggervalues_h7pg


def f_treatment_unit_numbers(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, a_nyatf_b1g=0):
    ##Unit 0 - per head
    unit0_pg = 1
    ##Unit 1 - adjusted head
    unit1_pg = unit0_pg * head_adjust
    ##Unit 2 - mob
    unit2_pg = unit0_pg / mobsize_pg
    ##Unit 3 - LW
    unit3_pg = unit0_pg * o_ffcfw_pg
    ##Unit 4 - CFW
    unit4_pg = unit0_pg * o_cfw_pg
    ##Unit 5 - nyatf
    unit5_pg = unit0_pg * a_nyatf_b1g
    ##Stack the triggers
    treatment_units_h8pg = np.stack(np.broadcast_arrays(unit0_pg, unit1_pg, unit2_pg, unit3_pg, unit4_pg, unit5_pg), axis=0)
    return treatment_units_h8pg



def f_operations_triggered(animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg):
    ##Test slice 0 of h5 axis
    slice0_h7h2pg = animal_triggervalues_h7pg[:, na, ...] <= operations_triggerlevels_h5h7h2pg[0,...]
    ##Test slice 1 of h5 axis
    slice1_h7h2pg = np.logical_or(animal_triggervalues_h7pg[:, na, ...] == operations_triggerlevels_h5h7h2pg[1, ...], operations_triggerlevels_h5h7h2pg[1, ...] == np.inf)
    ##Test slice 2 of h5 axis
    slice2_h7h2pg = animal_triggervalues_h7pg[:, na, ...] >= operations_triggerlevels_h5h7h2pg[2,...]
    ##Test across the conditions
    slices_all_h7h2pg = np.logical_and(slice0_h7h2pg, np.logical_and(slice1_h7h2pg, slice2_h7h2pg))
    ##Test across the rules (& collapse s7 axis)
    triggered_h2pg = np.all(slices_all_h7h2pg, axis=0)
    return triggered_h2pg

from memory_profiler import profile

@profile
def f_application_level(operation_triggered_h2pg, animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg):
    ## mask & remove the slices of the h7 axis that don't require calculation of the application level (not required because inputs do not include a range input)
    ## must be same mask for 'le' and 'ge'
    maskh7_h7 = fun.f_reduce_skipfew(np.any, operations_triggerlevels_h5h7h2pg[3,...] != np.inf, preserveAxis=0)#, keepdims=False), animal_triggervalues_h7pg.shape)
    ### mask the input arrays to minimise slices of h7
    animal_triggervalues_h7mask_h7pg = animal_triggervalues_h7pg[maskh7_h7]
    operations_triggerlevels_h7mask_h5h7h2pg = operations_triggerlevels_h5h7h2pg[:, maskh7_h7, ...]

    ##broadcast the input arrays so the 'required' mask can be applied
    operations_triggerlevels_casted_h5h7h2pg=np.broadcast_to(operations_triggerlevels_h7mask_h5h7h2pg, operations_triggerlevels_h7mask_h5h7h2pg.shape[0:2]+operation_triggered_h2pg.shape)
    animal_triggervalues_h7mask_h7h2pg = np.broadcast_to(animal_triggervalues_h7mask_h7pg[:,na,...], operations_triggerlevels_casted_h5h7h2pg.shape[1:])


    ## Calculate the application level for "less than or equal"
    ### The 'le' calculation is required only if the 'range' input is less than the le trigger value and both are not inf.
    required_h7h2pg = (operation_triggered_h2pg * (operations_triggerlevels_h7mask_h5h7h2pg[0, ...] != np.inf)
                       * (operations_triggerlevels_h7mask_h5h7h2pg[3, ...] != np.inf)
                       * (operations_triggerlevels_h7mask_h5h7h2pg[3, ...] < operations_triggerlevels_h7mask_h5h7h2pg[0, ...]))

    ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
    temporary_h7h2pg = np.ones_like(required_h7h2pg, dtype='float32')

    ##Level if animal trigger level is between 'range' and 'le'
    ### calculate the masked version of the triggerlevels because required 3 times in the calculation
    operations_triggerlevels_masked_h5h7h2pg = operations_triggerlevels_casted_h5h7h2pg[:, required_h7h2pg]
    temporary_h7h2pg[required_h7h2pg] = np.clip((animal_triggervalues_h7mask_h7h2pg[required_h7h2pg] - operations_triggerlevels_masked_h5h7h2pg[0, ...])/
                                                (operations_triggerlevels_masked_h5h7h2pg[3, ...] - operations_triggerlevels_masked_h5h7h2pg[0, ...]),0,1)
    ##Select the maximum across the h7 axis if the operation is triggered
    level_h2pg = np.max(temporary_h7h2pg,axis=0) * operation_triggered_h2pg   #mul by operation triggered so that level goes to 0 if operation is not triggered


    ## Repeat for 'ge' using same variable names as for 'le'
    ## Calculate the application level for "greater than or equal"
    ### The 'ge' calculation is required only if the 'range' input is greater than the ge trigger value and both are not inf.
    required_h7h2pg = (operation_triggered_h2pg * (operations_triggerlevels_h7mask_h5h7h2pg[2, ...] != -np.inf)
                       * (operations_triggerlevels_h7mask_h5h7h2pg[3, ...] != np.inf)
                       * (operations_triggerlevels_h7mask_h5h7h2pg[3, ...] > operations_triggerlevels_h7mask_h5h7h2pg[2, ...]))

    ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
    ### calculate the masked version of the triggerlevels because required 3 times in the calculation
    operations_triggerlevels_masked_h5h7h2pg = operations_triggerlevels_casted_h5h7h2pg[:, required_h7h2pg]
    temporary_h7h2pg = np.ones_like(required_h7h2pg, dtype='float32')

    ##Level if animal trigger level is between 'range' and 'le'
    temporary_h7h2pg[required_h7h2pg] = np.clip((animal_triggervalues_h7mask_h7h2pg[required_h7h2pg] - operations_triggerlevels_masked_h5h7h2pg[2, ...])/
                                                (operations_triggerlevels_masked_h5h7h2pg[3, ...] - operations_triggerlevels_masked_h5h7h2pg[2, ...]),0,1)
    ##Select the maximum across the h7 axis if the operation is triggered
    temporary_h2pg = np.max(temporary_h7h2pg,axis=0) * operation_triggered_h2pg   #mul by operation triggered so that level goes to 0 if operation is not triggered

    ##Select the maximum of the 'le' and 'ge' value
    level_h2pg = np.maximum(level_h2pg, temporary_h2pg)

    return level_h2pg


def f_mustering_required(application_level_h2pg, husb_operations_muster_propn_h2pg):
    ##Total mustering required for all operations
    musters_pg = np.sum(application_level_h2pg * husb_operations_muster_propn_h2pg, axis=0)
    ##Round up to the next integer
    musters_pg = np.ceil(musters_pg)
    return musters_pg


def f_husbandry_component(level, treatment_units, requirements, association, axes_tup):
    ##Number of treatment units for contract
    units = treatment_units[association]
    ##Infrastructure requirement for each animal class during the period
    component = np.sum(level * units * requirements, axis=axes_tup)
    return component


def f_husbandry_requisites(level_hpg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_requisites_prob_h6hpg,a_h8_h):
    ##Number of treatment units for requisites
    units_hpg = treatment_units_h8pg[a_h8_h]
    ##Labour requirement for each animal class during the period
    cost_pg = np.sum(level_hpg * units_hpg * husb_requisite_cost_h6pg[:,na, ...] *
                     husb_requisites_prob_h6hpg, axis = (0, 1))
    return cost_pg

def f_husbandry_labour(level_hpg, treatment_units_h8pg, units_per_labourhour_l2hpg, a_h8_h):
    ##Number of treatment units for contract
    units_hpg = treatment_units_h8pg[a_h8_h]
    ##Labour requirement for each animal class during the period
    hours_l2pg = np.sum(fun.f_divide(level_hpg * units_hpg , units_per_labourhour_l2hpg, dtype=level_hpg.dtype), axis=1)  #divide by units_per_labourhour_l2hpg because that is how many units can be done per hour eg how many sheep can be drenched per hr
    return hours_l2pg

def f_husbandry_infrastructure(level_hpg, husb_infrastructurereq_h1h2pg):
    ##Infrastructure requirement for each animal class during the period
    infrastructure_h1pg = np.sum(level_hpg * husb_infrastructurereq_h1h2pg, axis=1)
    return infrastructure_h1pg

def f_contract_cost(application_level_h2pg, treatment_units_h8pg, husb_operations_contract_cost_h2pg):
    ##Number of animal units for contract
    units_h2pg = treatment_units_h8pg[uinp.sheep['ia_h8_h2']]
    ##Contract cost for each animal class during the period
    cost_pg = np.sum(application_level_h2pg * units_h2pg * husb_operations_contract_cost_h2pg, axis=0)
    return cost_pg

def f_husbandry(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, operations_triggerlevels_h5h7h2pg, index_pg,
                age_start, period_is_shear_pg, a_next_s_pg, period_is_wean_pg, gender, o_ebg_p, wool_genes,
                husb_operations_muster_propn_h2pg, husb_requisite_cost_h6pg, husb_operations_requisites_prob_h6h2pg,
                operations_per_hour_l2h2pg, husb_operations_infrastructurereq_h1h2pg,
                husb_operations_contract_cost_h2pg, husb_muster_requisites_prob_h6h4pg,
                musters_per_hour_l2h4pg, husb_muster_infrastructurereq_h1h4pg,
                a_nyatf_b1g=0,period_is_joining_pg=False, animal_mated=False, period_is_endmating_pg=False, dtype=None):
    ##An array of the trigger values for the animal classes in each period - these values are compared against a threshold to determine if the husb is required
    animal_triggervalues_h7pg = f_animal_trigger_levels(index_pg, age_start, period_is_shear_pg, a_next_s_pg, period_is_wean_pg, gender,
                            o_ebg_p, wool_genes, period_is_joining_pg, animal_mated, period_is_endmating_pg).astype(dtype)
    ##The number of treatment units per animal in each period - each slice has a different unit eg mobsize, nyatf etc the treatment unit can be selected and applied for a given husb operation
    treatment_units_h8pg = f_treatment_unit_numbers(head_adjust, mobsize_pg, o_ffcfw_pg, o_cfw_pg, a_nyatf_b1g).astype(dtype)
    ##Is the husb operation triggered in the period for each class
    operation_triggered_h2pg = f_operations_triggered(animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg)
    ##The level of the operation in each period for the class of livestock (proportion of animals that receive treatment) - this accounts for the fact that just because the operation is triggered the operation may not be done to all animals
    start=time.time() #^delete this stuff once the function is faster
    application_level_h2pg = f_application_level(operation_triggered_h2pg, animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg)
    finish1=time.time()
    print('finish1 - level of husb : ', finish1-start)
    ##The number of times the mob must be mustered
    mustering_level_pg = f_mustering_required(application_level_h2pg, husb_operations_muster_propn_h2pg)
    ##The cost of requisites for the operations
    operations_requisites_cost_pg = f_husbandry_requisites(application_level_h2pg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_operations_requisites_prob_h6h2pg, uinp.sheep['ia_h8_h2'])
    ##The labour requirement for the operations
    operations_labourreq_l2pg = f_husbandry_labour(application_level_h2pg, treatment_units_h8pg, operations_per_hour_l2h2pg, uinp.sheep['ia_h8_h2'])
    ##The infrastructure requirements for the operations
    operations_infrastructurereq_h1pg = f_husbandry_infrastructure(application_level_h2pg, husb_operations_infrastructurereq_h1h2pg)
    ##Contract cost for husbandry
    contract_cost_pg = f_contract_cost(application_level_h2pg, treatment_units_h8pg, husb_operations_contract_cost_h2pg)
    ##The cost of requisites for mustering
    mustering_requisites_cost_pg = f_husbandry_requisites(mustering_level_pg, treatment_units_h8pg, husb_requisite_cost_h6pg, husb_muster_requisites_prob_h6h4pg, uinp.sheep['ia_h8_h4'])
    ##The labour requirement for mustering
    mustering_labourreq_l2pg = f_husbandry_labour(mustering_level_pg, treatment_units_h8pg, musters_per_hour_l2h4pg, uinp.sheep['ia_h8_h4'])
    ##The infrastructure requirements for mustering
    mustering_infrastructurereq_h1pg = f_husbandry_infrastructure(mustering_level_pg, husb_muster_infrastructurereq_h1h4pg)
    finish2=time.time()
    print('finish2: ', finish2-finish1)
    ##Total cost of husbandry
    husbandry_cost_pg = operations_requisites_cost_pg + mustering_requisites_cost_pg + contract_cost_pg
    ##Labour requirement for husbandry
    husbandry_labour_l2pg = operations_labourreq_l2pg + mustering_labourreq_l2pg
    ##infrastructure requirement for husbandry
    husbandry_infrastructure_h1pg = operations_infrastructurereq_h1pg + mustering_infrastructurereq_h1pg
    finish3=time.time()
    print('finish3: ', finish3-finish2)
    return husbandry_cost_pg, husbandry_labour_l2pg, husbandry_infrastructure_h1pg



##################
#post processing #
##################

##Method 1 (still used)- add p and v axis together then sum p axis - this may be a good method for faster computers with more memory
def f_p2v_std(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
            period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
    try:
        days_period_p = days_period_p.astype(
            'float32')  # convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it cant be converted to float (because int object is not numpy)
    except AttributeError:
        pass
    ##mul everything
    production_ftvpany = (production_p * numbers_p * days_period_p * period_is_tvp
                          * on_hand_tvp * (dvp_pointer_p == index_vp) * (a_any1_p == index_any1tvp)
                          * (a_any2_p == index_any2any1tvp))
    return np.sum(production_ftvpany, axis=uinp.structure['i_p_pos']-sumadj)  # sum along p axis to leave just a v axis (sumadj is to handle nsire that has a p8 axis at the end)


# ##Method 4 - loop over v and sum p - this save p and v axis being on the same array but requires lots of looping so isn't much faster
# def f_p2v_loop(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1, period_is_tvp=True, a_ev_p=1, index_ftvp=1, a_p6_p=1, index_p6ftvp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it cant be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_ev_p==index_ftvp)
#                         * (a_p6_p==index_p6ftvp))
#
#     shape = production_ftpany.shape[0:3] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[4:]  # bit messy because need v t and all the other axis (but not p)
#     final=np.zeros(shape).astype('float32')
#     for i in range(np.max(dvp_pointer_p)+1):
#         temp_prod = np.sum(production_ftpany * (dvp_pointer_p==i), axis=uinp.structure['i_p_pos'])
#         final[:,:,:,i,...] = temp_prod  #asign to correct v slice
#     return final

# ##Method 3 - use groupby to sum p, this means p and v don't exist on the same array - not as fast as method 2
# import numpy_indexed as npi
# def f_p2v_groupby(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1, period_is_tvp=True, a_ev_p=1, index_ftvp=1, a_p6_p=1, index_p6ftvp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it cant be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_ev_p==index_ftvp)
#                         * (a_p6_p==index_p6ftvp))
#     ##convert p to v
#     shape = production_ftpany.shape[0:3] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[4:]  # bit messy because need v t and all the other axis (but not p)
#     result=np.zeros(shape).astype('float32')
#     shape = dvp_pointer_p.shape
#     for e1 in range(shape[-13]):
#         for g in range(shape[-1]):
#             result[:,:,:,:, :, e1:e1+1, :, :, :, :, :, :, :, :, :, :, :, g:g+1] = npi.GroupBy(dvp_pointer_p[:, 0, e1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, g], axis=0).sum(
#                                                                     production_ftpany[:, :, :, :, :, e1:e1+1, :, :, :, :, :, :, :, :, :, :, :, g:g+1], axis=uinp.structure['i_p_pos'])[1]
#     return result

##Method 2 (fastest)- sum sections of p axis to leave v (almost like sum if) this is fast because don't need p and v axis one same array
def f_p2v(production_p, dvp_pointer_p=1, numbers_p=1, on_hand_tp=True, days_period_p=1, period_is_tp=True, a_any1_p=1, index_any1tp=1, a_any2_p=1, index_any2any1tp=1):
    try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it cant be converted to float (because int object is not numpy)
    except AttributeError:
        pass
    ##mul everything - add t,f and p6 axis
    production_ftpany = (production_p * numbers_p * days_period_p * period_is_tp
                        * on_hand_tp * (a_any1_p==index_any1tp)
                        * (a_any2_p==index_any2any1tp))
    ##convert p to v - info at this link https://stackoverflow.com/questions/50121980/numpy-conditional-sum
    ##basically we are summing the p axis for each dvp. the tricky part (which has caused the requirement for the loops) is that dvp pointer is not the same for each axis eg dvp is effected by e axis.
    ##so we need to loop though all the axis in the dvp and sum p and assign to a final array.
    ##if the axis is size 1 (ie singleton) then we want to take all of that axis ie ':' because just because the dvp pointer has singleton doesnt mean param array has singleton so need to take all slice of the param (unless that is an active dvp axis because that means dvp timing may differ for different slices along that axis so it must be summed in the loop)
    shape = production_ftpany.shape[0:uinp.structure['i_p_pos']] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[uinp.structure['i_p_pos']+1:]  # bit messy because need v t and all the other axis (but not p)
    result=np.zeros(shape).astype('float32')
    shape = dvp_pointer_p.shape
    for a1 in range(shape[-14]):
        a1_slc = slice(a1,a1+1) if shape[-14]>1 else slice(0,None) #used for param because we want to keep axis
        for e1 in range(shape[-13]):
            e1_slc = slice(e1, e1 + 1) if shape[-13] > 1 else slice(0, None)
            for b1 in range(shape[-12]):
                b1_slc = slice(b1, b1 + 1) if shape[-12] > 1 else slice(0, None)
                for n in range(shape[-11]):
                    n_slc = slice(n, n + 1) if shape[-11] > 1 else slice(0, None)
                    for w in range(shape[-10]):
                        w_slc = slice(w, w + 1) if shape[-10] > 1 else slice(0, None)
                        for z in range(shape[-9]):
                            z_slc = slice(z, z + 1) if shape[-9] > 1 else slice(0, None)
                            for i in range(shape[-8]):
                                i_slc = slice(i, i + 1) if shape[-8] > 1 else slice(0, None)
                                for d in range(shape[-7]):
                                    d_slc = slice(d, d + 1) if shape[-7] > 1 else slice(0, None)
                                    for a0 in range(shape[-6]):
                                        a0_slc = slice(a0, a0 + 1) if shape[-6] > 1 else slice(0, None)
                                        for e0 in range(shape[-5]):
                                            e0_slc = slice(e0, e0 + 1) if shape[-5] > 1 else slice(0, None)
                                            for b0 in range(shape[-4]):
                                                b0_slc = slice(b0, b0 + 1) if shape[-4] > 1 else slice(0, None)
                                                for x in range(shape[-3]):
                                                    x_slc = slice(x, x + 1) if shape[-3] > 1 else slice(0, None)
                                                    for y in range(shape[-2]):
                                                        y_slc = slice(y, y + 1) if shape[-2] > 1 else slice(0, None)
                                                        for g in range(shape[-1]):
                                                            g_slc = slice(g, g + 1) if shape[-1] > 1 else slice(0, None)
                                                            result[..., a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc] \
                                                                = np.add.reduceat(production_ftpany[..., a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                                  , np.r_[0, np.where(np.diff(dvp_pointer_p[:, a1, e1, b1, n, w, z, i, d, a0, e0, b0, x, y, g]))[0] + 1], axis=uinp.structure['i_p_pos']) #np.r_ basically concats two 1d arrays (so here we are just adding 0 to the start of the array)
    return result





def f_cum_dvp(arr,dvp_pointer,axis=0,shift=0):
    '''This function does accumulative max but it resets at each dvp.
    '''
    final = np.zeros_like(arr)
    for i in range(np.max(dvp_pointer)+1):  #plus 1 so that the last dvp is counted for
        arr1 = arr * (dvp_pointer==i) #sets the p slices to 0 if not in the given dvp
        arr1 = np.roll(arr1,shift,axis) #this is only used for the dams on hand calculation, this rolls the period is sale array 1 unit along the p axis.
                                        # This is required so that period is onhand == true in the period that sale occurs and false after that.
                                        # Because sale occurs at the end of a given period so the sheep are technically onhand for the period sale occurs.
        arr1 = np.maximum.accumulate(arr1,axis=axis)
        arr1 = arr1 * (dvp_pointer==i) #sets the cum max to 0 for other dvp not of interest
        final += arr1
    return final

def f_lw_distribution(ffcfw_condensed_va1e1b1nwzida0e0b0xyg, ffcfw_va1e1b1nwzida0e0b0xyg, i_n_len, i_n_fvp_period, dvp_type_next_tvgw=0):
    '''distributing animals on LW at the start of dvp0
    ^ this function will need altering if the dvp_type definition changes'''
    ##add second w axis - the condensed w axis becomes axis -1 and the end of period w stays in the normal place
    ffcfw_condensed_va1e1b1nwzida0e0b0xygw = fun.f_reshape_expand(np.moveaxis(ffcfw_condensed_va1e1b1nwzida0e0b0xyg,uinp.structure['i_w_pos'],-1), uinp.structure['i_n_pos']-1, right_pos=pinp.sheep['i_z_pos']-1)
    ##Calculate the difference between the 3 (or more if not dvp0) condensed weights and the middle weight (slice 0)
    diff = ffcfw_condensed_va1e1b1nwzida0e0b0xygw - f_dynamic_slice(ffcfw_condensed_va1e1b1nwzida0e0b0xygw, -1, 0, 1)
    ##Calculate the spread that would generate the average weight
    # spread =  1 - fun.f_divide((ffcfw_condensed_va1e1b1nwzida0e0b0xygw - ffcfw_va1e1b1nwzida0e0b0xyg[..., na]), diff, dtype=diff.dtype)
    spread =  fun.f_divide((diff - (ffcfw_condensed_va1e1b1nwzida0e0b0xygw - ffcfw_va1e1b1nwzida0e0b0xyg[..., na])), diff, dtype=diff.dtype)
    ## select the minimum value that is greater than 0, make others 0
    temporary = np.copy(spread)
    temporary[spread <= 0] = np.inf
    spread = spread * (spread == np.min(temporary, axis=-1, keepdims = True))
    ##Bound the spread
    spread_bounded = np.clip(spread, 0, 1)
    ##Set values for the standard pattern to be the remainder from the closest. (consolidated w axis)
    spread_bounded[..., :int(i_n_len ** i_n_fvp_period)] =  1 - np.maximum(spread_bounded[..., int(i_n_len ** i_n_fvp_period):-int(i_n_len ** i_n_fvp_period)], spread_bounded[..., -int(i_n_len ** i_n_fvp_period):])
    ##Set the distribution to 0 if lw_end is below the condensed minimum weight
    distribution_va1e1b1nwzida0e0b0xygw = spread_bounded * (ffcfw_va1e1b1nwzida0e0b0xyg[..., na] >= np.min(ffcfw_condensed_va1e1b1nwzida0e0b0xygw, axis = -1, keepdims=True))
    ##Set default for DVPs that don’t require distributing to 1 (these are masked later to remove those that are not required)
    distribution_va1e1b1nwzida0e0b0xygw = fun.f_update(distribution_va1e1b1nwzida0e0b0xygw, 1, dvp_type_next_tvgw!=0)
    return distribution_va1e1b1nwzida0e0b0xygw

def f_lw_distribution_2prog(ffcfw_prog_g2w9, ffcfw_yatf_vg1):
    ###maximum(0, ) removes points where yatf weight is greater than the rolled progeny weight
    distribution_2prog_vg1w9 = 1- fun.f_divide((ffcfw_yatf_vg1[..., na] - ffcfw_prog_g2w9)
                                               , np.abs(np.roll(ffcfw_prog_g2w9,-1,axis=-1) - ffcfw_prog_g2w9))
    ###remove if the yatf weight is above the weight in the next progeny slice (ie the division is >1).
    distribution_2prog_vg1w9[distribution_2prog_vg1w9 < 0] = 0
    ###remove if the yatf weight is below the progeny weight in that slice (ie the division is negative).
    distribution_2prog_vg1w9[distribution_2prog_vg1w9 > 1] = 0
    ###set the distribution for the other of the target pair
    distribution_2prog_vg1w9[..., 1:] += (1 - distribution_2prog_vg1w9[..., :-1]) * (distribution_2prog_vg1w9[..., :-1] > 0)
    return distribution_2prog_vg1w9

def f_create_production_param(group, production_vg, a_kcluster_vg_1=1, index_ktvg_1=1, a_kcluster_vg_2=1, index_kktvg_2=1, numbers_start_vg=1, mask_vg=True, pos_offset=0):
    '''convert production to per animal including impact of death. And apply the k clustering'''
    if group=='sire':
        return fun.f_divide(production_vg, numbers_start_vg, dtype=production_vg.dtype)
    elif group=='dams':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * mask_vg
                                  , axis = (uinp.parameters['i_b1_pos']-pos_offset, pinp.sheep['i_e1_pos']-pos_offset), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1),
                                     axis=(uinp.parameters['i_b1_pos']-pos_offset, pinp.sheep['i_e1_pos']-pos_offset), keepdims=True), dtype=production_vg.dtype)
    elif group=='offs':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2)
                                  , axis = (uinp.parameters['i_d_pos'], uinp.parameters['i_b0_pos'], uinp.structure['i_e0_pos']), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2),
                                     axis=(uinp.parameters['i_d_pos'], uinp.parameters['i_b0_pos'], uinp.structure['i_e0_pos']), keepdims=True), dtype=production_vg.dtype)
