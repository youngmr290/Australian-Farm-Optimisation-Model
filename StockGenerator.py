# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John


Code structure
1. when reshaping input arrays use the i_len_? values but in the rest of the code use the len_? values.
2. sim params use their own pos and len variables.

Notes
1. Inputs need to be put together. One sheet in property.xl and one in universsal.xl (and paramater sheet in universal so basically three sheets total)
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
from scipy import stats
# from numba import jit

# import FeedBudget as fdb
import Functions as fun
# import Periods as per
import PropertyInputs as pinp
import StockFunctions as sfun
import UniversalInputs as uinp






##^these function can possibly be moved to sheep routines once steve is done.
def f_feedsupply_adjust(attempts,feedsupply,itn):
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
    #^line can be deleted once working.
    # params_c2=parameters['i_gfw_c2']
    # y=parameters['i_gfw_y']
    # var_pos=parameters['i_cx_pos']
    # len_ax1=parameters['i_cx_len']
    # len_ax2=parameters['i_cx_len2']

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


def f_g2g(array_g,group,left_pos=0,len_ax1=0,len_ax2=0,len_ax3=0,swap=False,right_pos=-1,left_pos2=0,right_pos2=-1):
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
        return array_g[...,mask_sire_inc_g0]
    elif group == 'dams':
        ##create mask g?g
        mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis =1)
        return array_g[...,mask_dams_inc_g1]
    elif group == 'offs':
        ##create mask g?g
        mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis =1)
        return array_g[...,mask_offs_inc_g3]
    elif group == 'yatf':
        ##create mask g?g
        mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis =1)
        return array_g[...,mask_yatf_inc_g2]


def f_reshape_expand(array,left_pos=0,len_ax0=0,len_ax1=0,len_ax2=0,swap=False,ax1=0,ax2=1,right_pos=0,left_pos2=0,right_pos2=0
                     , left_pos3=0,right_pos3=0, condition = None, axis = 0):
    '''
    Parameters
    ----------
    array : array
        parameter array - input from excel.
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
    *note: mask applied last (after expanding and reshaping)

    Returns
    -------
    Reshapes, swaps axis if required, expands and applys a mask to a given axis if required.
    '''
    ##make array incase it is a single number
    array = np.array([array])
    if len_ax2>0:
        shape=(len_ax0,len_ax1,len_ax2)
        array = array.reshape(shape)
    elif len_ax1>0:
        shape=(len_ax0,len_ax1)
        array = array.reshape(shape)
    else:
        pass#don't need to reshpae

    ##swap axis if neccessary
    if swap:
        array = np.swapaxes(array, ax1, ax2)
    ##get axis into correct position 1
    if left_pos != None or left_pos != 0:
        extra_axes = tuple(range((left_pos + 1), right_pos))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##get axis into correct position 2 (some arrays need singleton axis added in multiple places ie seperated by a used axis)
    if left_pos2 != None or left_pos2 != 0:
        extra_axes = tuple(range((left_pos2 + 1), right_pos2))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##get axis into correct position 3 (some arrays need singleton axis added in multiple places ie seperated by a used axis)
    if left_pos3 != None or left_pos3 != 0:
        extra_axes = tuple(range((left_pos3 + 1), right_pos3))
    else: extra_axes = ()
    array = np.expand_dims(array, axis = extra_axes)
    ##apply mask if required
    try:
        if condition != None:
            if type(condition) == bool:
                condition= np.asarray([condition]) #convert to numpy if it is singular input (this will do nothing if already np array)
                array = np.compress(condition, array, axis)
    except ValueError: 
        if (condition != None).all():
            array = np.compress(condition, array, axis)
    return array

def f_DSTw(scan_std):
    '''
    Parameters
    ----------
    scan_std : np array
        scanning percentage of genotypes.

    Returns
    -------
    Proportion of dry, single, twins & triplets. Numpy way of making this formula: y=int+ax+bx^2+cx^3+dx^4

    '''
    scan_powers = uinp.sheep['i_scan_powers'][:,na,na]  #scan powers are the exponential powers used in the quadratic formula ie ^0, ^1, ^2, ^3, ^4
    scan_power_syg = scan_std ** scan_powers #puts the variable to the powers ie x^0, x^1, x^2, x^3, x^4
    dstwtr_l0yg = np.sum(uinp.sheep['i_scan_coeff_l0s'][...,na,na] * scan_power_syg, axis = -3) #add the coefficients and sum all the elements of the equation ie int+ax+bx^2+cx^3+dx^4
    return dstwtr_l0yg

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
    lamb_numbers_b0yg = np.zeros((6,lss.shape[-2],lss.shape[-1])) #^where can i reference 6? is there an input somewhere? (need i_b0_len)
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




#^some of these may have to occur inside the sim function because they may change between trials (it would be good if they didn't though because creating the indexes takes a bit of time)
### _create numpy index for param dicts ^creating indexes is a bit slow
##the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
keys_a = pinp.sheep['i_wean_times'][pinp.sheep['i_mask_a']]
keys_b0 = uinp.structure['i_btrt_idx_offs']
keys_b1 = uinp.structure['i_lsln_idx_dams']
keys_g0 = pinp.sheep['i_groups_sire']
keys_g1 = pinp.sheep['i_groups_dams']
keys_g3 = pinp.sheep['i_groups_offs']
keys_i = pinp.sheep['i_tol_idx'][pinp.sheep['i_mask_i']]
keys_lw0 = uinp.structure['i_w_idx_sire']
keys_lw1 = uinp.structure['i_w_idx_dams']
keys_lw3 = uinp.structure['i_w_idx_dams']
keys_n0 = uinp.structure['i_n_idx_sire']
keys_n1 = uinp.structure['i_n_idx_dams']
keys_n3 = uinp.structure['i_n_idx_dams']
keys_p6 = np.array([pinp.feed_inputs['feed_periods'].index[:-1]])
keys_v = np.asarray(uinp.structure['sheep_pools'])
keys_y0 = uinp.parameters['i_gen_merit_sire']
keys_y1 = uinp.parameters['i_gen_merit_dams']
keys_y3 = uinp.parameters['i_gen_merit_dams']
    







# def simulation():
#     """
#     A function to wrap the simulation that can be called by SheepPyomo.

#     Called after the sensitivty variables have been updated.
#     It populates the arrays by looping through the time periods
#     Globally define arrays are used to transfer results to sheep_paramters()

#     Returns
#     -------
#     None.
#     """

######################
##date               #
######################
na=np.newaxis
## _define the periods
n_sim_periods, date_start_p, date_end_p, p_index_p, step \
= sfun.sim_periods(pinp.sheep['i_startyear'], uinp.structure['i_sim_periods_year'], uinp.structure['i_age_max'])
date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
date_end_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_end_p, axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
##day of the year
doy_pa1e1b1nwzida0e0b0xyg = (date_start_pa1e1b1nwzida0e0b0xyg - date_start_pa1e1b1nwzida0e0b0xyg.astype('datetime64[Y]')).astype(int) + 1 #plus one to include current day eg 7th - 1st = 6 plus 1 = 7th day of year
##day length ^not used yet
dl_pa1e1b1nwzida0e0b0xyg = sfun.f_daylength(doy_pa1e1b1nwzida0e0b0xyg, pinp.sheep['i_latitude'])


###################################
## calculate masks                #
###################################
##masks required for initialising arrays
mask_sire_inc_g0 = np.any(uinp.structure['i_mask_g0g3'] * pinp.sheep['i_g3_inc'], axis =1)
mask_dams_inc_g1 = np.any(uinp.structure['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis =1)
mask_offs_inc_g3 = np.any(uinp.structure['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis =1)

# ###################################
# ### axis len                      #
# ###################################
## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
len_m1 = int(step / np.timedelta64(1, 'D')) #convert timedelta to float by dividing by one day
len_m2 = uinp.structure['i_lag_wool']
len_m3 = uinp.structure['i_lag_organs']
len_q0	 = pinp.sheep['i_eqn_exists_q0q1'].shape[1]
len_q1	 = len(pinp.sheep['i_eqn_reportvars_q1'])
len_q2	 = np.max(pinp.sheep['i_eqn_reportvars_q1'])
len_p = len(date_start_p)
len_a1 = np.count_nonzero(pinp.sheep['i_mask_a'])
len_e1 = np.max(pinp.sheep['i_join_cycles_ig1'])
len_b1 = len(uinp.structure['i_mask_b0_b1'])
len_n0 = uinp.structure['i_n0_len']
len_n1 = uinp.structure['i_n1_len']
len_n2 = uinp.structure['i_n1_len'] #same as dams
len_n3 = uinp.structure['i_n3_len']
len_w0 = uinp.structure['i_w0_len']
len_w1 = uinp.structure['i_w1_len']
len_w2 = uinp.structure['i_w1_len'] #same as dams
len_w3 = uinp.structure['i_w3_len']
len_z = np.count_nonzero(pinp.sheep['i_mask_z'])
len_i = np.count_nonzero(pinp.sheep['i_mask_i'])
len_d = uinp.parameters['i_d_len']
len_a0 = np.count_nonzero(pinp.sheep['i_mask_a'])
len_e0 = np.max(pinp.sheep['i_join_cycles_ig1'])
len_b0 = np.count_nonzero(uinp.structure['i_mask_b0_b1'])
len_x = pinp.sheep['i_x_len']
len_y = np.count_nonzero(uinp.parameters['i_mask_y'])
len_g0 = np.count_nonzero(mask_sire_inc_g0)
len_g1 = np.count_nonzero(mask_dams_inc_g1)
len_g2 = np.count_nonzero(mask_dams_inc_g1)
len_g3 = np.count_nonzero(mask_offs_inc_g3)

###################################
### index arrays                  #
###################################
index_p = np.arange(300)#asarray(300)
index_e1 = np.arange(np.max(pinp.sheep['i_join_cycles_ig1']))
index_e1b1nwzida0e0b0xyg = np.expand_dims(index_e1, axis = tuple(range(1,-pinp.sheep['i_e1_pos'])))
index_m1 = np.arange(len_m1)
index_m0 = np.arange(12)*2  #2hourly steps for chill calculations
index_z = np.arange(len_z)



############################
### initialise arrays      #
############################
'''only if assigned with a slice'''
##unique array shapes required to initialise arrays
qg0 = (len_q0, len_q1, len_q2, len_p, 1, 1, 1, 1, 1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g0)
qg1 = (len_q0, len_q1, len_q2, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g1)
qg2 = (len_q0, len_q1, len_q2, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, len_d, 1, 1, 1, len_x, len_y, len_g1)
qg3 = (len_q0, len_q1, len_q2, len_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y, len_g3)
# g1 = (len_p, len_a, len_e, len_b1, len_g1_n, len_g1_w, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g1)
m2g0 = (len_m2, len_p, 1, 1, 1, 1, 1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g0)
m2g1 = (len_m2, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g1)
m2g2 = (len_m2, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y, len_g2)
m2g3 = (len_m2, len_p, 1, 1, 1, len_n3, len_w3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y, len_g3)
m3g0 = (len_m3, len_p, 1, 1, 1, 1, 1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g0)
m3g1 = (len_m3, len_p, len_a1, len_e1, len_b1, len_n1, len_w1, len_z, len_i, 1, 1, 1, 1, 1, len_y, len_g1)
m3g2 = (len_m3, len_p, len_a1, len_e1, len_b1, len_n2, len_w2, len_z, len_i, 1, 1, 1, 1, len_x, len_y, len_g2)
m3g3 = (len_m3, len_p, 1, 1, 1, len_n3, len_n3, len_z, len_i, len_d, len_a0, len_e0, len_b0, len_x, len_y, len_g3)

###sires
omer_history_start_m3g0 = np.zeros(m3g0, dtype = 'float64')
d_cfw_history_start_m2g0 = np.zeros(m2g0, dtype = 'float64')
# nw_pa1e1b1nwzida0e0b0xyg0 = np.expand_dims(np.zeros(1, dtype = 'float64'), axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
# ###Dams
# ldr_start_g1 = np.zeros(g1, dtype = 'float64')
# lb_start_g1 = np.zeros(g1, dtype = 'float64')
# w_f_start_g1 = np.zeros(g1, dtype = 'float64')
# nw_f_start_g1 = np.zeros(g1, dtype = 'float64')
# nec_cum_start_dams = np.zeros(g1, dtype = 'float64')
# cf_w_b_mu_start_dams = np.zeros(g1, dtype = 'float64')
# cf_w_w_mu_start_g1 = np.zeros(g1, dtype = 'float64')
# cf_conception_mu_start_g1 = np.zeros(g1, dtype = 'float64')
omer_history_start_m3g1 = np.zeros(m3g1, dtype = 'float64')
d_cfw_history_start_m2g1 = np.zeros(m2g1, dtype = 'float64')
# nw_pa1e1b1nwzida0e0b0xyg1 = np.expand_dims(np.zeros(1, dtype = 'float64'), axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
# ###yatf
omer_history_start_m3g2 = np.zeros(m3g2, dtype = 'float64')
d_cfw_history_start_m2g2 = np.zeros(m2g2, dtype = 'float64')
# nw_pa1e1b1nwzida0e0b0xyg2 = np.expand_dims(np.zeros(1, dtype = 'float64'), axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
# ###Offspring
omer_history_start_m3g3 = np.zeros(m3g3, dtype = 'float64')
d_cfw_history_start_m2g3 = np.zeros(m2g3, dtype = 'float64')
# nw_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(np.zeros(1, dtype = 'float64'), axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))
###report variables
###empty arrays to store different return values from the eqation systems in the p loop.
r_compare_q0q1q2psire = np.zeros(qg0, dtype = 'float64')
r_compare_q0q1q2pdams = np.zeros(qg1, dtype = 'float64')
r_compare_q0q1q2pyatf = np.zeros(qg2, dtype = 'float64')
r_compare_q0q1q2poffs = np.zeros(qg3, dtype = 'float64')



################################################
#  management, age, date, timing inputs inputs #
################################################
##Shearing date
###sire
date_shear_sida0e0b0xyg0 = f_g2g(pinp.sheep['i_date_shear_sixg0'],'sire',uinp.parameters['i_x_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len'],swap=True,left_pos2=pinp.sheep['i_i_pos'],right_pos2=uinp.parameters['i_x_pos'])[:,pinp.sheep['i_mask_i'],...]
###dam
date_shear_sida0e0b0xyg1 = f_g2g(pinp.sheep['i_date_shear_sixg1'],'dams',uinp.parameters['i_x_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len'],swap=True,left_pos2=pinp.sheep['i_i_pos'],right_pos2=uinp.parameters['i_x_pos'])[:,pinp.sheep['i_mask_i'],...]
###off
date_shear_sida0e0b0xyg3 = f_g2g(pinp.sheep['i_date_shear_sixg3'],'offs',uinp.parameters['i_x_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len'],swap=True,left_pos2=pinp.sheep['i_i_pos'],right_pos2=uinp.parameters['i_x_pos'])[:,pinp.sheep['i_mask_i'],...]
##join
join_cycles_ida0e0b0xyg1 = f_g2g(pinp.sheep['i_join_cycles_ig1'],'dams',pinp.sheep['i_i_pos'])[pinp.sheep['i_mask_i'],...]
##lamb and lost
gbal_oa1e1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['i_gbal_og1'],'dams',uinp.structure['i_p_pos']) #need axis up to p so that p association can be applied
##scanning
scan_oa1e1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['i_scan_og1'],'dams',uinp.structure['i_p_pos']) #need axis up to p so that p association can be applied
##post weaning management
wean_oa1e1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['i_wean_og1'],'dams',uinp.structure['i_p_pos']) #need axis up to p so that p association can be applied
##age weaning
age_wean_a0e0b0xyg3 = f_g2g(pinp.sheep['i_age_wean_a0g3'],'offs',pinp.sheep['i_a0_pos']).astype('timedelta64[D]')[pinp.sheep['i_mask_a']]
##association between offspring and sire/dam (used to determine the wean age of sire and dams based on the inputted wean age of offs)
a_g3_g0 = f_g2g(pinp.sheep['ia_g3_g0'],'sire')
a_g3_g1 = f_g2g(pinp.sheep['ia_g3_g1'],'dams')
##date first lamb is born - need to apply i mask to these inputs
date_born1st_ida0e0b0xyg0 = f_g2g(pinp.sheep['i_date_born1st_ig0'],'sire',pinp.sheep['i_i_pos']).astype('datetime64[D]')[pinp.sheep['i_mask_i'],...]
date_born1st_ida0e0b0xyg1 = f_g2g(pinp.sheep['i_date_born1st_ig1'],'dams',pinp.sheep['i_i_pos']).astype('datetime64[D]')[pinp.sheep['i_mask_i'],...]
date_born1st_oa1e1b1nwzida0e0b0xyg2 = f_g2g(pinp.sheep['i_date_born1st_oig2'],'yatf',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'],pinp.sheep['i_o_len'],swap=True,left_pos2=uinp.structure['i_p_pos'],right_pos2=pinp.sheep['i_i_pos']).astype('datetime64[D]')[:,:,:,:,:,:,:,pinp.sheep['i_mask_i'],...] #left2 = e1-1 because e1 needs to be included for the calculation following
date_born1st_ida0e0b0xyg3 = f_g2g(pinp.sheep['i_date_born1st_idg3'],'offs',uinp.parameters['i_d_pos'],uinp.parameters['i_d_len'],pinp.sheep['i_i_len'],swap=True).astype('datetime64[D]')[pinp.sheep['i_mask_i'],...]
##mating
sire_propn_oa1e1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['i_ram_propn_oig1'],'dams',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_o_len'],swap=True,left_pos2=uinp.structure['i_p_pos'],right_pos2=pinp.sheep['i_i_pos'])[:,:,:,:,:,:,:,pinp.sheep['i_mask_i'],...]
sire_periods_g0p8 = np.swapaxes(pinp.sheep['i_sire_periods_p8g0'][pinp.sheep['i_mask_p8']], 0, 1)


############################
### feed supply inputs     #
############################
##feedsupply
###feedsupply option selected
a_r_zida0e0b0xyg0 = f_g2g(pinp.sheep['ia_r1_zig0'],'sire',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True)
a_r_zida0e0b0xyg1 = f_g2g(pinp.sheep['ia_r1_zig1'],'dams',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True)
a_r_zida0e0b0xyg3 = f_g2g(pinp.sheep['ia_r1_zig3'],'offs',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True)
###feed variation for dams
a_r2_k0e1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['ia_r2_k0ig1'],'dams',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k0_len'],swap=True,left_pos2=pinp.sheep['i_a1_pos'],right_pos2=pinp.sheep['i_i_pos'])
a_r2_k1b1nwzida0e0b0xyg1 = f_g2g(pinp.sheep['ia_r2_k1ig1'],'dams',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k1_len'],swap=True,left_pos2=pinp.sheep['i_e1_pos'],right_pos2=pinp.sheep['i_i_pos'])
a_r2_k2nwzida0e0b0xyg1 = f_g2g(pinp.sheep['ia_r2_k2ig1'],'dams',pinp.sheep['i_i_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k2_len'],swap=True,left_pos2=uinp.parameters['i_b1_pos'],right_pos2=pinp.sheep['i_i_pos'])  #add axis between g and i and i and b1
###feed variation for offs
a_r2_idk0e0b0xyg3 = f_g2g(pinp.sheep['ia_r2_ik0g3'],'offs',pinp.sheep['i_a0_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k0_len'],left_pos2=pinp.sheep['i_i_pos'],right_pos2=pinp.sheep['i_a0_pos'])
a_r2_ik3a0e0b0xyg3 = f_g2g(pinp.sheep['ia_r2_ik3g3'],'offs',uinp.parameters['i_d_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k3_len'])
a_r2_ida0e0k4xyg3 = f_g2g(pinp.sheep['ia_r2_ik4g3'],'offs',uinp.parameters['i_b0_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k4_len'],left_pos2=pinp.sheep['i_i_pos'],right_pos2=uinp.parameters['i_b0_pos'])  #add axis between g and bo and b0 and i
a_r2_ida0e0b0k5yg3 = f_g2g(pinp.sheep['ia_r2_ik5g3'],'offs',uinp.parameters['i_x_pos'],pinp.sheep['i_i_len'], pinp.sheep['i_k5_len'],left_pos2=pinp.sheep['i_i_pos'],right_pos2=uinp.parameters['i_x_pos'])  #add axis between g and bo and b0 and i

##std feed options
feedoptions_r1pj0 = np.rollaxis(pinp.feedsupply['i_feedoptions_r1pj0'].reshape(pinp.feedsupply['i_j0_len'],pinp.feedsupply['i_r1_len'],pinp.feedsupply['i_feedoptions_r1pj0'].shape[-1]), 0, 3)[:,0:len_p,:] #slice off extra p periods so it is the same length as the sim periods
##feed variation
feedoptions_var_r2p = pinp.feedsupply['i_feedoptions_var_r2p'][:,0:len_p] #slice off extra p periods so it is the same length as the sim periods
##an association between the k2 cluster (feed variation) and reproductive management (scanning, gbal & weaning). 
a_k2_vlsb1 = uinp.structure['ia_k2_vlsb1'].reshape(uinp.structure['i_len_v'], uinp.structure['i_len_l'], uinp.structure['i_len_s'], uinp.structure['ia_k2_vlsb1'].shape[-1])


###################################
###group independent              #  type(pinp.sheep['i_mask_z']).dtype
###################################
nyatf_b1nwzida0e0b0xyg = f_reshape_expand(uinp.structure['a_nyatf_b1'], uinp.parameters['i_b1_pos'])
##nfoet expanded
nfoet_b1nwzida0e0b0xyg = f_reshape_expand(uinp.structure['a_nfoet_b1'], uinp.parameters['i_b1_pos'])
##legume proportion in each period
legume_p6a1e1b1nwzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_legume_p6z'], pinp.sheep['i_z_pos'], pinp.sheep['i_p6_len'], pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (assosiation section)
##estimated foo and dmd for the midas periods - apply z mask
paststd_foo_p6a1e1b1j0wzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_paststd_foo_p6zj0'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], len_ax2=pinp.feedsupply['i_j0_len'], swap=True, ax1=1, ax2=2, left_pos2=uinp.structure['i_n_pos'], right_pos2=pinp.sheep['i_z_pos'], left_pos3=uinp.structure['i_p_pos'], right_pos3=uinp.structure['i_n_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (assosiation section), axis order doesnt matter because sliced when used
paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_paststd_dmd_p6zj0'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], len_ax2=pinp.feedsupply['i_j0_len'], swap=True, ax1=1, ax2=2, left_pos2=uinp.structure['i_n_pos'], right_pos2=pinp.sheep['i_z_pos'], left_pos3=uinp.structure['i_p_pos'], right_pos3=uinp.structure['i_n_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (assosiation section), axis order doesnt matter because sliced when used
pasture_stage_p6a1e1b1j0wzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_pasture_stage_p6z'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (assosiation section)
##season type
i_season_propn_z=np.array([pinp.sheep['i_season_propn_z']]) #convert to np array - this is required if inputs only have one season
season_propn_zida0e0b0xyg = f_reshape_expand(i_season_propn_z, pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #minum 1 because p axis needs to be added
season_propn_zida0e0b0xyg = season_propn_pa1e1b1nwzida0e0b0xyg/sum(i_season_propn_z[pinp.sheep['i_mask_z']]) #adjust probability of each season to account for some seasons being masked out
##wind speed
ws_m4a1e1b1nwzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_ws_m4'], uinp.structure['i_p_pos']) 
##expected stocking density
density_p6a1e1b1nwzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_density_p6z'], pinp.sheep['i_z_pos'], pinp.sheep['i_p6_len'], pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (assosiation section)
##nutrition adjustment for expected stocking density
density_nwzida0e0b0xyg1 = f_reshape_expand(uinp.structure['i_density_g1_n'], uinp.structure['i_n_pos'])
density_nwzida0e0b0xyg3 = f_reshape_expand(uinp.structure['i_density_g3_n'], uinp.structure['i_n_pos'])
##Calculation of rainfall distribution across the week - i_rain_distribution_m4m1 = how much rain falls on each day of the week sorted in order of quantity of rain. SO the most rain falls on the day with the highest rainfall.
rain_m4a1e1b1nwzida0e0b0xygm1 = f_reshape_expand(pinp.sheep['i_rain_m4'][...,na] * pinp.sheep['i_rain_distribution_m4m1'] * (7/30.4), uinp.structure['i_p_pos']-1,right_pos=-1) #-1 because p is -16 when m1 axis is included
##Mean daily temperature
temp_ave_m4a1e1b1nwzida0e0b0xyg= f_reshape_expand(pinp.sheep['i_temp_ave_m4'], uinp.structure['i_p_pos'])
##Mean daily maximum temperature
temp_max_m4a1e1b1nwzida0e0b0xyg= f_reshape_expand(pinp.sheep['i_temp_max_m4'], uinp.structure['i_p_pos'])
##Mean daily minimum temperature
temp_min_m4a1e1b1nwzida0e0b0xyg= f_reshape_expand(pinp.sheep['i_temp_min_m4'], uinp.structure['i_p_pos'])
##latitude
lat_deg = pinp.sheep['i_latitude']
lat_rad = np.radians(pinp.sheep['i_latitude'])

############################
### sim param arrays       # '''csiro params '''
############################
##convert input params from c to g
###production params
agedam_propn_da0e0b0xyg0, agedam_propn_da0e0b0xyg1, agedam_propn_da0e0b0xyg2, agedam_propn_da0e0b0xyg03 = f_c2g(uinp.parameters['i_agedam_propn_std_dc2'], uinp.parameters['i_agedam_propn_y'], uinp.parameters['i_agedam_propn_pos']) #yatf and off never used
aw_propn_yg0, aw_propn_yg1, aw_propn_yg2, aw_propn_yg3 = f_c2g(uinp.parameters['i_aw_propn_wean_c2'], uinp.parameters['i_aw_wean_y'])
bw_propn_yg0, bw_propn_yg1, bw_propn_yg2, bw_propn_yg3 = f_c2g(uinp.parameters['i_bw_propn_wean_c2'], uinp.parameters['i_bw_wean_y'])
#^   btrt_yg0, btrt_yg1, btrt_yg2, btrt_yg3 = f_c2g(uinp.parameters['i_scan_std_c2'], uinp.parameters['i_scan_std_y'])
cfw_propn_yg0, cfw_propn_yg1, cfw_propn_yg2, cfw_propn_yg3 = f_c2g(uinp.parameters['i_cfw_propn_c2'], uinp.parameters['i_cfw_propn_y'])
scan_std_yg0, scan_std_yg1, scan_std_yg2, scan_std_yg3 = f_c2g(uinp.parameters['i_scan_std_c2'], uinp.parameters['i_scan_std_y'])
lss_std_yg0, lss_std_yg1, lss_std_yg2, lss_std_yg3 = f_c2g(uinp.parameters['i_lss_std_c2'], uinp.parameters['i_lss_std_y'])
lstr_std_yg0, lstr_std_yg1, lstr_std_yg2, lstr_std_yg3 = f_c2g(uinp.parameters['i_lstr_std_c2'], uinp.parameters['i_lstr_std_y'])
lstw_std_yg0, lstw_std_yg1, lstw_std_yg2, lstw_std_yg3 = f_c2g(uinp.parameters['i_lstw_std_c2'], uinp.parameters['i_lstw_std_y'])
mw_propn_yg0, mw_propn_yg1, mw_propn_yg2, mw_propn_yg3 = f_c2g(uinp.parameters['i_mw_propn_wean_c2'], uinp.parameters['i_mw_wean_y'])
sfd_yg0, sfd_yg1, sfd_yg2, sfd_yg3 = f_c2g(uinp.parameters['i_sfd_c2'], uinp.parameters['i_sfd_y'])
sfw_yg0, sfw_yg1, sfw_yg2, sfw_yg3 = f_c2g(uinp.parameters['i_sfw_c2'], uinp.parameters['i_sfw_y'])
srw_yg0, srw_yg1, srw_yg2, srw_yg3 = f_c2g(uinp.parameters['i_srw_c2'], uinp.parameters['i_srw_y'])

###sim params
ca_sire, ca_dams, ca_yatf, ca_offs = f_c2g(uinp.parameters['i_ca_c2'], uinp.parameters['i_ca_y'], uinp.parameters['i_ca_pos'], uinp.parameters['i_ca_len'])
cb0_sire, cb0_dams, cb0_yatf, cb0_offs = f_c2g(uinp.parameters['i_cb0_c2'], uinp.parameters['i_cb0_y'], uinp.parameters['i_cb0_pos'], uinp.parameters['i_cb0_len'], uinp.parameters['i_cb0_len2'])
cc_sire, cc_dams, cc_yatf, cc_offs = f_c2g(uinp.parameters['i_cc_c2'], uinp.parameters['i_cc_y'], uinp.parameters['i_cc_pos'], uinp.parameters['i_cc_len'])
cd_sire, cd_dams, cd_yatf, cd_offs = f_c2g(uinp.parameters['i_cd_c2'], uinp.parameters['i_cd_y'], uinp.parameters['i_cd_pos'], uinp.parameters['i_cd_len'])
ce_sire, ce_dams, ce_yatf, ce_offs = f_c2g(uinp.parameters['i_ce_c2'], uinp.parameters['i_ce_y'], uinp.parameters['i_ce_pos'], uinp.parameters['i_ce_len'], uinp.parameters['i_ce_len2'])
cf_sire, cf_dams, cf_yatf, cf_offs = f_c2g(uinp.parameters['i_cf_c2'], uinp.parameters['i_cf_y'], uinp.parameters['i_cf_pos'], uinp.parameters['i_cf_len'])
cg_sire, cg_dams, cg_yatf, cg_offs = f_c2g(uinp.parameters['i_cg_c2'], uinp.parameters['i_cg_y'], uinp.parameters['i_cg_pos'], uinp.parameters['i_cg_len'])
ch_sire, ch_dams, ch_yatf, ch_offs = f_c2g(uinp.parameters['i_ch_c2'], uinp.parameters['i_ch_y'], uinp.parameters['i_ch_pos'], uinp.parameters['i_ch_len'])
ci_sire, ci_dams, ci_yatf, ci_offs = f_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'], uinp.parameters['i_ci_len'])
ck_sire, ck_dams, ck_yatf, ck_offs = f_c2g(uinp.parameters['i_ck_c2'], uinp.parameters['i_ck_y'], uinp.parameters['i_ck_pos'], uinp.parameters['i_ck_len'])
cl0_sire, cl0_dams, cl0_yatf, cl0_offs = f_c2g(uinp.parameters['i_cl0_c2'], uinp.parameters['i_cl0_y'], uinp.parameters['i_cl0_pos'], uinp.parameters['i_cl0_len'], uinp.parameters['i_cl0_len2'])
cl1_sire, cl1_dams, cl1_yatf, cl1_offs = f_c2g(uinp.parameters['i_cl1_c2'], uinp.parameters['i_cl1_y'], uinp.parameters['i_cl1_pos'], uinp.parameters['i_cl1_len'], uinp.parameters['i_cl1_len2'])
cl_sire, cl_dams, cl_yatf, cl_offs = f_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'], uinp.parameters['i_cl_len'])
cm_sire, cm_dams, cm_yatf, cm_offs = f_c2g(uinp.parameters['i_cm_c2'], uinp.parameters['i_cm_y'], uinp.parameters['i_cm_pos'], uinp.parameters['i_cm_len'])
cn_sire, cn_dams, cn_yatf, cn_offs = f_c2g(uinp.parameters['i_cn_c2'], uinp.parameters['i_cn_y'], uinp.parameters['i_cn_pos'], uinp.parameters['i_cn_len'])
cp_sire, cp_dams, cp_yatf, cp_offs = f_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'], uinp.parameters['i_cp_len'])
cr_sire, cr_dams, cr_yatf, cr_offs = f_c2g(uinp.parameters['i_cr_c2'], uinp.parameters['i_cr_y'], uinp.parameters['i_cr_pos'], uinp.parameters['i_cr_len'])
crd_sire, crd_dams, crd_yatf, crd_offs = f_c2g(uinp.parameters['i_crd_c2'], uinp.parameters['i_crd_y'], uinp.parameters['i_crd_pos'], uinp.parameters['i_crd_len'])
cu0_sire, cu0_dams, cu0_yatf, cu0_offs = f_c2g(uinp.parameters['i_cu0_c2'], uinp.parameters['i_cu0_y'], uinp.parameters['i_cu0_pos'], uinp.parameters['i_cu0_len'])
cu1_sire, cu1_dams, cu1_yatf, cu1_offs = f_c2g(uinp.parameters['i_cu1_c2'], uinp.parameters['i_cu1_y'], uinp.parameters['i_cu1_pos'], uinp.parameters['i_cu1_len'], uinp.parameters['i_cu1_len2'])
cu2_sire, cu2_dams, cu2_yatf, cu2_offs = f_c2g(uinp.parameters['i_cu2_c2'], uinp.parameters['i_cu2_y'], uinp.parameters['i_cu2_pos'], uinp.parameters['i_cu2_len'], uinp.parameters['i_cu2_len2'])
cw_sire, cw_dams, cw_yatf, cw_offs = f_c2g(uinp.parameters['i_cw_c2'], uinp.parameters['i_cw_y'], uinp.parameters['i_cw_pos'], uinp.parameters['i_cw_len'])
cx_sire, cx_dams, cx_yatf, cx_offs = f_c2g(uinp.parameters['i_cx_c2'], uinp.parameters['i_cx_y'], uinp.parameters['i_cx_pos'], uinp.parameters['i_cx_len'], uinp.parameters['i_cx_len2'])
##pasture params
cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu3_len'], uinp.pastparameters['i_cu3_len2'])
cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu4_len'], uinp.pastparameters['i_cu4_len2'])
##Convert the cl0 & cl1 to cb1 (dams and yatf only need cb1, sires and offs dont have b1 axis)
cb1_dams = cl0_dams[:,uinp.structure['a_nfoet_b1']] + cl1_dams[:,uinp.structure['a_nyatf_b1']]
cb1_yatf = cl0_yatf[:,uinp.structure['a_nfoet_b1']] + cl1_yatf[:,uinp.structure['a_nyatf_b1']]
###Alter select slices only for yatf (yatf dont have cb0 axis - instead they use cb1 so it allings with dams)
cb1_yatf[12, ...] = np.expand_dims(cb0_yatf[12, uinp.structure['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array alings with b1
cb1_yatf[13, ...] = np.expand_dims(cb0_yatf[13, uinp.structure['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array alings with b1
cb1_yatf[15, ...] = np.expand_dims(cb0_yatf[15, uinp.structure['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array alings with b1
cb1_yatf[17, ...] = np.expand_dims(cb0_yatf[17, uinp.structure['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array alings with b1
cb1_yatf[18, ...] = np.expand_dims(cb0_yatf[18, uinp.structure['ia_b0_b1']], axis = tuple(range(uinp.parameters['i_cl1_pos']+1,-3))) #add singleton axis between b0 and x so that b0 array alings with b1




############################
## calc for associations   #
############################
##date joined (when the rams go in)
date_joined_oa1e1b1nwzida0e0b0xyg1 = (date_born1st_oa1e1b1nwzida0e0b0xyg2) - cp_dams[1,...,0:1,:].astype('timedelta64[D]') #take slice 0 from y axis because cp1 is not affected by genetic merit
##expand feed periods over all the years of the sim so that an association between sim period can be made.
feedperiods_p6 = np.array(pinp.feed_inputs['feed_periods']['date']).astype('datetime64[D]') #convert from df to numpy
feedperiods_p6 = feedperiods_p6 + np.timedelta64(365,'D') * ((date_start_p[0].astype(object).year -1) - feedperiods_p6[0].astype(object).year) #this is to make sure the fisrt sim period date is greater than the first feed period date.
feedperiods_p6 = np.ravel(feedperiods_p6  + (np.arange(np.ceil(uinp.structure['i_age_max'] +1)) * np.timedelta64(365,'D') )[...,na]) #expand then ravel to return 1d array of the feed period dates expanded the lenght of the sim.

###############################
# Feed variation period calcs #
###############################
##early pregnancy fvp start - The pre-joining accumulation of the dams from the previous reproduction cycle - this date must correspond to the start date of period
prejoining_aprox_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 - uinp.structure['prejoin_offset'] #approx date of prejoining - adjusted to be the start of a sim period in the next step
idx = np.searchsorted(date_start_p, prejoining_aprox_oa1e1b1nwzida0e0b0xyg1)-1 #gets the sim period index for the period before the prejoining
prejoining_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx]
fvp_1_start_oa1e1b1nwzida0e0b0xyg1 = prejoining_oa1e1b1nwzida0e0b0xyg1
fvp_1_type_oa1e1b1nwzida0e0b0xyg1 = np.full(fvp_1_start_oa1e1b1nwzida0e0b0xyg1.shape,1)
##late pregnancy fvp start - Scanning if carried out, day 90 from joining (ram in) if not scanned.
fvp_2_start_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') + pinp.sheep['i_scan_day'][scan_oa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]') 
fvp_2_type_oa1e1b1nwzida0e0b0xyg1 = np.full(fvp_2_start_oa1e1b1nwzida0e0b0xyg1.shape,2)
## lactation fvp start - average date of lambing (with e axis)
fvp_3_start_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	
fvp_3_type_oa1e1b1nwzida0e0b0xyg1 = np.full(fvp_3_start_oa1e1b1nwzida0e0b0xyg1.shape,3)
##post weaning recovery fvp start - weaning date of offspring
fvp_4_start_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + age_wean_a0e0b0xyg3	
fvp_4_type_oa1e1b1nwzida0e0b0xyg1 = np.full(fvp_4_start_oa1e1b1nwzida0e0b0xyg1.shape,4)
## break of season fvp ^the following two lines of code will have to change once season type is included into the feedperiod inputs (the input will have z axis so the reshaping will need to be done in two steps ie pass in pos2 arg) and apply z mask
fvp_0_start_y = pinp.feed_inputs['feed_periods'].loc[0,'date'].to_datetime64().astype('datetime64[D]') + (np.arange(np.ceil(uinp.structure['i_age_max'])) * np.timedelta64(365,'D'))
fvp_0_start_ya1e1b1nwzida0e0b0xyg = f_reshape_expand(fvp_0_start_y, left_pos=uinp.structure['i_p_pos'])
##first need to manually expand arrays because arrays need to be same size to stack/concat but cant use broadcast function because fvp 4 array has a different length o/y axis
###create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
shape = np.maximum.reduce([fvp_1_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],fvp_2_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_3_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_4_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_0_start_ya1e1b1nwzida0e0b0xyg.shape[1:]]) #create shape which has the max size, this is used for o array
fvp_0_start_ya1e1b1nwzida0e0b0xyg = np.broadcast_to(fvp_0_start_ya1e1b1nwzida0e0b0xyg,(fvp_0_start_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape))
fvp_0_type_ya1e1b1nwzida0e0b0xyg = np.full(fvp_0_start_ya1e1b1nwzida0e0b0xyg.shape,0)
##broadcast fvp 0-3 - makes sure arrays are same size
fvp_1234_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_arrays(fvp_1_start_oa1e1b1nwzida0e0b0xyg1,fvp_2_start_oa1e1b1nwzida0e0b0xyg1,fvp_3_start_oa1e1b1nwzida0e0b0xyg1,fvp_4_start_oa1e1b1nwzida0e0b0xyg1)
fvp_1234_type_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_arrays(fvp_1_type_oa1e1b1nwzida0e0b0xyg1,fvp_2_type_oa1e1b1nwzida0e0b0xyg1,fvp_3_type_oa1e1b1nwzida0e0b0xyg1,fvp_4_type_oa1e1b1nwzida0e0b0xyg1)
##stack sort into date order
fvp_start_oa1e1b1nwzida0e0b0xyg1 = np.concatenate(([*fvp_1234_start_oa1e1b1nwzida0e0b0xyg1,fvp_0_start_ya1e1b1nwzida0e0b0xyg]),axis=0)
fvp_type_oa1e1b1nwzida0e0b0xyg1 = np.concatenate(([*fvp_1234_type_oa1e1b1nwzida0e0b0xyg1,fvp_0_type_ya1e1b1nwzida0e0b0xyg]),axis=0)
ind=np.argsort(fvp_start_oa1e1b1nwzida0e0b0xyg1, axis=0)
fvp_date_start_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_oa1e1b1nwzida0e0b0xyg1, ind, axis=0) 
fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_oa1e1b1nwzida0e0b0xyg1, ind, axis=0)
##proportion of each sim period in each feed period ^might want to swap the f and p axis 
fvp_length_fa1e1b1nwzida0e0b0xyg1 = np.append(fvp_date_start_fa1e1b1nwzida0e0b0xyg1,np.broadcast_to(date_end_pa1e1b1nwzida0e0b0xyg[-1:]+1,(1,)+tuple(fvp_date_start_fa1e1b1nwzida0e0b0xyg1.shape[1:])),axis=0)[1:]-fvp_date_start_fa1e1b1nwzida0e0b0xyg1
propn_pfa1e1b1nwzida0e0b0xyg1=fun.range_allocation_np(np.append(date_start_pa1e1b1nwzida0e0b0xyg,date_end_pa1e1b1nwzida0e0b0xyg[-1:]+1,axis=0),fvp_date_start_fa1e1b1nwzida0e0b0xyg1,fvp_length_fa1e1b1nwzida0e0b0xyg1) #the function needs the end of the last period so appended that to the start array. +1 so that i get the start of the next period not the last day of the current period


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
'''
key to help understand the results from the associations

Name         	    When it increments
Date_joined            Joining
Date_joined2           prejoining
Date_mated             Joining
Date_scan              prejoining
Date_born              birth
Date_born2             joining
Date_wean              birth
Date_wean2             prejoining
Date_prejoin_next      prejoining
Date_prejoin	Joining   prejoining

'''


##joining oppotunity association
a_nextprejoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, prejoining_oa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 0)
a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, prejoining_oa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_joined_oa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
# a_nextjoining_o_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_joined_oa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]'), date_start_p, 0)
a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_born1st_oa1e1b1nwzida0e0b0xyg2.astype('datetime64[D]'), date_end_p, 1)
# a_nextyatf_o_pa1e1b1nwzida0e0b0xyg2 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_born1st_oa1e1b1nwzida0e0b0xyg2.astype('datetime64[D]'), date_start_p, 0)
##dam age association, note this is the same as joining opp (just using a new variable name to avoid confusion in the rest of the code)
a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2 = a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2

##MIDAS feed period for each sim period
a_prevfeedperiod_p = np.apply_along_axis(sfun.f_next_prev_association, 0, feedperiods_p6, date_end_p, 1) % (len(pinp.feed_inputs['feed_periods'])-1) #% 10 required to convert association back to only the number of feed periods, -1 because the end feed period date is included

##shearing opp ^possibly need to fill in axis between p and i
a_prev_s_pida0e0b0xyg0 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg0.astype('datetime64[D]'), date_end_p, 1)
a_next_s_pida0e0b0xyg0 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg0.astype('datetime64[D]'), date_start_p, 0)
a_prev_s_pida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)
a_next_s_pida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg1.astype('datetime64[D]'), date_start_p, 0)
a_prev_s_pida0e0b0xyg3 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg3.astype('datetime64[D]'), date_end_p, 1)
a_next_s_pida0e0b0xyg3 = np.apply_along_axis(sfun.f_next_prev_association, 0, date_shear_sida0e0b0xyg3.astype('datetime64[D]'), date_start_p, 0)
##p7 to p association - used for equation systems
a_g0_p7_p = np.apply_along_axis(sfun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g0_p7'].astype('datetime64[D]'), date_end_p, 1)
a_g1_p7_p = np.apply_along_axis(sfun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g1_p7'].astype('datetime64[D]'), date_end_p, 1)
a_g2_p7_p = np.apply_along_axis(sfun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g2_p7'].astype('datetime64[D]'), date_end_p, 1)
a_g3_p7_p = np.apply_along_axis(sfun.f_next_prev_association, 0, pinp.sheep['i_eqn_date_g3_p7'].astype('datetime64[D]'), date_end_p, 1)
##month of each period (0 - 11 not 1 -12 because this is association array)
a_m4_p = date_start_p.astype('datetime64[M]').astype(int) % 12 
##feed variation period 
a_fvp_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, fvp_date_start_fa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]'), date_end_p, 1)


############################
### apply associations     #
############################
'''
The association applied determines when the increment to the next opportunity will occur:
    eg if you use a_prev_joining the date in the p slice will increment at joining each time.

'''
###management for weaning, gbal and scan options
wean_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(wean_oa1e1b1nwzida0e0b0xyg1,a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
gbal_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(gbal_oa1e1b1nwzida0e0b0xyg1,a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
scan_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(scan_oa1e1b1nwzida0e0b0xyg1,a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
###date, age, timing
date_born1st_pa1e1b1nwzida0e0b0xyg2=np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2,a_prevbirth_o_pa1e1b1nwzida0e0b0xyg2,0)
date_born1st2_pa1e1b1nwzida0e0b0xyg2=np.take_along_axis(date_born1st_oa1e1b1nwzida0e0b0xyg2,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0) #increments at prejoining
date_prejoin_next_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(prejoining_oa1e1b1nwzida0e0b0xyg1,a_nextprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
date_prejoin_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(prejoining_oa1e1b1nwzida0e0b0xyg1,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
date_joined_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(date_joined_oa1e1b1nwzida0e0b0xyg1,a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1,0)
date_joined2_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(date_joined_oa1e1b1nwzida0e0b0xyg1,a_prevprejoining_o_pa1e1b1nwzida0e0b0xyg1,0)
##yatf sim params - turn d to p axis
ce_yatf = np.expand_dims(ce_yatf, axis = tuple(range(uinp.structure['i_p_pos'],uinp.parameters['i_d_pos'])))
ce_yatf = np.take_along_axis(ce_yatf,a_prevbirth_d_pa1e1b1nwzida0e0b0xyg2[na,...],uinp.parameters['i_d_pos'])
##feed period
legume_pa1e1b1nwzida0e0b0xyg = legume_p6a1e1b1nwzida0e0b0xyg[a_prevfeedperiod_p,...]
##expected stocking density
density_pa1e1b1nwzida0e0b0xyg = density_p6a1e1b1nwzida0e0b0xyg[a_prevfeedperiod_p,...]
##select which equation is used for the sheep sim functions for each period
eqn_used_g0_q1p = pinp.sheep['i_eqn_used_g0_q1p7'][:, a_g0_p7_p]
eqn_used_g1_q1p = pinp.sheep['i_eqn_used_g1_q1p7'][:, a_g1_p7_p]
eqn_used_g2_q1p = pinp.sheep['i_eqn_used_g2_q1p7'][:, a_g2_p7_p]
eqn_used_g3_q1p = pinp.sheep['i_eqn_used_g3_q1p7'][:, a_g3_p7_p]
##convert foo and dmd for each feed period to each sim period
paststd_foo_pa1e1b1j0wzida0e0b0xyg = paststd_foo_p6a1e1b1j0wzida0e0b0xyg[a_prevfeedperiod_p,...]
paststd_dmd_pa1e1b1j0wzida0e0b0xyg = paststd_dmd_p6a1e1b1j0wzida0e0b0xyg[a_prevfeedperiod_p,...]
pasture_stage_pa1e1b1j0wzida0e0b0xyg = pasture_stage_p6a1e1b1j0wzida0e0b0xyg[a_prevfeedperiod_p,...]
##mating
sire_propn_pa1e1b1nwzida0e0b0xyg1=np.take_along_axis(sire_propn_oa1e1b1nwzida0e0b0xyg1,a_prevjoining_o_pa1e1b1nwzida0e0b0xyg1,0) #np.takealong uses the number in the second array as the index for the first array. and returns a same shaped array
##weather 
ws_pa1e1b1nwzida0e0b0xyg = ws_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
rain_pa1e1b1nwzida0e0b0xygm1 = rain_m4a1e1b1nwzida0e0b0xygm1[a_m4_p]
temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
##feed variation ^dont know if these arrays are needed
fvp_type_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)
fvp_date_start_pa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_date_start_fa1e1b1nwzida0e0b0xyg1,a_fvp_pa1e1b1nwzida0e0b0xyg1,0)

###########################
##genotype calculations   #
###########################
##calc proportion of dry, singles, twin and triplets
dstwtr_l0yg0 = f_DSTw(scan_std_yg0)
dstwtr_l0yg1 = f_DSTw(scan_std_yg1)
dstwtr_l0yg2 = f_DSTw(scan_std_yg2)
dstwtr_l0yg3 = f_DSTw(scan_std_yg3)
##calc propn of offs in each BTRT b0 category - 11, 22, 33, 21, 32, 31 -
btrt_propn_b0xyg0 = f_btrt0(dstwtr_l0yg0,lss_std_yg0,lstw_std_yg0,lstr_std_yg0)
btrt_propn_b0xyg1 = f_btrt0(dstwtr_l0yg1,lss_std_yg1,lstw_std_yg1,lstr_std_yg1)
btrt_propn_b0xyg2 = f_btrt0(dstwtr_l0yg2,lss_std_yg2,lstw_std_yg2,lstr_std_yg2)
btrt_propn_b0xyg3 = f_btrt0(dstwtr_l0yg3,lss_std_yg3,lstw_std_yg3,lstr_std_yg3)
###calc adjustments sfw ^ ce12 &13 should be scaled by relsize (similar to ce15)
adja_sfw_d_a0e0b0xyg0 = np.sum(ce_sire[12, ...] * agedam_propn_da0e0b0xyg0, axis = 0)
adja_sfw_d_a0e0b0xyg1 = np.sum(ce_dams[12, ...] * agedam_propn_da0e0b0xyg1, axis = 0)
adja_sfw_d_pa1e1b1nwzida0e0b0xyg2 = ce_yatf[12,...]
adja_sfw_d_da0e0b0xyg3 = ce_offs[12, ...]
adja_sfw_b0_xyg0 = np.sum(cb0_sire[12, ...] * btrt_propn_b0xyg0, axis = 0)
adja_sfw_b0_xyg1 = np.sum(cb0_dams[12, ...] * btrt_propn_b0xyg1, axis = 0)
adja_sfw_b0_b0xyg2 = cb1_yatf[12, ...]
adja_sfw_b0_b0xyg3 = cb0_offs[12, ...]
###apply adjustments sfw
sfw_a0e0b0xyg0 = sfw_yg0 + adja_sfw_d_a0e0b0xyg0 + adja_sfw_b0_xyg0
sfw_a0e0b0xyg1 = sfw_yg1 + adja_sfw_d_a0e0b0xyg1 + adja_sfw_b0_xyg1
sfw_pa1e1b1nwzida0e0b0xyg2 = sfw_yg2 + adja_sfw_d_pa1e1b1nwzida0e0b0xyg2 + adja_sfw_b0_b0xyg2
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
srw_xyg0 = srw_yg0 * cx_sire[11, 0:1, ...]  #11 is the srw parameter, 0:1 is the sire gender slice (retaining the axis).
srw_xyg1 = srw_yg1 * cx_dams[11, 1:2, ...]
srw_xyg2 = srw_yg2 * cx_yatf[11, ...] #all gender slices
srw_xyg3 = srw_yg3 * cx_offs[11, ...] #all gender slices

##Standard birth weight -
w_b_std_b0xyg0 = srw_xyg0 * np.sum(cb0_sire[15, ...] * btrt_propn_b0xyg0, axis = uinp.parameters['i_b0_pos'], keepdims=True) * cx_sire[15, 0:1, ...]
w_b_std_b0xyg1 = srw_xyg1 * np.sum(cb0_dams[15, ...] * btrt_propn_b0xyg1, axis = uinp.parameters['i_b0_pos'], keepdims=True) * cx_dams[15, 1:2, ...]
w_b_std_b0xyg3 = srw_xyg3 * cb0_offs[15, ...] * cx_offs[15, ...]
##fetal param - normal birthweight young - used as target birthweight duing pregnancy if sheep fed well. Therefore average gender effect.
w_b_std_y_b1nwzida0e0b0xyg1 = srw_xyg2 * cb1_yatf[15, ...] #gender not considers therefore no cx (gender neutral = 1)

##wool growth efficiency (sfw same for all animals)
wge_a0e0b0xyg0 = sfw_a0e0b0xyg0 / (srw_xyg0 / cx_sire[11, 0:1, ...]) #wge is sfw divided by srw of a ewe of given genotype therefore convert srw_sire back to dam by dividing by cx
wge_a0e0b0xyg1 = sfw_a0e0b0xyg1 / srw_xyg1
wge_pa1e1b1nwzida0e0b0xyg2 = sfw_pa1e1b1nwzida0e0b0xyg2 / srw_xyg2[1:2, ...] #take female slice of srw
wge_da0e0b0xyg3 = sfw_da0e0b0xyg3 / srw_xyg3[1:2, ...] #take female slice of srw

##Legume impact on efficiency
lgf_eff_pa1e1b1nwzida0e0b0xyg0 = 1 + ck_sire[14,...] * legume_pa1e1b1nwzida0e0b0xyg
lgf_eff_pa1e1b1nwzida0e0b0xyg1 = 1 + ck_dams[14,...] * legume_pa1e1b1nwzida0e0b0xyg
lgf_eff_pa1e1b1nwzida0e0b0xyg2 = 1 + ck_yatf[14,...] * legume_pa1e1b1nwzida0e0b0xyg
lgf_eff_pa1e1b1nwzida0e0b0xyg3 = 1 + ck_offs[14,...] * legume_pa1e1b1nwzida0e0b0xyg

##Day length factor on wool
dlf_wool_pa1e1b1nwzida0e0b0xyg0 = 1 + cw_sire[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
dlf_wool_pa1e1b1nwzida0e0b0xyg1 = 1 + cw_dams[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
dlf_wool_pa1e1b1nwzida0e0b0xyg2 = 1 + cw_yatf[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)
dlf_wool_pa1e1b1nwzida0e0b0xyg3 = 1 + cw_offs[6,...] * (dl_pa1e1b1nwzida0e0b0xyg - 12)

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
##convert i_adjp to adjp - add necessary axes for 'a' and 'w'
adjp_lw_initial_a0e0b0xyg = np.expand_dims(pinp.sheep['i_adjp_lw_initial_a'], axis = tuple(range(1,-pinp.sheep['i_a0_pos'])))
adjp_lw_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_lw_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_lw_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_lw_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_cfw_initial_a0e0b0xyg = np.expand_dims(pinp.sheep['i_adjp_cfw_initial_a'], axis = tuple(range(1,-pinp.sheep['i_a0_pos'])))
adjp_cfw_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_cfw_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_cfw_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_cfw_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fd_initial_a0e0b0xyg = np.expand_dims(pinp.sheep['i_adjp_fd_initial_a'], axis = tuple(range(1,-pinp.sheep['i_a0_pos'])))
adjp_fd_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fd_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fd_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_fd_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fl_initial_a0e0b0xyg = np.expand_dims(pinp.sheep['i_adjp_fl_initial_a'], axis = tuple(range(1,-pinp.sheep['i_a0_pos'])))
adjp_fl_initial_wzida0e0b0xyg0 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w0'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fl_initial_wzida0e0b0xyg1 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w1'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))
adjp_fl_initial_wzida0e0b0xyg3 = np.expand_dims(uinp.structure['i_adjp_fl_initial_w3'], axis = tuple(range(1,-uinp.structure['i_w_pos'])))


##convert variable from c2 to g (yatf is not used, only here because it is return from the function) then addjust by initial lw pattern
lw_initial_yg0, lw_initial_yg1, lw_initial_yatf, lw_initial_yg3 = f_c2g(uinp.parameters['i_lw_initial_c2'], uinp.parameters['i_lw_initial_y'])
lw_initial_wzida0e0b0xyg0 = lw_initial_yg0 * (1 + adjp_lw_initial_wzida0e0b0xyg0)
lw_initial_wzida0e0b0xyg1 = lw_initial_yg1 * (1 + adjp_lw_initial_wzida0e0b0xyg1)
lw_initial_wzida0e0b0xyg3 = lw_initial_yg3 * (1 + adjp_lw_initial_wzida0e0b0xyg3)
cfw_initial_yg0, cfw_initial_yg1, cfw_initial_yatf, cfw_initial_yg3 = f_c2g(uinp.parameters['i_cfw_initial_c2'], uinp.parameters['i_cfw_initial_y'])
cfw_initial_wzida0e0b0xyg0 = cfw_initial_yg0 * (1 + adjp_cfw_initial_wzida0e0b0xyg0)
cfw_initial_wzida0e0b0xyg1 = cfw_initial_yg1 * (1 + adjp_cfw_initial_wzida0e0b0xyg1)
cfw_initial_wzida0e0b0xyg3 = cfw_initial_yg3 * (1 + adjp_cfw_initial_wzida0e0b0xyg3)
fd_initial_yg0, fd_initial_yg1, fd_initial_yatf, fd_initial_yg3 = f_c2g(uinp.parameters['i_fd_initial_c2'], uinp.parameters['i_fd_initial_y'])
fd_initial_wzida0e0b0xyg0 = fd_initial_yg0 * (1 + adjp_fd_initial_wzida0e0b0xyg0)
fd_initial_wzida0e0b0xyg1 = fd_initial_yg1 * (1 + adjp_fd_initial_wzida0e0b0xyg1)
fd_initial_wzida0e0b0xyg3 = fd_initial_yg3 * (1 + adjp_fd_initial_wzida0e0b0xyg3)
fl_initial_yg0, fl_initial_yg1, fl_initial_yatf, fl_initial_yg3 = f_c2g(uinp.parameters['i_fl_initial_c2'], uinp.parameters['i_fl_initial_y'])
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
adja_lw_initial_x_xyg3 = cx_offs[17, ...]
adja_cfw_initial_x_wzida0e0b0xyg0 = cx_sire[12, 0:1, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0
adja_cfw_initial_x_wzida0e0b0xyg1 = cx_dams[12, 1:2, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
adja_cfw_initial_x_wzida0e0b0xyg3 = cx_offs[12, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
adja_fd_initial_x_xyg0 = cx_sire[13, 0:1, ...]
adja_fd_initial_x_xyg1 = cx_dams[13, 1:2, ...]
adja_fd_initial_x_xyg3 = cx_offs[13, ...]
adja_fl_initial_x_wzida0e0b0xyg0 = cx_sire[12, 0:1, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 #Should be fl_initial / sfw  So more understandable to think of the eqn as being fl_initial * cx[0] (cfw adj due to gender) / sfw
adja_fl_initial_x_wzida0e0b0xyg1 = cx_dams[12, 1:2, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1
adja_fl_initial_x_wzida0e0b0xyg3 = cx_offs[12, ...] * fl_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
##adjust for dam age. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr)
adja_lw_initial_d_a0e0b0xyg0 = np.sum(ce_sire[17, ...] * agedam_propn_da0e0b0xyg0, axis=0) #d axis lost when summing
adja_lw_initial_d_a0e0b0xyg1 = np.sum(ce_dams[17, ...] * agedam_propn_da0e0b0xyg1, axis=0)
adja_lw_initial_d_da0e0b0xyg3 = ce_offs[17, ...]
adja_cfw_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[12, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_da0e0b0xyg0, axis=uinp.parameters['i_d_pos'], keepdims=True)
adja_cfw_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[12, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_da0e0b0xyg1, axis=uinp.parameters['i_d_pos'], keepdims=True)
adja_cfw_initial_d_wzida0e0b0xyg3 = ce_offs[12, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
adja_fd_initial_d_a0e0b0xyg0 = np.sum(ce_sire[13, ...] * agedam_propn_da0e0b0xyg0, axis=0) #d axis lost when summing
adja_fd_initial_d_a0e0b0xyg1 = np.sum(ce_dams[13, ...] * agedam_propn_da0e0b0xyg1, axis=0)
adja_fd_initial_d_da0e0b0xyg3 = ce_offs[13, ...]
adja_fl_initial_d_wzida0e0b0xyg0 = np.sum(ce_sire[12, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * agedam_propn_da0e0b0xyg0, axis=uinp.parameters['i_d_pos'], keepdims=True) #Should be fl_initial / sfw  So more understandable to think of the eqn as being fl_initial * cx[0] (cfw adj due to gender) / sfw
adja_fl_initial_d_wzida0e0b0xyg1 = np.sum(ce_dams[12, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * agedam_propn_da0e0b0xyg1, axis=uinp.parameters['i_d_pos'], keepdims=True)
adja_fl_initial_d_wzida0e0b0xyg3 = ce_offs[12, ...] * fl_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
##adjust for btrt. Note cfw changes throughout the year therefore the adjustment factor will not be the same all yr hence divide by std_fw (same for fl) eg the impact of gender on cfw will be much less after only a small time (the parameter is a yearly factor eg male sheep have 0.02 kg more wool each yr)
adja_lw_initial_b0_xyg0 = np.sum(cb0_sire[17, ...] * btrt_propn_b0xyg0, axis=0) #d axis lost when summing
adja_lw_initial_b0_xyg1 = np.sum(cb0_dams[17, ...] * btrt_propn_b0xyg1, axis=0)
adja_lw_initial_b0_b0xyg3 = cb0_offs[17, ...]
adja_cfw_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[12, ...] * cfw_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_propn_b0xyg0, axis=uinp.parameters['i_b0_pos'], keepdims=True)
adja_cfw_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[12, ...] * cfw_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_propn_b0xyg1, axis=uinp.parameters['i_b0_pos'], keepdims=True)
adja_cfw_initial_b0_wzida0e0b0xyg3 = cb0_offs[12, ...] * cfw_initial_wzida0e0b0xyg3 / sfw_da0e0b0xyg3
adja_fd_initial_b0_xyg0 = np.sum(cb0_sire[13, ...] * btrt_propn_b0xyg0, axis=0) #d axis lost when summing
adja_fd_initial_b0_xyg1 = np.sum(cb0_dams[13, ...] * btrt_propn_b0xyg1, axis=0)
adja_fd_initial_b0_b0xyg3 = cb0_offs[13, ...]
adja_fl_initial_b0_wzida0e0b0xyg0 = np.sum(cb0_sire[12, ...] * fl_initial_wzida0e0b0xyg0 / sfw_a0e0b0xyg0 * btrt_propn_b0xyg0, axis=uinp.parameters['i_b0_pos'], keepdims=True) #Should be fl_initial / sfw  So more understandable to think of the eqn as being fl_initial * cx[0] (cfw adj due to gender) / sfw
adja_fl_initial_b0_wzida0e0b0xyg1 = np.sum(cb0_dams[12, ...] * fl_initial_wzida0e0b0xyg1 / sfw_a0e0b0xyg1 * btrt_propn_b0xyg1, axis=uinp.parameters['i_b0_pos'], keepdims=True)
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
##calc aw, bw and mw (adipose, bone and muscel weight)
aw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * aw_propn_yg0
aw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * aw_propn_yg1
aw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * aw_propn_yg3
bw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * bw_propn_yg0
bw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * bw_propn_yg1
bw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * bw_propn_yg3
mw_initial_wzida0e0b0xyg0 = lw_initial_wzida0e0b0xyg0 * mw_propn_yg0
mw_initial_wzida0e0b0xyg1 = lw_initial_wzida0e0b0xyg1 * mw_propn_yg1
mw_initial_wzida0e0b0xyg3 = lw_initial_wzida0e0b0xyg3 * mw_propn_yg3


#######################
##Age, date, timing 1 #
#######################

## date mated (when the ewe actually concieves)
date_mated_pa1e1b1nwzida0e0b0xyg1 = date_joined_pa1e1b1nwzida0e0b0xyg1.astype('datetime64[D]') + (cf_dams[4, ..., 0:1, :] * (index_e1b1nwzida0e0b0xyg + 0.5)).astype('timedelta64[D]')
##Age of dam when first lamb is born
agedam_lamb1st_a1e1b1nwzida0e0b0xyg3 = np.swapaxes(date_born1st_oa1e1b1nwzida0e0b0xyg2 - date_born1st_ida0e0b0xyg1,0,uinp.parameters['i_d_pos'])[0,...] #replace the d axis with the o axis then remove the d axis by taking slice 0 (note the d axis was not active)
agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 = agedam_lamb1st_a1e1b1nwzida0e0b0xyg3[...,a_g3_g0]
agedam_lamb1st_a1e1b1nwzida0e0b0xyg1 = agedam_lamb1st_a1e1b1nwzida0e0b0xyg3[...,a_g3_g1]

###convert from date of first lamb born to average date born of all lambs
date_born_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be concieved anytime within joining cycle
date_born_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be concieved anytime within joining cycle
date_born_pa1e1b1nwzida0e0b0xyg2 = date_born1st_pa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be concieved anytime within joining cycle. e_index is to account for ewe cycles.
date_born_e1b1nwzida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_offs[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be concieved anytime within joining cycle
##wean age - used to calc wean date and also to calc m1 stuff, sire and dams have no active a0 slice therefore just take the first slice
age_wean_e0b0xyg0 = np.rollaxis(age_wean_a0e0b0xyg3[0, ...,a_g3_g0],0,age_wean_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple sclices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
age_wean_e0b0xyg1 = np.rollaxis(age_wean_a0e0b0xyg3[0, ...,a_g3_g1],0,age_wean_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple sclices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
##wean date (weaning input is counting from the date of the first lamb (not the date of the average lamb in the first cycle = date_born_self))
date_weaned_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + age_wean_e0b0xyg0
date_weaned_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + age_wean_e0b0xyg1
date_weaned_pa1e1b1nwzida0e0b0xyg2 = date_born1st_pa1e1b1nwzida0e0b0xyg2 + age_wean_a0e0b0xyg3
date_weaned_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + age_wean_a0e0b0xyg3
##age start open (not capped at weaning or before birth) used to calc m1 stuff
age_start_open_pa1e1b1nwzida0e0b0xyg0 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_ida0e0b0xyg0
age_start_open_pa1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_ida0e0b0xyg1
age_start_open_pa1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_e1b1nwzida0e0b0xyg3
age_start_open_pa1e1b1nwzida0e0b0xyg2 = date_start_pa1e1b1nwzida0e0b0xyg - date_born_pa1e1b1nwzida0e0b0xyg2
##age start
age_start_pa1e1b1nwzida0e0b0xyg0 = np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg0) - date_born_ida0e0b0xyg0 #use date weaned because the simulation for these animals is starting at weaning.
age_start_pa1e1b1nwzida0e0b0xyg1 = np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg1) - date_born_ida0e0b0xyg1 #use date weaned because the simulation for these animals is starting at weaning.
age_start_pa1e1b1nwzida0e0b0xyg3 = np.maximum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg3) - date_born_e1b1nwzida0e0b0xyg3 #use date weaned because the simulation for these animals is starting at weaning.
age_start_pa1e1b1nwzida0e0b0xyg2 = np.maximum(np.array([0]).astype('timedelta64[D]'),np.minimum(date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2) - date_born_pa1e1b1nwzida0e0b0xyg2) #use min and max so that the min age is 0 and the max age is the age at weaning
##Age_end: age at the beginning of the last day of the given period
##age end, minus one to allow the plus one in the next step when period date is less than weaning date (the minus one ensurs that when the p_date is less than weaning the animal gets 0 days in the given period)
age_end_pa1e1b1nwzida0e0b0xyg0 = np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg0 -1) - date_born_ida0e0b0xyg0 #use date weaned because the simulation for these animals is starting at weaning.
age_end_pa1e1b1nwzida0e0b0xyg1 = np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg1 -1) - date_born_ida0e0b0xyg1 #use date weaned because the simulation for these animals is starting at weaning.
age_end_pa1e1b1nwzida0e0b0xyg3 = np.maximum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg3 -1) - date_born_e1b1nwzida0e0b0xyg3 #use date weaned because the simulation for these animals is starting at weaning.
age_end_pa1e1b1nwzida0e0b0xyg2 = np.maximum(np.array([0]).astype('timedelta64[D]'),np.minimum(date_end_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2 -1) - date_born_pa1e1b1nwzida0e0b0xyg2)  #use min and max so that the min age is 0 and the max age is the age at weaning

##age mid period , plus one to get the age at the end of the last day of the period ie needed to get the full len of period.
age_pa1e1b1nwzida0e0b0xyg0 = (age_start_pa1e1b1nwzida0e0b0xyg0 + age_end_pa1e1b1nwzida0e0b0xyg0 +1) /2
age_pa1e1b1nwzida0e0b0xyg1 = (age_start_pa1e1b1nwzida0e0b0xyg1 + age_end_pa1e1b1nwzida0e0b0xyg1 +1) /2
age_pa1e1b1nwzida0e0b0xyg2 = (age_start_pa1e1b1nwzida0e0b0xyg2 + age_end_pa1e1b1nwzida0e0b0xyg2 +1) /2
age_pa1e1b1nwzida0e0b0xyg3 = (age_start_pa1e1b1nwzida0e0b0xyg3 + age_end_pa1e1b1nwzida0e0b0xyg3 +1) /2

##days in each period for each animal
days_period_pa1e1b1nwzida0e0b0xyg0 = age_end_pa1e1b1nwzida0e0b0xyg0 +1 - age_start_pa1e1b1nwzida0e0b0xyg0
days_period_pa1e1b1nwzida0e0b0xyg1 = age_end_pa1e1b1nwzida0e0b0xyg1 +1 - age_start_pa1e1b1nwzida0e0b0xyg1
days_period_pa1e1b1nwzida0e0b0xyg2 = age_end_pa1e1b1nwzida0e0b0xyg2 +1 - age_start_pa1e1b1nwzida0e0b0xyg2
days_period_pa1e1b1nwzida0e0b0xyg3 = age_end_pa1e1b1nwzida0e0b0xyg3 +1 - age_start_pa1e1b1nwzida0e0b0xyg3

##Age of foetus (start of period, end of period and mid period - days)
age_f_start_open_pa1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1
age_f_start_pa1e1b1nwzida0e0b0xyg1 = np.maximum(np.array([0]).astype('timedelta64[D]'), np.minimum(cp_dams[1, 0:1, :].astype('timedelta64[D]'), date_start_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1))
age_f_end_pa1e1b1nwzida0e0b0xyg1 = np.maximum(np.array([0]).astype('timedelta64[D]'), np.minimum(cp_dams[1, 0:1, :].astype('timedelta64[D]') - 1, date_end_pa1e1b1nwzida0e0b0xyg - date_mated_pa1e1b1nwzida0e0b0xyg1)) #cp -1 so that the period_days formula below is correct when p_date - date_mated is greater than cp (because plus 1)
# age_f_pa1e1b1nwzida0e0b0xyg1 = (age_f_start_pa1e1b1nwzida0e0b0xyg1 + age_f_end_pa1e1b1nwzida0e0b0xyg1 +1) / 2 ^not used anymore


############################
### Daily steps            #    ^this requires some things below. maybe just merge it in with the age date time section?
############################    or maybe make this a function
##add m1 axis
date_start_pa1e1b1nwzida0e0b0xygm1 = date_start_pa1e1b1nwzida0e0b0xyg[...,na] + index_m1
doy_pa1e1b1nwzida0e0b0xygm1= doy_pa1e1b1nwzida0e0b0xyg[...,na] + index_m1
##age open ie not capped at weaning
age_m1_pa1e1b1nwzida0e0b0xyg0m1 = (age_start_open_pa1e1b1nwzida0e0b0xyg0[..., na] + index_m1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
age_m1_pa1e1b1nwzida0e0b0xyg1m1 = (age_start_open_pa1e1b1nwzida0e0b0xyg1[..., na] + index_m1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
age_m1_pa1e1b1nwzida0e0b0xyg2m1 = (age_start_open_pa1e1b1nwzida0e0b0xyg2[..., na] + index_m1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
age_m1_pa1e1b1nwzida0e0b0xyg3m1 = (age_start_open_pa1e1b1nwzida0e0b0xyg3[..., na] + index_m1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
##make if age<=weaning make it nan - need nan so that the values are not included in the mean calculations which determine the average production for a given m1 period.
age_m1_pa1e1b1nwzida0e0b0xyg0m1[np.less_equal(age_m1_pa1e1b1nwzida0e0b0xyg0m1,age_wean_e0b0xyg0[..., na]/np.timedelta64(1, 'D'))] = np.nan
age_m1_pa1e1b1nwzida0e0b0xyg1m1[age_m1_pa1e1b1nwzida0e0b0xyg1m1<=(age_wean_e0b0xyg1[..., na]/np.timedelta64(1, 'D'))] = np.nan
age_m1_pa1e1b1nwzida0e0b0xyg3m1[age_m1_pa1e1b1nwzida0e0b0xyg3m1<=(age_wean_a0e0b0xyg3[..., na]/np.timedelta64(1, 'D'))] = np.nan
##if age is greater than weaning make nan and <= 0 make nan
age_m1_pa1e1b1nwzida0e0b0xyg2m1[age_m1_pa1e1b1nwzida0e0b0xyg2m1<=0] = np.nan
age_m1_pa1e1b1nwzida0e0b0xyg2m1[age_m1_pa1e1b1nwzida0e0b0xyg2m1>=(age_wean_a0e0b0xyg3[..., na]/np.timedelta64(1, 'D'))] = np.nan
##Age of foetus with minor axis (days)
age_f_m1_pa1e1b1nwzida0e0b0xyg1m1 = (age_f_start_open_pa1e1b1nwzida0e0b0xyg1[...,na] + index_m1) /np.timedelta64(1, 'D') #divide by 1 day to get int. need int so that i can put np.nan into array
age_f_m1_pa1e1b1nwzida0e0b0xyg1m1[age_f_m1_pa1e1b1nwzida0e0b0xyg1m1 <= 0] = np.nan
age_f_m1_pa1e1b1nwzida0e0b0xyg1m1[age_f_m1_pa1e1b1nwzida0e0b0xyg1m1 > cp_dams[1, 0, :, na]] = np.nan
##adjusted age of young (adjusted by intake factor - basically the factor of how age of young effect dam intake, the adjustment factor basically alters the age of the young to influnce intake.)
age_y_adj_pa1e1b1nwzida0e0b0xyg1m1 = age_m1_pa1e1b1nwzida0e0b0xyg2m1 + np.maximum(0, (date_start_pa1e1b1nwzida0e0b0xygm1 - date_weaned_pa1e1b1nwzida0e0b0xyg2[..., na]) /np.timedelta64(1, 'D')) * (ci_dams[21, ..., na] - 1) #minus 1 because the ci factor is applied to the age post weaning but using the open date means it has already been included once ie we want x + y *ci but using date open gives  x  + y + y*ci, x = age to weaning, y = age between period and weaning, therefore minus 1 x  + y + y*(ci-1)
age_y_adj_pa1e1b1nwzida0e0b0xyg1m1[age_y_adj_pa1e1b1nwzida0e0b0xyg1m1 <= 0] = np.nan
##Foetal age relative to parturition with minor axis
relage_f_pa1e1b1nwzida0e0b0xyg1m1 = age_f_m1_pa1e1b1nwzida0e0b0xyg1m1 / cp_dams[1, 0, :, na]
##Age of lamb relative to peak intake-with minor function
pimi_pa1e1b1nwzida0e0b0xyg1m1 = age_y_adj_pa1e1b1nwzida0e0b0xyg1m1 / ci_dams[8, ..., na]
##Age of lamb relative to peak lactation-with minor function
lmm_pa1e1b1nwzida0e0b0xyg1m1 = (age_m1_pa1e1b1nwzida0e0b0xyg2m1 + cl_dams[1, ..., na]) / cl_dams[2, ..., na]
##Chill index for lamb survival
chill_index_pa1e1b1nwzida0e0b0xygm1 = (481 + (11.7 + 3.1 * ws_pa1e1b1nwzida0e0b0xyg[..., na] ** 0.5) * (40 - temp_ave_pa1e1b1nwzida0e0b0xyg[..., na]) + 418 * (1-np.exp(-0.04 * rain_pa1e1b1nwzida0e0b0xygm1)))
##Proportion of SRW with age
srw_age_pa1e1b1nwzida0e0b0xyg0 = np.nanmean(np.exp(-cn_sire[1, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg0m1 / srw_xyg0[..., na] ** cn_sire[2, ..., na]), axis = -1)
srw_age_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(-cn_dams[1, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg1m1 / srw_xyg1[..., na] ** cn_dams[2, ..., na]), axis = -1)
srw_age_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(np.exp(-cn_yatf[1, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg2m1 / srw_xyg2[..., na] ** cn_yatf[2, ..., na]), axis = -1)
srw_age_pa1e1b1nwzida0e0b0xyg3 = np.nanmean(np.exp(-cn_offs[1, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg3m1 / srw_xyg3[..., na] ** cn_offs[2, ..., na]), axis = -1)
##age factor wool
af_wool_pa1e1b1nwzida0e0b0xyg0 = np.nanmean(cw_sire[5, ..., na] + (1 - cw_sire[5, ..., na])*(1-np.exp(-cw_sire[12, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg0m1)), axis = -1)
af_wool_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(cw_dams[5, ..., na] + (1 - cw_dams[5, ..., na])*(1-np.exp(-cw_dams[12, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg1m1)), axis = -1)
af_wool_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(cw_yatf[5, ..., na] + (1 - cw_yatf[5, ..., na])*(1-np.exp(-cw_yatf[12, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg2m1)), axis = -1)
af_wool_pa1e1b1nwzida0e0b0xyg3 = np.nanmean(cw_offs[5, ..., na] + (1 - cw_offs[5, ..., na])*(1-np.exp(-cw_offs[12, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg3m1)), axis = -1)
##Day length factor on efficiency
dlf_eff_pa1e1b1nwzida0e0b0xyg = np.average(lat_deg / 40 * np.sin(2 * np.pi * doy_pa1e1b1nwzida0e0b0xygm1 / 365), axis = -1)
##Pattern of maintenance with age
mr_age_pa1e1b1nwzida0e0b0xyg0 = np.nanmean(np.maximum(cm_sire[4, ..., na], np.exp(-cm_sire[3, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg0m1)), axis = -1)
mr_age_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.maximum(cm_dams[4, ..., na], np.exp(-cm_dams[3, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg1m1)), axis = -1)
mr_age_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(np.maximum(cm_offs[4, ..., na], np.exp(-cm_offs[3, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg2m1)), axis = -1)
mr_age_pa1e1b1nwzida0e0b0xyg3 = np.nanmean(np.maximum(cm_yatf[4, ..., na], np.exp(-cm_yatf[3, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg3m1)), axis = -1)
##Impact of rainfall on 'cold' intake increment
rain_intake_pa1e1b1nwzida0e0b0xyg0 = np.average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygm1 / ci_sire[18, ..., na]), axis = -1)
rain_intake_pa1e1b1nwzida0e0b0xyg1 = np.average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygm1 / ci_dams[18, ..., na]), axis = -1)
rain_intake_pa1e1b1nwzida0e0b0xyg2 = np.average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygm1 / ci_offs[18, ..., na]), axis = -1)
rain_intake_pa1e1b1nwzida0e0b0xyg3 = np.average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygm1 / ci_yatf[18, ..., na]), axis = -1)
##Proportion of peak intake due to time from birth
pi_age_y_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(cb1_dams[19, ..., na] * (pimi_pa1e1b1nwzida0e0b0xyg1m1) ** ci_dams[9, ..., na] * np.exp(ci_dams[9, ..., na] * (1 - pimi_pa1e1b1nwzida0e0b0xyg1m1)), axis = -1)
##Peak milk production pattern (time from birth)
mp_age_y = np.nanmean(cb1_dams[0, ..., na] * lmm_pa1e1b1nwzida0e0b0xyg1m1 ** cl_dams[3, ..., na] * np.exp(cl_dams[3, ..., na]* (1 - lmm_pa1e1b1nwzida0e0b0xyg1m1)), axis = -1)
##Suckling volume pattern
mp2_age_y = np.nanmean(nyatf_b1nwzida0e0b0xyg * cl_dams[6, ..., na] * ( cl_dams[12, ..., na] + cl_dams[13, ..., na] * np.exp(-cl_dams[14, ..., na] * age_m1_pa1e1b1nwzida0e0b0xyg2m1)), axis = -1)
##Pattern of conception efficiency (doy)
crg_doy_pa1e1b1nwzida0e0b0xyg1 = np.average(np.maximum(0,1 - cb1_dams[1, ..., na] * (1 - np.sin(2 * np.pi * (doy_pa1e1b1nwzida0e0b0xygm1 + 10) / 365) * np.sin(lat_rad) / -0.57)), axis = -1)
##Rumen development factor on PI - yatf
piyf_pa1e1b1nwzida0e0b0xyg2 = np.nanmean(1/(1 + np.exp(-ci_yatf[3, ..., na] * (age_m1_pa1e1b1nwzida0e0b0xyg2m1 - ci_yatf[4, ..., na]))), axis = -1)
##Foetal normal weight pattern (mid period)
nwf_age_f_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(cp_dams[2, ..., na] * (1 - np.exp(cp_dams[3, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1m1)))), axis = -1)
##Conceptus weight pattern (mid period)
guw_age_f_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(cp_dams[6, ..., na] * (1 - np.exp(cp_dams[7, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1m1)))), axis = -1)
##Conceptus energy pattern (end of period)
ce_age_f_pa1e1b1nwzida0e0b0xyg1 = np.nanmean(np.exp(cp_dams[9, ..., na] * (1 - np.exp(cp_dams[10, ..., na] * (1 - relage_f_pa1e1b1nwzida0e0b0xyg1m1)))), axis = -1)

##genotype calc that requires af_wool. ME for minimum wool growth (with no intake, relsize = 1)
mew_min_pa1e1b1nwzida0e0b0xyg0 =cw_sire[14, ...] * sfw_a0e0b0xyg0[0, ...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg0 * dlf_wool_pa1e1b1nwzida0e0b0xyg0 * cw_sire[1, ...] / kw_yg0
mew_min_pa1e1b1nwzida0e0b0xyg1 =cw_dams[14, ...] * sfw_a0e0b0xyg1[0, ...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg1 * dlf_wool_pa1e1b1nwzida0e0b0xyg1 * cw_dams[1, ...] / kw_yg1
mew_min_pa1e1b1nwzida0e0b0xyg2 =cw_yatf[14, ...] * sfw_pa1e1b1nwzida0e0b0xyg2[0, ...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg2 * dlf_wool_pa1e1b1nwzida0e0b0xyg2 * cw_yatf[1, ...] / kw_yg2
mew_min_pa1e1b1nwzida0e0b0xyg3 =cw_offs[14, ...] * sfw_da0e0b0xyg3[0, ...] / 365 * af_wool_pa1e1b1nwzida0e0b0xyg3 * dlf_wool_pa1e1b1nwzida0e0b0xyg3 * cw_offs[1, ...] / kw_yg3


#######################
##Age, date, timing 2 #
#######################
##Days per period in the simulation - foetus
days_period_f_pa1e1b1nwzida0e0b0xyg1 = age_f_end_pa1e1b1nwzida0e0b0xyg1 +1 - age_f_start_pa1e1b1nwzida0e0b0xyg1

##proportion of days gestating during the period
gest_propn_pa1e1b1nwzida0e0b0xyg1 = days_period_f_pa1e1b1nwzida0e0b0xyg1 / days_period_pa1e1b1nwzida0e0b0xyg1

##proportion of days of lactating during the period
lact_propn_pa1e1b1nwzida0e0b0xyg1 = days_period_pa1e1b1nwzida0e0b0xyg2 / days_period_pa1e1b1nwzida0e0b0xyg1

##Is nutrition effecting lactation
lact_nut_effect_pa1e1b1nwzida0e0b0xyg1 = (age_pa1e1b1nwzida0e0b0xyg2 /np.timedelta64(1, 'D') > (cl_dams[16, ...] * cl_dams[2, ...]))

##Average daily CFW
d_cfw_ave_pa1e1b1nwzida0e0b0xyg0 = cw_sire[3, ...] * sfw_a0e0b0xyg0 * af_wool_pa1e1b1nwzida0e0b0xyg0 * days_period_pa1e1b1nwzida0e0b0xyg0 / 365
d_cfw_ave_pa1e1b1nwzida0e0b0xyg1 = cw_dams[3, ...] * sfw_a0e0b0xyg1 * af_wool_pa1e1b1nwzida0e0b0xyg1 * days_period_pa1e1b1nwzida0e0b0xyg1 / 365
d_cfw_ave_pa1e1b1nwzida0e0b0xyg2 = cw_yatf[3, ...] * sfw_pa1e1b1nwzida0e0b0xyg2 * af_wool_pa1e1b1nwzida0e0b0xyg2 * days_period_pa1e1b1nwzida0e0b0xyg2 / 365
d_cfw_ave_pa1e1b1nwzida0e0b0xyg3 = cw_offs[3, ...] * sfw_da0e0b0xyg3 * af_wool_pa1e1b1nwzida0e0b0xyg3 * days_period_pa1e1b1nwzida0e0b0xyg3 / 365

##Expected relative size
relsize_exp_a1e1b1nwzida0e0b0xyg0  = (srw_xyg0 - (srw_xyg0 - w_b_std_b0xyg0) * np.exp(cn_sire[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg0 /np.timedelta64(1, 'D')) / (srw_xyg0**cn_sire[2, ...]))) / srw_xyg0
relsize_exp_a1e1b1nwzida0e0b0xyg1  = (srw_xyg1 - (srw_xyg1 - w_b_std_b0xyg1) * np.exp(cn_dams[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg1/np.timedelta64(1, 'D')) / (srw_xyg1**cn_dams[2, ...]))) / srw_xyg1
relsize_exp_a1e1b1nwzida0e0b0xyg3  = (srw_xyg3 - (srw_xyg3 - w_b_std_b0xyg3) * np.exp(cn_offs[1, ...] * (agedam_lamb1st_a1e1b1nwzida0e0b0xyg3/np.timedelta64(1, 'D')) / (srw_xyg3**cn_offs[2, ...]))) / srw_xyg3

##adjust ce sim param (^ ce12 &13 should be scaled by relsize (similar to ce15)) - couldnt find a good way to do this because get error asigning multiple slices to na. ie if i = 2 (^instead of setting ce with relsize adjustment then akjusting birth weight could just adjust birthweight directly with relsize factor - to avoid doing this code below)
shape = (ce_sire.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg0.shape #get shape of the new ce array
ce_ca1e1b1nwzida0e0b0xyg0 = np.zeros(shape) #make a new array - same a ce with an active i axis
ce_ca1e1b1nwzida0e0b0xyg0[...] = np.expand_dims(ce_sire, axis = tuple(range(-relsize_exp_a1e1b1nwzida0e0b0xyg0.ndim,-ce_sire.ndim+1))) #assign the origional ce value - note the i axis has been added
ce_ca1e1b1nwzida0e0b0xyg0[15, ...] = 1 - cp_sire[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg0) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
ce_sire = ce_ca1e1b1nwzida0e0b0xyg0 #rename to keep consistent

shape = (ce_dams.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg1.shape #get shape of the new ce array
ce_ca1e1b1nwzida0e0b0xyg1 = np.zeros(shape) #make a new array - same a ce with an active i axis
ce_ca1e1b1nwzida0e0b0xyg1[...] = np.expand_dims(ce_dams, axis = tuple(range(-relsize_exp_a1e1b1nwzida0e0b0xyg1.ndim,-ce_dams.ndim+1))) #assign the origional ce value - note the i axis has been added
ce_ca1e1b1nwzida0e0b0xyg1[15, ...] = 1 - cp_dams[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg1) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
ce_dams = ce_ca1e1b1nwzida0e0b0xyg1 #rename to keep consistent

shape = (ce_offs.shape[0],) + relsize_exp_a1e1b1nwzida0e0b0xyg3.shape #get shape of the new ce array
ce_ca1e1b1nwzida0e0b0xyg3 = np.zeros(shape) #make a new array - same a ce with an active i axis
ce_ca1e1b1nwzida0e0b0xyg3[...] = np.expand_dims(ce_offs, axis = tuple(range(-relsize_exp_a1e1b1nwzida0e0b0xyg3.ndim,-ce_offs.ndim+1))) #assign the origional ce value - note the i axis has been added
ce_ca1e1b1nwzida0e0b0xyg3[15, ...] = 1 - cp_offs[4, ...] * (1 - relsize_exp_a1e1b1nwzida0e0b0xyg3) #alter ce15 param, relsize has active i axis hence this is not a  simple assignment.
ce_offs = ce_ca1e1b1nwzida0e0b0xyg3 #rename to keep consistent

##birth weight expected - includes relsize factoer
w_b_exp_a1e1b1nwzida0e0b0xyg0 = w_b_std_b0xyg0 * np.sum(ce_sire[15, ...] * agedam_propn_da0e0b0xyg0, axis = uinp.parameters['i_d_pos'], keepdims = True)
w_b_exp_a1e1b1nwzida0e0b0xyg1 = w_b_std_b0xyg1 * np.sum(ce_dams[15, ...] * agedam_propn_da0e0b0xyg1, axis = uinp.parameters['i_d_pos'], keepdims = True)
w_b_exp_a1e1b1nwzida0e0b0xyg3 = w_b_std_b0xyg3 * ce_offs[15, ...]

##Normal weight max (if animal is well fed)
nw_max_pa1e1b1nwzida0e0b0xyg0 = srw_xyg0 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg0) + w_b_exp_a1e1b1nwzida0e0b0xyg0 * srw_age_pa1e1b1nwzida0e0b0xyg0
nw_max_pa1e1b1nwzida0e0b0xyg1 = srw_xyg1 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg1) + w_b_exp_a1e1b1nwzida0e0b0xyg1 * srw_age_pa1e1b1nwzida0e0b0xyg1
nw_max_pa1e1b1nwzida0e0b0xyg3 = srw_xyg3 * (1 - srw_age_pa1e1b1nwzida0e0b0xyg3) + w_b_exp_a1e1b1nwzida0e0b0xyg3 * srw_age_pa1e1b1nwzida0e0b0xyg3

##Change in normal weight max - the last period will be 0 by default but this is okay because nw hits an asymptope so change in will be 0 in the last period.
d_nw_max_pa1e1b1nwzida0e0b0xyg0 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg0)
d_nw_max_pa1e1b1nwzida0e0b0xyg0[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg0[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg0[0:-1, ...]) / (days_period_pa1e1b1nwzida0e0b0xyg0/np.timedelta64(1, 'D'))[0:-1, ...]
d_nw_max_pa1e1b1nwzida0e0b0xyg1 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg1)
d_nw_max_pa1e1b1nwzida0e0b0xyg1[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg1[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg1[0:-1, ...]) / (days_period_pa1e1b1nwzida0e0b0xyg1/np.timedelta64(1, 'D'))[0:-1, ...]
d_nw_max_pa1e1b1nwzida0e0b0xyg3 = np.zeros_like(nw_max_pa1e1b1nwzida0e0b0xyg3)
d_nw_max_pa1e1b1nwzida0e0b0xyg3[0:-1, ...] = (nw_max_pa1e1b1nwzida0e0b0xyg3[1:, ...] - nw_max_pa1e1b1nwzida0e0b0xyg3[0:-1, ...]) / (days_period_pa1e1b1nwzida0e0b0xyg3/np.timedelta64(1, 'D'))[0:-1, ...]


#########################
# management calc       #
#########################
##date scan
date_scan_pa1e1b1nwzida0e0b0xyg1 = date_joined2_pa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') + pinp.sheep['i_scan_day'][scan_pa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]')
##Expected stocking density
density_pa1e1b1nwzida0e0b0xyg0 = density_pa1e1b1nwzida0e0b0xyg
density_pa1e1b1nwzida0e0b0xyg1 = density_pa1e1b1nwzida0e0b0xyg * density_nwzida0e0b0xyg1
density_pa1e1b1nwzida0e0b0xyg2 = density_pa1e1b1nwzida0e0b0xyg * density_nwzida0e0b0xyg1  #yes this is meant to be the same as dams
density_pa1e1b1nwzida0e0b0xyg3 = density_pa1e1b1nwzida0e0b0xyg * density_nwzida0e0b0xyg3
##numbers
###Distribution of initial numbers across the a1 axis	
initial_a1e1b1nwzida0e0b0xyg = f_reshape_expand(pinp.sheep['i_initial_a1'], pinp.sheep['i_a1_pos'], condition = pinp.sheep['i_mask_a'], axis = pinp.sheep['i_a1_pos']) 
###Distribution of initial numbers across the b1 axis	
initial_b1nwzida0e0b0xyg = f_reshape_expand(uinp.structure['i_initial_b1'], uinp.parameters['i_b1_pos']) 
###Distribution of initial numbers across the y axis	
initial_yg = f_reshape_expand(uinp.parameters['i_initial_y'], pinp.parameters['i_y_pos'], condition = uinp.parameters['i_mask_y'], axis = pinp.parameters['i_y_pos']) 
###Distribution of initial numbers across the e1 axis	
initial_e1 = np.zeros(len_e1)
initial_e1[0] = 1 #create this to look like [1,0,…] with enough zeros to be the length of the e1 axis
initial_e1b1nwzida0e0b0xyg = f_reshape_expand(initial_e1, pinp.sheep['i_e1_pos']) 



#########################
# period is ...         #
#########################
period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is_between', date_prejoin_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_scan_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
date_born2_pa1e1b1nwzida0e0b0xyg2 = date_born1st2_pa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be concieved anytime within joining cycle. e_index is to account for ewe cycles.
period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is_between', date_scan_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_born2_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg) #use date born that increments at joining
period_between_birthwean_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is_between', date_born_pa1e1b1nwzida0e0b0xyg2, date_start_pa1e1b1nwzida0e0b0xyg, date_weaned_pa1e1b1nwzida0e0b0xyg2, date_end_pa1e1b1nwzida0e0b0xyg)
date_weaned2_pa1e1b1nwzida0e0b0xyg2 = date_born1st2_pa1e1b1nwzida0e0b0xyg2 + age_wean_a0e0b0xyg3 #this needs to increment at prejoining for period between weaning and prejoining, so that it is false after prejoining and before weaning.
period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is_between', date_weaned2_pa1e1b1nwzida0e0b0xyg2, date_start_pa1e1b1nwzida0e0b0xyg, date_prejoin_next_pa1e1b1nwzida0e0b0xyg1, date_end_pa1e1b1nwzida0e0b0xyg)
period_is_birth_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is', date_born_pa1e1b1nwzida0e0b0xyg2, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivelant of date lambed g1
prev_period_is_birth_pa1e1b1nwzida0e0b0xyg1 = np.roll(period_is_birth_pa1e1b1nwzida0e0b0xyg1,1,axis=uinp.structure['i_p_pos'])
period_is_mating_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is', date_mated_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivelant of date lambed g1
period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is_between', date_born_pa1e1b1nwzida0e0b0xyg2, date_start_pa1e1b1nwzida0e0b0xyg, date_born_pa1e1b1nwzida0e0b0xyg2+np.array([(6*7)]).astype('timedelta64[D]'), date_end_pa1e1b1nwzida0e0b0xyg) #This is within 6 weeks of the Birth period


period_is_prejoin_pa1e1b1nwzida0e0b0xyg1 = f_period_is_('period_is', date_prejoin_pa1e1b1nwzida0e0b0xyg1, date_start_pa1e1b1nwzida0e0b0xyg, date_end_p = date_end_pa1e1b1nwzida0e0b0xyg) #g2 date born is the equivelant of date lambed g1
period_is_start_fvp1_pa1e1b1nwzida0e0b0xyg1 =  np.logical_and(fvp_type_pa1e1b1nwzida0e0b0xyg1 == 1, np.roll(fvp_type_pa1e1b1nwzida0e0b0xyg1,1, axis=0)!=1)  #is this sim period the first period in type 1 of fvp's 

=
############################
### feed supply calcs      # ^apparently need to add something about break of season..? and need to add e variation
############################
##1)	Compile the standard pattern from the inputs
###sire
t_feedsupply_pj0zida0e0b0xyg0 = np.rollaxis(np.rollaxis(feedoptions_r1pj0[a_r_zida0e0b0xyg0],-1,0),-1,0) #had to rollaxis twice once for p and once for j0 (couldn't find a way to do both at the same time)
t_feedsupply_pj0wzida0e0b0xyg0 = np.expand_dims(t_feedsupply_pj0zida0e0b0xyg0, axis = tuple(range(uinp.structure['i_n_pos']+1,pinp.sheep['i_z_pos']))) #add w axis
t_feedsupply_pa1e1b1j0wzida0e0b0xyg0 = np.expand_dims(t_feedsupply_pj0wzida0e0b0xyg0, axis = tuple(range(uinp.structure['i_p_pos']+1,uinp.structure['i_n_pos']))) #add a1,e1,b1 axis. Note n and j are the same thing (as far a position goes)
###dams
t_feedsupply_pj0zida0e0b0xyg1 = np.rollaxis(np.rollaxis(feedoptions_r1pj0[a_r_zida0e0b0xyg1],-1,0),-1,0) #had to rollaxis twice once for p and once for j0 (couldn't find a way to do both at the same time)
t_feedsupply_pj0wzida0e0b0xyg1 = np.expand_dims(t_feedsupply_pj0zida0e0b0xyg1, axis = tuple(range(uinp.structure['i_n_pos']+1,pinp.sheep['i_z_pos']))) #add w axis
t_feedsupply_pa1e1b1j0wzida0e0b0xyg1 = np.expand_dims(t_feedsupply_pj0wzida0e0b0xyg1, axis = tuple(range(uinp.structure['i_p_pos']+1,uinp.structure['i_n_pos']))) #add  a1,e1,b1 axis. Note n and j are the same thing (as far a position goes)
###offs
t_feedsupply_pj0zida0e0b0xyg3 = np.rollaxis(np.rollaxis(feedoptions_r1pj0[a_r_zida0e0b0xyg3],-1,0),-1,0) #had to rollaxis twice once for p and once for j0 (couldn't find a way to do both at the same time)
t_feedsupply_pj0wzida0e0b0xyg3 = np.expand_dims(t_feedsupply_pj0zida0e0b0xyg3, axis = tuple(range(uinp.structure['i_n_pos']+1,pinp.sheep['i_z_pos']))) #add w axis
t_feedsupply_pa1e1b1j0wzida0e0b0xyg3 = np.expand_dims(t_feedsupply_pj0wzida0e0b0xyg3, axis = tuple(range(uinp.structure['i_p_pos']+1,uinp.structure['i_n_pos']))) #add a1,e1,b1 axis. Note n and j are the same thing (as far a position goes)

##2) calculate the feedsupply variaiton for each sheep class
t_fs_ageweaned_pk0k1k2j0wzida0e0b0xyg1 = np.rollaxis(feedoptions_var_r2p[a_r2_k0e1b1nwzida0e0b0xyg1],-1,0)
t_fs_cycle_pk0k1k2j0wzida0e0b0xyg1 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_k1b1nwzida0e0b0xyg1],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,pinp.sheep['i_e1_pos']))) #add k0
t_fs_lsln_pk0k1k2j0wzida0e0b0xyg1 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_k2nwzida0e0b0xyg1],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,uinp.parameters['i_b1_pos']))) #add k0,k1
t_fs_agedam_pa1e1b1j0wzik3a0e0b0xyg3 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_ik3a0e0b0xyg3],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,pinp.sheep['i_i_pos']))) #add from i to p
t_fs_ageweaned_pa1e1b1j0wzidk0e0b0xyg3 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_idk0e0b0xyg3],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,pinp.sheep['i_i_pos']))) #add from i to p
t_fs_btrt_a1e1b1j0wzida0e0k4xyg3 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_ida0e0k4xyg3],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,pinp.sheep['i_i_pos']))) #add from i to p
t_fs_gender_pa1e1b1j0wzida0e0b0k5yg3 = np.expand_dims(np.rollaxis(feedoptions_var_r2p[a_r2_ida0e0b0k5yg3],-1,0), axis = tuple(range(uinp.structure['i_p_pos']+1,pinp.sheep['i_i_pos']))) #add from i to p

##3)Based on the animal management selected (scan, gbal and wean) and whether the animals are differentially managed in this trial
###a) weaning age variation
a_k0_pa1e1b1nwzida0e0b0xyg1 = period_between_weanprejoin_pa1e1b1nwzida0e0b0xyg1 * pinp.sheep['i_dam_wean_diffman'] * f_reshape_expand(np.arange(len_a1)+1, pinp.sheep['i_a1_pos']) #len_a+1 because that is the association between k0 and a1
t_fs_ageweaned_pa1e1b1j0wzida0e0b0xyg1 = np.take_along_axis(t_fs_ageweaned_pk0k1k2j0wzida0e0b0xyg1, a_k0_pa1e1b1nwzida0e0b0xyg1, 1) 

###b)b.	Dams Cluster k1 – oestrus cycle (e1): The association required is
#^Have decided to drop this out of version 1. Will require multiple nutrition patterns in order to test value of scanning for foetal age

###c)Dams Cluster k2 – BTRT (b1)
####have to create a_t array so that it is maximum size of the arrays that are used it mask it. Then use broadcasting function to allow a smaller mask to be applied.
shape = np.maximum.reduce([period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1.shape,period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1.shape,period_between_birthwean_pa1e1b1nwzida0e0b0xyg1.shape]) #create shape which has the max size
a_t_pa1e1b1nwzida0e0b0xyg1 = np.zeros(shape)
period_between_joinscan_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_prejoinscan_pa1e1b1nwzida0e0b0xyg1)[1] #mask must be manually broadcasted then applied - for some reason numpy doesnt automatically broadcast them.
period_between_scanbirth_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_scanbirth_pa1e1b1nwzida0e0b0xyg1)[1]
period_between_birthwean_mask = np.broadcast_arrays(a_t_pa1e1b1nwzida0e0b0xyg1, period_between_birthwean_pa1e1b1nwzida0e0b0xyg1)[1]
####order matters because post wean does not have a cap ie it is over written by others
a_t_pa1e1b1nwzida0e0b0xyg1[...] = 3 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
a_t_pa1e1b1nwzida0e0b0xyg1[period_between_joinscan_mask] = 0 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
a_t_pa1e1b1nwzida0e0b0xyg1[period_between_scanbirth_mask] = 1 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
a_t_pa1e1b1nwzida0e0b0xyg1[period_between_birthwean_mask] = 2 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean

# a_t_pa1e1b1nwzida0e0b0xyg1[period_is_postscan_pa1e1b1nwzida0e0b0xyg1] = 1 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
# a_t_pa1e1b1nwzida0e0b0xyg1[period_is_postlactation_pa1e1b1nwzida0e0b0xyg1] = 2 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
# a_t_pa1e1b1nwzida0e0b0xyg1[period_is_postwean_pa1e1b1nwzida0e0b0xyg1] = 3 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean
# a_t_pa1e1b1nwzida0e0b0xyg1[period_is_prescan_pa1e1b1nwzida0e0b0xyg1] = 0 #t = 0 is prescan, 1 is postscan, 2 is lactation, 3 is post wean


####dams management in each period
scan_pa1e1b1nwzida0e0b0xyg1 = (scan_pa1e1b1nwzida0e0b0xyg1) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 1) * pinp.sheep['i_dam_lsln_diffman_t'][1]
gbal_pa1e1b1nwzida0e0b0xyg1 = (gbal_pa1e1b1nwzida0e0b0xyg1 -1 ) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 2) * pinp.sheep['i_dam_lsln_diffman_t'][2] + 1  #minus 1 then plus 1 ensures that thewean option before lactation is 1 
wean_pa1e1b1nwzida0e0b0xyg1 = (wean_pa1e1b1nwzida0e0b0xyg1 -1 ) * (a_t_pa1e1b1nwzida0e0b0xyg1 >= 3) * pinp.sheep['i_dam_lsln_diffman_t'][3] + 1  #minus 1 then plus 1 ensures that thewean option before weaning is 1 
####a_k2_vlsb1 states the feed variation slice for defferent management. In this step we slice a_k2_vlsb1 for the selected management in each period.
a_k2_pa1e1b1nwzida0e0b0xyg1 = np.rollaxis(a_k2_vlsb1[wean_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...], gbal_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...], scan_pa1e1b1nwzida0e0b0xyg1[:,:,:,0,...], ...],-1,3) #remove the singlton b1 axis from the association arrays because a populated b1 axis comes from a_k2_vlsb1
####select feed variation pattern
t_fs_lsln_pa1e1b1j0wzida0e0b0xyg1 = np.take_along_axis(t_fs_lsln_pk0k1k2j0wzida0e0b0xyg1, a_k2_pa1e1b1nwzida0e0b0xyg1, uinp.parameters['i_b1_pos']) 

###d) ^come back to offs
# t_fs_agedam_pj0zida0e0b0xg3 = t_fs_agedam_pj0zik3k0k4k5g3
# t_fs_ageweaned_pj0zida0e0b0xg3 = t_fs_ageweaned_pj0zik3k0k4k5g3
# t_fs_btrt_pj0zida0e0b0xg3 = t_fs_btrt_pj0zik3k0k4k5g3
# t_fs_btrt_pj0zida0e0b0xg3 = t_fs_btrt_pj0zik3k0k4k5g3

##4) add variation to std pattern
t_feedsupply_pa1e1b1j0wzida0e0b0xyg1 = (t_feedsupply_pa1e1b1j0wzida0e0b0xyg1 + t_fs_ageweaned_pa1e1b1j0wzida0e0b0xyg1 + t_fs_lsln_pa1e1b1j0wzida0e0b0xyg1) #cant use += for some reason



##6)Convert the ‘j0’ axis to an ‘n’ axis
###a- create a ‘j0’ by ‘n’ array that is the multipliers that weight each ‘j0’ for that level of ‘n’
nut_mult_g0_j0n = np.empty((3,uinp.structure['i_nut_spread_n0'].shape[0]))
nut_mult_g0_j0n[0, ...] = 1 - np.abs(uinp.structure['i_nut_spread_n0'])
nut_mult_g0_j0n[1, ...] = 1-(1 - np.abs(np.minimum(0, uinp.structure['i_nut_spread_n0'])))
nut_mult_g0_j0n[2, ...] = 1-(1 - np.maximum(0, uinp.structure['i_nut_spread_n0']))
nut_mult_g1_j0n = np.empty((3,uinp.structure['i_nut_spread_n1'].shape[0]))
nut_mult_g1_j0n[0, ...] = 1 - np.abs(uinp.structure['i_nut_spread_n1'])
nut_mult_g1_j0n[1, ...] = 1-(1 - np.abs(np.minimum(0, uinp.structure['i_nut_spread_n1'])))
nut_mult_g1_j0n[2, ...] = 1-(1 - np.maximum(0, uinp.structure['i_nut_spread_n1']))
nut_mult_g3_j0n = np.empty((3,uinp.structure['i_nut_spread_n3'].shape[0]))
nut_mult_g3_j0n[0, ...] = 1 - np.abs(uinp.structure['i_nut_spread_n3'])
nut_mult_g3_j0n[1, ...] = 1-(1 - np.abs(np.minimum(0, uinp.structure['i_nut_spread_n3'])))
nut_mult_g3_j0n[2, ...] = 1-(1 - np.maximum(0, uinp.structure['i_nut_spread_n3']))

###b- create add array if there is a confinement or feedlot pattern (i_nut_spread_n >=3)
nut_spread_g0_n = uinp.structure['i_nut_spread_n0']
nut_add_g0_n = np.zeros_like(nut_spread_g0_n)
nut_add_g0_n[nut_spread_g0_n >=3] = nut_spread_g0_n[nut_spread_g0_n >=3]
nut_mult_g0_j0n[:,nut_spread_g0_n >=3] = 0 #if nut_add exists then nut_mult=0

nut_spread_g1_n = uinp.structure['i_nut_spread_n1']
nut_add_g1_n = np.zeros_like(nut_spread_g1_n)
nut_add_g1_n[nut_spread_g1_n >=3] = nut_spread_g1_n[nut_spread_g1_n >=3]
nut_mult_g1_j0n[:,nut_spread_g1_n >=3] = 0 #if nut_add exists then nut_mult=0

nut_spread_g3_n = uinp.structure['i_nut_spread_n3']
nut_add_g3_n = np.zeros_like(nut_spread_g3_n)
nut_add_g3_n[nut_spread_g3_n >=3] = nut_spread_g3_n[nut_spread_g3_n >=3]
nut_mult_g3_j0n[:,nut_spread_g3_n >=3] = 0 #if nut_add exists then nut_mult=0

###c - feedsupply_std with n axis (instead of j axis).
nut_mult_g0_pk0k1k2j0nwzida0e0b0xyg = np.expand_dims(nut_mult_g0_j0n[na,na,na,na,...], axis = tuple(range(uinp.structure['i_n_pos']+1,0))) #expand axis to line up with feedsupply, add axis from g to n and j0 to p
nut_add_g0_pk0k1k2nwzida0e0b0xyg = np.expand_dims(nut_add_g0_n, axis = (tuple(range(uinp.structure['i_p_pos'],uinp.structure['i_n_pos'])) + tuple(range(uinp.structure['i_n_pos']+1,0)))) #add axis from p to n and n to g
t_feedsupply_pa1e1b1j0nwzida0e0b0xyg0 = np.expand_dims(t_feedsupply_pa1e1b1j0wzida0e0b0xyg0, axis = uinp.structure['i_n_pos']) #add n axis
feedsupply_std_pa1e1b1nwzida0e0b0xyg0 = np.sum(t_feedsupply_pa1e1b1j0nwzida0e0b0xyg0 * nut_mult_g0_pk0k1k2j0nwzida0e0b0xyg, axis = uinp.structure['i_n_pos']-1 ) + nut_add_g0_pk0k1k2nwzida0e0b0xyg #minus 1 because n axis was added therefore shifting j0 position (it was origionally in the same place). Sum across j0 axis and leave just the n axis

nut_mult_g1_pk0k1k2j0nwzida0e0b0xyg = np.expand_dims(nut_mult_g1_j0n[na,na,na,na,...], axis = tuple(range(uinp.structure['i_n_pos']+1,0))) #expand axis to line up with feedsupply, add axis from g to n and j0 to p
nut_add_g1_pk0k1k2nwzida0e0b0xyg = np.expand_dims(nut_add_g1_n, axis = (tuple(range(uinp.structure['i_p_pos'],uinp.structure['i_n_pos'])) + tuple(range(uinp.structure['i_n_pos']+1,0)))) #add axis from p to n and n to g
t_feedsupply_pa1e1b1j0nwzida0e0b0xyg1 = np.expand_dims(t_feedsupply_pa1e1b1j0wzida0e0b0xyg1, axis = uinp.structure['i_n_pos']) #add n axis
feedsupply_std_pa1e1b1nwzida0e0b0xyg1 = np.sum(t_feedsupply_pa1e1b1j0nwzida0e0b0xyg1 * nut_mult_g1_pk0k1k2j0nwzida0e0b0xyg, axis = uinp.structure['i_n_pos']-1 ) + nut_add_g1_pk0k1k2nwzida0e0b0xyg #minus 1 because n axis was added therefore shifting j0 position (it was origionally in the same place). Sum across j0 axis and leave just the n axis

nut_mult_g3_pk0k1k2j0nwzida0e0b0xyg = np.expand_dims(nut_mult_g3_j0n[na,na,na,na,...], axis = tuple(range(uinp.structure['i_n_pos']+1,0))) #expand axis to line up with feedsupply, add axis from g to n and j0 to p
nut_add_g3_pk0k1k2nwzida0e0b0xyg = np.expand_dims(nut_add_g3_n, axis = (tuple(range(uinp.structure['i_p_pos'],uinp.structure['i_n_pos'])) + tuple(range(uinp.structure['i_n_pos']+1,0)))) #add axis from p to n and n to g
t_feedsupply_pa1e1b1j0nwzida0e0b0xyg3 = np.expand_dims(t_feedsupply_pa1e1b1j0wzida0e0b0xyg3, axis = uinp.structure['i_n_pos']) #add n axis
feedsupply_std_pa1e1b1nwzida0e0b0xyg3 = np.sum(t_feedsupply_pa1e1b1j0nwzida0e0b0xyg3 * nut_mult_g3_pk0k1k2j0nwzida0e0b0xyg, axis = uinp.structure['i_n_pos']-1 ) + nut_add_g3_pk0k1k2nwzida0e0b0xyg #minus 1 because n axis was added therefore shifting j0 position (it was origionally in the same place). Sum across j0 axis and leave just the n axis

##7)Ensure that no feed supplies are outside the range 0 to 4
feedsupply_std_pa1e1b1nwzida0e0b0xyg0 = np.maximum(0, feedsupply_std_pa1e1b1nwzida0e0b0xyg0)
feedsupply_std_pa1e1b1nwzida0e0b0xyg0 = np.minimum(4, feedsupply_std_pa1e1b1nwzida0e0b0xyg0)
feedsupply_std_pa1e1b1nwzida0e0b0xyg1 = np.maximum(0, feedsupply_std_pa1e1b1nwzida0e0b0xyg1)
feedsupply_std_pa1e1b1nwzida0e0b0xyg1 = np.minimum(4, feedsupply_std_pa1e1b1nwzida0e0b0xyg1)
feedsupply_std_pa1e1b1nwzida0e0b0xyg3 = np.maximum(0, feedsupply_std_pa1e1b1nwzida0e0b0xyg3)
feedsupply_std_pa1e1b1nwzida0e0b0xyg3 = np.minimum(4, feedsupply_std_pa1e1b1nwzida0e0b0xyg3)








  #BTRT for b1 - code below not curently used but may be a little helpful later on.

    # def f_btrt1(dstwtr,lss,lstw,lstr): #^this function is inflexible ie if you want to add qradruplets
    #     '''
    #     Parameters
    #     ----------
    #     dstwtr : np array
    #         proportion of dry, singles, twin and triplets.
    #     lss : np array
    #         single survival.
    #     lstw : np array
    #         twin survival.
    #     lstr : np array
    #         triplet survival.

    #     Returns
    #     -------
    #     btrt_b1nwzida0e0b0xyg : np array
    #         probability of ewe with lambs in each btrt category (eg 11, 22, 21 ...).

    #     '''

    #     ##lamb numbers is the number of lambs in each b0 category, based on survival of s, tw and tr after birth.
    #     lamb_numbers_b1yg = np.zeros((11,lss.shape[-2],lss.shape[-1])) #^where can i reference 11? would be good to have a b1 slice count somewhere.
    #     lamb_numbers_b1yg[0,...] = lss
    #     lamb_numbers_b1yg[1,...] = lstw**2
    #     lamb_numbers_b1yg[2,...] = lstr**3
    #     lamb_numbers_b1yg[3,...] = 2 * lstw * (1 - lstw)  #the 2 is because it could be either lamb 1 that dies or lamb 2 that dies
    #     lamb_numbers_b1yg[4,...] = (3* lstr**2 * (1 - lstr))  # 3x because it could be either lamb 1, 2 or 3 that dies
    #     lamb_numbers_b1yg[5,...] = 3* lstr * (1 - lstr)**2  #the 3x because it could be either lamb 1, 2 or 3 that survives
    #     ##mul lamb numbers array with lambing percentage to get overall btrt
    #     btrt_b1yg = lamb_numbers_b1yg * dstwtr_l0yg[uinp.structure['a_nfoet_b1'] ]
    #     ##add singleton x axis
    #     btrt_b1nwzida0e0b0xyg = np.expand_dims(btrt_b1yg, axis = tuple(range((uinp.parameters['i_cl1_pos'] + 1), -2))) #note i_cl1_pos refers to b1 position
    #     return btrt_b1nwzida0e0b0xyg

    # ##calc BTRT b1 - 11, 22, 33, 21, 32, 31, 10, 20, 30, 00, nm - probability of ewes that fall in to each of the btrt categories
    # btrt_b1nwzida0e0b0xy0 = f_btrt1(dstwtr_l0yg0,lss_std_yg0,lstw_std_yg0,lstr_std_yg0)
    # btrt_b1nwzida0e0b0xy1 = f_btrt1(dstwtr_l0yg1,lss_std_yg1,lstw_std_yg1,lstr_std_yg1)
    # btrt_b1nwzida0e0b0xy2 = f_btrt1(dstwtr_l0yg2,lss_std_yg2,lstw_std_yg2,lstr_std_yg2)
    # btrt_b1nwzida0e0b0xy3 = f_btrt1(dstwtr_l0yg3,lss_std_yg3,lstw_std_yg3,lstr_std_yg3)



####################################
### initilise arrays for sim loop  # axis names not always track from now on because they change bwtween p=0 and p=1
####################################

##all groups
eqn_compare = pinp.sheep['i_eqn_compare']
##sire
ffcfw_start_sire = lw_initial_wzida0e0b0xyg0 - cfw_initial_wzida0e0b0xyg0
ffcfw_max_start_sire = ffcfw_start_sire
omer_history_start_m3g0[...] = np.nan
d_cfw_history_start_m2g0[...] = np.nan
cfw_start_sire = cfw_initial_wzida0e0b0xyg0
fd_start_sire = fd_initial_wzida0e0b0xyg0
fl_start_sire = fl_initial_wzida0e0b0xyg0
fd_min_start_sire = fd_initial_wzida0e0b0xyg0
aw_start_sire = aw_initial_wzida0e0b0xyg0
mw_start_sire = mw_initial_wzida0e0b0xyg0
bw_start_sire = bw_initial_wzida0e0b0xyg0
nw_start_sire = 0 #no dimensions to start
temp_lc_sire = np.array([0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.

##dams ^
# ldr_start_dams = np.array([1])
lb_start_dams = np.array([1])
# w_f_start_dams = np.array([0])
# nw_f_start_dams = np.array([0])
# nec_cum_start_dams = np.array([0])
# cf_w_b_mu_start_dams = np.array([0])
# cf_w_w_mu_start_dams = np.array([0])
# cf_conception_mu_start_dams = np.array([0])
guw_start_dams = np.array([0])
rc_birth_start_dams = np.array([0])
ffcfw_start_dams = lw_initial_wzida0e0b0xyg1 + cb0_dams[2, ...] - cfw_initial_wzida0e0b0xyg1
ffcfw_max_start_dams = ffcfw_start_dams
omer_history_start_m3g1[...] = np.nan
d_cfw_history_start_m2g1[...] = np.nan
cfw_start_dams = cfw_initial_wzida0e0b0xyg1
fd_start_dams = fd_initial_wzida0e0b0xyg1
fl_start_dams = fl_initial_wzida0e0b0xyg1
fd_min_start_dams = fd_initial_wzida0e0b0xyg1
aw_start_dams = aw_initial_wzida0e0b0xyg1
mw_start_dams = mw_initial_wzida0e0b0xyg1
bw_start_dams = bw_initial_wzida0e0b0xyg1
nw_start_dams = np.array([0])
temp_lc_dams = np.array([0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.
##yatf 
omer_history_start_m3g0[...] = np.nan
d_cfw_history_start_m2g2[...] = np.nan
nw_start_yatf = 0
ffcfw_start_yatf = 0
ffcfw_max_start_yatf = ffcfw_start_yatf
cfw_start_yatf = 0
nw_max_pa1e1b1nwzida0e0b0xyg2 = np.array([0]) 
temp_lc_yatf = np.array([0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.

##offs
ffcfw_start_offs = lw_initial_wzida0e0b0xyg3 - cfw_initial_wzida0e0b0xyg3
ffcfw_max_start_offs = ffcfw_start_offs
omer_history_start_m3g3[...] = np.nan
d_cfw_history_start_m2g3[...] = np.nan
cfw_start_offs = cfw_initial_wzida0e0b0xyg3
fd_start_offs = fd_initial_wzida0e0b0xyg3
fl_start_offs = fl_initial_wzida0e0b0xyg3
fd_min_start_offs = fd_initial_wzida0e0b0xyg3
aw_start_offs = aw_initial_wzida0e0b0xyg3
mw_start_offs = mw_initial_wzida0e0b0xyg3
bw_start_offs = bw_initial_wzida0e0b0xyg3
nw_start_offs = 0
temp_lc_offs = np.array([0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.





######################
### sim engine       #
######################

# ## initialise the arrays for the first period
# lw_ffcf = i_weaning_wt
# mw = 0.7 * lw_ffcf
# aw = 0.2 * lw_ffcf
# bw = 0.1 * lw_ffcf
# cfw = 0.6 #cfw at weaning
# fd = 19 #fd at weaning
# fl = 10 #fl at weaning

##set all arrays that are assigned using += to 0.




## Loop through each week of the simulation (p) for ewes
for p in range(1):
    # if p != 0:  # only carry this out with p<>0

    # ^ NOTE: need to calc all the yatf stuff including the ce param adjustment.

        ##set start values
# '''
# need to create rc part array (shape of rc start) and assing it to rc part end
# '''
        # variable_start = variable_end


        # ###check if the previous period was shearing for any of the sheep
        # if np.any(prev period os shearing):
        #     ####reset all wool parameters ^i dont get this. what if not all groups were shorn?
        # ###check if previous period was mating or lambing
        # if np.any(previous period mating or lambing):
        #     ####calc weight transfers, calc n transfers

        #     ####update weights and numbers
        # ###check if period is pre joining FVP
        # if np.any(period is prejoining fvp):
        #     ####weights and production
        #     weights & prodn[not mated, in utero] = weighted average
        #     ####reset animal numbers
        #     NM,IU[-1,-1] = 0
        #     NM,IU[0:-1,0:-1] = 0
        #     ####reset birthweight
        #     bw = 0
        #     ####reset reproduction params
        #     ldr = 1
        #     lb = 1
        # ###check if period is new FVP
        # if np.any(period is new FVP): #^not sure why there is the np.any???
        #     ####set all numbers a weight values to the prime

        # ###check if period is a new season
        # if np.any(period is a new season):
        #     ####set patterns for each seaon type to the same starting point

        # ##update lw target

    ##calculate dependent start values
    ###GFW (start)
    gfw_start_sire = cfw_start_sire / cw_sire[3, ...]
    gfw_start_dams = cfw_start_dams / cw_dams[3, ...]
    gfw_start_yatf = cfw_start_yatf / cw_yatf[3, ...]
    gfw_start_offs = cfw_start_offs / cw_offs[3, ...]

    ###LW (start -with fleece & conceptus)
    lw_start_sire = ffcfw_start_sire + gfw_start_sire
    lw_start_dams = ffcfw_start_dams + guw_start_dams + gfw_start_dams
    lw_start_yatf = ffcfw_start_yatf + gfw_start_yatf
    lw_start_offs = ffcfw_start_offs + gfw_start_offs

    ###Normal weight (start)
    nw_start_sire = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg0[p], np.maximum(nw_start_sire, ffcfw_start_sire + cn_sire[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg0[p]  - ffcfw_start_sire)))
    nw_start_dams = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg1[p], np.maximum(nw_start_dams, ffcfw_start_dams + cn_dams[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg1[p]  - ffcfw_start_dams)))
    nw_start_yatf = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg2[p], np.maximum(nw_start_yatf, ffcfw_start_yatf + cn_yatf[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg2[p]  - ffcfw_start_yatf)))
    nw_start_offs = np.minimum(nw_max_pa1e1b1nwzida0e0b0xyg3[p], np.maximum(nw_start_offs, ffcfw_start_offs + cn_offs[3, ...] * (nw_max_pa1e1b1nwzida0e0b0xyg3[p]  - ffcfw_start_offs)))

    ###Relative condition (start)
    rc_start_sire = ffcfw_start_sire / nw_start_sire
    rc_start_dams = ffcfw_start_dams / nw_start_dams
    rc_start_yatf = ffcfw_start_yatf / nw_start_yatf
    rc_start_offs = ffcfw_start_offs / nw_start_offs
    
    ###Relative conditon of dam at parturition - needs to be remembered between loops (milk production)  
    rc_birth_start_dams = rc_birth_start_dams * ~period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...] + rc_start_dams * period_is_birth_pa1e1b1nwzida0e0b0xyg1[p, ...]

    ###Relative size (start) - dams & sires	
    relsize_start_sire = np.minimum(1, nw_start_sire / srw_xyg0)		
    relsize_start_dams = np.minimum(1, nw_start_dams / srw_xyg1)		
    relsize_start_yatf = np.minimum(1, nw_start_yatf / srw_xyg2)		
    relsize_start_offs = np.minimum(1, nw_start_offs / srw_xyg3)

    ###Relative size for LWG (start). Capped by current LW
    relsize1_sire = np.minimum(ffcfw_max_start_sire, nw_max_pa1e1b1nwzida0e0b0xyg0[p]) / srw_xyg0
    relsize1_dams = np.minimum(ffcfw_max_start_dams, nw_max_pa1e1b1nwzida0e0b0xyg1[p]) / srw_xyg1
    relsize1_yatf = np.minimum(ffcfw_max_start_yatf, nw_max_pa1e1b1nwzida0e0b0xyg2[p]) / srw_xyg2
    relsize1_offs = np.minimum(ffcfw_max_start_offs, nw_max_pa1e1b1nwzida0e0b0xyg3[p]) / srw_xyg3
    ###PI Size factor (for cattle)
    zf_sire = np.maximum(1, 1 + cr_sire[7, ...] - relsize_start_sire)
    zf_dams = np.maximum(1, 1 + cr_dams[7, ...] - relsize_start_dams)
    zf_yatf = np.maximum(1, 1 + cr_yatf[7, ...] - relsize_start_yatf)
    zf_offs = np.maximum(1, 1 + cr_offs[7, ...] - relsize_start_offs)
    ###EVG Size factor (decreases steadily)
    z1f_sire = 1 / (1 + np.exp(-cg_sire[4, ...] * (relsize1_sire - cg_sire[5, ...])))
    z1f_dams = 1 / (1 + np.exp(-cg_dams[4, ...] * (relsize1_dams - cg_dams[5, ...])))
    z1f_yatf = 1 / (1 + np.exp(-cg_yatf[4, ...] * (relsize1_yatf - cg_yatf[5, ...])))
    z1f_offs = 1 / (1 + np.exp(-cg_offs[4, ...] * (relsize1_offs - cg_offs[5, ...])))
    ###EVG Size factor (increases at maturity)
    z2f_sire = np.clip((relsize1_sire - cg_sire[6, ...]) / (cg_sire[7, ...] - cg_sire[6, ...]), 0 ,1)
    z2f_dams = np.clip((relsize1_dams - cg_dams[6, ...]) / (cg_dams[7, ...] - cg_dams[6, ...]), 0 ,1)
    z2f_yatf = np.clip((relsize1_yatf - cg_yatf[6, ...]) / (cg_yatf[7, ...] - cg_yatf[6, ...]), 0 ,1)
    z2f_offs = np.clip((relsize1_offs - cg_offs[6, ...]) / (cg_offs[7, ...] - cg_offs[6, ...]), 0 ,1)




    ## conception, mortality and numbers
    ### base mortality
    mortality_base_sire = f_mortality_base(cd_sire, cg_sire, rc_start_sire, ebg_start_sire, d_nw_max_pa1e1b1nwzida0e0b0xyg0[p])
    mortality_base_dams = f_mortality_base(cd_dams, cg_dams, rc_start_dams, ebg_start_dams, d_nw_max_pa1e1b1nwzida0e0b0xyg1[p])
    mortality_base_yatf = f_mortality_base(cd_yatf, cg_yatf, rc_start_yatf, ebg_start_yatf, d_nw_max_pa1e1b1nwzida0e0b0xyg2[p])
    mortality_base_offs = f_mortality_base(cd_offs, cg_offs, rc_start_offs, ebg_start_offs, d_nw_max_pa1e1b1nwzida0e0b0xyg3[p])
    
    ### weaner mortality 
    eqn_group = 2
    eqn_system = 0 # CSIRO = 0
    ####sire
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
            temp0 = sfun.f_mortality_weaner_cs(cd_sire, cg_sire, age_pa1e1b1nwzida0e0b0xyg0[p], ebg_start_sire, d_nw_max_pa1e1b1nwzida0e0b0xyg0[p])
            if eqn_used:
                weaner_mortality_sire = temp0
            if eqn_compare:
                r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0
    ####dams
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
            temp0 = sfun.f_mortality_weaner_cs(cd_dams, cg_dams, age_pa1e1b1nwzida0e0b0xyg1[p], ebg_start_dams, d_nw_max_pa1e1b1nwzida0e0b0xyg1[p])
            if eqn_used:
                weaner_mortality_dams = temp0
            if eqn_compare:
                r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0
    ####offs
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
            temp0 = sfun.f_mortality_weaner_cs(cd_offs, cg_offs, age_pa1e1b1nwzida0e0b0xyg3[p], ebg_start_offs, d_nw_max_pa1e1b1nwzida0e0b0xyg3[p])
            if eqn_used:
                weaner_mortality_offs = temp0
            if eqn_compare:
                r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0

    ### dam mortality - Peri-natal Dam mortality 
    eqn_group = 3
    eqn_system = 0 # CSIRO = 0
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
            temp0 = sfun.f_mortality_dam_cs(cb1_dams, cg_dams, nw_start_dams, ebg_start_dams, days_periodpa1e1b1nwzida0e0b0xyg1[p], period_between_birth6wks_pa1e1b1nwzida0e0b0xyg1[p], gest_propn_pa1e1b1nwzida0e0b0xyg1[p])
            if eqn_used:
                mortality_dams = temp0
            if eqn_compare:
                r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0

    ### Peri-natal progeny mortality (progeny survival)
    eqn_group = 1
    eqn_system = 0 # CSIRO = 0
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
            temp0, temp1, temp2 = sfun.f_mortality_progeny_cs(cd_yatf, cb1_yatf, w_b_yatf, rc_start_dams, w_b_exp_y_dams, prev_period_is_birth_pa1e1b1nwzida0e0b0xyg1[p], chill_index_pa1e1b1nwzida0e0b0xygm1[p], nfoet_b1nwzida0e0b0xyg)
            if eqn_used:
                mortalityd_yatf = temp0
                mortalityx_yatf = temp1
                mortalityd_dams = temp2
            if eqn_compare:
                r_compare_q0q1q2pyatf[eqn_system, eqn_group, 0, p, ...] = temp0
                r_compare_q0q1q2pyatf[eqn_system, eqn_group, 1, p, ...] = temp1
    
  
  # ###calc preg tox losses if less than 6wks to lambing.
    # if date <= lambing - 42:
    #     mr[] += f_preg_tox_cs
    #     mr[] += f_preg_tox_mu
    # ###if period is lambing calc dystocia losses
    # if period_date == lambing:
    #     mr += f_dystocia_cs
    #     mr += f_dystocia_mu
    # ###if previous period was lamning calc ewe mortality
    # if period_date == lambing+7:
    #     mry += f_mortality_ewe_cs
    #     mry += f_mortality_ewe_mu
    # ###if previous period was mating calc conception and transfers
    # if period_date == mating+7:
    #     cr_ojexyl[mask] += sfun.conception(lw_ffcf[p,...], srw_j)[mask]
    #     # with a mask to a
    #     nlb_ojewbl += cr_ojexyl#convert conception in _xy format to _wb
    # ###calc numbers after mortality and repro
    # number[p,...] = sfun.transfers(number[p-1,...], sales
    #                 , ewe_mortality, cr, lamb_mortality, ....)
    # number[p] = (number[p-1] - sales[p-1]) * (1 - mortality) ....
    # ###equation system loop ^dont know this enough to build it yet

    ##mating
    n_sire_a1e1b1nwzida0e0b0xyg1p8 = f_sire_req(sire_propn_pa1e1b1nwzida0e0b0xyg1[p], sire_periods_g0p8, pinp.sheep['i_sire_recovery'], pinp.sheep['i_startyear'], date_end_pa1e1b1nwzida0e0b0xyg[p], period_is_prejoin_pa1e1b1nwzida0e0b0xyg1[p])



    ##conception Dams
    eqn_group = 1
    eqn_system = 0 # CSIRO = 0
    if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
        eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
        if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
            temp0 = sfun.f_conception_cs(cf_dams, cb1_dams, relsize_start_dams, rc_start_dams, crg_doy_pa1e1b1nwzida0e0b0xyg1[p], period_is_mating_pa1e1b1nwzida0e0b0xyg1[p])
            if eqn_used:
                conception_dams = temp0
            if eqn_compare:
                r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0

    ##feed supply loop
    # this loop is only required if a LW target is specified for the animals
    # if there is a target then the loop needs to continue until
    # the feed supply has converged on a value that generates a liveweight
    # change close to the target
    # The loop needs to execute at least once, then repeat if there
    # is a target and the result is not close enough to the target
    # if this period (p) is a new feed variation period (f) or a new MIDAS feed period (n):
    #     then feed_supply_jxyl = feed_supply_pjxyl[p,...]
    #     otherwise use feedsupply from last period (which was optimised for the target)
    # Feed supply loop start
    # ##thought about making this a function but that is more difficult to debug so i just use a break if there is no target/need for a loop
    # ##adjust feed supply
    # ###initial info ^this will need to be hooked up with correct inputs, if they are the same for each period they donn't need to be initilised below
    # target_lwc =
    # epsilon =
    # n_max_itn =
    feedsupply = 1
    # attempts = np.zeros(,n_max_itn,2) #^need to add the dimensions of lwc at the beginning
    for itn in range(n_max_itn):
        ##potential intake
        eqn_group = 4
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0 = sfun.f_potential_intake_cs(ci_sire, srw_xyg0, relsize_start_sire, rc_start_sire, temp_lc_sire, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p] 
                                                   , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg0[p])
                if eqn_used:
                    pi_sire = temp0
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0 = sfun.f_potential_intake_cs(ci_dams, srw_xyg1, relsize_start_dams, rc_start_dams, temp_lc_dams, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p] 
                                                  , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg1[p], rc_birth_start = rc_birth_start_dams, pi_age_y = pi_age_y_pa1e1b1nwzida0e0b0xyg1
                                                  , lb_start = lb_start_dams, lactation_in_period = period_between_birthwean_pa1e1b1nwzida0e0b0xyg1[p])
                if eqn_used:
                    pi_dams = temp0
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0 = sfun.f_potential_intake_cs(ci_offs, srw_xyg1, relsize_start_offs, rc_start_offs, temp_lc_offs, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p] 
                                                    , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg3[p])
                if eqn_used:
                    pi_offs = temp0
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0
        
        ###murdoch ^function doesnt exist yet, add args when it is built
        eqn_system = 1 # mu = 1
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0 = sfun.f_potential_intake_mu()
                if eqn_used:
                    pi_sire = temp0
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0 = sfun.f_potential_intake_mu()
                if eqn_used:
                    pi_dams = temp0
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0 = sfun.f_potential_intake_mu()
                if eqn_used:
                    pi_offs = temp0
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0

        ##feedsupply
        foo_sire, hr_sire, dmd_sire, intake_s_sire = sfun.f_feedsupply(cu3, cu4, feedsupply_std_pa1e1b1nwzida0e0b0xyg0[p], paststd_foo_pa1e1b1j0wzida0e0b0xyg[p], paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], legume_pa1e1b1nwzida0e0b0xyg[p], pi_sire, pasture_stage_pa1e1b1j0wzida0e0b0xyg[p], pinp.sheep['i_hd_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'], uinp.pastparameters['i_hd_std'])
        foo_dams, hr_dams, dmd_dams, intake_s_dams = sfun.f_feedsupply(cu3, cu4, feedsupply_std_pa1e1b1nwzida0e0b0xyg1[p], paststd_foo_pa1e1b1j0wzida0e0b0xyg[p], paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], legume_pa1e1b1nwzida0e0b0xyg[p], pi_dams, pasture_stage_pa1e1b1j0wzida0e0b0xyg[p], pinp.sheep['i_hd_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'], uinp.pastparameters['i_hd_std'])
        foo_yatf, hr_yatf, dmd_yatf, intake_s_yatf = sfun.f_feedsupply(cu3, cu4, feedsupply_std_pa1e1b1nwzida0e0b0xyg1[p], paststd_foo_pa1e1b1j0wzida0e0b0xyg[p], paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], legume_pa1e1b1nwzida0e0b0xyg[p], pi_yatf, pasture_stage_pa1e1b1j0wzida0e0b0xyg[p], pinp.sheep['i_hd_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'], uinp.pastparameters['i_hd_std']) #yatf use dam feedsupply_std
        foo_offs, hr_offs, dmd_offs, intake_s_offs = sfun.f_feedsupply(cu3, cu4, feedsupply_std_pa1e1b1nwzida0e0b0xyg3[p], paststd_foo_pa1e1b1j0wzida0e0b0xyg[p], paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], legume_pa1e1b1nwzida0e0b0xyg[p], pi_offs, pasture_stage_pa1e1b1j0wzida0e0b0xyg[p], pinp.sheep['i_hd_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'], uinp.pastparameters['i_hd_std'])

        ##relative availability
        eqn_group = 5
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0 = sfun.f_ra_cs(cr_sire, foo_sire, hf_sire, zf_sire)
                if eqn_used:
                    ra_sire = temp0
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0 = sfun.f_ra_cs(cr_dams, foo_dams, hf_dams, zf_dams)
                if eqn_used:
                    meme_dams = temp0
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0 = sfun.f_ra_cs(cr_offs, foo_offs, hf_offs, zf_offs)
                if eqn_used:
                    meme_offs = temp0
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0

        eqn_system = 0 # Murdoch = 1
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0 = sfun.f_ra_mu(cu2_sire, foo_sire, hf_sire, zf_sire)
                if eqn_used:
                    ra_sire = temp0
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0 = sfun.f_ra_mu(cu2_dams, foo_dams, hf_dams, zf_dams)
                if eqn_used:
                    meme_dams = temp0
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0 = sfun.f_ra_mu(cu2_offs, foo_offs, hf_offs, zf_offs)
                if eqn_used:
                    meme_offs = temp0
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0


        ##intake
        mei_sire, intake_f_sire, md_solid_sire, mei_propn_milk_sire, mei_propn_herb_sire, mei_propn_supp_sire = sfun.f_intake(cr_sire, pi_sire, ra_sire, dmd_sire,  md_herb_sire, feedsupply_sire, intake_s_sire, pinp.sheep['i_md_supp'], legume_pa1e1b1nwzida0e0b0xyg[p], pinp.sheep['i_sf'])
        mei_dams, intake_f_dams, md_solid_dams, mei_propn_milk_dams, mei_propn_herb_dams, mei_propn_supp_dams = sfun.f_intake(cr_dams, pi_dams, ra_dams, dmd_dams,  md_herb_dams, feedsupply_dams, intake_s_dams, pinp.sheep['i_md_supp'], legume_pa1e1b1nwzida0e0b0xyg[p], pinp.sheep['i_sf'])
        mei_offs, intake_f_offs, md_solid_offs, mei_propn_milk_offs, mei_propn_herb_offs, mei_propn_supp_offs = sfun.f_intake(cr_offs, pi_offs, ra_offs, dmd_offs,  md_herb_offs, feedsupply_offs, intake_s_offs, pinp.sheep['i_md_supp'], legume_pa1e1b1nwzida0e0b0xyg[p], pinp.sheep['i_sf'])


        ##energy
        eqn_group = 6
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_sire, cx_sire, cm_sire, lw_start_sire, mr_age_pa1e1b1nwzida0e0b0xyg0, mei_sire, omer_history_start_sire, days_period_pa1e1b1nwzida0e0b0xyg0, md_solid_sire, pinp.sheep['i_md_supp'], md_herb_sire, lgf_eff_pa1e1b1nwzida0e0b0xyg0[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness'], density_pa1e1b1nwzida0e0b0xyg0[p], foo_sire, feedsupply_sire, intake_f_sire, dmd_sire)
                if eqn_used:
                    meme_sire = temp0
                    omer_history_sire = temp1
                    km_sire = temp2
                    kg_fodd_sire = temp3
                    kg_supp_sire = temp4  # temp5 is not used for sires
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0  # more of the return variable could be retained
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_dams, cx_dams, cm_dams, lw_start_dams, mr_age_pa1e1b1nwzida0e0b0xyg1, mei_dams, omer_history_start_dams, days_period_pa1e1b1nwzida0e0b0xyg1, md_solid_dams, pinp.sheep['i_md_supp'], md_herb_dams, lgf_eff_pa1e1b1nwzida0e0b0xyg1[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness'], density_pa1e1b1nwzida0e0b0xyg1[p], foo_dams, feedsupply_dams, intake_f_dams, dmd_dams)
                if eqn_used:
                    meme_dams = temp0
                    omer_history_dams = temp1
                    km_dams = temp2
                    kg_fodd_dams = temp3
                    kg_supp_dams = temp4
                    kl_dams = temp5
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0  # more of the return variable could be retained
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_offs, cx_offs, cm_offs, lw_start_offs, mr_age_pa1e1b1nwzida0e0b0xyg3, mei_offs, omer_history_start_offs, days_period_pa1e1b1nwzida0e0b0xyg3, md_solid_offs, pinp.sheep['i_md_supp'], md_herb_offs, lgf_eff_pa1e1b1nwzida0e0b0xyg3[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness'], density_pa1e1b1nwzida0e0b0xyg3[p], foo_offs, feedsupply_offs, intake_f_offs, dmd_offs)
                if eqn_used:
                    meme_offs = temp0
                    omer_history_offs = temp1
                    km_offs = temp2
                    kg_fodd_offs = temp3
                    kg_supp_offs = temp4 # temp5 is not used for offspring
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0  # more of the return variable could be retained


        
       
 

        ##foetal growth - dams
        eqn_group = 8
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0, temp1, temp2, temp3, temp4, temp5, temp6, temp7 = sfun.f_foetus_cs(cp_dams, cb1_dams, cx_dams, kc_yg1, nfoet_b1nwzida0e0b0xyg, rc_start_dams, relsize_start_dams, rc_start_dams, nec_cum_start_dams, w_b_std_y_b1nwzida0e0b0xyg1, w_f_start_dams, nw_f_start_dams, nwf_age_f_pa1e1b1nwzida0e0b0xyg1[p], guw_age_f_pa1e1b1nwzida0e0b0xyg1[p], ce_age_f_pa1e1b1nwzida0e0b0xyg1[p], days_period_f_pa1e1b1nwzida0e0b0xyg1[p], period_is_birth_pa1e1b1nwzida0e0b0xyg1[p])
                if eqn_used:
                    w_f_dams = temp0
                    nec_cum_dams = temp1
                    mec_dams = temp2
                    nec_dams = temp3
                    w_b_yatf = temp4
                    w_b_exp_y_dams = temp5
                    nw_f_dams = temp6  
                    guw_dams = temp7  
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0  
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 1, p, ...] = temp1  
        
         
        ##milk production
        mp2_dams, mel_dams, nel_dams, ldr_dams, lb_dams = f_milk(cl_dams, srw_xyg1, relsize_start_dams, rc_birth_start_dams, mei_dams, meme_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p], rc_start_dams, ffcfw_start_yatf, lb_start_dams, ldr_start_dams, age_pa1e1b1nwzida0e0b0xyg2, mp_age_y,  mp2_age_y, uinp.parameters['i_x_pos'], days_period_pa1e1b1nwzida0e0b0xyg2[p], kl_dams, lact_nut_effect_pa1e1b1nwzida0e0b0xyg1)
        mp2_yatf = mp2_dams / nyatf_b1nwzida0e0b0xyg
        
        ##potential intake - yatf
        eqn_group = 4
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                temp0 = sfun.f_potential_intake_cs(ci_yatf, srw_xyg2, relsize_start_yatf, rc_start_yatf, temp_lc_yatf, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p] 
                                                   , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg2[p]
                                                   , mp2 = mp2_yatf, piyf = piyf_pa1e1b1nwzida0e0b0xyg2[p], lactation_in_period = lactation_in_period_yatf[p])
                if eqn_used:
                    pi_yatf = temp0
                if eqn_compare:
                    r_compare_q0q1q2pyatf[eqn_system, eqn_group, 0, p, ...] = temp0


        ##intake - yatf
        ra_mu_yatf = sfun.f_ra_cs(cr_yatf, hf_yatf, zf_yatf, foo_yatf)
        ra_mu_yatf = sfun.f_ra_mu(cu2_yatf)
        mei_yatf, intake_f_yatf, md_solid_yatf, mei_propn_milk_yatf, mei_propn_herb_yatf, mei_propn_supp_yatf = sfun.f_intake(cr_yatf, pi_yatf, ra_yatf, dmd_yatf,  md_herb_yatf, feedsupply_yatf, intake_s_yatf, pinp.sheep['i_md_supp'], legume_pa1e1b1nwzida0e0b0xyg[p], pinp.sheep['i_sf'], mp2_yatf)

        ##energy - yatf
        eqn_group = 6
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                temp0, temp1, temp2, temp3, temp4, temp5 = sfun.f_energy_cs(ck_yatf, cx_yatf, cm_yatf, lw_start_yatf, mr_age_pa1e1b1nwzida0e0b0xyg2[p], mei_yatf, omer_history_start_yatf, days_period_pa1e1b1nwzida0e0b0xyg2[p], md_solid_yatf, pinp.sheep['i_md_supp'], md_herb_yatf, lgf_eff_pa1e1b1nwzida0e0b0xyg2[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p,...], pinp.sheep['i_steepness'], density_pa1e1b1nwzida0e0b0xyg2[p], foo_yatf, feedsupply_yatf, intake_f_yatf, dmd_yatf, mei_propn_milk_yatf)
                if eqn_used:
                    meme_yatf = temp0
                    omer_history_yatf = temp1
                    km_yatf = temp2
                    kg_fodd_yatf = temp3
                    kg_supp_yatf = temp4  # temp5 is not used for yatf
                if eqn_compare:
                    r_compare_q0q1q2pyatf[eqn_system, eqn_group, 0, p, ...] = temp0  # more of the return variable could be retained



        ##wool production
        d_cfw_sire, d_fd_sire, d_fl_sire, d_cfw_history_sire, mew_sire, new_sire = sfun.f_fibre(cw_sire, cc_sire, ffcfw_start_sire, relsize_start_sire, d_cfw_history_start_m2g0, mei_sire, mew_min_pa1e1b1nwzida0e0b0xyg0[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg0[p, ...], sfd_a0e0b0xyg0, wge_a0e0b0xyg0, af_wool_pa1e1b1nwzida0e0b0xyg0[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...],  kw_yg0, days_period_pa1e1b1nwzida0e0b0xyg0[p])
        d_cfw_dams, d_fd_dams, d_fl_dams, d_cfw_history_dams, mew_dams, new_dams = sfun.f_fibre(cw_dams, cc_dams, ffcfw_start_dams, relsize_start_dams, d_cfw_history_start_m2g1, mei_dams, mew_min_pa1e1b1nwzida0e0b0xyg1[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg1[p, ...], sfd_a0e0b0xyg1, wge_a0e0b0xyg1, af_wool_pa1e1b1nwzida0e0b0xyg1[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...],  kw_yg1, days_period_pa1e1b1nwzida0e0b0xyg1[p], mec_dams, mel_dams, gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
        d_cfw_yatf, d_fd_yatf, d_fl_yatf, d_cfw_history_yatf, mew_yatf, new_yatf = sfun.f_fibre(cw_yatf, cc_yatf, ffcfw_start_yatf, relsize_start_yatf, d_cfw_history_start_m2g2, mei_yatf, mew_min_pa1e1b1nwzida0e0b0xyg2[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg2[p, ...], sfd_a0e0b0xyg2, wge_a0e0b0xyg2, af_wool_pa1e1b1nwzida0e0b0xyg2[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...],  kw_yg2, days_period_pa1e1b1nwzida0e0b0xyg2[p])
        d_cfw_offs, d_fd_offs, d_fl_offs, d_cfw_history_offs, mew_offs, new_offs = sfun.f_fibre(cw_offs, cc_offs, ffcfw_start_offs, relsize_start_offs, d_cfw_history_start_m2g3, mei_offs, mew_min_pa1e1b1nwzida0e0b0xyg3[p], d_cfw_ave_pa1e1b1nwzida0e0b0xyg3[p, ...], sfd_a0e0b0xyg3, wge_a0e0b0xyg3, af_wool_pa1e1b1nwzida0e0b0xyg3[p, ...], dlf_eff_pa1e1b1nwzida0e0b0xyg[p, ...],  kw_yg3, days_period_pa1e1b1nwzida0e0b0xyg3[p])
        
        

        ##energy to offset chilling
        mem_sire, temp_lc_sire, kg_sire, level_sire = sfun.f_chill_cs(cc_sire, ck_sire, ffcfw_start_sire, rc_start_sire, fl_start_sire, mei_sire, meme_sire, mew_sire, new_sire, km_sire, kg_supp_sire, kg_fodd_sire, mei_propn_supp_sire, mei_propn_herb_sire, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p], temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygm1[p], index_m0)
        mem_dams, temp_lc_dams, kg_dams, level_dams = sfun.f_chill_cs(cc_dams, ck_dams, ffcfw_start_dams, rc_start_dams, fl_start_dams, mei_dams, meme_dams, mew_dams, new_dams, km_dams, kg_supp_dams, kg_fodd_dams, mei_propn_supp_dams, mei_propn_herb_dams, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p], temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygm1[p], index_m0, guw = guw_dams, kl = kl_dams,	mei_propn_milk	= mei_propn_milk_dams, mec = mec_dams, mel = mel_dams, nec = nec_dams, nel = nel_dams, gest_propn	= gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn = lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
        mem_yatf, temp_lc_yatf, kg_yatf, level_yatf = sfun.f_chill_cs(cc_yatf, ck_yatf, ffcfw_start_yatf, rc_start_yatf, fl_start_yatf, mei_yatf, meme_yatf, mew_yatf, new_yatf, km_yatf, kg_supp_yatf, kg_fodd_yatf, mei_propn_supp_yatf, mei_propn_herb_yatf, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p], temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygm1[p], index_m0)
        mem_offs, temp_lc_offs, kg_offs, level_offs = sfun.f_chill_cs(cc_offs, ck_offs, ffcfw_start_offs, rc_start_offs, fl_start_offs, mei_offs, meme_offs, mew_offs, new_offs, km_offs, kg_supp_offs, kg_fodd_offs, mei_propn_supp_offs, mei_propn_herb_offs, temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p], temp_min_pa1e1b1nwzida0e0b0xyg[p], ws_pa1e1b1nwzida0e0b0xyg[p], rain_pa1e1b1nwzida0e0b0xygm1[p], index_m0)
 

        ##calc lwc
        eqn_group = 7
        eqn_system = 0 # CSIRO = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_sire, rc_start_sire, mei_sire, mem_sire, mew_sire, z1f_sire, z2f_sire, kg_sire)
                if eqn_used:
                    ebg_sire = temp0
                    evg_history_sire = temp1
                    pg_sire = temp2
                    fg_fodd_sire = temp3
                    level_sire = temp4
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0  
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 1, p, ...] = temp1  
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_dams, rc_start_dams, mei_dams, mem_dams, mew_dams, z1f_dams, z2f_dams, kg_dams, mec_dams, mel_dams, gest_propn_pa1e1b1nwzida0e0b0xyg1[p], lact_propn_pa1e1b1nwzida0e0b0xyg1[p])
                if eqn_used:
                    ebg_dams = temp0
                    evg_history_dams = temp1
                    pg_dams = temp2
                    fg_fodd_dams = temp3
                    level_dams = temp4
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0 
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 1, p, ...] = temp1  
            ###yatf
            eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_yatf, rc_start_yatf, mei_yatf, mem_yatf, mew_yatf, z1f_yatf, z2f_yatf, kg_yatf)
                if eqn_used:
                    ebg_yatf = temp0
                    evg_history_yatf = temp1
                    pg_yatf = temp2
                    fg_fodd_yatf = temp3
                    level_yatf = temp4
                if eqn_compare:
                    r_compare_q0q1q2pyatf[eqn_system, eqn_group, 0, p, ...] = temp0  
                    r_compare_q0q1q2pyatf[eqn_system, eqn_group, 1, p, ...] = temp1  
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0, temp1, temp2, temp3, temp4 = sfun.f_lwc_cs(cg_offs, rc_start_offs, mei_offs, mem_offs, mew_offs, z1f_offs, z2f_offs, kg_offs)
                if eqn_used:
                    ebg_offs = temp0
                    evg_history_offs = temp1
                    pg_offs = temp2
                    fg_fodd_offs = temp3
                    level_offs = temp4
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0  
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 1, p, ...] = temp1  
        

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
        eqn_group = 10
        eqn_system = 0 # Baxter and Clapperton = 0
        if pinp.sheep['i_eqn_exists_q0q1'][eqn_system, eqn_group]:  # proceed with call & assignment if this system exists for this group
            ###sire
            eqn_used = (eqn_used_g0_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg0[p,...] >0):
                temp0, temp1 = sfun.f_emissions_bc(ch_sire, intake_f_sire, intake_s_sire, md_solid_sire, level_sire)
                if eqn_used:
                    ch4_total_sire = temp0
                    ch4_animal_sire = temp1
                if eqn_compare:
                    r_compare_q0q1q2psire[eqn_system, eqn_group, 0, p, ...] = temp0  
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg1[p,...] >0):
                temp0, temp1 = sfun.f_emissions_bc(ch_dams, intake_f_dams, intake_s_dams, md_solid_dams, level_dams)
                if eqn_used:
                    ch4_total_dams = temp0
                    ch4_animal_dams = temp1
                if eqn_compare:
                    r_compare_q0q1q2pdams[eqn_system, eqn_group, 0, p, ...] = temp0  
            ###yatf
            eqn_used = (eqn_used_g2_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg2[p,...] >0):
                temp0, temp1 = sfun.f_emissions_bc(ch_yatf, intake_f_yatf, intake_s_yatf, md_solid_yatf, level_yatf)
                if eqn_used:
                    ch4_total_yatf = temp0
                    ch4_animal_yatf = temp1
                if eqn_compare:
                    r_compare_q0q1q2pyatf[eqn_system, eqn_group, 0, p, ...] = temp0  
            ###offs
            eqn_used = (eqn_used_g3_q1p[eqn_group, p] == eqn_system)
            if (eqn_used or eqn_compare) and np.any(days_period_pa1e1b1nwzida0e0b0xyg3[p,...] >0):
                temp0, temp1 = sfun.f_emissions_bc(ch_offs, intake_f_offs, intake_s_offs, md_solid_offs, level_offs)
                if eqn_used:
                    ch4_total_offs = temp0
                    ch4_animal_offs = temp1
                if eqn_compare:
                    r_compare_q0q1q2poffs[eqn_system, eqn_group, 0, p, ...] = temp0  

    # ##end values



def f_mortality_base(cd, cg, rc_start, ebg_start, d_nw_max):
    return cd[1, ...] + cd[2, ...] * np.max(0, cd[3, ...] - rc_start) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))



def f_mortality_weaner_cs(cd, cg, age, ebg_start, d_nw_max):
    return cd[13, ...] * sfun.f_ramp(age, cd[15, ...], cd[14, ...]) * ((cd[16, ...] * d_nw_max) > (ebg_start* cg[18, ...]))


def f_mortality_dam_cs(cb1, cg, nw_start, ebg_start, days_period, period_is_6wpp, gest_propn):
    ##(Twin) Dam mortality in last 6 weeks (preg tox)
    t_mort = days_period * gest_propn /42 * sfun.f_sig(-42 * ebg_start * cg[18, ...] / nw_start, cb1[4, ...], cb1[5, ...])
    ##If not last 6 weeks then = 0
    mort = t_mort * period_is_6wpp
    return mort
    
def f_mortality_progeny_cs(cd, cb1, w_b, rc_start, w_b_exp_y, prev_is_birth, chill_index_panym1):
    ##Progeny losses due to large progeny (dystocia)
    mortalityd_yatf = sfun.f_sig(w_b / w_b_exp_y * np.maximum(1, rc_start), cb1[6, ...], cb1[7, ...]) * prev_is_birth
    ##dam mort due to large progeny (dystocia)
    mortalityd_dams = mortalityd * cd[21,...] / nfoet_b1any
    mortalityd_yatf = mortalityd_yatf * (1- cd[21,...])
    ##Exposure index
    xo = cd[8, ..., na] - cd[9, ..., na] * rc_start[..., na] + cd[10, ..., na] * chill_index_panym1 + cb1[11, ..., na]
    ##Progeny mortality at birth from exposure
    mortalityx = np.average(np.exp(xo) / (1 - np.exp(xo)) ,axis = -1) * prev_is_birth
    return mortalityd_yatf, mortalityx, mortalityd_dams







# m = np.arange(2*3*5).reshape((2,3,5))
# slc = [slice(None)] * len(m.shape)
# slc[2] = slice( 0,-2)
# m[tuple(slc)] 
# m[:,:,0:-2]

def f_dynamic_slice(arr, axis, start, stop):
    sl = [slice(None)] * arr.ndim
    sl[axis] = slice( start, stop)
    return arr[tuple(sl)]


def f_conception_cs(cf, cb1, relsize_start, rc_start, crg_doy, period_is_mating):
    ##Conception greater than or equal to x
    relsize_start_e1sliced = f_dynamic_slice(relsize_start, pinp.sheep['i_e1_pos'], 0, 1) #take slice from e1 axis
    relsize_start_e1b1sliced = f_dynamic_slice(relsize_start_e1sliced, uinp.parameters['i_b1_pos'], -1, None) #take slice from b1 axis
    rc_start_e1sliced = f_dynamic_slice(rc_start, pinp.sheep['i_e1_pos'], 0, 1) #take slice from e1 axis
    rc_start_e1b1sliced = f_dynamic_slice(rc_start_e1sliced, uinp.parameters['i_b1_pos'], -1, None) #take slice from b1 axis
    crg = crg_doy * sfun.f_sig( relsize_start_e1b1sliced * rc_start_e1b1sliced, cb1[2, ...], cb1[3, ...])
    ##Remove conception from the LSLN (b1) with progeny losses	
    slc = [slice(None)] * len(crg.shape)
    slc[uinp.parameters['i_b1_pos']] = slice(4,None)
    crg[tuple(slc)] = 0 #cant use dynamic slice funtion here because need to assign to slice
    ##Define the temp array shape	
    cr_temp = crg
    ##Conception equal to x (temporary array as if this period is joining)
    slc[uinp.parameters['i_b1_pos']] = slice(0,-1)
    cr_temp[tuple(slc)] = np.maximum(0, f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 0, -1) - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, None))    # (difference between '>x' and '>x+1')
    ##Dams that don't retain to 3rd trimester but do not return to service are added to 00 slice rather than staying in NM slice	
    slc[uinp.parameters['i_b1_pos']] = slice(0,1)
    cr_temp[tuple(slc)] = np.minimum(1 - f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 0, 1), f_dynamic_slice(crg, uinp.parameters['i_b1_pos'], 1, 2)) * (cf[5, ...] / (1 - cf[5, ...]))
    ##Proportion of animals with conception equal to x (if this period is mating)	
    conception = cr_temp * period_is_mating
    ##Number remaining not-mated (cr[-1])
    return -np.sum(conception, axis = (pinp.sheep['i_e1_pos'], uinp.parameters['i_b1_pos']), keepdims=True)




def f_emissions_bc(ch, intake_f, intake_s, md_solid, level):
    ##Methane production total
    ch4_total = ch[1, ...] * (intake_f + intake_s)*((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))
    ##Methane production animal component
    ch4_animal = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)
    return ch4_total, ch4_animal




def f_history(history, new_value, days_in_period):
    '''
    The idea that the f_history is implementing is for traits that have a lag from increased nutrition to increased production. 
    The representation being that the production today is an average of the non-lagged estimated production from the last x days (where x is either len_m2 or len_m3, either of which can be 1 but are expected to be 25).
    The history function is keeping track of the last x days of estimated production (slice 0 is the most recent day, slice -1 is the oldest day of the history). 
    The process is:
        1. move the most recent history back (to make space for the production from this period: days_period) 
        2. make the 0:days_period = this period estimated production
    '''
    offset = np.minimum(days_in_period, history.shape[0])
    history = np.roll(history, offset, axis = 0)
    history[:offset, ...] = new_value
    lagged = np.nanmean(history, axis = 0)
    return lagged, history



def f_fibre(cw, cc, ffcfw_start, relsize_start, d_cfw_history_start_m2pa1e1b1nwzida0e0b0xyg, mei, mew_min_pa1e1b1nwzida0e0b0xyg, d_cfw_ave_pa1e1b1nwzida0e0b0xyg, sfd_a0e0b0xyg, wge_a0e0b0xyg
            , af_wool_pa1e1b1nwzida0e0b0xyg, dlf_eff_pa1e1b1nwzida0e0b0xyg,  kw_yg, days_period_pa1e1b1nwzida0e0b0xyg
            , mec=0, mel=0, gest_propn_pa1e1b1nwzida0e0b0xyg=0, lact_propn_pa1e1b1nwzida0e0b0xyg=0):
    ##ME available for wool growth
    mew_xs_pa1e1b1nwzida0e0b0xyg = np.maximum(mew_min_pa1e1b1nwzida0e0b0xyg * relsize_start, mei - (mec * gest_propn_pa1e1b1nwzida0e0b0xyg + mel * lact_propn_pa1e1b1nwzida0e0b0xyg))
    ##Wool growth (protein weight) wo (without) lag
    d_cfw_wolag_pa1e1b1nwzida0e0b0xyg = cw[8, ...] * wge_a0e0b0xyg * af_wool_pa1e1b1nwzida0e0b0xyg * dlf_eff_pa1e1b1nwzida0e0b0xyg * mew_xs_pa1e1b1nwzida0e0b0xyg
    ##Wool growth (protein weight) wo lag
    d_cfw_pa1e1b1nwzida0e0b0xyg, d_cfw_history_m2pa1e1b1nwzida0e0b0xyg = f_history(d_cfw_history_start_m2pa1e1b1nwzida0e0b0xyg, d_cfw_wolag_pa1e1b1nwzida0e0b0xyg, days_period_pa1e1b1nwzida0e0b0xyg)
    ##Wool growth (protein weight) rolling average
    d_gfw = d_cfw_pa1e1b1nwzida0e0b0xyg / cw[3, ...]
    ##Net energy required for wool
    new = cw[1, ...] * (d_cfw_pa1e1b1nwzida0e0b0xyg - cw[2, ...] * relsize_start) / cw[3, ...]
    ##ME required for wool (above basal)
    mew = new / kw_yg
    ##Fibre diameter for the days growth
    d_fd_pa1e1b1nwzida0e0b0xyg = sfd_a0e0b0xyg * (d_cfw_pa1e1b1nwzida0e0b0xyg / d_cfw_ave_pa1e1b1nwzida0e0b0xyg) ** cw[13, ...]
    ##Surface Area
    area = cc[1, ...] * ffcfw_start ** (2/3)
    ##Daily fibre length growth
    d_fl_pa1e1b1nwzida0e0b0xyg = 400 * d_cfw_pa1e1b1nwzida0e0b0xyg / (np.pi * cw[10, ...] * cw[11, ...] * area * (d_fd_pa1e1b1nwzida0e0b0xyg / 10**6) ** 2)
    return d_cfw_pa1e1b1nwzida0e0b0xyg, d_fd_pa1e1b1nwzida0e0b0xyg, d_fl_pa1e1b1nwzida0e0b0xyg, d_cfw_history_m2pa1e1b1nwzida0e0b0xyg, mew, new



def f_foo_convert(cu3, cu4, foo, legume):
    ##Convert FOO to hand shears measurement
    foo_shears = np.max(0, np.min(foo, cu3[2] + cu3[0] * foo + cu3[1] * legume))
    ##Estimate height of pasture
    height = np.max(0, np.exp(cu3[3] + cu4[0] * foo + cu4[1] * legume + cu4[2] * foo * legume) + cu4[5] + cu4[4] * foo)
    ##Height density (height per unit FOO)
    hd = height / foo_shears
    return foo_shears, hd


def f_feedsupply(cu3, cu4, feedsupply_std_a1e1b1nwzida0e0b0xyg, paststd_foo_a1e1b1j0wzida0e0b0xyg, paststd_dmd_a1e1b1j0wzida0e0b0xyg, legume_a1e1b1nwzida0e0b0xyg, pi, pasture_stage_a1e1b1j0wzida0e0b0xyg, i_hd_scalar, i_region, i_n_pasture_stage, i_hd_std):
    ##level of pasture
    level_a1e1b1nwzida0e0b0xyg = np.trunc(np.minimum(2, feedsupply_std_a1e1b1nwzida0e0b0xyg)).astype('int') #note np.trunc rounds down to the nerest int (need to specify int type for the take along axis functin below)
    ##next level up of pasture
    next_level_a1e1b1nwzida0e0b0xyg = np.minimum(2, level_a1e1b1nwzida0e0b0xyg + 1)
    ##decimal component of feedsupply
    proportion_a1e1b1nwzida0e0b0xyg = feedsupply_std_a1e1b1nwzida0e0b0xyg % 1
    ##pasture conversion scenario
    conversion_scenario_a1e1b1j0wzida0e0b0xyg = i_region * i_n_pasture_stage + pasture_stage_a1e1b1j0wzida0e0b0xyg
    ##foo as measured
    paststd_foo_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    paststd_foo_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_foo_pa1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    foo_a1e1b1nwzida0e0b0xyg = paststd_foo_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_foo_next_a1e1b1nwzida0e0b0xyg - paststd_foo_a1e1b1nwzida0e0b0xyg)
    ##foo corrected to hand shears and estimated height
    foo, hd = f_foo_convert(cu3[..., conversion_scenario], cu4[..., conversion_scenario], foo_a1e1b1nwzida0e0b0xyg, legume_a1e1b1nwzida0e0b0xyg)
    ##height ratio                    
    hr = i_hr_scalar * hd / i_hd_std
    ##dmd
    paststd_dmd_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_a1e1b1j0wzida0e0b0xyg, level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    paststd_dmd_next_a1e1b1nwzida0e0b0xyg = np.take_along_axis(paststd_dmd_pa1e1b1j0wzida0e0b0xyg, next_level_a1e1b1nwzida0e0b0xyg, uinp.structure['i_n_pos'])
    dmd_a1e1b1nwzida0e0b0xyg = paststd_dmd_a1e1b1nwzida0e0b0xyg + proportion_a1e1b1nwzida0e0b0xyg * (paststd_dmd_next_a1e1b1nwzida0e0b0xyg - paststd_dmd_a1e1b1nwzida0e0b0xyg)
    ##proportion of PI that is offered as supp
    supp_propn_a1e1b1nwzida0e0b0xyg = proportion_a1e1b1nwzida0e0b0xyg * (feedsupply_std_a1e1b1nwzida0e0b0xyg > 2) + (feedsupply_std_a1e1b1nwzida0e0b0xyg == 4)   # the proportion of diet if the value is above 2 and equal to 1.0 if fs==4
    intake_s = pi * supp_propn_a1e1b1nwzida0e0b0xyg
    return foo_a1e1b1nwzida0e0b0xyg, hr, dmd_a1e1b1nwzida0e0b0xyg, intake_s



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


def f_chill_cs(cc, ck, ffcfw_start, rc_start, fl_start, mei, meme, mew, new, km, kg_supp, kg_fodd, mei_propn_supp, mei_propn_herb, temp_ave_pa1e1b1nwzida0e0b0xyg, temp_max_pa1e1b1nwzida0e0b0xyg, temp_min_pa1e1b1nwzida0e0b0xyg, ws_pa1e1b1nwzida0e0b0xyg, rain_pa1e1b1nwzida0e0b0xygm1, index_m0, guw	= 0, kl = 0,	mei_propn_milk	= 0, mec = 0, mel = 0, nec = 0, nel = 0, gest_propn	= 0, lact_propn = 0):
    ##Animal is below maintenance
    belowmaint = mei < (meme + mec + mel + mew)
    ##Efficiency for growth (before ECold)
    kge = f_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk)
    ##Sinusoidal variation in temp & wind
    sin_var_m0 = np.sin(2 * np.pi / 12 *(index_m0 - 3))
    ##Ambient temp (2 hourly)
    temperature_pa1e1b1nwzida0e0b0xygm0 = temp_ave_pa1e1b1nwzida0e0b0xyg[..., na] + (temp_max_pa1e1b1nwzida0e0b0xyg[..., na] - temp_min_m4pa1e1b1nwzida0e0b0xyg[..., na]) / 2 * sin_var_m0
    ##Wind velocity (2 hourly)
    wind_pa1e1b1nwzida0e0b0xygm0 = ws_pa1e1b1nwzida0e0b0xyg[..., na] * (1 + 0.35 * sin_var_m0)
    ##Proportion of sky that is clear
    sky_clear_pa1e1b1nwzida0e0b0xygm1 = 0.7 * np.exp(-0.25 * rain_pa1e1b1nwzida0e0b0xygm1)
    ##radius of animal
    radius = cc[2, ...] * ffcfw_start ** (1/3)
    ##surface area of animal
    area = cc[1, ...] * ffcfw_start ** (2/3)
    ##Impact of wet fleece on insulation
    wetflc_pa1e1b1nwzida0e0b0xygm1 = cc[5, ..., na] + (1 - cc[5, ..., na]) * np.exp(-cc[6, ..., na] * rain_pa1e1b1nwzida0e0b0xygm1 / fl_start[..., na])
    ##Insulation of air (2 hourly)
    in_air_pa1e1b1nwzida0e0b0xygm0 = radius[..., na] / (radius[..., na] + fl_start[..., na]) / (cc[7, ..., na] + cc[8, ..., na] * np.sqrt(wind_pa1e1b1nwzida0e0b0xygm0))
    ##Insulation of coat (2 hourly)
    in_coat_pa1e1b1nwzida0e0b0xygm0 = radius[..., na] * np.log((radius[..., na] + fl_start[..., na]) / radius[..., na]) / (cc[9, ..., na] - cc[10, ..., na] * np.sqrt(wind_pa1e1b1nwzida0e0b0xygm0))
    ##Insulation of  tissue
    in_tissue = cc[3, ...] * (rc_start - cc[4, ...] * (rc_start - 1))
    ##Insulation of  air + coat (2 hourly)
    in_ext_pa1e1b1nwzida0e0b0xygm0m1 = wetflc_pa1e1b1nwzida0e0b0xygm1[..., na, :] * (in_air_pa1e1b1nwzida0e0b0xygm0[..., na] + in_coat_pa1e1b1nwzida0e0b0xygm0[..., na])
    ##Impact of clear night skies on ME loss
    sky_temp_pa1e1b1nwzida0e0b0xygm0m1 = sky_clear_pa1e1b1nwzida0e0b0xygm1[..., na, :] * cc[13,..., na, na] * exp(-cc[14, ..., na, na] * np.min(0, cc[15, ..., na, na] - temperature_pa1e1b1nwzida0e0b0xygm0[..., na]) ** 2)
    ##Heat production per m2
    heat = ((mei - nec * gest_propn - nel * lact_propn - new - kge * (mei
            - (meme + mec * gest_propn + mel * lact_propn + mew))
            + cc[16, ...] * guw) / area)
    ##Lower critical temperature (2 hourly)
    temp_lc_pa1e1b1nwzida0e0b0xygm0m1 = cc[11, ..., na, na]+ cc[12, ..., na, na] - heat[..., na, na] * (in_tissue[..., na, na] + in_ext_pa1e1b1nwzida0e0b0xygm0m1) + sky_temp_pa1e1b1nwzida0e0b0xygm0m1
    ##Lower critical temperature (period)
    temp_lc_pa1e1b1nwzida0e0b0xyg = np.average(temp_lc_pa1e1b1nwzida0e0b0xygm0m1, axis = (-1,-2))
    ##Extra ME required to keep warm
    mecold_pa1e1b1nwzida0e0b0xyg = area * np.average(sfun.f_dim(temp_lc_pa1e1b1nwzida0e0b0xygm0m1, temperature_pa1e1b1nwzida0e0b0xygm0[..., na]) /(in_tissue[..., na, na] + in_ext_pa1e1b1nwzida0e0b0xygm0m1), axis = (-1,-2))
    ##ME requirement for maintenance (inc ECold)
    mem = meme + mecold_pa1e1b1nwzida0e0b0xyg
    ##Animal is below maintenance (incl ecold)
    belowmaint = mei < (mem + mec + mel + mew)
    ##Efficiency for growth (inc ECold) -different to the second line because belowmaint includes ecold
    kg = f_kg(ck, belowmaint, kl_cs, km_cs, kg_supp, kg_fodd, mei_propn_supp,
              mei_propn_herb, mei_propn_milk)
    return mem, temp_lc_pa1e1b1nwzida0e0b0xyg, kg


def f_lwc_cs(cg, rc_start, mei, mem, mew, z1f, z2f, kg, mec = 0,
              mel = 0, gest_propn = 0, lact_propn = 0):
    ##Net energy gain (based on ME)
    neg = kg * (mei - (mem + mec * gest_propn + mel_dams * lact_propn + mew))
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
    return ebg, evg, pg, fg



 

def f_sire_req(sire_propn_a1e1b1nwzida0e0b0xyg1, sire_periods_g0p8, i_sire_recovery, i_startyear, date_end_p, period_is_prejoin_a1e1b1nwzida0e0b0xyg1):
    ##Date at end of period adjusted to start year
    t_date_end_a1e1b1nwzida0e0b0xyg = date_end_p - (365 * (date_end_p.astype('datetime64[Y]').astype(int) + 1970 - i_startyear)).astype('timedelta64[D]')
    ##Date_end falls within the ram mating periods
    sire_required_a1e1b1nwzida0e0b0xygp8 = np.logical_and(t_date_end_a1e1b1nwzida0e0b0xyg[...,na] >= sire_periods_g0p8.astype('datetime64[D]') , t_date_end_a1e1b1nwzida0e0b0xyg[...,na] <= (sire_periods_g0p8.astype('datetime64[D]') + i_sire_recovery))
    ##Number of rams required per ewe (if this period is joining)
    n_sires = sire_required_a1e1b1nwzida0e0b0xygp8 * sire_propn_a1e1b1nwzida0e0b0xyg1[..., na] * period_is_prejoin_a1e1b1nwzida0e0b0xyg1[..., na]
    return n_sires


def f_comb(n,k):
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0








# def parameters():
#     """Parameter generation for the pyomo variables


#     Returns
#     -------
#     dictionaries for pyomo
#     """
# parameters = np.zeros((len(output_required),len(activities0)), dtype = 'float64')
#     # Loop through the number of variables
#     for a in activites:
#         ### create array masks  for the pyomo variable
#         ''' For each pyomo variable create a mask that represents the animals
#         The arrays can then be summed across the axes for that mask '''
#         mask = sfun.create_mask(i_activity_definition)

#         ### apply each mask to each simulation output
#         #output_required is a list of the arrays that are required as parameters
#         for n, o in enumerate(output_required):
#             parameters[n,a] = np.sum(o[mask])

# return parameters

# ''' Or to allow one function call per constraint this function could
# generate the array and then multiple functions that just return the
# required row of the array.'''


##################
#post processing #
##################




