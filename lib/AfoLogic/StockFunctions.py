"""
author: Young



"""
import datetime as dt
import numpy as np
import pandas as pd
from scipy import stats
import math
import time


# from dateutil.relativedelta import relativedelta

from . import Functions as fun
from . import FeedsupplyFunctions as fsfun
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Sensitivity as sen

na=np.newaxis


def f1_sim_periods(periods_per_year, oldest_animal, len_o):
    '''
    Define the days for the simulation periods.
    The year has 52 weeks with 7 days in a week. The extra day of the year is ignored
    All calculations are based on a day of the year rather than a date and the periods are weeks of the year
    This saves managing the difficulties associated with the extra day in the year and in leap years.

    Parameters:
    start_year = int: year to start simulation.
    periods_per_year = int: number of periods per year.
    oldest_animal = float: age of the oldest animal to be simulated (yrs).

    Returns:
    n_sim_periods: total number of sim periods
    date_start_p: array of period dates (1D periods)
    date_start_P: array of period dates that go beyond the end of the simulation (used when rounding dates to the end/start of a generator period)
    date_end_p: array of period end dates (1D periods) (date of the last day in the period)
    date_end_P: array of period end dates that go beyond the end of the simulation (used when rounding dates to the end/start of a generator period)
    index_p: index of the periods (for pyomo)
    step: days in each period

    Number of weeks is 52 and the range is 0 to 51
    '''
    n_sim_periods = int(oldest_animal * periods_per_year)
    step = 364/periods_per_year
    index_p = np.arange(n_sim_periods)
    date_start_p = index_p * step
    date_start_P = np.arange(len_o * periods_per_year) * step
    date_end_p = index_p * step + step-1 #end date is the day before the next period start date
    date_end_P = np.arange(len_o * periods_per_year) * step + step-1 #end date is the day before the next start date
    return n_sim_periods, date_start_p.astype(int), date_start_P.astype(int), date_end_p.astype(int), date_end_P.astype(int), index_p, step


def f1_period_is_(period_is, date_array, date_start_p=0, date_array2 = 0, date_end_p=0):
    '''
    Parameters
    ----------
    period_is: string - type of period is calc to return.
    date_start_p: start date of each period (must have all axis).
    date_end_p: end date of each period (must have all axis).
    date_array: array of dates of interest e.g. mating dates.
    date_array2: array of end dates used to determine if period is between.

    Returns
    -------
    period_is: boolean array shaped like the date array with the addition of the p axis. This is true if a given date from date array is within the date of a given period and false if not.

    period_is_any: 1D boolean array shape of the period dates array. True if any of the dates in the date array fall into a given period.

    period_is_between: return true if the period is between two dates (it is inclusive ie if an activity occurs during the period that period will be treated as between the two dates)
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


###################################
#input and manipulation functions #
###################################

def f1_c2g(params_c2, y, a_c2_c0, i_g3_inc, var_pos=0, condition=None, axis=0, dtype=False):
    '''
    Parameters
    ----------
    params_c2 : array - parameter array - input from excel.
    y : array - sensitivity array for genetic merit.
    a_c2_c0 :
    i_g3_inc : the offspring genotypes that are included in this trial
    var_pos : int - position of last axis when inserted into all axis.
    condition :
    axis:
    dtype :

    Returns
    -------
    param array for each genotype. Grouped by sheep group ie sire, offs, dams, yatf.

    '''

    ##these inputs are used for each param so they don't need to be passed into the function.
    i_mul_g0_c0 = sinp.stock['i_mul_g0c0']
    i_mul_g1_c0 = sinp.stock['i_mul_g1c0']
    i_mul_g2_c0 = sinp.stock['i_mul_g2c0']
    i_mul_g3_c0 = sinp.stock['i_mul_g3c0']
    i_mask_g0g3 = sinp.stock['i_mask_g0g3']
    i_mask_g1g3 = sinp.stock['i_mask_g1g3']
    i_mask_g2g3 = sinp.stock['i_mask_g2g3']
    i_mask_g3g3 = sinp.stock['i_mask_g3g3']

    ##convert params from c2 to c0
    params_c2 = params_c2.astype(float) #this is so that blank cells are converted to nan not none type because none type can't be multiplied etc
    params_c0 = params_c2[...,a_c2_c0]
    ##add y axis
    ###if y is not numpy ie was read in as an int because it was a single cell, it needs to be converted
    if type(y) == int:
        y = np.asarray([y])
    ###y is a 2d array however currently it only has one slice so it is read in as a 1d array. So need to add second slice
    if y.ndim == 1 and params_c0.ndim != 1:
        y=y[...,na]
    ###apply y mask
    y=y[...,uinp.parameters['i_mask_y']]
    params_c0 = np.multiply(params_c0[...,na,:],  y[...,na]) #na here is to account for c0 axis
    ##get axis into correct position (-2 because y & g are in correct position)
    if var_pos != None or var_pos != 0:
        extra_axes = tuple(range((var_pos + 1), -2))
    else: extra_axes = ()
    allaxis_params_c0 = np.expand_dims(params_c0, axis = extra_axes)
    ##create mask g?c0
    mask_sire_inc_g0 = np.any(i_mask_g0g3 * i_g3_inc, axis = 1)
    mask_dams_inc_g1 = np.any(i_mask_g1g3 * i_g3_inc, axis = 1)
    mask_yatf_inc_g2 = np.any(i_mask_g2g3 * i_g3_inc, axis = 1)
    mask_offs_inc_g3 = np.any(i_mask_g3g3 * i_g3_inc, axis = 1)
    ##create array with the proportion of each pure genotype required to have each actual genotype included in the analysis
    mul_sire_genotypes_g0c0 = i_mul_g0_c0[mask_sire_inc_g0]
    mul_dams_genotypes_g0c0 = i_mul_g1_c0[mask_dams_inc_g1]
    mul_yatf_genotypes_g0c0 = i_mul_g2_c0[mask_yatf_inc_g2]
    mul_offs_genotypes_g0c0 = i_mul_g3_c0[mask_offs_inc_g3]
    ##convert params from c0 to g. nansum required when the selected c0 info is not filled out ^may be an issue if params are missing and mixed breed sheep is selected because it won't catch the error
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


def f1_DSTw_adjust(propn_source_b1, cycles_source, cycles_destination, axis_b1, dry_slice=1):
    '''
    A numpy based calculation that adjusts the proportion of animals across the b1 axis from cycles_source
    to cycles_destination
    The conversion keeps the litter size constant (the proportion of single, twin & triplet) and varies the
    proportion of dry animals assuming that conception would be the same in each cycle.
    Parameters
    ----------
    propn_source_b1 : np array - The proportion of dams in each b1 slice if mated for the source number of cycles.
    cycles_source: int, optional - the number of cycles from which the input proportions have been estimated.
    cycles_destination: int, optional - the number of cycles for which the prediction is required.
    axis_b1 : int, optional - the axis that has the proportion of each litter size
    dry_slice: int, optional - the slice of axis that is the dry animals. If dry_slice is != 0, implies NM exists.

    Returns
    -------
    propn_destination_b1 : np array - Proportion of dry, single, twins & triplets.

    '''
    ##create slices for the b1 axis
    slc0 = [slice(None)] * len(propn_source_b1.shape)
    slc0[axis_b1] = slice(0,1)
    slc1 = [slice(None)] * len(propn_source_b1.shape)
    slc1[axis_b1] = slice(1,None)
    ##store the proportion in slice 0 for reset later if there is a NM slice (because NM shouldn't change with cycles)
    nm_propn = propn_source_b1[tuple(slc0)]
    ##roll the b1 axis so that the dry slice is in [0:1]
    t_source = np.roll(propn_source_b1, -dry_slice, axis=axis_b1)

    ##convert the litter size proportion from the source number of cycles to the destination number
    t_destination = np.zeros_like(propn_source_b1)
    dry_propn_source = t_source[tuple(slc0)]
    dry_propn_destination = dry_propn_source ** (cycles_destination / cycles_source)
    t_destination[tuple(slc0)] = dry_propn_destination
    t_destination[tuple(slc1)] = t_source[tuple(slc1)] * fun.f_divide((1 - dry_propn_destination), (1 - dry_propn_source))
    ##roll the b1 axis back to starting position
    propn_destination_b1 = np.roll(t_destination, dry_slice, axis=axis_b1)
    ##If the NM slice exists, reset it to starting value (default is it exists as part of the b1 axis)
    #todo check if this code does anything. Perhaps the NM slice is getting overwritten in f_conception_mu2()
    if dry_slice != 0:
        propn_destination_b1[tuple(slc0)] = nm_propn
    return propn_destination_b1


def f1_DSTw(scan_g, cycles=1):
    '''
    A numpy based calculation that returns the proportion of empty, single, twin & triplet bearing dams
    after the requested number of cycles from a scanning percentage of the calibration number of cycles (2).
    Prediction uses a polynomial formula y=intercept+ax+bx^2+cx^3+dx^4, where x is the scanning %
    The polynomial has been fitted to data measured in the Triplets project.

    Parameters
    ----------
    scan_g : np array - scanning percentage of genotypes if mated for the number of calibration cycles.
    cycles: int, optional - the number of cycles for which the prediction is required.
    Returns
    -------
    Proportion of dry, single, twins & triplets.

    '''
    calibration_cycles = 2  #The data used to calibrate the coefficients used are assumed to have been derived from mating for 2 cycles.

    ## predict the proportion of dry, single, twins & triplets if the dams were mated for the calibration period (2 cycles)
    scan_powers_s = uinp.sheep['i_scan_powers']  #scan powers are the exponential powers used in the polynomial formula ie ^0, ^1, ^2, ^3, ^4
    scan_power_gs = scan_g[...,na] ** scan_powers_s #raises scan_std to scan_powers_s ie x^0, x^1, x^2, x^3, x^4
    dstwtr_cal_gl0 = np.sum(uinp.sheep['i_scan_coeff_l0s'] * scan_power_gs[...,na,:], axis = -1) #add the coefficients and sum all the elements of the equation ie intercept+ax+bx^2+cx^3+dx^4

    ##convert the litter size proportion for the calibration period to the prediction period (the prediction period is the value of the 'cycles' argument)
    dstwtr_gl0 = f1_DSTw_adjust(dstwtr_cal_gl0, cycles_source=calibration_cycles, cycles_destination=cycles
                                , axis_b1=-1, dry_slice=0)
    # t_dstwtr_gl0 = np.zeros_like(dstwtr_cal_gl0)
    # t_dry_propn_cal_gl0 = dstwtr_cal_gl0[..., 0:1]
    # t_dry_propn_gl0 = t_dry_propn_cal_gl0 ** (cycles / calibration_cycles)
    # t_dstwtr_gl0[..., 0:1] = t_dry_propn_gl0
    # t_dstwtr_gl0[..., 1:] = dstwtr_cal_gl0[..., 1:] * (1 - t_dry_propn_gl0) / (1 - t_dry_propn_cal_gl0)

    ##set values between 0 & 1 and adjust proportion of singles so that total is 1 i.e. not using the predicted singles proportion.
    dstwtr_gl0 = np.clip(dstwtr_gl0, 0, 1)
    t_mask = [True, False, True, True]
    dstwtr_gl0[..., 1] = 1 - np.sum(dstwtr_gl0 * t_mask, axis = -1) # mask out the Singles value in the sum of the array across the l0 axis
    return dstwtr_gl0

def f1_RR_propn_logistic(RR_g, cb1, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1):
    '''
    Returns the proportion of empty, single, twin & triplet bearing dams after the requested number of cycles
    from a scanning percentage or litter size of the calibration number of cycles (2).
    The prediction is based on the logistic equation used in the LMAT reproduction function (f_conception_MU2)
    It requires calculating the roots of a cubic equation which has been fitted to the cut-offs for the
    transformed logistic equation.
    Solving the cubic and back transforming (using natural logarithm) is done in f_solve_cubic_for_logistic()
    See Moment to Moment8:pg 28 for derivation of the coefficients of the cubic equation.
    Approach is also implemented in Excel 'Components combined - latest v2.xlsx'

    Parameters
    ----------
    RR_g : np array - scanning percentage of genotypes if mated for the number of calibration cycles.
    cycles: int, optional - the number of cycles for which the prediction is required.
    Returns
    -------
    Proportion of dry, single, twins & triplets.

    '''

    ## calculate the coefficients of the cubic equation ax3 + bx2 + cx + d = 0
    ### requires some intermediate values y & z calculated from the cut-off coefficients in cb1_dams
    cut1_g = cb1[:,:,2:3,...] - cb1[:,:,1:2,...]
    cut2_g = cb1[:,:,3:4,...] - cb1[:,:,2:3,...]
    ### calculations are done with the exp of the cut-off values
    y = np.exp(cut1_g)
    z = np.exp(cut2_g)
    ### a,b,c,&d are calculated as derived
    a = (0 - RR_g) * (y**2 * z)
    b = (1 - RR_g) * (y**2 * z + y * z + y)
    c = (2 - RR_g) * (y * z + y + 1)
    d = (3 - RR_g)

    ##solve the cubic and calculate values that are the fitted values for the cutoffs (equivalent of cb1_dams[25])
    cutoff0 = fun.f_solve_cubic_for_logistic_multidim(a,b,c,d)

    ###If the repro rate was 0 then replace the cutoff in those slices with the value 10 (because they will be nan)
    ####The choice of 10 ensure that the back transformed value is close to 0 but not 0
    cutoff0[RR_g == 0] = 10

    ##calc conception propn
    cp = f1_cp_from_cutoff(cutoff0, cb1, nfoet_b1any, nyatf_b1any, b1_pos, cycles)
    return cp

def f1_LS_propn_logistic(LS_g, cb1, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1):
    '''
    Returns the proportion of empty, single, twin & triplet bearing dams after the requested number of cycles
    from a litter size.
    The prediction is based on the logistic equation used in the LMAT reproduction function (f_conception_MU2)
    It requires calculating the roots of a cubic equation which has been fitted to the cut-offs for the
    transformed logistic equation.
    Solving the cubic and back transforming (using natural logarithm) is done in f_solve_cubic_for_logistic()
    See Moment to Moment8:pg 28 for derivation of the coefficients of the cubic equation.
    Approach is also implemented in Excel 'Components combined - latest v2.xlsx'

    Parameters
    ----------
    LS_g : np array - scanning percentage of genotypes if mated for the number of calibration cycles.
    Returns
    -------
    Proportion of empty, single, twins & triplets.

    '''

    ## calculate the coefficients of the cubic equation ax3 + bx2 + cx + d = 0
    ### requires some intermediate values y & z calculated from the cut-off coefficients in cb1_dams
    cut1_g = cb1[:,:,2:3,...] - cb1[:,:,1:2,...]
    cut2_g = cb1[:,:,3:4,...] - cb1[:,:,2:3,...]
    ### calculations are done with the exp of the cut-off values
    y = np.exp(cut1_g)
    z = np.exp(cut2_g)
    ### a,b,c,&d are calculated as derived
    a = (1-LS_g)*(y**2*z)+(y*z)+y
    b = (1-LS_g)* (y**2*z + y*z +y) + 2*(y*z + y + 1)
    c = (2-LS_g)*(y*z + y + 1) +3
    d = (3-LS_g)

    ##solve the cubic and calculate values that are the fitted values for the cutoffs (equivalent of cb1_dams[25])
    cutoff0 = fun.f_solve_cubic_for_logistic_multidim(a, b, c, d)

    ###If the litter size was 0 then replace the cutoff in those slices with the value 10 (because they will be nan)
    ####The choice of 10 ensure that the back transformed value is close to 0 but not 0
    cutoff0[LS_g == 0] = 10

    ##calc conception propn
    cp = f1_cp_from_cutoff(cutoff0, cb1, nfoet_b1any, nyatf_b1any, b1_pos, cycles)
    return cp


def f1_cp_from_cutoff(cutoff0, cb1, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1):
    '''
    Calculating the conception proportion (cp) from the transformed cutoffs.

    :param cutoff0: transformed estimates of proportion empty
    :return:
    '''

    ## calculate the difference between the cut-off coefficients from cb1_dams (remembering the NM slice in cb1)
    cut1_g = cb1[:,:,2:3,...] - cb1[:,:,1:2,...]
    cut2_g = cb1[:,:,3:4,...] - cb1[:,:,2:3,...]

    ###calculate the cut-off values from the fitted value and the differences
    cutoff1 = cutoff0 + cut1_g
    cutoff2 = cutoff1 + cut2_g
    cutoff3 = cb1[:,:,4:5,...]  #this is a high number to ensure that all dams are less than or equal to the maximum number of foetuses

    boundaries = np.zeros_like(cb1)
    boundaries = fun.f_update(boundaries, cutoff0, nfoet_b1any == 0)
    boundaries = fun.f_update(boundaries, cutoff1, nfoet_b1any == 1)
    boundaries = fun.f_update(boundaries, cutoff2, nfoet_b1any == 2)
    boundaries = fun.f_update(boundaries, cutoff3, nfoet_b1any == 3)

    ##back transform (get y values (propn) based on x values (boundaries) - y=1/(1+e^-x)) to probability of having less than or equal to the number of foetuses in the corresponding b slice
    ### Note: LMAT equations predict 'less than or equal', and GrazPlan predict 'greater than or equal'
    cpl = fun.f_back_transform(boundaries)

    ##Set proportions to 0 for dams that gave birth and lost - this is required so that numbers in pp calculate correctly
    cpl *= (nfoet_b1any == nyatf_b1any)

    ##probability of a given number of foetuses (calculated from the difference in the cumulative probability)
    ##Calculate probability from cumulative probability by the difference between the array and the array
    ### values offset by one slice (difference between '<x' and '<x-1').
    ### To make the end case work requires setting cpl[NM] to 0 prior to the calculation (& max(0,calc))
    ###Define the temp array shape & populate with values from cpg (values are required for the proportion of the highest parity dams)
    slc_nm = [slice(None)] * len(cpl.shape)
    slc_nm[b1_pos] = slice(0, 1)
    cpl[tuple(slc_nm)] = 0
    cp = np.maximum(0, cpl - np.roll(cpl, 1, axis=b1_pos))

    ## Adjust the predicted proportions from the calibration number of cycles to 1 cycle (default values for the function)
    ### The prediction equations from the LMAT trial are based on mating for 2 cycles. AFO calculates for each cycle
    calibration_cycles = 2  #The data used to calibrate the coefficients used are assumed to have been derived from mating for 2 cycles.
    cp = f1_DSTw_adjust(cp, cycles_source=calibration_cycles, cycles_destination=cycles, axis_b1=b1_pos)

    return cp



def f1_btrt0(dstwtr_propn,pss,pstw,pstr): #^this function is inflexible ie if you want to add quadruplets
    '''
    Parameters
    ----------
    dstwtr_propn : np array, proportion of dams that are dry, singles, twin and triplets prior to birth.
    pss : np array, survival of single born progeny at birth.
    pstw : np array, survival of twin born progeny at birth.
    pstr : np array, survival of triplet born progeny at birth.

    Returns
    -------
    btrt_b0xyg : np array, proportion of progeny in each btrt category (e.g. 11, 22, 21 ...).
    progeny_total_xyg: np array, total number of progeny alive after birth per ewe mated

    '''
    ##progeny numbers is the number of alive progeny in each b0 slice per dam giving birth to that litter size
    ### value is the number of alive progeny in an outcome multiplied by the probability of the outcome.
    ### probability is based on survival of s, tw and tr at birth.
    progeny_numbers_b0yg = np.zeros((uinp.parameters['i_b0_len'],pss.shape[-2],pss.shape[-1]))
    progeny_numbers_b0yg[0,...] = pss
    progeny_numbers_b0yg[1,...] = 2 * pstw**2 #number of progeny surviving when there are no deaths is 2, therefore 2p^2
    progeny_numbers_b0yg[2,...] = 3 * pstr**3 #number of progeny surviving when there are no deaths is 3, therefore 3p^3
    progeny_numbers_b0yg[3,...] = 2 * pstw * (1 - pstw)  #the 2 is because it could be either progeny 1 that dies or progeny 2 that dies
    progeny_numbers_b0yg[4,...] = 2 * (3* pstr**2 * (1 - pstr))  #the 2x is because there are 2 progeny surviving in the litter and the 3x because it could be either progeny 1, 2 or 3 that dies
    progeny_numbers_b0yg[5,...] = 3* pstr * (1 - pstr)**2  #the 3x because it could be either progeny 1, 2 or 3 that survives
    ##mul progeny numbers array with number of dams giving birth to that litter size to get the number of progeny surviving per dam giving birth.
    a_nfoet_b0 = sinp.stock['a_nfoet_b1'][sinp.stock['i_mask_b0_b1']] #create association between l0 and b0
    btrt_b0yg = progeny_numbers_b0yg * dstwtr_propn[a_nfoet_b0]
    ##add singleton x axis
    btrt_b0xyg = np.expand_dims(btrt_b0yg, axis = tuple(range((uinp.parameters['i_cb0_pos'] + 1), -2))) #note i_cb0_pos refers to b0 position
    ##finally convert proportion from 'per dam' to 'per progeny'
    ###total number of progeny surviving per dam (similar to number of progeny marked)
    progeny_total_xyg = np.sum(btrt_b0xyg, axis=0)
    ###proportion of the progeny of each BTRT as a proportion of the progeny born alive
    btrt_propn_b0xyg = btrt_b0xyg / progeny_total_xyg
    return btrt_propn_b0xyg, progeny_total_xyg


def f1_btrt1(dstwtr_l0yg,pss,pstw,pstr): #^this function is inflexible ie if you want to add quadruplets
    '''
    Return proportion of progeny reared with each BTRT based on the BT proportion and lamb survival
    Used to scale the lifetime production of offspring that are the flock replacements.
    Parameters
    ----------
    dstwtr_l0yg : np array - proportion of dry, singles, twin and triplets.
    pss : np array - single survival.
    pstw : np array - twin survival.
    pstr : np array - triplet survival.

    Returns
    -------
    btrt_b1nwzida0e0b0xyg : np array
        probability of ewe with lambs in each btrt category (e.g. 11, 22, 21 ...).

    '''

    ##progeny numbers is the number of progeny in each b1 category per animal born, based on peri-natal survival of s, tw and tr.
    progeny_numbers_b1yg = np.zeros((len(sinp.stock['a_nfoet_b1']), pss.shape[-2], pss.shape[-1]))
    progeny_numbers_b1yg[2,...] = sinp.stock['a_nfoet_b1'][2] * pss
    progeny_numbers_b1yg[3,...] = sinp.stock['a_nfoet_b1'][3] * pstw**2
    progeny_numbers_b1yg[4,...] = sinp.stock['a_nfoet_b1'][4] * pstr**3
    progeny_numbers_b1yg[5,...] = sinp.stock['a_nfoet_b1'][5] * 2 * pstw * (1 - pstw)  #the 2 is because it could be either progeny 1 that dies or progeny 2 that dies
    progeny_numbers_b1yg[6,...] = sinp.stock['a_nfoet_b1'][6] * 3* pstr**2 * (1 - pstr)  # 3x because it could be either progeny 1, 2 or 3 that dies
    progeny_numbers_b1yg[7,...] = sinp.stock['a_nfoet_b1'][7] * 3* pstr * (1 - pstr)**2  #the 3x because it could be either progeny 1, 2 or 3 that survives
    ##adjust numbers to proportion of total progeny reared
    progeny_propn_b1yg = progeny_numbers_b1yg / np.sum(progeny_numbers_b1yg, axis=0, keepdims=True)
    ##mul progeny numbers array with birth type proportion to get overall btrt per progeny reared.
    btrt_b1yg = progeny_propn_b1yg * dstwtr_l0yg[sinp.stock['a_nfoet_b1'] ]
    ##add singleton x axis
    btrt_b1nwzida0e0b0xyg = np.expand_dims(btrt_b1yg, axis = tuple(range((uinp.parameters['i_cl1_pos'] + 1), -2))) #note i_cl1_pos refers to b1 position
    return btrt_b1nwzida0e0b0xyg

def f1_lsln(dstwtr_l0yg,pss,pstw,pstr): #^this function is inflexible ie if you want to add quadruplets
    '''
    Return proportion of dams with each LSLN based on the BT proportion and lamb survival
    Parameters
    ----------
    dstwtr_l0yg : np array - proportion of dry, singles, twin and triplets.

    Returns
    -------
    btrt_b1nwzida0e0b0xyg : np array
        probability of ewe with lambs in each btrt category (e.g. 11, 22, 21 ...).

    '''

    ##dam numbers is the number of dams in each b1 category per animal at birth, based on peri-natal survival of s, tw and tr.
    dam_numbers_b1yg = np.zeros((len(sinp.stock['a_nfoet_b1']), pss.shape[-2], pss.shape[-1]))
    dam_numbers_b1yg[1,...] = 1
    dam_numbers_b1yg[2,...] = pss
    dam_numbers_b1yg[3,...] = pstw**2
    dam_numbers_b1yg[4,...] = pstr**3
    dam_numbers_b1yg[5,...] = 2 * pstw * (1 - pstw)  #the 2 is because it could be either progeny 1 that dies or progeny 2 that dies
    dam_numbers_b1yg[6,...] = 3* pstr**2 * (1 - pstr)  # 3x because it could be either progeny 1, 2 or 3 that dies
    dam_numbers_b1yg[7,...] = 3* pstr * (1 - pstr)**2  #the 3x because it could be either progeny 1, 2 or 3 that survives
    dam_numbers_b1yg[8,...] = (1 - pss)
    dam_numbers_b1yg[9,...] = (1 - pstw)**2
    dam_numbers_b1yg[10,...] = (1 - pstr)**3
    ##mul progeny numbers array with birth type proportion to get overall btrt
    lsln_b1yg = dam_numbers_b1yg * dstwtr_l0yg[sinp.stock['a_nfoet_b1'] ]
    ##add singleton x axis
    lsln_b1nwzida0e0b0xyg = np.expand_dims(lsln_b1yg, axis = tuple(range((uinp.parameters['i_cl1_pos'] + 1), -2))) #note i_cl1_pos refers to b1 position
    return lsln_b1nwzida0e0b0xyg



####################
#DVP/FVP functions #
####################
def f1_fvpdvp_adj(fvp_start_fa1e1b1nwzida0e0b0xyg, fvp_type_fa1e1b1nwzida0e0b0xyg, date_weaned_ida0e0b0xyg, date_start_p,
                  other_vtype, condense_vtype, step):
    '''
    Handle dvps and fvps that fall before weaning or after the end of the generator. If the dvp/fvp falls before weaning
    or after the end of the gen for all axis then it can be removed. If it only occurs for some axis then the period
    is kept but moved to either weaning date or end of generator. Because each fvp needs to exist for all axis.
    '''
    ##handle pre weaning fvps
    ###mask any that occur before weaning (except the start fvp) and set to last date of generator and type to 0 so they are essentially ignored.
    pre_wean_fvp_mask = np.logical_and(fvp_start_fa1e1b1nwzida0e0b0xyg <= date_weaned_ida0e0b0xyg, fvp_start_fa1e1b1nwzida0e0b0xyg > date_start_p[0])
    duplicate_fvp_mask_f = np.logical_not(fun.f_reduce_skipfew(np.all, pre_wean_fvp_mask,preserveAxis=0)) #pre wean mask across all axis
    fvp_start_fa1e1b1nwzida0e0b0xyg = fvp_start_fa1e1b1nwzida0e0b0xyg[duplicate_fvp_mask_f] #remove fvps that are before weaning for all axis
    fvp_type_fa1e1b1nwzida0e0b0xyg = fvp_type_fa1e1b1nwzida0e0b0xyg[duplicate_fvp_mask_f] #remove fvps that are before weaning for all axis
    ###fvps that are before weaning for only some axis get set to weaning date plus 1 period offset if there are multiple
    pre_wean_fvp_mask = np.logical_and(fvp_start_fa1e1b1nwzida0e0b0xyg <= date_weaned_ida0e0b0xyg, fvp_start_fa1e1b1nwzida0e0b0xyg > date_start_p[0])
    new_fvp_prewean_fa1e1b1nwzida0e0b0xyg = date_weaned_ida0e0b0xyg + (np.cumsum(pre_wean_fvp_mask, axis=0)-1) * step #if multiple fvps occur at weaning they need to be incremented by 7 days
    idx_fa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, new_fvp_prewean_fa1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the new period, side=right so that if the date is already the start of a period it remains in that period.
    new_fvp_prewean_fa1e1b1nwzida0e0b0xyg = date_start_p[idx_fa1e1b1nwzida0e0b0xyg]
    fvp_start_fa1e1b1nwzida0e0b0xyg[pre_wean_fvp_mask] = new_fvp_prewean_fa1e1b1nwzida0e0b0xyg[pre_wean_fvp_mask] #fvps that occur before weaning for some axis are set to wean date plus offset if multiple fvps.
    fvp_type_fa1e1b1nwzida0e0b0xyg[pre_wean_fvp_mask] = other_vtype #fvps that occur before weaning for some axis are set to type other - so that nothing is triggered.

    ##handle post gen fvps
    ###mask any that occur on the last day of the generator.
    post_fvp_mask = fvp_start_fa1e1b1nwzida0e0b0xyg >= date_start_p[-1]
    duplicate_fvp_mask_f = np.logical_not(fun.f_reduce_skipfew(np.all, post_fvp_mask,preserveAxis=0)) #post gen mask across all axis
    fvp_start_fa1e1b1nwzida0e0b0xyg = fvp_start_fa1e1b1nwzida0e0b0xyg[duplicate_fvp_mask_f] #remove fvps that are on the last period of the generator
    fvp_type_fa1e1b1nwzida0e0b0xyg = fvp_type_fa1e1b1nwzida0e0b0xyg[duplicate_fvp_mask_f] #remove fvps that are on the last period of the generator
    ###if multiple fvps occur on the last period of the gen (only for some axis and hence aren't removed) date gets offset by 1 period.
    post_fvp_mask = fvp_start_fa1e1b1nwzida0e0b0xyg >= date_start_p[-1]
    new_fvp_post_fa1e1b1nwzida0e0b0xyg = date_start_p[-1] - (np.cumsum(post_fvp_mask, axis=0)-1) * step #if multiple fvps occur at weaning they need to be incremented by 7 days
    idx_fa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, new_fvp_post_fa1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the new period, side=right so that if the date is already the start of a period it remains in that period.
    new_fvp_post_fa1e1b1nwzida0e0b0xyg = date_start_p[idx_fa1e1b1nwzida0e0b0xyg]
    fvp_start_fa1e1b1nwzida0e0b0xyg[post_fvp_mask] = new_fvp_post_fa1e1b1nwzida0e0b0xyg[post_fvp_mask] #fvps that occur before weaning for some axis are set to wean date plus offset if multiple fvps.
    fvp_type_fa1e1b1nwzida0e0b0xyg[post_fvp_mask] = condense_vtype #set to condense type to make sure extra dvps don't cause issues with masking or feed supply

    return fvp_start_fa1e1b1nwzida0e0b0xyg, fvp_type_fa1e1b1nwzida0e0b0xyg

################
#Sim functions #
################

def f1_nv_components(paststd_foo_p6a1e1b1j0wzida0e0b0xyg, paststd_dmd_p6a1e1b1j0wzida0e0b0xyg, paststd_hf_p6a1e1b1j0wzida0e0b0xyg,
                     suppstd_p6a1e1b1nwzida0e0b0xyg, legume_p6a1e1b1nwzida0e0b0xyg, cr, cu0, zf=1):
    '''
    Generates the relationship between diet NV and, FOO & diet quality (in each feed period and weather-year).

    The function generates multiple discrete data points from which FOO & diet M/D can be predicted from diet NV by interpolation.
    This relationship is required because the same nutritive value can be achieved with various combinations of FOO and DMD.
    The combination selected affects the animal requirements because:

        #. FOO affects the energy requirement associated with walking to find feed.
        #. M/D affects the efficiency of utilising energy for maintenance and production.

    The range from lowest NV to highest NV is associated with increasing FOO & DMD inputs across the j0 axis and then
    increasing supplementary feeding from the input level up to ad-lib.

    The mei returned from this function is nv for a potential intake of 1 (nv = mei * PI). The nv is scaled
    for the actual PI outside this function.
    '''
    from scipy.interpolate import interp1d
    ##inputs
    n_pos = sinp.stock['i_n_pos']

    ##Generate the levels of FOO, DMD & Supplement for the conversion arrays.
    len_j1 = 150
    index_j1 = np.arange(len_j1) / len_j1
    propn = 2/3   #proportion of the length of j1 that is in the range of the 2 levels of j0.
    ###add upper level to sup
    max_supp_p6a1e1b1nwzida0e0b0xyg = np.ones_like(suppstd_p6a1e1b1nwzida0e0b0xyg)
    suppstd_p6a1e1b1j0wzida0e0b0xyg = np.concatenate([suppstd_p6a1e1b1nwzida0e0b0xyg, max_supp_p6a1e1b1nwzida0e0b0xyg], axis=n_pos)
    ###create the foo, dmd & supp for each level of j1
    foo_p6a1e1b1j1wzida0e0b0xyg = interp1d([0, propn], paststd_foo_p6a1e1b1j0wzida0e0b0xyg, axis=n_pos)(np.minimum(index_j1,propn))
    hf_p6a1e1b1j1wzida0e0b0xyg = interp1d([0, propn], paststd_hf_p6a1e1b1j0wzida0e0b0xyg, axis=n_pos)(np.minimum(index_j1,propn))
    dmd_p6a1e1b1j1wzida0e0b0xyg = interp1d([0, propn], paststd_dmd_p6a1e1b1j0wzida0e0b0xyg, axis=n_pos)(np.minimum(index_j1,propn))
    supp_p6a1e1b1j1wzida0e0b0xyg = interp1d([propn,1], suppstd_p6a1e1b1j0wzida0e0b0xyg, axis=n_pos)(np.maximum(index_j1,propn))

    ## calculate the M/D of diet from DMD, FOO & proportion of supplement
    past_md_p6a1e1b1j1wzida0e0b0xyg = fsfun.f1_dmd_to_md(dmd_p6a1e1b1j1wzida0e0b0xyg)

    ##relative availability - uses dams equation system in p=0
    eqn_group = 5
    eqn_system = 0 # CSIRO = 0
    if uinp.sheep['i_eqn_used_g1_q1p7'][eqn_group, 0] == eqn_system:
        ra_p6a1e1b1j1wzida0e0b0xyg = fsfun.f_ra_cs(foo_p6a1e1b1j1wzida0e0b0xyg, hf_p6a1e1b1j1wzida0e0b0xyg, cr, zf)
    eqn_system = 1 # Murdoch = 1
    if uinp.sheep['i_eqn_used_g1_q1p7'][eqn_group, 0] == eqn_system:
        ra_p6a1e1b1j1wzida0e0b0xyg = fsfun.f_ra_mu(foo_p6a1e1b1j1wzida0e0b0xyg, hf_p6a1e1b1j1wzida0e0b0xyg, zf, cu0)

    ##relative ingestibility (quality)
    eqn_group = 6
    eqn_system = 0 # CSIRO = 0
    if uinp.sheep['i_eqn_used_g1_q1p7'][eqn_group, 0] == eqn_system:
        rq_p6a1e1b1j1wzida0e0b0xyg = fsfun.f_rq_cs(dmd_p6a1e1b1j1wzida0e0b0xyg, legume_p6a1e1b1nwzida0e0b0xyg, cr, pinp.sheep['i_sf'])

    ##relative intake
    ri_p6a1e1b1j1wzida0e0b0xyg = fsfun.f_rel_intake(ra_p6a1e1b1j1wzida0e0b0xyg, rq_p6a1e1b1j1wzida0e0b0xyg, legume_p6a1e1b1nwzida0e0b0xyg, cr)

    ##mei which is nv for a PI of 1.
    mei_p6zj1 = f_intake(1, ri_p6a1e1b1j1wzida0e0b0xyg, past_md_p6a1e1b1j1wzida0e0b0xyg, False
                         , supp_p6a1e1b1j1wzida0e0b0xyg, pinp.sheep['i_md_supp'])[0] #slice the first return arg

    return mei_p6zj1, foo_p6a1e1b1j1wzida0e0b0xyg, dmd_p6a1e1b1j1wzida0e0b0xyg, supp_p6a1e1b1j1wzida0e0b0xyg


def f1_feedsupply(feedsupplyw_ta1e1b1nwzida0e0b0xyg, confinementw_ta1e1b1nwzida0e0b0xyg, nv_a1e1b1j1wzida0e0b0xyg,
                  foo_a1e1b1j1wzida0e0b0xyg, dmd_a1e1b1j1wzida0e0b0xyg, supp_a1e1b1j1wzida0e0b0xyg, pi_a1e1b1nwzida0e0b0xyg,
                  mp2=0):
    ##calc mei (mei = nv * pi)
    mei = feedsupplyw_ta1e1b1nwzida0e0b0xyg * pi_a1e1b1nwzida0e0b0xyg + mp2 #add mp2 because pi doesn't include milk.

    ##interp to calc foo, dmd and supp that correspond with given feedsupply
    ## this only works if nv, foo, dmd and supp have no active axis except j1, z & g and feedsupply has a singleton j1/n axis.
    ###test that criteria above are met - slice off g & z so they are not included because they are looped
    j1_pos = sinp.stock['i_n_pos']
    z_pos = sinp.stock['i_z_pos']
    g_pos = -1
    slc = [slice(None)] * len(nv_a1e1b1j1wzida0e0b0xyg.shape)
    slc[g_pos] = slice(0, 1)
    slc[z_pos] = slice(0, 1)
    nv_test = len(nv_a1e1b1j1wzida0e0b0xyg[tuple(slc)].ravel())==nv_a1e1b1j1wzida0e0b0xyg.shape[j1_pos]
    foo_test = len(foo_a1e1b1j1wzida0e0b0xyg[tuple(slc)].ravel())==foo_a1e1b1j1wzida0e0b0xyg.shape[j1_pos]
    dmd_test = len(dmd_a1e1b1j1wzida0e0b0xyg[tuple(slc)].ravel())==dmd_a1e1b1j1wzida0e0b0xyg.shape[j1_pos]
    supp_test = len(supp_a1e1b1j1wzida0e0b0xyg[tuple(slc)].ravel())==supp_a1e1b1j1wzida0e0b0xyg.shape[j1_pos]
    if not(nv_test and foo_test and dmd_test and supp_test):
        raise ValueError('Axis error in feed extrapolation: j1 can be the only active axis')
    ###interp - has to be done in a loop because g & z axis can be active in the nv array
    foo = np.zeros_like(feedsupplyw_ta1e1b1nwzida0e0b0xyg)
    dmd = np.zeros_like(feedsupplyw_ta1e1b1nwzida0e0b0xyg)
    supp = np.zeros_like(feedsupplyw_ta1e1b1nwzida0e0b0xyg)
    slc1 = [slice(None)] * len(feedsupplyw_ta1e1b1nwzida0e0b0xyg.shape)
    slc2 = [slice(None)] * len(nv_a1e1b1j1wzida0e0b0xyg.shape)
    slc3 = [slice(None)] * len(foo_a1e1b1j1wzida0e0b0xyg.shape) #foo, dmd & sup only have z axis not g axis
    for g in range(feedsupplyw_ta1e1b1nwzida0e0b0xyg.shape[g_pos]):
        slc1[g_pos] = slice(g, g+1)
        slc2[g_pos] = slice(g, g+1)
        for z in range(feedsupplyw_ta1e1b1nwzida0e0b0xyg.shape[z_pos]):
            slc1[z_pos] = slice(z, z+1)
            slc2[z_pos] = slice(z, z+1)
            slc3[z_pos] = slice(z, z+1)
            foo[tuple(slc1)] = np.interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].ravel(), nv_a1e1b1j1wzida0e0b0xyg[tuple(slc2)].squeeze(),
                                   foo_a1e1b1j1wzida0e0b0xyg[tuple(slc3)].squeeze()).reshape(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].shape)
            dmd[tuple(slc1)] = np.interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].ravel(), nv_a1e1b1j1wzida0e0b0xyg[tuple(slc2)].squeeze(),
                                   dmd_a1e1b1j1wzida0e0b0xyg[tuple(slc3)].squeeze()).reshape(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].shape)
            supp[tuple(slc1)] = np.interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].ravel(),nv_a1e1b1j1wzida0e0b0xyg[tuple(slc2)].squeeze(),
                                    supp_a1e1b1j1wzida0e0b0xyg[tuple(slc3)].squeeze()).reshape(feedsupplyw_ta1e1b1nwzida0e0b0xyg[tuple(slc1)].shape)
    #old method - made generator 36% slower due to looping
    # axis = sinp.stock['i_n_pos']
    # foo = fun.f_nD_interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg,nv_a1e1b1j1wzida0e0b0xyg,foo_a1e1b1j1wzida0e0b0xyg,axis)
    # dmd = fun.f_nD_interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg,nv_a1e1b1j1wzida0e0b0xyg,dmd_a1e1b1j1wzida0e0b0xyg,axis)
    # supp = fun.f_nD_interp(feedsupplyw_ta1e1b1nwzida0e0b0xyg,nv_a1e1b1j1wzida0e0b0xyg,supp_a1e1b1j1wzida0e0b0xyg,axis)

    ##if confinement then no pasture
    foo = fun.f_update(foo,0,confinementw_ta1e1b1nwzida0e0b0xyg)
    dmd = fun.f_update(dmd,0,confinementw_ta1e1b1nwzida0e0b0xyg)
    ##if confinement then all diet is made up from supp therefore scale supp accordingly
    supp = fun.f_update(supp, feedsupplyw_ta1e1b1nwzida0e0b0xyg / pinp.sheep['i_md_supp'], confinementw_ta1e1b1nwzida0e0b0xyg)

    ##supplement intake
    intake_s = pi_a1e1b1nwzida0e0b0xyg * supp
    ##ME intake from supplement
    mei_supp = intake_s * pinp.sheep['i_md_supp']
    ##ME intake of solid food
    mei_solid = mei - mp2
    ##ME intake from herbage
    mei_herb = mei_solid - mei_supp
    ##M/D of herbage
    md_herb = fsfun.f1_dmd_to_md(dmd)  # will be 0 if in confinement
    ##herb intake
    intake_f = fun.f_divide(mei_herb, md_herb) #func to stop div/0 error if confinement
    ##M/D of the diet (solids)
    md_solid = fun.f_divide(mei_solid, intake_f + intake_s) #yatf have 0 solid intake at start of life.
    ##Proportion of ME as milk
    mei_propn_milk = fun.f_divide(mp2, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals
    ##Proportion of ME as supp
    mei_propn_supp = fun.f_divide(mei_supp, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals
    ##Proportion of ME as herbage
    mei_propn_herb = fun.f_divide(mei_herb, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals

    return mei, foo, dmd, mei_solid, md_solid, md_herb, intake_f, intake_s, mei_propn_milk, mei_propn_supp, mei_propn_herb


def f1_feedsupply_adjust(attempts,feedsupply,itn):
    ##create empty array to put new feedsupply into, this is done so it doesn't have the itn axis (probably could just create from attempts array shape without last axis)
    feedsupply_new = np.zeros_like(feedsupply)
    ##which feedsupplies can be calculated using binary method - must have a negative and positive error
    binary_mask = np.nanmin(attempts[...,1], axis=-1)/np.nanmax(attempts[...,1], axis=-1) < 0 #axis -1 is the itn axis ie take the min and max error from the previous iterations
    if np.any(binary_mask):
        ##calc new feedsupply binary - take half of the two feedsupplies that have resulted in the error closest to 0. Only adds the binary result to slices that have a negative and a positive value (done using the mask created above)
        ###feedsupply with negative error that is closest to 0 - this is a little complex because applying a max function to a masked array
        mask_attempts = np.ma.masked_array(attempts[...,1],np.logical_or(attempts[...,1]>0, np.isnan(attempts[...,1]))) #np.ma has a true and false the other way around (e.g. false means keep data) therefore the <> sign is opposite to what you want
        neg_bool = np.ma.getdata(np.nanmax(mask_attempts, axis=-1, keepdims=True)==attempts[...,1]) #returns a mask that states the error that is negative but closest to 0
        neg_bool = neg_bool * binary_mask[...,na] #this just makes sure the neg mask only has a true in the same slice as the pos array (so it can be applied to the feed supply array below)
        ###feedsupply with positive error that is closest to 0 - this is a little complex because applying a max function to a masked array
        mask_attempts= np.ma.masked_array(attempts[...,1],np.logical_or(attempts[...,1]<0, np.isnan(attempts[...,1]))) #np.ma has a true and false the other way around (e.g. false means keep data) therefore the <> sign is opposite to what you want
        pos_bool=np.ma.getdata(np.nanmin(mask_attempts, axis=-1,keepdims=True)==attempts[...,1]) #returns a maks that states the error that is negative but closest to 0
        pos_bool = pos_bool * binary_mask[...,na] #this just makes sure the pos mask only has a true in the same place as the neg mask.
        ##calc feedsupply
        feedsupply_new[binary_mask] = (attempts[...,0][neg_bool] + attempts[...,0][pos_bool])/2
    ##calc feedsupply using interpolation
    ###first determine the slope, slope is always positive ie as feedsupply increases error increase because error = lwc - target and more feed means higher lwc.
    if itn==0:
        slope=pinp.sheep['i_feedsupply_slope_std']/1000 #g/hd/d/unit of feedsupply
        slope = slope*7 #convert to gen period
    else:
        ####linregress only works on 1d array and can't use apply_over_axis because needs x and y. maybe there is a better way but i looked for a while and found nothing
        slope=np.empty_like(feedsupply)
        feedsupply_all_itn = attempts[...,0]
        error_all_itn = attempts[...,1]
        for i in np.ndindex(error_all_itn.shape[:-1]): #not exactly sure how this is working but it is creating tuple of each combo of slices in each axis.
            x= feedsupply_all_itn[i] #indexing with tuple works correctly if we are interested in the last axis otherwise it doesn't work properly for some reason.ie t[(0,0)] == t[0,0,:] but t[:,(0,0)] != t[:,0,0]
            y= error_all_itn[i]
            if np.amax(x[:itn+1]) == np.amin(x[:itn+1]):
                ####if all x are the same then use the default slope - probably means nv has hit the max value.
                slope[i] = pinp.sheep['i_feedsupply_slope_std']/1000*7 #div 1000 to convert to kg and mul 7 to convert from days to gen period.
            else:
                slope[i] = stats.linregress(x[:itn+1],y[:itn+1])[0] #slice 0 to get slope
    ####change in feedsupply = error / slope. It is assumed that the most recent itn has the most accurate feedsupply
    feedsupply_new[~binary_mask] = feedsupply[~binary_mask] + ((2 * -attempts[...,itn,1]) / slope)[~binary_mask] # x 2 to overshoot then switch to binary.
    return np.minimum(20, feedsupply_new) #stop nv going above 20


def f1_rev_update(trait_name, trait_value, rev_trait_value):
    trait_idx = sinp.structuralsa['i_rev_trait_name'].tolist().index(trait_name)
    if sinp.structuralsa['i_rev_trait_inc'][trait_idx]:
        if sinp.structuralsa['i_rev_create']:
            rev_trait_value[trait_name] = trait_value.copy() #have to copy so that traits (e.g. mort) that are added to using += do not also update the rev value
        else:
            trait_value = rev_trait_value[trait_name]
    return trait_value


def f1_history(history, new_value, days_in_period):
    '''
    The idea that the f1_history is implementing is for traits that have a lag from increased nutrition to increased production.
    The representation being that the production today is an average of the non-lagged estimated production from the last x days (where x is either len_p2 or len_p3, either of which can be 1 but are expected to be 25).
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


def f_potential_intake_cs(ci, cl, srw, relsize_start, rc_start, temp_lc, temp_ave, temp_max, temp_min, rain_intake
                          , rc_birth_start=1, pi_age_y=0, lb_start=0, mp2=0, piyf=1, period_between_birthwean=1, sam_pi=1):
    '''

    :param ci:
    :param cl:
    :param srw:
    :param relsize_start:
    :param rc_start:
    :param temp_lc_dams:
    :param temp_ave:
    :param temp_max:
    :param temp_min:
    :param rain_intake:
    :param rc_birth_start:
    :param pi_age_y:
    :param lb_start:
    :param mp2:
    :param piyf:
    :param period_between_birthwean:
    :param sam_pi: sensitivity multiplier on PI. Applied as an intermediate SAM so that it can be differentially applied by age
    :return pi:
    '''
    ##Condition factor on PI
    picf= np.minimum(1, rc_start * (ci[20, ...] - rc_start) / (ci[20, ...] - 1))
    ##Lactation adjustment (RC at parturition) - only active for dams
    la = 1 + ci[15, ...] * (rc_birth_start - 1)
    ##Lactation factor on PI - dam only
    pilf = 1 + pi_age_y * la * lb_start
    ##Temperature function
    piax = np.arccos(np.clip((temp_ave - temp_lc) / (0.5 * (temp_max - temp_min)),-1,1))
    ##Temperature below the lower critical temp
    tlow = piax * (temp_lc - temp_ave) + 0.5 * np.sin(piax) * (temp_max - temp_min) / np.pi
    ##Temperature factor on PI - high temperatures
    pitf_high = 1 - ci[5, ...] * (temp_ave - ci[6, ...])
    ##Temperature factor on PI - low temperatures
    pitf_low = 1 + ci[17,...] * tlow * rain_intake
    ##Temperature factor on PI
    pitf = np.minimum(1, pitf_high) * np.maximum(1, pitf_low)
    ##Potential intake
    pi = ci[1, ...] * srw * relsize_start * (ci[2, ...] - relsize_start) * picf * pitf * pilf * sam_pi
    ##Potential intake of pasture - young at foot only. Note milk intake is not removed because PI of yatf is for solids
    pi = pi * piyf     # milk DM intake = mp2 / cl[6, ...] * cl[25, ...]
    ##Potential intake of pasture - young at foot only
    pi = pi * period_between_birthwean
    return np.maximum(0,pi)


def f_potential_intake_mu(srw):
    pi = 0.028 * srw
    return np.maximum(0,pi)


def f_intake(pi, ri, md_herb, confinement, intake_s, i_md_supp, mp2=0):
    ##Pasture intake
    intake_f = np.maximum(0, pi - intake_s) * ri * np.logical_not(confinement)
    ##ME intake from forage	
    mei_forage = 0
    ##ME intake from supplement	
    mei_supp = intake_s * i_md_supp
    ##ME intake from herbage	
    mei_herb = intake_f * md_herb
    ##ME intake of solid food	
    mei_solid = mei_forage + mei_herb + mei_supp
    ##ME intake total
    mei = mei_solid + mp2
    ##md solid (includes sup) - this is just used for the stubble generator
    md_solid = mei_solid / (intake_f + intake_s)

    ##for stub sim
    ###ME intake from supplement
    mei_supp = intake_s * pinp.sheep['i_md_supp']
    ###ME intake from herbage
    mei_herb = mei_solid - mei_supp
    ###Proportion of ME as milk
    mei_propn_milk = fun.f_divide(mp2, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals
    ###Proportion of ME as supp
    mei_propn_supp = fun.f_divide(mei_supp, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals
    ###Proportion of ME as herbage
    mei_propn_herb = fun.f_divide(mei_herb, mei) #func to stop div/0 error when some animals don't exist e.g. tol1 animals exist before tol2 animals

    return mei, mei_solid, intake_f, md_solid, mei_propn_milk, mei_propn_herb, mei_propn_supp


def f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb
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


def f_energy_cs(ck, cx, cm, lw_start, ffcfw_start, mr_age, mei, omer_history_start, days_period, md_solid, i_md_supp,
                md_herb, lgf_eff, dlf_eff, i_steepness, density, foo, confinement, intake_f, dmd, mei_propn_milk=0, sam_kg=1, sam_mr=1):
    ##Efficiency for maintenance	
    km = (ck[1, ...] + ck[2, ...] * md_solid) * (1-mei_propn_milk) + ck[3, ...] * mei_propn_milk
    ##Efficiency for lactation - dam only	
    kl =  ck[5, ...] + ck[6, ...] * md_solid
    ##Efficiency for growth (supplement) including the sensitivity scalar
    kg_supp = ck[16, ...] * i_md_supp * sam_kg
    ##Efficiency for growth (fodder) including the sensitivity scalar
    kg_fodd = ck[13, ...] * lgf_eff * (1+ ck[15, ...] * dlf_eff) * md_herb * sam_kg
    ##Energy required at maint for metabolism	
    emetab = cx[10, ...] * cm[2, ...] * ffcfw_start ** 0.75 * mr_age * (1 + cm[5, ...] * mei_propn_milk)
    ##Distance walked (horizontal equivalent)	
    distance = (1 + np.tan(np.deg2rad(i_steepness))) * np.minimum(1, cm[17, ...] / density) / (cm[8, ...] * foo + cm[9, ...])
    ##Set Distance walked to 0 if in confinement	
    distance = distance * np.logical_not(confinement)
    ##Energy required for movement	
    emove = cm[16, ...] * distance * lw_start
    ##Energy required for grazing (chewing and walking around)
    egraze = cm[6, ...] * ffcfw_start * intake_f * (cm[7, ...] - dmd) + emove
    ##Energy associated with organ activity
    omer, omer_history = f1_history(omer_history_start, cm[1, ...] * mei, days_period)
    ##ME requirement for maintenance (before ECold)
    meme = ((emetab + egraze) / km + omer) * sam_mr
    return meme, omer_history, km, kg_fodd, kg_supp, kl


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


def f1_carryforward_u1(cu1, cg, ebg, period_between_joinstartend, period_between_mated90, period_between_d90birth
                      , period_between_birthwean, days_period, period_propn=1):
    ##Select coefficient to increment the carry forward quantity based on the current period
    ### can only be the coefficient from one of the periods and the later period overwrites the earlier period.
    coeff_cf1 = fun.f_update(0, cu1[1,...], period_between_joinstartend) #note cu1 has already had the first axis (production parameter) sliced when it was passed in
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[2,...], period_between_mated90)
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[3,...], period_between_d90birth)
    coeff_cf1 = fun.f_update(coeff_cf1, cu1[4,...], period_between_birthwean)
    ##Calculate the increment (d_cf) from the coefficient, the change in LW (kg/d) and the days per period
    d_cf = coeff_cf1 * ebg * cg[18, ...] * days_period * period_propn
    return d_cf


def f_birthweight_cs(cx, w_b_yatf, w_f_dams, period_is_birth):
    ##set BW = foetal weight at end of period (if born)	
    t_w_b = w_f_dams * cx[15, ...] * period_is_birth
    ##update birth weight if it is birth period
    w_b_yatf = fun.f_update(w_b_yatf, t_w_b, period_is_birth)
    return w_b_yatf


def f_birthweight_mu(cu1, cb1, cg, cx, ce, w_b, cf_w_b_dams, ffcfw_birth_dams, ebg_dams, days_period, gest_propn
                     , period_between_joinscan, period_between_scanbirth, period_is_birth):
    ##Carry forward BW increment	
    d_cf_w_b = f1_carryforward_u1(cu1[16, ...], cg, ebg_dams, False, period_between_joinscan, period_between_scanbirth
                                 , False, days_period, gest_propn)
    ##Increment the total carry forward BW
    cf_w_b_dams = cf_w_b_dams + d_cf_w_b
    ##estimate BW by including the intercept, the effect of dam weight at birth and other non-LW coefficients
    t_w_b_yatf = (cf_w_b_dams + cu1[16, -1, ...] + cu1[16, 0, ...] * ffcfw_birth_dams + cb1[16, ...]
                  + cx[16, ...] + ce[16, ...])
    ##Update w_b if period is birth
    w_b = fun.f_update(w_b, t_w_b_yatf, period_is_birth)
    return w_b, cf_w_b_dams


def f_weanweight_cs(w_w_yatf, ffcfw_start_yatf, ebg_yatf, days_period, period_is_wean):
    ##set WWt = yatf weight at weaning
    #todo ebg needs to be multiplied by cg[18] to allow for gut fill. cg needs to be passed as an arg (or ebg converted to lwg)
    t_w_w = (ffcfw_start_yatf + ebg_yatf * days_period) + sen.saa['wean_wt']  #Note:saa[wean_wt] doesn't have an associated MEI impact.
    ##update weaning weight if it is weaning period
    w_w_yatf = fun.f_update(w_w_yatf, t_w_w, period_is_wean)
    return w_w_yatf


def f_weanweight_mu(cu1, cb1, cg, cx, ce, nyatf, w_w, cf_w_w_dams, ffcfw_wean_dams, ebg_dams, foo, foo_ave_start
                    , days_period, day_of_lactation, period_between_joinscan, period_between_scanbirth
                    , period_between_birthwean, period_is_wean):
    ##Calculate average FOO to end of this period (increment the running average to date)
    foo_ave_end = fun.f_divide(foo_ave_start * day_of_lactation + foo * days_period, day_of_lactation + days_period)
    ##Carry forward WWt increment
    d_cf_w_w = f1_carryforward_u1(cu1[17, ...], cg, ebg_dams, False, period_between_joinscan, period_between_scanbirth
                                 , period_between_birthwean, days_period)
    ##Increment the total Carry forward WWt
    cf_w_w_dams = cf_w_w_dams + d_cf_w_w
    ##add intercept, impact of dam LW at weaning, FOO, BTRT, gender and dam age effects to the carry forward value
    t_w_w = (cf_w_w_dams + cu1[17, -1, ...] + cu1[17, 0, ...] * ffcfw_wean_dams + cu1[17, 5, ...] * foo_ave_end
             + cu1[17, 6, ...] * foo_ave_end ** 2 + cb1[17, ...] + cx[17, ...] + ce[17, ...]
             + sen.saa['wean_wt']) * (nyatf > 0)  #Note:saa[wean_wt] doesn't have an associated MEI impact.
    ##Update w_w if it is weaning	
    w_w = fun.f_update(w_w, t_w_w, period_is_wean)
    return w_w, cf_w_w_dams, foo_ave_end


#todo Consider combining into 1 function f_progenyflc_mu
def f_progenycfw_mu(cu1, cg, cfw_adj, cf_cfw_dams, ffcfw_birth_dams, ffcfw_birth_std_dams, ebg_dams, days_period
                    , gest_propn, period_between_mated90, period_between_d90birth, period_is_birth):
    ##impact on progeny CFW of the dam LW profile being different from the standard pattern
    ### LTW coefficients are multiplied by the difference in the LW profile from the standard profile. This only requires representing explicitly for LW at birth because the std LW change is 0. Std pattern is lambing in CS 3, so LW = normal weight
    ##Carry forward CFW increment
    d_cf_cfw = f1_carryforward_u1(cu1[12, ...], cg, ebg_dams, False, period_between_mated90, period_between_d90birth, False, days_period, gest_propn)
    ##Increment the total Carry forward CFW
    cf_cfw_dams = cf_cfw_dams + d_cf_cfw
    ##temporary calculation including difference in current dam LW (only used if period is birth)
    ### Birth coefficient multiplied by the difference from the standard pattern rather than absolute weight
    t_cfw_yatf = (cf_cfw_dams + cu1[12, -1, ...] + cu1[12, 0, ...] * (ffcfw_birth_dams - ffcfw_birth_std_dams))
    ##Update CFW if it is birth
    cfw_adj = fun.f_update(cfw_adj, t_cfw_yatf, period_is_birth)
    return cfw_adj, cf_cfw_dams


def f_progenyfd_mu(cu1, cg, fd_adj, cf_fd_dams, ffcfw_birth_dams, ffcfw_birth_std_dams, ebg_dams, days_period
                   , gest_propn, period_between_mated90, period_between_d90birth, period_is_birth):
    ##impact on progeny FD of the dam LW profile being different from the standard pattern
    ### LTW coefficients are multiplied by the difference in the LW profile from the standard profile. This only requires representing explicitly for LW at birth because the std LW change is 0. Std pattern is lambing in CS 3, so LW = normal weight
    ##Carry forward FD increment
    d_cf_fd = f1_carryforward_u1(cu1[13, ...], cg, ebg_dams, False, period_between_mated90, period_between_d90birth, False, days_period, gest_propn)
    ##Increment the total Carry forward FD
    cf_fd_dams = cf_fd_dams + d_cf_fd
    ##temporary calculation including difference in current dam LW (only used if period is birth)
    ### Birth coefficient multiplied by the difference from the standard pattern rather than absolute weight
    t_fd_yatf = (cf_fd_dams + cu1[13, -1, ...] + cu1[13, 0, ...] * (ffcfw_birth_dams - ffcfw_birth_std_dams))
    ##Update FD if it is birth
    fd_adj = fun.f_update(fd_adj, t_fd_yatf, period_is_birth)
    return fd_adj, cf_fd_dams


def f_milk(cl, srw, relsize_start, rc_birth_start, mei, meme, mew_min, rc_start, ffcfw75_exp_yatf, lb_start, ldr_start
           , age_yatf, mp_age_y,  mp2_age_y, i_x_pos, days_period_yatf, kl, lact_nut_effect):
    ##Max milk prodn based on dam rc birth
    mpmax = srw** 0.75 * relsize_start * rc_birth_start * lb_start * mp_age_y
    ##Excess ME available for milk	
    mel_xs = np.maximum(0, (mei - (meme + mew_min * relsize_start))) * cl[5, ...] * kl
    ##Excess ME as a ratio of mpmax
    milk_ratio = fun.f_divide(mel_xs, mpmax) #func stops div0 error - and milk ratio is later discarded because days period f = 0
    ##Age or energy factor
    ad = np.maximum(age_yatf, milk_ratio / (2 * cl[22, ...]))
    ##Milk production based on energy available
    mp1 = cl[7, ...] * mpmax * fun.f_back_transform(-cl[19, ...] + cl[20, ...] * milk_ratio
                                                    + cl[21, ...] * ad * (milk_ratio - cl[22, ...] * ad)
                                                    - cl[23, ...] * rc_start * (milk_ratio - cl[24, ...] * rc_start))
    ##Milk production (per animal) based on suckling volume	(milk production per day of lactation)
    ### Based on the standard parameter values 'Suckling volume of young' is very rarely limiting milk production.
    mp2 = np.minimum(mp1, np.mean(fun.f_dynamic_slice(ffcfw75_exp_yatf, i_x_pos, 1, None), axis = i_x_pos, keepdims=True) * mp2_age_y)   # averages female and castrates weight, ffcfw75 is metabolic weight
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


def f_fibre(cw_g, cc_g, ffcfw_start_g, relsize_start_g, d_cfw_history_start_p2g, mei_g, mew_min_g, d_cfw_ave_g
            , sfd_a0e0b0xyg, wge_a0e0b0xyg, af_wool_g, dlf_wool_g,  kw_yg, days_period_g, sfw_ltwadj_g, sfd_ltwadj_g
            , rev_trait_value, mec_g1=0, mel_g1=0, gest_propn_g1=0, lact_propn_g1=0, sam_pi=1):
    ##adjust wge, cfw_ave, mew_min & sfd for the LTW adjustments (CFW is a scalar and FD is an addition)
    wge_a0e0b0xyg = wge_a0e0b0xyg * sfw_ltwadj_g
    d_cfw_ave_g = d_cfw_ave_g * sfw_ltwadj_g
    mew_min_g = mew_min_g * sfw_ltwadj_g
    sfd_a0e0b0xyg = sfd_a0e0b0xyg + sfd_ltwadj_g
    ##if passed, adjust wge by sam_pi so the intake sensitivity doesn't alter the wool growth outcome for the genotype
    ###this scaling could be applied to sfw but is applied here so that pi can be altered for a single age group
    ###which is required for the GEPEP analysis that is calibrating the adult intake and the fleece weight
    wge_a0e0b0xyg = wge_a0e0b0xyg / sam_pi
    ##ME available for wool growth
    mew_xs_g = np.maximum(mew_min_g * relsize_start_g, mei_g - (mec_g1 * gest_propn_g1 + mel_g1 * lact_propn_g1))
    ##Wool growth (protein weight-as shorn i.e. not DM) if there was no lag
    d_cfw_nolag_g = cw_g[8, ...] * wge_a0e0b0xyg * af_wool_g * dlf_wool_g * mew_xs_g
    ##Process the CFW REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    d_cfw_nolag_g = f1_rev_update('cfw', d_cfw_nolag_g, rev_trait_value)
    ##Wool growth (protein weight) with lag and updated history
    d_cfw_g, d_cfw_history_p2g = f1_history(d_cfw_history_start_p2g, d_cfw_nolag_g, days_period_g)
    ##Net energy required for wool
    new_g = cw_g[1, ...] * (d_cfw_g - cw_g[2, ...] * relsize_start_g) / cw_g[3, ...]
    ##ME required for wool (above basal growth rate)
    mew_g = new_g / kw_yg #can be negative because mem assumes 4g of wool is grown. If less is grown then mew 'returns' the energy.
    ##Fibre diameter for the days growth
    d_fd_g = sfd_a0e0b0xyg * fun.f_divide(d_cfw_g, d_cfw_ave_g) ** cw_g[13, ...]  #func to stop div/0 error when d_cfw_ave=0 so does d_cfw (only have a 0 when day period = 0)
    ##Process the FD REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    d_fd_g = f1_rev_update('fd', d_fd_g, rev_trait_value)
    ##Surface Area
    area = cc_g[1, ...] * ffcfw_start_g ** (2/3)
    ##Daily fibre length growth
    d_fl_g = 100 * fun.f_divide(d_cfw_g, cw_g[10, ...] * cw_g[11, ...] * area * np.pi * (0.5 * d_fd_g / 10**6) ** 2) #func to stop div/0 error, when d_fd==0 so does d_cfw
    return d_cfw_g, d_fd_g, d_fl_g, d_cfw_history_p2g, mew_g, new_g


def f_chill_cs(cc, ck, ffcfw_start, rc_start, sl_start, mei, meme, mew, new, km, kg_supp, kg_fodd, mei_propn_supp
               , mei_propn_herb, temp_ave_a1e1b1nwzida0e0b0xyg, temp_max_a1e1b1nwzida0e0b0xyg, temp_min_a1e1b1nwzida0e0b0xyg
               , ws_a1e1b1nwzida0e0b0xyg, rain_a1e1b1nwzida0e0b0xygp1, index_m0, guw = 0, kl = 0, mei_propn_milk = 0
               , mec = 0, mel = 0, nec = 0, nel = 0, gest_propn	= 0, lact_propn = 0):
    ##Animal is below maintenance
    belowmaint = mei < (meme + mec + mel + mew)
    ##Efficiency for growth (before ECold)
    kge = f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
    ##Sinusoidal variation in temp & wind (minimum temp at midnight (0:00 hrs) max temp at midday (12:00 hrs)
    sin_var_m0 = np.sin(2 * np.pi * (index_m0 - 6) / 24)
    ##Ambient temp (2 hourly)
    temperature_a1e1b1nwzida0e0b0xygm0 = temp_ave_a1e1b1nwzida0e0b0xyg[..., na] + (temp_max_a1e1b1nwzida0e0b0xyg[..., na] - temp_min_a1e1b1nwzida0e0b0xyg[..., na]) / 2 * sin_var_m0
    ##Wind velocity (2 hourly)
    wind_a1e1b1nwzida0e0b0xygm0 = ws_a1e1b1nwzida0e0b0xyg[..., na] * (1 + 0.35 * sin_var_m0)
    ##Proportion of sky that is clear
    sky_clear_a1e1b1nwzida0e0b0xygp1 = 0.7 * np.exp(-0.25 * rain_a1e1b1nwzida0e0b0xygp1)
    ##radius of animal
    radius = np.maximum(0.001,cc[2, ...] * ffcfw_start ** (1/3)) #max because realistic values of radius can be small for lambs - stops div0 error
    ##surface area of animal
    area = np.maximum(0.001,cc[1, ...] * ffcfw_start ** (2/3)) #max because area is in m2 so realistic values of area can be small for lambs
    ##Impact of wet fleece on insulation
    wetflc_a1e1b1nwzida0e0b0xygp1 = cc[5, ..., na] + (1 - cc[5, ..., na]) * np.exp(-cc[6, ..., na] * rain_a1e1b1nwzida0e0b0xygp1 / sl_start[..., na])
    ##Insulation of air (2 hourly)
    in_air_a1e1b1nwzida0e0b0xygm0 = radius[..., na] / (radius[..., na] + sl_start[..., na]) / (cc[7, ..., na] + cc[8, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of coat (2 hourly)
    in_coat_a1e1b1nwzida0e0b0xygm0 = radius[..., na] * np.log((radius[..., na] + sl_start[..., na]) / radius[..., na]) / (cc[9, ..., na] - cc[10, ..., na] * np.sqrt(wind_a1e1b1nwzida0e0b0xygm0))
    ##Insulation of  tissue
    in_tissue = cc[3, ...] * (rc_start - cc[4, ...] * (rc_start - 1))
    ##Insulation of  air + coat (2 hourly)
    in_ext_a1e1b1nwzida0e0b0xygm0p1 = wetflc_a1e1b1nwzida0e0b0xygp1[..., na, :] * (in_air_a1e1b1nwzida0e0b0xygm0[..., na] + in_coat_a1e1b1nwzida0e0b0xygm0[..., na])
    ##Impact of clear night skies on ME loss only during the nighttime hours of the m axis (5 slices)
    ###Note: the nighttime slices are different to Freer etal 2012 due to discrepancy in timing of the sinusoidal temperature
    night_mask_m0p1 = np.logical_or(index_m0 <= 4, index_m0 >= 20)[..., na]
    sky_temp_a1e1b1nwzida0e0b0xygm0p1 = night_mask_m0p1 * (sky_clear_a1e1b1nwzida0e0b0xygp1[..., na, :] * cc[13,..., na, na]
                                        * np.exp(-cc[14, ..., na, na]
                                                 * np.minimum(0, cc[15, ..., na, na] - temperature_a1e1b1nwzida0e0b0xygm0[..., na]) ** 2))
    ##Heat production per m2
    heat = (mei - nec * gest_propn - nel * lact_propn - new - kge * (mei
            - (meme + mec * gest_propn + mel * lact_propn + mew))
            + cc[16, ...] * guw) / area
    ##Lower critical temperature (2 hourly)
    temp_lc_a1e1b1nwzida0e0b0xygm0p1 = cc[11, ..., na, na]+ cc[12, ..., na, na] - heat[..., na, na] * (in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0p1) + sky_temp_a1e1b1nwzida0e0b0xygm0p1
    ##Lower critical temperature (period)
    temp_lc_a1e1b1nwzida0e0b0xyg = np.average(temp_lc_a1e1b1nwzida0e0b0xygm0p1, axis = (-1,-2))
    ##Extra ME required to keep warm
    mecold_a1e1b1nwzida0e0b0xyg = area * np.average(fun.f_dim(temp_lc_a1e1b1nwzida0e0b0xygm0p1, temperature_a1e1b1nwzida0e0b0xygm0[..., na]) /(in_tissue[..., na, na] + in_ext_a1e1b1nwzida0e0b0xygm0p1), axis = (-1,-2))
    ##ME requirement for maintenance (inc ECold)
    mem = meme + mecold_a1e1b1nwzida0e0b0xyg
    ##Animal is below maintenance (incl ecold)
    belowmaint = mei < (mem + mec + mel + mew)
    ##Efficiency for growth (inc ECold) -different to the second line because belowmaint includes ecold
    kg = f1_kg(ck, belowmaint, km, kg_supp, mei_propn_supp, kg_fodd, mei_propn_herb, kl, mei_propn_milk, lact_propn)
    return mem, temp_lc_a1e1b1nwzida0e0b0xyg, kg


def f_lwc_cs(cg, rc_start, mei, mem, mew, zf1, zf2, kg, rev_trait_value, mec = 0, mel = 0, gest_propn = 0, lact_propn = 0):
    ## requirement for maintenance
    maintenance = mem + mec * gest_propn + mel * lact_propn + mew
    ##Level of feeding (maint = 0)
    level = (mei / maintenance) - 1
    ##Energy intake that is surplus to maintenance
    surplus_energy = mei - maintenance
    ##Net energy gain (based on ME)
    neg = kg * surplus_energy
    ##Energy Value of gain
    evg = cg[8, ...] - zf1 * (cg[9, ...] - cg[10, ...] * (level - 1)) + zf2 * cg[11, ...] * (rc_start - 1)
    ##Protein content of gain (some uncertainty for sign associated with zf2.
    ### GrazFeed documentation had +ve however, this implies that PCG increases when BC > 1. So changed to -ve
    #todo check this equation when converting to a heat production based model.
    pcg = cg[12, ...] + zf1 * (cg[13, ...] - cg[14, ...] * (level - 1)) - zf2 * cg[15, ...] * (rc_start - 1)
    ##Empty bodyweight gain
    ebg = neg / evg
    ##Process the Liveweight REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    ebg = f1_rev_update('lwc', ebg, rev_trait_value)
    ##Protein gain
    pg = pcg * ebg
    ##fat gain
    fg = (neg - pg * cg[21, ...]) / cg[22, ...]
    return ebg, evg, pg, fg, level, surplus_energy


def f_lwc_mu(cg, rc_start, mei, mem, mew, zf1, zf2, kg, rev_trait_value, mec = 0, mel = 0, gest_propn = 0, lact_propn = 0):
    ## requirement for maintenance
    maintenance = mem + mec * gest_propn + mel * lact_propn + mew
    ##Level of feeding (maint = 0)
    level = (mei / maintenance) - 1
    ##Energy intake that is surplus to maintenance
    surplus_energy = mei - maintenance
    ##Net energy gain (based on ME)
    neg = kg * surplus_energy
    ##Energy Value of gain as calculated.
    c_evg = cg[8, ...] - zf1 * (cg[9, ...] - cg[10, ...] * (level - 1)) + zf2 * cg[11, ...] * (rc_start - 1)
    # evg = fun.f_update(evg , temporary, zf2 < 1)
    ## Scale from calculated to input evg based on zf2. If zf2 = 1 then scale based on the SAP
    evg = c_evg * (1 + sen.sap['evg_adult'] * zf2)
    ##Empty bodyweight gain
    ebg = neg / evg
    ##Process the Liveweight REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    ebg = f1_rev_update('lwc', ebg, rev_trait_value)
    # ##Protein gain
    # pg = pcg * ebg
    # ##fat gain
    # fg = (neg - pg * cg[21, ...]) / cg[22, ...]
    ## proportion of fat and lean is determined from the EVG based on energy and DM content of muscle and adipose
    adipose_propn = (evg - (cg[21, ...] * cg[19, ...])) / ((cg[22, ...] * cg[20, ...]) - (cg[21, ...] * cg[19, ...]))
    fg = ebg * adipose_propn * cg[20, ...]
    pg = (neg - fg * cg[22, ...]) / cg[21, ...]
    return ebg, evg, pg, fg, level, surplus_energy


def f_wbe(aw, mw, cg):
    ## calculate whole body energy content from weight of adipose tissue (aw) and muscle (mw), and the dry matter content and energy density.
    wbe = aw * cg[20, ...] * cg[22, ...] + mw * cg[19, ...] * cg[21, ...]
    return wbe


def f_emissions_bc(ch, intake_f, intake_s, md_solid, level):
    #todo these formulas need to be reviewed, then connected to emissions in Pyomo.
    # Compare with original Blaxter & Clapperton to check if error in the sign in Tech 2012 paper.
    # Compare the original MIDAS derivation of animal and feed components
    ##Methane production total
    ch4_total = ch[1, ...] * (intake_f + intake_s)*((ch[2, ...] + ch[3, ...] * md_solid) + (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid))
    ##Methane production animal component
    ch4_animal = ch[1, ...] * (intake_f + intake_s) * (level + 1) * (ch[4, ...] - ch[5, ...] * md_solid)
    return ch4_total, ch4_animal


# def f1_convert_fs2nv(fs_input, nv_p6f, feedsupply_f, a_p6_pa1e1b1nwzida0e0b0xyg):
#     ##convert a feed supply array (feed) and return an NV array using a conversion array (nv_p6f) for a corresponding feedsupply (feedsupply_f)
#     ## expect feed to have a p axis as axis 0.
#     ###the position of the feedsupply input in the conversion array
#     fs_col_pa1e1b1nwzida0e0b0xyg = np.searchsorted(feedsupply_f, fs_input, 'right') - 1
#     fs_col_pa1e1b1nwzida0e0b0xyg = np.maximum(0, fs_col_pa1e1b1nwzida0e0b0xyg)
#     ###the value from the conversion array in column fs_col in the row associated with the feed period for that generator period.
#     nv_pa1e1b1nwzida0e0b0xygf = nv_p6f[a_p6_pa1e1b1nwzida0e0b0xyg, :]
#     nv_pa1e1b1nwzida0e0b0xygf = np.take_along_axis(nv_pa1e1b1nwzida0e0b0xygf, fs_col_pa1e1b1nwzida0e0b0xyg[...,na], axis=-1)
#     return nv_pa1e1b1nwzida0e0b0xygf[...,0] #remove singleton f axis - no longer needed.
#
#
# def f1_convert_nv2fs(nv_input, nv_p6f, feedsupply_f, a_p6_pz):
#     ##convert a feed supply array (feed) and return an NV array using a conversion array (nv_p6f) for a corresponding feedsupply (feedsupply_f)
#     ## expect feed to have a p axis as axis 0.
#     ### multi dim search sorted requires the axes to be the same, so convert p6 to p in the lookup array
#     nv_pzf = nv_p6f[a_p6_pz, :]
#     ###the position of the feedsupply input in the conversion array
#     z_pos = sinp.stock['i_z_pos']
#     fs_col_pa1e1b1nwzida0e0b0xyg = fun.searchsort_multiple_dim(nv_pzf, nv_input, axis_a0=0, axis_v0=0, axis_a1=1, axis_v1=z_pos, side='right') - 1
#     fs_col_pa1e1b1nwzida0e0b0xyg = np.maximum(0, fs_col_pa1e1b1nwzida0e0b0xyg)
#     ###the value from the feedsupply array in column fs_col.
#     fs = feedsupply_f[fs_col_pa1e1b1nwzida0e0b0xyg]
#     return fs


def f_conception_cs(cf, cb1, relsize_mating, rc_mating, cpg_doy, nfoet_b1any, nyatf_b1any, period_is_mating, index_e1
                    , rev_trait_value, saa_rr, sam_rr):
    ''''
    Calculation of dam conception using CSIRO equation system

    Conception is the change in the numbers of animals in each slice of e & b as a proportion of the numbers
    in the NM slice (e[0]b[0]). The adjustment of the actual numbers occurs in f1_period_end_nums().
    This function calculates the change in the proportions (the total should add to 0)

    The general approach is to calculate the probability of conception greater than or equal to 1,2,3 foetuses
    Probability is calculated from a sigmoid relationship based on relative size * relative condition at joining
    The estimation of cumulative probability is scaled by a factor that varies with day of year that changes with
    litter size & latitude.
    The probability is an estimate of the number of dams carrying that number in the third trimester (or birth).
    Some dams conceive (and don't return to service) but don't carry to the third trimester
    due to abortion, this is taken into account.
    The values are altered by a sensitivity analysis on scanning percentage
    Conception (proportion of dams that are dry) and litter size (number of foetuses per pregnant dam) can be controlled for relative economic values

    :param cf:
    :param cb1: GrazPlan parameter stating the probability of conception with different number of foetuses.
    :param relsize_mating: Relative size at mating. This is a separate variable to relsize_start because mating
                           may occur mid period. Note: the e and b axis have been handled before passing in.
    :param rc_mating: Relative condition at mating. This is a separate variable to rc_start because mating
                      may occur mid period. Note: the e and b axis have been handled before passing in.
    :param cpg_doy: A scalar for the proportion of dry, single, twins & triplets based on day of the year.
    :param nfoet_b1any:
    :param nyatf_b1any:
    :param period_is_mating:
    :param index_e1:
    :param rev_trait_value:
    :param saa_rr:
    :return: Dam conception.
    '''
    if ~np.any(period_is_mating):
        conception = np.zeros_like(relsize_mating)
    else:
        b1_pos = sinp.stock['i_b1_pos']  #because used in many places in the function
        e1_pos = sinp.stock['i_e1_pos']  #because used in many places in the function

        ##back transform to probability of having greater than or equal to the number of foetuses in the corresponding b slice
        ## probability of at least a given number of foetuses including scaling for day of year
        cpg = cpg_doy * fun.f_sig(relsize_mating * rc_mating, cb1[2, ...], cb1[3, ...])
        ##Set proportions to 0 for dams that gave birth and lost - this is required so that numbers in pp calculate correctly
        cpg *= (nfoet_b1any == nyatf_b1any)

        ##Temporary array for probability of a given number of foetuses (calculated from the difference in the cumulative probability)
        ##Calculate probability from cumulative probability by the difference between the array and the array
        ### values offset by one slice (difference between '>x' and '>x+1').
        ### End cases work because GBAL have been set to 0 probability
        cp = np.maximum(0, cpg - np.roll(cpg, -1, axis=b1_pos))

        ##Apply scanning percentage sa to adjust the probability of the number of foetuses.
        ### Carried out here so that the sa affects the REV and is included in proportion of NM
        ### Achieved by calculating the impact of the sa on the scanning percentage and the change in the 'standardised'
        ### proportions of DST. Then adjusting the actual proportions of dry, singles and twins by that amount.
        ### Calculate the repro rate from the probabilities above (cp) and convert to an expected proportion of dry,
        ### singles, twins & triplets after 1 cycle
        #### convert the proportion of DST in cp to an equivalent scanning % after the calibration number of cycles.
        repro_rate = f1_convert_propn_to_2cycleRR(cp, nfoet_b1any, cycles = 1)
        ####remove singleton b1 axis by squeezing because it is replaced by the l0 axis in f1_DSTw
        repro_rate = np.squeeze(repro_rate, axis=b1_pos)
        saa_rr = np.squeeze(saa_rr, axis=b1_pos)
        sam_rr = np.squeeze(sam_rr, axis=b1_pos)
        ## apply the sa to the repro rate and convert the adjusted value to a proportion of dry, singles, twins & triplets after 1 cycle
        repro_rate_adj = fun.f_sa(repro_rate, sam_rr)
        repro_rate_adj = fun.f_sa(repro_rate_adj, saa_rr, 2, value_min=0) * (repro_rate > 0)     # only non-zero if original value was non-zero
        #### Convert the repro rate and adjusted repro rate to a 'standardised' proportion of DST after 1 cycle
        #### The proportions returned are in axis -1 and needs the slices altered (shape of l0 to b1) and moving to b1 position.
        propn_dst = np.moveaxis(f1_DSTw(repro_rate, cycles=1)[..., sinp.stock['a_nfoet_b1']], -1, b1_pos)
        propn_dst_adj = np.moveaxis(f1_DSTw(repro_rate_adj, cycles = 1)[..., sinp.stock['a_nfoet_b1']], -1, b1_pos)
        ####calculate the change in the expected proportions due to altering the scanning percentage,
        #### only apply to the conception slices - don't want to add any conception to the lambed and lost slices.
        propn_dst_change = (propn_dst_adj - propn_dst) * (nfoet_b1any==nyatf_b1any)
        ####apply the change to the original calculated proportions
        cp += propn_dst_change

        ##Process the Conception REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
        ###Conception is the proportion of dams that are dry and a change in conception is assumed to be converting
        ### a dry ewe into a single bearing ewe. It is calculated by altering the proportion of single bearing ewes b1[2].
        ### The proportion of Drys is then calculated (later) as the animals that didn't get pregnant.
        slc = [slice(None)] * len(cp.shape)
        slc[b1_pos] = slice(2,3)
        cp[tuple(slc)] = f1_rev_update('conception', cp[tuple(slc)], rev_trait_value)

        ##Process the Litter size REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
        ##The litter size REV is stored as the proportion of the pregnant dams that are single-, twin- & triplet-bearing
        ###Steps: Calculate litter size from cp, adjust litter size (if required) then recalculate cp from new litter size
        ### Calculating litter size (# of foetuses / dam pregnant) requires a mask for the pregnant dams that is the same shape as cp
        mask = nfoet_b1any.squeeze() > 0
        cp_masked = np.compress(mask, cp, b1_pos)
        ### Calculate the proportion of single-, twin- & triplet-bearing dams
        litter_propn = cp_masked / np.sum(cp_masked, b1_pos, keepdims=True)
        litter_propn = f1_rev_update('litter_size', litter_propn, rev_trait_value)
        ###calculate cp from the REV adjusted litter size. cp will only change from the original value if litter_size REV is active
        cp[:,:,:,mask,...] = litter_propn * np.sum(cp_masked, b1_pos, keepdims=True)

        ##Dams that implant (i.e. do not return to service) but don't retain to 3rd trimester/birth
        ## are added to 00 slice (b1[1:2]) so that they are removed from the NM slice.
        ###Number is based on a proportion (cf[5]) of the ewes that implant that lose their foetuses/embryos
        ###The number can't be more than the number of ewes that are not pregnant in the 3rd trimester (1 - propn_preg).
        propn_pregnant = np.sum(fun.f_dynamic_slice(cp, b1_pos, 2, None), axis=b1_pos, keepdims=True)
        slc[b1_pos] = slice(1,2)
        cp[tuple(slc)] = np.minimum((cf[5, ...] / (1 - cf[5, ...])) * propn_pregnant, 1 - propn_pregnant)

        ##If the period is mating then set conception = temporary probability array
        conception = cp * period_is_mating

        ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e1[0])
        slc = [slice(None)] * len(conception.shape)
        slc[e1_pos] = slice(0, 1)
        slc[b1_pos] = slice(0, 1)
        conception[tuple(slc)] = -np.sum(fun.f_dynamic_slice(conception, b1_pos, 1, None), axis=(e1_pos, b1_pos), keepdims=True)
    return conception


def f_conception_ltw(cf, cu0, relsize_mating, cs_mating, scan_std, doy_p, rr_doy, nfoet_b1any, nyatf_b1any, period_is_mating
                     , index_e1, rev_trait_value):
    '''
    Conception is the change in the numbers of animals in each slice of e & b as a proportion of the numbers
    in the NM slice (e[0]b[0]). The adjustment of the actual numbers occurs in f1_period_end_nums()
    This function calculates the change in the proportions (the total should add to 0)

    LTW system: The general calculation is scanning percentage is defined by a linear function of CS
    The standard value (CS 3) is determined by the genotype, relative size and adjusted based on day of the year.
    The slope with which RR is adjusted if CS is different from CS3 varies with day of year.
    The proportion of dry, single, twin & triplet is estimated as a function of the scanning percentage using f1_DSTw
    The proportion of ewes that implant but don't retain to the third trimester is accounted for.
    Note: sa is not applied in this function because it is applied to the input scan_std (which is also
    used to determine the BTRT effect on fleece)

    :param relsize_mating: Relative size at mating. This is a separate variable to relsize_start because mating
                           may occur mid period. Note: the e and b axis have been handled before passing in.
    :param cs_mating: Condition score at mating. Note: the e and b axis have been handled before passing in.
    :param rr_doy: A scalar for reproductive rate based on day of the year. Based on GrazPlan cpg_doy relationship
'''
    if ~np.any(period_is_mating):
        conception = np.zeros_like(relsize_mating)
    else:
        b1_pos = sinp.stock['i_b1_pos']  #because used in many places in the function
        e1_pos = sinp.stock['i_e1_pos']  #because used in many places in the function

        ##Adjust standard scanning percentage based on relative size (to reduce scanning percentage of younger animals)
        scan_std = scan_std * relsize_mating * rr_doy
        ##Slope of the RR vs CS relationship based on time of the year
        slope = np.maximum(cu0[4, ...], cu0[2, ...] + np.sin(2 * np.pi * doy_p / 364) * cu0[3, ...])
        ##Reproduction rate for dams as if mated for the number of cycles in the calibration data.
        repro_rate = scan_std + (cs_mating - 3) * slope

        ##Calculate the propn dry/single/twin for given repro rate.
        ###remove singleton b1 axis by squeezing because it is replaced by the l0 axis in f1_DSTw)
        repro_rate = np.squeeze(repro_rate, axis=b1_pos)
        ### Note: repro rate calculated above is based on a calibration with 2 cycles
        ### Require the proportions of dry, singles, twins & triplets for 1 cycle
        ### The proportions returned are in axis -1 and needs the slices altered (shape of l0 to b1) and moving to b1 position.
        cp = np.moveaxis(f1_DSTw(repro_rate, cycles = 1)[...,sinp.stock['a_nfoet_b1']], -1, b1_pos)
        ##Set proportions to 0 for dams that gave birth and lost - this is required so that numbers in pp behave correctly
        cp *= (nfoet_b1any == nyatf_b1any)
        ##Dams that implant (i.e. do not return to service) but don't retain to 3rd trimester are added to 00 slice rather than staying in NM slice
        ###Number is based on a proportion (cf[5]) of the ewes that implant (propn_preg) losing their foetuses/embryos
        ###The number can't be more than the number of ewes that are not pregnant in the 3rd trimester (1 - propn_preg).
        propn_pregnant = np.sum(fun.f_dynamic_slice(cp, b1_pos, 2, None), axis=b1_pos, keepdims=True)
        slc = [slice(None)] * len(cp.shape)
        slc[b1_pos] = slice(1,2)
        cp[tuple(slc)] = np.minimum((cf[5, ...] / (1 - cf[5, ...])) * propn_pregnant, 1 - propn_pregnant)
        ##If the period is mating then set conception = temporary probability array
        conception = cp * period_is_mating
        ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e1[0])
        slc = [slice(None)] * len(conception.shape)
        slc[e1_pos] = slice(0, 1)
        slc[b1_pos] = slice(0, 1)
        conception[tuple(slc)] = -np.sum(fun.f_dynamic_slice(conception, b1_pos, 1, None), axis=(e1_pos, b1_pos), keepdims=True)
        ##Process the Conception & Litter size REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
        conception[:, :, 1, ...] = f1_rev_update('conception', conception[:, :, 1, ...], rev_trait_value)
        conception[:, :, 2:, ...] = f1_rev_update('litter_size', conception[:, :, 2:, ...], rev_trait_value)
    return conception


def f_conception_mu2(cf, cb1, cu2, srw, maternallw_mating, lwc, age, nlb, cpl_doy, nfoet_b1any, nyatf_b1any
                      , period_is_mating, index_e1, rev_trait_value, saa_rr, sam_rr, saa_ls, saa_con, saa_preg_increment):
    ''''
    Calculation of dam conception using a back transformed logistic function. Using coefficients developed in
    Murdoch University trials.

    Conception is the change in the numbers of animals in each slice of e & b as a proportion of the numbers
    in the NM slice (e[0]b[0]). The adjustment of the actual numbers occurs in f1_period_end_nums()
    This function calculates the change in the proportions (the total should add to 0)

    The general approach is to calculate the probability of conception less than or equal to 1,2,3 foetuses
    similar to f_conception_cs() except LMAT is "less than" and probability is calculated from a back transformed
    calculation with linear and quadratic terms.
    The calculation includes terms for LW at joining, Age at joining and LW change during joining.
    The estimation of cumulative probability as fitted in the LMAT trial is scaled by a day of year factor, this
    is derived from the factor used for the GrazPlan adjustment but requires calculating in this function.
    The probability is an estimate of the number of dams carrying that number of young to birth if mated for the
    number of cycles assessed in the trial. The parameters could be altered to represent a single cycle however
    this correction hasn't been made (as of Apr 2022) and the adjustment is made in this function.
    Some dams conceive (and don't return to service) but don't carry to birth (the third trimester)
    due to abortion during pregnancy, this is taken into account.
    #todo The conversion of the prediction from 2 cycles back to one cycle doesn't include this loss
    # which then increases the proportion of empty ewes and reduces the expected RR.
    # The correction has been removed for now.
    The values are altered by a sensitivity analysis on scanning percentage
    Conception (proportion of dams that are dry) and litter size (number of foetuses per pregnant dam) can
    be controlled for relative economic values

    :param saa_ls:
    :param cf: Includes parameter for number of ewes that implant but don't retain to birth (the 3rd trimester).
    :param cb1: GrazPlan parameter stating the probability of conception with different number of foetuses.
    :param cu2: LMAT parameters controlling impact of LWJ, LWC during joining, NLB, Age at joining
    :param srw: Standard reference weight of the dam genotype, not including the adjustment assoicated with BTRT
    :param maternallw_mating: Maternal LW at mating. Allows that mating may occur mid-period.
                           Note: the e and b axis have been handled before passing in.
    :param lwc: Liveweight change of the dam during the generator period in g/hd/d.
    :param age: age of dam mid-period in days. Indexed for p. The i axis can be non-singleton
    :param nlb: Number of lambs born ASBV - mid-parent average (achieved by using nlb_g3).
    :param cpl_doy: The sine of the day of joining. This represents seasonality of reproduction
    :param nfoet_b1any:
    :param nyatf_b1any:
    :param period_is_mating:
    :param index_e1:
    :param rev_trait_value:
    :param saa_rr:
    :param sam_rr: combine sam on reproductive rate
    :return: Dam conception.
    '''
    if ~np.any(period_is_mating):
        conception = np.zeros_like(maternallw_mating)
    else:
        b1_pos = sinp.stock['i_b1_pos']  #because used in many places in the function
        e1_pos = sinp.stock['i_e1_pos']
        ##Select slice 24 (Ewe Lamb coefficients) or 25 (mature ewe coefficients) of cb1 & cu2 based on age of the dam
        cb1_sliced = fun.f_update(cb1[25, ...], cb1[24, ...], age < 364)
        cu2_sliced = fun.f_update(cu2[25, ...], cu2[24, ...], age < 364)
        ##Calculate the transformed estimates of proportion empty (slice cu2 allowing for active i axis)
        cutoff0 = cb1_sliced[:,:,1:2,...] + cu2_sliced[-1, ...] + (cu2_sliced[0, ...] * maternallw_mating
                                                                + cu2_sliced[1, ...] * maternallw_mating ** 2
                                                                + cu2_sliced[2, ...] * age
                                                                + cu2_sliced[3, ...] * age ** 2
                                                                + cu2_sliced[4, ...] * lwc
                                                                + cu2_sliced[5, ...] * lwc ** 2
                                                                + cu2_sliced[6, ...] * nlb
                                                                + cu2_sliced[7, ...] * nlb ** 2
                                                                + cu2_sliced[8, ...] * srw
                                                                + cu2_sliced[9, ...] * srw ** 2
                                                                + cu2_sliced[10, ...] * cpl_doy
                                                                + cu2_sliced[11, ...] * cpl_doy ** 2
                                                                  )

        ##calc conception propn
        cp = f1_cp_from_cutoff(cutoff0, cb1_sliced, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1)

        ##The following code is only required to apply SA and REV. A bit complicated because SA are for the entire repro period therefore need to adjust for one cycle

        ##Apply scanning percentage sa to adjust the probability of the number of foetuses using logistic function.
        ### Carried out here so that the sa affects the REV and is included in proportion of NM
        ### Calculate the RR to allow SA to be applied. Note: SA applied on basis of 2 cycles
        repro_rate = f1_convert_propn_to_2cycleRR(cp, nfoet_b1any, cycles = 1)
        #### apply the sa to the repro rate
        repro_rate_adj = fun.f_sa(repro_rate, sam_rr)
        repro_rate_adj = fun.f_sa(repro_rate_adj, saa_rr * (repro_rate > 0), 2, value_min=0)     # only adjust if original value was non-zero
        #### Back calculate the proportion of empty, single, twins & triplets using the Logistic function
        cp = f1_RR_propn_logistic(repro_rate_adj, cb1_sliced, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1)

        ##Set up slices and store values for the SA
        ###define empty slice (LSLN 00) and store proportion empty
        slc_empty = [slice(None)] * len(cp.shape)
        slc_empty[b1_pos] = slice(1, 2)
        empty = cp[tuple(slc_empty)]
        ###define slices for litter size from singles (LSLN 11) to the end
        slc_preg = [slice(None)] * len(cp.shape)
        slc_preg[b1_pos] = slice(2,None)

        ##Apply litter size sa to adjust the probability of the number of foetuses using logistic function.
        ### Carried out here prior to conception saa so that the proportions are still consistent with the logistic function
        ### Calculate the LS to allow SA to be applied. Note: SA applied on basis of 2 cycles
        litter_size = f1_convert_propn_to_LS(cp, nfoet_b1any)
        #### apply the sa to the repro rate
        litter_size_adj = fun.f_sa(litter_size, saa_ls * (litter_size > 0), 2, value_min=0)     # only adjust if original value was non-zero
        #### Back calculate the proportion of empty, single, twins & triplets using the Logistic function for the increased LS
        cp = f1_LS_propn_logistic(litter_size_adj, cb1_sliced, nfoet_b1any, nyatf_b1any, b1_pos, cycles=1)
        #### Scale the proportions so that the proportion of empty is same as prior to SA and the total is 1
        empty_adj = cp[tuple(slc_empty)]
        cp[tuple(slc_preg)] = cp[tuple(slc_preg)] * (1 - empty) / (1 - empty_adj)
        cp[tuple(slc_empty)] = empty

        ##Apply conception saa to adjust the proportion of ewes that are dry with constant litter size.
        ### Carried out here so that the sa affects the REV and is included in proportion of NM
        ### Apply the saa to the empty slice.
        cp[tuple(slc_empty)] = fun.f_sa(empty, saa_con * (empty > 0), 2, value_min=0)     # only adjust if original value was non-zero
        #### Adjust the pregnant slices to keep litter size constant and allow for the change in the number pregnant
        cp[tuple(slc_preg)] = cp[tuple(slc_preg)] * (1-cp[tuple(slc_empty)]) / (1-empty)

        ##Apply preg increment saa to increment an individual b1 slice at conception, so that the value of an extra lamb conceived of a given birth type can be calculate.
        ###Scale the increment by the proportion of dams that got pregnant this cycle
        ###This means the overall number of dams that are incremented depends on the proportion pregnant after the total number of mating cycles.
        saa_preg_increment = saa_preg_increment * (1 - cp[tuple(slc_empty)])
        ###Calculate the number available to increment relative to the saa value
        ###The dams are being moved from the previous slice to the current slice eg ##preg_increment_b1[4] is moving a dam from slice 3 (twins) to slice 4 (triplets)
        ###dams can only be moved if they exist in the previous slice.
        saa_preg_increment = np.minimum(saa_preg_increment, np.roll(cp, 1, axis = b1_pos))
        ###increment the target slice and reduce the source slice
        cp = cp + saa_preg_increment - np.roll(saa_preg_increment, -1, axis=b1_pos)

        ##Process the Conception REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
        ###Conception is the proportion of dams that are dry and a change in conception is assumed to be converting
        ### a dry ewe into a pregnant ewe with litter size as per rest of the flock. It is calculated by altering the proportion of empty ewes b1[1].
        empty_rev = f1_rev_update('conception', cp[tuple(slc_empty)], rev_trait_value)
        cp[tuple(slc_preg)] = cp[tuple(slc_preg)] * (1 - empty_rev) / (1 - empty)
        cp[tuple(slc_empty)] = empty_rev

        ##Process the Litter size REV: either save the trait value to the dictionary or over-write trait value with value from the dictionary
        ##The litter size REV is stored as the proportion of the pregnant dams that are single-, twin- & triplet-bearing
        ###Steps: Calculate litter size from cp, adjust litter size (if required) then recalculate cp from new litter size
        ### Calculating litter size (# of foetuses / dam pregnant) requires a mask for the pregnant dams that is the same shape as cp
        mask = nfoet_b1any.squeeze() > 0
        cp_masked = np.compress(mask, cp, b1_pos)
        ### Calculate the proportion of single-, twin- & triplet-bearing dams
        litter_propn = fun.f_divide(cp_masked, np.sum(cp_masked, b1_pos, keepdims=True))
        litter_propn = f1_rev_update('litter_size', litter_propn, rev_trait_value)
        ###calculate cp from the REV adjusted litter size. cp will only change from the original value if litter_size REV is active
        cp[:,:,:,mask,...] = litter_propn * np.sum(cp_masked, b1_pos, keepdims=True)

        ##Dams that implant (i.e. do not return to service) but don't retain to 3rd trimester/birth
        ## are added to 00 slice (b1[1:2]) so that they are removed from the NM slice.
        ###Number is based on a proportion (cf[5]) of the ewes that implant (propn_preg) losing their foetuses/embryos
        ###The number can't be more than the number of ewes that are not pregnant in the 3rd trimester (1 - propn_preg).
        ### This has been disabled because the prediction of 2 cycles back to one cycle is
        ### not representing this source of dry ewes.
        ### Disabled by setting the value of propn_pregnant to 0 rather remove the code because the proportion
        ###of empty needs to be set to 0 anyway (and the problem may be fixable)
        propn_pregnant = np.sum(fun.f_dynamic_slice(cp, b1_pos, 2, None), axis=b1_pos, keepdims=True) * 0
        slc = [slice(None)] * len(propn_pregnant.shape)
        slc[b1_pos] = slice(1,2)
        cp[tuple(slc)] = np.minimum((cf[5, ...] / (1 - cf[5, ...])) * propn_pregnant, 1 - propn_pregnant)

        ##If the period is mating then set conception = temporary probability array
        conception = cp * period_is_mating

        ##Subtract conception of 00, 11, 22 & 33 from the NM slice (in e1[0])
        slc = [slice(None)] * len(conception.shape)
        slc[e1_pos] = slice(0, 1)
        slc[b1_pos] = slice(0, 1)
        conception[tuple(slc)] = -np.sum(fun.f_dynamic_slice(conception, b1_pos, 1, None), axis=(e1_pos, b1_pos), keepdims=True)
    return conception


def f1_convert_propn_to_2cycleRR(dst_propn, nfoet_b1any, cycles = 1):
    '''
    Convert from a proportion of empty, singles, twins and triplets from specified number of cycles (usually 1)
    to an equivalent scanning percentage (repro rate) if mated for the calibration number of cycles (usually 2).
    Assumes that the litter size is constant and the factor that changes is the proportion of dams that are dry

    Parameters
    ----------
    dst_propn : np array - the proportion of dams in each b1 slice (NM, empty, single, twin, triplet ...).
    nfoet_b1any: np array - the number of foetuses in each b1 slice
    cycles: int, optional - the number of cycles from which the dst_propn has resulted (usually 1).
    Returns
    -------
    Reproductive rate (scanning percentage) if joined for 2 cycles

    '''
    ##The number of cycles for which the equivalent scanning percentage is required
    calibration_cycles = 2

    ##scanning percentage for the specified number of cycles
    repro_rate = np.sum(dst_propn * nfoet_b1any, axis = sinp.stock['i_b1_pos'], keepdims = True)

    ##convert number of cycles by scaling the proportion of drys (holding litter size constant)
    dry_propn = fun.f_dynamic_slice(dst_propn, sinp.stock['i_b1_pos'], 1, 2)
    dry_propn_cal = dry_propn ** (calibration_cycles / cycles)
    litter_size = fun.f_divide(repro_rate, 1 - dry_propn)
    repro_rate_cal = litter_size * (1 - dry_propn_cal)

    return repro_rate_cal


def f1_convert_propn_to_LS(dst_propn, nfoet_b1any):
    '''
    Convert from a proportion of singles, twins and triplets to an equivalent litter size
    Note: Litter size is independent of number of cycles

    Parameters
    ----------
    dst_propn : np array - the proportion of dams in each b1 slice (NM, empty, single, twin, triplet ...).
    nfoet_b1any: np array - the number of foetuses in each b1 slice
    Returns
    -------
    Litter size

    '''
    ##calculate litter size from repro rate and proportion empty
    repro_rate = np.sum(dst_propn * nfoet_b1any, axis = sinp.stock['i_b1_pos'], keepdims = True)
    empty_propn = fun.f_dynamic_slice(dst_propn, sinp.stock['i_b1_pos'], 1, 2)
    litter_size = fun.f_divide(repro_rate, 1 - empty_propn)

    return litter_size


def f_sire_req(sire_propn_a1e1b1nwzida0e0b0xyg1g0, sire_periods_g0p8, i_sire_recovery, date_end_p, period_is_prejoin_a1e1b1nwzida0e0b0xyg1):
    ##Date at end of period adjusted to start year
    t_date_end_a1e1b1nwzida0e0b0xyg = date_end_p % 364
    ##Date_end falls within the ram mating periods
    sire_required_a1e1b1nwzida0e0b0xyg1g0p8 = np.logical_and(t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] >= sire_periods_g0p8,
                                                             t_date_end_a1e1b1nwzida0e0b0xyg[...,na,na] <= (sire_periods_g0p8 + i_sire_recovery)) #add axis for p8 and g1
    ##Number of rams required per ewe (if this period is joining)
    n_sires = sire_required_a1e1b1nwzida0e0b0xyg1g0p8 * sire_propn_a1e1b1nwzida0e0b0xyg1g0[..., na] * period_is_prejoin_a1e1b1nwzida0e0b0xyg1[..., na,na] #add axis for g1 and p8
    return n_sires


##################
#Mortality CSIRO #
##################
'''The CSIRO system includes 
        1.  a base mortality for all animal classes which is a non reducible amount plus an increment
            The increment is a fixed value and occurs if the animals is below a threshold RC and the rate of LWC is below 20% of the normal weight gain
        2. a weaner mortality increment if the animal is less than 364 days old and the rate of LWC is less than 20% of normal weight gain
        3. progeny mortality that is the sum of
            a. mortality due to exposure at birth (mortalityx) that is a function of ewe RC at birth and the chill index at birth
            b. mortality due to difficult birth (mortalityd - dystocia) that depends on the lamb birth weight and ewe relative condition at birth
        4. dam mortality that is the sum of
            a. mortality due to preg toxemia in the last 6 weeks of pregnancy. This occurs for multiple bearing dams and is affected by rate of LW loss.
               This is calculated each week and the mortality is summed, rather than calculated from the sum of the LW change.
            b. mortality due to dystocia (calculated in f_mortality_progeny_cs). It is assumed that ewe death is associated with a fixed proportion of the lambs deaths from dystocia.
            '''
def f_mortality_base_cs(cd, cg, rc_start, cv_weight, ebg_start, sd_ebg, d_nw_max, days_period, rev_trait_value, sap_mortalityb):
    ## a minimum level of mortality per day that is increased if RC is below a threshold and LWG is below a threshold
    ### i.e. increased mortality only for thin animals that are growing slowly (< 20% of normal growth rate)
    ###distribution on ebg & rc_start, calculate mort and then average (axis =-1,-2)
    ebg_start_p0p0 = fun.f_distribution7(ebg_start, sd=sd_ebg)[...,na]
    rc_start_p0p0 = fun.f_distribution7(rc_start, cv=cv_weight)[...,na,:]
    mortalityb_p0p0 = (cd[1, ...,na,na] + cd[2, ...,na,na] *
                     np.maximum(0, cd[3, ...,na,na] - rc_start_p0p0) *
                     ((cd[16, ...,na,na] * d_nw_max[...,na,na]) > (ebg_start_p0p0 * cg[18, ...,na,na]))) * days_period[...,na,na] #mul by days period to convert from mort per day to per period
    ###average p1 axis
    mortalityb = np.mean(mortalityb_p0p0, axis=(-1,-2))
    ##apply sensitivity
    mortalityb = fun.f_sa(mortalityb, sap_mortalityb, sa_type = 1, value_min = 0)
    ##Process the Mortality REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    mortalityb = f1_rev_update('mortality', mortalityb, rev_trait_value)
    return mortalityb


def f_mortality_weaner_cs(cd, cg, age, ebg_start, sd_ebg, d_nw_max,days_period):
    ## mortality increases (cd[13]) for slow growing young animals (< 20% of normal growth rate).
    ### mortality does not increase with severity of under-nutrition, simply a switch based on growth rate
    ### the mortality increment varies with age. Full increment below 300 days (cd[14]) and ramping down to 0 at 364 days (cd[15])
    ###distribution on ebg - add distribution to ebg_start_p1 and then average (axis =-1)
    ebg_start_p1 = fun.f_distribution7(ebg_start, sd=sd_ebg)
    mort_weaner_p1 = cd[13, ...,na] * fun.f_ramp(age[...,na], cd[15, ...,na], cd[14, ...,na]
                                                ) * ((cd[16, ...,na] * d_nw_max[...,na])
                                                      > (ebg_start_p1 * cg[18, ...,na])) * days_period[...,na] #mul by days period to convert from mort per day to per period
    return np.mean(mort_weaner_p1, axis=-1)


def f_mortality_dam_cs():
    '''
    Peri natal (at birth) Dam mortality.
    Currently, the CSIRO system includes dam mortality due to PregTox (f_mortality_pregtox_cs) and dam mortality due to
    dystocia (included in f_mortality_progeny_cs) but there is no calculation of deaths from other causes,
    such as those that might affect twin & triplet bearing dams at birth.
    '''
    return 0


def f_mortality_pregtox_cs(cb1, cg, nw_start, ebg, sd_ebg, days_period, period_is_pregtox, gest_propn, saa_mortalitye):
    '''
    (Twin) Dam mortality in last 6 weeks (preg tox). This increments mortality associated with LWL in the base mortality function.

    Preg tox is short for pregnancy toxaemia. It is associated with ketosis where the ewe switches into burning fat
    (rather than carbohydrates) because they are losing weight. It is predominantly a problem for multiple bearing ewes
    because they have the highest energy demands close to lambing and the least capacity to eat more (because stomach
    capacity is restricted due to the volume of the conceptus). It is usually also more of a problem for ewes
    that start out in better condition.
    '''
    ###distribution on ebg - add distribution to ebg_start_p1 and then average (axis =-1)
    ebg_p1 = fun.f_distribution7(ebg, sd=sd_ebg)
    t_mort_p1 = days_period[..., na] * gest_propn[..., na] / 42 * fun.f_sig(-42 * ebg_p1 * cg[18, ..., na] / nw_start[..., na]
                                                                        , cb1[4, ..., na], cb1[5, ..., na]) #mul by days period to convert from mort per day to per period
    t_mort = np.mean(t_mort_p1, axis=-1)
    ##If not last 6 weeks then = 0
    mort = t_mort * period_is_pregtox
    ##Adjust by sensitivity on dam mortality - need to include periods_is so the saa only gets applied in one period.
    mort = fun.f_sa(mort, saa_mortalitye * period_is_pregtox, sa_type = 2, value_min = 0)
    return mort


def f_mortality_progeny_cs(cd, cb1, w_b, rc_birth, cv_weight, w_b_exp_y, period_is_birth, chill_index_p1, nfoet_b1
                           , rev_trait_value, sap_mortalityp, saa_mortalityx):
    '''Progeny losses due to large progeny or slow birth process (dystocia)

    Dystocia definition is a difficult birth that leads to brain damage, which can be due to physical trauma
    but also lack of oxygen. The difficult birth can be a larger single lamb but it is also quite prevalent
    in twins not due to large lambs but due to a slow birth because the ewe is lacking energy to push.
    '''
    ###distribution on w_b & rc_birth - add distribution to ebg_start_p1 and then average (axis =-1)
    w_b_p1p2 = fun.f_distribution7(w_b, cv=cv_weight)[...,na]
    rc_birth_p1p2 = fun.f_distribution7(rc_birth, cv=cv_weight)[...,na,:]
    mortalityd_yatf_p1p2 = fun.f_sig(fun.f_divide(w_b_p1p2, w_b_exp_y[...,na,na]) * np.maximum(1, rc_birth_p1p2)
                                     , cb1[6, ...,na,na], cb1[7, ...,na,na]) * period_is_birth[...,na,na]
    mortalityd_yatf = np.mean(mortalityd_yatf_p1p2, axis=(-1,-2))
    ##add sensitivity
    mortalityd_yatf = fun.f_sa(mortalityd_yatf, sap_mortalityp, sa_type = 1, value_min = 0)
    ##dam mort due to large progeny or lack of energy at birth (dystocia) - returns 0 mort if there is 0 nfoet also the fact that more prog die per dam when the dams has multiple nfoet (e.g. for a trip only one ewe dies for every 3 yatf)
    mortalityd_dams = fun.f_divide(np.mean(mortalityd_yatf, axis=sinp.stock['i_x_pos'], keepdims=True) * cd[21,...], nfoet_b1)
    ##Reduce progeny losses due to large progeny (dystocia) - so not double counting progeny losses associated with dam mortality
    mortalityd_yatf = mortalityd_yatf * (1- cd[21,...])
    ##Exposure index
    xo_p1p2 = (cd[8, ..., na,na] - cd[9, ..., na,na] * rc_birth_p1p2 + cd[10, ..., na,na] * chill_index_p1[..., na]
               + cb1[10, ..., na,na])  #Note: in CSIRO equations cb1 is slice [11] but the coefficient has been changed
    ##Progeny mortality at birth from exposure
    mortalityx = np.average(fun.f_back_transform(xo_p1p2), axis=(-1, -2)) * period_is_birth  #axis -1 & -2 are p1 & p2
    ##Apply SA to progeny mortality due to exposure
    mortalityx = fun.f_sa(mortalityx, sap_mortalityp, sa_type = 1, value_min = 0)
    mortalityx = fun.f_sa(mortalityx, saa_mortalityx, sa_type = 2, value_min = 0)
    ##Process the Ewe Rearing Ability REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    mortalityx = f1_rev_update('era', mortalityx, rev_trait_value)
    return mortalityx, mortalityd_yatf, mortalityd_dams

####################
#Mortality Murdoch #
####################
''' 
    The Murdoch Uni system includes:
        1. a base mortality for all animal classes which is a non reducible amount plus an increment
           The increment varies quadratically with both RC if below a threshold and ebg if below a threshold (relative to normal weight change)
        2. Weaner mortality is included in the base mortality through ebg being compared with normal growth rate.
        3. progeny mortality calculated from the LTW equations and is a function of birth weight, birth type and chill index at birth
        4. dam mortality: 
            a. f_mortality_dam_mu - a function of dam CS at birth. This is the increase in mortality for reproducing ewes and 
               is the same for single and twin bearing ewes.
            b. f_mortality_pregtox_cs - currently this is just 0 it needs to be built.
'''
def f_mortality_base_mu(cd, cg, rc_start, cv_weight, ebg_start, sd_ebg, d_nw_max, days_period, rev_trait_value, sap_mortalityb):
    ## a minimum level of mortality per day that is increased if RC is below a threshold and LWG is below a threshold
    ### the mortality rate increases in a quadratic function for lower RC & greater disparity between EBG and normal gain
    ###distribution on ebg & rc_start, calculate mort and then average (axis =-1,-2)
    ###distribution used to atempt to replicate real life where there is a spread within the mob. This is required because mortality is quadratic therefore it is in accurate to use mob average egb and rc.
    ebg_start_p1p2 = fun.f_distribution7(ebg_start, sd=sd_ebg)[...,na]
    rc_start_p1p2 = fun.f_distribution7(rc_start, cv=cv_weight)[...,na,:]
    ###calc mort scalars for the hybrid mortality function
    rc_mortality_scalar_p1p2 = (np.minimum(0, rc_start_p1p2 - cd[24, ...,na,na])
                                / (cd[23, ...,na,na] - cd[24, ...,na,na]))**2
    ebg_mortality_scalar_p1p2 = (np.minimum(0, (ebg_start_p1p2 * cg[18, ...,na,na] - d_nw_max[...,na,na]) - cd[26, ...,na,na])
                                 / (cd[25, ...,na,na] - cd[26, ...,na,na]))**2
    mortalityb_p1p2 = (cd[1, ...,na,na] + cd[22, ...,na,na] * rc_mortality_scalar_p1p2 * ebg_mortality_scalar_p1p2) * days_period[...,na,na]  #mul by days period to convert from mort per day to per period
    mortalityb = np.mean(mortalityb_p1p2, axis=(-1,-2))
    ##apply sensitivity
    mortalityb = fun.f_sa(mortalityb, sap_mortalityb, sa_type = 1, value_min = 0)
    ##Process the Mortality REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    mortalityb = f1_rev_update('mortality', mortalityb, rev_trait_value)
    return mortalityb


def f_mortality_weaner_mu(cu2, ce=0):
    ## The MU base mortality function accounts for the mortality increases for slow growing young animals
    ## Use coefficient cu2[20, 0, ...] & ce[20, ...]
    #todo incorporate Angus Campbell's mortality function as the MU weaner mortality function (to replace the base mortality for weaners)
    return 0


def f_mortality_dam_mu(cu2, ce, cb1, cs, cv_cs, period_is_birth, nfoet_b1, saa_mortalitye):
    ## transformed Dam mortality at birth due to low CS.
    ##The dam mortality is predicted as a transformed value (x axis) which is then back transformed to actual mortality (y axis), using the logit transformation
    ###distribution on cs_birth, calculate mort and then average (axis =-1)
    cs_p1 = fun.f_distribution7(cs, cv=cv_cs)
    ###calc mort
    t_mortalitye_mu_p1 = (cu2[22, 0, ...,na] * cs_p1 + cu2[22, 1, ...,na] * cs_p1 ** 2
                          + ce[22, ...,na] + cu2[22, -1, ...,na])
    ##Back transform the mortality
    mortalitye_mu_p1 = fun.f_back_transform(t_mortalitye_mu_p1)
    ##Average across the p1 axis (range of CS within the mob)
    mortalitye_mu = np.mean(mortalitye_mu_p1, axis=-1)
    ##Vertical shift in mortality based on litter size and only increase mortality if period is birth and reproducing ewes
    mortalitye_mu = (mortalitye_mu + cb1[22, ...]) * period_is_birth * (nfoet_b1 > 0)
    ##Adjust by sensitivity on dam mortality - need to include periods_is so the saa only gets applied in one period.
    mortalitye_mu = fun.f_sa(mortalitye_mu, saa_mortalitye * period_is_birth, sa_type = 2, value_min = 0)
    return mortalitye_mu


def f_mortality_dam_mu2(cu2, ce, cb1, cf_csc, csc, cs, cv_cs, period_between_scanprebirth
                        , period_is_prebirth, nfoet_b1, days_period, saa_mortalitye):
    '''
    Peri natal Dam mortality due to: CS at birth, CS change scanning to pre lambing, birth type & age of the dam.
    The mortality is incurred at the pre-birth period and can't be incurred each period because of the back transformation.
    '''
    ## The CS change (d_cf) is accumulated during scanning to pre-birth
    ### CS change is only incremented for multiple bearing ewes because the change in CS effect is associated
    ### with PregTox which is only applicable to multiple bearing ewes.
    d_cf = csc * days_period * period_between_scanprebirth * (nfoet_b1>1)
    ### Calculate the cumulative carried forward CS change
    cf_csc = cf_csc + d_cf
    ###The estimate of mortality includes a distribution for both the CS & CS change
    ### Mortality is calculated each loop and only retained if the period is pre-birth
    cs_p1p2 = fun.f_distribution7(cs, cv=cv_cs)[...,na]
    csc_p1p2 = fun.f_distribution7(cf_csc, cv=cv_cs)[...,na,:]
    ###calculate transformed mortality
    t_mortalitye_mu_p1p2 = (ce[23, ...,na,na] + cb1[23, ...,na,na] + cu2[23, -1, ...,na,na]
                                                                   + cu2[23, 0, ...,na,na] * cs_p1p2
                                                                   + cu2[23, 2, ...,na,na] * csc_p1p2)
    ##Back transform the mortality (Logit)
    mortalitye_mu_p1p2 = fun.f_back_transform(t_mortalitye_mu_p1p2)
    ##Average across the p1 & p2 axes (range of CS & CS change within the mob) if period is birth for reproducing ewes
    mortalitye_mu = np.mean(mortalitye_mu_p1p2, axis=(-1,-2)) * period_is_prebirth * (nfoet_b1 > 0)
    ##Adjust by sensitivity on peri-natal dam mortality - need to include periods_is so the saa only gets applied in one period.
    mortalitye_mu = fun.f_sa(mortalitye_mu, saa_mortalitye * period_is_prebirth, sa_type = 2, value_min = 0)
    return mortalitye_mu, cf_csc


def f_mortality_pregtox_mu(cb1, cg, nw_start, ebg, sd_ebg, days_period, period_is_pregtox, gest_propn, saa_mortalitye):
    '''
    (Twin) Dam mortality in last 6 weeks (preg tox). This increments mortality associated with LWL in the base mortality function.

    This is a copy of the CSIRO function except that the period has been reduced because the MU relationships include
    ewe mortality associated with loss of CS from scanning to pre-lambing (about day 135). So the CSIRO Preg Tox
    relationship is only active after that point.

    Preg tox is short for pregnancy toxaemia. It is associated with ketosis where the ewe switches into burning fat
    (rather than carbohydrates) because they are losing weight. It is predominantly a problem for multiple bearing ewes
    because they have the highest energy demands close to lambing and the least capacity to eat more (because stomach
    capacity is restricted due to the volume of the conceptus). It is usually also more of a problem for ewes
    that start out in better condition.
    '''
    ##distribution on ebg - add distribution to ebg_start_p1 and then average (axis =-1)
    ##the mortality rate per week is estimated as per the CSIRO equation (but applied for a shorter period).
    ebg_p1 = fun.f_distribution7(ebg, sd=sd_ebg)
    t_mort_p1 = days_period[..., na] * gest_propn[..., na] / 42 * fun.f_sig(-42 * ebg_p1 * cg[18, ..., na] / nw_start[..., na]
                                                                        , cb1[4, ..., na], cb1[5, ..., na]) #mul by days period to convert from mort per day to per period
    t_mort = np.mean(t_mort_p1, axis=-1)
    ##If not during the preg tox period then = 0
    mort = t_mort * period_is_pregtox
    # ##Adjust by sensitivity on dam mortality - need to include periods_is so the saa only gets applied in one period.
    # ###Removed here because mortalitye is already included in the MU system so this would be a double application
    # mort = fun.f_sa(mort, saa_mortalitye * period_is_pregtox, sa_type = 2, value_min = 0)
    return mort


def f_mortality_progeny_mu(cu2, cb1, cx, ce, w_b, w_b_std, cv_weight, foo, chill_index_p1, mob_size, period_is_birth
                           , rev_trait_value, sap_mortalityp, saa_mortalityx):
    '''
    Calculate the mortality of progeny at birth due to mis-mothering and exposure
    using the LTW & LMAT prediction equations (Oldham et al. 2011) with inclusion of chill index, FOO and mob size.
    Uses BW as a proportion of SRW (which is passed as the w_b argument)

    # Removed - The paddock level scalar is added (Young et al 2011) however, this is not calibrated for high chill environments (>1000)
    # The scalar adjusts the difference in survival if birth weight is different from the standard birthweight
    # this is to reflect the difference in survival observed in the LTW paddock trial compared with the plot scale trials.
    '''
    ##transformed survival for actual & standard
    ###distribution on w_b & rc_birth - add distribution to w_b & w_b_std and then average (axis =-1)
    w_b_p1p2 = fun.f_distribution7(w_b, cv=cv_weight)[...,na,:]
    ##removed the capacity to do a paddock level scalar because scalar is incorrect with chill index.
    ### leave code in case it is needed for backward compatibility to original LTW analysis (will need a diff slice)
    # w_b_std_p1p2 = fun.f_distribution7(w_b_std, cv=cv_weight)[...,na,:]

    t_survival_p1p2 = (cu2[8, 0, ...,na,na] * w_b_p1p2 + cu2[8, 1, ..., na,na] * w_b_p1p2 ** 2
                      + (cu2[8, 2, ..., na,na] + cx[9, ..., na, na]) * chill_index_p1[...,na]
                      + cu2[8, 4, ..., na,na] * foo[..., na,na] + cu2[8, 5, ..., na,na] * foo[..., na,na] ** 2
                      + cu2[8, -1, ..., na,na] + cb1[8, ..., na,na] + cb1[9, ..., na, na] * mob_size[..., na, na]
                      + cx[8, ..., na,na] + ce[8, ..., na,na])
    # t_survival_std_p1p2 = (cu2[8, 0, ..., na,na] * w_b_std_p1p2 + cu2[8, 1, ..., na,na] * w_b_std_p1p2 ** 2
    #                   + cu2[8, 2, ..., na,na] * chill_index_p1[...,na] + cu2[8, 4, ..., na,na] * foo[..., na,na]
    #                   + cu2[8, 5, ..., na,na] * foo[..., na,na] ** 2 + cu2[8, -1, ..., na,na] + cb1[8, ..., na,na]
    #                   + cx[8, ..., na,na] + cx[9, ..., na,na] * chill_index_p1[...,na] + ce[8, ..., na,na])
    ##back transform survival & convert to mortality
    mortalityx = (1 - np.average(fun.f_back_transform(t_survival_p1p2),axis = (-1,-2))) * period_is_birth #p1 axis averaged
    # mortalityx_std = (1 - np.average(fun.f_back_transform(t_survival_std_p1p2),axis = (-1,-2))) * period_is_birth #p1 axis averaged
    # ##Scale progeny survival using paddock level scalars
    # mortalityx = mortalityx_std + (mortalityx - mortalityx_std) * cb1[9, ...]
    ##Apply SA to progeny mortality at birth (LTW)
    mortalityx = fun.f_sa(mortalityx, sap_mortalityp, sa_type = 1, value_min = 0)
    mortalityx = fun.f_sa(mortalityx, saa_mortalityx, sa_type = 2, value_min = 0)
    ##Process the Ewe Rearing Ability REV: either save the trait value to the dictionary or overwrite trait value with value from the dictionary
    mortalityx = f1_rev_update('era', mortalityx, rev_trait_value)
    return mortalityx

#############################
#functions for end of loop #
###########################

def f1_period_start_prod(numbers, var, prejoin_tup, season_tup, period_is_startseason, mask_min_lw_z, mask_min_wa_lw_w,
                         mask_max_lw_z, mask_max_wa_lw_w, period_is_prejoin=0, group=None, scan_management=0, gbal=0,
                         drysretained_scan=1, drysretained_birth=1, stub_lw_idx=np.array(np.nan), len_gen_t=1, a_t_g=0,
                         period_is_startdvp=False):
    '''
    Production is weighted at prejoining across e&b axes and at season start across the z axis.

    Prejoining is slight more complex because there is the potential that drys would have been sold during the
    yr if the farmers management allowed them to be identified. If the drys were sold the animal at the start
    of the next repro cycle (prejoining) should be the weighted average of all the animals excluding drys.
    Because we don't know the animals are actually sold (since pyomo optimises this) we need to leave the
    main numbers variable untouched so we temporarily make the adjustment in this function. In the inputs the user
    inputs the expected number of drys that will be retained. In this function we scale the numbers of drys
    by that amount so that the new animal at the start of next prejoining reflects if drys were sold or retained.
    E.g. if drys were sold the prejoining animal would be a little bit lighter (because drys tend to weigh more).

    An extra step occurs if generating for stubble. For stubble the function selects the starting animal for the
    next period based on animal liveweight compare to the stubble trial. The animals that have te closest lw
    to the paddock trial become the starting animals next period.
    '''
    ##Set variable level = value at end of previous	
    var_start = var
    ##make sure numbers and var are same shape - this is required for the np.average func below
    numbers, var_start = np.broadcast_arrays(numbers,var_start)

    ##a)if generating with t axis reset the sale slices to the retained slice at the start of each dvp
    if np.any(period_is_startdvp) and len_gen_t>1:
        a_t_g = np.broadcast_to(a_t_g, var_start.shape)
        temporary = np.take_along_axis(var_start, a_t_g, axis=sinp.stock['i_p_pos']) #t is in the p pos
        var_start = fun.f_update(var_start, temporary, period_is_startdvp)

    ##b) Calculate temporary values as if period is start of season
    if np.any(period_is_startseason):
        var_start = f1_season_wa(numbers, var_start, season_tup, mask_min_lw_z, mask_min_wa_lw_w, mask_max_lw_z, mask_max_wa_lw_w, period_is_startseason)

    ##c) Calculated weighted average of var_start if period_is_prejoin (because the classes from the prior year are re-combined at pre-joining)
    ### If the dams have been scanned or assessed for gbal then the number of drys is adjusted based on the estimated management
    ### The adjustment for drys has to be done to the production levels at prejoining rather than to numbers at scan or birth
    ###because adjusting numbers (although more intuitive) affects the apparent mortality of the drys.
    ### Note: this is different to the approach taken for the proportion of dams mated that is done in f_end_numbers
    ###at the beginning of the prejoin DVP which doesn't affect apparent mortality.
    if group==1 and np.any(period_is_prejoin):
        ###inputs
        b1_pos = sinp.stock['i_b1_pos']
        nfoet_b1 = fun.f_expand(sinp.stock['a_nfoet_b1'],b1_pos)
        nyatf_b1 = fun.f_expand(sinp.stock['a_nyatf_b1'],b1_pos)
        ###scale numbers if drys are expected to have been sold at scanning (in the generator we don't know if drys are actually sold since pyomo optimises this, so this is just our best estimate)
        ###can't occur at prejoining rather than at scanning & birth
        temp = np.maximum(drysretained_scan, np.minimum(1, nfoet_b1)) * numbers
        scaled_numbers = fun.f_update(numbers, temp, scan_management >= 1) # only scale numbers if scanning occurs
        ###scale numbers if drys are expected to have been sold at birth (in the generator we don't know if drys are actually sold since pyomo optimises this, so this is just our best estimate)
        temp = np.maximum(drysretained_birth, np.minimum(1, nyatf_b1)) * scaled_numbers
        scaled_numbers = fun.f_update(scaled_numbers,temp, gbal >= 2)  # only scale numbers if differential management
        ###weighted average of e&b axis
        temporary = fun.f_weighted_average(var_start, scaled_numbers, prejoin_tup, keepdims=True, non_zero=True) #gets the weighted average of production in the different seasons
        ##Set values where it is beginning of FVP
        var_start = fun.f_update(var_start, temporary, period_is_prejoin)

    ##for stubble index the w axis to make the starting animal for the next period
    if np.all(np.logical_not(np.isnan(stub_lw_idx))):
        var_start[...] = np.take_along_axis(var_start, stub_lw_idx, sinp.stock['i_w_pos'])
    return var_start


def f1_season_wa(numbers, var, season, mask_min_lw_wz, mask_min_wa_lw_w, mask_max_lw_wz, mask_max_wa_lw_w, period_is_startseason):
    '''
    Perform weighted average across seasons, at the beginning of each season, so all seasons start from a common place.

    The animals with the lightest liveweight patterns (there could be multiple because depending on the fvp the w axis may be clustered)
     at the time of season start are assigned the lowest live weight from across the z axis rather than the weighted average,
     so that light animals are not lost in the postprocessing distribution. The same occurs with the heaviest animals.
    Don't need to worry about mortality in the different slices because this is not to do with condensing (in condensing we take the weights of animals with less than 10% mort).

    :param numbers: animal numbers from generator
    :param var: production variable of interest
    :param season: position of z axis
    :param mask_min_lw_wz: mask with Trues for the lightest animals across w and z
    :param mask_min_wa_lw_w: mask with Trues for the lightest animals across w after taking the weighted average of z
    :param mask_max_lw_wz: mask with Trues for the heaviest animals across w and z
    :param mask_max_wa_lw_w: mask with Trues for the heaviest animals across w after taking the weighted average of z
    :param period_is_startseason: boolean array with True for period is start of season
    :return: production variable with a singleton z axis
    '''
    ##weighted average along z axis
    temporary = fun.f_weighted_average(var,numbers,season,keepdims=True, non_zero=True)

    ##adjust production for min lw: the w slices with the minimum lw get assigned the production associated with the animal from the season with the lightest animal (this is so the light animals in the poor seasons are not disregarded when distributing in PP).
    ##use masked array to average the production from the z slices with the lightest animal (this is required in case multiple z slices have the same weight animals)
    masked_var = np.ma.masked_array(var, np.logical_not(mask_min_lw_wz))
    mean_var = np.mean(masked_var, axis=(season,sinp.stock['i_w_pos']),keepdims=True) #take the mean in case multiple season slices have the same weight light animal.
    mean_var = np.broadcast_to(mean_var, mask_min_wa_lw_w.shape) #broadcast the w axis
    temporary[mask_min_wa_lw_w] = mean_var[mask_min_wa_lw_w]

    ##adjust production for max lw: the w slices with the maximum lw get assigned the production associated with the animal from the season with the lightest animal (this is so the light animals in the poor seasons are not disregarded when distributing in PP).
    ##use masked array to average the production from the z slices with the lightest animal (this is required in case multiple z slices have the same weight animals)
    masked_var = np.ma.masked_array(var, np.logical_not(mask_max_lw_wz))
    mean_var = np.mean(masked_var, axis=(season,sinp.stock['i_w_pos']),keepdims=True) #take the mean in case multiple season slices have the same weight light animal.
    mean_var = np.broadcast_to(mean_var, mask_max_wa_lw_w.shape) #broadcast the w axis
    temporary[mask_max_wa_lw_w] = mean_var[mask_max_wa_lw_w]

    ##Update values if it is start of season
    var = fun.f_update(var, temporary, period_is_startseason)
    return var


def f1_condensed(var, lw_idx, condense_w_mask, i_n_len, i_w_len, i_n_fvp_period, period_is_condense, mask_gen_condensed_used=None, pkl_condensed_value=None, param_name=None):
    """
    Condense variable to x common points along the w axis when period_is_condense.
    Currently this function only handle 2 or 3 initial liveweights. The order of the returned W axis is M, H, L for 3 initial lws or H, L for 2 initial lws.

    Note: Animals in a given axis are condensed to starting weights determined by that activity. Meaning twins are
    distributed to starting weights that were calculated from all the w slices related to the twin activity.
    An alternative would be to distribute twins to starting weights that were calculated from all w slices across
    the e and b axis. This would mean that prejoining and condensing would have to be the same period and it would
    also complicate the 1n model since a twin on the std feed pattern may not pass to the std pattern in the next dvp.
    In the current structure at prejoining the e and b axis are weighted. This means that the w[0] activity is
    potentially created from a bigger spread of weights.

    Note 2: in the start functions the sale slices are overwritten by retained at the start of a dvp. Thus the condensed
    info is overwritten for the sale t slices (which is what we want). Distribution can occur with t axis but doesn't
    get used because only the retained t provides in the matrix.

    :param var: production variable being condensed
    :param lw_idx: index specifying the sorted order of the w axis
    :param condense_w_mask: mask which w slices can be used to build condensed animal e.g. w with mortality greater than 10% are excluded.
    :param i_n_len: number of nutrition options
    :param i_w_len: length of w axis
    :param i_n_fvp_period: number of fvps
    :param period_is_condense: bool array
    :param mask_gen_condensed_used: mask that specifies which w slices get updated by pkl condensed values.
    :return:
    """
    if np.any(period_is_condense):
        var = np.broadcast_to(var,lw_idx.shape).copy() #need to copy so array can be assigned to down below
        condense_w_mask = np.broadcast_to(condense_w_mask,lw_idx.shape)
        temporary = np.zeros_like(var)  #this is done to ensure that temp has the same size as var.
        ##test if array has diagonal and calc temp variables as if start of dvp - if there is not a diagonal use the alternative system for reallocating at the end of a DVP
        if i_n_len >= i_w_len:
            ###this method was the way we first tried - no longer used (might be used later if we add nutrient options back in)
            ### np.diagonal removes the n axis so it is added back in using the expand function, but that is a singleton, Therefore that is the reason that temp must be the same size as var. That will ensure that the new n axis is the same length as it used to before np diagonal
            temporary[...] = np.expand_dims(np.rollaxis(temporary.diagonal(axis1= sinp.stock['i_w_pos'], axis2= sinp.stock['i_n_pos']),-1,sinp.stock['i_w_pos']), sinp.stock['i_n_pos']) #roll w axis back into place and add na for n (np.diagonal removes the second axis in the diagonal and moves the other axis to the end)
        else:
            '''
            possible idea to handle more than 3 starting lws.
            Note: it is good to keep the medium lw as slice 0 because that means it is held the same across all dvps 
             and thus the user can make its pattern optimal.
                        
            if n_initial_lws==2:
                #in code already. Nothing needs to be changed for this part                
            
            else: #need to update the code with something like this.
                for i in n_initial_lws:
                #use i to assign and select correct patterns
                    if i == 0 or >2:
                        #medium = use current code with i to select correct pattern and assign
                        sl = [slice(None)] * temporary.ndim
                        sl_start_med = i*int(i_n_len ** i_n_fvp_period)
                        sl_end_med = (i+1)*int(i_n_len ** i_n_fvp_period)
                        sl[sinp.stock['i_w_pos']] = slice(sl_start_med, sl_end_med)
                        if i_n_len >= 3:
                            temporary[tuple(sl)] = f_dynamic_slice(var, sinp.stock['i_w_pos'], sl_start_med, sl_start_med+1)  # the pattern that is feed supply 1 (median) for the entire year (the top w pattern)
                        else:
                            temporary[tuple(sl)] = np.mean(var_sorted_mort, axis=sinp.stock['i_w_pos'], keepdims=True) ^this line wont work if more than 3 lws, somehow need to take mean of subsection of w axis depending on number of initial w. # average of all animals with less than 10% mort

                    elif i==1:
                        #calc high - using sorted var
                        sl = [slice(None)] * temporary.ndim
                        sl[sinp.stock['i_w_pos']] = slice(i*int(i_n_len ** i_n_fvp_period), (i+1)*int(i_n_len ** i_n_fvp_period))
                        temporary[tuple(sl)] = np.mean(f_dynamic_slice(var_sorted, sinp.stock['i_w_pos'], i_w_len - int(i_w_len / 10), -1), sinp.stock['i_w_pos'], keepdims=True)  # average of the top lw patterns

                    else: #i==2 (low w)
                        #calc low
                        numbers_start_sorted = np.take_along_axis(numbers_start_condense, lw_idx, axis=sinp.stock['i_w_pos'])
                        numbers_sorted = np.take_along_axis(numbers, lw_idx, axis=sinp.stock['i_w_pos'])
                        low_slice = np.argmax(np.sum(numbers_start_sorted, axis=prejoin_tup + (season_tup,), keepdims=True)
                                                     / np.sum(numbers_sorted, axis=prejoin_tup + (season_tup,), keepdims=True) > 0.9
                                                     , axis=sinp.stock['i_w_pos'])  # returns the index of the first w slice that has mort less the 10%.
                        low_slice = np.expand_dims(low_slice, axis=sinp.stock['i_w_pos']) #add singleton w axis back
                        sl = [slice(None)] * temporary.ndim
                        sl[sinp.stock['i_w_pos']] = slice(i*int(i_n_len ** i_n_fvp_period), (i+1)*int(i_n_len ** i_n_fvp_period))
                        temporary[tuple(sl)] = np.take_along_axis(var_sorted, low_slice, sinp.stock['i_w_pos'])
                                     
            '''
            #todo this function assumes a certain w axis order. we could change this and make it more flexible by using inputs for sinp.structuralsa['i_adjp_lw_initial_w1'].
            # this is how we did it for the mask_gen_condensed_values_used.
            ###sort var based on animal lw
            ma_var = np.ma.masked_array(var, np.logical_not(condense_w_mask))
            ma_var_sorted = np.take_along_axis(ma_var, lw_idx, axis=sinp.stock['i_w_pos']) #sort into production order (base on lw) so we can select the production of the lowest lw animals with mort less than 10% - note sorts in ascending order

            ###count number of w slices which have been masked out
            idx_min_lw = np.count_nonzero(condense_w_mask==0, axis=sinp.stock['i_w_pos'], keepdims=True) #index of the min lw with mort less than 10%
            n_lw = np.min(np.count_nonzero(condense_w_mask, axis=sinp.stock['i_w_pos'])) #number of included w slices

            ##to handle varying number of initial lws
            if sinp.structuralsa['i_w_start_len1'] == 2:
                ###add high pattern - this will not handle situations where the top 10% lw animals all have higher mortality than the threshold.
                temporary[...] = np.mean(
                    fun.f_dynamic_slice(ma_var_sorted,sinp.stock['i_w_pos'],i_w_len - int(math.ceil(n_lw / 10)), None), # ceil is used to handle cases where nutrition options is 1 (e.g. only 3 lw patterns)
                    sinp.stock['i_w_pos'],keepdims=True)  # average of the top lw patterns

                ###low pattern - production level of the lowest nutrition profile that has a mortality less than 10% for the year
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(-int(i_n_len ** i_n_fvp_period), None)
                temporary[tuple(sl)] = np.take_along_axis(ma_var_sorted, idx_min_lw, sinp.stock['i_w_pos']) #if you get an error here it probably means no animals had mort less than 10%

            else:
                ###add high pattern - this will not handle situations where the top 10% lw animals all have higher mortality than the threshold.
                temporary[...] = np.mean(fun.f_dynamic_slice(ma_var_sorted, sinp.stock['i_w_pos'], i_w_len - int(math.ceil(n_lw / 10)), None),  #ceil is used to handle cases where nutrition options is 1 (e.g. only 3 lw patterns)
                                         sinp.stock['i_w_pos'], keepdims=True)  # average of the top lw patterns

                ###add mid pattern (w 0 - 27) - use slice method in case w axis changes position (can't use MRYs dynamic slice function because we are assigning)
                ###the medium condense is the average of all animals with less than 10% mort.
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(0, int(i_n_len ** i_n_fvp_period))
                temporary[tuple(sl)] = np.mean(ma_var_sorted, axis=sinp.stock['i_w_pos'], keepdims=True)  # average of all animals with less than 10% mort

                ###low pattern - production level of the lowest nutrition profile that has a mortality less than 10% for the year
                sl = [slice(None)] * temporary.ndim
                sl[sinp.stock['i_w_pos']] = slice(-int(i_n_len ** i_n_fvp_period), None)
                temporary[tuple(sl)] = np.take_along_axis(ma_var_sorted, idx_min_lw, sinp.stock['i_w_pos']) #if you get an error here it probably means no animals had mort less than 10%

        ###update and use pkl condensed var
        ###Every trial creates a new pkl that is identified using the fs pkl number. Some trials use stored values and then essentially just create a copy. This is done so that the same fs numbers can be used.
        ###Note: can't create with w_start_len==2 and use for w_start_len==3 or visa versa (dont think this can happen with fs anyway so shouldnt be a problem).
        ###see google doc (randomness section) for more info.
        if param_name is not None:
            if sinp.structuralsa['i_use_pkl_condensed_start_condition']:
                ####store t length before updating
                t_pos = sinp.stock['i_p_pos'] #t is in p pos because p has been sliced
                i_t_len = temporary.shape[t_pos]
                ####update temporary with pickled value
                temporary_pkl = pkl_condensed_value[param_name]
                ####handle when the current trial has a different number of w slices or t slices than the create trial
                temporary_pkl = f1_adjust_pkl_condensed_axis_len(temporary_pkl, i_w_len, i_t_len)
                ###update temporary_pkl with temporary for desired w slices - high w and low w are not updated by pkl if
                ### the condensed animal calculated above has lower or higher weight (because we dont want to weight to vanish. this also handle cases when the fs is altered)
                temporary = fun.f_update(temporary_pkl, temporary, mask_gen_condensed_used)
            pkl_condensed_value[param_name] = temporary.copy()  # have to copy so that traits (e.g. mort) that are added to using += do not also update the value (not sure the copy is required here but have left it in since it was required for the rev)

        ###Update if the period is condense (shearing for offs and prejoining for dams)
        var = fun.f_update(var, temporary, period_is_condense)
    return var

def f1_adjust_pkl_condensed_axis_len(temporary, i_w_len, i_t_len):
    ####handle when the current trial has a different number of w slices than the create trial
    if i_w_len!=temporary.shape[sinp.stock['i_w_pos']]:
        #####cut back to 3 w slices that represent the start animals
        temporary = fun.f_dynamic_slice(temporary, sinp.stock['i_w_pos'], 0, None, int(temporary.shape[sinp.stock['i_w_pos']]/sinp.structuralsa['i_w_start_len1']))
        #####expand back to the number of w in the current trial
        a_s_w = (np.arange(i_w_len)/(i_w_len/sinp.structuralsa['i_w_start_len1'])).astype(int)
        a_s_twg = fun.f_expand(a_s_w, left_pos=sinp.stock['i_w_pos'], right_pos2=sinp.stock['i_w_pos'], left_pos2=-len(temporary.shape)-1)
        temporary = np.take_along_axis(temporary, a_s_twg, axis=sinp.stock['i_w_pos'])
    ####handle when the pkl condensed values dont have a t axis but the t axis is active - this can occur if the condensed params were saved in a trial where t was not active. The t axis still gets stored on the fs even if the generator didnt have an active t therefore it needs to be activated here.
    t_pos = sinp.stock['i_p_pos']  # t is in p pos because p has been sliced
    if i_t_len>temporary.shape[t_pos]:
        temporary = np.concatenate([temporary]*i_t_len, axis=t_pos) #wont work if pkl trial had t axis but current trial doesnt - to handle this would require passing in the a_t_g association.
    return temporary

def f1_gen_condensed_used(ffcfw, idx_sorted_w, condense_w_mask, n_fs, len_w, len_t, n_fvps_percondense
                          , period_is_condense_pa1e1b1nwzida0e0b0xyg, adjp_lw_initial_wzida0e0b0xyg, pkl_condensed_value, param_name):
    ###When using the pkl condensed values there may be cases when they do not have enough spread (e.g.
    ### the generated condensed animal is heavier than the pkl condensed animal this would result in weight vanishing in the distribution)
    ### in these cases the pkl condense values are overwritten by the generated condensed values.
    #### controls if the generated condensed values are used. False means pkl_condensed_values are used.
    w_pos = sinp.stock['i_w_pos']
    mask_gen_condensed_used = True #if it is not period is condense or pkl_condensed_values are not being used then this doesnt get used so just return True.
    if sinp.structuralsa['i_use_pkl_condensed_start_condition'] and np.any(
            period_is_condense_pa1e1b1nwzida0e0b0xyg):
        #####determine which w slices are the heaviest animal
        max_w_slices = np.isclose(adjp_lw_initial_wzida0e0b0xyg, np.max(adjp_lw_initial_wzida0e0b0xyg, axis=w_pos, keepdims=True))
        #####determine which w slices are the lightest animal
        min_w_slices = np.isclose(adjp_lw_initial_wzida0e0b0xyg, np.min(adjp_lw_initial_wzida0e0b0xyg, axis=w_pos, keepdims=True))

        #####calculate condensed lw (without using condensed pkl values).
        ffcfw_condensed = f1_condensed(ffcfw, idx_sorted_w, condense_w_mask, n_fs, len_w, n_fvps_percondense
                                             , period_is_condense_pa1e1b1nwzida0e0b0xyg)  # condensed lw at the end of the period
        #####get pkl condensed weight and handle when the current trial has a different number of w slices or t slices than the create trial
        pkl_ffcfw_condensed = f1_adjust_pkl_condensed_axis_len(pkl_condensed_value[param_name], len_w, len_t)
        #####calculate if the slices of the pickled values are to be updated with more extreme values from the generator
        update_high_with_gen = np.max(ffcfw_condensed, axis=w_pos, keepdims=True) > np.max(pkl_ffcfw_condensed, axis=w_pos, keepdims=True)
        update_low_with_gen = np.min(ffcfw_condensed, axis=w_pos, keepdims=True) < np.min(pkl_ffcfw_condensed, axis=w_pos, keepdims=True)
        #####create mask that controls if the generated condensed values are used. False means pkl_condensed_values are used
        mask_gen_condensed_used = np.logical_or(np.logical_and(update_high_with_gen, max_w_slices),
                                                     np.logical_and(update_low_with_gen, min_w_slices))
    return mask_gen_condensed_used

def f1_period_start_nums(numbers, prejoin_tup, season_tup, period_is_startseason, season_propn_z, group=None, nyatf_b1 = 0
                        , numbers_initial_repro=0, gender_propn_x=1, period_is_prejoin=0, period_is_birth=False, prevperiod_is_wean=False
                        ,len_gen_t=1, a_t_g=0, period_is_startdvp=False):

    #a)if generating with t axis reset the sale slices to the retained slice at the start of each dvp
    if np.any(period_is_startdvp) and len_gen_t>1:
        a_t_g = np.broadcast_to(a_t_g, numbers.shape)
        temporary = np.take_along_axis(numbers, a_t_g, axis=sinp.stock['i_p_pos']) #t is in the p pos
        numbers = fun.f_update(numbers, temporary, period_is_startdvp)

    ##b) reallocate for season type
    if np.any(period_is_startseason):
        temporary = np.sum(numbers * season_propn_z, axis = season_tup, keepdims=True) #Calculate temporary values as if period_is_break
        numbers = fun.f_update(numbers, temporary, period_is_startseason)  #Set values where it is beginning of season
    ##c)things for dams - prejoining and moving between classes
    if group==1 and np.any(period_is_prejoin):
        ###new repro cycle (prejoining)
        temporary = np.sum(numbers, axis = prejoin_tup, keepdims=True) * numbers_initial_repro #Calculate temporary values as if period_is_prejoin
        numbers = fun.f_update(numbers, temporary, period_is_prejoin)  #Set values where it is beginning of FVP
    ##d)things just for yatf
    if group==2:
        temp = nyatf_b1 * gender_propn_x   # nyatf is accounting for peri-natal mortality. But doesn't include the differential mortality of female and male offspring at birth
        numbers=fun.f_update(numbers, temp, period_is_birth)
        numbers=fun.f_update(numbers, 0, prevperiod_is_wean) #set numbers to 0 after weaning
    return numbers


def f1_period_end_nums(numbers, mortality, mortality_yatf=0, nfoet_b1 = 0, nyatf_b1 = 0, group=None
                      , conception = 0, gender_propn_x=1, period_is_mating = False
                      , period_is_matingend = False, period_is_birth=False, period_isbetween_prejoinmatingend=False
                      , propn_dams_mated=1):
    '''
    This adjusts numbers for things like conception and mortality that happen during a given period
    '''
    ##a) mortality (include np.maximum on mortality so that numbers can't become negative)
    ###For dams temporarily update the nm mort with mated mort between prejoining and end of mating. So that conception is calculated
    ### reflect the mated numbers. This is required because nm and mated might have a different feedsupply and conception needs to be based on the mated fs and hence mort.
    ### The back dating of the numbers scales the mortality correctly.
    if group==1:
        mortality = fun.f_update(mortality, mortality[:, :, :, 2:3, ...], period_isbetween_prejoinmatingend)
    numbers = numbers * np.maximum(0, 1-mortality)
    ##things for dams - prejoining and moving between classes
    if group==1:
        ###b) conception - conception is the change in numbers +ve for animals getting pregnancy and -ve in the NM e-0 slice (note the conception for e slice 1 and higher puts the negative numbers in the e-0 nm slice)
        if np.any(period_is_mating):
            temporary = numbers + conception * numbers[:, :, 0:1, 0:1, ...]  # numbers_dams[:,0,0,...] is the NM slice of cycle 0 ie the number of animals yet to be mated (conception will have negative value in nm slice)
            numbers = fun.f_update(numbers, temporary, np.any(period_is_mating, axis=sinp.stock['i_e1_pos'])) #needs to be previous period else conception is not calculated because numbers happens at beginning of p loop
        ###at the end of mating move any remaining numbers from nm to 00 slice (note only the nm slice for e-0 has numbers - this is handled in the conception function)
        ###Set temporary to copy of current numbers
        if np.any(period_is_matingend):
            temporary  = np.copy(numbers)
            temporary[:, :, 0:1, 1:2, ...] += numbers[:, :, 0:1, 0:1, ...]   # add the number remaining unmated to the dry slice in e1[0]
            temporary[:, :, :, 0:1, ...] = 0 #set the NM slice to 0 (because they have just been added to drys)
            ##handle the proportion mated. Note: if the inputs are set to optimise the proportion (np.inf) then it is treated as 100% mated
            mated_propn = np.minimum(1, propn_dams_mated) #maximum value of 1 because default is inf, otherwise propn to be mated.
            ### the number in the NM slice e1[0] is a proportion of the total numbers
            ### need a minimum number to keep nm in pyomo. Want a small number relative to mortality (after allowing for multiple slices getting the small number)
            ### Scale the numbers based on expected proportion mated so that the weighted average for production reflects expected management
            ### Note: scaling of numbers for expected management of drys occurs in f1_period_start_prod()
            temporary[:, :, 0:1, 0:1, ...] = np.maximum(0.00001, np.sum(temporary, axis=(sinp.stock['i_e1_pos'], sinp.stock['i_b1_pos']),
                                                                     keepdims=True) * (1 - mated_propn))
            ### the numbers in the other mated slices other than NM get scaled by the proportion mated
            temporary[:, :, :, 1:, ...] = np.maximum(0, temporary[:, :, :, 1:, ...] * mated_propn)
            ###update numbers with the temporary calculations if it is the end of mating
            numbers = fun.f_update(numbers, temporary, period_is_matingend)
        ###d) birth (account for birth status and if drys are retained)
        if np.any(period_is_birth):
            dam_propn_birth_b1 = fun.f_comb(nfoet_b1, nyatf_b1) * (1 - mortality_yatf) ** nyatf_b1 * mortality_yatf ** (nfoet_b1 - nyatf_b1) # the proportion of dams of each LSLN based on (progeny) mortality
            ##have to average x axis so that it is not active for dams - times by gender propn to give approx weighting (ie because offs are not usually entire males so they will get low weighting)
            temp = np.sum(dam_propn_birth_b1 * gender_propn_x, axis=sinp.stock['i_x_pos'], keepdims=True) * numbers[:,:,:,sinp.stock['ia_prepost_b1'],...]
            numbers = fun.f_update(numbers, temp, period_is_birth)  # calculated in the period after birth when progeny mortality due to exposure is calculated
    return numbers


#################
#post processing#
#################
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
        ph = fun.f_back_transform(ph)
#        ph = 1 / (1 + np.exp(-ph))
    ##predicted cv hauteur
    cvh = cu5_u5c5[0, i_eqn_cvh] * sl + cu5_u5c5[1, i_eqn_cvh] * ss + cu5_u5c5[2, i_eqn_cvh] * fd + cu5_u5c5[
        3, i_eqn_cvh] * pmb + cu5_u5c5[5, i_eqn_cvh] * vm + cu5_u5c5[6, i_eqn_cvh] * cvfd + cu5_u5c5[
          7, i_eqn_cvh] * cvsl + cu5_u5c5[8, i_eqn_cvh]
    ##predicted romaine
    romaine = cu5_u5c5[0, i_eqn_romaine] * sl + cu5_u5c5[1, i_eqn_romaine] * ss + cu5_u5c5[2, i_eqn_romaine] * fd + \
      cu5_u5c5[3, i_eqn_romaine] * pmb + cu5_u5c5[5, i_eqn_romaine] * vm + cu5_u5c5[6, i_eqn_romaine] * cvfd + \
      cu5_u5c5[7, i_eqn_romaine] * cvsl + cu5_u5c5[8, i_eqn_romaine]
    return ph, cvh, romaine


def f1_woolprice():
    '''Calculate the micron price guide (MPG) for a range of FD (where the FD is of the fleece component of the clip)
    Includes sensitivity on the average micron price guide (MPG) and the premium for finer wool
    The inputs for the function are:
    i_woolp_mpg_range_w5 - the percentile values for which the MPG is input
    i_woolp_mpg_w5 - the MPG for the base FD at each of the percentile levels
    i_woolp_mpg_percentile - the percentile level to use for this trial
    i_woolp_fdprem_range_w5 - the percentile values for which the FD premium is input
    i_woolp_fdprem_w4w5 - the FD premium at each of the percentile levels
    i_woolp_fdprem_percentile - the percentile level to use for this trial
    '''
    ##input value for micron price guide percentile to use (adjusted by SAV during the input process)
    mpg_percentile = uinp.sheep['i_woolp_mpg_percentile']
    ##extrapolate price for the std FD at selected percentile (can go beyond the data input range)
    mpg_stdfd = fun.np_extrap(mpg_percentile, uinp.sheep['i_woolp_mpg_range_w5'], uinp.sheep['i_woolp_mpg_w5'])
    ##adjust price for the std FD using sav
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sav['woolp_mpg'], 5)
    ##adjust price for the std FD using sam
    mpg_stdfd = fun.f_sa(mpg_stdfd, sen.sam['woolp_mpg'])
    ##FD percentile to use (adjusted by SAV during the input process)
    fd_percentile = uinp.sheep['i_woolp_fdprem_percentile']
    ##extrapolate FD premium at selected percentile for each FD (can go beyond the data input range)
    fdprem_w4 = np.array([fun.np_extrap(fd_percentile, uinp.sheep['i_woolp_fdprem_range_w5'], uinp.sheep['i_woolp_fdprem_w4w5'][i])
                          for i in range(uinp.sheep['i_woolp_fdprem_w4w5'].shape[0])])
    ##adjust FD premium using sav
    fdprem_w4 = fun.f_sa(fdprem_w4, sen.sav['woolp_fdprem'], 5)
    ##Wool price for the analysis (Note: fdprem is the price difference compared with the base FD)
    mpg_w4 = mpg_stdfd * (1 + fdprem_w4)

    ##STB scalar - the scalar is different for merino vs crossbreed because the component params differ.
    mer_stb_scalar_w4 = np.sum(uinp.sheep['i_woolp_component_mer_propn_w3'] * uinp.sheep['i_woolp_component_price_w4w3'] * (
                1 + np.interp(uinp.sheep['i_woolp_fd_range_w4'][:,na] + uinp.sheep['i_woolp_component_mer_fd_w3'], uinp.sheep['i_woolp_fd_range_w4'], fdprem_w4)) / (
                       1 + fdprem_w4[:,na]), axis=-1)
    xb_stb_scalar_w4 = np.sum(uinp.sheep['i_woolp_component_xb_propn_w3'] * uinp.sheep['i_woolp_component_price_w4w3'] * (
                1 + np.interp(uinp.sheep['i_woolp_fd_range_w4'][:,na] + uinp.sheep['i_woolp_component_xb_fd_w3'], uinp.sheep['i_woolp_fd_range_w4'], fdprem_w4)) / (
                       1 + fdprem_w4[:,na]), axis=-1)
    ###combine mer & xb. The selection of one or the other is based on the FD. If fd >= 26 then use XB.
    stb_scalar_w4 = mer_stb_scalar_w4
    stb_scalar_w4[uinp.sheep['i_woolp_fd_range_w4']>=26] = xb_stb_scalar_w4[uinp.sheep['i_woolp_fd_range_w4']>=26]
    return mpg_w4 * stb_scalar_w4


def f_wool_value(stb_mpg_w4, wool_price_scalar_c1w4tpg, cfw_pg, fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg,dtype=None):
    '''Calculate the net value of the wool on the sheep's back (cost of shearing is not included in these calculations)
    Includes adjusting price for FD, level of fault (VM & predicted hauteur) and all components of the clip (STB)
    FNF is 'free or nearly free' i.e. wool with no fault (low VM & high SS)
    STB is sweep the board i.e. including all the wool types that are produced (fleece, pieces, bellies ...)
    NIB is net in the bank i.e. all selling, testing & freight costs removed
    '''
    ##call function to calculate predicted hauteur (ph), CV of hauteur (cvh) and romaine
    ph_pg, cvh_pg, romaine_pg = f_wool_additional(fd_pg, sl_pg, ss_pg, vm_pg, pmb_pg)
    ##STB price for FNF (free or nearly free of fault)
    fnfstb_pg = np.interp(fd_pg, uinp.sheep['i_woolp_fd_range_w4'], stb_mpg_w4 ).astype(dtype)
    ##vm price adj
    vm_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_vm_adj_w4w6'], uinp.sheep['i_woolp_vm_range_w6']
                                           , uinp.sheep['i_woolp_fd_range_w4'], vm_pg,fd_pg).astype(dtype)
    ##predicted hauteur price adj
    ph_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_ph_adj_w4w7'], uinp.sheep['i_woolp_ph_range_w7']
                                           , uinp.sheep['i_woolp_fd_range_w4'], ph_pg,fd_pg).astype(dtype)
    ##cv hauteur price adj
    cvh_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_cvh_adj_w4w8'], uinp.sheep['i_woolp_cvh_range_w8']
                                            , uinp.sheep['i_woolp_fd_range_w4'], cvh_pg,fd_pg).astype(dtype)
    ##romaine price adj
    romaine_adj_pg = fun.f_bilinear_interpolate(uinp.sheep['i_woolp_romaine_adj_w4w9'], uinp.sheep['i_woolp_romaine_range_w9']
                                                , uinp.sheep['i_woolp_fd_range_w4'], romaine_pg,fd_pg).astype(dtype)
    ##wool price with adjustments
    woolp_stb_pg = fnfstb_pg * (1 + vm_adj_pg) * (1 + ph_adj_pg) * (1 - cvh_adj_pg) * (1 - romaine_adj_pg)
    ##stb net in the bank price
    woolp_stbnib_pg = woolp_stb_pg * (1 - uinp.sheep['i_wool_cost_pc']) - uinp.sheep['i_wool_cost_kg']
    ##wool value if shorn this period
    wool_value_pg = woolp_stbnib_pg * wool_price_scalar_c1w4tpg * cfw_pg
    return wool_value_pg, woolp_stbnib_pg


def f1_condition_score(rc_tpg, cn):
    ''' Estimate CS from LW. Works with scalars or arrays - provided they are broadcastable into ffcfw.

       ffcfw: (kg) Fleece free, conceptus free liveweight. normal_weight: (kg). cs_propn: (0.19) change in LW
       associated with 1 CS as a proportion of normal_weight.

       long version of the formula (use rc instead of using to following): 3 + (ffcfw - normal_weight) / (cs_propn * normal_weight)
       Returns: condition score - float
       '''
    return np.maximum(1, 3 + (rc_tpg - 1) / cn[5, ...]) #a minimum value of CS=1 is used to remove errors caused by low CS. A CS below 1 is unlikely because the animal would be dead


def f1_fat_score(rc_tpg, cn):
    ''' Calculate fat score from relative condition using relationship from van Burgel et al. 2011.
    Steps 1. calculate CS
          2. estimate GR tissue depth using relationship from van Burgel CS = 2.5 + 0.06 GR
          3. convert to fat score. FS1 = <5mm, FS2 6-10mm, FS3 11-15mm, FS4 16-20mm, FS5 >21mm'''
    condition_score = f1_condition_score(rc_tpg, cn)
    gr_depth = np.maximum(0, (condition_score - 2.5) / 0.06)
    fat_score = np.clip((gr_depth + 4)/5, 1, 5) #FS 1 is the lowest possible measurement.
    return fat_score


def f1_saleprice(score_pricescalar_s7s5s6, weight_pricescalar_s7s5s6, dtype=None):
    ##Sale price percentile to use (adjusted by sav)
    salep_percentile = uinp.sheep['i_salep_percentile']
    ##Max price in each grid (s7 axis) at selected percentile - 1d extrapolation along the s4 axis (can go beyond the input range)
    grid_max_s7 = (np.array([fun.np_extrap(salep_percentile, uinp.sheep['i_salep_percentile_range_s4'], uinp.sheep['i_salep_percentile_scalar_s7s4'][i])
                            for i in range(uinp.sheep['i_salep_percentile_scalar_s7s4'].shape[0])]) * uinp.sheep['i_salep_price_max_s7']).astype(dtype)
    ##Max price in grids (adj sav - overwrites the previous values)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sav['salep_max_s7'], 5)
    ##Max price in grids (adj sam - scales the values)
    grid_max_s7 = fun.f_sa(grid_max_s7, sen.sam['salep_max_s7'])
    ##Scalar for weight impact across the grid (sat adjusted)
    weight_scalar_s7s5s6 = weight_pricescalar_s7s5s6
    ##Scalar for score impact across the grid (sat adjusted)
    score_scalar_s7s5s6 = score_pricescalar_s7s5s6
    ##price for the analysis
    grid_s7s5s6 = grid_max_s7[:,na,na] * weight_scalar_s7s5s6 * score_scalar_s7s5s6
    return grid_s7s5s6


def f1_salep_mob(weight_s7tpg, scores_s7s6tpg, cvlw_s7s5tpg, cvscore_s7s6tpg,
                grid_weightrange_s7s5tpg, grid_scorerange_s7s6p5tpg, grid_priceslw_s7s5s6tpg):
    '''A function to calculate the average price of the mob based on the average specifications in the mob.
    This is to represent that the distribution of weight & specification reduces the mob average price
    This representation allows valuing individual animal management and reducing the mob distribution.
    Note: if the distribution extends below the lower range of weight or score in the grid these animals have zero value (ncv)'''

    ##loop on s7 to reduce memory
    saleprice_mobaverage_s7tpg = np.zeros_like(weight_s7tpg)
    for s7 in range(weight_s7tpg.shape[0]):
        ## Probability for each lw step in grid based on the mob average weight and the coefficient of variation (CV) of weight
        ### probability of being less than the upper value of the step (roll) - probability of less than the lower value of the step
        prob_lw_s5tpg = np.maximum(0, fun.f_norm_cdf(np.roll(grid_weightrange_s7s5tpg[s7,...], -1, axis = 0), weight_s7tpg[s7,...], cvlw_s7s5tpg[s7,...])
                              - fun.f_norm_cdf(grid_weightrange_s7s5tpg[s7,...], weight_s7tpg[s7,...], cvlw_s7s5tpg[s7,...]))
        ## Probability for each score step in grid (fat score/CS) based on the mob average score and the CV of quality score
        prob_score_s6tpg = np.maximum(0, fun.f_norm_cdf(np.roll(grid_scorerange_s7s6p5tpg[s7,...], -1, axis = 0), scores_s7s6tpg[s7,...], cvscore_s7s6tpg[s7,...])
                                 - fun.f_norm_cdf(grid_scorerange_s7s6p5tpg[s7,...], scores_s7s6tpg[s7,...], cvscore_s7s6tpg[s7,...]))
        ###adjust prob so that animals with score less than 1 get allocated to the score 1 slice - assumption is that score 1 is lowest
        prob_score_s6tpg[0,...] = prob_score_s6tpg[0,...] + (1-np.sum(prob_score_s6tpg, axis=0))

        ##Probability for each cell of grid (assuming that weight & score are independent allows multiplying weight and score probabilities)
        prob_grid_s5s6tpg = prob_lw_s5tpg[:,na, ...] * prob_score_s6tpg

        ##Average price for the mob is the sum of the probabilities in each cell of the grid and the price in that cell
        saleprice_mobaverage_s7tpg[s7,...] = np.sum(prob_grid_s5s6tpg * grid_priceslw_s7s5s6tpg[s7,...], axis = (0, 1))
    return saleprice_mobaverage_s7tpg


def f_sale_value(cn, cx, o_rc_tpg, o_ffcfw_tpg, dressp_adj_yg, dresspercent_adj_s6tpg,
                 dresspercent_adj_s7tpg, grid_price_s7s5s6tpg, price_scalar_c1s7tpg, month_scalar_s7tpg,
                 month_discount_s7tpg, price_type_s7tpg, cvlw_s7s5tpg, cvscore_s7s6tpg,
                 grid_weightrange_s7s5tpg, grid_scorerange_s7s6tpg, age_end_pg1, discount_age_s7tpg,sale_cost_pc_s7tpg,
                 sale_cost_hd_s7tpg, mask_s7x_s7tpg, sale_agemax_s7tpg1, sale_agemin_s7tpg1, sale_ffcfw_min_s7tpg,
                 sale_ffcfw_max_s7tpg, mask_s7g_s7tpg, dtype=None):
    ##Calculate condition score from relative condition
    cs_tpg = f1_condition_score(o_rc_tpg, cn)
    ##Calculate fat score from relative condition
    fs_tpg = f1_fat_score(o_rc_tpg, cn)
    ##Combine the scores into single array
    scores_s8tpg = np.stack([fs_tpg, cs_tpg], axis=0)
    ##Select the quality scores (s8) for each price grid (s7)
    scores_s7s6tpg = scores_s8tpg[uinp.sheep['ia_s8_s7']][:,na,...]
    ##Dressing percentage to adjust price grid from $/kg DW to $/kg LW
    ### It is easier to convert the price to $/kg LW than it is to convert a distribution of LW and fat score to a distribution of dressed weight and fat score
    ### because dressing percentage changes with fat score.
    dresspercent_for_price_s7s6tpg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[1, ...] + dresspercent_adj_s6tpg + dresspercent_adj_s7tpg[:,na,...]
    ##Dressing percentage is set to 100% if price type is $/kg LW or $/hd
    dresspercent_for_price_s7s6tpg = fun.f_update(dresspercent_for_price_s7s6tpg, 1, price_type_s7tpg[:,na,...] >= 1)
    ##Create the grid prices in $/kg LW
    grid_priceslw_s7s5s6tpg = grid_price_s7s5s6tpg * dresspercent_for_price_s7s6tpg[:,na,...]

    ## Calculate the 'lookup' weight of the average animal in the units of each grid (some grids the weight is dressed weight other grids are LW)
    ## start with dressing percentage and set to 1 later if the grid is kg LW
    ###Interploate DP adjustment based on the average FS of the animals
    dressp_adj_fs_tpg= np.interp(fs_tpg, uinp.sheep['i_salep_score_range_s8s6'][0, ...], uinp.sheep['i_salep_dressp_adj_s6']).astype(dtype)
    ###Average Dressing percentage including effects of genotype, fat score and age (which varies with the grid).
    dresspercent_for_wt_s7tpg = pinp.sheep['i_dressp'] + dressp_adj_yg + cx[1, ...] + dressp_adj_fs_tpg + dresspercent_adj_s7tpg
    ###Dressing percentage is 100% if price type is $/kg LW or $/hd
    dresspercent_wt_s7tpg = fun.f_update(dresspercent_for_wt_s7tpg, 1, price_type_s7tpg >= 1)
    ###Scale ffcfw to the units in the grid
    weight_for_lookup_s7tpg = o_ffcfw_tpg * dresspercent_wt_s7tpg

    ##Calculate mob average price in each grid from the mob average and the distribution of weight & score within the mob (this is just the price, not the total animal value)
    price_mobaverage_s7tpg = f1_salep_mob(weight_for_lookup_s7tpg, scores_s7s6tpg, cvlw_s7s5tpg, cvscore_s7s6tpg,
                                                      grid_weightrange_s7s5tpg, grid_scorerange_s7s6tpg, grid_priceslw_s7s5s6tpg)

    ##Scale price received based on month of sale
    price_mobaverage_s7tpg = price_mobaverage_s7tpg * (1+month_scalar_s7tpg)

    ## Apply the age based discount if the animal is greater than the threshold age
    ### Temporary value of the age based discount from the relevant month
    temporary_s7tpg = price_mobaverage_s7tpg * (1 + month_discount_s7tpg)
    ###Apply discount if age is greater than threshold age
    price_mobaverage_s7tpg = fun.f_update(price_mobaverage_s7tpg, temporary_s7tpg, age_end_pg1/30.4 > discount_age_s7tpg)  #divide 30 to convert to months

    ## Some grids are in $/hd. For these grids don't want to multiply grid value by weight (so set weight to 1)
    ### Convert weight to 1 if price is $/hd (price_type == 2)
    weight_for_value_s7tpg = fun.f_update(o_ffcfw_tpg, 1, price_type_s7tpg == 2)

    ## Calculate the net value per head from the gross value minus the selling costs
    ### Calculate gross value per head
    sale_value_s7tpg = price_mobaverage_s7tpg * weight_for_value_s7tpg
    ###add price variation - add before removing s7 axis incase scalar gets an active s7 axis.
    sale_value_c1s7tpg = sale_value_s7tpg * price_scalar_c1s7tpg
    ###Subtract the selling costs (some are percentage costs some are $/hd)
    sale_value_c1s7tpg = sale_value_c1s7tpg * (1 - sale_cost_pc_s7tpg) - sale_cost_hd_s7tpg

    ## Select the best net sale price from the relevant grids
    ###Mask the grids based on the maximum age, minimum age, the gender for each grid and genotype
    sale_value_c1s7tpg = sale_value_c1s7tpg * mask_s7x_s7tpg * mask_s7g_s7tpg * (age_end_pg1/30.4 <= sale_agemax_s7tpg1) * (age_end_pg1/30.4 >= sale_agemin_s7tpg1) #divide 30 to convert to months
    ###mask grids based on maimum and minimum ffcfw
    mask_ffcfw_s7tpg = np.logical_and(o_ffcfw_tpg>sale_ffcfw_min_s7tpg, o_ffcfw_tpg<sale_ffcfw_max_s7tpg)
    sale_value_c1s7tpg = sale_value_c1s7tpg * mask_ffcfw_s7tpg
    ###Select the maximum value across the grids
    sale_value = np.max(sale_value_c1s7tpg, axis=1)
    sale_grid = np.argmax(sale_value_c1s7tpg, axis=1)
    return sale_value, sale_grid

def f1_animal_trigger_levels(index_pg, age_start, period_is_shearing_pg, period_is_wean_pg, gender, o_ebg_tpg, wool_genes,
                            period_is_joining_pg, animal_mated, scan_option, period_is_endmating_pg):
    '''

    .. note:: animal_triggervalues_th7pg3 requires an h2 axis when it is applied, however, due to size the h2 axis is only added at the point it is used

    '''
    ##Trigger value 1 - week of year
    trigger1_pg = index_pg % 52
    ##Trigger value 2 - age
    trigger2_pg = np.trunc(age_start / 7)
    ##Trigger value 3 - Weeks from previous shearing
    trigger3_pg = index_pg - np.maximum.accumulate(index_pg*period_is_shearing_pg, axis=sinp.stock['i_p_pos'])
    ##Trigger value 4 - weeks to next shearing
    shear_idx = index_pg * period_is_shearing_pg
    shear_idx[np.logical_not(period_is_shearing_pg)] = np.max(index_pg) * 2 #set index to large number if not shearing
    trigger4_pg = np.flip(np.minimum.accumulate(np.flip(shear_idx, axis=sinp.stock['i_p_pos']), axis=sinp.stock['i_p_pos']), axis=sinp.stock['i_p_pos']) - index_pg
    ##Trigger value 5 - weeks from previous joining
    trigger5_pg = index_pg - np.maximum.accumulate(index_pg*period_is_joining_pg)
    ##Trigger value 6 - weeks from end of mating
    trigger6_pg = index_pg - np.maximum.accumulate(index_pg*period_is_endmating_pg)
    ##Trigger value 7 - weeks from previous weaning
    trigger7_pg = index_pg - np.maximum.accumulate(index_pg*period_is_wean_pg)
    ##Trigger value 8 - whether animals was mated
    trigger8_pg = animal_mated
    ##Trigger value 9 - scanning option being used
    trigger9_pg = scan_option
    ##Trigger value 10 - gender of the animal
    trigger10_pg = gender
    ##Trigger value 11 - rate of empty body gain
    trigger11_pg = o_ebg_tpg
    ##Trigger value 12 - the 'wooliness' of the genotype
    trigger12_pg = wool_genes
    ##Stack the triggers
    animal_triggervalues_h7tpg = np.stack(np.broadcast_arrays(trigger1_pg, trigger2_pg, trigger3_pg, trigger4_pg,
                                                             trigger5_pg, trigger6_pg, trigger7_pg, trigger8_pg, trigger9_pg,
                                                             trigger10_pg, trigger11_pg, trigger12_pg), axis = 0)
    return animal_triggervalues_h7tpg


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


def f1_adjust_triggervalues_for_t(animal_triggervalues_h7tpg, operations_triggerlevels_h5h7tpg, a_t_g):
    '''
    The t slice on period_is_shearing means that a randomness can be introduced in the husbandry.
    For example if animal classing is done 1 week before shearing but shearing for the t[2] (sale slice)
    occurs on the first period on a dvp then the husbandry will be triggered in the previous dvp
    but t[2] animals are in t[0] (retained) slice in the previous dvp therefore they do not incur the classing cost.

    The best solution we could come up with is to base husb triggers off the retained slice unless the trigger is ==0
    If the input value for time since 'x' or time to 'x' is 0 then you use t[:] if the value is anything other
    than 0 (i.e. it might be in a different DVP) then t[0] is used.

    For dams. there will still be some
    potential errors with dams changing g slice (for example crutching).

    Note: the generator t axis has been made to a singleton to reduce computational time in the husb calcs. Thus this
    function doesn't do anything for dams unless period_is_shear has a t axis.

    This function must be called each time the trigger_values are used. It needs to be called inside a h2
    loop so that triggervalues never has a full h2 axis (that would be too big).

    '''
    p_pos = sinp.stock['i_p_pos']
    ##only need to handle the t axis for groups that have a t axis (currently just offs)
    if animal_triggervalues_h7tpg.shape[p_pos-1]>1:

        #which of the trigger level inputs are operating on the current generator period which means we can use t[:] rather than the retained animal.
        #the slices h7[2:7] relate to time from previous or time to next, the values for these slices need to be 0 or default
        trigger_is_not_current_tpg = np.logical_not(np.all(np.logical_or(np.abs(operations_triggerlevels_h5h7tpg[:,2:7,...]) == np.inf,
                                                     operations_triggerlevels_h5h7tpg[:,2:7,...] == 0), axis=(0,1)))

        # select retained t slice if the trigger_is_not_current
        a_t_tpg = fun.f_expand(a_t_g, p_pos-2, right_pos=-1)
        t_animal_triggervalues_h7tpg = np.take_along_axis(animal_triggervalues_h7tpg, a_t_tpg[na], axis=p_pos-1)
        animal_triggervalues_h7tpg = fun.f_update(animal_triggervalues_h7tpg, t_animal_triggervalues_h7tpg, trigger_is_not_current_tpg)

    return animal_triggervalues_h7tpg


def f1_operations_triggered(animal_triggervalues_h7tpg, operations_triggerlevels_h5h7h2tpg, a_t_g):
    shape = (operations_triggerlevels_h5h7h2tpg.shape[2],) + animal_triggervalues_h7tpg.shape[1:]
    triggered_h2tpg = np.zeros(shape, dtype=bool)
    for h2 in range(operations_triggerlevels_h5h7h2tpg.shape[2]):
        ##adjust triggervalues for t axis
        adj_animal_triggervalues_h7tpg = f1_adjust_triggervalues_for_t(animal_triggervalues_h7tpg, operations_triggerlevels_h5h7h2tpg[:,:,h2,...], a_t_g)
        ##Test slice 0 of h5 axis
        slice0_h7tpg = adj_animal_triggervalues_h7tpg[:, ...] <= operations_triggerlevels_h5h7h2tpg[0, :, h2, ...]
        ##Test slice 1 of h5 axis
        slice1_h7tpg = np.logical_or(adj_animal_triggervalues_h7tpg[:, ...] == operations_triggerlevels_h5h7h2tpg[1, :, h2, ...],
                                    operations_triggerlevels_h5h7h2tpg[1, :, h2, ...] == np.inf)
        ##Test slice 2 of h5 axis
        slice2_h7tpg = adj_animal_triggervalues_h7tpg[:, ...] >= operations_triggerlevels_h5h7h2tpg[2, :, h2, ...]
        ##Test across the conditions
        slices_all_h7tpg = np.logical_and(slice0_h7tpg, np.logical_and(slice1_h7tpg, slice2_h7tpg))
        ##Test across the rules (& collapse s7 axis)
        triggered_h2tpg[h2,...] = np.all(slices_all_h7tpg, axis=0)
    return triggered_h2tpg


def f1_application_level(operation_triggered_h2pg, animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg, a_t_g):
    ##loop on h2 axis to save memory
    level_h2pg = np.ones_like(operation_triggered_h2pg, dtype='float32')
    for h2 in range(operation_triggered_h2pg.shape[0]):
        ##adjust triggervalues for t axis
        adj_animal_triggervalues_h7pg = f1_adjust_triggervalues_for_t(animal_triggervalues_h7pg, operations_triggerlevels_h5h7h2pg[:,:,h2,...], a_t_g)

        ## mask & remove the slices of the h7 axis that don't require calculation of the application level (not required because inputs do not include a range input)
        ## must be same mask for 'le' and 'ge'
        maskh7_h7 = fun.f_reduce_skipfew(np.any, operations_triggerlevels_h5h7h2pg[3,:, h2, ...] != np.inf, preserveAxis=0)

        ##slice operation_triggered array
        operation_triggered_pg = operation_triggered_h2pg[h2,...]

        ##if all values in mask are false (e.g. no range level needs to be calculated) then skip to next h2 (final array has 1 as default value so nothing needs to happen)
        if any(maskh7_h7):
            ### mask the input arrays to minimise slices of h7
            animal_triggervalues_h7mask_h7pg = adj_animal_triggervalues_h7pg[maskh7_h7]
            operations_triggerlevels_h7mask_h5h7pg = operations_triggerlevels_h5h7h2pg[:, maskh7_h7, h2, ...]


            ##broadcast the input arrays so the 'required' mask can be applied
            operations_triggerlevels_casted_h5h7pg=np.broadcast_to(operations_triggerlevels_h7mask_h5h7pg, operations_triggerlevels_h7mask_h5h7pg.shape[0:2]+operation_triggered_pg.shape)
            animal_triggervalues_h7mask_h7pg = np.broadcast_to(animal_triggervalues_h7mask_h7pg, operations_triggerlevels_casted_h5h7pg.shape[1:])


            ## Calculate the application level for "less than or equal"
            ### The 'le' calculation is required only if the 'range' input is less than the le trigger value and both are not inf.
            required_h7pg = (operation_triggered_pg * (operations_triggerlevels_h7mask_h5h7pg[0, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] < operations_triggerlevels_h7mask_h5h7pg[0, ...]))

            ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
            temporary_h7pg = np.ones_like(required_h7pg, dtype='float32')

            ##Level if animal trigger level is between 'range' and 'le'
            ### calculate the masked version of the triggerlevels because required 3 times in the calculation
            operations_triggerlevels_masked_h5h7pg = operations_triggerlevels_casted_h5h7pg[:, required_h7pg]
            temporary_h7pg[required_h7pg] = np.clip((animal_triggervalues_h7mask_h7pg[required_h7pg] - operations_triggerlevels_masked_h5h7pg[0, ...])/
                                                        (operations_triggerlevels_masked_h5h7pg[3, ...] - operations_triggerlevels_masked_h5h7pg[0, ...]),0,1)
            ##Select the maximum across the h7 axis if the operation is triggered
            level_pg = np.max(temporary_h7pg,axis=0) * operation_triggered_pg   #mul by operation triggered so that level goes to 0 if operation is not triggered


            ## Repeat for 'ge' using same variable names as for 'le'
            ## Calculate the application level for "greater than or equal"
            ### The 'ge' calculation is required only if the 'range' input is greater than the ge trigger value and both are not inf.
            required_h7pg = (operation_triggered_pg * (operations_triggerlevels_h7mask_h5h7pg[2, ...] != -np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] != np.inf)
                               * (operations_triggerlevels_h7mask_h5h7pg[3, ...] > operations_triggerlevels_h7mask_h5h7pg[2, ...]))

            ##Create blank versions for assignment - one is the default value for the calc below where the mask is false hence initialise with ones
            ### calculate the masked version of the triggerlevels because required 3 times in the calculation
            operations_triggerlevels_masked_h5h7pg = operations_triggerlevels_casted_h5h7pg[:, required_h7pg]
            temporary_h7pg = np.ones_like(required_h7pg, dtype='float32')

            ##Level if animal trigger level is between 'range' and 'le'
            temporary_h7pg[required_h7pg] = np.clip((animal_triggervalues_h7mask_h7pg[required_h7pg] - operations_triggerlevels_masked_h5h7pg[2, ...])/
                                                        (operations_triggerlevels_masked_h5h7pg[3, ...] - operations_triggerlevels_masked_h5h7pg[2, ...]),0,1)
            ##Select the maximum across the h7 axis if the operation is triggered
            temporary_pg = np.max(temporary_h7pg,axis=0) * operation_triggered_pg   #mul by operation triggered so that level goes to 0 if operation is not triggered

            ##Select the maximum of the 'le' and 'ge' value
            level_h2pg[h2,...] = np.maximum(level_pg, temporary_pg)
        ##if there is no range then level is just 1 * triggered
        else:
            level_h2pg[h2,...] = level_h2pg[h2,...] * operation_triggered_pg

    return level_h2pg


def f1_mustering_required(application_level_h2tpg, husb_operations_muster_propn_h2tpg):
    ##Total mustering required for all operations
    musters_tpg = np.sum(application_level_h2tpg * husb_operations_muster_propn_h2tpg, axis=0)
    ##Round up to the next integer
    musters_tpg = np.ceil(musters_tpg)
    return musters_tpg


# def f_husbandry_component(level, treatment_units, requirements, association, axes_tup):
#     ##Number of treatment units for contract
#     units = treatment_units[association]
#     ##Infrastructure requirement for each animal class during the period
#     component = np.sum(level * units * requirements, axis=axes_tup)
#     return component


def f1_husbandry_requisites(level_htpg, treatment_units_h8tpg, husb_requisite_cost_h6tpg, husb_requisites_prob_h6htpg,a_h8_h):
    ##Number of treatment units for requisites
    if type(a_h8_h)==int:
        units_htpg = treatment_units_h8tpg[a_h8_h:a_h8_h+1] #so the h axis is kept
    else:
        units_htpg = treatment_units_h8tpg[a_h8_h]
    ##Labour requirement for each animal class during the period
    ##calculated using loop to reduce memory
    cost_tpg = 0
    for h in range(level_htpg.shape[0]):
        cost_tpg += np.sum(level_htpg[h] * units_htpg[h] * husb_requisite_cost_h6tpg *
                     husb_requisites_prob_h6htpg[:,h], axis = 0)
    return cost_tpg


def f1_husbandry_labour(level_htpg, treatment_units_h8tpg, units_per_labourhour_l2htpg, a_h8_h):
    ##Number of treatment units for contract
    if type(a_h8_h)==int:
        units_htpg = treatment_units_h8tpg[a_h8_h:a_h8_h+1] #so the h axis is kept
    else:
        units_htpg = treatment_units_h8tpg[a_h8_h]
    ##Labour requirement for each animal class during the period
    ##calculated using loop to reduce memory
    hours_l2tpg = 0
    for h2 in range(level_htpg.shape[0]):
        hours_l2tpg += fun.f_divide(level_htpg[h2] * units_htpg[h2] , units_per_labourhour_l2htpg[:,h2], dtype=level_htpg.dtype) #divide by units_per_labourhour_l2hpg because that is how many units can be done per hour e.g. how many sheep can be drenched per hr
    return hours_l2tpg


def f1_husbandry_infrastructure(level_htpg, husb_infrastructurereq_h1h2tpg):
    ##Infrastructure requirement for each animal class during the period
    ##calculated using loop to reduce memory
    infrastructure_h1tpg = 0
    for h2 in range(level_htpg.shape[0]):
        infrastructure_h1tpg += level_htpg[h2] * husb_infrastructurereq_h1h2tpg[:,h2]
    return infrastructure_h1tpg


def f1_contract_cost(application_level_h2tpg, treatment_units_h8tpg, husb_operations_contract_cost_h2tpg):
    ##Number of animal units for contract
    units_h2tpg = treatment_units_h8tpg[uinp.sheep['ia_h8_h2']]
    ##Contract cost for each animal class during the period
    cost_tpg = np.sum(application_level_h2tpg * units_h2tpg * husb_operations_contract_cost_h2tpg, axis=0)
    return cost_tpg


def f_husbandry(head_adjust, mobsize_pg, o_ffcfw_tpg, o_cfw_tpg, operations_triggerlevels_h5h7h2tpg, index_pg,
                age_start, period_is_shear_pg, period_is_wean_pg, gender, o_ebg_tpg, wool_genes,
                husb_operations_muster_propn_h2tpg, husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg,
                operations_per_hour_l2h2tpg, husb_operations_infrastructurereq_h1h2tpg,
                husb_operations_contract_cost_h2tpg, husb_muster_requisites_prob_h6h4tpg,
                musters_per_hour_l2h4tpg, husb_muster_infrastructurereq_h1h4tpg, a_t_g=np.array([0]),
                a_nyatf_b1g=0,period_is_joining_pg=False, animal_mated=False, scan_option=0, period_is_endmating_pg=False, dtype=None):
    ##An array of the trigger values for the animal classes in each period - these values are compared against a threshold to determine if the husb is required
    animal_triggervalues_h7tpg = f1_animal_trigger_levels(index_pg, age_start, period_is_shear_pg, period_is_wean_pg, gender,
                            o_ebg_tpg, wool_genes, period_is_joining_pg, animal_mated, scan_option, period_is_endmating_pg).astype(dtype)
    ##The number of treatment units per animal in each period - each slice has a different unit e.g. mobsize, nyatf etc the treatment unit can be selected and applied for a given husb operation
    treatment_units_h8tpg = f_treatment_unit_numbers(head_adjust, mobsize_pg, o_ffcfw_tpg, o_cfw_tpg, a_nyatf_b1g).astype(dtype)
    ##Is the husb operation triggered in the period for each class
    operation_triggered_h2tpg = f1_operations_triggered(animal_triggervalues_h7tpg, operations_triggerlevels_h5h7h2tpg, a_t_g)
    ##The level of the operation in each period for the class of livestock (proportion of animals that receive treatment) - this accounts for the fact that just because the operation is triggered the operation may not be done to all animals
    application_level_h2tpg = f1_application_level(operation_triggered_h2tpg, animal_triggervalues_h7tpg, operations_triggerlevels_h5h7h2tpg, a_t_g)
    ##The number of times the mob must be mustered
    mustering_level_h4tpg = f1_mustering_required(application_level_h2tpg, husb_operations_muster_propn_h2tpg)[na,...] #needs a h4 axis for the functions below
    ##The cost of requisites for the operations
    operations_requisites_cost_tpg = f1_husbandry_requisites(application_level_h2tpg, treatment_units_h8tpg, husb_requisite_cost_h6tpg, husb_operations_requisites_prob_h6h2tpg, uinp.sheep['ia_h8_h2'])
    ##The labour requirement for the operations
    operations_labourreq_l2tpg = f1_husbandry_labour(application_level_h2tpg, treatment_units_h8tpg, operations_per_hour_l2h2tpg, uinp.sheep['ia_h8_h2'])
    ##The infrastructure requirements for the operations
    operations_infrastructurereq_h1tpg = f1_husbandry_infrastructure(application_level_h2tpg, husb_operations_infrastructurereq_h1h2tpg)
    ##Contract cost for husbandry
    contract_cost_tpg = f1_contract_cost(application_level_h2tpg, treatment_units_h8tpg, husb_operations_contract_cost_h2tpg)
    ##The cost of requisites for mustering
    mustering_requisites_cost_tpg = f1_husbandry_requisites(mustering_level_h4tpg, treatment_units_h8tpg, husb_requisite_cost_h6tpg, husb_muster_requisites_prob_h6h4tpg, uinp.sheep['ia_h8_h4'])
    ##The labour requirement for mustering
    mustering_labourreq_l2tpg = f1_husbandry_labour(mustering_level_h4tpg, treatment_units_h8tpg, musters_per_hour_l2h4tpg, uinp.sheep['ia_h8_h4'])
    ##The infrastructure requirements for mustering
    mustering_infrastructurereq_h1tpg = f1_husbandry_infrastructure(mustering_level_h4tpg, husb_muster_infrastructurereq_h1h4tpg)
    ##Total cost of husbandry
    husbandry_cost_tpg = operations_requisites_cost_tpg + mustering_requisites_cost_tpg + contract_cost_tpg
    ##Labour requirement for husbandry
    husbandry_labour_l2tpg = operations_labourreq_l2tpg + mustering_labourreq_l2tpg
    ##infrastructure requirement for husbandry
    husbandry_infrastructure_h1tpg = operations_infrastructurereq_h1tpg + mustering_infrastructurereq_h1tpg
    return husbandry_cost_tpg, husbandry_labour_l2tpg, husbandry_infrastructure_h1tpg


##################
#post processing #
##################

##Method 1 (still used)- add p and v axis together then sum p axis - this may be a good method for faster computers with more memory
def f1_p2v_std(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
            period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
    ## convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
    try:
        days_period_p = days_period_p.astype('float32')
    except AttributeError:
        pass
    ##mul everything
    production_ftvpany = (production_p * numbers_p * days_period_p * period_is_tvp
                          * on_hand_tvp * (dvp_pointer_p == index_vp) * (a_any1_p == index_any1tvp)
                          * (a_any2_p == index_any2any1tvp))
    ## sum along p axis to leave just v axis (sumadj is to handle nsire that has a p8 axis at the end)
    return np.sum(production_ftvpany, axis=sinp.stock['i_p_pos']-sumadj)


# ##Method 2 (fastest)- sum sections of p axis to leave v (almost like sum if) this is fast because don't need p and v axis in same array
# def f1_p2v(production_p, dvp_pointer_p=1, numbers_p=1, on_hand_tp=True, days_period_p=1, period_is_tp=True, a_any1_p=1, index_any1tp=1, a_any2_p=1, index_any2any1tp=1):
#     #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     try:
#         days_period_p = days_period_p.astype('float32')
#     except AttributeError:
#         pass
#     ##mul everything - add t,f and p6 axis
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tp
#                         * on_hand_tp * (a_any1_p==index_any1tp)
#                         * (a_any2_p==index_any2any1tp))
#     ##convert p to v - info at this link https://stackoverflow.com/questions/50121980/numpy-conditional-sum
#     ##basically we are summing the p axis for each dvp. the tricky part (which has caused the requirement for the loops) is that dvp pointer is not the same for each axis e.g. dvp is effected by e axis.
#     ##so we need to loop though all the axis in the dvp and sum p and assign to a final array.
#     ##if the axis is size 1 (ie singleton) then we want to take all of that axis ie ':' because just because the dvp pointer has singleton doesn't mean param array has singleton so need to take all slice of the param (unless that is an active dvp axis because that means dvp timing may differ for different slices along that axis so it must be summed in the loop)
#     shape = production_ftpany.shape[0:sinp.stock['i_p_pos']] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[sinp.stock['i_p_pos']+1:]  # bit messy because need v t and all the other axis (but not p)
#     result=np.zeros(shape).astype('float32')
#     shape = dvp_pointer_p.shape
#     for a1 in range(shape[-14]):
#         a1_slc = slice(a1,a1+1) if shape[-14]>1 else slice(0,None) #used for param because we want to keep axis
#         for e1 in range(shape[-13]):
#             e1_slc = slice(e1, e1 + 1) if shape[-13] > 1 else slice(0, None)
#             for b1 in range(shape[-12]):
#                 b1_slc = slice(b1, b1 + 1) if shape[-12] > 1 else slice(0, None)
#                 for n in range(shape[-11]):
#                     n_slc = slice(n, n + 1) if shape[-11] > 1 else slice(0, None)
#                     for w in range(shape[-10]):
#                         w_slc = slice(w, w + 1) if shape[-10] > 1 else slice(0, None)
#                         for z in range(shape[-9]):
#                             z_slc = slice(z, z + 1) if shape[-9] > 1 else slice(0, None)
#                             for i in range(shape[-8]):
#                                 i_slc = slice(i, i + 1) if shape[-8] > 1 else slice(0, None)
#                                 for d in range(shape[-7]):
#                                     d_slc = slice(d, d + 1) if shape[-7] > 1 else slice(0, None)
#                                     for a0 in range(shape[-6]):
#                                         a0_slc = slice(a0, a0 + 1) if shape[-6] > 1 else slice(0, None)
#                                         for e0 in range(shape[-5]):
#                                             e0_slc = slice(e0, e0 + 1) if shape[-5] > 1 else slice(0, None)
#                                             for b0 in range(shape[-4]):
#                                                 b0_slc = slice(b0, b0 + 1) if shape[-4] > 1 else slice(0, None)
#                                                 for x in range(shape[-3]):
#                                                     x_slc = slice(x, x + 1) if shape[-3] > 1 else slice(0, None)
#                                                     for y in range(shape[-2]):
#                                                         y_slc = slice(y, y + 1) if shape[-2] > 1 else slice(0, None)
#                                                         for g in range(shape[-1]):
#                                                             g_slc = slice(g, g + 1) if shape[-1] > 1 else slice(0, None)
#                                                             #calc the v length and only assign to that
#                                                             len_v = len(np.unique(dvp_pointer_p[:, a1, e1, b1, n, w, z, i, d, a0, e0, b0, x, y, g]))
#                                                             result[..., :len_v, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc] \
#                                                                 = np.add.reduceat(production_ftpany[..., a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
#                                                                                   , np.r_[0, np.where(np.diff(dvp_pointer_p[:, a1, e1, b1, n, w, z, i, d, a0, e0, b0, x, y, g]))[0] + 1], axis=sinp.stock['i_p_pos']) #np.r_ basically concats two 1d arrays (so here we are just adding 0 to the start of the array)
#     return result

##Method 2b - (similar speed to method 2) loop over v and other axis active in dvp pointer, mask p for the current v and sum. This method
# has replaced method 2 because this handles 0 day dvps.
def f1_p2v(production_p, dvp_pointer_p, numbers_p=np.array([1]), on_hand_tp=True, days_period_p=np.array([1]),
            period_is_tp=np.array([True]), a_any1_p=np.array([1]), index_any1tp=1, a_any2_p=np.array([1]), index_any2any1tp=1):
    try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
    except AttributeError:
        pass
    p_pos=sinp.stock['i_p_pos']
    ##broadcast everything - so that i can create final array and mask p
    final_shape_vp = np.broadcast(production_p, numbers_p, dvp_pointer_p, index_any1tp, index_any2any1tp, on_hand_tp, period_is_tp).shape
    ###remove p axis
    final_shape = final_shape_vp[:p_pos] + (np.max(dvp_pointer_p)+1,) + final_shape_vp[p_pos+1:]  # bit messy because need v t and all the other axis (but not p)
    ##initilise final array - it is assigned to by slice
    final=np.zeros(final_shape).astype('float32')

    ##broadcast arrays to dvp shape - needs all the axis that are active in the loop
    shape = dvp_pointer_p.shape
    a_any1_p = np.broadcast_to(a_any1_p, shape)
    a_any2_p = np.broadcast_to(a_any2_p, shape)
    production_p = np.broadcast_to(production_p, np.broadcast(production_p, dvp_pointer_p).shape)
    numbers_p = np.broadcast_to(numbers_p, np.broadcast(numbers_p, dvp_pointer_p).shape)
    days_period_p = np.broadcast_to(days_period_p, np.broadcast(days_period_p, dvp_pointer_p).shape)
    on_hand_tp = np.broadcast_to(on_hand_tp, np.broadcast(on_hand_tp, dvp_pointer_p).shape) #bit more complex because need to account for axes that 'shape' doesn't have.
    period_is_tp = np.broadcast_to(period_is_tp, np.broadcast(period_is_tp, dvp_pointer_p).shape)

    ##loop over each axis in dvp_pointer. Loop over all axis because active axis change for dams and offs. So this will handle if other axis get activated at a later date.
    for v in range(np.max(dvp_pointer_p)+1):
        for a1 in range(shape[-14]):
            a1_slc = slice(a1,a1 + 1) if shape[-14] > 1 else slice(0,None)  # used for param because we want to keep axis
            for e1 in range(shape[-13]):
                e1_slc = slice(e1,e1 + 1) if shape[-13] > 1 else slice(0,None)
                for b1 in range(shape[-12]):
                    b1_slc = slice(b1,b1 + 1) if shape[-12] > 1 else slice(0,None)
                    for n in range(shape[-11]):
                        n_slc = slice(n,n + 1) if shape[-11] > 1 else slice(0,None)
                        for w in range(shape[-10]):
                            w_slc = slice(w,w + 1) if shape[-10] > 1 else slice(0,None)
                            for z in range(shape[-9]):
                                z_slc = slice(z,z + 1) if shape[-9] > 1 else slice(0,None)
                                for i in range(shape[-8]):
                                    i_slc = slice(i,i + 1) if shape[-8] > 1 else slice(0,None)
                                    for d in range(shape[-7]):
                                        d_slc = slice(d,d + 1) if shape[-7] > 1 else slice(0,None)
                                        for a0 in range(shape[-6]):
                                            a0_slc = slice(a0,a0 + 1) if shape[-6] > 1 else slice(0,None)
                                            for e0 in range(shape[-5]):
                                                e0_slc = slice(e0,e0 + 1) if shape[-5] > 1 else slice(0,None)
                                                for b0 in range(shape[-4]):
                                                    b0_slc = slice(b0,b0 + 1) if shape[-4] > 1 else slice(0,None)
                                                    for x in range(shape[-3]):
                                                        x_slc = slice(x,x + 1) if shape[-3] > 1 else slice(0,None)
                                                        for y in range(shape[-2]):
                                                            y_slc = slice(y,y + 1) if shape[-2] > 1 else slice(0,None)
                                                            for g in range(shape[-1]):
                                                                g_slc = slice(g,g + 1) if shape[-1] > 1 else slice(0,None)
                                                                ##build mask - which p's in current v
                                                                mask_p = dvp_pointer_p[:, a1, e1, b1, n, w, z, i, d, a0, e0, b0, x, y, g]==v
                                                                ##calculation - using mask_p to make it faster.
                                                                final[...,v, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]\
                                                                    = np.sum(production_p[...,mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                             * numbers_p[..., mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                             * days_period_p[..., mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                             * period_is_tp[...,mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                             * on_hand_tp[...,mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]
                                                                             * (a_any1_p[...,mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]==index_any1tp)
                                                                             * (a_any2_p[...,mask_p, a1_slc, e1_slc, b1_slc, n_slc, w_slc, z_slc, i_slc, d_slc, a0_slc, e0_slc, b0_slc, x_slc, y_slc, g_slc]==index_any2any1tp)
                                                                             , axis=p_pos)
    return final

##method 6 - masked arrays (slow)
# def f1_p2v_loop2(production_p, dvp_pointer_p, numbers_p=1, on_hand_tp=True, days_period_p=np.array([1]),
#             period_is_tp=np.array([True]), a_any1_p=np.array([1]), index_any1tp=1, a_any2_p=np.array([1]), index_any2any1tp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     p_pos=sinp.stock['i_p_pos']
#     ##broadcast everything - so that i can create final array and mask p
#     final_shape_vp = np.broadcast(production_p, index_any1tp, index_any2any1tp, on_hand_tp, period_is_tp).shape
#     ###remove p axis
#     final_shape = final_shape_vp[:p_pos] + (np.max(dvp_pointer_p)+1,) + final_shape_vp[p_pos+1:]  # bit messy because need v t and all the other axis (but not p)
#
#     ##broadcast arrays to dvp shape - needs all the axis that are active in the loop
#     shape = dvp_pointer_p.shape
#     a_any1_p = np.broadcast_to(a_any1_p, shape)
#     a_any2_p = np.broadcast_to(a_any2_p, shape)
#     days_period_p = np.broadcast_to(days_period_p, shape)
#     on_hand_tp = np.broadcast_to(on_hand_tp, np.broadcast(on_hand_tp, dvp_pointer_p).shape) #bit more complex than above because need to account for t axis but only if it exists hence broadcast before getting shape.
#     period_is_tp = np.broadcast_to(a_any1_p, np.broadcast(period_is_tp, dvp_pointer_p).shape)
#
#     ##initilise final array - it is assigned to by slice
#     final=np.zeros(final_shape).astype('float32')
#
#     ##loop over each axis in dvp_pointer. Loop over all axis because active axis change for dams and offs. So this will handle if other axis get activated at a later date.
#     for v in range(np.max(dvp_pointer_p)+1):
#         ##build mask - which p's in current v
#         mask_p = dvp_pointer_p==v
#         ##build masked arrays
#         mask_production_p = np.broadcast_to(mask_p, production_p.shape)
#         ma_production_p = np.ma.masked_array(production_p, mask_production_p)
#         mask_numbers_p = np.broadcast_to(mask_p, numbers_p.shape)
#         ma_numbers_p = np.ma.masked_array(numbers_p, mask_numbers_p)
#         mask_days_period_p = np.broadcast_to(mask_p, days_period_p.shape)
#         ma_days_period_p = np.ma.masked_array(days_period_p, mask_days_period_p)
#         mask_on_hand_tp = np.broadcast_to(mask_p, on_hand_tp.shape)
#         ma_on_hand_tp = np.ma.masked_array(on_hand_tp, mask_on_hand_tp)
#         mask_a_any1_p = np.broadcast_to(mask_p, a_any1_p.shape)
#         ma_a_any1_p = np.ma.masked_array(a_any1_p, mask_a_any1_p)
#
#         ##calculation - using mask_p to make it faster.
#         final\
#                 = np.ma.sum(ma_production_p
#                      * ma_numbers_p
#                      * ma_days_period_p
#                      # * period_is_tp
#                      * ma_on_hand_tp
#                      * (ma_a_any1_p==index_any1tp)
#                      # * (a_any2_p==index_any2any1tp)
#                      , axis=sinp.stock['i_p_pos'])
#     return final


# ##Method 5 numexpr (slow - even with fv33n33 it is much slower than the current method
# import numexpr as ne
# def f1_p2v_5(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
#             period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
#     ## convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     try:
#         days_period_p = days_period_p.astype('float32')
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_any1_p==index_any1tvp)
#                         * (a_any2_p==index_any2any1tvp))* (dvp_pointer_p == index_vp)
#     ##mul everything and sum - both methods below are slow. but the least amount of calc in the ne.eval the better.
#     production_ftvany = ne.evaluate("sum(production_ftpany,axis=5)")
#     # production_ftvany = ne.evaluate("sum(production_p * numbers_p * days_period_p * period_is_tvp * on_hand_tvp * (dvp_pointer_p == index_vp) * (a_any1_p == index_any1tvp) * (a_any2_p == index_any2any1tvp),axis=5)")
#
#     return production_ftvany


# ##Method 4 - loop over v and sum p - this save p and v axis being on the same array but requires lots of looping so isn't much faster
# def f1_p2v_loop(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1,
#             period_is_tvp=True, a_any1_p=1, index_any1tvp=1, a_any2_p=1, index_any2any1tvp=1, sumadj=0):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
#     except AttributeError:
#         pass
#     ##mul everything
#     production_ftpany = (production_p * numbers_p * days_period_p * period_is_tvp
#                         * on_hand_tvp * (a_any1_p==index_any1tvp)
#                         * (a_any2_p==index_any2any1tvp))
#
#     shape = production_ftpany.shape[0:3] + (np.max(dvp_pointer_p)+1,) + production_ftpany.shape[4:]  # bit messy because need v t and all the other axis (but not p)
#     final=np.zeros(shape).astype('float32')
#     for i in range(np.max(dvp_pointer_p)+1):
#         temp_prod = np.sum(production_ftpany * (dvp_pointer_p==i), axis=sinp.stock['i_p_pos'])
#         final[:,:,:,i,...] = temp_prod  #asign to correct v slice
#     return final

# ##Method 3 - use groupby to sum p, this means p and v don't exist on the same array - not as fast as method 2
# import numpy_indexed as npi
# def f1_p2v_groupby(production_p, dvp_pointer_p=1, index_vp=1, numbers_p=1, on_hand_tvp=True, days_period_p=1, period_is_tvp=True, a_ev_p=1, index_ftvp=1, a_p6_p=1, index_p6ftvp=1):
#     try: days_period_p = days_period_p.astype('float32')  #convert int to float because float32 * int32 results in float64. Need the try/except because when days period is the default 1 it can't be converted to float (because int object is not numpy)
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
#                                                                     production_ftpany[:, :, :, :, :, e1:e1+1, :, :, :, :, :, :, :, :, :, :, :, g:g+1], axis=sinp.stock['i_p_pos'])[1]
#     return result

def f1_p2v_adj(production_v, a_p_v, a_v_p,update_production_v=None):
    """
    Adjustment for 0 day dvps.

    Mostly when a dvp is 0 days long (clashes with another dvp) we want it to have 0 production (e.g. 0 mei).
    This is how it works in the p2v function. However, in some cases (numbers and ffcfw) we want values in the
    0 day dvps because we need the animals to transfer to the next dvp. ffcfw is used in the numbers distribution
    so it needs numbers in case season start or condense dvp clash with another dvp.
    Numbers need to do a 1:1 transfer so the start and end numbers are the same.
    ffcfw needs to allow distribution to work correctly. So the condense dvp is always first in a clash (this is
    handled in the construction of the dvps) and the non 0 day dvp weights are used to populate the 0 day dvp.

    :param production_v: production param to update
    :param update_production_v: array used to update production array (if this is not provided then it uses the production array)
    :param a_p_v: association between p and v with v axis
    :param a_v_p: association between p and v with p axis
    """
    p_pos = sinp.stock['i_p_pos']
    ##if no array to update from then use the production array. This exists because numbers end are updated using numbers start to ensure numbers have a 1:1 transfer
    if type(update_production_v) == type(None):
        update_production_v = production_v
    ##work out which dvps are 0 day length
    dvp_0days_mask_va1e1b1nwzida0e0b0xyg1 = np.roll(a_p_v, shift=-1, axis=p_pos)==a_p_v
    ##build association to the dvp that fills each blank dvp
    a_v_v = np.take_along_axis(a_v_p, a_p_v, axis=p_pos)
    a_v_v = np.broadcast_to(a_v_v, update_production_v.shape) #handle cases where production has a t axis.
    ##update production in the 0 day dvps.
    production_v = fun.f_update(production_v, np.take_along_axis(update_production_v, a_v_v, axis=p_pos),
                                dvp_0days_mask_va1e1b1nwzida0e0b0xyg1) #intentionally numbers start - want numbers start and end to be the same for 0 day dvp so that 1:1 transfer happens.
    return production_v



def f1_cum_dvp(arr,dvp_pointer,axis=0,shift=0):
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


def f1_cum_sum_dvp(arr,dvp_pointer,axis=0,shift=0):
    '''This function does accumulative sum but it resets at each dvp.
    '''
    final = np.zeros_like(arr)
    for i in range(np.max(dvp_pointer)+1):  #plus 1 so that the last dvp is counted for
        arr1 = arr * (dvp_pointer==i) #sets the p slices to 0 if not in the given dvp
        arr1 = np.roll(arr1,shift,axis) #this is only used for the dams on hand calculation, this rolls the period is sale array 1 unit along the p axis.
                                        # This is required so that period is onhand == true in the period that sale occurs and false after that.
                                        # Because sale occurs at the end of a given period so the sheep are technically onhand for the period sale occurs.
        arr1 = np.cumsum(arr1,axis=axis)
        arr1 = arr1 * (dvp_pointer==i) #sets the cum max to 0 for other dvp not of interest
        final += arr1
    return final


def f1_lw_distribution(ffcfw_dest_w8g, ffcfw_source_w8g, mask_dest_wg=1, index_w8=None, dvp_type_next_tvgw=0, vtype=0): #, w_pos, i_n_len, i_n_fvp_period, dvp_type_next_tvgw=0, vtype=0):
    """Distribute animals between periods when the animals are changing on the period junction. This change can
    be either 1. condensing at prejoining when animals are 'condensed' from the final number of LW profiles back to
    the initial number or 2. averaging animals at season start when weights at the end of all the seasons are
    averaged to become the start weight for the next season.

    The animals are distributed to the 2 nearest neighbours, rather than a distribution across all destination LWs.
    The aim is that the average weight of the animals in the next period is equal to the average weight at the
    end of this period. However, this is only possible if the source weight being distributed is between the
    lowest and highest destination weights.

    If the weight is above the highest destination weight then the weight is rounded down & the extra weight is
    effectively lost. If the weight is below the lowest weight then that animal is not transferred to the next
    period and the animal is effectively lost.

    When the distribution is for condensing/pre-joining the destination weights are ffcfw_end for the condensed slices (w9).
    This is the condensed values of ffcfw_end (of this DVP) rather than ffcfw_start (of the next DVP) because ffcfw_start
    includes the averaging of LW that occurs as part of f_period_start_prod when the period is prejoin.
    Clustering based on the condensed weights of lw_start (after averaging across the b1 & e1 axes) would mean
    that the clustering is occurring based on the proportion of animals in the b1 & e1 axes in the generator
    (which are based on those animals following the same nutrition profile (H, M or L for that class)), whereas
    they may have been differentially managed in the matrix, hence the requirement to distribute based on ffcfw_end.
    The effect if the groups are clustered (i.e. not scanned) is to keep the heavier singles with the heavier twins,
    rather than distribute the heavy twins with the average singles (which might have a more similar weight).
    If this didn't occur the LW distribution would allow defacto ‘scanning’ through LW distribution. Although
    differential management based on LW can be done in practice, in reality there is a distribution of weight
    (within each class-dry,single,twin) so the split would be imprecise, whereas in the generator all the animals
    of a class (D,S,Tw) are all the same weight (for a given nutrition profile), so the generator could split
    them accurately for BTRT based on LW.

    Note: A distribution of liveweight at the end of a period would be technically correct i.e. a given class of
    animal that are offered a given feed supply for a period of time will result in spread of final weights, even
    if all the animals started at the same weight. However, in AFO it only results in a single weight. A fix could
    be to add a normal distribution to the final weights, however, this would significantly complicate debugging
    because it would make following a class of animals between periods very difficult.

    :param ffcfw_dest_w8g: The LW at the end of the DVP for the animals that define each w9 constraint (in w8 axis position)
    :param ffcfw_source_w8g: The LW at the end of the DVP, of the animals to be distributed
    :param mask_dest_wg: mask the destination slices for the distribution
    :param index_w8:
    :param dvp_type_next_tvgw: the DVP-type of the next DVP
    :param vtype: the distribution is only returned if the next DVP is of this vtype
    :return distribution_w8gw9: A v array with a proportion of each tw8g8 decision variable passing into each w9g9 constraint
    """
    ##set dtype
    dtype = ffcfw_dest_w8g.dtype

    ##ffcfw_destn should be the same for clustered W (but due to the season weighted average the decimals can be a tiny bit different).
    if index_w8 is not None:
        ###create an association that points to the first w slice that is not masked (can be masked by either clustering or mask_nut)
        a_wcluster_w8g = np.maximum.accumulate(index_w8 * mask_dest_wg, axis=sinp.stock['i_w_pos'])
        ###for each w, set it to the first weight in the cluster e.g. if w[0,1,2] are in the same w cluster then they will all be set to the w[0] value.
        ffcfw_dest_w8g = np.take_along_axis(ffcfw_dest_w8g, a_wcluster_w8g, axis=sinp.stock['i_w_pos'])

    ## Move w axis of dest_w8g to -1 and f_expand to retain the original ‘w’ as a singleton
    ffcfw_dest_wgw9 = fun.f_expand(np.moveaxis(ffcfw_dest_w8g, sinp.stock['i_w_pos'],-1), sinp.stock['i_n_pos']-1, right_pos=sinp.stock['i_z_pos']-1)
    ## create index for w9 based on shape of the (now) last axis (to be used later)
    index_w9 = np.arange(ffcfw_dest_wgw9.shape[-1])

    ## Create the distribution (result) array because it is assigned in slices
    distribution_nearest_w8gw9 = np.zeros_like(np.broadcast_arrays(ffcfw_source_w8g[...,na],ffcfw_dest_wgw9)[0]) #broadcast array returns list of two arrays so need to select one
    distribution_nextnearest_w8gw9 = np.zeros_like(distribution_nearest_w8gw9)

    ## Find the index of the destination slice that is nearest to the source weight
    #todo may be able to save memory here by storing sign(diff_w8gw9).astype(int) for later steps
    # then calculating in place the absolute value of diff_w8w9 = np.abs(diff_w8gw9, out=diff_w8gw9)
    # the idea being that it replaces a w8gw9-float with a w8gw9-int
    diff_w8gw9 = ffcfw_dest_wgw9 - ffcfw_source_w8g[...,na]
    diff_abs_w8gw9 = np.abs(diff_w8gw9)
    nearestw9_idx_w8g = np.argmin(diff_abs_w8gw9,axis = -1)

    ## If an index_w8 has been provided (because it is a square w8:w9) then test the nearest for equality
    ### If the source weight matches the destination then set index to the slice of the first clustered weight
    ### (so the slice distributes to itself or to the equivalent clustered slice)
    ### Covers two situations:
    ### 1. if destination weights are replicated
    ### 2. if a destination weight is masked
    ### in both cases nearestw9_idx will point to the next lowest unmasked weight e.g. if w9[0] and w9[54] are
    ### the same weight as w8[55] this code will make w8[55] distribute to w9[54] instead of w9[0].
    ### 8May22 unsure if this is achieving anything important
    if index_w8 is not None:
        ###create an association that points to the first w slice that is not masked (can be masked by either clustering or mask_nut)
        a_wcluster_w8g = np.maximum.accumulate(index_w8 * mask_dest_wg, axis=sinp.stock['i_w_pos'])
        ###points to the clustered/unmasked w to ensure that 1:1 distributing doesn't occur if a w9 slice is masked
        nearestw9_idx_w8g = fun.f_update(nearestw9_idx_w8g, a_wcluster_w8g, np.isclose(ffcfw_source_w8g, ffcfw_dest_w8g))

    ## The nearest destination weight for each source weight & the difference from each w8
    nearestw9_w8gw = np.take_along_axis(ffcfw_dest_wgw9, nearestw9_idx_w8g[...,na], axis=-1)
    diff_nearest_w8gw = np.take_along_axis(diff_w8gw9, nearestw9_idx_w8g[...,na], axis=-1)  #alternate calc: diff_nearest_w8gw = nearestw9_w8gw - ffcfw_source_w8g[...,na]

    ## Determine the index of the next nearest destination slice that is on the opposite side of the source (using masked array)
    ### mask the values for which the difference is the same sign as the difference of the nearest.
    mask = np.sign(diff_w8gw9) == np.sign(diff_nearest_w8gw)
    shape = np.broadcast(diff_abs_w8gw9, mask).shape
    diff_abs_w8gw9 = np.broadcast_to(diff_abs_w8gw9, shape)
    mask = np.broadcast_to(mask, shape)
    next_nearestw9_idx_w8g = np.argmin(np.ma.masked_array(diff_abs_w8gw9, mask), axis = -1)

    ## If an index_w8 has been provided then test for equality (as per nearest)
    if index_w8 is not None:
        a_wdest_w8 = np.maximum.accumulate(index_w8*mask_dest_wg, axis=sinp.stock['i_w_pos']) #points to the starting w to ensure that 1:1 distributing doesn't occur if a w9 slice is masked
        next_nearestw9_idx_w8g = fun.f_update(next_nearestw9_idx_w8g, a_wdest_w8, np.isclose(ffcfw_source_w8g, ffcfw_dest_w8g))

    ## the next_nearest destination weight
    next_nearestw9_w8gw = np.take_along_axis(ffcfw_dest_wgw9, next_nearestw9_idx_w8g[...,na], axis=-1)

    ## Calculate the proportion distributed to the nearest and assign to that w9 slice
    ### Handle the special cases in f_divide (option=1) where source and destination weights are the same,
    ### weights have converged or the dest and source weight is 0 for all slices (e.g. if animals don't exist or distribution doesn't occur in the dvp)
    #### nearest
    proportion = fun.f_divide(ffcfw_source_w8g[...,na] - next_nearestw9_w8gw
                              , nearestw9_w8gw - next_nearestw9_w8gw, dtype=dtype, option=1)
    # handle situation when the destination weights are replicated but source is not (not sure that this can occur)
    proportion = fun.f_update(proportion, 1, np.isclose(nearestw9_w8gw, next_nearestw9_w8gw))
    np.put_along_axis(distribution_nearest_w8gw9, nearestw9_idx_w8g[...,na], proportion, axis=-1)
    #### next nearest
    proportion = fun.f_divide(nearestw9_w8gw - ffcfw_source_w8g[...,na]
                              , nearestw9_w8gw - next_nearestw9_w8gw, dtype=dtype, option=1)
    np.put_along_axis(distribution_nextnearest_w8gw9, next_nearestw9_idx_w8g[...,na], proportion, axis=-1)

    ## Handle the special cases where source weight is less than the lowest destination weight
    ### the light animals are transferred such that total LW remains the same prior to and after the distribution.
    ### therefore the number of animals is reduced during the transfer by the ratio: source wt / lowest destination wt.
    ### to transfer the full number of the light animals the minimum destination weight will need to be altered.
    ratio_w8gw = fun.f_divide(ffcfw_source_w8g, np.min(ffcfw_dest_wgw9, axis=-1), dtype=dtype)[...,na]
    ### where the ratio is below 1 it is applied to the nearest w9 slice
    mask_w8gw9 = (ratio_w8gw < 1) * (nearestw9_idx_w8g[...,na] == index_w9)
    distribution_nearest_w8gw9 = fun.f_update(distribution_nearest_w8gw9, ratio_w8gw, mask_w8gw9)

    ## Combine the values into the return variable
    ### clip (0 to 1) to handle the special case where source weight > the maximum destination weight
    distribution_w8gw9 = np.clip(distribution_nearest_w8gw9 + distribution_nextnearest_w8gw9,0,1)
    # distribution_error = np.any(np.sum(distribution_w8gw9, axis=-1)>1)

    ##Set defaults for DVPs that don’t require distributing to 1 (these are masked later to remove those that are not required)
    distribution_w8gw9 = fun.f_update(distribution_w8gw9, np.array([1],dtype=np.float32), dvp_type_next_tvgw!=vtype) #make 1 an numpy array so it can be float32 to make f_update more data effcient.
    return distribution_w8gw9


def f1_create_production_param(group, production_vg, a_kcluster_vg_1=1, index_ktvg_1=1, a_kcluster_vg_2=1, index_kktvg_2=1, numbers_start_vg=1, mask_vg=True, pos_offset=0):
    '''Can convert total production to per animal production including impact of death if numbers have been included.
    Apply the k clustering and collapse the e, b & d axes
    If numbers_start are not included then only applies k clustering - this is usually done if production is already per head'''
    if group=='sire':
        return fun.f_divide(production_vg, numbers_start_vg, dtype=production_vg.dtype) * mask_vg
    elif group=='dams':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * mask_vg
                                  , axis = (sinp.stock['i_b1_pos']-pos_offset, sinp.stock['i_e1_pos']-pos_offset), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1),
                                     axis=(sinp.stock['i_b1_pos']-pos_offset, sinp.stock['i_e1_pos']-pos_offset), keepdims=True), dtype=production_vg.dtype)
    elif group=='offs':
        return fun.f_divide(np.sum(production_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2) * mask_vg
                                  , axis = (sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True)
                            , np.sum(numbers_start_vg * (a_kcluster_vg_1 == index_ktvg_1) * (a_kcluster_vg_2 == index_kktvg_2),
                                     axis=(sinp.stock['i_d_pos'], sinp.stock['i_b0_pos'], sinp.stock['i_e0_pos']), keepdims=True), dtype=production_vg.dtype)
