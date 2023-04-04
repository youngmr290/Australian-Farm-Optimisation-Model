# -*- coding: utf-8 -*-
"""
author: young

This module covers the labour supply. Note the labour requirements for various aspects of the farming system
are calculated and documented in the relevant modules.

To capture the dynamics of labour, the year is broken into labour periods :cite:p:`RN89`. The supply of
labour in each period by each labour source is calculated, and the labour required by each farm
activity is determined and assigned to the given period/s.

The amount of labour available in each period depends on the number of labour units and the hours worked
each day. Labour can be supplied by three sources:

#. Casual staff – In the unrestricted model, casual staff can come and go at any time throughout the
   year as required. However, the user can fix the number of casual staff employed
   during each period of the year.

#. Permanent staff – Permanent staff work on the property all year (with an allocation for leave).

#. Manager staff (commonly the farm owner) – The farm manager works on the property all year. They control
   the overall farm plan and thus spend a fixed amount of time each quarter on farm planning, learning,
   record-keeping, purchasing and selling, and other office work.

Farm labour tasks can be allocated to a specific labour source where required. For example, farm planning must
be completed by manager staff. Any labour source can complete unallocated tasks. To realistically reflect the
labour hierarchy, casual and permanent staff both require a certain amount of supervision from the farm manager.
The proportion of supervision is specified separately for seeding and harvesting. This is because during seeding
and harvest it is likely that less supervision is required. Casual staff are generally less experienced and/or
acquainted with the farm operation than permanent staff and thus require more supervision.

The importance of timeliness and the high labour requirement of seeding and harvest means staff often
work longer days during those periods :cite:p:`RN89`. To accommodate this, the user specifies the hours
worked by each type of staff on the weekdays and weekends for both standard periods and seeding and harvest periods.

The farm manager and permanent staff have four weeks of holiday each year. The holiday timing is flexible (optimised
by AFO). This is because managers and permanent staff tend to have a less defined schedule, often taking multiple
smaller holidays during the year or returning to the farm during holidays to check on things.
Additionally, in AFO, permanent and casual staff require supervision from the manager which means if the manager
is forced to take their holidays in one big chunk the model may not be able to access labour resulting in
inconsistencies if the period dates change.
All labour sources take days off for Christmas, New Year’s Day, and Easter. Permanent staff are
also allocated a certain number of sick days per year. The user has the ability to alter the length
and timing of worker leave

Casual staff are paid on a per hour basis and the manager and permanent staff are paid an annual wage.
All labour costs include superannuation and workers’ compensation insurance.
"""
#python modules
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

#AFO modules
# from LabourInputs import *
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Periods as per
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import Finance as fin


###################################################################
# make a df containing labour availability for each labour period #
###################################################################
na = np.newaxis


def f_labour_general(params,r_vals):
    '''
    Calculates labour supply, labour cost and supervision requirements.



    '''
    ###########################
    #inputs and initialisation#
    ###########################
    
    ##season inputs through input func
    harv_date_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)

    ##initialise period data
    lp_p5z = per.f_p_dates_df().values
    lp_start_p5z = per.f_p_dates_df().iloc[:-1].values#
    lp_end_p5z = per.f_p_dates_df().iloc[1:].values#
    lp_len_p5z = (lp_end_p5z - lp_start_p5z)

    ########
    #leave #
    ########
    
    ##manager leave
    leave_days = np.array([pinp.labour['leave_manager']])
    manager_leave_hours = int(leave_days * 2/7 * pinp.labour['daily_hours'].loc['weekends', 'Manager']
                            + leave_days * 5/7 * pinp.labour['daily_hours'].loc['weekdays', 'Manager'])

    ##perm leave
    ###normal leave
    leave_days = np.array([pinp.labour['leave_permanent']])
    perm_leave_hours = int(leave_days * 2/7 * pinp.labour['daily_hours'].loc['weekends', 'Permanent']
                         + leave_days * 5/7 * pinp.labour['daily_hours'].loc['weekdays', 'Permanent'])
    ###sick leave - x days split equally into each period
    perm_sick_leave_p5z = pinp.labour['sick_leave_permanent']/364 * lp_len_p5z

    ##########################
    #hours worked per period #
    ##########################
    
    ##determine possible labour days worked by the manager during the week and on weekend in a given labour periods. Note: casual labour has no leave.
    ###available days in the period minus leave multiplied by fraction of weekdays
    manager_weekdays_p5z = (lp_len_p5z) * 5/7
    perm_weekdays_p5z = (lp_len_p5z - perm_sick_leave_p5z) * 5/7
    cas_weekdays_p5z = (lp_len_p5z) * 5/7
    ###available days in the period minus leave multiplied by fraction of weekend days
    manager_weekend_p5z = (lp_len_p5z) * 2/7
    perm_weekend_p5z = (lp_len_p5z - perm_sick_leave_p5z) * 2/7
    cas_weekend_p5z = (lp_len_p5z) * 2/7

    ##set up stuff to calc hours work per period be each source
    seed_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    seeding_start_z = per.f_wet_seeding_start_date()
    seeding_end_z = seeding_start_z + np.sum(seed_period_lengths_pz, axis=0)
    seeding_occur_p5z =  np.logical_and(seeding_start_z <= lp_start_p5z, lp_start_p5z < seeding_end_z)
    harv_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1)
    harv_start_z = harv_date_z
    harv_end_z = harv_start_z + np.sum(harv_period_lengths_pz, axis=0)
    harv_occur_p5z =  np.logical_and(harv_start_z <= lp_start_p5z, lp_start_p5z < harv_end_z)

    ##manager hours
    ###seeding
    seeding_dailyhours = pinp.labour['daily_hours'].loc['seeding','Manager']
    manager_hrs_seeding = (lp_len_p5z) * seeding_occur_p5z * seeding_dailyhours
    ###harv
    harving_dailyhours = pinp.labour['daily_hours'].loc['harvest','Manager']
    manager_hrs_harv = (lp_len_p5z) * harv_occur_p5z * harving_dailyhours
    ###weekend hrs
    manager_hrs_weekend = manager_weekend_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekends','Manager']
    ###weekdays hrs
    manager_hrs_weekdays = manager_weekdays_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekdays','Manager']
    manager_hrs_total_p5z = manager_hrs_weekend + manager_hrs_weekdays + manager_hrs_seeding + manager_hrs_harv

    ##perm hours
    ###seeding
    seeding_dailyhours = pinp.labour['daily_hours'].loc['seeding','Permanent']
    perm_hrs_seeding = (lp_len_p5z - perm_sick_leave_p5z) * seeding_occur_p5z * seeding_dailyhours
    ###harv
    harving_dailyhours = pinp.labour['daily_hours'].loc['harvest','Permanent']
    perm_hrs_harv = (lp_len_p5z - perm_sick_leave_p5z) * harv_occur_p5z * harving_dailyhours
    ###weekend hrs
    perm_hrs_weekend = perm_weekend_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekends','Permanent']
    ###weekdays hrs
    perm_hrs_weekdays = perm_weekdays_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekdays','Permanent']
    perm_hrs_total_p5z = perm_hrs_weekend + perm_hrs_weekdays + perm_hrs_seeding + perm_hrs_harv

    ##cas hours
    ###seeding - seeding hours are the same for weekdays and weekends
    seeding_dailyhours = pinp.labour['daily_hours'].loc['seeding','Casual']
    cas_hrs_seeding = (cas_weekdays_p5z + cas_weekend_p5z) * seeding_occur_p5z * seeding_dailyhours
    ###harv - harvesting hours are the same for weekdays and weekends
    harving_dailyhours = pinp.labour['daily_hours'].loc['harvest','Casual']
    cas_hrs_harv = (cas_weekdays_p5z + cas_weekend_p5z) * harv_occur_p5z * harving_dailyhours
    ###weekend hrs
    cas_hrs_weekend = cas_weekend_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekends','Casual']
    ###weekdays hrs
    cas_hrs_weekdays = cas_weekdays_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekdays','Casual']
    ###total
    cas_hrs_total_p5z = cas_hrs_weekend + cas_hrs_weekdays + cas_hrs_seeding + cas_hrs_harv

    #############
    #supervision#
    #############
    
    ##work out the number of hours of supervision needed by casual staff
    perm_supervision_seed_p5z = seeding_occur_p5z * perm_hrs_total_p5z * pinp.labour['labour_eff'].loc['seedingharv','Permanent']
    perm_supervision_harv_p5z = harv_occur_p5z * perm_hrs_total_p5z * pinp.labour['labour_eff'].loc['seedingharv','Permanent']
    perm_supervision_norm_p5z = np.logical_not(np.logical_or(seeding_occur_p5z, harv_occur_p5z)) * perm_hrs_total_p5z * pinp.labour['labour_eff'].loc['normal','Permanent']
    perm_supervision_p5z = perm_supervision_norm_p5z + perm_supervision_harv_p5z + perm_supervision_seed_p5z

    ##work out the number of hours of supervision needed by casual staff
    cas_supervision_seed_p5z = seeding_occur_p5z * cas_hrs_total_p5z * pinp.labour['labour_eff'].loc['seedingharv','Casual']
    cas_supervision_harv_p5z = harv_occur_p5z * cas_hrs_total_p5z * pinp.labour['labour_eff'].loc['seedingharv','Casual']
    cas_supervision_norm_p5z = np.logical_not(np.logical_or(seeding_occur_p5z, harv_occur_p5z)) * cas_hrs_total_p5z * pinp.labour['labour_eff'].loc['normal','Casual']
    cas_supervision_p5z = cas_supervision_norm_p5z + cas_supervision_harv_p5z + cas_supervision_seed_p5z

    ##set bounds on casual staff
    seedharv_mask_pz = (seeding_occur_p5z + harv_occur_p5z)
    ###determine upper bounds for casual labour. note: casual labour requirements may be different during seeding and harvest compared to the rest
    max_casual_norm = pinp.labour['max_casual'] if pinp.labour['max_casual']!='inf' else np.inf #if inf need to convert to python inf
    max_casual_seedharv = pinp.labour['max_casual_seedharv'] if pinp.labour['max_casual']!='inf' else np.inf #if inf need to convert to python inf
    ub_cas_p5z = np.zeros(seeding_occur_p5z.shape, dtype=float)
    ub_cas_p5z[seedharv_mask_pz] = max_casual_seedharv
    ub_cas_p5z[np.logical_not(seedharv_mask_pz)] = max_casual_norm
    ###determine lower bounds for casual labour. note: casual labour requirements may be different during seeding and harvest compared to the rest
    lb_cas_p5z = np.zeros(seeding_occur_p5z.shape, dtype=float)
    lb_cas_p5z[seedharv_mask_pz] = pinp.labour['min_casual_seedharv']
    lb_cas_p5z[np.logical_not(seedharv_mask_pz)] = pinp.labour['min_casual']

    ##cost of casual for each labour period - wage plus super plus workers comp (multiplied by wage because super and others are %)
    ##differect to perm and manager because they are at a fixed level throughout the year ie same number of perm staff all yr.
    casual_cost_p5z = cas_hrs_total_p5z * (uinp.price['casual_cost'] + uinp.price['casual_cost'] * uinp.price['casual_super'] + uinp.price['casual_cost'] * uinp.price['casual_workers_comp'])
    casual_cost_zp5 = casual_cost_p5z.T

    ##labour cost cashflow period allocation and interest
    ### no enterprise is passed because fixed cost are for both enterprise and thus the interest is the average of both enterprises
    labour_cost_allocation_p7zp5, labour_wc_allocation_c0p7zp5 = fin.f_cashflow_allocation(lp_start_p5z.T, z_pos=-2)
    casual_cost_p7zp5 = casual_cost_zp5 * labour_cost_allocation_p7zp5
    casual_wc_c0p7zp5 = casual_cost_zp5 * labour_wc_allocation_c0p7zp5

    ########
    #z mask#
    ########
    ##make p5z8 mask (used to mask params with p5 axis and no p7 axis - params with p7 axis have been masked already in cash allocation)
    maskz8_p5z = zfun.f_season_transfer_mask(lp_start_p5z,z_pos=-1,mask=True)

    ##apply to params with only p5 period axis (p7z8 masking is handled elsewhere)
    perm_hrs_total_p5z = perm_hrs_total_p5z * maskz8_p5z
    perm_supervision_p5z = perm_supervision_p5z * maskz8_p5z
    cas_hrs_total_p5z = cas_hrs_total_p5z * maskz8_p5z
    cas_supervision_p5z = cas_supervision_p5z * maskz8_p5z
    manager_hrs_total_p5z = manager_hrs_total_p5z * maskz8_p5z

    #########
    ##keys  #
    #########
    ##keys
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = np.asarray(per.f_p_dates_df().index[:-1]).astype('str')

    ##index
    arrays = [keys_p5, keys_z]
    index_p5z = fun.cartesian_product_simple_transpose(arrays)
    tup_p5z = tuple(map(tuple, index_p5z))

    arrays = [keys_p7, keys_z, keys_p5]
    index_p7zp5 = fun.cartesian_product_simple_transpose(arrays)
    tup_p7zp5 = tuple(map(tuple, index_p7zp5))

    arrays = [keys_c0, keys_p7, keys_z, keys_p5]
    index_c0p7zp5 = fun.cartesian_product_simple_transpose(arrays)
    tup_c0p7zp5 = tuple(map(tuple, index_c0p7zp5))

    ##pyomo params
    params['permanent hours'] = dict(zip(tup_p5z, perm_hrs_total_p5z.ravel()))
    params['permanent supervision'] = dict(zip(tup_p5z, perm_supervision_p5z.ravel()))
    params['permanent_holiday_hours'] = perm_leave_hours
    params['casual hours'] = dict(zip(tup_p5z, cas_hrs_total_p5z.ravel()))
    params['casual supervision'] = dict(zip(tup_p5z, cas_supervision_p5z.ravel()))
    params['manager hours'] = dict(zip(tup_p5z, manager_hrs_total_p5z.ravel()))
    params['manager_holiday_hours'] = manager_leave_hours
    params['casual ub'] = dict(zip(tup_p5z, ub_cas_p5z.ravel()))
    params['casual lb'] = dict(zip(tup_p5z, lb_cas_p5z.ravel()))

    params['casual_cost'] =dict(zip(tup_p7zp5, casual_cost_p7zp5.ravel()))
    params['casual_wc'] =dict(zip(tup_c0p7zp5, casual_wc_c0p7zp5.ravel()))

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, maskz8_p5z, 'maskz8_p5z')
    fun.f1_make_r_val(r_vals, keys_p5, 'keys_p5')
    fun.f1_make_r_val(r_vals, casual_cost_p7zp5, 'casual_cost_p7zp5', mask_season_p7z[:,:,na], z_pos=-2)


def f_perm_cost(params, r_vals):
    '''
    Permanent and manager staff cost.

    Costs include bank interest.
    Permanent cost includes wage plus super plus workers comp and leave ls (multiplied by wage because super and others are %)
    '''

    ##cost allocation
    labour_start_c0 = per.f_cashflow_date() + 182 #fixed costs are incurred in the middle of the year and incur half a yr interest (in attempt to represent the even spread of fixed costs over the yr)
    ###call allocation/interset function - needs to be numpy
    ### no enterprise is passed because fixed cost are for both enterprise and thus the interest is the average of both enterprises
    labour_cost_allocation_p7z, labour_wc_allocation_c0p7z = fin.f_cashflow_allocation(labour_start_c0[:,na], z_pos=-1, c0_inc=True)

    ###perm
    perm_cost = (uinp.price['permanent_cost'] + uinp.price['permanent_cost'] * uinp.price['permanent_super'] \
    + uinp.price['permanent_cost'] * uinp.price['permanent_workers_comp'] + uinp.price['permanent_cost'] * uinp.price['permanent_ls_leave'])
    perm_cost_p7z = perm_cost * labour_cost_allocation_p7z
    perm_wc_c0p7z = perm_cost * labour_wc_allocation_c0p7z

    ###manager
    manager_cost = uinp.price['manager_cost']
    manager_cost_p7z = manager_cost * labour_cost_allocation_p7z
    manager_wc_c0p7z = manager_cost * labour_wc_allocation_c0p7z

    ##keys
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()

    arrays_p7z = [keys_p7, keys_z]
    arrays_c0p7z = [keys_c0, keys_p7, keys_z]

    ##params and report vals
    params['perm_cost'] = fun.f1_make_pyomo_dict(perm_cost_p7z, arrays_p7z)
    params['perm_wc'] = fun.f1_make_pyomo_dict(perm_wc_c0p7z, arrays_c0p7z)
    params['manager_cost'] = fun.f1_make_pyomo_dict(manager_cost_p7z, arrays_p7z)
    params['manager_wc'] = fun.f1_make_pyomo_dict(manager_wc_c0p7z, arrays_c0p7z)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, perm_cost_p7z, 'perm_cost_p7z', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, manager_cost_p7z, 'manager_cost_p7z', mask_season_p7z, z_pos=-1)













