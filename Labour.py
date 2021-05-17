# -*- coding: utf-8 -*-
"""
author: young
"""
#python modules
import pandas as pd
import numpy as np
import datetime 
from dateutil.relativedelta import relativedelta

#AFO modules
# from LabourInputs import *
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Periods as per
import Functions as fun


###################################################################
# make a df containing labour availability for each labour period #
###################################################################
na = np.newaxis


def labour_general(params,r_vals):
    '''
    Calculates labour supply, labour cost and supervision requirements.

    To capture the dynamics of labour, the year is broken into periods :cite:p:`RN89`. The supply of
    labour in each period by each labour source is calculated and the labour required by each farm
    activity is determined and assigned to the given period/s.

    The amount of time available to work in each period depends on the hours that each worker works
    each day. Labour can be supplied by three sources:

    #. Casual staff – In the unrestricted model casual staff can come and go at any time throughout the
       year as the model chooses. However, the user has the power to fix the number of casual staff employed
       during each part of the year.

    #. Permanent staff – Permanent staff work on the property all year.

    #. Manager staff (commonly the farm owner) – The farm manager works on the property all year. They control
       the overall farm plan and thus spend a fixed amount of time each quarter for farm planning, learning,
       record-keeping, purchasing and selling, and other office work.

    The farm manager and permanent staff have four weeks of holiday each year during December, January, and
    July. All labour sources take days off for Christmas, New Year’s Day, and Easter. Permanent staff are
    also allocated a certain number of sick days per year. The user has the ability to alter the length
    and timing of worker leave

    The timeliness and labour intensity of seeding and harvest means staff often work longer days during
    those periods :cite:p:`RN89`. To accommodate this, the user specifies the hours worked by each type of
    staff on the weekdays and weekends for both standard periods and seeding and harvest periods.

    Casual and permanent staff both require a certain amount of supervision from the farm manager. The
    proportion of supervision is specified separately for seeding and harvesting periods and all other
    labour periods. This is because during seeding and harvest it is likely that less supervision is
    required. Casual staff are generally less experienced and/or acquainted with the farm operation than
    permanent staff and thus require more supervision.

    Casual staff are paid on a per hour basis and the manager and permanent staff are paid an annual wage.
    All labour costs include superannuation and workers’ compensation.


    '''
    ##########################
    #inputs and initilisation#
    ##########################
    
    ##season inputs through input func
    harv_date_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)

    ##initilise periond data
    lp_p5z = per.p_dates_df().values
    lp_start_p5z = per.p_dates_df().iloc[:-1].values#.astype('datetime64[D]')
    lp_end_p5z = per.p_dates_df().iloc[1:].values#.astype('datetime64[D]')
    lp_len_p5z = (lp_end_p5z - lp_start_p5z).astype('timedelta64[D]').astype(float)

    ########
    #leave #
    ########
    
    ##manager leave
    # length = pd.to_timedelta(pinp.labour['leave_manager'], unit='D')
    length = np.array([pinp.labour['leave_manager']]).astype('timedelta64[D]')
    start = np.datetime64(pinp.labour['leave_manager_start_date'])
    manager_leave_alloc_p5z = fun.range_allocation_np(lp_p5z, start, length, True, shape=lp_p5z.shape)
    manager_leave_p5z = manager_leave_alloc_p5z * length.astype(float)
    manager_leave_p5z = manager_leave_p5z[:-1] #drop last row because it is just the end date of last period

    ##perm leave
    ###normal leave
    length = np.array([pinp.labour['leave_permanent']]).astype('timedelta64[D]')
    start = np.datetime64(pinp.labour['leave_permanent_start_date'])
    perm_leave_alloc_p5z = fun.range_allocation_np(lp_p5z, start, length, True, shape=lp_p5z.shape)
    perm_leave_p5z = perm_leave_alloc_p5z * length.astype(float)
    perm_leave_p5z = perm_leave_p5z[:-1] #drop last row because it is just the end date of last period
    ###sick leave - x days split equaly into each period
    perm_sick_leave_p5z = pinp.labour['sick_leave_permanent']/365 * lp_len_p5z
    ###total leave
    perm_leave_p5z = perm_leave_p5z + perm_sick_leave_p5z

    ##########################
    #hours worked per period #
    ##########################
    
    ##determine possible labour days worked by the manager during the week and on weekend in a given labour periods. Note: casual labour has no leave.
    ###available days in the period minus leave multiplied by fraction of weekdays
    manager_weekdays_p5z = (lp_len_p5z - manager_leave_p5z) * 5/7
    perm_weekdays_p5z = (lp_len_p5z - perm_leave_p5z) * 5/7
    cas_weekdays_p5z = (lp_len_p5z) * 5/7
    ###available days in the period minus leave multiplied by fraction of weekend days
    manager_weekend_p5z = (lp_len_p5z - manager_leave_p5z) * 2/7
    perm_weekend_p5z = (lp_len_p5z - perm_leave_p5z) * 2/7
    cas_weekend_p5z = (lp_len_p5z) * 2/7

    ##set up stuff to calc hours work per period be each source
    seed_period_lengths_pz = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    seeding_start_z = per.wet_seeding_start_date().astype(np.datetime64)
    seeding_end_z = seeding_start_z + np.sum(seed_period_lengths_pz, axis=0).astype('timedelta64[D]')
    seeding_occur_p5z =  np.logical_and(seeding_start_z <= lp_start_p5z, lp_start_p5z < seeding_end_z)
    harv_period_lengths_pz = pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1)
    harv_start_z = harv_date_z.astype(np.datetime64)
    harv_end_z = harv_start_z + np.sum(harv_period_lengths_pz, axis=0).astype('timedelta64[D]')
    harv_occur_p5z =  np.logical_and(harv_start_z <= lp_start_p5z, lp_start_p5z < harv_end_z)

    ##manager hours
    ###seeding
    seeding_dailyhours = pinp.labour['daily_hours'].loc['seeding','Manager']
    manager_hrs_seeding = (lp_len_p5z - manager_leave_p5z) * seeding_occur_p5z * seeding_dailyhours
    ###harv
    harving_dailyhours = pinp.labour['daily_hours'].loc['harvest','Manager']
    manager_hrs_harv = (lp_len_p5z - manager_leave_p5z) * harv_occur_p5z * harving_dailyhours
    ###weekend hrs
    manager_hrs_weekend = manager_weekend_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekends','Manager']
    ###weekdays hrs
    manager_hrs_weekdays = manager_weekdays_p5z * np.logical_not(np.logical_or(harv_occur_p5z, seeding_occur_p5z)) * pinp.labour['daily_hours'].loc['weekdays','Manager']
    manager_hrs_total_p5z = manager_hrs_weekend + manager_hrs_weekdays + manager_hrs_seeding + manager_hrs_harv

    ##perm hours
    ###seeding
    seeding_dailyhours = pinp.labour['daily_hours'].loc['seeding','Permanent']
    perm_hrs_seeding = (lp_len_p5z - perm_leave_p5z) * seeding_occur_p5z * seeding_dailyhours
    ###harv
    harving_dailyhours = pinp.labour['daily_hours'].loc['harvest','Permanent']
    perm_hrs_harv = (lp_len_p5z - perm_leave_p5z) * harv_occur_p5z * harving_dailyhours
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
    ub_cas_pz = np.zeros(seeding_occur_p5z.shape, dtype=float)
    ub_cas_pz[seedharv_mask_pz] = max_casual_seedharv
    ub_cas_pz[np.logical_not(seedharv_mask_pz)] = max_casual_norm
    ###determine lower bounds for casual labour. note: casual labour requirements may be different during seeding and harvest compared to the rest
    lb_cas_pz = np.zeros(seeding_occur_p5z.shape, dtype=float)
    lb_cas_pz[seedharv_mask_pz] = pinp.labour['min_casual_seedharv']
    lb_cas_pz[np.logical_not(seedharv_mask_pz)] = pinp.labour['min_casual']

    ##determine cashflow period each labour period alines with
    ###get cashflow period dates and names - used in the following loop
    p_dates = per.cashflow_periods()['start date']#get cashflow period dates
    p_dates_start_c = p_dates.values[:-1]
    p_dates_end_c = p_dates.values[1:]
    p_name = per.cashflow_periods()['cash period'].values[:-1].astype(str)#gets the period name
    ###determine cashflow allocation
    # index_c = np.arange(len(p_dates_start_c))
    length_c = p_dates_end_c - p_dates_start_c
    alloc_pzc = fun.range_allocation_np(lp_p5z[...,None],p_dates_start_c,length_c)[:-1]
    # cash_period_idx_pz = np.sum(alloc_pzc * index_c, axis=-1)
    # cashflow_alloc_p5z = p_name[cash_period_idx_pz]

    # p_name = np.broadcast_to(p_name[:,na], (p_name.shape + (lp_start_p5z.shape[-1],)))
    # cashflow_alloc_p5z = np.empty(lp_start_p5z.shape, dtype='S2')
    # for lp_date_z, lp_idx in zip(lp_start_p5z, np.arange(len(lp_start_p5z))):
    #     alloc_cz = np.logical_and(p_dates_start_c[:,na] <= lp_date_z, lp_date_z < p_dates_end_c[:,na])
    #     cashflow_alloc_p5z[lp_idx] = p_name[alloc_cz]

    ##cost of casual for each labour period - wage plus super plus workers comp (multipled by wage because super and others are %)
    ##differect to perm and manager because they are at a fixed level throughout the year ie same number of perm staff all yr.
    casual_cost_p5z = cas_hrs_total_p5z * (uinp.price['casual_cost'] + uinp.price['casual_cost'] * uinp.price['casual_super'] + uinp.price['casual_cost'] * uinp.price['casual_workers_comp'])
    casual_cost_p5zc = casual_cost_p5z[...,na] * alloc_pzc


    #########
    ##keys  #
    #########
    ##keys
    keys_c = np.array(sinp.general['cashflow_periods'])
    keys_p5 = np.asarray(per.p_dates_df().index[:-1]).astype('str')
    keys_z = pinp.f_keys_z()

    ##index
    arrays = [keys_p5,keys_c]
    index_p5c = fun.cartesian_product_simple_transpose(arrays)

    ################
    ##pyomo params #
    ################

    ##create season params in loop
    for z in range(len(keys_z)):
        ##create season key for params dict
        scenario = keys_z[z]
        params[scenario] = {}

        params[scenario]['permanent hours'] = dict(zip(keys_p5, perm_hrs_total_p5z[:,z]))
        params[scenario]['permanent supervision'] = dict(zip(keys_p5, perm_supervision_p5z[:,z]))
        params[scenario]['casual hours'] = dict(zip(keys_p5, cas_hrs_total_p5z[:,z]))
        params[scenario]['casual supervision'] = dict(zip(keys_p5, cas_supervision_p5z[:,z]))
        params[scenario]['manager hours'] = dict(zip(keys_p5, manager_hrs_total_p5z[:,z]))
        params[scenario]['casual ub'] = dict(zip(keys_p5, ub_cas_pz[:,z]))
        params[scenario]['casual lb'] = dict(zip(keys_p5, lb_cas_pz[:,z]))

        casual_cost_p5c = casual_cost_p5zc[:,z,:].ravel()
        tup_p5c = tuple(map(tuple, index_p5c))
        params[scenario]['casual_cost'] =dict(zip(tup_p5c, casual_cost_p5c))

    ##report values that are not season affected
    r_vals['keys_p5'] = keys_p5
    r_vals['casual_cost_p5zc'] = casual_cost_p5zc








#     labour_periods = per.p_dates_df()
#
#     for i, j in zip(labour_periods.index[:-1], labour_periods.index[1:]): #i is current period index, j is next period index
#         ##period length (days)
#         days = labour_periods.loc[j,'date'] - labour_periods.loc[i,'date']
#         labour_periods.loc[i,'days'] = days
#
#         ##leave manager, this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
#         ###if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
#         if labour_periods.loc[j , 'date'] < pinp.labour['leave_manager_start_date'] or labour_periods.loc[i , 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
#            labour_periods.loc[i , 'manager leave'] = datetime.timedelta(days = 0)
#         ###if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
#         elif labour_periods.loc[i , 'date'] < pinp.labour['leave_manager_start_date'] and labour_periods.loc[j, 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
#             labour_periods.loc[i , 'manager leave'] = datetime.timedelta(days = pinp.labour['leave_manager'])
#         ###if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
#         elif labour_periods.loc[i , 'date'] < pinp.labour['leave_manager_start_date'] and labour_periods.loc[j, 'date'] < pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
#             labour_periods.loc[i , 'manager leave'] = labour_periods.loc[j, 'date'] - pinp.labour['leave_manager_start_date']
#         ###if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
#         elif labour_periods.loc[i , 'date'] > pinp.labour['leave_manager_start_date'] and labour_periods.loc[j, 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
#             labour_periods.loc[i , 'manager leave'] = pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']) - labour_periods.loc[i , 'date']
#
#         ##leave permanent, very similar to above but also includes sick leave (10days/year split over each period), this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
#         ###if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
#         if labour_periods.loc[j, 'date'] < pinp.labour['leave_permanent_start_date'] or labour_periods.loc[i , 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
#            labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
#         ###if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
#         elif labour_periods.loc[i , 'date'] < pinp.labour['leave_permanent_start_date'] and labour_periods.loc[j, 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
#             labour_periods.loc[i , 'permanent leave'] = datetime.timedelta(days = pinp.labour['leave_permanent']) + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
#         ###if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
#         elif labour_periods.loc[i , 'date'] < pinp.labour['leave_permanent_start_date'] and labour_periods.loc[j, 'date'] < pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
#             labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[j, 'date'] - pinp.labour['leave_permanent_start_date']  + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
#         ###if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
#         elif labour_periods.loc[i , 'date'] > pinp.labour['leave_permanent_start_date'] and labour_periods.loc[j, 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
#             labour_periods.loc[i , 'permanent leave'] = pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']) - labour_periods.loc[i , 'date']  + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
#
#     ##determine possible labour days worked by the manager during the week and on weekend in a given labour periods
#     ###available days in the period minus leave multiplied by fraction of weekdays
#     labour_periods['manager weekdays'] = (labour_periods['days'] - labour_periods['manager leave']) * 5/7
#     ###available days in the period minus leave multiplied by fraction of weekend days
#     labour_periods['manager weekend'] = (labour_periods['days'] - labour_periods['manager leave']) * 2/7
#
#     ##determine possible labour days worked by permanent staff during the week and on weekend in a given labour perios
#     ###available days in the period minus leave multiplied by fraction of weekdays
#     labour_periods['permanent weekdays'] = (labour_periods['days'] - labour_periods['permanent leave']) * 5/7
#     ###available days in the period minus leave multiplied by fraction of weekend days
#     labour_periods['permanent weekend'] = (labour_periods['days'] - labour_periods['permanent leave']) * 2/7
#
#     ##function to determine possible labour days worked by casual staff during the week and on weekend in a given labour perios
#     ###available days in the period multiplied by fraction of weekdays
#     labour_periods['casual weekdays'] = labour_periods['days'] * 5/7
#     ###available days in the period multiplied by fraction of weekend days
#     labour_periods['casual weekend'] = labour_periods['days'] * 2/7
#
#     ##set upper limits on casual staff
#     max_casual = pinp.labour['max_casual'] if pinp.labour['max_casual']!='inf' else np.inf #if inf need to convert to python inf
#     max_casual_seedharv = pinp.labour['max_casual_seedharv'] if pinp.labour['max_casual']!='inf' else np.inf #if inf need to convert to python inf
#
#     ##get cashflow period dates and names - used in the following loop
#     p_dates = per.cashflow_periods()['start date']#get cashflow period dates
#     p_name = per.cashflow_periods()['cash period']#gets the period name
#
#     for i in labour_periods['date']: #loops through each period date
#         ##work out total hours available in each period for manager (owner)
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz): #checks if the date is a seed period
#             labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Manager'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
#         elif i in per.period_dates(harv_date_z,harv_period_lengths_pz):
#             labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Manager']
#         else:
#             labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekdays', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekends', 'Manager']
#         ##work out total hours available in each period for permanent staff
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz): #checks if the date is a seed period
#             labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Permanent'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding
#         elif i in per.period_dates(harv_date_z,harv_period_lengths_pz):
#             labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Permanent']
#         else:
#             labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekdays', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekends', 'Permanent']
#         ##work out total hours available in each period for casual staff
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz): #checks if the date is a seed period
#             labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['seeding', 'Casual'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
#         elif i in per.period_dates(harv_date_z,harv_period_lengths_pz):
#             labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['harvest', 'Casual']
#         else:
#             labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekdays', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
#             * pinp.labour['daily_hours'].loc['weekends', 'Casual']
#
#         ##work out the number of hours of supervision needed by permanent staff
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz) \
#         or i in per.period_dates(harv_date_z,harv_period_lengths_pz): #checks if the date is a seed period or harvest period
#             ###multiplys number of permanent hours in a given period by the percentage of supervision required for harvest and seeding
#             labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
#             * pinp.labour['labour_eff'].loc['seedingharv', 'Permanent']
#         else:
#             ###multiplys number of permanent hours in a given period by the percentage of supervision required for normal activities
#             labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
#             * pinp.labour['labour_eff'].loc['normal', 'Permanent']
#
#         ##function to work out the number of hours of supervision needed by casual staff
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz)   \
#         or i in per.period_dates(harv_date_z,harv_period_lengths_pz): #checks if the date is a seed period or harvest period
#             ###multiplys number of casual hours in a given period by the percentage of supervision required for harvest and seeding
#             labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
#             *pinp.labour['labour_eff'].loc['seedingharv', 'Casual']
#         else:
#             ###multiplys number of casual hours in a given period by the percentage of supervision required for normal activities
#             labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
#             *pinp.labour['labour_eff'].loc['normal', 'Casual']
#
#         ##determine bounds for casual labour, this is needed because casual labour requirements may be different during seeding and harvest compared to the rest
#         ###upper bound
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz) \
#         or i in per.period_dates(harv_date_z,harv_period_lengths_pz): #checks if the date is a seed period or harvest date
#             labour_periods.loc[labour_periods['date']==i , 'casual ub'] =  max_casual_seedharv
#         else:
#             labour_periods.loc[labour_periods['date']==i , 'casual ub'] = max_casual
#         ###lower bound
#         if i in per.period_dates(per.wet_seeding_start_date(),seed_period_lengths_pz) \
#         or i in per.period_dates(harv_date_z,harv_period_lengths_pz): #checks if the date is a seed period or harvest date
#             labour_periods.loc[labour_periods['date']==i , 'casual lb'] =  pinp.labour['min_casual_seedharv']
#         else:
#             labour_periods.loc[labour_periods['date']==i , 'casual lb'] = pinp.labour['min_casual']
#
#         ##determine cashflow period each labour period alines with
#         labour_periods.loc[labour_periods['date']==i , 'cashflow'] = fun.period_allocation(p_dates,p_name,i)
#
#     ##cost of casual for each labour period - wage plus super plus workers comp (multipled by wage because super and others are %)
#     ##differect to perm and manager because they are at a fixed level throughout the year ie same number of perm staff all yr.
#     labour_periods['casual_cost'] = labour_periods['casual hours'] * (uinp.price['casual_cost'] + uinp.price['casual_cost'] * uinp.price['casual_super'] + uinp.price['casual_cost'] * uinp.price['casual_workers_comp'])
#     ## drop last row, because it has na because it only contains the end date, therefore not a period
#     labour_periods.drop(labour_periods.tail(1).index,inplace=True)
#     ##create dicts for pyomo
#     params['permanent hours'] = labour_periods['permanent hours'].to_dict()
#     params['permanent supervision'] = labour_periods['permanent supervision'].to_dict()
#     params['casual_cost'] = dict(zip(zip(labour_periods['cashflow'].index,labour_periods['cashflow']),labour_periods['casual_cost']))
#     r_vals['casual_cost'] = pd.Series(params['casual_cost'])
#     params['casual hours'] = labour_periods['casual hours'].to_dict()
#     params['casual supervision'] = labour_periods['casual supervision'].to_dict()
#     params['manager hours'] = labour_periods['manager hours'].to_dict()
#     params['casual ub'] = labour_periods['casual ub'].to_dict()
#     params['casual lb'] = labour_periods['casual lb'].to_dict()
#     r_vals['keys_p5'] = per.p_date2_df().index.astype('object')
#
# # t_labour_periods=labour_general()



#permanent cost per cashflow period - wage plus super plus workers comp and leave ls (multipled by wage because super and others are %)
def perm_cost(params, r_vals):
    perm_cost = (uinp.price['permanent_cost'] + uinp.price['permanent_cost'] * uinp.price['permanent_super'] \
    + uinp.price['permanent_cost'] * uinp.price['permanent_workers_comp'] + uinp.price['permanent_cost'] * uinp.price['permanent_ls_leave']) / len(sinp.general['cashflow_periods'])
    perm_cost=dict.fromkeys(sinp.general['cashflow_periods'], perm_cost)
    params['perm_cost']=perm_cost
    r_vals['perm_cost_c']=np.array(list(perm_cost.values()))


#manager cost per cashflow period
def manager_cost(params, r_vals):
    manager_cost = uinp.price['manager_cost'] / len(sinp.general['cashflow_periods'])
    manager_cost=dict.fromkeys(sinp.general['cashflow_periods'], manager_cost)
    params['manager_cost']=manager_cost
    r_vals['manager_cost_c'] = np.array(list(manager_cost.values()))











