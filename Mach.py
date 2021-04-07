# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 17:06:04 2019

module: machinery module

extra - use mach_input rather than selecting a specific option from input sheet

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
   

@author: young
"""

#python modules
import pandas as pd
import datetime as dt
import numpy as np



#AFO modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import Periods as per
import Functions as fun

na = np.newaxis
# ################
# #mach option   #
# ################
# ##this selects the correct mach option from inputs and changes the dict name to mach_input, which is used in this module
# ##to access specific mach input data use mach_opt, and then assign mach_opt with a given option (done in property.xlsx)
# mach_opt = uinp.mach[pinp.mach['mach_option']]

# def select_mach_opt():
#     '''
    
#     Returns
#     -------
#     None.
#         This function is called from pyomo, which will update the mach option if necessary

#     '''
#     mach_opt = uinp.machine_options['mach_' + str(pinp.mach['mach_option'])]


################
#fuel price    #
################
#fuel price = price - rebate
def fuel_price():
    return uinp.price['diesel'] - uinp.price['diesel_rebate']



#######################################################################################################################################################
#######################################################################################################################################################
#feeding supplement
#######################################################################################################################################################
#######################################################################################################################################################
def sup_mach_cost():
    '''
    
    Returns
    -------
    Series.
            Cost of machinery to feed 1t of each grain to sheep.
    '''
    sup_cost=uinp.mach[pinp.mach['option']]['sup_feed'].copy() #need this so it doesnt alter inputs
    ##add fuel cost
    sup_cost['litres']=sup_cost['litres'] * fuel_price()
    ##sum the cost of r&m and fuel - note that rm is in the sup_cost df
    return sup_cost.sum(axis=1)

#######################################################################################################################################################
#######################################################################################################################################################
#seeding
#######################################################################################################################################################
#######################################################################################################################################################

#######################
#seed days per period #
#######################
##create a copy of periods df - so it doesn't alter the original period df that is used for labour stuff
# mach_periods = per.p_dates_df()#periods.copy()

def f_seed_days():
    '''
    Returns
    -------
    DataFrame - used in pyomo and also grazing days
        Determines the number of wet and dry seeding days in each period.
    '''
    mach_periods = per.p_dates_df()
    start_pz = mach_periods.values[:-1]
    end_pz = mach_periods.values[1:]
    length_pz = (end_pz - start_pz).astype('timedelta64[D]').astype(int)
    days = pd.DataFrame(length_pz, index=mach_periods.index[:-1], columns=mach_periods.columns)
    return days


    # for i in range(len(mach_periods)-1):
    #     days = (mach_periods.loc[mach_periods.index[i+1],'date'] - mach_periods.loc[mach_periods.index[i],'date']).days
    #     mach_periods.loc[mach_periods.index[i],'seed_days'] = days
    # ## drop last row, because it has na because it only contains the end date, therefore not a period
    # mach_periods.drop(mach_periods.tail(1).index,inplace=True)
    # if params:
    #     params[0]['seed_days'] = mach_periods['seed_days'].to_dict()
    # else: return mach_periods

def f_contractseeding_occurs():
    '''
    This function just sets the period when contract seeding must occur (period when wet seeding begins).
    Contract seeding is not hooked up to yield penalty because if your going to hire someone you will hire
    them at the optimum time. Contract seeding is hooked up to poc so this param stops the model having late seeding.
    '''
    contract_start_z = per.wet_seeding_start_date().astype(np.datetime64)
    mach_periods = per.p_dates_df()
    start_pz = mach_periods.values[:-1]
    end_pz = mach_periods.values[1:]
    contractseeding_occur_pz = np.logical_and(start_pz <= contract_start_z, contract_start_z < end_pz)
    contractseeding_occur_pz = pd.DataFrame(contractseeding_occur_pz,index=mach_periods.index[:-1],columns=mach_periods.columns)
    return contractseeding_occur_pz
    # params['contractseeding_occur'] = (mach_periods==contract_start).squeeze().to_dict()


# seed_days()
# def seed_days():
#     '''
#     Returns
#     -------
#     DataFrame - used in pyomo and also grazing days
#         Determines the number of wet and dry seeding days in each period.
#     '''
#     mach_periods = per.p_dates_df()
#     dry_seed_start = pinp.crop['dry_seed_start']
#     dry_seed_end = pinp.period['feed_periods'].loc[0,'date']#dry seeding finishes when the season breaks
#     dry_seed_len = dry_seed_end - dry_seed_start
#     ##determine the days of dry seeding occurring in each mach period
#     dry_days=fun.period_allocation(mach_periods['date'],mach_periods.index,dry_seed_start,dry_seed_len)['allocation'].dropna()*dry_seed_len #use the period allocation func to determine the proportion of total dry seeding occurring in each period
#     dry_days=dry_days/np.timedelta64(1,'D') #convert to int
#     wet_seed_start = per.wet_seeding_start_date()
#     seed_end = per.period_end_date(wet_seed_start, pinp.crop['seed_period_lengths'])
#     for i in range(len(mach_periods['date'])-1):
#         ##check wet seed dates
#         if wet_seed_start<= mach_periods.loc[i,'date'] < seed_end:
#             days = (mach_periods.loc[i+1,'date'] - mach_periods.loc[i,'date']).days
#         else:
#             days = 0
#         mach_periods.loc[i,'wet_seed_days'] = days
#     mach_periods['seed_days'] = mach_periods['wet_seed_days'].add(dry_days,fill_value=0)
#     ## drop last row, because it has na because it only contains the end date, therefore not a period
#     mach_periods.drop(mach_periods.tail(1).index,inplace=True) 
#     return mach_periods
# # seed_days()   

def f_grazing_days():
    '''
    Returns
    -------
    Dict for pyomo.
        Grazing days provided by wet seeding activity (ha/day/feed period)
        The maths behind this func is a little hard to explain - check google doc for better info
    '''
    ##inputs
    date_feed_periods = per.f_feed_periods().astype('datetime64')
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    mach_periods = per.p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    date_end_p5z = mach_periods.values[1:]
    seed_days_p5z = f_seed_days().values
    defer_period = np.array([pinp.crop['poc_destock']]).astype('timedelta64[D]') #days between seeding and destocking
    season_break_z = date_start_p6z[0]

    ##grazing days rectangle
    base_p6p5z = (np.minimum(date_end_p6z[:,na,:], date_start_p5z - defer_period) - np.maximum(season_break_z, date_start_p6z[:,na,:]))/ np.timedelta64(1, 'D')
    height_p5z = 1
    grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)

    ##triangular component
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(season_break_z, date_start_p5z - defer_period))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z - defer_period)
    base_p6p5z = (end_p6p5z - start_p6p5z)/ np.timedelta64(1, 'D')
    height_start_p6p5z = np.maximum(0, 1 - fun.f_divide((start_p6p5z - (date_start_p5z - defer_period))/ np.timedelta64(1, 'D'), seed_days_p5z))
    height_end_p6p5z = fun.f_divide(np.maximum(0,((date_end_p5z - defer_period) - end_p6p5z)/ np.timedelta64(1, 'D')), seed_days_p5z)
    grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

    ##total grazing days & convert to df
    total_grazing_days_p6p5z = grazing_days_tri_p6p5z + grazing_days_rect_p6p5z

    total_grazing_days_p6p5z = total_grazing_days_p6p5z.reshape(total_grazing_days_p6p5z.shape[0], -1)
    keys_z = pinp.f_keys_z()
    cols = pd.MultiIndex.from_product([mach_periods.index[:-1], keys_z])
    total_grazing_days = pd.DataFrame(total_grazing_days_p6p5z, index=pinp.period['i_fp_idx'], columns=cols)
    return total_grazing_days.stack(0)




    # ##drop last date from feed periods because it as the start date at the end
    # feed_periods_date = per.f_feed_periods()[:-1]
    # feed_periods_length = per.f_feed_periods(option=1)
    # ##run mach period func to get all the seeding day info
    # seed_days = f_seed_days()
    #
    # ##create df which all grazing days are added
    # grazing_days_df = pd.DataFrame(index=pinp.period['i_fp_idx'])
    #
    # ##loop through labour/mach periods.
    # for mach_p_start, seeding_days, mach_p_num in zip(mach_periods, seed_days,seed_days.index):
    #     grazing_days_list=[]
    #     season_break = feed_periods_date[0] #todo probs won't handle z axis
    #     effective_break = season_break + destock_days #accounts for the time before seeding that destocking must occur
    #     for i in range(len(feed_periods_date)):
    #         fp_end_date = feed_periods_date[i] + dt.timedelta(days = feed_periods_length[i]) #todo this will not handle Z axis either need to loop or maybe use numpy
    #         seed_end_date = mach_p_start + dt.timedelta(days = seeding_days)
    #         ##if the feed period finishes before the start of seeding it will receive a grazing day for each day since the break of season times the number of seeding days in the current seed period minus the grazing days in the previous periods
    #         if fp_end_date <= mach_p_start:
    #             fp_grazing_days = max((fp_end_date- effective_break).days * seeding_days - sum(grazing_days_list) , 0) #max required in case fp ends before effective break - don't want a negative value
    #         ##if the end date of the feed period is after the end date of the seeding period it will get the full grazing days minus the grazing days in the previous periods
    #         elif fp_end_date >= seed_end_date:
    #             fp_grazing_days = max((mach_p_start- effective_break).days * seeding_days + (0.5 * seeding_days * seeding_days)  - sum(grazing_days_list) ,0)
    #         ##if it isn't one of the conditions above then the feed period date must fall somewhere within the seed periods. This means the grazing days equal to the number of days between the season start and the start of the current seed period plus the diminishing number of days in the current period
    #         else:
    #             fp_grazing_days = max((mach_p_start- effective_break).days * seeding_days + (0.5 * seeding_days * seeding_days) -(0.5 * (seed_end_date-fp_end_date).days**2) - sum(grazing_days_list) , 0)
    #         grazing_days_list.append(fp_grazing_days)
    #     grazing_days_df[mach_p_num] = grazing_days_list #annoyingly both mach periods and feedperiods are defined as just numbers
    #     ##have to divide by the seeding days per period to return the grazing days if 1 ha was sown per period - this can now be multiplied by the ha sowed (done in pyomo)
    #     grazing_days_df[mach_p_num] = grazing_days_df[mach_p_num]/ seeding_days
    # params['grazing_days'] = grazing_days_df.stack().to_dict()
    #
#################################################
#seeding ha/day for each crop on each lmu  type #
#################################################

def f_seed_time_lmus():
    '''
    Returns
    -------
    DataFrame
        time taken to direct drill 1ha of wheat on each lmu.
    '''
    ##first turn seeding speed on each lmu to df so it can be manipulated (it was entered as a dict in machinputs)
    speed_lmu_df = pinp.mach['seeder_speed_lmu_adj']*uinp.mach[pinp.mach['option']]['seeder_speed_base']
    ##convert speed to rate of direct drill for wheat on each lmu type (hr/ha)
    rate_direct_drill = 1 / (speed_lmu_df * uinp.mach[pinp.mach['option']]['seeding_eff'] * uinp.mach[pinp.mach['option']]['seeder_width'] / 10)
    return rate_direct_drill

def f_overall_seed_rate(r_vals):
    '''
    Returns
    -------
    rate_direct_drill_day : Dict for pyomo
        Combines lmu seeding rate and crop adjustment to determine the seeding rate ha/day on each lmu type for each crop
    '''
    #convert seed time (hr/ha) to rate of direct drill per day (ha/day)
    seed_rate_lmus = 1 / f_seed_time_lmus().squeeze() * pinp.mach['daily_seed_hours']
    #adjusts the seeding rate (ha/day) for each different crop depending on its seeding speed vs wheat

    seedrate_df = pd.concat([uinp.mach[pinp.mach['option']]['seeder_speed_crop_adj']]*len(seed_rate_lmus),axis=1) #expands df for each lmu
    seedrate_df.columns = seed_rate_lmus.index #rename columns to lmu so i can mul
    seedrate_df=seedrate_df.mul(seed_rate_lmus)
    r_vals['seeding_rate'] = seedrate_df
    return seedrate_df.stack()

    
  

#################################################
#seeding 1 ha  cost                             #
#################################################
'''
the cost of seeding 1ha is the same for all crops, but the seeding rate (ha/day) may differ from crop to crop
cost of seeding 1ha is dependant on the lmu
'''   

def fuel_use_seeding():
    '''
    Returns
    -------
    DataFrame
        Fuel use L/ha used by tractor to seed on each lmu.
        Used to calculate fuel & oil cost and r&m cost in the functions below
    '''
    ##determine fuel use on base lmu (draft x tractor factor)
    base_lmu_seeding_fuel = uinp.mach[pinp.mach['option']]['draft_seeding'] * uinp.mach[pinp.mach['option']]['fuel_adj_tractor']
    ##determine fuel use on all soils by adjusting s5 fuel use with input adjustment factors
    df_seeding_fuel_lmu = base_lmu_seeding_fuel * pinp.mach['seeding_fuel_lmu_adj']
    #second multiply base cost by adj, to produce df with seeding fuel use for each lmu (L/ha)
    return df_seeding_fuel_lmu 
    
def tractor_cost_seeding():
    '''
    Returns
    -------
    DataFrame
        Tractor costs of seeding per ha ($/ha).
        -includes; fuel, oil, grease and r&m.
    '''
    fuel_used= fuel_use_seeding() 
    ##Tractor r&m during seeding for each lmu
    r_m_cost = fuel_used * fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    ##determine the $ cost of tractor fuel 
    tractor_fuel_cost = fuel_used * fuel_price()
    ##determine the $ cost of tractor oil and grease for seeding
    tractor_oil_cost = fuel_used * fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    return r_m_cost + tractor_fuel_cost + tractor_oil_cost


def maint_cost_seeder():
    '''
    Returns
    -------
    DataFrame
        Seeder r&m during seeding ($/ha).
    '''
    ##equals r&m on base lmu x lmu adj factor
    tillage_lmu_df = uinp.mach[pinp.mach['option']]['tillage_maint'] * pinp.mach['tillage_maint_lmu_adj']
    return  tillage_lmu_df

# def seeding_cost_lmu():
#     '''
#     Returns
#     -------
#     DataFrame
#         Total cost seeding on each lmu $/ha.
#     '''
#     return tractor_cost_seeding() + maint_cost_seeder()

def f_seed_cost_alloc():
    '''period allocation for seeding costs'''
    ##put inputs through season function
    seed_period_lengths_p5z = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    length_z = np.sum(seed_period_lengths_p5z, axis=0)
    ##gets the cost allocation
    p_dates_c = per.cashflow_periods()['start date'].values
    p_name_c = per.cashflow_periods()['cash period']
    length_z = length_z.astype('timedelta64[D]')
    start_z = per.wet_seeding_start_date().astype(np.datetime64)
    alloc_cz = fun.range_allocation_np(p_dates_c[...,None], start_z, length_z, True)
    keys_z = pinp.f_keys_z()
    alloc_cz = pd.DataFrame(alloc_cz, index=p_name_c, columns=keys_z)
    ## drop last row, because it has na because it only contains the end date, therefore not a period
    alloc_cz.drop(alloc_cz.tail(1).index,inplace=True)
    return alloc_cz

def f_seeding_cost(r_vals):
    '''
    Returns
    -------
    Dataframe for pyomo
        Returns the seeding cost allocation into cashflow periods.
    '''
    ##Total cost seeding on each lmu $/ha.
    seeding_cost_l = tractor_cost_seeding() + maint_cost_seeder()
    seeding_cost_l = seeding_cost_l.squeeze()
    ##gets the cost allocation
    alloc_cz = f_seed_cost_alloc()
    ##reindex with lmu so alloc can be mul with seeding_cost_l
    columns = pd.MultiIndex.from_product([seeding_cost_l.index, alloc_cz.columns])
    alloc_czl = alloc_cz.reindex(columns, axis=1, level=1)
    seeding_cost_czl = alloc_czl.mul(seeding_cost_l, axis=1, level=0)
    r_vals['seeding_cost'] = seeding_cost_czl
    return seeding_cost_czl





def f_contract_seed_cost(r_vals):
    '''
    Returns
    -------
    Dict
        Contract seeding cost in each cashflow period, currently, contract cost is the same for all lmus and crops.
    '''
    ##gets the cost allocation
    alloc_cz = f_seed_cost_alloc()
    ##cost to contract seed 1ha
    seed_cost = uinp.price['contract_seed_cost']
    contract_seed_cost = alloc_cz * seed_cost
    r_vals['contractseed_cost'] = contract_seed_cost
    return contract_seed_cost

########################################
#late seeding & dry seeding penalty    #
########################################

def f_yield_penalty():
    '''
    Yields
    ------
    Dict for pyomo
        Calcs the penalty in each mach period (dry seeding and wet) - kg/ha/period/crop.
        Used in pyomo to calculate loss of cashflow and reduction in stubble
    '''
    ##inputs
    seed_period_lengths_pz = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    dry_seeding_penalty_k = pinp.crop['yield_penalty']['dry_seeding_penalty']
    wet_seeding_penalty_k = pinp.crop['yield_penalty']['wet_seeding_penalty']

    ##general info
    mach_periods = per.p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]

    ##dry seeding penalty
    dry_seed_start = np.datetime64(pinp.crop['dry_seed_start'])
    dry_seed_end_z = per.f_feed_periods()[0].astype('datetime64') #dry seeding finishes when the season breaks
    period_is_dry_seeding_pz = np.logical_and(dry_seed_start <= mach_periods_start_pz, mach_periods_start_pz < dry_seed_end_z)
    dry_penalty_pzk = period_is_dry_seeding_pz[...,na] * dry_seeding_penalty_k.values

    ##wet seeding penalty - penalty = average penalty of period (= (start day + end day) / 2 * penalty)
    seed_start_z = per.wet_seeding_start_date().astype(np.datetime64)
    penalty_free_days_z = seed_period_lengths_pz[0].astype('timedelta64[D]')
    start_day_pz = 1 + (mach_periods_start_pz - (seed_start_z + penalty_free_days_z))/ np.timedelta64(1, 'D')
    end_day_pz = (mach_periods_end_pz - (seed_start_z + penalty_free_days_z))/ np.timedelta64(1, 'D')
    wet_penalty_pzk = (start_day_pz + end_day_pz)[...,na] / 2 * wet_seeding_penalty_k.values
    wet_penalty_pzk = np.clip(wet_penalty_pzk, 0, np.inf)

    ##combine dry and wet penalty
    penalty_pzk = dry_penalty_pzk + wet_penalty_pzk

    ##put into df
    penalty_pzk = penalty_pzk.reshape(penalty_pzk.shape[0], -1)
    keys_z = pinp.f_keys_z()
    cols = pd.MultiIndex.from_product([keys_z, wet_seeding_penalty_k.index])
    penalty = pd.DataFrame(penalty_pzk, index=mach_periods.index[:-1], columns=cols)
    return penalty.stack(1)


#     ##calc yield penalty
#     mach_periods = per.p_dates_df()
#     mach_penalty = pd.DataFrame()  #adds the average yield penalty for each crop for each period to the df
#     dry_seed_start = pinp.crop['dry_seed_start']
#     dry_seed_end = per.f_feed_periods()[0] #dry seeding finishes when the season breaks
#     seed_start = per.wet_seeding_start_date()
#     # seed_end = per.period_end_date(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths'])
#     penalty_free_days = dt.timedelta(days = seed_period_lengths_pz[0])
#     yield_penalty_df = pinp.crop['yield_penalty']
#     ##add the yield penalty for each period and each crop
#     for k, wet_penalty, dry_penalty in zip(yield_penalty_df.index, yield_penalty_df['wet_seeding_penalty'],yield_penalty_df['dry_seeding_penalty']):
#         for i in range(len(mach_periods['date'])-1):
#             period_start_date = mach_periods.loc[mach_periods.index[i],'date']
#             period_end = mach_periods.loc[mach_periods.index[i+1],'date']
#             ###check penalty free
#             if seed_start  <= period_start_date  < seed_start + penalty_free_days:
#                 penalty =  0
#             ###check wet seeding
#             elif seed_start + penalty_free_days <= period_start_date:#  < seed_end:
#                 #penalty = average penalty of period (= (start day + end day) / 2 * penalty)
#                 start_day = 1 + (period_start_date - (seed_start + penalty_free_days)).days
#                 end_day = (period_end - (seed_start + penalty_free_days)).days
#                 penalty =  (start_day + end_day) / 2  * wet_penalty
#             ###check dry seeding
#             elif dry_seed_start <= period_start_date <= dry_seed_end:
#                 penalty = dry_penalty
#             ###other periods (between season break and seeding start & before dry seeding) get 500 penalty
#             else: penalty = 500
#             mach_penalty.loc[mach_periods.index[i], k] = penalty
#     params['yield_penalty'] = mach_penalty.stack().to_dict()
# # x = (yield_penalty())


#######################################################################################################################################################
#######################################################################################################################################################
#harv / hay making
#######################################################################################################################################################
#######################################################################################################################################################


#################################################
#harvesting                                     #
#################################################

def harv_time_ha():
    '''
    Returns
    -------
    Dataframe
        Harvest rate for each crop (hr/ha).
        - has to be kept as a separate function because it is used in multiple places
        - harv sped is entered relative to a given yield for each crop
    '''
    harv_speed = uinp.mach[pinp.mach['option']]['harvest_speed']
    ##work rate hr/ha, determined from speed, size and eff
    return 10/ (harv_speed * uinp.mach[pinp.mach['option']]['harv_eff'] * uinp.mach[pinp.mach['option']]['harvester_width'])



def f_harv_rate_period():
    '''
    Returns
    -------
    Dict for pyomo
        Harv rate in each mach period for each crops.
        - account for crops that can be harvested early ie crops that can't be harvested early are given 0 harv rate in the first harv period
    '''
    ##season inputs through function
    harv_start_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype(np.datetime64) #when the first crop begins to be harvested (eg when harv periods start)
    harv_period_lengths_z = np.sum(pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done
    start_harvest_crops = pinp.crop['start_harvest_crops']
    start_harvest_crops_kz = pinp.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1).astype(np.datetime64) #start harvest for each crop

    ##harv occur - note: some crops are not harvested in the early harv period
    mach_periods_start_pz = per.p_dates_df().values[:-1]
    mach_periods_end_pz = per.p_dates_df().values[1:]
    harv_occur_pkz = np.logical_and(mach_periods_start_pz[:,na,:] < harv_end_z,
                                    mach_periods_end_pz[:,na,:] > start_harvest_crops_kz)
    ##make df
    keys_z = pinp.f_keys_z()
    col = pd.MultiIndex.from_product([start_harvest_crops.index, keys_z])
    harv_occur = harv_occur_pkz.reshape(harv_occur_pkz.shape[0],-1)
    harv_occur = pd.DataFrame(harv_occur, index=per.p_date2_df().index, columns=col)

    ##Grain harvested per hr (t/hr) for each crop.
    harv_rate = (uinp.mach_general['harvest_yield'] * (1 / harv_time_ha())).squeeze()

    ##combine harv rate and harv_occur
    harv_rate_period = harv_occur.mul(harv_rate, axis=1, level=0)
    return harv_rate_period.stack(0)

    # mach_periods = per.p_dates_df()
    # harv_rate_df = pd.DataFrame()
    # harv_end = per.period_end_date(harv_start_z, harv_period_lengths_pz)
    # ##Grain harvested per hr (t/hr) for each crop.
    # harv_rate = (uinp.mach_general['harvest_yield'] * (1 / harv_time_ha())).squeeze()
    # ##loops through dict which contains harv start date for each crop
    # ##this determines if the crop is allowed early harv
    # for k, crop_harv_date in zip(pinp.crop['start_harvest_crops'].index, start_harvest_crops_kz):
    #     if k=='h':
    #         continue # this is required because hay is included in the harvest dates (needed for stubble) but not in any of the other harvest info
    #     for i in range(len(mach_periods['date'])-1):
    #         period_start_date = mach_periods.loc[mach_periods.index[i],'date']
    #         period_end = mach_periods.loc[mach_periods.index[i+1],'date']
    #         ###if the period is a harvest period
    #         if harv_start_z <= period_start_date  < harv_end:
    #             ####if crop harv date is before the end of the current period then it is allowed to be harvested in that period hence it is given a harv rate
    #             if crop_harv_date < period_end:
    #                 harvest_rate =  harv_rate.squeeze()[k]
    #             else: harvest_rate = 0
    #         else: harvest_rate = 0
    #         harv_rate_df.loc[mach_periods.index[i], k] = harvest_rate
    # params['harv_rate_period'] = harv_rate_df.stack().to_dict()
# harv_rate_period()  


#adds the max number of harv hours for each crop for each period to the df  
def f_max_harv_hours():
    '''
    Maximum hours that can be spent harvesting in a given period per crop gear compliment.
    This is not crop specific.
    '''

    # ##inputs ^something like below could be used if switch to the improved method (documented in google)
    # harv_start_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0) #when the first crop begins to be harvested (eg when harv periods start)
    # harv_period_lengths_kz = np.sum(pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    # harv_end_z = harv_start_z.astype('datetime64') + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done
    # start_harvest_crops = pinp.crop['start_harvest_crops']
    # start_harvest_crops_kz = pinp.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1) #start harvest for each crop
    # length_kz = harv_end_z - start_harvest_crops_kz
    # mach_periods_pz = per.p_dates_df().values
    # harv_occur_propn_pkz = fun.range_allocation_np(mach_periods_pz, start_harvest_crops_kz, length_kz) ^this will return wrong shape
    # ##calchours per period for each crop
    # days_pz = per.p_dates_df().values[:-1] - per.p_dates_df().values[1:]
    # hours_pkz = days_pz * harv_occur_propn_pkz * pinp.mach['daily_harvest_hours'] this becomes a param
    # total_hours = np.max(hours_pkz, axis=1) #returns the max hours that can be spent harvesting any crop in a given period. this becomes a param

    ##inputs
    harv_start_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)
    harv_period_lengths_z = np.sum(pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z.astype('datetime64') + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done

    ##does any harvest occur in given period
    mach_periods_start_pz = per.p_dates_df()[:-1]
    mach_periods_end_pz = per.p_dates_df()[1:]
    harv_occur_pz = np.logical_and(harv_start_z <= mach_periods_start_pz, mach_periods_start_pz < harv_end_z)

    ##max harv hour per period
    days_pz = (mach_periods_end_pz.values - mach_periods_start_pz.values)/ np.timedelta64(1, 'D')
    max_hours_pz = days_pz * harv_occur_pz * pinp.mach['daily_harvest_hours']
    return max_hours_pz


#
#     harv_end = per.period_end_date(harv_start_z, harv_period_lengths_pz)
#     #loops through dict which contains harv start date for each crop
#     #this determines if the crop is allowed early harv
#     for i in range(len(mach_periods['date'])-1):
#         period_start_date = mach_periods.loc[mach_periods.index[i],'date']
#         period_end = mach_periods.loc[mach_periods.index[i+1],'date']
#         if harv_start_z <= period_start_date  < harv_end:
#             harv_days =  (period_end - period_start_date).days
#         else: harv_days = 0
#         #convert to hours.
#         mach_periods.loc[mach_periods.index[i], 'max_harv_hours'] = harv_days * pinp.mach['daily_harvest_hours']
#     ## drop last row, because it has na because it only contains the end date, therefore not a period
#     mach_periods.drop(mach_periods.tail(1).index,inplace=True)
#     params['max_harv_hours'] = mach_periods['max_harv_hours'].to_dict()
# #max_harv_hours()

def cost_harv():
    '''
    Returns
    -------
    Dataframe
        Harvesting cost ($/hr).
    '''
    ##fuel used L/hr - same for each crop
    fuel_used = uinp.mach[pinp.mach['option']]['harv_fuel_consumption'] 
    ##determine cost of fuel and oil and grease $/ha
    fuel_cost_hr = fuel_used * fuel_price()
    oil_cost_hr = fuel_used * uinp.mach[pinp.mach['option']]['oil_grease_factor_harv'] * fuel_price()
    ##determine fuel and oil cost per hr
    fuel_oil_cost_hr = fuel_cost_hr + oil_cost_hr 
    ##return fuel and oil cost plus r & m ($/hr)
    return fuel_oil_cost_hr + uinp.mach[pinp.mach['option']]['harvest_maint']

def f_harv_cost_alloc():
    '''allocation of harvest cost into cashflow period'''
    ##gets the cost allocation
    p_dates_c = per.cashflow_periods()['start date'].values
    harv_start_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype('datetime64')
    harv_lengths_z = np.sum(pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0).astype('timedelta64[D]')
    alloc_cz = fun.range_allocation_np(p_dates_c[...,None],harv_start_z,harv_lengths_z,True)
    ###make it a df
    p_name_c = per.cashflow_periods()['cash period']
    keys_z = pinp.f_keys_z()
    alloc_cz = pd.DataFrame(alloc_cz,index=p_name_c,columns=keys_z)
    ### drop last row, because it has na because it only contains the end date, therefore not a period
    alloc_cz.drop(alloc_cz.tail(1).index,inplace=True)
    return alloc_cz


def f_harvest_cost(r_vals):
    '''
    Returns
    -------
    Dict for pyomo
        Cost of harvest in each cashflow period ($/hr).

    '''
    ##allocation
    alloc_cz = f_harv_cost_alloc()

    ##reindex with lmu so alloc can be mul with seeding_cost_l
    harv_cost_k = cost_harv().squeeze()
    columns = pd.MultiIndex.from_product([harv_cost_k.index, alloc_cz.columns])
    alloc_czk = alloc_cz.reindex(columns, axis=1, level=1)
    harv_cost_czk = alloc_czk.mul(harv_cost_k, axis=1, level=0)
    r_vals['harvest_cost'] = harv_cost_czk
    return harv_cost_czk.stack(0)



    # #gets the date column of the cashflow periods df
    # p_dates = per.cashflow_periods()['start date']
    # #gets the period name
    # p_name = per.cashflow_periods()['cash period']
    # length = dt.timedelta(days = sum(harv_period_lengths_pz).astype(np.float64))
    # harvest_cost = fun.period_allocation_reindex(cost_df, p_dates, p_name, harv_start_z,length)
    # params['harvest_cost'] = harvest_cost.stack().to_dict()
    # r_vals['harvest_cost'] = harvest_cost



#########################
#contract harvesting    #
#########################

def f_contract_harv_rate():
    '''
    Returns
    -------
    Dict for pyomo.
        Grain harvested per hr by contractor (t/hr)
    '''
    yield_approx = uinp.mach_general['harvest_yield'] #these are the yields the contract harvester is calibrated to - they are used to convert time/ha to t/hr
    harv_speed = uinp.mach_general['contract_harvest_speed']
    ##work rate hr/ha, determined from speed, size and eff
    contract_harv_time_ha = 10 / (harv_speed * uinp.mach_general['contract_harvester_width'] * uinp.mach_general['contract_harv_eff'])
    ##overall t/hr
    harv_rate = (yield_approx * (1 / contract_harv_time_ha)).squeeze()
    return harv_rate
#print(contract_harv_rate())


def f_contract_harvest_cost_period(r_vals):
    '''
    Returns
    -------
    Dict for pyomo
        Cost of contract harvest in each cashflow period ($/hr).
    '''
    ##allocation
    alloc_cz = f_harv_cost_alloc()

    ##reindex with lmu so alloc can be mul with seeding_cost_l
    contract_harv_cost_k = uinp.price['contract_harv_cost'].squeeze() #contract harvesting cost for each crop ($/hr)
    columns = pd.MultiIndex.from_product([contract_harv_cost_k.index, alloc_cz.columns])
    alloc_czk = alloc_cz.reindex(columns, axis=1, level=1)
    contract_harv_cost_czk = alloc_czk.mul(contract_harv_cost_k, axis=1, level=0)
    r_vals['contract_harvest_cost'] = contract_harv_cost_czk
    return contract_harv_cost_czk.stack(0)

    # ##inputs through season input funciton
    # harv_start_z = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)
    # harv_period_lengths_pz = pinp.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1)
    # ##calc contract cost
    # cost_df = uinp.price['contract_harv_cost'] #contract harvesting cost for each crop ($/hr)
    # #gets the date column of the cashflow periods df
    # p_dates = per.cashflow_periods()['start date']
    # #gets the period name
    # p_name = per.cashflow_periods()['cash period']
    # length = dt.timedelta(days = sum(harv_period_lengths_pz).astype(np.float64))
    # contract_harvest_cost = fun.period_allocation_reindex(cost_df, p_dates, p_name, harv_start_z,length)
    # params['contract_harvest_cost'] = contract_harvest_cost.stack().to_dict()
    # r_vals['contract_harvest_cost'] = contract_harvest_cost


#########################
#make hay               #
#########################
def f_hay_making_cost():
    '''
    Returns
    -------
    series - used in pyomo
        Cost to make hay ($/t).
    Currently it is assumed that hay is allocated into the same cashflow periods in all seasons.
    '''
    ##cost allocation
    p_dates = per.cashflow_periods()['start date'] #gets the date column of the cashflow periods df
    p_name = per.cashflow_periods()['cash period'] #gets the period name
    start = pinp.crop['hay_making_date']
    length = dt.timedelta(days = pinp.crop['hay_making_len'])
    allocation = fun.period_allocation(p_dates, p_name, start,length).set_index('period')    ##convert from ha to tonne (cost per ha divide approx yield)
    mow_cost =  uinp.price['contract_mow_hay'] \
    / uinp.mach_general['approx_hay_yield'] 
    bail_cost =  uinp.price['contract_bail'] 
    cart_cost = uinp.price['cart_hay'] 
    total_cost = mow_cost + bail_cost + cart_cost
    return (allocation * total_cost).stack().droplevel(1) #drop level because i stacked to get it to a series but it was already 1d and i didn't want the col name as a key

#######################################################################################################################################################
#######################################################################################################################################################
#stubble handling
#######################################################################################################################################################
#######################################################################################################################################################

##############################
#cost per ha stubble handling#
##############################
def stubble_cost_ha():
    '''
    Returns
    -------
    Dataframe that is passed to crop.py which determine rotation cost of stubble handling
        Cost to handle stubble for 1 ha.
    '''
    start = pinp.mach['stub_handling_date'] #needed for allocation func
    length = dt.timedelta(days =pinp.mach['stub_handling_length']) #needed for allocation func
    p_dates = per.cashflow_periods()['start date'] #needed for allocation func
    p_name = per.cashflow_periods()['cash period'] #needed for allocation func
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period')
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + stubble rake(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['stubble_maint']
    return (allocation*cost).dropna()
#cc=stubble_cost_ha()

#######################################################################################################################################################
#######################################################################################################################################################
#fert
#######################################################################################################################################################
#######################################################################################################################################################

###########################
#fert application time   # used in labour crop also, defined here because it uses inputs from the different mach options which are consolidated at the top of this sheet
###########################

#time taken to spread 1ha (not including driving to and from paddock and filling up)
# hr/ha= 10/(width*speed*efficiency)
def time_ha():
      width_df = uinp.mach[pinp.mach['option']]['spreader_width']
      return 10/(width_df*uinp.mach[pinp.mach['option']]['spreader_speed']*uinp.mach[pinp.mach['option']]['spreader_eff'])

#time taken to driving to and from paddock and filling up
# hr/cubic m = ((ave distance to paddock *2)/speed + fill up time)/ spreader capacity  # *2 because to and from paddock
def time_cubic():
      return (((pinp.mach['ave_pad_distance'] *2) 
              /uinp.mach[pinp.mach['option']]['spreader_speed'] + uinp.mach[pinp.mach['option']]['time_fill_spreader'])
              /uinp.mach[pinp.mach['option']]['spreader_cap'])
     

###################
#application cost # *remember that lime application only happens every 4 yrs - accounted for in the passes inputs
################### *used in crop pyomo
#this is split into two sections - new feature of AFO
# 1- cost to drive around 1ha
# 2- cost per cubic metre ie to represent filling up and driving to and from paddock

def spreader_cost_hr():
    '''
    Returns
    -------
    Float
        Cost to spread fert for one hour.
        -used to determine fert application cost per hour and per ha
    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + spreader(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['spreader_maint']
    return cost

def fert_app_cost_ha():
    '''
    Returns
    -------
    Dataframe
        Application cost per ha - account for time to spread 1ha
        - multiplied by passes to account for the number of application (in crop module) 
    '''
    return spreader_cost_hr() * time_ha().stack().droplevel(1)

def fert_app_cost_t():
    '''
    Returns
    -------
    Dataframe
        Application cost per tonne, this is to account for filling up and driving to and from paddock.
    '''
    spreader_proportion = pinp.crop['fert_info']['spreader_proportion']
    conversion = pinp.crop['fert_info']['fert_density']
    ##spreader cost per hr is multiplied by the full time required to drive to and from paddock and fill up even though during filling up the tractor is just sitting, but this accounts for the loader time/cost
    cost_cubic = spreader_cost_hr() * time_cubic()
    ##mulitiplied by a factor (spreader_proportion) 0 or 1 if the fert is applied at seeding (or a fraction if applied at both seeding and another time)
    cost_t = cost_cubic / conversion * spreader_proportion #convert from meters cubed to tonne - divide by conversion (density) because lighter ferts require more filling up time per tonne
    return cost_t
#e=fert_app_t()
    
#######################################################################################################################################################
#######################################################################################################################################################
#chem
#######################################################################################################################################################
#######################################################################################################################################################

###########################
#chem application time   # used in labour crop, defined here because it uses inputs from the different mach options which are consolidated at the top of this sheet
###########################

##time taken to spray 1ha (use efficiency input to allow for driving to and from paddock and filling up)
## hr/ha= 10/(width*speed*efficiency)
def spray_time_ha():
      width_df = uinp.mach[pinp.mach['option']]['sprayer_width']
      return 10/(width_df*uinp.mach[pinp.mach['option']]['sprayer_speed']*uinp.mach[pinp.mach['option']]['sprayer_eff'])

   

###################
#application cost # 
################### *used in crop pyomo

def chem_app_cost_ha():
    '''
    Returns
    -------
    Float
          Application cost per ha - account for time to spread 1ha
        - multiplied by passes to account for the number of application (in crop module)
    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + sprayer(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['sprayer_maint']
    return cost



       
#######################################################################################################################################################
#######################################################################################################################################################
#Dep
#######################################################################################################################################################
#######################################################################################################################################################

#########################
#mach value             #
#########################
##harvest machine cost
def harvest_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'harvest allocation'])
    total_value = value * pinp.mach['number_harv_gear']
    return total_value


##value of gear used for seed. This is used to calculate the variable depreciation linked to seeding activity
def f_seeding_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'seeding allocation'])
    total_value = value * pinp.mach['number_seeding_gear']
    return total_value

    # sprayer_cost = uinp.mach[pinp.mach['option']]['clearing_value'].loc['sprayer','value'] * pinp.mach['sprayer_crop_allocation']
    # ##tractor
    # return sprayer_cost + uinp.mach[pinp.mach['option']]['clearing_value'].loc['silo','value'] + uinp.mach[pinp.mach['option']]['clearing_value'].loc['auger','value'] \
    # + uinp.mach[pinp.mach['option']]['clearing_value'].loc['tractor','value'] + uinp.mach[pinp.mach['option']]['clearing_value'].loc['seeder','value']

##total machine value - used to calc asset value, fixed dep and insurance
def f_total_clearing_value():
    harv_value = harvest_gear_clearing_value()
    seed_value = f_seeding_gear_clearing_value()
    other_value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'remaining allocation'])
    return harv_value + seed_value + other_value

#########################
#fixed depreciation     #
#########################


#total value of crop gear x dep rate x number of crop gear
def f_fix_dep():
    return f_total_clearing_value() * uinp.finance['fixed_dep']


####################################
#variable seeding depreciation     #
####################################

def f_seeding_dep():
    '''
    Returns
    -------
    Dict for pyomo
        Average variable dep for seeding $/ha.
    '''
    ##inputs
    seed_rate = f_seed_time_lmus().squeeze()
    ##first determine the approx time to seed all the crop - which is equal to dep area x average seeding rate (hr/ha)
    average_seed_rate = seed_rate.mean()
    seeding_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_seed_rate
    ##second, determine dep per hour - equal to crop gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    dep_hourly = seeding_gear_clearing_value * dep_rate / seeding_time
    ##third, convert to dep per ha for each soil type - equals cost per hr x seeding rate per hr
    dep_ha = dep_hourly * seed_rate
    return dep_ha


####################################
#variable harvest depreciation     #
####################################

def f_harvest_dep():
    '''
    Returns
    -------
    Float for pyomo
        Average variable dep for harvesting $/hr.
    '''
    ##first determine the approx time to harvest all the crop - which is equal to dep area x average harvest rate (hr/ha)
    average_harv_rate = harv_time_ha().squeeze().mean()
    average_harv_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_harv_rate 
    ##second, determine dep per hour - equal to harv gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    dep_hourly = harvest_gear_clearing_value() * dep_rate / average_harv_time
    return dep_hourly


#######################################################################################################################################################
#######################################################################################################################################################
#insurance on all gear
#######################################################################################################################################################
#######################################################################################################################################################
def f_insurance():
    '''

    Returns
    -------
    Dict for pyomo.
        cost of insurance for all machinery
    '''
    ##determine the insurance paid
    value_all_mach = f_total_clearing_value()
    insurance = value_all_mach * uinp.finance['equip_insurance']
    ##determine cash period
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    start = uinp.mach_general['insurance_date']
    allocation=fun.period_allocation(p_dates, p_name,start)
    return {allocation:insurance}


#######################################################################################################################################################
#######################################################################################################################################################
#params
#######################################################################################################################################################
#######################################################################################################################################################

##collates all the params
def f_mach_params(params,r_vals):
    seed_days = f_seed_days()
    contractseeding_occur = f_contractseeding_occurs()
    seedrate = f_overall_seed_rate(r_vals)
    seeding_cost = f_seeding_cost(r_vals).stack(0)
    contract_seed_cost = f_contract_seed_cost(r_vals)
    harv_rate_period = f_harv_rate_period()
    contract_harv_rate = f_contract_harv_rate()
    max_harv_hours = f_max_harv_hours()
    harvest_cost = f_harvest_cost(r_vals)
    contract_harvest_cost = f_contract_harvest_cost_period(r_vals)
    hay_making_cost = f_hay_making_cost()
    yield_penalty = f_yield_penalty()
    grazing_days = f_grazing_days()
    fixed_dep = f_fix_dep()
    harv_dep = f_harvest_dep()
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    seeding_dep = f_seeding_dep()
    insurance = f_insurance()
    mach_asset_value = f_total_clearing_value()

    ##add inputs that are params to dict
    params['number_seeding_gear'] = pinp.mach['number_seeding_gear']
    params['number_harv_gear'] = pinp.mach['number_harv_gear']
    params['seeding_occur'] = pinp.mach['seeding_occur']

    ##create non seasonal params
    params['seed_rate'] = seedrate.to_dict()
    params['contract_harv_rate'] = contract_harv_rate.to_dict()
    params['hay_making_cost'] = hay_making_cost.to_dict()
    params['fixed_dep'] = fixed_dep
    params['harv_dep'] = harv_dep
    params['seeding_gear_clearing_value'] = seeding_gear_clearing_value
    params['seeding_dep'] = seeding_dep.to_dict()
    params['mach_asset_value'] = mach_asset_value
    params['insurance'] = insurance

    ##create season params in loop
    keys_z = pinp.f_keys_z()
    for z in range(len(keys_z)):
        ##create season key for params dict
        scenario = keys_z[z]
        params[scenario] = {}
        params[scenario]['seed_days'] = seed_days[scenario].to_dict()
        params[scenario]['contractseeding_occur'] = contractseeding_occur[scenario].to_dict()
        params[scenario]['seeding_cost'] = seeding_cost[scenario].to_dict()
        params[scenario]['contract_seed_cost'] = contract_seed_cost[scenario].to_dict()
        params[scenario]['harv_rate_period'] = harv_rate_period[scenario].to_dict()
        params[scenario]['harvest_cost'] = harvest_cost[scenario].to_dict()
        params[scenario]['max_harv_hours'] = max_harv_hours[scenario].to_dict()
        params[scenario]['contract_harvest_cost'] = contract_harvest_cost[scenario].to_dict()
        params[scenario]['yield_penalty'] = yield_penalty[scenario].to_dict()
        params[scenario]['grazing_days'] = grazing_days[scenario].to_dict()


