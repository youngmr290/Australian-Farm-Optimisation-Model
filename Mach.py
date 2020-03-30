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



#MUDAS modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import Periods as per
import Functions as fun

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
#         This function is called from pyomo, which will update the mach option if nessecary

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
    sup_cost=uinp.mach[pinp.mach['option']]['sup_feed']
    ##add fuel cost
    sup_cost['litres']=sup_cost['litres'] * fuel_price()
    return sup_cost.sum(axis=1)

#######################################################################################################################################################
#######################################################################################################################################################
#seeding
#######################################################################################################################################################
#######################################################################################################################################################

#######################
#seed days per period #
#######################
##create a copy of periods df - so it doesn't alter the origional period df that is used for labour stuff
# mach_periods = per.p_dates_df()#periods.copy()

def seed_days():
    '''
    Returns
    -------
    DataFrame - used in pyomo and also grazing days
        Determines the number of wet and dry seeding days in each period.
    '''
    mach_periods = per.p_dates_df()
    for i in range(len(mach_periods['date'])-1):
        days = (mach_periods.loc[i+1,'date'] - mach_periods.loc[i,'date']).days
        mach_periods.loc[i,'seed_days'] = days
    ## drop last row, because it has na because it only contains the end date, therefore not a period
    mach_periods.drop(mach_periods.tail(1).index,inplace=True) 
    return mach_periods
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
#     dry_seed_end = pinp.feed_inputs['feed_periods'].loc[0,'date']#dry seeding finishes when the season breaks
#     dry_seed_len = dry_seed_end - dry_seed_start
#     ##determine the days of dry seeding occuring in each mach period
#     dry_days=fun.period_allocation(mach_periods['date'],mach_periods.index,dry_seed_start,dry_seed_len)['allocation'].dropna()*dry_seed_len #use the period allocation func to determine the proportion of total dry seeding occuring in each period
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

def grazing_days():
    '''
    Returns
    -------
    Dict for pyomo.
        Grazing days provided by wet seeding activity (ha/day/feed period)
        The maths behind this func is a little hard to explain - chech google doc for better info
    '''
    ##drop last date from feed periods because it as the start date at the end
    feed_periods=pinp.feed_inputs['feed_periods'].iloc[:-1]
    ##run mach period func to get all the seeding day info
    mach_periods = seed_days()
    ##create df which all grazing days are added
    grazing_days_df = pd.DataFrame()
    ##days between seeding and destocking
    destock_days = dt.timedelta(days = pinp.crop['poc_destock'])
    ##loop through labour/mach periods.
    for mach_p_start, seeding_days, mach_p_num in zip(mach_periods['date'], mach_periods['seed_days'],mach_periods.index):
        grazing_days_list=[]
        season_break = feed_periods.loc[0,'date']
        effective_break = season_break + destock_days #accounts for the time before seeding that destocking must occur
        for fp_date, fp_len in zip(feed_periods['date'], feed_periods['length']):
            fp_end_date = fp_date + dt.timedelta(days = fp_len)
            seed_end_date = mach_p_start + dt.timedelta(days = seeding_days)
            ##if the feed period finishes before the start of seeding it will recieve a grazing day for each day since the break of season times the number of seeding days in the current seed period minus the grazing days in the previous periods
            if fp_end_date <= mach_p_start:
                fp_grazing_days = max((fp_end_date- effective_break).days * seeding_days - sum(grazing_days_list) , 0) #max required incase fp ends before effective break - don't want a negitive value 
            ##if the end date of the feed period is after the end date of the seeding period it will get the full grazing days minus the grazing days in the previous periods
            elif fp_end_date >= seed_end_date:
                fp_grazing_days = max((mach_p_start- effective_break).days * seeding_days + (0.5 * seeding_days * seeding_days)  - sum(grazing_days_list) ,0)
            ##if it isn't one of the conditions above then the feed period date must fall somewhere within the seed periods. This means the grazing days equal to the number of days between the season start and the start of the current seed period plus the diminishing number of days in the current period
            else: 
                fp_grazing_days = max((mach_p_start- effective_break).days * seeding_days + (0.5 * seeding_days * seeding_days) -(0.5 * (seed_end_date-fp_end_date).days**2) - sum(grazing_days_list) , 0)
            grazing_days_list.append(fp_grazing_days)
        grazing_days_df[mach_p_num] = grazing_days_list #annoyingly both mach periods and feedperiods are defined as just numbers
        ##have to divide by the seeding days per period to return the grazing days if 1 ha was sown per period - this can now be multiplied by the ha sowed (done in pyomo)
        grazing_days_df[mach_p_num] = grazing_days_df[mach_p_num]/ seeding_days 
    return grazing_days_df.stack().to_dict()
        
#################################################
#seeding ha/day for each crop on each lmu  type #
#################################################

def seed_time_lmus():
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

def overall_seed_rate():
    '''
    Returns
    -------
    rate_direct_drill_day : Dict for pyomo
        Combines lmu seeding rate and crop adjustment to determine the seeding rate ha/day on each lmu type for each crop
    '''
    #convert seed time (hr/ha) to rate of direct drill per day (ha/day)
    seed_rate_lmus = 1 / seed_time_lmus().squeeze() * pinp.mach['daily_seed_hours'] 
    #adjusts the seeding rate (ha/day) for each different crop depending on its seeding speed vs wheat
    seedrate_df = pd.concat([uinp.mach[pinp.mach['option']]['seeder_speed_crop_adj']]*len(seed_rate_lmus),axis=1) #expands df for each lmu
    seedrate_df.columns = seed_rate_lmus.index #rename columns to lmu so i can mul
    seedrate_df=seedrate_df.mul(seed_rate_lmus)
    return seedrate_df.stack().to_dict()
    
  

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
    tractor_fuel_cost = fuel_used * fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
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

def seeding_cost_lmu():
    '''
    Returns
    -------
    DataFrame
        Total cost seeding on each lmu $/ha.
    '''
    return tractor_cost_seeding() + maint_cost_seeder()

def seeding_cost_period():
    '''
    Returns
    -------
    Dataframe for pyomo
        Returns the seeding cost allocation into cashflow periods.
    '''
    cost_df = seeding_cost_lmu()
    ##gets the date column of the cashflow periods df
    p_dates = per.cashflow_periods()['start date']
    ##gets the period name 
    p_name = per.cashflow_periods()['cash period']
    start = per.wet_seeding_start_date()
    length = dt.timedelta(days = sum(pinp.crop['seed_period_lengths']).astype(np.float64))
    return fun.period_allocation_reindex(cost_df, p_dates, p_name, start,length)

def contract_seed_cost():
    '''
    Returns
    -------
    Dict
        Contract seeding cost in each cashflow period, currently, contract cost is the same for all lmus and crops.
    '''
    ##gets the date column of the cashflow periods df
    p_dates = per.cashflow_periods()['start date']
    ##gets the period name 
    p_name = per.cashflow_periods()['cash period']
    seed_cost = uinp.price['contract_seed_cost']
    cash_period = fun.period_allocation(p_dates,p_name,per.wet_seeding_start_date())
    return {cash_period : seed_cost}


########################################
#late seeding & dry seeding penalty    #
########################################

def yield_penalty():
    '''
    Yields
    ------
    Dict for pyomo
        Calcs the penalty in each mach period (dry seeding and wet) - kg/ha/period/crop.
        Used in pyomo to calculate loss of cashflow and reduction in stubble
    '''
    mach_periods = per.p_dates_df()
    mach_penalty = pd.DataFrame()  #adds the average yield penalty for each crop for each period to the df 
    dry_seed_start = pinp.crop['dry_seed_start']
    dry_seed_end = pinp.feed_inputs['feed_periods'].loc[0,'date']#dry seeding finishes when the season breaks 
    seed_start = per.wet_seeding_start_date()
    seed_end = per.period_end_date(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths'])
    penalty_free_days = dt.timedelta(days = pinp.crop['seed_period_lengths'][0].astype(np.float64))
    yield_penalty_df = pinp.crop['yield_penalty'] 
    ##add the yield penalty for each period and each crop
    for k, wet_penalty, dry_penalty in zip(yield_penalty_df.index, yield_penalty_df['wet_seeding_penalty'],yield_penalty_df['dry_seeding_penalty']):
        for i in range(len(mach_periods['date'])-1):
            period_start_date = mach_periods.loc[i,'date']
            period_end = mach_periods.loc[i+1,'date']
            ###check penalty free 
            if seed_start  <= period_start_date  < seed_start + penalty_free_days:
                penalty =  0  
            ###check wet seeding 
            elif seed_start + penalty_free_days <= period_start_date:#  < seed_end:
                #penalty = average penalty of period (= (start day + end day) / 2 * penalty)
                start_day = 1 + (period_start_date - (seed_start + penalty_free_days)).days
                end_day = (period_end - (seed_start + penalty_free_days)).days
                penalty =  (start_day + end_day) / 2  * wet_penalty          
            ###check dry seeding
            elif dry_seed_start <= period_start_date <= dry_seed_end:
                penalty = dry_penalty
            ###other periods (between season break and seeding start & before dry seeding) get 500 penalty
            else: penalty = 500
            mach_penalty.loc[i, k] = penalty
    return mach_penalty.stack().to_dict()
# x = (yield_penalty())   


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
        - has to be kept as a seperate function because it is used in multiple places
        - harv sped is entered relative to a given yield for each crop
    '''
    harv_speed = uinp.mach[pinp.mach['option']]['harvest_speed']
    ##work rate hr/ha, determined from speed, size and eff
    return 10/ (harv_speed * uinp.mach[pinp.mach['option']]['harv_eff'] * uinp.mach[pinp.mach['option']]['harvester_width'])

def harv_rate_period():
    '''
    Returns
    -------
    Dict for pyomo
        Harv rate in each mach period for each crops.
        - account for crops that can be harvested early ie crops that can't be harvested early are given 0 harv rate in the first harv period
    '''
    mach_periods = per.p_dates_df()
    harv_rate_df = pd.DataFrame()
    harv_start = pinp.crop['harv_date']
    harv_end = per.period_end_date(harv_start, pinp.crop['harv_period_lengths'])
    ##Grain harvested per hr (t/hr) for each crop.
    harv_rate = uinp.mach_general['harvest_yield'] * (1 / harv_time_ha())
    ##loops through dict which contains harv start date for each crop
    ##this determines if the crop is allowed early harv
    for k, crop_harv_date in zip(pinp.crop['start_harvest_crops'].index, pinp.crop['start_harvest_crops']['date']):
        if k=='h':
            continue # this is required because hay is included in the harvest dates (needed for stubble) but not in any of the other harvest info
        for i in range(len(mach_periods['date'])-1):
            period_start_date = mach_periods.loc[i,'date']
            period_end = mach_periods.loc[i+1,'date']
            ###if the period is a harvest period
            if harv_start <= period_start_date  < harv_end:
                ####if crop harv date is before the end of the current period then it is allowed to be harvested in that period hence it is given a harv rate 
                if crop_harv_date < period_end: 
                    harvest_rate =  harv_rate.squeeze()[k]
                else: harvest_rate = 0
            else: harvest_rate = 0
            harv_rate_df.loc[i, k] = harvest_rate
    return harv_rate_df.stack().to_dict()
# harv_rate_period()  


#adds the max number of harv hours for each crop for each period to the df  
def max_harv_hours():
    mach_periods = per.p_dates_df()
    harv_start = pinp.crop['harv_date']
    harv_end = per.period_end_date(harv_start, pinp.crop['harv_period_lengths'])
    #loops through dict which contains harv start date for each crop
    #this determines if the crop is allowed early harv
    for i in range(len(mach_periods['date'])-1):
        period_start_date = mach_periods.loc[i,'date']
        period_end = mach_periods.loc[i+1,'date']
        if harv_start <= period_start_date  < harv_end:
            harv_days =  (period_end - period_start_date).days 
        else: harv_days = 0
        #convert to hours.
        mach_periods.loc[i, 'max_harv_hours'] = harv_days * pinp.mach['daily_harvest_hours']
    ## drop last row, because it has na because it only contains the end date, therefore not a period
    mach_periods.drop(mach_periods.tail(1).index,inplace=True) 
    return mach_periods['max_harv_hours'].to_dict()
#max_harv_hours()

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

def harvest_cost_period():
    '''
    Returns
    -------
    Dict for pyomo
        Cost of harvest in each cashflow period ($/hr).

    '''
    cost_df = cost_harv()
    #gets the date column of the cashflow periods df
    p_dates = per.cashflow_periods()['start date']
    #gets the period name 
    p_name = per.cashflow_periods()['cash period']
    start = pinp.crop['harv_date']
    length = dt.timedelta(days = sum(pinp.crop['harv_period_lengths']).astype(np.float64))
    return fun.period_allocation_reindex(cost_df, p_dates, p_name, start,length).stack().to_dict()


#########################
#contract harvesting    #
#########################

def contract_harv_rate():
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
    harv_rate = yield_approx * (1 / contract_harv_time_ha)
    return harv_rate.iloc[:,0].to_dict()
#print(contract_harv_rate())


def contract_harvest_cost_period():
    '''
    Returns
    -------
    Dict for pyomo
        Cost of contract harvest in each cashflow period ($/hr).
    '''
    cost_df = uinp.price['contract_harv_cost'] #contract harvesting cost for each crop ($/hr)
    #gets the date column of the cashflow periods df
    p_dates = per.cashflow_periods()['start date']
    #gets the period name 
    p_name = per.cashflow_periods()['cash period']
    start = pinp.crop['harv_date']
    length = dt.timedelta(days = sum(pinp.crop['harv_period_lengths']).astype(np.float64))
    return fun.period_allocation_reindex(cost_df, p_dates, p_name, start,length).stack().to_dict()


#########################
#make hay               #
#########################
def hay_making_cost():
    '''
    Returns
    -------
    Dict - used in pyomo 
        Cost to make hay ($/t).
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
    return (allocation * total_cost).stack().droplevel(1).to_dict() #drop level because i stacked to get it to a series but it was already 1d and i didn't want the col name as a key

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
#fert applicaation time   # used in labour crop also, defined here because it uses inputs from the differnt mach options which are consolidated at the top of this sheet
###########################

#time taked to spread 1ha (not including driving to and from paddock and filling up)
# hr/ha= 10/(width*speed*efficiency)
def time_ha():
      width_df = uinp.mach[pinp.mach['option']]['spreader_width']
      return 10/(width_df*uinp.mach[pinp.mach['option']]['spreader_speed']*uinp.mach[pinp.mach['option']]['spreader_eff'])

#time taken to driving to and from paddock and filling up
# hr/cubic m = ((ave distacne to paddock *2)/speed + fill up time)/ spreader capacity  # *2 because to and from paddock
def time_cubic():
      return (((pinp.mach['ave_pad_distance'] *2) 
              /uinp.mach[pinp.mach['option']]['spreader_speed'] + uinp.mach[pinp.mach['option']]['time_fill_spreader'])
              /uinp.mach[pinp.mach['option']]['spreader_cap'])
     

###################
#application cost # *remember that lime application only happens every 4 yrs - accounted for in the passes inputs
################### *used in crop pyomo
#this is split into two sections - new feature of midas
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
#chem applicaation time   # used in labour crop, defined here because it uses inputs from the differnt mach options which are consolidated at the top of this sheet
###########################

##time taked to spray 1ha (use efficiency input to allow for driving to and from paddock and filling up)
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
#fixed depreciation     #
#########################
    
def total_clearing_value():
    return sum(uinp.mach[pinp.mach['option']]['clearing_value']['value'])

#total valye of crop gear x dep rate x numver of crop gear
def fix_dep():
    return total_clearing_value() * uinp.finance['fixed_dep'] * pinp.mach['number_crop_gear']


####################################
#variable seeding depreciation     #
####################################
    
#not including harvester
def seeding_gear_clearing_value():
    #sprayer used for crop and pasture, so determine crop allocation
    sprayer_cost = uinp.mach[pinp.mach['option']]['clearing_value'].loc['sprayer','value'] * pinp.mach['sprayer_crop_allocation']
    return sprayer_cost + uinp.mach[pinp.mach['option']]['clearing_value'].loc['silo','value'] + uinp.mach[pinp.mach['option']]['clearing_value'].loc['auger','value'] \
    + uinp.mach[pinp.mach['option']]['clearing_value'].loc['tractor','value'] + uinp.mach[pinp.mach['option']]['clearing_value'].loc['seeder','value']
    
#
def seeding_dep():
    '''
    Returns
    -------
    Dict for pyomo
        Average variable dep for seeding $/ha.
    '''
    ##first determine the approx time to seed all the crop - which is equal to dep area x average seeding rate (hr/ha)
    average_seed_rate = seed_time_lmus().mean()
    seeding_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_seed_rate 
    ##second, determine dep per hour - equal to crop gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    dep_hourly = seeding_gear_clearing_value() * dep_rate / seeding_time
    ##third, convert to dep per ha for each soil type - equals cost per hr x seeding rate per hr
    dep_ha = dep_hourly * seed_time_lmus()
    return dep_ha.iloc[:,0].to_dict()


####################################
#variable harvest depreciation     #
####################################

def harvest_dep():
    '''
    Returns
    -------
    Float for pyomo
        Average variable dep for harvesting $/hr.
    '''
    ##first determine the approx time to harvest all the crop - which is equal to dep area x average harvest rate (hr/ha)
    average_harv_rate = harv_time_ha().mean()
    average_harv_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_harv_rate 
    ##second, determine dep per hour - equal to harv gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    dep_hourly = uinp.mach[pinp.mach['option']]['clearing_value'].loc['harvester','value'] * dep_rate / average_harv_time
    return dep_hourly.iloc[0]
#print(harv_time_ha())
#print(harvest_dep())

#######################################################################################################################################################
#######################################################################################################################################################
#insurance on all gear
#######################################################################################################################################################
#######################################################################################################################################################
def insurance():
    '''
    
    Returns
    -------
    Dict for pyomo.
        cost of insurance for all machinery
    '''
    ##determine the insurance paid
    value_all_mach = total_clearing_value()
    value_crop_gear = seeding_gear_clearing_value()
    total_value = value_all_mach - value_crop_gear + value_crop_gear * pinp.mach['number_crop_gear'] #accounts for the number of crop gear
    insurance = total_value * uinp.finance['equip_insurance']
    ##determine cash period
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    start = uinp.mach_general['insurance_date']
    allocation=fun.period_allocation(p_dates, p_name,start)
    return {allocation:insurance}



