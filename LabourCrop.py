# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:13:44 2019

Module - calcs for crop labour

@author: young
"""
#python modules
import pandas as pd
import numpy as np
import datetime as dt
import timeit

#AFO modules
# from LabourCropInputs import *
import Functions as fun
import Crop as crp
import Periods as per
import Mach as mac
import PropertyInputs as pinp
import UniversalInputs as uinp

########################
#phases                #
########################

#########################
#pack and prep time     #
#########################
#this function just combines all the needed elements to call the dict_period_total function.
#what is happening; i have a number of dicts that contain dates and the number of hours of labour for that date
#i want to combine and end up with the total hours of work done for each labour period
def prep_labour(params):
    p_dates = per.p_date2_df()['date']
    #gets the period name 
    p_name = per.p_date2_df().index
    #list of all the dicts that i want to combine
    dfs=pinp.labour['harvest_prep'],pinp.labour['fert_prep'] \
    , pinp.labour['spray_prep'], pinp.labour['seed_prep']
    params['prep_labour'] = fun.df_period_total(p_dates, p_name, *dfs) # '*' used to unpack list into separate items for func

###########################
#fert application time   #  this is similar to app cost done in mach sheet
###########################
#this is split into two sections - new feature of midas
# 1- time to drive around 1ha
# 2- time per cubic metre ie to represent filling up and driving to and from paddock

#allocation of fert costs into each cash period for each fert ie depending on the date diff ferts are in diff cash periods
def lab_allocation():
    start_df = pinp.crop['fert_info']['app_date'] 
    length_df = pinp.crop['fert_info']['app_len'].astype('timedelta64[D]') 
    p_dates = per.p_dates_df()['date']
    p_name = per.p_dates_df().index
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)


#time/per ha - needs to be multiplied by the number of phases and then added to phases df because the previous phases can effect number of passes and hence time
#also need to account for arable area
def fert_app_time_ha(params):
    ##fert passes - arable (arable area accounted for in passes function)
    passes_arable = crp.f_fert_passes()
    ##non arable fert passes
    passes_na = crp.f_nap_fert_passes() #on pasture phases only
    ##add fert for arable area and fert for nonarable area
    total_passes = pd.concat([passes_arable, passes_na], axis=1).sum(axis=1, level=0)
    ##adjust fert labour across each labour period
    time = lab_allocation().mul(mac.time_ha().squeeze()).stack() #time for 1 pass for each chem.
    ##adjust for passes
    time = total_passes.mul(time, axis=1,level=1) #total time
    time=time.sum(level=[0], axis=1).stack() #sum across fert type
    params['fert_app_time_ha'] = time.to_dict()  #add to precalc dict

#f=fert_app_time_ha()
#print(timeit.timeit(fert_app_time_ha,number=20)/20)

#time/t - need to convert m3 to tone and allocate into lab periods
def fert_app_time_t(params):
    spreader_proportion = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']])
    conversion = pd.DataFrame([pinp.crop['fert_info']['fert_density']])
    time = (mac.time_cubic() / conversion).mul(spreader_proportion.squeeze(),axis=1)
    params['fert_app_time_t'] = (time.iloc[0]*lab_allocation()).stack().to_dict()
#print(fert_app_time_t())    
    


###########################
#chem application time   #  this is similar to app cost done in mach sheet
###########################

def chem_lab_allocation():
    '''
    Returns
    -------
    DataFrame
        Collates all the data needed then calls the allocation function, which returns \
        the allocation of labour for chem application into labour periods.
    '''
    start_df = pinp.crop['chem_info']['app_date'] 
    length_df = pinp.crop['chem_info']['app_len'].astype('timedelta64[D]') 
    p_dates = per.p_dates_df()['date']
    p_name = per.p_dates_df().index
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)


def chem_app_time_ha(params):  
    '''
    Returns
    ----------
    Dict for pyomo
        Labour required by each rotation phase for spraying
        -arable area accounted for in crop.py
    '''
    ##passes
    passes = crp.f_chem_application()
    ##adjust chem labour across each labour period
    time = chem_lab_allocation().mul(mac.spray_time_ha()).stack() #time for 1 pass for each chem.
    ##adjust for passes
    time = passes.mul(time, axis=1,level=1) #total time
    time=time.sum(level=[0], axis=1).stack()
    params['chem_app_time_ha'] = time.to_dict()
    
# # t_chemlab=chem_app_time_ha()

    


###########################
#crop monitoring time     #
###########################

def f_crop_monitoring(params):
    '''
    Returns
    -------
    Dict for pyomo
    '''
    ##allocation
    crop_monitor = pinp.labour['crop_monitoring']
    date_start_d = crop_monitor.columns.values.astype('datetime64[D]')
    date_end_d = np.roll(date_start_d, -1)
    date_end_d[-1] = date_end_d[-1] + 365 #increment the first date by 1yr so it becomes the end date for the last period
    length_d = date_end_d - date_start_d
    monitoring_allocation_pd = fun.range_allocation_np(per.p_dates_df()['date']
                                                    , date_start_d, length_d, opposite=True)

    ## drop last row, because it has na because it only contains the end date, therefore not a period
    monitoring_allocation_pd = monitoring_allocation_pd[:-1]

    ##adjust to monitoring time per ha
    crop_monitor_kd = crop_monitor.values #convert to numpy
    crop_monitor_kd = crop_monitor_kd/pinp.general['pad_size'] * length_d.astype(float)/7

    ##convert date range to labour periods
    crop_monitor_kp = np.sum(crop_monitor_kd[:,np.newaxis,:] * monitoring_allocation_pd, axis=-1) #sum the d axis (monitoring date axis)

    ##convert to dict and expand landuse to rotation
    crop_monitor = pd.DataFrame(crop_monitor_kp, index=crop_monitor.index, columns=per.p_dates_df().index[:-1])
    phases_df = uinp.structure['phases']
    crop_monitor = pd.merge(phases_df, crop_monitor, how='left', left_on=uinp.cols()[-1], right_index = True) #merge with all the phases

    params['crop_monitoring'] = crop_monitor.drop(list(range(uinp.structure['phase_len'])), axis=1).stack().to_dict()















