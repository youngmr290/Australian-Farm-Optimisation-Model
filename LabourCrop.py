# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:13:44 2019

Module - calcs for crop labour

@author: young
"""
#python modules
import pandas as pd
import numpy as np#datetime
import timeit

#MUDAS modules
from LabourCropInputs import *
import Functions as fun
import Periods as per
import Mach as mac
import PropertyInputs as pinp
import UniversalInputs as uinp

########################
#phases                #
########################
##makes a df of all possible rotation phases
phases_df =uinp.structure['phases']
phases_df2=phases_df.copy() #make a copy so that it doesn't alter the phases df that exists outside this func
phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns, ['']])  #make the df multi index so that when it merges with other df below the indexs remanin seperate (otherwise it turn into a one leveled tuple)


#########################
#pack and prep time     #
#########################
#this function just combines all the needed elements to call the dict_period_total function.
#what is happening; i have a number of dicts that contain dates and the number of hours of labour for that date
#i want to combine and end up with the total hours of work done for each labour period
def prep_labour():
    p_dates = per.p_date2_df()['date']
    #gets the period name 
    p_name = per.p_date2_df().index
    #list of all the dicts that i want to combine
    dicts=crop_labour_input['harvest_prep'],crop_labour_input['fert_prep'] \
    , crop_labour_input['spray_prep'], crop_labour_input['seed_prep']
    return fun.dict_period_total(p_dates, p_name, *dicts) # '*' used to unpack list into seperate items for func

###########################
#fert applicaation time   #  this is similar to app cost done in mach sheet
###########################
#this is split into two sections - new feature of midas
# 1- time to drive around 1ha
# 2- time per cubic metre ie to represent filling up and driving to and from paddock

#allocation of fert costs into each cash period for each fert ie depending on the date diff ferts are in diff cash periods
def lab_allocation():
    start_dict = pinp.crop['fert_info']['app_date'] 
    length_dict = pinp.crop['fert_info']['app_len'].astype('timedelta64[D]') 
    p_dates = per.p_date2_df()['date']
    p_name = per.p_date2_df().index
    return fun.period_allocation2(start_dict, length_dict, p_dates, p_name)


#time/per ha - needs to be multiplied by the number of phases and then added to phases df because the previous phases can effect number of passes and hence time
#also need to account for arable area
def fert_app_time_ha():
    passes = pinp.crop['passes'].reset_index().pivot(index='fert',columns='index').T
    arable = pinp.crop['arable']
    arable=arable.reindex(passes.index, axis=0, level=1).stack()
    passes=passes.reindex(arable.index).T.mul(arable).T
    time = lab_allocation().mul(mac.time_ha().stack().droplevel(1)).stack()
    time = passes.reindex(time.index, axis=1,level=1).mul(time).unstack().swaplevel(0,2,axis=1) #swaplevel so that i can set index in the last step otherwise error because column names of the phases ie 0 -3 is the same as period numbers and they were both level 0 col index
    time = time.sum(level=[0,2], axis=1).replace(0, np.nan) #sum each fert - labour doesn't need to be seperated by fert type once joined with passes
                                                          #sum nan returns 0 therefore i need to convert 0 back to nan so that they are dropped when stacking to reduce dict size.
    phase_time = pd.merge(phases_df2, time, how='left', left_on=uinp.cols(), right_index = True)
    return phase_time.drop(list(range(uinp.structure['phase_len'])),axis=1,level=0).stack([0,1]).to_dict()  
    # return phase_time.set_index(list(range(rinp.rotation_data['phase_len']))).stack([0,1]).to_dict() 
#f=fert_app_time_ha()
#print(timeit.timeit(fert_app_time_ha,number=20)/20)

#time/t - need to convert m3 to tone and allocate into lab periods
def fert_app_time_t():
    spreader_proportion = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']])
    conversion = pd.DataFrame([pinp.crop['fert_info']['fert_density']])
    time = mac.time_cubic() * conversion * spreader_proportion
    return (time.iloc[0]*lab_allocation()).stack().to_dict()
#print(fert_app_time_t())    
    


###########################
#chem applicaation time   #  this is similar to app cost done in mach sheet
###########################

def chem_lab_allocation():
    '''
    Returns
    -------
    DataFrame
        Collates all the data needed then calls the allocation function, which returns \
        the allocation of labour for chem application into labour periods.
    '''
    start_dict = pinp.crop['chem_info']['app_date'] 
    length_dict = pinp.crop['chem_info']['app_len'].astype('timedelta64[D]') 
    p_dates = per.p_date2_df()['date']
    p_name = per.p_date2_df().index
    return fun.period_allocation2(start_dict, length_dict, p_dates, p_name)


def chem_app_time_ha():  
    '''
    Returns
    ----------
    Dict for pyomo
        Labour required by each rotation phase for spraying
    '''
    ##adjust passes for arable area.
    arable = pinp.crop['arable'] #read in arable area df
    passes = pinp.crop['chem_passes'].reset_index().pivot(index='chem',columns='current yr').T #passes over each ha for each chem type
    arable3=arable.reindex(passes.index, axis=0, level=1).stack() #reindex so it can be mul with passes
    passes=passes.reindex(arable3.index).T.mul(arable3).T
    ##adjust chem labour across each labour period
    time = chem_lab_allocation().mul(mac.spray_time_ha()).stack() #time for 1 pass for each chem.
    ##adjust for passes
    time = passes.reindex(time.index, axis=1,level=1).mul(time) #total time 
    time=time.sum(level=[0], axis=1).replace(0, np.nan).unstack() #sum each chem  - time doesn't need to be seperated by chem type once joined with passes #sum nan returns 0 therefore i need to convert 0 back to nan so that they are dropped when stacking to reduce dict size.
    ##merge to full rotation df
    phase_time = pd.merge(phases_df2, time, how='left', left_on=uinp.cols(), right_index = True) #merge with all the phases, requires because different phases have different application passes
    phase_time = phase_time.drop(list(range(uinp.structure['phase_len'])),axis=1,level=0).stack([1,0]) #adding level=0 does nothing but if not included you get a preformance warning.
    return phase_time.to_dict()
# t_chemlab=chem_app_time_ha()

    



