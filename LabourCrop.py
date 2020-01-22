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
import CropInputs as ci
import Inputs as inp
import RotationInputs as rinp

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
    length_dict = pinp.crop['fert_info']['app_len'] 
    p_dates = per.p_date2_df()['date']
    p_name = per.p_date2_df().index
    return fun.period_allocation2(start_dict, length_dict, p_dates, p_name)


#time/per ha - needs to be multiplied by the number of phases and then added to phases df because the previous phases can effect number of passes and hence time
#also need to account for arable area
def fert_app_time_ha():
    passes = ci.crop_input['excel_ranges']['passes'].reset_index().pivot(index='fert',columns='index').T
    arable = ci.crop_input['excel_ranges']['arable']
    arable=arable.reindex(passes.index, axis=0, level=1).stack()
    passes=passes.reindex(arable.index).T.mul(arable).T
    time = lab_allocation().mul(mac.time_ha().stack().droplevel(0)).stack()
    time = passes.reindex(time.index, axis=1,level=1).mul(time).unstack().swaplevel(0,2,axis=1) #swaplevel so that i can set index in the last step otherwise error because column names of the phases ie 0 -3 is the same as period numbers and they were both level 0 col index
    time = time.sum(level=[0,2], axis=1).replace(0, np.nan) #sum each fert - labour doesn't need to be seperated by fert type once joined with passes
                                                          #sum nan returns 0 therefore i need to convert 0 back to nan so that they are dropped when stacking to reduce dict size.
    phases_df =pd.Series(inp.input_data['rotations']['rot_phase']).str.split(expand=True) #makes a df of all possible rotation phases
    phases_df.columns = pd.MultiIndex.from_product([phases_df.columns, ['']]) #make the df multi index so that when it merges with other df below the indexs remanin seperate (otherwise it turn into a one leveled tuple)
    phase_time = pd.merge(phases_df, time, how='left', left_on=inp.cols(), right_index = True)
    return phase_time.set_index(list(range(rinp.rotation_data['phase_len']))).stack([0,1]).to_dict() 
#f=fert_app_time_ha()
#print(timeit.timeit(fert_app_time_ha,number=20)/20)

#time/t - need to convert m3 to tone and allocate into lab periods
def fert_app_time_t():
    spreader_proportion = pd.DataFrame([ci.crop_input['spreader_proportion']])
    conversion = pd.DataFrame([ci.crop_input['fert_density']])
    time = mac.time_cubic() * conversion * spreader_proportion
    return (time.iloc[0]*lab_allocation()).stack().to_dict()
#print(fert_app_time_t())    
    





