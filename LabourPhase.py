# -*- coding: utf-8 -*-
"""

author: young

Phase labour represents the labour associated with each rotation phase. This
includes the both crop and pasture phases. Phase labour includes the labour required
for seeding, harvest, spraying, fertilising and monitoring. The time each operation takes
is dependent on the rotation phase, LMU and machinery complement. The rate at which
seeding, harvest, spraying and fertilising can be done is calculated and documented in the machinery
section. In this section the machine time for each operation is converted to a labour required by including
a helper factor and allocating it to the labour periods. For example, the machinery time taken for
harvest is equal to the time the harvester is running but from a labour perspective there is a
header driver, chaser bin driver and often some helper labour for busy times such as moving paddocks.


"""
#python modules
import pandas as pd
import numpy as np
import datetime as dt
import timeit

#AFO modules
# from LabourCropInputs import *
import Functions as fun
import Phase as phs
import Periods as per
import Mach as mac
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp

na = np.newaxis

#########################
#pack and prep time     #
#########################
def f_prep_labour():
    '''
    Labour required for preparation and packing for each cropping operation.

    '''
    ##inputs
    labour_period = per.p_dates_df()
    lp_p5z = labour_period.values
    keys_p5 = labour_period.index[:-1]
    keys_z = pinp.f_keys_z()

    ##harvest_prep
    harvest_prep_dates_p8 = pinp.labour['harvest_prep'].index.values
    harvest_prep_length_p8 = pinp.labour['harvest_prep']['days'].values.astype('timedelta64[D]')
    harvest_prep_labour_p8 = pinp.labour['harvest_prep']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], harvest_prep_dates_p8, harvest_prep_length_p8, True)[:-1,:,:]
    harvest_prep_p5z = np.sum(alloc_p5zp8 * harvest_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##fert_prep
    fert_prep_dates_p8 = pinp.labour['fert_prep'].index.values
    fert_prep_length_p8 = pinp.labour['fert_prep']['days'].values.astype('timedelta64[D]')
    fert_prep_labour_p8 = pinp.labour['fert_prep']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], fert_prep_dates_p8, fert_prep_length_p8, True)[:-1,:,:]
    fert_prep_p5z = np.sum(alloc_p5zp8 * fert_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##spray_prep
    spray_prep_dates_p8 = pinp.labour['spray_prep'].index.values
    spray_prep_length_p8 = pinp.labour['spray_prep']['days'].values.astype('timedelta64[D]')
    spray_prep_labour_p8 = pinp.labour['spray_prep']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], spray_prep_dates_p8, spray_prep_length_p8, True)[:-1,:,:]
    spray_prep_p5z = np.sum(alloc_p5zp8 * spray_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##seed_prep
    seed_prep_dates_p8 = pinp.labour['seed_prep'].index.values
    seed_prep_length_p8 = pinp.labour['seed_prep']['days'].values.astype('timedelta64[D]')
    seed_prep_labour_p8 = pinp.labour['seed_prep']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], seed_prep_dates_p8, seed_prep_length_p8, True)[:-1,:,:]
    seed_prep_p5z = np.sum(alloc_p5zp8 * seed_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##sum all and make df
    prep_p5z = seed_prep_p5z + spray_prep_p5z + fert_prep_p5z + harvest_prep_p5z
    prep_p5z = pd.DataFrame(prep_p5z,index=keys_p5,columns=keys_z)
    return prep_p5z



###########################
#fert application time   #  this is similar to app cost done in mach sheet
###########################
#this is split into two sections - new feature of AFO
# 1- time to drive around 1ha
# 2- time per cubic metre ie to represent filling up and driving to and from paddock

#allocation of fert costs into each cash period for each fert ie depending on the date diff ferts are in diff cash periods
def f_lab_allocation():
    '''Allocation of fertiliser applications into each labour period'''

    fert_info = pinp.crop['fert_info']
    fert_date_f = fert_info['app_date'].values
    fert_length_f = fert_info['app_len'].values.astype('timedelta64[D]')
    p_dates_p5z = per.p_dates_df()
    keys_p5 = per.p_dates_df().index[:-1]
    shape_p5zf = p_dates_p5z.shape+fert_date_f.shape
    alloc_p5zf = fun.range_allocation_np(p_dates_p5z.values[...,na], fert_date_f, fert_length_f, True, shape=shape_p5zf)[:-1,...]
    ##put in df
    alloc_p5zf = alloc_p5zf.reshape(alloc_p5zf.shape[0], -1)
    keys_z = pinp.f_keys_z()
    cols = pd.MultiIndex.from_product([keys_z, fert_info.index])
    alloc_p5zf = pd.DataFrame(alloc_p5zf, index=keys_p5, columns=cols)
    return alloc_p5zf


#time/per ha - needs to be multiplied by the number of phases and then added to phases df because the previous phases can effect number of passes and hence time
#also need to account for arable area
def f_fert_app_time_ha():
    '''

    Fertilising labour part 1: time required per hectare.

    The labour required for fertilising is calculated in two parts. Part 1 is the time required per hectare
    for each rotation phase which represents the time taken spreading fertiliser in the paddock (calculated in Mach.py).
    This is adjusted for the number of fertiliser applications and allocated into a labour period/s.

    '''

    ##fert passes - arable (arable area accounted for in passes function)
    passes_arable = phs.f_fert_passes()
    ##non arable fert passes
    passes_na = phs.f_nap_fert_passes() #on pasture phases only
    ##add fert for arable area and fert for nonarable area, na_fert doesnt have season axis so need to reindex first
    passes_na = passes_na.unstack().reindex(passes_arable.unstack().index, axis=0, level=0).stack()
    total_passes = pd.concat([passes_arable, passes_na], axis=1).sum(axis=1, level=0)
    ##adjust fert labour across each labour period
    time = f_lab_allocation().mul(mac.time_ha().squeeze(), axis=1, level=1) #time for 1 pass for each chem.
    ##adjust for passes
    time = time.stack(1)
    total_passes = total_passes.reindex(time.index, axis=1, level=1).unstack(1)
    time = total_passes.mul(time.stack(), axis=1) #total time
    time=time.sum(level=[0,2], axis=1).stack(0) #sum across fert type
    return time

#f=fert_app_time_ha()
#print(timeit.timeit(fert_app_time_ha,number=20)/20)

#time/t - need to convert m3 to tone and allocate into lab periods
def f_fert_app_time_t():
    '''

    Fertilising labour part 2: time required per tonne.

    The labour required for fertilising is calculated in two parts. Part 2 is the time required per tonne
    which represents the time taken driving to and from the paddock and filling up (calculated in Mach.py).
    This is adjusted for the number of fertiliser applications and allocated into a labour period/s.


    '''

    spreader_proportion = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']])
    conversion = pd.DataFrame([pinp.crop['fert_info']['fert_density']])
    time = (mac.time_cubic() / conversion).mul(spreader_proportion.squeeze(),axis=1)
    allocation = f_lab_allocation()
    return allocation.mul(time.squeeze(), axis=1, level=1).stack(1)
#print(fert_app_time_t())    
    


###########################
#chem application time   #  this is similar to app cost done in mach sheet
###########################

def f_chem_lab_allocation():
    '''Allocation of chemical applications into each labour period'''
    chem_info = pinp.crop['chem_info']
    chem_date_f = chem_info['app_date'].values
    chem_length_f = chem_info['app_len'].values.astype('timedelta64[D]')
    p_dates_p5z = per.p_dates_df()
    keys_p5 = per.p_dates_df().index[:-1]
    shape_p5zf = p_dates_p5z.shape+chem_date_f.shape
    alloc_p5zf = fun.range_allocation_np(p_dates_p5z.values[...,na], chem_date_f, chem_length_f, True, shape=shape_p5zf)[:-1,...]
    ##put in df
    alloc_p5zf = alloc_p5zf.reshape(alloc_p5zf.shape[0], -1)
    keys_z = pinp.f_keys_z()
    cols = pd.MultiIndex.from_product([keys_z, chem_info.index])
    alloc_p5zf = pd.DataFrame(alloc_p5zf, index=keys_p5, columns=cols)
    return alloc_p5zf


def f_chem_app_time_ha():
    '''

    Calculate labour required for spraying for each rotation and allocate it into the labour periods.

    The labour required for spraying is calculated from the time to spray 1ha (calculated in Mach.py)
    and the number of chemical applications for each rotation phase (calculated in Phase.py).

    '''

    ##note arable area accounted for in crop.py

    ##passes
    passes = phs.f_chem_application()
    ##adjust chem labour across each labour period
    time = f_chem_lab_allocation() * mac.spray_time_ha() #time for 1 pass for each chem.
    ##adjust for passes
    time = time.stack(1)
    passes = passes.reindex(time.index, axis=1, level=1).unstack(1)
    time = passes.mul(time.stack(), axis=1) #total time
    time = time.stack(0).sum(level=[1], axis=1) #sum across fert type
    return time




###########################
#crop monitoring time     #
###########################

def f_crop_monitoring():
    '''
    Labour required for monitoring crop paddocks.

    For crop paddocks, monitoring time is broken into two section.
    Firstly, a fixed (irrelevant of crop area) labour requirement which is a user defined input stating
    the hours per week in each labour period. Secondly, a variable labour requirement which is incurred
    for each hectare of crop. This is also an input, but it can vary by crop type as well as period. The
    logic behind splitting the monitoring into two components is that typically farmers will spend longer
    examining a small number of paddocks, irrelevant of the total crop area and then examine the remaining
    paddocks much faster.

    '''
    ##allocation
    fixed_crop_monitor = pinp.labour['fixed_crop_monitoring']
    variable_crop_monitor = pinp.labour['variable_crop_monitoring']
    labour_periods_pz = per.p_dates_df().values
    date_start_d = fixed_crop_monitor.columns.values.astype('datetime64[D]')
    date_end_d = np.roll(date_start_d, -1)
    date_end_d[-1] = date_end_d[-1] + 365 #increment the first date by 1yr so it becomes the end date for the last period
    length_d = date_end_d - date_start_d
    shape_pzd = labour_periods_pz.shape + date_start_d.shape
    monitoring_allocation_pzd = fun.range_allocation_np(labour_periods_pz[...,na]
                                                    , date_start_d, length_d, opposite=True, shape=shape_pzd)

    ## drop last row, because it has na because it only contains the end date, therefore not a period
    monitoring_allocation_pzd = monitoring_allocation_pzd[:-1]

    ##variable monitoring
    ###adjust to monitoring time per ha
    variable_crop_monitor_kd = variable_crop_monitor.values #convert to numpy
    variable_crop_monitor_kd = variable_crop_monitor_kd/pinp.general['pad_size'] * length_d.astype(float)/7
    ###convert date range to labour periods
    variable_crop_monitor_kpz = np.sum(variable_crop_monitor_kd[:,na,na,:] * monitoring_allocation_pzd, axis=-1) #sum the d axis (monitoring date axis)
    ###convert to dict and expand landuse to rotation
    variable_crop_monitor_kpz = variable_crop_monitor_kpz.reshape(variable_crop_monitor_kpz.shape[0], -1)
    keys_z = pinp.f_keys_z()
    keys_p5 = per.p_dates_df().index[:-1]
    cols = pd.MultiIndex.from_product([keys_p5, keys_z])
    variable_crop_monitor = pd.DataFrame(variable_crop_monitor_kpz, index=variable_crop_monitor.index, columns=cols)
    phases_df = sinp.f_phases()
    phases_df.columns = pd.MultiIndex.from_product([phases_df.columns,['']])
    variable_crop_monitor = pd.merge(phases_df, variable_crop_monitor, how='left', left_on=sinp.end_col(), right_index = True) #merge with all the phases
    variable_crop_monitor = variable_crop_monitor.drop(list(range(sinp.general['phase_len'])), axis=1).stack(0)

    ##fixed monitoring
    ###adjust from hrs/week to hrs/period
    fixed_crop_monitor_d = fixed_crop_monitor.values
    fixed_crop_monitor_d = fixed_crop_monitor_d * length_d.astype(float)/7
    ###convert date range to labour periods
    fixed_crop_monitor_pz = np.sum(fixed_crop_monitor_d * monitoring_allocation_pzd, axis=-1) #sum the d axis (monitoring date axis)
    fixed_crop_monitor = pd.DataFrame(fixed_crop_monitor_pz, index=keys_p5, columns=keys_z)
    return variable_crop_monitor.stack(), fixed_crop_monitor.stack()

##collates all the params
def f1_labcrop_params(params,r_vals):
    prep_labour = f_prep_labour().stack()
    fert_app_time_t = f_fert_app_time_t().stack()
    fert_app_time_ha = f_fert_app_time_ha().stack()
    chem_app_time_ha = f_chem_app_time_ha().stack()
    variable_crop_monitor, fixed_crop_monitor = f_crop_monitoring()

    ##add params which are inputs
    params['harvest_helper'] = pinp.labour['harvest_helper'].squeeze().to_dict()
    params['daily_seed_hours'] = pinp.mach['daily_seed_hours']
    params['seeding_helper'] = pinp.labour['seeding_helper']
    params['prep_labour'] = prep_labour.to_dict()
    params['fert_app_time_t'] = fert_app_time_t.to_dict()
    params['fert_app_time_ha'] = fert_app_time_ha.to_dict()
    params['chem_app_time_ha'] = chem_app_time_ha.to_dict()
    params['variable_crop_monitor'] = variable_crop_monitor.to_dict()
    params['fixed_crop_monitor'] = fixed_crop_monitor.to_dict()













