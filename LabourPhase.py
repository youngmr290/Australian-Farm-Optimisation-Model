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
import SeasonalFunctions as zfun
import Periods as per
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Phase as phs
import Mach as mac
import RotationPhases as rps

na = np.newaxis

##general
def f_p5_p7_allocation():
    '''Allocation of labour in each p5 to each p7 period.'''
    labour_period_p5z = per.f_p_dates_df()
    labour_period_start_p5z = labour_period_p5z.values[:-1]
    labour_period_end_p5z = labour_period_p5z.values[1:]
    length_p5z = labour_period_start_p5z - labour_period_end_p5z
    ##allocate p5 to p7
    alloc_p7p5z = zfun.f1_z_period_alloc(labour_period_start_p5z[na,:,:],length_p5z[na,:,:],z_pos=-1)
    return alloc_p7p5z


#########################
#pack and prep time     #
#########################
def f_prep_labour():
    '''
    Labour required for preparation and packing for each cropping operation.

    Not linked to v_phase therefore no p7 axis required.
    '''
    ##inputs
    labour_period = per.f_p_dates_df()
    lp_p5z = labour_period.values
    keys_p5 = labour_period.index[:-1]
    keys_z = zfun.f_keys_z()

    ##harvest_prep
    harvest_prep_dates_p8 = pinp.labour['harvest_prep'].index.values
    harvest_prep_length_p8 = pinp.labour['harvest_prep']['days'].values
    harvest_prep_labour_p8 = pinp.labour['harvest_prep']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], harvest_prep_dates_p8, harvest_prep_length_p8)[:-1,:,:]
    harvest_prep_p5z = np.sum(alloc_p5zp8 * harvest_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##fert_prep
    fert_prep_dates_p8 = pinp.labour['fert_prep'].index.values
    fert_prep_length_p8 = pinp.labour['fert_prep']['days'].values
    fert_prep_labour_p8 = pinp.labour['fert_prep']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], fert_prep_dates_p8, fert_prep_length_p8)[:-1,:,:]
    fert_prep_p5z = np.sum(alloc_p5zp8 * fert_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##spray_prep
    spray_prep_dates_p8 = pinp.labour['spray_prep'].index.values
    spray_prep_length_p8 = pinp.labour['spray_prep']['days'].values
    spray_prep_labour_p8 = pinp.labour['spray_prep']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], spray_prep_dates_p8, spray_prep_length_p8)[:-1,:,:]
    spray_prep_p5z = np.sum(alloc_p5zp8 * spray_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##seed_prep
    seed_prep_dates_p8 = pinp.labour['seed_prep'].index.values
    seed_prep_length_p8 = pinp.labour['seed_prep']['days'].values
    seed_prep_labour_p8 = pinp.labour['seed_prep']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], seed_prep_dates_p8, seed_prep_length_p8)[:-1,:,:]
    seed_prep_p5z = np.sum(alloc_p5zp8 * seed_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##sum all and make df
    prep_p5z = seed_prep_p5z + spray_prep_p5z + fert_prep_p5z + harvest_prep_p5z

    ##mask z axis (not required for other params because they have p7 axis thus are already masked)
    maskz8_p5z = zfun.f_season_transfer_mask(lp_p5z[:-1,:],z_pos=-1,mask=True) #slice off the end date of the last period
    prep_p5z = prep_p5z * maskz8_p5z

    ##make df
    prep_p5z = pd.DataFrame(prep_p5z,index=keys_p5,columns=keys_z)
    return prep_p5z


###########################
#fert application time   #  this is similar to app cost done in mach sheet
###########################
#this is split into two sections - new feature of AFO
# 1- time to drive around 1ha
# 2- time per cubic metre ie to represent filling up and driving to and from paddock

#allocation of fert costs into each lab period for each fert ie depending on the date diff ferts are in diff lab periods
def f_fert_lab_allocation():
    '''Allocation of fertiliser applications into each labour period'''

    fert_info = pinp.crop['fert_info']
    fert_date_n = fert_info['app_date'].values
    fert_length_n = fert_info['app_len'].values
    p_dates_p5z = per.f_p_dates_df()
    shape_p5zn = p_dates_p5z.shape+fert_date_n.shape
    alloc_p5zn = fun.f_range_allocation_np(p_dates_p5z.values[...,na], fert_date_n, fert_length_n, shape=shape_p5zn)[:-1,...]
    ##allocate to p7
    alloc_p7p5z = f_p5_p7_allocation()
    alloc_p7p5zn = alloc_p5zn * alloc_p7p5z[...,na]
    ##put in df
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    keys_p7 = per.f_season_periods(keys=True)
    index_p7p5zn = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_z, fert_info.index])
    alloc_p7p5zn = pd.Series(alloc_p7p5zn.ravel(), index=index_p7p5zn)
    return alloc_p7p5zn

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
    ##add fert for arable area and fert for nonarable area, na_fert doesn't have season axis so need to reindex first
    passes_na = passes_na.unstack().reindex(passes_arable.unstack().index, axis=0, level=0).stack()
    total_passes_rzln = pd.concat([passes_arable, passes_na], axis=1).groupby(axis=1, level=0).sum().stack()
    ##time taken to cover 1ha while spreading
    time_ha_n = mac.time_ha().squeeze()
    ##adjust fert labour across each labour period
    alloc_p7p5zn = f_fert_lab_allocation()
    time_p7p5z_n = alloc_p7p5zn.mul(time_ha_n, level=-1).unstack() #time for 1 pass for each chem.
    ##adjust for passes
    time_p7p5z_rln = time_p7p5z_n.reindex(total_passes_rzln.unstack(1).index, axis=1, level=2)
    time_p7p5_rzln = time_p7p5z_rln.unstack().reorder_levels([0,3,1,2], axis=1).sort_index(axis=1)
    fert_app_time_ha_p7p5_rzln = time_p7p5_rzln.mul(total_passes_rzln, axis=1)
    fert_app_time_ha_p7p5_rzl = fert_app_time_ha_p7p5_rzln.groupby(axis=1, level=(0,1,2)).sum() #sum fert type
    fert_app_time_ha_rzlp5p7 = fert_app_time_ha_p7p5_rzl.unstack([1,0])

    ##create params for v_phase_increment
    increment_fert_app_time_ha_rzlp5p7 = rps.f_v_phase_increment_adj(fert_app_time_ha_rzlp5p7,p7_pos=-1,z_pos=-4,p5_pos=-2)

    return fert_app_time_ha_rzlp5p7, increment_fert_app_time_ha_rzlp5p7

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
    ##fert used in each rotation phase
    fert_total_rzln = phs.f1_total_fert_req()/1000 #convert to tonnes

    ##time per tonne
    spreader_proportion = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']])
    conversion = pd.DataFrame([pinp.crop['fert_info']['fert_density']])
    time_n = ((mac.time_cubic() / conversion).mul(spreader_proportion.squeeze(),axis=1)).squeeze()

    ##p5 and p7 allocation
    alloc_p7p5zn = f_fert_lab_allocation()
    time_p7p5z_n = alloc_p7p5zn.mul(time_n, level=-1).unstack() #time for 1 tonne for each fert.

    ##combine with total phase fert
    time_p7p5z_rln = time_p7p5z_n.reindex(fert_total_rzln.unstack(1).index, axis=1, level=2)
    time_p7p5_rzln = time_p7p5z_rln.unstack().reorder_levels([0,3,1,2], axis=1).sort_index(axis=1)
    fert_app_time_tonne_p7p5_rzln = time_p7p5_rzln.mul(fert_total_rzln, axis=1)
    fert_app_time_tonne_p7p5_rzl = fert_app_time_tonne_p7p5_rzln.groupby(axis=1, level=(0,1,2)).sum() #sum fert type
    fert_app_time_tonne_rzlp5p7 = fert_app_time_tonne_p7p5_rzl.unstack([1,0])

    ##create params for v_phase_increment
    increment_fert_app_time_tonne_rzlp5p7 = rps.f_v_phase_increment_adj(fert_app_time_tonne_rzlp5p7,p7_pos=-1,z_pos=-4,p5_pos=-2)

    return fert_app_time_tonne_rzlp5p7, increment_fert_app_time_tonne_rzlp5p7


#print(fert_app_time_t())
    


###########################
#chem application time   #  this is similar to app cost done in mach sheet
###########################

def f_chem_lab_allocation():
    '''Allocation of chemical applications into each labour period'''
    chem_info = pinp.crop['chem_info']
    chem_date_n = chem_info['app_date'].values
    chem_length_n = chem_info['app_len'].values
    p_dates_p5z = per.f_p_dates_df()
    shape_p5zn = p_dates_p5z.shape+chem_date_n.shape
    alloc_p5zn = fun.f_range_allocation_np(p_dates_p5z.values[...,na], chem_date_n, chem_length_n, shape=shape_p5zn)[:-1,...]
    
    ##allocate to p7
    alloc_p7p5z = f_p5_p7_allocation()
    alloc_p7p5zn = alloc_p5zn * alloc_p7p5z[...,na]
    
    ##put in df
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    keys_p7 = per.f_season_periods(keys=True)
    index_p7p5zn = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_z, chem_info.index])
    alloc_p7p5zn = pd.Series(alloc_p7p5zn.ravel(), index=index_p7p5zn)

    return alloc_p7p5zn


def f_chem_app_time_ha():
    '''

    Calculate labour required for spraying for each rotation and allocate it into the labour periods.

    The labour required for spraying is calculated from the time to spray 1ha (calculated in Mach.py)
    and the number of chemical applications for each rotation phase (calculated in Phase.py).

    '''

    ##note arable area accounted for in crop.py

    ##passes
    total_passes_rzln = phs.f_chem_application().stack()
    ##time for 1 pass for each chem
    time = mac.spray_time_ha()
    ##adjust fert labour across each labour period
    alloc_p7p5zn = f_chem_lab_allocation()
    time_p7p5z_n = alloc_p7p5zn.mul(time, level=-1).unstack() #time for 1 pass for each chem.
    ##adjust for passes
    time_p7p5z_rln = time_p7p5z_n.reindex(total_passes_rzln.unstack(1).index, axis=1, level=2)
    time_p7p5_rzln = time_p7p5z_rln.unstack().reorder_levels([0,3,1,2], axis=1).sort_index(axis=1)
    chem_app_time_p7p5_rzln = time_p7p5_rzln.mul(total_passes_rzln, axis=1)
    chem_app_time_p7p5_rzl = chem_app_time_p7p5_rzln.groupby(axis=1, level=(0,1,2)).sum() #sum chem type
    chem_app_time_rzlp5p7 = chem_app_time_p7p5_rzl.unstack([1,0])

    ##create params for v_phase_increment
    increment_chem_app_time_rzlp5p7 = rps.f_v_phase_increment_adj(chem_app_time_rzlp5p7,p7_pos=-1,z_pos=-4,p5_pos=-2)

    return chem_app_time_rzlp5p7, increment_chem_app_time_rzlp5p7




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
    ##p5 allocation
    fixed_crop_monitor = pinp.labour['fixed_crop_monitoring']
    variable_crop_monitor = pinp.labour['variable_crop_monitoring']
    labour_periods_pz = per.f_p_dates_df().values
    date_start_d = fixed_crop_monitor.columns.values
    date_end_d = np.roll(date_start_d, -1)
    date_end_d[-1] = date_end_d[-1] + 364 #increment the first date by 1yr so it becomes the end date for the last period
    length_d = date_end_d - date_start_d
    shape_pzd = labour_periods_pz.shape + date_start_d.shape
    monitoring_allocation_p5zd = fun.f_range_allocation_np(labour_periods_pz[...,na]
                                                    , date_start_d, length_d, shape=shape_pzd)

    ## drop last row, because it has na because it only contains the end date, therefore not a period
    monitoring_allocation_p5zd = monitoring_allocation_p5zd[:-1]

    ##adjust for p7 (season period) axis
    alloc_p7p5z = f_p5_p7_allocation()

    ##variable monitoring
    ###adjust to monitoring time per ha
    variable_crop_monitor_kd = variable_crop_monitor.values #convert to numpy
    variable_crop_monitor_kd = variable_crop_monitor_kd/pinp.general['pad_size'] * length_d.astype(float)/7
    ###convert date range to labour periods
    variable_crop_monitor_kp7p5z = np.sum(variable_crop_monitor_kd[:,na,na,na,:] * monitoring_allocation_p5zd * alloc_p7p5z[...,na], axis=-1) #sum the d axis (monitoring date axis)
    ###convert to df and expand landuse to rotation
    variable_crop_monitor_k_p7p5z = variable_crop_monitor_kp7p5z.reshape(variable_crop_monitor_kp7p5z.shape[0], -1)
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    keys_p7 = per.f_season_periods(keys=True)
    keys_k = variable_crop_monitor.index
    cols_p7p5z = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_z])
    variable_crop_monitor = pd.DataFrame(variable_crop_monitor_k_p7p5z, index=keys_k, columns=cols_p7p5z)
    phases_df = sinp.f_phases()
    phases_df.columns = pd.MultiIndex.from_product([phases_df.columns,[''],['']])
    variable_crop_monitor = pd.merge(phases_df, variable_crop_monitor, how='left', left_on=sinp.end_col(), right_index = True) #merge with all the phases
    variable_crop_monitor_r_p7p5z = variable_crop_monitor.drop(list(range(sinp.general['phase_len'])), axis=1)
    variable_crop_monitor_r_p7p5z.columns = cols_p7p5z #need to update cols because merging added levels
    variable_crop_monitor_r_p7p5z.index.name = None #remove index name (it got added because of the merge and causes issues later)
    variable_crop_monitor_p7p5zr = variable_crop_monitor_r_p7p5z.unstack().fillna(0) #nan exist because pasture was not included in the merge above
    ###create params for v_phase_increment
    increment_variable_crop_monitor_p7p5zr = rps.f_v_phase_increment_adj(variable_crop_monitor_p7p5zr,p7_pos=-4,z_pos=-2,p5_pos=-3)

    ##fixed monitoring
    ###adjust from hrs/week to hrs/period
    fixed_crop_monitor_d = fixed_crop_monitor.values
    fixed_crop_monitor_d = fixed_crop_monitor_d * length_d.astype(float)/7
    ###convert date range to labour periods
    fixed_crop_monitor_p5z = np.sum(fixed_crop_monitor_d * monitoring_allocation_p5zd, axis=-1) #sum the d axis (monitoring date axis)
    ###mask z axis
    maskz8_p5z = zfun.f_season_transfer_mask(labour_periods_pz[:-1,:],z_pos=-1,mask=True) #slice off the end date of the last period
    fixed_crop_monitor_p5z = fixed_crop_monitor_p5z * maskz8_p5z
    ###convert to df
    fixed_crop_monitor = pd.DataFrame(fixed_crop_monitor_p5z, index=keys_p5, columns=keys_z)
    return variable_crop_monitor_p7p5zr, increment_variable_crop_monitor_p7p5zr, fixed_crop_monitor.stack()

##collates all the params
def f1_labcrop_params(params,r_vals):
    prep_labour = f_prep_labour().stack()
    fert_app_time_t, increment_fert_app_time_t = f_fert_app_time_t()
    fert_app_time_ha, increment_fert_app_time_ha = f_fert_app_time_ha()
    chem_app_time_ha, increment_chem_app_time_ha = f_chem_app_time_ha()
    variable_crop_monitor, increment_variable_crop_monitor, fixed_crop_monitor = f_crop_monitoring()

    ##add params which are inputs
    params['harvest_helper'] = pinp.labour['harvest_helper'].squeeze().to_dict()
    params['daily_seed_hours'] = pinp.mach['daily_seed_hours']
    params['seeding_helper'] = pinp.labour['seeding_helper']
    params['prep_labour'] = prep_labour.to_dict()
    params['fert_app_time_t'] = fert_app_time_t.to_dict()
    params['increment_fert_app_time_t'] = increment_fert_app_time_t.to_dict()
    params['fert_app_time_ha'] = fert_app_time_ha.to_dict()
    params['increment_fert_app_time_ha'] = increment_fert_app_time_ha.to_dict()
    params['chem_app_time_ha'] = chem_app_time_ha.to_dict()
    params['increment_chem_app_time_ha'] = increment_chem_app_time_ha.to_dict()
    params['variable_crop_monitor'] = variable_crop_monitor.to_dict()
    params['increment_variable_crop_monitor'] = increment_variable_crop_monitor.to_dict()
    params['fixed_crop_monitor'] = fixed_crop_monitor.to_dict()
    # params['fert_req'] = fert_total.to_dict()













