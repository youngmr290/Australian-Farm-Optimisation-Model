# -*- coding: utf-8 -*-
"""

author: young

Phase labour represents the labour associated with each rotation phase. This
includes both the crop and pasture phases. Phase labour includes the labour required
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
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import Periods as per
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Phase as phs
from . import Mach as mac
from . import RotationPhases as rps

na = np.newaxis

##general
def f1_p5_alloc(item_start=0, item_length=1, z_pos=-1, is_phase_param=False):
    '''
    Allocation of item into labour periods (p5).

    Cant be allocated across the season start (length is automatically adjusted in f_range_allocation to stop allocation
    going from the end of the periods to the start of the periods. This is because gets complicated if it crosses junction)
    e.g. post harvest jobs must be completed before the next season start.
    An item can be allocated in to multiple season periods as long as the dv exists in the same multiple season periods
    e.g. v_phase exists in all children therefore labour requirement can cross nodes.

    Note: For params linked to v_phase activity the timing of an item is adjusted (in fun.f_range_allocation)
    so that no cost/labour/depn is incurred
    between season start and break of season. This stops the model getting double costs in medium/late breaks where
    phases are carried over past the start of the season to provide dry pas and stubble area (because it is also
    accounted for by v_phase_increment).

    - Arrays must be numpy and broadcastable.
    - p5 axis must be in pos 0
    - item start must contain all axes (including p5)

    :param item_start: item dates which are allocated into season periods. MUST contain all axis of the final array (singleton is fine)
    :param item_length: length (days) of item being allocated
    :param z_pos:
    :return:
    '''
    labour_period = per.f_p_dates_df()
    lp_p5z = labour_period.values
    len_p5 = lp_p5z.shape[0] - 1

    ##align axes
    p5_pos = -item_start.ndim
    lp_p5etc = fun.f_expand(lp_p5z, left_pos=z_pos, right_pos2=z_pos, left_pos2=p5_pos)
    shape = (len_p5,) + tuple(np.maximum.reduce([lp_p5etc.shape[1:], item_start.shape[1:]]))  # create shape which has the max size, this is used for o array

    break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    season_start = per.f_season_periods()[0, 0]  # slice season node to get season start
    alloc_p5etc = fun.f_range_allocation_np(lp_p5etc, item_start, item_length, shape=shape, is_phase_param=is_phase_param,
                                           break_z=break_z, season_start=season_start, z_pos=z_pos)

    return alloc_p5etc


def f_p5_p7_allocation(param=False):
    '''Allocation of labour periods (p5) to each p7 period.'''
    labour_period_p5z = per.f_p_dates_df()
    labour_period_start_p5z = labour_period_p5z.values[:-1]
    labour_period_end_p5z = labour_period_p5z.values[1:]
    length_p5z = labour_period_start_p5z - labour_period_end_p5z
    ##allocate p5 to p7
    alloc_p7p5z = zfun.f1_z_period_alloc(labour_period_start_p5z[na,:,:],length_p5z[na,:,:],z_pos=-1)

    if param==True:
        ##make df
        keys_z = zfun.f_keys_z()
        keys_p7 = per.f_season_periods(keys=True)
        keys_p5 = labour_period_p5z.index[:-1]
        new_index_p7p5z = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_z])
        alloc_p7p5z = pd.Series(alloc_p7p5z.ravel(), index=new_index_p7p5z)
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
    alloc_p5zp8 = f1_p5_alloc(harvest_prep_dates_p8[na,na,:], harvest_prep_length_p8, z_pos=-2, is_phase_param=False)
    harvest_prep_p5z = np.sum(alloc_p5zp8 * harvest_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##fert_prep
    fert_prep_dates_p8 = pinp.labour['fert_prep'].index.values
    fert_prep_length_p8 = pinp.labour['fert_prep']['days'].values
    fert_prep_labour_p8 = pinp.labour['fert_prep']['hours'].values
    alloc_p5zp8 = f1_p5_alloc(fert_prep_dates_p8[na,na,:], fert_prep_length_p8, z_pos=-2, is_phase_param=False)
    fert_prep_p5z = np.sum(alloc_p5zp8 * fert_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##spray_prep
    spray_prep_dates_p8 = pinp.labour['spray_prep'].index.values
    spray_prep_length_p8 = pinp.labour['spray_prep']['days'].values
    spray_prep_labour_p8 = pinp.labour['spray_prep']['hours'].values
    alloc_p5zp8 = f1_p5_alloc(spray_prep_dates_p8[na,na,:], spray_prep_length_p8, z_pos=-2, is_phase_param=False)
    spray_prep_p5z = np.sum(alloc_p5zp8 * spray_prep_labour_p8, axis=-1) #get rid of p8 axis

    ##seed_prep
    seed_prep_dates_p8 = pinp.labour['seed_prep'].index.values
    seed_prep_length_p8 = pinp.labour['seed_prep']['days'].values
    seed_prep_labour_p8 = pinp.labour['seed_prep']['hours'].values
    alloc_p5zp8 = f1_p5_alloc(seed_prep_dates_p8[na,na,:], seed_prep_length_p8, z_pos=-2, is_phase_param=False)
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

def f_fert_lab_allocation():
    '''Allocation of fertiliser applications into each labour period and season period'''

    ##calc p5 allocation
    fert_info = pinp.crop['fert_info']
    fert_date_n = fert_info['app_date'].values
    fert_length_n = fert_info['app_len'].values
    alloc_p5zn = f1_p5_alloc(fert_date_n[na,na,:], fert_length_n, z_pos=-2, is_phase_param=True)

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


def f_fert_app_time():
    '''
    Fertilising labour (hr/ha).

    The labour required for fertilising is calculated in two parts. Part 1 is the time required per hectare
    for each rotation phase which represents the time taken spreading fertiliser in the paddock (calculated in Mach.py).
    Part 2 is the time required per tonne
    which represents the time taken driving to and from the paddock and filling up (calculated in Mach.py).
    This is adjusted for the number of fertiliser applications and allocated into a labour period/s.

    '''

    ##total time spent fertilising per ha
    fert_time_rzln = phs.f1_fertilising_time()

    ##p5 and p7 allocation
    alloc_p7p5z_n = f_fert_lab_allocation().unstack()

    ##combine with total phase fert
    alloc_p7p5z_rln = alloc_p7p5z_n.reindex(fert_time_rzln.unstack(1).index, axis=1, level=2)
    alloc_p7p5_rzln = alloc_p7p5z_rln.unstack().reorder_levels([0,3,1,2], axis=1).sort_index(axis=1)
    fert_app_time_p7p5_rzln = alloc_p7p5_rzln.mul(fert_time_rzln.sort_index(), axis=1)
    fert_app_time_p7p5_rzl = fert_app_time_p7p5_rzln.groupby(axis=1, level=(0,1,2)).sum() #sum fert type
    fert_app_time_rzlp5p7 = fert_app_time_p7p5_rzl.unstack([1,0]).sort_index()

    ##create params for v_phase_change_increase
    increment_fert_app_time_rzlp5p7 = rps.f_v_phase_increment_adj(fert_app_time_rzlp5p7,p7_pos=-1,z_pos=-4,p5_pos=-2)

    return fert_app_time_rzlp5p7, increment_fert_app_time_rzlp5p7


###########################
#chem application time   #  this is similar to app cost done in mach sheet
###########################

def f_chem_lab_allocation():
    '''Allocation of chemical applications into each labour period'''
    ##calc p5 allocation
    chem_info = pinp.crop['chem_info']
    chem_date_n = chem_info['app_date'].values
    chem_length_n = chem_info['app_len'].values
    alloc_p5zn = f1_p5_alloc(chem_date_n[na, na, :], chem_length_n, z_pos=-2, is_phase_param=True)
    
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

    ##time spent spraying each rotation phase
    spray_time_rzln = phs.f1_spraying_time()

    ##p5 and p7 allocation
    alloc_p7p5z_n = f_chem_lab_allocation().unstack()

    ##adjust for passes
    alloc_p7p5z_rln = alloc_p7p5z_n.reindex(spray_time_rzln.unstack(1).index, axis=1, level=2)
    alloc_p7p5_rzln = alloc_p7p5z_rln.unstack().reorder_levels([0,3,1,2], axis=1).sort_index(axis=1)
    chem_app_time_p7p5_rzln = alloc_p7p5_rzln.mul(spray_time_rzln.sort_index(), axis=1)
    chem_app_time_p7p5_rzl = chem_app_time_p7p5_rzln.groupby(axis=1, level=(0,1,2)).sum() #sum chem type
    chem_app_time_rzlp5p7 = chem_app_time_p7p5_rzl.unstack([1,0]).sort_index()

    ##create params for v_phase_change_increase
    increment_chem_app_time_rzlp5p7 = rps.f_v_phase_increment_adj(chem_app_time_rzlp5p7,p7_pos=-1,z_pos=-4,p5_pos=-2)

    return chem_app_time_rzlp5p7, increment_chem_app_time_rzlp5p7




###########################
#crop monitoring time     #
###########################

def f_crop_monitoring():
    '''
    Labour required for monitoring crop paddocks.

    For crop paddocks, monitoring time is broken into two sections.
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
    date_start_d = fixed_crop_monitor.columns.values.astype(float)
    date_end_d = np.roll(date_start_d, -1)
    date_end_d[-1] = date_end_d[-1] + 364 #increment the first date by 1yr so it becomes the end date for the last period
    length_d = date_end_d - date_start_d
    monitoring_allocation_p5zd = f1_p5_alloc(date_start_d[na, na, :], length_d, z_pos=-2, is_phase_param=True)

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
    phases_df = pinp.phases_r.copy()
    phases_df.columns = pd.MultiIndex.from_product([phases_df.columns,[''],['']])
    variable_crop_monitor = pd.merge(phases_df, variable_crop_monitor, how='left', left_on=sinp.end_col(), right_index = True) #merge with all the phases
    variable_crop_monitor_r_p7p5z = variable_crop_monitor.drop(list(range(sinp.general['phase_len'])), axis=1)
    variable_crop_monitor_r_p7p5z.columns = cols_p7p5z #need to update cols because merging added levels
    variable_crop_monitor_r_p7p5z.index.name = None #remove index name (it got added because of the merge and causes issues later)
    variable_crop_monitor_p7p5zr = variable_crop_monitor_r_p7p5z.unstack().fillna(0) #nan exist because pasture was not included in the merge above
    ###create params for v_phase_change_increase
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
    fert_app_time, increment_fert_app_time = f_fert_app_time()
    chem_app_time_ha, increment_chem_app_time_ha = f_chem_app_time_ha()
    fert_chem_app_time = fert_app_time + chem_app_time_ha
    increment_fert_chem_app_time = increment_fert_app_time + increment_chem_app_time_ha
    variable_crop_monitor, increment_variable_crop_monitor, fixed_crop_monitor = f_crop_monitoring()

    ##add params which are inputs
    params['harvest_helper'] = pinp.labour['harvest_helper'].squeeze().to_dict()
    params['daily_seed_hours'] = pinp.mach['daily_seed_hours']
    params['seeding_helper'] = pinp.labour['seeding_helper']
    params['prep_labour'] = prep_labour.to_dict()
    params['fert_chem_app_time'] = fert_chem_app_time.to_dict()
    params['increment_fert_chem_app_time'] = increment_fert_chem_app_time.to_dict()
    params['variable_crop_monitor'] = variable_crop_monitor.to_dict()
    params['increment_variable_crop_monitor'] = increment_variable_crop_monitor.to_dict()
    params['fixed_crop_monitor'] = fixed_crop_monitor.to_dict()
    params['a_p5_p7'] = f_p5_p7_allocation(param=True).to_dict()













