"""

This module reads in rotation information that has been generated from RotGeneration.py and manipulates
it to produce the rotation parameters.

author: young
"""

#python modules
import pandas as pd
import numpy as np
na = np.newaxis
import timeit

#AFO modules - a bunch of other precalc modules import this module
import StructuralInputs as sinp
import PropertyInputs as pinp
import Functions as fun
import Periods as per
import SeasonalFunctions as zfun


def f1_rot_period_alloc(item_start=0, item_length=np.timedelta64(1, 'D'), z_pos=0):
    '''
    Allocation of item into rotation periods (m).

    - Arrays must be numpy and broadcastable.
    - M axis must be in pos 0
    - item start must contain all axes (including z and m)

    :param item_start: datetime64 item dates which are allocated into rotation periods. MUST contain all axis of the final array (singleton is fine)
    :param item_length: datetime64
    :param z_pos:
    :return:
    '''

    date_phase_periods_mz = per.f_phase_periods()
    len_m = date_phase_periods_mz.shape[0] - 1  # minus one because end date is not a period

    ##align axes
    m_pos = -item_start.ndim
    date_node_metc = fun.f_expand(date_phase_periods_mz, left_pos=z_pos, right_pos2=z_pos, left_pos2=m_pos)
    shape = (len_m,) + tuple(np.maximum.reduce([date_node_metc.shape[1:], item_start.shape[1:]]))  # create shape which has the max size, this is used for o array
    alloc_metc = fun.range_allocation_np(date_node_metc, item_start, item_length, opposite=True, shape=shape)

    ##mask z8
    mask_season_z8 = zfun.f_season_transfer_mask(date_node_metc[:-1,...],z_pos,mask=True) #slice off end date m1

    return alloc_metc * mask_season_z8


def f_v_phase_increment_adj(param, m_pos, r_pos, numpy=False):
    '''
    Adjust v_phase param for v_phase_increment.

    v_phase_increment must incur the requirement to date for labour and cash for the phase.
    This is making the assumption that any jobs carried out and any expenditure
    (fertiliser or chemical applied) will be applied even though the phase is selected later in the year.
    This stops the optimisation selecting the phase in the last node and receiving the income without
    incurring any costs. Note: Yield and stubble do not require increment params because it is not possible to harvest a
    rotation before the rotation is selected.

    Note dry sown rotations are excluded because they are selected in the final m period and passed to the starting m.

    :param param: parameter with m axis and r axis that is being adjusted.
    :param m_pos: axis position of m
    :param r_pos: for pandas this is r axis level, for numpy this is r axis pos
    :param numpy: Boolean, stating if param is numpy.
    '''

    param_increment = np.roll(np.cumsum(param.values, axis=m_pos),1, axis=m_pos) #include .values incase df is passed.
    slc = [slice(None)] * len(param_increment.shape)
    slc[m_pos] = slice(0,1)
    param_increment[tuple(slc)] = 0

    ##remove cumulative for dry sown phases
    phases_df = sinp.f_phases()
    landuse_r = phases_df.iloc[:,-1].values
    dry_sown_landuses = sinp.landuse['dry_sown']
    phase_is_not_drysown_r = np.logical_not(np.any(landuse_r[:,na]==list(dry_sown_landuses), axis=-1))

    if numpy:
        phase_is_not_drysown_r = fun.f_expand(phase_is_not_drysown_r, r_pos)
        param_increment = param_increment * phase_is_not_drysown_r
    else:
        phase_is_not_drysown_r = pd.Series(phase_is_not_drysown_r, index=phases_df.index)
        index = param.index
        cols = param.columns
        param_increment = pd.DataFrame(param_increment, index=index, columns=cols)
        param_increment = param_increment.mul(phase_is_not_drysown_r, axis=1-m_pos, level=r_pos)

    return param_increment


def f_season_params(params):
    '''
    Create params for phase period transfer.
    '''
    ##inputs
    keys_m = per.f_phase_periods(keys=True)
    keys_z = zfun.f_keys_z()
    phases_df = sinp.f_phases()
    keys_r = phases_df.index
    index_z = np.arange(len(keys_z))
    initiating_parent_z = zfun.f_initiating_parent_z()
    landuse_r = phases_df.iloc[:,-1].values
    dry_sown_landuses = sinp.landuse['dry_sown']
    phase_is_drysown_r = np.any(landuse_r[:,na]==list(dry_sown_landuses), axis=-1)

    ##z8z9 transfer
    start_phase_periods_mz = per.f_phase_periods()[:-1,:] #remove end date of last period
    mask_phase_provz8z9_mz8z9 = zfun.f_season_transfer_mask(start_phase_periods_mz, z_pos=-1)

    ##mask phases which transfer in each m
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1:
        ###if steady state then there is no m transfering
        mask_phases_rm = np.zeros((len(phases_df),len(keys_m)))
    else:
        ###if dsp no transfer at the end of yr to the start (different for dry sown landuses since m-1 is essentially the start for them)
        mask_phases_rm = np.ones((len(phases_df),len(keys_m)))
        mask_phases_rm[:,-1] = phase_is_drysown_r #only dry sown landuse pass from m[-1] to m[0] because m[-1] is the period when dry sown phases are selected.
        mask_phases_rm[:,-2] = np.logical_not(phase_is_drysown_r) #v_phase dry does not provide into m[-1]. if the model wants dry sown phases it can select via v_phase_increment.

    ##dry seeding link between season - dry seeding must happen in all seasons that brk after the season with dry seeding.
    ##the same amount of dry seeding must occur for all seasons with the same break therefore the end season passes back to the start for a given brk.
    ##the param below is used on the require side of phase_increment. It says if you want to dry sow in z0 you must
    ## dry sow in z1 and if you want to sow in z1 you must dry sow in z2 etc.
    ## The second part of the mask creation makes it so that the same dry sowing occurs for each season that has the same brk.
    mask_drynext_z8z9 = index_z[:,na] == index_z-1 #every season must have at least the same amount of dry sowing as the previous season.
    mask_drystart_z8z9 = (initiating_parent_z != np.roll(initiating_parent_z,-1))[:,na]*(index_z==initiating_parent_z[:,na])#each season with the same brk must have the same amount of dry sowiing.
    # mask_dryz8z9_z8z9 = np.logical_or(mask_drynext_z8z9, mask_drystart_z8z9)
    ###only for dry sown phases
    mask_drynext_z8z9 = mask_drynext_z8z9 * phase_is_drysown_r[:,na,na]
    mask_drystart_z8z9 = mask_drystart_z8z9 * phase_is_drysown_r[:,na,na]
    ##mask which seasons are active in each phase period
    mask_phase_z8_mz8 = zfun.f_season_transfer_mask(start_phase_periods_mz, z_pos=-1, mask=True)

    ##build params
    arrays = [keys_m, keys_z, keys_z]
    index_mz8z9 = fun.cartesian_product_simple_transpose(arrays)
    tup_mz8z9 = tuple(map(tuple,index_mz8z9))

    arrays = [keys_m, keys_z]
    index_mz = fun.cartesian_product_simple_transpose(arrays)
    tup_mz = tuple(map(tuple,index_mz))

    arrays = [keys_r, keys_m]
    index_rm = fun.cartesian_product_simple_transpose(arrays)
    tup_rm = tuple(map(tuple,index_rm))

    arrays = [keys_r, keys_z, keys_z]
    index_rz8z9 = fun.cartesian_product_simple_transpose(arrays)
    tup_rz8z9 = tuple(map(tuple,index_rz8z9))

    # params['p_childz_req_cashflow'] =dict(zip(tup_z8z9, mask_cashflow_reqz8z9_z8z9.ravel()*1))
    params['p_parentchildz_transfer_phase'] =dict(zip(tup_mz8z9, mask_phase_provz8z9_mz8z9.ravel()*1))
    params['p_mask_phases'] =dict(zip(tup_rm, mask_phases_rm.ravel()*1))
    params['p_dryz_link'] =dict(zip(tup_rz8z9, mask_drynext_z8z9.ravel()*1))
    params['p_dryz_link2'] =dict(zip(tup_rz8z9, mask_drystart_z8z9.ravel()*1))
    params['p_mask_childreqz'] =dict(zip(tup_mz, mask_phase_z8_mz8.ravel()*1))


def f_landuses_phases(params,report):
    '''
    * Read in the rotation list generated by RotGeneration.py
    * Create rotation area parameter for pyomo (simply each rotation phase uses 1ha of area).
    * Store rotation list and pasture phases list to report dictionary

    '''
    phases=sinp.f_phases()
    phases_rk = phases.set_index(5, append=True) #add landuse as index level
    params['phases_rk'] = dict.fromkeys(phases_rk.index,1)
    report['phases']=phases
    report['all_pastures']=sinp.landuse['All_pas'] #all_pas2 includes the cont pasture landuses


def f_rot_lmu_params(params):
    '''
    Create parameters for lmu area.

    '''
    ##area
    lmu_mask = pinp.general['i_lmu_area'] > 0
    params['lmu_area'] = dict(zip(pinp.general['i_lmu_idx'][lmu_mask], pinp.general['i_lmu_area'][lmu_mask]))


def f_rot_hist_params(params):
    '''
    Create parameters for landuse history provided and required by each rotation phase.

    '''
    rot_req = pd.read_excel('Rotation.xlsx', sheet_name='rotation_req', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_prov = pd.read_excel('Rotation.xlsx', sheet_name='rotation_prov', header= None, engine='openpyxl')#, index_col = [0,1]) #couldn't get it to read in with multiindex for some reason
    rot_req = rot_req.set_index([0,1])
    rot_prov = rot_prov.set_index([0,1])
    params['hist_prov'] = rot_prov.squeeze().to_dict()
    params['hist_req'] = rot_req.squeeze().to_dict()



