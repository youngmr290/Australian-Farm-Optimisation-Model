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



def f1_rot_period_alloc(item_start=0, item_length=np.timedelta64(1, 'D'), z_pos=0, keys=False, periods=False):
    '''
    Allocation of item into rotation periods (m).

    - Arrays must be numpy and broadcastable.
    - M axis must be in pos 0
    - item start must contain all axes (including z and m)

    :param item_start: datetime64 item dates which are allocated into rotation periods. MUST contain all axis of the final array (singleton is fine)
    :param item_length: datetime64
    :param z_pos:
    :param keys: Boolean if True this returns the m keys
    :param periods: Boolean if True this returns the m period dates
    :return:
    '''
    date_node_zm = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'],numpy=True,axis=0).astype('datetime64')  # treat z axis
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1:
        date_node_zm = date_node_zm[:,0] #if steady state then m axis is singleton (start and finish at the break of season).
    ###add end date of last node period - required for the allocation function
    end_zm = date_node_zm[:,0:1] + np.timedelta64(365, 'D')  # increment the first date by 1yr so it becomes the end date for the last period
    ###add dry seeding period
    dry_seed_start_m = np.array([pinp.crop['dry_seed_start']], dtype='datetime64')
    dry_seed_start_zm = np.broadcast_to(dry_seed_start_m[na,:], end_zm.shape) #expand z axis
    date_phase_node_mz = np.concatenate([date_node_zm, dry_seed_start_zm, end_zm], axis=1).T #put m in pos 0 because that how the allocation function requires
    len_m = date_phase_node_mz.shape[0] - 1 #minus one because end date is not a period

    ##return keys if wanted
    if periods:
        return date_phase_node_mz

    ##return keys if wanted
    if keys:
        keys_m = np.array(['m%s' % i for i in range(len_m)])
        return keys_m

    ##align axes
    m_pos = -item_start.ndim
    date_node_metc = fun.f_expand(date_phase_node_mz, left_pos=z_pos, right_pos2=z_pos, left_pos2=m_pos)
    shape = (len_m,) + tuple(np.maximum.reduce([date_node_metc.shape[1:], item_start.shape[1:]]))  # create shape which has the max size, this is used for o array
    alloc_metc = fun.range_allocation_np(date_node_metc, item_start, item_length, opposite=True, shape=shape)
    return alloc_metc

def f_v_phase_increment_adj(param, m_pos, numpy=False):
    '''
    Adjust v_phase param for v_phase_increment.

    v_phase_increment must incur the requirement to date for labour and cash for the phase.
    This is making the assumption that any jobs carried out and any expenditure
    (fertiliser or chemical applied) will be applied even though the phase is selected later in the year.
    This stops the optimisation selecting the phase in the last node and receiving the income without
    incurring any costs. Note: Yield and stubble do not require increment params because it is not possible to harvest a
    rotation before the rotation is selected.

    '''

    param_increment = np.roll(np.cumsum(param.values, axis=m_pos),1, axis=m_pos) #include .values incase df is passed.
    slc = [slice(None)] * len(param_increment.shape)
    slc[m_pos] = slice(0,1)
    param_increment[tuple(slc)] = 0

    if not numpy:
        index = param.index
        cols = param.columns
        param_increment = pd.DataFrame(param_increment, index=index, columns=cols)

    return param_increment


def f_m_z8z9_transfer(params=None, mask=False):
    '''
    Mask transfer between phase periods within a given season.

    Seasons are masked out until the point in the year when they are identified. At the point of identification
    the parent season provides the transfer parameters to the child season. This transfering method ensures the
    model has the same management across seasons until they are identified. For example, if there are two seasons, a
    good and a bad, that are identified in spring. Both seasons must have the same management through the beginning of
    the year until spring (because the farmer doesnt know if they are having the good or bad year until spring).

    The mask and params built in this function are generic to all constraints that transfer between phase periods.
    Thus these params get used in multiple modules.'''

    ##inputs
    date_initiate_z = zfun.f_seasonal_inp(pinp.general['i_date_initiate_z'], numpy=True, axis=0).astype('datetime64')
    bool_steady_state = pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1
    if bool_steady_state:
        len_z = 1
    else:
        len_z = np.count_nonzero(pinp.general['i_mask_z'])
    index_z = np.arange(len_z)
    fp_dates_p6z = per.f_feed_periods()[:-1,:] #slice off the end date slice
    date_node_zm = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'],numpy=True,axis=0).astype(
        'datetime64')  # treat z axis

    ##dams child parent transfer
    mask_fp_provz8z9_p6z8z9, mask_fp_z8var_p6z = \
    zfun.f_season_transfer_mask(fp_dates_p6z, date_node_zm, date_initiate_z, index_z, bool_steady_state, z_pos=-1)

    if mask:
        return mask_fp_z8var_p6z

    ##build params
    keys_p6 = np.asarray(pinp.period['i_fp_idx'])
    keys_z = zfun.f_keys_z()

    arrays = [keys_p6, keys_z, keys_z]
    index_p6z8z9 = fun.cartesian_product_simple_transpose(arrays)
    tup_p6z8z9 = tuple(map(tuple,index_p6z8z9))

    params['p_parentchildz_transfer_fp'] =dict(zip(tup_p6z8z9, mask_fp_provz8z9_p6z8z9.ravel()*1))


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



