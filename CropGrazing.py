'''
Author: Young

Crop grazing is an option that allows sheep to graze green crop, typically from June until august
however, this range can be altered in the inputs. Green crops
have a higher energy content than green pasture and grow more vertical allowing for easier grazing,
meaning a lower crop FOO is required to meet the livestock needs. However, a yield and stubble penalty is
associated with this activity. Trials have recorded varying yield penalties from -15% to +15% but the
consensus is that the yield penalty is minimal if the crop is grazed early and lightly. The level of the yield
penalty is an input which can be easily changed.
'''

##import python modules
import pandas as pd
import numpy as np

##import AFO modules
import PropertyInputs as pinp
import StructuralInputs as sinp
import UniversalInputs as uinp
import Periods as per
import FeedsupplyFunctions as fsfun
import Functions as fun


na = np.newaxis

#todo this module will need some postprocessing season clustering

def f_graze_crop_area():
    '''
    The area of each crop that can be grazed for 1ha each rotation phase.
    Each rotation phase only provide crop grazing for one crop on the arable areas.

    '''
    ##read phases
    phases_rh = sinp.f_phases().values

    ##lmu mask
    lmu_mask = pinp.general['i_lmu_area'] > 0

    ##propn of crop grazing possible for each landuse.
    landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    landuse_grazing_kl = pinp.cropgraze['i_cropgrazing_inc_landuse'][:, lmu_mask]

    ##graze = arable area
    arable_l = pinp.crop['arable'].squeeze().values[lmu_mask]

    ##area of crop grazing that 1ha of each landuse provides
    graze_area_kl = landuse_grazing_kl * arable_l

    ##merge to rot phases
    current_landuse_r = phases_rh[:,-1]
    a_r_k_rk = current_landuse_r[:,na] == landuse_idx_k
    cropgraze_area_rkl = graze_area_kl * a_r_k_rk[...,na]
    return cropgraze_area_rkl

def f_cropgraze_foo(foo=False):
    '''
    Calculates the FOO available for grazing on crop paddocks.

    Crop growth rate is an input for each feed period, LMU and land use.
    There are two main limitations of the representation:

        #. Impacts of rotation are not included in the estimation of crop growth.
        #. Impacts of grazing intensity are not included in the estimation of crop growth. This is likely not a big
           issue because crop tend to be grazed lightly and thus the rate of crop growth is likely to remain similar.
    '''
    ##read inputs
    lmu_mask = pinp.general['i_lmu_area'] > 0
    growth_kp6z = pinp.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1)
    wastage_k = pinp.cropgraze['i_cropgraze_wastage']
    growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl'][:,lmu_mask]
    consumption_factor_p6 = pinp.cropgraze['i_cropgraze_consumption_factor_p6']
    feed_period_lengths_p6z = per.f_feed_periods(option=1)

    ##adjust crop growth for lmu
    growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]

    ##calc total dry matter in each feed period
    total_dm_kp6zl = growth_kp6zl * feed_period_lengths_p6z[:,na].astype('float')

    if not foo:
        ##calc dry matter available for consumption provided by 1ha of crop
        crop_foo_provided_kp6zl = total_dm_kp6zl * consumption_factor_p6[:,na,na]

        ##calc foo required for animals to consume 1t - accounts for wastage
        crop_foo_required_k = 1000 * (1 + wastage_k)

        return crop_foo_provided_kp6zl, crop_foo_required_k

    else:
        ##crop foo mid way through feed peirod after consumption - used to calc vol in the next function.
        ##foo = cumulative sum of foo in previous periods minus foo consumed. Minus half the foo in the current period to get the foo in the middle of the period.
        crop_foo_kp6zl = np.cumsum(total_dm_kp6zl * (1-consumption_factor_p6[:,na,na]), axis=1) - total_dm_kp6zl/2 * (1-consumption_factor_p6[:,na,na])
        return crop_foo_kp6zl

def crop_md_vol(nv):
    '''
    Energy provided and volume required from 1t of crop grazing.
    '''

    ##inputs
    crop_dmd_kp6z = pinp.f_seasonal_inp(pinp.cropgraze['i_crop_dmd_kp6z'],numpy=True,axis=-1)
    crop_foo_kp6zl = f_cropgraze_foo(foo=True)
    hr = pinp.cropgraze['i_hr_crop']
    me_threshold_fp6z = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)
    crop_me_eff_gainlose = pinp.cropgraze['i_crop_me_eff_gainlose']

    ##nv stuff
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.

    ## md per tonne
    crop_md_kp6z = fsfun.dmd_to_md(crop_dmd_kp6z)

    ## vol
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        crop_ri_qual_kp6z = fsfun.f_rq_cs(crop_dmd_kp6z, legume=0)

    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    hf= fsfun.f_hf(hr) #height factor
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        crop_ri_quan_kp6zl = fsfun.f_ra_cs(crop_foo_kp6zl, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        crop_ri_quan_kp6zl = fsfun.f_ra_mu(crop_foo_kp6zl, hf)

    crop_ri_kp6zl = fsfun.f_rel_intake(crop_ri_quan_kp6zl, crop_ri_qual_kp6z[...,na], legume=0)
    crop_vol_kp6zl = fun.f_divide(1000, crop_ri_kp6zl)  # 1000 to convert to vol per tonne

    ##reduce me if nv is higher than livestock diet requirement.
    crop_md_fkp6zl = fsfun.f_effective_mei(1000, crop_md_kp6z[...,na], me_threshold_fp6z[:,na,...,na]
                                           , crop_ri_kp6zl, crop_me_eff_gainlose)
    ##crop cannot be grazed in the confinement pool hence me is 0
    crop_md_fkp6zl = crop_md_fkp6zl * nv_is_not_confinement_f[:,na,na,na,na]

    return crop_md_fkp6zl, crop_vol_kp6zl

def cropgraze_yield_penalty():
    '''
    Yield and stubble penalty associated with grazing 1ha of each crop on each lmu.

    The yield penalty is an inputted percentage of the average yield for each crop. The average yield calculated from
    all rotation phase. The stubble penalty is calculated from the yield
    using f_stubble_production.
    '''
    import Stubble as stub
    import Phase as phs
    ##inputs
    cropgraze_landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    base_yield_k1z = phs.f_base_yield().mean(axis=0, level=1)
    lmu_adj_k2l = phs.f1_mask_lmu(pinp.crop['yield_by_lmu'], axis=1)
    stubble_per_grain_k3 = stub.f_stubble_production()
    yield_reduction_propn = pinp.cropgraze['i_cropgraze_yield_reduction']

    ##give base yields and lmu_adj the correct k axis (k axis needs to be in the correct order and contain all crops so that numpy arrays align).
    ###yield
    yield_idx_bool_k1k = base_yield_k1z.index.values[:,na]==cropgraze_landuse_idx_k
    base_yield_kz = np.sum(base_yield_k1z.values[:,na,:] * yield_idx_bool_k1k[...,na], axis=0)
    ###lmu factor
    lmu_idx_bool_k2k = lmu_adj_k2l.index.values[:,na]==cropgraze_landuse_idx_k
    lmu_adj_kl = np.sum(lmu_adj_k2l.values[:,na,:] * lmu_idx_bool_k2k[...,na], axis=0)
    ###stubble
    stub_idx_bool_k3k = stubble_per_grain_k3.index.values[:,na]==cropgraze_landuse_idx_k
    stubble_per_grain_k = np.sum(stubble_per_grain_k3.values[:,na] * stub_idx_bool_k3k, axis=0)
    ##calc yield penalty for grazing 1ha of crop (note the penalty is allocated into grain pools in pyomo)
    crp_yield_kzl = base_yield_kz[...,na] * lmu_adj_kl[:,na,:]
    yield_penalty_kzl = crp_yield_kzl * yield_reduction_propn

    ##calc stubble reduction
    stubble_penalty_kzl = yield_penalty_kzl * stubble_per_grain_k[:,na,na]
    return yield_penalty_kzl, stubble_penalty_kzl


def f1_cropgraze_params(params, r_vals, nv):
    grazecrop_area_rkl = f_graze_crop_area()
    crop_foo_provided_kp6zl, crop_foo_required_k = f_cropgraze_foo()
    yield_penalty_kzl, stubble_penalty_kzl = cropgraze_yield_penalty()
    crop_md_fkp6zl, crop_vol_kp6zl = crop_md_vol(nv)

    ##keys
    keys_r = np.array(sinp.f_phases().index).astype('str')
    lmu_mask = pinp.general['i_lmu_area'] > 0
    keys_l = pinp.general['i_lmu_idx'][lmu_mask]
    keys_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    keys_p6 = pinp.period['i_fp_idx']
    keys_f  = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    keys_z = pinp.f_keys_z()

    ##array indexes
    ###rkl
    arrays = [keys_r, keys_k, keys_l]
    index_rkl = fun.cartesian_product_simple_transpose(arrays)
    tup_rkl = tuple(map(tuple, index_rkl))
    ###kzl
    arrays = [keys_k, keys_z, keys_l]
    index_kzl = fun.cartesian_product_simple_transpose(arrays)
    tup_kzl = tuple(map(tuple, index_kzl))
    ###kp6zl
    arrays = [keys_k, keys_p6, keys_z, keys_l]
    index_kp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_kp6zl = tuple(map(tuple, index_kp6zl))
    ###fkp6zl
    arrays = [keys_f, keys_k, keys_p6, keys_z, keys_l]
    index_fkp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_fkp6zl = tuple(map(tuple, index_fkp6zl))


    ##create params
    params['grazecrop_area_rkl'] =dict(zip(tup_rkl, grazecrop_area_rkl.ravel()))
    params['crop_foo_provided_kp6zl'] =dict(zip(tup_kp6zl, crop_foo_provided_kp6zl.ravel()))
    params['crop_foo_required_k'] =dict(zip(keys_k, crop_foo_required_k))
    params['yield_penalty_kzl'] =dict(zip(tup_kzl, yield_penalty_kzl.ravel()))
    params['stubble_penalty_kzl'] =dict(zip(tup_kzl, stubble_penalty_kzl.ravel()))
    params['crop_md_fkp6zl'] =dict(zip(tup_fkp6zl, crop_md_fkp6zl.ravel()))
    params['crop_vol_kp6zl'] =dict(zip(tup_kp6zl, crop_vol_kp6zl.ravel()))

