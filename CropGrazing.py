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

def f_cropgraze_DM(total_DM=False):
    '''
    Calculates the dry matter (DM) available for grazing on crop paddocks and the total DM used to calculate relative
    availability.

    The total DM is calculated from the initial DM plus growth minus the consumption. The initial DM is an inputted amount
    which represents germination and first growth over an establishing period. No grazing can occur during the establishing
    period. After the establishing period the crop grows at an inputted rate per day and a proportion of this growth
    becomes available for consumption. The total DM and the DM available for consumption are calculated assuming that
    the crop is sown on the first day of seeding however, this may not be the case. To account for this a FOO reduction is
    included that is linked to the crop sowing activity (see f_DM_reduction_seeding_time). It is not possible to adjust
    the DM used in the relative availability (volume) calculation. Thus it assumes that seeding occurs on the
    first day of seeding period (potentially overestimating DM) and the maximum DM is consumed in each
    period (potentially underestimating DM). These two limitations somewhat balance each other out.

    If DM is not consumed in the period is grows it is transferred to the following feed period. Currently, it
    doesn't incur a growth rate (e.g. 1t that isn't consumed in fp0 transfers to 1t in fp1). A possible improvement would
    be to include growth in the transfer activity.

    Crop growth rate is an input for each feed period, LMU and land use.
    There are two main limitations of the representation:

        #. Impacts of rotation are not included in the estimation of crop growth.
        #. Growth rate remains the same independent of selected grazing management (e.g. if AFO opt to graze less
           crop in the first period the growth rate does.

    :param DM: boolean when set to True calculates the total crop DM used to calculate relative availability.
    '''
    ##read inputs
    lmu_mask = pinp.general['i_lmu_area'] > 0
    growth_kp6z = pinp.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1)
    wastage_k = pinp.cropgraze['i_cropgraze_wastage']
    growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl'][:,lmu_mask]
    consumption_factor_p6z = pinp.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T
    date_feed_periods = per.f_feed_periods().astype('datetime64')
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    seeding_start_z = per.wet_seeding_start_date().astype(np.datetime64)
    initial_DM = pinp.cropgraze['i_cropgraze_initial_dm'] #used to calc total DM for relative availability (vol). The initial DM cant be consumed.
    establishment_days = pinp.cropgraze['i_cropgraze_defer_days'] #days between sowing and grazing

    ##adjust crop growth for lmu
    growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]

    ##calc total dry matter in each feed period - have to adjust the length of the feed period to account for
    # establishment period (growth is not calculated during the establishment period there is just an inputted initial DM)
    end_establishment_z = seeding_start_z + establishment_days
    date_start_adj_p6z = np.maximum(date_start_p6z, end_establishment_z)
    feed_period_lengths_p6z = np.maximum(0,(date_end_p6z - date_start_adj_p6z).astype('timedelta64[D]').astype('float'))
    total_dm_growth_kp6zl = growth_kp6zl * feed_period_lengths_p6z[:,na]

    if not total_DM:
        ##calc dry matter available for consumption provided by 1ha of crop
        crop_DM_provided_kp6zl = total_dm_growth_kp6zl * consumption_factor_p6z[:,na]

        ##calc foo required for animals to consume 1t - accounts for wastage
        crop_DM_required_k = 1000 / (1 - wastage_k)

        ##calc mask if DM can be transferred to following period (can only be transferred to periods when consumption is greater than 0)
        transfer_exists_p6z = (consumption_factor_p6z > 0)*1

        return crop_DM_provided_kp6zl, crop_DM_required_k, transfer_exists_p6z

    else:
        ##crop foo mid way through feed period after consumption - used to calc vol in the next function.
        ##DM = initial DM plus cumulative sum of DM in previous periods minus DM consumed. Minus half the DM in the current period to get the DM in the middle of the period.
        initial_DM_p6z = initial_DM * (end_establishment_z <= date_end_p6z)
        crop_DM_kp6zl =  initial_DM_p6z[...,na] + np.cumsum(total_dm_growth_kp6zl * (1-consumption_factor_p6z[:,na])
                                                            , axis=1) - total_dm_growth_kp6zl/2 * (1-consumption_factor_p6z[:,na])
        return crop_DM_kp6zl

def f_DM_reduction_seeding_time():
    '''
    Reduction in crop grazing DM available for consumption due to seeding time.

    Crop DM provided by each hectare of rotation is calculated assuming that seeding occurs on the first day of the
    seedig window. However, if seeding occurs later in the period there will be less DM. This function calculates the
    reduction in DM due to sowing later in the sowing period.
    '''
    ##inputs
    date_feed_periods = per.f_feed_periods().astype('datetime64')
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    mach_periods = per.p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    date_end_p5z = mach_periods.values[1:]
    seeding_start_z = per.wet_seeding_start_date().astype(np.datetime64)
    establishment_days = pinp.cropgraze['i_cropgraze_defer_days'] #days between sowing and grazing
    lmu_mask = pinp.general['i_lmu_area'] > 0
    growth_kp6z = pinp.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1)
    growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl'][:,lmu_mask]
    consumption_factor_p6z = pinp.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T


    crop_grazing_start_z = seeding_start_z + establishment_days
    seed_days_p5z = (date_end_p5z - date_start_p5z).astype('timedelta64[D]').astype(int)

    ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
    base_p6p5z = (np.minimum(date_end_p6z[:,na,:], date_start_p5z + establishment_days) \
                  - np.maximum(crop_grazing_start_z, date_start_p6z[:,na,:]))/ np.timedelta64(1, 'D')
    height_p5z = 1
    grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)

    ##grazing days triangular component (for p5) and allocation to feed periods (p6)
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(crop_grazing_start_z, date_start_p5z + establishment_days))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z + establishment_days)
    base_p6p5z = (end_p6p5z - start_p6p5z)/ np.timedelta64(1, 'D')
    height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z + establishment_days) - start_p6p5z)/ np.timedelta64(1, 'D')
                                                        , seed_days_p5z))
    height_end_p6p5z = fun.f_divide(np.maximum(0,((date_end_p5z + establishment_days) - end_p6p5z)/ np.timedelta64(1, 'D'))
                                    , seed_days_p5z)
    grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

    ##total grazing days crop growth couldnt occur due to seeding after the first day
    total_grazing_days_reduction_p6p5z = grazing_days_tri_p6p5z + grazing_days_rect_p6p5z

    ##reduction in DM available for consumption per day sowing occurs after seeding start
    ###adjust crop growth for lmu
    growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]
    DM_reduction_kp6p5zl = total_grazing_days_reduction_p6p5z[...,na] * growth_kp6zl[:,:,na,...] * consumption_factor_p6z[:,na,:,na]
    return DM_reduction_kp6p5zl




def crop_md_vol(nv):
    '''
    Energy provided and volume required from 1t of crop grazing.
    '''

    ##inputs
    crop_dmd_kp6z = pinp.f_seasonal_inp(pinp.cropgraze['i_crop_dmd_kp6z'],numpy=True,axis=-1)
    crop_DM_kp6zl = f_cropgraze_DM(total_DM=True)
    hr = pinp.cropgraze['i_hr_crop']
    me_threshold_fp6z = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)
    crop_me_eff_gainlose = pinp.cropgraze['i_crop_me_eff_gainlose']

    ##nv stuff
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.

    ## vol
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        crop_ri_qual_kp6z = fsfun.f_rq_cs(crop_dmd_kp6z, legume=0)

    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    hf= fsfun.f_hf(hr) #height factor
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        crop_ri_quan_kp6zl = fsfun.f_ra_cs(crop_DM_kp6zl, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        crop_ri_quan_kp6zl = fsfun.f_ra_mu(crop_DM_kp6zl, hf)

    crop_ri_kp6zl = fsfun.f_rel_intake(crop_ri_quan_kp6zl, crop_ri_qual_kp6z[...,na], legume=0)
    crop_vol_kp6zl = fun.f_divide(1000, crop_ri_kp6zl)  # 1000 to convert to vol per tonne
    crop_vol_fkp6zl = crop_vol_kp6zl * nv_is_not_confinement_f[:,na,na,na,na] #me from crop is 0 in the confinement pool

    ## md per tonne
    crop_md_kp6z = fsfun.dmd_to_md(crop_dmd_kp6z)
    ##reduce me if nv is higher than livestock diet requirement.
    crop_md_fkp6zl = fsfun.f_effective_mei(1000, crop_md_kp6z[...,na], me_threshold_fp6z[:,na,...,na]
                                           , nv['confinement_inc'], crop_ri_kp6zl, crop_me_eff_gainlose)
    ##crop cannot be grazed in the confinement pool hence me is 0
    crop_md_fkp6zl = crop_md_fkp6zl * nv_is_not_confinement_f[:,na,na,na,na]

    return crop_md_fkp6zl, crop_vol_fkp6zl

def cropgraze_yield_penalty():
    '''
    Yield and stubble penalty associated with the amount of crop consumed.

    The yield penalty is an inputted percentage of the average yield for each crop. The average yield calculated from
    all rotation phase. The stubble penalty is calculated from the yield
    using f_stubble_production.
    '''
    import Stubble as stub
    import Phase as phs
    ##inputs
    cropgraze_landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    stubble_per_grain_k3 = stub.f_stubble_production()
    yield_reduction_propn_kp6z = pinp.cropgraze['i_cropgraze_yield_reduction_kp6z']

    ##correct stubble k axis (k axis needs to be in the correct order and contain all crops so that numpy arrays align).
    stub_idx_bool_k3k = stubble_per_grain_k3.index.values[:,na]==cropgraze_landuse_idx_k
    stubble_per_grain_k = np.sum(stubble_per_grain_k3.values[:,na] * stub_idx_bool_k3k, axis=0)

    ##calc stubble reduction
    stubble_reduction_propn_kp6z = yield_reduction_propn_kp6z * stubble_per_grain_k[:,na,na]
    return yield_reduction_propn_kp6z, stubble_reduction_propn_kp6z


def f1_cropgraze_params(params, r_vals, nv):
    grazecrop_area_rkl = f_graze_crop_area()
    crop_DM_provided_kp6zl, crop_DM_required_k, transfer_exists_p6z = f_cropgraze_DM()
    yield_reduction_propn_kp6z, stubble_reduction_propn_kp6z = cropgraze_yield_penalty()
    crop_md_fkp6zl, crop_vol_fkp6zl = crop_md_vol(nv)
    DM_reduction_kp6p5zl = f_DM_reduction_seeding_time()

    ##keys
    keys_r = np.array(sinp.f_phases().index).astype('str')
    lmu_mask = pinp.general['i_lmu_area'] > 0
    keys_l = pinp.general['i_lmu_idx'][lmu_mask]
    keys_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    keys_p6 = pinp.period['i_fp_idx']
    keys_p5 = np.asarray(per.p_dates_df().index[:-1]).astype('str')
    keys_f  = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    keys_z = pinp.f_keys_z()

    ##array indexes
    ###rkl
    arrays = [keys_r, keys_k, keys_l]
    index_rkl = fun.cartesian_product_simple_transpose(arrays)
    tup_rkl = tuple(map(tuple, index_rkl))
    ###kp6z
    arrays = [keys_k, keys_p6, keys_z]
    index_kp6z = fun.cartesian_product_simple_transpose(arrays)
    tup_kp6z = tuple(map(tuple, index_kp6z))
    ###p6z
    arrays = [keys_p6, keys_z]
    index_p6z = fun.cartesian_product_simple_transpose(arrays)
    tup_p6z = tuple(map(tuple, index_p6z))
    ###kp6zl
    arrays = [keys_k, keys_p6, keys_z, keys_l]
    index_kp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_kp6zl = tuple(map(tuple, index_kp6zl))
    ###kp6p5zl
    arrays = [keys_k, keys_p6, keys_p5, keys_z, keys_l]
    index_kp6p5zl = fun.cartesian_product_simple_transpose(arrays)
    tup_kp6p5zl = tuple(map(tuple, index_kp6p5zl))
    ###fkp6zl
    arrays = [keys_f, keys_k, keys_p6, keys_z, keys_l]
    index_fkp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_fkp6zl = tuple(map(tuple, index_fkp6zl))


    ##create params
    params['grazecrop_area_rkl'] = dict(zip(tup_rkl, grazecrop_area_rkl.ravel()))
    params['crop_DM_provided_kp6zl'] = dict(zip(tup_kp6zl, crop_DM_provided_kp6zl.ravel()))
    params['DM_reduction_kp6p5zl'] = dict(zip(tup_kp6p5zl, DM_reduction_kp6p5zl.ravel()))
    params['crop_DM_required_k'] = dict(zip(keys_k, crop_DM_required_k))
    params['transfer_exists_p6z'] = dict(zip(tup_p6z, transfer_exists_p6z.ravel()))
    params['yield_reduction_propn_kp6z'] = dict(zip(tup_kp6z, yield_reduction_propn_kp6z.ravel()))
    params['stubble_reduction_propn_kp6z'] = dict(zip(tup_kp6z, stubble_reduction_propn_kp6z.ravel()))
    params['crop_md_fkp6zl'] = dict(zip(tup_fkp6zl, crop_md_fkp6zl.ravel()))
    params['crop_vol_kp6zl'] = dict(zip(tup_fkp6zl, crop_vol_fkp6zl.ravel()))

