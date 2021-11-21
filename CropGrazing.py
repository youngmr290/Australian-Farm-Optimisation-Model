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
import SeasonalFunctions as zfun


na = np.newaxis


# def f_graze_crop_area():
#     '''
#     The area of each crop that can be grazed for 1ha each rotation phase.
#     Each rotation phase only provide crop grazing for one crop on the arable areas.
#
#     '''
#     ##read phases
#     phases_rh = sinp.f_phases().values
#
#     ##lmu mask
#     lmu_mask = pinp.general['i_lmu_area'] > 0
#
#     ##propn of crop grazing possible for each landuse.
#     landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
#     landuse_grazing_kl = pinp.cropgraze['i_cropgrazing_inc_landuse'][:, lmu_mask]
#
#     ##graze = arable area
#     arable_l = pinp.crop['arable'].squeeze().values[lmu_mask]
#
#     ##area of crop grazing that 1ha of each landuse provides
#     graze_area_kl = landuse_grazing_kl * arable_l
#
#     ##merge to rot phases
#     current_landuse_r = phases_rh[:,-1]
#     a_r_k_rk = current_landuse_r[:,na] == landuse_idx_k
#     cropgraze_area_rkl = graze_area_kl * a_r_k_rk[...,na]
#     return cropgraze_area_rkl

def f_cropgraze_DM(total_DM=False):
    '''
    Calculates the dry matter (DM) available for grazing on crop paddocks and the total DM used to calculate relative
    availability.

    The total DM is calculated from the initial DM plus growth minus the consumption. The initial DM is an inputted amount
    which represents germination and first growth over an establishing period. No grazing can occur during the establishing
    period. After the establishing period the crop grows at an inputted rate per day and a proportion of this growth
    becomes available for consumption. Total DM (used to calc relative availability) is calculated assuming that all
    the crop available for grazing is consumed. Depending on grazing management this may under estimate DM
    however, crop grazing has high availability at low DM levels (due to upright growth) so this limitation is likely
    to be minor.

    If DM is not consumed in the period it grows it is transferred to the following feed period. Currently, it
    doesn't incur a growth rate (e.g. 1t that isn't consumed in fp0 transfers to 1t in fp1). A possible improvement would
    be to include growth in the transfer activity.

    Crop growth rate is an input for each feed period, LMU and land use.
    There are two main limitations of the representation:

        #. Impacts of rotation are not included in the estimation of crop growth.
        #. Growth rate is independent of selected grazing management (e.g. if the crop isn't grazed in the
           first period then the subsequent growth rate does not change).

    Both DM parameters are built with a z8z9 axis. This is because provision of crop grazing is hooked up to the
    seeding activity which is not transferred to future periods and thus doesnt uncluster but seeding provides
    crop DM in future periods which may be in nodes. Thus a z8z9 axis is required so that seeding in z[0] provides
    crop grazing in z[1] and other children.

    :param total_DM: boolean when set to True calculates the total crop DM used to calculate relative availability.
    '''
    ##read inputs
    lmu_mask = pinp.general['i_lmu_area'] > 0
    growth_kp6z = zfun.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1) #kg/d
    wastage_k = pinp.cropgraze['i_cropgraze_wastage']
    growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl'][:,lmu_mask]
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T
    date_feed_periods = per.f_feed_periods()
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    date_end_p5z = mach_periods.values[1:]
    seeding_start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
    initial_DM = pinp.cropgraze['i_cropgraze_initial_dm'] #used to calc total DM for relative availability (vol). The initial DM cant be consumed.
    establishment_days = np.timedelta64(pinp.cropgraze['i_cropgraze_defer_days'], 'D') #days between sowing and grazing
    end_establishment_z = seeding_start_z + establishment_days

    ##adjust crop growth for lmu (kg/d)
    growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]

    ##calc total dry matter accumulation in each feed period - the duration of growth in each feed period is adjusted to
    # account for the establishment period because the DM available at the end of the establishment period is an input.
    crop_grazing_start_z = seeding_start_z + establishment_days
    seed_days_p5z = (date_end_p5z - date_start_p5z).astype('timedelta64[D]').astype(int)

    ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
    base_p6p5z = (date_end_p6z[:,na,:] - np.maximum(np.maximum(date_end_p5z + establishment_days, crop_grazing_start_z)
                                                    , date_start_p6z[:,na,:]))/ np.timedelta64(1, 'D')
    height_p5z = 1
    grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)

    ##grazing days triangular component (for p5) and allocation to feed periods (p6)
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(crop_grazing_start_z, date_start_p5z + establishment_days))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z + establishment_days)
    base_p6p5z = (end_p6p5z - start_p6p5z)/ np.timedelta64(1, 'D')
    ###calculated based on seeding 1ha/day then divided by seed_days to scale the height back to 1 day (which is that seeding activity)
    height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z + establishment_days) - start_p6p5z)/ np.timedelta64(1, 'D')
                                                    , seed_days_p5z))
    height_end_p6p5z = np.maximum(0,fun.f_divide(((date_end_p5z + establishment_days) - end_p6p5z)/ np.timedelta64(1, 'D')
                                                    , seed_days_p5z))
    grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

    ##reduction in total grazing days due to seeding after the first day
    total_grazing_days_p6p5z = grazing_days_tri_p6p5z + grazing_days_rect_p6p5z
    total_dm_growth_kp6p5zl = growth_kp6zl[:,:,na,:,:] * total_grazing_days_p6p5z[...,na]

    ##landuse mask - some crops can't be grazed
    ###lmu mask
    lmu_mask = pinp.general['i_lmu_area'] > 0
    ###propn of crop grazing possible for each landuse.
    landuse_grazing_kl = pinp.cropgraze['i_cropgrazing_inc_landuse'][:, lmu_mask]

    ##season mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)

    if not total_DM:
        ##calc dry matter available for consumption provided by 1ha of crop
        crop_DM_provided_kp6p5zl = total_dm_growth_kp6p5zl * consumption_factor_p6z[:,na,:,na]

        ##season transfer (z8z9) param
        ##little more complicated because seeding in parent p5 provides into all children p6 so p5 axis is required.
        ##and cumultive max on p6 (needs to be done after adding p5)
        #todo this needs to also handle situations where z0 is providing to z2 where z2 is a grandchild.

        # season_start_z = per.f_season_periods()[0,:]  # slice season node to get season start
        # period_is_seasonstart_p6z = date_start_p6z == season_start_z
        # mask_p6z8z9= zfun.f_season_transfer_mask(
        #     date_start_p6z,period_is_seasonstart_pz=period_is_seasonstart_p6z,z_pos=-1)[0] #only want z8z9 mask
        # ###add p5 - If the seeding period (p5) is before the initiation node then z9 is included with the parent. If equal our after then it is not included
        # date_initiate_z = zfun.f_seasonal_inp(pinp.general['i_date_initiate_z'],numpy=True,axis=0).astype('datetime64[D]')
        # p5_prov_p5z9 = date_start_p5z < date_initiate_z
        # mask_p6z8z9 * p5_prov_p5z9[:,na,na,:]

        #todo cant get the stuff below to work needs p6p5z8z9 axis. and parent needs to provide grandchildren for certain p5 slices.
        maskz8_p5z = zfun.f_season_transfer_mask(date_start_p5z, z_pos=-1, mask=True)
        index_z = np.arange(maskz8_p5z.shape[-1])
        a_zcluster_p5z9 = np.maximum.accumulate(index_z * maskz8_p5z)
        crop_DM_provided_kp6p5zlz9 = np.take_along_axis(crop_DM_provided_kp6p5zl[...,na], a_zcluster_p5z9[na,na,:,na,na,:], axis=3)

        # a_zcluster_p5z[...,na] == index_z

        ##calc mask if crop can be grazed
        grazing_exists_p6z = (consumption_factor_p6z > 0)*1

        ##calc mask if DM can be transferred to following period (can only be transferred to periods when consumption is greater than 0)
        transfer_exists_p6z = grazing_exists_p6z * np.roll(grazing_exists_p6z, shift=-1, axis=0) #doesnt transfer into the first period or out of the last hence need to add the roll

        ##calc DM removal when animals consume 1t - accounts for wastage and trampling
        crop_DM_required_kp6z = 1000 / (1 - wastage_k[:,na,na]) * grazing_exists_p6z

        ##apply season mask (only apply here because these become params. The other part of the 'if' statement goes to another function)
        crop_DM_provided_p5kp6zl = crop_DM_provided_p5kp6zl * mask_fp_z8var_p6z[:,:,na]
        transfer_exists_p6z = transfer_exists_p6z * mask_fp_z8var_p6z
        crop_DM_required_kp6z = crop_DM_required_kp6z * mask_fp_z8var_p6z

        return crop_DM_provided_p5kp6zl * landuse_grazing_kl[:,na,na,na,:], crop_DM_required_kp6z, transfer_exists_p6z

    else:
        ##crop foo mid way through feed period after consumption - used to calc vol in the next function.
        ##DM = initial DM plus cumulative sum of DM in previous periods minus DM consumed. Minus half the DM in the current period to get the DM in the middle of the period.
        initial_DM_p6z = initial_DM * (end_establishment_z <= date_end_p6z)
        crop_foo_kp6p5zl =  initial_DM_p6z[:,na,:,na] + np.cumsum(total_dm_growth_kp6p5zl * (1-consumption_factor_p6z[:,na,:,na])
                                                            , axis=1) - total_dm_growth_kp6p5zl/2 * (1-consumption_factor_p6z[:,na,:,na])
        return crop_foo_kp6p5zl * landuse_grazing_kl[:,na,na,na,:]

# def f_DM_reduction_seeding_time():
#     '''
#     Reduction in crop grazing DM available for consumption due to seeding time.
#
#     Crop DM provided by each hectare of rotation is calculated assuming that seeding occurs on the first day of the
#     seedig window. However, if seeding occurs later in the period there will be less DM. This function calculates the
#     reduction in DM due to sowing later in the sowing period.
#     '''
#     ##inputs
#     date_feed_periods = per.f_feed_periods()
#     date_start_p6z = date_feed_periods[:-1]
#     date_end_p6z = date_feed_periods[1:]
#     mach_periods = per.f_p_dates_df()
#     date_start_p5z = mach_periods.values[:-1]
#     date_end_p5z = mach_periods.values[1:]
#     seeding_start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
#     establishment_days = pinp.cropgraze['i_cropgraze_defer_days'] #days between sowing and grazing
#     lmu_mask = pinp.general['i_lmu_area'] > 0
#     growth_kp6z = zfun.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1)
#     growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl'][:,lmu_mask]
#     consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T
#
#
#     crop_grazing_start_z = seeding_start_z + establishment_days
#     seed_days_p5z = (date_end_p5z - date_start_p5z).astype('timedelta64[D]').astype(int)
#
#     ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
#     base_p6p5z = (np.minimum(date_end_p6z[:,na,:], date_start_p5z + establishment_days) \
#                   - np.maximum(crop_grazing_start_z, date_start_p6z[:,na,:]))/ np.timedelta64(1, 'D')
#     height_p5z = 1
#     grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)
#
#     ##grazing days triangular component (for p5) and allocation to feed periods (p6)
#     start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(crop_grazing_start_z, date_start_p5z + establishment_days))
#     end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z + establishment_days)
#     base_p6p5z = (end_p6p5z - start_p6p5z)/ np.timedelta64(1, 'D')
#     height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z + establishment_days) - start_p6p5z)/ np.timedelta64(1, 'D')
#                                                     , seed_days_p5z))
#     height_end_p6p5z = np.maximum(0,fun.f_divide(((date_end_p5z + establishment_days) - end_p6p5z)/ np.timedelta64(1, 'D')
#                                                     , seed_days_p5z))
#     grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)
#
#     ##reduction in total grazing days due to seeding after the first day
#     total_grazing_days_reduction_p6p5z = grazing_days_tri_p6p5z + grazing_days_rect_p6p5z
#
#     ##reduction in DM available for consumption due to seeding after the first day
#     ###adjust crop growth for lmu
#     growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]
#     DM_reduction_kp6p5zl = total_grazing_days_reduction_p6p5z[...,na] * growth_kp6zl[:,:,na,...] * consumption_factor_p6z[:,na,:,na]
#
#     ##apply season mask
#     mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
#     DM_reduction_kp6p5zl = DM_reduction_kp6p5zl * mask_fp_z8var_p6z[:,na,:,na]
#
#     return DM_reduction_kp6p5zl




def crop_md_vol(nv):
    '''
    Energy provided and volume required from 1t of crop grazing.
    '''

    ##inputs
    crop_dmd_kp6z = zfun.f_seasonal_inp(pinp.cropgraze['i_crop_dmd_kp6z'],numpy=True,axis=-1)
    crop_foo_kp6zl = f_cropgraze_DM(total_DM=True)
    hr = pinp.cropgraze['i_hr_crop']
    me_threshold_fp6z = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)
    crop_me_eff_gainlose = pinp.cropgraze['i_crop_me_eff_gainlose']
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T

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
        crop_ri_quan_kp6zl = fsfun.f_ra_cs(crop_foo_kp6zl, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        crop_ri_quan_kp6zl = fsfun.f_ra_mu(crop_foo_kp6zl, hf)

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

    ##apply season mask and grazing exists mask
    ###calc mask if crop can be grazed
    grazing_exists_p6z = (consumption_factor_p6z > 0) * 1
    ###calc season mask
    date_start_p6z = per.f_feed_periods()[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    ###apply masks
    crop_md_fkp6zl = crop_md_fkp6zl * mask_fp_z8var_p6z[:,:,na] * grazing_exists_p6z[:,:,na]
    crop_vol_fkp6zl = crop_vol_fkp6zl * mask_fp_z8var_p6z[:,:,na] * grazing_exists_p6z[:,:,na]

    return crop_md_fkp6zl, crop_vol_fkp6zl

def f_cropgraze_yield_penalty():
    '''
    Yield and stubble penalty associated with the amount of crop consumed.

    The yield penalty is an inputted proportion of the dry matter consumed. The stubble penalty is calculated
    from the yield using f_cropresidue_production.
    '''
    import CropResidue as stub
    ##inputs
    cropgraze_landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    stubble_per_grain_k3 = stub.f_cropresidue_production()
    yield_reduction_propn_kp6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_yield_reduction_kp6z'], numpy=True, axis=-1)
    proportion_grain_harv_k = pd.Series(pinp.stubble['proportion_grain_harv'], index=pinp.stubble['i_stub_landuse_idx'])
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T

    ##correct stubble k axis (k axis needs to be in the correct order and contain all crops so that numpy arrays align).
    stub_idx_bool_k3k = stubble_per_grain_k3.index.values[:,na]==cropgraze_landuse_idx_k
    stubble_per_grain_k = np.sum(stubble_per_grain_k3.values[:,na] * stub_idx_bool_k3k, axis=0)

    ##adjust seeding penalty - crops that are not harvested eg fodder don't have yield penalty. But do have a stubble penalty
    ###correct stubble k axis (k axis needs to be in the correct order and contain all crops so that numpy arrays align).
    stub_idx_bool_k3k = proportion_grain_harv_k.index.values[:,na]==cropgraze_landuse_idx_k
    proportion_grain_harv_k = np.sum(proportion_grain_harv_k.values[:,na] * stub_idx_bool_k3k, axis=0)
    ###if calculating yield penalty for stubble then include all crop (eg include fodders)
    stub_yield_reduction_propn_kp6z = yield_reduction_propn_kp6z
    ###if calculating yield penalty for grain transfer then only include harvested crops (eg don't include fodders)
    yield_reduction_propn_kp6z = yield_reduction_propn_kp6z * (proportion_grain_harv_k>0)[:,na,na]

    ##calc stubble reduction (kg of stubble per kg of crop DM consumed)
    stubble_reduction_propn_kp6z = stub_yield_reduction_propn_kp6z * stubble_per_grain_k[:,na,na]

    ##apply season mask and grazing exists mask
    ###calc mask if crop can be grazed
    grazing_exists_p6z = (consumption_factor_p6z > 0) * 1
    ###calc season mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(per.f_feed_periods()[:-1], z_pos=-1, mask=True)
    ###apply masks
    yield_reduction_propn_kp6z = yield_reduction_propn_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z
    stubble_reduction_propn_kp6z = stubble_reduction_propn_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z

    return yield_reduction_propn_kp6z, stubble_reduction_propn_kp6z


def f1_cropgraze_params(params, r_vals, nv):
    # grazecrop_area_rkl = f_graze_crop_area()
    crop_DM_provided_p5kp6zl, crop_DM_required_kp6z, transfer_exists_p6z = f_cropgraze_DM()
    yield_reduction_propn_kp6z, stubble_reduction_propn_kp6z = f_cropgraze_yield_penalty()
    crop_md_fkp6zl, crop_vol_fkp6zl = crop_md_vol(nv)
    # DM_reduction_kp6p5zl = f_DM_reduction_seeding_time()

    ##keys
    keys_r = np.array(sinp.f_phases().index).astype('str')
    lmu_mask = pinp.general['i_lmu_area'] > 0
    keys_l = pinp.general['i_lmu_idx'][lmu_mask]
    keys_k = pinp.cropgraze['i_cropgraze_landuse_idx']
    # keys_m = per.f_season_periods(keys=True)
    keys_p6 = pinp.period['i_fp_idx']
    keys_p5 = np.asarray(per.f_p_dates_df().index[:-1]).astype('str')
    keys_f  = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    keys_z = zfun.f_keys_z()

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
    ###p5kp6zl
    arrays = [keys_p5, keys_k, keys_p6, keys_z, keys_l]
    index_p5kp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_p5kp6zl = tuple(map(tuple, index_p5kp6zl))
    # ###kp6p5zl
    # arrays = [keys_k, keys_p6, keys_p5, keys_z, keys_l]
    # index_kp6p5zl = fun.cartesian_product_simple_transpose(arrays)
    # tup_kp6p5zl = tuple(map(tuple, index_kp6p5zl))
    ###fkp6zl
    arrays = [keys_f, keys_k, keys_p6, keys_z, keys_l]
    index_fkp6zl = fun.cartesian_product_simple_transpose(arrays)
    tup_fkp6zl = tuple(map(tuple, index_fkp6zl))


    ##create params
    # params['grazecrop_area_rkl'] = dict(zip(tup_rkl, grazecrop_area_rkl.ravel()))
    params['crop_DM_provided_p5kp6zl'] = dict(zip(tup_p5kp6zl, crop_DM_provided_p5kp6zl.ravel()))
    # params['DM_reduction_kp6p5zl'] = dict(zip(tup_kp6p5zl, DM_reduction_kp6p5zl.ravel()))
    params['crop_DM_required_kp6z'] = dict(zip(tup_kp6z, crop_DM_required_kp6z.ravel()))
    params['transfer_exists_p6z'] = dict(zip(tup_p6z, transfer_exists_p6z.ravel()))
    params['yield_reduction_propn_kp6z'] = dict(zip(tup_kp6z, yield_reduction_propn_kp6z.ravel()))
    params['stubble_reduction_propn_kp6z'] = dict(zip(tup_kp6z, stubble_reduction_propn_kp6z.ravel()))
    params['crop_md_fkp6zl'] = dict(zip(tup_fkp6zl, crop_md_fkp6zl.ravel()))
    params['crop_vol_kp6zl'] = dict(zip(tup_fkp6zl, crop_vol_fkp6zl.ravel()))

