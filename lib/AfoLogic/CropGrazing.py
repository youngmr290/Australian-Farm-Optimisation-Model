'''
Author: Young

Crop grazing is an option that allows stock to graze green crops, by default, from June until August (user
customisable range). Green crops have a high energy content and grow erect allowing for easier grazing.
Therefore, crops can meet livestock energy needs at a lower FOO than an equivalent pasture. However, for
every kilogram of crop biomass consumed yield is reduced by 150 grams per hectare (user customisable),
with a corresponding effect on stubble production. Trials have recorded varying yield penalties from -15% to +15%, but the
consensus is that the yield penalty is minimal if the crop is grazed early and lightly. The level of the yield
penalty is an input which can be easily changed.
'''

##import python modules
import pandas as pd
import numpy as np

##import AFO modules
from . import PropertyInputs as pinp
from . import StructuralInputs as sinp
from . import UniversalInputs as uinp
from . import Periods as per
from . import FeedsupplyFunctions as fsfun
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import EmissionFunctions as efun

na = np.newaxis


# def f_graze_crop_area():
#     '''
#     The area of each crop that can be grazed for 1ha each rotation phase.
#     Each rotation phase only provide crop grazing for one crop on the arable areas.
#
#     '''
#     ##read phases
#     phases_rh = pinp.f1_phases().values
#
#     ##lmu mask
#     lmu_mask = pinp.general['i_lmu_area'] > 0
#
#     ##propn of crop grazing possible for each landuse.
#     landuse_idx_k = pinp.cropgraze['i_cropgraze_landuse_idx']
#     landuse_grazing_kl = pinp.cropgraze['i_cropgrazing_inc_landuse'][:, lmu_mask]
#
#     ##graze = arable area
#     arable_l = pinp.general['arable'][lmu_mask]
#
#     ##area of crop grazing that 1ha of each landuse provides
#     graze_area_kl = landuse_grazing_kl * arable_l
#
#     ##merge to rot phases
#     current_landuse_r = phases_rh[:,-1]
#     a_r_k_rk = current_landuse_r[:,na] == landuse_idx_k
#     cropgraze_area_rkl = graze_area_kl * a_r_k_rk[...,na]
#     return cropgraze_area_rkl

def f_cropgraze_DM(r_vals=None, total_DM=False):
    '''
    Calculates the dry matter (DM) available for grazing on crop paddocks and the total DM used to calculate relative
    availability.

    The total DM is calculated from the initial DM plus growth minus the consumption. The initial DM is an inputted amount
    which represents germination and first growth over an establishing period. No grazing can occur during the establishing
    period. After the establishing period, the crop grows at an inputted rate per day and a proportion of this growth
    becomes available for consumption. Total DM (used to calc relative availability) is calculated assuming that all
    the crop available for grazing is consumed. Depending on grazing management this may underestimate DM
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
    seeding activity which is not transferred to future periods and thus doesn't uncluster but seeding provides
    crop DM in future periods which may be in nodes. Therefore, a z8z9 axis is required so that seeding in z[0] provides
    crop grazing in z[1] and other children.

    P5 axis is required to represent sowing time. Sowing time impacts the DM provided in each p6 period.

    :param total_DM: boolean when set to True calculates the total crop DM used to calculate relative availability.
    '''
    ##read inputs
    cropgrazing_inc = pinp.cropgraze['i_cropgrazing_inc']
    growth_kp6z = zfun.f_seasonal_inp(np.moveaxis(pinp.cropgraze['i_crop_growth_zkp6'], source=0, destination=-1),numpy=True,axis=-1) #kg/d
    wastage_k = pinp.cropgraze['i_cropgraze_wastage']
    growth_lmu_factor_kl = pinp.cropgraze['i_cropgrowth_lmu_factor_kl']
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T
    date_feed_periods = per.f_feed_periods()
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    date_end_p5z = mach_periods.values[1:]
    seeding_start_z = per.f_wet_seeding_start_date()
    initial_DM = pinp.cropgraze['i_cropgraze_initial_dm'] #used to calc total DM for relative availability (vol). The initial DM cant be consumed.
    establishment_days = pinp.cropgraze['i_cropgraze_defer_days'] #days between sowing and grazing

    ##adjust crop growth for lmu (kg/d)
    growth_kp6zl = growth_kp6z[...,na] * growth_lmu_factor_kl[:,na,na,:]

    ##calc total dry matter accumulation in each feed period - the duration of growth in each feed period is adjusted to
    # account for the establishment period because the DM available at the end of the establishment period is an input.
    crop_grazing_start_z = seeding_start_z + establishment_days
    seed_days_p5z = (date_end_p5z - date_start_p5z)

    ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
    base_p6p5z = (date_end_p6z[:,na,:] - np.maximum(np.maximum(date_end_p5z + establishment_days, crop_grazing_start_z)
                                                    , date_start_p6z[:,na,:]))
    height_p5z = 1
    grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)

    ##grazing days triangular component (for p5) and allocation to feed periods (p6)
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(crop_grazing_start_z, date_start_p5z + establishment_days))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z + establishment_days)
    base_p6p5z = (end_p6p5z - start_p6p5z)
    ###calculated based on seeding 1ha/day then divided by seed_days to scale the height back to 1 day (which is that seeding activity)
    height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z + establishment_days) - start_p6p5z)
                                                    , seed_days_p5z))
    height_end_p6p5z = np.maximum(0,fun.f_divide(((date_end_p5z + establishment_days) - end_p6p5z)
                                                    , seed_days_p5z))
    grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

    ##reduction in total grazing days due to seeding after the first day
    total_grazing_days_p6p5z = grazing_days_tri_p6p5z + grazing_days_rect_p6p5z
    total_dm_growth_kp6p5zl = growth_kp6zl[:,:,na,:,:] * total_grazing_days_p6p5z[...,na]

    ##landuse mask - some crops can't be grazed
    ###lmu mask
    ###propn of crop grazing possible for each landuse.
    propn_area_grazed_kl = pinp.cropgraze['i_cropgraze_propn_area_grazed_kl']
    ###mask which z crop graing can occur
    propn_area_grazed_kl = propn_area_grazed_kl * cropgrazing_inc

    ##season mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)

    if not total_DM:
        ##calc dry matter available for consumption provided by 1ha of crop
        crop_DM_provided_kp6p5zl = total_dm_growth_kp6p5zl * consumption_factor_p6z[:,na,:,na]

        ##season transfer (z8z9) param
        ##little more complicated because seeding in parent p5 provides into all children p6 so p5 axis is required.
        ##and cumultive max on p6 (needs to be done after adding p5)
        maskz8_p5z = zfun.f_season_transfer_mask(date_start_p5z, z_pos=-1, mask=True)
        index_z = np.arange(maskz8_p5z.shape[-1])
        a_zcluster_p5z9 = np.maximum.accumulate(index_z * maskz8_p5z, axis=1)
        crop_DM_provided_kp6p5z8lz9 = crop_DM_provided_kp6p5zl[...,na] * (a_zcluster_p5z9[na,na,:,na,na,:] == index_z[:,na,na])


        ##calc mask if crop can be grazed
        ###p6 grazing only occurs in periods defined by user
        grazing_exists_p6z = (consumption_factor_p6z > 0)
        ###can only graze crop sown in p5 if p5 < p6
        grazing_exists_p6p5z = date_start_p5z + establishment_days <= date_end_p6z[:,na,:]
        grazing_exists_p6p5z = np.logical_and(grazing_exists_p6z[:,na,:], grazing_exists_p6p5z) * 1

        ##calc mask if DM can be transferred to following period (can only be transferred to periods when consumption is greater than 0)
        transfer_exists_p6p5z = grazing_exists_p6p5z * np.roll(grazing_exists_p6p5z, shift=-1, axis=0) #doesn't transfer into the first period or out of the last hence need to add the roll

        ##calc DM removal when animals consume 1t - accounts for wastage and trampling
        crop_DM_required_kp6p5z = 1000 / (1 - wastage_k[:,na,na,na]) * grazing_exists_p6p5z

        ##apply season mask (only apply here because these become params. The other part of the 'if' statement goes to another function)
        crop_DM_provided_kp6p5z8lz9 = crop_DM_provided_kp6p5z8lz9 * mask_fp_z8var_p6z[:,na,:,na,na]
        transfer_exists_p6p5z = transfer_exists_p6p5z * mask_fp_z8var_p6z[:,na,:]
        crop_DM_required_kp6p5z = crop_DM_required_kp6p5z * mask_fp_z8var_p6z[:,na,:]

        crop_DM_provided_kp6p5z8lz9 = crop_DM_provided_kp6p5z8lz9 * propn_area_grazed_kl[:,na,na,na,:,na]

        ##store report vals
        fun.f1_make_r_val(r_vals, crop_DM_provided_kp6p5z8lz9, 'crop_DM_provided_kp6p5z8lz9') #doesnt need unclustering because of z9 axis
        fun.f1_make_r_val(r_vals, crop_DM_required_kp6p5z, 'crop_DM_required_kp6p5z') #doesnt need unclustering because of z9 axis

        return crop_DM_provided_kp6p5z8lz9, crop_DM_required_kp6p5z, transfer_exists_p6p5z

    else:
        ##crop foo mid way through feed period after consumption - used to calc vol in the next function.
        ##DM = initial DM plus cumulative sum of DM in previous periods minus DM consumed. Minus half the DM in the current period to get the DM in the middle of the period.
        initial_DM_p6p5z = initial_DM * np.logical_and(date_start_p5z + establishment_days < date_end_p6z[:,na,:],
                                                     date_start_p5z + establishment_days >= date_start_p6z[:,na,:]) #only get initial DM in the fp when seeding first occurs.
        crop_foo_kp6p5zl =  initial_DM_p6p5z[...,na] + np.cumsum(total_dm_growth_kp6p5zl * (1-consumption_factor_p6z[:,na,:,na])
                                                            , axis=1) - total_dm_growth_kp6p5zl/2 * (1-consumption_factor_p6z[:,na,:,na])
        return crop_foo_kp6p5zl * (propn_area_grazed_kl[:,na,na,na,:]>0) #second part to set unused dv to 0 so it is removed from lp.

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




def crop_md_vol(nv, r_vals):
    '''
    Energy provided and volume required from 1t of crop grazing.

    p5 axis exists because sowing time influences crop DM which impacts ME and vol.
    '''

    ##inputs
    crop_dmd_kp6z = zfun.f_seasonal_inp(pinp.cropgraze['i_crop_dmd_kp6z'],numpy=True,axis=-1)
    crop_foo_kp6p5zl = f_cropgraze_DM(total_DM=True)
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
        crop_ri_quan_kp6p5zl = fsfun.f_ra_cs(crop_foo_kp6p5zl, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        crop_ri_quan_kp6p5zl = fsfun.f_ra_mu(crop_foo_kp6p5zl, hf)

    crop_ri_kp6p5zl = fsfun.f_rel_intake(crop_ri_quan_kp6p5zl, crop_ri_qual_kp6z[:,:,na,:,na], legume=0)
    crop_vol_kp6p5zl = fun.f_divide(1000, crop_ri_kp6p5zl)  # 1000 to convert to vol per tonne
    crop_vol_fkp6p5zl = crop_vol_kp6p5zl * nv_is_not_confinement_f[:,na,na,na,na,na] #me from crop is 0 in the confinement pool

    ## md per tonne
    crop_md_kp6z = fsfun.f1_dmd_to_md(crop_dmd_kp6z)
    ##reduce me if nv is higher than livestock diet requirement.
    crop_md_fkp6p5zl = fsfun.f_effective_mei(1000, crop_md_kp6z[:,:,na,:,na], me_threshold_fp6z[:,na,:,na,:,na]
                                           , nv['confinement_inc'], crop_ri_kp6p5zl, crop_me_eff_gainlose)
    ##crop cannot be grazed in the confinement pool hence me is 0
    crop_md_fkp6p5zl = crop_md_fkp6p5zl * nv_is_not_confinement_f[:,na,na,na,na,na]

    ##apply season mask and grazing exists mask
    ###calc mask if crop can be grazed
    grazing_exists_p6z = (consumption_factor_p6z > 0) * 1
    ###can only graze crop sown in p5 if p5 < p6
    date_start_p5z = per.f_p_dates_df().values[:-1]
    date_end_p6z = per.f_feed_periods()[1:]
    establishment_days = pinp.cropgraze['i_cropgraze_defer_days'] #days between sowing and grazing
    grazing_exists_p6p5z = date_start_p5z + establishment_days <= date_end_p6z[:,na,:]
    grazing_exists_p6p5z = np.logical_and(grazing_exists_p6z[:,na,:],grazing_exists_p6p5z)
    ###calc season mask
    date_start_p6z = per.f_feed_periods()[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    ###apply masks
    crop_md_fkp6p5zl = crop_md_fkp6p5zl * mask_fp_z8var_p6z[:,na,:,na] * grazing_exists_p6p5z[...,na]
    crop_vol_fkp6p5zl = crop_vol_fkp6p5zl * mask_fp_z8var_p6z[:,na,:,na] * grazing_exists_p6p5z[...,na]

    ##store report vals
    fun.f1_make_r_val(r_vals,crop_md_fkp6p5zl,'crop_md_fkp6p5zl',mask_fp_z8var_p6z[:,na,:,na],z_pos=-2)

    return crop_md_fkp6p5zl, crop_vol_fkp6p5zl

def f_cropgraze_emissions(r_vals):
    '''
    Livestock emissions liked to consuming 1t of green crop.

    '''

    ##inputs
    crop_dmd_kp6z = zfun.f_seasonal_inp(pinp.cropgraze['i_crop_dmd_kp6z'],numpy=True,axis=-1)
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'], numpy=True, axis=0).T
    i_grn_cp_p6z = zfun.f_seasonal_inp(pinp.pasture_inputs['annual']['CPGrn'], numpy=True, axis=1) #assuming that protien of green crop is the same as annual pastures

    ##livestock methane emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_ch4_cropgraze_kp6z = efun.f_stock_ch4_feed_nir(1000, crop_dmd_kp6z)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        crop_md_kp6z = fsfun.f1_dmd_to_md(crop_dmd_kp6z)
        stock_ch4_cropgraze_kp6z = efun.f_stock_ch4_feed_bc(1000, crop_md_kp6z)

    ##livestock nitrous oxide emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_cropgraze_kp6z = efun.f_stock_n2o_feed_nir(1000, crop_dmd_kp6z, i_grn_cp_p6z) #assuming that protien of green crop is the same as annual pastures

    co2e_cropgraze_kp6z = stock_ch4_cropgraze_kp6z * uinp.emissions['i_ch4_gwp_factor'] + stock_n2o_cropgraze_kp6z * uinp.emissions['i_n2o_gwp_factor']

    ##apply season mask and grazing exists mask
    ###calc mask if crop can be grazed
    grazing_exists_p6z = (consumption_factor_p6z > 0) * 1
    ###calc season mask
    date_start_p6z = per.f_feed_periods()[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    ###apply masks
    stock_n2o_cropgraze_kp6z = stock_n2o_cropgraze_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z
    stock_ch4_cropgraze_kp6z = stock_ch4_cropgraze_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z
    co2e_cropgraze_kp6z = co2e_cropgraze_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z

    ##store report vals
    fun.f1_make_r_val(r_vals,stock_n2o_cropgraze_kp6z,'stock_n2o_cropgraze_kp6z',mask_fp_z8var_p6z,z_pos=-1)
    fun.f1_make_r_val(r_vals,stock_ch4_cropgraze_kp6z,'stock_ch4_cropgraze_kp6z',mask_fp_z8var_p6z,z_pos=-1)

    return co2e_cropgraze_kp6z


def f_cropgraze_biomass_penalty(r_vals):
    '''
    Biomass penalty associated with the amount of crop consumed.

    The yield penalty is an inputted proportion of the dry matter consumed. Below it is converted to a biomass
    penalty.
    '''
    # import CropResidue as stub
    ##inputs
    # stubble_per_grain_k = stub.f_cropresidue_production().values
    biomass_reduction_propn_kp6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_yield_reduction_kp6z'], numpy=True, axis=-1)
    # proportion_grain_harv_k = pinp.stubble['proportion_grain_harv']
    consumption_factor_p6z = zfun.f_seasonal_inp(pinp.cropgraze['i_cropgraze_consumption_factor_zp6'],numpy=True,axis=0).T

    # ##adjust seeding penalty - crops that are not harvested e.g. fodder don't have yield penalty. But do have a stubble penalty
    # ###if calculating yield penalty for stubble then include all crop (e.g. include fodders)
    # stub_yield_reduction_propn_kp6z = yield_reduction_propn_kp6z
    # ###if calculating yield penalty for grain transfer then only include harvested crops (e.g. don't include fodders)
    # yield_reduction_propn_kp6z = yield_reduction_propn_kp6z * (proportion_grain_harv_k>0)[:,na,na]

    # ##calc stubble reduction (kg of stubble per kg of crop DM consumed)
    # stubble_reduction_propn_kp6z = stub_yield_reduction_propn_kp6z * stubble_per_grain_k[:,na,na]

    ##convert from yield penalty to biomass penalty - required because input is grain yield reduction per tonne of crop consumed
    harvest_index_k = pinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because yield penalty is inputted as the harvestable grain
    biomass_reduction_propn_kp6z = biomass_reduction_propn_kp6z / harvest_index_k[:,na,na]

    ##apply season mask and grazing exists mask
    ###calc mask if crop can be grazed - doesn't need to include p5 since no p5 set in the constraint.
    grazing_exists_p6z = (consumption_factor_p6z > 0) * 1
    ###calc season mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(per.f_feed_periods()[:-1], z_pos=-1, mask=True)
    ###apply masks
    biomass_reduction_propn_kp6z = biomass_reduction_propn_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z
    # stubble_reduction_propn_kp6z = stubble_reduction_propn_kp6z * mask_fp_z8var_p6z * grazing_exists_p6z

    ##store report vals
    fun.f1_make_r_val(r_vals,biomass_reduction_propn_kp6z,'crop_grazing_biomass_penalty_kp6z',mask_fp_z8var_p6z,z_pos=-1)


    return biomass_reduction_propn_kp6z #, stubble_reduction_propn_kp6z


def f1_cropgraze_params(params, r_vals, nv):
    # grazecrop_area_rkl = f_graze_crop_area()
    crop_DM_provided_kp6p5z8lz9, crop_DM_required_kp6p5z, transfer_exists_p6p5z = f_cropgraze_DM(r_vals=r_vals)
    biomass_reduction_propn_kp6z = f_cropgraze_biomass_penalty(r_vals)
    crop_md_fkp6p5zl, crop_vol_fkp6p5zl = crop_md_vol(nv, r_vals)
    co2e_cropgraze_kp6z = f_cropgraze_emissions(r_vals)
    # DM_reduction_kp6p5zl = f_DM_reduction_seeding_time()

    ##keys
    keys_l = pinp.general['i_lmu_idx']
    keys_k = sinp.landuse['C']
    keys_p6 = pinp.period['i_fp_idx']
    keys_p5 = np.asarray(per.f_p_dates_df().index[:-1]).astype('str')
    keys_f = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    keys_z = zfun.f_keys_z()

    ##array indexes
    ###DM prov
    arrays_kp6p5z8lz9 = [keys_k, keys_p6, keys_p5, keys_z, keys_l, keys_z]
    ###DM req
    arrays_kp6p5z = [keys_k, keys_p6, keys_p5, keys_z]
    ###yield & stub penalty
    arrays_kp6z = [keys_k, keys_p6, keys_z]
    ###transfer
    arrays_p6p5z = [keys_p6, keys_p5, keys_z]
    ###mei & pi
    arrays_fkp6p5zl = [keys_f, keys_k, keys_p6, keys_p5, keys_z, keys_l]

    ##create params
    params['crop_DM_provided_kp6p5z8lz9'] = fun.f1_make_pyomo_dict(crop_DM_provided_kp6p5z8lz9, arrays_kp6p5z8lz9)
    params['crop_DM_required_kp6p5z'] = fun.f1_make_pyomo_dict(crop_DM_required_kp6p5z, arrays_kp6p5z)
    params['transfer_exists_p6p5z'] = fun.f1_make_pyomo_dict(transfer_exists_p6p5z, arrays_p6p5z)
    params['biomass_reduction_propn_kp6z'] = fun.f1_make_pyomo_dict(biomass_reduction_propn_kp6z, arrays_kp6z)
    # params['stubble_reduction_propn_kp6z'] = fun.f1_make_pyomo_dict(stubble_reduction_propn_kp6z, arrays_kp6z)
    params['crop_md_fkp6p5zl'] = fun.f1_make_pyomo_dict(crop_md_fkp6p5zl, arrays_fkp6p5zl)
    params['crop_vol_kp6p5zl'] = fun.f1_make_pyomo_dict(crop_vol_fkp6p5zl, arrays_fkp6p5zl)
    params['co2e_cropgraze_kp6z'] = fun.f1_make_pyomo_dict(co2e_cropgraze_kp6z, arrays_kp6z)

