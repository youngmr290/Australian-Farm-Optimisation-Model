"""
author: young

At the end of the growing season AFO has the option of harvesting or baling each crop, which leaves
stubble for stock consumption, or crops can be left standing for fodder grazing. Stubble
and fodder are modelled in the same ways, as follows.

Stubble and fodder are a key feed source for sheep during the summer months. In general, sheep graze crop residues
selectively, preferring the higher quality components.  Thus, they tend to eat grain first, followed
by leaf and finally stem. To allow the optimisation of the quantity of the stubble grazed and to reflect selective
grazing the total crop residues are divided into ten categories. The higher categories are better
quality but generally lower quantity. Consumption of a higher quality category allows the consumption of a lower
category (e.g. sheep can not consume any of category B until some of category A has been consumed).

The total mass of crop residues at first
grazing (harvest for stubble and an inputted date for fodder) is calculated as a product of the biomass,
harvest index and proportion harvested (see f_biomass2residue). Over time if the feed is not consumed it
deteriorates in quality and quantity due to adverse effects of weather and the impact of sheep trampling.

Residue production can be positively impacted by frost because frost during the plants flowering stage
can damage cell tissue and reduce grain fill :cite:p:`RN144`. This results in less grain and more residue
due to not using energy resources to fill grain. Thus, the harvest index used to calculate biomass to residue
is adjusted by a frost factor. The frost factor can be customised for each
crop which is required because different crops flower at different times, changing the impact and probability of
frost biomass reduction. Frost factor can be customised for each LMU because frost effects can be altered by
the LMU topography and soil type. For example, sandy soils are more affected by frost because the lower
moisture holding capacity reduces the heat buffering from the soil.

To represent crop residues in AFO requires the proportion of total residue in each category and the DMD (quality)
of each category. The DMD of each category is an input which a proportion of the total residue is allocated to.
The proportion in each category was determined using AFO's residue simulator which leverages the AFO
stock generator (documented in a future section) in combination with trial liveweight data (Riggall 2017 pers comm).
Using AFO's stock generator, animals that reflect those in the paddock trial were simulated on large range of diet
qualities and daily intake, and liveweight change was determined. The liveweight change of the simulated animals
was compared with the actual liveweight change in the paddock trial to determine the daily feed quality.
Based on the number of
sheep, the sheep intake, and the total crop residue available in the trial, the proportion of residue in each category
was calculated.

The energy provided from consuming each crop residue category is calculated from DMD. Like pasture, crop residue
FOO is expressed in units of dry matter (excluding moisture), therefore feed energy is expressed as M/D
(does not require dry matter content conversion). The volume of each crop residue category is calculated
based on both the quality and availability of the feed.

Farmer often rake and burn crop residue in preparation for the following seeding. This is represented as a
cost see Phase.py for further information.

Stubble grazing optimisation in AFO includes:

    - The time to start grazing of each stubble
    - The class of stock that grazes the stubble
    - The duration of grazing
    - The amount of supplementary required in addition to stubble (to meet alternative LW profiles)

Stubble definitions:

    - Total Grain = HI * (above ground) biomass
    - Leaf + Stem = (1-HI) * biomass
    - Harvested grain = (1 - spilt%) * Total grain
    - Spilt grain = spilt% * Total grain
    - Stubble = Leaf + Stem + Spilt grain
    - Spilt grain as a proportion of the stubble = (HI * spilt %) / (1 - HI(1 - spilt%))
"""
#python modules
import numpy as np
import os.path
import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')

#AFO modules
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import FeedsupplyFunctions as fsfun
from . import EmissionFunctions as efun
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Sensitivity as SA
from . import Periods as per

na = np.newaxis

# def f_cropresidue_production():
#     '''
#     Stubble produced per kg of total grain (kgs of dry matter).
#
#     This is a separate function because it is used in CropGrazing.py and Mach.py to calculate stubble penalties.
#     '''
#     stubble_prod_data = 1 / pinp.stubble['i_harvest_index_ks2'][:,0] - 1 * pinp.stubble['i_propn_grain_harv_ks2'][:,0]  # subtract 1*harv propn to account for the tonne of grain that was harvested and doesn't become stubble.
#     stubble = pd.Series(data=stubble_prod_data, index=sinp.landuse['C'])
#     return stubble

def f_biomass2residue(residuesim=False):
    '''
    Residue produced (Stubble or standing fodder) per kg of biomass .

    This is a separate function because it is used in residue simulator.
    '''
    ##inputs
    harvest_index_ks2 = pinp.stubble['i_harvest_index_ks2']
    biomass_scalar_ks2 = pinp.stubble['i_biomass_scalar_ks2']
    propn_grain_harv_ks2 = pinp.stubble['i_propn_grain_harv_ks2']
    lmu_mask = pinp.general['i_lmu_area'] > 0
    frost_kl = pinp.crop['frost'].values[:,lmu_mask]

    ##calc biomass to product scalar
    ##if this is being calculated for sim then don't want to include frost (because don't want lmu axis and the frost input in AFO doesn't reflect the trial).
    ##the assumption is that a frosted crop will not be used in the stubble trial.
    if residuesim:
        biomass2residue_ks2 = (1 - harvest_index_ks2 * propn_grain_harv_ks2) * biomass_scalar_ks2
        return biomass2residue_ks2

    ##calc biomass to product scalar - adjusted for frost
    frost_harv_factor_kl = (1-frost_kl)
    harvest_index_kls2 = harvest_index_ks2[:,na,:] * frost_harv_factor_kl[:,:,na]
    biomass2residue_kls2 = (1 - harvest_index_kls2 * propn_grain_harv_ks2[:,na,:]) * biomass_scalar_ks2[:,na,:]

    return biomass2residue_kls2


def crop_residue_all(params, r_vals, nv, cat_propn_s1_ks2):
    '''
    Calculates the crop residue available, MD provided, volume required and the proportion of the way through
    the feed period that crop residue becomes available.


    '''
    ##general
    len_p6 = len(per.f_feed_periods()) - 1
    len_nv = nv['len_nv']
    index_p6 = np.arange(len_p6)

    ##nv stuff
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.
    me_threshold_fp6z = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)
    stub_me_eff_gainlose = pinp.stubble['i_stub_me_eff_gainlose']

    ##create mask which is stubble available. Stubble is available from the period harvest starts to the beginning of the following growing season.
    ##if the end date of the fp is after harvest then stubble is available.
    feed_period_dates_p6z = per.f_feed_periods()
    fp_end_p6z = feed_period_dates_p6z[1:]
    fp_start_p6z = feed_period_dates_p6z[:-1]
    harv_date_zk = zfun.f_seasonal_inp(pinp.crop['start_harvest_crops'].values, numpy=True, axis=1).swapaxes(0,1)
    period_is_harvest_p6zk = np.logical_and(fp_end_p6z[...,na] >= harv_date_zk, fp_start_p6z[...,na] <= harv_date_zk)
    idx_fp_start_stub_zk = fun.searchsort_multiple_dim(feed_period_dates_p6z, harv_date_zk, 1, 0, side='right') - 1

    idx_fp_end_stub_z = zfun.f_seasonal_inp(pinp.stubble['i_fp_end_stub_z'], numpy=True, axis=0)

    mask_stubble_exists_p6zk = np.logical_or(np.logical_and(index_p6[:,na,na]>=idx_fp_start_stub_zk, index_p6[:,na,na]<=idx_fp_end_stub_z[:,na]),
                                             np.logical_and(idx_fp_end_stub_z[:,na] < idx_fp_start_stub_zk,
                                                            np.logical_or(index_p6[:,na,na]>=idx_fp_start_stub_zk, index_p6[:,na,na]<=idx_fp_end_stub_z[:,na])))


    # #############################
    # # Total stubble production  #
    # #############################
    # ##calc yield - frost and seeding rate not accounted for because they don't effect stubble.
    # rot_yields_rkl_p7z = phs.f_rot_yield(for_stub=True)
    # ##calc stubble
    # residue_per_grain_k = f_cropresidue_production()
    # rot_stubble_rkl_p7z = rot_yields_rkl_p7z.mul(residue_per_grain_k, axis=0, level=1)

    #########################
    # deterioration         #
    #########################
    ##days since harvest (calculated from the end date of each fp)
    days_since_harv_p6zk = fp_end_p6z[...,na] - harv_date_zk
    days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] = days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] + 364  #add 364 to the periods at the start of the year because as far as stubble goes they are after harvest
    average_days_since_harv_p6zk = days_since_harv_p6zk - np.minimum(days_since_harv_p6zk, (fp_end_p6z - fp_start_p6z)[...,na])/2 #subtract half the length of current period to get the average days since harv. Minimum is to handle the period when harvest occurs.
    average_days_since_harv_p6zk = average_days_since_harv_p6zk.astype(float)

    ##calc the quantity decline % for each period - used in transfer constraints, need to average the number of days in the period of interest
    quant_declined_p6zk = (1 - pinp.stubble['quantity_decay']) ** average_days_since_harv_p6zk.astype(float)

    ##calc the quality decline % for each period
    ###quality of each category is inputted at harvest.
    qual_declined_p6zk = (1 - pinp.stubble['quality_deterioration']) ** average_days_since_harv_p6zk.astype(float)

    ###############
    # M/D & vol   #
    ###############
    '''
    This section creates a df that contains the M/D for each stubble category for each crop and 
    the equivalent for vol. This is used by live stock.
    
    1) passed in: stubble component composition, calculated by the sim (stored in an excel file)
    2) converts total dmd to dmd of category 
    3) calcs ri quantity and availability 
    4) calcs the md of each stubble category (dmd to MD)
    
    '''
    len_k = len(sinp.landuse['C'])
    len_s2 = len(pinp.stubble['i_idx_s2'])
    len_s1 = len(pinp.stubble['i_stub_cat_dmd_s1'])
    cat_propn_ks1s2 = cat_propn_s1_ks2.values.reshape(len_s1,len_k,len_s2).swapaxes(0,1)


    ##quality of each category in each period
    ###scale dmd at harvest to each period.
    stub_cat_qual_s1 = pinp.stubble['i_stub_cat_dmd_s1']
    dmd_cat_p6zks1 = stub_cat_qual_s1 * qual_declined_p6zk[...,na]

    ##crude protein of each category in each period
    ###scale cp at harvest to each period. Reduces at the same rate as DMD as per MIDAS.
    stub_cat_cp_s1 = pinp.stubble['i_stub_cat_cp_s1']
    cp_cat_p6zks1 = stub_cat_cp_s1 * qual_declined_p6zk[...,na]

    ##calc relative quality before converting dmd to md - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        ri_quality_p6zks1 = fsfun.f_rq_cs(dmd_cat_p6zks1, pinp.stubble['clover_propn_in_sward_stubble'])

    # ##ri availability (not calced anymore - stubble uses ra=1 now) - first calc stubble foo (stub available) this is the average from all rotations and lmus because we just need one value for foo (crop residue volume is assumed to be the same across lmu - the extra detail could be added)
    # ###try calc the base yield for each crop but if the crop is not one of the rotation phases then assign the average foo (this is only to stop error. it doesn't matter because the crop doesn't exist so the stubble is never used)
    # base_yields = rot_yields_rkl_p7z.droplevel(0, axis=0).groupby(axis=1, level=1).sum() #drop rotation index and sum p7 axis (just want total yield to calc pi)
    # base_yields = base_yields.replace(0,np.NaN) #replace 0 with nan so if yield inputs are missing (e.g. set to 0) the foo is still correct (nan gets skipped in pd.mean)
    # stub_foo_harv_zk = np.zeros((n_seasons, n_crops))
    # for crop, crop_idx in zip(pinp.stubble['i_stub_landuse_idx'], range(n_crops)):
    #     try:
    #         stub_foo_harv_zk[:, crop_idx] = base_yields.loc[crop].mean(axis=0) * residue_per_grain_k.loc[crop]
    #     except KeyError: #if the crop is not in any of the rotations assign average foo to stop error - this is not used so could assign any value.
    #         stub_foo_harv_zk[:,crop_idx] = base_yields.mean(axis=0) * residue_per_grain_k.mean()
    # stub_foo_harv_zk = np.nan_to_num(stub_foo_harv_zk) #replace nan with 0 (only wanted nan for the mean)
    ###adjust the foo for each category because the good stuff is eaten first therefore there is less foo when the sheep start eating the poorer stubble
    # cat_propn_rolled_ks1 = np.roll(cat_propn_ks1, shift=1, axis=1) #roll along the cat axis. So that the previous cat lines up with the current cat
    # cat_propn_rolled_ks1[:, 0] = 0 #set the first slice to 0 because no stubble is consumed before cat A is consumed e.g. there is 100% of foo available when sheep are consuming cat A
    # cat_cum_propn_ks1 = np.cumsum(cat_propn_rolled_ks1, axis=1) #cumulative sum of the component sizes.
    # stubble_foo_zks1 = stub_foo_harv_zk[..., na] *  (1 - cat_cum_propn_ks1)
    ###adjust for quantity delcine due to deterioration
    # stubble_foo_p6zks1 = stubble_foo_zks1 * quant_declined_p6zk[..., na]
    ###ri availability
    # hf = fsfun.f_hf(pinp.stubble['i_hr'])  # height factor
    # if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used - note that the equation system used is the one selected for dams in p1
    #     ri_availability_p6zks1 = fsfun.f_ra_cs(stubble_foo_p6zks1, hf)
    # elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used - note that the equation system used is the one selected for dams in p1
    #     ri_availability_p6zks1 = fsfun.f_ra_mu(stubble_foo_p6zks1, hf)

    ##combine ri quality and ri availability to calc overall vol (potential intake) - use ra=1 for stubble (same as stubble sim)
    ri_p6zks1 = fsfun.f_rel_intake(1, ri_quality_p6zks1, pinp.stubble['clover_propn_in_sward_stubble'])
    vol_p6zks1 = (1000 / ri_p6zks1) / (1 + SA.sap['pi'])
    vol_p6zks1 = vol_p6zks1 * mask_stubble_exists_p6zk[..., na] #stop md being provided if stubble doesn't exist
    vol_fp6zks1 = vol_p6zks1 * nv_is_not_confinement_f[:,na,na,na,na] #me from stubble is 0 in the confinement pool

    ##convert dmd to M/D
    ## Stubble doesn't include calculation of effective mei because stubble is generally low quality feed with a wide variation in quality within the sward.
    ## Therefore, there is scope to alter average diet quality by altering the grazing time and the proportion of the stubble consumed.
    md_p6zks1 = np.clip(fsfun.f1_dmd_to_md(dmd_cat_p6zks1), 0, np.inf)
    md_p6zks1 = md_p6zks1 * mask_stubble_exists_p6zk[...,na] #stop md being provided if stubble doesn't exist
    ##reduce me if nv is higher than livestock diet requirement.
    md_fp6zks1 = fsfun.f_effective_mei(1000, md_p6zks1, me_threshold_fp6z[...,na,na]
                                       , nv['confinement_inc'], ri_p6zks1, stub_me_eff_gainlose)

    md_fp6zks1 = md_fp6zks1 * nv_is_not_confinement_f[:,na,na,na,na] #me from stubble is 0 in the confinement pool

    ###########################
    # emissions               #
    ###########################
    ##livestock methane emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        ch4_stub_p6zks1 = efun.f_ch4_feed_nir(1000, dmd_cat_p6zks1)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        ch4_stub_p6zks1 = efun.f_ch4_feed_bc()

    ##livestock nitrous oxide emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        n2o_stub_p6zks1 = efun.f_n2o_feed_nir(1000, dmd_cat_p6zks1, cp_cat_p6zks1)

    co2e_stub_p6zks1 = ch4_stub_p6zks1 * uinp.emissions['i_ch4_gwp_factor'] + n2o_stub_p6zks1 * uinp.emissions['i_n2o_gwp_factor']

    ###########
    #trampling#
    ###########
    #for now this is just a single number however the input could be changed to per period
    tramp_effect_ks1s2 = pinp.stubble['trampling'][:,na,na] * cat_propn_ks1s2 #mul by cat propn because only want to include the trampling of the categry being consumed.

    ################################
    # allow access to next category#
    ################################

    ##Note: In the sim each category has a minimum of 1kg so that the following transfers always work.

    ##quantity of cat A stubble provided from 1t of total stubble at harvest
    cat_a_prov_p6zks1s2 = 1000 * cat_propn_ks1s2 * np.logical_and(np.arange(len(pinp.stubble['i_stub_cat_idx']))[:,na]==0
                                                      ,period_is_harvest_p6zk[...,na,na]) #Only cat A is provided at harvest

    ##amount of available stubble required to consume 1t of each cat in each fp
    stub_req_ks1s2 = 1000*(1+tramp_effect_ks1s2)

    ##amount of next category provide by consumption of current category.
    stub_prov_ks1s2 = np.roll(cat_propn_ks1s2, shift=-1,axis=1)/cat_propn_ks1s2*1000
    stub_prov_ks1s2[:,-1,:] = 0 #final cat doesn't provide anything


    ##############################
    #transfers between periods   #
    ##############################
    ##transfer a given cat to the next period. Only cat A is available at harvest - it comes from the rotation phase.
    stub_transfer_prov_p6zk = 1000 * np.roll(quant_declined_p6zk, shift=-1, axis=0)/quant_declined_p6zk #divide to capture only the decay during the curent period (quant_decline is the decay since harv)
    stub_transfer_prov_p6zk = stub_transfer_prov_p6zk * mask_stubble_exists_p6zk  #no transfer can occur when stubble doesn't exist
    stub_transfer_prov_p6zk = stub_transfer_prov_p6zk * np.roll(np.logical_not(period_is_harvest_p6zk), -1, 0) #last yrs stubble doesn't transfer past the following harv.

    ##transfer requirment - mask out harvest period because last years stubble can not be consumed after this years harvest.
    stub_transfer_req_p6zk = 1000 * mask_stubble_exists_p6zk   # No transfer can occur when stubble doesn't exist or at harvest.

    ###############
    #harvest p con# stop sheep consuming more than possible because harvest is not at the start of the period
    ###############
    #how far through each period does harv start? note: 0 for each period harv doesn't start in. Used to calc stub consumption limit in harv period
    fp_len_p6z = fp_end_p6z - fp_start_p6z
    cons_propn_p6zk = np.clip(fun.f_divide(fp_len_p6z[...,na] - (fp_end_p6z[...,na] - harv_date_zk), fp_len_p6z[...,na]),0, np.inf)
    cons_propn_p6zk[cons_propn_p6zk>=1] = 0 #cons_prop can not be 1 else div0 error in pyomo.

    ######################
    #apply season mask   #
    ######################
    ##mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(fp_start_p6z, z_pos=-1, mask=True)
    mask_fp_z8var_zp6 = mask_fp_z8var_p6z.T

    ##apply mask
    cons_propn_p6zk = cons_propn_p6zk * mask_fp_z8var_p6z[...,na]
    stub_transfer_prov_p6zk = stub_transfer_prov_p6zk * mask_fp_z8var_p6z[...,na]
    stub_transfer_req_p6zk = stub_transfer_req_p6zk * mask_fp_z8var_p6z[...,na]
    cat_a_prov_p6zks1s2 = cat_a_prov_p6zks1s2 * mask_fp_z8var_p6z[...,na,na,na]
    md_fp6zks1 = md_fp6zks1 * mask_fp_z8var_p6z[...,na,na]
    vol_fp6zks1 = vol_fp6zks1 * mask_fp_z8var_p6z[...,na,na]
    ch4_stub_p6zks1 = ch4_stub_p6zks1 * mask_fp_z8var_p6z[...,na,na]
    n2o_stub_p6zks1 = n2o_stub_p6zks1 * mask_fp_z8var_p6z[...,na,na]
    co2e_stub_p6zks1 = co2e_stub_p6zks1 * mask_fp_z8var_p6z[...,na,na]

    #########
    ##keys  #
    #########
    ##keys
    keys_k = sinp.landuse['C']
    keys_p6 = pinp.period['i_fp_idx']
    keys_s1 = pinp.stubble['i_stub_cat_idx']
    keys_s2 = pinp.stubble['i_idx_s2']
    keys_f  = np.array(['nv{0}' .format(i) for i in range(len_nv)])
    keys_z = zfun.f_keys_z()
    lmu_mask = pinp.general['i_lmu_area'] > 0
    keys_l = pinp.general['i_lmu_idx'][lmu_mask]

    ##array indexes
    ###stub transfer (cat b & c)
    arrays_ks1s2 = [keys_k, keys_s1, keys_s2]
    ###category A req
    arrays_p6zks1s2 = [keys_p6, keys_z, keys_k, keys_s1, keys_s2]
    ###md & vol
    arrays_fp6zks1 = [keys_f, keys_p6, keys_z, keys_k, keys_s1]
    ###emissions
    arrays_p6zks1 = [keys_p6, keys_z, keys_k, keys_s1]
    ###harv con & feed period transfer
    arrays_p6zk = [keys_p6, keys_z, keys_k]
    ###biomass to residue
    arrays_kls2 = [keys_k, keys_l, keys_s2]

    ################
    ##pyomo params #
    ################

    # ##stubble produced per tonne of grain yield - this is df so don't need to build index.
    # params['rot_stubble'] = rot_stubble_rkl_p7z.stack([0,1]).to_dict()

    ##'require' params ie consuming 1t of stubble B requires 1.002t from the constraint (0.002 accounts for trampling)
    params['transfer_req'] = fun.f1_make_pyomo_dict(stub_req_ks1s2, arrays_ks1s2)

    ###'provide' from cat to cat ie consuming 1t of cat A provides 2t of cat b
    params['transfer_prov'] = fun.f1_make_pyomo_dict(stub_prov_ks1s2, arrays_ks1s2)

    ###harv con
    params['cons_prop'] = fun.f1_make_pyomo_dict(cons_propn_p6zk, arrays_p6zk)

    ###feed period transfer
    params['stub_transfer_prov'] = fun.f1_make_pyomo_dict(stub_transfer_prov_p6zk, arrays_p6zk)
    params['stub_transfer_req'] = fun.f1_make_pyomo_dict(stub_transfer_req_p6zk, arrays_p6zk)

    ###category A transfer param
    params['cat_a_prov'] = fun.f1_make_pyomo_dict(cat_a_prov_p6zks1s2, arrays_p6zks1s2)

    ###category A transfer param
    biomass2residue_kls2 = f_biomass2residue()
    params['biomass2residue_kls2'] = fun.f1_make_pyomo_dict(biomass2residue_kls2, arrays_kls2)

    ##md
    params['md'] = fun.f1_make_pyomo_dict(md_fp6zks1, arrays_fp6zks1)

    ##vol
    params['vol'] = fun.f1_make_pyomo_dict(vol_fp6zks1, arrays_fp6zks1)

    ##emissions
    params['co2e_stub_p6zks1'] = fun.f1_make_pyomo_dict(co2e_stub_p6zks1, arrays_p6zks1)

    ###########
    #report   #
    ###########
    ##keys
    fun.f1_make_r_val(r_vals,keys_k,'keys_k1')
    fun.f1_make_r_val(r_vals,keys_s1,'keys_s1')
    fun.f1_make_r_val(r_vals,keys_s2,'keys_s2')

    ##store report vals
    fun.f1_make_r_val(r_vals,np.moveaxis(np.moveaxis(md_fp6zks1, 0, 2), 0, 1),'md_zp6fks1',mask_fp_z8var_zp6[:,:,na,na,na],z_pos=-5)
    fun.f1_make_r_val(r_vals,np.moveaxis(ch4_stub_p6zks1, 0, 1),'ch4_stub_zp6ks1',mask_fp_z8var_zp6[:,:,na,na],z_pos=-4)
    fun.f1_make_r_val(r_vals,np.moveaxis(n2o_stub_p6zks1, 0, 1),'n2o_stub_zp6ks1',mask_fp_z8var_zp6[:,:,na,na],z_pos=-4)


