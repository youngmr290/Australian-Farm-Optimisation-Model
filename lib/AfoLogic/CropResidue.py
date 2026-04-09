"""
author: young

At the end of the growing season AFO has the option of harvesting or baling each crop, which leaves
stubble for stock consumption, or crops can be left standing for fodder grazing. Stubble
and fodder are modelled in the same ways, as follows.

Stubble and fodder are a key feed source for sheep during the summer months. In general, sheep graze crop residues
selectively, preferring the higher quality components. They tend to eat grain first, followed by leaf and finally
stem. To allow the optimisation of the proportion of the stubble grazed - which is optimising the trade-off
between quantity and quality of the feed consumed - the total crop residues are divided into ten
categories (A, B, C ... J). Each category is a defined quality, which is the same across all crop
types. The higher categories are better quality but generally lower quantity. Lower quality categories
can’t be consumed until some higher quality categories have been consumed. This reflects the reality
of selective grazing and ensures the model can’t defer all the high quality stubble categories until
late in the autumn while grazing lower quality categories earlier in summer. If feed in a given
category is not consumed then it deteriorates in quality and quantity due to adverse effects of
weather and the impact of sheep trampling.

The total mass of crop residues at first grazing (harvest for stubble and an inputted date for
fodder) is calculated as a product of the biomass, harvest index and proportion harvested
(see f_biomass2residue below). The total mass of the residue is allocated between the
categories using the AFO’s residue simulator which leverages the AFO stock
generator (see :ref:`livestock_ref` for more information).

The proportion of total residue in each category is estimated with the :ref:`cropresiduesim` using
trial data of sequential animal liveweights grazing crop residue of a crop with known yield. See :ref:`cropresiduesim`
for more details.

The energy provided from consuming each crop residue category is calculated from DMD. Like pasture, crop residue
FOO is expressed in units of dry matter (excluding moisture), therefore feed energy is expressed as M/D
(does not require dry matter content conversion). The volume of each crop residue category is calculated
based on both the quality and availability of the feed.

Work with farmers suggested that hay residue is treated just as any other crop residue. However,
there are small differences between hay and a normally harvest crop. Hay is cut earlier
therefore there is no split or spilt grain but there is some higher quality regrowth.
The model captures the difference in quality through the :ref:`cropresiduesim` and the impact on residue volume
is handled through adjusting the harvest index and proportion harvested inputs.

Farmers often rake and burn crop residue in preparation for the following seeding. This is represented as a
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
#     stubble_prod_data = 1 / uinp.stubble['i_harvest_index_ks2'][:,0] - 1 * uinp.stubble['i_propn_grain_harv_ks2'][:,0]  # subtract 1*harv propn to account for the tonne of grain that was harvested and doesn't become stubble.
#     stubble = pd.Series(data=stubble_prod_data, index=sinp.general['i_idx_k1'])
#     return stubble

def f_biomass2residue():
    '''
    Residue produced (Stubble or standing fodder) per kg of biomass.

    The total mass of crop residues at first grazing (harvest for stubble and an inputted date for fodder) is
    calculated as a product of the biomass, harvest index and proportion harvested.

    .. note:: Residue production can be positively impacted by frost because frost during the plants flowering stage
              can damage cell tissue and reduce grain fill :cite:p:`RN144`. This results in less grain and more residue
              due to not using energy resources to fill grain. However, this is not currently captured in the
              calculation of residue as it is challenging to represent and isn't a significant error.

    This is a separate function because it is used in residue simulator.
    '''
    ##inputs
    harvest_index_ks2 = uinp.stubble['i_harvest_index_ks2']
    biomass_scalar_ks2 = uinp.stubble['i_biomass_scalar_ks2']
    propn_grain_harv_ks2 = uinp.stubble['i_propn_grain_harv_ks2']

    ##calc biomass to product scalar
    biomass2residue_ks2 = (1 - harvest_index_ks2 * propn_grain_harv_ks2) * biomass_scalar_ks2
    return biomass2residue_ks2


def crop_residue_all(params, r_vals, nv, cat_propn_s1_ks2):
    '''
    Calculates the crop residue available, MD provided, volume required and the proportion of the way through
    the feed period that crop residue becomes available.


    '''
    # -------------------------
    # General setup
    # -------------------------
    ##feed period
    feed_period_dates_p6z = per.f_feed_periods()
    fp_end_p6z = feed_period_dates_p6z[1:]
    fp_start_p6z = feed_period_dates_p6z[:-1]
    len_p6 = len(feed_period_dates_p6z) - 1
    index_p6 = np.arange(len_p6)

    ##nv stuff
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.
    me_threshold_fp6z = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)
    stub_me_eff_gainlose = uinp.stubble['i_stub_me_eff_gainlose']

    # -------------------------
    # (A) Domain masks early
    # -------------------------

    ##Within/between child-z masks for FEED PERIOD constraints
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p6z = fp_start_p6z==season_start_z
    _, _, mask_childz_within_fp_p6z, mask_childz_between_fp_p6z  = zfun.f_season_transfer_mask(
        fp_start_p6z, period_is_seasonstart_pz=period_is_seasonstart_p6z, z_pos=-1)

    ## Season mask — used everywhere
    mask_fp_z8var_p6z = mask_childz_within_fp_p6z | mask_childz_between_fp_p6z
    mask_fp_z8var_zp6 = mask_fp_z8var_p6z.T

    ##create mask which is stubble available. Stubble is available from the period harvest starts to the beginning of the following growing season.
    ##if the end date of the fp is after harvest then stubble is available.
    harv_date_zk = zfun.f_seasonal_inp(pinp.crop['start_harvest_crops'].values, numpy=True, axis=1).swapaxes(0,1)
    period_is_harvest_p6zk = np.logical_and(fp_end_p6z[...,na] >= harv_date_zk, fp_start_p6z[...,na] <= harv_date_zk)
    idx_fp_start_stub_zk = fun.searchsort_multiple_dim(feed_period_dates_p6z, harv_date_zk, 1, 0, side='right') - 1

    ##stubble is destocked X days before early break - you dont know the type of season in advanced therefore must destock all z before the early break just in case
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    destock_days_prior_brk = pinp.stubble['i_destock_stub']
    date_destocked = np.min(season_break_z) + 364 - destock_days_prior_brk #incremented to the next yr.
    in_period_p6z = (fp_start_p6z <= date_destocked) & (date_destocked < fp_end_p6z)
    idx_fp_end_stub_z = np.argmax(in_period_p6z, axis=0).astype(int)

    mask_stubble_exists_p6zk = np.logical_or(np.logical_and(index_p6[:,na,na]>=idx_fp_start_stub_zk, index_p6[:,na,na]<=idx_fp_end_stub_z[:,na]),
                                             np.logical_and(idx_fp_end_stub_z[:,na] < idx_fp_start_stub_zk,
                                                            np.logical_or(index_p6[:,na,na]>=idx_fp_start_stub_zk, index_p6[:,na,na]<=idx_fp_end_stub_z[:,na])))

    ##stub category transfer mask
    keys_s1 = uinp.stubble['i_stub_cat_idx']
    len_s1 = len(keys_s1)
    sc_index = np.arange(len_s1)
    mask_sc_has_next_s1 = sc_index < (len_s1-1) # A..I

    cat_is_A_s1 = (np.arange(len_s1) == 0)
    mask_cat_a_prov_p6zks1 = cat_is_A_s1 * period_is_harvest_p6zk[..., na]


    #########################
    # deterioration         #
    #########################
    ##days since harvest (calculated from the end date of each fp)
    days_since_harv_p6zk = fp_end_p6z[...,na] - harv_date_zk
    days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] = days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] + 364  #add 364 to the periods at the start of the year because as far as stubble goes they are after harvest
    average_days_since_harv_p6zk = days_since_harv_p6zk - np.minimum(days_since_harv_p6zk, (fp_end_p6z - fp_start_p6z)[...,na])/2 #subtract half the length of current period to get the average days since harv. Minimum is to handle the period when harvest occurs.
    average_days_since_harv_p6zk = average_days_since_harv_p6zk.astype(float)

    # todo better would be to deteriorate high categories less because more grain (if changed here need to change in cropresidue.py module)
    ##calc the quantity decline % for each period - used in transfer constraints, need to average the number of days in the period of interest
    quant_declined_since_harv_p6zk = (1 - uinp.stubble['quantity_decay']) ** average_days_since_harv_p6zk.astype(float)

    ##calc the quality decline % for each period
    ###quality of each category is inputted at harvest.
    qual_declined_p6zk = (1 - uinp.stubble['quality_deterioration']) ** average_days_since_harv_p6zk.astype(float)

    # -------------------------
    # Category proportions (k,s1,s2) (k already masked)
    # -------------------------
    len_s2 = len(uinp.stubble['i_idx_s2'])
    cat_propn_ks1s2 = cat_propn_s1_ks2.values.reshape(len_s1, -1, len_s2).swapaxes(0, 1)
    cat_propn_ks1s2 = cat_propn_ks1s2[pinp.crop_landuse_mask_k1, :, :]

    # -------------------------
    # Category transfer params + Trampling
    # -------------------------
    ##Note: In the sim each category has a minimum of 1kg so that the following transfers always work.

    ##quantity of cat A stubble provided from 1t of total stubble at harvest - only cat A is provided at harvest
    cat_a_prov_p6zks1s2 = 1000 * cat_propn_ks1s2 * mask_cat_a_prov_p6zks1[..., na]

    ##amount of available stubble required to consume 1t of each cat in each fp (includes trampling)
    #todo This trampling could be improved by spreading the quantity trampled across the lower quality categories
    tramp_effect_ks1s2 = uinp.stubble['trampling'][:,na,na] * cat_propn_ks1s2 #mul by cat propn because only want to include the trampling of the categry being consumed.
    stub_req_ks1s2 = 1000*(1+tramp_effect_ks1s2)

    ##amount of next category provide by consumption of current category.
    stub_prov_ks1s2 = np.roll(cat_propn_ks1s2, shift=-1,axis=1)/cat_propn_ks1s2*1000
    stub_prov_ks1s2[:, ~mask_sc_has_next_s1, :] = 0 #final cat doesn't provide anything

    ##############################
    #transfers between periods   #
    ##############################
    ##transfer a given cat to the next period. Only cat A is available at harvest - it comes from the rotation phase.
    stub_transfer_prov_p6zk = 1000 * np.roll(quant_declined_since_harv_p6zk, shift=-1, axis=0)/quant_declined_since_harv_p6zk #divide to capture only the decay during the curent period (quant_decline is the decay since harv)
    stub_transfer_prov_p6zk = stub_transfer_prov_p6zk * mask_stubble_exists_p6zk * np.roll(np.logical_not(period_is_harvest_p6zk), -1, axis=0) # Backup to ensure that last years stubble doesnt transfer past next harvest

    ##transfer requirment
    stub_transfer_req_p6zk = 1000 * mask_stubble_exists_p6zk

    ###############
    #Proportion of the fp where stubble grazing can't occur - stop sheep consuming more than possible because harvest is not at the start of the period
    ###############
    fp_len_p6z = fp_end_p6z - fp_start_p6z

    ## --- (1) Pre-harvest block: stubble can't be grazed before harvest happens in that period ---
    # fraction of period BEFORE harvest (blocked)
    stub_preharvest_block_frac_p6zk  = np.clip(fun.f_divide(fp_len_p6z[...,na] - (fp_end_p6z[...,na] - harv_date_zk), fp_len_p6z[...,na]),0, 1)
    ### if harvest is not in this period, set to 0 (no pre-harvest block for stubble availability)
    stub_preharvest_block_frac_p6zk[stub_preharvest_block_frac_p6zk>=1] = 0

    # --- (2) Post-destock block: once sheep are destocked, remainder of period is not grazeable ---
    # date_destocked is a scalar (same for all z); numpy broadcasts across (p6,z)
    stub_postdestock_block_frac_p6z  = np.clip(fun.f_divide(fp_len_p6z - (fp_end_p6z - date_destocked), fp_len_p6z),0, 1)
    # Convert from "fraction of period before destock" to "fraction of period after destock (not available for grazing)"
    stub_postdestock_block_frac_p6z = 1 - stub_postdestock_block_frac_p6z
    # if destock after the period ends -> no block
    stub_postdestock_block_frac_p6z[date_destocked >= fp_end_p6z] = 0
    # if destock before the period starts -> no blocked (because stubble doesnt exist.
    stub_postdestock_block_frac_p6z[date_destocked <= fp_start_p6z] = 0

    ##combine
    stub_unavailable_frac_p6zk  = stub_postdestock_block_frac_p6z[:,:,na] + stub_preharvest_block_frac_p6zk
    stub_unavailable_frac_p6zk = np.clip(stub_unavailable_frac_p6zk, 0, 1)

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

    ##quality of each category in each period
    ###scale dmd at harvest to each period.
    stub_cat_qual_s1 = uinp.stubble['i_stub_cat_dmd_s1']
    dmd_cat_p6zks1 = stub_cat_qual_s1 * qual_declined_p6zk[...,na]

    ##crude protein of each category in each period
    ###scale cp at harvest to each period. Reduces at the same rate as DMD as per MIDAS.
    stub_cat_cp_s1 = uinp.stubble['i_stub_cat_cp_s1']
    cp_cat_p6zks1 = stub_cat_cp_s1 * qual_declined_p6zk[...,na]

    ##calc relative quality before converting dmd to md - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        ri_quality_p6zks1 = fsfun.f_rq_cs(dmd_cat_p6zks1, uinp.stubble['clover_propn_in_sward_stubble'])

    # ##ri availability (not calced anymore - stubble uses ra=1 now) - first calc stubble foo (stub available) this is the average from all rotations and lmus because we just need one value for foo (crop residue volume is assumed to be the same across lmu - the extra detail could be added)
    # ###try calc the base yield for each crop but if the crop is not one of the rotation phases then assign the average foo (this is only to stop error. it doesn't matter because the crop doesn't exist so the stubble is never used)
    # base_yields = rot_yields_rkl_p7z.droplevel(0, axis=0).groupby(axis=1, level=1).sum() #drop rotation index and sum p7 axis (just want total yield to calc pi)
    # base_yields = base_yields.replace(0,np.NaN) #replace 0 with nan so if yield inputs are missing (e.g. set to 0) the foo is still correct (nan gets skipped in pd.mean)
    # stub_foo_harv_zk = np.zeros((n_seasons, n_crops))
    # for crop, crop_idx in zip(uinp.stubble['i_stub_landuse_idx'], range(n_crops)):
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
    # stubble_foo_p6zks1 = stubble_foo_zks1 * quant_declined_since_harv_p6zk[..., na]
    ###ri availability
    # hf = fsfun.f_hf(uinp.stubble['i_hr'])  # height factor
    # if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used - note that the equation system used is the one selected for dams in p1
    #     ri_availability_p6zks1 = fsfun.f_ra_cs(stubble_foo_p6zks1, hf)
    # elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used - note that the equation system used is the one selected for dams in p1
    #     ri_availability_p6zks1 = fsfun.f_ra_mu(stubble_foo_p6zks1, hf)

    ##combine ri quality and ri availability to calc overall vol (potential intake) - use ra=1 for stubble (same as stubble sim)
    ri_p6zks1 = fsfun.f_rel_intake(1, ri_quality_p6zks1, uinp.stubble['clover_propn_in_sward_stubble'])
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
        stock_ch4_stub_p6zks1 = efun.f_stock_ch4_feed_nir(1000, dmd_cat_p6zks1)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        stock_ch4_stub_p6zks1 = efun.f_stock_ch4_feed_bc(1000, md_p6zks1)

    ##livestock nitrous oxide emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_stub_p6zks1 = efun.f_stock_n2o_feed_nir(1000, dmd_cat_p6zks1, cp_cat_p6zks1)

    ##nitrous oxide emissions from crop residue breakdown, leaching nad burning - linked to both production (+ve) and consumption (-ve) of 1t of stubble
    burn_date = pinp.emissions['i_burn_date'] + 364 * (pinp.emissions['i_burn_date'] < harv_date_zk)
    decay_harv_to_burn_zk = (1 - uinp.stubble['quantity_decay'])**(burn_date - harv_date_zk)
    decay_consumption_to_burn_p6zk = decay_harv_to_burn_zk / quant_declined_since_harv_p6zk

    ##emissions from 1t of stubble at harvest that is not grazed
    residue_n2o_stub_zk, leach_n2o_stub_zk, n2o_burning_zk, ch4_burning_zk = efun.f_crop_residue_n2o_nir(1000, pinp.emissions['i_burn_propn_k'], decay_harv_to_burn_zk)
    ##emissions reduction from grazing 1t of stubble at grazing time - use -1000 because consumption reduces emissions from burning, decay and leaching.
    residue_n2o_stub_cons_p6zk, leach_n2o_stub_cons_p6zk, n2o_burning_cons_p6zk, ch4_burning_cons_p6zk = efun.f_crop_residue_n2o_nir(-1000, pinp.emissions['i_burn_propn_k'], decay_consumption_to_burn_p6zk)

    ##co2e
    ###linked to stubble at harvest activity
    co2e_stub_production_zk = (ch4_burning_zk * uinp.emissions['i_ch4_gwp_factor']
                              + (residue_n2o_stub_zk + leach_n2o_stub_zk + n2o_burning_zk) * uinp.emissions['i_n2o_gwp_factor'])

    ###linked to consumption activity
    co2e_stub_cons_p6zks1 = ((stock_ch4_stub_p6zks1 + ch4_burning_cons_p6zk[...,na]) * uinp.emissions['i_ch4_gwp_factor']
                             + (stock_n2o_stub_p6zks1 + residue_n2o_stub_cons_p6zk[...,na] + leach_n2o_stub_cons_p6zk[...,na]
                             + n2o_burning_cons_p6zk[...,na]) * uinp.emissions['i_n2o_gwp_factor'])
    ##combine for r_val - maybe later these can be split and reported separately
    residue_harv_n2o_zk = residue_n2o_stub_zk + leach_n2o_stub_zk + n2o_burning_zk
    residue_cons_n2o_p6zk = residue_n2o_stub_cons_p6zk + leach_n2o_stub_cons_p6zk + n2o_burning_cons_p6zk
    residue_harv_ch4_zk = ch4_burning_zk
    residue_cons_ch4_p6zk = ch4_burning_cons_p6zk

    ## Apply existence mask for interpretability
    stock_ch4_stub_p6zks1 *= mask_stubble_exists_p6zk[..., na]
    stock_n2o_stub_p6zks1 *= mask_stubble_exists_p6zk[..., na]
    co2e_stub_cons_p6zks1  *= mask_stubble_exists_p6zk[..., na]
    residue_cons_n2o_p6zk  *= mask_stubble_exists_p6zk
    residue_cons_ch4_p6zk  *= mask_stubble_exists_p6zk


    ######################
    #apply season mask   #
    ######################
    stub_unavailable_frac_p6zk = stub_unavailable_frac_p6zk * mask_fp_z8var_p6z[...,na]
    stub_transfer_prov_p6zk = stub_transfer_prov_p6zk * mask_fp_z8var_p6z[...,na]
    stub_transfer_req_p6zk = stub_transfer_req_p6zk * mask_fp_z8var_p6z[...,na]
    cat_a_prov_p6zks1s2 = cat_a_prov_p6zks1s2 * mask_fp_z8var_p6z[...,na,na,na]
    md_fp6zks1 = md_fp6zks1 * mask_fp_z8var_p6z[...,na,na]
    vol_fp6zks1 = vol_fp6zks1 * mask_fp_z8var_p6z[...,na,na]
    stock_ch4_stub_p6zks1 = stock_ch4_stub_p6zks1 * mask_fp_z8var_p6z[...,na,na]
    stock_n2o_stub_p6zks1 = stock_n2o_stub_p6zks1 * mask_fp_z8var_p6z[...,na,na]
    residue_cons_n2o_p6zk = residue_cons_n2o_p6zk * mask_fp_z8var_p6z[...,na]
    residue_cons_ch4_p6zk = residue_cons_ch4_p6zk * mask_fp_z8var_p6z[...,na]
    co2e_stub_cons_p6zks1 = co2e_stub_cons_p6zks1 * mask_fp_z8var_p6z[...,na,na]

    #########
    ##keys  #
    #########
    ##keys
    keys_k = sinp.general['i_idx_k1']
    keys_p6 = pinp.period['i_fp_idx']
    keys_s1 = uinp.stubble['i_stub_cat_idx']
    keys_s2 = uinp.stubble['i_idx_s2']
    keys_f  = np.array(['nv{0}' .format(i) for i in range(len_nv)])
    keys_z = zfun.f_keys_z()
    keys_l = pinp.general['i_lmu_idx']

    ##array indexes
    ###stub transfer (cat b & c)
    arrays_ks1s2 = [keys_k, keys_s1, keys_s2]
    ###category A req
    arrays_p6zks1s2 = [keys_p6, keys_z, keys_k, keys_s1, keys_s2]
    ###md & vol
    arrays_fp6zks1 = [keys_f, keys_p6, keys_z, keys_k, keys_s1]
    ###emissions
    arrays_p6zks1 = [keys_p6, keys_z, keys_k, keys_s1]
    arrays_zk = [keys_z, keys_k]
    ###harv con & feed period transfer
    arrays_p6zk = [keys_p6, keys_z, keys_k]
    ###biomass to residue
    arrays_ks2 = [keys_k, keys_s2]

    ################
    ##pyomo params #
    ################

    # ##stubble produced per tonne of grain yield - this is df so don't need to build index.
    # params['rot_stubble'] = rot_stubble_rkl_p7z.stack([0,1]).to_dict()

    ##'require' params ie consuming 1t of stubble B requires 1.002t from the constraint (0.002 accounts for trampling)
    params['cat_transfer_req'] = fun.f1_make_pyomo_dict(stub_req_ks1s2, arrays_ks1s2)

    ###'provide' from cat to cat ie consuming 1t of cat A provides 2t of cat b
    params['cat_transfer_prov'] = fun.f1_make_pyomo_dict(stub_prov_ks1s2, arrays_ks1s2)

    ###harv con
    params['stub_unavailable_frac_p6zk'] = fun.f1_make_pyomo_dict(stub_unavailable_frac_p6zk, arrays_p6zk)

    ###feed period transfer
    params['stub_transfer_prov'] = fun.f1_make_pyomo_dict(stub_transfer_prov_p6zk, arrays_p6zk)
    params['stub_transfer_req'] = fun.f1_make_pyomo_dict(stub_transfer_req_p6zk, arrays_p6zk)

    ###category A transfer param
    params['cat_a_prov'] = fun.f1_make_pyomo_dict(cat_a_prov_p6zks1s2, arrays_p6zks1s2)

    ###category A transfer param
    biomass2residue_ks2 = f_biomass2residue()
    params['biomass2residue_ks2'] = fun.f1_make_pyomo_dict(biomass2residue_ks2, arrays_ks2)

    ##md
    params['md'] = fun.f1_make_pyomo_dict(md_fp6zks1, arrays_fp6zks1)

    ##vol
    params['vol'] = fun.f1_make_pyomo_dict(vol_fp6zks1, arrays_fp6zks1)

    ##emissions
    params['co2e_stub_cons_p6zks1'] = fun.f1_make_pyomo_dict(co2e_stub_cons_p6zks1, arrays_p6zks1)
    params['co2e_stub_production_zk'] = fun.f1_make_pyomo_dict(co2e_stub_production_zk, arrays_zk)

    ##--- build active index masks for sparse Pyomo sets - more notes in corresponding pyomo module ---
    base_mask_p6zk = mask_stubble_exists_p6zk & mask_fp_z8var_p6z[..., na]
    # transfer_mask_p6zk = mask_transfer_p6zk & mask_fp_z8var_p6z[...,na]

    axis_keys_p6zk = [keys_p6, keys_z, keys_k]
    axis_keys_p6zks1 = [keys_p6, keys_z, keys_k, keys_s1]
    params['idx_stub_base_p6zk'] = fun.build_active_index(base_mask_p6zk, axis_keys_p6zk)
    # params['idx_stub_transfer_p6zk'] = fun.build_active_index(transfer_mask_p6zk, axis_keys_p6zk)
    params['idx_stub_cat_A_prov_p6zks1'] = fun.build_active_index(mask_cat_a_prov_p6zks1, axis_keys_p6zks1)

    base_within_p6zk = base_mask_p6zk & mask_childz_within_fp_p6z[..., na]
    base_between_p6zk = base_mask_p6zk & mask_childz_between_fp_p6z[..., na]

    params['idx_stub_base_within_p6zk'] = fun.build_active_index(base_within_p6zk, [keys_p6, keys_z, keys_k])
    params['idx_stub_base_between_p6zk'] = fun.build_active_index(base_between_p6zk, [keys_p6, keys_z, keys_k])


    ###active z in each p6. see extra notes in corresponding pyomo module.
    stub_z8_by_p6k = {}
    stub_k_by_p6z = {}
    for (p6, z, k) in params['idx_stub_base_p6zk']:
        stub_z8_by_p6k.setdefault((p6, k), []).append(z)
        stub_k_by_p6z.setdefault((p6, z), set()).add(k)
    params['stub_z8_by_p6k'] = {pk: sorted(zs) for pk, zs in stub_z8_by_p6k.items()}
    params['stub_k_by_p6z'] = {pz: sorted(ks) for pz, ks in stub_k_by_p6z.items()}
    ###########
    #report   #
    ###########
    ##keys
    fun.f1_make_r_val(r_vals,keys_k,'keys_k1')
    fun.f1_make_r_val(r_vals,keys_s1,'keys_s1')
    fun.f1_make_r_val(r_vals,keys_s2,'keys_s2')

    ##store report vals
    fun.f1_make_r_val(r_vals,np.moveaxis(np.moveaxis(md_fp6zks1, 0, 2), 0, 1),'md_zp6fks1',mask_fp_z8var_zp6[:,:,na,na,na],z_pos=-5)
    fun.f1_make_r_val(r_vals,np.moveaxis(stock_ch4_stub_p6zks1, 0, 1),'stock_ch4_stub_zp6ks1',mask_fp_z8var_zp6[:,:,na,na],z_pos=-4)
    fun.f1_make_r_val(r_vals,np.moveaxis(stock_n2o_stub_p6zks1, 0, 1),'stock_n2o_stub_zp6ks1',mask_fp_z8var_zp6[:,:,na,na],z_pos=-4)
    fun.f1_make_r_val(r_vals,biomass2residue_ks2,'biomass2residue_ks2')
    fun.f1_make_r_val(r_vals,residue_harv_n2o_zk,'residue_harv_n2o_zk')
    fun.f1_make_r_val(r_vals,residue_harv_ch4_zk,'residue_harv_ch4_zk')
    fun.f1_make_r_val(r_vals,np.moveaxis(residue_cons_n2o_p6zk, 0, 1),'residue_cons_n2o_zp6k',mask_fp_z8var_zp6[:,:,na],z_pos=-3)
    fun.f1_make_r_val(r_vals,np.moveaxis(residue_cons_ch4_p6zk, 0, 1),'residue_cons_ch4_zp6k',mask_fp_z8var_zp6[:,:,na],z_pos=-3)


