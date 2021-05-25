# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:10:54 2019


Stubble:

Total Grain = HI * (above ground) biomass
Leaf + Stem = (1-HI) * biomass
Harvested grain = (1 - spilt%) * Total grain
Spilt grain = spilt% * Total grain
Stubble = Leaf + Stem + Spilt grain

Spilt grain as a proportion of the stubble = (HI * spilt %) / (1 - HI(1 - spilt%))

@author: young
"""
#python modules
import numpy as np

import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')

#AFO modules
import Functions as fun
import FeedsupplyFunctions as fsfun
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Crop as crp
import Sensitivity as SA
import Periods as per

na = np.newaxis

def stubble_all(params):
    '''
    Calculates the stubble available, MD provided, volume required and the proportion of the way through
    the harvest period that stubble becomes available.

    Stubble is a key feed source for sheep during the summer months. In general sheep graze stubbles
    selectively, preferring the higher quality components.  Thus, they tend to eat grain first followed
    by leaf and finally stem.  In AFO, total stubble is split into four categories (A, B, C & D) to
    reflect the selectivity of grazing.  For cereals, category A is mainly grain and leaf blade,
    category B is mainly leaf blade, leaf sheath and cocky chaff, category C is mainly cocky chaff and
    stem and category D is the remaining fraction that is not grazed. The total mass of stubble at harvest
    time is calculated as a product of the grain yield. Overtime if the stubble is not consumed it
    deteriorates in quality and quantity due to adverse effects of weather and the impact of sheep trampling.

    To represent stubble in AFO requires the proportion of each category, the DMD of each category and the
    proportion of each component in each category. The DMD of each component has been determined by other
    work :cite:p:`RN108` and can be used to determine the proportion of each component in each category based
    on the DMD of each category. The quantity and DMD of each stubble category were determined using the AFO
    sheep generator (documented in a future section). The DMD of the feed was altered until the liveweight of
    the sheep in the simulation matched the that of an equivalent sheep in a stubble trial (Riggall 2017 pers comm).
    This provided the feed DMD and the intake required to achieve the given liveweight. Based on the number of
    sheep and the total stubble available in the trial the simulation results can be extrapolated to provide
    the stubble DMD for different levels of intake. These results can be summaries into the 4 stubble categories
    providing the DMD and proportion of the total stubble in each.

    The energy provided from consuming each stubble category is calculated from DMD. Like pasture, stubble
    FOO is expressed in units of dry matter (excluding moisture) therefore feed energy is expressed as MD
    (does not require dry matter content conversion). The volume of each stubble category is calculated
    based on both the quality and availability of the feed.

    Farmer often rack and burn stubbles in preparation for the following seeding. This is represented as a
    cost see Crop.py for further information.

    '''
    ##ev stuff
    confinement_inc = np.maximum(np.max(sinp.structuralsa['i_nut_spread_n1'][0:sinp.structuralsa['i_n1_len']]),
                                 np.max(sinp.structuralsa['i_nut_spread_n3'][0:sinp.structuralsa['i_n3_len']])) > 3 #if fs>3 then need to include confinement feeding
    ev_is_not_confinement_v = sinp.general['ev_is_not_confinement']
    ev_mask_v = np.logical_or(ev_is_not_confinement_v, confinement_inc)
    ev_is_not_confinement_v = ev_is_not_confinement_v[ev_mask_v]



    ##create mask which is stubble available. Stubble is available from the period harvest starts to the beginning of the following growing season.
    ##if the end date of the fp is after harvest then stubble is available.
    fp_end_p6z = per.f_feed_periods()[1:].astype('datetime64[D]')
    fp_start_p6z = per.f_feed_periods()[:-1].astype('datetime64[D]')
    harv_date_zk = pinp.f_seasonal_inp(pinp.crop['start_harvest_crops'].values, numpy=True, axis=1).swapaxes(0,1).astype(np.datetime64)
    mask_stubble_exists_p6zk = fp_end_p6z[...,na] > harv_date_zk  #need to use the full fp array that has the end date of the last period.

    #########################
    #dmd deterioration      #
    #########################
    stubble_per_grain = crp.f_stubble_production() #produces dict with stubble production per kg of yield for each grain used in the ri.availability section

    ##days since harvest (calculated from the end date of each fp)
    days_since_harv_p6zk = fp_end_p6z[...,na] - harv_date_zk.astype('datetime64[D]')
    days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] = days_since_harv_p6zk[days_since_harv_p6zk.astype(int)<0] + 365  #add 365 to the periods at the start of the year because as far as stubble goes they are after harvest
    average_days_since_harv_p6zk = days_since_harv_p6zk - np.minimum(days_since_harv_p6zk, (fp_end_p6z - fp_start_p6z)[...,na])/2 #subtract half the length of current period to get the average days since harv. Minimum is to handle the period when harvest occurs.
    average_days_since_harv_p6zk = average_days_since_harv_p6zk.astype(float)

    ##calc the quantity decline % for each period - used in transfer constraints, need to average the number of days in the period of interest
    quant_decline_p6zk = 1 - (1 - pinp.stubble['quantity_deterioration']) ** average_days_since_harv_p6zk.astype(float)

    ##calc dmd for each component in each period for each crop
    deterioration_factor_ks0 = pinp.stubble['quality_deterioration']
    dmd_component_harv_ks0 = pinp.stubble['component_dmd']
    dmd_component_p6zks0 = (1 - (deterioration_factor_ks0 * average_days_since_harv_p6zk[...,na])) * dmd_component_harv_ks0

    ###############
    # M/D & vol   #
    ###############
    '''
    This section creates a df that contains the M/D for each stubble category for each crop and 
    the equivalent for vol. This is used by live stock.
    
    1) read in stubble component composition, calculated by the sim (stored in an excel file)
    2) converts total dmd to dmd of category 
    3) calcs ri quantity and availability 
    4) calcs the md of each stubble category (dmd to MD)
    
    '''
    ##load in data from stubble sim spreadsheet.
    n_crops = len(pinp.crop['start_harvest_crops'])
    n_comp = dmd_component_harv_ks0.shape[1]
    n_cat = 4
    n_seasons = pinp.f_keys_z().shape[0]
    ks0s1 = (n_crops, n_comp, n_cat)
    stub_cat_component_proportion_ks0s1 = np.zeros(ks0s1)
    for crop, crop_idx in zip(pinp.crop['start_harvest_crops'].index, range(len(pinp.crop['start_harvest_crops']))):
        try: #required if the crop does not have stubble sim inputs
            stub_cat_component_proportion_ks0s1[crop_idx,...] = pd.read_excel('stubble sim.xlsx',sheet_name=crop,header=None, engine='openpyxl')
        except KeyError:
            pass

    ##quality of each category in each period - multiply quality by proportion of components in each category (a, b, c, d) then sum the components axis
    dmd_cat_p6zks1 = np.sum(dmd_component_p6zks0[...,na] * stub_cat_component_proportion_ks0s1, axis=-2)

    ##calc relative quality before converting dmd to md - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        ri_quality_p6zks1 = fsfun.f_rq_cs(dmd_cat_p6zks1, pinp.stubble['clover_propn_in_sward_stubble'])

    ##ri availability - first calc stubble foo (stub available) this is the average from all rotations because we just need one value for foo
    ###try calc the base yield for each crop but if the crop is not one of the rotation phases then assign the average foo (this is only to stop error. it doesnt matter because the crop doesnt exist so the stubble is never used)
    base_yields = crp.f_base_yield().droplevel(0, axis=0) #drop rotation index
    stub_foo_harv_zk = np.zeros((n_seasons, n_crops))
    for crop, crop_idx in zip(pinp.crop['start_harvest_crops'].index, range(len(pinp.crop['start_harvest_crops']))):
        try:
            stub_foo_harv_zk[:,crop_idx] = base_yields.loc[crop].mean() * stubble_per_grain[(crop,'a')]
        except KeyError: #if the crop is not in any of the rotations assign average foo to stop error - this is not used so could assign any value.
            stub_foo_harv_zk[:,crop_idx] = base_yields.mean()
    ###adjust the foo for each category because the good stuff is eaten first therefore there is less foo when the sheep start eating the poorer stubble
    cat_propn_ks1 = pinp.stubble['stub_cat_prop']
    cat_propn_rolled_ks1 = np.roll(cat_propn_ks1, shift=1, axis=1) #roll along the cat axis. So that the previous cat lines up with the current cat
    cat_propn_rolled_ks1[:, 0] = 0 #set the first slice to 0 because no stubble is consumed before cat A is consumed eg there is 100% of foo available when sheep are consuming cat A
    cat_cum_propn_ks1 = np.cumsum(cat_propn_rolled_ks1, axis=1) #cumulative sum of the component sizes.
    stubble_foo_zks1 = stub_foo_harv_zk[..., na] *  (1 - cat_cum_propn_ks1)
    ###adjust for quantity delcine due to deterioration
    stubble_foo_p6zks1 = stubble_foo_zks1 * (1 - quant_decline_p6zk[..., na])
    ###ri availability
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used - note that the equation system used is the one selected for dams in p1
        ri_availability_p6zks1 = fsfun.f_ra_cs(stubble_foo_p6zks1, pinp.stubble['i_hf'])

    ##combine ri quality and ri availability to calc overall vol (potential intake)
    ri_p6zks1 = fsfun.f_rel_intake(ri_availability_p6zks1, ri_quality_p6zks1, pinp.stubble['clover_propn_in_sward_stubble'])
    vol_p6zks1 = (1000 / ri_p6zks1) / (1 + SA.sap['pi'])
    vol_p6zks1 = vol_p6zks1 * mask_stubble_exists_p6zk[..., na] #stop md being provided if stubble doesnt exist

    ##convert dmd to M/D
    ## Stubble doesn't include calculation of effective mei because stubble is generally low quality feed with a wide variation in quality within the sward.
    ## Therefore, there is scope to alter average diet quality by altering the grazing time and the proportion of the stubble consumed.
    md_p6zks1 = np.clip(fsfun.dmd_to_md(dmd_cat_p6zks1) * 1000, 0, np.inf) #mul to convert to tonnes
    md_p6zks1 = md_p6zks1 * mask_stubble_exists_p6zk[...,na] #stop md being provided if stubble doesnt exist
    md_vp6zks1 = md_p6zks1 * ev_is_not_confinement_v[:,na,na,na,na] #me from stubble is 0 in the confinement pool

    ###########
    #trampling#
    ###########
    #for now this is just a single number however the input could be changed to per period, if this is changed some of the dict below would need to be dfs the stacked - so they account for period
    tramp_effect_ks1 = pinp.stubble['trampling'][:,na] * cat_propn_ks1

    ################################
    # allow access to next category#   #^this is a little inflexible ie you would need to add or remove code if a stubble cat was added or removed
    ################################

    cat_a_st_req_p6zk = (1/(1-quant_decline_p6zk))*(1+tramp_effect_ks1[:,0])*(1/cat_propn_ks1[:,0])*1000 #*1000 - to convert to tonnes
    cat_b_st_prov_k = cat_propn_ks1[:,1]/cat_propn_ks1[:,0]*1000 #cat b provided by consuming 1t of cat a.
    cat_b_st_req_k = 1000*(1+tramp_effect_ks1[:,1])
    cat_c_st_prov_k = cat_propn_ks1[:,2]/cat_propn_ks1[:,1]*1000
    cat_c_st_req_k = 1000*(1+tramp_effect_ks1[:,2])

    ##############################
    #transfers between periods   #
    ##############################
    ##transfer a given cat to the next period.
    per_transfer_p6zk = 1000 - quant_decline_p6zk*1000
    per_transfer_p6zk = per_transfer_p6zk * mask_stubble_exists_p6zk  #no transfer can occur when stubble doesnt exist

    ###############
    #harvest p con# stop sheep consuming more than possible because harvest is not at the start of the period
    ###############
    #how far through each period does harv start? note: 0 for each period harv doesn't start in. Used to calc stub consumption limit in harv period
    fp_len_p6z = fp_end_p6z - fp_start_p6z
    cons_propn_p6zk = np.clip((fp_len_p6z[...,na] - (fp_end_p6z[...,na] - harv_date_zk)) / fp_len_p6z[...,na],0, np.inf)
    cons_propn_p6zk[cons_propn_p6zk>1] = 0

    #########
    ##keys  #
    #########
    ##keys
    keys_k = np.array(pinp.crop['start_harvest_crops'].index)
    keys_p6 = pinp.period['i_fp_idx']
    keys_s1_cut = np.array(['b', 'c'])
    keys_s1_cut2 = np.array(['a'])
    keys_s1 = pinp.stubble['stub_cat_idx']
    keys_v  = np.asarray(sinp.general['sheep_pools'][ev_mask_v])
    keys_z = pinp.f_keys_z()


    ##array indexes
    ###ks1 - stub transfer (cat b & c)
    arrays = [keys_k, keys_s1_cut]
    index_bc_ks1 = fun.cartesian_product_simple_transpose(arrays)
    ###p6ks1 - category A req
    arrays = [keys_p6, keys_k, keys_s1_cut2]
    index_a_p6ks1 = fun.cartesian_product_simple_transpose(arrays)
    ###p6ks1 - md & vol
    arrays = [keys_p6, keys_k, keys_s1]
    index_p6ks1 = fun.cartesian_product_simple_transpose(arrays)
    ###p6ks1 - md & vol
    arrays = [keys_v, keys_p6, keys_k, keys_s1]
    index_vp6ks1 = fun.cartesian_product_simple_transpose(arrays)
    ###p6k - p7con & feed period transfer
    arrays = [keys_p6, keys_k]
    index_p6k = fun.cartesian_product_simple_transpose(arrays)

    ################
    ##pyomo params #
    ################

    ##'require' params ie consuming 1t of stubble B requires 1.002t from the constraint (0.002 accounts for trampling)
    stub_req_ks1 = np.stack([cat_b_st_req_k, cat_c_st_req_k], 1)
    stub_req_ks1 = stub_req_ks1.ravel()
    tup_ks1 = tuple(map(tuple, index_bc_ks1))
    params['transfer_req'] =dict(zip(tup_ks1, stub_req_ks1))

    ###'provide' from cat to cat ie consuming 1t of cat A provides 2t of cat b
    stub_prov_ks1 = np.stack([cat_b_st_prov_k, cat_c_st_prov_k], 1)
    stub_prov_ks1 = stub_prov_ks1.ravel()
    tup_ks1 = tuple(map(tuple, index_bc_ks1))
    params['transfer_prov'] =dict(zip(tup_ks1, stub_prov_ks1))

    ##create season params in loop
    for z in range(len(keys_z)):
        ##create season key for params dict
        params[keys_z[z]] = {}
        scenario = keys_z[z]

        ###p7con
        cons_propn_p6k = cons_propn_p6zk[:,z,:].ravel()
        tup_p6k = tuple(map(tuple, index_p6k))
        params[scenario]['cons_prop'] =dict(zip(tup_p6k, cons_propn_p6k))

        ###feed period transfer
        per_transfer_p6k = per_transfer_p6zk[:,z,:].ravel()
        tup_p6k = tuple(map(tuple, index_p6k))
        params[scenario]['per_transfer'] =dict(zip(tup_p6k, per_transfer_p6k))

        ###category A transfer 'require' param
        cat_a_st_req_p6k = cat_a_st_req_p6zk[:,z,:].ravel()
        tup_p6k = tuple(map(tuple, index_a_p6ks1))
        params[scenario]['cat_a_st_req'] =dict(zip(tup_p6k, cat_a_st_req_p6k))

        ##md
        md_vp6ks1 = md_vp6zks1[:,:,z,:,:].ravel()
        tup_vp6ks1 = tuple(map(tuple, index_vp6ks1))
        params[scenario]['md'] =dict(zip(tup_vp6ks1, md_vp6ks1))

        ##vol
        vol_p6ks1 = vol_p6zks1[:,z,:,:].ravel()
        tup_p6ks1 = tuple(map(tuple, index_p6ks1))
        params[scenario]['vol'] =dict(zip(tup_p6ks1, vol_p6ks1))


