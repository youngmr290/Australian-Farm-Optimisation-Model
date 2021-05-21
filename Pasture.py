# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:46:24 2019
@author: john

Description of this pasture module: This representation includes at optimisation (ie the following options are represented in the variables of the model)
    Growth rate of pasture (PGR) varies with FOO at the start of the period and grazing intensity during the period
        Grazing intensity operates by altering the average FOO during the period
    The nutritive value of the green feed consumed (as represented by ME & volume) varies with FOO & grazing intensity.
        Grazing intensity alters the average FOO during the period and the capacity of the animals to select a higher quality diet.
    Selective grazing of dry pasture. 2 dry pasture quality pools are represented and either can be selected for grazing
        Note: There is not a constraint that ensures that the high quality pool is grazed prior to the low quality pool (as there is in the stubble selective grazing)

This is the version that uses an extra axis on the array rather than a class.
"""


'''
import functions from other modules
'''
# import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

import PropertyInputs as pinp
import StockFunctions as sfun
import UniversalInputs as uinp
import StructuralInputs as sinp
import Functions as fun
import Periods as per
import Sensitivity as sen
import PastureFunctions as pfun

#todo Will need to add the foo reduction in the current year for manipulated pasture and a germination reduction in the following year.

def f_pasture(params, r_vals, ev):
    ######################
    ##background vars    #
    ######################
    na = np.newaxis

    ########################
    ##ev stuff             #
    ########################
    confinement_inc = np.maximum(np.max(sinp.structuralsa['i_nut_spread_n1'][0:sinp.structuralsa['i_n1_len']]),
                                 np.max(sinp.structuralsa['i_nut_spread_n3'][0:sinp.structuralsa['i_n3_len']])) > 3 #if fs>3 then need to include confinement feeding
    ev_is_not_confinement_v = sinp.general['ev_is_not_confinement']
    ev_mask_v = np.logical_or(ev_is_not_confinement_v, confinement_inc)
    ev_is_not_confinement_v = ev_is_not_confinement_v[ev_mask_v]
    len_v1 = np.count_nonzero(ev_is_not_confinement_v) #number of normal ev pools (doesnt including confinement)
    ########################
    ##phases               #
    ########################
    ## read the rotation phases information from inputs
    # phase_len       = sinp.general['phase_len']
    phases_rotn_df  = sinp.phases['phases']
    pasture_sets    = sinp.landuse['pasture_sets']
    pastures        = sinp.general['pastures'][pinp.general['pas_inc']]

    ########################
    ##constants required   #
    ########################
    ## define some parameters required to size arrays.
    n_feed_pools    = np.count_nonzero(ev_mask_v)
    n_dry_groups    = len(sinp.general['dry_groups'])           # Low & high quality groups for dry feed
    n_grazing_int   = len(sinp.general['grazing_int'])          # grazing intensity in the growth/grazing activities
    n_foo_levels    = len(sinp.general['foo_levels'])           # Low, medium & high FOO level in the growth/grazing activities
    n_feed_periods  = len(per.f_feed_periods()) - 1
    n_lmu           = len(pinp.general['lmu_area'])
    n_phases_rotn   = len(phases_rotn_df.index)
    n_pasture_types = len(pastures)   #^ need to sort timing of the definition of pastures
    # n_total_seasons = len(pinp.general['i_mask_z']) #used to reshape inputs
    if pinp.general['steady_state']:
        n_season_types = 1
    else:
        n_season_types = np.count_nonzero(pinp.general['i_mask_z'])

    index_f = np.arange(n_feed_periods)

    ## indexes required for advanced indexing
    l_idx = np.arange(n_lmu)
    z_idx = np.arange(n_season_types)
    t_idx = np.arange(n_pasture_types)
    r_idx = np.arange(n_phases_rotn)


    arable_l = np.array(pinp.crop['arable']).reshape(-1)
    # length_f  = np.array(pinp.period['feed_periods'].loc[:pinp.period['feed_periods'].index[-2],'length']) # not including last row because that is the start of the following year. #todo as above this will need z axis
    # feed_period_dates_f = np.array(i_feed_period_dates,dtype='datetime64[D]')
    length_fz  = np.array(per.f_feed_periods(option=1),dtype='float64')
    feed_period_dates_fz = fun.f_baseyr(per.f_feed_periods()).astype('datetime64[D]') #feed periods are all date to the base yr (eg 2019) - this is required for some of the allocation formulas

    # vgoflt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dgoflzt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # vdft    = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
    vfzt    = (n_feed_pools, n_feed_periods, n_season_types, n_pasture_types)
    # dft     = (n_dry_groups, n_feed_periods, n_pasture_types)
    # goflzt  = (n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # goft    = (n_grazing_int, n_foo_levels, n_feed_periods, n_pasture_types)
    gft     = (n_grazing_int, n_feed_periods, n_pasture_types)
    gt      = (n_grazing_int, n_pasture_types)
    oflzt   = (n_foo_levels, n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    # dflrt   = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    dflrzt  = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    flrt    = (n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    flrzt   = (n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    # frt     = (n_feed_periods, n_phases_rotn, n_pasture_types)
    rt      = (n_phases_rotn, n_pasture_types)
    flt     = (n_feed_periods, n_lmu, n_pasture_types)
    flzt    = (n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    lt      = (n_lmu, n_pasture_types)
    lzt     = (n_lmu, n_season_types, n_pasture_types)
    ft      = (n_feed_periods, n_pasture_types)
    fzt     = (n_feed_periods, n_season_types, n_pasture_types)
    zt      = (n_season_types, n_pasture_types)
    # t       = (n_pasture_types)

    ## define the vessels that will store the input data that require pre-defining
    ### all need pre-defining because inputs are in separate pasture type arrays
    i_phase_germ_dict = dict()
    i_grn_senesce_daily_ft      = np.zeros(ft,  dtype = 'float64')              # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_fzt       = np.zeros(fzt,  dtype = 'float64')             # proportion of green feed that senesces in period (due to a water deficit or completing life cycle)
    dry_decay_daily_fzt         = np.zeros(fzt,  dtype = 'float64')             # daily decline in dry foo in each period
    i_end_of_gs_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_decay_t               = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    # i_me_maintenance_vft        = np.zeros(vft,  dtype = 'float64')     # M/D level for target LW pattern
    c_pgr_gi_scalar_gft         = np.zeros(gft,  dtype = 'float64')     # pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_graze_propn_gt        = np.zeros(gt, dtype ='float64')        # proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_oflzt             = np.zeros(oflzt, dtype = 'float64')    # FOO level     for the FOO/growth/grazing variables.
    i_fxg_pgr_oflzt             = np.zeros(oflzt, dtype = 'float64')    # PGR level     for the FOO/growth/grazing variables.
    c_fxg_a_oflzt               = np.zeros(oflzt, dtype = 'float64')    # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_oflzt               = np.zeros(oflzt, dtype = 'float64')    # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_oflt               = np.zeros(oflt, dtype = 'float64')     # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_oflt               = np.zeros(oflt, dtype = 'float64')     # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_flzt              = np.zeros(flzt, dtype = 'float64') # green pasture digestibility in each period, LMU, season & pasture type.
    i_poc_intake_daily_flt      = np.zeros(flt, dtype = 'float64')  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_flt      = np.zeros(flt, dtype = 'float64')  # minimum foo at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lzt           = np.zeros(lzt,  dtype = 'float64') # scale the germination levels for each lmu
    i_restock_fooscalar_lt      = np.zeros(lt,  dtype = 'float64')  # scalar for FOO between LMUs when pastures are restocked after reseeding

    i_me_eff_gainlose_ft        = np.zeros(ft,  dtype = 'float64')  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_ft          = np.zeros(ft,  dtype = 'float64')  # green pasture trampling in each feed period as proportion of intake.
    i_dry_trampling_ft          = np.zeros(ft,  dtype = 'float64')  # dry pasture trampling   in each feed period as proportion of intake.
    i_base_ft                   = np.zeros(ft,  dtype = 'float64')  # lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_ft     = np.zeros(ft,  dtype = 'float64')  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_ft          = np.zeros(ft,  dtype = 'float64')  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_fzt  = np.zeros(fzt,  dtype = 'float64') # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_fzt           = np.zeros(fzt,  dtype = 'float64') # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_fzt         = np.zeros(fzt,  dtype = 'float64') # range in digestibility of dry feed if it is not grazed
    i_dry_foo_high_fzt          = np.zeros(fzt,  dtype = 'float64') # expected foo for the dry pasture in the high quality pool
    dry_decay_period_fzt        = np.zeros(fzt,  dtype = 'float64') # decline in dry foo for each period
    mask_dryfeed_exists_fzt     = np.zeros(fzt,  dtype = bool)      # mask for period when dry feed exists
    mask_greenfeed_exists_fzt   = np.zeros(fzt,  dtype = bool)      # mask for period when green feed exists
    i_germ_scalar_fzt           = np.zeros(fzt,  dtype = 'float64') # allocate the total germination between feed periods
    i_grn_cp_ft                 = np.zeros(ft,  dtype = 'float64')  # crude protein content of green feed
    i_dry_cp_ft                 = np.zeros(ft,  dtype = 'float64')  # crude protein content of dry feed
    i_poc_dmd_ft                = np.zeros(ft,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_ft                = np.zeros(ft,  dtype = 'float64')  # foo of pasture consumed on crop paddocks
    # grn_senesce_startfoo_ft     = np.zeros(ft,  dtype = 'float64')  # proportion of the FOO at the start of the period that senesces during the period
    # grn_senesce_pgrcons_ft      = np.zeros(ft,  dtype = 'float64')  # proportion of the (total or average daily) PGR that senesces during the period (consumption leads to a reduction in senescence)

    i_reseeding_date_start_zt   = np.zeros(zt, dtype = 'datetime64[D]')         # start date of seeding this pasture type
    i_reseeding_date_end_zt     = np.zeros(zt, dtype = 'datetime64[D]')         # end date of the pasture seeding window for this pasture type
    i_destock_date_zt           = np.zeros(zt, dtype = 'datetime64[D]')         # date of destocking this pasture type prior to reseeding
    i_destock_foo_zt            = np.zeros(zt, dtype = 'float64')               # kg of FOO that was not grazed prior to destocking for spraying prior to reseeding pasture (if spring sown)
    i_restock_date_zt           = np.zeros(zt, dtype = 'datetime64[D]')         # date of first grazing of reseeded pasture
    i_restock_foo_arable_t      = np.zeros(n_pasture_types, dtype = 'float64')  # FOO at restocking on the arable area of the resown pastures
    # reseeding_machperiod_t      = np.zeros(n_pasture_types, dtype = 'float64')  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
    i_germination_std_zt        = np.zeros(zt, dtype = 'float64')               # standard germination level for the standard soil type in a continuous pasture rotation
    # i_ri_foo_t                  = np.zeros(n_pasture_types, dtype = 'float64')  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    # poc_days_of_grazing_t       = np.zeros(n_pasture_types, dtype = 'float64')  # number of days after the pasture break that (moist) seeding can begin
    i_legume_zt                 = np.zeros(zt, dtype = 'float64')               # proportion of legume in the sward
    i_restock_grn_propn_t       = np.zeros(n_pasture_types, dtype = 'float64')  # Proportion of the FOO that is green when pastures are restocked after reseeding
    i_fec_maintenance_t         = np.zeros(n_pasture_types, dtype = 'float64')  # approximate feed energy concentration for maintenance (FEC = M/D * relative intake)

#    germination_flrzt           = np.zeros(flrzt,  dtype = 'float64')  # germination for each rotation phase (kg/ha)
    foo_grn_reseeding_flrzt     = np.zeros(flrzt,  dtype = 'float64')  # green FOO adjustment for destocking and restocking of the resown area (kg/ha)
    foo_dry_reseeding_flrzt     = np.zeros(flrzt,  dtype = 'float64')  # dry FOO adjustment for destocking and restocking of the resown area (kg/ha)
    foo_dry_reseeding_dflrzt    = np.zeros(dflrzt, dtype = 'float64')  # dry FOO adjustment allocated to the low & high quality dry feed pools (kg/ha)
    # dry_removal_t_ft            = np.zeros(ft,   dtype = 'float64')  # Total DM removal from the tonne consumed (includes trampling)

    ### define the array that links rotation phase and pasture type
    pasture_rt                  = np.zeros(rt, dtype = 'float64')


    ## create numpy index for param dicts ^creating indexes is a bit slow
    ### the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
    keys_d  = np.asarray(sinp.general['dry_groups'])
    keys_v  = np.asarray(sinp.general['sheep_pools'][ev_mask_v])
    keys_f  = pinp.period['i_fp_idx']
    keys_g  = np.asarray(sinp.general['grazing_int'])
    keys_l  = np.array(pinp.general['lmu_area'].index).astype('str')    # lmu index description
    keys_o  = np.asarray(sinp.general['foo_levels'])
    keys_p  = np.array(per.p_date2_df().index).astype('str')
    keys_r  = np.array(phases_rotn_df.index).astype('str')
    keys_t  = np.asarray(pastures)                                      # pasture type index description
    keys_k  = np.asarray(list(sinp.landuse['All']))                     #landuse
    keys_z  = pinp.f_keys_z()

    ### plrk
    arrays=[keys_p, keys_l, keys_r, keys_k]
    index_plrk=fun.cartesian_product_simple_transpose(arrays)
    index_plrk=tuple(map(tuple, index_plrk)) #create a tuple rather than a list because tuples are faster

    ### rt
    arrays=[keys_r, keys_t]
    index_rt=fun.cartesian_product_simple_transpose(arrays)
    index_rt=tuple(map(tuple, index_rt)) #create a tuple rather than a list because tuples are faster

    ### flrt
    arrays=[keys_f, keys_l, keys_r, keys_t]
    index_flrt=fun.cartesian_product_simple_transpose(arrays)
    index_flrt=tuple(map(tuple, index_flrt)) #create a tuple rather than a list because tuples are faster

    ### oflt
    arrays=[keys_o, keys_f, keys_l, keys_t]
    index_oflt=fun.cartesian_product_simple_transpose(arrays)
    index_oflt=tuple(map(tuple, index_oflt)) #create a tuple rather than a list because tuples are faster

    ### goflt
    arrays=[keys_g, keys_o, keys_f, keys_l, keys_t]
    index_goflt=fun.cartesian_product_simple_transpose(arrays)
    index_goflt=tuple(map(tuple, index_goflt)) #create a tuple rather than a list because tuples are faster

    ### vgoflt
    arrays=[keys_v, keys_g, keys_o, keys_f, keys_l, keys_t]
    index_vgoflt=fun.cartesian_product_simple_transpose(arrays)
    index_vgoflt=tuple(map(tuple, index_vgoflt)) #create a tuple rather than a list because tuples are faster

    ### dgoflt
    arrays=[keys_d, keys_g, keys_o, keys_f, keys_l, keys_t]
    index_dgoflt=fun.cartesian_product_simple_transpose(arrays)
    index_dgoflt=tuple(map(tuple, index_dgoflt)) #create a tuple rather than a list because tuples are faster

    ### dflrt
    arrays=[keys_d, keys_f, keys_l, keys_r, keys_t]
    index_dflrt=fun.cartesian_product_simple_transpose(arrays)
    index_dflrt=tuple(map(tuple, index_dflrt)) #create a tuple rather than a list because tuples are faster

    ### vdft
    arrays=[keys_v, keys_d, keys_f, keys_t]
    index_vdft=fun.cartesian_product_simple_transpose(arrays)
    index_vdft=tuple(map(tuple, index_vdft)) #create a tuple rather than a list because tuples are faster

    ### dft
    arrays=[keys_d, keys_f, keys_t]
    index_dft=fun.cartesian_product_simple_transpose(arrays)
    index_dft=tuple(map(tuple, index_dft)) #create a tuple rather than a list because tuples are faster

    ### vf
    arrays=[keys_v, keys_f]
    index_vf=fun.cartesian_product_simple_transpose(arrays)
    index_vf=tuple(map(tuple, index_vf)) #create a tuple rather than a list because tuples are faster

    ### fl
    arrays=[keys_f, keys_l]
    index_fl=fun.cartesian_product_simple_transpose(arrays)
    index_fl=tuple(map(tuple, index_fl)) #create a tuple rather than a list because tuples are faster

    ### ft
    arrays=[keys_f, keys_t]
    index_ft=fun.cartesian_product_simple_transpose(arrays)
    index_ft=tuple(map(tuple, index_ft)) #create a tuple rather than a list because tuples are faster

    ###########
    #map_excel#
    ###########
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''

    ## map data from excel file into arrays
    ### loop through each pasture type
    for t, pasture in enumerate(pastures):
        exceldata = pinp.pasture_inputs[pasture]           # assign the pasture data to exceldata
        ## map the Excel data into the numpy arrays
        i_germination_std_zt[...,t]         = pinp.f_seasonal_inp(exceldata['GermStd'], numpy=True)
        # i_ri_foo_t[t]                       = exceldata['RIFOO']
        i_end_of_gs_zt[...,t]               = pinp.f_seasonal_inp(exceldata['EndGS'], numpy=True)
        i_dry_decay_t[t]                    = exceldata['PastDecay']
        i_poc_intake_daily_flt[...,t]       = exceldata['POCCons']
        i_legume_zt[...,t]                  = pinp.f_seasonal_inp(exceldata['Legume'], numpy=True)
        i_restock_grn_propn_t[t]            = exceldata['FaG_PropnGrn']
        i_grn_dmd_senesce_redn_fzt[...,t]   = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigRednSenesce'],0,1), numpy=True, axis=1)
        i_dry_dmd_ave_fzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryAve'],0,1), numpy=True, axis=1)
        i_dry_dmd_range_fzt[...,t]          = pinp.f_seasonal_inp(np.swapaxes(exceldata['DigDryRange'],0,1), numpy=True, axis=1)
        i_dry_foo_high_fzt[...,t]           = pinp.f_seasonal_inp(np.swapaxes(exceldata['FOODryH'],0,1), numpy=True, axis=1)
        i_germ_scalar_fzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarFP'],0,1), numpy=True, axis=1)

        i_grn_cp_ft[...,t]                  = exceldata['CPGrn']
        i_dry_cp_ft[...,t]                  = exceldata['CPDry']
        i_poc_dmd_ft[...,t]                 = exceldata['DigPOC']
        i_poc_foo_ft[...,t]                 = exceldata['FOOPOC']
        i_germ_scalar_lzt[...,t]            = pinp.f_seasonal_inp(np.swapaxes(exceldata['GermScalarLMU'],0,1), numpy=True, axis=1)
        i_restock_fooscalar_lt[...,t]       = exceldata['FaG_LMU']  #todo may need a z axis

        i_lmu_conservation_flt[...,t]       = exceldata['ErosionLimit']

        i_reseeding_date_start_zt[...,t]    = pinp.f_seasonal_inp(exceldata['Date_Seeding'], numpy=True)
        i_reseeding_date_end_zt[...,t]      = pinp.f_seasonal_inp(exceldata['pas_seeding_end'], numpy=True)
        i_destock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_Destocking'], numpy=True)
        i_destock_foo_zt[...,t]             = pinp.f_seasonal_inp(exceldata['FOOatSeeding'], numpy=True) #ungrazed foo when destocked for reseeding
        i_restock_date_zt[...,t]            = pinp.f_seasonal_inp(exceldata['Date_ResownGrazing'], numpy=True)
        i_restock_foo_arable_t[t]           = exceldata['FOOatGrazing']

        i_grn_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_ft[...,t]       = np.asfarray(exceldata['SenescePropn'])
        i_grn_senesce_eos_fzt[...,t]        = pinp.f_seasonal_inp(np.asfarray(exceldata['SenesceEOS']), numpy=True, axis=1)
        i_base_ft[...,t]                    = np.asfarray(exceldata['BaseLevelInput'])
        i_grn_dmd_declinefoo_ft[...,t]      = np.asfarray(exceldata['DigDeclineFOO'])
        i_grn_dmd_range_ft[...,t]           = np.asfarray(exceldata['DigSpread'])
        i_foo_graze_propn_gt[..., t]        = np.asfarray(exceldata['FOOGrazePropn'])
        #### impact of grazing intensity (at the other levels) on PGR during the period
        c_pgr_gi_scalar_gft[...,t]      = 1 - i_foo_graze_propn_gt[..., na, t] ** 2 * (1 - np.asfarray(exceldata['PGRScalarH']))

        i_fxg_foo_oflzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowFOO'],0,-1), numpy=True, axis=-1)
        i_fxg_foo_oflzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedFOO'],0,-1), numpy=True, axis=-1)
        i_me_eff_gainlose_ft[...,t]     = exceldata['MaintenanceEff'][:,0]
        # i_me_maintenance_vft[...,t]     = exceldata['MaintenanceEff'].iloc[:,1:].to_numpy().T  # replaced by the ev_cutoff. Still used in PastureTest
        i_fec_maintenance_t[t]          = exceldata['MaintenanceFEC']
        ## # i_fxg_foo_oflt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_oflzt[0,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['LowPGR'],0,-1), numpy=True, axis=-1)
        i_fxg_pgr_oflzt[1,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)
        i_fxg_pgr_oflzt[2,...,t]        = pinp.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_flzt[...,t]           = pinp.f_seasonal_inp(np.moveaxis(exceldata['DigGrn'],0,-1), numpy=True, axis=-1)  # numpy array of inputs for green pasture digestibility on each LMU.

        i_phase_germ_dict[pasture]      = pd.DataFrame(exceldata['GermPhases'])  #DataFrame with germ scalar and resown
        # i_phase_germ_dict[pasture].reset_index(inplace=True)                                # replace index read from Excel with numbers to match later merging
        # i_phase_germ_dict[pasture].columns.values[range(phase_len)] = [*range(phase_len)]   # replace the pasture columns read from Excel with numbers to match later merging

        ### define the link between rotation phase and pasture type while looping on pasture
        pasture_rt[:,t]                 = phases_rotn_df.iloc[:,-1].isin(pasture_sets[pasture])

    ##season inputs not required in t loop above
    harv_date_z         = pinp.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype(np.datetime64)
    i_pasture_stage_p6z = np.rint(pinp.f_seasonal_inp(np.moveaxis(pinp.sheep['i_pasture_stage_p6z'],0,-1), numpy=True, axis=-1)
                                  ).astype(int) #it would be better if z axis was treated after pas_stage has been used (like in stock.py) because it is used as an index. But there wasn't any way to do this without doubling up a lot of code. This is only a limitation in the weighted average version of model.
    ### pasture params used to convert foo for rel availability
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)

    ## one time data manipulation for the inputs just read
    ### calculate dry_decay_period (used in reseeding and green&dry)
    dry_decay_daily_fzt[...] = i_dry_decay_t
    for t in range(n_pasture_types):
        for z in range(n_season_types):
            dry_decay_daily_fzt[0:i_end_of_gs_zt[z,t], z, t] = 1  #couldn't do this without loops - advanced indexing doesnt appear to work when taking multiple slices
    dry_decay_period_fzt[...] = 1 - (1 - dry_decay_daily_fzt) ** length_fz[...,na]

    ###create dry pasture exists mask - in the current structure dry pasture only exists after the growing season.
    # todo this is a limitation of pasture (green and dry pasture don't exist simultaneously) this is okay for wa but may need work for places with perennials.
    mask_dryfeed_exists_fzt[...] = index_f[:, na, na] > i_end_of_gs_zt   #green exists in the period which is the end of growing season hence >
    mask_greenfeed_exists_fzt[...] = np.logical_not(mask_dryfeed_exists_fzt)

    ###create equation coefficients for pgr = a+b*foo
    i_fxg_foo_oflzt[2,...]  = 100000 #large number so that the np.searchsorted doesn't go above
    c_fxg_b_oflzt[0,...] =  fun.f_divide(i_fxg_pgr_oflzt[0,...], i_fxg_foo_oflzt[0,...])
    c_fxg_b_oflzt[1,...] =   fun.f_divide((i_fxg_pgr_oflzt[1,...] - i_fxg_pgr_oflzt[0,...])
                            , (i_fxg_foo_oflzt[1,...] - i_fxg_foo_oflzt[0,...]))
    c_fxg_b_oflzt[2,...] =  0
    c_fxg_a_oflzt[0,...] =  0
    c_fxg_a_oflzt[1,...] =  i_fxg_pgr_oflzt[0,...] - c_fxg_b_oflzt[1,...] * i_fxg_foo_oflzt[0,...]
    c_fxg_a_oflzt[2,...] =  i_fxg_pgr_oflzt[1,...] # because slope = 0

    ## proportion of start foo that senesces during the period, different formula than excel
    grn_senesce_startfoo_fzt = 1 - ((1 - i_grn_senesce_daily_ft[:,na,:]) **  length_fz[...,na])

    ## average senescence over the period for the growth and consumption
    grn_senesce_pgrcons_fzt = 1 - ((1 -(1 - i_grn_senesce_daily_ft[:,na,:]) ** (length_fz[...,na]+1))
                                   /        i_grn_senesce_daily_ft[:,na,:]-1) / length_fz[...,na]



    #################################################
    #Calculate germination and reseeding parameters #
    #################################################

    ##germination: create an array called p_germination_flrt being the parameters to be passed to pyomo.
    ##reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    ##Results are stored in p_...._reseeding

    #todo currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed
    ## intermediate calculations are not stored, however, if they were stored the 'key variables' could change the values of the intermediate calcs which could then be fed into the parameter calculations (in a separate method)
    ## the above would provide more options for KVs and provide another step that may not need to be recalculated

    ## define instantiate arrays that are assigned in slices
    na_erosion_flrt      = np.zeros(flrt,  dtype = 'float64')
    na_phase_area_flrzt  = np.zeros(flrzt, dtype = 'float64')
    grn_restock_foo_flzt = np.zeros(flzt,  dtype = 'float64')
    dry_restock_foo_flzt = np.zeros(flzt,  dtype = 'float64')
    foo_na_destock_fzt   = np.zeros(fzt,   dtype = 'float64')
    germ_scalar_rt       = np.zeros(rt,    dtype='float64')
    resown_rt            = np.zeros(rt,    dtype='int')

    # ### set initial values to 0 because function is called multiple times
    # foo_grn_reseeding_flrt[...] = 0          # array has been initialised, reset all values to 0
    # foo_dry_reseeding_flrt[...] = 0
    # germination_flrt[...]       = 0

    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df

    ## germination and resowing to the rotation phases
    for t, pasture in enumerate(pastures):
        phase_germresow_df['germ_scalar']=0 #set default to 0
        phase_germresow_df['resown']=False #set default to false
        ###loop through each combo of landuses and pastures (i_phase_germ), then check which rotations fall into each germ/resowing category. Then populate the rot phase df with the necessary germination and resowing param.
        for ix_row in i_phase_germ_dict[pasture].index:
            ix_bool = pd.Series(data=True,index=range(len(phase_germresow_df)))
            for ix_col in range(i_phase_germ_dict[pasture].shape[1]-2):    #-2 because two of the cols are germ and resowing
                c_set = sinp.landuse[i_phase_germ_dict[pasture].iloc[ix_row,ix_col]]
                ix_bool &= phase_germresow_df.loc[:,ix_col].reset_index(drop=True).isin(c_set) #had to drop index so that it would work (just said false when the index was different between series)
            ### maps the relevant germ scalar and resown bool to the rotation phase
            phase_germresow_df.loc[list(ix_bool),'germ_scalar'] = i_phase_germ_dict[pasture].iloc[ix_row, -2]  #have to make bool into a list for some reason it doesn't like a series
            phase_germresow_df.loc[list(ix_bool),'resown'] = i_phase_germ_dict[pasture].iloc[ix_row, -1]
        ### Convert germ and resow into a numpy - each pasture goes in a different slice
        germ_scalar_rt[:,t] = phase_germresow_df['germ_scalar'].to_numpy()    # extract the germ_scalar from the dataframe
        resown_rt[:,t] = phase_germresow_df['resown'].to_numpy()              # extract the resown boolean from the dataframe

    ## germination on the arable area of pasture paddocks based on std germ, rotation scalar, lmu scalar and distribution across periods
    arable_germination_flrzt = i_germination_std_zt                 \
                              *   i_germ_scalar_lzt[:, na, ...]     \
                              *      germ_scalar_rt[:, na, :]       \
                              *   i_germ_scalar_fzt[:, na, na, ...]
    arable_germination_flrzt[np.isnan(arable_germination_flrzt)]  = 0.0

    ## germination on the non arable area is the maximum germination across phases (continuous pasture) for the first pasture type (annuals)
    ### todo a potential error here when if the allocation of germination across periods varies by rotation phase (because taking max of each period)
    max_germination_flz = np.max(arable_germination_flrzt[..., 0], axis=2)  #use germination_flrzt because it includes any sensitivity that is carried out

    ## germination on the non arable area of pasture paddocks. Grows pasture type 0 that can be grazed during the growing season
    na_germination_flrz = max_germination_flz[..., na, :] * np.any(pasture_rt[:, na, :], axis = -1)
    ## set germination in first period to germination on arable area
    germination_flrzt = arable_germination_flrzt * arable_l[:, na, na, na]
    ## add germination on the non-arable area to the first pasture type
    germination_flrzt[..., 0] += na_germination_flrz * (1 - arable_l[:,na,na])

    #todo test the calculation of FOO on the resown area when the full set of rotation phases is included
    ## the green feed to remove from matrix when pasture is destocked.
    foo_arable_destock_zt = i_destock_foo_zt
    foo_na_destock_zt =  i_destock_foo_zt
    ## the periods from which to remove based on date destocked.
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na]  # which feed period does destocking occur & the proportion that destocking occurs during the period.
                                                          , i_destock_date_zt)
    ## the change (reduction) in green and dry FOO on the arable and non-arable areas when pasture is destocked for spraying prior to reseeding
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = pfun.update_reseeding_foo(foo_grn_reseeding_flrzt
                                                                               , foo_dry_reseeding_flrzt
                                                                               ,              resown_rt
                                                                               ,               period_zt
                                                                               ,       1 - proportion_zt
                                                                               ,  -foo_arable_destock_zt
                                                                               ,      -foo_na_destock_zt) # Assumes that all feed lost is green

    ##FOO on the arable area of each LMU when reseeded pasture is restocked (this is calculated from input values)
    foo_arable_restock_lt =  i_restock_fooscalar_lt * i_restock_foo_arable_t

    ## calc foo on non arable area when the area is restocked after reseeding
    ### FOO on non-arable areas at restocking equals foo at destocking plus any germination occurring in the destocked period plus growth from destocking to grazing
    #### FOO at destocking is an input, allocate the input to the destocking feed period
    foo_na_destock_fzt[period_zt,z_idx[:,na],t_idx] = foo_na_destock_zt

    #### the period from destocking to restocking (for germination and growth)
    destock_duration_zt = i_restock_date_zt - i_destock_date_zt
    shape_fzt = feed_period_dates_fz.shape + (i_destock_date_zt.shape[-1],)
    periods_destocked_fzt = fun.range_allocation_np(feed_period_dates_fz[...,na]
                                                    ,   i_destock_date_zt
                                                    , destock_duration_zt
                                                    ,     shape=shape_fzt)[0:n_feed_periods,...]
    days_each_period_fzt = periods_destocked_fzt * length_fz[..., na]
    #### period when restocking occurs and the proportion through the period that it occurs
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na], i_restock_date_zt)

    ### germination during destocked period (this is the germination of pasture type 1 but it includes a t axis because the destocked period can vary with pasture type)
    germination_destocked_flzt = max_germination_flz[..., na] * periods_destocked_fzt[:, na, ...]

    ### Calculate the FOO profile on the non arable area from destocking through to restocking
    #### need to loop through t because FOO at destocking and reseeding date can change
    for t in range(n_pasture_types):
        ### green FOO to start the profile is FOO at destocking plus germination that occurs during the destocking period
        #### assumes FOO at destocking of pasture type 0 on the non arable area is equivalent to the pasture itself.
        grn_foo_na_initial_flzt = foo_na_destock_fzt[:, na, :, t:t + 1] + germination_destocked_flzt[..., t: t+1]
        ##FOO at the end of the destocked period is calculated from the FOO profile from destocking to restocking
        grn_restock, dry_restock = pfun.calc_foo_profile(grn_foo_na_initial_flzt  # axes are aligned in the function
                                                         , dry_decay_period_fzt[..., 0:1]
                                                         , days_each_period_fzt[...,t]
                                                         , i_fxg_foo_oflzt[..., 0:1]
                                                         , c_fxg_a_oflzt[..., 0:1]
                                                         , c_fxg_b_oflzt[..., 0:1]
                                                         , i_grn_senesce_eos_fzt[..., 0:1]
                                                         , grn_senesce_startfoo_fzt[..., 0:1]
                                                         , grn_senesce_pgrcons_fzt[..., 0:1])
        #### assign the growth to a variable to store all the pasture types
        grn_restock_foo_flzt[...,t:t+1] = grn_restock
        dry_restock_foo_flzt[...,t:t+1] = dry_restock
    ### combine dry and grn foo because the proportion of green at restocking is an input
    #### foo is calculated at the start of period, +1 to get end period FOO.
    foo_na_restock_lzt = grn_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx]   \
                        + dry_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx] #foo is calc at the start of period, +1 to get end period foo.

    ## increment the change in green and dry foo on the arable and non-arable areas when pasture is restocked after reseeding
    ### Note: the function call includes += for the green and dry foo variables
    ## combine the non-arable and arable foo to get the resulting foo in the green and dry pools when paddocks are restocked. Spread between periods based on date grazed. (arable proportion accounted for in function)
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = pfun.update_reseeding_foo(foo_grn_reseeding_flrzt  #axes aligned in function
                                                                               , foo_dry_reseeding_flrzt
                                                                               ,                 resown_rt
                                                                               ,               period_zt
                                                                               ,       1 - proportion_zt
                                                                               ,   foo_arable_restock_lt[:,na,:]
                                                                               ,       foo_na_restock_lzt
                                                                               , propn_grn=i_restock_grn_propn_t)

    ## split the change in dry FOO between the high & low quality FOO pools
    foo_dry_reseeding_dflrzt[0,...] = foo_dry_reseeding_flrzt * 0.5  # a 50% split assumes the dry feed removed at destocking and added at restocking is average quality.
    foo_dry_reseeding_dflrzt[1,...] = foo_dry_reseeding_flrzt * 0.5

    ### sow param determination
    ### determine the labour periods pas seeding occurs
    i_seeding_length_zt = i_reseeding_date_end_zt - i_reseeding_date_start_zt
    period_dates_p5z = per.p_dates_df().values
    shape_p5zt = period_dates_p5z.shape + (i_seeding_length_zt.shape[-1],)
    reseeding_machperiod_p5zt  = fun.range_allocation_np(        period_dates_p5z[...,na]
                                                        ,i_reseeding_date_start_zt
                                                        ,      i_seeding_length_zt
                                                        , True,   shape=shape_p5zt)
    ### combine with rotation reseeding requirement
    pas_sown_lrt = resown_rt * arable_l[:, na, na]
    pas_sow_plrzt = pas_sown_lrt[...,na,:] * reseeding_machperiod_p5zt[:, na, na,...]
    pas_sow_plrz = np.sum(pas_sow_plrzt, axis=-1) #sum the t axis. the different pastures are tracked by the rotation.
    pas_sow_plrkz = pas_sow_plrz[..., na,:] * (keys_k[:, na]==phases_rotn_df.iloc[:,-1].values[:, na,na]) #add k (landuse axis) this is required for sow param


    ## area of pasture being grazed and growing
    ### calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    arable_phase_area_flrzt = (1 - (resown_rt[:,na,:] * periods_destocked_fzt[:, na, na, ...]))  \
                             * arable_l[:, na, na, na] * pasture_rt[:, na, :]
    ###the non arable area is all allocated to the first pasture type (annuals)
    na_phase_area_flrzt[...,0] = np.sum((1 - (resown_rt[:,na,:] * periods_destocked_fzt[:, na, na, ...]))
                                        * (1 - arable_l[:, na, na, na]) * pasture_rt[:, na, :]
                                        , axis = -1)
    phase_area_flrzt = arable_phase_area_flrzt + na_phase_area_flrzt


    ############################################################
    ## erosion limit. The minimum FOO at the end of each period#
    ############################################################
    arable_erosion_flrt = i_lmu_conservation_flt[..., na,:]  \
                                    *  arable_l[:, na, na]  \
                                    * pasture_rt
    na_erosion_flrt[...,0] = np.sum(i_lmu_conservation_flt[..., na,:]
                                    *         (1-arable_l[:, na, na])
                                    *           pasture_rt
                                    , axis = -1)
    erosion_flrt = arable_erosion_flrt + na_erosion_flrt

    ##############
    ## PGR & FOO #
    ##############
    ''' Populates the parameter arrays for green and dry feed.

    Pasture growth, consumption and senescence of green feed.
    Consumption & deferment of dry feed.
    '''

    ## initialise numpy arrays used only in this method
    senesce_propn_dgoflzt      = np.zeros(dgoflzt, dtype = 'float64')
    nap_dflrzt                 = np.zeros(dflrzt,  dtype = 'float64')
    me_threshold_vfzt          = np.zeros(vfzt,    dtype = 'float64')   # the threshold for the EV pools which define the animals feed quality requirements

    ## create numpy array of threshold values from the ev dictionary
    ### note: v in pasture is f in StockGen and f in pasture is p6 in StockGen
    ev_cutoff_vfzt = np.swapaxes(ev['ev_cutoff_p6fz'][..., na], axis1=0, axis2=1)
    ev_max_vfzt = ev['ev_max_p6z'][na,...,na]
    ev = np.concatenate([ev_cutoff_vfzt, ev_max_vfzt], axis=0)
    me_threshold_vfzt[0:len_v1, ...] = ev #assign to all slices except confinement (if it is active)
    ### if the threshold is below the expected maintenance quality set to the maintenance quality
    ### switching from one below maintenance feed to another that is further below maintenance doesn't affect average efficiency
    me_threshold_vfzt[me_threshold_vfzt < i_fec_maintenance_t] = i_fec_maintenance_t

    ## dry, DM decline (high = low pools)
    #todo look at masking the dry transfer to only those periods that dry exist (decay eos > 0)
    dry_transfer_t_fzt = 1000 * (1-dry_decay_period_fzt)

    ## FOO on the non-arable areas in crop paddocks is ungrazed FOO of pasture type 0 (annual), therefore calculate the profile based on the pasture type 0 values
    grn_foo_start_ungrazed_flzt , dry_foo_start_ungrazed_flzt \
         = pfun.calc_foo_profile(max_germination_flz[..., na], dry_decay_period_fzt[..., 0:1], length_fz
                                 , i_fxg_foo_oflzt[..., 0:1], c_fxg_a_oflzt[..., 0:1], c_fxg_b_oflzt[..., 0:1]
                                 , i_grn_senesce_eos_fzt[..., 0:1], grn_senesce_startfoo_fzt[..., 0:1], grn_senesce_pgrcons_fzt[..., 0:1])

    ### non arable pasture becomes available to graze at the beginning of the first harvest period
    # harvest_period  = fun.period_allocation(pinp.period['feed_periods']['date'], range(len(pinp.period['feed_periods'])), pinp.period['harv_date']) #use range(len()) to get the row number that harvest occurs has to be row number not index name because it is used to index numpy below
    harv_period_z, harv_proportion_z = fun.period_proportion_np(feed_period_dates_fz, harv_date_z)
    index = pd.MultiIndex.from_arrays([keys_f[harv_period_z], keys_z])
    harvest_period_prop = pd.Series(harv_proportion_z, index=index).unstack()
    # params['p_harvest_period_prop']  = dict([(pinp.period['feed_periods'].index[harv_period_z], harv_proportion_z)])

    ### all pasture from na area goes into the Low pool (#1) because it is rank & low quality
    nap_dflrzt[0,harv_period_z,l_idx[:,na,na], r_idx[:,na], z_idx, 0] = (
                                            dry_foo_start_ungrazed_flzt[harv_period_z, l_idx[:,na], z_idx, 0][:,na,:]
                                           * (1-arable_l[:, na,na])
                                           * (1-np.sum(pasture_rt[:, na, :], axis=-1)))    # sum pasture proportion across the t axis to get area of crop

    ## green initial FOO for the 'grnha' decision variables
    foo_start_grnha_oflzt = i_fxg_foo_oflzt
#    foo_start_grnha_oflzt = np.maximum(i_fxg_foo_oflzt, i_base_ft[:, na, na, :])  # to ensure that final foo can not be below the base level
    max_foo_flzt                 = np.maximum(i_fxg_foo_oflzt[1,...], grn_foo_start_ungrazed_flzt)     #maximum of ungrazed foo and foo from the medium foo level
    foo_start_grnha_oflzt[2,...] = np.maximum.accumulate(max_foo_flzt,axis=0)                          #maximum accumulated along the feed periods axis, i.e. max to date
    # foo_start_grnha_oflt[...]   = np.maximum(foo_start_grnha_oflt
    #                                          , i_base_ft[:, na,:])         # to ensure that final foo can not be below 0
    foo_start_grnha_oflzt = foo_start_grnha_oflzt * mask_greenfeed_exists_fzt[:, na, ...]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.


    ## green, pasture growth for the 'grnha' decision variables
    pgr_grnday_oflzt = np.maximum(0.01, i_fxg_pgr_oflzt)                  # use maximum to ensure that the pgr is non zero (because foo_days requires dividing by pgr)
    pgr_grnha_goflzt = pgr_grnday_oflzt * length_fz[:,na, :, na] * c_pgr_gi_scalar_gft[:, na, :, na, na, :]

    ## green, final foo from initial, pgr and senescence
    ### foo at end of period if ungrazed
    foo_end_ungrazed_grnha_oflzt  = foo_start_grnha_oflzt * (1 - grn_senesce_startfoo_fzt[:, na, ...])   \
                                   + pgr_grnha_goflzt[0, ...] * (1 - grn_senesce_pgrcons_fzt[:, na, ...])
    ### foo at end of period with range of grazing intensity prior to eos senescence
    foo_endprior_grnha_goflzt = (foo_end_ungrazed_grnha_oflzt
                                 - (foo_end_ungrazed_grnha_oflzt - i_base_ft[:, na, na,: ])
                                 * i_foo_graze_propn_gt[:, na, na, na, na, :])
    senesce_eos_grnha_goflzt = foo_endprior_grnha_goflzt * i_grn_senesce_eos_fzt[:, na, ...]
    foo_end_grnha_goflzt = foo_endprior_grnha_goflzt - senesce_eos_grnha_goflzt
    #apply mask to remove any green foo at the end of period in periods when green pas doesnt exist.
    foo_end_grnha_goflzt = foo_end_grnha_goflzt * mask_greenfeed_exists_fzt[:, na, ...]

    ## green, removal & dmi
    ### divide by (1 - grn_senesce_pgrcons) to allows for consuming feed reducing senescence
    removal_grnha_goflzt =np.maximum(0, (foo_start_grnha_oflzt * (1 - grn_senesce_startfoo_fzt[:, na, ...])
                                         + pgr_grnha_goflzt * (1 - grn_senesce_pgrcons_fzt[:, na, :])
                                         - foo_endprior_grnha_goflzt)
                                        / (1 - grn_senesce_pgrcons_fzt[:, na, :]))
    cons_grnha_t_goflzt  = removal_grnha_goflzt / (1 + i_grn_trampling_ft[:, na, na, :])

    ## green, dmd & md from input values and impact of foo & grazing intensity
    ### sward digestibility is reduced with higher FOO (based on start FOO)
    ### diet digestibility is reduced with higher FOO if grazing intensity is greater than 25%
    #### Low FOO or low grazing intensity is input
    #### High FOO with 100% grazing is reduced by half the range in digestibility.
    #### Between low and high FOO, and between 25% & 100% grazing intensity is a linear interpolation
    dmd_sward_grnha_goflzt = (i_grn_dig_flzt - i_grn_dmd_range_ft[:, na, na, :] /2
                              * fun.f_divide(foo_start_grnha_oflzt - foo_start_grnha_oflzt[0,...]
                                              , foo_start_grnha_oflzt[-1,...] - foo_start_grnha_oflzt[0,...]))
    dmd_diet_grnha_goflzt = (i_grn_dig_flzt - i_grn_dmd_range_ft[:, na, na, :] /2
                             * fun.f_divide(foo_start_grnha_oflzt - foo_start_grnha_oflzt[0,...]
                                            , foo_start_grnha_oflzt[-1,...] - foo_start_grnha_oflzt[0,...])
                             * (i_foo_graze_propn_gt[:, na, na, na, na, :] - 0.25)/(1 - 0.25))  # 0.25 is grazing intensity that gives diet quality == input value.
    grn_md_grnha_goflzt = fun.dmd_to_md(dmd_diet_grnha_goflzt)

    ## green, mei & volume
    ###Average FOO is calculated using FOO at the end prior to EOS senescence (which assumes all pasture senesces after grazing)
    foo_ave_grnha_goflzt = (foo_start_grnha_oflzt + foo_endprior_grnha_goflzt)/2
    ### pasture params used to convert foo for rel availability
    pasture_stage_flzt = i_pasture_stage_p6z[:, na, :, na]
    ### adjust foo and calc hf
    foo_ave_grnha_goflzt, hf = sfun.f_foo_convert(cu3, cu4, foo_ave_grnha_goflzt, pasture_stage_flzt, i_legume_zt, z_pos=-2)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        grn_ri_availability_goflzt = sfun.f_ra_cs(foo_ave_grnha_goflzt, hf)
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        grn_ri_quality_goflzt = sfun.f_rq_cs(dmd_diet_grnha_goflzt, i_legume_zt)
    grn_ri_goflzt = sfun.f_rel_intake(grn_ri_availability_goflzt, grn_ri_quality_goflzt, i_legume_zt)

    me_cons_grnha_vgoflzt = fun.f_effective_mei( cons_grnha_t_goflzt
                                              ,  grn_md_grnha_goflzt
                                              ,   me_threshold_vfzt[:, na, na,:, na, ...]
                                              ,        grn_ri_goflzt
                                              , i_me_eff_gainlose_ft[:, na, na, :])
    me_cons_grnha_vgoflzt = me_cons_grnha_vgoflzt * mask_greenfeed_exists_fzt[:, na, ...]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    me_cons_grnha_vgoflzt = me_cons_grnha_vgoflzt * ev_is_not_confinement_v[:,na,na,na,na,na,na] #me from pasture is 0 in the confinement pool

    volume_grnha_goflzt    =  cons_grnha_t_goflzt / grn_ri_goflzt              # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
    volume_grnha_goflzt = volume_grnha_goflzt * mask_greenfeed_exists_fzt[:, na, ...]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.

    ## dry, dmd & foo of feed consumed
    ### do sensitivity adjustment for dry_dmd_input based on increasing/reducing the reduction in dmd from the maximum (starting value)
    dry_dmd_adj_fzt  = (i_dry_dmd_ave_fzt - np.max(i_dry_dmd_ave_fzt, axis=0)) * sen.sam['dry_dmd_decline','annual']
    dry_dmd_high_fzt = np.max(i_dry_dmd_ave_fzt, axis=0) + dry_dmd_adj_fzt + i_dry_dmd_range_fzt/2
    dry_dmd_low_fzt  = np.max(i_dry_dmd_ave_fzt, axis=0) + dry_dmd_adj_fzt - i_dry_dmd_range_fzt/2
    dry_dmd_dfzt     = np.stack((dry_dmd_low_fzt, dry_dmd_high_fzt), axis=0)    # create an array with a new axis 0 by stacking the existing arrays

    dry_foo_high_fzt = i_dry_foo_high_fzt * 3/4
    dry_foo_low_fzt  = i_dry_foo_high_fzt * 1/4                               # assuming half the foo is high quality and the remainder is low quality
    dry_foo_dfzt     = np.stack((dry_foo_low_fzt, dry_foo_high_fzt),axis=0)  # create an array with a new axis 0 by stacking the existing arrays

    ## dry, volume of feed consumed per tonne
    ### adjust foo and calc hf
    pasture_stage_fzt = i_pasture_stage_p6z[...,na]
    dry_foo_dfzt, hf = sfun.f_foo_convert(cu3, cu4, dry_foo_dfzt, pasture_stage_fzt, i_legume_zt, z_pos=-2)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        dry_ri_availability_dfzt = sfun.f_ra_cs(dry_foo_dfzt, hf)

    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        dry_ri_quality_dfzt = sfun.f_rq_cs(dry_dmd_dfzt, i_legume_zt)
    dry_ri_dfzt = sfun.f_rel_intake(dry_ri_availability_dfzt, dry_ri_quality_dfzt, i_legume_zt)  #set the minimum RI to 0.05

    dry_volume_t_dfzt = 1000 / dry_ri_dfzt                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed
    dry_volume_t_dfzt = dry_volume_t_dfzt * mask_dryfeed_exists_fzt  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.

    ## dry, ME consumed per kg consumed
    dry_md_dfzt        = fun.dmd_to_md(dry_dmd_dfzt)
    dry_md_vdfzt       = np.stack([dry_md_dfzt] * n_feed_pools, axis = 0)
    ## convert to effective quality per tonne
    dry_mecons_t_vdfzt = fun.f_effective_mei( 1000                                    # parameters for the dry feed grazing activities: Total ME of the tonne consumed
                               ,          dry_md_vdfzt
                               ,     me_threshold_vfzt[:, na, ...]
                               ,           dry_ri_dfzt
                               , i_me_eff_gainlose_ft[:,na,:])
    dry_mecons_t_vdfzt = dry_mecons_t_vdfzt * mask_dryfeed_exists_fzt  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    dry_mecons_t_vdfzt = dry_mecons_t_vdfzt * ev_is_not_confinement_v[:,na,na,na,na] #me from pasture is 0 in the confinement pool

    ## dry, animal removal
    dry_removal_t_ft  = 1000 * (1 + i_dry_trampling_ft)

    ## senescence from green to dry - green, total senescence for the period
    ## the senesced feed that is available to stock is that which senesces at the end of the growing season (i.e. not during the growing season)
    ##todo may need revisiting for perennial pastures where green & dry feed are part of a mixed diet.
    senesce_total_grnha_goflzt    = senesce_eos_grnha_goflzt
    grn_dmd_senesce_goflzt        =       dmd_sward_grnha_goflzt       \
                                   + i_grn_dmd_senesce_redn_fzt[:, na, ...]
    senesce_propn_dgoflzt[1,...]  = np.clip(( grn_dmd_senesce_goflzt                     # senescence to high pool. np.clip reduces the range of the dmd to the range of dmd in the dry feed pools
                                             -      dry_dmd_low_fzt[:, na, :])
                                            / (    dry_dmd_high_fzt[:, na,:]
                                               -    dry_dmd_low_fzt[:, na,:]), 0, 1)
    senesce_propn_dgoflzt[0,...] = 1- senesce_propn_dgoflzt[1,...]                       # senescence to low pool
    senesce_grnha_dgoflzt        = senesce_total_grnha_goflzt * senesce_propn_dgoflzt       # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    senesce_grnha_dgoflzt        = senesce_grnha_dgoflzt * mask_greenfeed_exists_fzt[:, na, ...]  # apply mask - green pasture only senesces when green pas exists.


    ######
    #poc #
    ######
    '''
        The amount of pasture consumption that can occur on crop paddocks each day before seeding
        - this is adjusted for lmu and feed period
        The quality of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
        The relative intake of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    ### poc is assumed to be annual hence the 0 slice in the last axis
    ## con
    poc_con_fl = i_poc_intake_daily_flt[..., 0] / 1000 #divide 1000 to convert to tonnes of foo per ha
    ## md per tonne
    poc_md_f = fun.dmd_to_md(i_poc_dmd_ft[..., 0]) * 1000 #times 1000 to convert to mj per tonne
    poc_md_vf = poc_md_f * ev_is_not_confinement_v[:,na] #me from pasture is 0 in the confinement pool

    ## vol
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        poc_ri_qual_fz = sfun.f_rq_cs(i_poc_dmd_ft[..., na, 0], i_legume_zt[..., 0])
    
    ### adjust foo and calc hf
    i_poc_foo_fz, hf = sfun.f_foo_convert(cu3, cu4, i_poc_foo_ft[:,na,0], i_pasture_stage_p6z, i_legume_zt[...,0], z_pos=-1)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        poc_ri_quan_fz = sfun.f_ra_cs(i_poc_foo_fz, hf)

    poc_ri_fz = sfun.f_rel_intake(poc_ri_quan_fz, poc_ri_qual_fz, i_legume_zt[..., 0])
    poc_vol_fz = fun.f_divide(1000, poc_ri_fz)  # 1000 to convert to vol per tonne


    ###########
    #params   #
    ###########
    ##non seasonal
    pasture_area = pasture_rt.ravel() * 1  # times 1 to convert from bool to int eg if the phase is pasture then 1ha of pasture is recorded.
    params['pasture_area_rt'] = dict(zip(index_rt,pasture_area))

    erosion_rav_flrt = erosion_flrt.ravel()
    params['p_erosion_flrt'] = dict(zip(index_flrt,erosion_rav_flrt))

    dry_removal_t_rav_ft = dry_removal_t_ft.ravel()
    params['p_dry_removal_t_ft'] = dict(zip(index_ft,dry_removal_t_rav_ft))

    poc_con_rav_fl = poc_con_fl.ravel()
    params['p_poc_con_fl'] = dict(zip(index_fl,poc_con_rav_fl))

    poc_md_rav_vf = poc_md_vf.ravel()
    params['p_poc_md_vf'] = dict(zip(index_vf,poc_md_rav_vf))

    ##create season params in loop
    for z in range(len(keys_z)):
        ##create season key for params dict
        params[keys_z[z]] = {}
        scenario = keys_z[z]

        ###create param from dataframe
        params[scenario]['p_harvest_period_prop'] = harvest_period_prop[scenario].to_dict()

        ###create param from numpy

        ##convert the change in dry and green FOO at destocking and restocking into a pyomo param (for the area that is resown)
        foo_dry_reseeding_rav_dflrt = foo_dry_reseeding_dflrzt[...,z,:].ravel()
        params[scenario]['p_foo_dry_reseeding_dflrt'] = dict(zip(index_dflrt,foo_dry_reseeding_rav_dflrt))
        foo_grn_reseeding_rav_flrt = foo_grn_reseeding_flrzt[...,z,:].ravel()
        params[scenario]['p_foo_grn_reseeding_flrt'] = dict(zip(index_flrt,foo_grn_reseeding_rav_flrt))

        pas_sow_rav_plrk = pas_sow_plrkz[...,z].ravel()
        params[scenario]['p_pas_sow_plrk'] = dict(zip(index_plrk,pas_sow_rav_plrk))

        phase_area_rav_flrt = phase_area_flrzt[...,z,:].ravel()
        params[scenario]['p_phase_area_flrt'] = dict(zip(index_flrt,phase_area_rav_flrt))

        dry_transfer_t_rav_ft = dry_transfer_t_fzt[...,z,:].ravel()
        params[scenario]['p_dry_transfer_t_ft'] = dict(zip(index_ft,dry_transfer_t_rav_ft))

        germination_rav_flrt = germination_flrzt[...,z,:].ravel()
        params[scenario]['p_germination_flrt'] = dict(zip(index_flrt,germination_rav_flrt))

        nap_rav_dflrt = nap_dflrzt[...,z,:].ravel()
        params[scenario]['p_nap_dflrt'] = dict(zip(index_dflrt,nap_rav_dflrt))

        foo_start_grnha_rav_oflt = foo_start_grnha_oflzt[...,z,:].ravel()
        params[scenario]['p_foo_start_grnha_oflt'] = dict(zip(index_oflt ,foo_start_grnha_rav_oflt))

        foo_end_grnha_rav_goflt = foo_end_grnha_goflzt[...,z,:].ravel()
        params[scenario]['p_foo_end_grnha_goflt'] = dict( zip(index_goflt ,foo_end_grnha_rav_goflt))

        me_cons_grnha_rav_vgoflt = me_cons_grnha_vgoflzt[...,z,:].ravel()
        params[scenario]['p_me_cons_grnha_vgoflt'] = dict(zip(index_vgoflt,me_cons_grnha_rav_vgoflt))

        dry_mecons_t_rav_vdft = dry_mecons_t_vdfzt[...,z,:].ravel()
        params[scenario]['p_dry_mecons_t_vdft'] = dict(zip(index_vdft,dry_mecons_t_rav_vdft))

        volume_grnha_rav_goflt = volume_grnha_goflzt[...,z,:].ravel()
        params[scenario]['p_volume_grnha_goflt'] = dict(zip(index_goflt,volume_grnha_rav_goflt))

        dry_volume_t_rav_dft = dry_volume_t_dfzt[...,z,:].ravel()
        params[scenario]['p_dry_volume_t_dft'] = dict(zip(index_dft,dry_volume_t_rav_dft))

        senesce_grnha_rav_dgoflt = senesce_grnha_dgoflzt[...,z,:].ravel()
        params[scenario]['p_senesce_grnha_dgoflt'] = dict(zip(index_dgoflt,senesce_grnha_rav_dgoflt))

        poc_vol_rav_f = poc_vol_fz[...,z].ravel()
        params[scenario]['p_poc_vol_f'] = dict(zip(keys_f,poc_vol_rav_f))

    ###########
    #report   #
    ###########
    ##keys
    r_vals['keys_d'] = keys_d
    r_vals['keys_v'] = keys_v
    r_vals['keys_f'] = keys_f
    r_vals['keys_g'] = keys_g
    r_vals['keys_l'] = keys_l
    r_vals['keys_o'] = keys_o
    r_vals['keys_p'] = keys_p
    r_vals['keys_r'] = keys_r
    r_vals['keys_t'] = keys_t
    r_vals['keys_k'] = keys_k

    ##store report vals
    r_vals['pasture_area_rt'] = pasture_rt
    r_vals['keys_pastures'] = pastures
    r_vals['days_p6z'] = length_fz

    r_vals['pgr_grnha_goflzt'] = pgr_grnha_goflzt #store for reporting
    r_vals['foo_end_grnha_goflzt'] = foo_endprior_grnha_goflzt #store for reporting. Green FOO prior to eos senescence
    r_vals['cons_grnha_t_goflzt'] = cons_grnha_t_goflzt #store for reporting
    r_vals['fec_grnha_vgoflzt'] = fun.f_divide(me_cons_grnha_vgoflzt, volume_grnha_goflzt) #store for reporting
    r_vals['fec_dry_vdfzt'] = fun.f_divide(dry_mecons_t_vdfzt, dry_volume_t_dfzt)  #store for reporting