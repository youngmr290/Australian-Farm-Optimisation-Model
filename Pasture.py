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
import Functions as fun
import Periods as per
import Sensitivity as sen
import PastureFunctions as pfun

def f_pasture(params, r_vals, ev):
    ######################
    ##background vars    #
    ######################
    na = np.newaxis

    ########################
    ##phases               #
    ########################
    ## read the rotation phases information from inputs
    phase_len       = uinp.structure['phase_len']
    phases_rotn_df  = uinp.structure['phases']
    pasture_sets    = uinp.structure['pasture_sets']
    pastures        = uinp.structure['pastures'][pinp.general['pas_inc']]

    ########################
    ##constants required   #
    ########################
    ## define some parameters required to size arrays.
    n_feed_pools    = len(uinp.structure['sheep_pools'])
    n_dry_groups    = len(uinp.structure['dry_groups'])           # Low & high quality groups for dry feed
    n_grazing_int   = len(uinp.structure['grazing_int'])          # grazing intensity in the growth/grazing activities
    n_foo_levels    = len(uinp.structure['foo_levels'])           # Low, medium & high FOO level in the growth/grazing activities
    n_feed_periods  = len(per.f_feed_periods()) - 1
    n_lmu           = len(pinp.general['lmu_area'])
    n_phases_rotn   = len(phases_rotn_df.index)
    n_pasture_types = len(pastures)   #^ need to sort timing of the definition of pastures

    index_f = np.arange(n_feed_periods)

    # i_feed_period_dates   = list(pinp.period['feed_periods']['date']) #todo this will need z axis. it is a df so convert to numpy: idx = pd.IndexSlice then: df.loc[idx[:, 'date'], :].values
    t_list = np.arange(n_pasture_types)

    arable_l = np.array(pinp.crop['arable']).reshape(-1)
    # length_f  = np.array(pinp.period['feed_periods'].loc[:pinp.period['feed_periods'].index[-2],'length']) # not including last row because that is the start of the following year. #todo as above this will need z axis
    # feed_period_dates_f = np.array(i_feed_period_dates,dtype='datetime64[D]')
    length_fz  = np.array(per.f_feed_periods(option=1),dtype='float64')
    feed_period_dates_fz = np.array(per.f_feed_periods(option=2),dtype='datetime64[D]')


    vgoflt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dgoflt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    vdft   = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
    vft    = (n_feed_pools, n_feed_periods, n_pasture_types)
    dft    = (n_dry_groups, n_feed_periods, n_pasture_types)
    goflt  = (n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    goft   = (n_grazing_int, n_foo_levels, n_feed_periods, n_pasture_types)
    gft    = (n_grazing_int, n_feed_periods, n_pasture_types)
    gt     = (n_grazing_int, n_pasture_types)
    oflt   = (n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dflrt  = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    flrt   = (n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    frt    = (n_feed_periods, n_phases_rotn, n_pasture_types)
    rt     = (n_phases_rotn, n_pasture_types)
    flt    = (n_feed_periods, n_lmu, n_pasture_types)
    lt     = (n_lmu, n_pasture_types)
    ft     = (n_feed_periods, n_pasture_types)
    # t      = (n_pasture_types)

    ## define the vessels that will store the input data that require pre-defining
    ### all need pre-defining because inputs are in separate pasture type arrays
    i_phase_germ_dict = dict()
    i_grn_senesce_daily_ft          = np.zeros(ft,  dtype = 'float64')  # proportion of green feed that senesces each period (due to leaf drop)
    # i_grn_senesce_eos_ft            = np.zeros(ft,  dtype = 'float64')  # proportion of green feed that senesces in period (due to completing life cycle)
    dry_decay_daily_ft              = np.zeros(ft,  dtype = 'float64')  # daily decline in dry foo in each period
    i_end_of_gs_t                   = np.zeros(n_pasture_types, dtype = 'int')  # the period number when the pasture senesces
    i_dry_decay_t                   = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    i_me_maintenance_vft            = np.zeros(vft,  dtype = 'float64')  # M/D level for target LW pattern
    c_pgr_gi_scalar_gft             = np.zeros(gft,  dtype = 'float64')  # numpy array of pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_graze_propn_gt            = np.zeros(gt, dtype ='float64')  # numpy array of proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_oflt                  = np.zeros(oflt, dtype = 'float64')  # numpy array of FOO level       for the FOO/growth/grazing variables.
    i_fxg_pgr_oflt                  = np.zeros(oflt, dtype = 'float64')  # numpy array of PGR level       for the FOO/growth/grazing variables.
    c_fxg_a_oflt                    = np.zeros(oflt, dtype = 'float64')  # numpy array of coefficient a   for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_oflt                    = np.zeros(oflt, dtype = 'float64')  # numpy array of coefficient b   for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_oflt                   = np.zeros(oflt, dtype = 'float64')  # numpy array of coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_oflt                   = np.zeros(oflt, dtype = 'float64')  # numpy array of coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_flt                   = np.zeros(flt, dtype = 'float64')  # numpy array of inputs for green pasture digestibility on each LMU.
    i_poc_intake_daily_flt          = np.zeros(flt, dtype = 'float64')  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_flt          = np.zeros(flt, dtype = 'float64')  # minimum foo prior at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lt                = np.zeros(lt,  dtype = 'float64')  # scale the germination levels for each lmu
    i_reseeding_fooscalar_lt        = np.zeros(lt,  dtype = 'float64')  # scalar for FOO at the first grazing for the lmus
    i_dry_dmd_reseeding_lt          = np.zeros(lt,  dtype = 'float64')  # Average digestibility of any dry FOO at the first grazing (if there is any)

    i_me_eff_gainlose_ft            = np.zeros(ft,  dtype = 'float64')  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_ft              = np.zeros(ft,  dtype = 'float64')  # numpy array of inputs for green pasture trampling in each feed period.
    i_dry_trampling_ft              = np.zeros(ft,  dtype = 'float64')  # numpy array of inputs for dry pasture trampling   in each feed period.
    i_grn_senesce_eos_ft            = np.zeros(ft,  dtype = 'float64')  # proportion of green feed that senesces in period (due to completing life cycle)
    i_base_ft                       = np.zeros(ft,  dtype = 'float64')  # lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_ft         = np.zeros(ft,  dtype = 'float64')  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_ft              = np.zeros(ft,  dtype = 'float64')  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_ft       = np.zeros(ft,  dtype = 'float64')  # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_ft                = np.zeros(ft,  dtype = 'float64')  # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_ft              = np.zeros(ft,  dtype = 'float64')  # range in digestibility of dry feed if it is not grazed
    i_dry_foo_high_ft               = np.zeros(ft,  dtype = 'float64')  # expected foo for the dry pasture in the high quality pool
    dry_decay_period_ft             = np.zeros(ft,  dtype = 'float64')  # decline in dry foo for each period
    mask_dryfeed_exists_ft          = np.zeros(ft,  dtype = bool)       # mask for period when dry feed exists
    mask_greenfeed_exists_ft        = np.zeros(ft,  dtype = bool)       # mask for period when green feed exists
    i_grn_cp_ft                     = np.zeros(ft,  dtype = 'float64')  # crude protein content of green feed
    i_dry_cp_ft                     = np.zeros(ft,  dtype = 'float64')  # crude protein content of dry feed
    i_poc_dmd_ft                    = np.zeros(ft,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_ft                    = np.zeros(ft,  dtype = 'float64')  # foo of pasture consumed on crop paddocks
    grn_senesce_startfoo_ft         = np.zeros(ft,  dtype = 'float64')  # proportion of the FOO at the start of the period that senesces during the period
    grn_senesce_pgrcons_ft          = np.zeros(ft,  dtype = 'float64')  # proportion of the (total or average daily) PGR that senesces during the period (consumption leads to a reduction in senescence)

    i_reseeding_date_seed_t         = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of seeding this pasture type (will be read in from inputs)
    i_seeding_end_t                 = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of seeding this pasture type (will be read in from inputs)
    i_reseeding_date_destock_t      = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of destocking this pasture type prior to reseeding (will be read in from inputs)
    i_reseeding_ungrazed_destock_t  = np.zeros(n_pasture_types, dtype = 'float64')  # kg of FOO that was not grazed prior to seeding occurring (if spring sown)
    i_reseeding_date_grazing_t      = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of first grazing of reseeded pasture (will be read in from inputs)
    i_reseeding_foo_grazing_t       = np.zeros(n_pasture_types, dtype = 'float64')  # FOO at time of first grazing
    # reseeding_machperiod_t          = np.zeros(n_pasture_types, dtype = 'float64')  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
    i_germination_std_t             = np.zeros(n_pasture_types, dtype = 'float64')  # standard germination level for the standard soil type in a continuous pasture rotation
    # i_ri_foo_t                      = np.zeros(n_pasture_types, dtype = 'float64')  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    # poc_days_of_grazing_t           = np.zeros(n_pasture_types, dtype = 'float64')  # number of days after the pasture break that (moist) seeding can begin
    i_legume_t                      = np.zeros(n_pasture_types, dtype = 'float64')  # proportion of legume in the sward
    i_grn_propn_reseeding_t         = np.zeros(n_pasture_types, dtype = 'float64')  # Proportion of the FOO available at the first grazing that is green
    i_fec_maintenance_t             = np.zeros(n_pasture_types, dtype = 'float64')  # approximate M/D for maintenance

    ### define the numpy arrays that will be the output from the pre-calcs for pyomo
    germination_flrt              = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: germination (kg/ha)
    foo_grn_reseeding_flrt        = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: feed lost and gained during destocking and then grazing of resown pasture (kg/ha)
    foo_dry_reseeding_flrt        = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: high quality dry feed gained from grazing of resown pasture (kg/ha)
    foo_dry_reseeding_dflrt       = np.zeros(dflrt, dtype = 'float64')  # parameters for rotation phase variable: low & high quality dry feed gained from grazing of resown pasture (kg/ha)
    dry_removal_t_dft             = np.zeros(dft,   dtype = 'float64')  # parameters for the dry feed grazing activities: Total DM removal from the tonne consumed (includes trampling)

    ### define the array that links rotation phase and pasture type
    pasture_rt                    = np.zeros(rt, dtype = 'float64')


    ## create numpy index for param dicts ^creating indexes is a bit slow
    ### the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
    keys_d                       = np.asarray(uinp.structure['dry_groups'])
    keys_v                       = np.asarray(uinp.structure['sheep_pools'])
    keys_f                       = np.asarray(per.f_feed_periods().index[:-1])
    keys_g                       = np.asarray(uinp.structure['grazing_int'])
    keys_l                       = pinp.general['lmu_area'].index.to_numpy() # lmu index description
    keys_o                       = np.asarray(uinp.structure['foo_levels'])
    keys_p                       = np.asarray(per.p_date2_df().index)
    keys_r                       = phases_rotn_df.index.to_numpy()
    keys_t                       = np.asarray(pastures)                      # pasture type index description
    keys_k                       = np.asarray(list(uinp.structure['All']))   #landuse

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

    ### flt
    arrays=[keys_f, keys_l, keys_t]
    index_flt=fun.cartesian_product_simple_transpose(arrays)
    index_flt=tuple(map(tuple, index_flt)) #create a tuple rather than a list because tuples are faster

    ### fl
    arrays=[keys_f, keys_l]
    index_fl=fun.cartesian_product_simple_transpose(arrays)
    index_fl=tuple(map(tuple, index_fl)) #create a tuple rather than a list because tuples are faster

    ### ft
    arrays=[keys_f, keys_t]
    index_ft=fun.cartesian_product_simple_transpose(arrays)
    index_ft=tuple(map(tuple, index_ft)) #create a tuple rather than a list because tuples are faster

    ### frt
    arrays=[keys_f, keys_r, keys_t]
    index_frt=fun.cartesian_product_simple_transpose(arrays)
    index_frt=tuple(map(tuple, index_frt)) #create a tuple rather than a list because tuples are faster

    ###########
    #map_excel#
    ###########
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''
#     global grn_senesce_startfoo_ft
#     global grn_senesce_pgrcons_ft
# #    global i_end_of_gs_t
#     global t_list


    ## map data from excel file into arrays
    ### loop through each pasture type
    for t, pasture in enumerate(pastures):
        exceldata = pinp.pasture_inputs[pasture]           # assign the pasture data to exceldata
        ## map the Excel data into the numpy arrays
        i_germination_std_t[t]              = exceldata['GermStd']
        # i_ri_foo_t[t]                       = exceldata['RIFOO']
        i_end_of_gs_t[t]                    = exceldata['EndGS']
        i_dry_decay_t[t]                    = exceldata['PastDecay']
        # poc_days_of_grazing_t[t]            = exceldata['POCDays']
        i_poc_intake_daily_flt[...,t]       = exceldata['POCCons']
        i_legume_t[t]                       = exceldata['Legume']
        i_grn_propn_reseeding_t[t]          = exceldata['FaG_PropnGrn']

        i_grn_dmd_senesce_redn_ft[...,t]    = exceldata['DigRednSenesce']
        i_dry_dmd_ave_ft[...,t]             = exceldata['DigDryAve']
        i_dry_dmd_range_ft[...,t]           = exceldata['DigDryRange']
        i_dry_foo_high_ft[...,t]            = exceldata['FOODryH']
        i_grn_cp_ft[...,t]                  = exceldata['CPGrn']
        i_dry_cp_ft[...,t]                  = exceldata['CPDry']
        i_poc_dmd_ft[...,t]                 = exceldata['DigPOC']
        i_poc_foo_ft[...,t]                 = exceldata['FOOPOC']
        i_germ_scalar_lt[...,t]             = exceldata['GermScalarLMU']
        i_reseeding_fooscalar_lt[...,t]     = exceldata['FaG_LMU']
        i_dry_dmd_reseeding_lt[...,t]       = exceldata['FaG_digDry']

        i_lmu_conservation_flt[...,t]       = exceldata['ErosionLimit']

        i_reseeding_date_seed_t[t]          = exceldata['Date_Seeding']
        i_seeding_end_t[t]                  = exceldata['seeding_length']
        i_reseeding_date_destock_t[t]       = exceldata['Date_Destocking']
        i_reseeding_ungrazed_destock_t[t]   = exceldata['FOOatSeeding'] #ungrazed foo when destocked for reseeding
        i_reseeding_date_grazing_t[t]       = exceldata['Date_ResownGrazing']
        i_reseeding_foo_grazing_t[t]        = exceldata['FOOatGrazing']

        i_grn_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_ft[...,t]       = np.asfarray(exceldata['SenescePropn'])
        i_grn_senesce_eos_ft[...,t]         = np.asfarray(exceldata['SenesceEOS'])
        i_base_ft[...,t]                    = np.asfarray(exceldata['BaseLevelInput'])
        i_grn_dmd_declinefoo_ft[...,t]      = np.asfarray(exceldata['DigDeclineFOO'])
        i_grn_dmd_range_ft[...,t]           = np.asfarray(exceldata['DigSpread'])
        i_foo_graze_propn_gt[..., t]        = np.asfarray(exceldata['FOOGrazePropn'])
        #### impact of grazing intensity (at the other levels) on PGR during the period
        c_pgr_gi_scalar_gft[...,t]    = 1 - i_foo_graze_propn_gt[..., t].reshape(-1, 1) ** 2 \
                                        * (1 - np.asfarray(exceldata['PGRScalarH']))

        i_fxg_foo_oflt[0,:,:,t]             = exceldata['LowFOO'].to_numpy()
        i_fxg_foo_oflt[1,:,:,t]             = exceldata['MedFOO'].to_numpy()
        i_me_eff_gainlose_ft[...,t]         = exceldata['MaintenanceEff'].iloc[:,0].to_numpy()
        # i_me_maintenance_vft[...,t]         = exceldata['MaintenanceEff'].iloc[:,1:].to_numpy().T
        i_fec_maintenance_t[t]               = exceldata['MaintenanceFEC']
        ## # i_fxg_foo_oflt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_oflt[0,:,:,t]             = exceldata['LowPGR'].to_numpy()
        i_fxg_pgr_oflt[1,:,:,t]             = exceldata['MedPGR'].to_numpy()
        i_fxg_pgr_oflt[2,:,:,t]             = exceldata['MedPGR'].to_numpy()  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_flt[...,t]                = exceldata['DigGrn'].to_numpy()  # numpy array of inputs for green pasture digestibility on each LMU.

        i_phase_germ_dict[pasture]          = exceldata['GermPhases'].copy()  #DataFrame with germ scalar and resown
        i_phase_germ_dict[pasture].reset_index(inplace=True)                                                  # replace index read from Excel with numbers to match later merging
        i_phase_germ_dict[pasture].columns.values[range(phase_len)] = [*range(phase_len)]         # replace the pasture columns read from Excel with numbers to match later merging

        ### define the link between rotation phase and pasture type while looping on pasture
        pasture_rt[:,t] = phases_rotn_df.iloc[:,-1].isin(pasture_sets[pasture])
    ##create pasture area param. used to bound SR.
    pasture_area = pasture_rt.ravel() * 1 #times 1 to convert from bool to int eg if the phase is pasture then 1ha of pasture is recorded.
    params['pasture_area_rt'] = dict(zip(index_rt ,pasture_area))
    ## one time data manipulation for the inputs just read
    ### calculate dry_decay_period (used in reseeding and green&dry)
    dry_decay_daily_ft[...] = i_dry_decay_t
    for t in range(n_pasture_types):
        dry_decay_daily_ft[0:i_end_of_gs_t[t],t_list[t]] = 1
    dry_decay_period_ft[...] = 1 - (1 - dry_decay_daily_ft)               \
                              ** length_f.reshape(-1,1)

    ###create dry pasture exists mask - in the current structure dry pasture only exists after the growing season.
    # todo this is a limitation of pasture (green and dry pasture don't exist simultaneously) this is okay for wa but may need work for places with perennials.
    mask_dryfeed_exists_ft[...] = index_f[:, na] > i_end_of_gs_t   #green exists in the period which is the end of growing season hence >
    mask_greenfeed_exists_ft[...] = np.logical_not(mask_dryfeed_exists_ft)

    ###create equation coefficients for pgr = a+b*foo
    i_fxg_foo_oflt[2,...]  = 100000 #large number so that the np.searchsorted doesn't go above
    c_fxg_b_oflt[0,...] =  i_fxg_pgr_oflt[0,...]       \
                         / i_fxg_foo_oflt[0,...]
    c_fxg_b_oflt[1,...] = (   i_fxg_pgr_oflt[1,...]
                           -  i_fxg_pgr_oflt[0,...])        \
                           /( i_fxg_foo_oflt[1,...]
                             -i_fxg_foo_oflt[0,...])
    c_fxg_b_oflt[2,...] =  0
    c_fxg_a_oflt[0,...] =  0
    c_fxg_a_oflt[1,...] =  i_fxg_pgr_oflt[0,...]        \
                         -   c_fxg_b_oflt[1,...]        \
                         * i_fxg_foo_oflt[0,...]
    c_fxg_a_oflt[2,...] =  i_fxg_pgr_oflt[1,...] # because slope = 0

    # proportion of start foo that senesces during the period, different formula than excel
    grn_senesce_startfoo_ft =1 - ((1 -     i_grn_senesce_daily_ft) **  length_f.reshape(-1,1))
    # average senescence over the period for the growth and consumption
    grn_senesce_pgrcons_ft  =1 - ((1 -(1 - i_grn_senesce_daily_ft) ** (length_f.reshape(-1,1)+1))
                                  /        i_grn_senesce_daily_ft-1) / length_f.reshape(-1,1)

    ##store report vals
    r_vals['pasture_area_rt'] = pasture_rt
    r_vals['keys_pastures'] = pastures
    r_vals['days_p6'] = length_f
    return

    #################################################
    #Calculate germination and reseeding parameters #
    #################################################
    ''' 
    Calculate germination and reseeding parameters

    germination: create an array called p_germination_flrt being the parameters to be passed to pyomo.
    reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    Results are stored in p_...._reseeding

    requires phases_rotn_df as a global variable
    ^ currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed
    ## intermediate calculations are not stored, however, if they were stored the 'key variables' could change the values of the intermediate calcs which could then be fed into the parameter calculations (in a separate method)
    ## the above would provide more options for KVs and provide another step that may not need to be recalculated
    '''

    ## define instantiate arrays that are assigned in slices
    na_erosion_flrt     = np.zeros(flrt, dtype = 'float64')
    na_phase_area_flrt  = np.zeros(flrt, dtype = 'float64')
    grn_destock_foo_flt = np.zeros(flt,  dtype = 'float64')
    dry_destock_foo_flt = np.zeros(flt,  dtype = 'float64')
#    foo_na_flt          = np.zeros(flt,   dtype = 'float64')
    foo_na_destock_ft   = np.zeros(ft,   dtype = 'float64')
    germ_scalar_rt      = np.zeros(rt, dtype='float64')
    resown_rt           = np.zeros(rt, dtype='int')

    # ### set initial values to 0 because function is called multiple times
    # foo_grn_reseeding_flrt[...] = 0          # array has been initialised, reset all values to 0
    # foo_dry_reseeding_flrt[...] = 0
    # germination_flrt[...]       = 0

    ### set variables used in multiple locations
    destock_duration_t = (i_reseeding_date_grazing_t
                        - i_reseeding_date_destock_t)
    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df


    ## germination and resowing to the rotation phases
    for t, pasture in enumerate(pastures):
        phase_germresow_df['germ_scalar']=0 #set default to 0
        phase_germresow_df['resown']=False #set default to false
        ###loop through each combo in of landuses and pastures (i_phase_germ), then check which rotations fall into each germ/resowing category. Then populate the rot phase df with the necessary germination and resowing param.
        for ix_row in i_phase_germ_dict[pasture].index:
            ix_bool = pd.Series(data=True,index=range(len(phase_germresow_df)))
            for ix_col in range(i_phase_germ_dict[pasture].shape[1]-2):    #-2 because two of the cols are germ and resowing
                c_set = uinp.structure[i_phase_germ_dict[pasture].iloc[ix_row,ix_col]]
                ix_bool &= phase_germresow_df.loc[:,ix_col].reset_index(drop=True).isin(c_set) #had to drop index so that it would work (just said false when the index was different between series)
            ### maps the relevant germ scalar and resown bool to the rotation phase
            phase_germresow_df.loc[list(ix_bool),'germ_scalar'] = i_phase_germ_dict[pasture].loc[ix_row, 'germ_scalar']  #have to make bool into a list for some reason it doesn't like a series
            phase_germresow_df.loc[list(ix_bool),'resown'] = i_phase_germ_dict[pasture].loc[ix_row, 'resown']
        ### Convert germ and resow into a numpy - each pasture goes in a different slice
        germ_scalar_rt[:,t] = phase_germresow_df['germ_scalar'].to_numpy()#.reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
        resown_rt[:,t] = phase_germresow_df['resown'].to_numpy()#.reshape(-1,1)                       # extract the resown boolean from the dataframe and transpose (reshape to a column vector)

    ## germination on the arable area based on std germ, the rotation scalar and the lmu scalar
    arable_germination_lrt = i_germination_std_t        \
                            * i_germ_scalar_lt[:, na,:]  \
                            * germ_scalar_rt      # create an array rot phase x lmu
    arable_germination_lrt[np.isnan(arable_germination_lrt)]  = 0.0
    ## germination on the non arable area based on annual pasture
    na_germination_lr = np.sum(i_germination_std_t[0:1]
                               * i_germ_scalar_lt[:, na,0:1]
                               * pasture_rt, axis = -1)
    germination_flrt[0,...]    = arable_germination_lrt \
                                * arable_l.reshape(-1,1,1)     # set germination in first period to germination on arable area
    germination_flrt[0,...,0] += na_germination_lr  \
                                * (1-arable_l.reshape(-1,1)) # add non-arable area
    ##create the pyomo parameter
    germination_rav_flrt = germination_flrt.ravel()
    params['p_germination_flrt'] = dict(zip(index_flrt ,germination_rav_flrt))

    ## reseeding impacts due to destocking & then grazing
    ### calculate the green feed lost when pasture is destocked. Spread between periods based on date destocked
    ### # include proportion arable
    foo_arable_destock_t = i_reseeding_ungrazed_destock_t
    foo_na_destock_t =  i_reseeding_ungrazed_destock_t
    period_t, proportion_t = fun.period_proportion_np(feed_period_dates_f  # which feed period does destocking occur & the proportion that destocking occurs during the period.
                                                      ,i_reseeding_date_destock_t)
    foo_grn_reseeding_flrt, foo_dry_reseeding_flrt = pfun.update_reseeding_foo(foo_grn_reseeding_flrt, foo_dry_reseeding_flrt,
                                                                               resown_rt, period_t, 1-proportion_t,
                                                                               -foo_arable_destock_t, -foo_na_destock_t) # call function to remove the FOO lost for the periods. Assumed that all feed lost is green

    ### calculate the green & dry foo when pasture first grazed after reseeding. Spread between periods based on date grazed
    #### foo for the reseeded area is simply an input adjusted for each lmu
    foo_arable_reseed_lt = i_reseeding_fooscalar_lt       \
                          * i_reseeding_foo_grazing_t                 # FOO at the first grazing for each lmu (kg/ha)
    #### for non arable area (not resown) foo at first grazing is foo at destocking plus growth from destocking to grazing
    foo_na_destock_ft[period_t,t_list] = foo_na_destock_t
    #####  growth from destocking to grazing
    periods_destocked_ft = fun.range_allocation_np(feed_period_dates_f
                    , i_reseeding_date_destock_t
                    , destock_duration_t)[0:n_feed_periods]
    days_each_period_ft = periods_destocked_ft  \
                         * length_f[:, na]
    period_t, proportion_t = fun.period_proportion_np(feed_period_dates_f
                                                      ,i_reseeding_date_grazing_t)       # which feed period does grazing occur
    ##### growth of annual but with start foo varying with pasture type
    for t in range(n_pasture_types):
        foo_na_flt = foo_na_destock_ft[:, na,t:t+1]  #broadcast into array
        grn_destock, dry_destock = pfun.calc_foo_profile(foo_na_flt, dry_decay_period_ft, days_each_period_ft[t],
                                                         i_fxg_foo_oflt, c_fxg_a_oflt, c_fxg_b_oflt, i_grn_senesce_eos_ft,
                                                         grn_senesce_startfoo_ft, grn_senesce_pgrcons_ft)
        ###### assign the growth from the annual slice
        grn_destock_foo_flt[...,t] = grn_destock[...,0]
        dry_destock_foo_flt[...,t] = dry_destock[...,0]
    grn_destock_foo_lt = grn_destock_foo_flt[period_t+1,...]
    dry_destock_foo_lt = dry_destock_foo_flt[period_t+1,...]
    foo_na_reseed_lt =  grn_destock_foo_lt + dry_destock_foo_lt

    ###combine the non-arable and arable foo to get the resulting foo in the green and dry pools after reseeding
    foo_grn_reseeding_flrt, foo_dry_reseeding_flrt = pfun.update_reseeding_foo(foo_grn_reseeding_flrt, foo_dry_reseeding_flrt,
                                                                               resown_rt, period_t, 1-proportion_t, foo_arable_reseed_lt,
                                                                               foo_na_reseed_lt, propn_grn=i_grn_propn_reseeding_t) # call function to update green & dry feed in the periods.

    ##convert dry seeding pas into pyomo param
    foo_dry_reseeding_dflrt[0,...] = foo_dry_reseeding_flrt
    foo_dry_reseeding_dflrt[1,...] = foo_dry_reseeding_flrt
    foo_dry_reseeding_rav_dflrt = foo_dry_reseeding_dflrt.ravel()
    params['p_foo_dry_reseeding_dflrt'] = dict(zip(index_dflrt ,foo_dry_reseeding_rav_dflrt))
    ##convert green seeding pas into pyomo param
    foo_grn_reseeding_rav_flrt = foo_grn_reseeding_flrt.ravel()
    params['p_foo_grn_reseeding_flrt'] = dict(zip(index_flrt ,foo_grn_reseeding_rav_flrt))

    ### sow param determination
    ### determine the labour periods pas seeding occurs
    i_seeding_length_t = i_seeding_end_t - i_reseeding_date_seed_t
    period_dates            = per.p_dates_df()['date']
    reseeding_machperiod_pt  = fun.range_allocation_np(period_dates
                                    ,i_reseeding_date_seed_t
                                    ,i_seeding_length_t
                                    ,True)
    ### combine with rotation reseeding requirement
    pas_sown_lrt = resown_rt * arable_l.reshape(-1,1,1)
    pas_sow_plrt = pas_sown_lrt * reseeding_machperiod_pt[:, na, na,:]
    pas_sow_plr = np.sum(pas_sow_plrt, axis=-1) #sum the t axis. the different pastures are tracked by the rotation.
    pas_sow_plrk = pas_sow_plr[..., na] * (keys_k==phases_rotn_df.iloc[:,-1].values[:, na]) #add k (landuse axis) this is required for sow param
    pas_sow_rav_plrk = pas_sow_plrk.ravel()
    params['p_pas_sow_plrk'] = dict(zip(index_plrk ,pas_sow_rav_plrk))

    ## area of pasture being grazed and growing
    ### calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    arable_phase_area_flrt = (1-(resown_rt
                                 * periods_destocked_ft[:, na, na,:]))  \
                            * arable_l.reshape(-1,1,1)  \
                            * pasture_rt
    na_phase_area_flrt[...,0] = np.sum((1-(resown_rt
                                           * periods_destocked_ft[:, na, na,:]))
                                       * (1-arable_l.reshape(-1,1,1))
                                       * pasture_rt
                                       , axis = -1)
    phase_area_flrt = arable_phase_area_flrt + na_phase_area_flrt
    phase_area_rav_flrt = phase_area_flrt.ravel()
    params['p_phase_area_flrt'] = dict(zip(index_flrt ,phase_area_rav_flrt))

    ############
    ## erosion #
    ############
    arable_erosion_flrt = i_lmu_conservation_flt[..., na,:]  \
                         * arable_l.reshape(-1,1,1)  \
                         * pasture_rt
    na_erosion_flrt[...,0] = np.sum(i_lmu_conservation_flt[..., na,:]
                                    * (1-arable_l.reshape(-1,1,1))
                                    * pasture_rt
                                    , axis = -1)
    erosion_flrt = arable_erosion_flrt + na_erosion_flrt
    erosion_rav_flrt = erosion_flrt.ravel()
    params['p_erosion_flrt'] = dict(zip(index_flrt ,erosion_rav_flrt))

    ##############
    ## PGR & FOO #
    ##############
    ''' Populates the parameter arrays for green and dry feed.

    Pasture growth, consumption and senescence of green feed.
    Consumption & deferment of dry feed.
    '''

    ## initialise numpy arrays used only in this method
    grn_dmd_selectivity_goflt = np.zeros(goflt,  dtype = 'float64')
    senesce_propn_dgoflt      = np.zeros(dgoflt, dtype = 'float64')
    nap_dflrt                 = np.zeros(dflrt,  dtype = 'float64')
    dry_transfer_t_dft        = np.zeros(dft,    dtype = 'float64')
    me_maintenance_vft        = np.zeros(vft,    dtype = 'float64')
#    foo_start_grnha_oflt      = np.zeros(oflt,   dtype = 'float64')

    ## create numpy array of threshold values from the ev dictionary
    me_maintenance_vft[0:-1, ...] = ev['ev_cutoff_p6f'].T[..., na]
    me_maintenance_vft[-1, ...] = ev['ev_max_p6'][..., na]
    me_maintenance_vft[me_maintenance_vft < i_fec_maintenance_t] = i_fec_maintenance_t

    ## dry, DM decline (high = low pools)
    dry_transfer_t_dft[...] = 1000 * (1-dry_decay_period_ft)
    dry_transfer_t_rav_dft = dry_transfer_t_dft.ravel()
    params['p_dry_transfer_t_dft'] = dict(zip(index_dft ,dry_transfer_t_rav_dft))

    ## non-arable areas in crop paddocks (the annual pasture available if not grazed)
    ### # is maximum ungrazed pasture in the growing season
    ### # _maximum foo achievable for each lmu & feed period (ungrazed pasture that germinates at the maximum level on that lmu)
    ### non arable pasture becomes available to graze at the beginning of the first harvest period
    ### ^a potential error here when germination is spread across periods (because taking max of each period)
    germination_pass_flt = np.max(germination_flrt, axis=2)  #use p_germination because it includes any sensitivity that is carried out
    grn_foo_start_ungrazed_flt , dry_foo_start_ungrazed_flt \
         = pfun.calc_foo_profile(germination_pass_flt, dry_decay_period_ft, length_f,
                                 i_fxg_foo_oflt, c_fxg_a_oflt, c_fxg_b_oflt, i_grn_senesce_eos_ft,
                                 grn_senesce_startfoo_ft, grn_senesce_pgrcons_ft)
    ### all pasture from na area into the Low pool (#1) because it is rank
    harvest_period  = fun.period_allocation(pinp.period['feed_periods']['date'], range(len(pinp.period['feed_periods'])), pinp.period['harv_date']) #use range(len()) to get the row number that harvest occurs has to be row number not index name because it is used to index numpy below
    period, proportion = fun.period_proportion_np(pinp.period['feed_periods']['date'], pinp.period['harv_date'])
    params['p_harvest_period_prop']  = dict([(pinp.period['feed_periods'].index[period], proportion)])
    nap_dflrt[0,harvest_period,...,0] = dry_foo_start_ungrazed_flt[harvest_period,:, na,0]  \
                                           * (1-arable_l.reshape(-1,1))  \
                                           * (1-np.sum(pasture_rt, axis=1))    # sum pasture proportion across the t axis to get area of crop
    nap_rav_dflrt = nap_dflrt.ravel()
    params['p_nap_dflrt'] = dict(zip(index_dflrt ,nap_rav_dflrt))

    ## green initial FOO
    foo_start_grnha_oflt = np.maximum(i_fxg_foo_oflt
                                       , i_base_ft[:, na, :])  # to ensure that final foo can not be below the base level
    max_foo_flt                 = np.maximum(i_fxg_foo_oflt[1,...], grn_foo_start_ungrazed_flt)      #maximum of ungrazed foo and foo from the medium foo level
    foo_start_grnha_oflt[2,...] = np.maximum.accumulate(max_foo_flt,axis=0)                          #maximum accumulated along the feed periods axis, i.e. max to date
    # foo_start_grnha_oflt[...]   = np.maximum(foo_start_grnha_oflt
    #                                          , i_base_ft[:, na,:])         # to ensure that final foo can not be below 0
    foo_start_grnha_oflt = foo_start_grnha_oflt * mask_greenfeed_exists_ft[:, na,:]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    foo_start_grnha_rav_oflt = foo_start_grnha_oflt.ravel()
    params['p_foo_start_grnha_oflt'] = dict(zip(index_oflt ,foo_start_grnha_rav_oflt))

    ## green, pasture growth
    pgr_grnday_oflt = np.maximum(0.01, i_fxg_pgr_oflt)                  # use maximum to ensure that the pgr is non zero (because foo_days requires dividing by pgr)
    pgr_grnha_goflt =       pgr_grnday_oflt     \
                     *          length_f.reshape((-1,1,1))      \
                     * c_pgr_gi_scalar_gft[:, na,:, na,:]

    ## green, final foo from initial, pgr and senescence
    ### foo at end of period if ungrazed
    foo_ungrazed_grnha_oflt  = foo_start_grnha_oflt    * (1-grn_senesce_startfoo_ft[:, na,:])   \
                              + pgr_grnha_goflt[0,...] * (1- grn_senesce_pgrcons_ft[:, na,:])
    ### foo at end of period with range of grazing intensity prior to eos senescence
    foo_endprior_grnha_goflt =    foo_ungrazed_grnha_oflt \
                               - (foo_ungrazed_grnha_oflt
                                  -           i_base_ft[:, na,:]) \
                                *  i_foo_graze_propn_gt[:, na, na, na, :] \
                               +              i_base_ft[:, na, :]
    senesce_eos_grnha_goflt = foo_endprior_grnha_goflt * i_grn_senesce_eos_ft[:, na,:]
    foo_end_grnha_goflt = foo_endprior_grnha_goflt - senesce_eos_grnha_goflt
    foo_end_grnha_goflt = foo_end_grnha_goflt * mask_greenfeed_exists_ft[:, na,:]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    foo_end_grnha_rav_goflt = foo_end_grnha_goflt.ravel()
    params['p_foo_end_grnha_goflt'] = dict( zip(index_goflt ,foo_end_grnha_rav_goflt))

    ## green, removal & dmi
    ### divide by (1 - grn_senesce_pgrcons) to allows for consuming feed reducing senescence (but it also converts to per day, so have to multiply by days in period)
    removal_grnha_goflt =np.maximum(0,   foo_start_grnha_oflt
                               * (1 - grn_senesce_startfoo_ft[:, na,:])
                                      +          pgr_grnha_goflt
                               * (1 -  grn_senesce_pgrcons_ft[:, na,:])
                                      - foo_endprior_grnha_goflt)          \
                          / (1 -       grn_senesce_pgrcons_ft[:, na,:])
    cons_grnha_t_goflt  =      removal_grnha_goflt   \
                         /(1+i_grn_trampling_ft[:, na,:])

    ## green, dmd & md from average and change due to foo & grazing intensity
    ### # to calculate foo_days requires calculating number of days in current period and adding days from the previous period (if required)

    ###set the default to Clip between -1 and 0 for low FOO level
    min_oflt = np.zeros(n_foo_levels).reshape((-1,1,1,1))
    max_oflt = np.ones(n_foo_levels).reshape((-1,1,1,1))
    # ### and clip between 0 and 1 for high FOO level
    # min_oflt[0,...] = 0
    # max_oflt[0,...] = 0

    propn_period_oflt               = (  foo_start_grnha_oflt
                                       - foo_start_grnha_oflt [  0:1,...])            \
                                     /         pgr_grnha_goflt[0, ...]
    propn_periodprev_oflt           = (  foo_start_grnha_oflt   [ : ,1: ,:,:]
                                       - foo_start_grnha_oflt   [0:1,1: ,:,:]
                                       -       pgr_grnha_goflt[0, : ,1:  ,:,:])     \
                                     /         pgr_grnha_goflt[0, : , :-1,:,:]      # pgr from the previous period
    foo_days_grnha_oflt             = np.clip(propn_period_oflt,min_oflt,max_oflt)              \
                                     *              length_f.reshape((-1,1,1))
    foo_days_grnha_oflt[:,1:,:,:]  += np.clip(propn_periodprev_oflt,min_oflt,max_oflt)          \
                                     *                  length_f[:-1].reshape(-1,1,1) # length from previous period
    ### convert monthly decline to daily decline
    grn_dmd_declinefoo_ft           = i_grn_dmd_declinefoo_ft / 30.5
    ### change in sward average digestibility due to increasing foo
    grn_dmd_fooadj_oflt             = ((1 - grn_dmd_declinefoo_ft[:, na,:])
                                         **     foo_days_grnha_oflt) - 1
    dmd_sward_grnha_goflt           =            i_grn_dig_flt                        \
                                     +      grn_dmd_fooadj_oflt                       \
    ### change in digestibility associated with diet selection (altered by level of grazing)
    grn_dmd_range_oflt              = i_grn_dmd_range_ft[:, na, :] - grn_dmd_fooadj_oflt * 2   # Sward ave DMD reduction (due to deferment) increases the range
    grn_dmd_selectivity_goflt[1,...] = 0.5000 * grn_dmd_range_oflt
    grn_dmd_selectivity_goflt[2,...] = 0.3333 * grn_dmd_range_oflt              #^ could improve this by making the selectivity proportion a formula based on proportion grazed
    grn_dmd_selectivity_goflt[3,...] = 0
    dmd_diet_grnha_goflt =      dmd_sward_grnha_goflt                       \
                          + grn_dmd_selectivity_goflt
    grn_md_grnha_goflt = fun.dmd_to_md(dmd_diet_grnha_goflt)

    ## green, mei & volume
    foo_ave_grnha_goflt      = (foo_start_grnha_oflt
                                + foo_end_grnha_goflt)/2
    ### pasture params used to convert foo for rel availability
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu3_len'], uinp.pastparameters['i_cu3_len2']).astype(float) #have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu4_len'], uinp.pastparameters['i_cu4_len2']).astype(float) #have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)
    pasture_stage_flt = pinp.sheep['i_pasture_stage_p6z'][:, na, na]#^this is what the line below will need to look like when season axis is added: pasture_stage_flt = f_reshape_expand(pinp.sheep['i_pasture_stage_p6z'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos'])
    ### adjust foo and calc hf
    foo_ave_grnha_goflt, hf = sfun.f_foo_convert(cu3, cu4, foo_ave_grnha_goflt, pinp.sheep['i_hr_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'],uinp.pastparameters['i_hd_std'], i_legume_t, pasture_stage_flt)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        grn_ri_availability_goflt = sfun.f_ra_cs(foo_ave_grnha_goflt, hf)
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        grn_ri_quality_goflt     = sfun.f_rq_cs(dmd_diet_grnha_goflt, i_legume_t)
    grn_ri_goflt             = np.maximum( 0.05                                        # set the minimum RI to 0.05
                                          ,     grn_ri_quality_goflt
                                          *grn_ri_availability_goflt)
    #todo set me_cons to 0 in the confinement pool when the pool is added
    me_cons_grnha_vgoflt     = fun.f_effective_mei(      cons_grnha_t_goflt
                                                 ,     grn_md_grnha_goflt
                                                 ,   me_maintenance_vft[:, na, na,:, na,:]
                                                 ,           grn_ri_goflt
                                                 ,i_me_eff_gainlose_ft[:, na,:])
    me_cons_grnha_vgoflt = me_cons_grnha_vgoflt * mask_greenfeed_exists_ft[:, na,:]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    me_cons_grnha_rav_vgoflt = me_cons_grnha_vgoflt.ravel()
    params['p_me_cons_grnha_vgoflt'] = dict( zip(index_vgoflt ,me_cons_grnha_rav_vgoflt))

    volume_grnha_goflt    =  cons_grnha_t_goflt / grn_ri_goflt              # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
    volume_grnha_goflt = volume_grnha_goflt * mask_greenfeed_exists_ft[:, na,:]  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    volume_grnha_rav_goflt = volume_grnha_goflt.ravel()
    params['p_volume_grnha_goflt'] = dict(zip(index_goflt ,volume_grnha_rav_goflt))

    ## dry, dmd & foo of feed consumed
    ### do sensitivity adjustment for dry_dmd_input based on increasing/reducing the reduction in dmd from the maximum (starting value)
    dry_dmd_adj_ft  = (        i_dry_dmd_ave_ft
                       -np.max(i_dry_dmd_ave_ft, axis=0)) * sen.sam['dry_dmd_decline','annual']
    dry_dmd_high_ft = np.max(i_dry_dmd_ave_ft, axis=0) + dry_dmd_adj_ft + i_dry_dmd_range_ft/2
    dry_dmd_low_ft  = np.max(i_dry_dmd_ave_ft, axis=0) + dry_dmd_adj_ft - i_dry_dmd_range_ft/2
    dry_dmd_dft     = np.stack((dry_dmd_low_ft, dry_dmd_high_ft), axis=0)    # create an array with a new axis 0 by stacking the existing arrays

    dry_foo_high_ft = i_dry_foo_high_ft * 3/4
    dry_foo_low_ft  = i_dry_foo_high_ft * 1/4                               # assuming half the foo is high quality and the remainder is low quality
    dry_foo_dft     = np.stack((dry_foo_low_ft, dry_foo_high_ft),axis=0)  # create an array with a new axis 0 by stacking the existing arrays

    ## dry, volume of feed consumed per tonne
    ### adjust foo and calc hf
    pasture_stage_ft = pinp.sheep['i_pasture_stage_p6z'][:, na]#^this is what the line below will need to look like when season axis is added: pasture_stage_flt = f_reshape_expand(pinp.sheep['i_pasture_stage_p6z'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos'])
    dry_foo_dft, hf = sfun.f_foo_convert(cu3, cu4, dry_foo_dft, pinp.sheep['i_hr_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'],uinp.pastparameters['i_hd_std'], i_legume_t, pasture_stage_ft) 
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        dry_ri_availability_dft = sfun.f_ra_cs(dry_foo_dft, hf)

    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        dry_ri_quality_dft     = sfun.f_rq_cs(dry_dmd_dft, i_legume_t)
    dry_ri_dft              = dry_ri_quality_dft * dry_ri_availability_dft
    dry_ri_dft[dry_ri_dft<0.05] = 0.05 #set the minimum RI to 0.05
    dry_volume_t_dft  = 1000 / dry_ri_dft                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed
    dry_volume_t_dft = dry_volume_t_dft * mask_dryfeed_exists_ft  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    dry_volume_t_rav_dft = dry_volume_t_dft.ravel()
    params['p_dry_volume_t_dft'] = dict(zip(index_dft ,dry_volume_t_rav_dft))

    ## dry, ME consumed per kg consumed
    dry_md_dft           = fun.dmd_to_md(dry_dmd_dft)
    dry_md_vdft          = np.stack([dry_md_dft] * n_feed_pools, axis = 0)
    ## convert to effective quality per tonne
    dry_mecons_t_vdft  = fun.f_effective_mei( 1000                                    # parameters for the dry feed grazing activities: Total ME of the tonne consumed
                            ,           dry_md_vdft
                            ,   me_maintenance_vft[:, na,:,:]
                            ,           dry_ri_dft
                            ,i_me_eff_gainlose_ft)
    dry_mecons_t_vdft = dry_mecons_t_vdft * mask_dryfeed_exists_ft  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    dry_mecons_t_rav_vdft = dry_mecons_t_vdft.ravel()
    params['p_dry_mecons_t_vdft'] = dict(zip(index_vdft ,dry_mecons_t_rav_vdft))

    ## dry, animal removal
    dry_removal_t_dft[...]  = 1000 * (1 + i_dry_trampling_ft)
    dry_removal_t_rav_dft = dry_removal_t_dft.ravel()
    params['p_dry_removal_t_dft'] = dict( zip(index_dft ,dry_removal_t_rav_dft))

    ## senescence from green to dry
    ### green, total senescence for the period
    # senesce_total_grnha_goflt   = foo_start_grnha_oflt    \
    #                              +      pgr_grnha_goflt   \
    #                              -  removal_grnha_goflt   \
    #                              -  foo_end_grnha_goflt
    ## the senesced feed that is available to stock is that which senesces at the end of the growing season (i.e. not during the growing season)
    ##^ may need revisiting for perennial pastures where green & dry feed are part of a mixed diet.
    senesce_total_grnha_goflt    = senesce_eos_grnha_goflt
    grn_dmd_senesce_goflt        =               dmd_sward_grnha_goflt       \
                                  + i_grn_dmd_senesce_redn_ft[:, na,:]
    senesce_propn_dgoflt[1,...]  = np.clip(( grn_dmd_senesce_goflt                     # senescence to high pool. np.clip reduces the range of the dmd to the range of dmd in the dry feed pools
                                            -    dry_dmd_low_ft[:, na,:])
                                          /(    dry_dmd_high_ft[:, na,:]
                                            -    dry_dmd_low_ft[:, na,:]), 0, 1)
    senesce_propn_dgoflt[0,...] = 1- senesce_propn_dgoflt[1,...]                       # senescence to low pool
    senesce_grnha_dgoflt        = senesce_total_grnha_goflt * senesce_propn_dgoflt       # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    senesce_grnha_dgoflt        = senesce_grnha_dgoflt * mask_greenfeed_exists_ft[:, na,:]  # apply mask - green pasture only senesces when green pas exists.
    senesce_grnha_rav_dgoflt    = senesce_grnha_dgoflt.ravel()
    params['p_senesce_grnha_dgoflt'] = dict( zip(index_dgoflt ,senesce_grnha_rav_dgoflt))

    ##store report vals
    r_vals['pgr_grnha_goflt'] = pgr_grnha_goflt #store for reporting
    r_vals['foo_end_grnha_goflt'] = foo_end_grnha_goflt #store for reporting
    r_vals['cons_grnha_t_goflt'] = cons_grnha_t_goflt #store for reporting

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
    poc_con_fl = i_poc_intake_daily_flt[...,0] / 1000 #divide 1000 to convert to tonnes of foo per ha
    poc_con_rav_fl = poc_con_fl.ravel()
    params['p_poc_con_fl'] = dict(zip(index_fl, poc_con_rav_fl))
    ## md per tonne
    poc_md_f = fun.dmd_to_md(i_poc_dmd_ft[...,0]) * 1000 #times 1000 to convert to mj per tonne
    poc_md_rav_f = poc_md_f.ravel()
    params['p_poc_md_f'] = dict(zip(keys_f ,poc_md_rav_f))
    ## vol
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        ri_qual_f     = sfun.f_rq_cs(i_poc_dmd_ft[...,0], i_legume_t[...,0])
    
    ### pasture params used to convert foo for rel availability
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu3_len'], uinp.pastparameters['i_cu3_len2']).astype(float) #have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu4_len'], uinp.pastparameters['i_cu4_len2']).astype(float) #have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)
    pasture_stage_flt = pinp.sheep['i_pasture_stage_p6z']# currently no active z ^this is what the line below will need to look like when season axis is added: pasture_stage_flt = f_reshape_expand(pinp.sheep['i_pasture_stage_p6z'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos'])
    ### adjust foo and calc hf
    i_poc_foo_f, hf = sfun.f_foo_convert(cu3, cu4, i_poc_foo_ft[...,0], pinp.sheep['i_hr_scalar'], pinp.sheep['i_region'], uinp.pastparameters['i_n_pasture_stage'],uinp.pastparameters['i_hd_std'], i_legume_t, pasture_stage_flt) 
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        ri_quan_f = sfun.f_ra_cs(i_poc_foo_f, hf)
        poc_vol_f = fun.f_divide(1,(ri_qual_f*ri_quan_f)) * 1000 #times 1000 to convert to vol to per tonne
    poc_vol_rav_f = poc_vol_f.ravel()
    params['p_poc_vol_f'] = dict(zip(keys_f ,poc_vol_rav_f))
    
    ###########
    #report   #
    ###########
    ##keys
    r_vals['keys_d'] = keys_d
    r_vals['keys_v'] = keys_v
    r_vals['keys_f'] = keys_f
    r_vals['keys_g'] =  keys_g
    r_vals['keys_l'] = keys_l
    r_vals['keys_o'] = keys_o
    r_vals['keys_p'] = keys_p
    r_vals['keys_r'] = keys_r
    r_vals['keys_t'] = keys_t
    r_vals['keys_k'] = keys_k