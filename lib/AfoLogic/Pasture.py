'''
Pasture documentation can be found in PastureFunctions.py
'''


'''
import functions from other modules
'''
# import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import FeedsupplyFunctions as fsfun
from . import EmissionFunctions as efun
from . import Periods as per
from . import Sensitivity as sen
from . import PastureFunctions as pfun
from . import Trees as tree

#1. todo add labour required for feed budgeting. Inputs are currently in the sheep sheet of Property.xls (would be best if this can be built in phase_labour module)
#2. todo Will need to add the foo reduction in the current year for manipulated pasture and a germination reduction in the following year.

def f_pasture(params, r_vals, nv):
    ######################
    ##background vars    #
    ######################
    na = np.newaxis

    ########################
    ##nv stuff             #
    ########################
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement period is included the last nv pool is confinement.

    ########################
    ##phases               #
    ########################
    ## read the rotation phases information from inputs
    # phase_len       = sinp.general['phase_len']
    phases_rotn_df  = pinp.phases_r
    pasture_sets    = sinp.landuse['pasture_sets']
    pastures        = sinp.general['pastures'][pinp.general['pas_inc_t']]

    ########################
    ##constants required   #
    ########################
    ## define some parameters required to size arrays.
    n_feed_pools    = len_nv
    n_dry_groups    = len(sinp.general['dry_groups'])           # Low & high quality groups for dry feed
    n_grazing_int   = len(sinp.general['grazing_int'])          # grazing intensity in the growth/grazing activities
    n_foo_levels    = len(sinp.general['foo_levels'])           # Low, medium & high FOO level in the growth/grazing activities
    n_feed_periods  = len(per.f_feed_periods()) - 1
    n_lmu           = np.count_nonzero(pinp.general['i_lmu_area'])
    n_phases_rotn   = len(phases_rotn_df.index)
    n_pasture_types = len(pastures)   #^ need to sort timing of the definition of pastures
    # n_total_seasons = len(pinp.general['i_mask_z']) #used to reshape inputs
    if sinp.structuralsa['steady_state']:
        n_season_types = 1
    else:
        n_season_types = np.count_nonzero(pinp.general['i_mask_z'])
    len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
    index_p6 = np.arange(n_feed_periods)

    ## indexes required for advanced indexing
    l_idx = np.arange(n_lmu)
    z_idx = np.arange(n_season_types)
    t_idx = np.arange(n_pasture_types)
    r_idx = np.arange(n_phases_rotn)


    arable_l = pinp.general['arable']
    length_p6z  = per.f_feed_periods(option=1)
    feed_period_dates_p6z = per.f_feed_periods()
    date_start_p6z = feed_period_dates_p6z[:-1]
    date_end_p6z = feed_period_dates_p6z[1:]
    date_mid_p6z = date_start_p6z + (date_end_p6z - date_start_p6z)/2

    # fgop6lt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
    dgop6lzt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # fdp6t    = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
    fp6zt    = (n_feed_pools, n_feed_periods, n_season_types, n_pasture_types)
    # dp6t     = (n_dry_groups, n_feed_periods, n_pasture_types)
    # gop6lzt  = (n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,  n_season_types, n_pasture_types)
    # gop6t    = (n_grazing_int, n_foo_levels, n_feed_periods, n_pasture_types)
    gp6zt     = (n_grazing_int, n_feed_periods, n_season_types, n_pasture_types)
    gt      = (n_grazing_int, n_pasture_types)
    op6lzt   = (n_foo_levels, n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    qop6lzt   = (len_q, n_foo_levels, n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    # dp6lrt   = (n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    qdp6lrzt  = (len_q, n_dry_groups, n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    # p6lrt    = (n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
    p6lrzt   = (n_feed_periods, n_lmu, n_phases_rotn, n_season_types, n_pasture_types)
    # p6rt     = (n_feed_periods, n_phases_rotn, n_pasture_types)
    rt      = (n_phases_rotn, n_pasture_types)
    rzt      = (n_phases_rotn, n_season_types, n_pasture_types)
    # p6lt     = (n_feed_periods, n_lmu, n_pasture_types)
    p6lzt    = (n_feed_periods, n_lmu, n_season_types, n_pasture_types)
    lt      = (n_lmu, n_pasture_types)
    lzt     = (n_lmu, n_season_types, n_pasture_types)
    p6t      = (n_feed_periods, n_pasture_types)
    p6zt     = (n_feed_periods, n_season_types, n_pasture_types)
    zt      = (n_season_types, n_pasture_types)
    # t       = (n_pasture_types)

    ## define the vessels that will store the input data that require pre-defining
    ### all need pre-defining because inputs are in separate pasture type arrays
    i_phase_germ_dict = dict()
    i_grn_senesce_daily_p6zt      = np.zeros(p6zt,  dtype = 'float64')              # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_p6zt       = np.zeros(p6zt,  dtype = 'float64')             # proportion of green feed that senesces in period (due to a water deficit or completing life cycle)
    dry_decay_daily_p6zt         = np.zeros(p6zt,  dtype = 'float64')             # daily decline in dry foo in each period
    i_end_of_gs_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_exists_zt              = np.zeros(zt, dtype = 'int')                   # the period number when the pasture senesces due to lack of water or end of life cycle
    i_dry_decay_t               = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    # i_me_maintenance_vp6t        = np.zeros(vp6t,  dtype = 'float64')     # M/D level for target LW pattern
    c_pgr_gi_scalar_gp6zt         = np.zeros(gp6zt,  dtype = 'float64')     # pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_graze_propn_gt        = np.zeros(gt, dtype ='float64')        # proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_op6lzt             = np.zeros(op6lzt, dtype = 'float64')    # FOO level     for the FOO/growth/grazing variables.
    i_fxg_pgr_qop6lzt             = np.zeros(qop6lzt, dtype = 'float64')    # PGR level     for the FOO/growth/grazing variables.
    c_fxg_a_qop6lzt               = np.zeros(qop6lzt, dtype = 'float64')    # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_qop6lzt               = np.zeros(qop6lzt, dtype = 'float64')    # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_op6lt               = np.zeros(op6lt, dtype = 'float64')     # coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_op6lt               = np.zeros(op6lt, dtype = 'float64')     # coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_p6lzt              = np.zeros(p6lzt, dtype = 'float64') # green pasture digestibility in each period, LMU, season & pasture type.
    i_soil_production_lzt              = np.zeros(lzt, dtype = 'float64') # green pasture digestibility in each period, LMU, season & pasture type.
    i_poc_intake_daily_p6lzt      = np.zeros(p6lzt, dtype = 'float64')  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_p6lzt      = np.zeros(p6lzt, dtype = 'float64')  # minimum foo at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lzt           = np.zeros(lzt,  dtype = 'float64') # scale the mobilisation of below ground reserves for each lmu
    i_restock_fooscalar_lt      = np.zeros(lt,  dtype = 'float64')  # scalar for FOO between LMUs when pastures are restocked after reseeding
    i_pasture_coverage_lt      = np.zeros(lt,  dtype = 'float64')  # scalar for pasture coverage i.e. what propn of the lmu that each pasture grows on.

    i_me_eff_gainlose_p6zt        = np.zeros(p6zt,  dtype = 'float64')  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_p6t          = np.zeros(p6t,  dtype = 'float64')  # green pasture trampling in each feed period as proportion of intake.
    i_dry_trampling_p6t          = np.zeros(p6t,  dtype = 'float64')  # dry pasture trampling   in each feed period as proportion of intake.
    i_base_p6zt                   = np.zeros(p6zt,  dtype = 'float64')  # lowest level that pasture can be consumed in each period
    i_grn_dmd_range_p6zt          = np.zeros(p6zt,  dtype = 'float64')  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_p6zt  = np.zeros(p6zt,  dtype = 'float64') # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_p6zt           = np.zeros(p6zt,  dtype = 'float64') # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_p6zt         = np.zeros(p6zt,  dtype = 'float64') # range in digestibility of dry feed if it is not grazed
    i_dry_dmd_eogs_zt           = np.zeros(zt,  dtype = 'float64') # dmd of dry pasture at the end of the growing season.
    i_dry_dmd_brk_zt           = np.zeros(zt,  dtype = 'float64') # dmd of dry pasture at the lastest season brk.
    i_dry_foo_high_p6zt          = np.zeros(p6zt,  dtype = 'float64') # expected foo for the dry pasture in the high quality pool
    dry_decay_period_p6zt        = np.zeros(p6zt,  dtype = 'float64') # decline in dry foo for each period
    mask_dryfeed_exists_p6zt     = np.zeros(p6zt,  dtype = bool)      # mask for period when dry feed exists
    mask_greenfeed_exists_p6zt   = np.zeros(p6zt,  dtype = bool)      # mask for period when green feed exists
    i_germ_scalar_p6zt           = np.zeros(p6zt,  dtype = 'float64') # allocate the total mobilisation of below ground reserves between feed periods
    i_grn_cp_p6zt                 = np.zeros(p6zt,  dtype = 'float64')  # crude protein content of green feed
    i_dry_cp_p6zt                 = np.zeros(p6zt,  dtype = 'float64')  # crude protein content of dry feed
    i_poc_dmd_p6zt                = np.zeros(p6zt,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_p6zt                = np.zeros(p6zt,  dtype = 'float64')  # foo of pasture consumed on crop paddocks
    # grn_senesce_startfoo_p6t     = np.zeros(p6t,  dtype = 'float64')  # proportion of the FOO at the start of the period that senesces during the period
    # grn_senesce_pgrcons_p6t      = np.zeros(p6t,  dtype = 'float64')  # proportion of the (total or average daily) PGR that senesces during the period (consumption leads to a reduction in senescence)

    i_destock_date_zt           = np.zeros(zt, dtype = 'float64')         # date of destocking this pasture type prior to reseeding
    i_destock_foo_zt            = np.zeros(zt, dtype = 'float64')               # kg of FOO that was not grazed prior to destocking for spraying prior to reseeding pasture (if spring sown)
    i_restock_date_zt           = np.zeros(zt, dtype = 'float64')         # date of first grazing of reseeded pasture
    i_restock_foo_arable_t      = np.zeros(n_pasture_types, dtype = 'float64')  # FOO at restocking on the arable area of the resown pastures
    # reseeding_machperiod_t      = np.zeros(n_pasture_types, dtype = 'float64')  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
    i_germination_std_zt        = np.zeros(zt, dtype = 'float64')               # standard level of mobilisation of below ground reserves for the standard soil type in a continuous pasture rotation
    # i_ri_foo_t                  = np.zeros(n_pasture_types, dtype = 'float64')  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    # poc_days_of_grazing_t       = np.zeros(n_pasture_types, dtype = 'float64')  # number of days after the pasture break that (moist) seeding can begin
    i_legume_zt                 = np.zeros(zt, dtype = 'float64')               # proportion of legume in the sward
    i_hr_scalar_zt              = np.ones(zt, dtype = 'float64')               # Scalar for the pasture height ratio
    i_pasture_stage_p6zt        = np.zeros(p6zt,  dtype = 'float64')  # 0 is establishing pasture, 1 is vegetative pasture. Value is used to convert FOO & height for the local pasture pasture measured using the local system to the measurements used in GrazPlan
    i_restock_grn_propn_t       = np.zeros(n_pasture_types, dtype = 'float64')  # Proportion of the FOO that is green when pastures are restocked after reseeding
    i_nv_maintenance_t         = np.zeros(n_pasture_types, dtype = 'float64')  # approximate nutritive value for maintenance (NV = M/D * relative intake)

#    germination_p6lrzt           = np.zeros(p6lrzt,  dtype = 'float64')  # germination for each rotation phase (kg/ha)
    # foo_dry_reseeding_dp6lrzt    = np.zeros(dp6lrzt, dtype = 'float64')  # dry FOO adjustment allocated to the low & high quality dry feed pools (kg/ha)
    # dry_removal_t_p6t            = np.zeros(p6t,   dtype = 'float64')  # Total DM removal from the tonne consumed (includes trampling)

    ### define the array that links rotation phase and pasture type
    pasture_rt                  = np.zeros(rt, dtype = 'float64')


    ## create numpy index for param dicts ^creating indexes is a bit slow
    ### the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
    keys_d  = np.asarray(sinp.general['dry_groups'])
    keys_f  = np.array(['nv{0}' .format(i) for i in range(len_nv)])
    keys_p6  = np.asarray(pinp.period['i_fp_idx'])
    keys_g  = np.asarray(sinp.general['grazing_int'])
    keys_l  = pinp.general['i_lmu_idx']   # lmu index description
    keys_p7 = per.f_season_periods(keys=True)
    keys_o  = np.asarray(sinp.general['foo_levels'])
    keys_p5  = np.array(per.f_p_date2_df().index).astype('str')
    keys_r  = np.array(phases_rotn_df.index).astype('str')
    keys_t  = np.asarray(pastures)                                      # pasture type index description
    keys_k  = np.asarray(list(sinp.general['i_idx_k']))                     #landuse
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    keys_z  = zfun.f_keys_z()

    ### rt
    arrays_rt=[keys_r, keys_t]

    ### mp6lrzt
    arrays_p7p6lrzt=[keys_p7, keys_p6, keys_l, keys_r, keys_z, keys_t]

    ### mp6lrzt
    arrays_p7qp6lrzt=[keys_p7, keys_q, keys_p6, keys_l, keys_r, keys_z, keys_t]

    ### op6lzt
    arrays_qop6lzt=[keys_q, keys_o, keys_p6, keys_l, keys_z, keys_t]

    ### gop6lzt
    arrays_qgop6lzt=[keys_q, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]

    ### fgop6lzt
    arrays_qfgop6lzt=[keys_q, keys_f, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]

    ### dgop6lzt
    arrays_qdgop6lzt=[keys_q, keys_d, keys_g, keys_o, keys_p6, keys_l, keys_z, keys_t]

    ### mdp6lrzt
    arrays_p7qdp6lrzt=[keys_p7, keys_q, keys_d, keys_p6, keys_l, keys_r, keys_z, keys_t]
    arrays_qp6lrzt=[keys_q, keys_p6, keys_l, keys_r, keys_z, keys_t]

    ### fdp6zt
    arrays_fdp6zt=[keys_f, keys_d, keys_p6, keys_z, keys_t]

    ### dp6zt
    arrays_dp6zt=[keys_d, keys_p6, keys_z, keys_t]

    ### fp6z
    arrays_fp6z=[keys_f, keys_p6, keys_z]

    ### p6l
    arrays_p6lz=[keys_p6, keys_l, keys_z]

    ### p6zt
    arrays_p6zt=[keys_p6, keys_z, keys_t]

    ###p6z
    arrays_p6z8 = [keys_p6, keys_z]

    ###p6z8z9
    arrays_p6z8z9 = [keys_p6, keys_z, keys_z]


    ###########
    #map_excel#
    ###########
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''

    ## map data from excel file into arrays
    ### loop through each pasture type
    for t, pasture in enumerate(pastures):
        exceldata = pinp.pasture_inputs[pasture]           # assign the pasture data to exceldata
        ## map the Excel data into the numpy arrays
        i_germination_std_zt[...,t]         = zfun.f_seasonal_inp(exceldata['GermStd'], numpy=True)
        i_pasture_coverage_lt[...,t]         = exceldata['i_pasture_coverage']
        # i_ri_foo_t[t]                       = exceldata['RIFOO']
        i_end_of_gs_zt[...,t]               = zfun.f_seasonal_inp(exceldata['EndGS'], numpy=True)
        i_dry_exists_zt[...,t]               = zfun.f_seasonal_inp(exceldata['i_dry_exists'], numpy=True)
        i_dry_decay_t[t]                    = exceldata['PastDecay']
        i_poc_intake_daily_p6lzt[...,t]       = zfun.f_seasonal_inp(exceldata['POCCons'], numpy=True, axis=2)
        i_legume_zt[...,t]                  = zfun.f_seasonal_inp(exceldata['Legume'], numpy=True)
        i_hr_scalar_zt[...,t]                  = zfun.f_seasonal_inp(exceldata['hr_scalar'], numpy=True)
        i_pasture_stage_p6zt[...,t]         = zfun.f_seasonal_inp(exceldata['i_pasture_stage_p6z'], numpy=True, axis=1
                                             ) #it would be better if z axis was treated after pas_stage has been used (like in stock.py) because it is used as an index. But there wasn't any way to do this without doubling up a lot of code. This is only a limitation in the weighted average version of model.
        i_restock_grn_propn_t[t]            = exceldata['FaG_PropnGrn']
        i_grn_dmd_senesce_redn_p6zt[...,t]   = zfun.f_seasonal_inp(np.swapaxes(exceldata['DigRednSenesce'],0,1), numpy=True, axis=1)
        i_dry_dmd_ave_p6zt[...,t]            = zfun.f_seasonal_inp(np.swapaxes(exceldata['DigDryAve'],0,1), numpy=True, axis=1)
        i_dry_dmd_range_p6zt[...,t]          = zfun.f_seasonal_inp(np.swapaxes(exceldata['DigDryRange'],0,1), numpy=True, axis=1)
        i_dry_dmd_eogs_zt[...,t]          = zfun.f_seasonal_inp(exceldata['i_dry_dmd_eogs_z'], numpy=True, axis=0)
        i_dry_dmd_brk_zt[...,t]          = zfun.f_seasonal_inp(exceldata['i_dry_dmd_brk_z'], numpy=True, axis=0)
        i_dry_foo_high_p6zt[...,t]           = zfun.f_seasonal_inp(np.swapaxes(exceldata['FOODryH'],0,1), numpy=True, axis=1)
        i_germ_scalar_p6zt[...,t]            = zfun.f_seasonal_inp(np.swapaxes(exceldata['GermScalarFP'],0,1), numpy=True, axis=1)

        i_grn_cp_p6zt[...,t]                  = zfun.f_seasonal_inp(exceldata['CPGrn'], numpy=True, axis=1)
        i_dry_cp_p6zt[...,t]                  = zfun.f_seasonal_inp(exceldata['CPDry'], numpy=True, axis=1)
        i_poc_dmd_p6zt[...,t]                 = zfun.f_seasonal_inp(exceldata['DigPOC'], numpy=True, axis=1)
        i_poc_foo_p6zt[...,t]                 = zfun.f_seasonal_inp(exceldata['FOOPOC'], numpy=True, axis=1)
        i_germ_scalar_lzt[...,t]            = zfun.f_seasonal_inp(np.swapaxes(exceldata['GermScalarLMU'],0,1), numpy=True, axis=1)
        i_restock_fooscalar_lt[...,t]       = exceldata['FaG_LMU']  #todo may need a z axis

        i_lmu_conservation_p6lzt[...,t]       = zfun.f_seasonal_inp(np.moveaxis(exceldata['ErosionLimit'],0,-1), numpy=True, axis=-1)

        i_destock_date_zt[...,t]            = zfun.f_seasonal_inp(exceldata['Date_Destocking'], numpy=True)
        i_destock_foo_zt[...,t]             = zfun.f_seasonal_inp(exceldata['FOOatSeeding'], numpy=True) #ungrazed foo when destocked for reseeding
        i_restock_date_zt[...,t]            = zfun.f_seasonal_inp(exceldata['Date_ResownGrazing'], numpy=True)
        i_restock_foo_arable_t[t]           = exceldata['FOOatGrazing']

        i_grn_trampling_p6t[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_p6t[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_p6zt[...,t]       = zfun.f_seasonal_inp(exceldata['SenescePropn'], numpy=True, axis=1)
        i_grn_senesce_eos_p6zt[...,t]        = zfun.f_seasonal_inp(np.asfarray(exceldata['SenesceEOS']), numpy=True, axis=1)
        i_base_p6zt[...,t]                    = zfun.f_seasonal_inp(exceldata['BaseLevelInput'], numpy=True, axis=1)
        i_grn_dmd_range_p6zt[...,t]           = zfun.f_seasonal_inp(np.moveaxis(exceldata['DigSpread'],0,-1), numpy=True, axis=1)
        i_foo_graze_propn_gt[..., t]        = np.asfarray(exceldata['FOOGrazePropn'])
        #### impact of grazing intensity (at the other levels) on PGR during the period
        PGRScalarH_p6z = zfun.f_seasonal_inp(exceldata['PGRScalarH'], numpy=True, axis=1)
        c_pgr_gi_scalar_gp6zt[...,t]      = 1 - i_foo_graze_propn_gt[..., na, na, t] ** 2 * (1 - PGRScalarH_p6z)

        i_fxg_foo_op6lzt[0,...,t]        = zfun.f_seasonal_inp(np.moveaxis(exceldata['LowFOO'],0,-1), numpy=True, axis=-1)
        i_fxg_foo_op6lzt[1,...,t]        = zfun.f_seasonal_inp(np.moveaxis(exceldata['MedFOO'],0,-1), numpy=True, axis=-1)
        i_me_eff_gainlose_p6zt[...,t]     = zfun.f_seasonal_inp(exceldata['MaintenanceEff'][:,:,0], numpy=True, axis=1)
        i_nv_maintenance_t[t]          = exceldata['MaintenanceNV']
        ## # i_fxg_foo_op6lt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_qop6lzt[:,0,...,t]        = zfun.f_seasonal_inp(np.moveaxis(exceldata['LowPGR'],0,-1), numpy=True, axis=-1)
        i_fxg_pgr_qop6lzt[:,1,...,t]        = zfun.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)
        i_fxg_pgr_qop6lzt[:,2,...,t]        = zfun.f_seasonal_inp(np.moveaxis(exceldata['MedPGR'],0,-1), numpy=True, axis=-1)  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_p6lzt[...,t]           = zfun.f_seasonal_inp(np.moveaxis(exceldata['DigGrn'],0,-1), numpy=True, axis=-1)  # numpy array of inputs for green pasture digestibility on each LMU.
        i_soil_production_lzt[...,t]           = zfun.f_seasonal_inp(exceldata['i_soil_production_zl'], numpy=True, axis=0).T  # numpy array of inputs for green pasture digestibility on each LMU.

        ###to handle different length rotation phases (ie simulation is shorter than pinp) the germ df needs to be sliced.
        offset = exceldata['GermPhases'].shape[-1] - len(phases_rotn_df.columns) - 1 #minus 1 because germ inputs has extra col
        ###to handle if there is only one rotation
        exceldata['GermPhases'] = exceldata['GermPhases'][na,:] if exceldata['GermPhases'].ndim==1 else exceldata['GermPhases']
        i_phase_germ_dict[pasture]      = pd.DataFrame(exceldata['GermPhases'][:,offset:])  #DataFrame with germ scalar and resown bool
        # i_phase_germ_dict[pasture].reset_index(inplace=True)                                # replace index read from Excel with numbers to match later merging
        # i_phase_germ_dict[pasture].columns.values[range(phase_len)] = [*range(phase_len)]   # replace the pasture columns read from Excel with numbers to match later merging

        ### define the link between rotation phase and pasture type while looping on pasture
        pasture_rt[:,t]                 = phases_rotn_df.iloc[:,-1].isin(pasture_sets[pasture])

    ##season inputs not required in t loop above
    harv_date_z         = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)
    i_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    ### pasture params used to convert foo for rel availability
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].astype(float)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].astype(float)

    ##universal emission inputs
    RBG_t = uinp.emissions['i_RBG_t'][pinp.general['pas_inc_t']]  # below ground-residue to above ground residue ratio
    NCAG_t = uinp.emissions['i_NCAG_t'][pinp.general['pas_inc_t']]  # nitrogen content of above-ground crop residue
    NCBG_t = uinp.emissions['i_NCBG_t'][pinp.general['pas_inc_t']]  # nitrogen content of below-ground crop residue

    ##scale pasture production based on adjacent tree plantings
    tree_production_scalar_l = tree.f_adjacent_land_production_scalar()
    i_fxg_pgr_qop6lzt = i_fxg_pgr_qop6lzt * tree_production_scalar_l[:,na,na]
    
    ##scale pasture production for q (sequence year) for MP model. version 1: all pasture types scaled the same.
    ##this pgr sam has a p6 and q which is good for raw version but too complicated to explain in the web app because a user might have node[0] in spring so if they changed pgr in p6[2] q[0] they would actually be changing the following calendar year. A simpler method is below.
    q_pgr_scalar_qp6 = sen.sam['q_pgr_scalar_Qp6'][0:len_q,:]  # have to slice len_q because SAM was initiliased with a big number (because q is unknown because it can be changed by SA)
    i_fxg_pgr_qop6lzt = i_fxg_pgr_qop6lzt * q_pgr_scalar_qp6[:,na,:,na,na,na]

    ##scale pasture production for the current year from node[0] to the end of the growing season. version 1: all pasture types scaled the same.
    ## this only adjusts p6 periods that occur after node 1 (ie when external condition change). Therefore it doesnt carry over to the following calendar year if node1 occurs at the end of the current calendar year (note q[0] can be in tow calendar years)
    q_pgr_scalar = sen.sam['q_pgr_scalar']
    date_season_node1_z = per.f_season_periods()[0, :]
    q_pgr_sam_mask_p6z = date_start_p6z >= date_season_node1_z
    q_pgr_scalar_p6z = np.ones((n_feed_periods, n_season_types))
    q_pgr_scalar_p6z[q_pgr_sam_mask_p6z] = q_pgr_scalar
    i_fxg_pgr_qop6lzt = i_fxg_pgr_qop6lzt * q_pgr_scalar_qp6[:, na, :, na, na, na] * q_pgr_scalar_p6z[:, na, :, na]

    ##create dry and green pasture exists mask
    ##in the late brk season dry feed can occur in fp0&1.
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    idx_fp_start_gs_z = fun.searchsort_multiple_dim(feed_period_dates_p6z, season_break_z, 1, 0, side='right') - 1
    mask_dryfeed_exists_p6zt[...] = np.logical_or(index_p6[:, na, na] >= i_dry_exists_zt, index_p6[:, na, na]<idx_fp_start_gs_z[...,na])   #mask periods when dry feed is available to livestock.
    mask_greenfeed_exists_p6zt[...] = np.logical_or(np.logical_and(index_p6[:,na,na]>=idx_fp_start_gs_z[...,na], index_p6[:, na, na] <= i_end_of_gs_zt),       #green exists in the period which is the end of growing season hence <=
                                                    np.logical_and(i_end_of_gs_zt < idx_fp_start_gs_z[...,na],
                                                                   np.logical_or(index_p6[:,na,na]>=idx_fp_start_gs_z[...,na], index_p6[:,na,na]<=i_end_of_gs_zt)))   #this handles if green feed starts mid fp and wraps around to start fps.

    ### calculate dry_decay_period (used in reseeding and green&dry)
    ### dry_decay_daily is decay of dry foo at the start of the period that was either
    ### dry at the end of the last period or transferred in from senescence in the previous period.
    ### dry_decay_daily does not effect green feed that senesces during the current period.
    dry_decay_daily_p6zt[...] = i_dry_decay_t
    dry_decay_daily_p6zt[~mask_dryfeed_exists_p6zt] = 1
    dry_decay_period_p6zt[...] = 1 - (1 - dry_decay_daily_p6zt) ** length_p6z[...,na]
    ### allowance for the decay of dry feed in the days prior to being consumed
    ### because only the feed at the end of period is decayed by dry_decay_period_p6zt
    ### scales total removal to allow for an equal portion of the feed being grazed each day
    #### can use f_divide because consumption is masked for the periods in which dry_decay_daily is 1 which should lead to infinite removal scalar (which causes error)
    removal_scalar_dry_decay_daily_p6zt = fun.f_divide(1, 1 - dry_decay_daily_p6zt)
    removal_scalar_dry_decay_p6zt = fun.f_divide((1 - removal_scalar_dry_decay_daily_p6zt ** length_p6z[..., na])
                                     / (1 - removal_scalar_dry_decay_daily_p6zt)
                                     , length_p6z[..., na])
    ## dry, DM decline (high = low pools)
    ###dry transfer prov is the amount of dry feed that is transferred into the current period from the previous (1000 - decay)
    dry_transfer_prov_t_p6zt = 1000 * (1-dry_decay_period_p6zt) #note: parent needs to provide if ANY child has dry feed next period
    ###dry transfer required is the amount of dry feed required in the current period to transfer into the next period (1000 mask by dry exists)
    dry_transfer_req_t_p6zt = 1000 * mask_dryfeed_exists_p6zt #this parameter exists so that the constraint wont be built for fp when no dry feed exists.

    ###adjust pgr for soil by weather-year scalar - this accounts for the fact that relativesoil production can vary due to weather-year
    i_fxg_pgr_qop6lzt = i_fxg_pgr_qop6lzt * i_soil_production_lzt
    ###create equation coefficients for pgr = a+b*foo
    i_fxg_foo_op6lzt[2,...]  = 100000 #large number so that the np.searchsorted doesn't go above
    c_fxg_b_qop6lzt[:,0,...] =  fun.f_divide(i_fxg_pgr_qop6lzt[:,0,...], i_fxg_foo_op6lzt[0,...])
    c_fxg_b_qop6lzt[:,1,...] =   fun.f_divide((i_fxg_pgr_qop6lzt[:,1,...] - i_fxg_pgr_qop6lzt[:,0,...])
                            , (i_fxg_foo_op6lzt[1,...] - i_fxg_foo_op6lzt[0,...]))
    c_fxg_b_qop6lzt[:,2,...] =  0
    c_fxg_a_qop6lzt[:,0,...] =  0
    c_fxg_a_qop6lzt[:,1,...] =  i_fxg_pgr_qop6lzt[:,0,...] - c_fxg_b_qop6lzt[:,1,...] * i_fxg_foo_op6lzt[0,...]
    c_fxg_a_qop6lzt[:,2,...] =  i_fxg_pgr_qop6lzt[:,1,...] # because slope = 0

    ## proportion of start foo that senesces during the period, different formula than excel
    grn_senesce_startfoo_p6zt = 1 - ((1 - i_grn_senesce_daily_p6zt) **  length_p6z[...,na])

    ##average proportion of growth senescenced per period.
    ## The first days growth decays for all the days and the last days growth only decays for 1 day (or maybe 0 days). The assumption is that a fixed percentage of the growth is consumed on the day it is grown.
    ## geometric series (https://en.wikipedia.org/wiki/Geometric_series) is used to calculate the foo at the end of the period after scenescense. Our geometric sum is starting from i=1 (not i=0) therefore subtract ar^0. Therefore the formula is a(((1-r^(n+1))/(1-r))-1) where r = (1-senesce).
    ## Then need to convert from final foo to scenescensed foo, then calculate average the scenescense per period.
    ## Formula is based on 1kg/da of growth.
    grn_senesce_pgrcons_p6zt = 1 - fun.f_divide(((1 -(1 - i_grn_senesce_daily_p6zt) ** (length_p6z[...,na]+1))
                                   / i_grn_senesce_daily_p6zt-1), length_p6z[...,na])



    ###############################################################################
    #Calculate the mobilisation of below ground reserves and reseeding parameters #
    ###############################################################################


    ## define instantiate arrays that are assigned in slices
    # na_erosion_p6lrt      = np.zeros(p6lrt,  dtype = 'float64')
    # na_phase_area_p6lrzt  = np.zeros(p6lrzt, dtype = 'float64')
    # grn_restock_foo_p6lzt = np.zeros(p6lzt,  dtype = 'float64')
    # dry_restock_foo_p6lzt = np.zeros(p6lzt,  dtype = 'float64')



    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df
    #todo max_germination_p6lz is being used later as an arg for f1_calc_foo_profile() to calculate dry pasture on non-arable area that should only have germination_p6lrzt.
    germination_p6lrzt, max_germination_p6lzt = pfun.f_germination(i_germination_std_zt, i_germ_scalar_lzt
                                                                , i_germ_scalar_p6zt, pasture_rt, arable_l
                                                                , pastures, phase_germresow_df, i_phase_germ_dict
                                                                ,  i_destock_date_zt, i_break_z, rzt)

    resown_rt = np.zeros(rt)
    seeding_freq_r = pinp.seeding_freq_r
    landuse_r = phases_rotn_df.iloc[:,-1].values
    a_k_rk = landuse_r[:,na] == keys_k
    for t,pasture in enumerate(pastures):
        pasture_landuses = list(sinp.landuse['pasture_sets'][pasture])
        landuse_is_pas_k = np.in1d(keys_k, pasture_landuses)  #if landuse is a pasture
        resown_rt[:,t] = np.sum(landuse_is_pas_k * a_k_rk, axis=-1) * seeding_freq_r
    foo_grn_reseeding_qp6lrzt, foo_dry_reseeding_qdp6lrzt, periods_destocked_p6zt = pfun.f_reseeding(
        i_destock_date_zt, i_restock_date_zt, i_destock_foo_zt, i_restock_grn_propn_t, resown_rt, feed_period_dates_p6z
        , i_restock_fooscalar_lt, i_restock_foo_arable_t, dry_decay_period_p6zt, i_fxg_foo_op6lzt, c_fxg_a_qop6lzt
        , c_fxg_b_qop6lzt, i_grn_senesce_eos_p6zt, grn_senesce_startfoo_p6zt, grn_senesce_pgrcons_p6zt, max_germination_p6lzt
        , length_p6z, n_feed_periods, p6lrzt, p6zt, t_idx, z_idx, len_q)

    ## area of green pasture being grazed and growing
    phase_area_p6lrzt = pfun.f1_green_area(resown_rt, pasture_rt, periods_destocked_p6zt, arable_l, i_pasture_coverage_lt)

    ## erosion limit. The minimum FOO at the end of each period#
    erosion_p6lrzt = pfun.f_erosion(i_lmu_conservation_p6lzt, arable_l, pasture_rt)

    ## initialise numpy arrays used only in this method
    # senesce_propn_dgop6lzt      = np.zeros(dgop6lzt, dtype = 'float64')
    nap_qdp6lrzt                 = np.zeros(qdp6lrzt,  dtype = 'float64')
    me_threshold_fp6zt          = np.zeros(fp6zt,    dtype = 'float64')   # the threshold for the nv pools which define the animals feed quality requirements

    ## create numpy array of threshold values from the nv dictionary
    me_threshold_fp6zt[...] = np.swapaxes(nv['nv_cutoff_ave_p6fz'], axis1=0, axis2=1)[...,na]
    ###threshold is the greater of the maintenance or the NV required in the pool because switching from one below
    ### maintenance feed to another that is further below maintenance doesn't affect average efficiency.
    me_threshold_fp6zt = fun.f_update(me_threshold_fp6zt, i_nv_maintenance_t, me_threshold_fp6zt < i_nv_maintenance_t)

    ##  Calculate the FOO profile for ungrazed green & dry pastures at the start of each period
    ### Variable is used elsewhere with the assumption that ungrazed is the maximum foo that can be achieved.
    grn_foo_start_ungrazed_qp6lzt, dry_foo_start_ungrazed_qp6lzt = pfun.f1_calc_foo_profile(
        max_germination_p6lzt, dry_decay_period_p6zt, length_p6z[...,na], i_fxg_foo_op6lzt
        , c_fxg_a_qop6lzt, c_fxg_b_qop6lzt, i_grn_senesce_eos_p6zt, grn_senesce_startfoo_p6zt
        , grn_senesce_pgrcons_p6zt, len_q)

    ### non-arable pasture becomes available to graze at the beginning of the first harvest period
    # harvest_period  = fun.period_allocation(pinp.period['feed_periods']['date'], range(len(pinp.period['feed_periods'])), pinp.period['harv_date']) #use range(len()) to get the row number that harvest occurs has to be row number not index name because it is used to index numpy below
    harv_period_z, harv_proportion_z = fun.period_proportion_np(feed_period_dates_p6z, harv_date_z)
    index = pd.MultiIndex.from_arrays([keys_p6[harv_period_z], keys_z])
    harvest_period_prop = pd.Series(harv_proportion_z, index=index).unstack()
    # params['p_harvest_period_prop']  = dict([(pinp.period['feed_periods'].index[harv_period_z], harv_proportion_z)])

    ## FOO on the non-arable areas in crop paddocks is ungrazed FOO of pasture type 0 (annual) because other pasture types are difficult to establish on a non-arable area.
    ### all senescing pasture from na area on crop paddocks goes into the Low pool (slice 0) because it is rank & low quality
    nap_qdp6lrzt[:,0,...,0] = (dry_foo_start_ungrazed_qp6lzt[...,na,:,0] * (harv_period_z==index_p6[:,na,na,na])
                                           * (1-arable_l[:, na,na])
                                           * (1-np.sum(pasture_rt[:, na, :], axis=-1)))    # sum pasture proportion across the t axis to get area of crop

    ## Pasture growth, consumption of green feed.
    me_cons_grnha_qfgop6lzt, volume_grnha_qfgop6lzt, foo_start_grnha_qop6lzt, foo_end_grnha_qgop6lzt, senesce_period_grnha_qgop6lzt \
    , senesce_eos_grnha_qgop6lzt, dmd_sward_grnha_qgop6lzt, pgr_grnha_qgop6lzt, foo_endprior_grnha_qgop6lzt, cons_grnha_t_qgop6lzt \
    , foo_ave_grnha_qgop6lzt, dmd_diet_grnha_qgop6lzt = pfun.f_grn_pasture(
        cu3, cu4, i_fxg_foo_op6lzt, i_fxg_pgr_qop6lzt, c_pgr_gi_scalar_gp6zt, grn_foo_start_ungrazed_qp6lzt, foo_grn_reseeding_qp6lrzt
        , i_foo_graze_propn_gt, grn_senesce_startfoo_p6zt, grn_senesce_pgrcons_p6zt, i_grn_senesce_eos_p6zt
        , i_base_p6zt, i_grn_trampling_p6t, i_grn_dig_p6lzt, i_grn_dmd_range_p6zt, i_pasture_stage_p6zt
        , i_legume_zt, i_hr_scalar_zt, me_threshold_fp6zt, i_me_eff_gainlose_p6zt, mask_greenfeed_exists_p6zt
        , length_p6z, nv_is_not_confinement_f, len_q)
    volume_grnha_qfgop6lzt = volume_grnha_qfgop6lzt / (1 + sen.sap['pi'])

    ##livestock methane emissions linked to the consumption of green pasture - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_ch4_grnpas_qgop6lzt = efun.f_stock_ch4_feed_nir(cons_grnha_t_qgop6lzt*1000, dmd_diet_grnha_qgop6lzt)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        grn_md_grnha_qgop6lzt = fsfun.f1_dmd_to_md(dmd_diet_grnha_qgop6lzt)
        stock_ch4_grnpas_qgop6lzt = efun.f_stock_ch4_feed_bc(cons_grnha_t_qgop6lzt*1000, grn_md_grnha_qgop6lzt)

    ##livestock nitrous oxide emissions linked to the consumption of green pasture - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_grnpas_qgop6lzt = efun.f_stock_n2o_feed_nir(cons_grnha_t_qgop6lzt*1000, dmd_diet_grnha_qgop6lzt, i_grn_cp_p6zt[:,na,:,:])

    ##residue nitrous oxide emissions - two separate calls because consumption doesnt reduce below ground biomass emissions (it does reduce future growth but doesnt the existing below ground biomass does not change)
    grnpas_n2o_residue_growth_qgop6lzt = efun.f_pas_residue_n2o_nir(pgr_grnha_qgop6lzt, RBG_t, NCAG_t, NCBG_t)
    grnpas_n2o_residue_cons_qgop6lzt = efun.f_pas_residue_n2o_nir(- cons_grnha_t_qgop6lzt*1000, RBG_t, NCAG_t, NCBG_t)
    grnpas_n2o_residue_qgop6lzt = grnpas_n2o_residue_growth_qgop6lzt + grnpas_n2o_residue_cons_qgop6lzt

    ##residue nitrous oxide emissions from consumption of 1t or growth of 1kg of pasture
    ## used for grn germination, dry pas consumption, nap production and consumption
    pas_n2o_residue_cons_t = efun.f_pas_residue_n2o_nir(-1000, RBG_t, NCAG_t, NCBG_t)
    pas_n2o_residue_growth_t = efun.f_pas_residue_n2o_nir(1, RBG_t, NCAG_t, NCBG_t)
    n2o_pas_residue_v_phase_growth_qp6lrzt = pas_n2o_residue_growth_t * (np.sum(nap_qdp6lrzt, axis=1) + germination_p6lrzt + np.maximum(0, foo_grn_reseeding_qp6lrzt))  #max so that any foo removed at destocking is treated like it has been decayed because it gets sprayed out.

    ##adjust dmd of dry feed post growing season - this doesnt do anything for perennials that are growing the whole yr.
    ##For annual pastures the dmd of dry feed is calculated based on the days since senescence.
    ##For perennials the dry dmd is based on the quality of the green feed that is senesced. Thus for perenials the code below does nothing. For perenials the dmd is just inputted.
    date_eogs_zt = np.take_along_axis(feed_period_dates_p6z[:,:,na],i_end_of_gs_zt[na]+1, axis=0)[0,...] #+1 because input is the last period when grn exists. [0] to remove singleton p6 axis.
    date_end_dry = np.max(pinp.general['i_break']) + 364 #use the latest season brk because a late brk season could follow the current season
    max_deterioration_period_zt = (date_end_dry - date_eogs_zt).astype(int)
    daily_deterioration_zt = 1-(i_dry_dmd_brk_zt / i_dry_dmd_eogs_zt)**(1/max_deterioration_period_zt)
    average_days_since_eogs_p6zt = date_mid_p6z[...,na] + 364 * (date_mid_p6z[...,na]<date_eogs_zt) - date_eogs_zt
    dry_dmd_p6zt = i_dry_dmd_eogs_zt * (1-daily_deterioration_zt)**average_days_since_eogs_p6zt.astype(int)
    ###update dry dmd if end of growing season
    dry_dmd_ave_p6zt = fun.f_update(i_dry_dmd_ave_p6zt,dry_dmd_p6zt, np.logical_not(mask_greenfeed_exists_p6zt))

    ## dry, dmd & foo of feed consumed
    dry_mecons_t_fdp6zt, dry_volume_t_fdp6zt, dry_dmd_dp6zt, dry_foo_dp6zt = pfun.f_dry_pasture(
        cu3, cu4, dry_dmd_ave_p6zt, i_dry_dmd_range_p6zt, i_dry_foo_high_p6zt, me_threshold_fp6zt, i_me_eff_gainlose_p6zt
        , mask_dryfeed_exists_p6zt, i_pasture_stage_p6zt, nv_is_not_confinement_f, i_legume_zt, i_hr_scalar_zt, n_feed_pools)
    dry_volume_t_fdp6zt = dry_volume_t_fdp6zt / (1 + sen.sap['pi'])

    ## dry, animal removal, mask consumption in periods where dry doesn't exist to remove the decision variable in pyomo.
    dry_removal_t_p6zt  = (1000 * (1 + i_dry_trampling_p6t[:,na,:])
                           * removal_scalar_dry_decay_p6zt
                           * mask_dryfeed_exists_p6zt)

    ## Senescence of green feed into the dry pool.
    senesce_grnha_qdgop6lzt = pfun.f1_senescence(senesce_period_grnha_qgop6lzt, senesce_eos_grnha_qgop6lzt, dry_decay_period_p6zt
                                              , dmd_sward_grnha_qgop6lzt, i_grn_dmd_senesce_redn_p6zt, dry_dmd_dp6zt
                                              , mask_greenfeed_exists_p6zt)


    ##livestock methane emissions linked to the consumption of 1t of dry pasture - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_ch4_drypas_dp6zt = efun.f_stock_ch4_feed_nir(1000, dry_dmd_dp6zt)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        dry_md_dp6zt = fsfun.f1_dmd_to_md(dry_dmd_dp6zt)
        stock_ch4_drypas_dp6zt = efun.f_stock_ch4_feed_bc(1000, dry_md_dp6zt)

    ##livestock nitrous oxide emissions linked to the consumption of 1t of dry pasture - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_drypas_dp6zt = efun.f_stock_n2o_feed_nir(1000, dry_dmd_dp6zt, i_dry_cp_p6zt)


    ######
    #poc #
    ######
    ##call poc function - info about poc can be found in function doc string.
    poc_con_p6lz, poc_md_fp6z, poc_vol_fp6z = pfun.f_poc(cu3, cu4, i_poc_intake_daily_p6lzt, i_poc_dmd_p6zt, i_poc_foo_p6zt
                                                         , i_legume_zt, i_hr_scalar_zt, i_pasture_stage_p6zt
                                                         , nv_is_not_confinement_f, me_threshold_fp6zt, i_me_eff_gainlose_p6zt)
    poc_vol_fp6z = poc_vol_fp6z/ (1 + sen.sap['pi'])

    ##livestock methane emissions linked to the consumption of 1t of poc - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_ch4_poc_p6z = efun.f_stock_ch4_feed_nir(1000, i_poc_dmd_p6zt[:, :, 0]) #slice t for annual
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        poc_md_p6z = fsfun.f1_dmd_to_md(i_poc_dmd_p6zt[..., 0])
        stock_ch4_poc_p6z = efun.f_stock_ch4_feed_bc(1000, poc_md_p6z)

    ##livestock nitrous oxide emissions linked to the consumption of 1t of poc - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_poc_p6z = efun.f_stock_n2o_feed_nir(1000, i_poc_dmd_p6zt[:, :, 0], i_grn_cp_p6zt[:, :, 0]) #slice t for annual


    ######
    #pnc #
    ######
    #todo remove this once new pasture is done - this is just temporary handling of foo when pnc is desocked. (not a great method because we are just guessing the amount of foo to reduce).
    ## PNC removes foo when the phase is changed. The assumption is that the phase is changed on the first day of the p7 period. POC handles any grazing between the start of p7 and seeding.
    ## The FOO removed is just equal to the germination with the assumption that any growth is grazed.
    ###p7 p6 allocation - only the first p6 period in each p7 has a true because phase changes on the first day of p7.
    date_season_node_p7z = per.f_season_periods()[:-1,:]
    p6_is_startp7_p7p6z = date_start_p6z == date_season_node_p7z[:,na,:]

    ###foo change when a2 is removed or annual phase is incremented - phase change occurs at the very start of a p7 period so only accounts for foo before the p7 period.
    foo_phase_change_p6lrzt = np.cumsum(np.roll(germination_p6lrzt, shift=1, axis=0), axis=0) #roll foward one so that current period germ isnt included because phase change occurs on the first day so germ for the current period hasnt occured yet.
    foo_phase_change_p7p6lrzt = foo_phase_change_p6lrzt * p6_is_startp7_p7p6z[:,:,na,na,:,na]
    ###foo removed when pnc is reduced (amount reduced is the amount of germination so far)
    # phase_is_a2_r = landuse_r == sinp.general['i_a2_idx']
    phase_is_annual_r = np.isin(landuse_r, list(sinp.landuse['pasture_sets']['annual']))
    foo_removed_pas_reduce_p7p6lrzt = foo_phase_change_p7p6lrzt * phase_is_annual_r[:,na,na]
    ###foo added when annual pasture is selected (the amount added is only the amount of germination that has already occured)
    foo_added_annual_increase_p7p6lrzt = foo_phase_change_p7p6lrzt * phase_is_annual_r[:,na,na]


    ######################
    #apply season mask   #
    ######################
    ##season transfer (z8z9) param
    season_start_z = per.f_season_periods()[0,:] #slice season node to get season start
    period_is_seasonstart_p6z = date_start_p6z==season_start_z
    mask_provwithinz8z9_p6z8z9, mask_provbetweenz8z9_p6z8z9, mask_reqwithinz8_p6z8, mask_reqbetweenz8_p6z8 = zfun.f_season_transfer_mask(
        date_start_p6z, period_is_seasonstart_pz=period_is_seasonstart_p6z, z_pos=-1)

    ##mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    mask_fp_z8var_p6lrzt = mask_fp_z8var_p6z[:,na,na,:,na]
    mask_fp_z8var_p6lzt = mask_fp_z8var_p6z[:,na,:,na]
    mask_fp_z8var_p6zt = mask_fp_z8var_p6z[:,:,na]

    ##apply mask
    erosion_p6lrzt = erosion_p6lrzt * mask_fp_z8var_p6lrzt
    poc_con_p6lz = poc_con_p6lz * mask_fp_z8var_p6z[:,na,:]
    poc_md_fp6z = poc_md_fp6z * mask_fp_z8var_p6z
    dry_removal_t_p6zt = dry_removal_t_p6zt * mask_fp_z8var_p6zt
    foo_dry_reseeding_qdp6lrzt = foo_dry_reseeding_qdp6lrzt * mask_fp_z8var_p6lrzt
    foo_grn_reseeding_qp6lrzt = foo_grn_reseeding_qp6lrzt * mask_fp_z8var_p6lrzt
    phase_area_p6lrzt = phase_area_p6lrzt * mask_fp_z8var_p6lrzt
    dry_transfer_prov_t_p6zt = dry_transfer_prov_t_p6zt * mask_fp_z8var_p6zt
    dry_transfer_req_t_p6zt = dry_transfer_req_t_p6zt * mask_fp_z8var_p6zt
    germination_p6lrzt = germination_p6lrzt * mask_fp_z8var_p6lrzt
    foo_removed_pas_reduce_p7p6lrzt = foo_removed_pas_reduce_p7p6lrzt * mask_fp_z8var_p6lrzt
    foo_added_annual_increase_p7p6lrzt = foo_added_annual_increase_p7p6lrzt * mask_fp_z8var_p6lrzt
    nap_qdp6lrzt = nap_qdp6lrzt * mask_fp_z8var_p6lrzt
    foo_start_grnha_qop6lzt = foo_start_grnha_qop6lzt * mask_fp_z8var_p6lzt
    foo_end_grnha_qgop6lzt = foo_end_grnha_qgop6lzt * mask_fp_z8var_p6lzt
    me_cons_grnha_qfgop6lzt = me_cons_grnha_qfgop6lzt * mask_fp_z8var_p6lzt
    dry_mecons_t_fdp6zt = dry_mecons_t_fdp6zt * mask_fp_z8var_p6zt
    volume_grnha_qfgop6lzt = volume_grnha_qfgop6lzt * mask_fp_z8var_p6lzt
    dry_volume_t_fdp6zt = dry_volume_t_fdp6zt * mask_fp_z8var_p6zt
    senesce_grnha_qdgop6lzt = senesce_grnha_qdgop6lzt * mask_fp_z8var_p6lzt
    poc_vol_fp6z = poc_vol_fp6z * mask_fp_z8var_p6z
    stock_ch4_grnpas_qgop6lzt = stock_ch4_grnpas_qgop6lzt * mask_fp_z8var_p6lzt
    stock_n2o_grnpas_qgop6lzt = stock_n2o_grnpas_qgop6lzt * mask_fp_z8var_p6lzt
    stock_ch4_drypas_dp6zt = stock_ch4_drypas_dp6zt * mask_fp_z8var_p6zt
    stock_n2o_drypas_dp6zt = stock_n2o_drypas_dp6zt * mask_fp_z8var_p6zt
    stock_ch4_poc_p6z = stock_ch4_poc_p6z * mask_fp_z8var_p6z
    stock_n2o_poc_p6z = stock_n2o_poc_p6z * mask_fp_z8var_p6z
    grnpas_n2o_residue_qgop6lzt = grnpas_n2o_residue_qgop6lzt * mask_fp_z8var_p6lzt
    n2o_pas_residue_v_phase_growth_qp6lrzt = n2o_pas_residue_v_phase_growth_qp6lrzt * mask_fp_z8var_p6lrzt


    #############################################
    #adjust params with r axis for rot period   #
    #############################################
    ##p7 allocation
    alloc_p7p6z = zfun.f1_z_period_alloc(date_start_p6z[na,:,:], length_p6z[na,:,:], z_pos=-1)
    alloc_p7p6lrzt = alloc_p7p6z[:,:,na,na,:,na]
    alloc_p7qp6lrzt = alloc_p7p6z[:,na,:,na,na,:,na]
    alloc_p7qdp6lrzt = alloc_p7p6z[:,na,na,:,na,na,:,na]

    ##apply allocation
    erosion_p7p6lrzt = erosion_p6lrzt * alloc_p7p6lrzt
    foo_dry_reseeding_p7qdp6lrzt = foo_dry_reseeding_qdp6lrzt * alloc_p7qdp6lrzt
    foo_grn_reseeding_p7qp6lrzt = foo_grn_reseeding_qp6lrzt * alloc_p7qp6lrzt
    phase_area_p7p6lrzt = phase_area_p6lrzt * alloc_p7p6lrzt
    germination_p7p6lrzt = germination_p6lrzt * alloc_p7p6lrzt
    nap_p7qdp6lrzt = nap_qdp6lrzt * alloc_p7qdp6lrzt


    #####################
    #calc co2e params   #
    #####################
    co2e_grnpas_qgop6lzt = (stock_ch4_grnpas_qgop6lzt * uinp.emissions['i_ch4_gwp_factor']
                           + (stock_n2o_grnpas_qgop6lzt + grnpas_n2o_residue_qgop6lzt) * uinp.emissions['i_n2o_gwp_factor'])
    co2e_drypas_cons_dp6zt = (stock_ch4_drypas_dp6zt * uinp.emissions['i_ch4_gwp_factor']
                              + (stock_n2o_drypas_dp6zt + pas_n2o_residue_cons_t) * uinp.emissions['i_n2o_gwp_factor'])
    co2e_poc_p6z = stock_ch4_poc_p6z * uinp.emissions['i_ch4_gwp_factor'] + stock_n2o_poc_p6z * uinp.emissions['i_n2o_gwp_factor']
    co2e_pas_residue_v_phase_growth_qp6lrzt = n2o_pas_residue_v_phase_growth_qp6lrzt * uinp.emissions['i_n2o_gwp_factor']


    ###########
    #params   #
    ###########
    ##non seasonal
    params['pasture_area_rt'] = fun.f1_make_pyomo_dict(pasture_rt * 1, arrays_rt)    # times 1 to convert from bool to int e.g. if the phase is pasture then 1ha of pasture is recorded.

    ##create season params

    params['p_mask_childz_within_fp'] = fun.f1_make_pyomo_dict(mask_reqwithinz8_p6z8 * 1, arrays_p6z8)
    params['p_mask_childz_between_fp'] = fun.f1_make_pyomo_dict(mask_reqbetweenz8_p6z8 * 1, arrays_p6z8)
    params['p_parentz_provwithin_fp'] = fun.f1_make_pyomo_dict(mask_provwithinz8z9_p6z8z9 * 1, arrays_p6z8z9)
    params['p_parentz_provbetween_fp'] = fun.f1_make_pyomo_dict(mask_provbetweenz8z9_p6z8z9 * 1, arrays_p6z8z9)

    params['p_erosion_p7p6lrzt'] = fun.f1_make_pyomo_dict(erosion_p7p6lrzt, arrays_p7p6lrzt)

    params['p_harvest_period_prop'] = harvest_period_prop.stack().to_dict()

    params['p_dry_removal_t_p6zt'] = fun.f1_make_pyomo_dict(dry_removal_t_p6zt, arrays_p6zt)

    params['p_foo_dry_reseeding_p7qdp6lrzt'] = fun.f1_make_pyomo_dict(foo_dry_reseeding_p7qdp6lrzt, arrays_p7qdp6lrzt)
    params['p_foo_grn_reseeding_p7qp6lrzt'] = fun.f1_make_pyomo_dict(foo_grn_reseeding_p7qp6lrzt, arrays_p7qp6lrzt)

    params['p_phase_area_p7p6lrzt'] = fun.f1_make_pyomo_dict(phase_area_p7p6lrzt, arrays_p7p6lrzt)

    params['p_dry_transfer_prov_t_p6zt'] = fun.f1_make_pyomo_dict(dry_transfer_prov_t_p6zt, arrays_p6zt)

    params['p_dry_transfer_req_t_p6zt'] = fun.f1_make_pyomo_dict(dry_transfer_req_t_p6zt, arrays_p6zt)

    params['p_germination_p7p6lrzt'] = fun.f1_make_pyomo_dict(germination_p7p6lrzt, arrays_p7p6lrzt)

    params['p_foo_removed_pas_reduce_p7p6lrzt'] = fun.f1_make_pyomo_dict(foo_removed_pas_reduce_p7p6lrzt, arrays_p7p6lrzt)

    params['p_foo_added_annual_increase_p7p6lrzt'] = fun.f1_make_pyomo_dict(foo_added_annual_increase_p7p6lrzt, arrays_p7p6lrzt)

    params['p_nap_p7qdp6lrzt'] = fun.f1_make_pyomo_dict(nap_p7qdp6lrzt, arrays_p7qdp6lrzt)

    params['p_foo_start_grnha_qop6lzt'] = fun.f1_make_pyomo_dict(foo_start_grnha_qop6lzt, arrays_qop6lzt)

    params['p_foo_end_grnha_qgop6lzt'] = fun.f1_make_pyomo_dict(foo_end_grnha_qgop6lzt, arrays_qgop6lzt)

    params['p_me_cons_grnha_qfgop6lzt'] = fun.f1_make_pyomo_dict(me_cons_grnha_qfgop6lzt, arrays_qfgop6lzt)

    params['p_dry_mecons_t_fdp6zt'] = fun.f1_make_pyomo_dict(dry_mecons_t_fdp6zt, arrays_fdp6zt)

    params['p_volume_grnha_qfgop6lzt'] = fun.f1_make_pyomo_dict(volume_grnha_qfgop6lzt, arrays_qfgop6lzt)

    params['p_dry_volume_t_fdp6zt'] = fun.f1_make_pyomo_dict(dry_volume_t_fdp6zt, arrays_fdp6zt)

    params['p_senesce_grnha_qdgop6lzt'] = fun.f1_make_pyomo_dict(senesce_grnha_qdgop6lzt, arrays_qdgop6lzt)

    params['p_poc_vol_fp6z'] = fun.f1_make_pyomo_dict(poc_vol_fp6z, arrays_fp6z)

    params['p_poc_con_p6lz'] = fun.f1_make_pyomo_dict(poc_con_p6lz, arrays_p6lz)

    params['p_poc_md_fp6z'] = fun.f1_make_pyomo_dict(poc_md_fp6z, arrays_fp6z)

    params['p_co2e_grnpas_qgop6lzt'] = fun.f1_make_pyomo_dict(co2e_grnpas_qgop6lzt, arrays_qgop6lzt)
    params['p_co2e_drypas_cons_dp6zt'] = fun.f1_make_pyomo_dict(co2e_drypas_cons_dp6zt, arrays_dp6zt)
    params['p_co2e_poc_p6z'] = fun.f1_make_pyomo_dict(co2e_poc_p6z, arrays_p6z8)
    params['p_co2e_pas_residue_v_phase_growth_qp6lrzt'] = fun.f1_make_pyomo_dict(co2e_pas_residue_v_phase_growth_qp6lrzt, arrays_qp6lrzt)


    ###########
    #report   #
    ###########
    ##maskz8 used to uncluster lp_vars
    fun.f1_make_r_val(r_vals,mask_fp_z8var_p6z,'mask_fp_z8var_p6z')

    ##keys
    fun.f1_make_r_val(r_vals,keys_d,'keys_d')
    fun.f1_make_r_val(r_vals,keys_f,'keys_f')
    fun.f1_make_r_val(r_vals,keys_p6,'keys_p6')
    fun.f1_make_r_val(r_vals,keys_g,'keys_g')
    fun.f1_make_r_val(r_vals,keys_l,'keys_l')
    fun.f1_make_r_val(r_vals,keys_o,'keys_o')
    fun.f1_make_r_val(r_vals,keys_p5,'keys_p5')
    fun.f1_make_r_val(r_vals,keys_r,'keys_r')
    fun.f1_make_r_val(r_vals,keys_t,'keys_t')
    fun.f1_make_r_val(r_vals,keys_k,'keys_k')

    ##store report vals
    fun.f1_make_r_val(r_vals,date_start_p6z % 364,'fp_date_start_p6z') #mod 364 so that all dates are from the start of the yr (makes it easier to compare in the report)
    fun.f1_make_r_val(r_vals,pasture_rt,'pasture_area_rt')
    fun.f1_make_r_val(r_vals,pastures,'keys_pastures')
    fun.f1_make_r_val(r_vals,length_p6z,'days_p6z',mask_fp_z8var_p6z,z_pos=-1)
    fun.f1_make_r_val(r_vals,pgr_grnha_qgop6lzt,'pgr_grnha_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,foo_endprior_grnha_qgop6lzt,'foo_end_grnha_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)#Green FOO prior to eos senescence
    fun.f1_make_r_val(r_vals,foo_start_grnha_qop6lzt,'foo_start_grnha_qop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,i_foo_graze_propn_gt,'i_foo_graze_propn_gt')
    fun.f1_make_r_val(r_vals,cons_grnha_t_qgop6lzt,'cons_grnha_t_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,me_cons_grnha_qfgop6lzt,'me_cons_grnha_qfgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,fun.f_divide(me_cons_grnha_qfgop6lzt, volume_grnha_qfgop6lzt),'nv_grnha_qfgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,fun.f_divide(dry_mecons_t_fdp6zt, dry_volume_t_fdp6zt),'nv_dry_fdp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,foo_ave_grnha_qgop6lzt,'foo_ave_grnha_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,dmd_diet_grnha_qgop6lzt,'dmd_diet_grnha_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,dry_foo_dp6zt,'dry_foo_dp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,dry_dmd_dp6zt,'dry_dmd_dp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,dry_mecons_t_fdp6zt,'dry_mecons_t_fdp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,poc_md_fp6z,'poc_md_fp6z',mask_fp_z8var_p6z,z_pos=-1)
    ###emissions
    fun.f1_make_r_val(r_vals,stock_ch4_grnpas_qgop6lzt,'stock_ch4_grnpas_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,stock_n2o_grnpas_qgop6lzt,'stock_n2o_grnpas_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,stock_ch4_drypas_dp6zt,'stock_ch4_drypas_dp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,stock_n2o_drypas_dp6zt,'stock_n2o_drypas_dp6zt',mask_fp_z8var_p6zt,z_pos=-2)
    fun.f1_make_r_val(r_vals,stock_ch4_poc_p6z,'stock_ch4_poc_p6z',mask_fp_z8var_p6z,z_pos=-1)
    fun.f1_make_r_val(r_vals,stock_n2o_poc_p6z,'stock_n2o_poc_p6z',mask_fp_z8var_p6z,z_pos=-1)
    fun.f1_make_r_val(r_vals,pas_n2o_residue_cons_t,'pas_n2o_residue_cons_t')
    fun.f1_make_r_val(r_vals,grnpas_n2o_residue_qgop6lzt,'grnpas_n2o_residue_qgop6lzt',mask_fp_z8var_p6lzt,z_pos=-2)
    fun.f1_make_r_val(r_vals,np.swapaxes(n2o_pas_residue_v_phase_growth_qp6lrzt,2,4),'n2o_pas_residue_v_phase_growth_qp6zrlt',mask_fp_z8var_p6z[:,:,na,na,na],z_pos=-4)


