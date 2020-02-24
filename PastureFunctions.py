# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:46:24 2019
@author: john

Description of this pasture module: This representation includes at optimisation (ie the folowing options are represented in the variables of the model)
    Growth rate of pasture (PGR) varies with FOO at the start of the period and grazing intensity during the period
        Grazing intensity operates by altering the average FOO during the period
    The nutritive value of the green feed consumed (as represneted by ME & volume) varies with FOO & grazing intensity.
        Grazing intensity alters the average FOO during the period and the capacity of the animals to select a higher quality diet.
    Selective grazing of dry pasture. 2 dry pasture quality pools are represented and either can be selected for grazing
        Note: There is not a constraint that ensures that the high quality pool is grazed prior to the low quality pool (as there is in the stubble selective grazing)

This is the version that uses an extra axis on the array rather than a class.
"""


'''
import functions from other modules
'''
import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

import PropertyInputs as pinp
import FeedBudget as fdb
import UniversalInputs as uinp
import Functions as fun
import Periods as per
import Sensitivity as sen


########################
#phases                #
########################
# read the rotation phases information from inputs
phase_len       = uinp.structure['phase_len']
phases_rotn_df  = uinp.structure['phases']


########################
#constants required    #
########################
## define some parameters required to size arrays.
n_feed_pools        = uinp.n_feed_pools
n_dry_groups        = 2             # Low & high quality groups for dry feed
n_grazing_int       = 4             # 0, med & high grazing intensity in the growth/grazing activities
n_foo_levels        = 3             # Low, medium & high FOO level in the growth/grazing activities
n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1
n_lmu               = len(pinp.general['lmu_area'])
n_phases_rotn       = len(phases_rotn_df.index)
n_pasture_types     = pinp.n_pasture_types   #^ need to sort timing of the definition of pastures

length_f  = np.array(pinp.feed_inputs['feed_periods'].loc[:n_feed_periods-1,'length']) # converted to np. to get @jit working

egoflt = (n_feed_pools,               n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,                n_pasture_types)
dgoflt = (              n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,                n_pasture_types)
edft   = (n_feed_pools, n_dry_groups,                              n_feed_periods,                       n_pasture_types)
eft    = (n_feed_pools,                                            n_feed_periods,                       n_pasture_types)
dft    = (              n_dry_groups,                              n_feed_periods,                       n_pasture_types)
goflt  = (                            n_grazing_int, n_foo_levels, n_feed_periods, n_lmu,                n_pasture_types)
goft   = (                            n_grazing_int, n_foo_levels, n_feed_periods,                       n_pasture_types)
gft    = (                            n_grazing_int,               n_feed_periods,                       n_pasture_types)
gt     = (                            n_grazing_int,                                                     n_pasture_types)
oflt   = (                                           n_foo_levels, n_feed_periods, n_lmu,                n_pasture_types)
flrt   = (                                                         n_feed_periods, n_lmu, n_phases_rotn, n_pasture_types)
frt    = (                                                         n_feed_periods,        n_phases_rotn, n_pasture_types)
flt    = (                                                         n_feed_periods, n_lmu,                n_pasture_types)
lt     = (                                                                         n_lmu,                n_pasture_types)
ft     = (                                                         n_feed_periods,                       n_pasture_types)
# t      = (                                                                                               n_pasture_types)


def init_and_read_excel(filename, landuses):
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''
    ## set global on all variables required outside this function
    global i_reseeding_date_seed_t
    global i_reseeding_date_destock_t
    global i_reseeding_ungrazed_destock_t
    global i_reseeding_date_grazing_t
    global i_reseeding_foo_grazing_t
    global reseeding_machperiod_t

    global i_germ_scalar_lt
    global i_reseeding_fooscalar_lt
    global i_dry_dmd_reseeding_lt

    global i_grn_dmd_senesce_redn_ft
    global i_dry_dmd_ave_ft
    global i_dry_dmd_range_ft
    global i_dry_foo_high_ft
    global i_grn_cp_ft
    global i_dry_cp_ft
    global i_poc_dmd_ft
    global i_poc_foo_ft
    global i_grn_trampling_ft
    global i_dry_trampling_ft
    global i_grn_senesce_eos_ft
    global i_base_ft
    global i_grn_dmd_declinefoo_ft
    global i_grn_dmd_range_ft
    global grn_senesce_startfoo_ft
    global grn_senesce_pgrcons_ft
    global i_me_eff_gainlose_ft

    global i_me_maintenance_eft
    global p_dry_removal_t_dft

    global i_fxg_foo_oflt
    global i_fxg_pgr_oflt
    global c_fxg_a_oflt
    global c_fxg_b_oflt
    # global c_fxg_ai_oflt
    # global c_fxg_bi_oflt
    global p_foo_start_grnha_oflt

    global p_germination_flrt
    global p_foo_grn_reseeding_flrt
    global p_foo_dryh_reseeding_flrt
    global p_foo_dryl_reseeding_flrt

    global p_index_flrt     # and other indexes that are required

    global i_germination_std_t
    global i_ri_foo_t
    global i_end_of_gs_t
    global i_dry_decay_t
    # global poc_days_of_grazing_t
    global i_poc_intake_daily_flt
    global i_legume_t
    global i_grn_propn_reseeding_t
    global i_lmu_conservation_flt

    global i_phase_germ_df

    global i_foo_end_propn_gt
    global c_pgr_gi_scalar_gft
    global i_grn_dig_flt


    ### -define the vessels that will store the input data that require pre-defining
    ## all need pre-defining because inputs are read for each pasture type separately

    i_me_maintenance_eft            = np.zeros(eft,    dtype = np.float64)  # M/D level for target LW pattern
    c_pgr_gi_scalar_gft             = np.zeros(gft,    dtype = np.float64)  # numpy array of pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
    i_foo_end_propn_gt              = np.zeros(gt,     dtype = np.float64)  # numpy array of proportion of available feed consumed for each grazing intensity level.

    i_fxg_foo_oflt                  = np.zeros(oflt,   dtype = np.float64)  # numpy array of FOO level       for the FOO/growth/grazing variables.
    i_fxg_pgr_oflt                  = np.zeros(oflt,   dtype = np.float64)  # numpy array of PGR level       for the FOO/growth/grazing variables.
    p_foo_start_grnha_oflt          = np.zeros(oflt,   dtype = np.float64)  # parameters for the growth/grazing activities: initial FOO
    c_fxg_a_oflt                    = np.zeros(oflt,   dtype = np.float64)  # numpy array of coefficient a   for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_oflt                    = np.zeros(oflt,   dtype = np.float64)  # numpy array of coefficient b   for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_oflt                   = np.zeros(oflt,   dtype = np.float64)  # numpy array of coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_oflt                   = np.zeros(oflt,   dtype = np.float64)  # numpy array of coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO

    i_grn_dig_flt                   = np.zeros(flt,    dtype = np.float64)  # numpy array of inputs for green pasture digestibility on each LMU.
    i_poc_intake_daily_flt          = np.zeros(flt,    dtype = np.float64)  # intake per day of pasture on crop paddocks prior to seeding
    i_lmu_conservation_flt          = np.zeros(flt,    dtype = np.float64)  # minimum foo prior at end of each period to reduce risk of wind & water erosion

    i_germ_scalar_lt                = np.zeros(lt,     dtype = np.float64)  # scale the germination levels for each lmu
    i_reseeding_fooscalar_lt        = np.zeros(lt,     dtype = np.float64)  # scalar for FOO at the first grazing for the lmus
    i_dry_dmd_reseeding_lt          = np.zeros(lt,     dtype = np.float64)  # Average digestibility of any dry FOO at the first grazing (if there is any)

    i_me_eff_gainlose_ft            = np.zeros(ft,     dtype = np.float64)  # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_grn_trampling_ft              = np.zeros(ft,     dtype = np.float64)  # numpy array of inputs for green pasture trampling in each feed period.
    i_dry_trampling_ft              = np.zeros(ft,     dtype = np.float64)  # numpy array of inputs for dry pasture trampling   in each feed period.
    i_grn_senesce_daily_ft          = np.zeros(ft,     dtype = np.float64)  # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_ft            = np.zeros(ft,     dtype = np.float64)  # proportion of green feed that senesces in period (due to completing life cycle)
    i_base_ft                       = np.zeros(ft,     dtype = np.float64)  # lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_ft         = np.zeros(ft,     dtype = np.float64)  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_ft              = np.zeros(ft,     dtype = np.float64)  # range in digestibility within the sward for green feed
    i_grn_dmd_senesce_redn_ft       = np.zeros(ft,     dtype = np.float64)  # reduction in digestibility of green feed when it senesces
    i_dry_dmd_ave_ft                = np.zeros(ft,     dtype = np.float64)  # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_dry_dmd_range_ft              = np.zeros(ft,     dtype = np.float64)  # range in digestibility of dry feed if it is not grazed
    i_dry_foo_high_ft               = np.zeros(ft,     dtype = np.float64)  # expected foo for the dry pasture in the high quality pool
    i_grn_cp_ft                     = np.zeros(ft,     dtype = np.float64)  # crude protein content of green feed
    i_dry_cp_ft                     = np.zeros(ft,     dtype = np.float64)  # crude protein content of dry feed
    i_poc_dmd_ft                    = np.zeros(ft,     dtype = np.float64)  # digestibility of pasture consumed on crop paddocks
    i_poc_foo_ft                    = np.zeros(ft,     dtype = np.float64)  # foo of pasture consumed on crop paddocks

    i_reseeding_date_seed_t         = np.zeros(n_pasture_types, dtype = dt.datetime)  # date of seeding this pasture type (will be read in from inputs)
    i_reseeding_date_destock_t      = np.zeros(n_pasture_types, dtype = dt.datetime)  # date of destocking this pasture type prior to reseeding (will be read in from inputs)
    i_reseeding_ungrazed_destock_t  = np.zeros(n_pasture_types, dtype = np.float64)  # kg of FOO that was not grazed prior to seeding occurring (if spring sown)
    i_reseeding_date_grazing_t      = np.zeros(n_pasture_types, dtype = dt.datetime)  # date of first grazing of reseeded pasture (will be read in from inputs)
    i_reseeding_foo_grazing_t       = np.zeros(n_pasture_types, dtype = np.float64)  # FOO at time of first grazing
    # reseeding_machperiod_t          = np.zeros(n_pasture_types, dtype = np.float64)  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
    i_germination_std_t             = np.zeros(n_pasture_types, dtype = np.float64)  # standard germination level for the standard soil type in a continuous pasture rotation
    i_ri_foo_t                      = np.zeros(n_pasture_types, dtype = np.float64)  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    i_end_of_gs_t                   = np.zeros(n_pasture_types, dtype = np.float64)  # the period number when the pasture senesces
    i_dry_decay_t                   = np.zeros(n_pasture_types, dtype = np.float64)  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)
    # poc_days_of_grazing_t           = np.zeros(n_pasture_types, dtype = np.float64)  # number of days after the pasture break that (moist) seeding can begin
    i_legume_t                      = np.zeros(n_pasture_types, dtype = np.float64)  # proportion of legume in the sward
    i_grn_propn_reseeding_t         = np.zeros(n_pasture_types, dtype = np.float64)  # Proportion of the FOO available at the first grazing that is green

    ## define the numpy arrays that will be the output from the pre-calcs for pyomo
    p_germination_flrt              = np.zeros(flrt,   dtype = np.float64)  # parameters for rotation phase variable: germination (kg/ha)
    p_foo_grn_reseeding_flrt        = np.zeros(flrt,   dtype = np.float64)  # parameters for rotation phase variable: feed lost and gained during destocking and then grazing of resown pasture (kg/ha)
    p_foo_dryh_reseeding_flrt       = np.zeros(flrt,   dtype = np.float64)  # parameters for rotation phase variable: high quality dry feed gained from grazing of resown pasture (kg/ha)
    p_foo_dryl_reseeding_flrt       = np.zeros(flrt,   dtype = np.float64)  # parameters for rotation phase variable: low quality dry feed gained from grazing of resown pasture (kg/ha)
    p_dry_removal_t_dft             = np.zeros(egoflt, dtype = np.float64)  # parameters for the dry feed grazing activities: Total DM removal from the tonne consumed (includes trampling)

    index_t                       = np.asarray(landuses)                      # pasture type index description
    index_l                       = pinp.general['lmu_area'].index.to_numpy() # lmu index description
    index_f                       = [*range(n_feed_periods)]
    index_r                       = uinp.structure['phases'].index.to_numpy()

    #^ an option to create the index for the parameter arrays is
    p_index_flrt                    =np.ix_(index_f, index_l, index_r, index_t) #or perhaps the cartesian_products function

    ### _read data for each pasture type from excel file into arrays
    for t,landuse in enumerate(landuses):
        exceldata = pinp.pasture_inputs[landuse]           # assign the pasture data to exceldata
        ## map the Excel data into the numpy arrays
        i_germination_std_t[t]              = exceldata['GermStd']
        i_ri_foo_t[t]                       = exceldata['RIFOO']
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
        i_reseeding_date_destock_t[t]       = exceldata['Date_Destocking']
        i_reseeding_ungrazed_destock_t[t]   = exceldata['FOOatSeeding']
        i_reseeding_date_grazing_t[t]       = exceldata['Date_ResownGrazing']
        i_reseeding_foo_grazing_t[t]        = exceldata['FOOatGrazing']

        i_grn_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_dry_trampling_ft[...,t].fill       (exceldata['Trampling'])
        i_grn_senesce_daily_ft[...,t]       = np.asfarray(exceldata['SenescePropn'])
        i_grn_senesce_eos_ft[...,t]         = np.asfarray(exceldata['SenesceEOS'])
        i_base_ft[...,t]                    = np.asfarray(exceldata['BaseLevelInput'])
        i_grn_dmd_declinefoo_ft[...,t]      = np.asfarray(exceldata['DigDeclineFOO'])
        i_grn_dmd_range_ft[...,t]           = np.asfarray(exceldata['DigSpread'])
        i_foo_end_propn_gt[...,t]           = np.asfarray(exceldata['FOOGrazePropn'])
        c_pgr_gi_scalar_gft[...,t]    = 1 - i_foo_end_propn_gt[...,t].reshape(-1,1)**2        \
                                       * (1 - np.asfarray(exceldata['PGRScalarH']))

        i_fxg_foo_oflt[0,:,:,t]             = exceldata['LowFOO'].to_numpy()
        i_fxg_foo_oflt[1,:,:,t]             = exceldata['MedFOO'].to_numpy()
        i_me_eff_gainlose_ft[...,t]         = exceldata['MaintenanceEff'].iloc[:,0].to_numpy()
        i_me_maintenance_eft[...,t]         = exceldata['MaintenanceEff'].iloc[:,1:].to_numpy().T
        ## # i_fxg_foo_oflt[-1,...] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        i_fxg_pgr_oflt[0,:,:,t]             = exceldata['LowPGR'].to_numpy()
        i_fxg_pgr_oflt[1,:,:,t]             = exceldata['MedPGR'].to_numpy()
        i_fxg_pgr_oflt[2,:,:,t]             = exceldata['MedPGR'].to_numpy()  #PGR for high (last entry) is the same as PGR for medium
        i_grn_dig_flt[...,t]                = exceldata['DigGrn'].to_numpy()  # numpy array of inputs for green pasture digestibility on each LMU.

        ### _NEEDS WORK
        i_phase_germ_df                     = exceldata['GermPhases']       #DataFrame with germ scalar and resown

    ## Some one time data manipulation for the inputs just read
    # i_phase_germ_df.index = [*range(len(i_phase_germ_df.index))]              # replace index read from Excel with numbers to match later merging
    i_phase_germ_df.reset_index(inplace=True)                                                  # replace index read from Excel with numbers to match later merging
    i_phase_germ_df.columns.values[range(phase_len)] = [*range(phase_len)]         # replace the landuse columns read from Excel with numbers to match later merging

    i_fxg_foo_oflt[2,...]  = 10000 #large number so that the np.searchsorted doesn't go above
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

    grn_senesce_startfoo_ft =1 -((1 -     i_grn_senesce_daily_ft) **  length_f.reshape(-1,1))          # proportion of start foo that senescences during the period, different formula than excel
    grn_senesce_pgrcons_ft  =1 -((1 -(1 - i_grn_senesce_daily_ft) ** (length_f.reshape(-1,1)+1))   \
                                 /        i_grn_senesce_daily_ft-1) / length_f.reshape(-1,1)     # proportion of the total growth & consumption that senescences during the period

def calculate_germ_and_reseed():
    ''' Calculate germination and reseeding parameters

    germination: create an array called p_germination_flrt being the parameters to be passed to pyomo.
    reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    Results are stored in p_...._reseeding

    requires phases_rotn_df as a global variable
    ^ currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed
    ## intermediate calculations are not stored, however, if they were stored the 'key variables' could change the values of the intermediate calcs which could then be fed into the parameter calculations (in a separate method)
    ## the above would provide more options for KVs and provide another step that may not need to be recalculated
    '''
    ## set global on all variables required outside this function
    global phase_germresow_df
    global p_germination_flrt
    global p_phase_area_frt
    global p_foo_grn_reseeding_flrt
    global p_foo_dryh_reseeding_flrt
    global p_foo_dryl_reseeding_flrt
    global reseeding_machperiod_t

    def update_reseeding_foo(period_t, proportion_t, total, propn_grn_t=1, dmd_dry_lt=0):
        ''' Update p_foo parameters with values for destocking & subsequent grazing (reseeding)

        period_t & proportion_t - an array [type] : the first period affected by the destocking or subsequent grazing.
        total_t                 - an array either [lmu] or [lmu,type] : foo to be spread between the period and the subsequent period.
        propn_grn_t             - an array [type] : proportion of the total feed available for grazing that is green.
        dmd_dry_lt              - an array [lmu,type] : dmd of dry feed (if any).

        the adjustments are spread between periods to allow for the pasture growth that can occur from the green feed
        and the amount of grazing available if the feed is dry
        '''
        ## set global on all variables required outside this function
        global p_foo_grn_reseeding_flrt
        global p_foo_dryh_reseeding_flrt
        global p_foo_dryl_reseeding_flrt

        total_lt      = np.zeros((n_lmu,n_pasture_types), dtype = np.float64)   # create the array total_lt with the required shape
        total_lt[:,:] = total                                                   # broadcast total_t into total_lt (to handle total not having an lmu axis)

        foo_change_lrt = total_lt[:,np.newaxis,:] * phase_germresow_df['resown'].to_numpy().reshape(-1,1).astype(float)  # create an array (phase x lmu) that is the value to be added for any phase that is resown
        foo_change_lrt[np.isnan(foo_change_lrt)] = 0

        ## ^This loop can be removed - no, advanced indexing returns a copy not a view so it can't be assigned to
        ## # 1. p_foo_grn_reseeding_flrt[period_t,:,:,*range(n_pasture_types)] += might work (see AdvanceIndexing.py)
        ## # 2. removed if on propn_grn_t and doing all the dry calculations even if prop_grn_t = 1. (makes the code look much neater)
        for t in range(n_pasture_types):
            proportion  = proportion_t[t]
            propn_grn   =  propn_grn_t[t]
            foo_change  = foo_change_lrt[...,t]
            period      =     period_t[t]
            next_period = (period+1) % n_feed_periods

            ave_dmd     = i_dry_dmd_ave_ft[period,t]
            range_dmd   = i_dry_dmd_range_ft[period,t]
            high_dmd    = ave_dmd+range_dmd/2
            low_dmd     = ave_dmd-range_dmd/2
            propn_high_l  = (dmd_dry_lt[...,t] - low_dmd) /  (high_dmd - low_dmd)
            propn_low_l   = 1 - propn_high_l

            p_foo_grn_reseeding_flrt[period,:,:,t]        += foo_change *    proportion  * propn_grn      # add the amount of green for the first period
            p_foo_grn_reseeding_flrt[next_period,:,:,t]   += foo_change * (1-proportion) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
            p_foo_dryh_reseeding_flrt[period,:,:,t]       += foo_change *    proportion  * (1-propn_grn) * propn_high_l        # add the amount of high quality dry for the first period
            p_foo_dryh_reseeding_flrt[next_period,:,:,t]  += foo_change * (1-proportion) * (1-propn_grn) * propn_high_l        # add the remainder to the next period (wrapped if past the 10th period)
            p_foo_dryl_reseeding_flrt[period,:,:,t]       += foo_change *    proportion  * (1-propn_grn) * propn_low_l         # add the amount of high quality dry for the first period
            p_foo_dryl_reseeding_flrt[next_period,:,:,t]  += foo_change * (1-proportion) * (1-propn_grn) * propn_low_l         # add the remainder to the next period (wrapped if past the 10th period)

    ##reset all initial values to 0              ^ required even if deleted in the other functions
    p_foo_grn_reseeding_flrt[...]  = 0          # array has been initialised, reset all values to 0
    p_foo_dryh_reseeding_flrt[...] = 0
    p_foo_dryl_reseeding_flrt[...] = 0

    ## map the germination and resowing to the rotation phases   ^ this needs to be revamped along with the germination inputs (see notes on rotatioin sets in book 2-2-20)
    phase_germresow_df = phases_rotn_df.copy() #copy bit needed so future changes dont alter initial df
    rp=np.empty([len(phase_germresow_df),n_pasture_types])
    resown_r=np.empty([len(phase_germresow_df),n_pasture_types])
    ###loop through each phase in the germ df then check if each phase isin the set.
    for t in range(n_pasture_types):
        phase_germresow_df['germ_scalar']=0 #set default to 0
        phase_germresow_df['resown']=False #set default to false
        for ix_row in i_phase_germ_df.index:
            ix_bool = pd.Series(data=True,index=range(len(phase_germresow_df)))
            for ix_col in range(i_phase_germ_df.shape[1]-2):    #-2 because two of the cols are germ and resowing
                c_set = uinp.structure[i_phase_germ_df.iloc[ix_row,ix_col]]
                ix_bool &= phase_germresow_df.loc[:,ix_col].reset_index(drop=True).isin(c_set) #had to drop index so that it would work (just said false when the index was different between series)
            #maps the relevant germ scalar and resown bool to the roation phase
            phase_germresow_df.loc[list(ix_bool),'germ_scalar'] = i_phase_germ_df.loc[ix_row, 'germ_scalar']  #have to make bool into a list for some reason it doesn't like a series
            phase_germresow_df.loc[list(ix_bool),'resown'] = i_phase_germ_df.loc[ix_row, 'resown']
        ###Now convert germ and resow into a numpy - each pasture goes on a different level
        rp[:,t] = phase_germresow_df['germ_scalar'].to_numpy()#.reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
        resown_r[:,t] = phase_germresow_df['resown'].to_numpy()#.reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
    lmu_lrt             = i_germ_scalar_lt[:,np.newaxis,:]              \
                          * sen.sam_germ_l.reshape(-1,1,1) # lmu germination scalar x SA on lmu scalar
    germination_lrt         = i_germination_std_t        \
                          * np.multiply(rp,lmu_lrt)          \
                          * sen.sam_germ                             # create an array rot phase x lmu
    germination_lrt[np.isnan(germination_lrt)]  = 0.0
    p_germination_flrt[0,...]           = germination_lrt    # set germination in first period to germination

    ## retain the (labour) period during which this pasture is reseeded. For machinery expenditure
    period_dates            = per.p_dates_df()['date']
    period_name             = per.p_dates_df().index
    reseeding_machperiod_t  = fun.period_allocation(period_dates,period_name,i_reseeding_date_seed_t)

    ## set the period definitions to the feed periods
    feed_period_dates   = list(pinp.feed_inputs['feed_periods']['date'])
    feed_period_name    = pinp.feed_inputs['feed_periods'].index

    ## calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    duration            = (i_reseeding_date_grazing_t
                         - i_reseeding_date_destock_t)
    periods_destocked   = fun.range_allocation( feed_period_dates     # proportion of each period that is not being grazed because destocked for reseeding
                                               ,feed_period_name
                                               ,i_reseeding_date_destock_t
                                               ,duration)
    p_phase_area_frt    = 1 - np.multiply(resown_r,periods_destocked[:,np.newaxis,:])  # parameters for rotation phase variable: area of pasture in each period (is 0 for resown phases during periods that resown pasture is not grazed )
                                                        

    ## calculate the green feed lost when pasture is destocked. Spread between periods based on date destocked
    period, proportion  = fun.period_proportion( feed_period_dates  # which feed period does destocking occur & the proportion that destocking occurs during the period.
                                                ,feed_period_name
                                                ,i_reseeding_date_destock_t)
    update_reseeding_foo(period, 1-proportion, -i_reseeding_ungrazed_destock_t)                                       # call function to remove the FOO lost for the periods. Assumed that all feed lost is green

    ## calculate the green & dry feed available when pasture first grazed after reseeding. Spread between periods based on date grazed
    reseed_foo_lt  =    i_reseeding_fooscalar_lt       \
                    *             sen.sam_pgr_l.reshape(-1,1)   \
                    * i_reseeding_foo_grazing_t                 # FOO at the first grazing for each lmu (kg/ha)
    period, proportion  = fun.period_proportion( feed_period_dates
                                                ,feed_period_name
                                                ,i_reseeding_date_grazing_t)       # which feed period does grazing occur
    update_reseeding_foo(period, 1-proportion, reseed_foo_lt,
                         propn_grn=i_grn_propn_reseeding_t, dmd_dry=i_dry_dmd_reseeding_lt)                            # call function to update green & dry feed in the periods.
    ## possible idea to create the dataframe with the key for the dict
    # m=np.array(['lmu1,','lmu2,','lmu3,'], dtype=np.object)
    ## if these are the index of a dataframe (df) then m = df.index.to_numpy()

    # n=np.array(['fp1','fpd2','fpd3'], dtype=np.object)

    # names = m.reshape(-1,1) + n.reshape(1,-1)

    #  np.ravel(names)
    #  array(['lmu1,fp1', 'lmu1,fpd2', 'lmu1,fpd3', 'lmu2,fp1', 'lmu2,fpd2', 'lmu2,fpd3', 'lmu3,fp1', 'lmu3,fpd2', 'lmu3,fpd3'], dtype=object)


# define a function that loops through feed periods to generate the foo profile for a specified germination and consumption
# examined some options to get rid of @jit error "UnsupportedError: Use of unknown opcode 'BUILD_LIST_UNPACK'"
# tried changing all inputs to the function to only include numpy arrays but didn't fix issue
# next step was to
#   try the bool filter approach to remove the 2nd loop
#   simplify the function to only be doing the f loop & maybe put it inside the function that calls the loop - not important here because it is not taking much time)
# @jit("float64[:,:](float64[:,:],float64[:,:])",nopython=True, nogil=True)
# @jit(nopython=True, nogil=True)
# @jit()
def calc_foo_profile(germination_flt, sam_pgr):
    '''
    Calculate the FOO level at the start of each feed period from the germination & sam on PGR provided

    Parameters
    ----------
    germination_flt - An array[feed_period,lmu,type] : kg of green feed germinating in the period.
    sam_pgr         - An array[feed_period,lmu,type] : SA multiplier for pgr.

    Returns
    -------
    An array[feed_period,lmu,type]: foo at the start of the period.
    '''
    ## reshape the inputs passed and set some initial variables that are required
    foo_start_flt       = np.zeros(flt, dtype = np.float64)
    foo_end_flt         = np.zeros(flt, dtype = np.float64)
    pgr_daily_l         = np.zeros(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array

    foo_end_flt[-1,:,:] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    ## loop through the pasture types
    for t in range(n_pasture_types):   #^ is this loop required?
        ## loop through the feed periods and calculate the foo at the start of each period
        for f in range(n_feed_periods):
            foo_start_flt[f,:,t]      = germination_flt[f,:,t] + foo_end_flt[f-1,:,t]
            ## alternative approach (a1)
            ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
            for l in [*range(n_lmu)]: #loop through lmu
                idx             = np.searchsorted(i_fxg_foo_oflt[:,f,l,t], foo_start_flt[f,l,t], side='left')   # find where foo_starts fits into the input data
                pgr_daily_l[l]  = sam_pgr[f,l,t] * (    c_fxg_a_oflt[idx,f,l,t]
                                                    +   c_fxg_b_oflt[idx,f,l,t]
                                                    * foo_start_flt[f,l,t])
            foo_end_flt[f,:,t]  = (  foo_start_flt[f,:,t]      * (1 - grn_senesce_startfoo_ft[f,t])
                                   + pgr_daily_l * length_f[f] * (1 -  grn_senesce_pgrcons_ft[f,t])) \
                                 *                               (1 -    i_grn_senesce_eos_ft[f,t])
    return foo_start_flt

## the following method generates the PGR & FOO parameters for the growth variables. Stored in a numpy array(lmu, feed period, FOO level, grazing intensity)
## def green_consumption:

def green_and_dry():
    ''' Populates the parameter arrays for green and dry feed.

    Pasture growth, consumption and senescence of green feed.
    Consumption & deferment of dry feed.


    Returns:
    -------
    The parameters in the existing variables.
    '''
    ## set global on all variables required outside this function
    global p_foo_start_grnha_oflt
    global p_foo_end_grnha_goflt
    global p_me_cons_grnha_egoflt
    global p_volume_grnha_egoflt
    global p_senesce2h_grnha_goflt
    global p_senesce2l_grnha_goflt
    global p_senesce_grnha_dgoflt
    global p_dry_mecons_t_edft
    global p_dry_volume_t_dft
    global p_dry_removal_t_dft
    global p_dry_transfer_t_dft

    ### _initialise numpy arrays used in this method
    grn_dmd_selectivity_goft = np.zeros(goft,   dtype = np.float64)
    senesce_propn_dgoflt     = np.zeros(dgoflt, dtype = np.float64)  #

    ### _set sensitivity variables used
    sam_pgr_flt                  = np.asfarray(  sen.sam_pgr
                                               * sen.sam_pgr_f.reshape(-1, 1, 1)
                                               * sen.sam_pgr_l.reshape( 1,-1, 1)
                                               * sen.sam_pgr_t.reshape( 1, 1,-1))
    ### _maximum foo achievable for each lmu & feed period (ungrazed pasture that germinates at the maximum level on that lmu)
    germination_pass_flt            = np.max(p_germination_flrt, axis=2)                                    #use p_germination because it includes any sensitivity that is carried out
    foo_start_ungrazed_flt          = calc_foo_profile(germination_pass_flt, sam_pgr_flt)# ^ passing the consumption value in a numpy array in an attempt to get the function @jit compatible
    max_foo_flt                     = np.maximum(i_fxg_foo_oflt[1,...], foo_start_ungrazed_flt)                  #maximum of ungrazed foo and foo from the medium foo level
    p_foo_start_grnha_oflt[2,...]   = np.maximum.accumulate(max_foo_flt,axis=1)                                #maximum accumulated along the feed periods axis, i.e. max to date
    p_foo_start_grnha_oflt          = np.maximum(p_foo_start_grnha_oflt
                                                 ,          i_base_ft[:,np.newaxis,:])         # to ensure that final foo can not be below 0
    ### _green, pasture growth
    pgr_grnday_oflt = np.maximum(0.01, i_fxg_pgr_oflt                  # use maximum to ensure that the pgr is non zero (because foo_days requires dividing by pgr)
                                      *  sam_pgr_flt)
    pgr_grnha_goflt =       pgr_grnday_oflt     \
                     *          length_f.reshape(-1,1,1)       \
                     * c_pgr_gi_scalar_gft[:,np.newaxis,:,np.newaxis,:]

    ### _green, final foo from initial, pgr and senescence
    foo_ungrazed_grnha_oflt  = p_foo_start_grnha_oflt         *(1-grn_senesce_startfoo_ft[:,np.newaxis,:])   \
                              +        pgr_grnha_goflt[0,...] *(1- grn_senesce_pgrcons_ft[:,np.newaxis,:])
    foo_endprior_grnha_goflt =  foo_ungrazed_grnha_oflt   \
                              -(foo_ungrazed_grnha_oflt
                                -           i_base_ft[:,np.newaxis,:])      \
                              *    i_foo_end_propn_gt[:,np.newaxis,np.newaxis,np.newaxis,:]
    p_foo_end_grnha_goflt    =      foo_endprior_grnha_goflt                   \
                              * (1 - i_grn_senesce_eos_ft[:,np.newaxis,:])

    ### _green, removal & dmi
    removal_grnha_goflt =np.maximum(0,   p_foo_start_grnha_oflt 
                               * (1 - grn_senesce_startfoo_ft[:,np.newaxis,:])
                                      +          pgr_grnha_goflt 
                               * (1 -  grn_senesce_pgrcons_ft[:,np.newaxis,:])
                                      - foo_endprior_grnha_goflt)          \
                         /       (1 -  grn_senesce_pgrcons_ft[:,np.newaxis,:])
    cons_grnha_t_goflt  =      removal_grnha_goflt   \
                         /(1+i_grn_trampling_ft[:,np.newaxis,:])

    ### _green, dmd & md from average and change due to foo & grazing intensity
    ## # to calculate foo_days requires calculating number of days in current period and adding days from the previous period (if required)
    min=-1; max = 0         #Clip between -1 and 0
    if (p_foo_start_grnha_oflt > p_foo_start_grnha_oflt[1:2,...]):       # 1:2 to retain the axis, but it only points to index level 1.
        min += 1; max +=1   #Clip between 0 and 1
    propn_period_oflt               = (  p_foo_start_grnha_oflt
                                       - p_foo_start_grnha_oflt[1:2,...])            \
                                     /           pgr_grnha_goflt[0,...]
    propn_periodprev_oflt           = (  p_foo_start_grnha_oflt [   : ,1:  ,:,:]
                                       - p_foo_start_grnha_oflt [  1:2,1:  ,:,:]
                                       -         pgr_grnha_goflt[0, : ,1:  ,:,:])     \
                                     /           pgr_grnha_goflt[0, : , :-1,:,:]      # pgr from the previous period
    foo_days_grnha_oflt             = np.clip(propn_period_oflt,min,max)              \
                                     *              length_f.reshape(-1,1,1)
    foo_days_grnha_oflt[:,1:,:,:]  += np.clip(propn_periodprev_oflt,min,max)          \
                                     *                  length_f[:-1].reshape(-1,1,1) # length from previous period
    grn_dmd_swardscalar_oflt        = (1 - i_grn_dmd_declinefoo_ft[:,np.newaxis,:])   \
                                     **          foo_days_grnha_oflt                  # multiplier on digestibility of the sward due to level of FOO (associated with destocking)
    grn_dmd_range_ft                = (       i_grn_dmd_range_ft
                                       *sen.sam_grn_dmd_range_f).reshape(-1,1)
    grn_dmd_selectivity_goft[1,...] = -0.5 * grn_dmd_range_ft                         # addition to digestibility associated with diet selection (level of grazing)
    grn_dmd_selectivity_goft[2:...] = 0
    grn_dmd_selectivity_goft[3,...] = +0.5 * grn_dmd_range_ft
    dmd_grnha_goflt                 =            i_grn_dig_flt                        \
                                     * grn_dmd_swardscalar_oflt                       \
                                     + grn_dmd_selectivity_goft[:,:,:,np.newaxis,:]
    grn_md_grnha_goflt              = fdb.dmd_to_md(dmd_grnha_goflt)

    ### _green, mei & volume
    foo_ave_grnha_goflt      = (p_foo_start_grnha_oflt
                                + p_foo_end_grnha_goflt)/2
    grn_ri_availability_goflt= fdb.ri_availability(foo_ave_grnha_goflt, i_ri_foo_t)
    grn_ri_quality_goflt     = fdb.ri_quality(dmd_grnha_goflt, i_legume_t)
    grn_ri_goflt             = np.maximum( 0.05                                        # set the minimum RI to 0.05
                                          ,     grn_ri_quality_goflt
                                          *grn_ri_availability_goflt)

    p_me_cons_grnha_egoflt   = fdb.effective_mei(      cons_grnha_t_goflt
                                                 ,     grn_md_grnha_goflt
                                                 , i_me_maintenance_eft[:,np.newaxis,np.newaxis,:,np.newaxis,:]
                                                 ,           grn_ri_goflt
                                                 ,i_me_eff_gainlose_ft[:,np.newaxis,:])
    p_volume_grnha_egoflt    = np.tile( cons_grnha_t_goflt / grn_ri_goflt              # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
                                       ,(n_feed_pools,1,1,1,1,1))
    ### _dry, DM decline (high = low pools)
    dry_decay_daily_ft                    = np.tile(  i_dry_decay_t                    # fill the _t array to _ft shape ^ alternative is to instantiate the array and assign with [...]
                                                    ,(n_feed_periods,1))
    dry_decay_daily_ft[0:i_end_of_gs_t-1] = 1
    dry_decay_period_ft                   = 1 - (1 - dry_decay_daily_ft)               \
                                           **                 length_f.reshape(-1,1)
    p_dry_transfer_t_dft[...]             = 1000 * (1-dry_decay_period_ft)  # parameters for the dry feed transfer activities: quantity transferred
    ### _dry, dmd & foo of feed consumed
    dry_dmd_adj_ft  = np.max(i_dry_dmd_ave_ft,axis=0) * (1 -sen.sam_dry_dmd_decline)   \
                     +       i_dry_dmd_ave_ft         *     sen.sam_dry_dmd_decline    # do sensitivity adjustment for dry_dmd_input based on increasing/reducing the reduction in dmd from the maximum (starting value)
    dry_dmd_high_ft = dry_dmd_adj_ft + i_dry_dmd_range_ft/2
    dry_dmd_low_ft  = dry_dmd_adj_ft - i_dry_dmd_range_ft/2
    dry_dmd_dft     = np.stack((dry_dmd_high_ft, dry_dmd_low_ft),axis=0)    # create an array with a new axis 0 by stacking the existing arrays
    ## # ^ could implement a dry foo sensitivity analysis here
    dry_foo_high_ft = i_dry_foo_high_ft * 3/4
    dry_foo_low_ft  = i_dry_foo_high_ft * 1/4                               # assuming half the foo is high quality and the remainder is low quality
    dry_foo_dft     = np.stack((dry_foo_high_ft, dry_foo_low_ft),axis=0)  # create an array with a new axis 0 by stacking the existing arrays

    ### _dry, volume of feed consumed per tonne
    dry_ri_availability_dft = fdb.ri_availability(dry_foo_dft,i_ri_foo_t)
    dry_ri_quality_dft      = fdb.ri_quality(dry_dmd_dft, i_legume_t)
    dry_ri_dft              = dry_ri_quality_dft * dry_ri_availability_dft
    dry_ri_dft[dry_ri_dft<0.05] = 0.05 #set the minimum RI to 0.05
    p_dry_volume_t_dft  = 1000 / dry_ri_dft                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed

    ### _dry, ME consumed per tonne consumed
    dry_md_dft           = fdb.dmd_to_md(dry_dmd_dft)
    dry_md_edft          = np.stack([dry_md_dft * 1000] * n_feed_pools, axis = 0)
    p_dry_mecons_t_edft  = fdb.effective_mei( 1000                                    # parameters for the dry feed grazing activities: Total ME of the tonne consumed
                                             ,           dry_md_edft
                                             , i_me_maintenance_eft[:,np.newaxis,:,:]
                                             ,           dry_ri_dft
                                             ,i_me_eff_gainlose_ft)

    ### _dry, animal removal
    p_dry_removal_t_dft[...]  = 1000 * (1 + i_dry_trampling_ft)

    ### _senescence from green to dry
    ## # _green, total senescence
    senesce_total_grnha_goflt   = p_foo_start_grnha_oflt    \
                                 +        pgr_grnha_goflt   \
                                 -    removal_grnha_goflt   \
                                 -  p_foo_end_grnha_goflt
    grn_dmd_senesce_goflt       =               dmd_grnha_goflt       \
                                 - i_grn_dmd_senesce_redn_ft[:,np.newaxis,:]
    senesce2h_propn_goflt       = ( grn_dmd_senesce_goflt
                                   -    dry_dmd_low_ft[:,np.newaxis,:])       \
                                 /(    dry_dmd_high_ft[:,np.newaxis,:]
                                   -    dry_dmd_low_ft[:,np.newaxis,:])
    senesce_propn_dgoflt[0,...]  = ( grn_dmd_senesce_goflt                     # senescence to high pool
                                    -    dry_dmd_low_ft[:,np.newaxis,:])       \
                                  /(    dry_dmd_high_ft[:,np.newaxis,:]
                                    -    dry_dmd_low_ft[:,np.newaxis,:])
    senesce_propn_dgoflt[1,...] = 1- senesce_propn_dgoflt[0,...]              # senescence to low pool
    p_senesce2h_grnha_goflt     = senesce_total_grnha_goflt *    senesce2h_propn_goflt     # parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    p_senesce2l_grnha_goflt     = senesce_total_grnha_goflt * (1-senesce2h_propn_goflt)    # parameters for the growth/grazing activities: quantity of green that senesces to the low pool
    p_senesce_grnha_dgoflt      = senesce_total_grnha_goflt *      senesce_propn_dgoflt                                   # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool


def poc_con():             #^ This doesn't look right. I think that some calculations are required to calculate the area of the triangle
    '''
    Returns
    -------
    Dict for pyomo.
        The amount of pasture consumption that can occur on crop paddocks each day before seeding
        - this is adjusted for lmu and feed period
    '''
    df_poc_con = i_poc_intake_daily_flt
    return df_poc_con.stack().to_dict()

def poc_md():
    '''
    Returns
    -------
    Dict for pyomo.
        The quality of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    p_md_ft=list(map(fdb.dmd_to_md,  i_poc_dmd_ft)) #could use list comp but thought it was a good place to practise map
    return dict(enumerate(p_md_ft))  # may need np.ndenumerate() to use with an array

def poc_vol():
    '''
    Returns
    -------
    Dict for pyomo.
        The relative intake of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    # ri_qual = np.asarray([fdb.ri_quality(dmd, i_legume_t) for dmd in i_poc_dmd_ft])       #could use map ie list(map(fdb.ri_quality, md, repeat(annual.legume))) (repeat is imported from itertools)
    # ri_quan = np.asarray([fdb.ri_availability(foo, i_ri_foo_t) for foo in i_poc_foo_ft])
    ri_qual_ft = fdb.ri_quality(i_poc_dmd_ft, i_legume_t)       # passing a numpy array
    ri_quan_ft = fdb.ri_availability(i_poc_foo_ft, i_ri_foo_t)
    p_poc_vol_ft = 1/(ri_qual_ft*ri_quan_ft)
    return dict(enumerate(p_poc_vol_ft))  # may need np.ndenumerate() to use with an array
