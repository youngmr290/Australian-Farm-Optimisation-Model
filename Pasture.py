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
# import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

import PropertyInputs as pinp
import FeedBudget as fdb
import UniversalInputs as uinp
import Functions as fun
import Periods as per


########################
##phases               #
########################
## read the rotation phases information from inputs
phase_len       = uinp.structure['phase_len']
phases_rotn_df  = uinp.structure['phases']
pasture_sets    = uinp.structure['pasture_sets']
pastures        = uinp.structure['pastures']

########################
##constants required   #
########################
## define some parameters required to size arrays.
n_feed_pools    = len(uinp.structure['sheep_pools'])
n_dry_groups    = len(uinp.structure['dry_groups'])           # Low & high quality groups for dry feed
n_grazing_int   = len(uinp.structure['grazing_int'])          # grazing intensity in the growth/grazing activities
n_foo_levels    = len(uinp.structure['foo_levels'])           # Low, medium & high FOO level in the growth/grazing activities
n_feed_periods  = len(pinp.feed_inputs['feed_periods']) - 1
n_lmu           = len(pinp.general['lmu_area'])
n_phases_rotn   = len(phases_rotn_df.index)
n_pasture_types = len(pastures)   #^ need to sort timing of the definition of pastures

i_feed_period_dates   = list(pinp.feed_inputs['feed_periods']['date'])
t_list = [*range(n_pasture_types)]

arable_l = np.array(pinp.crop['arable']).reshape(-1)
length_f  = np.array(pinp.feed_inputs['feed_periods'].loc[:n_feed_periods-1,'length']) # converted to np. to get @jit working
feed_period_dates_f = np.array(i_feed_period_dates,dtype='datetime64[D]')


egoflt = (n_feed_pools, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
dgoflt = (n_dry_groups, n_grazing_int, n_foo_levels, n_feed_periods, n_lmu, n_pasture_types)
edft   = (n_feed_pools, n_dry_groups, n_feed_periods, n_pasture_types)
eft    = (n_feed_pools, n_feed_periods, n_pasture_types)
dft    = (n_dry_groups, n_feed_periods, n_pasture_types)
dlt    = (n_dry_groups, n_lmu, n_pasture_types)
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

### -define the vessels that will store the input data that require pre-defining
## all need pre-defining because inputs are in separate pasture type arrays
i_phase_germ_dict = dict()

i_me_maintenance_eft            = np.zeros(eft,  dtype = 'float64')  # M/D level for target LW pattern
c_pgr_gi_scalar_gft             = np.zeros(gft,  dtype = 'float64')  # numpy array of pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)
i_foo_end_propn_gt              = np.zeros(gt,   dtype = 'float64')  # numpy array of proportion of available feed consumed for each grazing intensity level.

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
i_grn_cp_ft                     = np.zeros(ft,  dtype = 'float64')  # crude protein content of green feed
i_dry_cp_ft                     = np.zeros(ft,  dtype = 'float64')  # crude protein content of dry feed
i_poc_dmd_ft                    = np.zeros(ft,  dtype = 'float64')  # digestibility of pasture consumed on crop paddocks
i_poc_foo_ft                    = np.zeros(ft,  dtype = 'float64')  # foo of pasture consumed on crop paddocks

i_reseeding_date_seed_t         = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of seeding this pasture type (will be read in from inputs)
i_seeding_end_t                 = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of seeding this pasture type (will be read in from inputs)
i_reseeding_date_destock_t      = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of destocking this pasture type prior to reseeding (will be read in from inputs)
i_reseeding_ungrazed_destock_t  = np.zeros(n_pasture_types, dtype = 'float64')  # kg of FOO that was not grazed prior to seeding occurring (if spring sown)
i_reseeding_date_grazing_t      = np.zeros(n_pasture_types, dtype = 'datetime64[D]')  # date of first grazing of reseeded pasture (will be read in from inputs)
i_reseeding_foo_grazing_t       = np.zeros(n_pasture_types, dtype = 'float64')  # FOO at time of first grazing
# reseeding_machperiod_t          = np.zeros(n_pasture_types, dtype = 'float64')  # labour/machinery period in which reseeding occurs ^ instantiation may not be required
i_germination_std_t             = np.zeros(n_pasture_types, dtype = 'float64')  # standard germination level for the standard soil type in a continuous pasture rotation
i_ri_foo_t                      = np.zeros(n_pasture_types, dtype = 'float64')  # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
# poc_days_of_grazing_t           = np.zeros(n_pasture_types, dtype = 'float64')  # number of days after the pasture break that (moist) seeding can begin
i_legume_t                      = np.zeros(n_pasture_types, dtype = 'float64')  # proportion of legume in the sward
i_grn_propn_reseeding_t         = np.zeros(n_pasture_types, dtype = 'float64')  # Proportion of the FOO available at the first grazing that is green

## define the numpy arrays that will be the output from the pre-calcs for pyomo
germination_flrt              = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: germination (kg/ha)
foo_grn_reseeding_flrt        = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: feed lost and gained during destocking and then grazing of resown pasture (kg/ha)
foo_dry_reseeding_flrt        = np.zeros(flrt,  dtype = 'float64')  # parameters for rotation phase variable: high quality dry feed gained from grazing of resown pasture (kg/ha)
foo_dry_reseeding_dflrt       = np.zeros(dflrt, dtype = 'float64')  # parameters for rotation phase variable: low & high quality dry feed gained from grazing of resown pasture (kg/ha)
dry_removal_t_dft             = np.zeros(dft,   dtype = 'float64')  # parameters for the dry feed grazing activities: Total DM removal from the tonne consumed (includes trampling)

## define the array that links rotation phase and pasture type
pasture_rt                    = np.zeros(rt, dtype = 'float64')


### _create numpy index for param dicts ^creating indexes is a bit slow
##the array returned must be of type object, if string the dict keys become a numpy string and when indexed in pyomo it doesn't work.
index_d                       = np.asarray(uinp.structure['dry_groups'])
index_e                       = np.asarray(uinp.structure['sheep_pools'])
index_f                       = np.asarray([*range(n_feed_periods)], dtype='object')
index_g                       = np.asarray(uinp.structure['grazing_int'])
index_l                       = pinp.general['lmu_area'].index.to_numpy() # lmu index description
index_o                       = np.asarray(uinp.structure['foo_levels'])
index_p                       = np.asarray(per.p_date2_df().index)
index_r                       = uinp.structure['phases'].index.to_numpy()
index_t                       = np.asarray(pastures)                      # pasture type index description

## plrt
arrays=[index_p, index_l, index_r, index_t]
index_plrt=fun.cartesian_product_simple_transpose(arrays)
index_plrt=tuple(map(tuple, index_plrt)) #create a tuple rather than a list because tuples are faster

## flrt
arrays=[index_f, index_l, index_r, index_t]
index_flrt=fun.cartesian_product_simple_transpose(arrays)
index_flrt=tuple(map(tuple, index_flrt)) #create a tuple rather than a list because tuples are faster

## oflt
arrays=[index_o, index_f, index_l, index_t]
index_oflt=fun.cartesian_product_simple_transpose(arrays)
index_oflt=tuple(map(tuple, index_oflt)) #create a tuple rather than a list because tuples are faster

## goflt
arrays=[index_g, index_o, index_f, index_l, index_t]
index_goflt=fun.cartesian_product_simple_transpose(arrays)
index_goflt=tuple(map(tuple, index_goflt)) #create a tuple rather than a list because tuples are faster

## egoflt
arrays=[index_e, index_g, index_o, index_f, index_l, index_t]
index_egoflt=fun.cartesian_product_simple_transpose(arrays)
index_egoflt=tuple(map(tuple, index_egoflt)) #create a tuple rather than a list because tuples are faster

## dgoflt
arrays=[index_d, index_g, index_o, index_f, index_l, index_t]
index_dgoflt=fun.cartesian_product_simple_transpose(arrays)
index_dgoflt=tuple(map(tuple, index_dgoflt)) #create a tuple rather than a list because tuples are faster

## dflrt
arrays=[index_d, index_f, index_l, index_r, index_t]
index_dflrt=fun.cartesian_product_simple_transpose(arrays)
index_dflrt=tuple(map(tuple, index_dflrt)) #create a tuple rather than a list because tuples are faster

## edft
arrays=[index_e, index_d, index_f, index_t]
index_edft=fun.cartesian_product_simple_transpose(arrays)
index_edft=tuple(map(tuple, index_edft)) #create a tuple rather than a list because tuples are faster

## dft
arrays=[index_d, index_f, index_t]
index_dft=fun.cartesian_product_simple_transpose(arrays)
index_dft=tuple(map(tuple, index_dft)) #create a tuple rather than a list because tuples are faster

## dlt
arrays=[index_d, index_l, index_t]
index_dlt=fun.cartesian_product_simple_transpose(arrays)
index_dlt=tuple(map(tuple, index_dlt)) #create a tuple rather than a list because tuples are faster

## flt
arrays=[index_f, index_l, index_t]
index_flt=fun.cartesian_product_simple_transpose(arrays)
index_flt=tuple(map(tuple, index_flt)) #create a tuple rather than a list because tuples are faster

## fl
arrays=[index_f, index_l]
index_fl=fun.cartesian_product_simple_transpose(arrays)
index_fl=tuple(map(tuple, index_fl)) #create a tuple rather than a list because tuples are faster

## ft
arrays=[index_f, index_t]
index_ft=fun.cartesian_product_simple_transpose(arrays)
index_ft=tuple(map(tuple, index_ft)) #create a tuple rather than a list because tuples are faster

## frt
arrays=[index_f, index_r, index_t]
index_frt=fun.cartesian_product_simple_transpose(arrays)
index_frt=tuple(map(tuple, index_frt)) #create a tuple rather than a list because tuples are faster

def map_excel(filename):
    '''Instantiate variables required and read inputs for the pasture variables from an excel file'''
    global grn_senesce_startfoo_ft
    global grn_senesce_pgrcons_ft
    global p_erosion_flt

    global i_end_of_gs_t
    global t_list
    ### -define the vessels that will store the input data that require pre-defining
    ## all need pre-defining because inputs are in separate pasture type arrays

    i_grn_senesce_daily_ft          = np.zeros(ft,  dtype = 'float64')  # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_ft            = np.zeros(ft,  dtype = 'float64')  # proportion of green feed that senesces in period (due to completing life cycle)
    dry_decay_daily_ft              = np.zeros(ft,  dtype = 'float64')  # daily decline in dry foo in each period
    i_end_of_gs_t                   = np.zeros(n_pasture_types, dtype = 'int')  # the period number when the pasture senesces
    i_dry_decay_t                   = np.zeros(n_pasture_types, dtype = 'float64')  # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)

    ### _map data from excel file into arrays
    ##loop through each pasture type
    for t, pasture in enumerate(pastures):
        exceldata = pinp.pasture_inputs[pasture]           # assign the pasture data to exceldata
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
        i_seeding_end_t[t]                  = exceldata['seeding_length']
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

        i_phase_germ_dict[pasture]          = exceldata['GermPhases'].copy()  #DataFrame with germ scalar and resown
        i_phase_germ_dict[pasture].reset_index(inplace=True)                                                  # replace index read from Excel with numbers to match later merging
        i_phase_germ_dict[pasture].columns.values[range(phase_len)] = [*range(phase_len)]         # replace the pasture columns read from Excel with numbers to match later merging

        ## define the link between rotation phase and pasture type while looping on pasture
        pasture_rt[:,t] = phases_rotn_df.iloc[:,-1].isin(pasture_sets[pasture])

    ### _one time data manipulation for the inputs just read
    ## calculate dry_decay_period (used in reseeding and green&dry)
    dry_decay_daily_ft[...] = i_dry_decay_t
    for t in range(n_pasture_types):
        dry_decay_daily_ft[0:i_end_of_gs_t[t]-1,t_list[t]] = 1
    dry_decay_period_ft[...] = 1 - (1 - dry_decay_daily_ft)               \
                              ** length_f.reshape(-1,1)

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

    grn_senesce_startfoo_ft =1 -((1 -     i_grn_senesce_daily_ft) **  length_f.reshape(-1,1))          # proportion of start foo that senescences during the period, different formula than excel
    grn_senesce_pgrcons_ft  =1 -((1 -(1 - i_grn_senesce_daily_ft) ** (length_f.reshape(-1,1)+1))   \
                                 /        i_grn_senesce_daily_ft-1) / length_f.reshape(-1,1)     # proportion of the total growth & consumption that senescences during the period

## define a function that loops through feed periods to generate the foo profile for a specified germination and consumption
def calc_foo_profile(germination_flt, dry_decay_ft, length_of_periods_f):
    '''
    Calculate the FOO level at the start of each feed period from the germination & sam on PGR provided

    Parameters
    ----------
    germination_flt     - An array[feed_period,lmu,type] : kg of green feed germinating in the period.
    dry_decay_ft        - An array[feed_period,type]     : decay rate of dry feed
    length_of_periods_f - An array[feed_period]          : days in each period
    Returns
    -------
    An array[feed_period,lmu,type]: foo at the start of the period.
    '''
    ## reshape the inputs passed and set some initial variables that are required
    grn_foo_start_flt   = np.zeros(flt, dtype = 'float64')
    grn_foo_end_flt     = np.zeros(flt, dtype = 'float64')
    dry_foo_start_flt   = np.zeros(flt, dtype = 'float64')
    dry_foo_end_flt     = np.zeros(flt, dtype = 'float64')
    pgr_daily_l         = np.zeros(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array

    grn_foo_end_flt[-1,:,:] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    dry_foo_end_flt[-1,:,:] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    ## loop through the pasture types
    for t in range(n_pasture_types):   #^ is this loop required?
        ## loop through the feed periods and calculate the foo at the start of each period
        for f in range(n_feed_periods):
            grn_foo_start_flt[f,:,t]      = germination_flt[f,:,t] + grn_foo_end_flt[f-1,:,t]
            dry_foo_start_flt[f,:,t]      =                          dry_foo_end_flt[f-1,:,t]
            ## alternative approach (a1)
            ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
            for l in [*range(n_lmu)]: #loop through lmu
                idx = np.searchsorted(i_fxg_foo_oflt[:,f,l,t], grn_foo_start_flt[f,l,t], side='left')   # find where foo_starts fits into the input data
                pgr_daily_l[l] = (c_fxg_a_oflt[idx,f,l,t]
                                  + c_fxg_b_oflt[idx,f,l,t]
                                  * grn_foo_start_flt[f,l,t])
            grn_foo_end_flt[f,:,t] = (grn_foo_start_flt[f,:,t]
                                      * (1 - grn_senesce_startfoo_ft[f,t])
                                      + pgr_daily_l * length_of_periods_f[f]
                                      * (1 -  grn_senesce_pgrcons_ft[f,t])) \
                                    * (1 - i_grn_senesce_eos_ft[f,t])
            senescence_l = grn_foo_start_flt[f,:,t]  \
                          +    pgr_daily_l * length_of_periods_f[f]  \
                          -  grn_foo_end_flt[f,:,t]
            dry_foo_end_flt[f,:,t] = dry_foo_start_flt[f,:,t] \
                                    * (1 - dry_decay_ft[f,t]) \
                                    + senescence_l
    return grn_foo_start_flt, dry_foo_start_flt


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
    global p_germination_flrt
    global p_phase_area_flrt
    global p_pas_sow_plrt
    global p_foo_dry_reseeding_dflrt
    global p_foo_grn_reseeding_flrt
    global p_erosion_flrt
    # global reseeding_machperiod_t

    ## define instantiate arrays that are assigned in slices
    na_erosion_flrt     = np.zeros(flrt, dtype = 'float64')
    na_phase_area_flrt  = np.zeros(flrt, dtype = 'float64')
    grn_destock_foo_flt = np.zeros(flt,  dtype = 'float64')
    dry_destock_foo_flt = np.zeros(flt,  dtype = 'float64')
    foo_na_flt          = np.zeros(flt,   dtype = 'float64')
    foo_na_destock_ft   = np.zeros(ft,   dtype = 'float64')
    germ_scalar_rt      = np.zeros(rt, dtype='float64')
    resown_rt           = np.zeros(rt, dtype='int')

    ## set initial values to 0 because function is called multiple times
    foo_grn_reseeding_flrt[...] = 0          # array has been initialised, reset all values to 0
    foo_dry_reseeding_flrt[...] = 0
    germination_flrt[...]       = 0

    ## set variables used in multiple locations
    destock_duration_t = (i_reseeding_date_grazing_t
                        - i_reseeding_date_destock_t)
    phase_germresow_df = phases_rotn_df.copy() #copy needed so subsequent changes don't alter initial df

    def update_reseeding_foo(period_t, proportion_t, foo_arable, foo_na, propn_grn=1): #, dmd_dry=0):
        ''' Update p_foo parameters with values for destocking & subsequent grazing (reseeding)

        period_t     - an array [type] : the first period affected by the destocking or subsequent grazing.
        proportion_t - an array [type] : the proportion of the period that has occurred prior to the destocking or subsequent grazing.
        foo_arable   - an array either [lmu] or [lmu,type] : foo on arable area.
        foo_na       - an array either [lmu] or [lmu,type] : foo on non arable area to be spread between the period and the subsequent period.
        propn_grn    - a scalar or an array [type] : proportion of the total feed available for grazing that is green.
        # dmd_dry      - a scalar or an array [lmu,type] : dmd of dry feed (if any).

        the adjustments are spread between periods to allow for the pasture growth that can occur from the green feed
        and the amount of grazing available if the feed is dry
        '''
        foo_arable_lt      = np.zeros(lt, dtype = 'float64')             # create the array foo_arable_lt with the required shape
        foo_arable_lt[...] = foo_arable                                  # broadcast foo_arable into foo_arable_lt (to handle foo_arable not having an lmu axis)
        foo_na_lt          = np.zeros(lt, dtype = 'float64')             # create the array foo_na_l with the required shape
        foo_na_lt[...]     = foo_na                                      # broadcast foo_na into foo_na_l (to handle foo_arable not having an lmu axis)
        propn_grn_t        = np.ones(n_pasture_types, dtype = 'float64') # create the array propn_grn_t with the required shape
        propn_grn_t[:]     = propn_grn                                   # broadcast propn_grn into propn_grn_t (to handle propn_grn not having an pasture type axis)

        ## the arable foo allocated to the rotation phases
        foo_arable_lrt = foo_arable_lt[:,np.newaxis,:]  \
            * arable_l.reshape(-1,1,1)  \
            * resown_rt
        foo_arable_lrt[np.isnan(foo_arable_lrt)] = 0

        foo_na_lr = np.sum(foo_na_lt[:,np.newaxis,:]
                           * (1-arable_l.reshape(-1,1,1))
                           * resown_rt, axis = -1)
        foo_na_lr[np.isnan(foo_na_lr)] = 0
        foo_change_lrt         = foo_arable_lrt
        foo_change_lrt[...,0] += foo_na_lr  #assuming all non-arable is pasture 0 (annuals)

        ## ^This loop might be able to be removed
        ## # 1. p_foo_grn_reseeding_flrt[period_t,:,:,*range(n_pasture_types)] += might work (see AdvanceIndexing.py & AdvIndex2.py)
        ## #  . p_foo_grn_reseeding_flrt[period_t,:,:,t_list]
        ## # 2. removed if on propn_grn_t and doing all the dry calculations even if prop_grn_t = 1. (makes the code look much neater)
        for t in range(n_pasture_types):
            proportion  = proportion_t[t]
            propn_grn   =  propn_grn_t[t]
            foo_change  = foo_change_lrt[...,t]
            period      =     period_t[t]
            next_period = (period+1) % n_feed_periods

            foo_grn_reseeding_flrt[period,:,:,t]        \
              += foo_change *    proportion  * propn_grn      # add the amount of green for the first period
            foo_grn_reseeding_flrt[next_period,:,:,t]   \
              += foo_change * (1-proportion) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
            foo_dry_reseeding_flrt[period,:,:,t]       \
              += foo_change *    proportion  * (1-propn_grn) * 0.5  # assume 50% in high & 50% into low pool. for the first period
            foo_dry_reseeding_flrt[next_period,:,:,t]  \
              += foo_change * (1-proportion) * (1-propn_grn) * 0.5  # add the remainder to the next period (wrapped if past the 10th period)

    ### germination and resowing to the rotation phases
    for t, pasture in enumerate(pastures):
        phase_germresow_df['germ_scalar']=0 #set default to 0
        phase_germresow_df['resown']=False #set default to false
        for ix_row in i_phase_germ_dict[pasture].index:
            ix_bool = pd.Series(data=True,index=range(len(phase_germresow_df)))
            for ix_col in range(i_phase_germ_dict[pasture].shape[1]-2):    #-2 because two of the cols are germ and resowing
                c_set = uinp.structure[i_phase_germ_dict[pasture].iloc[ix_row,ix_col]]
                ix_bool &= phase_germresow_df.loc[:,ix_col].reset_index(drop=True).isin(c_set) #had to drop index so that it would work (just said false when the index was different between series)
            ## maps the relevant germ scalar and resown bool to the rotation phase
            phase_germresow_df.loc[list(ix_bool),'germ_scalar'] = i_phase_germ_dict[pasture].loc[ix_row, 'germ_scalar']  #have to make bool into a list for some reason it doesn't like a series
            phase_germresow_df.loc[list(ix_bool),'resown'] = i_phase_germ_dict[pasture].loc[ix_row, 'resown']
        ## Convert germ and resow into a numpy - each pasture goes in a different slice
        germ_scalar_rt[:,t] = phase_germresow_df['germ_scalar'].to_numpy()#.reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
        resown_rt[:,t] = phase_germresow_df['resown'].to_numpy()#.reshape(-1,1)                       # extract the resown boolean from the dataframe and transpose (reshape to a column vector)

    ## germination on the arable area based on std germ, the rotation scalar and the lmu scalar
    arable_germination_lrt = i_germination_std_t        \
                            * i_germ_scalar_lt[:,np.newaxis,:]  \
                            * germ_scalar_rt      # create an array rot phase x lmu
    arable_germination_lrt[np.isnan(arable_germination_lrt)]  = 0.0
    ## germination on the non arable area based on annual pasture
    na_germination_lr = np.sum(i_germination_std_t[0:1]
                               * i_germ_scalar_lt[:,np.newaxis,0:1]
                               * pasture_rt, axis = -1)
    germination_flrt[0,...]    = arable_germination_lrt \
                                * arable_l.reshape(-1,1,1)     # set germination in first period to germination on arable area
    germination_flrt[0,...,0] += na_germination_lr  \
                                * (1-arable_l.reshape(-1,1)) # add non-arable area
    ##create the pyomo parameter
    germination_rav_flrt = germination_flrt.ravel()
    p_germination_flrt = dict(zip(index_flrt ,germination_rav_flrt))

    ### reseeding impacts due to destoking & then grazing
    ## calculate the green feed lost when pasture is destocked. Spread between periods based on date destocked
    ## # include proportion arable
    foo_arable_destock_t = i_reseeding_ungrazed_destock_t
    foo_na_destock_t =  i_reseeding_ungrazed_destock_t
    period_t, proportion_t = fun.period_proportion_np(feed_period_dates_f  # which feed period does destocking occur & the proportion that destocking occurs during the period.
                                                      ,i_reseeding_date_destock_t)
    update_reseeding_foo(period_t, 1-proportion_t,
                         -foo_arable_destock_t, -foo_na_destock_t)                                       # call function to remove the FOO lost for the periods. Assumed that all feed lost is green

    ## calculate the green & dry feed available when pasture first grazed after reseeding. Spread between periods based on date grazed
    foo_arable_reseed_lt = i_reseeding_fooscalar_lt       \
                          * i_reseeding_foo_grazing_t                 # FOO at the first grazing for each lmu (kg/ha)
    ## calculate the foo at the end of the destock period on the ungrazed area
    ## # foo at destocking plus growth from destocking to grazing
    periods_destocked_ft = fun.range_allocation_np(feed_period_dates_f
                    , i_reseeding_date_destock_t
                    , destock_duration_t)[0:n_feed_periods]
    foo_na_destock_ft[period_t,t_list] = foo_na_destock_t
    days_each_period_ft = periods_destocked_ft  \
                         * length_f[:,np.newaxis]
    period_t, proportion_t = fun.period_proportion_np(feed_period_dates_f
                                                      ,i_reseeding_date_grazing_t)       # which feed period does grazing occur
    ##  growth from destocking to grazing
    ## # growth of annual but with start foo varying with pasture type
    for t in range(n_pasture_types):
        foo_na_flt = foo_na_destock_ft[:,np.newaxis,t:t+1]  #broadcast into array
        grn_destock, dry_destock = calc_foo_profile(foo_na_flt
                                                    , dry_decay_period_ft
                                                    , days_each_period_ft)
        ## # assign the growth from the annual slice
        grn_destock_foo_flt[...,t] = grn_destock[...,0]
        dry_destock_foo_flt[...,t] = dry_destock[...,0]
    grn_destock_foo_lt = grn_destock_foo_flt[period_t+1,...]
    dry_destock_foo_lt = dry_destock_foo_flt[period_t+1,...]
    foo_na_reseed_lt =  grn_destock_foo_lt + dry_destock_foo_lt

    update_reseeding_foo(period_t, 1-proportion_t
                         , foo_arable_reseed_lt, foo_na_reseed_lt
                         , propn_grn=i_grn_propn_reseeding_t)                            # call function to update green & dry feed in the periods.

    ##convert dry seeding pas into pyomo param
    foo_dry_reseeding_dflrt[0,...] = foo_dry_reseeding_flrt
    foo_dry_reseeding_dflrt[1,...] = foo_dry_reseeding_flrt
    foo_dry_reseeding_rav_dflrt = foo_dry_reseeding_dflrt.ravel()
    p_foo_dry_reseeding_dflrt = dict(zip(index_dflrt ,foo_dry_reseeding_rav_dflrt))
    ##convert green seeding pas into pyomo param
    foo_grn_reseeding_rav_flrt = foo_grn_reseeding_flrt.ravel()
    p_foo_grn_reseeding_flrt = dict(zip(index_flrt ,foo_grn_reseeding_rav_flrt))

    ### sow param determination
    ## determine the labour periods pas seeding occurs
    i_seeding_length_t = i_seeding_end_t - i_reseeding_date_seed_t
    period_dates            = per.p_dates_df()['date']
    reseeding_machperiod_pt  = fun.range_allocation_np(period_dates
                                    ,i_reseeding_date_seed_t
                                    ,i_seeding_length_t
                                    ,True)
    ## combine with rotation reseeding requirement
    pas_sown_lrt = resown_rt * arable_l.reshape(-1,1,1)
    pas_sow_plrt = pas_sown_lrt * reseeding_machperiod_pt[:,np.newaxis,np.newaxis,:]
    pas_sow_rav_plrt = pas_sow_plrt.ravel()
    p_pas_sow_plrt = dict(zip(index_plrt ,pas_sow_rav_plrt))

    ### area of pasture being grazed and growing
    ## calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    arable_phase_area_flrt = (1-(resown_rt
                                 * periods_destocked_ft[:,np.newaxis,np.newaxis,:]))  \
                            * arable_l.reshape(-1,1,1)  \
                            * pasture_rt
    na_phase_area_flrt[...,0] = np.sum((1-(resown_rt
                                           * periods_destocked_ft[:,np.newaxis,np.newaxis,:]))
                                       * (1-arable_l.reshape(-1,1,1))
                                       * pasture_rt
                                       , axis = -1)
    phase_area_flrt = arable_phase_area_flrt + na_phase_area_flrt
    phase_area_rav_flrt = phase_area_flrt.ravel()
    p_phase_area_flrt = dict(zip(index_flrt ,phase_area_rav_flrt))

    ### erosion
    arable_erosion_flrt = i_lmu_conservation_flt[...,np.newaxis,:]  \
                         * arable_l.reshape(-1,1,1)  \
                         * pasture_rt
    na_erosion_flrt[...,0] = np.sum(i_lmu_conservation_flt[...,np.newaxis,:]
                                    * (1-arable_l.reshape(-1,1,1))
                                    * pasture_rt
                                    , axis = -1)
    erosion_flrt = arable_erosion_flrt + na_erosion_flrt
    erosion_rav_flrt = erosion_flrt.ravel()
    p_erosion_flrt = dict(zip(index_flrt ,erosion_rav_flrt))

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
    ## set global on the parameters (dicts) used in pyomo
    global p_foo_start_grnha_oflt
    global p_foo_end_grnha_goflt
    global p_me_cons_grnha_egoflt
    global p_volume_grnha_egoflt
    global p_senesce_grnha_dgoflt
    global p_dry_mecons_t_edft
    global p_dry_volume_t_dft
    global p_dry_removal_t_dft
    global p_dry_transfer_t_dft
    global p_nap_dflrt

    ### _initialise numpy arrays used only in this method
    grn_dmd_selectivity_goft = np.zeros(goft,   dtype = 'float64')
    senesce_propn_dgoflt     = np.zeros(dgoflt, dtype = 'float64')
    nap_dflrt                 = np.zeros(dflrt,    dtype = 'float64')
    dry_transfer_t_dft       = np.zeros(dft,    dtype = 'float64')
    foo_start_grnha_oflt     = np.zeros(oflt,   dtype = 'float64')

    ### _dry, DM decline (high = low pools)
    dry_transfer_t_dft[...] = 1000 * (1-dry_decay_period_ft)
    dry_transfer_t_rav_dft = dry_transfer_t_dft.ravel()
    p_dry_transfer_t_dft = dict(zip(index_dft ,dry_transfer_t_rav_dft))

    ### _non-arable areas in crop paddocks (the annual pasture available if not grazed)
    ## # is maximum ungrazed pasture in the growing season
    ## # _maximum foo achievable for each lmu & feed period (ungrazed pasture that germinates at the maximum level on that lmu)
    ## ^a potential error here when germination is spread across periods (because taking max of each period)
    germination_pass_flt = np.max(germination_flrt, axis=2)  #use p_germination because it includes any sensitivity that is carried out
    grn_foo_start_ungrazed_flt , dry_foo_start_ungrazed_flt \
         = calc_foo_profile(germination_pass_flt, dry_decay_period_ft, length_f)# ^ passing the consumption value in a numpy array in an attempt to get the function @jit compatible
    ## all pasture from na area into the Low pool (#1) because it is rank
    harvest_period  = fun.period_allocation(pinp.feed_inputs['feed_periods']['date'], pinp.feed_inputs['feed_periods'].index, pinp.crop['harv_date'])
    nap_dflrt[1,...,0] = dry_foo_start_ungrazed_flt[harvest_period,:,np.newaxis,0]  \
                       * (1-arable_l.reshape(-1,1))  \
                       * (1-np.sum(pasture_rt, axis=1))
    nap_rav_dflrt = nap_dflrt.ravel()
    p_nap_dflrt = dict(zip(index_dflrt ,nap_rav_dflrt))

    ### _green initial FOO
    max_foo_flt                 = np.maximum(i_fxg_foo_oflt[1,...], grn_foo_start_ungrazed_flt)                  #maximum of ungrazed foo and foo from the medium foo level
    foo_start_grnha_oflt[2,...] = np.maximum.accumulate(max_foo_flt,axis=1)                                #maximum accumulated along the feed periods axis, i.e. max to date
    foo_start_grnha_oflt[...]   = np.maximum(foo_start_grnha_oflt
                                             , i_base_ft[:,np.newaxis,:])         # to ensure that final foo can not be below 0
    foo_start_grnha_rav_oflt = foo_start_grnha_oflt.ravel()
    p_foo_start_grnha_oflt = dict(zip(index_oflt ,foo_start_grnha_rav_oflt))

    ### _green, pasture growth
    pgr_grnday_oflt = np.maximum(0.01, i_fxg_pgr_oflt)                  # use maximum to ensure that the pgr is non zero (because foo_days requires dividing by pgr)
    pgr_grnha_goflt =       pgr_grnday_oflt     \
                     *          length_f.reshape(-1,1,1)       \
                     * c_pgr_gi_scalar_gft[:,np.newaxis,:,np.newaxis,:]

    ### _green, final foo from initial, pgr and senescence
    foo_ungrazed_grnha_oflt  = foo_start_grnha_oflt    *(1-grn_senesce_startfoo_ft[:,np.newaxis,:])   \
                              + pgr_grnha_goflt[0,...] *(1- grn_senesce_pgrcons_ft[:,np.newaxis,:])
    foo_endprior_grnha_goflt =  foo_ungrazed_grnha_oflt \
                              -(foo_ungrazed_grnha_oflt
                                - i_base_ft[:,np.newaxis,:]) \
                              * i_foo_end_propn_gt[:,np.newaxis,np.newaxis,np.newaxis,:]
    foo_end_grnha_goflt = foo_endprior_grnha_goflt * (1-i_grn_senesce_eos_ft[:,np.newaxis,:])
    foo_end_grnha_rav_goflt = foo_end_grnha_goflt.ravel()
    p_foo_end_grnha_goflt = dict( zip(index_goflt ,foo_end_grnha_rav_goflt))

    ### _green, removal & dmi
    removal_grnha_goflt =np.maximum(0,   foo_start_grnha_oflt
                               * (1 - grn_senesce_startfoo_ft[:,np.newaxis,:])
                                      +          pgr_grnha_goflt
                               * (1 -  grn_senesce_pgrcons_ft[:,np.newaxis,:])
                                      - foo_endprior_grnha_goflt)          \
                         /       (1 -  grn_senesce_pgrcons_ft[:,np.newaxis,:])
    cons_grnha_t_goflt  =      removal_grnha_goflt   \
                         /(1+i_grn_trampling_ft[:,np.newaxis,:])

    ### _green, dmd & md from average and change due to foo & grazing intensity
    ## # to calculate foo_days requires calculating number of days in current period and adding days from the previous period (if required)

    ##set the default to Clip between -1 and 0 for low FOO level
    min_oflt = np.ones(n_foo_levels).reshape(-1,1,1,1) * -1
    max_oflt = np.zeros(n_foo_levels).reshape(-1,1,1,1)
    # and clip between 0 and 1 for high FOO level
    min_oflt[2,...] = 0
    max_oflt[2,...] = 1

    propn_period_oflt               = (  foo_start_grnha_oflt
                                       - foo_start_grnha_oflt[1:2,...])            \
                                     /           pgr_grnha_goflt[0,...]
    propn_periodprev_oflt           = (  foo_start_grnha_oflt [   : ,1:  ,:,:]
                                       - foo_start_grnha_oflt [  1:2,1:  ,:,:]
                                       -         pgr_grnha_goflt[0, : ,1:  ,:,:])     \
                                     /           pgr_grnha_goflt[0, : , :-1,:,:]      # pgr from the previous period
    foo_days_grnha_oflt             = np.clip(propn_period_oflt,min_oflt,max_oflt)              \
                                     *              length_f.reshape(-1,1,1)
    foo_days_grnha_oflt[:,1:,:,:]  += np.clip(propn_periodprev_oflt,min_oflt,max_oflt)          \
                                     *                  length_f[:-1].reshape(-1,1,1) # length from previous period
    grn_dmd_swardscalar_oflt        = (1 - i_grn_dmd_declinefoo_ft[:,np.newaxis,:])   \
                                     **          foo_days_grnha_oflt                  # multiplier on digestibility of the sward due to level of FOO (associated with destocking)
    grn_dmd_range_ft                = (       i_grn_dmd_range_ft)
    grn_dmd_selectivity_goft[1,...] = -0.5 * grn_dmd_range_ft                         # addition to digestibility associated with diet selection (level of grazing)
    grn_dmd_selectivity_goft[2,...] = 0
    grn_dmd_selectivity_goft[3,...] = +0.5 * grn_dmd_range_ft
    dmd_grnha_goflt                 =            i_grn_dig_flt                        \
                                     * grn_dmd_swardscalar_oflt                       \
                                     + grn_dmd_selectivity_goft[:,:,:,np.newaxis,:]
    grn_md_grnha_goflt              = fdb.dmd_to_md(dmd_grnha_goflt)

    ### _green, mei & volume
    foo_ave_grnha_goflt      = (foo_start_grnha_oflt
                                + foo_end_grnha_goflt)/2
    grn_ri_availability_goflt= fdb.ri_availability(foo_ave_grnha_goflt, i_ri_foo_t)
    grn_ri_quality_goflt     = fdb.ri_quality(dmd_grnha_goflt, i_legume_t)
    grn_ri_goflt             = np.maximum( 0.05                                        # set the minimum RI to 0.05
                                          ,     grn_ri_quality_goflt
                                          *grn_ri_availability_goflt)

    me_cons_grnha_egoflt   = fdb.effective_mei(      cons_grnha_t_goflt
                                                 ,     grn_md_grnha_goflt
                                                 , i_me_maintenance_eft[:,np.newaxis,np.newaxis,:,np.newaxis,:]
                                                 ,           grn_ri_goflt
                                                 ,i_me_eff_gainlose_ft[:,np.newaxis,:])
    me_cons_grnha_rav_egoflt = me_cons_grnha_egoflt.ravel()
    p_me_cons_grnha_egoflt = dict( zip(index_egoflt ,me_cons_grnha_rav_egoflt))

    volume_grnha_egoflt    =  cons_grnha_t_goflt / grn_ri_goflt              # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
    volume_grnha_rav_egoflt = volume_grnha_egoflt.ravel()
    p_volume_grnha_egoflt = dict(zip(index_egoflt ,volume_grnha_rav_egoflt))

    ### _dry, dmd & foo of feed consumed
    dry_dmd_adj_ft  = np.max(i_dry_dmd_ave_ft,axis=0)    \
                     +       i_dry_dmd_ave_ft                               # do sensitivity adjustment for dry_dmd_input based on increasing/reducing the reduction in dmd from the maximum (starting value)
    dry_dmd_high_ft = dry_dmd_adj_ft + i_dry_dmd_range_ft/2
    dry_dmd_low_ft  = dry_dmd_adj_ft - i_dry_dmd_range_ft/2
    dry_dmd_dft     = np.stack((dry_dmd_high_ft, dry_dmd_low_ft),axis=0)    # create an array with a new axis 0 by stacking the existing arrays

    dry_foo_high_ft = i_dry_foo_high_ft * 3/4
    dry_foo_low_ft  = i_dry_foo_high_ft * 1/4                               # assuming half the foo is high quality and the remainder is low quality
    dry_foo_dft     = np.stack((dry_foo_high_ft, dry_foo_low_ft),axis=0)  # create an array with a new axis 0 by stacking the existing arrays

    ### _dry, volume of feed consumed per tonne
    dry_ri_availability_dft = fdb.ri_availability(dry_foo_dft,i_ri_foo_t)
    dry_ri_quality_dft      = fdb.ri_quality(dry_dmd_dft, i_legume_t)
    dry_ri_dft              = dry_ri_quality_dft * dry_ri_availability_dft
    dry_ri_dft[dry_ri_dft<0.05] = 0.05 #set the minimum RI to 0.05
    dry_volume_t_dft  = 1000 / dry_ri_dft                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed
    dry_volume_t_rav_dft = dry_volume_t_dft.ravel()
    p_dry_volume_t_dft = dict(zip(index_dft ,dry_volume_t_rav_dft))

    ### _dry, ME consumed per tonne consumed
    dry_md_dft           = fdb.dmd_to_md(dry_dmd_dft)
    dry_md_edft          = np.stack([dry_md_dft * 1000] * n_feed_pools, axis = 0)
    dry_mecons_t_edft  = fdb.effective_mei( 1000                                    # parameters for the dry feed grazing activities: Total ME of the tonne consumed
                            ,           dry_md_edft
                            , i_me_maintenance_eft[:,np.newaxis,:,:]
                            ,           dry_ri_dft
                            ,i_me_eff_gainlose_ft)
    dry_mecons_t_rav_edft = dry_mecons_t_edft.ravel()
    p_dry_mecons_t_edft = dict(zip(index_edft ,dry_mecons_t_rav_edft))

    ### _dry, animal removal
    dry_removal_t_dft[...]  = 1000 * (1 + i_dry_trampling_ft)
    dry_removal_t_rav_dft = dry_removal_t_dft.ravel()
    p_dry_removal_t_dft = dict( zip(index_dft ,dry_removal_t_rav_dft))

    ### _senescence from green to dry
    ## # _green, total senescence
    senesce_total_grnha_goflt   = foo_start_grnha_oflt    \
                                 +        pgr_grnha_goflt   \
                                 -    removal_grnha_goflt   \
                                 -  foo_end_grnha_goflt
    grn_dmd_senesce_goflt       =               dmd_grnha_goflt       \
                                 - i_grn_dmd_senesce_redn_ft[:,np.newaxis,:]
    senesce_propn_dgoflt[0,...]  = ( grn_dmd_senesce_goflt                     # senescence to high pool
                                    -    dry_dmd_low_ft[:,np.newaxis,:])       \
                                  /(    dry_dmd_high_ft[:,np.newaxis,:]
                                    -    dry_dmd_low_ft[:,np.newaxis,:])
    senesce_propn_dgoflt[1,...] = 1- senesce_propn_dgoflt[0,...]              # senescence to low pool
    senesce_grnha_dgoflt      = senesce_total_grnha_goflt *      senesce_propn_dgoflt                                   # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    senesce_grnha_rav_dgoflt = senesce_grnha_dgoflt.ravel()
    p_senesce_grnha_dgoflt = dict( zip(index_dgoflt ,senesce_grnha_rav_dgoflt))


def poc_con():
    '''
    Returns
    -------
    Dict for pyomo.
        The amount of pasture consumption that can occur on crop paddocks each day before seeding
        - this is adjusted for lmu and feed period
    '''
    poc_con_fl = i_poc_intake_daily_flt[...,0]
    poc_con_rav_fl = poc_con_fl.ravel()
    p_poc_con_fl = dict(zip(index_fl, poc_con_rav_fl))
    return p_poc_con_fl

def poc_md():
    '''
    Returns
    -------
    Dict for pyomo.
        The quality of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    poc_md_f = fdb.dmd_to_md(i_poc_dmd_ft[...,0])
    poc_md_rav_f = poc_md_f.ravel()
    p_poc_md_f = dict(zip(index_f ,poc_md_rav_f))
    return p_poc_md_f

def poc_vol():
    '''
    Returns
    -------
    Dict for pyomo.
        The relative intake of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    ri_qual_f = fdb.ri_quality(i_poc_dmd_ft[...,0], i_legume_t[...,0])       # passing a numpy array
    ri_quan_f = fdb.ri_availability(i_poc_foo_ft[...,0], i_ri_foo_t[...,0])
    poc_vol_f = 1/(ri_qual_f*ri_quan_f)
    poc_vol_rav_f = poc_vol_f.ravel()
    p_poc_vol_f = dict(zip(index_f ,poc_vol_rav_f))
    return p_poc_vol_f
