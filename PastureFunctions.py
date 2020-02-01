# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:46:24 2019
module: pasture input module - contains all input data likely to vary for different regions or farms

extra info: - this could eventually interact with a user interface
            - interacts with kv's

key: green section title is major title
     '#' around a title is a minor section title
     std '#' comment about a given line of code

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
import Sensitivity as sen


########################
#phases                #
########################
# read the rotation phases information from inputs
phase_len       = uinp.structure['phase_len']
phases_rotn_df  = pd.Series(uinp.structure['rotations']['rot_phase']).str.split(expand=True)[:-1]   # drop the last row of the df which is the blank line at the bottom of the table


########################
#constants required    #
########################
## define some parameters required to size arrays. Will need to be read in prior to defining the Class
n_pasture_types     = 3             # Annual, Lucerne, Tedera

n_foo_levels        = 3             # Low, medium & high FOO level in the growth/grazing activities
n_grazing_int       = 4             # 0, med & high grazing intensity in the growth/grazing activities
n_feed_pools        = 4             # number of feed pools (by quality groups)
n_dry_groups        = 2             # Low & high quality groups for dry feed
n_phases_rotn       = len(phases_rotn_df.index)
n_lmu               = len(pinp.general['lmu_area'])
n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1
length_f  = np.array(pinp.feed_inputs['feed_periods'].loc[:n_feed_periods-1,'length']) # converted to np. to get @jit working

########################
#Pasture Class         #
########################
'''
Define the pasture class that contains the input data and the precalcs that generate the arrays of output data that will populate pyomo
Types are set at initialisation to catch input errors at the time rather than wait till it causes an error
The class expects the following variables to already be defined
    n_feed_periods: int (number of feed periods)
    n_phases_rotn: int (number of rotation phases)
    phases_rotn_df: dataframe of landuse sequences (rotation phase names)
    n_lmus: int (number of lmus)

'''
def init_and_read_excel(filename, landuse):
    '''Instantiate variables required and read inputs for the pasture class from an excel file'''
    ## set global on all variables required outside this function
    global i_reseeding
    global i_lmu
    global i_feed_period
    global i_fxg_foo_lfo
    global i_fxg_pgr_lfo
    global c_fxg_a_lfo
    global c_fxg_b_lfo
    # global c_fxg_ai_lfo
    # global c_fxg_bi_lfo
    global i_grn_trampling_f
    global i_dry_trampling_f
    global p_germination_rlf
    global p_foo_grn_reseeding_rlf
    global p_foo_dryh_reseeding_rlf
    global p_foo_dryl_reseeding_rlf
    global i_germination_std
    global i_ri_foo
    global i_end_of_gs
    global i_dry_decay
    global poc_days_of_grazing
    global i_poc_intake_daily
    global i_legume
    global i_grn_propn_reseeding
    global i_lmu_conservation
    global i_phase_germ_df
    global i_grn_senesce_eos_f
    global i_base_f
    global i_grn_dmd_declinefoo_f
    global i_grn_dmd_range_f
    global i_foo_end_propn_g
    global i_pgr_gi_scalar_fg
    global i_me_eff_gainlose_f
    global i_me_maintenance_fe
    global grn_senesce_startfoo_f
    global grn_senesce_pgrcons_f
    global i_grn_dig_lf


    ## define the vessels that will store the input data that require pre-defining
    i_reseeding            = dict()                                                        # to create a dict to store data about reseeding of the pasture. Populated when data read in from Excel
    i_lmu               = pd.DataFrame(index = pinp.general['lmu_area'].index)     # a dataframe to store data relevant to lmus
    i_feed_period       = pd.DataFrame(index = range(n_feed_periods))                   # a dataframe to store data relevant to feed periods
    i_fxg_foo_lfo            = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of FOO level       for the FOO/growth/grazing variables.
    i_fxg_pgr_lfo            = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of PGR level       for the FOO/growth/grazing variables.
    c_fxg_a_lfo              = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient a   for the FOO/growth/grazing variables. PGR = a + b FOO
    c_fxg_b_lfo              = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient b   for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_ai_lfo             = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
    # c_fxg_bi_lfo             = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
    i_grn_trampling_f        = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # ^ should this include the ,1 or just do newaxis when it is used numpy array of inputs for green pasture trampling in each feed period.
    i_dry_trampling_f        = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # ^ should this include the ,1 or just do newaxis when it is used numpy array of inputs for dry pasture trampling   in each feed period.
    ## define the numpy arrays that will be the output from the pre-calcs for pyomo
    p_germination_rlf          = np.zeros((n_phases_rotn, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: germination (kg/ha)
    p_foo_grn_reseeding_rlf    = np.zeros((n_phases_rotn, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: feed lost and gained during destocking and then grazing of resown pasture (kg/ha)
    p_foo_dryh_reseeding_rlf   = np.zeros((n_phases_rotn, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: high quality dry feed gained from grazing of resown pasture (kg/ha)
    p_foo_dryl_reseeding_rlf   = np.zeros((n_phases_rotn, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: low quality dry feed gained from grazing of resown pasture (kg/ha)

    ## read data from excel file into a dictionary
    exceldata = fun.xl_all_named_ranges(filename, landuse)           # read all range names from the Excel file from the specified sheet
    ## map the Excel data into the python variables
    # included :bool                             = exceldata['SA.Past_inc']  # this pasture is included
    i_germination_std                   = exceldata['GermStd']      # standard germination level for the standard soil type in a continuous pasture rotation
    i_ri_foo                              = exceldata['RIFOO']        # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
    # trampling                           = exceldata['Trampling']    # removed in lieu of green & dry array by period. the amount of pasture trampled per unit of pasture consumed
    i_end_of_gs                           = exceldata['EndGS']        # the period number when the pasture senesces
    i_dry_decay                         = exceldata['PastDecay']    # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)
    # poc_days_of_grazing: int                   = exceldata['POCDays']      # number of days after the pasture break that (moist) seeding can begin
    i_poc_intake_daily                           = exceldata['POCCons']      # intake per day of pasture on crop paddocks prior to seeding
    i_legume                              = exceldata['Legume']       # proportion of legume in the sward
    i_grn_propn_reseeding                        = exceldata['FaG_PropnGrn'] # Proportion of the FOO available at the first grazing that is green
    #                                            = exceldata['']
    # i_feed_period['included']               = exceldata['SA.PastGP_inc']# growth of this pasture in this period is included
    i_feed_period['grn_dmd_senesce_redn']   = exceldata['DigRednSenesce']   # reduction in digestibility of green feed when it senesces
    i_feed_period['dry_dmd_average']        = exceldata['DigDryAve']    # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
    i_feed_period['dry_dmd_range']          = exceldata['DigDryRange']  # range in digestibility of dry feed if it is not grazed
    i_feed_period['dry_foo_high']           = exceldata['FOODryH']      # expected foo for the dry pasture in the high quality pool
    i_feed_period['grn_crudeprotein']       = exceldata['CPGrn']        # crude protein content of green feed
    i_feed_period['dry_crudeprotein']       = exceldata['CPDry']        # crude protein content of dry feed
    i_feed_period['poc_dmd']                = exceldata['DigPOC']       # digestibility of pasture consumed on crop paddocks
    i_feed_period['poc_foo']                = exceldata['FOOPOC']       # foo of pasture consumed on crop paddocks
    # i_feed_period['']                       = exceldata['']
    # i_lmu['included']                       = exceldata['SA.PastL_inc']     # this pasture is included on this lmu
    i_lmu['germ_scalar']                    = exceldata['GermScalarLMU']    # scale the germination levels for each lmu
    i_lmu['reseeding_foo_scalar']           = exceldata['FaG_LMU']          # scalar for FOO at the first grazing for the lmus
    i_lmu['dmd_dry']                        = exceldata['FaG_digDry']       # Average digestibility of any dry FOO at the first grazing (if there is any)
    # i_lmu['']                               = exceldata['']
    i_lmu_conservation                      = exceldata['ErosionLimit']     # minimum foo prior at end of each period to reduce risk of wind & water erosion

    i_reseeding['seed_date']                   = exceldata['Date_Seeding']         # date of seeding this pasture type (will be read in from inputs)
    i_reseeding['destock_date']                = exceldata['Date_Destocking']      # date of seeding this pasture type (will be read in from inputs)
    i_reseeding['ungrazed_at_destocking']      = exceldata['FOOatSeeding']         # kg of FOO that was not grazed prior to seeding occurring (if spring sown)
    i_reseeding['grazing_date']                = exceldata['Date_ResownGrazing']   # date of first grazing of reseeded pasture (will be read in from inputs)
    i_reseeding['grazing_foo']                 = exceldata['FOOatGrazing']         # FOO at time of first grazing

    i_phase_germ_df                            = exceldata['GermPhases']       #DataFrame with germ scalar and resown

    ## inputs read into numpy arrays
    i_grn_trampling_f.fill                        (exceldata['Trampling'])
    i_dry_trampling_f.fill                        (exceldata['Trampling'])
    i_grn_senesce_daily_f                 = np.asfarray(exceldata['SenescePropn'])    # proportion of green feed that senesces each period (due to leaf drop)
    i_grn_senesce_eos_f              = np.asfarray(exceldata['SenesceEOS'])      # proportion of green feed that senesces in period (due to completing life cycle)
    i_base_f                    = np.asfarray(exceldata['BaseLevelInput'])     #Lowest level that pasture can be consumed in each period
    i_grn_dmd_declinefoo_f          = np.asfarray(exceldata['DigDeclineFOO'])  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
    i_grn_dmd_range_f                = np.asfarray(exceldata['DigSpread'])       # range in digestibility within the sward for green feed
    i_foo_end_propn_g                 = np.asfarray(exceldata['FOOGrazePropn'])   # numpy array of proportion of available feed consumed for each grazing intensity level.
    i_pgr_gi_scalar_fg         =1-i_foo_end_propn_g**2*        \
                                     (1-exceldata['PGRScalarH'].to_numpy()).reshape(-1,1)      # # numpy array of pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)

    i_fxg_foo_lfo[:,:,0]                         = exceldata['LowFOO'].to_numpy().T
    i_fxg_foo_lfo[:,:,1]                         = exceldata['MedFOO'].to_numpy().T
    i_me_eff_gainlose_f                            = exceldata['MaintenanceEff[0]'].to_numpy()     # Reduction in efficiency if M/D is above requirement for target LW pattern
    i_me_maintenance_fe                            = exceldata['MaintenanceEff[1:-1]'].to_numpy()  # M/D level for target LW pattern
    ## # i_fxg_foo_lfo[:,:,-1] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
    ## # it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
    i_fxg_pgr_lfo[:,:,0]                         = exceldata['LowPGR'].to_numpy().T
    i_fxg_pgr_lfo[:,:,1]                         = exceldata['MedPGR'].to_numpy().T
    i_fxg_pgr_lfo[:,:,-1]                        = exceldata['MedPGR'].to_numpy().T  #PGR for high (last entry) is the same as PGR for medium
    i_grn_dig_lf                                 = exceldata['DigGrn'].to_numpy().T  # numpy array of inputs for green pasture digestibility on each LMU.
    ## Some one time data manipulation for the inputs just read
    i_phase_germ_df.index = [*range(len(i_phase_germ_df.index))]              # replace index read from Excel with numbers to match later merging
    i_phase_germ_df.columns.values[range(phase_len)] = [*range(phase_len)]         # replace the landuse columns read from Excel with numbers to match later merging
    i_fxg_foo_lfo[:,:,-1]  = 10000 #large number so that the searchsorted doesn't go above
    c_fxg_b_lfo[:,:,0] =   i_fxg_pgr_lfo[:,:,0]       \
                           /  i_fxg_foo_lfo[:,:,0]
    c_fxg_b_lfo[:,:,1] = ( i_fxg_pgr_lfo[:,:,1]
                             -i_fxg_pgr_lfo[:,:,0])        \
                           /( i_fxg_foo_lfo[:,:,1]
                             -i_fxg_foo_lfo[:,:,0])
    c_fxg_b_lfo[:,:,2] =  0

    c_fxg_a_lfo[:,:,0] =  0
    c_fxg_a_lfo[:,:,1] =  i_fxg_pgr_lfo[:,:,0]        \
                             -   c_fxg_b_lfo[:,:,1]        \
                             * i_fxg_foo_lfo[:,:,0]
    c_fxg_a_lfo[:,:,2] =  i_fxg_pgr_lfo[:,:,1] # because slope = 0

    grn_senesce_startfoo_f          =1-((1-i_grn_senesce_daily_f) ** length_f)          # proportion of start foo that senescences during the period, different formula than excel
    grn_senesce_pgrcons_f           =1-((1-(1-i_grn_senesce_daily_f) ** (length_f+1))   \
                                     /        i_grn_senesce_daily_f -1 ) / length_f     # proportion of the total growth & consumptioin that senescences during the period

def calculate_germ_and_reseed():
    ''' Calculate germination and reseeding parameters

    germination: create an array called p_germination(r,l) being the parameters to be passed to pyomo.
    reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    Results are stored in p_...._reseeding

    requires phases_rotn_df as a global variable
    ^ currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed
    ## intermediate calculations are not stored, however, if they were stored the 'key variables' could change the values of the intermediate calcs which could then be fed into the parameter calculations (in a separate method)
    ## the above would provide more options for KVs and provide another step that may not need to be recalculated
    '''
    ## set global on all variables required outside this function
    global phase_germ_df
    global p_germination_rlf
    global i_reseeding
    global p_phase_area_rf
    global p_foo_grn_reseeding_rlf
    global p_foo_dryh_reseeding_rlf
    global p_foo_dryl_reseeding_rlf

    def update_reseeding_foo(period, proportion, total, propn_grn=1, dmd_dry=0):
        ''' Update p_foo_grn_reseeding, p_foo_dryh_reseeding, p_foo_dryl__reseeding with values for destocking & subsequent grazing in the relevant feed periods

        period & proportion refer to the first period affected by the destocking or subsequent grazing
        total is the foo to be spread between the period and the subsequent period, can be a single value or a list. If list it must be by lmu
        propn_grn is a single value
        dmd_dry is a list of values by lmu

        the adjustments are spread between periods to allow for the pasture growth that can occur from the green feed
        and the amount of grazing available if the feed is dry
        '''
        ## set global on all variables required outside this function
        global p_foo_grn_reseeding_rlf
        global p_foo_dryh_reseeding_rlf
        global p_foo_dryl_reseeding_rlf

        foo_change = phase_germ_df['resown'].to_numpy().reshape(-1,1).astype(float) * total                            # create an array (phase x lmu) that is the value to be added for any phase that is resown
        foo_change[np.isnan(foo_change)] = 0

        next_period = (period+1) % n_feed_periods
        p_foo_grn_reseeding_rlf[:,:,period]        += foo_change *    proportion  * propn_grn      # add the amount of green for the first period
        p_foo_grn_reseeding_rlf[:,:,next_period]   += foo_change * (1-proportion) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
        if propn_grn < 1:
            ave_dmd     = i_feed_period['dry_dmd_average'][period]
            range_dmd   = i_feed_period['dry_dmd_range'][period]
            high_dmd    = ave_dmd+range_dmd/2
            low_dmd     = ave_dmd-range_dmd/2
            propn_high  = (dmd_dry-low_dmd)/(high_dmd-low_dmd)
            propn_low   = (high_dmd-dmd_dry)/(high_dmd-low_dmd)
            p_foo_dryh_reseeding_rlf[:,:,period]       += foo_change *    proportion  * (1-propn_grn) * propn_high        # add the amount of high quality dry for the first period
            p_foo_dryh_reseeding_rlf[:,:,next_period]  += foo_change * (1-proportion) * (1-propn_grn) * propn_high        # add the remainder to the next period (wrapped if past the 10th period)
            p_foo_dryl_reseeding_rlf[:,:,period]       += foo_change *    proportion  * (1-propn_grn) * propn_low         # add the amount of high quality dry for the first period
            p_foo_dryl_reseeding_rlf[:,:,next_period]  += foo_change * (1-proportion) * (1-propn_grn) * propn_low         # add the remainder to the next period (wrapped if past the 10th period)

    ##reset all initial values to 0              ^ required even if deleted in the other functions
    p_foo_grn_reseeding_rlf[...]  = 0          # array has been initialised, reset all values to 0
    p_foo_dryh_reseeding_rlf[...] = 0
    p_foo_dryl_reseeding_rlf[...] = 0

    ## map the germination phases to the rotation phases
    phase_germ_df      = pd.merge(phases_rotn_df,i_phase_germ_df,
                                   how     = 'left',
                                   left_on = [*range(phase_len)],
                                   right_on= [*range(phase_len)],
                                   sort    = False)
    phase_germ_df = phase_germ_df.set_index([*range(phase_len)])                                     # ^not sure the reason for the set_index may not be necessary
    rp                  = phase_germ_df['germ_scalar'].to_numpy().reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
    lmu                 = i_lmu['germ_scalar'].to_numpy()  \
                         * sen.sam_germ_l  # lmu germination scalar x SA on lmu scalar
    germination         = i_germination_std        \
                         * np.multiply(rp,lmu)          \
                         * sen.sam_germ                             # create an array rot phase x lmu
    germination[np.isnan(germination)]  = 0.0
    p_germination_rlf[:,:,0]           = germination                                                                       # set germination in first period to germination

    ## retain the (labour) period during which this pasture is reseeded. For machinery expenditure
    period_dates    = per.p_dates_df()['date']
    period_name     = per.p_dates_df().index
    date            = i_reseeding['seed_date']
    i_reseeding['seeding_period'] = fun.period_allocation(period_dates,period_name,date)

    ## set the period definitions to the feed periods
    feed_period_dates   = list(pinp.feed_inputs['feed_periods']['date'])
    feed_period_name    = pinp.feed_inputs['feed_periods'].index

    ## calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    phase_area_df               = pd.merge(phases_rotn_df,i_phase_germ_df,
                                     how='left', left_on=[*range(phase_len)], right_on=[*range(phase_len)], sort=False)
    destock_date                = i_reseeding['destock_date']
    duration                    = i_reseeding['grazing_date'] - destock_date
    periods_destocked           = fun.range_allocation(feed_period_dates, feed_period_name, destock_date, duration)     # proportion of each period that is not being grazed because destocked for reseeding
    phase_area_rf                  = 1 - fun.create_array_from_dfs(phase_area_df['resown'],periods_destocked['allocation'])
    p_phase_area_rf        = phase_area_rf   # parameters for rotation phase variable: area of pasture in each period (is 0 for resown phases during periods that resown pasture is not grazed )

    ## calculate the green feed lost when pasture is destocked. Spread between periods based on date destocked
    ungrazed_foo        = i_reseeding['ungrazed_at_destocking']
    period, proportion  = fun.period_proportion(feed_period_dates, feed_period_name, destock_date)  # which feed period does destocking occur & the proportion that destocking occurs during the period.
    update_reseeding_foo(period, 1-proportion, -ungrazed_foo)                                       # call function to remove the FOO lost for the periods. Assumed that all feed lost is green

    ## calculate the green & dry feed available when pasture first grazed after reseeding. Spread between periods based on date grazed
    reseed_foo          = i_lmu['reseeding_foo_scalar'] * sen.sam_pgr_l * i_reseeding['grazing_foo'] # FOO at the first grazing for each lmu (kg/ha)
    period, proportion  = fun.period_proportion(feed_period_dates, feed_period_name, i_reseeding['grazing_date'])       # which feed period does grazing occur
    update_reseeding_foo(period, 1-proportion, reseed_foo.values,
                         propn_grn=i_grn_propn_reseeding,dmd_dry=i_lmu['dmd_dry'].values)                            # call function to update green & dry feed in the periods.
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
def calc_foo_profile(germination, sam_pgr):
    '''
    Calculate the FOO level at the start of each feed period from the germination & sam on PGR provided

    Parameters
    ----------
    germination - An array[lmu, feed_period]                    : kg of green feed germinating in the period.
    sam_pgr     - An array[lmu, feed_period]                    : SA multiplier for pgr.

    Returns
    -------
    An array[lmu, feed_period]: foo at the start of the period.
    '''
    ## reshape the inputs passed and set some initial variables that are required
    array_shape     = germination.shape
    foo_start       = np.zeros(array_shape, dtype=np.float64)
    foo_end         = np.zeros(array_shape, dtype=np.float64)
    ## loop through the feed periods and calculate the foo at the start of each period
    pgr_daily=np.zeros(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array
    foo_end[:,-1] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    for f in range(n_feed_periods):
        foo_start[:,f]      = germination[:,f] + foo_end[:,f-1]
        ## alternative approach (a1)
        ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
        for l in [*range(n_lmu)]: #loop through lmu
            idx             = np.searchsorted(i_fxg_foo_lfo[l,f,:],foo_start[l,f], side='left')   # find where foo_starts fits into the input data
            pgr_daily[l]    = sam_pgr[l,f] * (  c_fxg_a_lfo[l,f,idx]
                                              + c_fxg_b_lfo[l,f,idx] * foo_start[l,f])
        foo_end[:,f]        = (foo_start[:,f]            * (1-grn_senesce_startfoo_f[f])
                               + pgr_daily * length_f[f] * (1-grn_senesce_pgrcons_f[f])) \
                             *(1-i_grn_senesce_eos_f[f])
    return foo_start

## the following method generates the PGR & FOO parameters for the growth variables. Stored in a numpy array(lmu, feed period, FOO level, grazing intensity)
## def green_consumption:

def green_and_dry():
    ''' Populates the parameter arrays for the pasture growth, consumption and senescence of green feed

    Returns:
    -------
    The parameters in the existing variables.
    '''
    ## set global on all variables required outside this function
    global p_foo_start_grnha_lfo
    global p_foo_end_grnha_lfog
    global p_me_cons_grnha_lfoge
    global p_volume_grnha_lfoge
    global p_senesce2h_grnha_lfog
    global p_senesce2l_grnha_lfog
    global p_senesce_grnha_lfogd
    global p_dry_mecons_t_fde
    global p_dry_volume_t_fd
    global p_dry_removal_t_fd
    global p_dry_transfer_t_fd

    ## initialise numpy arrays used in this method
    grn_dmd_selectivity_fog = np.zeros((       n_feed_periods, n_foo_levels, n_grazing_int),               dtype=np.float64)
    senesce_propn_lfogd     = np.zeros((n_lmu, n_feed_periods, n_foo_levels, n_grazing_int, n_dry_groups), dtype=np.float64)  #

    ## set sensitivity variables used
    sam_pgr_lf                  = np.asfarray(  sen.sam_pgr
                                              * sen.sam_pgr_l.reshape(-1,1)
                                              * sen.sam_pgr_f.reshape(1,-1))
    ## calculate the maximum foo achievable for each lmu & feed period (ungrazed pasture that germinates at the maximum level on that lmu)
    germination_pass_lf             = np.max(p_germination_rlf, axis=0)                                    #use p_germination because it includes any sensitivity that is carried out
    foo_start_ungrazed_lf           = calc_foo_profile(germination_pass_lf, sam_pgr_lf)# ^ passing the consumption value in a numpy array in an attempt to get the function @jit compatible
    max_foo_lf                      = np.maximum(i_fxg_foo_lfo[:,:,-2], foo_start_ungrazed_lf)                  #maximum of ungrazed foo and foo from the medium foo level
    p_foo_start_grnha_lfo           = i_fxg_foo_lfo                                                      # parameters for the growth/grazing activities: initial FOO
    p_foo_start_grnha_lfo[:,:,-1]   = np.maximum.accumulate(max_foo_lf,axis=1)                                #maximum accumulated along the feed periods axis, i.e. max to date
    p_foo_start_grnha_lfo           = np.maximum(p_foo_start_grnha_lfo
                                                 ,          i.base_f.reshape(-1,1))         # to ensure that final foo can not be below 0
    ## calculate green feed parameters
    pgr_grnday_lfo         = np.maximum(0.01, i_fxg_pgr_lfo                  # use maximum to ensure that the pgr is non zero (because foo_days requires dividing by pgr)
                                             *  sam_pgr_lf[...,np.newaxis])
    pgr_grnha_lfog         =       pgr_grnday_lfo[...,np.newaxis]     \
                            *          length_f.reshape(-1,1,1)       \
                            * i_pgr_gi_scalar_fg[:,np.newaxis,:]
    foo_ungrazed_grn_ha_lfo = p_foo_start_grnha_lfo         *(1-grn_senesce_startfoo_f.reshape(-1,1))   \
                             +        pgr_grnha_lfog[...,0] *(1- grn_senesce_pgrcons_f.reshape(-1,1))
    foo_endprior_grnha_lfog=    foo_ungrazed_grn_ha_lfo[...,np.newaxis] -   \
                            -  (foo_ungrazed_grn_ha_lfo[...,np.newaxis]
                                -            i_base_f.reshape(-1,1,1))      \
                            *       i_foo_end_propn_g
    p_foo_end_grnha_lfog   =    foo_endprior_grnha_lfog                   \
                            * (1-i_grn_senesce_eos_f.reshape(-1,1,1))
    removal_grnha_lfog     =np.maximum(0, p_foo_start_grnha_lfo[...,np.newaxis] *(1-grn_senesce_startfoo_f.reshape(-1,1,1))
                                         +        pgr_grnha_lfog                *(1- grn_senesce_pgrcons_f.reshape(-1,1,1))
                                         -  foo_prior_grnha_lfog)          \
                            / (1-grn_senesce_pgrcons_f.reshape(-1,1,1))
    cons_grnha_t_lfog      =      removal_grnha_lfog   \
                            /(1+i_grn_trampling_f.reshape(-1,1,1))
    senesce_total_grnha_lfog  = p_foo_start_grnha_lfo[...,np.newaxis]      \
                               +        pgr_grnha_lfog
                               -    removal_grnha_lfog
                               -  p_foo_end_grnha_lfog
    ## to calculate foo_days requires calculating number of days in current period and adding days from the previous period (if required)
    min=-1; max = 0         #Clip between -1 and 0
    if (p_foo_start_grnha_lfo > p_foo_start_grnha_lfo[:,:,1,np.newaxis]):
        min += 1; max +=1   #Clip between 0 and 1
    propn_period_lfo           = (  p_foo_start_grnha_lfo
                                  - p_foo_start_grnha_lfo[ : , : ,1:2])    \
                                /           pgr_grnha_lfog[: , : , : , 0]
    propn_periodprev_lfo       = (  p_foo_start_grnha_lfo [:,1:  , : ]
                                  - p_foo_start_grnha_lfo [:,1:  ,1:2]
                                  -         pgr_grnha_lfog[:,1:  ,: , 0])    \
                                /           pgr_grnha_lfog[:, :-1,: , 0]     # pgr from the previous period
    foo_days_grnha_lfo         = np.clip(    propn_period_lfo,min,max)             \
                                *                  length_f.reshape(-1,1)
    foo_days_grnha_lfo[:,1:,:]+= np.clip(propn_periodprev_lfo,min,max) \
                                *                  length_f[:-1].reshape(-1,1)      # length from previous period
    grn_dmd_swardscalar_lfo              = (1-i_grn_dmd_declinefoo_f.reshape(-1,1))**foo_days_grnha_lfo    # multiplier on digestibility of the sward due to level of FOO (associated with destocking)
    grn_dmd_range = (  i_grn_dmd_range_f
                 *sen.sam_grn_dmd_range_f).reshape(-1,1)
    grn_dmd_selectivity_fog[:,:,1]        = -0.5 * grn_dmd_range                            # addition to digestibility associated with diet selection (level of grazing)
    grn_dmd_selectivity_fog[:,:,2]        = 0
    grn_dmd_selectivity_fog[:,:,3]        = +0.5 * grn_dmd_range
    dmd_grnha_lfog          =    i_grn_dig_lf[...,np.newaxis,np.newaxis]    \
                              *      grn_dmd_swardscalar_lfo[...,np.newaxis]              \
                              +  grn_dmd_selectivity_fog
    grn_md_grnha_lfog       = fdb.dmd_to_md(dmd_grnha_lfog)
    p_me_cons_grnha_lfoge   =  cons_grnha_t_lfog               \
                                  * grn_md_grnha_lfog               \
                                  - np.maximum(0
                                               ,(            grn_md_grnha_lfog[...,np.newaxis]
                                                 -  i_me_maintenance_fe[:,np.newaxis,np.newaxis,:])
                                               *(1-i_me_eff_gainlose_f.reshape(-1,1,1,1)))           # parameters for the growth/grazing activities: Total ME of feed consumed from the hectare
    foo_ave_grnha_lfog      = (p_foo_start_grnha_lfo[...,np.newaxis]
                               + p_foo_end_grnha_lfog)/2
    grn_ri_availability_lfog= fdb.ri_availability(foo_ave_grnha_lfog,i_ri_foo)
    grn_ri_quality_lfog     = fdb.ri_quality(dmd_grnha_lfog, i_legume)
    grn_ri_lfog             = grn_ri_quality_lfog * grn_ri_availability_lfog
    grn_ri_lfog[grn_ri_lfog<0.05]     = 0.05 #set the minimum RI to 0.05
    p_volume_grnha_lfoge   = p_me_cons_grnha_lfoge / grn_ri_lfog[...,np.newaxis]   # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare

    ## transfer: decline in DM for dry feed (same for high & low pools)
    dry_decay_daily_f                         = [i_dry_decay] * n_feed_periods
    dry_decay_daily_f[0:i_end_of_gs-1]   = [1] * (i_end_of_gs-1)
    dry_decay_period                        = [1 - (1 - decay)**length_f[f] for f,decay in enumerate(dry_decay_daily_f)]
    p_dry_transfer_t_fd[...]           = 1000 * (1-np.c_[dry_decay_period])  # parameters for the dry feed transfer activities: quantity transferred
    ## consumption: quality & FOO of the feed consumed
    dry_dmd_ave          = i_feed_period['dry_dmd_average']
    dry_dmd_range        = i_feed_period['dry_dmd_range']
    dry_dmd_high     = dry_dmd_ave+dry_dmd_range/2
    dry_dmd_low      = dry_dmd_ave-dry_dmd_range/2
    dry_dmd_input    = np.c_[dry_dmd_high, dry_dmd_low]      # create a numpy array that arranges the 2 arguments as columns
    dry_dmd          = np.max(dry_dmd_input,axis=0) - (np.max(dry_dmd_input,axis=0) - dry_dmd_input) * sen.sam_dmd_decline_dry  # do sensitivity adjustment for dry_dmd_input based on increasing the reduction in dmd from the maximum (starting value)

    dry_foo_high     = i_feed_period['dry_foo_high']
    dry_foo_low      = dry_foo_high / 2                      # assuming half the foo is high quality and the remainder is low quality
    dry_foo_input    = np.c_[dry_foo_high, dry_foo_low]      # create a numpy array that arranges the 2 arguments as columns
    dry_foo          = dry_foo_input                         # do sensitivity adjustment for dry_foo_input. Currently not implemented
    ## ME consumed per tonne of dry feed consumed
    dry_md                   = fdb.dmd_to_md(dry_dmd)
    p_dry_mecons_t_fde  = dry_md * 1000                # parameters for the dry feed grazing activities: Total ME of the tonne consumed
    ## Volume of feed consumed per tonne
    dry_ri_availability     = fdb.ri_availability(dry_foo,i_ri_foo)
    dry_ri_quality          = fdb.ri_quality(dry_dmd, i_legume)
    dry_ri                  = dry_ri_quality * dry_ri_availability
    dry_ri[dry_ri<0.05]     = 0.05 #set the minimum RI to 0.05
    p_dry_volume_t_fd  = 1000 / dry_ri                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed
    ## Removal of dry feed
    p_dry_removal_t_fd[...]  = 1000 * (1 + i_dry_trampling_f.reshape(-1,1))   # parameters for the dry feed grazing activities: Total DM removal from the tonne consumed (includes trampling)
    ## Senescence from green to dry
    senesce_tota_grnhal_lfog  = senesce_grnha_lfog + senesce_eos_grnha_lfog
    grn_dmd_senesce_lfog = dmd_grnha_lfog       \
                          - i_feed_period['grn_dmd_senesce_redn'].to_numpy().reshape(-1,1,1)
    senesce2h_propn_lfog        = ( grn_dmd_senesce_lfog - dry_dmd_low)       \
                                 /( dry_dmd_high         - dry_dmd_low)
    senesce_propn_lfogd[...,1] = ( grn_dmd_senesce_lfog - dry_dmd_low)       \
                                 /( dry_dmd_high         - dry_dmd_low)             # senescence to high pool
    senesce_propn_lfogd[...,0] = 1- senesce_propn_lfogd[...,1]                      # senescence to low pool
    p_senesce2h_grnha_lfog = senesce_total_grnha_lfog *    senesce2h_propn_lfog     # parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    p_senesce2l_grnha_lfog = senesce_total_grnha_lfog * (1-senesce2h_propn_lfog)    # parameters for the growth/grazing activities: quantity of green that senesces to the low pool
    p_senesce_grnha_lfogd  = senesce_total_grnha_lfog[np.newaxis]
                            * senesce_propn_lfogd                                   # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool


def poc_con():
    '''
    Returns
    -------
    Dict for pyomo.
        The amount of pasture consumption that can occur on crop paddocks each day before seeding
        - this is adjusted for lmu and feed period
    '''
    df_poc_con = i_poc_intake_daily
    return df_poc_con.stack().to_dict()

def poc_md():
    '''
    Returns
    -------
    Dict for pyomo.
        The quality of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    md=list(map(fdb.dmd_to_md,  i_feed_period['poc_dmd'])) #could use list comp but thought it was a good place to practise map
    poc_md = dict(enumerate(md))
    return poc_md

def poc_vol():
    '''
    Returns
    -------
    Dict for pyomo.
        The relitive intake of pasture on crop paddocks each day before seeding
        - this is adjusted for feed period
    '''
    # ri_qual = np.asarray([fdb.ri_quality(dmd, i_legume) for dmd in i_feed_period['poc_dmd']])       #could use map ie list(map(fdb.ri_quality, md, repeat(annual.legume))) (repeat is imported from itertools)
    # ri_quan = np.asarray([fdb.ri_availability(foo, i_ri_foo) for foo in i_feed_period['poc_foo']])
    dmd_f = i_feed_period['poc_dmd'].to_numpy
    foo_f = i_feed_period['poc_foo'].to_numpy
    ri_qual = fdb.ri_quality(dmd_f, i_legume)       # passing a numpy array
    ri_quan = fdb.ri_availability(foo_f, i_ri_foo)
    poc_vol = dict(enumerate(1/(ri_qual*ri_quan)))
    return poc_vol





















