# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:46:24 2019
module: pasture input module - contains all input data likely to vary for different regions or farms

extra info: - this could eventually interact with a user interface
            - interacts with kv's

key: green section title is major title
     '#' around a title is a minor section title
     std '#' comment about a given line of code

Version Control:
Version     Date        Person  Change
   1.1      10Dec19     John    read_inputs_from_excel: Added mapping the Excel inputs to the python variables
   1.2      25Dec19     John    added reseeding pasture methods (tested with the new rotation phases)
                                added reading the rotation phases from Excel (copy of code from crop.py)
   1.3      26Dec19     JMY     calculate_germination: alter the indexing of the rotn dataframe for the merging with the germination information
                                                       changed the feed period in which the germination is added (was 1 now 0) because forgot to allow for python starting at 0
                                                       change the germ_phase_df to convert nan to 0 and change to float type (to get it ready for the reseeding calculations)
   1.4      27Dec19     JMY     changed name to PastureFunctions and moved calling the functions to Pasture
   1.5      28Dec19     JMY     update_reseeding_foo: Completed the dry foo adjustment.
                                changed proportion of green at grazing after reseeding to a single cell rahter than a dataframe by lmu
   1.6      29Dec19     JMY     calculate_reseeding: debugged operation. phase_area was missing a feed period (because feed_periods didn't include the start of teh following year')
                                                     added an axis=1 to align with lmu
                                                     Added sam_pgr to the FOO available at the first grazing
   1.7      31Dec19     JMY     dry_feed: added method

Known problems:
Fixed   Date    ID by   Problem
        25Dec19 John   -The spraytopped and manipulated pasture needs to have FOO reduction related to the chemical manipulation.
                       -Perhaps also germination reduction in the following pasture phase.
                       -calculate_germination method: The df indexes are messed up with the new rotation phase definitions
                        (or maybe the number of years in a phase definition)
                       -rotn_phase_df is read in from Excel in this module and in crop.py This should be sorted so Excel is not read twice
  1.3   26Dec19 JMY    -calculate_germination: The MultiIndex of the rotn dataframe is not aligning with germination dataframe
        26Dec19 JMY    -germ_phases_df: when it is created in the merge the order appears to change even though sort=False is invoked.
                        Then the germination doesnt align with the rotation phases
  1.3   26Dec19 JMY    -update_reseeding_foo: cant update the reseeding_grn_foo because it generates a typecast error.
                        Fixed by removing nan and setting type to float (in calculate_germination method)
  1.4   28Dec19 JMY    -calculate_germination: the calculation of the necessary germination to meet the erosion limit
                        is not currently connected. It is using the actual FOO level as the minimum germination required.
                        Fixed by removing the idea that rotation phases that dont generate enough pasture should be increased to meet the constraint (those phases will be infeasible or subsidised by the other phases)

@author: john

Description of this pasture module: This representation includes at optimisation (ie the folowing options are represented in the variables of the model)
    Growth rate of pasture (PGR) varies with FOO at the start of the period and grazing intensity during the period
        Grazing intensity operates by altering the average FOO during the period
    The nutritive value of the green feed consumed (as represneted by ME & volume) varies with FOO & grazing intensity.
        Grazing intensity alters the average FOO during the period and the capacity of the animals to select a higher quality diet.
    Selective grazing of dry pasture. 2 dry pasture quality pools are represented and either can be selected for grazing
        Note: There is not a constraint that ensures that the high quality pool is grazed prior to the low quality pool (as there is in the stubble selective grazing)

"""

'''
import functions from other modules
'''
# import datetime as dt
import timeit
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
rotn_phases_df  = pd.Series(uinp.structure['rotations']['rot_phase']).str.split(expand=True)[:-1]   # drop the last row of the df which is the blank line at the bottom of the table


########################
#constants required    #
########################
#define some parameters required to size arrays. Will need to be read in prior to defining the Class
n_foo_levels        = 3                                     # Low, medium & high FOO level in the growth/grazing activities
n_grazing_int       = 4                                     # 0, med & high grazing intensity in the growth/grazing activities
n_feed_pools        = 4                                     # number of feed pools (by quality groups)
n_dry_groups        = 2                                     # Low & high quality groups for dry feed
n_rotn_phases       = len(rotn_phases_df.index)
n_lmu               = len(pinp.general['lmu_area'])
n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1
np_period_length_f  = np.array(pinp.feed_inputs['feed_periods'].loc[:n_feed_periods-1,'length']) # converted to np. to get @jit working

########################
#Pasture Class         #
########################
'''
Define the pasture class that contains the input data and the precalcs that generate the arrays of output data that will populate pyomo
Types are set at initialisation to catch input errors at the time rather than wait till it causes an error
The class expects the following variables to already be defined
    n_feed_periods: int (number of feed periods)
    n_rotn_phases: int (number of rotation phases)
    rotn_phases_df: dataframe of landuse sequences (rotation phase names)
    n_lmus: int (number of lmus)

'''
class PastDetailed:
    ''' pasture in the detailed format: FOO x grazing intensity '''
    def __init__(self,landuse, landuse_set,filename):
        # define the vessels that will store the input data
        self.landuse:str            = landuse                                                       # the landuse name of this pasture to locate it in the rotation phases
        self.landuse_set:set()      = landuse_set                                                   # the set of landuses for this pasture to locate it in the rotation phases
        self.inputfile:str          = filename                                                      # filename of an excel file in a defined layout that contains the input data for this pasture type (Must be in directory with python)
        self.reseeding_data         = dict()                                                        # to create a dict to store data about reseeding of the pasture. Populated when data read in from Excel
        self.lmu_data               = pd.DataFrame(index = pinp.general['lmu_area'].index)     # a dataframe to store data relevant to lmus
        self.feed_period_data       = pd.DataFrame(index = range(n_feed_periods))                   # a dataframe to store data relevant to feed periods
        self.germ_phase_data        = pd.DataFrame()                                                # create a dataframe to store data about the reseeding phases. Populated when read in from Excel
        self.phase_desc:str         = pd.DataFrame()                                                # create a dataframe to hold the rotation phases
        self.base_data_f            = np.zeros((                          n_feed_periods              ), dtype=np.float64)  #Lowest level that pasture can be consumed in each period
        self.grn_dmd_decline_foo_f  = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # decline in digestibility of green feed if pasture is not grazed (and foo increases)
        self.grn_dmd_range_f        = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # range in digestibility within the sward for green feed
        self.grn_senesce_eos_f      = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # proportion of green feed that senesces each period (due to old leaf drop)
        self.fxg_foo_lfo            = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of FOO level       for the FOO/growth/grazing variables.
        self.fxg_pgr_lfo            = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of PGR level       for the FOO/growth/grazing variables.
        self.fxg_a_lfo              = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient a   for the FOO/growth/grazing variables. PGR = a + b FOO
        self.fxg_b_lfo              = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient b   for the FOO/growth/grazing variables. PGR = a + b FOO
        # self.fxg_ai_lfo             = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient a for the FOO/growth/grazing variables. PGR = a + b FOO
        # self.fxg_bi_lfo             = np.zeros((                   n_lmu, n_feed_periods, n_foo_levels), dtype=np.float64)  # numpy array of coefficient b for the FOO/growth/grazing variables. PGR = a + b FOO
        self.grn_dig_lf             = np.zeros((                   n_lmu, n_feed_periods              ), dtype=np.float64)  # numpy array of inputs for green pasture digestibility on each LMU.
        self.grn_trampling_f        = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # ^ should this include the ,1 or just do newaxis when it is used numpy array of inputs for green pasture trampling in each feed period.
        self.dry_trampling_f        = np.zeros((                          n_feed_periods              ), dtype=np.float64)  # ^ should this include the ,1 or just do newaxis when it is used numpy array of inputs for dry pasture trampling   in each feed period.
        # define the numpy arrays that will be the output from the pre-calcs for pyomo
        self.p_germination          = np.zeros((n_rotn_phases, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: germination (kg/ha)
        self.p_reseeding_grn_foo    = np.zeros((n_rotn_phases, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: feed lost and gained during destocking and then grazing of resown pasture (kg/ha)
        self.p_reseeding_dryh_foo   = np.zeros((n_rotn_phases, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: high quality dry feed gained from grazing of resown pasture (kg/ha)
        self.p_reseeding_dryl_foo   = np.zeros((n_rotn_phases, n_lmu, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: low quality dry feed gained from grazing of resown pasture (kg/ha)
        self.p_phase_area_rf        = np.zeros((n_rotn_phases,     1, n_feed_periods), dtype=np.float64)    # parameters for rotation phase variable: area of pasture in each period (is 0 for resown phases during periods that resown pasture is not grazed )

        self.i_cons_propn_g         = np.zeros((                                                   n_grazing_int              ), dtype=np.float64)  # numpy array of proportion of available feed consumed for each grazing intensity level.
        self.i_pgr_gi_scalar_fg     = np.zeros((                      n_feed_periods,              n_grazing_int              ), dtype=np.float64)  # numpy array of pgr scalar for foo level.
        self.p_grn_ha_foo_start_lfo = np.zeros((               n_lmu, n_feed_periods,n_foo_levels                             ), dtype=np.float64)   # parameters for the growth/grazing activities: initial FOO
        self.p_grn_ha_foo_end_lfog  = np.zeros((               n_lmu, n_feed_periods,n_foo_levels, n_grazing_int              ), dtype=np.float64)   # parameters for the growth/grazing activities: final FOO
        self.p_grn_ha_me_cons_lfog  = np.zeros((               n_lmu, n_feed_periods,n_foo_levels, n_grazing_int, n_feed_pools), dtype=np.float64)   # parameters for the growth/grazing activities: Total ME of feed consumed from the hectare
        self.p_grn_ha_volume_lfog   = np.zeros((               n_lmu, n_feed_periods,n_foo_levels, n_grazing_int              ), dtype=np.float64)   # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
        self.p_grn_ha_senesce_to_h  = np.zeros((               n_lmu, n_feed_periods,n_foo_levels, n_grazing_int              ), dtype=np.float64)   # parameters for the growth/grazing activities: quantity of green that senesces to the high pool
        self.p_grn_ha_senesce_to_l  = np.zeros((               n_lmu, n_feed_periods,n_foo_levels, n_grazing_int              ), dtype=np.float64)   # parameters for the growth/grazing activities: quantity of green that senesces to the low pool

        self.p_dry_t_me_cons        = np.zeros((                      n_feed_periods,n_dry_groups,                n_feed_pools), dtype=np.float64)   # parameters for the dry feed grazing activities: Total ME of the tonne consumed
        self.p_dry_t_volume         = np.zeros((                      n_feed_periods,n_dry_groups,                            ), dtype=np.float64)   # parameters for the dry feed grazing activities: Total volume of the tonne consumed
        self.p_dry_t_removal        = np.zeros((                      n_feed_periods,n_dry_groups,                            ), dtype=np.float64)   # parameters for the dry feed grazing activities: Total DM removal from the tonne consumed (includes trampling)
        self.p_dry_t_transfer       = np.zeros((                      n_feed_periods,n_dry_groups,                            ), dtype=np.float64)   # parameters for the dry feed transfer activities: quantity transferred
        # self.p_

########################
#methods               #
########################
        
    def read_inputs_from_excel(self):
        '''Read inputs for the pasture class from an excel file and store in the object'''
        exceldata = fun.xl_all_named_ranges(self.inputfile, self.landuse)           # read all range names from the Excel file from the specified sheet
        self.t_exceldata                                = exceldata                 # keep a copy of all the Excel data until this is operating
        # map the Excel data into the python variables
        # self.included :bool                             = exceldata['SA.Past_inc']  # this pasture is included
        self.i_germination_std: float                   = exceldata['GermStd']      # standard germination level for the standard soil type in a continuous pasture rotation
        self.i_ri_foo: int                              = exceldata['RIFOO']        # to reduce foo to allow for differences in measurement methods for FOO. The target is to convert the measurement to the system developing the intake equations
        # self.trampling: float                           = exceldata['Trampling']    # removed in lieu of green & dry array by period. the amount of pasture trampled per unit of pasture consumed
        self.i_end_of_gs: int                           = exceldata['EndGS']        # the period number when the pasture senesces
        self.i_dry_decay: float                         = exceldata['PastDecay']    # decay rate of dry pasture during the dry feed phase (Note: 100% during growing season)
        # self.poc_days_of_grazing: int                   = exceldata['POCDays']      # number of days after the pasture break that (moist) seeding can begin
        self.poc_intake_daily                           = exceldata['POCCons']      # intake per day of pasture on crop paddocks prior to seeding
        self.legume: float                              = exceldata['Legume']       # proportion of legume in the sward
        self.reseeding_propn_grn                        = exceldata['FaG_PropnGrn'] # Proportion of the FOO available at the first grazing that is green
        # self.                                           = exceldata['']
        # self.feed_period_data['included']               = exceldata['SA.PastGP_inc']# growth of this pasture in this period is included
        self.feed_period_data['grn_dmd_redn_senesce']   = exceldata['DigRednSenesce']   # reduction in digestibility of green feed when it senesces
        self.feed_period_data['dry_dmd_average']        = exceldata['DigDryAve']    # average digestibility of dry feed. Note the reduction in this value determines the reduction in quality of ungrazed dry feed in each of the dry feed quality pools. The average digestibility of the dry feed sward will depend on selective grazing which is an optimised variable.
        self.feed_period_data['dry_dmd_range']          = exceldata['DigDryRange']  # range in digestibility of dry feed if it is not grazed
        self.feed_period_data['dry_foo_high']           = exceldata['FOODryH']      # expected foo for the dry pasture in the high quality pool
        self.feed_period_data['grn_crude_protein']      = exceldata['CPGrn']        # crude protein content of green feed
        self.feed_period_data['dry_crude_protein']      = exceldata['CPDry']        # crude protein content of dry feed
        self.feed_period_data['poc_dmd']                = exceldata['DigPOC']       # digestibility of pasture consumed on crop paddocks
        self.feed_period_data['poc_foo']                = exceldata['FOOPOC']       # foo of pasture consumed on crop paddocks
        # self.feed_period_data['']                       = exceldata['']
        # self.lmu_data['included']                       = exceldata['SA.PastL_inc']     # this pasture is included on this lmu
        self.lmu_data['germ_scalar']                    = exceldata['GermScalarLMU']    # scale the germination levels for each lmu
        self.lmu_data['reseeding_foo_scalar']           = exceldata['FaG_LMU']          # scalar for FOO at the first grazing for the lmus
        self.lmu_data['dmd_dry']                        = exceldata['FaG_digDry']       # Average digestibility of any dry FOO at the first grazing (if there is any)
        # self.lmu_data['']                               = exceldata['']
        self.lmu_conservation_data                      = exceldata['ErosionLimit']     # minimum foo prior at end of each period to reduce risk of wind & water erosion

        self.reseeding_data['seed_date']                = exceldata['Date_Seeding']         # date of seeding this pasture type (will be read in from inputs)
        self.reseeding_data['destock_date']             = exceldata['Date_Destocking']      # date of seeding this pasture type (will be read in from inputs)
        self.reseeding_data['ungrazed_at_destocking']   = exceldata['FOOatSeeding']         # kg of FOO that was not grazed prior to seeding occurring (if spring sown)
        self.reseeding_data['grazing_date']             = exceldata['Date_ResownGrazing']   # date of first grazing of reseeded pasture (will be read in from inputs)
        self.reseeding_data['grazing_foo']              = exceldata['FOOatGrazing']         # FOO at time of first grazing

        self.germ_phase_data                            = exceldata['GermPhases']       #DataFrame with germ scalar and resown

        ## inputs read into numpy arrays
        self.grn_trampling_f.fill                        (exceldata['Trampling'])
        self.dry_trampling_f.fill                        (exceldata['Trampling'])
        grn_senesce_daily_f                 = np.asfarray(exceldata['SenescePropn'])    # proportion of green feed that senesces each period (due to old leaf drop)
        self.grn_senesce_eos_f              = np.asfarray(exceldata['SenesceEOS'])      # ^ alternative is array[...] = exceldata[] without the np.asfarray. It would then use the previous definition of the array
        self.base_data_f                    = np.asfarray(exceldata['BaseLevelInput'])
        self.grn_dmd_decline_foo_f          = np.asfarray(exceldata['DigDeclineFOO'])
        self.grn_dmd_range_f                = np.asfarray(exceldata['DigSpread'])       # range in digestibility within the sward for green feed
        self.i_cons_propn_g                 = np.asfarray(exceldata['FOOGrazePropn'])
        self.i_pgr_gi_scalar_fg=1-self.i_cons_propn_g**2*(1-
                                              np.asfarray(exceldata['PGRScalarH'])).reshape(-1,1)      # Scale PGR =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)

        self.fxg_foo_lfo[:,:,0]                         = exceldata['LowFOO'].to_numpy().T
        self.fxg_foo_lfo[:,:,1]                         = exceldata['MedFOO'].to_numpy().T
        self.fxg_foo_lfo[:,:,-1]                        = 10000 #large number so that the searchsorted doesn't go above
        ### self.fxg_foo_lfo[:,:,-1] is calculated later and is the maximum foo that can be achieved (on that lmu in that period)
        ### it is affected by sa on pgr so it must be calculated during the experiment where sam might be altered.
        self.fxg_pgr_lfo[:,:,0]                         = exceldata['LowPGR'].to_numpy().T
        self.fxg_pgr_lfo[:,:,1]                         = exceldata['MedPGR'].to_numpy().T
        self.fxg_pgr_lfo[:,:,-1]                        = exceldata['MedPGR'].to_numpy().T  #PGR for high (last entry) is the same as PGR for medium
        self.grn_dig_lf                                 = exceldata['DigGrn'].to_numpy().T
        ## Some one time data manipulation for the inputs just read
        self.germ_phase_data.index = [*range(len(self.germ_phase_data.index))]              # replace index read from Excel with numbers to match later merging
        self.germ_phase_data.columns.values[range(phase_len)] = [*range(phase_len)]         # replace the landuse columns read from Excel with numbers to match later merging
        self.fxg_b_lfo[:,:,0] =   self.fxg_pgr_lfo[:,:,0]     \
                               /  self.fxg_foo_lfo[:,:,0]
        self.fxg_b_lfo[:,:,1] = ( self.fxg_pgr_lfo[:,:,1]
                                 -self.fxg_pgr_lfo[:,:,0])    \
                               /( self.fxg_foo_lfo[:,:,1]
                                 -self.fxg_foo_lfo[:,:,0])
        self.fxg_b_lfo[:,:,2] =  0

        self.fxg_a_lfo[:,:,0] =  0
        self.fxg_a_lfo[:,:,1] = self.fxg_pgr_lfo[:,:,0] - self.fxg_b_lfo[:,:,1] * self.fxg_foo_lfo[:,:,0]
        self.fxg_a_lfo[:,:,2] = self.fxg_pgr_lfo[:,:,1] # because slope = 0

        self.grn_senesce_f          = 1 - ((1 - grn_senesce_daily_f) ** np_period_length_f) # senescence for the period, different formula than excel

    def calculate_germination(self):
        ''' create an array called p_germination(r,l) being the parameters to be passed to pyomo.

        requires rotn_phases_df as a global variable
        ^ currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed
        '''
        # map the germination phases to the rotation phases
        germ_phases_df              = pd.merge(rotn_phases_df,self.germ_phase_data,
                                         how='left', left_on=[*range(phase_len)], right_on=[*range(phase_len)], sort=False)
        self.germ_phases_df         = germ_phases_df.set_index([*range(phase_len)])                                     # retain the dataframe for use in reseeding method.
        rp                          = np.array(germ_phases_df['germ_scalar'].values).reshape(-1,1)                      # extract the germ_scalar from the dataframe and transpose (reshape to a column vector)
        lmu                         = np.array(self.lmu_data['germ_scalar'].values) * sen.sam_germ_l  # lmu germination scalar x SA on lmu scalar
        germination                 = self.i_germination_std * np.multiply(rp,lmu) * sen.sam_germ                             # create an array rot phase x lmu
        germination[np.isnan(germination)] = 0.0
        self.p_germination[:,:,0]   = germination                                                                       # set germination in first period to germination

        ##possible idea to create the dataframe with the key for the dict
        # m=np.array(['lmu1,','lmu2,','lmu3,'], dtype=np.object)

        # n=np.array(['fp1','fpd2','fpd3'], dtype=np.object)

        # names = m.reshape(-1,1) + n.reshape(1,-1)

       #  np.ravel(names)
       #  array(['lmu1,fp1', 'lmu1,fpd2', 'lmu1,fpd3', 'lmu2,fp1', 'lmu2,fpd2', 'lmu2,fpd3', 'lmu3,fp1', 'lmu3,fpd2', 'lmu3,fpd3'], dtype=object)

    # the following method generates the FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    # intermediate calculations are not stored, however, if they were stored the 'key variables' could change the values of the intermediate calcs which could then be fed into the parameter calculations (in a separate method)
    # the above would provide more options for KVs and provide another step that may not need to be recalculated
    def calculate_reseeding(self):
        '''Generates the green & dry FOO that is lost and gained from reseeding pasture

        Results are stored in p_reseeding'''
        #reset all initial values to 0              ^ required even if deleted in the other functions
        self.p_reseeding_grn_foo[...]  = 0          # array has been initialised, reset all values to 0
        self.p_reseeding_dryh_foo[...] = 0
        self.p_reseeding_dryl_foo[...] = 0
        self.p_phase_area_rf[...]      = 0
        def update_reseeding_foo(period, proportion, total, propn_grn=1, dmd_dry=0):
            ''' Populate p_reseeding_grn_foo, p_reseeding_dryh_foo, p_reseeding_dryl_foo with values for destocking & subsequent grazing in the relevant feed periods

            period & proportion refer to the first period affected by the destocking or subsequent grazing
            total is the foo to be spread between the period and the subsequent period, can be a single value or a list. If list it must be by lmu
            propn_grn is a single value
            dmd_dry is a list of values by lmu

            the adjustments are spread between periods to allow for the pasture growth that can occur from the green feed
            and the amount of grazing available if the feed is dry
            '''
            foo_change = np.array(self.germ_phases_df['resown'].values.reshape(-1,1).astype(float) * total)                             # create an array (phase x lmu) that is the value to be added for any phase that is resown
            foo_change[np.isnan(foo_change)] = 0

            next_period = (period+1) % n_feed_periods
            self.p_reseeding_grn_foo[:,:,period]        += foo_change *    proportion  * propn_grn      # add the amount of green for the first period
            self.p_reseeding_grn_foo[:,:,next_period]   += foo_change * (1-proportion) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
            if propn_grn < 1:
                ave_dmd     = self.feed_period_data['dry_dmd_average'][period]
                range_dmd   = self.feed_period_data['dry_dmd_range'][period]
                high_dmd    = ave_dmd+range_dmd/2
                low_dmd     = ave_dmd-range_dmd/2
                propn_high  = (dmd_dry-low_dmd)/(high_dmd-low_dmd)
                propn_low   = (high_dmd-dmd_dry)/(high_dmd-low_dmd)
                self.p_reseeding_dryh_foo[:,:,period]       += foo_change *    proportion  * (1-propn_grn) * propn_high        # add the amount of high quality dry for the first period
                self.p_reseeding_dryh_foo[:,:,next_period]  += foo_change * (1-proportion) * (1-propn_grn) * propn_high        # add the remainder to the next period (wrapped if past the 10th period)
                self.p_reseeding_dryl_foo[:,:,period]       += foo_change *    proportion  * (1-propn_grn) * propn_low         # add the amount of high quality dry for the first period
                self.p_reseeding_dryl_foo[:,:,next_period]  += foo_change * (1-proportion) * (1-propn_grn) * propn_low         # add the remainder to the next period (wrapped if past the 10th period)

        # retain the (labour) period during which this pasture is reseeded. For machinery expenditure
        period_dates    = per.p_dates_df()['date']
        period_name     = per.p_dates_df().index
        date            = self.reseeding_data['seed_date']
        self.reseeding_data['seeding_period'] = fun.period_allocation(period_dates,period_name,date)

        # set the period definitions to the feed periods
        feed_period_dates   = list(pinp.feed_inputs['feed_periods']['date'])
        feed_period_name    = pinp.feed_inputs['feed_periods'].index

        # calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
        phase_area_df               = pd.merge(rotn_phases_df,self.germ_phase_data,
                                         how='left', left_on=[*range(phase_len)], right_on=[*range(phase_len)], sort=False)
        destock_date                = self.reseeding_data['destock_date']
        duration                    = self.reseeding_data['grazing_date'] - destock_date
        periods_destocked           = fun.range_allocation(feed_period_dates, feed_period_name, destock_date, duration)     # proportion of each period that is not being grazed because destocked for reseeding
        phase_area                  = 1 - fun.create_array_from_dfs(phase_area_df['resown'],periods_destocked['allocation'])
        self.p_phase_area_rf        = phase_area

        # calculate the green feed lost when pasture is destocked. Spread between periods based on date destocked
        ungrazed_foo        = self.reseeding_data['ungrazed_at_destocking']
        period, proportion  = fun.period_proportion(feed_period_dates, feed_period_name, destock_date)  # which feed period does destocking occur & the proportion that destocking occurs during the period.
        update_reseeding_foo(period, 1-proportion, -ungrazed_foo)                                       # call function to remove the FOO lost for the periods. Assumed that all feed lost is green

        # calculate the green & dry feed available when pasture first grazed after reseeding. Spread between periods based on date grazed
        reseed_foo          = self.lmu_data['reseeding_foo_scalar'] * sen.sam_pgr_l * self.reseeding_data['grazing_foo'] # FOO at the first grazing for each lmu (kg/ha)
        period, proportion  = fun.period_proportion(feed_period_dates, feed_period_name, self.reseeding_data['grazing_date'])       # which feed period does grazing occur
        update_reseeding_foo(period, 1-proportion, reseed_foo.values,
                             propn_grn=self.reseeding_propn_grn,dmd_dry=self.lmu_data['dmd_dry'].values)                            # call function to update green & dry feed in the periods.

    # define a function that loops through feed periods to generate the foo profile for a specified germination and consumption
    # examined some options to get rid of @jit error "UnsupportedError: Use of unknown opcode 'BUILD_LIST_UNPACK'"
    # tried changing all inputs to the function to only include numpy arrays but didn't fix issue
    # next step was to
    #   try the bool filter approach to remove the 2nd loop
    #   simplify the function to only be doing the f loop & maybe put it inside the function that calls the loop - not important here because it is not taking much time)
    # @jit("float64[:,:](float64[:,:],float64[:,:])",nopython=True, nogil=True)
    # @jit(nopython=True, nogil=True)
    # @jit()
    def calc_foo_profile(self, germination, consumption, sam_pgr):
        '''
        Calculate the FOO level at the start of each feed period from the germination, consumption & sam on PGR provided

        Parameters
        ----------
        germination - An array[lmu, feed_period]                    : kg of green feed germinating in the period.
        consumption - An array or broadcastable to [lmu,feed_period]: kg of green feed consumed in the period.
        sam_pgr     - An array[lmu, feed_period]                    : SA multiplier for pgr.

        Returns
        -------
        An array[lmu, feed_period]: foo at the start of the period.
        '''
        ## reshape the inputs passed and set some initial variables that are required
        consumption = np.asarray(consumption)                       # check that consumption is an array. To allow a scalar to be based as the value for consumption
        array_shape     = germination.shape
        if consumption.shape != array_shape:
            consumption = np.full(array_shape,consumption) # create an array based on shape of germination and fill with the value from consumption
        foo_start       = np.zeros(array_shape, dtype=np.float64)
        foo_end         = np.zeros(array_shape, dtype=np.float64)
        ## loop through the feed periods and calculate the foo at the start of each period
        pgr_daily=np.arange(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array
        foo_end[:,-1] = 0
        for f in range(n_feed_periods):
            foo_start[:,f]      = germination[:,f] + foo_end[:,f-1]
            ## alternative approach (a1)
            ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
            for l in [*range(n_lmu)]: #loop through lmu
                idx             = np.searchsorted(self.fxg_foo_lfo[l,f,:],foo_start[l,f], side='left')   # find where foo_starts fits into the input data
                pgr_daily[l]    = sam_pgr[l,f] * (  self.fxg_a_lfo[l,f,idx]
                                                  + self.fxg_b_lfo[l,f,idx] * foo_start[l,f])
            senescence          = (foo_start[:,f] + pgr_daily * np_period_length_f[f]/2 - consumption[:,f]/2) * self.grn_senesce_f[f]
            foo_end[:,f]        = (foo_start[:,f] + pgr_daily * np_period_length_f[f] - senescence - consumption[:,f]) \
                                 *(1-self.grn_senesce_eos_f[f])
        return foo_start

    ## the following method generates the PGR & FOO parameters for the growth variables. Stored in a numpy array(lmu, feed period, FOO level, grazing intensity)
    ## def green_consumption:

    def green_feed(self):
        ''' Populates the parameter arrays for the pasture growth, consumption and senescence of green feed

        Returns:
        -------
        The parameters in the existing variables.
        '''
        #initialise numpy arrays used in this method
        diet_dig_a_fog                      = np.zeros((n_feed_periods, n_foo_levels, n_grazing_int),dtype=float)
        #reset all initial values to 0    ^ probably not necessary now that arrays aren't populated with +=
        self.p_grn_ha_foo_start_lfo[...]    = 0
        self.p_grn_ha_foo_end_lfog[...]     = 0
        self.p_grn_ha_me_cons_lfog[...]     = 0
        self.p_grn_ha_volume_lfog[...]      = 0
        self.p_grn_ha_senesce_to_h[...]     = 0
        self.p_grn_ha_senesce_to_l[...]     = 0

        sam_pgr_lf                          = np.asfarray(sen.sam_pgr * sen.sam_pgr_l[:,np.newaxis] * sen.sam_pgr_f)
        self.t_sam_pgr = sam_pgr_lf
        ## calculate the maximum foo achievable for each lmu & feed period (ungrazed pasture that germinates at the maximum level on that lmu)
        germination_to_pass                 = np.max(self.p_germination, axis=0)                                    #use p_germination because it includes any sensitivity that is carried out
        foo_start_ungrazed                  = self.calc_foo_profile(germination_to_pass, np.asarray([0]),sam_pgr_lf)# ^ passing the consumption value in a numpy array in an attempt to get the function @jit compatible
        max_c                               = np.maximum(self.fxg_foo_lfo[:,:,-2], foo_start_ungrazed)                  #maximum of ungrazed foo and foo from the medium foo level
        self.fxg_foo_lfo[:,:,-1]            = np.maximum.accumulate(max_c,axis=1)                                #maximum accumulated along the feed periods axis, i.e. max to date
        self.p_grn_ha_foo_start_lfo         =             self.fxg_foo_lfo                                                          #foo_start only has one
        grn_day_pgr_lfo                     = np.maximum(0.01, self.fxg_pgr_lfo                  # use maximum to ensure that the pgr is non zero
                                                         *          sam_pgr_lf[...,np.newaxis])
        grn_ha_pgr_lfog                     =           grn_day_pgr_lfo[...,np.newaxis]     \
                                             *     np_period_length_f.reshape(-1,1,1)       \
                                             * self.i_pgr_gi_scalar_fg[:,np.newaxis,:]
        self.t_pgr_lfog = grn_ha_pgr_lfog[3,...]
        grn_ha_senesce_lfog                 = (self.p_grn_ha_foo_start_lfo[...,np.newaxis]
                                               +            grn_ha_pgr_lfog / 2) \
                                             *        self.grn_senesce_f.reshape(-1,1,1)
        self.t_senesce_lfog = grn_ha_senesce_lfog[3,...]
        grn_ha_removal_lfog                 =np.maximum(0,      self.i_cons_propn_g                     # removal can't be below 0
                                                        *(self.p_grn_ha_foo_start_lfo[...,np.newaxis]
                                                          +            grn_ha_pgr_lfog
                                                          -        grn_ha_senesce_lfog
                                                          -        self.base_data_f.reshape(-1,1,1)))
        self.t_remove_lfog = grn_ha_removal_lfog[3,...]
        grn_ha_senesce_eos_lfog             =      self.grn_senesce_eos_f.reshape(-1,1,1) \
                                             * (self.p_grn_ha_foo_start_lfo[...,np.newaxis]
                                                +            grn_ha_pgr_lfog
                                                -        grn_ha_senesce_lfog
                                                -        grn_ha_removal_lfog)
        self.t_eos_lfog = grn_ha_senesce_eos_lfog[3,...]
        self.p_grn_ha_foo_end_lfog          = self.p_grn_ha_foo_start_lfo[...,np.newaxis]    \
                                             +             grn_ha_pgr_lfog    \
                                             -         grn_ha_senesce_lfog    \
                                             -         grn_ha_removal_lfog    \
                                             -     grn_ha_senesce_eos_lfog
        self.t_end_lfog = self.p_grn_ha_foo_end_lfog[3,...]
        grn_ha_cons_t_lfog                  =           grn_ha_removal_lfog   \
                                             / (1 + self.grn_trampling_f.reshape(-1,1,1))
        self.t_cons_t_lfog = grn_ha_cons_t_lfog[3,...]
        grn_ha_foo_days_lfo                 = np.clip((  self.p_grn_ha_foo_start_lfo
                                                       - self.p_grn_ha_foo_start_lfo[:,:,1,np.newaxis])   \
                                                       /             grn_day_pgr_lfo,
                                                     -np_period_length_f.reshape(-1,1),
                                                      np_period_length_f.reshape(-1,1))
        self.t_foo_days_lfo = grn_ha_foo_days_lfo   #^delete later
        m_sward_dig_lfo                     = (1-self.grn_dmd_decline_foo_f.reshape(-1,1))**grn_ha_foo_days_lfo    # multiplier on digestibility of the sward due to level of FOO (associated with destocking)
        diet_dig_a_fog[:,:,1]               = -0.5 * self.grn_dmd_range_f.reshape(-1,1)                            # addition to digestibility associated with diet selection (level of grazing)
        diet_dig_a_fog[:,:,2]               = 0
        diet_dig_a_fog[:,:,3]               = +0.5 * self.grn_dmd_range_f.reshape(-1,1)
        self.t_diet_dig = diet_dig_a_fog
        grn_ha_dig_lfog                     = self.grn_dig_lf[...,np.newaxis,np.newaxis] \
                                             * m_sward_dig_lfo[...,np.newaxis]           \
                                             +  diet_dig_a_fog
        self.p_grn_ha_me_cons_lfog          = grn_ha_cons_t_lfog * fdb.dmd_to_md(grn_ha_dig_lfog)

        # print('sam_pgr',sam_pgr_lf)
        # for f in range(10): print('f',grn_ha_pgr_lfog[0,f,0,0],'   ',grn_ha_pgr_lfog[0,f,1,0],'   ',grn_ha_pgr_lfog[0,f,2,2])
        # print('pgr',grn_ha_pgr_lfog)

    def dry_feed(self):
        ''' Populates the parameter arrays associated with dry feed consumption & deferment

        Returns:
        -------
        The parameters in the existing variables.
        '''
        #reset all initial values to 0    ^ probably not necessary now that arrays aren't populated with +=
        self.p_dry_t_me_cons[...]   = 0
        self.p_dry_t_volume[...]    = 0
        self.p_dry_t_removal[...]   = 0
        self.p_dry_t_transfer[...]  = 0

        # transfer: decline in DM for dry feed (same for high & low pools)
        dry_decay_daily                         = [self.i_dry_decay] * n_feed_periods
        dry_decay_daily[0:self.i_end_of_gs-1]   = [1] * (self.i_end_of_gs-1)
        dry_decay_period                        = [1 - (1 - dry_decay_daily[i])**np_period_length_f[i] for i,n in enumerate(dry_decay_daily)]
        self.p_dry_t_transfer[...]              = 1000 * (1-np.c_[dry_decay_period])
        # consumption: quality & FOO of the feed consumed
        ave_dmd          = self.feed_period_data['dry_dmd_average']
        range_dmd        = self.feed_period_data['dry_dmd_range']
        dry_dmd_high     = ave_dmd+range_dmd/2
        dry_dmd_low      = ave_dmd-range_dmd/2
        dry_dmd_input    = np.c_[dry_dmd_high, dry_dmd_low]      # create a numpy array that arranges the 2 arguments as columns
        dry_dmd          = np.max(dry_dmd_input,axis=0) - (np.max(dry_dmd_input,axis=0) - dry_dmd_input) * sen.sam_dmd_decline_dry  # do sensitivity adjustment for dry_dmd_input based on increasing the reduction in dmd from the maximum (starting value)

        dry_foo_high     = self.feed_period_data['dry_foo_high']
        dry_foo_low      = dry_foo_high / 2                      # assuming half the foo is high quality and the remainder is low quality
        dry_foo_input    = np.c_[dry_foo_high, dry_foo_low]      # create a numpy array that arranges the 2 arguments as columns
        dry_foo          = dry_foo_input                         # do sensitivity adjustment for dry_foo_input. Currently not implemented
        # ME consumed per tonne of dry feed consumed
        dry_md                = fdb.dmd_to_md(dry_dmd)
        self.p_dry_t_me_cons  = dry_md * 1000
        # Volume of feed consumed per tonne
        dry_ri_availability = fdb.ri_availability(dry_foo,self.i_ri_foo)
        dry_ri_quality      = fdb.ri_quality(dry_dmd, self.legume)
        dry_ri              = dry_ri_quality * dry_ri_availability
        dry_ri[dry_ri<0.05] = 0.05 #set the minimum RI to 0.05
        self.p_dry_t_volume = 1000 / dry_ri
        # Removal of dry feed
        self.p_dry_t_removal[...]  = 1000 * (1 + self.dry_trampling_f.reshape(-1,1))

# annual = PastDetailed('annual', {'a', 'ar', 'a3', 'a4', 'a5', 's', 'sr', 's3', 's4', 's5', 'm', 'm3', 'm4'},'GSMInputs.xlsx')        # create an instance of the Pasture class and pass the landuse name and the filename for the Excel file that stores the data
# annual.read_inputs_from_excel()                         # read inputs from Excel file and map to the python variables
# annual.calculate_germination()                          # calculate the germination for each rotation phase
# annual.calculate_reseeding()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment


    def poc_con(self):
        '''
        Returns
        -------
        Dict for pyomo.
            The amount of pasture consumption that can occur on crop paddocks each day before seeding
            - this is adjusted for lmu and feed period
        '''
        df_poc_con = self.poc_intake_daily
        return df_poc_con.stack().to_dict()

    def poc_md(self):
        '''
        Returns
        -------
        Dict for pyomo.
            The quality of pasture on crop paddocks each day before seeding
            - this is adjusted for feed period
        '''
        md=list(map(fdb.dmd_to_md,  self.feed_period_data['poc_dmd'])) #could use list comp but thought it was a good place to practise map 
        return dict(enumerate(md))
    def poc_vol(self):
        '''
        Returns
        -------
        Dict for pyomo.
            The relitive intake of pasture on crop paddocks each day before seeding
            - this is adjusted for feed period
        '''
        ri_qual = np.asarray([fdb.ri_quality(dmd, self.legume) for dmd in self.feed_period_data['poc_dmd']])       #could use map ie list(map(fdb.ri_quality, md, repeat(annual.legume))) (repeat is imported from itertools)
        ri_quan = np.asarray([fdb.ri_availability(foo, self.i_ri_foo) for foo in self.feed_period_data['poc_foo']])
        self.poc_vol = dict(enumerate(1/(ri_qual*ri_quan)))
        return self.poc_vol
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
