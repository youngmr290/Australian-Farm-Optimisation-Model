# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young

This is where all sensitivity values must be initialised.

"""

import numpy as np
import PropertyInputs as pinp
import UniversalInputs as uinp 

##create dict - store sa variables in dict so they can easily be changed in the exp loop
sam = dict()
sap = dict()
saa = dict()
sat = dict()
sav = dict()
sar = dict()

######
#SAP #
######

##Global
sap['pi']=0 #global potential intake ^this has not been applied globally to the model yet (currently only in stubble)

######
#SAM #
######
## Annual module sensitivity variables - these need to have the same name for each pasture type
sam['germ','annual']                    = 1.0                                                                       # SA multiplier for germination on all lmus in all periods
sam['germ_l','annual']                  = np.ones((len(pinp.general['lmu_area'])),              dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam['pgr','annual']                     = 1.0                                                                       # SA multiplier for growth on all lmus in all periods
sam['pgr_f','annual']                   = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam['pgr_l','annual']                   = np.ones((len(pinp.general['lmu_area'])),              dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam['dry_dmd_decline','annual']         = 1.0                                                                       # SA multiplier for the decline in digestibility of dry feed
sam['grn_dmd_declinefoo_f','annual']    = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam['grn_dmd_range_f','annual']         = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam['grn_dmd_senesce_f','annual']       = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
# sa_feed_period_inc_t      = True    # growth of this pasture in this period is included
# sa_lmu_inc_t              = True    # this pasture is included on this lmu

##livestock
sam['woolp_mpg'] = 1.0                   # sa multiplier for wool price at std micron
sam['salep_max'] = 1.0                        #max sale price in grid


######
#SAP #
######


######
#SAA #
######


######
#SAT #
######
sat['salep_weight_scalar'] = 0 #Scalar for LW impact across grid 1
sat['salep_score_scalar'] = 0 #Scalar for score impact across the grid


######
#SAR #
######

##sheep
sar['mortalityp'] = 0          #Scale the calculated progeny mortality at birth in the target range 0 to 100% - this is a high level sa it impacts within a calculation not on an input
sar['mortalitye'] = 0          #Scale the calculated dam mortality at birth in the target range 0 to 100% - this is a high level sa it impacts within a calculation not on an input

######
#SAV #
######
##if you initialise an array it must be type object (so that you can assign int/float/bool into the array)
##area
sav['bnd_total_pas_area'] = '-'  #Total pasture area for bound. '-' is default so it will chuck an error if the bound is turned on without a specified area
sav['bnd_pasarea_inc'] = '-'   #SA to turn on the pasture area bound
##Sheep
sav['fec_inc'] = '-'   #SA to store fec report report values
sav['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
sav['eqn_used_g0_q1p7'] = np.full(uinp.sheep['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g1_q1p7'] = np.full(uinp.sheep['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g2_q1p7'] = np.full(uinp.sheep['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g3_q1p7'] = np.full(uinp.sheep['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['TOL_inc']         = np.full(pinp.sheep_inp['i_mask_i'].shape, '-', dtype=object)   # SA value for the inclusion of each TOL
sav['g3_included']         = np.full(pinp.sheep_inp['i_g3_inc'].shape, '-', dtype=object) # SA value for the inclusion of each offspring genotype
sav['scan_og1']         = np.full(pinp.sheep_inp['i_scan_og1'].shape, '-', dtype=object) # SA value for the scanning management option
sav['woolp_mpg_percentile'] = '-'              #sa value for the wool price percentile
sav['woolp_mpg'] = '-'                     # sa value for wool price at std micron
sav['woolp_fdprem_percentile'] = '-'           # sa value for fd premium percentile (premium received by fd compared to std)
sav['woolp_fdprem'] = '-'                     # sa value for fd premium
sav['salep_percentile'] = '-'                     #Value for percentile for all sale grids
sav['salep_max'] = '-'                        #max sale price in grid
sav['nut_mask_dams'] = np.full(pinp.sheep_inp['i_sai_lw_dams_owi'].shape, '-', dtype=object)  #masks the nutrition options available eg high low high - the options selected are available for each starting weight
sav['nut_mask_offs'] = np.full(pinp.sheep_inp['i_sai_lw_offs_swix'].shape, '-', dtype=object)  #masks the nutrition options available eg high low high - the options selected are available for each starting weight