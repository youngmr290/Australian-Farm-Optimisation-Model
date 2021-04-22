# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young

This is where all sensitivity values must be initialised.

"""

import numpy as np
import PropertyInputs as pinp
import UniversalInputs as uinp
import Periods as per

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
sap['pi']=0 #global potential intake (this increases animal intake without altering animal energy profile, to alter the energy profile use ci[1:2,...] todo this has not been applied globally to the model yet (currently only in stubble)

######
#SAM #
######
## Annual module sensitivity variables - these need to have the same name for each pasture type
sam['germ','annual']                    = 1.0                                                          # SA multiplier for germination on all lmus in all periods
sam['germ_l','annual']                  = np.ones((len(pinp.general['lmu_area'])),  dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam['pgr','annual']                     = 1.0                                                          # SA multiplier for growth on all lmus in all periods
sam['pgr_f','annual']                   = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam['pgr_l','annual']                   = np.ones((len(pinp.general['lmu_area'])),  dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam['dry_dmd_decline','annual']         = 1.0                                                          # SA multiplier for the decline in digestibility of dry feed
sam['grn_dmd_declinefoo_f','annual']    = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam['grn_dmd_range_f','annual']         = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam['grn_dmd_senesce_f','annual']       = np.ones((len(per.f_feed_periods()) - 1),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing
# sa_feed_period_inc_t      = True    # growth of this pasture in this period is included
# sa_lmu_inc_t              = True    # this pasture is included on this lmu

##livestock
sam['woolp_mpg'] = 1.0                      # sa multiplier for wool price at std micron
sam['salep_max'] = 1.0                      #max sale price in grid
sam['kg'] = 1.0                             #energy efficiency of adults (zf2==1)
sam['mr'] = 1.0                             #Maintenance requirement of adults (zf2==1)
sam['pi'] = 1.0                             #Potential intake of adults (zf2==1)
sam['LTW_dams'] = 1.0                       #adjust impact of life time wool fleece effects
sam['LTW_offs'] = 1.0                       #adjust impact of life time wool fleece effects

##stock parameters
sam['ci_c2'] = np.ones(uinp.parameters_inp['i_ci_c2'].shape, dtype=np.float64)  #intake params for genotypes
sam['sfw_c2'] = 1.0                       #std fleece weight genotype params

######
#SAP #
######
sap['evg'] = 0.0               #energy content of liveweight gain - this is a high level sa, it impacts within a calculation not on an input and is only implemented on adults
sap['mortalityp'] = 0          #Scale the calculated progeny mortality at birth relative - this is a high level sa, it impacts within a calculation not on an input
sap['mortalitye'] = 0          #Scale the calculated dam mortality at birth - this is a high level sa, it impacts within a calculation not on an input
sap['mortalityb'] = 0          #Scale the calculated base mortality - this is a high level sa, it impacts within a calculation not on an input


######
#SAA #
######
##stock
saa['husb_cost_h2'] = np.zeros(uinp.sheep_inp['i_husb_operations_contract_cost_h2'].shape, dtype=np.float64)  #SA value for husbandry costs.
saa['husb_labour_l2h2'] = np.zeros(uinp.sheep_inp['i_husb_operations_labourreq_l2h2'].shape, dtype=np.float64)  #SA value for husbandry labour.

##stock parameters
saa['sfd_c2'] = 0.0                       #std fibre diameter genotype params
saa['cl0_c2'] = np.zeros(uinp.parameters_inp['i_cl0_c2'].shape, dtype=np.float64)  #SA value for litter size genotype params.
saa['scan_std_c2'] = 0                #std scanning percentage of a genotype. Controls the MU repro, initial propn of sing/twin/trip prog required to replace the dams, the lifetime productivity of the dams as affected by their BTRT..

######
#SAT #
######
sat['salep_weight_scalar'] = 0 #Scalar for LW impact across grid 1
sat['salep_score_scalar'] = 0  #Scalar for score impact across the grid


######
#SAR #
######


######
#SAV #
######
##if you initialise an array it must be type object (so that you can assign int/float/bool into the array)
##general
sav['steady_state']      = '-'                  #SA to alter if the model is steady state

##finance
sav['minroe']      = '-'                  #SA to alter the minroe (applied to both steady-state and dsp minroe inputs)

##bounds
sav['bnd_total_pas_area'] = '-'  #Total pasture area for bound. '-' is default so it will chuck an error if the bound is turned on without a specified area
sav['bnd_pasarea_inc'] = '-'   #SA to turn on the pasture area bound
sav['bnd_mateyearlings_inc'] = '-'   #SA to bound no mating of ewe yearlings.
sav['bnd_sellyearlings_inc'] = '-'   #SA to bound no selling of ewe yearlings.

##pasture
sav['pas_inc'] = np.full_like(pinp.general_inp['pas_inc'], '-', dtype=object) #SA value for pastures included mask

##Stock
sav['fec_inc'] = '-'    #SA to store FEC report values
sav['lw_inc'] = '-'     #SA to store LW report values
sav['ffcfw_inc'] = '-'  #SA to store FFCFW report values
sav['eqn_compare']      = '-'                  #SA to alter if the different equation systems in the sheep sim are run and compared
sav['eqn_used_g0_q1p7'] = np.full(uinp.sheep_inp['i_eqn_used_g0_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g1_q1p7'] = np.full(uinp.sheep_inp['i_eqn_used_g1_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g2_q1p7'] = np.full(uinp.sheep_inp['i_eqn_used_g2_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['eqn_used_g3_q1p7'] = np.full(uinp.sheep_inp['i_eqn_used_g3_q1p7'].shape, '-', dtype=object) #SA value for which equation system to use
sav['TOL_inc']          = np.full(pinp.sheep_inp['i_mask_i'].shape, '-', dtype=object)      # SA value for the inclusion of each TOL
sav['g3_included']      = np.full(pinp.sheep_inp['i_g3_inc'].shape, '-', dtype=object)      # SA value for the inclusion of each offspring genotype
sav['genotype']         = np.full(pinp.sheep_inp['a_c2_c0'].shape, '-', dtype=object)       # this is the selection of the genotypes of the sires for B, M & T
sav['scan_og1']         = np.full(pinp.sheep_inp['i_scan_og1'].shape, '-', dtype=object)    # SA value for the scanning management option
sav['woolp_mpg_percentile'] = '-'               #sa value for the wool price percentile
sav['woolp_mpg'] = '-'                          # sa value for wool price at std micron
sav['woolp_fdprem_percentile'] = '-'            # sa value for fd premium percentile (premium received by fd compared to std)
sav['woolp_fdprem'] = '-'                       # sa value for fd premium
sav['salep_percentile'] = '-'                   #Value for percentile for all sale grids
sav['salep_max'] = '-'                          #max sale price in grid
sav['nut_mask_dams'] = np.full(pinp.sheep_inp['i_sai_lw_dams_owi'].shape, '-', dtype=object)    #masks the nutrition options available eg high low high - the options selected are available for each starting weight
sav['nut_mask_offs'] = np.full(pinp.sheep_inp['i_sai_lw_offs_swix'].shape, '-', dtype=object)   #masks the nutrition options available eg high low high - the options selected are available for each starting weight
sav['nut_spread_n1'] = np.full(pinp.sheep_inp['i_nut_spread_n1'].shape, '-', dtype=object)      #nut spread dams
sav['nut_spread_n3'] = np.full(pinp.sheep_inp['i_nut_spread_n3'].shape, '-', dtype=object)      #nut spread dams
sav['drys_sold'] = '-'   #SA to force drys to be sold
sav['drys_retained'] = '-'   #SA to force drys to be retained
sav['r1_izg1'] = '-'   #SA to change the base feed option for dams

##stock parameters
sav['srw_c2'] = np.full(uinp.parameters_inp['i_srw_c2'].shape, '-', dtype=object)  #SA value for srw of each c2 genotype.
