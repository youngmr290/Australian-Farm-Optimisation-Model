# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young
"""

import numpy as np
import PropertyInputs as pinp
import UniversalInputs as uinp 

##create dict - store sa variables in dict so they can easily be changed in the exp loop
sap = dict()
sam = dict()
saa = dict()

######
#SAP #
######

##Global
sap['pi']=0 #global potential intake

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

######
#SAA #
######