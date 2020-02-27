# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young
"""

import numpy as np
import PropertyInputs as pinp

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

## Pasture module sensitivity variables 
### ^ The pasture sensitivity variable need a t component added. This requires accessing the number of pastures included, which is controlled in the Pasture module
sam['germ']                    = 1.0                                                                       # SA multiplier for germination on all lmus in all periods
sam['germ_l']                  = np.ones((len(pinp.general['lmu_area'])),              dtype=np.float64)  # SA multiplier for germination on each lmus in all periods
sam['pgr']                     = 1.0                                                                       # SA multiplier for growth on all lmus in all periods
sam['pgr_f']                   = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier for growth in each feed period
sam['pgr_l']                   = np.ones((len(pinp.general['lmu_area'])),              dtype=np.float64)  # SA multiplier for growth on each lmus in all periods
sam['pgr_t']                   = np.ones((pinp.n_pasture_types),                       dtype=np.float64)  # SA multiplier for growth of each pasture type
sam['dry_dmd_decline']         = 1.0                                                                       # SA multiplier for the decline in digestibility of dry feed
sam['grn_dmd_decline_l']       = np.ones((len(pinp.general['lmu_area'])),              dtype=np.float64)  # SA multiplier for the decline in digestibility of green feed
sam['grn_dmd_declinefoo_f']    = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam['grn_dmd_range_f']         = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on range in digestibility of green feed
sam['grn_dmd_senesce_f']       = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),  dtype=np.float64)  # SA multiplier on reduction in digestibility when senescing

# sa_feed_period_inc_t      = True    # growth of this pasture in this period is included
# sa_lmu_inc_t              = True    # this pasture is included on this lmu

######
#SAA #
######