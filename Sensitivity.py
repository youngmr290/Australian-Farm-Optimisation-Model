# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:57:56 2020


@author: young
"""

import numpy as np
import PropertyInputs as pinp


#global potential intake
sap_pi=0

########################################
# Pasture module sensitivity variables #
########################################

sam_germ :float             = 1.0                                                              # SA multiplier for germination on all lmus in all periods
sam_pgr :float              = 1.0                                                              # SA multiplier for growth on all lmus in all periods
sam_dmd_decline_dry:float   = 1.0                                                              # SA multiplier for the decline in digestibility of dry feed
sam_pgr_f                   = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),   dtype=np.float64)   # SA multiplier for growth in each feed period
sam_dmd_decline_foo_f       = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),   dtype=np.float64)   # SA multiplier on decline in digestibility if green feed is not grazed (to increase FOO)
sam_dmd_range_f             = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),   dtype=np.float64)   # SA multiplier on range in digestibility of green feed
sam_dmd_senesce_f           = np.ones((len(pinp.feed_inputs['feed_periods']) - 1),   dtype=np.float64)   # SA multiplier on reduction in digestibility when senescing
sam_germ_l                  = np.ones((len(pinp.general['lmu_area'])),               dtype=np.float64)   # SA multiplier for germination on each lmus in all periods
sam_pgr_l                   = np.ones((len(pinp.general['lmu_area'])),               dtype=np.float64)   # SA multiplier for growth on each lmus in all periods
sam_dmd_decline_grn_l       = np.ones((len(pinp.general['lmu_area'])),               dtype=np.float64)   # SA multiplier for the decline in digestibility of green feed
