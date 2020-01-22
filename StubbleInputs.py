# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:10:49 2019

need to move this to excel

***note - hay, faba, lupins are not accuratley calibrated ***


@author: young
"""
#inputs used in simulation and stubble calc
stubble_inputs={}


####################
# stubble inputs   #
####################
#sim stuff 
stubble_inputs['step_size']= 1 #the intake % per step. ie x% pf the stubble components are consumed each step

stubble_inputs['clover_propn_in_sward_stubble'] = 0.1 #goes with RIg to influence pi


#component dmd at harvest
#harvest index - harvest index (HI) is the ratio of harvested grain to total shoot dry matter
#proportion of the grain harvested
#quality deterioration (%/day)
#stubble cat prop - proportion of each stubble category that makes up all the stubble ie must add to 100 (this is done automatically by calculatiing d based on the other inputs)
#stub qual - quality that you want each cat to be - this is determined bases on exp data (currently from ed)
#quantity deterioration - %/day decline in available stub
#trampling - % of stubble lost per kg consumed, currently the same for each cat - note grain feeding may increase trampling effect
stubble_inputs['crop_stub']={'w':{'component_dmd':{'grain':85
                                                  ,'blade':58
                                                  ,'sheath':37
                                                  ,'chaff':37
                                                  ,'stem':23}
                                   ,'harvest_index': 0.42
                                   ,'proportion_grain_harv': 0.94
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.05
                                                  ,'b':0.04
                                                  ,'c':0.1
                                                  ,'d':0} # d is calculted in stub sim module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':80  
                                                    ,'b':51
                                                    ,'c':45
                                                    ,'d':30}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'b':{'component_dmd':{'grain':85
                                                  ,'blade':58
                                                  ,'sheath':37
                                                  ,'chaff':37
                                                  ,'stem':23}
                                   ,'harvest_index': 0.44
                                   ,'proportion_grain_harv': 0.94
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.05
                                                  ,'b':0.04
                                                  ,'c':0.1
                                                  ,'d':0}  # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':80  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':51
                                                    ,'c':45
                                                    ,'d':30}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'o':{'component_dmd':{'grain':85
                                                  ,'blade':58
                                                  ,'sheath':37
                                                  ,'chaff':37
                                                  ,'stem':23}
                                   ,'harvest_index': 0.40
                                   ,'proportion_grain_harv': 0.94
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.05
                                                  ,'b':0.04
                                                  ,'c':0.1
                                                  ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':80  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':51
                                                    ,'c':45
                                                    ,'d':30}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'z':{'component_dmd':{'grain':95
                                                  ,'blade':64
                                                  ,'sheath':0 #canola doesn't have this
                                                  ,'chaff':52 #this is pod for canola
                                                  ,'stem':31}
                                   ,'harvest_index': 0.2
                                   ,'proportion_grain_harv': 0.97
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.02
                                                  ,'b':0.03
                                                  ,'c':0.1
                                                  ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':89  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':68
                                                    ,'c':46
                                                    ,'d':33}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'r':{'component_dmd':{'grain':95
                                                  ,'blade':64
                                                  ,'sheath':0 #canola doesn't have this
                                                  ,'chaff':52 #this is pod for canola
                                                  ,'stem':31}
                                   ,'harvest_index': 0.2
                                   ,'proportion_grain_harv': 0.97
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.02
                                                  ,'b':0.03
                                                  ,'c':0.1
                                                  ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':89  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':68
                                                    ,'c':46
                                                    ,'d':33}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'f':{'component_dmd':{'grain':90
                                                  ,'blade':64
                                                  ,'sheath':0 #canola doesn't have this
                                                  ,'chaff':58 #this is pod for canola
                                                  ,'stem':35}
                                   ,'harvest_index': 0.3
                                   ,'proportion_grain_harv': 0.90
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.05
                                                  ,'b':0.05
                                                  ,'c':0.05
                                                  ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':85  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':65
                                                    ,'c':56
                                                    ,'d':43}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}

                        ,'l':{'component_dmd':{'grain':90
                                              ,'blade':64
                                              ,'sheath':0 #canola doesn't have this
                                              ,'chaff':58 #this is pod for canola
                                              ,'stem':35}
                               ,'harvest_index': 0.3
                               ,'proportion_grain_harv': 0.90
                               ,'quality_deterioration':{'grain':0.05
                                              ,'blade':0.3
                                              ,'sheath':0.3
                                              ,'chaff':0.3
                                              ,'stem':0.3}
                               ,'stub_cat_prop':{'a':0.05
                                              ,'b':0.05
                                              ,'c':0.05
                                              ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                               ,'stub_cat_qual':{'a':85  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                ,'b':65
                                                ,'c':56
                                                ,'d':43}
                               ,'quantity_deterioration':0.4 #%/day
                               ,'trampling':2}

                        ,'h':{'component_dmd':{'grain':85
                                                  ,'blade':58
                                                  ,'sheath':37
                                                  ,'chaff':37
                                                  ,'stem':23}
                                   ,'harvest_index': 0.40
                                   ,'proportion_grain_harv': 0.94
                                   ,'quality_deterioration':{'grain':0.05
                                                  ,'blade':0.3
                                                  ,'sheath':0.3
                                                  ,'chaff':0.3
                                                  ,'stem':0.3}
                                   ,'stub_cat_prop':{'a':0.05
                                                  ,'b':0.04
                                                  ,'c':0.1
                                                  ,'d':0} # d is calculted in stub module d = 1-(a+b+c)
                                   ,'stub_cat_qual':{'a':80  #quality that you want each cat to be - this is determined bases on exp data (currently from ed)
                                                    ,'b':51
                                                    ,'c':45
                                                    ,'d':30}
                                   ,'quantity_deterioration':0.4 #%/day
                                   ,'trampling':2}}
