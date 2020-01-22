# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:13:42 2019

module - crop labour inputs

@author: young
"""
crop_labour_input = {}

#########################
#helper time            #
#########################

#proportion of seeding time that requires a helper ie Ferrying fuel, grain, fertiliser, repairs & maintenance, Ferrying vehicles between paddocks, Changing machine settings between paddocks
crop_labour_input['seeding_helper'] = 0.5

#harvest helper time per crop
#includes chaser bin driver ie 100% helper time
crop_labour_input['harvest_helper'] = {'w' : 1.3, \
                                             'b' : 1.2, \
                                             'o' : 1.2, \
                                             'r' : 1.2, \
                                             'z' : 1.2, \
                                             'l' : 1.2, \
                                             'f' : 1.2}

#########################
#pack and prep time     #
#########################
#harv gear , month first
crop_labour_input['harvest_prep'] = { '11,5,2019': 20, \
                                      '10,5,2019' : 20}

#fert gear
crop_labour_input['fert_prep'] = { '4,5,2019': 13, \
                                      '5,5,2019': 3, \
                                      '8,5,2019' : 3}

#spray gear
crop_labour_input['spray_prep'] = { '4,5,2019': 13, \
                                      '5,5,2019': 3, \
                                      '9,5,2019' : 3}

#seeding gear
crop_labour_input['seed_prep'] = { '3,5,2019': 40, \
                                      '4,5,2019': 40, \
                                      '6,16,2019' : 5} #after seeding finishes