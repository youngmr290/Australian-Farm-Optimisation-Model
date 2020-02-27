# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:00:25 2019

@author: John
"""
#python modules
from pyomo.environ import *

#MUDAS modules
from CreateModel import *
import Pasture as pas

pastures = ['annual']        # ^should be from UniversalInputs.py or perhaps PropertyInputs see also PropertyInputs.py

# Conservation limit over feed period(i) & soil type(j) = 50 constraints
# Note: the MIDAS version was effectively just 5 constraints achieved
# by subtracting the conservation limit from the period 1 low quality Dry foo row (and relying on non negative FOO)
sum(sum(foo(f,l)>=erosion_limit(f,l) for f in feed_period) for l in lmu)




#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Params
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

pas.init_and_map_excel('Property.xlsx', pastures)                         # read inputs from Excel file and map to the python variables
pas.calculate_germ_and_reseed()                          # calculate the germination for each rotation phase
pas.green_and_dry()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment

try:
    model.del_component(model.p_poc_con)
except AttributeError:
    pass
model.p_poc_con = Param(model.s_feed_periods ,model.s_lmus, initialize=,default=0, doc='consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')

try:
    model.del_component(model.p_poc_md)
except AttributeError:
    pass
model.p_poc_md = Param(model.s_feed_periods, initialize=,default=0, doc='md of pasture on crop paddocks for each feed period')

try:
    model.del_component(model.p_poc_vol)
except AttributeError:
    pass
model.p_poc_vol = Param(model.s_feed_periods, initialize=,default=0, doc='vol (ri intake) of pasture on crop paddocks for each feed period')




#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
# Variables
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

#Pasture on crop paddocks before seeding
model.v_sheep_pascroppaddocks = Var(model.s_feed_periods, model.s_sheep_pools, bounds = (0,None) , doc='a given sheep pool grazing 1t of foo on a crop paddock before seeding')

