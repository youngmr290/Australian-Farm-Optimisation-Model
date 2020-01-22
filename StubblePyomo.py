# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:10:55 2019

@author: young
"""

#python modules
from pyomo.environ import *

#MUDAS modules
import Stubble as stub
from CreateModel import *


####################
#params            #
####################

#this param has inf values - this may cause a problem, if it does do a replace in the stubble sheet line 151
model.p_stub_vol = Param(model.s_sheep_pools, model.s_feed_periods, model.s_stub_cat, model.s_crops, initialize=stub.stub_vol, doc='amount of intake volume required by 1t of each stubble category for each crop')

model.stub_vol.pprint()



###################
#local global#
###################

###################
#constraint global#
###################
##limit stubble production in harvet period. Sheep must consume a certain amount of pasture to consume stubble in harv period. Ie if harvest occurs 90% of the way through p7 then if a sheep is to consume 1t of stubble they must consume 9t of pasture. because it is incorrect to allow them to fill their p7 intake just from 10t of stubble
#def harv_st_con_limit():
    