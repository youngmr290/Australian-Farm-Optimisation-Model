# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:03:17 2020

@author: young
"""
#python modules
from pyomo import environ as pe

#MUDAS modules
from CreateModel import model
import SupFeed as sup


#########
#param  #
#########    
try:
    model.del_component(model.p_buy_grain_price)
except AttributeError:
    pass
model.p_buy_grain_price = pe.Param(model.s_crops, model.s_cashflow_periods, model.s_grain_pools, initialize=sup.buy_grain_price().to_dict(), default = 0.0, doc='price to buy grain from neighbour')
