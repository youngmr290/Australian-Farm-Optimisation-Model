# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 10:58:48 2019

module: finance module - info on debit and credit and overdraws

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
     
@author: young
"""
#python modules
from pyomo.environ import *

#MUDAS modules
from CreateModel import *


'''
#variables
'''
#credit for a given time period (time period defined by cashflow set)
model.v_credit = Var(model.s_cashflow_periods, bounds = (0.0, None), doc = 'amount of net positive cashflow in a given period')
#debit for a given time period (time period defined by cashflow set)
model.v_debit = Var(model.s_cashflow_periods, bounds = (0.0, None), doc = 'amount of net negitive cashflow in a given period')
##dep
model.v_dep = Var(bounds = (0.0, None), doc = 'transfers total dep to objective')
##dep
model.v_asset = Var(bounds = (0.0, None), doc = 'transfers total value of asset to objective to ensure opportuninty cost is represented')
##minroe
model.v_minroe = Var(bounds = (0.0, None), doc = 'total expenditure, used to ensure min returen is met')


def finpyomo_local():
    ####################
    #Local constrain   #
    ####################
    ##debit can't be more than a specified amount ie farmers will draw a maximum from the bank throughout yr
    def overdraw(model,c): 
        return model.v_debit[c] <= pinp.finance['overdraw_limit']
    try:
        model.del_component(model.con_overdraw)
    except AttributeError:
        pass
    model.con_overdraw = Constraint(model.s_cashflow_periods, rule=overdraw, doc='overdraw limit')

