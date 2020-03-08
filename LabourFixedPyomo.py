# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:07:19 2019

module: labour pyomo module - contains pyomo params, variables and constraits

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

@author: young
"""

#python modules
from pyomo.environ import *

#MUDAS modules
import LabourFixed as lfix
import PropertyInputs as pinp
# from LabourPyomo import *
from CreateModel import *


def labfxpyomo_local():
    #########
    #param  #
    #########    
    ##call labour fixed function then extract info out of df in the section below
    labour_periods_fx=lfix.fixed()
    
    try:
        model.del_component(model.p_super_labour)
    except AttributeError:
        pass
    model.p_super_labour = Param(model.s_periods, initialize= labour_periods_fx['super'].to_dict(), doc='hours of labour required to complete super and wc activities')
    
    try:
        model.del_component(model.p_bas_labour)
    except AttributeError:
        pass
    model.p_bas_labour = Param(model.s_periods, initialize= labour_periods_fx['bas'].to_dict(), doc='hours of labour required to complete bas activities')
    
    try:
        model.del_component(model.p_planning_labour)
    except AttributeError:
        pass
    model.p_planning_labour = Param(model.s_periods, initialize= labour_periods_fx['planning'].to_dict(), doc='hours of labour required to complete planning activities')
    
    try:
        model.del_component(model.p_tax_labour)
    except AttributeError:
        pass
    model.p_tax_labour = Param(model.s_periods, initialize= labour_periods_fx['tax'].to_dict(), doc='hours of labour required to complete tax activities')
    
    try:
        model.del_component(model.p_learn_labour)
    except AttributeError:
        pass
    model.p_learn_labour = Param(initialize= pinp.labour['learn'], doc='hours of labour required to complete learning activities')

    ###################################
    #local constraints                #
    ###################################
    ##constraint makes sure the model allocate the labour learn to labour periods, because labour learn timing is optimised (others are fixed timing determined in input sheet)
    try:
        model.del_component(model.labour_learn_period)
    except AttributeError:
        pass
    def labour_learn_period(model):
        return sum(model.v_learn_allocation[i] * model.p_learn_labour for i in model.s_periods ) - model.p_learn_labour >= 0
    model.labour_learn_period = Constraint(rule = labour_learn_period, doc='constrains the amount of labour learn in each period')

############
#variables #
############  
model.v_learn_allocation = Var(model.s_periods, bounds = (0,1) , doc='proportion of learning done each labour period')



