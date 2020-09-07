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

def labfx_precalcs(params):
    lfix.fixed(params)
    params['learn'] = pinp.labour['learn']
    
def labfxpyomo_local(params):
    #########
    #param  #
    #########    
    
    try:
        model.del_component(model.p_super_labour)
    except AttributeError:
        pass
    model.p_super_labour = Param(model.s_labperiods, initialize= params['super'], doc='hours of labour required to complete super and wc activities')
    
    try:
        model.del_component(model.p_bas_labour)
    except AttributeError:
        pass
    model.p_bas_labour = Param(model.s_labperiods, initialize= params['bas'], doc='hours of labour required to complete bas activities')
    
    try:
        model.del_component(model.p_planning_labour)
    except AttributeError:
        pass
    model.p_planning_labour = Param(model.s_labperiods, initialize= params['planning'], doc='hours of labour required to complete planning activities')
    
    try:
        model.del_component(model.p_tax_labour)
    except AttributeError:
        pass
    model.p_tax_labour = Param(model.s_labperiods, initialize= params['tax'], doc='hours of labour required to complete tax activities')
    
    try:
        model.del_component(model.p_learn_labour)
    except AttributeError:
        pass
    model.p_learn_labour = Param(initialize= params['learn'], doc='hours of labour required to complete learning activities')

    ###################################
    #local constraints                #
    ###################################
    ##constraint makes sure the model allocate the labour learn to labour periods, because labour learn timing is optimised (others are fixed timing determined in input sheet)
    try:
        model.del_component(model.labour_learn_period)
    except AttributeError:
        pass
    def labour_learn_period(model):
        # return -sum(model.v_learn_allocation[i] * model.p_learn_labour for i in model.s_labperiods ) + model.p_learn_labour <= 0
        return -sum(model.v_learn_allocation[p] for p in model.s_labperiods)  <= -1
    model.labour_learn_period = Constraint(rule = labour_learn_period, doc='constrains the amount of labour learn in each period')

############
#variables #
############  
model.v_learn_allocation = Var(model.s_labperiods, bounds = (0,1) , doc='proportion of learning done each labour period')



