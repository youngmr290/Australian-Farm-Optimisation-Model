# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:07:19 2019

module: labour pyomo module - contains pyomo params, variables and constraints

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

@author: young
"""

#python modules
from pyomo.environ import *

#AFO modules
import LabourFixed as lfix
from CreateModel import *
import PropertyInputs as pinp

def labfx_precalcs(params, report):
    lfix.fixed(params)
    params['learn'] = pinp.labour['learn']
    
def labfxpyomo_local(params):
    ############
    # variables #
    ############
    try:
        model.del_component(model.v_learn_allocation)
        model.del_component(model.v_learn_allocation_index)
    except AttributeError:
        pass
    model.v_learn_allocation = Var(model.s_labperiods,bounds=(0,1),doc='proportion of learning done each labour period')

    #########
    #param  #
    #########

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    try:
        model.del_component(model.p_super_labour)
    except AttributeError:
        pass
    model.p_super_labour = Param(model.s_labperiods, initialize= params[season]['super'], mutable=True, doc='hours of labour required to complete super and wc activities')
    
    try:
        model.del_component(model.p_bas_labour)
    except AttributeError:
        pass
    model.p_bas_labour = Param(model.s_labperiods, initialize= params[season]['bas'], mutable=True, doc='hours of labour required to complete bas activities')
    
    try:
        model.del_component(model.p_planning_labour)
    except AttributeError:
        pass
    model.p_planning_labour = Param(model.s_labperiods, initialize= params[season]['planning'], mutable=True, doc='hours of labour required to complete planning activities')
    
    try:
        model.del_component(model.p_tax_labour)
    except AttributeError:
        pass
    model.p_tax_labour = Param(model.s_labperiods, initialize= params[season]['tax'], mutable=True, doc='hours of labour required to complete tax activities')
    
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
        model.del_component(model.con_labour_learn_period)
    except AttributeError:
        pass
    def labour_learn_period(model):
        # return -sum(model.v_learn_allocation[i] * model.p_learn_labour for i in model.s_labperiods ) + model.p_learn_labour <= 0
        return -sum(model.v_learn_allocation[p] for p in model.s_labperiods)  <= -1
    model.con_labour_learn_period = Constraint(rule = labour_learn_period, doc='constrains the amount of labour learn in each period')



