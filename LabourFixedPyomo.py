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
import pyomo.environ as pe

#AFO modules
import LabourFixed as lfix
import PropertyInputs as pinp

def labfx_precalcs(params, report):
    lfix.fixed(params)
    params['learn'] = pinp.labour['learn']
    
def f1_labfxpyomo_local(params, model):
    ############
    # variables #
    ############
    model.v_learn_allocation = pe.Var(model.s_sequence_year, model.s_sequence, model.s_labperiods,
                                   bounds=(0,1), doc='proportion of learning done each labour period')

    #########
    #param  #
    #########
    model.p_super_labour = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['super'], mutable=False, doc='hours of labour required to complete super and wc activities')
    
    model.p_bas_labour = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['bas'], mutable=False, doc='hours of labour required to complete bas activities')
    
    model.p_planning_labour = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['planning'], mutable=False, doc='hours of labour required to complete planning activities')
    
    model.p_tax_labour = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['tax'], mutable=False, doc='hours of labour required to complete tax activities')
    
    model.p_learn_labour = pe.Param(initialize= params['learn'], doc='hours of labour required to complete learning activities')

    ###################################
    #local constraints                #
    ###################################
    ##constraint makes sure the model allocate the labour learn to labour periods, because labour learn timing is optimised (others are fixed timing determined in input sheet)
    ##Has to be the same across the z axis so that learn labour cant be done before the season is identified and after that season is identified for the parent.
    def labour_learn_period(model,q,s):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return -sum(model.v_learn_allocation[q,s,p] for p in model.s_labperiods) <= -1
        else:
            return pe.Constraint.Skip

    model.con_labour_learn_period = pe.Constraint(model.s_sequence_year, model.s_sequence, rule = labour_learn_period, doc='constrains the allocation of labour learn to a total of 1')



