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
from LabourFixed import *
from LabourPyomo import *
from CreateModel import *


'''
pyomo parameters
'''
#hours of labour required to complete super and wc activities
model.super_labour_requirment = Param(model.periods, initialize= labour_periods['super'].to_dict(), doc='hours of labour required to complete super and wc activities')

#hours of labour required to complete bas activities
model.bas_labour_requirment = Param(model.periods, initialize= labour_periods['bas'].to_dict(), doc='hours of labour required to complete bas activities')

#hours of labour required to complete planning activities
model.planning_labour_requirment = Param(model.periods, initialize= labour_periods['planning'].to_dict(), doc='hours of labour required to complete planning activities')

#hours of labour required to complete tax activities
model.tax_labour_requirment = Param(model.periods, initialize= labour_periods['tax'].to_dict(), doc='hours of labour required to complete tax activities')

#hours of labour required to complete learning activities
model.learn_labour_requirment = Param(initialize= labour_fixed_input_data['labour_learn'], doc='hours of labour required to complete learning activities')


'''
pyomo variables
'''
#amount of labour learn required in each period, this variable is used in the core model to link labour requirment with labour supply.
model.labour_learn = Var(model.periods, bounds = (0,None) , doc='proportion of learning done each labour period')

#amount of labour_planning required in each period, this variable is used in the core model to link labour requirment with labour supply.
model.labour_planning = Var(model.periods, bounds = (0,None) , doc='proportion of learning done each labour period')

#amount of labour_super required in each period, this variable is used in the core model to link labour requirment with labour supply.
model.labour_super = Var(model.periods, bounds = (0,None) , doc='proportion of learning done each labour period')

#amount of labour tax required in each period, this variable is used in the core model to link labour requirment with labour supply.
model.labour_tax = Var(model.periods, bounds = (0,None) , doc='proportion of learning done each labour period')

#amount of labour_bas required in each period, this variable is used in the core model to link labour requirment with labour supply.
model.labour_bas = Var(model.periods, bounds = (0,None) , doc='proportion of learning done each labour period')

#proportion of required learning done in each period, this variable exists so the model can optimise the timing of labour learn as a farmer would do in real life.
model.quantity_learn = Var(model.periods, bounds = (0,1) , doc='proportion of learning done each labour period')


'''
pyomo constraints
'''

#constraint to link labour supply with learn labour requirment. labour supplied only by permanent and farmer staff because casual dont need to learn (can be changed though)
def labour_learn(model, i):
    return model.labour_learn[i] - model.quantity_learn[i] * model.learn_labour_requirment >= 0
model.labour_learn_con = Constraint(model.periods, rule = labour_learn, doc='requirment of learn is filled by labour learn variable')

#constraint to link labour supply with planning labour requirment. labour supplied only by permanent and farmer staff because casual dont plan
def labour_planning(model, i):
    return model.labour_planning[i] - model.planning_labour_requirment[i] >= 0
model.labour_planning_con = Constraint(model.periods, rule = labour_planning, doc='requirment of planning is filled by labour planning variable')

#constraint to link labour supply with planning labour requirment. labour supplied only by permanent and farmer staff because casual dont plan
def labour_super(model, i):
    return model.labour_super[i] - model.super_labour_requirment[i] >= 0
model.labour_super_con = Constraint(model.periods, rule = labour_super, doc='requirment of super and wc is filled by labour super and wc variable')

#constraint to link labour supply with planning labour requirment. labour supplied only by permanent and farmer staff because casual dont plan
def labour_tax(model, i):
    return model.labour_tax[i] - model.tax_labour_requirment[i] >= 0
model.labour_tax_con = Constraint(model.periods, rule = labour_tax, doc='requirment of tax is filled by labour tax variable')

#constraint to link labour supply with planning labour requirment. labour supplied only by permanent and farmer staff because casual dont plan
def labour_bas(model, i):
    return model.labour_bas[i] - model.bas_labour_requirment[i] >= 0
model.labour_bas_con = Constraint(model.periods, rule = labour_bas, doc='requirment of BAS is filled by labour BAS variable')

#constraint makes sure the model allocate the labour learn to labour periods, because labour learn timing is optimised (others are fixed timing determined in input sheet)
def labour_learn_period(model):
    return sum(model.quantity_learn[i] * model.learn_labour_requirment for i in model.periods ) - model.learn_labour_requirment >= 0
model.labour_learn_period = Constraint(rule = labour_learn_period, doc='constrains the amount of labour learn in each period')


#model.pprint()