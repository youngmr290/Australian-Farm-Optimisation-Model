# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:55:51 2019

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
import Labour as lab
from CreateModel import *
import PropertyInputs as pinp

def lab_precalcs(params):
    lab.labour_general(params)
    lab.perm_cost(params)
    lab.manager_cost(params)
    params['min_perm'] = pinp.labour['min_perm'] 
    params['max_perm'] = pinp.labour['max_perm']
    params['min_managers'] = pinp.labour['min_managers'] 
    params['max_managers'] = pinp.labour['max_managers']
                    


def labpyomo_local(params):
    #########
    #param  #
    #########    
    ##called here , used below to generate params
    try:
        model.del_component(model.p_perm_hours)
    except AttributeError:
        pass
    model.p_perm_hours = Param(model.s_labperiods, initialize= params['permanent hours'], doc='hours worked by a permanent staff in each period')
    
    try:
        model.del_component(model.p_perm_supervison)
    except AttributeError:
        pass
    model.p_perm_supervison = Param(model.s_labperiods, initialize= params['permanent supervision'], doc='hours of supervision required by a permanent staff in each period')
    
    try:
        model.del_component(model.p_perm_cost)
    except AttributeError:
        pass
    model.p_perm_cost = Param(model.s_cashflow_periods, initialize = params['perm_cost'], default = 0.0, doc = 'cost of a permanent staff for 1 yr')
    
    try:
        model.del_component(model.p_casual_cost_index)
        model.del_component(model.p_casual_cost)
    except AttributeError:
        pass
    model.p_casual_cost = Param(model.s_labperiods, model.s_cashflow_periods,  initialize = params['casual_cost'], default = 0.0, doc = 'cost of a casual staff for each labour period')
    
    try:
        model.del_component(model.p_casual_hours)
    except AttributeError:
        pass
    model.p_casual_hours = Param(model.s_labperiods, initialize= params['casual hours'], doc='hours worked by a casual staff in each period')
    
    try:
        model.del_component(model.p_casual_supervison)
    except AttributeError:
        pass
    model.p_casual_supervison = Param(model.s_labperiods, initialize= params['casual supervision'], doc='hours of supervision required by a casual staff in each period')
    
    try:
        model.del_component(model.p_manager_hours)
    except AttributeError:
        pass
    model.p_manager_hours = Param(model.s_labperiods, initialize= params['manager hours'], doc='hours worked by a manager in each period')
    
    try:
        model.del_component(model.p_manager_cost)
    except AttributeError:
        pass
    model.p_manager_cost = Param(model.s_cashflow_periods, initialize = params['manager_cost'], doc = 'cost of a manager for 1 yr')
    
    try:
        model.del_component(model.p_casual_upper)
    except AttributeError:
        pass
    model.p_casual_upper = Param(model.s_labperiods, initialize = params['casual ub'],  doc = 'casual availability upper bound')
    
    try:
        model.del_component(model.p_casual_lower)
    except AttributeError:
        pass
    model.p_casual_lower = Param(model.s_labperiods, initialize = params['casual lb'], doc = 'casual availability lower bound')

###############################
#local constraints            #
###############################
#to constrain the amount of casual labour in each period
#this can't be done with variable bounds because it's not a constant value for each period (seeding and harv may differ)
    try:
        model.del_component(model.casual_bounds)
    except AttributeError:
        pass
    def casual_labour_availability(model, p):
        return  (model.p_casual_lower[p], model.v_quantity_casual[p], model.p_casual_upper[p])
    model.casual_bounds = Constraint(model.s_labperiods, rule = casual_labour_availability, doc='bounds the casual labour in each period')
    
    #manager, this is a little more complex because also need to subtract the supervision hours off of the manager supply of workable hours
    try:
        model.del_component(model.labour_transfer_manager)
    except AttributeError:
        pass
    def labour_transfer_manager(model,p):
        return -(model.v_quantity_manager * model.p_manager_hours[p]) + (model.p_perm_supervison[p] * model.v_quantity_perm) + (model.p_casual_supervison[p] * model.v_quantity_casual[p])      \
    + model.v_sheep_labour_manager[p] + model.v_crop_labour_manager[p] + model.v_fixed_labour_manager[p]  <= 0
    model.labour_transfer_manager = Constraint(model.s_labperiods, rule = labour_transfer_manager, doc='labour from manager to sheep and crop and fixed')
    
    #permanent 
    try:
        model.del_component(model.labour_transfer_permanent)
    except AttributeError:
        pass
    def labour_transfer_permanent(model,p):
        return -(model.v_quantity_perm *  model.p_perm_hours[p])    \
        + model.v_sheep_labour_permanent[p] + model.v_crop_labour_permanent[p] + model.v_fixed_labour_permanent[p] <= 0
    model.labour_transfer_permanent = Constraint(model.s_labperiods, rule = labour_transfer_permanent, doc='labour from permanent staff to sheep and crop and fixed')
    
    #casual note perm and manager can do casual tasks - variables may need to change name so to be less confusing
    try:
        model.del_component(model.labour_transfer_casual)
    except AttributeError:
        pass
    def labour_transfer_casual(model,p):
        return -(model.v_quantity_casual[p] *  model.p_casual_hours[p])  \
            + model.v_sheep_labour_casual[p] + model.v_crop_labour_casual[p] + model.v_fixed_labour_casual[p]  <= 0
    model.labour_transfer_casual = Constraint(model.s_labperiods, rule = labour_transfer_casual, doc='labour from casual staff to sheep and crop and fixed')

############
#variable  #
############    
#Amount of casual. Casual labour can be optimised for each period 
model.v_quantity_casual = Var(model.s_labperiods, bounds = (0,None) , doc='number of casual labour used in each labour period')

#Amount of permanent labour. 
model.v_quantity_perm = Var(bounds=(pinp.labour['min_perm'],pinp.labour['max_perm']), doc='number of permanent labour used in each labour period')

#Amount of manager labour 
model.v_quantity_manager = Var(bounds=(pinp.labour['min_managers'],pinp.labour['max_managers']), doc='number of manager/owner labour used in each labour period')

#manager pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.v_sheep_labour_manager = Var(model.s_labperiods, bounds = (0,None), doc='manager labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.v_crop_labour_manager = Var(model.s_labperiods, bounds = (0,None), doc='manager labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.v_fixed_labour_manager = Var(model.s_labperiods, bounds = (0,None), doc='manager labour used by fixed activities in each labour period')

#permanent pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.v_sheep_labour_permanent = Var(model.s_labperiods, bounds = (0,None), doc='permanent labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.v_crop_labour_permanent = Var(model.s_labperiods, bounds = (0,None), doc='permanent labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.v_fixed_labour_permanent = Var(model.s_labperiods, bounds = (0,None), doc='permanent labour used by fixed activities in each labour period')

#casual pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.v_sheep_labour_casual = Var(model.s_labperiods, bounds = (0,None), doc='casual labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.v_crop_labour_casual = Var(model.s_labperiods, bounds = (0,None), doc='casual labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.v_fixed_labour_casual = Var(model.s_labperiods, bounds = (0,None), doc='casual labour used by fixed activities in each labour period')

#^happens ing core model now - there isnt a transfefr but the different activities provide for the same con eg perm labour provides into the casual labour constraint.
# #transfer labour between pools
# #transfer labour from manager to casual, because jobs are tasked to particular pools and jobs tasked to casual can be done by the manager or permanent staff 
# model.v_manager_casual_labour_transfer = Var(model.s_labperiods, bounds = (0,None), doc='manager labour used to complete jobs tasked to casual')

# #transfer labour from manager to permanent 
# model.v_manager_permanent_labour_transfer = Var(model.s_labperiods, bounds = (0,None), doc='manager labour used to complete jobs tasked to permanent')

# #transfer labour from permanent to casual 
# model.v_permanent_casual_labour_transfer = Var(model.s_labperiods, bounds = (0,None), doc='permanent labour used to complete jobs tasked to casual')



#######################
#labour cost function #
#######################

#sum the cost of perm, casual and manager labour. When i tried to do it all in one function it didn't work (it should be possible though )
def casual(model,c):
    return sum( model.v_quantity_casual[p] * model.p_casual_cost[p,c] for p in model.s_labperiods) 
def perm(model,c):
    return model.v_quantity_perm * model.p_perm_cost[c] 
def manager(model,c):
    return model.v_quantity_manager * model.p_manager_cost[c] 
def labour_cost(model,c):
    return casual(model,c) + perm(model,c) + manager(model,c)





