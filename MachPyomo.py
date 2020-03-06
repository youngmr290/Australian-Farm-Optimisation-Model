# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 09:49:18 2019

module: machinery pyomo module

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
   

@author: young
"""

#python modules
from pyomo.environ import *
import sys
#MUDAS modules
import Mach as mac
from CreateModel import *

print('Status:  running machpyomo')
def machpyomo_local():
    
    ##called here to update mach option if necessary
    mac.select_mach_opt()
    
    #########
    #param  #
    #########    
    try:
        model.del_component(model.p_seeding_rate)
    except AttributeError:
        pass
    model.p_seeding_rate = Param(model.s_landuses, model.s_lmus, initialize=mac.overall_seed_rate(), default = 0.0, doc='rate of seeding ha/day provided by one crop gear')
    
    try:
        model.del_component(model.p_seed_days)
    except AttributeError:
        pass
    model.p_seed_days = Param(model.s_periods, initialize=mac.seed_days()['seed_days'].to_dict(), default = 0.0, doc='number of seeding days in each period')
    
    try:
        model.del_component(model.p_seeding_cost)
    except AttributeError:
        pass
    model.p_seeding_cost = Param(model.s_cashflow_periods, model.s_lmus, initialize=mac.seeding_cost_period().stack().to_dict(), default = 0.0, doc='cost of seeding 1ha')
    
    try:
        model.del_component(model.p_contract_seeding_cost)
    except AttributeError:
        pass
    model.p_contract_seeding_cost = Param(model.s_cashflow_periods, initialize=mac.contract_seed_cost(), default = 0.0, doc='cost of contract seeding 1ha')
    
    try:
        model.del_component(model.p_seeding_dep)
    except AttributeError:
        pass
    model.p_seeding_dep = Param(model.s_lmus, initialize=mac.seeding_dep(), default = 0.0, doc='depreciation cost of seeding 1ha')
    
    try:
        model.del_component(model.p_harv_rate)
    except AttributeError:
        pass
    model.p_harv_rate = Param(model.s_periods, model.s_crops, initialize=mac.harv_rate_period(), default = 0.0, doc='rate of harv t/hr provided by one crop gear each period')
    
    try:
        model.del_component(model.p_contractharv_rate)
    except AttributeError:
        pass
    model.p_contractharv_rate = Param(model.s_crops, initialize=mac.contract_harv_rate(), default = 0.0, doc='rate of harv t/hr provided by one crop gear each period')
    
    try:
        model.del_component(model.p_harv_hrs_max)
    except AttributeError:
        pass
    model.p_harv_hrs_max = Param(model.s_periods, initialize= mac.max_harv_hours(), default = 0.0, doc='max hours of harvest per period')
    
    try:
        model.del_component(model.p_harv_cost)
    except AttributeError:
        pass
    model.p_harv_cost = Param(model.s_cashflow_periods, model.s_crops, initialize=mac.harvest_cost_period(), default = 0.0, doc='cost of harvesting 1hr')
    
    try:
        model.del_component(model.p_contractharv_cost)
    except AttributeError:
        pass
    model.p_contractharv_cost = Param(model.s_cashflow_periods, model.s_crops, initialize=mac.contract_harvest_cost_period(), default = 0.0, doc='cost of contract harvesting 1hr')
    
    try:
        model.del_component(model.p_contracthay_cost)
    except AttributeError:
        pass
    model.p_contracthay_cost = Param(model.s_cashflow_periods, initialize=mac.hay_making_cost(), default = 0.0, doc='cost of contract making hay $/t')
    
    try:
        model.del_component(model.p_yield_penalty)
    except AttributeError:
        pass
    model.p_yield_penalty = Param(model.s_periods, model.s_crops, initialize=mac.yield_penalty(), default = 0.0, doc='kg/ha/day penalty for late sowing in each period')
    
    try:
        model.del_component(model.p_seeding_grazingdays)
    except AttributeError:
        pass
    model.p_seeding_grazingdays = Param(model.s_feed_periods, model.s_periods, initialize=mac.grazing_days(), default = 0.0, doc='pasture grazing days per feed period provided by 1ha of seeding in each seed period')

    ###################################
    #local constraints                #
    ###################################
    ##days of seeding is limited by seed period length and the number of crop gear (ie 2 crop gear gives you double the penalty free days)
    ##constraint to limit the number of seed days in each period ()
    ##includes a factor to account for days that are too wet or dry to seed (this used to be accounted for in seeding rate, but that meant a labour cost was occured for time that was too wet or dry)
    try:
        model.del_component(model.seed_period_days)
    except AttributeError:
        pass
    def seed_period_days(model,p):
        return sum(sum(model.v_seeding_machdays[p,k,l] for k in model.s_crops)for l in model.s_lmus) <= \
        model.p_seed_days[p] * pinp.mach['number_crop_gear'] * pinp.mach['seeding_occur']
    model.seed_period_days = Constraint(model.s_periods, rule=seed_period_days, doc='constrain the number of seeding days per seed period')
    
    ##constraint to limit the number of hours of harvest to the amount that can be supplied by x crop gear
    try:
        model.del_component(model.harv_hours_limit)
    except AttributeError:
        pass
    def harv_hours_limit(model, p):
        return sum(model.v_harv_hours[p, k] for k in model.s_harvcrops) <= model.p_harv_hrs_max[p] * pinp.mach['number_crop_gear']
    model.harv_hours_limit = Constraint(model.s_periods, rule=harv_hours_limit, doc='constrain the number of hours of harvest x crop gear can provide')

############
#variable  #
############    
#number of seeding days in each period on each crop and lmu
model.v_seeding_machdays = Var(model.s_periods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha of each rotation')
#number of ha seeded using contractor
model.v_contractseeding_ha = Var(model.s_periods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha contract seeding for each crop')
#number of hours harvesting for each crop - there is a constraint to limit this to the hours available in the harvest period
model.v_harv_hours = Var(model.s_periods, model.s_harvcrops, bounds=(0,None), doc='number of hours of harvesting')
#number of contract hours harvesting for each crop
model.v_contractharv_hours = Var(model.s_harvcrops, bounds=(0,None), doc='number of contract hours of harvesting')
#tonnes of hay made
model.v_hay_made = Var(bounds=(0,None), doc='tonnes of hay made')

###################################
#functions for core model         #
###################################
def sow_supply(model,k,l):
    '''
    Parameters
    ----------
    model 
    k : Set
        Crop.
    l : Set
        LMU.

    Returns
    -------
    Function for pyomo
        - determine sow supply
        - contract_seed + farmer_seed 
    '''
    return sum(model.v_contractseeding_ha[p,k,l] + (model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l])  for p in model.s_periods)
     

def ha_pasture_crop_paddocks(model,f,l):
    '''
    Returns
    -------
    Pyomo function.
        Total hectares that can be grazed on crop paddocks before harvest
    '''
    ##number of grazable pasture ha provided by contract seeding
    ha_contract= sum(sum(model.p_seeding_grazingdays[f,p] * model.v_contractseeding_ha[p,k,l] for k in model.s_crops) for p in model.s_periods)
    ##number of grazable pasture ha provided by farmer seeding
    ha_personal= sum(sum(model.p_seeding_grazingdays[f,p] * model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] for k in model.s_crops) for p in model.s_periods)
    return ha_contract + ha_personal

#function to determine late seeding penalty, this will be passed to core model
def late_seed_penalty(model,k):
    return  sum(sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] * model.p_yield_penalty[p, k] for l in model.s_lmus) for p in model.s_periods)

#function to determine late seeding stubble penalty, this will be passed to core model
def stubble_penalty(model,k):
    return  sum(sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] * model.p_yield_penalty[p, k] for l in model.s_lmus) for p in model.s_periods) \
    * model.stubble[k]

def harv_supply(model,k):
    #total harvest availability for each crop, period doesn't matter i think hence sum
    farmer_harv = sum(model.v_harv_hours[p, k] * model.p_harv_rate[p, k]  for p in model.s_periods  )
    contract_harv = model.v_contractharv_hours[k] * model.p_contractharv_rate[k] 
    return farmer_harv + contract_harv

# #make hay, this will be passed to core model
# def make_hay(model):
#     return model.v_hay_made

#function to determine seeding cost, this will be passed to core model
def seeding_cost(model,c):
    #contract cost
    contract_cost = sum(sum(sum(model.v_contractseeding_ha[p,k,l] * model.p_contract_seeding_cost[c] for l in model.s_lmus) for p in model.s_periods) for k in model.s_crops) 
    #cost per ha x number of days seeding x ha per day
    seeding_cost = sum(sum(sum(model.p_seeding_cost[c,l] * model.v_seeding_machdays[p,k,l] * model.p_seeding_rate[k,l] for l in model.s_lmus) for p in model.s_periods) for k in model.s_crops)  
    return contract_cost + seeding_cost
 
#function to determine harv cost, this will be passed to core model
def harvesting_cost(model,c):
    ##contract cost and owner cost (cost per hr x number of hours)
    return sum(sum(model.v_contractharv_hours[k] * model.p_contractharv_cost[c,k] + model.p_harv_cost[c,k] * model.v_harv_hours[p, k] for k in model.s_harvcrops) for p in model.s_periods)

#includes hay cost
def mach_cost(model,c):
    hay_cost = model.v_hay_made * model.p_contracthay_cost[c]
    return harvesting_cost(model,c) + seeding_cost(model,c) + hay_cost

#function to determine derpriciation cost, this will be passed to core model
#equals seeding dep plus harv dep plus fixed dep
def total_dep(model):
    #fixed dep = total sale value of equipment x fixed rate of dep, number of crop fear acounted for before this step
    fixed_dep = mac.fix_dep()
    #cost per ha seeding dep x number of days seeding x ha per day
    seeding_depreciation = sum(sum(sum(model.p_seeding_dep[l] * model.v_seeding_machdays[p,k,l] * model.p_seeding_rate[k,l] for l in model.s_lmus) for p in  model.s_periods) for k in model.s_crops) 
    #cost of harv dep = hourly dep x early and late harv hours 
    harv_dep= mac.harvest_dep() * sum(sum(model.v_harv_hours[p ,k] for k in model.s_harvcrops) for p in model.s_periods)
    return seeding_depreciation + fixed_dep + harv_dep 











