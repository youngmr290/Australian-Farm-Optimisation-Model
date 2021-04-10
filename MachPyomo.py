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
#AFO modules
import Mach as mac
from CreateModel import *
import PropertyInputs as pinp

def mach_precalcs(params, r_vals):
    mac.f_mach_params(params, r_vals)



def machpyomo_local(params):
    ############
    #variable  #
    ############
    #number of seeding days in each period on each crop and lmu
    try:
        model.del_component(model.v_seeding_machdays)
        model.del_component(model.v_seeding_machdays_index)
    except AttributeError:
        pass
    model.v_seeding_machdays = Var(model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of days of seeding')
    #number of ha seeded for each pasture
    try:
        model.del_component(model.v_seeding_pas)
        model.del_component(model.v_seeding_pas_index)
    except AttributeError:
        pass
    model.v_seeding_pas = Var(model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha of pasture seeded')
    #number of ha seeded for each crop
    try:
        model.del_component(model.v_seeding_crop)
        model.del_component(model.v_seeding_crop_index)
    except AttributeError:
        pass
    model.v_seeding_crop = Var(model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha of crop seeded')
    #number of ha seeded using contractor
    try:
        model.del_component(model.v_contractseeding_ha)
        model.del_component(model.v_contractseeding_ha_index)
    except AttributeError:
        pass
    model.v_contractseeding_ha = Var(model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha contract seeding for each crop')
    #number of hours harvesting for each crop - there is a constraint to limit this to the hours available in the harvest period
    try:
        model.del_component(model.v_harv_hours)
        model.del_component(model.v_harv_hours_index)
    except AttributeError:
        pass
    model.v_harv_hours = Var(model.s_labperiods, model.s_harvcrops, bounds=(0,None), doc='number of hours of harvesting')
    #number of contract hours harvesting for each crop
    try:
        model.del_component(model.v_contractharv_hours)
        model.del_component(model.v_contractharv_hours_index)
    except AttributeError:
        pass
    model.v_contractharv_hours = Var(model.s_harvcrops, bounds=(0,None), doc='number of contract hours of harvesting')
    #tonnes of hay made
    try:
        model.del_component(model.v_hay_made)
    except AttributeError:
        pass
    model.v_hay_made = Var(bounds=(0,None), doc='tonnes of hay made')

    
    #########
    #param  #
    #########

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    try:
        model.del_component(model.p_seeding_rate_index)
        model.del_component(model.p_seeding_rate)
    except AttributeError:
        pass
    model.p_seeding_rate = Param(model.s_landuses, model.s_lmus, initialize=params['seed_rate'], default = 0.0, doc='rate of seeding ha/day provided by one crop gear')
    
    try:
        model.del_component(model.p_contractseeding_occur)
    except AttributeError:
        pass
    model.p_contractseeding_occur = Param(model.s_labperiods, initialize=params[season]['contractseeding_occur'], default = 0.0, mutable=True, doc='period/s when contract seeding can occur')
    
    try:
        model.del_component(model.p_seed_days)
    except AttributeError:
        pass
    model.p_seed_days = Param(model.s_labperiods, initialize=params[season]['seed_days'], default = 0.0, mutable=True, doc='number of seeding days in each period')

    try:
        model.del_component(model.p_seeding_cost_index)
        model.del_component(model.p_seeding_cost)
    except AttributeError:
        pass
    model.p_seeding_cost = Param(model.s_cashflow_periods, model.s_lmus, initialize=params[season]['seeding_cost'], default = 0.0, mutable=True, doc='cost of seeding 1ha')
    
    try:
        model.del_component(model.p_contract_seeding_cost)
    except AttributeError:
        pass
    model.p_contract_seeding_cost = Param(model.s_cashflow_periods, initialize=params[season]['contract_seed_cost'], default = 0.0, mutable=True, doc='cost of contract seeding 1ha')
    
    try:
        model.del_component(model.p_harv_rate_index)
        model.del_component(model.p_harv_rate)
    except AttributeError:
        pass
    model.p_harv_rate = Param(model.s_labperiods, model.s_crops, initialize=params[season]['harv_rate_period'], default = 0.0, mutable=True, doc='rate of harv t/hr provided by one crop gear each period')
    
    try:
        model.del_component(model.p_contractharv_rate)
    except AttributeError:
        pass
    model.p_contractharv_rate = Param(model.s_crops, initialize=params['contract_harv_rate'], default = 0.0, doc='rate of harv t/hr provided by one crop gear each period')
    
    try:
        model.del_component(model.p_harv_hrs_max)
    except AttributeError:
        pass
    model.p_harv_hrs_max = Param(model.s_labperiods, initialize= params[season]['max_harv_hours'], default = 0.0, mutable=True, doc='max hours of harvest per period')
    
    try:
        model.del_component(model.p_harv_cost_index)
        model.del_component(model.p_harv_cost)
    except AttributeError:
        pass
    model.p_harv_cost = Param(model.s_cashflow_periods, model.s_crops, initialize=params[season]['harvest_cost'], default = 0.0, mutable=True, doc='cost of harvesting 1hr')
    
    try:
        model.del_component(model.p_contractharv_cost_index)
        model.del_component(model.p_contractharv_cost)
    except AttributeError:
        pass
    model.p_contractharv_cost = Param(model.s_cashflow_periods, model.s_crops, initialize=params[season]['contract_harvest_cost'], default = 0.0, mutable=True, doc='cost of contract harvesting 1hr')
    
    try:
        model.del_component(model.p_contracthay_cost)
    except AttributeError:
        pass
    model.p_contracthay_cost = Param(model.s_cashflow_periods, initialize=params['hay_making_cost'], default = 0.0, doc='cost of contract making hay $/t')
    
    try:
        model.del_component(model.p_yield_penalty_index)
        model.del_component(model.p_yield_penalty)
    except AttributeError:
        pass
    model.p_yield_penalty = Param(model.s_labperiods, model.s_crops, initialize=params[season]['yield_penalty'], default = 0.0, mutable=True, doc='kg/ha/day penalty for late sowing in each period')
    
    try:
        model.del_component(model.p_seeding_grazingdays_index)
        model.del_component(model.p_seeding_grazingdays)
    except AttributeError:
        pass
    model.p_seeding_grazingdays = Param(model.s_feed_periods, model.s_labperiods, initialize=params[season]['grazing_days'], default = 0.0, mutable=True, doc='pasture grazing days per feed period provided by 1ha of seeding in each seed period')

    try:
        model.del_component(model.p_fixed_dep)
    except AttributeError:
        pass
    model.p_fixed_dep = Param(initialize=params['fixed_dep'],default=0.0, doc='fixed depreciation of all machinery for 1 yr')

    try:
        model.del_component(model.p_seeding_dep)
    except AttributeError:
        pass
    model.p_seeding_dep = Param(model.s_lmus,initialize=params['seeding_dep'],default=0.0,
                                doc='depreciation cost of seeding 1ha')

    try:
        model.del_component(model.p_harv_dep)
    except AttributeError:
        pass
    model.p_harv_dep = Param(initialize=params['harv_dep'],default=0.0, doc='depreciation cost of harvesting 1hr')

    try:
        model.del_component(model.p_mach_asset)
    except AttributeError:
        pass
    model.p_mach_asset = Param(initialize=params['mach_asset_value'], default = 0.0, doc='asset value associated with crop gear')

    try:
        model.del_component(model.p_mach_insurance)
    except AttributeError:
        pass
    model.p_mach_insurance = Param(model.s_cashflow_periods, initialize=params['insurance'], default = 0.0, doc='insurance paid on all machinery')
    
    try:
        model.del_component(model.p_number_seeding_gear)
    except AttributeError:
        pass
    model.p_number_seeding_gear = Param(initialize=params['number_seeding_gear'], default = 0.0, doc='number of crop gear')
    
    try:
        model.del_component(model.p_number_harv_gear)
    except AttributeError:
        pass
    model.p_number_harv_gear = Param(initialize=params['number_harv_gear'], default = 0.0, doc='number of crop gear')

    try:
        model.del_component(model.p_seeding_occur)
    except AttributeError:
        pass
    model.p_seeding_occur = Param(initialize=params['seeding_occur'], default = 0.0, doc='proportion of time seeding can occur each period')

    ###################################
    #local constraints                #
    ###################################
    ##days of seeding is limited by seed period length and the number of crop gear (ie 2 crop gear gives you double the penalty free days)
    ##constraint to limit the number of seed days in each period ()
    ##includes a factor to account for days that are too wet or dry to seed (this used to be accounted for in seeding rate, but that meant a labour cost was occured for time that was too wet or dry)
    try:
        model.del_component(model.con_seed_period_days)
    except AttributeError:
        pass
    def seed_period_days(model,p):
        return sum(sum(model.v_seeding_machdays[p,k,l] for k in model.s_crops)for l in model.s_lmus) <= \
        model.p_seed_days[p] * model.p_number_seeding_gear * model.p_seeding_occur
    model.con_seed_period_days = Constraint(model.s_labperiods, rule=seed_period_days, doc='constrain the number of seeding days per seed period')
    
    ##constraint to limit the number of hours of harvest to the amount that can be supplied by x crop gear
    try:
        model.del_component(model.con_harv_hours_limit)
    except AttributeError:
        pass
    def harv_hours_limit(model, p):
        return sum(model.v_harv_hours[p, k] for k in model.s_harvcrops) <= model.p_harv_hrs_max[p] * model.p_number_harv_gear
    model.con_harv_hours_limit = Constraint(model.s_labperiods, rule=harv_hours_limit, doc='constrain the number of hours of harvest x crop gear can provide')
    
    ##link sow supply to crop and pas variable - this has to be done because crop is not by period and pasture is
    try:
        model.del_component(model.con_sow_supply_index)
        model.del_component(model.con_sow_supply)
    except AttributeError:
        pass
    def sow_supply(model,p,k1,l):
        return -model.v_contractseeding_ha[p,k1,l] * model.p_contractseeding_occur[p] - model.p_seeding_rate[k1,l] * model.v_seeding_machdays[p,k1,l]   \
                + model.v_seeding_pas[p,k1,l] + model.v_seeding_crop[p,k1,l] <=0
    model.con_sow_supply = Constraint(model.s_labperiods, model.s_landuses, model.s_lmus, rule=sow_supply, doc='link sow supply to crop and pas variable')

###################################
#functions for core model         #
###################################   

def ha_pasture_crop_paddocks(model,f,l):
    '''
    Returns
    -------
    Pyomo function.
        Total hectares that can be grazed on crop paddocks before seeding
        *note poc is only on crop paddocks but the seeding activity includes pastures, to stop pasture paddocks providing poc only loop through the crop set
    '''
    ##number of grazable pasture ha provided by contract seeding
    ha_contract= sum(sum(model.p_seeding_grazingdays[f,p] * model.v_contractseeding_ha[p,k,l] for k in model.s_crops) for p in model.s_labperiods)
    ##number of grazable pasture ha provided by farmer seeding
    ha_personal= sum(sum(model.p_seeding_grazingdays[f,p] * model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] for k in model.s_crops) for p in model.s_labperiods)
    return ha_contract + ha_personal

#function to determine late seeding penalty, this will be passed to core model
def late_seed_penalty(model,g,k):
    return  sum(sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] * model.p_yield_penalty[p, k] for l in model.s_lmus) for p in model.s_labperiods)  \
                * model.p_grainpool_proportion[k,g]
#function to determine late seeding stubble penalty, this will be passed to core model
def stubble_penalty(model,k,s):
    return  sum(sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[p,k,l] * model.p_yield_penalty[p, k] * model.p_rot_stubble[k,s]\
                    for l in model.s_lmus) for p in model.s_labperiods if model.p_rot_stubble[k,s] !=0) \
    

def harv_supply(model,k):
    #total harvest availability for each crop, period doesn't matter i think hence sum
    farmer_harv = sum(model.v_harv_hours[p, k] * model.p_harv_rate[p, k]  for p in model.s_labperiods  )
    contract_harv = model.v_contractharv_hours[k] * model.p_contractharv_rate[k] 
    return farmer_harv + contract_harv

# #make hay, this will be passed to core model
# def make_hay(model):
#     return model.v_hay_made

#function to determine seeding cost, this will be passed to core model
def seeding_cost(model,c):
    #contract cost
    contract_cost = sum(sum(sum(model.v_contractseeding_ha[p,k1,l] * model.p_contract_seeding_cost[c] for l in model.s_lmus) for p in model.s_labperiods) for k1 in model.s_landuses) 
    #cost per ha x number of days seeding x ha per day
    seeding_cost = sum(sum(sum(model.p_seeding_cost[c,l] * model.v_seeding_machdays[p,k1,l] * model.p_seeding_rate[k1,l] for l in model.s_lmus) for p in model.s_labperiods) for k1 in model.s_landuses)  
    return contract_cost + seeding_cost
 
#function to determine harv cost, this will be passed to core model
def harvesting_cost(model,c):
    ##contract cost and owner cost (cost per hr x number of hours)
    return sum(model.v_contractharv_hours[k] * model.p_contractharv_cost[c,k] + sum(model.p_harv_cost[c,k] * model.v_harv_hours[p, k] for p in model.s_labperiods) for k in model.s_harvcrops)

#includes hay cost
def mach_cost(model,c):
    hay_cost = model.v_hay_made * model.p_contracthay_cost[c]
    return harvesting_cost(model,c) + seeding_cost(model,c) + hay_cost + model.p_mach_insurance[c]

#function to determine derpriciation cost, this will be passed to core model
#equals seeding dep plus harv dep plus fixed dep
def total_dep(model):
    #fixed dep = total sale value of equipment x fixed rate of dep, number of crop fear accounted for before this step
    fixed_dep = model.p_fixed_dep
    #cost per ha seeding dep x number of days seeding x ha per day
    seeding_depreciation = sum(sum(sum(model.p_seeding_dep[l] * model.v_seeding_machdays[p,k,l] * model.p_seeding_rate[k,l] for l in model.s_lmus) for p in  model.s_labperiods) for k in model.s_crops) 
    #cost of harv dep = hourly dep x early and late harv hours 
    harv_dep = model.p_harv_dep * sum(sum(model.v_harv_hours[p ,k] for k in model.s_harvcrops) for p in model.s_labperiods)
    return seeding_depreciation + fixed_dep + harv_dep

def mach_asset(model):
    return model.p_mach_asset









