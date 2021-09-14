# -*- coding: utf-8 -*-
"""

author: young

"""

#python modules
import pyomo.environ as pe
import sys
#AFO modules
import Mach as mac
import PropertyInputs as pinp

def mach_precalcs(params, r_vals):
    mac.f_mach_params(params, r_vals)



def f1_machpyomo_local(params, model):
    ############
    #variable  #
    ############
    #number of seeding days in each period on each crop and lmu
    model.v_seeding_machdays = pe.Var(model.s_season_types, model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of days of seeding')
    #number of ha seeded for each pasture
    model.v_seeding_pas = pe.Var(model.s_labperiods, model.s_landuses, model.s_lmus, model.s_season_types, bounds=(0,None), doc='number of ha of pasture seeded')
    #number of ha seeded for each crop
    model.v_wet_seeding_crop = pe.Var(model.s_labperiods, model.s_landuses, model.s_lmus, model.s_season_types, bounds=(0,None), doc='number of ha of crop wet sown')
    #number of ha seeded for each crop
    model.v_dry_seeding_crop = pe.Var(model.s_labperiods, model.s_landuses, model.s_lmus, model.s_season_types, bounds=(0,None), doc='number of ha of crop dry sown')
    #number of ha seeded using contractor
    model.v_contractseeding_ha = pe.Var(model.s_season_types, model.s_labperiods, model.s_landuses, model.s_lmus, bounds=(0,None), doc='number of ha contract seeding for each crop')
    #number of hours harvesting for each crop - there is a constraint to limit this to the hours available in the harvest period
    model.v_harv_hours = pe.Var(model.s_season_types, model.s_labperiods, model.s_harvcrops, bounds=(0,None), doc='number of hours of harvesting')
    #number of contract hours harvesting for each crop
    model.v_contractharv_hours = pe.Var(model.s_season_types, model.s_harvcrops, bounds=(0,None), doc='number of contract hours of harvesting')
    #tonnes of hay made
    model.v_hay_made = pe.Var(model.s_season_types, bounds=(0,None), doc='tonnes of hay made')

    
    #########
    #param  #
    #########
    model.p_seeding_rate = pe.Param(model.s_landuses, model.s_lmus, initialize=params['seed_rate'], default = 0.0, doc='rate of seeding ha/day provided by one crop gear')
    
    model.p_contractseeding_occur = pe.Param(model.s_labperiods, model.s_season_types, initialize=params['contractseeding_occur'], default = 0.0, mutable=False, doc='period/s when contract seeding can occur')
    
    model.p_seed_days = pe.Param(model.s_labperiods, model.s_season_types, initialize=params['seed_days'], default = 0.0, mutable=False, doc='number of seeding days in each period')

    model.p_seeding_cost = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, model.s_lmus, initialize=params['seeding_cost'], default = 0.0, mutable=False, doc='cost of seeding 1ha')
    
    model.p_contract_seeding_cost = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, initialize=params['contract_seed_cost'], default = 0.0, mutable=False, doc='cost of contract seeding 1ha')
    
    model.p_harv_rate = pe.Param(model.s_labperiods, model.s_crops, model.s_season_types, initialize=params['harv_rate_period'], default = 0.0, mutable=False, doc='rate of harv t/hr provided by one crop gear each period')
    
    model.p_contractharv_rate = pe.Param(model.s_crops, initialize=params['contract_harv_rate'], default = 0.0, doc='rate of harv t/hr provided by contractor')
    
    model.p_harv_hrs_max = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['max_harv_hours'], default = 0.0, mutable=False, doc='max hours of harvest per period')
    
    model.p_harv_cost = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, model.s_crops, initialize=params['harvest_cost'], default = 0.0, mutable=False, doc='cost of harvesting 1hr')
    
    model.p_contractharv_cost = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, model.s_crops, initialize=params['contract_harvest_cost'], default = 0.0, mutable=False, doc='cost of contract harvesting 1hr')
    
    model.p_contracthay_cost = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, initialize=params['hay_making_cost'], default = 0.0, doc='cost of contract making hay $/t')
    
    model.p_yield_penalty = pe.Param(model.s_labperiods, model.s_crops, model.s_season_types, initialize=params['yield_penalty'], default = 0.0, mutable=False, doc='kg/ha/day penalty for late sowing in each period')
    
    model.p_stubble_penalty = pe.Param(model.s_labperiods, model.s_crops, model.s_season_types, initialize=params['stubble_penalty'], default = 0.0, mutable=False, doc='kg/ha/day penalty for late sowing in each period')

    model.p_poc_grazingdays = pe.Param(model.s_feed_periods, model.s_labperiods, model.s_season_types, initialize=params['poc_grazing_days'], default = 0.0, mutable=False, doc='pasture grazing days per feed period provided by 1ha of seeding in each seed period')

    model.p_fixed_dep = pe.Param(initialize=params['fixed_dep'],default=0.0, doc='fixed depreciation of all machinery for 1 yr')

    model.p_seeding_dep = pe.Param(model.s_lmus,initialize=params['seeding_dep'],default=0.0,
                                doc='depreciation cost of seeding 1ha')

    model.p_harv_dep = pe.Param(initialize=params['harv_dep'],default=0.0, doc='depreciation cost of harvesting 1hr')

    model.p_mach_asset = pe.Param(initialize=params['mach_asset_value'], default = 0.0, doc='asset value associated with crop gear')

    model.p_mach_insurance = pe.Param(model.s_enterprises, model.s_cashflow_periods, model.s_season_types, initialize=params['insurance'], default = 0.0, doc='insurance paid on all machinery')
    
    model.p_number_seeding_gear = pe.Param(initialize=params['number_seeding_gear'], default = 0.0, doc='number of crop gear')
    
    model.p_number_harv_gear = pe.Param(initialize=params['number_harv_gear'], default = 0.0, doc='number of harvest gear')

    model.p_seeding_occur = pe.Param(initialize=params['seeding_occur'], default = 0.0, doc='proportion of time seeding can occur each period')

    ###################################
    #call local constraints           #
    ###################################
    f_con_seed_period_days(model)
    f_con_harv_hours_limit(model)
    f_con_sow_supply(model)



def f_con_seed_period_days(model):
    '''
    Constraint which acts to bound the variable that is the number of days seeding for each crop on each LMU
    in each machinery period.

    The number of days of seeding is limited by the length of each machinery period and the number
    of crop gear (ie two seeders allows you to seed twice as much). The constraint includes a factor
    to account for days that are too wet or dry to seed.
    '''
    def seed_period_days(model,p,z):
        return sum(sum(model.v_seeding_machdays[z,p,k,l] for k in model.s_crops)for l in model.s_lmus) <= \
        model.p_seed_days[p,z] * model.p_number_seeding_gear * model.p_seeding_occur
    model.con_seed_period_days = pe.Constraint(model.s_labperiods, model.s_season_types, rule=seed_period_days, doc='constrain the number of seeding days per seed period')

def f_con_harv_hours_limit(model):
    '''
    Constraint which acts to bound the variable that is the hours of harvesting for each crop on each LMU
    in each machinery period.

    The number of hours of harvest is limited by the max harvest hours in each harvest period and the number
    of harvest gear (ie two harvesters allows you to harvest twice as much).
    '''
    def harv_hours_limit(model, p, z):
        return sum(model.v_harv_hours[z,p,k] for k in model.s_harvcrops) <= model.p_harv_hrs_max[p,z] * model.p_number_harv_gear
    model.con_harv_hours_limit = pe.Constraint(model.s_labperiods, model.s_season_types, rule=harv_hours_limit, doc='constrain the number of hours of harvest x crop gear can provide')

def f_con_sow_supply(model):
    '''
    Constraint between the hectares sown and the supply.

    The hectares of pasture and crop seeded must be less than the amount supplied by either farmers machinery or
    contract services. The amount supplied from the farmers equipment is limited by the seeding days and the
    rate of seeding per day.
    '''
    def sow_supply(model,p5,k,l,z):
        return - model.v_contractseeding_ha[z,p5,k,l] * model.p_contractseeding_occur[p5,z] \
               - model.v_seeding_machdays[z,p5,k,l] * model.p_seeding_rate[k,l]    \
               + model.v_seeding_pas[p5,k,l,z]        \
               + model.v_wet_seeding_crop[p5,k,l,z]   \
               + model.v_dry_seeding_crop[p5,k,l,z]  <=0
    model.con_sow_supply = pe.Constraint(model.s_labperiods, model.s_landuses, model.s_lmus, model.s_season_types, rule=sow_supply, doc='link sow supply to crop and pas variable')

###################################
#functions for core model         #
###################################   

def f_ha_days_pasture_crop_paddocks(model,f,l,z):
    '''
    Calculate the total hectare grazing days that can be grazed on crop paddocks before seeding based on the
    seeding activities selected.

    Used in global constraint (con_poc_available). See CorePyomo

    Note: poc is only on crop paddocks but the seeding activity includes pastures, to stop pasture paddocks providing poc only loop through the crop set
    '''

    ##number of grazable pasture ha provided by contract seeding
    ha_days_contract= sum(sum(model.p_poc_grazingdays[f,p,z] * model.v_contractseeding_ha[z,p,k,l] for k in model.s_crops) for p in model.s_labperiods)
    ##number of grazable pasture ha provided by farmer seeding
    ha_days_personal= sum(sum(model.p_poc_grazingdays[f,p,z] * model.p_seeding_rate[k,l] * model.v_seeding_machdays[z,p,k,l] for k in model.s_crops) for p in model.s_labperiods)
    return ha_days_contract + ha_days_personal

#function to determine late seeding penalty, this will be passed to core model
def f_late_seed_penalty(model,g,k,z):
    '''
    Calculate the yield penalty (kg) based on the timeliness of the selected contract and farmer seeding activities.

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''

    farmer_penalty = sum(sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[z,p,k,l] * model.p_yield_penalty[p,k,z]
                             for l in model.s_lmus) for p in model.s_labperiods)  * model.p_grainpool_proportion[k,g]

    contract_penalty = sum(sum(model.v_contractseeding_ha[z,p,k,l] * model.p_yield_penalty[p, k, z] for l in model.s_lmus)
                           for p in model.s_labperiods) * model.p_grainpool_proportion[k,g]

    return farmer_penalty + contract_penalty

#function to determine late seeding stubble penalty, this will be passed to core model
def f_stubble_penalty(model,k,z):
    '''
    Calculate the stubble production penalty (kg) based on the timeliness of the selected contract and farmer seeding activities.

    Used in global constraint (con_stubble_a). See CorePyomo
    '''
    farmer_penalty = sum(model.p_seeding_rate[k,l] * model.v_seeding_machdays[z,p,k,l] * model.p_stubble_penalty[p,k,z]
                         for l in model.s_lmus for p in model.s_labperiods
                         if pe.value(model.p_stubble_penalty[p,k,z]) != 0)

    contract_penalty = sum(model.v_contractseeding_ha[z,p,k,l] * model.p_stubble_penalty[p,k,z]
                           for l in model.s_lmus for p in model.s_labperiods
                           if pe.value(model.p_stubble_penalty[p,k,z]) != 0)

    return farmer_penalty + contract_penalty
    

def f_harv_supply(model,k,z):
    '''
    Calculate the total hectares of each crop that can be harvested based on the allocation of harvesting
    time.

    Used in global constraint (con_harv). See CorePyomo
    '''

    #total harvest availability for each crop, period doesn't matter i think hence sum
    farmer_harv = sum(model.v_harv_hours[z,p,k] * model.p_harv_rate[p,k,z]  for p in model.s_labperiods  )
    contract_harv = model.v_contractharv_hours[z,k] * model.p_contractharv_rate[k]
    return farmer_harv + contract_harv

#function to determine seeding cost, this will be passed to core model
def f1_seeding_cost(model,c0,p7,z):
    #contract cost
    contract_cost = sum(sum(sum(model.v_contractseeding_ha[z,p,k1,l] * model.p_contract_seeding_cost[c0,p7,z] for l in model.s_lmus) for p in model.s_labperiods) for k1 in model.s_landuses)
    #cost per ha x number of days seeding x ha per day
    seeding_cost = sum(sum(sum(model.p_seeding_cost[c0,p7,z,l] * model.v_seeding_machdays[z,p,k1,l] * model.p_seeding_rate[k1,l] for l in model.s_lmus) for p in model.s_labperiods) for k1 in model.s_landuses)
    return contract_cost + seeding_cost
 
#function to determine harv cost, this will be passed to core model
def f1_harvesting_cost(model,c0,p7,z):
    ##contract cost and owner cost (cost per hr x number of hours)
    return sum(model.v_contractharv_hours[z,k] * model.p_contractharv_cost[c0,p7,z,k] + sum(model.p_harv_cost[c0,p7,z,k] * model.v_harv_hours[z, p, k] for p in model.s_labperiods) for k in model.s_harvcrops)

#includes hay cost
def f_mach_cost(model,c0,p7,z):
    '''
    Calculate the cost of machinery for insurance, seeding, harvesting and making hay based on the level
    of machinery activities selected.

    Used in global constraint (con_cashflow). See CorePyomo
    '''

    hay_cost = model.v_hay_made[z] * model.p_contracthay_cost[c0,p7,z]
    return f1_harvesting_cost(model,c0,p7,z) + f1_seeding_cost(model,c0,p7,z) + hay_cost + model.p_mach_insurance[c0,p7,z]

#function to determine derpriciation cost, this will be passed to core model
#equals seeding dep plus harv dep plus fixed dep
def f_total_dep(model,z):
    '''
    Calculate the total depreciation of farm machinery.

    Used in global constraint (con_dep). See CorePyomo
    '''

    #fixed dep = total sale value of equipment x fixed rate of dep, number of crop fear accounted for before this step
    fixed_dep = model.p_fixed_dep
    #cost per ha seeding dep x number of days seeding x ha per day
    seeding_depreciation = sum(sum(sum(model.p_seeding_dep[l] * model.v_seeding_machdays[z,p,k,l] * model.p_seeding_rate[k,l] for l in model.s_lmus) for p in  model.s_labperiods) for k in model.s_crops)
    #cost of harv dep = hourly dep x early and late harv hours 
    harv_dep = model.p_harv_dep * sum(sum(model.v_harv_hours[z,p,k] for k in model.s_harvcrops) for p in model.s_labperiods)
    return seeding_depreciation + fixed_dep + harv_dep

def f_mach_asset(model):
    '''
    Calculate the total asset value of farm machinery.

    Used in global constraint (con_asset). See CorePyomo
    '''

    return model.p_mach_asset









