# -*- coding: utf-8 -*-
"""
author: young
"""
#python modules
from pyomo import environ as pe

#AFO modules
import SupFeed as sup
import PropertyInputs as pinp

def sup_precalcs(params, r_vals):
    '''
    Call crop labour precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''

    sup.f_sup_params(params,r_vals)

    
def suppyomo_local(params, model):
    ''' Builds pyomo variables, parameters'''

    ############
    # variable #
    ############
    model.v_buy_grain = pe.Var(model.s_season_types, model.s_crops, model.s_grain_pools, bounds=(0,None),
                               doc='tonnes of grain in each pool purchased for sup feeding')
    model.v_sup_con = pe.Var(model.s_season_types, model.s_crops, model.s_grain_pools, model.s_feed_pools, model.s_feed_periods,
                             bounds=(0,None), doc='tonnes of grain consumed in each pool')

    #########
    #param  #
    ######### 

    ##sup cost
    model.p_sup_cost = pe.Param(model.s_cashflow_periods, model.s_crops, model.s_feed_periods, model.s_season_types, initialize=params['total_sup_cost'], default = 0.0, mutable=True, doc='cost of storing and feeding 1t of sup each period')
    
    ##sup dep
    model.p_sup_dep = pe.Param(model.s_crops, initialize= params['storage_dep'], default = 0.0, doc='depreciation of storing 1t of sup each period')
    
    ##sup asset
    model.p_sup_asset = pe.Param(model.s_crops, initialize=params['storage_asset'], default = 0.0, doc='asset value associated with storing 1t of sup each period')
    
    ##sup labour
    model.p_sup_labour = pe.Param(model.s_labperiods, model.s_feed_periods, model.s_crops, model.s_season_types, initialize=params['sup_labour'], default = 0.0, mutable=True, doc='labour required to feed each sup in each feed period')
    
    ##sup vol
    model.p_sup_vol = pe.Param(model.s_crops, initialize=params['vol_tonne'] , default = 0.0, doc='vol per tonne of grain fed')
    
    ##sup md
    model.p_sup_md = pe.Param(model.s_crops, initialize=params['md_tonne'] , default = 0.0, doc='md per tonne of grain fed')
    
    ##price buy grain
    model.p_buy_grain_price = pe.Param(model.s_crops, model.s_cashflow_periods, model.s_grain_pools, initialize=params['buy_grain_price'], default = 0.0, doc='price to buy grain from neighbour')


#######################################################################################################################################################
#######################################################################################################################################################
#functions for core model
#######################################################################################################################################################
#######################################################################################################################################################
def sup_cost(model,c,z):
    '''
    Calculate the total cost of feeding the selected level of supplement.

    Used in global constraint (con_cashflow). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_cost[c,k,p6,z] for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods)

def sup_me(model,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of supplement.

    Used in global constraint (con_me). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_md[k]for g in model.s_grain_pools for k in model.s_crops)

def sup_vol(model,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of supplement.

    Used in global constraint (con_vol). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_vol[k] for g in model.s_grain_pools for k in model.s_crops)

def sup_dep(model,z):
    '''
    Calculate the total depreciation of silos.

    Used in global constraint (con_dep). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_dep[k] for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods)

def sup_asset(model,z):
    '''
    Calculate the total asset value of silos.

    Used in global constraint (con_asset). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_asset[k] for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods)
    
def sup_labour(model,p5,z):
    '''
    Calculate the total labour required for supplementary feeding.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    return sum(model.v_sup_con[z,k,g,f,p6] * model.p_sup_labour[p5,p6,k,z] for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    