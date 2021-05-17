# -*- coding: utf-8 -*-
"""
author: young
"""
#python modules
from pyomo import environ as pe

#AFO modules
from CreateModel import model
import SupFeed as sup
import PropertyInputs as pinp

def sup_precalcs(params, r_vals):
    '''
    Call crop labour precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''

    sup.f_sup_params(params,r_vals)

    
def suppyomo_local(params):
    ''' Builds pyomo variables, parameters'''

    ############
    # variable #
    ############
    try:
        model.del_component(model.v_buy_grain_index)
        model.del_component(model.v_buy_grain)
    except AttributeError:
        pass
    model.v_buy_grain = pe.Var(model.s_crops,model.s_grain_pools,bounds=(0,None),
                               doc='tonnes of grain in each pool purchased for sup feeding')
    try:
        model.del_component(model.v_sup_con_index)
        model.del_component(model.v_sup_con)
    except AttributeError:
        pass
    model.v_sup_con = pe.Var(model.s_crops,model.s_grain_pools,model.s_feed_pools,model.s_feed_periods,bounds=(0,None),
                             doc='tonnes of grain consumed in each pool')

    #########
    #param  #
    ######### 

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    ##sup cost
    try:
        model.del_component(model.p_sup_cost_index)
        model.del_component(model.p_sup_cost)
    except AttributeError:
        pass
    model.p_sup_cost = pe.Param(model.s_cashflow_periods, model.s_crops, model.s_feed_periods, initialize=params[season]['total_sup_cost'], default = 0.0, mutable=True, doc='cost of storing and feeding 1t of sup each period')
    
    ##sup dep
    try:
        # model.del_component(model.p_sup_dep_index)
        model.del_component(model.p_sup_dep)
    except AttributeError:
        pass
    model.p_sup_dep = pe.Param(model.s_crops, initialize= params['storage_dep'], default = 0.0, doc='depreciation of storing 1t of sup each period')
    
    ##sup asset
    try:
        # model.del_component(model.p_sup_asset_index)
        model.del_component(model.p_sup_asset)
    except AttributeError:
        pass
    model.p_sup_asset = pe.Param(model.s_crops, initialize=params['storage_asset'], default = 0.0, doc='asset value associated with storing 1t of sup each period')
    
    ##sup labour
    try:
        model.del_component(model.p_sup_labour_index)
        model.del_component(model.p_sup_labour)
    except AttributeError:
        pass
    model.p_sup_labour = pe.Param(model.s_labperiods, model.s_feed_periods, model.s_crops, initialize=params[season]['sup_labour'], default = 0.0, mutable=True, doc='labour required to feed each sup in each feed period')
    
    ##sup vol
    try:
        model.del_component(model.p_sup_vol)
    except AttributeError:
        pass
    model.p_sup_vol = pe.Param(model.s_crops, initialize=params['vol_tonne'] , default = 0.0, doc='vol per tonne of grain fed')
    
    ##sup md
    try:
        model.del_component(model.p_sup_md)
    except AttributeError:
        pass
    model.p_sup_md = pe.Param(model.s_crops, initialize=params['md_tonne'] , default = 0.0, doc='md per tonne of grain fed')
    
    ##price buy grain
    try:
        model.del_component(model.p_buy_grain_price_index)
        model.del_component(model.p_buy_grain_price)
    except AttributeError:
        pass
    model.p_buy_grain_price = pe.Param(model.s_crops, model.s_cashflow_periods, model.s_grain_pools, initialize=params['buy_grain_price'], default = 0.0, doc='price to buy grain from neighbour')


#######################################################################################################################################################
#######################################################################################################################################################
#functions for core model
#######################################################################################################################################################
#######################################################################################################################################################
def sup_cost(model,c):
    '''
    Calculate the total cost of feeding the selected level of supplement.

    Used in global constraint (con_cashflow). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_cost[c,k,f] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)

def sup_me(model,v,f):
    '''
    Calculate the total energy provided to each ev pool by feed the selected amount of supplement.

    Used in global constraint (con_me). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_md[k]for g in model.s_grain_pools for k in model.s_crops)

def sup_vol(model,v,f):
    '''
    Calculate the total volume required by each ev pool to feed the selected amount of supplement.

    Used in global constraint (con_vol). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_vol[k] for g in model.s_grain_pools for k in model.s_crops)

def sup_dep(model):
    '''
    Calculate the total depreciation of silos.

    Used in global constraint (con_dep). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_dep[k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)

def sup_asset(model):
    '''
    Calculate the total asset value of silos.

    Used in global constraint (con_asset). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_asset[k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)
    
def sup_labour(model,p):
    '''
    Calculate the total labour required for supplementary feeding.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_labour[p,f,k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    