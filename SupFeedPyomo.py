# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 08:03:17 2020

@author: young
"""
#python modules
from pyomo import environ as pe

#MUDAS modules
from CreateModel import model
import SupFeed as sup

def sup_precalcs(params, r_vals):
    ##call sup functions
    sup.sup_cost(params, r_vals)
    vol_md = sup.sup_md_vol(params)  
    sup.sup_labour(params)
    sup.buy_grain_price(params, r_vals)
    
    
def suppyomo_local(params):
    #########
    #param  #
    ######### 
    
    ##sup cost
    try:
        model.del_component(model.p_sup_cost_index)
        model.del_component(model.p_sup_cost)
    except AttributeError:
        pass
    model.p_sup_cost = pe.Param(model.s_cashflow_periods, model.s_feed_periods, model.s_crops, initialize=params['total_sup_cost'], default = 0.0, doc='cost of storing and feeding 1t of sup each period')
    
    ##sup dep
    try:
        model.del_component(model.p_sup_dep_index)
        model.del_component(model.p_sup_dep)
    except AttributeError:
        pass
    model.p_sup_dep = pe.Param(model.s_feed_periods, model.s_crops, initialize= params['storage_dep'], default = 0.0, doc='depreciation of storing 1t of sup each period')
    
    ##sup asset
    try:
        model.del_component(model.p_sup_asset_index)
        model.del_component(model.p_sup_asset)
    except AttributeError:
        pass
    model.p_sup_asset = pe.Param(model.s_feed_periods, model.s_crops, initialize=params['storage_asset'], default = 0.0, doc='asset value associated with storing 1t of sup each period')
    
    ##sup labour
    try:
        model.del_component(model.p_sup_labour_index)
        model.del_component(model.p_sup_labour)
    except AttributeError:
        pass
    model.p_sup_labour = pe.Param(model.s_crops, model.s_labperiods, model.s_feed_periods, initialize=params['sup_labour'], default = 0.0, doc='labour required to feed each sup in each feed period')
    
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
#variable
#######################################################################################################################################################
#######################################################################################################################################################
model.v_buy_grain = pe.Var(model.s_crops, model.s_grain_pools, bounds=(0,None), doc='tonnes of grain in each pool purchased for sup feeding')
model.v_sup_con = pe.Var(model.s_crops,  model.s_grain_pools, model.s_feed_pools, model.s_feed_periods, bounds=(0,None), doc='tonnes of grain consumed in each pool')

#######################################################################################################################################################
#######################################################################################################################################################
#functions for core model
#######################################################################################################################################################
#######################################################################################################################################################
def sup_cost(model,c):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_cost[c,f,k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)

def sup_me(model,v,f):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_md[k]for g in model.s_grain_pools for k in model.s_crops)

def sup_vol(model,v,f):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_vol[k] for g in model.s_grain_pools for k in model.s_crops)

def sup_dep(model):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_dep[f,k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)

def sup_asset(model):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_asset[f,k] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)
    
def sup_labour(model,p):
    return sum(model.v_sup_con[k,g,v,f] * model.p_sup_labour[k,p,f] for v in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for f in model.s_feed_periods)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    