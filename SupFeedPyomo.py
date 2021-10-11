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

    
def f1_suppyomo_local(params, model):
    ''' Builds pyomo variables, parameters'''

    ############
    # variable #
    ############
    model.v_buy_grain = pe.Var(model.s_sequence_year, model.s_sequence, model.s_phase_periods, model.s_season_types, model.s_crops, model.s_grain_pools, bounds=(0,None),
                               doc='tonnes of grain in each pool purchased for sup feeding')
    model.v_sup_con = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_crops, model.s_grain_pools, model.s_feed_pools, model.s_feed_periods,
                             bounds=(0,None), doc='tonnes of grain consumed in each pool')

    #########
    #param  #
    ######### 

    ##sup cost
    model.p_sup_cost = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_feed_periods, model.s_crops, initialize=params['total_sup_cost'], default = 0.0, mutable=True, doc='cost of storing and feeding 1t of sup each period')
    
    ##sup wc
    model.p_sup_wc = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_feed_periods, model.s_crops, initialize=params['total_sup_wc'], default = 0.0, mutable=True, doc='wc of storing and feeding 1t of sup each period')
    
    ##sup dep
    model.p_sup_dep = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_season_types, model.s_crops,
                               initialize= params['storage_dep'], default = 0.0, doc='depreciation of storing 1t of sup each period')
    
    ##sup asset
    model.p_sup_asset = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_season_types, model.s_crops,
                                 initialize=params['storage_asset'], default = 0.0, doc='asset value associated with storing 1t of sup each period')
    
    ##sup labour
    model.p_sup_labour = pe.Param(model.s_labperiods, model.s_feed_periods, model.s_crops, model.s_season_types, initialize=params['sup_labour'], default = 0.0, mutable=True, doc='labour required to feed each sup in each feed period')
    
    ##sup vol
    model.p_sup_vol = pe.Param(model.s_crops, initialize=params['vol_tonne'] , default = 0.0, doc='vol per tonne of grain fed')
    
    ##sup md
    model.p_sup_md = pe.Param(model.s_crops, initialize=params['md_tonne'] , default = 0.0, doc='md per tonne of grain fed')
    
    ##price buy grain
    model.p_buy_grain_price = pe.Param(model.s_phase_periods, model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_grain_pools, model.s_crops, initialize=params['buy_grain_price'], default = 0.0, doc='price to buy grain from neighbour')

    ##wc buy grain
    model.p_buy_grain_wc = pe.Param(model.s_phase_periods, model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_grain_pools, model.s_crops, initialize=params['buy_grain_wc'], default = 0.0, doc='wc to buy grain from neighbour')

    ##buy_grain_prov_mz
    model.p_buy_grain_prov = pe.Param(model.s_phase_periods, model.s_season_types, initialize=params['buy_grain_prov_mz'], default = 0.0, doc='phase periods when buying grain provides into grain transfer (this param exists so that grain is only provided when it is purchased - otherwise it could provide grain in a peirod when it didnt pay eg get free grain)')

    ##a_p6_m
    model.p_a_p6_m = pe.Param(model.s_phase_periods, model.s_feed_periods, model.s_season_types, initialize=params['a_p6_m'], default = 0.0, doc='link between p6 and m')


#######################################################################################################################################################
#######################################################################################################################################################
#functions for core model
#######################################################################################################################################################
#######################################################################################################################################################
def f_sup_cost(model,q,s,c0,p7,z):
    '''
    Calculate the total cost of feeding the selected level of supplement.

    Used in global constraint (con_cashflow). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_cost[c0,p7,z,p6,k]
               for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods
               if pe.value(model.p_sup_cost[c0,p7,z,p6,k])!=0)

def f_sup_wc(model,q,s,c0,p7,z):
    '''
    Calculate the total wc of feeding the selected level of supplement.

    Used in global constraint (con_workingcap). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_wc[c0,p7,z,p6,k]
               for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods
               if pe.value(model.p_sup_wc[c0,p7,z,p6,k])!=0)

def f_sup_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of supplement.

    Used in global constraint (con_me). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_md[k] for g in model.s_grain_pools for k in model.s_crops)

def f_sup_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of supplement.

    Used in global constraint (con_vol). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_vol[k] for g in model.s_grain_pools for k in model.s_crops)

def f_sup_dep(model,q,s,p7,z):
    '''
    Calculate the total depreciation of silos.

    Used in global constraint (con_dep). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_dep[p7,p6,z,k]
               for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods
               if pe.value(model.p_sup_dep[p7,p6,z,k])!=0)

def f_sup_asset(model,q,s,p7,z):
    '''
    Calculate the total asset value of silos.

    Used in global constraint (con_asset). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_asset[p7,p6,z,k]
               for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods
               if pe.value(model.p_sup_asset[p7,p6,z,k])!=0)
    
def f_sup_labour(model,q,s,p5,z):
    '''
    Calculate the total labour required for supplementary feeding.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    return sum(model.v_sup_con[q,s,z,k,g,f,p6] * model.p_sup_labour[p5,p6,k,z]
               for f in model.s_feed_pools for g in model.s_grain_pools for k in model.s_crops for p6 in model.s_feed_periods
               if pe.value(model.p_sup_labour[p5,p6,k,z])!=0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    