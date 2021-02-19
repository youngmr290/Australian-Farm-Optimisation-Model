# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:35 2020

@author: John
"""

#python modules
import pyomo.environ as pe
import time
import numpy as np

#AFO modules
from CreateModel import model
import StockGenerator as sgen


def stock_precalcs(params, r_vals, ev):
    sgen.generator(params, r_vals, ev)



def stockpyomo_local(params):

    ##these sets require info from the stock module
    try:
        model.del_component(model.s_wean_times)
    except AttributeError:
        pass
    model.s_wean_times = pe.Set(initialize=params['a_idx'], doc='weaning options')
    try:
        model.del_component(model.s_tol)
    except AttributeError:
        pass
    model.s_tol = pe.Set(initialize=params['i_idx'], doc='birth groups (times of lambing)')
    try:
        model.del_component(model.s_sire_periods)
    except AttributeError:
        pass
    model.s_sire_periods = pe.Set(initialize=params['p8_idx'], doc='sire periods')
    try:
        model.del_component(model.s_groups_sire)
    except AttributeError:
        pass
    model.s_groups_sire = pe.Set(initialize=params['g_idx_sire'], doc='genotype groups of sires')
    try:
        model.del_component(model.s_gen_merit_sire)
    except AttributeError:
        pass
    model.s_gen_merit_sire = pe.Set(initialize=params['y_idx_sire'], doc='genetic merit of sires')
    try:
        model.del_component(model.s_k2_birth_dams)
    except AttributeError:
        pass
    model.s_k2_birth_dams = pe.Set(initialize=params['k2_idx_dams'], doc='Cluster for LSLN & oestrus cycle based on scanning, global & weaning management')
    try:
        model.del_component(model.s_dvp_dams)
    except AttributeError:
        pass
    model.s_dvp_dams = pe.Set(ordered=True, initialize=params['dvp_idx_dams'], doc='Decision variable periods for dams') #ordered so they can be indexed in constraint to determine previous period
    try:
        model.del_component(model.s_lw_dams)
    except AttributeError:
        pass
    model.s_lw_dams = pe.Set(initialize=params['w_idx_dams'], doc='Standard LW patterns dams')
    try:
        model.del_component(model.s_groups_dams)
    except AttributeError:
        pass
    model.s_groups_dams = pe.Set(initialize=params['g_idx_dams'], doc='genotype groups of dams')
    try:
        model.del_component(model.s_groups_prog)
    except AttributeError:
        pass
    model.s_groups_prog = pe.Set(initialize=params['g_idx_dams'], doc='genotype groups of prog') #same as dams and offs
    try:
        model.del_component(model.s_gen_merit_dams)
    except AttributeError:
        pass
    model.s_gen_merit_dams = pe.Set(initialize=params['y_idx_dams'], doc='genetic merit of dams')
    try:
        model.del_component(model.s_sale_dams)
    except AttributeError:
        pass
    model.s_sale_dams = pe.Set(initialize=params['t_idx_dams'], doc='Sales within the year for dams')
    try:
        model.del_component(model.s_dvp_offs)
    except AttributeError:
        pass
    model.s_dvp_offs = pe.Set(ordered=True, initialize=params['dvp_idx_offs'], doc='Decision variable periods for offs') #ordered so they can be indexed in constraint to determine previous period
    try:
        model.del_component(model.s_damage)
    except AttributeError:
        pass
    model.s_damage = pe.Set(initialize=params['d_idx'], doc='age of mother - offs')
    try:
        model.del_component(model.s_k3_damage_offs)
    except AttributeError:
        pass
    model.s_k3_damage_offs = pe.Set(initialize=params['k3_idx_offs'], doc='age of mother - offs')
    try:
        model.del_component(model.s_k5_birth_offs)
    except AttributeError:
        pass
    model.s_k5_birth_offs = pe.Set(initialize=params['k5_idx_offs'], doc='Cluster for BTRT & oestrus cycle based on scanning, global & weaning management')
    try:
        model.del_component(model.s_groups_offs)
    except AttributeError:
        pass
    model.s_groups_offs = pe.Set(initialize=params['g_idx_offs'], doc='genotype groups of offs')
    try:
        model.del_component(model.s_gen_merit_offs)
    except AttributeError:
        pass
    model.s_gen_merit_offs = pe.Set(initialize=params['y_idx_offs'], doc='genetic merit of offs')
    try:
        model.del_component(model.s_gender)
    except AttributeError:
        pass
    model.s_gender = pe.Set(initialize=params['x_idx_offs'], doc='gender of offs')

    #####################
    ##  setup variables # #variables that use dynamic sets must be defined each iteration of exp
    #####################
    print('set up variables')
    ##animals
    try:
        model.del_component(model.v_sire)
    except AttributeError:
        pass
    model.v_sire = pe.Var(model.s_groups_sire, bounds = (0,None) , doc='number of sire animals')
    try:
        model.del_component(model.v_dams_index)
        model.del_component(model.v_dams)
    except AttributeError:
        pass
    model.v_dams = pe.Var(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, bounds = (0,None) , doc='number of dam animals')
    try:
        model.del_component(model.v_offs_index)
        model.del_component(model.v_offs)
    except AttributeError:
        pass
    model.v_offs = pe.Var(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types,
                          model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs,
                          model.s_groups_offs, bounds = (0,None) , doc='number of offs animals')
    try:
        model.del_component(model.v_prog_index)
        model.del_component(model.v_prog)
    except AttributeError:
        pass
    model.v_prog = pe.Var(model.s_k5_birth_offs, model.s_sale_prog, model.s_lw_prog, model.s_season_types,
                          model.s_tol, model.s_damage, model.s_wean_times, model.s_gender,
                          model.s_groups_prog, bounds = (0,None) , doc='number of offs animals')

    ##purchases
    try:
        model.del_component(model.v_purchase_dams_index)
        model.del_component(model.v_purchase_dams)
    except AttributeError:
        pass
    model.v_purchase_dams = pe.Var(model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, bounds = (0,None) , doc='number of purchased dam animals')
    try:
        model.del_component(model.v_purchase_offs_index)
        model.del_component(model.v_purchase_offs)
    except AttributeError:
        pass
    model.v_purchase_offs = pe.Var(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, bounds = (0,None) , doc='number of purchased offs animals')


    ######################
    ### setup parameters #
    ######################
    print('set up params')
    param_start = time.time()

    ##nsire - mating
    try:
        model.del_component(model.p_nsires_req_index)
        model.del_component(model.p_nsires_req)
    except AttributeError:
        pass
    model.p_nsires_req = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                               model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_sire, model.s_sire_periods,
                               initialize=params['p_nsire_req_dams'], default=0.0, doc='requirement for sires for mating')

    try:
        model.del_component(model.p_nsires_prov_index)
        model.del_component(model.p_nsires_prov)
    except AttributeError:
        pass
    model.p_nsires_prov = pe.Param(model.s_groups_sire, model.s_sire_periods,
                               initialize=params['p_nsire_prov_sire'], default=0.0, doc='sires available for mating')

    ##progeny
    try:
        model.del_component(model.p_npw_index)
        model.del_component(model.p_npw)
    except AttributeError:
        pass
    model.p_npw = pe.Param(model.s_k5_birth_offs, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                              model.s_season_types, model.s_tol, model.s_damage, model.s_gender, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_prog, model.s_tol,
                              initialize=params['p_npw_dams'], default=0.0, doc='number of progeny weaned')
    try:
        model.del_component(model.p_npw_req_index)
        model.del_component(model.p_npw_req)
    except AttributeError:
        pass
    model.p_npw_req = pe.Param(model.s_sale_prog, model.s_damage, model.s_gender, model.s_groups_prog,
                              initialize=params['p_npw_req_prog'], default=0.0, doc='number of yatf required by the prog activity')
    try:
        model.del_component(model.p_progprov_dams_index)
        model.del_component(model.p_progprov_dams)
    except AttributeError:
        pass
    model.p_progprov_dams = pe.Param(model.s_k3_damage_offs, model.s_sale_prog, model.s_lw_prog,model.s_season_types, model.s_tol,
                              model.s_damage, model.s_wean_times, model.s_gender, model.s_gen_merit_dams, model.s_groups_prog, model.s_groups_dams, model.s_lw_dams,
                              initialize=params['p_progprov_dams'], default=0.0, doc='number of progeny provided to dams')
    try:
        model.del_component(model.p_progreq_dams_index)
        model.del_component(model.p_progreq_dams)
    except AttributeError:
        pass
    model.p_progreq_dams = pe.Param(model.s_k2_birth_dams, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_dams, model.s_lw_dams,
                              model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                              initialize=params['p_progreq_dams'], default=0.0, doc='number of progeny required by dams')
    try:
        model.del_component(model.p_progprov_offs_index)
        model.del_component(model.p_progprov_offs)
    except AttributeError:
        pass
    model.p_progprov_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_prog, model.s_lw_prog, model.s_season_types,
                              model.s_tol, model.s_damage, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, model.s_lw_offs,
                              initialize=params['p_progprov_offs'], default=0.0, doc='number of progeny provided to dams')
    try:
        model.del_component(model.p_progreq_offs_index)
        model.del_component(model.p_progreq_offs)
    except AttributeError:
        pass
    model.p_progreq_offs = pe.Param(model.s_dvp_offs, model.s_lw_offs, model.s_tol, model.s_gender, model.s_lw_offs,
                              initialize=params['p_progreq_offs'], default=0.0, doc='number of progeny required by dams')


    ##stock - dams
    try:
        model.del_component(model.p_numbers_prov_dams_index)
        model.del_component(model.p_numbers_prov_dams)
    except AttributeError:
        pass
    model.p_numbers_prov_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_prov_dams'], default=0.0, doc='numbers provided by each dam activity into the next period')

    try:
        model.del_component(model.p_numbers_provthis_dams_index)
        model.del_component(model.p_numbers_provthis_dams)
    except AttributeError:
        pass
    model.p_numbers_provthis_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams,
                                         model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                         model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_provthis_dams'], default=0.0,
                                         doc='numbers provided by each dam transfer activity into this period')

    try:
        model.del_component(model.p_numbers_req_dams_index)
        model.del_component(model.p_numbers_req_dams)
    except AttributeError:
        pass
    model.p_numbers_req_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_req_dams'], default=0.0, doc='numbers required by each dam activity in the current period')

    ##stock - offs
    try:
        model.del_component(model.p_numbers_prov_offs_index)
        model.del_component(model.p_numbers_prov_offs)
    except AttributeError:
        pass
    model.p_numbers_prov_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol,
                                 model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, model.s_lw_offs,
                                 initialize=params['p_numbers_prov_offs'], default=0.0, doc='numbers provided into the current period from the previous periods activities')
    try:
        model.del_component(model.p_numbers_req_offs_index)
        model.del_component(model.p_numbers_req_offs)
    except AttributeError:
        pass
    model.p_numbers_req_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_offs, model.s_lw_offs,
                                        model.s_tol, model.s_gender, model.s_groups_offs, model.s_lw_offs,
                                        initialize=params['p_numbers_req_offs'], default=0.0, doc='requirement of off in the current period')

    ##energy intake
    try:
        model.del_component(model.p_mei_sire_index)
        model.del_component(model.p_mei_sire)
    except AttributeError:
        pass
    model.p_mei_sire = pe.Param(model.s_feed_periods, model.s_feed_pools, model.s_groups_sire, initialize=params['p_mei_sire'],
                                  default=0.0, doc='energy requirement sire')
    try:
        model.del_component(model.p_mei_dams_index)
        model.del_component(model.p_mei_dams)
    except AttributeError:
        pass
    model.p_mei_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_feed_pools, model.s_sale_dams,
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_mei_dams'], default=0.0, doc='energy requirement dams')
    try:
        model.del_component(model.p_mei_offs_index)
        model.del_component(model.p_mei_offs)
    except AttributeError:
        pass
    model.p_mei_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_feed_pools, model.s_sale_offs,
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times, 
                               model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_mei_offs'], default=0.0, doc='energy requirement offs')

    ##potential intake
    try:
        model.del_component(model.p_pi_sire_index)
        model.del_component(model.p_pi_sire)
    except AttributeError:
        pass
    model.p_pi_sire = pe.Param(model.s_feed_periods, model.s_feed_pools, model.s_groups_sire, initialize=params['p_pi_sire'],
                                  default=0.0, doc='pi sire')
    try:
        model.del_component(model.p_pi_dams_index)
        model.del_component(model.p_pi_dams)
    except AttributeError:
        pass
    model.p_pi_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_feed_pools, model.s_sale_dams,
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_pi_dams'], default=0.0, doc='pi dams')
    try:
        model.del_component(model.p_pi_offs_index)
        model.del_component(model.p_pi_offs)
    except AttributeError:
        pass
    model.p_pi_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_feed_pools, model.s_sale_offs,
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times, 
                               model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_pi_offs'], default=0.0, doc='pi offs')

    
    ##cashflow
    try:
        model.del_component(model.p_cashflow_sire_index)
        model.del_component(model.p_cashflow_sire)
    except AttributeError:
        pass
    model.p_cashflow_sire = pe.Param(model.s_cashflow_periods, model.s_groups_sire, initialize=params['p_cashflow_sire'], 
                                  default=0.0, doc='cashflow sire')
    try:
        model.del_component(model.p_cashflow_dams_index)
        model.del_component(model.p_cashflow_dams)
    except AttributeError:
        pass
    model.p_cashflow_dams = pe.Param(model.s_k2_birth_dams, model.s_cashflow_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cashflow_dams'], default=0.0, doc='cashflow dams')
    try:
        model.del_component(model.p_cashflow_prog_index)
        model.del_component(model.p_cashflow_prog)
    except AttributeError:
        pass
    model.p_cashflow_prog = pe.Param(model.s_cashflow_periods, model.s_sale_prog, model.s_lw_prog, model.s_season_types,
                                     model.s_tol, model.s_wean_times, model.s_gender, model.s_groups_dams,
                                  initialize=params['p_cashflow_prog'], default=0.0, doc='cashflow prog - made up from just sale value')
    try:
        model.del_component(model.p_cashflow_offs_index)
        model.del_component(model.p_cashflow_offs)
    except AttributeError:
        pass
    model.p_cashflow_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_cashflow_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_cashflow_offs'], default=0.0, doc='cashflow offs')
    ##cost - minroe
    try:
        model.del_component(model.p_cost_sire)
    except AttributeError:
        pass
    model.p_cost_sire = pe.Param(model.s_groups_sire, initialize=params['p_cost_sire'], 
                                  default=0.0, doc='husbandry cost sire')
    try:
        model.del_component(model.p_cost_dams_index)
        model.del_component(model.p_cost_dams)
    except AttributeError:
        pass
    model.p_cost_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cost_dams'], default=0.0, doc='husbandry cost dams')
    try:
        model.del_component(model.p_cost_offs_index)
        model.del_component(model.p_cost_offs)
    except AttributeError:
        pass
    model.p_cost_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_cost_offs'], default=0.0, doc='husbandry cost offs')

    ##asset value stock
    try:
        model.del_component(model.p_asset_sire)
    except AttributeError:
        pass
    model.p_asset_sire = pe.Param(model.s_groups_sire, initialize=params['p_assetvalue_sire'], default=0.0, doc='Asset value of sire')
    try:
        model.del_component(model.p_asset_dams_index)
        model.del_component(model.p_asset_dams)
    except AttributeError:
        pass
    model.p_asset_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_assetvalue_dams'], default=0.0, doc='Asset value of dams')
    try:
        model.del_component(model.p_asset_offs_index)
        model.del_component(model.p_asset_offs)
    except AttributeError:
        pass
    model.p_asset_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                                 model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                                 initialize=params['p_assetvalue_offs'], default=0.0, doc='Asset value of offs')

    ##labour - sire
    try:
        model.del_component(model.p_lab_anyone_sire_index)
        model.del_component(model.p_lab_anyone_sire)
    except AttributeError:
        pass
    model.p_lab_anyone_sire = pe.Param(model.s_labperiods, model.s_groups_sire,  initialize=params['p_labour_anyone_sire'], default=0.0, doc='labour requirement sire that can be done by anyone')
    try:
        model.del_component(model.p_lab_perm_sire_index)
        model.del_component(model.p_lab_perm_sire)
    except AttributeError:
        pass
    model.p_lab_perm_sire = pe.Param(model.s_labperiods, model.s_groups_sire, initialize=params['p_labour_perm_sire'], default=0.0, doc='labour requirement sire that can be done by perm staff')
    try:
        model.del_component(model.p_lab_manager_sire_index)
        model.del_component(model.p_lab_manager_sire)
    except AttributeError:
        pass
    model.p_lab_manager_sire = pe.Param(model.s_labperiods, model.s_groups_sire, initialize=params['p_labour_manager_sire'], default=0.0, doc='labour requirement sire that can be done by manager')
    
    ##labour - dams
    try:
        model.del_component(model.p_lab_anyone_dams_index)
        model.del_component(model.p_lab_anyone_dams)
    except AttributeError:
        pass
    model.p_lab_anyone_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                       initialize=params['p_labour_anyone_dams'], default=0.0, doc='labour requirement dams that can be done by anyone')
    try:
        model.del_component(model.p_lab_perm_dams_index)
        model.del_component(model.p_lab_perm_dams)
    except AttributeError:
        pass
    model.p_lab_perm_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                     initialize=params['p_labour_perm_dams'], default=0.0, doc='labour requirement dams that can be done by perm staff')
    try:
        model.del_component(model.p_lab_manager_dams_index)
        model.del_component(model.p_lab_manager_dams)
    except AttributeError:
        pass
    model.p_lab_manager_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                        initialize=params['p_labour_manager_dams'], default=0.0, doc='labour requirement dams that can be done by manager')
    
    ##labour - offs
    try:
        model.del_component(model.p_lab_anyone_offs_index)
        model.del_component(model.p_lab_anyone_offs)
    except AttributeError:
        pass
    model.p_lab_anyone_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_anyone_offs'], default=0.0, doc='labour requirement offs - anyone')
    try:
        model.del_component(model.p_lab_perm_offs_index)
        model.del_component(model.p_lab_perm_offs)
    except AttributeError:
        pass
    model.p_lab_perm_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_perm_offs'], default=0.0, doc='labour requirement offs - perm')
    try:
        model.del_component(model.p_lab_manager_offs_index)
        model.del_component(model.p_lab_manager_offs)
    except AttributeError:
        pass
    model.p_lab_manager_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_manager_offs'], default=0.0, doc='labour requirement offs - manager')

    ##infrastructure
    try:
        model.del_component(model.p_infra_sire_index)
        model.del_component(model.p_infra_sire)
    except AttributeError:
        pass
    model.p_infra_sire = pe.Param(model.s_infrastructure, model.s_groups_sire, initialize=params['p_infrastructure_sire'], 
                                  default=0.0, doc='sire requirement for infrastructure (based on number of times yarded and shearing activity)')
    try:
        model.del_component(model.p_infra_dams_index)
        model.del_component(model.p_infra_dams)
    except AttributeError:
        pass
    model.p_infra_dams = pe.Param(model.s_k2_birth_dams, model.s_infrastructure, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_infrastructure_dams'], default=0.0, doc='Dams requirement for infrastructure (based on number of times yarded and shearing activity)')
    try:
        model.del_component(model.p_infra_offs_index)
        model.del_component(model.p_infra_offs)
    except AttributeError:
        pass
    model.p_infra_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_infrastructure, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_infrastructure_offs'], default=0.0, doc='offs requirement for infrastructure (based on number of times yarded and shearing activity)')

    ##dse
    try:
        model.del_component(model.p_dse_sire_index)
        model.del_component(model.p_dse_sire)
    except AttributeError:
        pass
    model.p_dse_sire = pe.Param(model.s_feed_periods, model.s_groups_sire, initialize=params['p_dse_sire'],
                                  default=0.0, doc='number of dse for each sire activity')
    try:
        model.del_component(model.p_dse_dams_index)
        model.del_component(model.p_dse_dams)
    except AttributeError:
        pass
    model.p_dse_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_dse_dams'], default=0.0, doc='number of dse for each dam activity')
    try:
        model.del_component(model.p_dse_offs_index)
        model.del_component(model.p_dse_offs)
    except AttributeError:
        pass
    model.p_dse_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_dse_offs'], default=0.0, doc='number of dse for each offs activity')

    try:
        model.del_component(model.p_asset_stockinfra)
    except AttributeError:
        pass
    model.p_asset_stockinfra = pe.Param(model.s_infrastructure, initialize=params['p_infra'], default=0.0, doc='Asset value of infra')

    try:
        model.del_component(model.p_dep_stockinfra)
    except AttributeError:
        pass
    model.p_dep_stockinfra = pe.Param(model.s_infrastructure, initialize=0, default=0.0, doc='Depreciation of the asset value')

    try:
        model.del_component(model.p_rm_stockinfra_fix_index)
        model.del_component(model.p_rm_stockinfra_fix)
    except AttributeError:
        pass
    model.p_rm_stockinfra_fix = pe.Param(model.s_infrastructure, model.s_cashflow_periods,initialize=params['p_rm_stockinfra_fix'], default=0.0, doc='Fixed cost of R&M of the infrastructure')
    try:
        model.del_component(model.p_rm_stockinfra_var_index)
        model.del_component(model.p_rm_stockinfra_var)
    except AttributeError:
        pass
    model.p_rm_stockinfra_var = pe.Param(model.s_infrastructure, model.s_cashflow_periods,initialize=params['p_rm_stockinfra_var'], default=0.0, doc='Variable cost of R&M of the infrastructure (per animal mustered/shorn)')
    # try:
    #     model.del_component(model.p_lab_stockinfra)
    # except AttributeError:
    #     pass
    # model.p_lab_stockinfra = Param(model.s_infrastructure, model.s_labperiods, initialize=, default=0.0, doc='Labour required for R&M of the infrastructure (per animal mustered/shorn)')


    ##purchases
    try:
        model.del_component(model.p_cost_purch_sire_index)
        model.del_component(model.p_cost_purch_sire)
    except AttributeError:
        pass
    model.p_cost_purch_sire = pe.Param(model.s_cashflow_periods, model.s_groups_sire,
                                   initialize=params['p_purchcost_sire'], default=0.0, doc='cost of purchased sires')
    # model.p_numberpurch_dam = Param(model.s_dvp_dams, model.s_wean_times, model.s_k2_birth_dams, model.s_lw_dams,
    #                           model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception,
    #                           model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default=0.0, doc='purchase transfer - ie how a purchased dam is allocated into damR')
    # model.p_cost_purch_dam = Param(model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, model.s_cashflow_periods,
    #                                initialize=, default=0.0, doc='cost of purchased dams')
    # model.p_numberpurch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_k3_damage_offs,
    #                                  model.s_wean_times, model.s_k5_birth_offs, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
    #                                  initialize=, default=0.0, doc='purchase transfer - ie how a purchased offs is allocated into offsR')
    # model.p_cost_purch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, model.s_cashflow_periods,
    #                                initialize=, default=0.0, doc='cost of purchased offs')
    # ##transfers
    # model.p_offs2dam_numbers = Param(model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_k3_damage_offs,
    #                                  model.s_wean_times, model.s_k5_birth_offs, model.s_gender, model.s_gen_merit_offs, model.s_co_cfw,
    #                                  model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_dams, model.s_dvp_dams, model.s_wean_times,
    #                                  model.s_k2_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
    #                                  initialize=, default=0.0, doc='Proportion of the offs distributed to each of the starting LWs at the beginning of the current dam feed variation period')
    # model.p_dam2sire_numbers = Param(model.s_dvp_dams, model.s_wean_times, model.s_k2_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol,
    #                                  model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams,
    #                                  initialize=, default=0.0, doc='Proportion of the animals distributed to each of the starting LWs of the recipient animals at the beginning of the recipients next feed variation period')

    # textbuffer = StringIO()
    # model.p_numbers_prov_dams.pprint(textbuffer)
    # textbuffer.write('\n')
    # with open('number_prov.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())
    #
    # textbuffer = StringIO()
    # model.p_numbers_req_dams.pprint(textbuffer)
    # textbuffer.write('\n')
    # with open('number_prov.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


    end_params = time.time()
    print('params time: ',end_params-param_start)

    ########################
    ### set up constraints #
    ########################
    '''pyomo summary:
            - if a set has a 9 on the end of it, it is a special constraint set. And it is used to link with a decision variable set (the corresponding letter without 9 eg g? and g9). The
              set without a 9 must be summed.
            - if a given set doesnt have a corresponding 9 set, then you have two options
                1. transfer from one decision variable to another 1:1 (or at another ratio determined be the param - but it means that it transfers to the same set eg x1_dams transfers to x1_prog)
                2. treat all decision variable in a set the same. Done by summing. eg the npw provided by each dam t slice can be treated the same because it doesnt make a difference
                   if the progeny came from a dam that gets sold vs retained. (for most of the livestock it has been built in a way that doesnt need summing except for the sets which have a corresponding 9 set).
    
    speed info:
    - constraint.skip is fast, the trick is designing the code efficiently so that is knows when to skip.
    - in method 2 i use the param to determine when the constraint should be skipped, this still requires looping through the param
    - in method 3 i use the numpy array to determine when the constraint should be skipped. This is messier and requires some extra code but it is much more efficient reducing time 2x.
    - using if statements to save summing 0 values is faster but it still takes time to evaluate the if therefore it saves time to select the minimum number of if statements
    - constraints can only be skipped on based on the req param. if the provide side is 0 and you skip the constraint then that would mean there would be no restriction for the require variable.
    '''

    print('set up constraints')


    ##turn sets into list so they can be indexed (required for advanced method to save time)
    l_k29 = list(model.s_k2_birth_dams)
    l_v1 = list(model.s_dvp_dams)
    l_a = list(model.s_wean_times)
    l_z = list(model.s_season_types)
    l_i = list(model.s_tol)
    l_x = list(model.s_gender)
    l_y1 = list(model.s_gen_merit_dams)
    l_g9 = list(model.s_groups_dams)
    l_w9 = list(model.s_lw_dams)
    l_k3 = list(model.s_k3_damage_offs)
    l_k5 = list(model.s_k5_birth_offs)
    l_v3 = list(model.s_dvp_offs)
    l_g3 = list(model.s_groups_offs)
    l_w9_offs = list(model.s_lw_offs)


    try:
        model.del_component(model.con_damR_index)
        model.del_component(model.con_damR)
    except AttributeError:
        pass
    def damR(model,k29,v1,a,z,i,y1,g9,w9):
        v1_prev = l_v1[l_v1.index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k29 = l_k29.index(k29)
        t_v1 = l_v1.index(v1)
        t_a = l_a.index(a)
        t_z = l_z.index(z)
        t_i = l_i.index(i)
        t_y1 = l_y1.index(y1)
        t_g9 = l_g9.index(g9)
        t_w9 = l_w9.index(w9)
        if not np.any(params['numbers_req_numpyversion_k2k2tva1nw8ziyg1g9w9'][:,t_k29,:,t_v1,t_a,:,:,t_z,t_i,t_y1,:,t_g9,t_w9]):
            return pe.Constraint.Skip
        ##need to use both provide & require in this if statement (even though it is slower) because there are situations eg dvp4 (prejoining) where prov will have a value and req will not.
        ##but the prov parameter is necessary as it allows other dam permutations on this constraint
        return sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,t1,v1,a,n1,w8,z,i,y1,g1,g9,w9]
                   - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,g9,w9]
                   - model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_provthis_dams[k28,k29,t1,v1,a,n1,w8,z,i,y1,g1,g9,w9]
                   for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams
                   for n1 in model.s_nut_dams for w8 in model.s_lw_dams for g1 in model.s_groups_dams
                   if model.p_numbers_req_dams[k28, k29, t1, v1, a, n1, w8, z, i, y1, g1,g9, w9] != 0
                   or model.p_numbers_prov_dams[k28, k29, t1, v1_prev, a, n1, w8, z, i, y1, g1, g9, w9] != 0
                   or model.p_numbers_provthis_dams[k28, k29, t1, v1, a, n1, w8, z, i, y1, g1, g9, w9] != 0) <=0

    start=time.time()
    model.con_damR = pe.Constraint(model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                   model.s_groups_dams, model.s_lw_dams, rule=damR, doc='transfer dam to dam from last dvp to current dvp.')
    end=time.time()
    print('con_damR method 3: ',end-start)


    try:
        model.del_component(model.con_offR_index)
        model.del_component(model.con_offR)
    except AttributeError:
        pass
    def offR(model,k3,k5,v3,a,z,i,x,y3,g3,w9):
        v3_prev = l_v1[l_v3.index(v3) - 1]  #used to get the activity number from the last period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k3 = l_k3.index(k3)
        t_k5 = l_k5.index(k5)
        t_v3 = l_v3.index(v3)
        t_i = l_i.index(i)
        t_x = l_x.index(x)
        t_g3 = l_g3.index(g3)
        t_w9 = l_w9_offs.index(w9)
        if not np.any(params['numbers_req_numpyversion_k3k5vw8ixg3w9'][t_k3,t_k5,t_v3,:,t_i,t_x,t_g3,t_w9]):
            return pe.Constraint.Skip
        return sum(model.v_offs[k3,k5,t3,v3,n3,w8,z,i,a,x,y3,g3] * model.p_numbers_req_offs[k3,k5,v3,w8,i,x,g3,w9]
                   - model.v_offs[k3,k5,t3,v3_prev,n3,w8,z,i,a,x,y3,g3] * model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z,i,a,x,y3,g3,w9]
                    for t3 in model.s_sale_offs for n3 in model.s_nut_offs for w8 in model.s_lw_offs
                   if model.p_numbers_req_offs[k3,k5,v3,w8,i,x,g3,w9] != 0 or model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z,i,a,x,y3,g3,w9] != 0) <=0 #need to use both in the if statement (even though it is slower) because there are situations eg dvp4 (prejoining) where prov will have a value and req will not.
    start=time.time()
    model.con_offR = pe.Constraint(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_offs, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gender,
                                   model.s_gen_merit_dams, model.s_groups_offs, model.s_lw_offs, rule=offR, doc='transfer off to off from last dvp to current dvp.')
    end=time.time()
    print('con_offR method 3: ',end-start)


    try:
        model.del_component(model.con_progR_index)
        model.del_component(model.con_progR)
    except AttributeError:
        pass
    def progR(model, k5, a, z, i9, d, x, y1, g1, w9):
        if any(model.p_npw_req[t2,d,x,g1] for t2 in model.s_sale_prog):
            return (- sum(model.v_dams[k5, t1, v1, a, n1, w18, z, i, y1, g1]  * model.p_npw[k5, t1, v1, a, n1, w18, z, i, d, x, y1, g1, w9, i9] #pass in the k5 set to dams - each slice of k5 aligns with a slice in k2 eg 11 and 22. we don't need other k2 slices eg nm
                        for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams for w18 in model.s_lw_dams for i in model.s_tol
                             if model.p_npw[k5, t1, v1, a, n1, w18, z, i, d, x, y1, g1, w9, i9]!=0)
                    + sum(model.v_prog[k5, t2, w9, z, i9, d, a, x, g1] * model.p_npw_req[t2,d,x,g1] for t2 in model.s_sale_prog if model.p_npw_req[t2,d,x,g1]!=0))<=0
        else:
            return pe.Constraint.Skip
    # def progR(model, k5, a, z, i9, d, x, y1, g1, w9):
    #     if any(model.p_npw_req[t2,d,x] for t2 in model.s_sale_prog):
    #         return (- sum(model.v_dams[k2, t1, v1, a, n1, w18, z, i, y1, g1]  * model.p_npw[k2, k5, t1, v1, a, n1, w18, z, i, d, x, y1, g1, w9, i9]
    #                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams for w18 in model.s_lw_dams for i in model.s_tol
    #                          if model.p_npw[k2, k5, t1, v1, a, n1, w18, z, i, d, x, y1, g1, w9, i9]!=0)
    #                 + sum(model.v_prog[k5, t2, w9, z, i9, d, a, x, g1] * model.p_npw_req[t2,d,x] for t2 in model.s_sale_prog if model.p_npw_req[t2,d,x]!=0))<=0
    #     else:
    #         return pe.Constraint.Skip
    start = time.time()
    model.con_progR = pe.Constraint(model.s_k5_birth_offs, model.s_wean_times, model.s_season_types, model.s_tol,
                                    model.s_damage, model.s_gender, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_prog, rule=progR,
                                   doc='transfer npw from dams to prog.')
    end = time.time()
    print('con_progR ',end-start)

    try:
        model.del_component(model.con_prog2damsR_index)
        model.del_component(model.con_prog2damsR)
    except AttributeError:
        pass
    def prog2damR(model, k3, k5, v1, z, i, y1, g9, w9):
        if v1==l_v1[0] and any(model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9] for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for w18 in model.s_lw_dams for g1 in model.s_groups_dams):
            return (sum(- model.v_prog[k5, t2, w28, z, i, d, a0, x, g2] * model.p_progprov_dams[k3, t2, w28, z, i, d, a0, x, y1, g2,g9,w9]
                        for d in model.s_damage for a0 in model.s_wean_times for x in model.s_gender for w28 in model.s_lw_prog for t2 in model.s_sale_prog for g2 in model.s_groups_prog
                        if model.p_progprov_dams[k3, t2, w28, z, i, d, a0, x, y1, g2,g9,w9]!= 0)
                       + sum(model.v_dams[k2, t1, v1, a1, n1, w18, z, i, y1, g1]  * model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9]
                        for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for a1 in model.s_wean_times for n1 in model.s_nut_dams for w18 in model.s_lw_dams for g1 in model.s_groups_dams
                             if model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9]!= 0))<=0
        else:
            return pe.Constraint.Skip
    start = time.time()
    model.con_prog2damsR = pe.Constraint(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_dams, model.s_season_types,
                                   model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=prog2damR,
                                   doc='transfer prog to dams in dvp 0.')
    end = time.time()
    print('con_prog2damsR ',end-start)

    try:
        model.del_component(model.con_prog2offsR_index)
        model.del_component(model.con_prog2offsR)
    except AttributeError:
        pass
    def prog2offsR(model, k3, k5, v3, z, i, a, x, y3, g3, w9):
        if v3==l_v3[0] and any(model.p_progreq_offs[v3, w38, i, x, w9] for w38 in model.s_lw_offs):
            return (sum(- model.v_prog[k5, t2, w28, z, i, d, a, x, g3] * model.p_progprov_offs[k3, k5, t2, w28, z, i, d, a, x, y3, g3, w9] #use g3 (same as g2)
                        for d in model.s_damage for w28 in model.s_lw_prog for t2 in model.s_sale_prog
                        if model.p_progprov_offs[k3, k5, t2, w28, z, i, d, a, x, y3, g3, w9]!= 0)
                       + sum(model.v_offs[k3,k5,t3,v3,n3,w38,z,i,a,x,y3,g3]  * model.p_progreq_offs[v3, w38, i, x, w9]
                        for t3 in model.s_sale_offs for n3 in model.s_nut_dams for w38 in model.s_lw_offs if model.p_progreq_offs[v3, w38, i, x, w9]!= 0))<=0
        else:
            return pe.Constraint.Skip
    start = time.time()
    model.con_prog2offsR = pe.Constraint(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_dams, model.s_season_types, model.s_tol,
                                   model.s_wean_times, model.s_gender, model.s_gen_merit_dams, model.s_groups_offs, model.s_lw_offs, rule=prog2offsR,
                                   doc='transfer prog to off in dvp 0.')
    end = time.time()
    print('con_prog2offR ',end-start)



    try:
        model.del_component(model.con_matingR_index)
        model.del_component(model.con_matingR)
    except AttributeError:
        pass
    def mating(model,g0,p8):
        return - model.v_sire[g0] * model.p_nsires_prov[g0, p8] + sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_nsires_req[k2,t1,v1,a,n1,w1,z,i,y1,g1,g0,p8]
                  for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for a in model.s_wean_times for n1 in model.s_nut_dams
                   for w1 in model.s_lw_dams for z in model.s_season_types for i in model.s_tol for y1 in model.s_gen_merit_dams  for g1 in model.s_groups_dams
                   if model.p_nsires_req[k2,t1,v1,a,n1,w1,z,i,y1,g1,g0,p8]!=0) <=0
    model.con_matingR = pe.Constraint(model.s_groups_sire, model.s_sire_periods, rule=mating, doc='sire requirement for mating')

    try:
        model.del_component(model.con_stockinfra)
    except AttributeError:
        pass
    def stockinfra(model,h1):
        return -model.v_infrastructure[h1] + sum(model.v_sire[g0] * model.p_infra_sire[h1,g0] for g0 in model.s_groups_sire if model.p_infra_sire[h1,g0]!=0)  \
               + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_infra_dams[k2,h1,t1,v1,a,n1,w1,z,i,y1,g1]
                         for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                         for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams if model.p_infra_dams[k2,h1,t1,v1,a,n1,w1,z,i,y1,g1]!=0)
                    + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_infra_offs[k3,k5,h1,t3,v3,n3,w3,z,i,a,x,y3,g3]
                          for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                          for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs if model.p_infra_offs[k3,k5,h1,t3,v3,n3,w3,z,i,a,x,y3,g3]!=0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol) <=0
    model.con_stockinfra = pe.Constraint(model.s_infrastructure, rule=stockinfra, doc='Requirement for infrastructure (based on number of times yarded and shearing activity)')

    end_cons=time.time()
    print('time con: ', end_cons-end_params)

    # textbuffer = StringIO()
    # model.con_offR.pprint(textbuffer)
    # textbuffer.write('\n')
    # with open('cons.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())

#####################
##  setup variables # these variables only need initialising once ie sets wont change within and iteration of exp.
#####################
##infrastructure
model.v_infrastructure = pe.Var(model.s_infrastructure, bounds = (0,None) , doc='amount of infrastructure required for given animal enterprise (based on number of sheep through infra)')

# ##################################
# ### setup core model constraints #
# ##################################

def stock_me(model,f,p6):
    return sum(model.v_sire[g0] * model.p_mei_sire[p6,f,g0] for g0 in model.s_groups_sire)\
           + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_mei_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if model.p_mei_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_mei_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_mei_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)


def stock_pi(model,f,p6):
    return sum(model.v_sire[g0] * model.p_pi_sire[p6,f,g0] for g0 in model.s_groups_sire)\
           + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_pi_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if model.p_pi_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_pi_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_pi_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)

def stock_cashflow(model,c):
    infrastructure = sum(model.p_rm_stockinfra_fix[h1,c] + model.p_rm_stockinfra_var[h1,c] * model.v_infrastructure[h1]
                         for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cashflow_sire[c,g0] for g0 in model.s_groups_sire) \
           + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_cashflow_dams[k2,c,t1,v1,a,n1,w1,z,i,y1,g1]
                      for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                      for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if model.p_cashflow_dams[k2,c,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_prog[k5, t2, w2, z, i, d, a, x, g2] * model.p_cashflow_prog[c, t2, w2, z, i, a, x, g2]
                      for k5 in model.s_k5_birth_offs for t2 in model.s_sale_prog for w2 in model.s_lw_prog for d in model.s_damage
                      for x in model.s_gender for g2 in model.s_groups_prog if model.p_cashflow_prog[c, t2, w2, z, i, a, x, g2] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_cashflow_offs[k3,k5,c,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_cashflow_offs[k3,k5,c,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[c,g0] for g0 in model.s_groups_sire)
    return stock - infrastructure - purchases

#     purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[g0,c] for g0 in model.s_groups_sire)  \
#                 + sum(sum(model.v_purchase_dams[v1,w1,z,i,g1] * model.p_cost_purch_dam[v1,w1,z,i,g1,c] for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
#                     + sum(model.v_purchase_offs[v3,w3,z,i,g3] * model.p_cost_purch_offs[v3,w3,z,i,g3,c] for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
#                     for z in model.s_season_types for i in model.s_tol)
#     return stock - infrastructure - purchases


def stock_cost(model):
    infrastructure = sum(model.p_rm_stockinfra_fix[h1,c] + model.p_rm_stockinfra_var[h1,c] * model.v_infrastructure[h1]
                         for h1 in model.s_infrastructure for c in model.s_cashflow_periods)
    stock = sum(model.v_sire[g0] * model.p_cost_sire[g0] for g0 in model.s_groups_sire) \
            + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_cost_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if model.p_cost_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_cost_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_cost_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[c,g0] for g0 in model.s_groups_sire for c in model.s_cashflow_periods)
    return  stock + infrastructure + purchases
#
#
def stock_labour_anyone(model,p5):
    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_lab_anyone_sire[p5,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_anyone_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if model.p_lab_anyone_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_anyone_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_lab_anyone_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    return stock

def stock_labour_perm(model,p5):
    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_lab_perm_sire[p5,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_perm_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if model.p_lab_perm_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_perm_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_lab_perm_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    return stock

def stock_labour_manager(model,p5):
    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_lab_manager_sire[p5,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_manager_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if model.p_lab_manager_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_manager_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_lab_manager_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    return stock
#


def stock_dep(model):
    return sum(model.p_dep_stockinfra[h1]  * model.v_infrastructure[h1] for h1 in model.s_infrastructure)

def stock_asset(model):
    infrastructure = sum(model.p_asset_stockinfra[h1] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_asset_sire[g0] for g0 in model.s_groups_sire) \
            + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_asset_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if model.p_asset_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] != 0)
                + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_asset_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if model.p_asset_offs[k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3] != 0)
               for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    # purchases = sum(sum(model.v_purchase_dams[v1,w1,z,i,g1] * sum(model.p_cost_purch_dam[v1,w1,z,i,g1,c] for c in model.s_cashflow_periods) for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
    #                 +sum(model.v_purchase_offs[v3,w3,z,i,g3] * sum(model.p_cost_purch_offs[v3,w3,z,i,g3,c] for c in model.s_cashflow_periods) for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
    return  stock + infrastructure #+ purchases


##################################
# old methods used to find speed #
##################################

    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR(model,k29,v1,a,z,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     con = sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0 or model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    #         # + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,g1_new]
    #         #       - model.v_dams2sire[v1_prev,a,b1,n1,w1,z,i,y1,g1,g1_new] * model.p_dam2sire_numbers[v1,a,b1,n1,w1,z,i,y1,g1,g1_new]
    #         #       for n1 in model.s_nut_dams for g1_new in model.s_groups_dams) \
    #         # - model.v_purchase_dams[v1,w1,z,i,g1] * model.p_numberpurch_dam[v1,a,b1,w1,z,i,y1,g1] \ #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
    #         # - sum(model.v_offs2dam[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off] * model.p_offs2dam_numbers[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off,v1,a,b1,w1,z,i,y1,g1]
    #         #       for v3 in model.s_dvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for z3 in model.s_season_types for i3 in model.s_tol for d in model.s_k3_damage_offs for a3 in model.s_wean_times
    #         #       for b3 in model.s_k5_birth_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs for g1_off in model.s_groups_dams)  #have to track off sets so only they are summed.
    #     ###if statement required to handle the constraints that don't exist due to lw clustering
    #     # if sum(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) ==0 and sum(model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #     #         for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9] !=0)==0:
    #     #     pass
    #     if type(con)==bool:
    #         return pe.Constraint.Skip
    #     else: return con

    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR1(model,k29,v1,a,z,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     con = sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    #     if type(con)==bool:
    #         return pe.Constraint.Skip
    #     else: return con
    # start=time.time()
    # # model.con_damR = pe.Constraint(model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR1, doc='transfer of off to dam and dam from last dvp to current dvp.')
    # end=time.time()
    # print('method 1: ',end-start)

    # ##method 2
    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR2(model,k29,v1,a,z,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     ##skip constraint if the require param is 0
    #     if not any(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams):
    #         return pe.Constraint.Skip
    #     return sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    # start=time.time()
    # model.con_damR = pe.Constraint(model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR2, doc='transfer of off to dam and dam from last dvp to current dvp.')
    # end=time.time()
    # print('method 2: ',end-start)



