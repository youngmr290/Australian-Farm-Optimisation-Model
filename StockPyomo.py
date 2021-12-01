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
import StockGenerator as sgen
import PropertyInputs as pinp


def stock_precalcs(params, r_vals, nv, pkl_fs_info):
    sgen.generator(params, r_vals, nv, pkl_fs_info)



def f1_stockpyomo_local(params, model):

    ##these sets require info from the stock module
    model.s_wean_times = pe.Set(initialize=params['a_idx'], doc='weaning options')
    model.s_tol = pe.Set(initialize=params['i_idx'], doc='birth groups (times of lambing)')
    model.s_sire_periods = pe.Set(initialize=params['p8_idx'], doc='sire periods')
    model.s_groups_sire = pe.Set(initialize=params['g_idx_sire'], doc='genotype groups of sires')
    model.s_gen_merit_sire = pe.Set(initialize=params['y_idx_sire'], doc='genetic merit of sires')
    model.s_k2_birth_dams = pe.Set(initialize=params['k2_idx_dams'], doc='Cluster for LSLN & oestrus cycle based on scanning, global & weaning management')
    model.s_dvp_dams = pe.Set(ordered=True, initialize=params['dvp_idx_dams'], doc='Decision variable periods for dams') #ordered so they can be indexed in constraint to determine previous period
    model.s_lw_dams = pe.Set(initialize=params['w_idx_dams'], doc='Standard LW patterns dams')
    model.s_groups_dams = pe.Set(initialize=params['g_idx_dams'], doc='genotype groups of dams')
    model.s_groups_prog = pe.Set(initialize=params['g_idx_dams'], doc='genotype groups of prog') #same as dams and offs
    model.s_gen_merit_dams = pe.Set(initialize=params['y_idx_dams'], doc='genetic merit of dams')
    model.s_sale_dams = pe.Set(initialize=params['t_idx_dams'], doc='Sales within the year for dams')
    model.s_dvp_offs = pe.Set(ordered=True, initialize=params['dvp_idx_offs'], doc='Decision variable periods for offs') #ordered so they can be indexed in constraint to determine previous period
    model.s_damage = pe.Set(initialize=params['d_idx'], doc='age of mother - offs')
    model.s_k3_damage_offs = pe.Set(initialize=params['k3_idx_offs'], doc='age of mother - offs')
    model.s_k5_birth_offs = pe.Set(initialize=params['k5_idx_offs'], doc='Cluster for BTRT & oestrus cycle based on scanning, global & weaning management')
    model.s_lw_offs = pe.Set(initialize=params['w_idx_offs'], doc='Standard LW patterns offs')
    model.s_groups_offs = pe.Set(initialize=params['g_idx_offs'], doc='genotype groups of offs')
    model.s_gen_merit_offs = pe.Set(initialize=params['y_idx_offs'], doc='genetic merit of offs')
    model.s_gender = pe.Set(initialize=params['x_idx_offs'], doc='gender of offs')

    #####################
    ##  setup variables # #variables that use dynamic sets must be defined each iteration of exp
    #####################
    ##animals
    model.v_sire = pe.Var(model.s_sequence_year, model.s_sequence, model.s_groups_sire, bounds = (0,None) , doc='number of sire animals') #assumption is that no tactical management of numbers of dams mated and hence sires across seasons so no z axis. Does need q & s axis though for multiperiod model.
    model.v_dams = pe.Var(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, bounds = (0,None) , doc='number of dam animals')
    model.v_offs = pe.Var(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                          model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs,
                          model.s_groups_offs, bounds = (0,None) , doc='number of offs animals')
    model.v_prog = pe.Var(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs,
                          model.s_sale_prog, model.s_lw_prog, model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender,
                          model.s_groups_prog, bounds = (0,None) , doc='number of offs animals')

    ##purchases
    model.v_purchase_dams = pe.Var(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, bounds = (0,None) , doc='number of purchased dam animals')
    model.v_purchase_offs = pe.Var(model.s_sequence_year, model.s_sequence, model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, bounds = (0,None) , doc='number of purchased offs animals')

    ##infrastructure
    model.v_infrastructure = pe.Var(model.s_sequence_year, model.s_sequence, model.s_infrastructure, model.s_season_types, bounds=(0,None),
                                    doc='amount of infrastructure required for given animal enterprise (based on number of sheep through infra)')

    ######################
    ### setup parameters #
    ######################
    param_start = time.time()

    ##nsire - mating
    model.p_nsires_req = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, 
                                  model.s_groups_sire, model.s_sire_periods, initialize=params['p_nsire_req_dams'], 
                                  default=0.0, mutable=False, doc='requirement for sires for mating')

    model.p_nsires_prov = pe.Param(model.s_season_types, model.s_groups_sire, model.s_sire_periods,
                               initialize=params['p_nsire_prov_sire'], default=0.0, mutable=False, doc='sires available for mating')

    ##progeny
    model.p_npw = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, 
                           model.s_nut_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gender, model.s_gen_merit_dams, 
                           model.s_groups_dams, model.s_lw_prog, model.s_tol, initialize=params['p_npw_dams']
                           , default=0.0, mutable=False, doc='number of progeny weaned')
    model.p_npw_req = pe.Param(model.s_k3_damage_offs, model.s_sale_prog, model.s_gender, model.s_groups_prog,
                              initialize=params['p_npw_req_prog'], default=0.0, doc='number of yatf required by the prog activity')
    model.p_progprov_dams = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_prog, model.s_lw_prog, 
                                     model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_dams, 
                                     model.s_groups_prog, model.s_groups_dams, model.s_lw_dams,
                                     initialize=params['p_progprov_dams'], default=0.0, mutable=False, doc='number of progeny provided to dams')
    model.p_progreq_dams = pe.Param(model.s_k2_birth_dams, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_dams, model.s_lw_dams,
                              model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                              initialize=params['p_progreq_dams'], default=0.0, doc='number of progeny required by dams')
    model.p_progprov_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_prog, model.s_lw_prog,
                                     model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, 
                                     model.s_gen_merit_offs, model.s_groups_offs, model.s_lw_offs,
                                     initialize=params['p_progprov_offs'], default=0.0, mutable=False, doc='number of progeny provided to dams')
    model.p_progreq_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_lw_offs, model.s_season_types,
                                    model.s_tol, model.s_gender, model.s_groups_offs, model.s_lw_offs,
                              initialize=params['p_progreq_offs'], default=0.0, doc='number of progeny required by dams')


    ##stock - dams
    model.p_numbers_prov_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams, 
                                         model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                                         model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_prov_dams'], default=0.0, mutable=False, doc='numbers provided by each dam activity into the next period')

    model.p_numbers_provthis_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams,
                                         model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                         model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_provthis_dams'], default=0.0, mutable=False,
                                         doc='numbers provided by each dam transfer activity into this period')

    model.p_numbers_req_dams = pe.Param(model.s_k2_birth_dams, model.s_k2_birth_dams, model.s_sale_dams, model.s_dvp_dams,
                                        model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types,
                                        model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                        initialize=params['p_numbers_req_dams'], default=0.0, doc='numbers required by each dam activity in the current period')

    ##stock - offs
    model.p_numbers_prov_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_dvp_offs, 
                                         model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times, 
                                         model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, model.s_lw_offs,
                                 initialize=params['p_numbers_prov_offs'], default=0.0, mutable=False, doc='numbers provided into the current period from the previous periods activities')
    model.p_numbers_req_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_offs, model.s_lw_offs,
                                        model.s_season_types, model.s_tol, model.s_gender, model.s_groups_offs, model.s_lw_offs,
                                        initialize=params['p_numbers_req_offs'], default=0.0, doc='requirement of off in the current period')

    ##energy intake
    model.p_mei_sire = pe.Param(model.s_feed_periods, model.s_feed_pools, model.s_season_types, model.s_groups_sire,
                                initialize=params['p_mei_sire'],
                                  default=0.0, mutable=False, doc='energy requirement sire')
    model.p_mei_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_feed_pools, model.s_sale_dams,
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_mei_dams'],
                                default=0.0, mutable=False, doc='energy requirement dams')
    model.p_mei_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_feed_pools, model.s_sale_offs,
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times,
                               model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_mei_offs'],
                                default=0.0, mutable=False, doc='energy requirement offs')

    ##potential intake
    model.p_pi_sire = pe.Param(model.s_feed_periods, model.s_feed_pools, model.s_season_types, model.s_groups_sire,
                               initialize=params['p_pi_sire'],
                                  default=0.0, mutable=False, doc='pi sire')
    model.p_pi_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_feed_pools, model.s_sale_dams,
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_pi_dams'],
                               default=0.0, mutable=False, doc='pi dams')
    model.p_pi_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_feed_pools, model.s_sale_offs,
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times,
                               model.s_gender, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_pi_offs'],
                               default=0.0, mutable=False, doc='pi offs')

    ##cashflow
    model.p_cashflow_sire = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_groups_sire, initialize=params['p_cashflow_sire'],
                                  default=0.0, mutable=False, doc='cashflow sire')
    model.p_cashflow_dams = pe.Param(model.s_k2_birth_dams, model.s_enterprises, model.s_season_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cashflow_dams'], default=0.0, mutable=False, doc='cashflow dams')
    model.p_cashflow_prog = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_enterprises, model.s_season_periods, model.s_sale_prog, model.s_lw_prog,
                                     model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_groups_dams,
                                  initialize=params['p_cashflow_prog'], default=0.0, mutable=False, doc='cashflow prog - made up from just sale value')
    model.p_cashflow_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_enterprises, model.s_season_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_cashflow_offs'], default=0.0, mutable=False, doc='cashflow offs')

    ##working capital
    model.p_wc_sire = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_groups_sire, initialize=params['p_cashflow_sire'],
                                  default=0.0, mutable=False, doc='wc sire')
    model.p_wc_dams = pe.Param(model.s_k2_birth_dams, model.s_enterprises, model.s_season_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_wc_dams'], default=0.0, mutable=False, doc='wc dams')
    model.p_wc_prog = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_enterprises, model.s_season_periods, model.s_sale_prog, model.s_lw_prog,
                                     model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_groups_dams,
                                  initialize=params['p_wc_prog'], default=0.0, mutable=False, doc='wc prog - made up from just sale value')
    model.p_wc_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_enterprises, model.s_season_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_wc_offs'], default=0.0, mutable=False, doc='wc offs')

    ##cost - minroe
    model.p_cost_sire = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_groups_sire, initialize=params['p_cost_sire'],
                                  default=0.0, mutable=False, doc='husbandry cost sire')
    model.p_cost_dams = pe.Param(model.s_k2_birth_dams, model.s_enterprises, model.s_season_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cost_dams'], default=0.0, mutable=False, doc='husbandry cost dams')
    model.p_cost_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_enterprises, model.s_season_periods,
                                 model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                                 model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                                 initialize=params['p_cost_offs'], default=0.0, mutable=False, doc='husbandry cost offs')

    ##asset value stock
    model.p_asset_sire = pe.Param(model.s_season_periods, model.s_season_types, model.s_groups_sire, initialize=params['p_assetvalue_sire'],
                                  default=0.0, mutable=False, doc='Asset value of sire')
    model.p_asset_dams = pe.Param(model.s_k2_birth_dams, model.s_sale_dams, model.s_season_periods, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_assetvalue_dams'], default=0.0, mutable=False, doc='Asset value of dams')
    model.p_asset_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_sale_offs, model.s_season_periods, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                                 model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                                 initialize=params['p_assetvalue_offs'], default=0.0, mutable=False, doc='Asset value of offs')

    ##labour - sire
    model.p_lab_anyone_sire = pe.Param(model.s_labperiods, model.s_season_types, model.s_groups_sire,  initialize=params['p_labour_anyone_sire'], default=0.0, mutable=False, doc='labour requirement sire that can be done by anyone')
    model.p_lab_perm_sire = pe.Param(model.s_labperiods, model.s_season_types, model.s_groups_sire, initialize=params['p_labour_perm_sire'], default=0.0, mutable=False, doc='labour requirement sire that can be done by perm staff')
    model.p_lab_manager_sire = pe.Param(model.s_labperiods, model.s_season_types, model.s_groups_sire, initialize=params['p_labour_manager_sire'], default=0.0, mutable=False, doc='labour requirement sire that can be done by manager')
    
    ##labour - dams
    model.p_lab_anyone_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                       model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_labour_anyone_dams'],
                                       default=0.0, mutable=False, doc='labour requirement dams that can be done by anyone')
    model.p_lab_perm_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                     model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_labour_perm_dams'],
                                     default=0.0, mutable=False, doc='labour requirement dams that can be done by perm staff')
    model.p_lab_manager_dams = pe.Param(model.s_k2_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                        model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_labour_manager_dams'],
                                        default=0.0, mutable=False, doc='labour requirement dams that can be done by manager')
    
    ##labour - offs
    model.p_lab_anyone_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_anyone_offs'], default=0.0, mutable=False, doc='labour requirement offs - anyone')
    model.p_lab_perm_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_perm_offs'], default=0.0, mutable=False, doc='labour requirement offs - perm')
    model.p_lab_manager_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_manager_offs'], default=0.0, mutable=False, doc='labour requirement offs - manager')

    ##infrastructure
    model.p_infra_sire = pe.Param(model.s_infrastructure, model.s_season_types, model.s_groups_sire, initialize=params['p_infrastructure_sire'],
                                  default=0.0, mutable=False, doc='sire requirement for infrastructure (based on number of times yarded and shearing activity)')
    model.p_infra_dams = pe.Param(model.s_k2_birth_dams, model.s_infrastructure, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_infrastructure_dams'], default=0.0, mutable=False, doc='Dams requirement for infrastructure (based on number of times yarded and shearing activity)')
    model.p_infra_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_infrastructure, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_infrastructure_offs'], default=0.0, mutable=False, doc='offs requirement for infrastructure (based on number of times yarded and shearing activity)')
    model.p_rm_stockinfra_fix = pe.Param(model.s_infrastructure, model.s_enterprises, model.s_season_periods, model.s_season_types,
                                         initialize=params['p_rm_stockinfra_fix'], default=0.0, doc='Fixed cost of R&M of the infrastructure')
    model.p_rm_stockinfra_var = pe.Param(model.s_infrastructure, model.s_enterprises, model.s_season_periods, model.s_season_types,
                                         initialize=params['p_rm_stockinfra_var'], default=0.0, doc='Variable cost of R&M of the infrastructure (per animal mustered/shorn)')
    # model.p_lab_stockinfra = Param(model.s_infrastructure, model.s_labperiods, initialize=, default=0.0, doc='Labour required for R&M of the infrastructure (per animal mustered/shorn)')

    ##dse
    model.p_dse_sire = pe.Param(model.s_feed_periods, model.s_season_types, model.s_groups_sire, initialize=params['p_dse_sire'],
                                  default=0.0, mutable=False, doc='number of dse for each sire activity')
    model.p_dse_dams = pe.Param(model.s_k2_birth_dams, model.s_feed_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams,
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_dse_dams'], default=0.0, mutable=False, doc='number of dse for each dam activity')
    model.p_dse_offs = pe.Param(model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_feed_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_dse_offs'], default=0.0, mutable=False, doc='number of dse for each offs activity')

    model.p_wg_propn_p6z = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_wg_propn_p6z'],
                                    default=0.0, mutable=False, doc='proportion of each feed period used to calc dse')

    ##asset value
    model.p_asset_stockinfra = pe.Param(model.s_season_periods, model.s_infrastructure, initialize=params['p_infra'], default=0.0, doc='Asset value of infra')

    ##purchases
    model.p_cost_purch_sire = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_groups_sire,
                                   initialize=params['p_purchcost_sire'], default=0.0, mutable=False, doc='cost of purchased sires')
    model.p_wc_purch_sire = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, model.s_groups_sire,
                                   initialize=params['p_purchcost_wc_sire'], default=0.0, mutable=False, doc='working capital cost of purchased sires')
    # model.p_numberpurch_dam = Param(model.s_dvp_dams, model.s_wean_times, model.s_k2_birth_dams, model.s_lw_dams,
    #                           model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception,
    #                           model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default=0.0, doc='purchase transfer - ie how a purchased dam is allocated into damR')
    # model.p_cost_purch_dam = Param(model.s_dvp_dams, model.s_lw_dams, model.s_tol, model.s_groups_dams, model.s_season_periods,
    #                                initialize=, default=0.0, doc='cost of purchased dams')
    # model.p_numberpurch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_tol, model.s_k3_damage_offs,
    #                                  model.s_wean_times, model.s_k5_birth_offs, model.s_gender, model.s_gen_merit_offs, model.s_groups_offs,
    #                                  initialize=, default=0.0, doc='purchase transfer - ie how a purchased offs is allocated into offsR')
    # model.p_cost_purch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_tol, model.s_groups_offs, model.s_season_periods,
    #                                initialize=, default=0.0, doc='cost of purchased offs')

    ##season - current and prev versions of the param are required because in the numbers constraint they are indexed by v_prev and in the prog constraints they are indexed by v.
    # model.p_childz_req = pe.Param(model.s_season_types, model.s_season_types, initialize=params['p_childz_req'],
    #                               default=0.0, mutable=False, doc='z8z9 numbers required')
    model.p_parentz_provwithin_dams = pe.Param(model.s_k2_birth_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                  model.s_season_types, initialize=params['p_parentz_provwithin_dams'], default=0.0,
                                  mutable=False, doc='Transfer of z8 dv in the previous dvp to z9 constraint in the current dvp')
    model.p_parentz_provbetween_dams = pe.Param(model.s_k2_birth_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                  model.s_season_types, initialize=params['p_parentz_provbetween_dams'], default=0.0,
                                  mutable=False, doc='Transfer of z8 dv in the previous dvp to z9 constraint in the current dvp')
    model.p_mask_childz_within_dams = pe.Param(model.s_k2_birth_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                  initialize=params['p_mask_childz_within_dams'], default=0.0,
                                  mutable=False, doc='mask child season require params')
    model.p_mask_childz_between_dams = pe.Param(model.s_k2_birth_dams, model.s_dvp_dams, model.s_season_types, model.s_groups_dams,
                                  initialize=params['p_mask_childz_between_dams'], default=0.0,
                                  mutable=False, doc='mask child season require params')
    model.p_parentz_provwithin_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_season_types,
                                                  model.s_gender, model.s_groups_offs, model.s_season_types,
                                                  initialize=params['p_parentz_provwithin_offs'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous dvp to z9 constraint in the current dvp')
    model.p_parentz_provbetween_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_season_types,
                                                  model.s_gender, model.s_groups_offs, model.s_season_types,
                                                  initialize=params['p_parentz_provbetween_offs'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous dvp to z9 constraint in the current dvp')
    model.p_mask_childz_within_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_season_types,
                                                  model.s_gender, model.s_groups_offs,
                                                  initialize=params['p_mask_childz_within_offs'], default=0.0,
                                                  mutable=False, doc='mask child season require params')
    model.p_mask_childz_between_offs = pe.Param(model.s_k3_damage_offs, model.s_dvp_offs, model.s_season_types,
                                                  model.s_gender, model.s_groups_offs,
                                                  initialize=params['p_mask_childz_between_offs'], default=0.0,
                                                  mutable=False, doc='mask child season require params')

    ##write param to text file.
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
    # print('params time: ',end_params-param_start)

    ########################
    #call local constraint #
    ########################
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

    ##call local constraint functions
    f_con_off_withinR(model, params, l_v3, l_k3, l_k5, l_z, l_i, l_x, l_g3, l_w9_offs)
    f_con_off_betweenR(model, params, l_v3, l_k3, l_k5, l_z, l_i, l_x, l_g3, l_w9_offs)
    f_con_dam_withinR(model, params, l_v1, l_k29, l_a, l_z, l_i, l_y1, l_g9, l_w9)
    f_con_dam_betweenR(model, params, l_v1, l_k29, l_a, l_z, l_i, l_y1, l_g9, l_w9)
    f_con_progR(model)
    f_con_prog2damsR(model,l_v1)
    f_con_prog2offsR(model,l_v3)
    f_con_matingR(model)
    f_con_stockinfra(model)

########################
# local constraints    #
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
- you can use set filter to build filtered sets instead od skipping the constraint however this made little speed difference. 
- using if statements to save summing 0 values is faster but it still takes time to evaluate the if therefore it saves time to select the minimum number of if statements
- constraints can only be skipped on based on the req param. if the provide side is 0 and you skip the constraint then that would mean there would be no restriction for the require variable.
'''

def f_con_off_withinR(model, params, l_v3, l_k3, l_k5, l_z, l_i, l_x, l_g3, l_w9_offs):
    '''
    Within year numbers/transfers of offspring to offspring in the following decision variable period.

    '''
    def offwithinR(model,q,s,k3,k5,v3,a,z9,i,x,y3,g3,w9):
        v3_prev = l_v3[l_v3.index(v3) - 1]  #used to get the activity number from the last period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k3 = l_k3.index(k3)
        t_k5 = l_k5.index(k5)
        t_v3 = l_v3.index(v3)
        t_z = l_z.index(z9)
        t_i = l_i.index(i)
        t_x = l_x.index(x)
        t_g3 = l_g3.index(g3)
        t_w9 = l_w9_offs.index(w9)
        if np.any(params['numbers_req_numpyversion_k3k5vw8zixg3w9'][t_k3,t_k5,t_v3,:,t_z,t_i,t_x,t_g3,t_w9]) \
           and pe.value(model.p_mask_childz_within_offs[k3,v3,z9,x,g3])\
           and pe.value(model.p_wyear_inc_qs[q,s]):
            ###note: dont need to multiply the child params/variables by p_mask_child because the whole constraint is skipped
            ### and the params are already masked by mask_z8 so the only bit missing is the 'between' period which is handled by skipping.
            return sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w8,z9,i,a,x,y3,g3] * model.p_numbers_req_offs[k3,k5,v3,w8,z9,i,x,g3,w9]
                       - sum(model.v_offs[q,s,k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3] * model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3,w9]
                          * model.p_parentz_provwithin_offs[k3,v3_prev,z8,x,g3,z9] for z8 in model.s_season_types)
                       for t3 in model.s_sale_offs for n3 in model.s_nut_offs for w8 in model.s_lw_offs
                       if pe.value(model.p_numbers_req_offs[k3,k5,v3,w8,z9,i,x,g3,w9]) != 0
                       or pe.value(model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z9,i,a,x,y3,g3,w9]) #doesnt need to use z8 because in the within constraint because z only provides to itsself and children with the same w patten.
                       ) <=0 #need to use both in the if statement (even though it is slower) because there are situations eg dvp4 (prejoining) where prov will have a value and req will not.
        else:
            return pe.Constraint.Skip
    start_con_offwithinR=time.time()
    model.con_offwithinR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_offs, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gender,
                                   model.s_gen_merit_dams, model.s_groups_offs, model.s_lw_offs, rule=offwithinR, doc='transfer off to off from last dvp to current dvp.')
    end_con_offwithinR=time.time()
    # print('con_offwithinR: ',end_con_offR - start_con_offR)

def f_con_off_betweenR(model, params, l_v3, l_k3, l_k5, l_z, l_i, l_x, l_g3, l_w9_offs):
    '''
    Between year numbers/transfers of offspring to offspring in the following decision variable period.

    '''
    def offbetweenR(model,q,s9,k3,k5,v3,a,z9,i,x,y3,g3,w9):
        v3_prev = l_v3[l_v3.index(v3) - 1]  #used to get the activity number from the last period
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]  #used to get the activity number from the last period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k3 = l_k3.index(k3)
        t_k5 = l_k5.index(k5)
        t_v3 = l_v3.index(v3)
        t_z = l_z.index(z9)
        t_i = l_i.index(i)
        t_x = l_x.index(x)
        t_g3 = l_g3.index(g3)
        t_w9 = l_w9_offs.index(w9)
        if np.any(params['numbers_req_numpyversion_k3k5vw8zixg3w9'][t_k3,t_k5,t_v3,:,t_z,t_i,t_x,t_g3,t_w9]) \
           and pe.value(model.p_mask_childz_between_offs[k3,v3,z9,x,g3]) \
           and pe.value(model.p_wyear_inc_qs[q,s9]):
            ###note: dont need to multiply the child params/variables by p_mask_child because the whole constraint is skipped
            ### and the params are already masked by mask_z8 so the only bit missing is the 'between' period which is handled by skipping.
            return sum(model.v_offs[q,s9,k3,k5,t3,v3,n3,w8,z9,i,a,x,y3,g3] * model.p_numbers_req_offs[k3,k5,v3,w8,z9,i,x,g3,w9]
                       - sum(model.v_offs[q_prev,s8,k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3] * model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3,w9]
                           * model.p_parentz_provbetween_offs[k3,v3_prev,z8,x,g3,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                           + model.v_offs[q_prev,s8,k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3] * model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3,w9]
                           * model.p_parentz_provbetween_offs[k3,v3_prev,z8,x,g3,z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                             for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q,s8])!=0)
                       for t3 in model.s_sale_offs for n3 in model.s_nut_offs for w8 in model.s_lw_offs
                       if pe.value(model.p_numbers_req_offs[k3,k5,v3,w8,z9,i,x,g3,w9]) != 0
                       or any(pe.value(model.p_numbers_prov_offs[k3,k5,t3,v3_prev,n3,w8,z8,i,a,x,y3,g3,w9]) != 0 for z8 in model.s_season_types)) <=0 #need to use both in the if statement (even though it is slower) because there are situations eg dvp4 (prejoining) where prov will have a value and req will not.
        else:
            return pe.Constraint.Skip
    start_con_offbetweenR=time.time()
    model.con_offbetweenR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_offs, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gender,
                                   model.s_gen_merit_dams, model.s_groups_offs, model.s_lw_offs, rule=offbetweenR, doc='transfer off to off from last dvp to current dvp.')
    end_con_offbetweenR=time.time()
    # print('con_offbetweenR: ',end_con_offR - start_con_offR)

def f_con_dam_withinR(model, params, l_v1, l_k29, l_a, l_z, l_i, l_y1, l_g9, l_w9):
    '''
    Within year numbers/transfers of

    a) Dams to dams in the current decision variable period (only selected when a dam is changing its sire
       group e.g. BBB to BBT).
    b) Dams to dams in the following decision variable period.

    '''
    def damwithinR(model,q,s,k29,v1,a,z9,i,y1,g9,w9):
        v1_prev = l_v1[l_v1.index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k29 = l_k29.index(k29)
        t_v1 = l_v1.index(v1)
        t_a = l_a.index(a)
        t_z = l_z.index(z9)
        t_i = l_i.index(i)
        t_y1 = l_y1.index(y1)
        t_g9 = l_g9.index(g9)
        t_w9 = l_w9.index(w9)
        if np.any(params['numbers_req_numpyversion_k2k2tva1nw8ziyg1g9w9'][:,t_k29,:,t_v1,t_a,:,:,t_z,t_i,t_y1,:,t_g9,t_w9])\
           and any(pe.value(model.p_mask_childz_within_dams[k28,v1,z9,g1]) for k28 in model.s_k2_birth_dams for g1 in model.s_groups_dams)\
           and pe.value(model.p_wyear_inc_qs[q,s]):

            ###note: dont need to multiply the child params/variables by p_mask_child because the whole constraint is skipped
            ### and the params are already masked by mask_z8 so the only bit missing is the 'between' period which is handled by skipping.
            return sum(model.v_dams[q,s,k28,t1,v1,a,n1,w8,z9,i,y1,g1] * model.p_numbers_req_dams[k28,k29,t1,v1,a,n1,w8,z9,i,y1,g1,g9,w9]
                       - model.v_dams[q,s,k28,t1,v1,a,n1,w8,z9,i,y1,g1] * model.p_numbers_provthis_dams[k28,k29,t1,v1,a,n1,w8,z9,i,y1,g1,g9,w9]
                       - sum(model.v_dams[q,s,k28,t1,v1_prev,a,n1,w8,z8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z8,i,y1,g1,g9,w9]
                          * model.p_parentz_provwithin_dams[k28,v1_prev,z8,g1,z9] for z8 in model.s_season_types)
                       for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams
                       for w8 in model.s_lw_dams for g1 in model.s_groups_dams
                       if pe.value(model.p_numbers_req_dams[k28, k29, t1, v1, a, n1, w8, z9, i, y1, g1,g9, w9]) != 0
                       or pe.value(model.p_numbers_prov_dams[k28, k29, t1, v1_prev, a, n1, w8, z9, i, y1, g1, g9, w9]) #doesnt need to use z8 because in the within constraint because z only provides to itsself and children with the same w patten.
                       or pe.value(model.p_numbers_provthis_dams[k28, k29, t1, v1, a, n1, w8, z9, i, y1, g1, g9, w9]) != 0
                       ) <=0 #need to use both in the if statement (even though it is slower) because there are situations eg dvp4 (prejoining) where prov will have a value and req will not.
        else:
            return pe.Constraint.Skip

    start_con_damR=time.time()
    model.con_dam_withinR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                   model.s_groups_dams, model.s_lw_dams, rule=damwithinR, doc='transfer dam to dam from last dvp to current dvp.')
    end_con_damR=time.time()
    print('con_damwithinR: ',end_con_damR-start_con_damR)

def f_con_dam_betweenR(model, params, l_v1, l_k29, l_a, l_z, l_i, l_y1, l_g9, l_w9):
    '''
    Between year numbers/transfers of

    a) Dams to dams in the current decision variable period (only selected when a dam is changing its sire
       group e.g. BBB to BBT).
    b) Dams to dams in the following decision variable period.

    Note: this constraint is only active when season start is included as a dvp (eg not always active in steady state model).
    '''
    def dambetweenR(model,q,s9,k29,v1,a,z9,i,y1,g9,w9):
        v1_prev = l_v1[l_v1.index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster because don't need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k29 = l_k29.index(k29)
        t_v1 = l_v1.index(v1)
        t_a = l_a.index(a)
        t_z = l_z.index(z9)
        t_i = l_i.index(i)
        t_y1 = l_y1.index(y1)
        t_g9 = l_g9.index(g9)
        t_w9 = l_w9.index(w9)
        if np.any(params['numbers_req_numpyversion_k2k2tva1nw8ziyg1g9w9'][:,t_k29,:,t_v1,t_a,:,:,t_z,t_i,t_y1,:,t_g9,t_w9])\
           and any(pe.value(model.p_mask_childz_between_dams[k28,v1,z9,g1]) for k28 in model.s_k2_birth_dams for g1 in model.s_groups_dams)\
           and pe.value(model.p_wyear_inc_qs[q,s9]):
            ###note: dont need to multiply the child params/variables by p_mask_child because the whole constraint is skipped
            ### and the params are already masked by mask_z8 so the only bit missing is the 'between' period which is handled by skipping.
            return sum(model.v_dams[q,s9,k28,t1,v1,a,n1,w8,z9,i,y1,g1] * model.p_numbers_req_dams[k28,k29,t1,v1,a,n1,w8,z9,i,y1,g1,g9,w9]
                       - model.v_dams[q,s9,k28,t1,v1,a,n1,w8,z9,i,y1,g1] * model.p_numbers_provthis_dams[k28,k29,t1,v1,a,n1,w8,z9,i,y1,g1,g9,w9]
                       - sum(model.v_dams[q_prev,s8,k28,t1,v1_prev,a,n1,w8,z8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z8,i,y1,g1,g9,w9]
                           * model.p_parentz_provbetween_dams[k28,v1_prev,z8,g1,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                           + model.v_dams[q_prev,s8,k28,t1,v1_prev,a,n1,w8,z8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z8,i,y1,g1,g9,w9]
                           * model.p_parentz_provbetween_dams[k28,v1_prev,z8,g1,z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                             for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q,s8])!=0)
                       for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams
                       for w8 in model.s_lw_dams for g1 in model.s_groups_dams
                       if pe.value(model.p_numbers_req_dams[k28, k29, t1, v1, a, n1, w8, z9, i, y1, g1, g9, w9]) != 0
                       or any(pe.value(model.p_numbers_prov_dams[k28, k29, t1, v1_prev, a, n1, w8, z8, i, y1, g1, g9, w9]) != 0 for z8 in model.s_season_types)#need to use z8 because at season start all z's provide the initiating z's.
                       or pe.value(model.p_numbers_provthis_dams[k28, k29, t1, v1, a, n1, w8, z9, i, y1, g1, g9, w9]) != 0
                       ) <=0
        else:
            return pe.Constraint.Skip
    start_con_damR=time.time()
    model.con_dam_betweenR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams,
                                   model.s_groups_dams, model.s_lw_dams, rule=dambetweenR, doc='sason start - transfer dam to dam from last dvp to current dvp.')
    end_con_damR=time.time()
    print('con_dambetweenR: ',end_con_damR-start_con_damR)

def f_con_progR(model):
    '''
    Numbers/transfer of dam yatf to progeny. At weaning yatf are weaned from the dams and temporarily transferred to
    a progeny variable before being transferred to either offspring or dam variables (see prog2dams and prog2offs).

    '''
    def progR(model, q,s,k3, k5, a, z, i9, x, y1, g1, w9):
        if any(model.p_npw_req[k3, t2, x, g1] for t2 in model.s_sale_prog):
            return (- sum(model.v_dams[q,s,k5, t1, v1, a, n1, w18, z, i, y1, g1] * model.p_npw[k3, k5, t1, v1, a, n1, w18, z, i, x, y1, g1, w9, i9] #pass in the k5 set to dams - each slice of k5 aligns with a slice in k2 eg 11 and 22. we don't need other k2 slices eg nm
                          for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams for w18 in model.s_lw_dams for i in model.s_tol
                          if pe.value(model.p_npw[k3, k5, t1, v1, a, n1, w18, z, i, x, y1, g1, w9, i9])!=0)
                    + sum(model.v_prog[q,s,k3, k5, t2, w9, z, i9, a, x, g1] * model.p_npw_req[k3, t2, x, g1] for t2 in model.s_sale_prog
                          if pe.value(model.p_npw_req[k3, t2,x,g1])!=0))<=0
        else:
            return pe.Constraint.Skip
    start_con_progR = time.time()
    model.con_progR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_wean_times, model.s_season_types, model.s_tol,
                                    model.s_gender, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_prog, rule=progR,
                                   doc='transfer npw from dams to prog.')
    end_con_progR = time.time()
    # print('con_progR: ',end_con_progR-start_con_progR)

def f_con_prog2damsR(model, l_v1):
    '''
    Numbers/transfer of progeny to dams. This transfer only happens in dvp0.

    .. note:: Originally this constraint was made such that a dam required a certain proportion of single, twin &
        triplet progeny. However the requirement was the same for all initial lw patterns of the dams at weaning, this
        caused problems when scanning for multiples (which has the effect of differentiating the prog) because the
        multiples are too light to provide sufficient numbers of the high dam starting weight and the single prog are
        too heavy to provide sufficient numbers of the low dam starting weight. It would be possible to reduce the
        maximum initial dam weight and increase the minimum so it works but if we reduce the maximum weight to the
        highest weight that can be provided by triplets and increase the minimum weight to the lightest
        of the singles then we will have reduced the weight range significantly. So that removes the benefit of improving
        nutrition of dams to increase progeny weaning weight and the light progeny (that are below the lowest dam initial
        weight) can not be distributed and are therefore dropped altogether.
        The alternative implemented (3 May 21) is to sum the k2 axis for the progreq_dams constraint so that the
        total number of dams required can be supplied by progeny of any BTRT. The impact of this design is that the
        optimisation is able to select to sell the heavy singles and retain only the lighter twins as the replacements
        without incurring a lifetime penalty for the extra twin born lambs that are retained. This is not represented
        because the dams are generated using a predetermined mix of birth type - although the proportion can be altered
        prior to running the generator so you can calibrate the dams to represent a high proportion of twin birth type.
        The impact of this error will be offset by the expected higher reproductive rate of the twin born progeny.
        A further option might be to calculate the proportion of single, twins and trips required for each of the w slices
        separately based on an average of the progeny providing to each of those slices.

    .. note:: the k3 axis is summed (same as k5 discussed above) so that the proportion of replacements selected from
        maidens and adults is not fixed. However, the input on the proportion of the flock replaced is still used to mask
        whether that age group of dams (particularly the yearlings) can provide replacements.

    .. note:: A similar problem doesn’t exist for the offspring because the offspring have a k5 axis which is the
        BTRT of the animals. Therefore, the twin born progeny distribute only to twin born offspring (provided that
        the dams have been scanned to identify the twins) and there is not the problem associated with pre-determined
        proportions. However, this is not possible for the dams due to model size, including a k5 axis for the dams
        would require generating the dams with an active b0 axis and including a k5 axis in pyomo, both of which would
        significantly increase model size.


    '''
    ##k5 is a set which contains the common k slices (11,22,33) between prog and dams. It is being summed which means any b0 prog can provide a dam.
    ## the same happens for k3. See doc string for further explanation.
    def prog2damR(model, q,s,v1, z, i, y1, g9, w9):
        if v1==l_v1[0] and any(model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9] for k5 in model.s_k5_birth_offs
                               for k3 in model.s_k3_damage_offs for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams
                               for w18 in model.s_lw_dams for g1 in model.s_groups_dams):
            return (sum(- model.v_prog[q,s,k3, k5, t2, w28, z, i, a0, x, g2] * model.p_progprov_dams[k3, k5, t2, w28, z, i, a0, x, y1, g2,g9,w9]
                        for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for a0 in model.s_wean_times
                        for x in model.s_gender for w28 in model.s_lw_prog for t2 in model.s_sale_prog for g2 in model.s_groups_prog
                        if pe.value(model.p_progprov_dams[k3, k5, t2, w28, z, i, a0, x, y1, g2,g9,w9])!= 0)
                       + sum(model.v_dams[q,s,k2, t1, v1, a1, n1, w18, z, i, y1, g1]  * model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9]
                        for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams
                             for a1 in model.s_wean_times for n1 in model.s_nut_dams for w18 in model.s_lw_dams for g1 in model.s_groups_dams
                             if pe.value(model.p_progreq_dams[k2, k3, k5, t1, w18, z, i, y1, g1, g9, w9])!= 0))<=0
        else:
            return pe.Constraint.Skip
    start_con_prog2damsR = time.time()
    model.con_prog2damsR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dvp_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                         model.s_lw_dams, rule=prog2damR, doc='transfer prog to dams in dvp 0.')
    end_con_prog2damsR = time.time()
    # print('con_prog2damsR: ',end_con_prog2damsR-start_con_prog2damsR)

def f_con_prog2offsR(model, l_v3):
    '''
    Numbers/transfer of progeny to offs. This transfer only happens in dvp0.

    '''
    def prog2offsR(model,q,s, k3, k5, v3, z, i, a, x, y3, g3, w9):
        if v3==l_v3[0] and any(model.p_progreq_offs[k3, v3, w38, z, i, x, g3, w9] for w38 in model.s_lw_offs):
            return (sum(- model.v_prog[q,s,k3, k5, t2, w28, z, i, a, x, g3] * model.p_progprov_offs[k3, k5, t2, w28, z, i, a, x, y3, g3, w9] #use g3 (same as g2)
                        for w28 in model.s_lw_prog for t2 in model.s_sale_prog
                        if pe.value(model.p_progprov_offs[k3, k5, t2, w28, z, i, a, x, y3, g3, w9])!= 0)
                       + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w38, z, i,a,x,y3,g3] * model.p_progreq_offs[k3, v3, w38, z, i, x, g3, w9]
                        for t3 in model.s_sale_offs for n3 in model.s_nut_dams for w38 in model.s_lw_offs
                             if pe.value(model.p_progreq_offs[k3, v3, w38, z, i, x, g3, w9])!= 0))<=0
        else:
            return pe.Constraint.Skip
    start_con_prog2offR = time.time()
    model.con_prog2offsR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs, model.s_k5_birth_offs, model.s_dvp_dams, model.s_season_types, model.s_tol,
                                   model.s_wean_times, model.s_gender, model.s_gen_merit_dams, model.s_groups_offs, model.s_lw_offs, rule=prog2offsR,
                                   doc='transfer prog to off in dvp 0.')
    end_con_prog2offR = time.time()
    # print('con_prog2offR: ',end_con_prog2offR-start_con_prog2offR)

def f_con_matingR(model):
    '''
    Sire requirements for mating. Links the number of dams being joined during each mating period with the sire activities.
    The mating periods are necessary to represent because sires may be able to mate with more than one group of
    dams if joining of different groups is sufficiently dispersed. However, if the mating periods are close together
    the same sires may not be ready to use again. These constraints are the link between the number of sires and
    the availability of those sires in multiple periods.

    '''
    def mating(model,q,s,z,g0,p8):
        return - model.v_sire[q,s,g0] * model.p_nsires_prov[z,g0,p8] + sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_nsires_req[k2,t1,v1,a,n1,w1,z,i,y1,g1,g0,p8]
                  for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for a in model.s_wean_times for n1 in model.s_nut_dams
                   for w1 in model.s_lw_dams for i in model.s_tol for y1 in model.s_gen_merit_dams  for g1 in model.s_groups_dams
                   if pe.value(model.p_nsires_req[k2,t1,v1,a,n1,w1,z,i,y1,g1,g0,p8])!=0) <=0
    model.con_matingR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_groups_sire, model.s_sire_periods, rule=mating, doc='sire requirement for mating')

def f_con_stockinfra(model):
    '''
    Ensures enough infrastructure exists for all the animals and the associated events (eg mustering, shearing, etc).
    '''
    def stockinfra(model,q,s,h1,z):
        return -model.v_infrastructure[q,s,h1,z] + sum(model.v_sire[q,s,g0] * model.p_infra_sire[h1,z,g0] for g0 in model.s_groups_sire if model.p_infra_sire[h1,z,g0]!=0)  \
               + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_infra_dams[k2,h1,t1,v1,a,n1,w1,z,i,y1,g1]
                         for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                         for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                         if pe.value(model.p_infra_dams[k2,h1,t1,v1,a,n1,w1,z,i,y1,g1])!=0)
                    + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_infra_offs[k3,k5,h1,t3,v3,n3,w3,z,i,a,x,y3,g3]
                          for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                          for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                          if pe.value(model.p_infra_offs[k3,k5,h1,t3,v3,n3,w3,z,i,a,x,y3,g3])!=0)
               for a in model.s_wean_times for i in model.s_tol) <=0
    model.con_stockinfra = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_infrastructure, model.s_season_types, rule=stockinfra, doc='Requirement for infrastructure (based on number of times yarded and shearing activity)')

    end_cons=time.time()
    # print('time stock con: ', end_cons-end_params)

    ##write constraint to text file
    # textbuffer = StringIO()
    # model.con_offR.pprint(textbuffer)
    # textbuffer.write('\n')
    # with open('cons.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())


##################################
### setup core model constraints #
##################################

def f_stock_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy required by livestock in each nv pool in each feed period.

    Used in global constraint (con_me). See CorePyomo
    '''

    return sum(model.v_sire[q,s,g0] * model.p_mei_sire[p6,f,z,g0] for g0 in model.s_groups_sire)\
           + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_mei_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if pe.value(model.p_mei_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_mei_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_mei_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)


def f_stock_pi(model,q,s,p6,f,z):
    '''
    Calculate the total volume provided by livestock in each nv pool in each feed period.

    Used in global constraint (con_vol). See CorePyomo
    '''

    return sum(model.v_sire[q,s,g0] * model.p_pi_sire[p6,f,z,g0] for g0 in model.s_groups_sire)\
           + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_pi_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if pe.value(model.p_pi_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_pi_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_pi_offs[k3,k5,p6,f,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)

def f_stock_cashflow(model,q,s,c0,p7,z):
    '''
    Calculate the net cashflow (income - expenses) of livestock and their associated activities.

    Used in global constraint (con_cashflow). See CorePyomo
    '''

    infrastructure = sum(model.p_rm_stockinfra_fix[h1,c0,p7,z] + model.p_rm_stockinfra_var[h1,c0,p7,z] * model.v_infrastructure[q,s,h1,z]
                         for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_cashflow_sire[c0,p7,z,g0] for g0 in model.s_groups_sire) \
           + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_cashflow_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]
                      for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                      for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if pe.value(model.p_cashflow_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_prog[q,s,k3, k5, t2, w2, z, i, a, x, g2] * model.p_cashflow_prog[k3, k5, c0,p7, t2, w2, z, i, a, x, g2]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t2 in model.s_sale_prog for w2 in model.s_lw_prog
                      for x in model.s_gender for g2 in model.s_groups_prog if model.p_cashflow_prog[k3, k5, c0,p7, t2, w2, z, i, a, x, g2] != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_cashflow_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_cashflow_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    purchases = sum(model.v_sire[q,s,g0] * model.p_cost_purch_sire[c0,p7,z,g0] for g0 in model.s_groups_sire)
    return stock - infrastructure - purchases

#     purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[g0,c] for g0 in model.s_groups_sire)  \
#                 + sum(sum(model.v_purchase_dams[v1,w1,i,g1] * model.p_cost_purch_dam[v1,w1,i,g1,c] for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
#                     + sum(model.v_purchase_offs[v3,w3,i,g3] * model.p_cost_purch_offs[v3,w3,i,g3,c] for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
#                     for z in model.s_season_types for i in model.s_tol)
#     return stock - infrastructure - purchases

def f_stock_wc(model,q,s,c0,p7,z):
    '''
    Calculate the net wc (income - expenses) of livestock and their associated activities.

    Used in global constraint (con_wc). See CorePyomo
    '''

    infrastructure = sum(model.p_rm_stockinfra_fix[h1,c0,p7,z] + model.p_rm_stockinfra_var[h1,c0,p7,z] * model.v_infrastructure[q,s,h1,z]
                         for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_wc_sire[c0,p7,z,g0] for g0 in model.s_groups_sire) \
           + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_wc_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]
                      for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                      for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                     if pe.value(model.p_wc_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_prog[q,s,k3, k5, t2, w2, z, i, a, x, g2] * model.p_wc_prog[k3, k5, c0,p7, t2, w2, z, i, a, x, g2]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t2 in model.s_sale_prog for w2 in model.s_lw_prog
                      for x in model.s_gender for g2 in model.s_groups_prog if model.p_wc_prog[k3, k5, c0,p7, t2, w2, z, i, a, x, g2] != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_wc_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_wc_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    purchases = sum(model.v_sire[q,s,g0] * model.p_wc_purch_sire[c0,p7,z,g0] for g0 in model.s_groups_sire)
    return stock - infrastructure - purchases

#     purchases = sum(model.v_sire[g0] * model.p_wc_purch_sire[g0,c] for g0 in model.s_groups_sire)  \
#                 + sum(sum(model.v_purchase_dams[v1,w1,i,g1] * model.p_wc_purch_dam[v1,w1,i,g1,c] for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
#                     + sum(model.v_purchase_offs[v3,w3,i,g3] * model.p_wc_purch_offs[v3,w3,i,g3,c] for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
#                     for z in model.s_season_types for i in model.s_tol)
#     return stock - infrastructure - purchases


def f_stock_cost(model,q,s,c0,p7,z):
    '''
    Calculate the total cost of livestock (husbandry & infrastructure).

    Used in global constraint (con_minroe). See CorePyomo
    '''

    infrastructure = sum(model.p_rm_stockinfra_fix[h1,c0,p7,z] + model.p_rm_stockinfra_var[h1,c0,p7,z] * model.v_infrastructure[q,s,h1,z]
                         for h1 in model.s_infrastructure for c0 in model.s_enterprises)
    stock = sum(model.v_sire[q,s,g0] * model.p_cost_sire[c0,p7,z,g0] for g0 in model.s_groups_sire) \
            + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_cost_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if pe.value(model.p_cost_dams[k2,c0,p7,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_cost_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_cost_offs[k3,k5,c0,p7,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    purchases = sum(model.v_sire[q,s,g0] * model.p_cost_purch_sire[c0,p7,z,g0]
                    for g0 in model.s_groups_sire for c0 in model.s_enterprises)
    return  stock + infrastructure + purchases
#
#
def f_stock_labour_anyone(model,q,s,p5,z):
    '''
    Calculate the total 'anyone' labour required for livestock activities.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_lab_anyone_sire[p5,z,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_anyone_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if pe.value(model.p_lab_anyone_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_anyone_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_lab_anyone_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    return stock

def f_stock_labour_perm(model,q,s,p5,z):
    '''
    Calculate the total 'permanent' labour required for livestock activities.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_lab_perm_sire[p5,z,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_perm_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if pe.value(model.p_lab_perm_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_perm_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_lab_perm_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    return stock

def f_stock_labour_manager(model,q,s,p5,z):
    '''
    Calculate the total 'manager' labour required for livestock activities.

    Used in global constraint (con_labour_any). See CorePyomo
    '''

    # infrastructure = sum(model.p_lab_stockinfra[h1,p5] * model.v_infrastructure[h1,p5] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_lab_manager_sire[p5,z,g0] for g0 in model.s_groups_sire)\
            + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_lab_manager_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if pe.value(model.p_lab_manager_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_lab_manager_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_lab_manager_offs[k3,k5,p5,t3,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    return stock
#


def f_stock_asset(model,q,s,p7,z):
    '''
    Calculate the total asset value of livestock.

    Used in global constraint (con_asset). See CorePyomo
    '''

    infrastructure = sum(model.p_asset_stockinfra[p7,h1] for h1 in model.s_infrastructure)
    stock = sum(model.v_sire[q,s,g0] * model.p_asset_sire[p7,z,g0] for g0 in model.s_groups_sire) \
            + sum(sum(model.v_dams[q,s,k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_asset_dams[k2,t1,p7,v1,a,n1,w1,z,i,y1,g1]
                     for k2 in model.s_k2_birth_dams for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for n1 in model.s_nut_dams
                     for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams
                      if pe.value(model.p_asset_dams[k2,t1,p7,v1,a,n1,w1,z,i,y1,g1]) != 0)
                + sum(model.v_offs[q,s,k3,k5,t3,v3,n3,w3,z,i,a,x,y3,g3]  * model.p_asset_offs[k3,k5,t3,p7,v3,n3,w3,z,i,a,x,y3,g3]
                      for k3 in model.s_k3_damage_offs for k5 in model.s_k5_birth_offs for t3 in model.s_sale_offs for v3 in model.s_dvp_offs
                      for n3 in model.s_nut_offs for w3 in model.s_lw_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs
                      if pe.value(model.p_asset_offs[k3,k5,t3,p7,v3,n3,w3,z,i,a,x,y3,g3]) != 0)
               for a in model.s_wean_times for i in model.s_tol)
    # purchases = sum(sum(model.v_purchase_dams[q,s,v1,w1,i,g1] * sum(model.p_cost_purch_dam[v1,w1,i,g1,c] for c in model.s_season_periods) for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
    #                 +sum(model.v_purchase_offs[q,s,v3,w3,i,g3] * sum(model.p_cost_purch_offs[v3,w3,i,g3,c] for c in model.s_season_periods) for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
    return stock + infrastructure #+ purchases


##################################
# old methods used to find speed #
##################################

    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR(model,k29,v1,a,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     con = sum(model.v_dams[k28,t1,v1,a,n1,w8,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] !=0 or model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9] !=0) <= 0
    #         # + sum(model.v_dams2sire[v1,a,b1,n1,w1,i,y1,g1,g1_new]
    #         #       - model.v_dams2sire[v1_prev,a,b1,n1,w1,i,y1,g1,g1_new] * model.p_dam2sire_numbers[v1,a,b1,n1,w1,i,y1,g1,g1_new]
    #         #       for n1 in model.s_nut_dams for g1_new in model.s_groups_dams) \
    #         # - model.v_purchase_dams[v1,w1,i,g1] * model.p_numberpurch_dam[v1,a,b1,w1,i,y1,g1] \ #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
    #         # - sum(model.v_offs2dam[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off] * model.p_offs2dam_numbers[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off,v1,a,b1,w1,i,y1,g1]
    #         #       for v3 in model.s_dvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for z3 in model.s_season_types for i3 in model.s_tol for d in model.s_k3_damage_offs for a3 in model.s_wean_times
    #         #       for b3 in model.s_k5_birth_offs for x in model.s_gender for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs for g1_off in model.s_groups_dams)  #have to track off sets so only they are summed.
    #     ###if statement required to handle the constraints that don't exist due to lw clustering
    #     # if sum(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] !=0) ==0 and sum(model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9]
    #     #         for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9] !=0)==0:
    #     #     pass
    #     if type(con)==bool:
    #         return pe.Constraint.Skip
    #     else: return con

    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR1(model,k29,v1,a,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     con = sum(model.v_dams[k28,t1,v1,a,n1,w8,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] !=0) <= 0
    #     if type(con)==bool:
    #         return pe.Constraint.Skip
    #     else: return con
    # start=time.time()
    # # model.con_damR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR1, doc='transfer of off to dam and dam from last dvp to current dvp.')
    # end=time.time()
    # print('method 1: ',end-start)

    # ##method 2
    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR2(model,k29,v1,a,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     ##skip constraint if the require param is 0
    #     if not any(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams):
    #         return pe.Constraint.Skip
    #     return sum(model.v_dams[k28,t1,v1,a,n1,w8,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_k2_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,i,y1,g1,w9] !=0) <= 0
    # start=time.time()
    # model.con_damR = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR2, doc='transfer of off to dam and dam from last dvp to current dvp.')
    # end=time.time()
    # print('method 2: ',end-start)



