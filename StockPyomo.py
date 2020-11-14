# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:35 2020

@author: John
"""

#python modules
import pyomo.environ as pe
import time
from io import StringIO
import numpy as np

#MUDAS modules
from CreateModel import model
import StockGenerator as sgen






def sheep_pyomo_local(params,report):


    sgen.generator(params,report)

    ##these sets require info from the stock module
    model.s_wean_times = pe.Set(initialize=params['a_idx'], doc='weaning options')
    model.s_tol = pe.Set(initialize=params['i_idx'], doc='birth groups (times of lambing)')
    model.s_sire_periods = pe.Set(initialize=params['p8_idx'], doc='sire periods')
    model.s_groups_sire = pe.Set(initialize=params['g_idx_sire'], doc='geneotype groups of sires')
    model.s_gen_merit_sire = pe.Set(initialize=params['y_idx_sire'], doc='genetic merit of sires')
    model.s_birth_dams = pe.Set(initialize=params['k2_idx_dams'], doc='Cluster for LSLN & oestrus cycle based on scanning, global & weaning management')
    model.s_dvp_dams = pe.Set(ordered=True, initialize=params['dvp_idx_dams'], doc='Decision variable periods for dams') #ordered so they can be indexed in constraint to determine previous period
    model.s_groups_dams = pe.Set(initialize=params['g_idx_dams'], doc='geneotype groups of dams')
    model.s_gen_merit_dams = pe.Set(initialize=params['y_idx_dams'], doc='genetic merit of dams')
    model.s_sale_dams = pe.Set(initialize=params['t_idx_dams'], doc='Sales within the year for damss')
    model.s_dvp_offs = pe.Set(ordered=True, initialize=params['dvp_idx_offs'], doc='Decision variable periods for offs') #ordered so they can be indexed in constraint to determine previous period
    model.s_damage_offs = pe.Set(initialize=params['k3_idx_offs'], doc='age of mother - offs')
    model.s_birth_offs = pe.Set(initialize=params['k5_idx_offs'], doc='Cluster for BTRT & oestrus cycle based on scanning, global & weaning management')
    model.s_groups_offs = pe.Set(initialize=params['g_idx_offs'], doc='geneotype groups of offs')
    model.s_gen_merit_offs = pe.Set(initialize=params['y_idx_offs'], doc='genetic merit of offs')
    model.s_gender_offs = pe.Set(initialize=params['x_idx_offs'], doc='gender of offs')

    #####################
    ##  setup variables # #variables that use dynamic sets must be defined each itteration of exp
    #####################
    print('set up variables')
    ##animals
    model.v_sire = pe.Var(model.s_groups_sire, bounds = (0,None) , doc='number of sire animals')
    model.v_dams = pe.Var(model.s_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, bounds = (0,None) , doc='number of dam animals')
    model.v_offs = pe.Var(model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types,
                          model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs,
                          model.s_groups_offs, bounds = (0,None) , doc='number of offs animals')
    ##animal transfers
    model.v_offs2dam = pe.Var(model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types,
                              model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs,
                              model.s_groups_offs, model.s_groups_dams, bounds = (0,None) , doc='transfer of animals from the offspring variables to the dam variables.')
    model.v_dams2sire = pe.Var(model.s_dvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams,
                               model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_sire,
                               bounds = (0,None) , doc='transfer of animals from the dam variables of one sire genotype to another sire genotype')


    ##purchases
    model.v_purchase_dams = pe.Var(model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, bounds = (0,None) , doc='number of purchased dam animals')
    model.v_purchase_offs = pe.Var(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, bounds = (0,None) , doc='number of purchased offs animals')
    ##mating
    # model.v_sirecap = pe.Var(model.s_groups_sire, model.s_sire_periods, bounds = (0,None) , doc='number of sire animals')

    ######################
    ### setup parameters #
    ######################
    print('set up params')
    param_start = time.time()

    ##stock - dams
    try:
        model.del_component(model.p_numbers_prov_dams_index)
        model.del_component(model.p_numbers_prov_dams)
    except AttributeError:
        pass
    model.p_numbers_prov_dams = pe.Param(model.s_birth_dams, model.s_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_prov_dams'], default=0.0, doc='numbers provided by each dam activity into the next period')

    try:
        model.del_component(model.p_numbers_req_dams_index)
        model.del_component(model.p_numbers_req_dams)
    except AttributeError:
        pass
    model.p_numbers_req_dams = pe.Param(model.s_birth_dams, model.s_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams, model.s_lw_dams,
                                         initialize=params['p_numbers_req_dams'], default=0.0, doc='numbers required by each dam activity in the current period')

    # model.p_npw = Param(model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams,
    #                           model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_dvp_offs, model.s_lw_offs,
    #                           model.s_season_types, model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs,
    #                           model.s_gen_merit_offs, model.s_groups_offs,
    #                           initialize=, default=0.0, doc='number of prodgeny weaned in each off class')

    model.p_n_sires = pe.Param(model.s_birth_dams, model.s_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                               model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_sire, model.s_sire_periods,
                               initialize=params['p_nsire_dams'], default=0.0, doc='requirement for sires for mating')

    ##stock - offs
    model.p_numbers_prov_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol,
                                 model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, model.s_lw_offs,
                                 initialize=params['p_numbers_prov_offs'], default=0.0, doc='numbers provided into the current period from the previous periods activities')
    model.p_numbers_req_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_lw_offs,
                                        model.s_groups_offs, model.s_lw_offs,
                                        initialize=params['p_numbers_req_offs'], default=0.0, doc='requirment of off in the current period')

    ##energy intake
    try:
        model.del_component(model.p_mei_sire)
    except AttributeError:
        pass
    model.p_mei_sire = pe.Param(model.s_feed_periods, model.s_sheep_pools, model.s_groups_sire, initialize=params['p_mei_sire'], 
                                  default=0.0, doc='energy requirement sire')
    try:
        model.del_component(model.p_mei_dams)
    except AttributeError:
        pass
    model.p_mei_dams = pe.Param(model.s_birth_dams, model.s_feed_periods, model.s_sheep_pools, model.s_sale_dams, 
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_mei_dams'], default=0.0, doc='energy requirement dams')
    try:
        model.del_component(model.p_mei_offs)
    except AttributeError:
        pass
    model.p_mei_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_feed_periods, model.s_sheep_pools, model.s_sale_offs, 
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times, 
                               model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_mei_offs'], default=0.0, doc='energy requirement offs')

    ##potential intake
    try:
        model.del_component(model.p_pi_sire)
    except AttributeError:
        pass
    model.p_pi_sire = pe.Param(model.s_feed_periods, model.s_sheep_pools, model.s_groups_sire, initialize=params['p_pi_sire'], 
                                  default=0.0, doc='pi sire')
    try:
        model.del_component(model.p_pi_dams)
    except AttributeError:
        pass
    model.p_pi_dams = pe.Param(model.s_birth_dams, model.s_feed_periods, model.s_sheep_pools, model.s_sale_dams, 
                               model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams, model.s_season_types, 
                               model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, initialize=params['p_pi_dams'], default=0.0, doc='pi dams')
    try:
        model.del_component(model.p_pi_offs)
    except AttributeError:
        pass
    model.p_pi_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_feed_periods, model.s_sheep_pools, model.s_sale_offs, 
                               model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_wean_times, 
                               model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, initialize=params['p_pi_offs'], default=0.0, doc='pi offs')

    
    ##cashflow
    try:
        model.del_component(model.p_cashflow_sire)
    except AttributeError:
        pass
    model.p_cashflow_sire = pe.Param(model.s_cashflow_periods, model.s_groups_sire, initialize=params['p_cashflow_sire'], 
                                  default=0.0, doc='cashflow sire')
    try:
        model.del_component(model.p_cashflow_dams)
    except AttributeError:
        pass
    model.p_cashflow_dams = pe.Param(model.s_birth_dams, model.s_cashflow_periods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cashflow_dams'], default=0.0, doc='cashflow dams')
    try:
        model.del_component(model.p_cashflow_offs)
    except AttributeError:
        pass
    model.p_cashflow_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_cashflow_periods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_cashflow_offs'], default=0.0, doc='cashflow offs')
    ##cost - minroe
    try:
        model.del_component(model.p_cost_sire)
    except AttributeError:
        pass
    model.p_cost_sire = pe.Param(model.s_groups_sire, initialize=params['p_cost_sire'], 
                                  default=0.0, doc='husbandry cost sire')
    try:
        model.del_component(model.p_cost_dams)
    except AttributeError:
        pass
    model.p_cost_dams = pe.Param(model.s_birth_dams, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_cost_dams'], default=0.0, doc='husbandry cost dams')
    try:
        model.del_component(model.p_cost_offs)
    except AttributeError:
        pass
    model.p_cost_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_cost_offs'], default=0.0, doc='husbandry cost offs')

    ##labour - sire
    try:
        model.del_component(model.p_lab_anyone_sire)
    except AttributeError:
        pass
    model.p_lab_anyone_sire = pe.Param(model.s_labperiods, model.s_groups_sire,  initialize=params['p_labour_anyone_sire'], default=0.0, doc='labour requirment sire that can be done by anyone')
    try:
        model.del_component(model.p_lab_perm_sire)
    except AttributeError:
        pass
    model.p_lab_perm_sire = pe.Param(model.s_labperiods, model.s_groups_sire, initialize=params['p_labour_perm_sire'], default=0.0, doc='labour requirment sire that can be done by perm staff')
    try:
        model.del_component(model.p_lab_manager_sire)
    except AttributeError:
        pass
    model.p_lab_manager_sire = pe.Param(model.s_labperiods, model.s_groups_sire, initialize=params['p_labour_manager_sire'], default=0.0, doc='labour requirment sire that can be done by manager')
    
    ##labour - dams
    try:
        model.del_component(model.p_lab_anyone_dams)
    except AttributeError:
        pass
    model.p_lab_anyone_dams = pe.Param(model.s_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                       initialize=params['p_labour_anyone_dams'], default=0.0, doc='labour requirment dams that can be done by anyone')
    try:
        model.del_component(model.p_lab_perm_dams)
    except AttributeError:
        pass
    model.p_lab_perm_dams = pe.Param(model.s_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                     initialize=params['p_labour_perm_dams'], default=0.0, doc='labour requirment dams that can be done by perm staff')
    try:
        model.del_component(model.p_lab_manager_dams)
    except AttributeError:
        pass
    model.p_lab_manager_dams = pe.Param(model.s_birth_dams, model.s_labperiods, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, model.s_lw_dams,
                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                        initialize=params['p_labour_manager_dams'], default=0.0, doc='labour requirment dams that can be done by manager')
    
    ##labour - offs
    try:
        model.del_component(model.p_lab_anyone_offs)
    except AttributeError:
        pass
    model.p_lab_anyone_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_anyone_offs'], default=0.0, doc='labour requirment offs - anyone')
    try:
        model.del_component(model.p_lab_perm_offs)
    except AttributeError:
        pass
    model.p_lab_perm_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_perm_offs'], default=0.0, doc='labour requirment offs - perm')
    try:
        model.del_component(model.p_lab_manager_offs)
    except AttributeError:
        pass
    model.p_lab_manager_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_labperiods, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_labour_manager_offs'], default=0.0, doc='labour requirment offs - manager')

    ##infrastructure
    try:
        model.del_component(model.p_infra_sire)
    except AttributeError:
        pass
    model.p_infra_sire = pe.Param(model.s_infrastructure, model.s_groups_sire, initialize=params['p_infrastructure_sire'], 
                                  default=0.0, doc='sire requirement for infrastructure (based on number of times yarded and shearing activity)')
    try:
        model.del_component(model.p_infra_dams)
    except AttributeError:
        pass
    model.p_infra_dams = pe.Param(model.s_birth_dams, model.s_infrastructure, model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_nut_dams, 
                                  model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                                  initialize=params['p_infrastructure_dams'], default=0.0, doc='Dams requirement for infrastructure (based on number of times yarded and shearing activity)')
    try:
        model.del_component(model.p_infra_offs)
    except AttributeError:
        pass
    model.p_infra_offs = pe.Param(model.s_damage_offs, model.s_birth_offs, model.s_infrastructure, model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs,
                             model.s_season_types, model.s_tol, model.s_wean_times, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                             initialize=params['p_infrastructure_offs'], default=0.0, doc='offs requirement for infrastructure (based on number of times yarded and shearing activity)')


    # try:
    #     model.del_component(model.p_asset_stockinfra)
    # except AttributeError:
    #     pass
    # model.p_asset_stockinfra = Param(model.s_infrastructure, initialize=, default=0.0, doc='Asset value per animal mustered  or shorn')
    # try:
    #     model.del_component(model.p_dep_stockinfra)
    # except AttributeError:
    #     pass
    # model.p_dep_stockinfra = Param(model.s_infrastructure, initialize=, default=0.0, doc='Depreciation of the asset value')
    # try:
    #     model.del_component(model.p_rm_stockinfra)
    # except AttributeError:
    #     pass
    # model.p_rm_stockinfra = Param(model.s_infrastructure, model.s_cashflow_periods,initialize=, default=0.0, doc='Cost of R&M of the infrastructure (per animal mustered/shorn)')
    # try:
    #     model.del_component(model.p_lab_stockinfra)
    # except AttributeError:
    #     pass
    # model.p_lab_stockinfra = Param(model.s_infrastructure, model.s_labperiods, initialize=, default=0.0, doc='Labour required for R&M of the infrastructure (per animal mustered/shorn)')

    # try:
    #     model.del_component(model.p_asset_sire)
    # except AttributeError:
    #     pass
    # model.p_asset_sire = Param(model.s_groups_sire, initialize=, default=0.0, doc='Asset value of sire')
    # try:
    #     model.del_component(model.p_asset_dams)
    # except AttributeError:
    #     pass
    # model.p_asset_dams = Param(model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams,
    #                            model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception,
    #                            model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default=0.0, doc='Asset value of dams')
    # try:
    #     model.del_component(model.p_asset_offs)
    # except AttributeError:
    #     pass
    # model.p_asset_offs = Param(model.s_sale_offs, model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types,
    #                            model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs,
    #                            model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default=0.0, doc='Asset value of offs')


    ##purchases
    # model.p_cost_purch_sire = Param(model.s_groups_sire, model.s_cashflow_periods,
    #                                initialize=, default=0.0, doc='cost of purchased sires')
    # model.p_numberpurch_dam = Param(model.s_dvp_dams, model.s_wean_times, model.s_birth_dams, model.s_lw_dams,
    #                           model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception,
    #                           model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default=0.0, doc='purchase transfer - ie how a purchased dam is allocated into damR')
    # model.p_cost_purch_dam = Param(model.s_dvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, model.s_cashflow_periods,
    #                                initialize=, default=0.0, doc='cost of purchased dams')
    # model.p_numberpurch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs,
    #                                  model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
    #                                  initialize=, default=0.0, doc='purchase transfer - ie how a purchased offs is allocated into offsR')
    # model.p_cost_purch_offs = Param(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, model.s_cashflow_periods,
    #                                initialize=, default=0.0, doc='cost of purchased offs')
    # ##transfers
    # model.p_offs2dam_numbers = Param(model.s_dvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs,
    #                                  model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_co_cfw,
    #                                  model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_dams, model.s_dvp_dams, model.s_wean_times,
    #                                  model.s_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
    #                                  initialize=, default=0.0, doc='Proportion of the offs distributed to each of the starting LWs at the beginning of the current dam feed variation period')
    # model.p_dam2sire_numbers = Param(model.s_dvp_dams, model.s_wean_times, model.s_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol,
    #                                  model.s_gen_merit_dams, model.s_groups_dams, model.s_groups_dams,
    #                                  initialize=, default=0.0, doc='Proportion of the animals distributed to each of the starting LWs of the recipient animals at the beginning of the recipients next feed variation period')

    textbuffer = StringIO()
    model.p_numbers_prov_dams.pprint(textbuffer)
    textbuffer.write('\n')
    with open('number_prov.txt', 'w') as outputfile:
        outputfile.write(textbuffer.getvalue())


    end_params = time.time()
    print('params time: ',end_params-param_start)

    ########################
    ### set up constraints #
    ########################
    print('set up constraints')

    # try:
    #     model.del_component(model.con_damR_index)
    #     model.del_component(model.con_damR)
    # except AttributeError:
    #     pass
    # def damR(model,k29,v1,a,z,i,y1,g1,w9):
    #     v1_prev = list(model.s_dvp_dams)[list(model.s_dvp_dams).index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
    #     con = sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0 or model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    #         # + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,g1_new]
    #         #       - model.v_dams2sire[v1_prev,a,b1,n1,w1,z,i,y1,g1,g1_new] * model.p_dam2sire_numbers[v1,a,b1,n1,w1,z,i,y1,g1,g1_new]
    #         #       for n1 in model.s_nut_dams for g1_new in model.s_groups_dams) \
    #         # - model.v_purchase_dams[v1,w1,z,i,g1] * model.p_numberpurch_dam[v1,a,b1,w1,z,i,y1,g1] \ #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
    #         # - sum(model.v_offs2dam[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off] * model.p_offs2dam_numbers[v3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,g1_off,v1,a,b1,w1,z,i,y1,g1]
    #         #       for v3 in model.s_dvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for z3 in model.s_season_types for i3 in model.s_tol for d in model.s_damage_offs for a3 in model.s_wean_times
    #         #       for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs for g1_off in model.s_groups_dams)  #have to track off sets so only they are summed.
    #     ###if statement required to handle the constraints that dont exist due to lw clustering
    #     # if sum(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) ==0 and sum(model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #     #         for t1 in model.s_sale_dams for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams if model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9] !=0)==0:
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
    #                for t1 in model.s_sale_dams for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    #     if type(con)==bool:
    #         return pe.Constraint.Skip
    #     else: return con
    # start=time.time()
    # # model.con_damR = pe.Constraint(model.s_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR1, doc='transfer of off to dam and dam from last dvp to current dvp.')
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
    #     if not any(model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams):
    #         return pe.Constraint.Skip
    #     return sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
    #                - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
    #                for t1 in model.s_sale_dams for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
    #                if model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9] !=0) <= 0
    # start=time.time()
    # model.con_damR = pe.Constraint(model.s_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR2, doc='transfer of off to dam and dam from last dvp to current dvp.')
    # end=time.time()
    # print('method 2: ',end-start)

    ##method 3
    ##info:
    ##constraint.skip is fast, the trick is designing the code efficiently so that is knows when to skip.
    ##in method 2 i use the param to determine when the constraint should be skipped, this still requires looping throught the param
    ##in method 3 i use the numpy array to determine when the constraint should be skipped. This is messier and requires some extra code but it is much more efficient reducing time 2x.
    ##for method 3 1second or less is spent skipping the constraints the remaining 10seconds is the time taken to build the remaining 900 constraints.
    ##to save any further time will require making the building of the constraint faster. however i cant think of a way to do this because i am already including if statements for params with 0 value.
    ##significant time can be saved by using if statements that only evaluate one item
    l_k29 = list(model.s_birth_dams)
    l_v1 = list(model.s_dvp_dams)
    l_a = list(model.s_wean_times)
    l_z = list(model.s_season_types)
    l_i = list(model.s_tol)
    l_y1 = list(model.s_gen_merit_dams)
    l_g1 = list(model.s_groups_dams)
    l_g9 = list(model.s_groups_dams)
    l_w9 = list(model.s_lw_dams)
    try:
        model.del_component(model.con_damR_index)
        model.del_component(model.con_damR)
    except AttributeError:
        pass
    def damR3(model,k29,v1,a,z,i,y1,g1,g9,w9):
        v1_prev = l_v1[l_v1.index(v1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period
        ##skip constraint if the require param is 0 - using the numpy array because it is 2x faster becasue dont need to loop through activity keys eg k28
        ###get the index number - required so numpy array can be indexed
        t_k29 = l_k29.index(k29)
        t_v1 = l_v1.index(v1)
        t_a = l_a.index(a)
        t_z = l_z.index(z)
        t_i = l_i.index(i)
        t_y1 = l_y1.index(y1)
        t_g1 = l_g1.index(g1)
        t_g9 = l_g9.index(g9)
        t_w9 = l_w9.index(w9)
        if not np.any(params['numbers_req_numpyvesion_k2k2tva1nw8ziyg1g9w9'][:,t_k29,:,t_v1,t_a,:,:,t_z,t_i,t_y1,t_g1,t_g9,t_w9]):
            return pe.Constraint.Skip
        return sum(model.v_dams[k28,t1,v1,a,n1,w8,z,i,y1,g1] * model.p_numbers_req_dams[k28,k29,v1,a,n1,w8,z,i,y1,g1,w9]
                   - model.v_dams[k28,t1,v1_prev,a,n1,w8,z,i,y1,g1] * model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9]
                    for t1 in model.s_sale_dams for k28 in model.s_birth_dams
                    for n1 in model.s_nut_dams for w8 in model.s_lw_dams if
                    model.p_numbers_req_dams[k28, k29, v1, a, n1, w8, z, i, y1, g1, w9] != 0) <=0
                   # for t1 in model.s_sale_dams for k28 in model.s_birth_dams for n1 in model.s_nut_dams for w8 in model.s_lw_dams
                   # if model.p_numbers_prov_dams[k28,k29,t1,v1_prev,a,n1,w8,z,i,y1,g1,w9] !=0)) <= 0
    start=time.time()
    model.con_damR = pe.Constraint(model.s_birth_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_lw_dams, rule=damR3, doc='transfer of off to dam and dam from last dvp to current dvp.')
    end=time.time()
    print('method 3: ',end-start)

    end_cons=time.time()
    print('time con: ', end_cons-end_params)

    # model.con_damR.pprint(textbuffer)
    # textbuffer.write('\n')
    # with open('con_damR.txt', 'w') as outputfile:
    #     outputfile.write(textbuffer.getvalue())



    # try:
    #     model.del_component(model.con_offsR)
    # except AttributeError:
    #     pass
    # def offsR(model,k3,k5,v3,w3,z,i,d,a,b3,x,y3,g3):
    #     f3_prev = list(model.s_dvp_offs)[list(model.s_dvp_offs).index(v3) - 1]  #used to get the activity number from the last period - to determine the number of off provided into this period
    #     return sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,g1] for n3 in model.s_nut_offs for g1 in model.s_groups_dams) \
    #         + sum(model.v_offs[k3,k5,t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3]
    #                - model.v_offs[k3,k5,t3,f3_prev,n3,w3,z,i,d,a,b3,x,y3,g3] * model.p_numbers_prov_offs[k3,k5,t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3]
    #                for t3 in model.s_sale_offs for n3 in model.s_nut_offs) \
    #         - model.v_purchase_offs[v3,w3,z,i,g3] * model.p_numberpurch_offs[v3,w3,z,i,d,a,b3,x,y3,g3] \ #p_numpurch allocates the purchased offs into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
    #         - sum(model.v_dams[t1,v1,a1,b1,n1,w1,z1,i1,y1,g1] * model.p_npw[t1,v1,a1,b1,n1,w1,z1,i1,y1,g1,v3,w3,z,i,d,a,b3,x,y3,g3] #have to distinguish between a1 and a3 so only the a from dams is summed.
    #                   for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for a1 in model.s_wean_times for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for z1 in model.s_season_types for i1 in model.s_tol for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)   <=0
    # model.con_offsR = pe.Constraint(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs,
    #                                  model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
    #                                  model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, rule=offsR, doc='transfer of off to next dvp.')
    try:
        model.del_component(model.con_offsR)
    except AttributeError:
        pass
    def sireR(model,g0):
        return model.v_sire[g0]
    model.con_offsR = pe.Constraint(model.s_dvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs,
                                     model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs,
                                     model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, rule=offsR, doc='transfer of off to next dvp.')
    #
    #
    # try:
    #     model.del_component(model.con_matingR)
    # except AttributeError:
    #     pass
    # def mating(model,g0,p8):
    #     return sum(sum(model.v_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for t1 in model.s_sale_dams for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
    #              * p_n_sires[v1,b1,g1,g0,p8] for v1 in model.s_dvp_dams for b1 in model.s_birth_dams for g1 in model.s_groups_dams) \
    #         - model.v_sire[g0] <=0 #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
    # model.con_matingR = pe.Constraint(model.s_groups_sire, model.s_sire_periods, rule=mating, doc='sire requirment for mating')
    #
    try:
        model.del_component(model.con_stockinfra)
    except AttributeError:
        pass
    def stockinfra(model,h1):
        return -model.v_infrastructure[h1] + sum(model.v_sire[g0] * model.p_infra_sire[g0,h1] for g0 in model.s_groups_sire)  \
               + sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams
                             * model.p_infra_dams[t1,v1,b1,z,i,g1,h1] for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for b1 in model.s_birth_dams for g1 in model.s_groups_dams)  \
               + sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) * model.p_infra_offs[t3,v3,b3,z,i,g3,h3] for t3 in model.s_sale_offs for v3 in model.s_dvp_offs for b3 in model.s_birth_offs for g3 in model.s_groups_offs)
               for z in model.s_season_types for i in model.s_tol) <=0
    model.con_stockinfra = pe.Constraint(model.s_infrastructure, rule=stockinfra, doc='Requirement for infrastructure (based on number of times yarded and shearing activity)')


#####################
##  setup variables # these variables only need initialising once ie sets wont change within and iteration of exp.
#####################
##infrastructure
model.v_infrastructure = pe.Var(model.s_infrastructure, bounds = (0,None) , doc='amount of infustructure required for given animal enterprise (based on number of sheep through infra)')

# ##################################
# ### setup core model constraints #
# ##################################

def stock_me(model,f,p6):
    return sum(model.v_sire[g0] * model.p_mei_sire[p6,f,g0] for g0 in model.s_groups_sire)\
          + sum(sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1] * model.p_mei_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1]
                        for t1 in model.s_sale_dams for v1 in model.s_dvp_dams for k2 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)
          + sum(sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3]
                * model.p_mei_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for t3 in model.s_sale_offs)
                + model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,g1_new] for g1_new in model.s_groups_dams)
                * model.p_mei_trans_offs[v3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for v3 in model.s_dvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs)
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)


def stock_pi(model,f,p6):
    return sum(model.v_sire[g0] * model.p_pi_sire[p6,f,g0] for g0 in model.s_groups_sire)\
        + sum(sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                    * model.p_pi_dams[k2,p6,f,t1,v1,a,n1,w1,z,i,y1,g1] for t1 in model.s_sale_dams)
                    # + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams)
                    # * model.p_pi_trans_dams[v1,a,b1,n1,w1,z,i,y1,g1,p6,v]
                    for v1 in model.s_dvp_dams for k2 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)
          + sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3]
                * model.p_pi_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for t3 in model.s_sale_offs)
                + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams)
                * model.p_pi_trans_offs[v3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for v3 in model.s_dvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs)
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)

def stock_cashflow(model,c):
    # infrastructure = sum(model.p_rm_stockinfra[h3,c] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cash_sire[g0,c] for g0 in model.s_groups_sire) \
          + sum(sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                    * model.p_cash_dams[k2,c,t1,v1,a,n1,w1,z,i,y1,g1] for t1 in model.s_sale_dams)
                    for v1 in model.s_dvp_dams  for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)
          + sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_cash_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs)
                + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for g1_new in model.s_groups_dams)
                * model.p_cash_trans_offs[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,c] for v3 in model.s_dvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs)
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)

    return stock #- infastrucure - purchases

# def stock_cashflow(model,c):
#     infastrucure = sum(model.p_rm_stockinfra[h3,c] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
#     stock = sum(model.v_sire[g0] * model.p_cash_sire[g0,c] for g0 in model.s_groups_sire)
#           + sum(sum(sum(sum(model.v_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in model.s_co_ww)
#                     * model.p_cash_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7,c] for t1 in model.s_sale_dams)
#                     + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for g1_new in model.s_groups_dams)
#                     * model.p_cash_trans_dams[v1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7,c]
#                     for v1 in model.s_dvp_dams  for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)
#           + sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_cash_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs)
#                 + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for g1_new in model.s_groups_dams)
#                 * model.p_cash_trans_offs[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,c] for v3 in model.s_dvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs)
#           for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
#     purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[g0,c] for g0 in model.s_groups_sire)  \
#                 + sum(sum(model.v_purchase_dams[v1,w1,z,i,g1] * model.p_cost_purch_dam[v1,w1,z,i,g1,c] for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
#                     + sum(model.v_purchase_offs[v3,w3,z,i,g3] * model.p_cost_purch_offs[v3,w3,z,i,g3,c] for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
#                     for z in model.s_season_types for i in model.s_tol)
#     return stock - infastrucure - purchases
#
#
def stock_cost(model):
    infrastrucure = sum(model.p_rm_stockinfra[h3,c] for c in model.s_cashflow_periods * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cost_sire[g0] for g0 in model.s_groups_sire)+\
            sum(sum(sum(sum(model.v_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
                    * model.p_cost_dams[t1,v1,b1,z,i,g1] for t1 in model.s_sale_dams)
                    + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams)
                    * model.p_cost_trans_dams[v1,b1,z,i,g1] for v1 in model.s_dvp_dams for b1 in model.s_birth_dams  for g1 in model.s_groups_dams)
           + sum(sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b2,x,y3,g3,r4,r5,r6,r7] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
                 * model.p_cost_offs[t3,v3,b3,z,i,g3] for t3 in model.s_sale_offs)
                 + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b2,x,y3,g3,r4,r5,r6,r7,g1_new] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams)
                 * model.p_cost_trans_offs[v3,b3,z,i,g3] for v3 in model.s_dvp_offs for b3 in model.s_birth_offs for g3 in model.s_groups_offs)
                 for z in model.s_season_types for i in model.s_tol)
    purchases = sum(sum(model.v_purchase_dams[v1,w1,z,i,g1] * sum(model.p_cost_purch_dam[v1,w1,z,i,g1,c] for c in model.s_cashflow_periods) for v1 in model.s_dvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
                    +sum(model.v_purchase_offs[v3,w3,z,i,g3] * sum(model.p_cost_purch_offs[v3,w3,z,i,g3,c] for c in model.s_cashflow_periods) for v3 in model.s_dvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
                    for z in model.s_season_types for i in model.s_tol)
    return infrastrucure + stock + purchases
#
#
def stock_labour_anyone(model,p5):
    # infastrucure = sum(model.p_lab_stockinfra[h3,p5] * model.v_infrastructure[h3,p5] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_lab_anyone_sire[g0,p5] for g0 in model.s_groups_sire)\
          + sum(sum(sum(model.v_dams[k2,t1,v1,a,n1,w1,z,i,y1,g1]
                    * model.p_lab_anyone_dams[k2,p5,t1,v1,a,n1,w1,z,i,y1,g1] for t1 in model.s_sale_dams for y1 in model.s_gen_merit_dams)
                    # + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,g1_new] for y1 in model.s_gen_merit_dams for g1_new in model.s_groups_dams)
                    # * model.p_lab_trans_dams[v1,a,b1,n1,w1,z,i,g1,p5]
                    for v1 in model.s_dvp_dams for k2 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
          # + sum(sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3] for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
          #       * model.p_lab_offs[t3,v3,n3,w3,z,i,d,a,b3,x,g3,p5] for t3 in model.s_sale_offs)
          #       + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,g1_new] for y3 in model.s_gen_merit_offs for g1_new in model.s_groups_dams)
          #       * model.p_lab_trans_offs[v3,n3,w3,z,i,d,a,b3,x,g3,p5] for v3 in model.s_dvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for g3 in model.s_groups_offs)
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    return stock
#


# def stock_dep(model):
#     return sum(model.p_dep_stockinfra[h3]  * model.v_infrastructure[h3] for h3 in model.s_infrastructure)

#
# def stock_asset(model):
#     infastrucure = sum(model.p_asset_stockinfra[h3] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
#     stock = sum(model.v_sire[g0] * model.p_asset_sire[g0] for g0 in model.s_groups_sire)
#           + sum(sum(sum(sum(model.v_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww)
#                     * model.p_asset_dams[t1,v1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7] for t1 in model.s_sale_dams)
#                     + sum(model.v_dams2sire[v1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for g1_new in model.s_groups_dams)
#                     * model.p_asset_trans_dams[v1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7]
#                     for v1 in model.s_dvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)
#           + sum(sum(model.v_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_asset_offs[t3,v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs)
#                 + sum(model.v_offs2dam[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for g1_new in model.s_groups_dams)
#                 * model.p_asset_trans_offs[v3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for v3 in model.s_dvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs)
#           for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
#     return infastrucure + stock
#
#






