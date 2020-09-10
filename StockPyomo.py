# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:35 2020

@author: John
"""

#python modules
import pyomo.environ as pe


#MUDAS modules
from CreateModel import model
import StockGenerator as sgen

def sheep_precalcs(params,report):
    sgen.sheep_sim()
    sgen.sheep_parameters()
    
    
    
# def sheep_pyomo_local(params):

#####################
##  setup variables # #variables that use dynamic sets must be defined each itteration of exp
#####################
##animals
model.v_sire = pe.Var(model.s_groups_sire, bounds = (0,None) , doc='number of sire animals')
model.v_dams = pe.Var(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                      model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                      model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, bounds = (0,None) , doc='number of dam animals')
model.v_offs = pe.Var(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, \
                      model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                      model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, bounds = (0,None) , doc='number of offs animals')
##animal transfers
model.v_offs2dam = pe.Var(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, \
                      model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                      model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_dams, bounds = (0,None) , doc='transfer of animals from the offspring variables to the dam variables.')
model.v_dams2sire = pe.Var(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                      model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                      model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_sire, bounds = (0,None) , doc='transfer of animals from the dam variables of one sire genotype to another sire genotype')


##purchases
model.v_purchase_dams = pe.Var(model.s_fvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, bounds = (0,None) , doc='number of purchased dam animals')
model.v_purchase_offs = pe.Var(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, bounds = (0,None) , doc='number of purchased offs animals')
##mating
model.v_sirecap = pe.Var(model.s_groups_sire, model.s_sire_periods bounds = (0,None) , doc='number of sire animals')
######################
### setup parameters #
######################
##infrastructure
try:
    model.del_component(model.p_asset_stockinfra)
except AttributeError:
    pass
model.p_asset_stockinfra = Param(model.s_infrastructure, initialize=, default = 0.0, doc='Asset value per animal mustered  or shorn')
try:
    model.del_component(model.p_dep_stockinfra)
except AttributeError:
    pass
model.p_dep_stockinfra = Param(model.s_infrastructure, initialize=, default = 0.0, doc='Depreciation of the asset value')
try:
    model.del_component(model.p_rm_stockinfra)
except AttributeError:
    pass
model.p_rm_stockinfra = Param(model.s_infrastructure, model.s_cashflow_periods,initialize=, default = 0.0, doc='Cost of R&M of the infrastructure (per animal mustered/shorn)')
try:
    model.del_component(model.p_lab_stockinfra)
except AttributeError:
    pass
model.p_lab_stockinfra = Param(model.s_infrastructure, model.s_labperiods, initialize=, default = 0.0, doc='Labour required for R&M of the infrastructure (per animal mustered/shorn)')
##stock - all
try:
    model.del_component(model.p_asset_sire)
except AttributeError:
    pass
model.p_asset_sire = Param(model.s_groups_sire, initialize=, default = 0.0, doc='Asset value of sire')
try:
    model.del_component(model.p_asset_dams)
except AttributeError:
    pass
model.p_asset_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                           model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                           model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='Asset value of dams')
try:
    model.del_component(model.p_asset_trans_dams)
except AttributeError:
    pass
model.p_asset_trans_dams = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                           model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                           model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='Asset value of transfer dams')
try:
    model.del_component(model.p_asset_offs)
except AttributeError:
    pass
model.p_asset_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                           model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                           model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='Asset value of offs')
try:
    model.del_component(model.p_asset_trans_offs)
except AttributeError:
    pass
model.p_asset_trans_offs = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                           model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                           model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='Asset value of transferred offs')
try:
    model.del_component(model.p_infra_sire)
except AttributeError:
    pass
model.p_infra_sire = Param(model.s_groups_sire, model.s_infrastructure, initialize=, default = 0.0, doc='sire requirement for infrastructure (based on number of times yarded and shearing activity)')
try:
    model.del_component(model.p_infra_dams)
except AttributeError:
    pass
model.p_infra_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_birth_dams, model.s_season_types, 
                           model.s_tol, model.s_groups_dams, model.s_infrastructure, initialize=, default = 0.0, doc='Dams requirement for infrastructure (based on number of times yarded and shearing activity)')
try:
    model.del_component(model.p_infra_offs)
except AttributeError:
    pass
model.p_infra_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_season_types, model.s_tol, model.s_birth_offs,  
                          model.s_groups_offs, model.s_infrastructure, initialize=, default = 0.0, doc='offs requirement for infrastructure (based on number of times yarded and shearing activity)')
try:
    model.del_component(model.p_cash_sire)
except AttributeError:
    pass
model.p_cash_sire = Param(model.s_groups_sire, initialize=, default = 0.0, doc='Income (wool and sale sheep) and expenses (husbandry) sire')
try:
    model.del_component(model.p_cash_dams)
except AttributeError:
    pass
model.p_cash_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                          model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_cashflow_periods,
                          initialize=, default = 0.0, doc='Income (wool and sale sheep) and expenses (husbandry) dams')
try:
    model.del_component(model.p_cash_trans_dams)
except AttributeError:
    pass
model.p_cash_trans_dams = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                          model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_cashflow_periods,
                          initialize=, default = 0.0, doc='Income (wool and sale sheep) and expenses (husbandry) dams')
try:
    model.del_component(model.p_cash_offs)
except AttributeError:
    pass
model.p_cash_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                          model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                          model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_cashflow_periods, 
                          initialize=, default = 0.0, doc='Income (wool and sale sheep) and expenses (husbandry) offs')
try:
    model.del_component(model.p_cash_trans_offs)
except AttributeError:
    pass
model.p_cash_trans_offs = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                          model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                          model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_cashflow_periods, 
                          initialize=, default = 0.0, doc='Income (wool and sale sheep) and expenses (husbandry) offs')
try:
    model.del_component(model.p_cost_sire)
except AttributeError:
    pass
model.p_cost_sire = Param(model.s_groups_sire, initialize=, default = 0.0, doc='husbandry cost sire')
try:
    model.del_component(model.p_cost_dams)
except AttributeError:
    pass
model.p_cost_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_birth_dams, 
                          model.s_season_types, model.s_tol, model.s_groups_dams,
                          initialize=, default = 0.0, doc='husbandry cost dams')
try:
    model.del_component(model.p_cost_trans_dams)
except AttributeError:
    pass
model.p_cost_trans_dams = Param(model.s_fvp_dams, model.s_birth_dams, 
                          model.s_season_types, model.s_tol, model.s_groups_dams,
                          initialize=, default = 0.0, doc='husbandry cost of transfer dams')
try:
    model.del_component(model.p_cost_offs)
except AttributeError:
    pass
model.p_cost_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_lw_offs, model.s_season_types, 
                          model.s_tol, model.s_birth_offs, 
                          model.s_groups_offs, initialize=, default = 0.0, doc='husbandry cost offs')
try:
    model.del_component(model.p_cost_trans_offs)
except AttributeError:
    pass
model.p_cost_trans_offs = Param(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, 
                          model.s_tol, model.s_birth_offs, 
                          model.s_groups_offs, initialize=, default = 0.0, doc='husbandry cost of transfer offs')
try:
    model.del_component(model.p_mei_sire)
except AttributeError:
    pass
model.p_mei_sire = Param(model.s_groups_sire, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='energy requirement sire')
try:
    model.del_component(model.p_mei_dams)
except AttributeError:
    pass
model.p_mei_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_feed_periods, model.s_sheep_pools, 
                         initialize=, default = 0.0, doc='energy requirement dams')
try:
    model.del_component(model.p_mei_trans_dams)
except AttributeError:
    pass
model.p_mei_trans_dams = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_feed_periods, model.s_sheep_pools, 
                         initialize=, default = 0.0, doc='energy requirement of transfered  dams')
try:
    model.del_component(model.p_mei_offs)
except AttributeError:
    pass
model.p_mei_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='energy requirement offs')
try:
    model.del_component(model.p_mei_trans_offs)
except AttributeError:
    pass
model.p_mei_trans_offs = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='energy requirement transfered offs')
try:
    model.del_component(model.p_pi_sire)
except AttributeError:
    pass
model.p_pi_sire = Param(model.s_groups_sire, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity sire')
try:
    model.del_component(model.p_pi_dams)
except AttributeError:
    pass
model.p_pi_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity dams')
try:
    model.del_component(model.p_pi_trans_dams)
except AttributeError:
    pass
model.p_pi_trans_dams = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_feed_periods, model.s_sheep_pools, 
                        initialize=, default = 0.0, doc='intake capacity of transferred dams')
try:
    model.del_component(model.p_pi_offs)
except AttributeError:
    pass
model.p_pi_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity offs')
try:
    model.del_component(model.p_pi_trans_offs)
except AttributeError:
    pass
model.p_pi_trans_offs = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_feed_periods, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity of transfered offs')
try:
    model.del_component(model.p_lab_sire)
except AttributeError:
    pass
model.p_lab_sire = Param(model.s_groups_sire, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment sire')
try:
    model.del_component(model.p_lab_dams)
except AttributeError:
    pass
model.p_lab_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_groups_dams, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment dams')
try:
    model.del_component(model.p_lab_trans_dams)
except AttributeError:
    pass
model.p_lab_trans_dams = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_groups_dams, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment transfer dams')
try:
    model.del_component(model.p_lab_offs)
except AttributeError:
    pass
model.p_lab_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, 
                         model.s_groups_offs, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment offs')
try:
    model.del_component(model.p_lab_trans_offs)
except AttributeError:
    pass
model.p_lab_trans_offs = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, 
                         model.s_groups_offs, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment transfer offs')
##stock - dams
model.p_numbers_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                          model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='transfer of dam activity to the next feed variation period') 
model.p_npw = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_fvp_offs, model.s_lw_offs, 
                          model.s_season_types, model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, 
                          model.s_gen_merit_offs, model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, 
                          initialize=, default = 0.0, doc='number of prodgeny weaned in each off class') 
model.p_n_sires = Param(model.s_fvp_dams, model.s_birth_dams, model.s_groups_dams, model.s_groups_sire, model.s_sire_periods, initialize=, default = 0.0, doc='requirement for sires for mating') 

##stock - offs
model.p_numbers_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs, 
                             model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, 
                             model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='transfer of offs activity to the next feed variation period') 
##purchases
model.p_cost_purch_sire = Param(model.s_groups_sire, model.s_cashflow_periods, 
                               initialize=, default = 0.0, doc='cost of purchased sires') 
model.p_numberpurch_dam = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_lw_dams, 
                          model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, 
                          model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='purchase transfer - ie how a purchased dam is allocated into damR') 
model.p_cost_purch_dam = Param(model.s_fvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, model.s_cashflow_periods, 
                               initialize=, default = 0.0, doc='cost of purchased dams') 
model.p_numberpurch_offs = Param(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs, 
                                 model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, 
                                 model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='purchase transfer - ie how a purchased offs is allocated into offsR') 
model.p_cost_purch_offs = Param(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, model.s_cashflow_periods, 
                               initialize=, default = 0.0, doc='cost of purchased offs') 
##transfers
model.p_offs2dam_numbers = Param(model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs, 
                                 model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_co_cfw, 
                                 model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_dams, model.s_fvp_dams, model.s_wean_times, 
                                 model.s_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, 
                                 model.s_co_conception, model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl,                                 
                                 initialize=, default = 0.0, doc='Proportion of the offs distributed to each of the starting LWs at the beginning of the current dam feed variation period') 
model.p_dam2sire_numbers = Param(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol, 
                                 model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, model.s_co_bw, model.s_co_ww, model.s_co_cfw, 
                                 model.s_co_fd, model.s_co_min_fd, model.s_co_fl, model.s_groups_dams,                                
                                 initialize=, default = 0.0, doc='Proportion of the offs distributed to each of the starting LWs at the beginning of the current dam feed variation period') 

#####################
##  setup variables # these variables only need initialising once ie sets wont change within and iteration of exp.
#####################
##infrastructure
model.v_infrastructure = pe.Var(model.s_infrastructure, bounds = (0,None) , doc='amount of infustructure required for given animal enterprise (based on number of sheep through infra)')


########################
### set up constraints #
########################
## def constraint_function:
## ^ or define outside the main function and just call here
## call constraint function
#model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')
try:
    model.del_component(model.con_stockinfra)
except AttributeError:
    pass
def stockinfra(model,h3):
    return -model.v_infrastructure[h3] + sum(model.v_sire[g0] * model.p_infra_sire[g0,h3] for g0 in model.s_groups_sire)  \
           + sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)  * model.p_infra_dams[t1,f1,b1,z,i,g1,h3] for t1 in model.s_sale_dams for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for g1 in model.s_groups_dams)  \
           + sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) * model.p_infra_offs[t3,f3,b3,z,i,g3,h3] for t3 in model.s_sale_offs for f3 in model.s_fvp_offs for b3 in model.s_birth_offs for g3 in model.s_groups_offs) 
           for z in model.s_season_types for i in model.s_tol) <=0
model.con_stockinfra = pe.Constraint(model.s_infrastructure, rule=stockinfra, doc='Requirement for infrastructure (based on number of times yarded and shearing activity)')

try:
    model.del_component(model.con_damR)
except AttributeError:
    pass
def damR(model,f1,a,b1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7):
    f1_prev = list(model.s_fvp_dams)[list(model.s_fvp_dams).index(f1) - 1]  #used to get the activity number from the last period - to determine the number of dam provided into this period 
    return sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] 
               - model.v_dams[t1,f1_prev,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] * p_numbers_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7]   
               for t1 in model.s_sale_dams for n1 in model.s_nut_dams) \
        + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new]
              - model.v_dams2sire[f1_prev,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] * model.p_dam2sire_numbers[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new]
              for n1 in model.s_nut_dams for g1_new in model.s_groups_dams) \
        - model.v_purchase_dams[f1,w1,z,i,g1] * model.p_numberpurch_dam[f1,a,b1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] \ #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
        - sum(model.v_offs2dam[f3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,r4_off,r5_off,r6_off,r7_off,g1_off] * model.p_offs2dam_numbers[f3,n3,w3,z3,i3,d,a3,b3,x,y3,g3,r4_off,r5_off,r6_off,r7_off,g1_off,f1,a,b1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] 
              for f3 in model.s_fvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for z3 in model.s_season_types for i3 in model.s_tol for d in model.s_damage_offs for a3 in model.s_wean_times
              for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs for r4_off in model.s_co_cfw
              for r5_off in model.s_co_fd for r6_off in model.s_co_min_fd for r7_off in model.s_co_fl for g1_off in model.s_groups_dams) <= 0 #have to track off sets so only they are summed.
model.con_damR = pe.Constraint(model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_co_conception, model.s_co_bw, model.s_co_ww, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, rule=damR, doc='transfer of off to dam and dam to next fvp.')


try:
    model.del_component(model.con_offsR)
except AttributeError:
    pass
def offsR(model,f3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7):
    f3_prev = list(model.s_fvp_offs)[list(model.s_fvp_offs).index(f3) - 1]  #used to get the activity number from the last period - to determine the number of off provided into this period 
    return sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1] for n3 in model.s_nut_offs for g1 in model.s_groups_dams) \
        + sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] 
               - model.v_offs[t3,f3_prev,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * p_numbers_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7]   
               for t3 in model.s_sale_offs for n3 in model.s_nut_offs) \
        - model.v_purchase_offs[f3,w3,z,i,g3] * model.p_numberpurch_offs[f3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] \ #p_numpurch allocates the purchased offs into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
        - sum(sum(model.v_dams[t1,f1,a1,b1,n1,w1,z1,i1,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) \# note r4567 for off is not meant to be summed with the r's from dams - the r's in the off are used to allocate the off betwen the cons
                  * model.p_npw[t1,f1,a1,b1,n1,w1,z1,i1,y1,g1,f3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] \#have to distinguish between a1 and a3 so only the a from dams is summed.
                  for t1 in model.s_sale_dams for f1 in model.s_fvp_dams for a1 in model.s_wean_times for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for z1 in model.s_season_types for i1 in model.s_tol for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams)   <=0 
model.con_offsR = pe.Constraint(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_damage_offs, 
                                 model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, model.s_groups_offs, 
                                 model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, rule=offsR, doc='transfer of off to next fvp.')


try:
    model.del_component(model.con_matingR)
except AttributeError:
    pass
def mating(model,g0,p8):
    return sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for t1 in model.s_sale_dams for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)   
             * p_n_sires[f1,b1,g1,g0,p8] for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for g1 in model.s_groups_dams) \
        - model.v_sire[g0] <=0 #p_numpurch allocates the purchased dams into certain sets, in this case it is correct to multiply a var with less sets to a param with more sets
model.con_matingR = pe.Constraint(model.s_groups_sire, model.s_sire_periods, rule=mating, doc='sire requirment for mating')


##################################
### setup core model constraints #
##################################
## def constraint_function:
## ^ or define outside the main function and just call here
## call constraint function
#model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')

def stock_dep(model):
    infastrucure = sum(model.p_dep_stockinfra[h3]  * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    return infastrucure + stock

def stock_asset(model):
    infastrucure = sum(model.p_asset_stockinfra[h3] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_asset_sire[g0] for g0 in model.s_groups_sire) 
          + sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww) 
                    * model.p_asset_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7] for t1 in model.s_sale_dams) 
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for g1_new in model.s_groups_dams)
                    * model.p_asset_trans_dams[f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7]
                    for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_asset_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs)
                + sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for g1_new in model.s_groups_dams) 
                * model.p_asset_trans_offs[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
    return infastrucure + stock

def stock_cashflow(model,c):
    infastrucure = sum(model.p_rm_stockinfra[h3,c] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cash_sire[g0,c] for g0 in model.s_groups_sire) 
          + sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in model.s_co_ww) 
                    * model.p_cash_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7,c] for t1 in model.s_sale_dams)
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for g1_new in model.s_groups_dams)
                    * model.p_cash_trans_dams[f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7,c]
                    for f1 in model.s_fvp_dams  for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_cash_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs)
                + sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for g1_new in model.s_groups_dams) 
                * model.p_cash_trans_offs[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,c] for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
    purchases = sum(model.v_sire[g0] * model.p_cost_purch_sire[g0,c] for g0 in model.s_groups_sire)  \
                + sum(sum(model.v_purchase_dams[f1,w1,z,i,g1] * model.p_cost_purch_dam[f1,w1,z,i,g1,c] for f1 in model.s_fvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
                    + sum(model.v_purchase_offs[f3,w3,z,i,g3] * model.p_cost_purch_offs[f3,w3,z,i,g3,c] for f3 in model.s_fvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
                    for z in model.s_season_types for i in model.s_tol)
    return stock - infastrucure - purchases
        

def stock_cost(model): 
    infastrucure = sum(model.p_rm_stockinfra[h3,c] for c in model.s_cashflow_periods * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cost_sire[g0] for g0 in model.s_groups_sire)
            sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)  
                    * model.p_cost_dams[t1,f1,b1,z,i,g1] for t1 in model.s_sale_dams) 
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams)  
                    * model.p_cost_trans_dams[f1,b1,z,i,g1] for f1 in model.s_fvp_dams for b1 in model.s_birth_dams  for g1 in model.s_groups_dams)  
           + sum(sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b2,x,y3,g3,r4,r5,r6,r7] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                 * model.p_cost_offs[t3,f3,b3,z,i,g3] for t3 in model.s_sale_offs)
                 + sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b2,x,y3,g3,r4,r5,r6,r7,g1_new] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                 * model.p_cost_trans_offs[f3,b3,z,i,g3] for f3 in model.s_fvp_offs for b3 in model.s_birth_offs for g3 in model.s_groups_offs)
                 for z in model.s_season_types for i in model.s_tol) 
    purchases = sum(sum(model.v_purchase_dams[f1,w1,z,i,g1] * sum(model.p_cost_purch_dam[f1,w1,z,i,g1,c] for c in model.s_cashflow_periods) for f1 in model.s_fvp_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams)
                    +sum(model.v_purchase_offs[f3,w3,z,i,g3] * sum(model.p_cost_purch_offs[f3,w3,z,i,g3,c] for c in model.s_cashflow_periods) for f3 in model.s_fvp_offs for w3 in model.s_lw_offs for g3 in model.s_groups_offs)
                    for z in model.s_season_types for i in model.s_tol)
    return infastrucure + stock + purchases
    

def stock_labour(model,p5):
    infastrucure = sum(model.p_lab_stockinfra[h3,p5] * model.v_infrastructure[h3,p5] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_lab_sire[g0,p5] for g0 in model.s_groups_sire) 
          + sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                    * model.p_lab_dams[t1,f1,a,b1,n1,w1,z,i,g1,p5] for t1 in model.s_sale_dams) 
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                    * model.p_lab_trans_dams[f1,a,b1,n1,w1,z,i,g1,p5] for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for g1 in model.s_groups_dams) 
          + sum(sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                * model.p_lab_offs[t3,f3,n3,w3,z,i,d,a,b3,x,g3,p5] for t3 in model.s_sale_offs)
                + sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                * model.p_lab_trans_offs[f3,n3,w3,z,i,d,a,b3,x,g3,p5] for f3 in model.s_fvp_offs for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)
    return infastrucure + stock

                       
def stock_me(model,v,p6):
    return sum(model.v_sire[g0] * model.p_mei_sire[g0,p6,v] for g0 in model.s_groups_sire) 
          + sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                    * model.p_mei_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,p6,v] for t1 in model.s_sale_dams) 
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                    * model.p_mei_trans_dams[f1,a,b1,n1,w1,z,i,y1,g1,p6,v]
                    for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                * model.p_mei_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for t3 in model.s_sale_offs) 
                + model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                * model.p_mei_trans_offs[f3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)


def stock_pi(model,v,f):
    return sum(model.v_sire[g0] * model.p_pi_sire[g0,p6,v] for g0 in model.s_groups_sire) 
          + sum(sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                    * model.p_pi_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,p6,v] for t1 in model.s_sale_dams)
                    + sum(model.v_dams2sire[f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7,g1_new] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                    * model.p_pi_trans_dams[f1,a,b1,n1,w1,z,i,y1,g1,p6,v] for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) 
                * model.p_pi_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for t3 in model.s_sale_offs) 
                + sum(model.v_offs2dam[f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7,g1_new] for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl for g1_new in model.s_groups_dams) 
                * model.p_pi_trans_offs[f3,n3,w3,z,i,d,a,b3,x,y3,g3,p6,v] for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol)








