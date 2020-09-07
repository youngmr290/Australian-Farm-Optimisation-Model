# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:35 2020

@author: John
"""

#python modules
import pyomo.environ as pe


#MUDAS modules
from CreateModel import model
import SheepSim as ssim

def sheep_precalcs(params,report):
    ssim.sheep_sim()
    ssim.sheep_parameters()
    
#####################
##  setup variables #
#####################
##animals
model.v_sire = pe.Var(model.s_sale_sire, model.s_fvp_sire, model.s_nut_sire, model.s_lw_sire, model.s_season_types, \
                      model.s_tol, model.s_gen_merit_sire, model.s_groups_sire, bounds = (0,None) , doc='number of sire animals')
model.v_dams = pe.Var(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_e_cycles, model.s_lsln_dams, 
                      model.s_nut_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                      model.s_conception, model.s_bw, model.s_ww, model.s_cfw, model.s_fd, model.s_min_fd, model.s_fl, bounds = (0,None) , doc='number of dams animals')
model.v_offs = pe.Var(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, \
                      model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_e_cycles, model.s_btrt_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                      model.s_groups_offs, model.s_cfw, model.s_fd, model.s_min_fd, model.s_fl, bounds = (0,None) , doc='number of offs animals')

##infrastructure
model.v_infrastructure = pe.Var(model.s_infrastructure, bounds = (0,None) , doc='amount of infustructure required for given animal enterprise')
    
    
# def sheep_pyomo_local(params):

######################
### setup parameters #
######################

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
model.p_rm_stockinfra = Param(model.s_infrastructure, initialize=, default = 0.0, doc='Cost of R&M of the infrastructure (per animal mustered/shorn)')
try:
    model.del_component(model.p_lab_stockinfra)
except AttributeError:
    pass
model.p_lab_stockinfra = Param(model.s_infrastructure, model.s_labperiods, initialize=, default = 0.0, doc='Labour required for R&M of the infrastructure (per animal mustered/shorn)')

try:
    model.del_component(model.p_asset_sire)
except AttributeError:
    pass
model.p_asset_sire = Param(model.s_, initialize=, default = 0.0, doc='Asset value of sire')
try:
    model.del_component(model.p_asset_dams)
except AttributeError:
    pass
model.p_asset_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_e_cycles, model.s_lsln_dams, 
                           model.s_nut_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams,
                           model.s_conception, model.s_bw, model.s_ww, model.s_cfw, model.s_fd, model.s_min_fd, model.s_fl, initialize=, default = 0.0, doc='Asset value of dams')
try:
    model.del_component(model.p_asset_offs)
except AttributeError:
    pass
model.p_asset_offs = Param(model.s_, initialize=, default = 0.0, doc='Asset value of offs')

try:
    model.del_component(model.p_infra_sire)
except AttributeError:
    pass
model.p_infra_sire = Param(model.s_, initialize=, default = 0.0, doc='sire requirement for infrastructure (based on number of times yarded and shearing activity)')
try:
    model.del_component(model.p_infra_dams)
except AttributeError:
    pass
model.p_infra_dams = Param(model.s_, initialize=, default = 0.0, doc='Dams requirement for infrastructure (based on number of times yarded and shearing activity)')
try:
    model.del_component(model.p_infra_offs)
except AttributeError:
    pass
model.p_infra_offs = Param(model.s_, initialize=, default = 0.0, doc='offs requirement for infrastructure (based on number of times yarded and shearing activity)')

try:
    model.del_component(model.p_inc_wool_sire)
except AttributeError:
    pass
model.p_inc_wool_sire = Param(model.s_, initialize=, default = 0.0, doc='wool income sire')
try:
    model.del_component(model.p_inc_wool_dams)
except AttributeError:
    pass
model.p_inc_wool_dams = Param(model.s_, initialize=, default = 0.0, doc='wool income dams')
try:
    model.del_component(model.p_inc_wool_offs)
except AttributeError:
    pass
model.p_inc_wool_offs = Param(model.s_, initialize=, default = 0.0, doc='wool income offs')

try:
    model.del_component(model.p_inc_sales_sire)
except AttributeError:
    pass
model.p_inc_sales_sire = Param(model.s_, initialize=, default = 0.0, doc='sale income sire')
try:
    model.del_component(model.p_inc_sales_dams)
except AttributeError:
    pass
model.p_inc_sales_dams = Param(model.s_, initialize=, default = 0.0, doc='sale income dams')
try:
    model.del_component(model.p_inc_sales_offs)
except AttributeError:
    pass
model.p_inc_sales_offs = Param(model.s_, initialize=, default = 0.0, doc='sale income offs')

try:
    model.del_component(model.p_cost_husb_sire)
except AttributeError:
    pass
model.p_cost_husb_sire = Param(model.s_, initialize=, default = 0.0, doc='husbandry costs sire')
try:
    model.del_component(model.p_cost_husb_dams)
except AttributeError:
    pass
model.p_cost_husb_dams = Param(model.s_, initialize=, default = 0.0, doc='husbandry costs dams')
try:
    model.del_component(model.p_cost_husb_offs)
except AttributeError:
    pass
model.p_cost_husb_offs = Param(model.s_, initialize=, default = 0.0, doc='husbandry costs offs')

try:
    model.del_component(model.p_mei_sire)
except AttributeError:
    pass
model.p_mei_sire = Param(model.s_, initialize=, default = 0.0, doc='energy requirement sire')
try:
    model.del_component(model.p_mei_dams)
except AttributeError:
    pass
model.p_mei_dams = Param(model.s_, initialize=, default = 0.0, doc='energy requirement dams')
try:
    model.del_component(model.p_mei_offs)
except AttributeError:
    pass
model.p_mei_offs = Param(model.s_, initialize=, default = 0.0, doc='energy requirement offs')


try:
    model.del_component(model.p_pi_sire)
except AttributeError:
    pass
model.p_pi_sire = Param(model.s_, initialize=, default = 0.0, doc='intake capacity sire')
try:
    model.del_component(model.p_pi_dams)
except AttributeError:
    pass
model.p_pi_dams = Param(model.s_, initialize=, default = 0.0, doc='intake capacity dams')
try:
    model.del_component(model.p_pi_offs)
except AttributeError:
    pass
model.p_pi_offs = Param(model.s_, initialize=, default = 0.0, doc='intake capacity offs')

try:
    model.del_component(model.p_lab_sire)
except AttributeError:
    pass
model.p_lab_sire = Param(model.s_, initialize=, default = 0.0, doc='labour requirment sire')
try:
    model.del_component(model.p_lab_dams)
except AttributeError:
    pass
model.p_lab_dams = Param(model.s_, initialize=, default = 0.0, doc='labour requirment dams')
try:
    model.del_component(model.p_lab_offs)
except AttributeError:
    pass
model.p_lab_offs = Param(model.s_, initialize=, default = 0.0, doc='labour requirment offs')





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
    return -model.v_infrastructure[h3] + sum(model.v_sire[] * model.p_infra_sire[] for ... in ...  \
           + sum(model.v_dams[] * model.p_infra_dams[] for ... in ...  \
           + sum(model.v_offs[] * model.p_infra_offs[] for ... in ...  <=0
model.con_stockinfra = pe.Constraint(model.s_infrastructure, rule=stockinfra, doc='Requirement for infrastructure (based on number of times yarded and shearing activity)')

##################################
### setup core model constraints #
##################################
## def constraint_function:
## ^ or define outside the main function and just call here
## call constraint function
#model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')

def shp_dep(model):
    infastrucure = sum(sum(model.p_dep_stockinfra[] for in  * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    sheep = 
    return infastrucure + sheep
        
def shp_asset(model):
    infastrucure = sum(sum(model.p_asset_stockinfra[] for ... in ... * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    sheep = 
    return infastrucure + sheep

def shp_cost(model,c):
    infastrucure = sum(model.p_rm_stockinfra[h3] * model.v_infrastructure[h3]
    sheep = 
    return infastrucure + sheep
                       
def shp_labour(model,p5):
    infastrucure = sum(model.p_lab_stockinfra[h3,p5] * model.v_infrastructure[h3,p5] for h3 in model.s_infrastructure)
    sheep = 
    return infastrucure + sheep
                       
def shp_me(model,v,f):
    model.v_sire[] * model.p_me_sire[]

def shp_pi(model,v,f):
    model.v_sire[] * model.p_pi_sire[]





