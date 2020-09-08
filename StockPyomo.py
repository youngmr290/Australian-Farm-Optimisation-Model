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
##purchases
model.v_purchase_dams = pe.Var(model.s_fvp_dams, model.s_lw_dams, model.s_season_types, model.s_tol, model.s_groups_dams, bounds = (0,None) , doc='number of purchased dam animals')
model.v_purchase_offs = pe.Var(model.s_fvp_offs, model.s_lw_offs, model.s_season_types, model.s_tol, model.s_groups_offs, bounds = (0,None) , doc='number of purchased offs animals')

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
##stock
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
    model.del_component(model.p_asset_offs)
except AttributeError:
    pass
model.p_asset_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                           model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                           model.s_groups_offs, model.s_co_cfw, model.s_co_fd, model.s_co_min_fd, model.s_co_fl, initialize=, default = 0.0, doc='Asset value of offs')
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
    model.del_component(model.p_cash_offs)
except AttributeError:
    pass
model.p_cash_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
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
    model.del_component(model.p_cost_offs)
except AttributeError:
    pass
model.p_cost_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_lw_offs, model.s_season_types, 
                          model.s_tol, model.s_birth_offs, 
                          model.s_groups_offs, initialize=, default = 0.0, doc='husbandry cost offs')


try:
    model.del_component(model.p_mei_sire)
except AttributeError:
    pass
model.p_mei_sire = Param(model.s_groups_sire, model.s_sheep_pools, initialize=, default = 0.0, doc='energy requirement sire')
try:
    model.del_component(model.p_mei_dams)
except AttributeError:
    pass
model.p_mei_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                         model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_sheep_pools, 
                         initialize=, default = 0.0, doc='energy requirement dams')
try:
    model.del_component(model.p_mei_offs)
except AttributeError:
    pass
model.p_mei_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_sheep_pools, initialize=, default = 0.0, doc='energy requirement offs')
try:
    model.del_component(model.p_pi_sire)
except AttributeError:
    pass
model.p_pi_sire = Param(model.s_groups_sire, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity sire')
try:
    model.del_component(model.p_pi_dams)
except AttributeError:
    pass
model.p_pi_dams = Param(model.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_gen_merit_dams, model.s_groups_dams, model.s_sheep_pools,, initialize=, default = 0.0, doc='intake capacity dams')
try:
    model.del_component(model.p_pi_offs)
except AttributeError:
    pass
model.p_pi_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, model.s_gen_merit_offs, 
                         model.s_groups_offs, model.s_sheep_pools, initialize=, default = 0.0, doc='intake capacity offs')
try:
    model.del_component(model.p_lab_sire)
except AttributeError:
    pass
model.p_lab_sire = Param(model.s_groups_sire, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment sire')
try:
    model.del_component(model.p_lab_dams)
except AttributeError:
    pass
model.p_lab_dams = Parammodel.s_sale_dams, model.s_fvp_dams, model.s_wean_times, model.s_birth_dams, model.s_nut_dams, model.s_lw_dams, 
                        model.s_season_types, model.s_tol, model.s_groups_dams, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment dams')
try:
    model.del_component(model.p_lab_offs)
except AttributeError:
    pass
model.p_lab_offs = Param(model.s_sale_offs, model.s_fvp_offs, model.s_nut_offs, model.s_lw_offs, model.s_season_types, 
                         model.s_tol, model.s_damage_offs, model.s_wean_times, model.s_birth_offs, model.s_gender_offs, 
                         model.s_groups_offs, model.s_labperiods, initialize=, default = 0.0, doc='labour requirment offs')



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

##################################
### setup core model constraints #
##################################
## def constraint_function:
## ^ or define outside the main function and just call here
## call constraint function
#model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')

def stock_dep(model):
    infastrucure = sum(model.p_dep_stockinfra[h3]  * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = model.v_sire[g] * 
    return infastrucure + stock
        
def stock_asset(model):
    infastrucure = sum(model.p_asset_stockinfra[h3] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_asset_sire[g0] for g0 in model.s_groups_sire) 
          + sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww) 
                    * model.p_asset_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7] for t1 in model.s_sale_dams for f1 in model.s_fvp_dams for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_asset_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
    return infastrucure + stock

def stock_cashflow(model,c):
    infastrucure = sum(model.p_rm_stockinfra[h3,c] * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(model.v_sire[g0] * model.p_cash_sire[g0,c] for g0 in model.s_groups_sire) 
          + sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in model.s_co_ww) 
                    * model.p_cash_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r4,r5,r6,r7,c] for t1 in model.s_sale_dams for f1 in model.s_fvp_dams  for b1 in model.s_birth_dams for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for g1 in model.s_groups_dams) 
          + sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] * model.p_cash_offs[t3,f3,n3,w3,z,i,d,a,b3,x,y3,g3,r4,r5,r6,r7] for t3 in model.s_sale_offs for f3 in model.s_fvp_offs  for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for b3 in model.s_birth_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for g3 in model.s_groups_offs) 
          for a in model.s_wean_times for z in model.s_season_types for i in model.s_tol for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)
    return infastrucure + stock
                       
def stock_cost(model): 
    infastrucure = sum(model.p_rm_stockinfra[h3,c]for c in model.s_cashflow_periods * model.v_infrastructure[h3] for h3 in model.s_infrastructure)
    stock = sum(sum(sum(model.v_dams[t1,f1,a,b1,n1,w1,z,i,y1,g1,r1,r2,r3,r4,r5,r6,r7] for a in model.s_wean_times for n1 in model.s_nut_dams for w1 in model.s_lw_dams for y1 in model.s_gen_merit_dams for r1 in model.s_co_conception for r2 in model.s_co_bw for r3 in  model.s_co_ww for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl)  * model.p_cost_dams[t1,f1,b1,z,i,g1] for t1 in model.s_sale_dams for f1 in model.s_fvp_dams for b1 in model.s_birth_dams  for g1 in model.s_groups_dams)  \
           + sum(sum(model.v_offs[t3,f3,n3,w3,z,i,d,a,b2,x,y3,g3,r4,r5,r6,r7] for a in model.s_wean_times for n3 in model.s_nut_offs for w3 in model.s_lw_offs for d in model.s_damage_offs for x in model.s_gender_offs for y3 in model.s_gen_merit_offs for r4 in model.s_co_cfw for r5 in model.s_co_fd for r6 in model.s_co_min_fd for r7 in model.s_co_fl) * model.p_cost_offs[t3,f3,b3,z,i,g3] for t3 in model.s_sale_offs for f3 in model.s_fvp_offs for b3 in model.s_birth_offs)
                 for g3 in model.s_groups_offs) for z in model.s_season_types for i in model.s_tol) <=0
    return infastrucure + stock
    

def stock_labour(model,p5):
    infastrucure = sum(model.p_lab_stockinfra[h3,p5] * model.v_infrastructure[h3,p5] for h3 in model.s_infrastructure)
    stock = 
    return infastrucure + stock
                       
def stock_me(model,v,f):
    model.v_sire[] * model.p_me_sire[]

def stock_pi(model,v,f):
    model.v_sire[] * model.p_pi_sire[]


