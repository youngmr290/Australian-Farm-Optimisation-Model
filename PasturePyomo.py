# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:00:25 2019

@author: John
"""
#python modules
from pyomo import environ as pe

#MUDAS modules
from CreateModel import model
import Pasture as pas
import UniversalInputs as uinp

pastures = uinp.structure['pastures']       



def paspyomo_local():
    pas.init_and_map_excel('Property.xlsx', pastures)                         # read inputs from Excel file and map to the python variables
    pas.calculate_germ_and_reseed()                          # calculate the germination for each rotation phase
    pas.green_and_dry()                            # calculate the FOO lost when destocked and the FOO gained when grazed after establishment
    
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Params
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    try:
        model.del_component(model.p_germination_index_index_0_index_0)
        model.del_component(model.p_germination_index_index_0)
        model.del_component(model.p_germination_index)
        model.del_component(model.p_germination)
    except AttributeError:
        pass
    model.p_germination = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_germination_flrt, default=0, doc='pasture germination for each rotation')
    
    try:
        model.del_component(model.p_foo_grn_reseeding_index_index_0_index_0)
        model.del_component(model.p_foo_grn_reseeding_index_index_0)
        model.del_component(model.p_foo_grn_reseeding_index)
        model.del_component(model.p_foo_grn_reseeding)
    except AttributeError:
        pass
    model.p_foo_grn_reseeding = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_foo_grn_reseeding_flrt, default=0, doc='Change in grn FOO due to destocking and seeding of pastures')
    
    try:
        model.del_component(model.p_foo_dry_reseeding_index_index_0_index_0_index_0)
        model.del_component(model.p_foo_dry_reseeding_index_index_0_index_0)
        model.del_component(model.p_foo_dry_reseeding_index_index_0)
        model.del_component(model.p_foo_dry_reseeding_index)
        model.del_component(model.p_foo_dry_reseeding)
    except AttributeError:
        pass
    model.p_foo_dry_reseeding = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=pas.p_foo_dry_reseeding_dflrt, default=0, doc='Change in dry FOO due to destocking and seeding of pastures')
    
    try:
        model.del_component(model.p_foo_end_grnha_index_index_0_index_0_index_0)
        model.del_component(model.p_foo_end_grnha_index_index_0_index_0)
        model.del_component(model.p_foo_end_grnha_index_index_0)
        model.del_component(model.p_foo_end_grnha_index)
        model.del_component(model.p_foo_end_grnha)
    except AttributeError:
        pass
    model.p_foo_end_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_foo_end_grnha_goflt, default=0, doc='Green Foo at the end of the period')
    
    try:
        model.del_component(model.p_foo_start_grnha_index_index_0_index_0)
        model.del_component(model.p_foo_start_grnha_index_index_0)
        model.del_component(model.p_foo_start_grnha_index)
        model.del_component(model.p_foo_start_grnha)
    except AttributeError:
        pass
    model.p_foo_start_grnha = pe.Param(model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_foo_start_grnha_oflt, default=0, doc='Green Foo at the start of the period')
    
    try:
        model.del_component(model.p_senesce_grnha_index_index_0_index_0_index_0_index_0)
        model.del_component(model.p_senesce_grnha_index_index_0_index_0_index_0)
        model.del_component(model.p_senesce_grnha_index_index_0_index_0)
        model.del_component(model.p_senesce_grnha_index_index_0)
        model.del_component(model.p_senesce_grnha_index)
        model.del_component(model.p_senesce_grnha)
    except AttributeError:
        pass
    model.p_senesce_grnha = pe.Param(model.s_dry_groups, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_senesce_grnha_dgoflt, default=0, doc='Green pasture senescence into high and low quality dry pasture pools')
    
    try:
        model.del_component(model.p_me_cons_grnha_index_index_0_index_0_index_0_index_0)
        model.del_component(model.p_me_cons_grnha_index_index_0_index_0_index_0)
        model.del_component(model.p_me_cons_grnha_index_index_0_index_0)
        model.del_component(model.p_me_cons_grnha_index_index_0)
        model.del_component(model.p_me_cons_grnha_index)
        model.del_component(model.p_me_cons_grnha)
    except AttributeError:
        pass
    model.p_me_cons_grnha = pe.Param(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_me_cons_grnha_egoflt, default=0, doc='Total ME from grazing a hectare')
    
    try:
        model.del_component(model.p_volume_grnha_index_index_0_index_0_index_0_index_0)
        model.del_component(model.p_volume_grnha_index_index_0_index_0_index_0)
        model.del_component(model.p_volume_grnha_index_index_0_index_0)
        model.del_component(model.p_volume_grnha_index_index_0)
        model.del_component(model.p_volume_grnha_index)
        model.del_component(model.p_volume_grnha)
    except AttributeError:
        pass
    model.p_volume_grnha = pe.Param(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_volume_grnha_egoflt, default=0, doc='Total Vol from grazing a hectare')
    
    try:
        model.del_component(model.p_dry_mecons_t_index_index_0_index_0)
        model.del_component(model.p_dry_mecons_t_index_index_0)
        model.del_component(model.p_dry_mecons_t_index)
        model.del_component(model.p_dry_mecons_t)
    except AttributeError:
        pass
    model.p_dry_mecons_t = pe.Param(model.s_sheep_pools, model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_mecons_t_edft, default=0, doc='Total ME from grazing a tonne of dry feed')
    
    try:
        model.del_component(model.p_dry_volume_t_index_index_0)
        model.del_component(model.p_dry_volume_t_index)
        model.del_component(model.p_dry_volume_t)
    except AttributeError:
        pass
    model.p_dry_volume_t = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_volume_t_dft, default=0, doc='Total Vol from grazing a tonne of dry feed')
    
    try:
        model.del_component(model.p_dry_transfer_t_index_index_0)
        model.del_component(model.p_dry_transfer_t_index)
        model.del_component(model.p_dry_transfer_t)
    except AttributeError:
        pass
    model.p_dry_transfer_t = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_transfer_t_dft, default=0, doc='quantity of dry feed transfered out of the period to the next')
    
    try:
        model.del_component(model.p_dry_removal_t_index_index_0)
        model.del_component(model.p_dry_removal_t_index)
        model.del_component(model.p_dry_removal_t)
    except AttributeError:
        pass
    model.p_dry_removal_t = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=pas.p_dry_removal_t_dft, default=0, doc='quantity of dry feed removed for sheep to consume 1t')
    
    try:
        model.del_component(model.p_nap_index_index_0)
        model.del_component(model.p_nap_index)
        model.del_component(model.p_nap)
    except AttributeError:
        pass
    model.p_nap = pe.Param(model.s_dry_groups, model.s_lmus, model.s_pastures, initialize=pas.p_nap_dlt, default=0, doc='non arable pasture')
    
    try:
        model.del_component(model.p_erosion_index_index_0)
        model.del_component(model.p_erosion_index)
        model.del_component(model.p_erosion)
    except AttributeError:
        pass
    model.p_erosion = pe.Param(model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=pas.p_erosion_flt, default=0, doc='erosion limit in each period')
    
    try:
        model.del_component(model.p_phase_area_index_index_0)
        model.del_component(model.p_phase_area_index)
        model.del_component(model.p_phase_area)
    except AttributeError:
        pass
    model.p_phase_area = pe.Param(model.s_feed_periods, model.s_phases, model.s_pastures, initialize=pas.p_phase_area_frt, default=0, doc='pasture area in each rotation for each feed period')
    
    try:
        model.del_component(model.p_pas_sow_index_index_0_index_0)
        model.del_component(model.p_pas_sow_index_index_0)
        model.del_component(model.p_pas_sow_index)
        model.del_component(model.p_pas_sow)
    except AttributeError:
        pass
    model.p_pas_sow = pe.Param(model.s_periods, model.s_lmus, model.s_phases, model.s_landuses, initialize=pas.p_pas_sow_plrt, default=0, doc='pasture sown for each rotation')
    
    try:
        model.del_component(model.p_poc_con_index_index_0)
        model.del_component(model.p_poc_con_index)
        model.del_component(model.p_poc_con)
    except AttributeError:
        pass
    model.p_poc_con = pe.Param(model.s_feed_periods ,model.s_lmus, model.s_pastures, initialize=pas.poc_con(),default=0, doc='consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')
    
    try:
        model.del_component(model.p_poc_md_index)
        model.del_component(model.p_poc_md)
    except AttributeError:
        pass
    model.p_poc_md = pe.Param(model.s_feed_periods, model.s_pastures, initialize=pas.poc_md(),default=0, doc='md of pasture on crop paddocks for each feed period')
    
    try:
        model.del_component(model.p_poc_vol_index)
        model.del_component(model.p_poc_vol)
    except AttributeError:
        pass
    model.p_poc_vol = pe.Param(model.s_feed_periods, model.s_pastures, initialize=pas.poc_vol(),default=0, doc='vol (ri intake) of pasture on crop paddocks for each feed period')
    
    
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Local constraints
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    try:
        model.del_component(model.con_greenpas_index_index_0)
        model.del_component(model.con_greenpas_index)
        model.del_component(model.con_greenpas)
    except AttributeError:
        pass
    def greenpas(model,f,l,t):
        return sum(model.v_phase_area[r,l] * (model.p_germination[f,l,r,t] + model.p_foo_grn_reseeding[f,l,r,t])for r in model.s_phases if model.p_germination[f,l,r,t] !=0 or model.p_foo_grn_reseeding[f,l,r,t] !=0)         \
                       + sum(sum(sum(model.v_greenpas_ha[e,g,o,f,l,t] for e in model.s_sheep_pools) * (-model.p_foo_start_grnha[o,f,l,t] + model.p_foo_end_grnha[g,o,f,l,t])for g in model.s_grazing_int) for o in model.s_foo_levels) >=0
    model.con_greenpas = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule = greenpas, doc='green pasture of each type available on each soil type in each feed period')
    
    try:
        model.del_component(model.con_drypas_index_index_0)
        model.del_component(model.con_drypas_index)
        model.del_component(model.con_drypas)
    except AttributeError:
        pass
    def drypas(model,d,f,t):
        return sum(sum(model.v_greenpas_ha[e,g,o,f,l,t] * model.p_senesce_grnha[d,g,o,f,l,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus)         \
                            - model.v_drypas_consumed[e,d,f,t] * model.p_dry_removal_t[d,f,t] for e in model.s_sheep_pools) + model.v_drypas_transfer[d,f,t] * (model.p_dry_transfer_t[d,f,t] -1000) >=0
    model.con_drypas = pe.Constraint(model.s_dry_groups, model.s_feed_periods, model.s_pastures, rule = drypas, doc='High and low quality dry pasture of each type available in each period')
    
    try:
        model.del_component(model.con_pasarea_index_index_0)
        model.del_component(model.con_pasarea_index)
        model.del_component(model.con_pasarea)
    except AttributeError:
        pass
    def pasarea(model,f,l,t):
        return sum(model.v_phase_area[r,l] * model.p_phase_area[f,r,t] for r in model.s_phases)   \
                        - sum(model.v_greenpas_ha[e,g,o,f,l,t] for e in model.s_sheep_pools for g in model.s_grazing_int for o in model.s_foo_levels) >=0
    model.con_pasarea = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule = pasarea, doc='Pasture area row for growth constraint of each type on each soil for each feed period (ha)')
    
    #^once r is added to erosion limit, add an if statement with in sum for r in phases ; if model.p_erosion[f,l,r,t] != 0
    try:
        model.del_component(model.con_erosion)
    except AttributeError:
        pass
    def erosion(model,t):
        return sum(sum(sum(model.v_greenpas_ha[e,g,o,f,l,t] for e in model.s_sheep_pools) *  (model.p_foo_end_grnha[g,o,f,l,t] + sum(model.p_senesce_grnha[d,g,o,f,l,t] for d in model.s_dry_groups))for g in model.s_grazing_int for o in model.s_foo_levels) \
                   - sum(model.v_phase_area[r,l] for r in model.s_phases) * model.p_erosion[f,l,t] for f in model.s_feed_periods for l in model.s_lmus) >=0
    model.con_erosion = pe.Constraint(model.s_pastures, rule = erosion, doc='total pasture available of each type on each soil type in each feed period')
    
    
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
### Variables
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

model.v_greenpas_ha = pe.Var(model.s_sheep_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, bounds = (0,None) , doc='hectares grazed each period for each grazing intensity on each soil in each period')
model.v_drypas_consumed = pe.Var(model.s_sheep_pools, model.s_dry_groups, model.s_feed_periods, model.s_pastures, bounds = (0,None) , doc='tonnes of low and high quality dry feed consumed by each sheep pool in each feed period')
model.v_drypas_transfer = pe.Var(model.s_dry_groups, model.s_feed_periods, model.s_pastures, bounds = (0,None) , doc='tonnes of low and high quality dry feed transferred to the following periods in each feed period')
model.v_poc = pe.Var(model.s_sheep_pools, model.s_feed_periods, model.s_lmus, bounds = (0,None) , doc='tonnes of poc consumed by each sheep pool in each period on each lmu')


#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
### Functions for coremodel
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

##############
#sow         #
##############
def cropsow(model,p,k,l):
    return sum(model.p_pas_sow[p,l,r,k]*model.v_phase_area[r,l] for r in model.s_phases if model.p_pas_sow[p,l,r,k] != 0) 

##############
#MD          #
##############
def pas_md(model,e,f):
    return sum(model.v_greenpas_ha[e,g,o,f,l,t] * model.p_me_cons_grnha[e,g,o,f,l,t] + model.v_drypas_consumed[e,d,f,t] * model.p_dry_mecons_t[e,d,f,t] \
               + model.v_poc[e,f,l] * model.p_poc_md[f] for d in model.s_dry_groups for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus for t in model.s_pastures)

##############
#Vol         #
##############
def pas_vol(model,e,f):
    return sum(model.v_greenpas_ha[e,g,o,f,l,t] * model.p_volume_grnha[g,o,f,l,t] + model.v_drypas_consumed[e,d,f,t] * model.p_dry_volume_t[d,f,t] \
               + model.v_poc[e,f,l] * model.p_poc_vol[f] for d in model.s_dry_groups for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus for t in model.s_pastures)

