# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 15:00:25 2019

@author: John
"""
#python modules
from pyomo import environ as pe
import PropertyInputs as pinp

#AFO modules
import Pasture as pas


def paspyomo_precalcs(params, r_vals, nv):
    pas.f_pasture(params, r_vals, nv)

def paspyomo_local(params, model):
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Variables
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    model.v_greenpas_ha = pe.Var(model.s_feed_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods,
                                 model.s_lmus, model.s_season_types, model.s_pastures,bounds=(0,None),
                                 doc='hectares grazed each period for each grazing intensity on each soil in each period')
    model.v_drypas_consumed = pe.Var(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_season_types,
                                     model.s_pastures, bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed consumed by each sheep pool in each feed period')
    model.v_drypas_transfer = pe.Var(model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed at end of the period transferred to the following periods in each feed period')
    model.v_nap_consumed = pe.Var(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_season_types,
                                  model.s_pastures, bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks consumed by each sheep pool in each feed period')
    model.v_nap_transfer = pe.Var(model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures,bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks transferred to the following periods in each feed period')
    model.v_poc = pe.Var(model.s_feed_pools, model.s_feed_periods, model.s_lmus, model.s_season_types, bounds=(0,None),
                         doc='tonnes of poc consumed by each sheep pool in each period on each lmu')

    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Params
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################

    model.p_pasture_area = pe.Param(model.s_phases, model.s_pastures, initialize=params['pasture_area_rt'], default=0, doc='pasture area of each rotation')
    
    model.p_germination = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_germination_p6lrzt'], default=0, mutable=False, doc='pasture germination for each rotation')

    model.p_foo_grn_reseeding = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_foo_grn_reseeding_p6lrzt'], default=0, mutable=False, doc='Change in grn FOO due to destocking and restocking of resown pastures')
    
    model.p_foo_dry_reseeding = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_foo_dry_reseeding_dp6lrzt'], default=0, mutable=False, doc='Change in dry FOO due to destocking and seeding of pastures')
    
    model.p_foo_end_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_foo_end_grnha_gop6lzt'], default=0, mutable=False, doc='Green Foo at the end of the period')
    
    model.p_foo_start_grnha = pe.Param(model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_foo_start_grnha_op6lzt'], default=0, mutable=False, doc='Green Foo at the start of the period')
    
    model.p_senesce_grnha = pe.Param(model.s_dry_groups, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_senesce_grnha_dgop6lzt'], default=0, mutable=False, doc='Green pasture senescence into high and low quality dry pasture pools')
    
    model.p_me_cons_grnha = pe.Param(model.s_feed_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_me_cons_grnha_fgop6lzt'], default=0, mutable=False, doc='Total ME from grazing a hectare')
    
    model.p_volume_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_volume_grnha_gop6lzt'], default=0, mutable=False, doc='Total Vol from grazing a hectare')
    
    model.p_dry_mecons_t = pe.Param(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_mecons_t_fdp6zt'], default=0, mutable=False, doc='Total ME from grazing a tonne of dry feed')
    
    model.p_dry_volume_t = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_volume_t_dp6zt'], default=0, mutable=False, doc='Total Vol from grazing a tonne of dry feed')
    
    model.p_dry_transfer_prov_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_transfer_prov_t_p6zt'], default=0, mutable=False, doc='quantity of dry feed transferred out of the previous period to the current (allows for decay)')

    model.p_dry_transfer_req_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_transfer_req_t_p6zt'], default=0, mutable=False, doc='quantity of dry feed required to transfer a tonne of dry feed to the following period (this parameter is always 1000 unless dry feed doesnt exist)')

    model.p_dry_removal_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_removal_t_p6zt'], default=0, doc='quantity of dry feed removed for sheep to consume 1t, accounts for trampling')
    
    model.p_nap = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_nap_dp6lrzt'], default=0, mutable=False, doc='pasture on non arable areas in crop paddocks')
    
    model.p_nap_prop = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_harvest_period_prop'], default=0, mutable=False, doc='proportion of the way through each period nap becomes available')
    
    model.p_erosion = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_erosion_p6lrzt'], default=0, doc='erosion limit in each period')
    
    model.p_phase_area = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_phase_area_p6lrzt'], default=0, mutable=False, doc='pasture area in each rotation for each feed period')
    
    model.p_pas_sow = pe.Param(model.s_labperiods, model.s_lmus, model.s_phases, model.s_landuses, model.s_season_types, initialize=params['p_pas_sow_p5lrkz'], default=0, mutable=False, doc='pasture sown for each rotation')
    
    model.p_poc_con = pe.Param(model.s_feed_periods ,model.s_lmus, model.s_season_types, initialize=params['p_poc_con_p6lz'],default=0, doc='available consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')

    model.p_poc_md = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, initialize=params['p_poc_md_fp6z'],default=0, doc='md of pasture on crop paddocks for each feed period')
    
    model.p_poc_vol = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_poc_vol_p6z'],default=0, mutable=False, doc='vol (ri intake) of pasture on crop paddocks for each feed period')
    
    
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Local constraints
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    l_fp = list(model.s_feed_periods)#have to convert to a list first because indexing of an ordered set starts at 1
    def greenpas(model,p6,l,z,t):
        p6s = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if any(model.p_foo_start_grnha[o,p6,l,z,t] for o in model.s_foo_levels):
            return sum(model.v_phase_area[z,r,l] * (-model.p_germination[p6,l,r,z,t] - model.p_foo_grn_reseeding[p6,l,r,z,t]) for r in model.s_phases
                       if pe.value(model.p_germination[p6,l,r,z,t])!=0 or model.p_foo_grn_reseeding[p6,l,r,z,t]!=0)         \
                            + sum(model.v_greenpas_ha[f,g,o,p6,l,z,t] * model.p_foo_start_grnha[o,p6,l,z,t]   \
                            - model.v_greenpas_ha[f,g,o,p6s,l,z,t] * model.p_foo_end_grnha[g,o,p6s,l,z,t] for f in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
        else:
            return pe.Constraint.Skip
    #todo the greenpas (FOO) and pasarea (ha) could be replaced by a grnha constraint that passes area and foo together. Needs a FooB (base level) and reseeding foo removal and addition associated with the reseeding rotation phases
    model.con_greenpas = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = greenpas, doc='green pasture of each type available on each soil type in each feed period')

    def drypas(model,d,p6,z,t):
        p6s = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if model.p_dry_removal_t[p6,z,t] == 0 and model.p_dry_transfer_req_t[p6,z,t] == 0:
            return pe.Constraint.Skip
        else:
            return sum(sum(- model.v_greenpas_ha[f,g,o,p6s,l,z,t] * model.p_senesce_grnha[d,g,o,p6s,l,z,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus)        \
                       + model.v_drypas_consumed[f,d,p6,z,t] * model.p_dry_removal_t[p6,z,t] for f in model.s_feed_pools) \
                   - model.v_drypas_transfer[d,p6s,z,t] * model.p_dry_transfer_prov_t[p6s,z,t] \
                   + model.v_drypas_transfer[d,p6,z,t] * model.p_dry_transfer_req_t[p6,z,t] <=0
    model.con_drypas = pe.Constraint(model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, rule = drypas, doc='High and low quality dry pasture of each type available in each period')

    def nappas(model,d,p6,z,t):
        p6s = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if model.p_dry_removal_t[p6,z,t] == 0 and model.p_dry_transfer_req_t[p6,z,t] == 0:
            return pe.Constraint.Skip
        else:
            return sum(sum(- model.v_phase_area[z,r,l] * model.p_nap[d,p6,l,r,z,t] for r in model.s_phases for l in model.s_lmus if pe.value(model.p_nap[d,p6,l,r,z,t]) != 0)
                       + model.v_nap_consumed[f,d,p6,z,t] * model.p_dry_removal_t[p6,z,t] for f in model.s_feed_pools) \
                   - model.v_nap_transfer[d,p6s,z,t] * model.p_dry_transfer_prov_t[p6s,z,t] \
                   + model.v_nap_transfer[d,p6,z,t] * model.p_dry_transfer_req_t[p6,z,t] <=0
    model.con_nappas = pe.Constraint(model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, rule = nappas, doc='High and low quality dry pasture of each type available in each period')
    
    def pasarea(model,p6,l,z,t):
        return sum(-model.v_phase_area[z,r,l] * model.p_phase_area[p6,l,r,z,t] for r in model.s_phases if pe.value(model.p_phase_area[p6,l,r,z,t]) != 0)   \
                        + sum(model.v_greenpas_ha[f,g,o,p6,l,z,t] for f in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
    model.con_pasarea = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = pasarea, doc='Pasture area row for growth constraint of each type on each soil for each feed period (ha)')
    
    def erosion(model,p6,l,z,t):
        #senescence is included here because it is passed into the dry feed pool in the following fp. Thus senesced feed is not included in green or dry pasture in the period it senesced.
        return sum(sum(model.v_greenpas_ha[f,g,o,p6,l,z,t] for f in model.s_feed_pools) * -(model.p_foo_end_grnha[g,o,p6,l,z,t] +
                   sum(model.p_senesce_grnha[d,g,o,p6,l,z,t] for d in model.s_dry_groups)) for g in model.s_grazing_int for o in model.s_foo_levels) \
                -  sum(model.v_drypas_transfer[d,p6,z,t] * 1000 for d in model.s_dry_groups) \
                + sum(model.v_phase_area[z,r,l]  * model.p_erosion[p6,l,r,z,t] for r in model.s_phases if pe.value(model.p_erosion[p6,l,r,z,t]) != 0) <=0
    model.con_erosion = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = erosion, doc='total pasture available of each type on each soil type in each feed period')

    


#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
### Functions for coremodel
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

##############
#sow         #
##############
def passow(model,p,k,l,z):
    if any(model.p_pas_sow[p,l,r,k,z] for r in model.s_phases):
        return sum(model.p_pas_sow[p,l,r,k,z]*model.v_phase_area[z,r,l] for r in model.s_phases if pe.value(model.p_pas_sow[p,l,r,k,z]) != 0)
    else:
        return 0

##############
#ME          #
##############
def pas_me(model,p6,f,z):
    return sum(sum(model.v_greenpas_ha[f,g,o,p6,l,z,t] * model.p_me_cons_grnha[f,g,o,p6,l,z,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus) \
               + sum(model.v_drypas_consumed[f,d,p6,z,t] * model.p_dry_mecons_t[f,d,p6,z,t] for d in model.s_dry_groups) for t in model.s_pastures) \
               + sum(model.v_poc[f,p6,l,z] * model.p_poc_md[f,p6,z] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast

def nappas_me(model,p6,f,z):
    return sum(model.v_nap_consumed[f,d,p6,z,t] * model.p_dry_mecons_t[f,d,p6,z,t] for d in model.s_dry_groups for t in model.s_pastures)

##############
#Vol         #
##############
def pas_vol(model,p6,f,z):
    return sum(sum(model.v_greenpas_ha[f,g,o,p6,l,z,t] * model.p_volume_grnha[g,o,p6,l,z,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus) \
               + sum(model.v_drypas_consumed[f,d,p6,z,t] * model.p_dry_volume_t[d,p6,z,t] \
               + model.v_nap_consumed[f,d,p6,z,t] * model.p_dry_volume_t[d,p6,z,t] for d in model.s_dry_groups) for t in model.s_pastures)\
               + sum(model.v_poc[f,p6,l,z] * model.p_poc_vol[p6,z] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast
