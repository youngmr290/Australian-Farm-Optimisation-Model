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


def paspyomo_precalcs(params,r_vals,ev):
    pas.f_pasture(params,r_vals,ev)

def paspyomo_local(params, model):
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Variables
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    model.v_greenpas_ha = pe.Var(model.s_feed_pools,model.s_grazing_int,model.s_foo_levels,model.s_feed_periods,
                                 model.s_lmus,model.s_pastures,bounds=(0,None),
                                 doc='hectares grazed each period for each grazing intensity on each soil in each period')
    model.v_drypas_consumed = pe.Var(model.s_feed_pools,model.s_dry_groups,model.s_feed_periods,model.s_pastures,
                                     bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed consumed by each sheep pool in each feed period')
    model.v_drypas_transfer = pe.Var(model.s_dry_groups,model.s_feed_periods,model.s_pastures,bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed at end of the period transferred to the following periods in each feed period')
    model.v_nap_consumed = pe.Var(model.s_feed_pools,model.s_dry_groups,model.s_feed_periods,model.s_pastures,
                                  bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks consumed by each sheep pool in each feed period')
    model.v_nap_transfer = pe.Var(model.s_dry_groups,model.s_feed_periods,model.s_pastures,bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks transferred to the following periods in each feed period')
    model.v_poc = pe.Var(model.s_feed_pools,model.s_feed_periods,model.s_lmus,bounds=(0,None),
                         doc='tonnes of poc consumed by each sheep pool in each period on each lmu')

    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Params
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]


    model.p_pasture_area = pe.Param(model.s_phases, model.s_pastures, initialize=params['pasture_area_rt'], default=0, doc='pasture area of each rotation')
    
    model.p_germination = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params[season]['p_germination_flrt'], default=0, mutable=False, doc='pasture germination for each rotation')

    model.p_foo_grn_reseeding = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params[season]['p_foo_grn_reseeding_flrt'], default=0, mutable=False, doc='Change in grn FOO due to destocking and restocking of resown pastures')
    
    model.p_foo_dry_reseeding = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params[season]['p_foo_dry_reseeding_dflrt'], default=0, mutable=False, doc='Change in dry FOO due to destocking and seeding of pastures')
    
    model.p_foo_end_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=params[season]['p_foo_end_grnha_goflt'], default=0, mutable=False, doc='Green Foo at the end of the period')
    
    model.p_foo_start_grnha = pe.Param(model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=params[season]['p_foo_start_grnha_oflt'], default=0, mutable=False, doc='Green Foo at the start of the period')
    
    model.p_senesce_grnha = pe.Param(model.s_dry_groups, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=params[season]['p_senesce_grnha_dgoflt'], default=0, mutable=False, doc='Green pasture senescence into high and low quality dry pasture pools')
    
    model.p_me_cons_grnha = pe.Param(model.s_feed_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=params[season]['p_me_cons_grnha_vgoflt'], default=0, mutable=False, doc='Total ME from grazing a hectare')
    
    model.p_volume_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_pastures, initialize=params[season]['p_volume_grnha_goflt'], default=0, mutable=False, doc='Total Vol from grazing a hectare')
    
    model.p_dry_mecons_t = pe.Param(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=params[season]['p_dry_mecons_t_vdft'], default=0, mutable=False, doc='Total ME from grazing a tonne of dry feed')
    
    model.p_dry_volume_t = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_pastures, initialize=params[season]['p_dry_volume_t_dft'], default=0, mutable=False, doc='Total Vol from grazing a tonne of dry feed')
    
    model.p_dry_transfer_t = pe.Param(model.s_feed_periods, model.s_pastures, initialize=params[season]['p_dry_transfer_t_ft'], default=0, mutable=False, doc='quantity of dry feed transferred out of the period to the next')
    
    model.p_dry_removal_t = pe.Param(model.s_feed_periods, model.s_pastures, initialize=params['p_dry_removal_t_ft'], default=0, doc='quantity of dry feed removed for sheep to consume 1t, accounts for trampling')
    
    model.p_nap = pe.Param(model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params[season]['p_nap_dflrt'], default=0, mutable=False, doc='pasture on non arable areas in crop paddocks')
    
    model.p_nap_prop = pe.Param(model.s_feed_periods, initialize=params[season]['p_harvest_period_prop'], default=0, mutable=False, doc='proportion of the way through each period nap becomes available')
    
    model.p_erosion = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params['p_erosion_flrt'], default=0, doc='erosion limit in each period')
    
    model.p_phase_area = pe.Param(model.s_feed_periods, model.s_lmus, model.s_phases, model.s_pastures, initialize=params[season]['p_phase_area_flrt'], default=0, mutable=False, doc='pasture area in each rotation for each feed period')
    
    model.p_pas_sow = pe.Param(model.s_labperiods, model.s_lmus, model.s_phases, model.s_landuses, initialize=params[season]['p_pas_sow_plrk'], default=0, mutable=False, doc='pasture sown for each rotation')
    
    model.p_poc_con = pe.Param(model.s_feed_periods ,model.s_lmus, initialize=params['p_poc_con_fl'],default=0, doc='available consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')

    model.p_poc_md = pe.Param(model.s_feed_pools, model.s_feed_periods, initialize=params['p_poc_md_vf'],default=0, doc='md of pasture on crop paddocks for each feed period')
    
    model.p_poc_vol = pe.Param(model.s_feed_periods, initialize=params[season]['p_poc_vol_f'],default=0, mutable=False, doc='vol (ri intake) of pasture on crop paddocks for each feed period')
    
    
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    ### Local constraints
    #####################################################################################################################################################################################################
    #####################################################################################################################################################################################################
    l_fp = list(model.s_feed_periods)#have to convert to a list first because indexing of an ordered set starts at 1
    def greenpas(model,f,l,t):
        fs = l_fp[l_fp.index(f) - 1] #need the activity level from last feed period
        if any(model.p_foo_start_grnha[o,f,l,t] for o in model.s_foo_levels):
            return sum(model.v_phase_area[r,l] * (-model.p_germination[f,l,r,t] - model.p_foo_grn_reseeding[f,l,r,t]) for r in model.s_phases
                       if pe.value(model.p_germination[f,l,r,t])!=0 or model.p_foo_grn_reseeding[f,l,r,t]!=0)         \
                            + sum(model.v_greenpas_ha[v,g,o,f,l,t] * model.p_foo_start_grnha[o,f,l,t]   \
                            - model.v_greenpas_ha[v,g,o,fs,l,t] * model.p_foo_end_grnha[g,o,fs,l,t] for v in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
        else:
            return pe.Constraint.Skip
    #todo the greenpas (FOO) and pasarea (ha) could be replaced by a grnha constraint that passes area and foo together. Needs a FooB (base level) and reseeding foo removal and addition associated with the reseeding rotation phases
    model.con_greenpas = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule = greenpas, doc='green pasture of each type available on each soil type in each feed period')

    def drypas(model,d,f,t):
        fs = l_fp[l_fp.index(f) - 1] #need the activity level from last feed period
        return sum(sum(model.v_greenpas_ha[v,g,o,fs,l,t] * -model.p_senesce_grnha[d,g,o,fs,l,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus)        \
                       + model.v_drypas_consumed[v,d,f,t] * model.p_dry_removal_t[f,t] for v in model.s_feed_pools) \
                       - model.v_drypas_transfer[d,fs,t] * model.p_dry_transfer_t[fs,t] + model.v_drypas_transfer[d,f,t] * 1000 <=0 #minus 1000 is what you are transferring into constraint, p_dry_transfer is how much you get in the current period if you transferred 1t from previous period (not 1000 because you have to account for deterioration)
    model.con_drypas = pe.Constraint(model.s_dry_groups, model.s_feed_periods, model.s_pastures, rule = drypas, doc='High and low quality dry pasture of each type available in each period')
    
    def nappas(model,d,f,t):
        fs = l_fp[l_fp.index(f) - 1] #need the activity level from last feed period
        return sum(sum(sum(model.v_phase_area[r,l] * -model.p_nap[d,f,l,r,t] for r in model.s_phases if pe.value(model.p_nap[d,f,l,r,t]) != 0)for l in model.s_lmus)        \
                       + model.v_nap_consumed[v,d,f,t] * model.p_dry_removal_t[f,t] for v in model.s_feed_pools) \
                       - model.v_nap_transfer[d,fs,t] * model.p_dry_transfer_t[fs,t] + model.v_nap_transfer[d,f,t] * 1000 <=0 #minus 1000 is what you are transferring into constraint, p_dry_transfer is how much you get in the current period if you transferred 1t from previous period (not 1000 because you have to account for deterioration)
    model.con_nappas = pe.Constraint(model.s_dry_groups, model.s_feed_periods, model.s_pastures, rule = nappas, doc='High and low quality dry pasture of each type available in each period')
    
    def pasarea(model,f,l,t):
        return sum(-model.v_phase_area[r,l] * model.p_phase_area[f,l,r,t] for r in model.s_phases if pe.value(model.p_phase_area[f,l,r,t]) != 0)   \
                        + sum(model.v_greenpas_ha[v,g,o,f,l,t] for v in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
    model.con_pasarea = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule = pasarea, doc='Pasture area row for growth constraint of each type on each soil for each feed period (ha)')
    
    def erosion(model,f,l,t):
        #senescence is included here because it is passed into the dry feed pool in the following fp. Thus senesced feed is not included in green or dry pasture in the period it senesced.
        return sum(sum(model.v_greenpas_ha[v,g,o,f,l,t] for v in model.s_feed_pools) * -(model.p_foo_end_grnha[g,o,f,l,t] +
                   sum(model.p_senesce_grnha[d,g,o,f,l,t] for d in model.s_dry_groups)) for g in model.s_grazing_int for o in model.s_foo_levels) \
                -  sum(model.v_drypas_transfer[d,f,t] * 1000 for d in model.s_dry_groups) \
                + sum(model.v_phase_area[r,l]  * model.p_erosion[f,l,r,t] for r in model.s_phases if pe.value(model.p_erosion[f,l,r,t]) != 0) <=0
    model.con_erosion = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule = erosion, doc='total pasture available of each type on each soil type in each feed period')

    


#####################################################################################################################################################################################################
#####################################################################################################################################################################################################
### Functions for coremodel
#####################################################################################################################################################################################################
#####################################################################################################################################################################################################

##############
#sow         #
##############
def passow(model,p,k,l):
    if any(model.p_pas_sow[p,l,r,k] for r in model.s_phases):
        return sum(model.p_pas_sow[p,l,r,k]*model.v_phase_area[r,l] for r in model.s_phases if pe.value(model.p_pas_sow[p,l,r,k]) != 0)
    else:
        return 0

##############
#ME          #
##############
def pas_me(model,v,f):
    return sum(sum(model.v_greenpas_ha[v,g,o,f,l,t] * model.p_me_cons_grnha[v,g,o,f,l,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus) \
               + sum(model.v_drypas_consumed[v,d,f,t] * model.p_dry_mecons_t[v,d,f,t] for d in model.s_dry_groups) for t in model.s_pastures) \
               + sum(model.v_poc[v,f,l] * model.p_poc_md[v,f] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast

def nappas_me(model,v,f):
    return sum(model.v_nap_consumed[v,d,f,t] * model.p_dry_mecons_t[v,d,f,t] for d in model.s_dry_groups for t in model.s_pastures)

##############
#Vol         #
##############
def pas_vol(model,v,f):
    return sum(sum(model.v_greenpas_ha[v,g,o,f,l,t] * model.p_volume_grnha[g,o,f,l,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus) \
               + sum(model.v_drypas_consumed[v,d,f,t] * model.p_dry_volume_t[d,f,t] \
               + model.v_nap_consumed[v,d,f,t] * model.p_dry_volume_t[d,f,t] for d in model.s_dry_groups) for t in model.s_pastures)\
               + sum(model.v_poc[v,f,l] * model.p_poc_vol[f] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast
