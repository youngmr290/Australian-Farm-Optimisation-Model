# -*- coding: utf-8 -*-
"""
author: young
"""

#python modules
import pyomo.environ as pe

#AFO modules
from . import CropResidue as stub
from . import StructuralInputs as sinp
from . import PyomoFunctions as fpy

def stub_precalcs(params, r_vals, nv, cat_propn_s1_ks2):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    stub.crop_residue_all(params, r_vals, nv, cat_propn_s1_ks2)
    
    
    
def f1_stubpyomo_local(params, model, MP_lp_vars):
    ''' Builds pyomo variables, parameters and constraints'''
    ########################
    # active index sets    #
    ########################
    '''
    Stubble consumption and transfer are only biologically feasible in a subset of feed periods, seasons, 
    and categories (e.g. after harvest, before the end of the stubble season, and excluding terminal categories). 
    To reflect this, the model constructs sparse index sets from precomputed domain masks and builds variables 
    and constraints only on those sets.
    
    Some constraints require summing over related indices that may not exist everywhere, such as:

        - transfers from the previous feed period (p6_prev), and

        - summation over child season types (z8) where stubble is present.

    Rather than performing membership checks inside constraint rules (e.g. (p6,z,k) in s_stub_base_p6zk), 
    which is slow and error-prone at model build time, we pre-build indexed mappings that identify valid 
    combinations (e.g. mapping (p6,k) to the set of valid z values). These mappings ensure that transfer 
    terms are only generated where stubble actually exists.

    This approach allows constraints to be written once for all relevant feed periods while automatically 
    evaluating transfer contributions to zero when predecessor periods or seasons have no stubble. 
    It preserves biological realism, avoids invalid references (such as transfers from pre-harvest periods), 
    and significantly improves model build performance by keeping the formulation sparse.

    Note: although V_transfer could be masked slightly more because technically you cant transfer out of the final 
    period this has not been included because it got a bit complex and there can be cases where z unclustering could 
    cause errors. Therefore just use the same mask for everything.
    '''
    ##Build sets that only have active items based on masks built in the precalcs.

    model.s_stub_base_p6zk = fpy.build_active_set(base_index=params['idx_stub_base_p6zk'])
    model.s_stub_base_p6zks1 = fpy.build_active_set(base_index=params['idx_stub_base_p6zk'], suffix_sets=[model.s_stub_cat])
    model.s_stub_base_fp6zks1 = fpy.build_active_set(base_index=params['idx_stub_base_p6zk'], prefix_sets=[model.s_feed_pools], suffix_sets=[model.s_stub_cat])

    model.s_stub_base_qszp6fks1s2 = fpy.build_active_set(
        base_index=params['idx_stub_base_p6zk'],
        prefix_sets=[model.s_active_qs], suffix_sets=[model.s_feed_pools, model.s_stub_cat, model.s_biomass_uses],
        order=(0, 1, 3, 2, 5, 4, 6, 7))

    model.s_stub_base_qszp6ks1s2 = fpy.build_active_set(
        base_index=params['idx_stub_base_p6zk'],
        prefix_sets=[model.s_active_qs], suffix_sets=[model.s_stub_cat, model.s_biomass_uses],
        order=(0, 1, 3, 2, 4, 5, 6))

    model.s_stub_cat_A_prov_p6zks1s2 = fpy.build_active_set(base_index=params['idx_stub_cat_A_prov_p6zks1'], suffix_sets=[model.s_biomass_uses])

    # model.s_stub_transfer_p6zk = fpy.build_active_set(base_index=params['idx_stub_transfer_p6zk'])
    # model.s_stub_transfer_qszp6ks1s2 = fpy.build_active_set(
    #     base_index=params['idx_stub_transfer_p6zk'],
    #     prefix_sets=[model.s_active_qs], suffix_sets=[model.s_stub_cat, model.s_biomass_uses],
    #     order=(0, 1, 3, 2, 4, 5, 6)
    # )

    ###for constraints
    model.s_stub_within_qsp6zks1s2 = fpy.build_active_set(
        base_index=params['idx_stub_base_within_p6zk'],
        prefix_sets=[model.s_active_qs],
        suffix_sets=[model.s_stub_cat, model.s_biomass_uses],
    )
    model.s_stub_between_qsp6zks1s2 = fpy.build_active_set(
        base_index=params['idx_stub_base_between_p6zk'],
        prefix_sets=[model.s_active_qs_between_con],
        suffix_sets=[model.s_stub_cat, model.s_biomass_uses],
    )

    ###special set use in the constraints when summing the z8 axis - this stops summing z8 that dont exist. The alternative would be to
    def f1_init_z_by_p6k(model, p6, k):
        return params['stub_z8_by_p6k'].get((p6, k), [])
    model.s_stub_z8_by_p6k = pe.Set(model.s_feed_periods, model.s_crops, initialize=f1_init_z_by_p6k)

    def init_stub_k_by_p6z(m, p6, z):
        return params['stub_k_by_p6z'].get((p6, z), [])
    model.s_stub_k_by_p6z = pe.Set(model.s_feed_periods, model.s_season_types, initialize=init_stub_k_by_p6z)

    ###################
    # variable         #
    ###################
    ##stubble consumption
    model.v_stub_con = pe.Var(model.s_stub_base_qszp6fks1s2, bounds=(0.0,None),
                              doc='consumption of 1t of stubble')
    ##stubble transfer
    model.v_stub_transfer = pe.Var(model.s_stub_base_qszp6ks1s2, bounds=(0.0,None),
                                   doc='transfer of 1t of stubble to following period - 1t of stubble at the start of the period that is not consumed but is decayed')


    ####################
    #define parameters #
    ####################
    # model.p_rot_stubble = pe.Param(model.s_phases, model.s_crops, model.s_lmus, model.s_season_periods, model.s_season_types,
    #                                initialize=params['rot_stubble'], default=0.0, doc='stubble produced per ha of each rotation')

    model.p_stub_unavailable_frac_p6zk = pe.Param(model.s_stub_base_p6zk, initialize=params['stub_unavailable_frac_p6zk'],
                                 default = 0.0, mutable=False, doc='proportion of each feed period where grazing cannot occur either because harvest only occurs part way or because stubbles are destocked part way')
    
    model.p_stub_md = pe.Param(model.s_stub_base_fp6zks1, initialize=params['md'],
                               default = 0.0, mutable=False, doc='md from 1t of each stubble categories for each crop')

    model.p_stub_vol = pe.Param(model.s_stub_base_fp6zks1, initialize=params['vol'],
                                default = 0.0, mutable=False, doc='amount of intake volume required by 1t of each stubble category for each crop')
    
    model.p_a_prov = pe.Param(model.s_stub_cat_A_prov_p6zks1s2, initialize=params['cat_a_prov'],
                             default = 0.0, mutable=False, doc='cat A stubble provided at harvest from 1t of stubble')

    model.p_biomass2residue = pe.Param(model.s_crops, model.s_biomass_uses, initialize=params['biomass2residue_ks2'],
                             default = 0.0, mutable=False, doc='conversion of biomass to crop residue for each biomass use (harvesting as normal, baling for hay and grazing as fodder)')

    model.p_bc_prov = pe.Param(model.s_crops, model.s_stub_cat, model.s_biomass_uses, initialize=params['cat_transfer_prov'], default = 0.0,
                               doc='stubble B provided from 1t of cat A and stubble C provided from 1t of cat B')

    model.p_bc_req = pe.Param(model.s_crops, model.s_stub_cat, model.s_biomass_uses, initialize=params['cat_transfer_req'], default = 0.0,
                              doc='stubble required from the row inorder to consume cat B or cat C')

    model.p_stub_transfer_prov = pe.Param(model.s_stub_base_p6zk, initialize=params['stub_transfer_prov'],
                                   default = 0.0, mutable=False, doc='stubble available for consumption. Transferred in from last period or harvest.')

    model.p_stub_transfer_req = pe.Param(model.s_stub_base_p6zk, initialize=params['stub_transfer_req'],
                                   default = 0.0, mutable=False, doc='stubble required for transfer to the next period')

    model.co2e_stub_cons_p6zks1 = pe.Param(model.s_stub_base_p6zks1, initialize=params['co2e_stub_cons_p6zks1'],
                                default = 0.0, mutable=False, doc='kgs of co2e saved by the consumption of 1t stubble for each crop')

    model.co2e_stub_production_zk = pe.Param(model.s_season_types, model.s_crops, initialize=params['co2e_stub_production_zk'],
                                default = 0.0, mutable=False, doc='kgs of co2e produced from a 1t of crop residue at harvest')


    ########################
    #call local constraint #
    ########################
    f_con_cropresidue_within(model)
    f_con_cropresidue_between(model, MP_lp_vars)



###################
#local constraint #
###################
def f_con_cropresidue_within(model):
    ''' Links the consumption of a given category with the provision of another category or the transfer of
    stubble to the following period. E.g. category A consumption provides category B. Category B can either be
    consumed (hence providing category C) or transferred to the following period.
    '''
    ##stubble transfer from category to category and period to period
    ##s2 required because cat propn can vary across s2

    stub_cats = list(model.s_stub_cat)
    prev_sc = {stub_cats[i]: stub_cats[i - 1] for i in range(len(stub_cats))}

    p6_list = list(model.s_feed_periods)
    prev_p6 = {p6_list[i]: p6_list[i - 1] for i in range(len(p6_list))}

    def cropresidue_transfer_within(model,q,s,p6,z9,k,sc,s2):
        sc_prev = prev_sc[sc]
        p6_prev = prev_p6[p6]

        # Transfer in from previous feed period (only where stubble exists in p6_prev for crop k)
        transfer_in = pe.quicksum(
            model.v_stub_transfer[q, s, z8, p6_prev, k, sc, s2]
            * model.p_stub_transfer_prov[p6_prev, z8, k]
            * model.p_parentz_provwithin_fp[p6_prev, z8, z9]
            for z8 in model.s_stub_z8_by_p6k[p6_prev, k]
        )

        # Category A provision from biomass use (harvest / bale / graze-as-fodder)
        if (p6, z9, k, sc, s2) in model.s_stub_cat_A_prov_p6zks1s2:
            harvest_prov = pe.quicksum(
                model.v_use_biomass[q, s, p7, z9, k, l, s2]
                * model.p_a_p6_p7[p7, p6, z9]
                * model.p_biomass2residue[k, s2]
                for p7 in model.s_season_periods
                for l in model.s_lmus
            ) * model.p_a_prov[p6, z9, k, sc, s2]
        else:
            harvest_prov = 0

        # Transfer out to next feed period (stub remaining at end of p6)
        transfer_out = (
                model.v_stub_transfer[q, s, z9, p6, k, sc, s2]
                * model.p_stub_transfer_req[p6, z9, k]
        )

        # Within-period category balance: consuming prev cat provides this cat; consuming this cat requires this cat
        consume_balance = pe.quicksum(
            - model.v_stub_con[q, s, z9, p6, f, k, sc_prev, s2] * model.p_bc_prov[k, sc_prev, s2]
            + model.v_stub_con[q, s, z9, p6, f, k, sc, s2] * model.p_bc_req[k, sc, s2]
            for f in model.s_feed_pools
        )

        # Supply must cover transfers and consumption requirements
        return -transfer_in - harvest_prov + transfer_out + consume_balance <= 0

    model.con_cropresidue_within = pe.Constraint(model.s_stub_within_qsp6zks1s2, rule=cropresidue_transfer_within, doc='stubble transfer between feed periods and stubble transfer between categories.')


def f_con_cropresidue_between(model, MP_lp_vars):
    ''' Links the consumption of a given category with the provision of another category, or the transfer of
    stubble to the following period. E.g. category A consumption provides category B. Category B can either be
    consumed (hence providing category C) or transferred to the following period.
    '''
    ##stubble transfer from category to category and period to period
    ##s2 required because cat propn can vary across s2
    stub_cats = list(model.s_stub_cat)
    prev_sc = {stub_cats[i]: stub_cats[i - 1] for i in range(len(stub_cats))}

    p6_list = list(model.s_feed_periods)
    prev_p6 = {p6_list[i]: p6_list[i - 1] for i in range(len(p6_list))}
    l_q = list(model.s_sequence_year_between_con)
    def cropresidue_transfer_between(model,q,s9,p6,z9,k,sc,s2):
        sc_prev = prev_sc[sc]
        p6_prev = prev_p6[p6]
        ###adjust q_prev for multi-period model
        if sinp.structuralsa['model_is_MP']:
            ####yr0 is SE so q_prev is q
            if q == l_q[0]:
                q_prev = q
                v_stub_transfer_hist = MP_lp_vars[str('v_stub_transfer')]  # q[0] is provided by the MP set up run.
            ####the final year is provided by both the previous year and itself (the final year is in equilibrium). Therefore the final year needs two constraints. This is achieved by making the q set 1 year longer than the modeled period (len_MP + 1). Then adjusting q and q_prev for the final q so that the final year is also in equilibrium.
            elif q == l_q[-1]:
                q = l_q[l_q.index(q) - 1]
                q_prev = q
                v_stub_transfer_hist = model.v_stub_transfer
            else:
                q_prev = l_q[l_q.index(q) - 1]
                v_stub_transfer_hist = model.v_stub_transfer
        else:
            q_prev = l_q[l_q.index(q) - 1]
            v_stub_transfer_hist = model.v_stub_transfer

        # --- transfer in from previous feed period across sequences/season types ---
        transfer_in = pe.quicksum(
            v_stub_transfer_hist[q_prev, s8, z8, p6_prev, k, sc, s2]
            * model.p_stub_transfer_prov[p6_prev, z8, k]
            * model.p_parentz_provbetween_fp[p6_prev, z8, z9]
            * (model.p_sequence_prov_qs8zs9[q_prev, s8, z8, s9]
               + model.p_endstart_prov_qsz[q_prev, s8, z8])
            for s8 in model.s_sequence
            if pe.value(model.p_wyear_inc_qs[q_prev, s8]) != 0
            for z8 in model.s_stub_z8_by_p6k[p6_prev, k]
        )

        # --- provision from biomass use (note the 1000 scaling in your original) ---
        if (p6, z9, k, sc, s2) in model.s_stub_cat_A_prov_p6zks1s2:
            harvest_prov = pe.quicksum(
                model.v_use_biomass[q, s9, p7, z9, k, l, s2]
                * 1000
                * model.p_a_p6_p7[p7, p6, z9]
                * model.p_biomass2residue[k, s2]
                for p7 in model.s_season_periods
                for l in model.s_lmus
            ) * model.p_a_prov[p6, z9, k, sc, s2]
        else:
            harvest_prov = 0

        # --- transfer out to next feed period ---
        transfer_out = (
            model.v_stub_transfer[q, s9, z9, p6, k, sc, s2]
            * model.p_stub_transfer_req[p6, z9, k]
        )

        # --- within-period category balance ---
        consume_balance = pe.quicksum(
            - model.v_stub_con[q, s9, z9, p6, f, k, sc_prev, s2] * model.p_bc_prov[k, sc_prev, s2]
            + model.v_stub_con[q, s9, z9, p6, f, k, sc, s2] * model.p_bc_req[k, sc, s2]
            for f in model.s_feed_pools
        )

        return -transfer_in - harvest_prov + transfer_out + consume_balance <= 0

    model.con_cropresidue_between = pe.Constraint(model.s_stub_between_qsp6zks1s2, rule=cropresidue_transfer_between, doc='stubble transfer between feed periods and stubble transfer between categories.')


###################
#constraint global#
###################
##stubble transfer from category to category and period to period
# def f_cropresidue_req_a(model,q,s,p7,z,k,sc):
#     '''
#     Calculate the total stubble required to consume the selected volume category A stubble in each period.
#
#     Used in global constraint (con_cropresidue_a). See CorePyomo
#     '''
#
#     return sum(model.v_stub_transfer[q,s,p6,z,k,sc] * model.p_a_req[p6,z,k,sc] * model.p_a_p6_p7[p7,p6,z]
#                for p6 in model.s_feed_periods if pe.value(model.p_a_req[p6,z,k,sc]) !=0)
#     # return sum(model.v_stub_con[q,s,f,p6,z,k,sc] * model.p_a_req[p6,z,k,sc] * model.p_a_p6_p7[p7,p6,z]
#     #            for f in model.s_feed_pools for p6 in model.s_feed_periods if pe.value(model.p_a_req[p6,z,k,sc]) !=0)


##stubble md
def f_cropresidue_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of stubble.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_stub_con[q,s,z,p6,f,k,sc,s2] * model.p_stub_md[f,p6,z,k,sc]
               for k in model.s_stub_k_by_p6z[p6, z]
               for sc in model.s_stub_cat
               for s2 in model.s_biomass_uses)
    
##stubble vol
def f_cropresidue_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of stubble.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_stub_con[q,s,z,p6,f,k,sc,s2] * model.p_stub_vol[f,p6,z,k,sc]
               for k in model.s_stub_k_by_p6z[p6, z]
               for sc in model.s_stub_cat
               for s2 in model.s_biomass_uses)


def f_cropresidue_consumption_emissions(model,q,s,p6,z):
    '''
    Calculate the emissions linked to consumption of stubble

    Used in global constraint (con_emissions). See BoundPyomo
    '''
    return sum(model.v_stub_con[q,s,z,p6,f,k,sc,s2] * model.co2e_stub_cons_p6zks1[p6,z,k,sc]
               for k in model.s_stub_k_by_p6z[p6, z]
               for f in model.s_feed_pools
               for sc in model.s_stub_cat
               for s2 in model.s_biomass_uses
            )

def f_cropresidue_production_emissions(model,q,s,p7,z):
    '''
    Calculate the emissions linked to the residue at harvest activity.
    
    This is separate function to above because it has p7 axis not p6.

    Used in global constraint (con_emissions). See BoundPyomo
    '''
    return sum(model.v_use_biomass[q,s,p7,z,k,l,s2] * model.p_biomass2residue[k,s2] * model.co2e_stub_production_zk[z,k]
               for l in model.s_lmus for k in model.s_crops for s2 in model.s_biomass_uses)


def f1_stub_consumed_in_harvest_period(model, q, s, p6, f, z):
    return pe.quicksum(
        (model.p_stub_unavailable_frac_p6zk[p6, z, k] / (1 - model.p_stub_unavailable_frac_p6zk[p6, z, k]))
        * model.v_stub_con[q, s, z, p6, f, k, sc, s2]
        * model.p_stub_md[f, p6, z, k, sc]
        for k in model.s_stub_k_by_p6z[p6, z]   # only k where (p6,z,k) exists
        for sc in model.s_stub_cat
        for s2 in model.s_biomass_uses
        if model.p_stub_unavailable_frac_p6zk[p6, z, k] != 0     # avoid building zero terms
    )

