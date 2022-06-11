"""
author: Young
"""
##python modules
from pyomo import environ as pe

##AFO modules
import Pasture as pas
import FeedsupplyFunctions as fsfun


def paspyomo_precalcs(params, r_vals, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param r_vals: dictionary which stores all report values.
    :param nv: dictionary which stores nutrient pool info from StockGenerator.py.

    '''

    pas.f_pasture(params, r_vals, nv)

def f1_paspyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''
    ###################
    # variable         #
    ###################
    model.v_greenpas_ha = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_pools, model.s_grazing_int,
                                 model.s_foo_levels, model.s_feed_periods,
                                 model.s_lmus, model.s_season_types, model.s_pastures,bounds=(0,None),
                                 doc='hectares grazed each period for each grazing intensity on each soil in each period')
    model.v_drypas_consumed = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_pools, model.s_dry_groups,
                                     model.s_feed_periods, model.s_season_types, model.s_lmus,
                                     model.s_pastures, bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed consumed by each sheep pool in each feed period')
    model.v_drypas_transfer = pe.Var(model.s_sequence_year, model.s_sequence, model.s_dry_groups, model.s_feed_periods,
                                     model.s_season_types, model.s_lmus, model.s_pastures, bounds=(0,None),
                                     doc='tonnes of low and high quality dry feed at end of the period transferred to the following periods in each feed period')
    model.v_nap_consumed = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_pools, model.s_dry_groups,
                                  model.s_feed_periods, model.s_season_types, model.s_pastures, bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks consumed by each sheep pool in each feed period')
    model.v_nap_transfer = pe.Var(model.s_sequence_year, model.s_sequence, model.s_dry_groups, model.s_feed_periods,
                                  model.s_season_types, model.s_pastures,bounds=(0,None),
                                  doc='tonnes of low and high quality dry pasture on crop paddocks transferred to the following periods in each feed period')
    model.v_poc = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_pools, model.s_feed_periods, model.s_lmus,
                         model.s_season_types, bounds=(0,None),
                         doc='tonnes of poc consumed by each sheep pool in each period on each lmu')

    ####################
    #define parameters #
    ####################
    model.p_pasture_area = pe.Param(model.s_phases, model.s_pastures, initialize=params['pasture_area_rt'],
                                    default=0, doc='pasture area of each rotation')
    
    model.p_germination = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_lmus, model.s_phases,
                                   model.s_season_types, model.s_pastures, initialize=params['p_germination_p7p6lrzt'],
                                   default=0, mutable=False, doc='pasture germination for each rotation')

    model.p_foo_grn_reseeding = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_lmus, model.s_phases,
                                         model.s_season_types, model.s_pastures, initialize=params['p_foo_grn_reseeding_p7p6lrzt'],
                                         default=0, mutable=False, doc='Change in grn FOO due to destocking and restocking of resown pastures')
    
    model.p_foo_dry_reseeding = pe.Param(model.s_season_periods, model.s_dry_groups, model.s_feed_periods, model.s_lmus,
                                         model.s_phases, model.s_season_types, model.s_pastures, initialize=params['p_foo_dry_reseeding_p7dp6lrzt'],
                                         default=0, mutable=False, doc='Change in dry FOO due to destocking and seeding of pastures')
    
    model.p_foo_end_grnha = pe.Param(model.s_grazing_int, model.s_foo_levels, model.s_feed_periods, model.s_lmus,
                                     model.s_season_types, model.s_pastures, initialize=params['p_foo_end_grnha_gop6lzt'],
                                     default=0, mutable=False, doc='Green Foo at the end of the period')
    
    model.p_foo_start_grnha = pe.Param(model.s_foo_levels, model.s_feed_periods, model.s_lmus, model.s_season_types,
                                       model.s_pastures, initialize=params['p_foo_start_grnha_op6lzt'], default=0,
                                       mutable=False, doc='Green Foo at the start of the period')
    
    model.p_senesce_grnha = pe.Param(model.s_dry_groups, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods,
                                     model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_senesce_grnha_dgop6lzt'],
                                     default=0, mutable=False, doc='Green pasture senescence into high and low quality dry pasture pools')
    
    model.p_me_cons_grnha = pe.Param(model.s_feed_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods,
                                     model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_me_cons_grnha_fgop6lzt'],
                                     default=0, mutable=False, doc='Total ME from grazing a hectare')
    
    model.p_volume_grnha = pe.Param(model.s_feed_pools, model.s_grazing_int, model.s_foo_levels, model.s_feed_periods,
                                    model.s_lmus, model.s_season_types, model.s_pastures, initialize=params['p_volume_grnha_fgop6lzt'],
                                    default=0, mutable=False, doc='Total Vol from grazing a hectare')
    
    model.p_dry_mecons_t = pe.Param(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_season_types,
                                    model.s_pastures, initialize=params['p_dry_mecons_t_fdp6zt'], default=0,
                                    mutable=False, doc='Total ME from grazing a tonne of dry feed')
    
    model.p_dry_volume_t = pe.Param(model.s_feed_pools, model.s_dry_groups, model.s_feed_periods, model.s_season_types,
                                    model.s_pastures, initialize=params['p_dry_volume_t_fdp6zt'], default=0,
                                    mutable=False, doc='Total Vol from grazing a tonne of dry feed')
    
    model.p_dry_transfer_prov_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures,
                                           initialize=params['p_dry_transfer_prov_t_p6zt'], default=0, mutable=False,
                                           doc='quantity of dry feed transferred out of the previous period to the current (allows for decay)')

    model.p_dry_transfer_req_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures,
                                          initialize=params['p_dry_transfer_req_t_p6zt'], default=0, mutable=False,
                                          doc='quantity of dry feed required to transfer a tonne of dry feed to the following period (this parameter is always 1000 unless dry feed does not exist)')

    model.p_dry_removal_t = pe.Param(model.s_feed_periods, model.s_season_types, model.s_pastures, initialize=params['p_dry_removal_t_p6zt'],
                                     default=0, doc='quantity of dry feed removed for sheep to consume 1t, accounts for trampling')
    
    model.p_nap = pe.Param(model.s_season_periods, model.s_dry_groups, model.s_feed_periods, model.s_lmus, model.s_phases,
                           model.s_season_types, model.s_pastures, initialize=params['p_nap_p7dp6lrzt'], default=0, mutable=False,
                           doc='pasture on non arable areas in crop paddocks')
    
    model.p_nap_prop = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_harvest_period_prop'],
                                default=0, mutable=False, doc='proportion of the way through each period nap becomes available')
    
    model.p_erosion = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types,
                               model.s_pastures, initialize=params['p_erosion_p7p6lrzt'], default=0, doc='erosion limit in each period')
    
    model.p_phase_area = pe.Param(model.s_season_periods, model.s_feed_periods, model.s_lmus, model.s_phases, model.s_season_types, model.s_pastures,
                                  initialize=params['p_phase_area_p7p6lrzt'], default=0, mutable=False, doc='pasture area in each rotation for each feed period')
    
    model.p_poc_con = pe.Param(model.s_feed_periods ,model.s_lmus, model.s_season_types, initialize=params['p_poc_con_p6lz'],
                               default=0, doc='available consumption of pasture on 1ha of a crop paddock each day for each lmu in each feed period')

    model.p_poc_md = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, initialize=params['p_poc_md_fp6z'],
                              default=0, doc='md of pasture on crop paddocks for each feed period')
    
    model.p_poc_vol = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, initialize=params['p_poc_vol_fp6z'],
                               default=0, mutable=False, doc='vol (ri intake) of pasture on crop paddocks for each feed period')
    
    model.p_parentz_provwithin_fp = pe.Param(model.s_feed_periods, model.s_season_types, model.s_season_types,
                                                  initialize=params['p_parentz_provwithin_fp'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous fp to z9 constraint in the current fp within years')
    model.p_parentz_provbetween_fp = pe.Param(model.s_feed_periods, model.s_season_types, model.s_season_types,
                                                  initialize=params['p_parentz_provbetween_fp'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous fp to z9 constraint in the current fp between years')
    model.p_mask_childz_within_fp = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_mask_childz_within_fp'],
                                           default=0.0, mutable=False, doc='mask child season require in each fp within year')
    model.p_mask_childz_between_fp = pe.Param(model.s_feed_periods, model.s_season_types, initialize=params['p_mask_childz_between_fp'],
                                            default=0.0, mutable=False, doc='mask child season require in each fp between years')

    
    ########################
    #call local constraint #
    ########################
    f_con_greenpas_within(model)
    f_con_greenpas_between(model)
    f_con_drypas_within(model)
    f_con_drypas_between(model)
    f_con_nappas(model)
    f_con_pasarea(model)
    f_con_erosion(model)


###################
#local constraint #
###################
def f_con_greenpas_within(model):
    '''
    Constrain the green pasture available on each soil type in each feed period within a given season.
    Determined by rotation selection (germination and resowing), growth and consumption on each
    hectare of pasture landuse.
    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def greenpas(model,q,s,p6,l,z9,t):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if pe.value(model.p_mask_childz_within_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s]) and any(model.p_foo_start_grnha[o,p6,l,z9,t] for o in model.s_foo_levels):
            return sum(model.v_phase_area[q,s,p7,z9,r,l] * (-model.p_germination[p7,p6,l,r,z9,t] - model.p_foo_grn_reseeding[p7,p6,l,r,z9,t])
                       for r in model.s_phases for p7 in model.s_season_periods
                       if pe.value(model.p_germination[p7,p6,l,r,z9,t])!=0 or model.p_foo_grn_reseeding[p7,p6,l,r,z9,t]!=0)         \
                   + sum(model.v_greenpas_ha[q,s,f,g,o,p6,l,z9,t] * model.p_foo_start_grnha[o,p6,l,z9,t]   \
                         - sum(model.v_greenpas_ha[q,s,f,g,o,p6_prev,l,z8,t] * model.p_foo_end_grnha[g,o,p6_prev,l,z8,t]
                               * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types)
                         for f in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
        else:
            return pe.Constraint.Skip
    #todo the greenpas (FOO) and pasarea (ha) could be replaced by a grnha constraint that passes area and foo together. Needs a FooB (base level) and reseeding foo removal and addition associated with the reseeding rotation phases
    model.con_greenpas_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = greenpas, doc='Within seasons - green pasture of each type available on each soil type in each feed period')

def f_con_greenpas_between(model):
    '''
    Constrain the green pasture available on each soil type in each feed period between a given season.
    Determined by rotation selection (germination and resowing), growth and consumption on each
    hectare of pasture landuse.
    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def greenpas(model,q,s9,p6,l,z9,t):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
        if pe.value(model.p_mask_childz_between_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s9]) and any(model.p_foo_start_grnha[o,p6,l,z9,t] for o in model.s_foo_levels):
            return sum(model.v_phase_area[q,s9,p7,z9,r,l] * (-model.p_germination[p7,p6,l,r,z9,t] - model.p_foo_grn_reseeding[p7,p6,l,r,z9,t])
                       for r in model.s_phases for p7 in model.s_season_periods
                       if pe.value(model.p_germination[p7,p6,l,r,z9,t])!=0 or model.p_foo_grn_reseeding[p7,p6,l,r,z9,t]!=0)         \
                   + sum(model.v_greenpas_ha[q,s9,f,g,o,p6,l,z9,t] * model.p_foo_start_grnha[o,p6,l,z9,t]   \
                         - sum(model.v_greenpas_ha[q_prev,s8,f,g,o,p6_prev,l,z8,t] * model.p_foo_end_grnha[g,o,p6_prev,l,z8,t]
                               * model.p_parentz_provbetween_fp[p6_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                               + model.v_greenpas_ha[q_prev,s8,f,g,o,p6_prev,l,z8,t] * model.p_foo_end_grnha[g,o,p6_prev,l,z8,t]
                               * model.p_parentz_provbetween_fp[p6_prev, z8, z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                               for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)
                         for f in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
        else:
            return pe.Constraint.Skip
    #todo the greenpas (FOO) and pasarea (ha) could be replaced by a grnha constraint that passes area and foo together. Needs a FooB (base level) and reseeding foo removal and addition associated with the reseeding rotation phases
    model.con_greenpas_between = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = greenpas, doc='Between seasons - green pasture of each type available on each soil type in each feed period')

def f_con_drypas_within(model):
    '''
    Constrains the high and low quality dry pasture available in each period. Determined by senesced green pasture
    in the current period, dry pasture transferred from previous period and livestock consumption. Pasture decay and
    trampling are factored into the consumption and transfer activities (e.g. the transfer activity removes 1000kg
    from the previous period and provides 1000 - decay - trampling kg into the current period).

    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def drypas_within(model,q,s,d,p6,z9,l,t):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if pe.value(model.p_mask_childz_within_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s]) and (model.p_dry_removal_t[p6,z9,t] != 0 or model.p_dry_transfer_req_t[p6,z9,t] != 0):
            return sum(model.v_phase_area[q,s,p7,z9,r,l] * model.p_foo_dry_reseeding[p7,d,p6,l,r,z9,t]
                       for r in model.s_phases for p7 in model.s_season_periods)   \
                 + sum(-sum(model.v_greenpas_ha[q,s,f,g,o,p6_prev,l,z8,t] * model.p_senesce_grnha[d,g,o,p6_prev,l,z8,t]
                            * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types
                            for g in model.s_grazing_int for o in model.s_foo_levels)
                       + model.v_drypas_consumed[q,s,f,d,p6,z9,l,t] * model.p_dry_removal_t[p6,z9,t] for f in model.s_feed_pools) \
                 - sum(model.v_drypas_transfer[q,s,d,p6_prev,z8,l,t] * model.p_dry_transfer_prov_t[p6_prev,z8,t]
                       * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types) \
                 + model.v_drypas_transfer[q,s,d,p6,z9,l,t] * model.p_dry_transfer_req_t[p6,z9,t] <=0
        else:
            return pe.Constraint.Skip
    model.con_drypas_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dry_groups, model.s_feed_periods,
                                            model.s_season_types, model.s_lmus, model.s_pastures, rule = drypas_within, doc='Within seasons: High and low quality dry pasture of each type available in each period')

def f_con_drypas_between(model):
    '''
    Constrains the high and low quality dry pasture available in each period. Determined by senesced green pasture
    in the current period, dry pasture transferred from previous period and livestock consumption. Pasture decay and
    trampling are factored into the consumption and transfer activities (e.g. the transfer activity removes 1000kg
    from the previous period and provides 1000 - decay - trampling kg into the current period).

    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def drypas_between(model,q,s9,d,p6,z9,l,t):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
        if pe.value(model.p_mask_childz_between_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s9]) and (model.p_dry_removal_t[p6,z9,t] != 0 or model.p_dry_transfer_req_t[p6,z9,t] != 0):
            return sum(model.v_phase_area[q,s9,p7,z9,r,l] * model.p_foo_dry_reseeding[p7,d,p6,l,r,z9,t]
                       for r in model.s_phases for p7 in model.s_season_periods)   \
                 + sum(-sum(model.v_greenpas_ha[q_prev,s8,f,g,o,p6_prev,l,z8,t] * model.p_senesce_grnha[d,g,o,p6_prev,l,z8,t]
                            * model.p_parentz_provbetween_fp[p6_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                            + model.v_greenpas_ha[q_prev,s8,f,g,o,p6_prev,l,z8,t] * model.p_senesce_grnha[d,g,o,p6_prev,l,z8,t]
                            * model.p_parentz_provbetween_fp[p6_prev, z8, z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                            for z8 in model.s_season_types for s8 in model.s_sequence for g in model.s_grazing_int
                            for o in model.s_foo_levels if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)
                       + model.v_drypas_consumed[q,s9,f,d,p6,z9,l,t] * model.p_dry_removal_t[p6,z9,t] for f in model.s_feed_pools) \
                 - sum(model.v_drypas_transfer[q_prev,s8,d,p6_prev,z8,l,t] * model.p_dry_transfer_prov_t[p6_prev,z8,t]
                       * model.p_parentz_provbetween_fp[p6_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                       + model.v_drypas_transfer[q_prev,s8,d,p6_prev,z8,l,t] * model.p_dry_transfer_prov_t[p6_prev,z8,t]
                       * model.p_parentz_provbetween_fp[p6_prev, z8, z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                       for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0) \
                 + model.v_drypas_transfer[q,s9,d,p6,z9,l,t] * model.p_dry_transfer_req_t[p6,z9,t] <=0
        else:
            return pe.Constraint.Skip
    model.con_drypas_between = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dry_groups, model.s_feed_periods,
                                             model.s_season_types, model.s_lmus, model.s_pastures, rule = drypas_between, doc='Between seasons: High and low quality dry pasture of each type available in each period')

def f_con_nappas(model):
    '''
    Constrains the high and low quality dry pasture available from non arable areas of crop paddocks in each period.
    Determined by rotation and ungrazed growth rate of pasture during the growing season. This pasture becomes available
    for livestock to graze once harvest is done.
    Pasture decay and trampling are factored into the consumption and transfer activities (e.g. the transfer activity removes 1000kg
    from the previous period and provides 1000 - decay - trampling kg into the current period).

    This has to be a separate constraint from dry_pas so that nap doesn’t provide pasture in the erosion constraint.

    Note: Dry pasture does not transfer into the new season because once green
    feed is available stock will not graze old dry feed. Therefore no 'between' constraint exists.

    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def nappas(model,q,s,d,p6,z9,t):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if (model.p_dry_removal_t[p6,z9,t] == 0 and model.p_dry_transfer_req_t[p6,z9,t] == 0) or not pe.value(model.p_wyear_inc_qs[q, s]):
            return pe.Constraint.Skip
        else:
            return sum(sum(- model.v_phase_area[q,s,p7,z9,r,l] * model.p_nap[p7,d,p6,l,r,z9,t]
                           for r in model.s_phases for l in model.s_lmus for p7 in model.s_season_periods
                           if pe.value(model.p_nap[p7,d,p6,l,r,z9,t]) != 0)
                       + model.v_nap_consumed[q,s,f,d,p6,z9,t] * model.p_dry_removal_t[p6,z9,t] for f in model.s_feed_pools) \
                   - sum(model.v_nap_transfer[q,s,d,p6_prev,z8,t] * model.p_dry_transfer_prov_t[p6_prev,z8,t]
                         * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types)   \
                   + model.v_nap_transfer[q,s,d,p6,z9,t] * model.p_dry_transfer_req_t[p6,z9,t] <=0
    model.con_nappas = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_dry_groups, model.s_feed_periods, model.s_season_types, model.s_pastures, rule = nappas, doc='High and low quality dry pasture of each type available in each period')

def f_con_pasarea(model):
    '''
    Constrains the pasture area (used in con_greenpas) on each LMU based on the rotation selected.
    This accounts for arable area and destocking for reseeding.
    '''
    def pasarea(model,q,s,p6,l,z,t):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return sum(-model.v_phase_area[q,s,p7,z,r,l] * model.p_phase_area[p7,p6,l,r,z,t]
                       for r in model.s_phases for p7 in model.s_season_periods if pe.value(model.p_phase_area[p7,p6,l,r,z,t]) != 0)   \
                 + sum(model.v_greenpas_ha[q,s,f,g,o,p6,l,z,t] for f in model.s_feed_pools for g in model.s_grazing_int for o in model.s_foo_levels) <=0
        else:
            return pe.Constraint.Skip
    model.con_pasarea = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = pasarea, doc='Pasture area row for growth constraint of each type on each soil for each feed period (ha)')

def f_con_erosion(model):
    '''
    Constraint on the erosion limit at the end of the period. This ensure that at the end of the growing season
    paddocks have some cover as a sustainability measure.
    Doesn't include nap because nap is only on crop paddocks (the nap on crop paddocks is all low quality and doesn't
    tend to get grazed so there is no need for an erosion constraint on non-arable crop area).
    '''
    def erosion(model,q,s,p6,l,z,t):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            #senescence is included here because it is passed into the dry feed pool in the following fp. Thus senesced feed is not included in green or dry pasture in the period it senesced.
            return sum(sum(model.v_greenpas_ha[q,s,f,g,o,p6,l,z,t] for f in model.s_feed_pools) * -(model.p_foo_end_grnha[g,o,p6,l,z,t] +
                       sum(model.p_senesce_grnha[d,g,o,p6,l,z,t] for d in model.s_dry_groups)) for g in model.s_grazing_int for o in model.s_foo_levels) \
                    -  sum(model.v_drypas_transfer[q,s,d,p6,z,l,t] * 1000 for d in model.s_dry_groups) \
                    + sum(model.v_phase_area[q,s,p7,z,r,l]  * model.p_erosion[p7,p6,l,r,z,t]
                          for r in model.s_phases for p7 in model.s_season_periods if pe.value(model.p_erosion[p7,p6,l,r,z,t]) != 0) <=0
        else:
            return pe.Constraint.Skip
    model.con_erosion = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_lmus, model.s_season_types, model.s_pastures, rule = erosion, doc='total pasture available of each type on each soil type in each feed period')

    
###################
#constraint global#
###################
# ##sow
# def f_passow(model,p5,k,l,z):
#     '''
#     Calculate the hectares of pasture that are resown.
#
#     Used in global constraint (con_passow). See CorePyomo.
#     '''
#     if any(model.p_pas_sow[p5,l,r,k,z] for r in model.s_phases):
#         return sum(model.p_pas_sow[p5,l,r,k,z]*model.v_phase_area[p7,z,r,l] for r in model.s_phases if pe.value(model.p_pas_sow[p5,l,r,k,z]) != 0)
#     else:
#         return 0

##ME
def f_pas_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected level of dry and green pasture consumption.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(sum(sum(model.v_greenpas_ha[q,s,f,g,o,p6,l,z,t] * model.p_me_cons_grnha[f,g,o,p6,l,z,t] for g in model.s_grazing_int for o in model.s_foo_levels) \
               + sum(model.v_drypas_consumed[q,s,f,d,p6,z,l,t] * model.p_dry_mecons_t[f,d,p6,z,t] for d in model.s_dry_groups) for t in model.s_pastures) \
               + model.v_poc[q,s,f,p6,l,z] * model.p_poc_md[f,p6,z] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast

def f_nappas_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected level of non arable pasture consumption.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_nap_consumed[q,s,f,d,p6,z,t] * model.p_dry_mecons_t[f,d,p6,z,t] for d in model.s_dry_groups for t in model.s_pastures)

##Vol
def f_pas_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of pasture.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(sum(model.v_greenpas_ha[q,s,f,g,o,p6,l,z,t] * model.p_volume_grnha[f,g,o,p6,l,z,t] for g in model.s_grazing_int for o in model.s_foo_levels for l in model.s_lmus) \
               + sum(sum(model.v_drypas_consumed[q,s,f,d,p6,z,l,t] * model.p_dry_volume_t[f,d,p6,z,t] for l in model.s_lmus) \
               +         model.v_nap_consumed[q,s,f,d,p6,z,t] * model.p_dry_volume_t[f,d,p6,z,t] for d in model.s_dry_groups) for t in model.s_pastures)\
           + sum(model.v_poc[q,s,f,p6,l,z] * model.p_poc_vol[f,p6,z] for l in model.s_lmus) #have to sum lmu here again, otherwise other axis will broadcast
