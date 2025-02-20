"""

author: young


"""

# python modules
import pyomo.environ as pe

# AFO modules
from . import PropertyInputs as pinp
from . import Saltbush as slp
from . import StructuralInputs as sinp


def saltbush_precalcs(params, r_vals, nv):
    '''
    Call saltbush precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param r_vals: dictionary which stores all report values.
    :param nv: dictionary which stores information about the feed pools.

    '''
    slp.f_saltbush_precalcs(params, r_vals, nv)


def f1_saltbushpyomo_local(params,model, MP_lp_vars):
    ''' Builds pyomo variables, parameters and constraints'''

    ############
    # variable #
    ############

    model.v_slp_ha = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_lmus, bounds=(0,None),
                                          doc='tonnes of crop consumed by livestock in a p6 that was sown in a p5 (p5 axis tracks when the crop being grazed was sown)')

    model.v_tonnes_sb_consumed = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods,
                                        model.s_feed_pools, model.s_lmus,bounds=(0,None),
                                        doc='tonnes of saltbush consumed by livestock in a p6')

    model.v_tonnes_sb_transfer = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods, model.s_lmus,
                                          bounds=(0,None), doc='tonnes of saltbush DM transferred to next feed period')

    #########
    # param #
    #########
    model.p_phase_slp_area = pe.Param(model.s_phases, initialize=params['phase_slp_area_r'], default=0, mutable=False,
                                     doc='area of salt land pasture provided by 1ha of each rotation phase')

    model.p_slp_cost = pe.Param(model.s_season_periods, model.s_season_types,
                                initialize=params['slp_estab_cost_p7z'], default=0.0, mutable=True,
                                doc='cost of establishing 1ha of salt land pasture')

    model.p_slp_wc = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types,
                              initialize=params['slp_estab_wc_c0p7z'], default=0.0, mutable=True,
                              doc='wc of establishing 1ha of salt land pasture')

    model.p_max_growth_per_ha = pe.Param(model.s_season_types, model.s_feed_periods, model.s_lmus,
                                     initialize=params['max_growth_per_ha_zp6l'], default=0, mutable=False,
                                     doc='Maximum saltbush growth per period (maximum growth occurs if grazed)')

    model.p_sb_transfer_provide = pe.Param(model.s_season_types, model.s_feed_periods,
                                     initialize=params['transfer_prov_zp6'], default=0, mutable=False,
                                     doc='amount of sb foo provided by the transfer of 1t from the previous period')


    model.p_sb_md = pe.Param(model.s_season_types, model.s_feed_periods, model.s_feed_pools,
                                     initialize=params['sb_me_zp6f'], default=0, mutable=False,
                                     doc='Energy provided from 1t of saltbush grazing (just the saltbush not the understory)')

    model.p_sb_vol = pe.Param(model.s_season_types, model.s_feed_periods, model.s_feed_pools,
                                     initialize=params['sb_vol_zp6f'], default=0, mutable=False,
                                     doc='Volume required to consume 1t of saltbush (just the saltbush not the understory). Note this accounts for adverse effects of salt on animal intake')

    model.p_sb_selectivity_zp6 = pe.Param(model.s_season_types, model.s_feed_periods,
                                     initialize=params['sb_selectivity_zp6'], default=0, mutable=False,
                                     doc='The ratio of the volume of saltbush consumed to the total volume of feed consumed by stock grazing slp (saltbush and understory)')

    model.co2e_sb_zp6 = pe.Param(model.s_season_types, model.s_feed_periods,
                                     initialize=params['co2e_sb_zp6'], default=0, mutable=False,
                                     doc='Emissions from consuming 1t of saltbush (just the saltbush not the understory).')

    ###################################
    #call local constraints           #
    ###################################
    f_con_slp_area(model)
    f_con_saltbush_within(model)
    f_con_saltbush_between(model, MP_lp_vars)



###################
#local constraint #
###################
def f_con_slp_area(model):
    '''
    Constrains the SLP area on each LMU based on the rotation selected.

    This constraint essentially calculates the hectares of salt land pasture based on the rotation phases selected. The
    p7 axis is not required on v_slp_area because SLP is a continuous phase and therefore exists in all p7.
    Removing the p7 axis makes the calculations in saltbush pyomo simpler.
    '''
    def slp_area(model,q,s,z,p7,l):
        if pe.value(model.p_wyear_inc_qs[q, s]) and pinp.general['pas_inc_t'][3] and pe.value(model.p_mask_season_p7z[p7,z]):
            return sum(-model.v_phase_area[q,s,p7,z,r,l] * model.p_phase_slp_area[r]
                       for r in model.s_phases if pe.value(model.p_phase_slp_area[r]) != 0)   \
                 + model.v_slp_ha[q,s,z,l] ==0
        else:
            return pe.Constraint.Skip
    model.con_slp_area = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_season_periods, model.s_lmus, rule=slp_area, doc='Pasture area row for growth constraint of each type on each soil for each feed period (ha)')

def f_con_saltbush_within(model):
    '''
    Constrain the saltbush feed available on each soil type in each feed period within a given season.
    Saltbush can be eaten or transferred to the next period.
    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def saltbush_foo(model,q,s,z9,p6,l):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        if pe.value(model.p_wyear_inc_qs[q,s]) and pe.value(model.p_mask_childz_within_fp[p6,z9]) and pinp.general['pas_inc_t'][3]:
            return - model.v_slp_ha[q,s,z9,l] * model.p_max_growth_per_ha[z9,p6,l] \
                    - sum(model.v_tonnes_sb_transfer[q,s,z8,p6_prev,l] * model.p_sb_transfer_provide[z8,p6_prev]
                          * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types)  \
                    + sum(model.v_tonnes_sb_consumed[q,s,z9,p6,f,l] * 1000 for f in model.s_feed_pools)     \
                    + model.v_tonnes_sb_transfer[q,s,z9,p6,l] * 1000 <=0
        else:
            return pe.Constraint.Skip
    model.con_saltbush_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods, model.s_lmus,
                                              rule=saltbush_foo, doc='Within seasons - saltbush feed available in each feed period')

def f_con_saltbush_between(model, MP_lp_vars):
    '''
    Constrain the saltbush feed available on each soil type in each feed period between a given season.
    Saltbush can be eaten or transferred to the next period.
    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    def saltbush_foo(model,q,s9,z9,p6,l):
        p6_prev = l_fp[l_fp.index(p6) - 1] #need the activity level from last feed period
        l_q = list(model.s_sequence_year_between_con)
        ###adjust q_prev for multi-period model
        if sinp.structuralsa['model_is_MP']:
            ####yr0 is SE so q_prev is q
            if q == l_q[0]:
                q_prev = q
                v_tonnes_sb_transfer_hist = MP_lp_vars[str('v_tonnes_sb_transfer')]  # q[0] is provided by the MP set up run.
            ####the final year is provided by both the previous year and itself (the final year is in equilibrium). Therefore the final year needs two constraints. This is achieved by making the q set 1 year longer than the modeled period (len_MP + 1). Then adjusting q and q_prev for the final q so that the final year is also in equilibrium.
            elif q == l_q[-1]:
                q = l_q[l_q.index(q) - 1]
                q_prev = q
                v_tonnes_sb_transfer_hist = model.v_tonnes_sb_transfer
            else:
                q_prev = l_q[l_q.index(q) - 1]
                v_tonnes_sb_transfer_hist = model.v_tonnes_sb_transfer
        else:
            q_prev = l_q[l_q.index(q) - 1]
            v_tonnes_sb_transfer_hist = model.v_tonnes_sb_transfer

        if pe.value(model.p_wyear_inc_qs[q,s9]) and pe.value(model.p_mask_childz_between_fp[p6,z9]) and pinp.general['pas_inc_t'][3]:
            return - model.v_slp_ha[q,s9,z9,l] * model.p_max_growth_per_ha[z9,p6,l]  \
                   + sum(model.v_tonnes_sb_consumed[q,s9,z9,p6,f,l] * 1000 for f in model.s_feed_pools)     \
                   + model.v_tonnes_sb_transfer[q,s9,z9,p6,l] * 1000 <= sum(v_tonnes_sb_transfer_hist[q_prev,s8,z8,p6_prev,l] * model.p_sb_transfer_provide[z8,p6_prev]
                         * model.p_parentz_provbetween_fp[p6_prev,z8,z9]
                         * (model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9] + model.p_endstart_prov_qsz[q_prev,s8,z8])
                         for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)
        else:
            return pe.Constraint.Skip
    model.con_saltbush_between = pe.Constraint(model.s_sequence_year_between_con, model.s_sequence, model.s_season_types, model.s_feed_periods, model.s_lmus,
                                              rule=saltbush_foo, doc='between seasons - saltbush feed available in each feed period')


def parametrize_con_saltbush_between(model, z8, MP_lp_vars):
    '''
    Constrain the saltbush feed available on each soil type in each feed period between a given season.
    Saltbush can be eaten or transferred to the next period.
    '''
    ##convert feed period set to a list so it can be indexed
    l_fp = list(model.s_feed_periods)
    v_tonnes_sb_transfer_hist = MP_lp_vars[str('v_tonnes_sb_transfer')]
    l_fp = list(model.s_feed_periods)
    p6 = l_fp[0]
    p6_prev = l_fp[-1]
    # q_prev = q
    for q in model.s_sequence_year_between_con:
        for s9 in model.s_sequence:
            for z9 in model.s_season_types:
                # for p6 in model.s_feed_periods:
                    # p6_prev = l_fp[l_fp.index(p6) - 1]
                q_prev = q

                if pe.value(model.p_wyear_inc_qs[q,s9]) and pe.value(model.p_mask_childz_between_fp[p6,z9]) and pinp.general['pas_inc_t'][3]:
                    weighted_rhs = sum(v_tonnes_sb_transfer_hist[q_prev,s8,z8_,p6_prev,l] * model.p_sb_transfer_provide[z8_,p6_prev]
                        * model.p_parentz_provbetween_fp[p6_prev,z8_,z9]
                        * (model.p_sequence_prov_qs8zs9[q_prev,s8,z8_,s9] + model.p_endstart_prov_qsz[q_prev,s8,z8_])
                        for z8_ in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)
                    new_rhs = sum(v_tonnes_sb_transfer_hist[q_prev,s8,z8_,p6_prev,l] * model.p_sb_transfer_provide[z8_,p6_prev]
                        * model.p_parentz_provbetween_fp[p6_prev,z8_,z9]
                        * (model.p_sequence_prov_qs8zs9[q_prev,s8,z8_,s9] + model.p_endstart_prov_qsz[q_prev,s8,z8_])
                        for z8_ in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)
                    print(f"Changing rhs from {weighted_rhs} to {new_rhs}")
                    print(model.con_saltbush_between[q,s9,z9,p6].body)
                    model.con_saltbush_between[q,s9,z9,p6].set_value(model.con_saltbush_between[q,s9,z9,p6].body <= new_rhs)
###################################
#functions for core pyomo         #
###################################

def f_saltbush_cost(model,q,s,z,p7):
    '''
    Calculate the total cost required per hectare of salt land pasture.

    Used in global constraint (con_profit). See CorePyomo
    '''
    return sum(model.v_slp_ha[q,s,z,l] * model.p_slp_cost[p7,z] for l in model.s_lmus)

def f_saltbush_wc(model,q,s,z,c0,p7):
    '''
    Calculate the total working capital required per hectare of salt land pasture.

    Used in global constraint (con_wc). See CorePyomo
    '''
    return sum(model.v_slp_ha[q,s,z,l] * model.p_slp_wc[c0,p7,z] for l in model.s_lmus)

def f_saltbush_me(model,q,s,z,p6,f):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of saltbush.

    Used in global constraint (con_me). See CorePyomo
    '''
    if pinp.general['pas_inc_t'][3]:
        return sum(model.v_tonnes_sb_consumed[q,s,z,p6,f,l] * model.p_sb_md[z,p6,f] for l in model.s_lmus)
    else:
        return 0

def f_saltbush_vol(model,q,s,z,p6,f):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of saltbush.

    Used in global constraint (con_vol). See CorePyomo
    '''
    if pinp.general['pas_inc_t'][3]:
        return sum(model.v_tonnes_sb_consumed[q,s,z,p6,f,l] * model.p_sb_vol[z,p6,f] for l in model.s_lmus)
    else:
        return 0

def f_saltbush_emissions(model,q,s,z,p6):
    '''
    Calculate the total emissions produced from consumption of saltbush.

    Used in global constraint (con_emissions). See BoundPyomo
    '''
    return sum(model.v_tonnes_sb_consumed[q,s,z,p6,f,l] * model.co2e_sb_zp6[z,p6,f]
               for f in model.s_feed_pools for l in model.s_lmus)

def f_saltbush_selection(model,q,s,z,p6,f,l):
    '''
    Calculate the amount of understory required to consume 1t of saltbush in each feed period (this is based on the
    animals diet selection which changes during the year)

    Used in global constraint (con_link_understory_saltbush_consumption). See CorePyomo
    '''
    return model.v_tonnes_sb_consumed[q,s,z,p6,f,l] * model.p_sb_vol[z,p6,f] * (1-model.p_sb_selectivity_zp6[z,p6])









