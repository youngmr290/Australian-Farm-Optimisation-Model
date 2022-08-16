"""
author: young

"""

#python modules
import pyomo.environ as pe
import numpy as np

#AFO modules
import RotationPhases as rps
import PropertyInputs as pinp

def rotation_precalcs(params, report):
    '''
    Call rotation precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    rps.f_rot_lmu_params(params)
    rps.f_landuses_phases(params,report)
    rps.f_season_params(params)
    rps.f_phase_link_params(params)
    rps.f_rot_hist_params(params)

def f1_rotationpyomo(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    #############
    #variables  #
    #############
    ##Amount of each phase on each soil, Positive Variable.
    model.v_phase_area = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, model.s_phases,model.s_lmus, bounds=(0,None),doc='cumulative total area (ha) of phase, selected up to and including the current m period')

    ##Amount of each phase added in each rotation period on each soil.
    model.v_phase_change_increase = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, model.s_phases,model.s_lmus, bounds=(0,None),doc='Increased area (ha) of phase, selected in the current m period')

    ##Amount of each phase reduced in each rotation period on each soil.
    model.v_phase_change_reduce = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, model.s_phases,model.s_lmus, bounds=(0,None),doc='Increased area (ha) of phase, selected in the current m period')

    ####################
    #define parameters #
    ####################
    model.p_area = pe.Param(model.s_lmus, initialize=params['lmu_area'], doc='available area on farm for each soil')
    model.p_landuse_area = pe.Param(model.s_phases, model.s_landuses, initialize=params['phases_rk'], doc='landuse in each phase')
    model.p_hist_prov = pe.Param(params['hist_prov'].keys(), initialize=params['hist_prov'], default=0, doc='history provided by  each rotation') #use keys instead of sets to reduce size of param
    model.p_hist_req = pe.Param(params['hist_req'].keys(), initialize=params['hist_req'], default=0, doc='history required by  each rotation') #use keys instead of sets to reduce size of param
    # model.p_mask_phases = pe.Param(model.s_phases, model.s_season_periods, initialize=params['p_mask_phases'], doc='mask phases that transfer in each phase period')
    # model.p_dryz_link = pe.Param(model.s_phases, model.s_season_types, model.s_season_types, initialize=params['p_dryz_link'], doc='dry link between seasons (only occurs between m[-1])')
    # model.p_dryz_link2 = pe.Param(model.s_phases, model.s_season_types, model.s_season_types, initialize=params['p_dryz_link2'], doc='dry link between seasons (only occurs between m[-1])')
    model.p_parentz_provwithin_phase = pe.Param(model.s_season_periods, model.s_season_types, model.s_season_types,
                                             initialize=params['p_parentz_provwithin_phase'], default=0.0, mutable=False,
                                             doc='Transfer of z8 dv in the previous phase period to z9 constraint in the current phase period within years')
    model.p_parentz_provbetween_phase = pe.Param(model.s_season_periods, model.s_season_types, model.s_season_types,
                                              initialize=params['p_parentz_provbetween_phase'], default=0.0, mutable=False,
                                              doc='Transfer of z8 dv in the previous phase period to z9 constraint in the current phase period between years')
    model.p_mask_childz_within_phase = pe.Param(model.s_season_periods, model.s_season_types, initialize=params['p_mask_childz_within_phase'],
                                           default=0.0, mutable=False, doc='mask child require within season in each phase period')
    model.p_mask_childz_between_phase = pe.Param(model.s_season_periods, model.s_season_types, initialize=params['p_mask_childz_between_phase'],
                                           default=0.0, mutable=False, doc='mask child require between season in each phase period')
    model.p_phase_area_transfers = pe.Param(model.s_season_periods, model.s_season_types, model.s_phases, initialize=params['p_phase_area_transfers_p7zr'],
                                           default=0.0, mutable=False, doc='mask phase transfer to force a change at season break')
    model.p_phase_can_increase = pe.Param(model.s_season_periods, model.s_phases, initialize=params['p_phase_can_increase_p7r'],
                                           default=0.0, mutable=False, doc='mask which phases can change_increase in each p7')
    model.p_phase_can_reduce = pe.Param(model.s_season_periods, model.s_phases, initialize=params['p_phase_can_reduce_p7r'],
                                           default=0.0, mutable=False, doc='mask which phases can change_reduce in each p7')

    ###################
    #call constraints #
    ###################
    f_con_rotation_between(params, model)
    f_phase_link_within(model)
    f_phase_link_between(model)
    f_con_area(model)
    # f_con_dry_link(model)



#######################################################################################################################################################
#######################################################################################################################################################
#local constraints
#######################################################################################################################################################
#######################################################################################################################################################
######################
#rotation constraints#
######################
##rotation constraints are usually the same each loop. but if the lmu mask changes they need to be built again
##thus they are just built each loop. Maybe this could be changed if running lots of rotations.

def f_con_rotation_between(params, model):
    '''
    Creates the constraint between history provided at the end of the year and the history required at the beginning
    of the year for each rotation phase on each LMU.

    The rotation constraints are to ensure that the rotation phases that are selected in the optimisation can
    be arranged into an actual rotation. All phases except the continuous rotations require at least one other
    phase. Eg. a canola(z)-wheat(w) rotation would be generated from 2 phases  Y Y N E z and Y Y E N w. To represent
    this requires ensuring that each rotaion phase selected has a preceding phase that has landuses in the same
    order as the target rotation phase (except for year 0). This is called the history required and history required.

    '''

    def rot_phase_link(model,q,s9,p7,l,h,z9):
        l_p7 = list(model.s_season_periods)
        p7_end = l_p7[-1]
        l_q = list(model.s_sequence_year)
        q_prev = l_q[l_q.index(q) - 1]

        if p7 == p7_end and pe.value(model.p_wyear_inc_qs[q,s9]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return sum(model.v_phase_area[q_prev,s8,p7_end,z8,r,l]*model.p_hist_prov[r,h] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                       + model.v_phase_area[q_prev,s8,p7_end,z8,r,l] * model.p_hist_prov[r,h] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                       for r in model.s_phases for s8 in model.s_sequence for z8 in model.s_season_types
                       if ((r,)+(h,)) in params['hist_prov'].keys()) \
                 + sum(model.v_phase_area[q,s9,p7,z9,r,l]*model.p_hist_req[r,h] for r in model.s_phases
                       if ((r,)+(h,)) in params['hist_req'].keys())<=0
        else:
            return pe.Constraint.Skip

    model.con_rotation_hist_con = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_rotconstraints, model.s_season_types, rule=rot_phase_link, doc='rotation phases constraint')


def f_phase_link_within(model):
    '''
    The phase link constraint is used to force the selection of v_phase_change when a phase change is required.
    This is necessary because v_phase_change incurs costs and includes the requirement for seeding.

    The phase area selected in each phase_period must be at least the area selected in the previous period and the
    increment area incurs the costs to date (so that selection at later nodes is not ‘cheaper’ than
    earlier selection).

    The transfer of the phases selected in the parent weather-year to the child weather-years is achieved in
    the same manner as the transfers of stock, pasture and cashflow with 2 differences:

        a.	the inclusion of v_phase_change_increase which allows extra area of a phase to be selected in each node.
        b.	the constraint is equal-to rather than less-than. This is necessary to cover a situation in which the cashflow
            parameter of v_phase_change_increase is earning money. In this situation the model would be unbounded with a less-than constraint.

    '''
    def phase_link_within(model,q,s,p7,l,r,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last season period
        if pe.value(model.p_wyear_inc_qs[q,s]) and pe.value(model.p_mask_childz_within_phase[p7,z9]):
            return model.v_phase_area[q,s,p7,z9,r,l]  \
                   - model.v_phase_change_increase[q,s,p7,z9,r,l] * model.p_phase_can_increase[p7,r] \
                   + model.v_phase_change_reduce[q,s,p7,z9,r,l] * model.p_phase_can_reduce[p7,r] \
                   - sum(model.v_phase_area[q,s,p7_prev,z8,r,l] * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
                         for z8 in model.s_season_types) * model.p_phase_area_transfers[p7_prev,z9,r] == 0 #p_phase_area_transfers ensures no transfer at break of season except for dry sown phases
        else:
            return pe.Constraint.Skip
    model.con_phase_link_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_phases, model.s_season_types, rule=phase_link_within, doc='rotation phases constraint')

def f_phase_link_between(model):
    '''
    This is the between year version of above. This is required because in the late breaks the phase from the previous
    year needs to carry over so that dry pasture and stubble can exist.
    '''
    def phase_link_between(model,q,s9,p7,l,r,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
        if pe.value(model.p_wyear_inc_qs[q,s9]) and pe.value(model.p_mask_childz_between_phase[p7,z9]):
            return model.v_phase_area[q,s9,p7,z9,r,l]  \
                   - model.v_phase_change_increase[q,s9,p7,z9,r,l] * model.p_phase_can_increase[p7,r] \
                   + model.v_phase_change_reduce[q,s9,p7,z9,r,l] * model.p_phase_can_reduce[p7,r] \
                   - sum(model.v_phase_area[q,s8,p7_prev,z8,r,l]
                         * model.p_parentz_provbetween_phase[p7_prev, z8, z9]
                         * (model.p_sequence_prov_qs8zs9[q_prev, s8, z8, s9] + model.p_endstart_prov_qsz[q_prev, s8, z8])
                            for s8 in model.s_sequence for z8 in model.s_season_types) * model.p_phase_area_transfers[p7_prev,z9,r] \
                   == 0 #end of the previous yr is controlled by between constraint
        else:
            return pe.Constraint.Skip
    model.con_phase_link_between = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_phases, model.s_season_types, rule=phase_link_between, doc='rotation phases constraint')


# def f_con_dry_link(model):
#     '''
#     Link between dry seeding in different breaks.
#
#     If dry seeding occurs in a given season it must also occur in all other seasons that have not yet broken.
#     For example, if dry sowing occurs before the earliest break then at least the same amount must occur in all
#     other seasons. However, if dry seeding occurs in a season with a medium break it doesn't need to happen in a season
#     with an early break but it must happen in a season with a later break.
#
#     This constraint only occurs for m[-1] because that is the period when dry sowing phases are selected.
#     This constraint is required because in m[-1] all seasons are identified so nothing forces dry seeding to be
#     the same across seasons.
#
#     '''
#     #this one forces the current season to have at least as much dry seeding as the previous season
#     def dry_phase_link1(model,q,s,p7,l,r,z9):
#         l_p7 = list(model.s_season_periods)
#         ##only build the constraint for m[-1]
#         if p7 == l_p7[-1] and any(model.p_dryz_link[r,z8,z9] for z8 in model.s_season_types) and pe.value(model.p_mask_childz_phase[p7,z9]):
#             return - model.v_phase_change_increase[q,s,p7,z9,r,l] \
#                    + sum(model.v_phase_change_increase[q,s,p7,z8,r,l] * model.p_dryz_link[r,z8,z9]
#                          for z8 in model.s_season_types) <= 0
#         else:
#             return pe.Constraint.Skip
    # model.con_dry_link1 = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_phases, model.s_season_types, rule=dry_phase_link1, doc='link dry seeding between season types')

    # #this one forces each season with the same break to have the same amount of dry seeding (by forcing the end to equal the start)
    # def dry_phase_link2(model,q,s,p7,l,r,z9):
    #     l_p7 = list(model.s_season_periods)
    #     ##only build the constraint for m[-1]
    #     if p7 == l_p7[-1] and any(model.p_dryz_link2[r,z8,z9] for z8 in model.s_season_types) and pe.value(model.p_mask_childz_phase[p7,z9]):
    #         return - model.v_phase_change_increase[q,s,p7,z9,r,l] \
    #                + sum(model.v_phase_change_increase[q,s,p7,z8,r,l] * model.p_dryz_link2[r,z8,z9]
    #                      for z8 in model.s_season_types) <= 0
    #     else:
    #         return pe.Constraint.Skip
    # model.con_dry_link2 = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_phases, model.s_season_types, rule=dry_phase_link2, doc='link dry seeding between season types')




########
# Area #
########
def f_con_area(model):
    '''
    Creates the constraint between farm area and rotation area on each LMU.

    Constrains the maximum area of all rotations on each lmu by the area of each LMU on the modelled property.
    The area of rotation on a given soil can't be more than the amount of that soil available on the farm.
    '''

    def area_rule(model, q,  s, p7, l, z):
        if pe.value(model.p_mask_season_p7z[p7,z]) and pe.value(model.p_wyear_inc_qs[q, s]):
            return sum(model.v_phase_area[q,s,p7,z,r,l] for r in model.s_phases) <= model.p_area[l]
        else:
            return pe.Constraint.Skip
    model.con_area = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_lmus, model.s_season_types, rule=area_rule, doc='rotation area constraint')
    


