"""
Contains constraints for core equations and objective.

author: young
"""
import time
import pyomo.environ as pe
import numpy as np
# import networkx
# import pyomo.pysp.util.rapper as rapper
# import pyomo.pysp.plugins.csvsolutionwriter as csvw
# import pyomo.pysp.plugins.jsonsolutionwriter as jsonw
import os
import shutil

# AFO modules - should only be pyomo modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
import PhasePyomo as phspy
import MachPyomo as macpy
import LabourPyomo as labpy
import LabourPhasePyomo as lphspy
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import CropResiduePyomo as stubpy
import StockPyomo as stkpy
import MVF as mvf
import Sensitivity as sen
import Finance as fin
import CropGrazingPyomo as cgzpy


def coremodel_all(trial_name,model):
    '''
    Wraps all of the core model into a function so it can be run multiple times in a loop

    Returns
    -------
    None.

    '''
    ##################
    #call constraints#
    ##################
    # Labour fixed
    f_con_labour_fixed_anyone(model)
    f_con_labour_fixed_manager(model)
    # Labour crop
    f_con_labour_phase_anyone(model)
    f_con_labour_phase_perm(model)
    # labour Sheep
    f_con_labour_sheep_anyone(model)
    f_con_labour_sheep_perm(model)
    f_con_labour_sheep_manager(model)
    # stubble & nap consumption at harvest
    f_con_harv_stub_nap_cons(model)
    # # stubble
    # f_con_stubble_a(model)
    # sow landuse
    f_con_phasesow(model)
    # harvest and make hay
    f_con_harv(model)
    f_con_makehay(model)
    # feed supply
    f_con_poc_available(model)
    f_con_vol(model)
    f_con_me(model)
    #crop grazing
    # f_con_cropgraze_area(model)
    #biomass
    f_con_biomass_transfer(model)
    #grain
    f_con_product_transfer(model)
    #cashflow
    f_con_cashflow(model)
    f_con_workingcap_within(model)
    f_con_workingcap_between(model)
    f_con_dep(model)
    f_con_asset(model)
    f_con_minroe(model)

    #############
    # objective #
    #############
    '''
    maximise credit in the last period of cashflow (rather than indexing directly with ND$FLOW, i index with the last name in the cashflow periods in case cashflow periods change) 
    minus dep (variable and fixed)
    '''
    model.profit = pe.Objective(rule=f_objective,sense=pe.maximize)
    # model.profit.pprint()

    #########
    # solve #
    #########

    ##sometimes if there is a bug when solved it is good to write lp here - because the code doesn't run to the other place where lp written
    directory_path = os.path.dirname(os.path.abspath(__file__))
    model.write(os.path.join(directory_path, 'Output/test.lp'),io_options={'symbolic_solver_labels': True})  # comment this out when not debugging

    ##tells the solver you want duals and rc
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.slack = pe.Suffix(direction=pe.Suffix.IMPORT)
    ##solve - uses cplex if it exists else glpk - tee=True will print out solver information.
    if not shutil.which("cplex") == None:
        ##solve with cplex if it exists
        solver = pe.SolverFactory('cplex')
    else:
        ##solve with glpk
        solver = pe.SolverFactory('glpk')
        solver.options['tmlim'] = 100  # limit solving time to 100sec in case solver stalls.
    solver_result = solver.solve(model,tee=True)  # turn to true for solver output - may be useful for troubleshooting
    try:  # to handle infeasible (there is no profit component when infeasible)
        obj = pe.value(model.profit)
    except ValueError:
        obj = 0

    ##this prints trial name, overall profit and feasibility for each trial
    print("Displaying Solution for trial: %s\n" % trial_name,'-' * 60,'\n%s' % obj)

    ##this check if the solver is optimal - if infeasible or error the model will save a file in Output/infeasible/ directory. This will be accessed in reporting to stop you reporting infeasible trials.
    ##the model will keep running the next trials even if one is infeasible.
    if (solver_result.solver.status == pe.SolverStatus.ok) and (
            solver_result.solver.termination_condition == pe.TerminationCondition.optimal):
        print('OPTIMAL LP SOLUTION FOUND')  # Do nothing when the solution in optimal and feasible
        ###trys to delete the infeasible file because the trial is now optimal
        try:
            os.remove('Output/infeasible/%s.txt' % trial_name)
        except FileNotFoundError:
            pass
    elif (solver_result.solver.termination_condition == pe.TerminationCondition.infeasible):
        print('***INFEASIBLE LP SOLUTION***')
        ###save infeasible file
        with open('Output/infeasible/%s.txt' % trial_name,'w') as f:
            f.write("Solver Status: {0}".format(solver_result.solver.termination_condition))
    else:  # Something else is wrong - solver may have stalled.
        print('***Solver Status: error (other)***')
        ###save infeasible file
        with open('Output/infeasible/%s.txt' % trial_name,'w') as f:
            f.write("Solver Status: {0}".format(solver_result.solver.termination_condition))

    return obj

##############
#constriants #
##############
def f_con_labour_fixed_anyone(model):
    '''
    Tallies labour used for fixed activities that can be completed by anyone (casual/permanent/manager) and ensures
    that there is sufficient labour available to carry out the jobs.
    '''
    def labour_fixed_casual(model,q,s,p,w,z):
        return -model.v_fixed_labour_casual[q,s,p,w,z] - model.v_fixed_labour_permanent[q,s,p,w,z] - \
               model.v_fixed_labour_manager[q,s,p,w,z] \
               + model.p_super_labour[p,z] + model.p_tax_labour[p,z] + model.p_bas_labour[p,z] <= 0

    model.con_labour_fixed_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,
                                                  rule=labour_fixed_casual,
                                                  doc='link between labour supply and requirement by fixed jobs for casual and above')


def f_con_labour_fixed_manager(model):
    '''
    Tallies labour used for fixed activities that can only be completed by the manager and ensures that there is
    sufficient labour available to carry out the jobs.
    '''
    def labour_fixed_manager(model,q,s,p,w,z):
        return -model.v_fixed_labour_manager[q,s,p,w,z] + model.p_planning_labour[p,z] + (
                    model.p_learn_labour * model.v_learn_allocation[q,s,p,z]) <= 0

    model.con_labour_fixed_manager = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['mngr'],model.s_season_types,
                                                   rule=labour_fixed_manager,
                                                   doc='link between labour supply and requirement by fixed jobs for manager')


def f_con_labour_phase_anyone(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by anyone (casual/permanent/manager) and ensures
    that there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_anyone(model,q,s,p,w,z):
        return -model.v_phase_labour_casual[q,s,p,w,z] - model.v_phase_labour_permanent[q,s,p,w,z] - model.v_phase_labour_manager[
            q,s,p,w,z] + lphspy.f_mach_labour_anyone(model,q,s,p,z) <= 0

    model.con_labour_crop_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,
                                                 rule=labour_crop_anyone,
                                                 doc='link between labour supply and requirement by crop jobs for all labour sources')


def f_con_labour_phase_perm(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_perm(model,q,s,p,w,z):
        return - model.v_phase_labour_permanent[q,s,p,w,z] - model.v_phase_labour_manager[q,s,p,w,z] + lphspy.f_mach_labour_perm(
            model,q,s,p,z) <= 0

    model.con_labour_crop_perm = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['perm'],model.s_season_types,rule=labour_crop_perm,
                                               doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')


def f_con_labour_sheep_anyone(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by anyone (casual/permanent/manager) and
    ensures that there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_cas(model,q,s,p,w,z):
        return -model.v_sheep_labour_casual[q,s,p,w,z] - model.v_sheep_labour_permanent[q,s,p,w,z] - \
               model.v_sheep_labour_manager[q,s,p,w,z] + suppy.f_sup_labour(model,q,s,p,z) + stkpy.f_stock_labour_anyone(model,q,s,p,z) <= 0

    model.con_labour_sheep_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,rule=labour_sheep_cas,
                                                  doc='link between labour supply and requirement by sheep jobs for all labour sources')


def f_con_labour_sheep_perm(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_perm(model,q,s,p,w,z):
        return - model.v_sheep_labour_permanent[q,s,p,w,z] - model.v_sheep_labour_manager[q,s,p,w,z] + stkpy.f_stock_labour_perm(
            model,q,s,p,z) <= 0

    model.con_labour_sheep_perm = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['perm'],model.s_season_types,rule=labour_sheep_perm,
                                                doc='link between labour supply and requirement by sheep jobs for perm labour sources')


def f_con_labour_sheep_manager(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_manager(model,q,s,p,w,z):
        return - model.v_sheep_labour_manager[q,s,p,w,z] + stkpy.f_stock_labour_manager(model,q,s,p,z) <= 0

    model.con_labour_sheep_manager = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['mngr'],model.s_season_types,
                                                   rule=labour_sheep_manager,
                                                   doc='link between labour supply and requirement by sheep jobs for manager labour sources')


# def f_con_cropgraze_area(model):
#     '''
#     Constrains the area of crop grazed to the amount provided by the selected rotations.
#     '''
#     if pinp.cropgraze['i_cropgrazing_inc']:
#         def cropgraze_area(model,p7,k,l,z):
#             return -sum(model.v_phase_area[p7,z,r,l] * model.p_cropgrazing_area[r,k,l] for r in model.s_phases if pe.value(model.p_cropgrazing_area[r,k,l])!=0)\
#                    + model.v_grazecrop_ha[p7,k,z,l] <= 0
#
#         model.con_cropgraze_area = pe.Constraint(model.s_season_periods, model.s_crops, model.s_lmus, model.s_season_types, rule=cropgraze_area,
#                                                        doc='link rotation area to the area of crop that can be grazed')


def f_con_harv_stub_nap_cons(model):
    '''
    Constrains the ME from stubble and non arable pasture in the feed period that harvest occurs. To consume ME from
    stubble and non arable pasture sheep must also consume a proportion (depending on when harvest occurs within
    the feed period) of their total intake from pasture. This stops sheep consuming all their energy intake
    for a given period from stubble and non arable pasture when they don’t become available until after harvest
    (the logic behind this is explained in the stubble section of this document).
    '''
    def harv_stub_nap_cons(model,q,s,p6,z):
        if any(model.p_nap_prop[p6,z] or model.p_harv_prop[p6,z,k] for k in model.s_crops):
            return sum(-paspy.f_pas_me(model,q,s,p6,f,z)
                       + sum(model.p_harv_prop[p6,z,k] / (1 - model.p_harv_prop[p6,z,k])
                             * model.v_stub_con[q,s,f,p6,z,k,sc,s2] * model.p_stub_md[f,p6,z,k,sc]
                             for k in model.s_crops for sc in model.s_stub_cat for s2 in model.s_biomass_uses)
                       + model.p_nap_prop[p6,z] / (1 - model.p_nap_prop[p6,z]) * paspy.f_nappas_me(model,q,s,p6,f,z)
                       for f in model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip

    model.con_harv_stub_nap_cons = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_season_types,rule=harv_stub_nap_cons,
                                                 doc='limit stubble and nap consumption in the period harvest occurs')


# def f_con_stubble_a(model):
#     '''
#     Constrains the amount of stubble required to consume 1t of category A to no more than the total amount of
#     stubble produced from each rotation.
#     '''
#     ##has to have p7 axis and transfer because penalties occur at seeding time and have to transfer as seasons are unclustered (same as yield)
#     def stubble_a(model,q,s,p7,k,z9):
#         l_p7 = list(model.s_season_periods)
#         p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
#         p7_end = l_p7[-1]
#         if pe.value(model.p_wyear_inc_qs[q,s]):
#             return (-sum(model.v_phase_area[q,s,p7,z9,r,l] * model.p_rot_stubble[r,k,l,p7,z9]
#                          for r in model.s_phases for l in model.s_lmus
#                          if pe.value(model.p_rot_stubble[r,k,l,p7,z9]) != 0)
#                     + macpy.f_stubble_penalty(model,q,s,p7,k,z9) + cgzpy.f_grazecrop_stubble_penalty(model,q,s,p7,k,z9)
#                     + sum(model.v_stub_harv[q,s,p6,z9,k] * 1000 * model.p_a_p6_p7[p7,p6,z9] for p6 in model.s_feed_periods)
#                     - model.v_stub_debit[q,s,p7,k,z9] *1000 * (p7 != p7_end) #cant debit in the final period otherwise unlimited stubble.
#
#                     + sum(model.v_stub_debit[q,s,p7_prev,k,z8] * 1000 * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
#                           for z8 in model.s_season_types) <= 0)
#         else:
#             return pe.Param.Skip
#
#     model.con_stubble_a = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops, model.s_season_types, rule=stubble_a,
#                                         doc='Total stubble at harvest. Provides Cat A at harvest.')


def f_con_phasesow(model):
    '''
    Ensures that the seeding requirements for each rotation phase selected can be met by the capacity of the
    machinery and/or with the help of contract work.

    Links seeding requirement with machinery sowing activity (which accounts for machinery use and labour
    needed to sow pasture).
    No p5 set in the constraint because model can optimise sowing time (can only optimise within the periods provided eg
    dry sowing activity only provides sowing capacity before the break).

    p_sow_prov links k and p5 so that:

        #. dry sown landuses are sown before the break.
        #. pasture is sown in the correct p5 periods based on the inputted reseeding date.
        #. wet sown crops are sown after the break of season.

    v_seeding_machdays is bound in con_seed_period_days to ensure that the correct days of seeding are provided in
    each p7 and p5 period.

    Notes:

        #. The requirement for seeding is based on v_phase_increment rather than v_phase_area
        #. a phase can only be sown in the phase_period for which the phase_increment is selected. If there is
           insufficient seeding capacity then the selection of v_phase_increment must be made in a later phase_period.

    Note: this is an equals to constraint to stop the model sowing without a landuse so it can get poc and crop
          grazing (both of those activities are provided by seeding).
    '''
    def sow_link(model,q,s,p7,k,l,z):
        if type(phspy.f_phasesow_req(model,q,s,p7,k,l,z)) == int:  # if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip  # skip constraint if no crop is being sown on given rotation
        else:
            return - sum(model.v_contractseeding_ha[q,s,z,p5,k,l] * model.p_contractseeding_occur[p5,z] * model.p_sow_prov[p7,p5,z,k] for p5 in model.s_labperiods) \
                   - sum(model.v_seeding_machdays[q,s,z,p5,k,l] * model.p_seeding_rate[k,l] * model.p_sow_prov[p7,p5,z,k] for p5 in model.s_labperiods) \
                   + phspy.f_phasesow_req(model,q,s,p7,k,l,z) == 0

    model.con_phasesow = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_landuses, model.s_lmus,model.s_season_types,rule=sow_link,
                                      doc='link between mach sow provide and rotation crop sow require')


# def f_con_passow(model):
#     '''
#     Links pasture seeding requirement with machinery sowing activity (which accounts for machinery use and labour
#     needed to sow pasture). Requires a p set because the timing of sowing pasture is not optimisable (pasture
#     sowing can occur in any period so the user specifies the periods when a given pasture must be sown)
#
#     Pasture sow has separate constraint from crop sow because pas sow has a p axis so that user can specify period
#     when pasture is sown (pasture has no yield penalty so model doesn't optimise seeding time like it does for crop)
#     '''
#
#     def passow_link(model,p5,k,l,z):
#         if type(paspy.f_passow(model,p5,k,l,
#                              z)) == int:  # if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
#             return pe.Constraint.Skip  # skip constraint if no pasture is being sown
#         else:
#             return -model.v_seeding_pas[p5,k,l,z] + paspy.f_passow(model,p5,k,l,z) <= 0
#
#     model.con_passow = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,model.s_landuses,model.s_lmus,model.s_season_types,
#                                      rule=passow_link,doc='link between mach sow provide and rotation pas sow require')


def f_con_harv(model):
    '''
    Links the harvest requirement for each rotation with harvesting capacity. Harvest capacity can be provided from
    farmer labour and machinery or contract services.
    '''
    ##Transfer unharvested grain in case a season node occurs between two harvest periods. The harvest requirement needs to uncluster to the new seasons.
    def harv(model,q,s,p7,k,s2,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1]  # need the activity level from last feed period
        p7_end = l_p7[-1]
        return (-macpy.f_harv_supply(model,q,s,p7,k,z9)
                + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] * model.p_biomass2product[k,l,s2] #adjust with biomass2product because harv dv are based on grain yield not biomass
                      for l in model.s_lmus)
                - model.v_unharvested_yield[q,s,p7,k,z9] * (p7 != p7_end) #must be harvested before the beginning of the next yr - therefore no transfer
                + sum(model.v_unharvested_yield[q,s,p7_prev,k,z8] * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
                      for z8 in model.s_season_types)
                <= 0)

    model.con_harv = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops, ['Harv'], model.s_season_types, rule=harv,doc='harvest constraint')


def f_con_makehay(model):
    '''
    Constrains the hay making requirement for each rotation by hay making capacity. Hay making capacity is provided
    by contract services.
    '''
    ##Transfer unharvested grain in case a season node occurs between two harvest periods. The harvest requirement needs to uncluster to the new seasons.
    def hay(model,q,s,p7,s2,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        p7_end = l_p7[-1]
        return (-model.v_hay_made[q,s,z9] * model.p_hay_made_prov[p7,z9]
                   + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] * model.p_biomass2product[k,l,s2]
                         for k in model.s_crops for l in model.s_lmus)
               - model.v_hay_tobe_made[q,s,p7,z9] * (p7 != p7_end) #must be baled before the beginning of the next yr - therefore no transfer
               + sum(model.v_hay_tobe_made[q,s,p7_prev,z8] * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
                     for z8 in model.s_season_types)
               <= 0)

    model.con_makehay = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, ['Bale'], model.s_season_types,rule=hay,doc='make hay constraint')


def f_con_biomass_transfer(model):
    '''
    Tracks the biomass of each phase and allows the transfer of biomass to either grain, hay or fodder for grazing.
    Biomass penalties associated with untimely sowing and/or crop grazing are accounted for here as well.
    '''
    ##Must pass between p7 because need to transfer penalties through the season

    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def biomass_transfer(model,q,s,p7,k,l,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        p7_start = l_p7[0]
        p7_end = l_p7[-1]

        return -phspy.f_rotation_biomass(model,q,s,p7,k,l,z9) + macpy.f_late_seed_penalty(model,q,s,p7,k,l,z9) \
               + cgzpy.f_grazecrop_biomass_penalty(model,q,s,p7,k,l,z9) \
               - model.v_biomass_debit[q,s,p7,z9,k,l] * 1000 * (p7 != p7_end) \
               + model.v_biomass_credit[q,s,p7,z9,k,l] * 1000 \
               + sum((model.v_biomass_debit[q,s,p7_prev,z8,k,l] * 1000 - model.v_biomass_credit[q,s,p7_prev,z8,k,l] * 1000 * (p7 != p7_start)) * model.p_parentz_provwithin_phase[p7_prev,z8,z9
                     ]   # p7!=p7[0] to stop biomass tranfer from last yr to current yr else unbounded solution.
                     for z8 in model.s_season_types) \
               + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] for s2 in model.s_biomass_uses) * 1000 <= 0

    model.con_biomass_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops,
                                               model.s_lmus, model.s_season_types,rule=biomass_transfer, doc='constrain biomass transfer')


def f_con_product_transfer(model):
    '''
    Links rotation product (grain or baled biomass), feed requirement for supplementary feeding and the sale and
    purchase of grain/hay.
    Grain fed to the sheep is purchased unless it is produced on farm.
    The net grain is either purchased or sold depending on the final balance.
    '''
    ##Must pass between p7 because harvest could be in different p7's for different crops and sale/buy timing could
    ## be in different p7 to harvest

    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def product_transfer(model,q,s,p7,g,k,s2,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        p7_start = l_p7[0]
        p7_end = l_p7[-1]

        return -phspy.f_rotation_product(model,q,s,p7,g,k,s2,z9) \
               + sum(model.v_sup_con[q,s,z9,k,g,f,p6] * model.p_sup_s2[k,s2] * model.p_a_p6_p7[p7,p6,z9] * 1000
                     for f in model.s_feed_pools for p6 in model.s_feed_periods) \
               - model.v_grain_debit[q,s,p7,z9,k,s2,g] * 1000 * (p7 != p7_end) \
               + model.v_grain_credit[q,s,p7,z9,k,s2,g] * 1000 \
               + sum((model.v_grain_debit[q,s,p7_prev,z8,k,s2,g] * 1000 - model.v_grain_credit[q,s,p7_prev,z8,k,s2,g] * 1000 * (p7 != p7_start)) * model.p_parentz_provwithin_phase[p7_prev,z8,z9
                     ]   # p7!=p7[0] to stop grain tranfer from last yr to current yr else unbounded solution.
                     for z8 in model.s_season_types) \
               - model.v_buy_grain[q,s,p7,z9,k,s2,g] * model.p_buy_grain_prov[p7,z9] * 1000 + model.v_sell_grain[q,s,p7,z9,k,s2,g] * 1000 <= 0

    model.con_product_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_grain_pools,
                                               model.s_crops,model.s_biomass_uses,model.s_season_types,rule=product_transfer,
                                             doc='constrain grain transfer between rotation and sup feeding')


def f1_grain_income(model,q,s,p7,z,c1):
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    return sum(
            model.v_sell_grain[q,s,p7,z,k,s2,g] * model.p_grain_price[p7,z,g,k,s2,c1] - model.v_buy_grain[q,s,p7,z,k,s2,g] * model.p_buy_grain_price[
            p7,z,g,k,s2,c1] for k in model.s_crops for s2 in model.s_biomass_uses for g in model.s_grain_pools)

def f1_grain_wc(model,q,s,c0,p7,z):
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    return sum(
        model.v_sell_grain[q,s,p7,z,k,s2,g] * model.p_grain_wc[c0,p7,z,g,k,s2] - model.v_buy_grain[q,s,p7,z,k,s2,g] * model.p_buy_grain_wc[
            c0,p7,z,g,k,s2] for k in model.s_crops for s2 in model.s_biomass_uses for g in model.s_grain_pools)


def f_con_poc_available(model):
    '''
    Constrains the foo consumed on crop paddocks before seeding. The foo available is determined by the number of
    hectares sown and the number of days each hectare can be grazed (calculated in Mach.py) multiplied
    by the foo available to be consumed on each hectare each day (calculated in Pasture.py).
    '''
    def poc(model,q,s,f,l,z):
        return -macpy.f_ha_days_pasture_crop_paddocks(model,q,s,f,l,z) * model.p_poc_con[f,l,z] + sum(
            model.v_poc[q,s,v,f,l,z] for v in model.s_feed_pools) <= 0

    model.con_poc_available = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_lmus,model.s_season_types,rule=poc,
                                            doc='constraint between poc available and consumed')


def f_con_me(model):
    '''
    All livestock activities require energy. Provided by pastures, stubbles, supplementary feeding, grazing of green
    crops and other fodder crops.

    The livestock require energy and the quantity required per head depends on the target liveweight profile of
    the animals (are the animals gaining or losing condition), the proportion of ewes in the flock
    (because ewes generally require more energy than dry animals) and the level of productivity (particularly the
    level of reproduction).

    The combination of a minimum energy requirement and maximum intake capacity sets a minimum diet quality for the
    animal that must be provided by the diet selected. This ensures that the diet selected is feasible for the animal
    to consume i.e. the diet selected cannot consist purely of a large quantity of low quality (cheap) fodder that
    is beyond the capacity of the animal to consume.

    '''
    def me(model,q,s,p6,f,z):
        return -paspy.f_pas_me(model,q,s,p6,f,z) - paspy.f_nappas_me(model,q,s,p6,f,z) - suppy.f_sup_me(model,q,s,p6,f,z) \
               - stubpy.f_stubble_me(model,q,s,p6,f,z) - cgzpy.f_grazecrop_me(model,q,s,p6,f,z) \
               + stkpy.f_stock_me(model,q,s,p6,f,z) - mvf.f_mvf_me(model,q,s,p6,f) <= 0

    model.con_me = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_feed_pools,model.s_season_types,rule=me,
                                 doc='constraint between me available and consumed')


def f_con_vol(model):
    '''
    All livestock activities provide volume to consume feed. The volume required by each feed changes depending on
    quality and quality of availability.

    The combination of a minimum energy requirement and maximum intake capacity sets a minimum diet quality for the
    animal that must be provided by the diet selected. This ensures that the diet selected is feasible for the animal
    to consume i.e. the diet selected cannot consist purely of a large quantity of low quality (cheap) fodder that
    is beyond the capacity of the animal to consume.

    '''
    def vol(model,q,s,p6,f,z):
        return paspy.f_pas_vol(model,q,s,p6,f,z) + suppy.f_sup_vol(model,q,s,p6,f,z) + stubpy.f_stubble_vol(model,q,s,p6,f,z) \
               + cgzpy.f_grazecrop_vol(model,q,s,p6,f,z) \
               - stkpy.f_stock_pi(model,q,s,p6,f,z) \
               + mvf.f_mvf_vol(model,q,s,p6,f) <= 0

    model.con_vol = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_feed_pools,model.s_season_types,rule=vol,
                                  doc='constraint between me available and consumed')


def f_con_cashflow(model):
    '''
    Tallies all cashflow in each period and transfers to the next period. Cashflow periods exist so that a transfer can
    exist between parent and child seasons.
    '''
    def cash_flow(model,q,s,c1,p7,z9):
        p7_start = list(model.s_season_periods)[0]
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        return ((-f1_grain_income(model,q,s,p7,z9,c1) + phspy.f_rotation_cost(model,q,s,p7,z9) + labpy.f_labour_cost(model,q,s,p7,z9)
                + macpy.f_mach_cost(model,q,s,p7,z9) + suppy.f_sup_cost(model,q,s,p7,z9) + model.p_overhead_cost[p7,z9]
                - stkpy.f_stock_cashflow(model,q,s,p7,z9,c1)
                - model.v_debit[q,s,c1,p7,z9] + model.v_credit[q,s,c1,p7,z9])
                + sum((model.v_debit[q,s,c1,p7_prev,z8] - model.v_credit[q,s,c1,p7_prev,z8]) * model.p_parentz_provwithin_season[p7_prev,z8,z9] * (p7!=p7_start)  #end cashflow doesnot provide start cashflow else unbounded.
                      for z8 in model.s_season_types)) <= 0

    model.con_cashflow_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_c1, model.s_season_periods, model.s_season_types,rule=cash_flow,
                                                doc='transfer of cash between periods')


def f_con_workingcap_within(model):
    '''
    Tallies working capital and transfers to the next period. Cashflow periods exist so that a transfer can
    exist between parent and child seasons.

    '''
    def working_cap_within(model,q,s,c0,p7,z9):
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        if pe.value(model.p_mask_childz_within_season[p7,z9]) and pe.value(model.p_wyear_inc_qs[q,s]):
            return (-f1_grain_wc(model,q,s,c0,p7,z9) + phspy.f_rotation_wc(model,q,s,c0,p7,z9) + labpy.f_labour_wc(model,q,s,c0,p7,z9)
                    + macpy.f_mach_wc(model,q,s,c0,p7,z9) + suppy.f_sup_wc(model,q,s,c0,p7,z9) + model.p_overhead_wc[c0,p7,z9]
                    - stkpy.f_stock_wc(model,q,s,c0,p7,z9)
                    - model.v_wc_debit[q,s,c0,p7,z9]
                    + model.v_wc_credit[q,s,c0,p7,z9]
                    + sum((model.v_wc_debit[q,s,c0,p7_prev,z8] - model.v_wc_credit[q,s,c0,p7_prev,z8]) #end working capital doesnot provide start else unbounded constraint.
                          * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                         for z8 in model.s_season_types)) <= 0
        else:
            return pe.Constraint.Skip
    model.con_workingcap_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types,rule=working_cap_within,
                                       doc='working capital transfer within year')


def f_con_workingcap_between(model):
    '''
    Tallies working capital and transfers to the next period. Cashflow periods exist so that a transfer can
    exist between parent and child seasons.

    Cashflow at the end of the previous yr becomes the starting balance for working capital (cashflow broadcasts
    to both c0 slices). This only happens between years in the sequence. End to start doesn't carry over. This is
    because it was decided that the working capital constraint is more useful if there is no starting balance at
    the start of the sequence. This means an expensive strategy with a high reward can be bounded using wc (if the end
    cashflow became the start then a high expense high income strategy would not trigger the constraint).

    '''
    def working_cap_between(model,q,s9,c0,p7,z9):
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
        if pe.value(model.p_mask_childz_between_season[p7,z9]) and pe.value(model.p_wyear_inc_qs[q,s9]):
            return (-f1_grain_wc(model,q,s9,c0,p7,z9) + phspy.f_rotation_wc(model,q,s9,c0,p7,z9) + labpy.f_labour_wc(model,q,s9,c0,p7,z9)
                    + macpy.f_mach_wc(model,q,s9,c0,p7,z9) + suppy.f_sup_wc(model,q,s9,c0,p7,z9) + model.p_overhead_wc[c0,p7,z9]
                    - stkpy.f_stock_wc(model,q,s9,c0,p7,z9)
                    - model.v_wc_debit[q,s9,c0,p7,z9]
                    + model.v_wc_credit[q,s9,c0,p7,z9]
                    + sum(sum((model.v_debit[q,s8,c1,p7_prev,z8] - model.v_credit[q,s8,c1,p7_prev,z8]) * model.p_prob_c1[c1]
                              for c1 in model.s_c1)#end cashflow become start wc (only within a sequence).
                          * model.p_parentz_provbetween_season[p7_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                        # + (model.v_debit[q,s8,c1,p7_prev,z8] - model.v_credit[q,s8,c1,p7_prev,z8]) #end cashflow become start wc.
                        #   * model.p_parentz_provbetween_season[p7_prev,z8,z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                          for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q,s8])!=0)) <= 0
        else:
            return pe.Constraint.Skip
    model.con_workingcap_between = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types,rule=working_cap_between,
                                       doc='working capital transfer between years')


def f_con_dep(model):
    '''Tallies the depreciation of capital, which is then passed to the objective.'''
    def dep(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        return (macpy.f_total_dep(model,q,s,p7,z9) + suppy.f_sup_dep(model,q,s,p7,z9) - model.v_dep[q,s,p7,z9]
                + sum(model.v_dep[q,s,p7_prev,z9] * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                      for z8 in model.s_season_types) * (p7!=p7_start) #end doesn't carry over
                <= 0)

    model.con_dep = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=dep,
                                  doc='tallies depreciation from all activities so it can be transferred to objective')


def f_con_asset(model):
    '''Tallies the total asset value to ensure that there is a minimum ROI on farm assets. The asset value multiplied
    by opportunity cost on capital is then passed to the objective.
    '''
    def asset(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        return (suppy.f_sup_asset(model,q,s,p7,z9) + macpy.f_mach_asset(model,p7) + stkpy.f_stock_asset(model,q,s,p7,z9)) * uinp.finance['opportunity_cost_capital'] \
               - model.v_asset[q,s,p7,z9] \
               + sum(model.v_asset[q,s,p7_prev,z8] * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                     for z8 in model.s_season_types) * (p7!=p7_start) <= 0 #end doesn't carry over

    model.con_asset = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=asset,
                                    doc='tallies asset from all activities so it can be transferred to objective to represent ROE')


def f_con_minroe(model):
    '''Tallies the total expenditure to ensure that there is a minimum ROI on cash expenditure.'''
    def minroe(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        return ((phspy.f_rotation_cost(model,q,s,p7,z9) + labpy.f_labour_cost(model,q,s,p7,z9) + macpy.f_mach_cost(model,q,s,p7,z9)
                + suppy.f_sup_cost(model,q,s,p7,z9) + stkpy.f_stock_cost(model,q,s,p7,z9))
                * fin.f_min_roe()
                - model.v_minroe[q,s,p7,z9]
                + sum(model.v_minroe[q,s,p7_prev,z8] *model.p_parentz_provwithin_season[p7_prev,z8,z9]
                      for z8 in model.s_season_types) * (p7 != p7_start)) <= 0  # end doesn't carry over

    model.con_minroe = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=minroe,
                                     doc='tallies total expenditure to ensure minimum roe is met')


def f_objective(model):
    '''
    The objective of the model is to maximise long run average profit. The
    objective function includes the net cash flow, a cost to represent a minimum return on operating costs incurred,
    the cost of depreciation and the opportunity cost on the farm assets (total value of all assets times the discount
    rate  (to ensure that the assets generate a minimum ROI)).
    '''
    variables = model.component_objects(pe.Var,active=True)

    p7_end = list(model.s_season_periods)[-1]
    return (sum((model.v_credit[q,s,c1,p7_end,z] - model.v_debit[q,s,c1,p7_end,z]
               - model.v_dep[q,s,p7_end,z] - model.v_minroe[q,s,p7_end,z] - model.v_asset[q,s,p7_end,z])
                * model.p_season_prob_qsz[q,s,z] * model.p_prob_c1[c1]
               for q in model.s_sequence_year for s in model.s_sequence for c1 in model.s_c1 for z in model.s_season_types)  # have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
               -0.00001 * sum(sum(v[s] for s in v) for v in variables))


