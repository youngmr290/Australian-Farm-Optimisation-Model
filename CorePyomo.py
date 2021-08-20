"""
Contains constraints for core equations and objective.

author: young
"""
import time
import pyomo.environ as pe
import numpy as np
import networkx
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
import StubblePyomo as stubpy
import StockPyomo as stkpy
import MVF as mvf
import Sensitivity as sen
import Finance as fin
import CropGrazingPyomo as cgzpy


def coremodel_all(params,trial_name,model):
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
    # stubble
    f_con_stubble_a(model)
    # sow landuse
    f_con_cropsow(model)
    f_con_passow(model)
    # harvest and make hay
    f_con_harv(model)
    f_con_makehay(model)
    # feed supply
    f_con_poc_available(model)
    f_con_vol(model)
    f_con_me(model)
    #crop grazing
    f_con_cropgraze_area(model)
    #grain
    f_con_grain_transfer(model)
    #cashflow
    f_con_cashflow(model)
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
    model.write('Output/test.lp',io_options={'symbolic_solver_labels': True})  # comment this out when not debugging

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
    def labour_fixed_casual(model,p,w,z):
        return -model.v_fixed_labour_casual[p,w,z] - model.v_fixed_labour_permanent[p,w,z] - \
               model.v_fixed_labour_manager[p,w,z] \
               + model.p_super_labour[p,z] + model.p_tax_labour[p,z] + model.p_bas_labour[p,z] <= 0

    model.con_labour_fixed_anyone = pe.Constraint(model.s_labperiods,['any'],model.s_season_types,
                                                  rule=labour_fixed_casual,
                                                  doc='link between labour supply and requirement by fixed jobs for casual and above')


def f_con_labour_fixed_manager(model):
    '''
    Tallies labour used for fixed activities that can only be completed by the manager and ensures that there is
    sufficient labour available to carry out the jobs.
    '''
    def labour_fixed_manager(model,p,w,z):
        return -model.v_fixed_labour_manager[p,w,z] + model.p_planning_labour[p,z] + (
                    model.p_learn_labour * model.v_learn_allocation[p,z]) <= 0

    model.con_labour_fixed_manager = pe.Constraint(model.s_labperiods,['mngr'],model.s_season_types,
                                                   rule=labour_fixed_manager,
                                                   doc='link between labour supply and requirement by fixed jobs for manager')


def f_con_labour_phase_anyone(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by anyone (casual/permanent/manager) and ensures
    that there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_anyone(model,p,w,z):
        return -model.v_phase_labour_casual[p,w,z] - model.v_phase_labour_permanent[p,w,z] - model.v_phase_labour_manager[
            p,w,z] + lphspy.f_mach_labour_anyone(model,p,z) <= 0

    model.con_labour_crop_anyone = pe.Constraint(model.s_labperiods,['any'],model.s_season_types,
                                                 rule=labour_crop_anyone,
                                                 doc='link between labour supply and requirement by crop jobs for all labour sources')


def f_con_labour_phase_perm(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_perm(model,p,w,z):
        return - model.v_phase_labour_permanent[p,w,z] - model.v_phase_labour_manager[p,w,z] + lphspy.f_mach_labour_perm(
            model,p,z) <= 0

    model.con_labour_crop_perm = pe.Constraint(model.s_labperiods,['perm'],model.s_season_types,rule=labour_crop_perm,
                                               doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')


def f_con_labour_sheep_anyone(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by anyone (casual/permanent/manager) and
    ensures that there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_cas(model,p,w,z):
        return -model.v_sheep_labour_casual[p,w,z] - model.v_sheep_labour_permanent[p,w,z] - \
               model.v_sheep_labour_manager[p,w,z] + suppy.f_sup_labour(model,p,z) + stkpy.f_stock_labour_anyone(model,p,z) <= 0

    model.con_labour_sheep_anyone = pe.Constraint(model.s_labperiods,['any'],model.s_season_types,rule=labour_sheep_cas,
                                                  doc='link between labour supply and requirement by sheep jobs for all labour sources')


def f_con_labour_sheep_perm(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_perm(model,p,w,z):
        return - model.v_sheep_labour_permanent[p,w,z] - model.v_sheep_labour_manager[p,w,z] + stkpy.f_stock_labour_perm(
            model,p,z) <= 0

    model.con_labour_sheep_perm = pe.Constraint(model.s_labperiods,['perm'],model.s_season_types,rule=labour_sheep_perm,
                                                doc='link between labour supply and requirement by sheep jobs for perm labour sources')


def f_con_labour_sheep_manager(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_manager(model,p,w,z):
        return - model.v_sheep_labour_manager[p,w,z] + stkpy.f_stock_labour_manager(model,p,z) <= 0

    model.con_labour_sheep_manager = pe.Constraint(model.s_labperiods,['mngr'],model.s_season_types,
                                                   rule=labour_sheep_manager,
                                                   doc='link between labour supply and requirement by sheep jobs for manager labour sources')


def f_con_cropgraze_area(model):
    '''
    Constrains the area of crop grazed to the amount provided by the selected rotations.
    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        def cropgraze_area(model,k,l,z):
            return -sum(model.v_phase_area[z,r,l] * model.p_cropgrazing_area[r,k,l] for r in model.s_phases if pe.value(model.p_cropgrazing_area[r,k,l])!=0)\
                   + model.v_grazecrop_ha[k,z,l] <= 0

        model.con_cropgraze_area = pe.Constraint(model.s_crops, model.s_lmus, model.s_season_types, rule=cropgraze_area,
                                                       doc='link rotation area to the area of crop that can be grazed')


def f_con_harv_stub_nap_cons(model):
    '''
    Constrains the ME from stubble and non arable pasture in the feed period that harvest occurs. To consume ME from
    stubble and non arable pasture sheep must also consume a proportion (depending on when harvest occurs within
    the feed period) of their total intake from pasture. This stops sheep consuming all their energy intake
    for a given period from stubble and non arable pasture when they don’t become available until after harvest
    (the logic behind this is explained in the stubble section of this document).
    '''
    def harv_stub_nap_cons(model,p6,z):
        if any(model.p_nap_prop[p6,z] or model.p_harv_prop[p6,z,k] for k in model.s_crops):
            return sum(-paspy.f_pas_me(model,p6,f,z) + sum(model.p_harv_prop[p6,z,k] / (1 - model.p_harv_prop[p6,z,k])
                                                         * model.v_stub_con[f,p6,z,k,s] * model.p_stub_md[f,p6,z,k,s]
                                                         for k in model.s_crops for s in model.s_stub_cat)
                       + model.p_nap_prop[p6,z] / (1 - model.p_nap_prop[p6,z]) * paspy.f_nappas_me(model,p6,f,z) for f in
                       model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip

    model.con_harv_stub_nap_cons = pe.Constraint(model.s_feed_periods,model.s_season_types,rule=harv_stub_nap_cons,
                                                 doc='limit stubble and nap consumption in the period harvest occurs')


def f_con_stubble_a(model):
    '''
    Constrains the amount of stubble required to consume 1t of category A to no more than the total amount of
    stubble produced from each rotation.
    '''
    def stubble_a(model,k,s,z):
        if model.p_rot_stubble[k] != 0:
            return -phspy.f1_total_rot_yield(model,k,z) * model.p_rot_stubble[k]  \
                   + macpy.f_stubble_penalty(model,k,z) + cgzpy.f_grazecrop_stubble_penalty(model,k,z) \
                   + stubpy.f_stubble_req_a(model,z,k,s) <= 0
        else:
            return pe.Constraint.Skip

    model.con_stubble_a = pe.Constraint(model.s_crops,model.s_stub_cat,model.s_season_types,rule=stubble_a,
                                        doc='links rotation stubble production with consumption of cat A')


def f_con_cropsow(model):
    '''
    Ensures that the seeding requirements for each crop rotation selected can be met by the capacity of the
    machinery and/or with the help of contract work.

    No p5 set because model can optimise crop sowing time
    '''
    def cropsow_link(model,k,l,z):
        if type(phspy.f_cropsow(model,k,l,
                              z)) == int:  # if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip  # skip constraint if no crop is being sown on given rotation
        else:
            return sum(-model.v_seeding_crop[p,k,l,z] for p in model.s_labperiods) + phspy.f_cropsow(model,k,l,z) <= 0

    model.con_cropsow = pe.Constraint(model.s_crops,model.s_lmus,model.s_season_types,rule=cropsow_link,
                                      doc='link between mach sow provide and rotation crop sow require')


def f_con_passow(model):
    '''
    Links pasture seeding requirement with machinery sowing activity (which accounts for machinery use and labour
    needed to sow pasture). Requires a p set because the timing of sowing pasture is not optimisable (pasture
    sowing can occur in any period so the user specifies the periods when a given pasture must be sown)

    Pasture sow has separate constraint from crop sow because pas sow has a p axis so that user can specify period
    when pasture is sown (pasture has no yield penalty so model doesnt optimise seeding time like it does for crop)
    '''

    def passow_link(model,p5,k,l,z):
        if type(paspy.f_passow(model,p5,k,l,
                             z)) == int:  # if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip  # skip constraint if no pasture is being sown
        else:
            return -model.v_seeding_pas[p5,k,l,z] + paspy.f_passow(model,p5,k,l,z) <= 0

    model.con_passow = pe.Constraint(model.s_labperiods,model.s_landuses,model.s_lmus,model.s_season_types,
                                     rule=passow_link,doc='link between mach sow provide and rotation pas sow require')


def f_con_harv(model):
    '''
    Links the harvest requirement for each rotation with harvesting capacity. Harvest capacity can be provided from
    farmer labour and machinery or contract services.
    '''
    def harv(model,k,z):
        return -macpy.f_harv_supply(model,k,z) + sum(
            phspy.f_rotation_yield_transfer(model,g,k,z) / 1000 for g in model.s_grain_pools) <= 0

    model.con_harv = pe.Constraint(model.s_harvcrops,model.s_season_types,rule=harv,doc='harvest constraint')


def f_con_makehay(model):
    '''
    Constrains the hay making requirement for each rotation by hay making capacity. Hay making capacity is provided
    by contract services.
    '''
    def harv(model,k,z):
        return sum(
            -model.v_hay_made[z] + phspy.f_rotation_yield_transfer(model,g,k,z) / 1000 for g in model.s_grain_pools) <= 0

    model.con_makehay = pe.Constraint(model.s_haycrops,model.s_season_types,rule=harv,doc='make hay constraint')


def f_con_grain_transfer(model):
    '''
    Tracks the grain transferred between different activities including the yield production from all rotations, late
    seeding penalties and grain feed to sheep. Grain fed to the sheep is purchased unless it is produced on farm.
    The net grain is either purchased or sold depending on the final balance.
    '''
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def grain_transfer(model,g,k,z):
        return -phspy.f_rotation_yield_transfer(model,g,k,z) + macpy.f_late_seed_penalty(model,g,k,z) \
               + cgzpy.f_grazecrop_yield_penalty(model,g,k,z) + sum(model.v_sup_con[z,k,g,f,p6] * 1000 for f in model.s_feed_pools for p6 in model.s_feed_periods) \
               - model.v_buy_grain[z,k,g] * 1000 + model.v_sell_grain[z,k,g] * 1000 <= 0

    model.con_grain_transfer = pe.Constraint(model.s_grain_pools,model.s_crops,model.s_season_types,rule=grain_transfer,
                                             doc='constrain grain transfer between rotation and sup feeding')


def f1_grain_income(model,c,z):
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    return sum(
        model.v_sell_grain[z,k,g] * model.p_grain_price[k,c,g] - model.v_buy_grain[z,k,g] * model.p_buy_grain_price[
            k,c,g] for k in model.s_crops for g in model.s_grain_pools)


def f_con_poc_available(model):
    '''
    Constrains the foo consumed on crop paddocks before seeding. The foo available is determined by the number of
    hectares sown and the number of days each hectare can be grazed (calculated in Mach.py) multiplied
    by the foo available to be consumed on each hectare each day (calculated in Pasture.py).
    '''
    def poc(model,f,l,z):
        return -macpy.f_ha_days_pasture_crop_paddocks(model,f,l,z) * model.p_poc_con[f,l,z] + sum(
            model.v_poc[v,f,l,z] for v in model.s_feed_pools) <= 0

    model.con_poc_available = pe.Constraint(model.s_feed_periods,model.s_lmus,model.s_season_types,rule=poc,
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
    def me(model,p6,f,z):
        return -paspy.f_pas_me(model,p6,f,z) - paspy.f_nappas_me(model,p6,f,z) - suppy.f_sup_me(model,p6,f,z) \
               - stubpy.f_stubble_me(model,p6,f,z) - cgzpy.f_grazecrop_me(model,p6,f,z) \
               + stkpy.f_stock_me(model,p6,f,z) - mvf.f_mvf_me(model,p6,f) <= 0

    model.con_me = pe.Constraint(model.s_feed_periods,model.s_feed_pools,model.s_season_types,rule=me,
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
    def vol(model,p6,f,z):
        return paspy.f_pas_vol(model,p6,f,z) + suppy.f_sup_vol(model,p6,f,z) + stubpy.f_stubble_vol(model,p6,f,z) \
               + cgzpy.f_grazecrop_vol(model,p6,f,z) \
               - stkpy.f_stock_pi(model,p6,f,z) \
               + mvf.f_mvf_vol(model,p6,f) <= 0

    model.con_vol = pe.Constraint(model.s_feed_periods,model.s_feed_pools,model.s_season_types,rule=vol,
                                  doc='constraint between me available and consumed')


def f_con_cashflow(model):
    '''
    Tallies all cashflow in each bi-monthly period from each module. If the balance is positive interest is earned.
    If the balance is negitive interest is paid.
    '''
    def cash_flow(model,i,z):
        c = sinp.general['cashflow_periods']
        ##j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        ##this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period)
        j = [1] * len(c)
        j[0] = 0
        # todo Revisit the interest calculation at some stage because it didn't tally with the back of envelope estimate by $1000
        return (-f1_grain_income(model,c[i],z) + phspy.f_rotation_cost(model,c[i],z) + labpy.f_labour_cost(model,c[i],z)
                + macpy.f_mach_cost(model,c[i],z) + suppy.f_sup_cost(model,c[i],z) + model.p_overhead_cost[c[i]]
                - stkpy.f_stock_cashflow(model,c[i],z)
                - model.v_debit[c[i]] + model.v_credit[c[i]]
                + (model.v_debit[c[i - 1]] * fin.debit_interest() - model.v_credit[c[i - 1]] * fin.credit_interest()) * j[i] # mul by j so that credit in ND doesnt provide into JF otherwise it will be unbounded because it will get interest
                ) <= 0

    model.con_cashflow = pe.Constraint(range(len(model.s_cashflow_periods)),model.s_season_types,rule=cash_flow,
                                       doc='cashflow')


def f_con_dep(model):
    '''Tallies the depreciation of capital, which is then passed to the objective.'''
    def dep(model,z):
        return macpy.f_total_dep(model,z) + suppy.f_sup_dep(model,z) - model.v_dep[z] <= 0

    model.con_dep = pe.Constraint(model.s_season_types,rule=dep,
                                  doc='tallies depreciation from all activities so it can be transferred to objective')


def f_con_asset(model):
    '''Tallies the total asset value to ensure that there is a minimum ROI on farm assets. The asset value multiplied
    by opportunity cost on capital is then passed to the objective.
    '''
    def asset(model,z):
        return (suppy.f_sup_asset(model,z) + macpy.f_mach_asset(model) + stkpy.f_stock_asset(model,z)) * uinp.finance[
            'opportunity_cost_capital'] \
               - model.v_asset[z] <= 0

    model.con_asset = pe.Constraint(model.s_season_types,rule=asset,
                                    doc='tallies asset from all activities so it can be transferred to objective to represent ROE')


def f_con_minroe(model):
    '''Tallies the total expenditure to ensure that there is a minimum ROI on cash expenditure.'''
    def minroe(model,z):
        return (sum(phspy.f_rotation_cost(model,c,z) + labpy.f_labour_cost(model,c,z) + macpy.f_mach_cost(model,c,z)
                    + suppy.f_sup_cost(model,c,z) for c in model.s_cashflow_periods) + stkpy.f_stock_cost(model,z)) * fin.f_min_roe() \
               - model.v_minroe[z] <= 0

    model.con_minroe = pe.Constraint(model.s_season_types,rule=minroe,
                                     doc='tallies total expenditure to ensure minimum roe is met')


def f_objective(model):
    '''
    The objective of the model is to maximise long run average profit. The
    objective function includes the net cash flow, a cost to represent a minimum return on operating costs incurred,
    the cost of depreciation and the opportunity cost on the farm assets (total value of all assets times the discount
    rate  (to ensure that the assets generate a minimum ROI)).
    '''
    c = sinp.general['cashflow_periods']
    i = len(c) - 1  # minus one because index starts from 0
    return model.v_credit[c[i]] - model.v_debit[c[i]] - sum(
        model.v_dep[z] + model.v_minroe[z] + model.v_asset[z] for z in
        model.s_season_types)  # have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
