"""
Contains constraints for core equations and objective.

author: young
"""
import time
import pyomo.environ as pe
import pyomo.core as pc
import numpy as np
# import networkx
# import pyomo.pysp.util.rapper as rapper
# import pyomo.pysp.plugins.csvsolutionwriter as csvw
# import pyomo.pysp.plugins.jsonsolutionwriter as jsonw
import os
import shutil

# AFO modules - should only be pyomo modules
from . import UniversalInputs as uinp
from . import PropertyInputs as pinp
from . import StructuralInputs as sinp
from . import PhasePyomo as phspy
from . import MachPyomo as macpy
from . import LabourPyomo as labpy
from . import LabourPhasePyomo as lphspy
from . import PasturePyomo as paspy
from . import SupFeedPyomo as suppy
from . import CropResiduePyomo as stubpy
from . import StockPyomo as stkpy
from . import MVF as mvf
from . import Sensitivity as sen
from . import Finance as fin
from . import CropGrazingPyomo as cgzpy
from . import SaltbushPyomo as slppy
from . import TreePyomo as treepy
from . import relativeFile

def coremodel_all(trial_name, model, method, nv, print_debug_output, MP_lp_vars):
    '''
    Wraps all of the core model into a function so it can be run multiple times in a loop


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
    # sow landuse
    f_con_phasesow(model)
    # harvest and make hay
    f_con_harv(model)
    f_con_makehay(model)
    # feed supply
    f_con_poc_available(model)
    f_con_link_understory_saltbush_consumption(model)
    f_con_link_pasture_supplement_consumption(model, nv)
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
    f_con_totalcap_within(model)
    f_con_totalcap_between(model)
    f_con_dep(model)
    f_con_asset(model)
    f_con_minroe(model)


    ###this last constraint is for the MP model to constrain the starting point
    if sinp.structuralsa['model_is_MP']:
        f_con_MP(model, MP_lp_vars)


    #############
    # objective #
    #############
    '''
    maximise credit in the last period of cashflow (rather than indexing directly with ND$FLOW, i index with the last name in the cashflow periods in case cashflow periods change) 
    minus dep (variable and fixed)
    '''
    model.utility = pe.Objective(rule=f_objective,sense=pe.maximize)
    # model.utility.pprint()

    #########
    # solve #
    #########
    ##add warmstart guesses - currently warm start file is only read for MIP (i think it might be somethig to do with the link between pyomo and cplex)
    import pickle as pkl
    ###read in past trial (need to decide which trial and change trial_name)
    # with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"rb") as f:
    #     lp_vars = pkl.load(f)
    ###update current variables with old solution (might need to add some error handling for cases when variables differ between the old trial and current trial)
    # for v in model.component_objects(pe.Var, active=True):
    #     for s in v:
    #         prev=lp_vars[str(v)][s]
    #         v[s] = prev

    ##sometimes if there is a bug when solved it is good to write lp here - because the code doesn't run to the other place where lp written
    ## print_debug_output can be set to True in RunAfoRaw.
    if print_debug_output==True:
        output_path = relativeFile.find(__file__, "../../Output", "test.lp")
        model.write(output_path,io_options={'symbolic_solver_labels': True})

    ##tells the solver you want duals and rc
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
    # model.slack = pe.Suffix(direction=pe.Suffix.IMPORT)
    ##solve - solver choice is passed in as an argument so the user can change it. -
    if method=="CPLEX" and not shutil.which("cplex") == None:
        ##solve with cplex if it exists
        solver = pe.SolverFactory('cplex')
        solver_result = solver.solve(model, warmstart=True, tee=False)  # tee=True for solver output - may be useful for troubleshooting, currently warmstart doesnt do anything (could only get it to work for MIP)
    elif method=="glpk":
        ##solve with glpk to see options enter glpsol --help into command prompt.
        solver = pe.SolverFactory('glpk')
        solver.options['tmlim'] = 100  # limit solving time to 100sec in case solver stalls.
        # solver.options['norelax'] = ""
        # solver.options['dual'] = ""
        # solver.options['nopresol'] = ""
        solver_result = solver.solve(model, tee=True)  # tee=True for solver output - may be useful for troubleshooting
    elif method=="cbc":
        solver = pe.SolverFactory('cbc')
        solver.options['seconds'] = 60  # limit solving time. Occasionally CBC takes a long time to find optimum solution but it gets very close to optimum quickly.
        solver_result = solver.solve(model, tee=True) #tee=True will print out solver information
    elif method=="ipopt":
        solver = pe.SolverFactory('ipopt')
        solver_result = solver.solve(model, tee=True) #tee=True will print out solver information.
    else:
        import highspy
        # solver = appsi.solvers.Highs()
        solver = pe.SolverFactory('appsi_highs')
        try:
            solver_result = solver.solve(model)
        except RuntimeError: #incase trial is infeasible (highs just throws an error but we want AFO to keep running so stop Highs trying to load solution)
            solver_result = solver.solve(model, load_solutions=False)


    ##calc profit - profit = terminal wealth (this is the objective without risk) + minroe + asset_cost
    try:  # to handle infeasible (there is no profit component when infeasible)
        p7_end = list(model.s_season_periods)[-1]
        utility = pe.value(model.utility)
        profit = pe.value(sum((model.v_terminal_wealth[q,s,z,c1] + model.v_minroe[q,s,p7_end,z] + model.v_asset_cost[q,s,p7_end,z])
                                       * model.p_season_prob_qsz[q,s,z] * model.p_prob_c1[c1] * model.p_discount_factor_q[q]
                                       for q in model.s_sequence_year for s in model.s_sequence for c1 in model.s_c1
                                       for z in model.s_season_types if pe.value(model.p_wyear_inc_qs[q,s])))
    except ValueError:
        utility = 0
        profit = 0

    ##this prints trial name, overall profit and feasibility for each trial
    print(f'\nDisplaying profit and obj for trial: {trial_name}')
    print(f'Profit: {profit}   Obj: {utility}')
    print('-' * 60)

    ##this check if the solver is optimal - if infeasible or error the model will save a file in Output/infeasible/ directory. This will be accessed in reporting to stop you reporting infeasible trials.
    ##the model will keep running the next trials even if one is infeasible.
    if (solver_result.solver.status == pe.SolverStatus.ok) and (
            solver_result.solver.termination_condition == pe.TerminationCondition.optimal):
        print('OPTIMAL LP SOLUTION FOUND')  # Do nothing when the solution in optimal and feasible
        trial_infeasible = False
    elif (solver_result.solver.termination_condition == pe.TerminationCondition.infeasible):
        print('***INFEASIBLE LP SOLUTION***')
        trial_infeasible = True
    else:  # Something else is wrong - solver may have stalled.
        print('***Solver Status: error (other)***')
        trial_infeasible = True

    return profit, utility, trial_infeasible

##############
#constriants #
##############
def f_con_labour_fixed_anyone(model):
    '''
    Tallies labour used for fixed activities that can be completed by anyone (casual/permanent/manager) and ensures
    that there is sufficient labour available to carry out the jobs.
    '''
    def labour_fixed_casual(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return -model.v_fixed_labour_casual[q,s,p,w,z] - model.v_fixed_labour_permanent[q,s,p,w,z] - \
                   model.v_fixed_labour_manager[q,s,p,w,z] \
                   + model.p_super_labour[p,z] + model.p_tax_labour[p,z] + model.p_bas_labour[p,z] <= 0
        else:
            return pe.Constraint.Skip
    model.con_labour_fixed_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,
                                                  rule=labour_fixed_casual,
                                                  doc='link between labour supply and requirement by fixed jobs for casual and above')


def f_con_labour_fixed_manager(model):
    '''
    Tallies labour used for fixed activities that can only be completed by the manager and ensures that there is
    sufficient labour available to carry out the jobs.
    '''
    def labour_fixed_manager(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return -model.v_fixed_labour_manager[q,s,p,w,z] + model.p_planning_labour[p,z] + (
                        model.p_learn_labour * model.v_flex_labour_allocation[q,s,p]) <= 0
        else:
            return pe.Constraint.Skip
    model.con_labour_fixed_manager = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['mngr'],model.s_season_types,
                                                   rule=labour_fixed_manager,
                                                   doc='link between labour supply and requirement by fixed jobs for manager')


def f_con_labour_phase_anyone(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by anyone (casual/permanent/manager) and ensures
    that there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_anyone(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return -model.v_phase_labour_casual[q,s,p,w,z] - model.v_phase_labour_permanent[q,s,p,w,z] - model.v_phase_labour_manager[
                q,s,p,w,z] + lphspy.f_mach_labour_anyone(model,q,s,p,z) <= 0
        else:
            return pe.Constraint.Skip
    model.con_labour_crop_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,
                                                 rule=labour_crop_anyone,
                                                 doc='link between labour supply and requirement by crop jobs for all labour sources')


def f_con_labour_phase_perm(model):
    '''
    Tallies labour used in the crop enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_crop_perm(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return - model.v_phase_labour_permanent[q,s,p,w,z] - model.v_phase_labour_manager[q,s,p,w,z] + lphspy.f_mach_labour_perm(
                model,q,s,p,z) <= 0
        else:
            return pe.Constraint.Skip
    model.con_labour_crop_perm = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['perm'],model.s_season_types,rule=labour_crop_perm,
                                               doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')


def f_con_labour_sheep_anyone(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by anyone (casual/permanent/manager) and
    ensures that there is sufficient labour available to carry out the jobs.

    Labour required for fixed R&M (each year irrelevant of stock numbers) of infrastructure get optimised into a
    labour period.

    Labour for variable R&M (depends on stock numbers) gets incurred in the labour period when the stock use the
    infrastructure. If this is limiting then it could be separated into its own parameter and optimised into a
    labour period like fixed R&M labour.
    '''
    def labour_sheep_cas(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return (-model.v_sheep_labour_casual[q,s,p,w,z] - model.v_sheep_labour_permanent[q,s,p,w,z]
                    - model.v_sheep_labour_manager[q,s,p,w,z] + suppy.f_sup_labour(model,q,s,p,z) + stkpy.f_stock_labour_anyone(model,q,s,p,z)
                    +model.p_lab_infra_rm_fixed * model.v_flex_labour_allocation[q,s,p]<= 0)
        else:
            return pe.Constraint.Skip
    model.con_labour_sheep_anyone = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['any'],model.s_season_types,rule=labour_sheep_cas,
                                                  doc='link between labour supply and requirement by sheep jobs for all labour sources')


def f_con_labour_sheep_perm(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by permanent or manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_perm(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return - model.v_sheep_labour_permanent[q,s,p,w,z] - model.v_sheep_labour_manager[q,s,p,w,z] + stkpy.f_stock_labour_perm(
                model,q,s,p,z) <= 0
        else:
            return pe.Constraint.Skip
    model.con_labour_sheep_perm = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_labperiods,['perm'],model.s_season_types,rule=labour_sheep_perm,
                                                doc='link between labour supply and requirement by sheep jobs for perm labour sources')


def f_con_labour_sheep_manager(model):
    '''
    Tallies labour used in the sheep enterprise that can be completed by manager staff and ensures that
    there is sufficient labour available to carry out the jobs.
    '''
    def labour_sheep_manager(model,q,s,p,w,z):
        if pe.value(model.p_wyear_inc_qs[q, s]):
            return - model.v_sheep_labour_manager[q,s,p,w,z] + stkpy.f_stock_labour_manager(model,q,s,p,z) <= 0
        else:
            return pe.Constraint.Skip
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
    Stubble and non-arable pasture may become available part way through a feed period, this means sheep could
    fill their entire energy requirement for that feed period solely with stubble and non-arable pasture.
    This is unreasonable because stubble can only be grazed after harvest.
    To solve this a constraint is implemented which forces a
    proportion (depending on when harvest occurs in the feed period) of the sheep
    energy intake to be consumed from pasture.

    The constraint is built so that consumption of any stubble requires a certain amount of pasture consumed.
    However, if one crop is harvested before another, consuming stubble from the latter crop still
    requires a certain amount of pasture to be consumed. This is a slight limitation because in reality
    the sheep may have been consuming crop A stubble. However consuming crop A stubble doesn't provide
    the ability to consume crop B stubble.

    '''
    def harv_stub_nap_cons(model,q,s,p6,z):

        nap_exists = (model.p_nap_prop[p6, z] != 0)
        stub_exists = len(model.s_stub_k_by_p6z[p6, z]) > 0
        if (nap_exists or stub_exists) and pe.value(model.p_wyear_inc_qs[q, s]):
            return sum(
                -paspy.f_pas_me(model,q,s,p6,f,z)
                + stubpy.f1_stub_consumed_in_harvest_period(model,q,s,p6,f,z)
                + model.p_nap_prop[p6,z] / (1 - model.p_nap_prop[p6,z]) * paspy.f_nappas_me(model,q,s,p6,f,z)
                for f in model.s_feed_pools
            ) <= 0
        else:
            return pe.Constraint.Skip
    model.con_harv_stub_nap_cons = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_season_types,rule=harv_stub_nap_cons,
                                                 doc='limit stubble and nap consumption in the period harvest occurs')



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

        #. The requirement for seeding is based on v_phase_change_increase rather than v_phase_area
        #. a phase can only be sown in the phase_period for which the phase_increment is selected. If there is
           insufficient seeding capacity then the selection of v_phase_change_increase must be made in a later phase_period.

    Note: this is an equals to constraint to stop the model sowing without a landuse so it can get poc and crop
          grazing (both of those activities are provided by seeding).
    '''
    def sow_link(model,q,s,p7,k,l,z):
        if not pe.value(model.p_wyear_inc_qs[q, s]) or (
                type(phspy.f_phasesow_req(model,q,s,p7,k,l,z)) == int and all(model.p_sow_prov[p7,p5,z,k]==0 for p5 in model.s_labperiods)):  # if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
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
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return (-macpy.f_harv_supply(model,q,s,p7,k,z9)
                    + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] * model.p_biomass2product[k,s2] #adjust with biomass2product because harv dv are based on grain yield not biomass
                          for l in model.s_lmus)
                    - model.v_unharvested_yield[q,s,p7,k,z9] * ((p7 != p7_end)*1) #must be harvested before the beginning of the next yr - therefore no transfer
                    + sum(model.v_unharvested_yield[q,s,p7_prev,k,z8] * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
                          for z8 in model.s_season_types)
                    <= 0)
        else:
            return pe.Constraint.Skip
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
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return (-model.v_hay_made[q,s,z9] * model.p_hay_made_prov[p7,z9]
                       + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] * model.p_biomass2product[k,s2]
                             for k in model.s_crops for l in model.s_lmus)
                   - model.v_hay_tobe_made[q,s,p7,z9] * ((p7 != p7_end)*1) #must be baled before the beginning of the next yr - therefore no transfer
                   + sum(model.v_hay_tobe_made[q,s,p7_prev,z8] * model.p_parentz_provwithin_phase[p7_prev,z8,z9]
                         for z8 in model.s_season_types)
                   <= 0)
        else:
            return pe.Constraint.Skip
    model.con_makehay = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, ['Bale'], model.s_season_types,rule=hay,doc='make hay constraint')


def f_con_biomass_transfer(model):
    '''
    Tracks the biomass of each phase and allows the transfer of biomass to either grain, hay or fodder for grazing.
    Biomass penalties associated with untimely sowing and/or crop grazing are accounted for here as well.

    Biomass doesn't carry between seasons because the choice biomass use must before or at harvest time.
    '''
    ##Must pass between p7 because need to transfer penalties through the season

    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def biomass_transfer(model,q,s,p7,k,l,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        p7_start = l_p7[0]
        p7_end = l_p7[-1]
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return -phspy.f_rotation_biomass(model,q,s,p7,k,l,z9) + macpy.f_late_seed_penalty(model,q,s,p7,k,l,z9) \
                   + cgzpy.f_grazecrop_biomass_penalty(model,q,s,p7,k,l,z9) \
                   - model.v_biomass_debit[q,s,p7,z9,k,l] * 1000 * ((p7 != p7_end)*1) \
                   + model.v_biomass_credit[q,s,p7,z9,k,l] * 1000 \
                   + sum((model.v_biomass_debit[q,s,p7_prev,z8,k,l] * 1000 - model.v_biomass_credit[q,s,p7_prev,z8,k,l]
                          * 1000 * ((p7 != p7_start)*1)) * model.p_parentz_provwithin_phase[p7_prev,z8,z9
                         ]   # p7!=p7[0] to stop biomass tranfer from last yr to current yr else unbounded solution.
                         for z8 in model.s_season_types) \
                   + sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] for s2 in model.s_biomass_uses) * 1000 <= 0
        else:
            return pe.Constraint.Skip
    model.con_biomass_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops,
                                               model.s_lmus, model.s_season_types,rule=biomass_transfer, doc='constrain biomass transfer')


def f_con_product_transfer(model):
    '''
    Links rotation product (grain or baled biomass), feed requirement for supplementary feeding and the sale and
    purchase of grain/hay.
    Grain fed to the sheep is purchased unless it is produced on farm.
    The net grain is either purchased or sold depending on the final balance.

    This constraint exists (as well as biomass transfer) because supplement requirement needs to be tracked and
    fulfilled by either purchasing or harvesting grain.

    Currently, the supplement fed during the year is either purchased or transferred from the cropping enterprise at the
    end of the season. This means we are underestimating variable and fixed storage costs in good seasons because less
    supplement is fed but in reality the same amount would be stored since a farmer doesn't know it is going to be a good
    year when making storage decisions.
    '''
    ##Must pass between p7 because harvest could be in different p7's for different crops and sale/buy timing could
    ## be in different p7 to harvest

    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def product_transfer(model,q,s,p7,g,k4,s2,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last feed period
        p7_start = l_p7[0]
        p7_end = l_p7[-1]
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return ((-phspy.f_rotation_product(model,q,s,p7,g,k4,s2,z9) if k4 in model.s_crops else 0) \
                   + (sum(model.v_sup_con[q,s,z9,k4,g,f,p6] * model.p_sup_s2[k4,s2] * model.p_a_p6_p7[p7,p6,z9] * 1000
                         for f in model.s_feed_pools for p6 in model.s_feed_periods) if k4 in model.s_supp_feeds else 0) \
                   - model.v_product_debit[q,s,p7,z9,k4,s2,g] * 1000 * ((p7 != p7_end)*1) \
                   + model.v_product_credit[q,s,p7,z9,k4,s2,g] * 1000 \
                   + sum((model.v_product_debit[q,s,p7_prev,z8,k4,s2,g] * 1000 - model.v_product_credit[q,s,p7_prev,z8,k4,s2,g]
                          * 1000 * ((p7 != p7_start)*1)) * model.p_parentz_provwithin_phase[p7_prev,z8,z9
                         ]   # p7!=p7[0] to stop grain transfer from last yr to current yr else unbounded solution.
                         for z8 in model.s_season_types) \
                   - (model.v_buy_product[q,s,p7,z9,k4,s2,g] * model.p_buy_product_prov[p7,z9] * 1000 if k4 in model.s_supp_feeds else 0)
                   + (model.v_sell_product[q,s,p7,z9,k4,s2,g] * 1000 if k4 in model.s_crops else 0) <= 0)
        else:
            return pe.Constraint.Skip
    model.con_product_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_grain_pools,
                                               model.s_crops_and_supp, model.s_biomass_uses, model.s_season_types, rule=product_transfer,
                                             doc='constrain grain transfer between rotation and sup feeding')


def f1_grain_income(model,q,s,p7,z,c1):
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    return sum(sum(model.v_sell_product[q,s,p7,z,k1,s2,g] * model.p_grain_price[q,p7,z,g,k1,s2,c1] for k1 in model.s_crops)
               - sum(model.v_buy_product[q,s,p7,z,k3,s2,g] * model.p_buy_grain_price[q,p7,z,g,k3,s2,c1] for k3 in model.s_supp_feeds)
               for s2 in model.s_biomass_uses for g in model.s_grain_pools)

def f1_grain_wc(model,q,s,c0,p7,z):
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    return sum(sum(model.v_sell_product[q,s,p7,z,k1,s2,g] * model.p_grain_wc[q,c0,p7,z,g,k1,s2] for k1 in model.s_crops)
               - sum(model.v_buy_product[q,s,p7,z,k3,s2,g] * model.p_buy_grain_wc[q,c0,p7,z,g,k3,s2] for k3 in model.s_supp_feeds)
               for s2 in model.s_biomass_uses for g in model.s_grain_pools)

def f1_sup_minroe(model,q,s,p7,z):
    ##cost of grain for livestock enterprise. Note grain purchased cost more because of transport fees than grain transferred from crop enterprise.
    return sum(((sum(model.v_sup_con[q,s,z,k3,g,f,p6] for f in model.s_feed_pools for p6 in model.s_feed_periods)
                 - model.v_buy_product[q,s,p7,z,k3,s2,g]) * sum(model.p_grain_price[q,p7,z,g,k3,s2,c1] * model.p_prob_c1[c1] for c1 in model.s_c1) if k3 in model.s_crops else 0)
               + model.v_buy_product[q,s,p7,z,k3,s2,g] * sum(model.p_buy_grain_price[q,p7,z,g,k3,s2,c1]  * model.p_prob_c1[c1] for c1 in model.s_c1)
               for k3 in model.s_supp_feeds for s2 in model.s_biomass_uses for g in model.s_grain_pools)

def f_con_poc_available(model):
    '''
    Constrains the foo consumed on crop paddocks before seeding. The foo available is determined by the number of
    hectares sown and the number of days each hectare can be grazed (calculated in Mach.py) multiplied
    by the foo available to be consumed on each hectare each day (calculated in Pasture.py).
    '''
    def poc(model,q,s,f,l,z):
        if pe.value(model.p_wyear_inc_qs[q, s]) and pinp.crop['i_poc_inc']:
            return -macpy.f_ha_days_pasture_crop_paddocks(model,q,s,f,l,z) * model.p_poc_con[f,l,z] + sum(
                model.v_poc[q,s,v,f,l,z] for v in model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip
    model.con_poc_available = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_lmus,model.s_season_types,rule=poc,
                                            doc='constraint between poc available and consumed')


def f_con_link_understory_saltbush_consumption(model):
    '''
    Constrains the consumption of understory and saltbush based on the estimated diet selection of animals grazing
    salt land pasture. Saltbush info comes from saltbushpyomo and understory comes from pasturepyomo.
    '''
    def link_us_sb(model,q,s,z,p6,f,l):
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p6z[p6,z]) and pinp.general['pas_inc_t'][3]:
            return - slppy.f_saltbush_selection(model,q,s,z,p6,f,l) \
                   + sum(model.v_greenpas_ha[q, s, f, g, o, p6, l, z, 'understory'] * model.p_volume_grnha[q,f, g, o, p6, l, z, 'understory'] * model.p_sb_selectivity_zp6[z,p6]
                        for g in model.s_grazing_int for o in model.s_foo_levels) \
                   + sum(model.v_drypas_consumed[q, s, f, d, p6, z, l, 'understory'] * model.p_dry_volume_t[f, d, p6, z, 'understory'] * model.p_sb_selectivity_zp6[z,p6]
                         for d in model.s_dry_groups) == 0
        else:
            return pe.Constraint.Skip
    model.con_link_understory_saltbush_consumption = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types,
                                                                   model.s_feed_periods,model.s_feed_pools, model.s_lmus,rule=link_us_sb,
                                                                   doc='link between the consumption of understory and saltbush')


def f_con_link_pasture_supplement_consumption(model,nv):
    '''
    Constrains the consumption of paddock feed with supplement if the animal is not in confinement.
    This is to represent the fact that livestock will still eat pasture if they're being fed supplement.

    Note: this constraint can make the model infeasible for n11 because the optimal fs cant be met without supplement.
    To fix this run n33. This may still be infeasible due to sires who are always n33. But sires nut can be changed in sinp
    or sires can be removed from ME and vol in stockpyomo.
    '''
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.
    l_f = list(model.s_feed_pools)
    def link_pas_sup(model,q,s,z,p6,f):
        f_idx = l_f.index(f)
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p6z[p6,z]) and nv_is_not_confinement_f[f_idx] and uinp.supfeed['i_sup_selectivity_included']:
            return - (paspy.f_pas_me2(model,q,s,p6,f,z) + stubpy.f_cropresidue_me(model,q,s,p6,f,z)) * model.p_max_sup_selectivity[p6,z] \
                   + suppy.f_sup_me(model,q,s,p6,f,z) * (1-model.p_max_sup_selectivity[p6,z]) <= 0
        else:
            return pe.Constraint.Skip
    model.con_link_pasture_supplement_consumption = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types,
                                                                   model.s_feed_periods,model.s_feed_pools,rule=link_pas_sup,
                                                                   doc='link between the consumption of paddock feed and supplement when trail feeding.')


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
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p6z[p6,z]):
            return -paspy.f_pas_me(model,q,s,p6,f,z) - paspy.f_nappas_me(model,q,s,p6,f,z) - suppy.f_sup_me(model,q,s,p6,f,z) \
                   - stubpy.f_cropresidue_me(model,q,s,p6,f,z) - cgzpy.f_grazecrop_me(model,q,s,p6,f,z) - slppy.f_saltbush_me(model,q,s,z,p6,f) \
                   + stkpy.f_stock_me(model,q,s,p6,f,z) - mvf.f_mvf_me(model,q,s,p6,f) <= 0
        else:
            return pe.Constraint.Skip
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
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p6z[p6,z]):
            return paspy.f_pas_vol(model,q,s,p6,f,z) + suppy.f_sup_vol(model,q,s,p6,f,z) + stubpy.f_cropresidue_vol(model,q,s,p6,f,z) \
                   + cgzpy.f_grazecrop_vol(model,q,s,p6,f,z) + slppy.f_saltbush_vol(model,q,s,z,p6,f) \
                   - stkpy.f_stock_pi(model,q,s,p6,f,z) \
                   + mvf.f_mvf_vol(model,q,s,p6,f) <= 0
        else:
            return pe.Constraint.Skip
    model.con_vol = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods,model.s_feed_pools,model.s_season_types,rule=vol,
                                  doc='constraint between me available and consumed')


def f_con_cashflow(model):
    '''
    Tallies all cashflow in each period and transfers to the next period. Season periods exist so that a transfer can
    exist between parent and child seasons.
    '''
    def cashflow(model,q,s,c1,p7,z9):
        p7_start = list(model.s_season_periods)[0]
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return ((-f1_grain_income(model,q,s,p7,z9,c1) + phspy.f_rotation_cost(model,q,s,p7,z9) + labpy.f_labour_cost(model,q,s,p7,z9)
                    + macpy.f_mach_cost(model,q,s,p7,z9) + suppy.f_sup_feeding_cost(model,q,s,p7,z9) + model.p_overhead_cost[p7,z9] + slppy.f_saltbush_cost(model,q,s,z9,p7)
                    - stkpy.f_stock_cashflow(model,q,s,p7,z9,c1) - treepy.f_tree_cashflow(model,p7,z9)
                    - model.v_debit[q,s,c1,p7,z9] + model.v_credit[q,s,c1,p7,z9])
                    + sum((model.v_debit[q,s,c1,p7_prev,z8] - model.v_credit[q,s,c1,p7_prev,z8]) * model.p_parentz_provwithin_season[p7_prev,z8,z9] * ((p7!=p7_start)*1)  #end cashflow doesnot provide start cashflow else unbounded.
                          for z8 in model.s_season_types)) <= 0
        else:
            return pe.Constraint.Skip
    model.con_cashflow_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_c1, model.s_season_periods, model.s_season_types,rule=cashflow,
                                                doc='transfer of cashflow between periods')

def f1_start_asset_value(model,q,s,p7,z):
    '''total value of assets at the start of the sequence (q[0], p7[0]).'''
    p7_start = list(model.s_season_periods)[0]
    q_start = list(model.s_sequence_year)[0]
    if q==q_start and p7==p7_start:
        return macpy.f_mach_asset(model,p7) + (-model.v_tradevalue[q, s, p7, z]) #tradevalue in p7[0] is the opening sheep assets
    else:
        return 0

def f_con_totalcap_within(model):
    '''
    Tallies total capital and transfers to the next period. Cashflow periods exist so that a transfer can
    exist between parent and child seasons.

    Total capital is a combination of assets and working capital. Working capital is the sum of the
    expenses minus any income, since the previous 'main' income (e.g. harvest or shearing).
    This constraint exists so the model user can examine how the farm is structured under different levels of finance.
    The default is to allow a large amount of finance so that this constraint it not impacting the solution.

    Start asset value is included to ensure that retaining sheep from the end of last year
    until the start of the current cashflow period doesn't reduce wc (on farm this does not work because there is no
    concept of start and end of cashflow like there is in the model). If start asset value was not included the model
    shift strategical off spring sales that would usually occur at shearing until the start of next season which would
    reduce wc because sales after main shearing offset wc costs (this would be a problem in both SE and DSP).

    Note: trade value is not included because it is a valid tactic to sell more animals in a poor season and then buy 
    them back after peak debt or in SQ start the following year understocked.
    
     '''
    def total_cap_within(model,q,s,c0,p7,z9):
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        if pe.value(model.p_mask_childz_within_season[p7,z9]) and pe.value(model.p_wyear_inc_qs[q,s]) and uinp.finance['i_working_capital_constraint_included']:
            return (-f1_grain_wc(model,q,s,c0,p7,z9) + phspy.f_rotation_wc(model,q,s,c0,p7,z9) + labpy.f_labour_wc(model,q,s,c0,p7,z9) + slppy.f_saltbush_wc(model,q,s,z9,c0,p7)
                    + macpy.f_mach_wc(model,q,s,c0,p7,z9) + suppy.f_sup_wc(model,q,s,c0,p7,z9) + model.p_overhead_wc[c0,p7,z9]
                    - stkpy.f_stock_wc(model,q,s,c0,p7,z9) - treepy.f_tree_wc(model,c0,p7,z9) + f1_start_asset_value(model,q,s,p7,z9)
                    - model.v_wc_debit[q,s,c0,p7,z9]
                    + model.v_wc_credit[q,s,c0,p7,z9]
                    + sum((model.v_wc_debit[q,s,c0,p7_prev,z8] - model.v_wc_credit[q,s,c0,p7_prev,z8]) #end working capital doesnot provide start else unbounded constraint.
                          * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                         for z8 in model.s_season_types)) <= 0
        else:
            return pe.Constraint.Skip
    model.con_totalcap_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types,rule=total_cap_within,
                                       doc='working capital transfer within year')


def f_con_totalcap_between(model):
    '''
    Tallies total capital and transfers to the next period. Cashflow periods exist so that a transfer can
    exist between parent and child seasons.

    Cashflow at the end of the previous yr becomes the starting balance for working capital (cashflow broadcasts
    to both c0 slices). This only happens between years in the sequence. End to start doesn't carry over. This is
    because it was decided that the working capital constraint is more useful if there is no starting balance at
    the start of the sequence. This means an expensive strategy with a high reward can be bounded using wc (if the end
    cashflow became the start then a high expense high income strategy would not trigger the constraint).

    '''
    def total_cap_between(model,q,s9,c0,p7,z9):
        p7_prev = list(model.s_season_periods)[list(model.s_season_periods).index(p7) - 1]  # previous cashperiod - have to convert to a list first because indexing of an ordered set starts at 1
        l_q = list(model.s_sequence_year_between_con)
        ###adjust q_prev for multi-period model
        if sinp.structuralsa['model_is_MP']:
            ####yr0 is SE so q_prev is q. Note dont need to use lp_var in this between constraint because it is not a production constraint.
            if q == l_q[0]:
                q_prev = q
            ####the final year is provided by both the previous year and itself (the final year is in equilibrium). Therefore the final year needs two constraints. This is achieved by making the q set 1 year longer than the modeled period (len_MP + 1). Then adjusting q and q_prev for the final q so that the final year is also in equilibrium.
            elif q == l_q[-1]:
                q = l_q[l_q.index(q) - 1]
                q_prev = q
            else:
                q_prev = l_q[l_q.index(q) - 1]
        else:
            q_prev = l_q[l_q.index(q) - 1]

        if pe.value(model.p_mask_childz_between_season[p7,z9]) and pe.value(model.p_wyear_inc_qs[q,s9]) and uinp.finance['i_working_capital_constraint_included']:
            return (-f1_grain_wc(model,q,s9,c0,p7,z9) + phspy.f_rotation_wc(model,q,s9,c0,p7,z9) + labpy.f_labour_wc(model,q,s9,c0,p7,z9) + slppy.f_saltbush_wc(model,q,s9,z9,c0,p7)
                    + macpy.f_mach_wc(model,q,s9,c0,p7,z9) + suppy.f_sup_wc(model,q,s9,c0,p7,z9) + model.p_overhead_wc[c0,p7,z9]
                    - stkpy.f_stock_wc(model,q,s9,c0,p7,z9) - treepy.f_tree_wc(model,c0,p7,z9) + f1_start_asset_value(model,q,s9,p7,z9)
                    - model.v_wc_debit[q,s9,c0,p7,z9]
                    + model.v_wc_credit[q,s9,c0,p7,z9]
                    + sum(sum((model.v_debit[q_prev,s8,c1,p7_prev,z8] - model.v_credit[q_prev,s8,c1,p7_prev,z8]) * model.p_prob_c1[c1]
                              for c1 in model.s_c1)#end cashflow become start wc (only within a sequence).
                          * model.p_parentz_provbetween_season[p7_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                        # note - there is not end to start transfer because there is no opening balance.
                          for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)) <= 0
        else:
            return pe.Constraint.Skip
    model.con_totalcap_between = pe.Constraint(model.s_sequence_year_between_con, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types,rule=total_cap_between,
                                       doc='working capital transfer between years')


def f_con_dep(model):
    '''Tallies the depreciation of capital, which is then passed to the objective.'''
    def dep(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return (macpy.f_seeding_harv_dep(model,q,s,p7,z9) + phspy.f_rotation_depn(model,q,s,p7,z9)
                    + suppy.f_sup_dep(model,q,s,p7,z9) - model.v_dep[q,s,p7,z9]
                    + sum(model.v_dep[q,s,p7_prev,z8] * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                          for z8 in model.s_season_types) * ((p7!=p7_start)*1) #end doesn't carry over
                    <= 0)
        else:
            return pe.Constraint.Skip
    model.con_dep = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=dep,
                                  doc='tallies depreciation from all activities so it can be transferred to objective')


def f_con_asset(model):
    '''Tallies the total asset value to ensure that there is a minimum ROI on farm assets that are selected.
    The asset value multiplied by opportunity cost on capital is then passed to the objective.
    '''
    def asset_cost(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return (suppy.f_sup_asset(model,q,s,p7,z9) + macpy.f_mach_asset(model,p7) + stkpy.f_stock_asset(model,q,s,p7,z9)) * uinp.finance['opportunity_cost_capital'] \
                   - model.v_asset_cost[q,s,p7,z9] \
                   + sum(model.v_asset_cost[q,s,p7_prev,z8] * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                         for z8 in model.s_season_types) * ((p7!=p7_start)*1) <= 0 #end doesn't carry over
        else:
            return pe.Constraint.Skip
    model.con_asset = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=asset_cost,
                                    doc='tallies asset from all activities so it can be transferred to objective to represent ROA')


def f_con_minroe(model):
    '''Tallies the total expenditure to ensure that there is a minimum ROI on cash expenditure.'''
    #todo Does/should minroe include 1. fixed costs. Because minroe is not tallying with the total expenses in the pnl report
    def minroe(model,q,s,p7,z9):
        l_p7 = list(model.s_season_periods)
        p7_prev = l_p7[l_p7.index(p7) - 1] #need the activity level from last period
        p7_start = l_p7[0]
        if pe.value(model.p_wyear_inc_qs[q, s]) and pe.value(model.p_mask_season_p7z[p7,z9]):
            return ((phspy.f_rotation_cost(model,q,s,p7,z9) + labpy.f_labour_cost(model,q,s,p7,z9) + macpy.f_mach_cost(model,q,s,p7,z9)
                     + suppy.f_sup_feeding_cost(model,q,s,p7,z9) + stkpy.f_stock_cost(model,q,s,p7,z9) + slppy.f_saltbush_cost(model,q,s,z9,p7)
                     + f1_sup_minroe(model,q,s,p7,z9))
                    * fin.f1_min_roe()
                    - model.v_minroe[q,s,p7,z9]
                    + sum(model.v_minroe[q,s,p7_prev,z8] * model.p_parentz_provwithin_season[p7_prev,z8,z9]
                          for z8 in model.s_season_types) * ((p7 != p7_start)*1)) <= 0  # end doesn't carry over
        else:
            return pe.Constraint.Skip
    model.con_minroe = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types,rule=minroe,
                                     doc='tallies total expenditure to ensure minimum roe is met')


def f_objective(model):
    r'''
    The objective of the model is to maximise expected utility. In the case of
    risk neutrality, expected utility is equivalent to expected profit, meaning
    that the optimal farm management plan is that which maximises farm profit.
    In the case of risk aversion, utility increases at a diminishing rate as
    profit increases. Thus, when farm profit is low, an extra dollar of profit
    provides more utility than when farm profit is high. This means a risk
    averse farmer aims to reduce profit variation (i.e. increase profit in poor
    years at the cost of reduced profit in the good years). For example, if the
    crop and stock enterprise on the modelled farm are similar but grain prices
    are more volatile, then risk aversion will shift resources towards the
    stock enterprise to reduce risk (profit variation).

    The expected return used to calculate utility includes the net cash flow for
    a given price and weather scenario, minus a cost to represent a minimum
    return on operating costs incurred (MINROE), minus the cost of depreciation,
    and minus the opportunity cost on the farm assets (total value of all
    assets times the discount rate, to ensure that the assets generate a minimum
    ROA). MINROE and asset opportunity costs are discussed in more detail in the
    finance section, and their inclusion is controlled by the user.

    Constant absolute risk-aversion (CARA) and constant relative risk-aversion
    (CRRA) are two well known utility functions. Both have been previously used
    in stochastic farm modelling (see :cite:p:`KINGWELL1994, kingwell1996`) and
    both methods are included in AFO. A third expo-power utility function is
    also included in AFO. This provides a flexible middle ground between
    CARA and CRRA, overcoming many of their limitations and allowing risk aversion to change with the scale of returns.
    The expo-power function we are using is based on :cite:p:`holt2002`
    variant of the original function introduced by :cite:p:`saha1993`.
    All three utility functions are nonlinear. To accommodate this in AFO, a
    piecewise-linear approximation is used, with 13 linear segments applied over
    the relevant range of profit values.

    As an alternative to expected-utility maximisation, AFO also allows a
    mean–CVaR (Conditional Value at Risk) objective. In this case, the model
    chooses a farm plan by balancing expected profit against downside (tail)
    risk in low-profit states.

    **CARA is a negative exponential curve:**

    :math:`U = 1-exp(-a*x)`

    where :math:`U` is utility, :math:`a` is the Pratt–Arrow coefficient of
    absolute risk aversion, and :math:`x` is the return to management and
    capital. The coefficient :math:`a` is a user input that controls the level
    of risk aversion. Earlier farm modelling work by :cite:p:`KINGWELL1994` used values of 0.000003 and
    0.000005 to represent moderate and high levels of risk aversion
    respectively, based on profit levels typical at the time.

    To ensure comparability under contemporary farm conditions—where average profit levels are substantially
    higher—we calibrate CARA parameters based on relative risk aversion evaluated at a representative
    modern profit level. We target coefficients of relative risk aversion consistent with Kingwell’s classifications:

    - Moderate risk aversion: 0.78
    - High risk aversion: 1.30

    Given the relationship :math:`R = a x`, the corresponding CARA parameter is
    computed as: :math:`a = R / x`. where :math:`x` is the representative annual farm profit. This
    ensures updated parameters express the *same behavioural strength* of risk
    aversion as earlier work, but scaled to current economic settings.

    **CRRA is a power utility function:**

    :math:`U = W^(1-R) / (1-R)`

    where :math:`U` is utility, :math:`W` is terminal wealth, and :math:`R`
    is the relative risk aversion coefficient. The CRRA method affects
    risk aversion through wealth, meaning that MINROE and asset opportunity
    costs influence the implied level of risk aversion. :cite:p:`kingwell1996`
    used values of :math:`R` between 0.1 and 3.0 to represent low to high
    risk aversion. A key limitation is that CRRA cannot be applied when the
    terminal wealth level becomes negative.

    **Expo-power Utility:**

    :math: `U(x) = (1 - exp(-α x^{1-r})) / α`

    where:

    - :math:`x` is the profit in each weather–price state.
    - :math:`r` is a shape parameter controlling how risk aversion
      changes with the size of profit. For `0 < r < 1`, the expo-power utility function exhibits decreasing absolute
      risk aversion (DARA) and increasing relative risk aversion (IRRA).
    - :math:`a` is a scale parameter determining the overall level of risk
      aversion. It is calibrated so that the coefficient of relative risk aversion at
      the reference profit level matches the levels used in the CARA formulation.

    The curvature parameter :math:`r` for the expo-power utility is based on
    empirical estimates from farmer risk-preference studies.
    Results using agricultural lotteries yield :math:`r = 0.641`
    (:cite:p:`loduca2024`). Comparable estimates can be derived from earlier
    studies using equivalent expo-power formulations, including
    :math:`r = 0.635` from :cite:p:`saha1994` and :math:`r = 0.643` from
    :cite:p:`bocqueho2014`. These values cluster tightly around 0.64, and broader
    evidence from European farming systems suggests a plausible range between
    0.6 and 0.8 (:cite:p:`rommel2022`). Accordingly, we adopt
    :math:`r = 0.64` as a representative baseline value.
    This value reflects moderate curvature and means relative risk aversion increases
    with income.

    The parameter :math:`a` is not taken directly from the literature but is
    calibrated so that the resulting coefficient of relative risk aversion at the
    reference profit level (AUD 500,000) matches the same “moderate” and “high”
    risk aversion levels used for the CARA specification (0.78 and 1.30), ensuring
    comparability across utility forms.

    In this formulation, AFO treats :math:`x` (profit) as the “thing at risk,”
    rather than full terminal wealth. This simplifies both interpretation and
    implementation, and aligns with the existing CARA-based specification.
    Wealth can be incorporated in future developments by replacing
    :math:`x` with :math:`W = W_0 + x`.

    **Mean–CVaR Tail-Risk Method:**

    Let :math:`W` denote terminal profit in each weather–price state, and define
    loss as :math:`L = -W`. For a confidence level :math:`\alpha \in (0,1)`
    (for example, :math:`\alpha = 0.95`) and a risk-aversion coefficient
    :math:`\lambda > 0`, AFO uses the following CVaR definition:

    .. math::

       \text{CVaR}_\alpha(L)
       = \eta + \frac{1}{1-\alpha}\,\mathbb{E}\big[(L - \eta)^+\big],

    where :math:`\eta` is an endogenous VaR-like threshold and
    :math:`(L - \eta)^+` denotes the positive part of :math:`L - \eta`.
    The objective in this mode is

    .. math::

       \max \ \mathbb{E}[W] \;-\; \lambda \,\text{CVaR}_\alpha(L),

    so that higher values of :math:`\lambda` place more weight on limiting extreme
    losses in bad years. This mean–CVaR specification does not rely on a utility
    transformation of profit. However, the same terminal profit measure is used
    as in the CARA/CRRA/expo-power cases, so results remain comparable across
    risk specifications.

    **Calibration and scaling of terminal wealth points**

    The piecewise-linear approximation to the utility function is constructed over
    a grid of terminal profit values given by ``tw_points``. These breakpoints
    should be chosen to cover the range of profit outcomes implied by the price–
    weather states in the model, with some buffer for unusually good or bad years.
    Changing the minimum or maximum values of ``tw_points`` does not alter the
    underlying risk-aversion parameters (e.g. CARA :math:`a`, CRRA :math:`R`,
    expo-power :math:`r` and :math:`a`), but it does affect how accurately the
    utility curve is approximated in the extreme tails.

    For CRRA and expo-power, AFO applies utility to total wealth,

    .. math::

       W = W_0 + x,

    where :math:`x` is terminal profit and :math:`W_0` is an initial wealth level
    specified by the user (``i_initial_wealth``). This shift ensures that wealth
    remains positive even when profit is negative. However, it also means that the
    effective scale at which risk aversion is evaluated depends on both profit and
    initial wealth.

    In the CARA and expo-power specifications, risk-aversion coefficients are
    calibrated at a representative profit level :math:`x_\text{ref}`. For example,
    the expo-power coefficient :math:`a` is chosen so that the coefficient of
    relative risk aversion at :math:`x_\text{ref} = 650{,}000` matches Kingwell's
    "moderate" or "high" risk-aversion benchmarks. When an initial wealth
    :math:`W_0` is introduced, the relevant reference argument becomes

    .. math::

       W_\text{ref} = W_0 + x_\text{ref}.

    If :math:`W_0` or :math:`x_\text{ref}` is changed substantially, the original
    calibration of :math:`a` will no longer imply the same level of relative risk
    aversion at the new reference wealth. In that case, :math:`a` should be
    recomputed using the expo-power relative risk aversion expression

    .. math::

       R(W) = r + a (1 - r)\, W^{\,1-r},

    by solving for :math:`a` at the desired target :math:`R(W_\text{ref})`. An
    analogous consideration applies to the CARA coefficient :math:`a`, where the
    calibration :math:`a = R / x_\text{ref}` assumes a particular reference profit
    scale.

    For CRRA, the parameter :math:`R` is itself the coefficient of relative risk
    aversion and does not depend on wealth. Introducing :math:`W_0` therefore does
    not change :math:`R` in the Arrow–Pratt sense, although it does change the
    absolute sensitivity to a given dollar change in profit relative to total
    wealth. Users who alter ``tw_points``, initial wealth or the representative
    profit level should be aware of these scale effects and, if necessary,
    recalibrate the CARA and expo-power coefficients to preserve their intended
    behavioural interpretation.


    '''

    ##terminal wealth transfer constraint - combine cashflow with depreciation, MINROE and asset value
    p7_end = list(model.s_season_periods)[-1]
    def terminal_wealth(model,q,s,z,c1):
        variables = model.component_objects(pe.Var,active=True) #this has to get called for each constraint (something to do with generator objects)
        if pe.value(model.p_wyear_inc_qs[q,s]):
            return (model.v_terminal_wealth[q,s,z,c1] - model.v_credit[q,s,c1,p7_end,z] + model.v_debit[q,s,c1,p7_end,z] # have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
                    + model.v_dep[q,s,p7_end,z] + model.v_minroe[q,s,p7_end,z] + model.v_asset_cost[q,s,p7_end,z]
                    - model.v_tradevalue[q, s, p7_end, z]
                    # all variables with positive bounds (ie variables that can be negative e.g. terminal_wealth are excluded) put a small neg number into objective. This stop cplex selecting variables that don't contribute to the objective (cplex selects variables to remove slack on constraints).
                    + 0.00001 * sum(sum(v[idx] for idx in v if idx[0]==q) for v in variables #only sum for given q (this is required for the MP model otherwise the variable bnd on the first year doesnt work because len q affect v_terminal wealth).
                                       if v._rule_bounds._initializer.val[0] is not None and v._rule_bounds._initializer.val[0]>=0)) ==0 #had to make this == constraint so that the model soved correctly when risk was being used (otherwise there was a few cases where there was slack here ie the solver was not finding optimum)
        else:                                                                                                  #note; _rule_bounds.val[0] is the lower bound of each variable
            return pe.Constraint.Skip
    model.con_terminal_wealth = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1, rule=terminal_wealth,
                                     doc='tallies up terminal wealth so it can be transferred to the utility function.')

    ##terminal wealth at each segment - smaller steps at low profit when utility function is steep
    ###need to ensure lowest segement capture the lowest terminal wealth state - this can be checked by running model with price variation on but no risk.
    tw_points = (
            list(range(-500000, -200000, 50000)) +
            list(range(-200000, 500000, 100000)) +
            list(range(500000, 1200000, 200000)))
    tw_points.append(2000000) #add a high number to end to handle if profit is very high. Note utility will be linear for any values in this last segment, thus shouldn't be common to have profit in this seg
    tw_points = sorted(set(tw_points))  # ensure sorted and unique
    tw_points = np.array(tw_points)

    # Default: risk-neutral linear utility (used also for CVaR case)
    utility_u = tw_points
    ##CARA/CRRA utility function
    if uinp.general['i_utility_method']==1: #CARA
        a=uinp.general['i_cara_risk_coef']
        utility_u = (1-np.exp(-a*tw_points))*100 #need to multiply by 100 so the solver works (not 100% sure but seems to be due to the small change in obj making it hard to solve)
    ##CRRA utility function
    elif uinp.general['i_utility_method']==2: #CRRA
        Rr = uinp.general['i_crra_risk_coef']
        initial_welth = uinp.general['i_initial_wealth']
        t_tw_points = tw_points+initial_welth
        utility_u = t_tw_points**(1-Rr) / (1-Rr)
    ##Expo-power Utility
    elif uinp.general['i_utility_method']==3:
        r = uinp.general['i_r_expo_risk_coef']
        a = uinp.general['i_a_expo_risk_coef']
        initial_welth = uinp.general['i_initial_wealth']
        t_tw_points = tw_points+initial_welth
        utility_u = (1 - np.exp(-a * t_tw_points**(1-r))) / a

    keys_u = np.array(['u%s' % i for i in range(len(tw_points))])
    p_tw_points = dict(zip(keys_u, tw_points))
    p_utility = dict(zip(keys_u, utility_u))
    model.s_utility_points = pe.Set(initialize=keys_u, doc='utility segments')
    model.v_utility_points = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1, model.s_utility_points, bounds = (0, None), doc = 'propn of utility from each segment')
    model.p_tw_points = pe.Param(model.s_utility_points, initialize=p_tw_points, default = 0.0, doc='terminal wealth at the beginning of each segment')
    model.p_utility = pe.Param(model.s_utility_points, initialize=p_utility, default = 0.0, doc='utility provided by each level of terminal wealth')

    def terminal_wealth_transfer(model,q,s,z,c1):
        if pe.value(model.p_wyear_inc_qs[q,s]):
            return -model.v_terminal_wealth[q,s,z,c1] + sum(model.v_utility_points[q,s,z,c1,u] * model.p_tw_points[u]
                                                            for u in model.s_utility_points) <=0
        else:
            return pe.Constraint.Skip
    model.con_terminal_wealth_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1, rule=terminal_wealth_transfer,
                                     doc='transfers terminal wealth to utility')

    def utility_propn(model,q,s,z,c1):
        if pe.value(model.p_wyear_inc_qs[q,s]):
            return sum(model.v_utility_points[q,s,z,c1,u] for u in model.s_utility_points) ==1
        else:
            return pe.Constraint.Skip
    model.con_utility_segment_propn = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1, rule=utility_propn,
                                     doc='ensures terminal wealth points tally to 1. Required to stop utility function being unbounded.')

    if uinp.general['i_utility_method'] <=3:
        # Default objective (risk-neutral or CARA/CRRA/Expo)
        return sum(sum(model.v_utility_points[q,s,z,c1,u] * model.p_utility[u] for u in model.s_utility_points)
                   * model.p_season_prob_qsz[q,s,z] * model.p_prob_c1[c1] * model.p_discount_factor_q[q]
                   for q in model.s_sequence_year for s in model.s_sequence for c1 in model.s_c1 for z in model.s_season_types
                   if pe.value(model.p_wyear_inc_qs[q,s]))

    # CVaR risk function (only used if method 4 selected) ---
    if uinp.general['i_utility_method'] == 4:
        alpha_tail = uinp.general['i_tail_alpha']   # e.g. 0.95
        lambda_tail = uinp.general['i_tail_lambda'] # risk aversion to tail

        # VaR on loss (single global)
        model.v_eta = pe.Var(bounds=(None, None), doc='VaR level on loss')

        # Shortfall variables per state
        model.v_shortfall = pe.Var(
            model.s_sequence_year, model.s_sequence,
            model.s_season_types, model.s_c1,
            bounds=(0, None),
            doc='(loss - eta)+ per state'
        )

        def shortfall_rule(model, q, s, z, c1):
            if pe.value(model.p_wyear_inc_qs[q, s]):
                # Define loss as negative terminal wealth:
                W = model.v_terminal_wealth[q, s, z, c1]
                L = -W
                return model.v_shortfall[q, s, z, c1] >= L - model.v_eta
            else:
                return pe.Constraint.Skip

        model.con_shortfall = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1,
            rule=shortfall_rule, doc='defines shortfall for CVaR')

        # helper inlined: state probability (including discount)
        def state_prob(q, s, z, c1):
            return (model.p_season_prob_qsz[q, s, z] * model.p_prob_c1[c1] * model.p_discount_factor_q[q])

        # Expected wealth
        exp_wealth = sum(
            model.v_terminal_wealth[q, s, z, c1] * state_prob(q, s, z, c1)
            for q in model.s_sequence_year for s in model.s_sequence
            for z in model.s_season_types for c1 in model.s_c1
            if pe.value(model.p_wyear_inc_qs[q, s])
        )

        # Expected shortfall
        exp_shortfall = sum(
            model.v_shortfall[q, s, z, c1] * state_prob(q, s, z, c1)
            for q in model.s_sequence_year for s in model.s_sequence
            for z in model.s_season_types for c1 in model.s_c1
            if pe.value(model.p_wyear_inc_qs[q, s])
        )

        CVaR = model.v_eta + (1.0 / (1.0 - alpha_tail)) * exp_shortfall

        # Objective: mean - lambda * CVaR
        return exp_wealth - lambda_tail * CVaR


def f_con_MP(model, MP_lp_vars):
    ''''
    These constraints are to bound the variable levels in the first node of the first year in the MP model. This
    is when farm conditions have changed but management has not yet reacted. Therefore, key production management is
    bound to be as per normal (based on the Initial MP run).
    '''
    len_p7 = len(model.s_season_periods)

    def MP_rotation_q0_lower(model,q,s,p7,z,r,l):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        ## give 1% flex on the bnd to allow for any rounding.
        if q == 'q0' and p7 == 'zm0' and len_p7>1:
            return (model.v_phase_area[q,s,p7,z,r,l] <=
                    MP_lp_vars[str('v_phase_area')]['q0',s,p7,z,r,l] * 1.01)
        else:
            return pe.Constraint.Skip
    model.con_MP_rotation_q0_lower = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                   model.s_season_types, model.s_phases, model.s_lmus, rule=MP_rotation_q0_lower,
                                            doc='phase area bnd for node 1 in the MP model')

    def MP_rotation_q0_upper(model,q,s,p7,z,r,l):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        ## give 1% flex on the bnd to allow for any rounding.
        if q == 'q0' and p7 == 'zm0' and len_p7>1:
            return (model.v_phase_area[q,s,p7,z,r,l] >=
                    MP_lp_vars[str('v_phase_area')]['q0',s,p7,z,r,l] * 0.99)
        else:
            return pe.Constraint.Skip
    model.con_MP_rotation_q0_upper = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_season_periods,
                                                   model.s_season_types, model.s_phases, model.s_lmus, rule=MP_rotation_q0_upper,
                                            doc='phase area bnd for node 1 in the MP model')

    def MP_dams_sale_lower(model, q, s, k2, t1, v1, a, z, i, y1, g1):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        ## bnd sale and not mated animals so the model can't choose to not mate then sell at the second node.
        ## therefore give 5% flex on the bnd to allow for mort.
        if q == 'q0' and len_p7>1 and (t1=='t0' or t1=='t1' or k2=='NM-0') and model.p_dvp_is_node1_vzg1[v1, z, g1]:
            return (sum(model.v_dams[q, s, k2, t1, v1, a, n1, w1, z, i, y1, g1]
                       for n1 in model.s_nut_dams for w1 in model.s_lw_dams) <=
                    sum(MP_lp_vars[str('v_dams')]['q0', s, k2, t1, v1, a, n1, w1, z, i, y1, g1]
                        for n1 in model.s_nut_dams for w1 in model.s_lw_dams)*1.05)
        else:
            return pe.Constraint.Skip
    model.con_MP_dams_sale_lower = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams,
                                            model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol,
                                            model.s_gen_merit_dams, model.s_groups_dams, rule=MP_dams_sale_lower,
                                            doc='dams numbers bnd for node 1 in the MP model')

    def MP_dams_sale_upper(model, q, s, k2, t1, v1, a, z, i, y1, g1):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        ## bnd sale and not mated animals so the model can't choose to not mate then sell at the second node.
        ## therefore give 5% flex on the bnd to allow for mort.
        if q == 'q0' and len_p7>1 and (t1=='t0' or t1=='t1' or k2=='NM-0') and model.p_dvp_is_node1_vzg1[v1, z, g1]:
            return (sum(model.v_dams[q, s, k2, t1, v1, a, n1, w1, z, i, y1, g1]
                       for n1 in model.s_nut_dams for w1 in model.s_lw_dams) >=
                    sum(MP_lp_vars[str('v_dams')]['q0', s, k2, t1, v1, a, n1, w1, z, i, y1, g1]
                        for n1 in model.s_nut_dams for w1 in model.s_lw_dams)*0.95)
        else:
           return pe.Constraint.Skip
    model.con_MP_dams_sale_upper = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k2_birth_dams,
                                            model.s_sale_dams, model.s_dvp_dams, model.s_wean_times, model.s_season_types, model.s_tol,
                                            model.s_gen_merit_dams, model.s_groups_dams, rule=MP_dams_sale_upper,
                                            doc='dams numbers bnd for node 1 in the MP model')

    ##need to have upper and lower bnd here for mortality. If we force lighter animal it might die more therefore cant sell exactly the same as normal.
    def MP_offs_sale_lower(model, q, s, k3, k5, t3, v3, z, i, a, x, y3, g3):
        ##bnd the first node in q[1] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        if q == 'q0' and len_p7>1 and t3!='t0' and model.p_dvp_is_node1_k3vzxg3[k3,v3,z,x,g3]:
            return (sum(model.v_offs[q, s, k3, k5, t3, v3, n3, w3, z, i, a, x, y3, g3]
                       for n3 in model.s_nut_offs for w3 in model.s_lw_offs) <=
                    sum(MP_lp_vars[str('v_offs')]['q0', s, k3, k5, t3, v3, n3, w3, z, i, a, x, y3, g3]
                       for n3 in model.s_nut_offs for w3 in model.s_lw_offs)*1.01)
        else:
            return pe.Constraint.Skip
    model.con_MP_offs_sale_lower = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs,
                                            model.s_k5_birth_offs, model.s_sale_offs,
                                            model.s_dvp_offs, model.s_season_types, model.s_tol, model.s_wean_times,
                                            model.s_gender, model.s_gen_merit_offs,
                                            model.s_groups_offs, rule=MP_offs_sale_lower,
                                            doc='offs numbers bnd for node 1 in the MP model')

    def MP_offs_sale_upper(model, q, s, k3, k5, t3, v3, z, i, a, x, y3, g3):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        if q == 'q0' and len_p7>1 and t3!='t0' and model.p_dvp_is_node1_k3vzxg3[k3,v3,z,x,g3]:
            return (sum(model.v_offs[q, s, k3, k5, t3, v3, n3, w3, z, i, a, x, y3, g3]
                       for n3 in model.s_nut_offs for w3 in model.s_lw_offs) >=
                    sum(MP_lp_vars[str('v_offs')]['q0', s, k3, k5, t3, v3, n3, w3, z, i, a, x, y3, g3]
                       for n3 in model.s_nut_offs for w3 in model.s_lw_offs)*0.99)
        else:
            return pe.Constraint.Skip
    model.con_MP_offs_sale_upper = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs,
                                            model.s_k5_birth_offs, model.s_sale_offs,
                                            model.s_dvp_offs, model.s_season_types, model.s_tol, model.s_wean_times,
                                            model.s_gender, model.s_gen_merit_offs,
                                            model.s_groups_offs, rule=MP_offs_sale_upper,
                                            doc='offs numbers bnd for node 1 in the MP model')

    def MP_prog_bound(model, q, s, k3, k5, t2, z, i, a, x, g2):
        ##bnd the first node in q[0] (this is when farm conditions have changed but management has not changed) (unless only one p7 period because that means the management can change in p7[0])
        ##doesnt need upper and lower because there is no LW bnd for prog.
        if q == 'q0' and len_p7>1 and t2=='t0' and model.p_dvp_is_node1_k3zg2[k3,z,g2]:
            return (sum(model.v_prog[q,s,k3, k5, t2, w2, z, i, a, x, g2] for w2 in model.s_lw_prog) ==
                    sum(MP_lp_vars[str('v_prog')]['q0',s,k3, k5, t2, w2, z, i, a, x, g2] for w2 in model.s_lw_prog))
        else:
            return pe.Constraint.Skip
    model.con_MP_prog_bound = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_k3_damage_offs,
                                            model.s_k5_birth_offs, model.s_sale_prog,
                                            model.s_season_types, model.s_tol, model.s_wean_times,
                                            model.s_gender, model.s_groups_prog, rule=MP_prog_bound,
                                            doc='prog numbers bnd for node 1 in the MP model')


    ##origional version (it worked but model seemed to sometime not solve because to degenerate):
    # #Unfortunately there was a case when == constraint made the model infeasible (maybe a rounding error). So now there is an upper and lower bnd on each variable.
    # i = 0
    # list_v = []
    # list_s = []
    # list_idx = []
    # list_bnd = []
    # for v in model.component_objects(pe.Var, active=True):
    #     for s in v:
    #         # print(v,s)
    #         list_idx.append(i)
    #         list_v.append(v)
    #         list_s.append(s)
    #         ###make it so that all q in mp model look at q0 from MP_lp_vars (because lp vars are from SE model that only has q[0])
    #         s_adjusted = ("q0",) + s[1:]
    #         list_bnd.append(MP_lp_vars[str(v)][s_adjusted])
    #         i = i + 1
    #
    # def MP_upper(model, idx):
    #     v = list_v[idx]
    #     s = list_s[idx]
    #     q = s[0]
    #     if q == 'q0' and (str(v)=='v_dams'  or str(v)=='v_offs' or str(v)=='v_phase_area' or
    #                                           str(v)=='v_greenpas_ha' or str(v)=='v_drypas_transfer' or
    #                                           str(v)=='v_tonnes_sb_transfer' or str(v)=='v_stub_transfer'): #only need to constrain variables that transfer from q_prev
    #         bnd = list_bnd[idx]
    #         return v[s] >= bnd * 0.99 #minus 1 to give the model a tiny bit of wriggle room for rounding issues
    #     ###bnd rotation variables in p7[0] (unless only one p7 period because that means the management can change in p7[0])
    #     elif q == 'q0' and len_p7>1 and str(v) == "v_phase_area" and s[2] == 'zm0':
    #         bnd = list_bnd[idx]
    #         return v[s] >= bnd * 0.99 #minus 1 to give the model a tiny bit of wriggle room for rounding issues
    #     else:
    #         return pe.Constraint.Skip
    #
    # # model.con_MP_upper = pe.Constraint(list_idx, rule=MP_upper)
    #
    # def MP_lower(model, idx):
    #     v = list_v[idx]
    #     s = list_s[idx]
    #     q = s[0]
    #     if q == 'q0' and (str(v)=='v_dams'  or str(v)=='v_offs' or str(v)=='v_phase_area' or
    #                                           str(v)=='v_greenpas_ha' or str(v)=='v_drypas_transfer' or
    #                                           str(v)=='v_tonnes_sb_transfer' or str(v)=='v_stub_transfer'): #only need to constrain variables that transfer from q_prev
    #         bnd = list_bnd[idx]
    #         return v[s] <= bnd * 1.01 #plus 1 to give the model a tiny bit of wriggle room for rounding issues
    #     ###bnd rotation variables in p7[0] (unless only one p7 period because that means the management can change in p7[0])
    #     elif q == 'q0' and len_p7>1 and str(v) == "v_phase_area" and s[2] == 'zm0':
    #         bnd = list_bnd[idx]
    #         return v[s] <= bnd * 1.01 #plus 1 to give the model a tiny bit of wriggle room for rounding issues
    #     else:
    #         return pe.Constraint.Skip
    #
    # # model.con_MP_lower = pe.Constraint(list_idx, rule=MP_lower)



    # ##Notes:
    # ##1.
    # ## there has been cases where including risk stop the model solving correctly.
    # ## This can be helped by customising the segments to more closely fit the expected profit.
    # ## The solver also seems to prefers if all segments are the same size.
    # ##2.
    # ## there is no point having very big segments because all levels of terminal wealth within a segment have a linear
    # ## relationship with utility therefore to reflect risk aversion terminal wealth due to different price (c1) and
    # ## season (z) need to fall into different segments. Therefore, the size of the segments should reflect the variation
    # ## between c1 and z.
    #
    # ##piecewise utility function - two options CRRA and CARA
    # if not uinp.general['i_inc_risk']:
    #     breakpoints = [-500000, 20000001]
    #     def f(model, i0, i1, i2, i3, x):
    #         '''If no risk utility is equal to profit'''
    #         return x
    # elif uinp.general['i_utility_method']==1: #CARA
    #     a=uinp.general['i_cara_risk_coef']
    #     breakpoints = list(range(-500000, 1000000, 75000)) #majority of segments in expected profit range - these need to line up with terminal wealth before initial wealth is added.
    #     breakpoints.append(20000001) #add a high number to end to handle if profit is very high. Note utility will be linear for any values in this last segment, thus shouldn't be common to have profit in this seg
    #     def f(model, i0, i1, i2, i3, x):
    #         '''CARA/CRRA utility function'''
    #         return 1-np.exp(-a*x)
    # elif uinp.general['i_utility_method']==2: #CRRA
    #     Rr = uinp.general['i_crra_risk_coef']
    #     initial_welth = uinp.general['i_crra_initial_wealth']
    #     breakpoints = list(range(-500000, 1000000, 75000)) #majority of segments in expected profit range - these need to line up with terminal wealth before initial wealth is added.
    #     breakpoints.append(20000001) #add a high number to end to handle if profit is very high. Note utility will be linear for any values in this last segment, thus shouldn't be common to have profit in this seg
    #     def f(model, i0, i1, i2, i3, x):
    #         '''CRRA utility function'''
    #         ##This method doesnt handle negative terminal wealth (x).
    #         ##The function also returns a very small number at high Rr which seem to trip out the solver.
    #         x+=initial_welth
    #         return x**(1-Rr) / (1-Rr)
    # else:
    #     raise ValueError("Specify a valid risk method or turn risk off.")
    #
    # model.con = pc.Piecewise(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_c1, #sets
    #                          model.v_utility, model.v_terminal_wealth, # range and domain variables
    #                          pw_pts=breakpoints,
    #                          pw_constr_type='UB',
    #                          f_rule=f,
    #                          pw_repn='CC')
    #
    # ##objective function (maximise utility)
    # return sum(model.v_utility[q,s,z,c1] * model.p_season_prob_qsz[q,s,z] * model.p_prob_c1[c1] * model.p_discount_factor_q[q]
    #            for q in model.s_sequence_year for s in model.s_sequence for c1 in model.s_c1 for z in model.s_season_types)  # have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
