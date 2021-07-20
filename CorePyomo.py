# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 18:40:57 2019


module: core module - contains constraints for core equations and objective

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
     

@author: young
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

#AFO modules - should only be pyomo modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
import CropPyomo as crppy
import MachPyomo as macpy
import LabourPyomo as labpy
import LabourCropPyomo as lcrppy
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
import StockPyomo as stkpy
import MVF as mvf
import Sensitivity as sen
 
import Finance as fin


def coremodel_all(params, trial_name, model):
    '''
    Wraps all of the core model into a function so it can be run multiple times in a loop

    Returns
    -------
    None.

    '''
    #######################################################################################################################################################
    #######################################################################################################################################################
    #core model (pyomo constraints)
    #######################################################################################################################################################
    #######################################################################################################################################################
    
    ######################
    #Labour fixed        #
    ######################
    ##Fixed labour jobs that can be completed by anyone ie this constraint links labour fixed casual and perm and manager supply and requirement.
    def labour_fixed_casual(model, p, w, z):
        return -model.v_fixed_labour_casual[p,w,z] - model.v_fixed_labour_permanent[p,w,z] - model.v_fixed_labour_manager[p,w,z] \
               +  model.p_super_labour[p,z] + model.p_tax_labour[p,z] + model.p_bas_labour[p,z] <= 0
    model.con_labour_fixed_anyone = pe.Constraint(model.s_labperiods, ['any'], model.s_season_types, rule = labour_fixed_casual, doc='link between labour supply and requirement by fixed jobs for casual and above')
    
    ##Fixed labour jobs that must be completed by the manager ie this constraint links labour fixed manager supply and requirement.
    def labour_fixed_manager(model, p, w, z):
        return -model.v_fixed_labour_manager[p,w,z] +  model.p_planning_labour[p,z] + (model.p_learn_labour * model.v_learn_allocation[p,z]) <= 0
    model.con_labour_fixed_manager = pe.Constraint(model.s_labperiods, ['mngr'], model.s_season_types, rule = labour_fixed_manager, doc='link between labour supply and requirement by fixed jobs for manager')
    
    ######################
    #Labour crop         #
    ######################
    ##labour crop - can be done by anyone
    def labour_crop_anyone(model, p, w, z):
        return -model.v_crop_labour_casual[p,w,z] - model.v_crop_labour_permanent[p,w,z] - model.v_crop_labour_manager[p,w,z] + lcrppy.mach_labour_anyone(model,p, z) <= 0
    model.con_labour_crop_anyone = pe.Constraint(model.s_labperiods, ['any'], model.s_season_types, rule = labour_crop_anyone, doc='link between labour supply and requirement by crop jobs for all labour sources')
    
    ##labour crop - can be done by perm and manager
    def labour_crop_perm(model,p,w,z):
        return - model.v_crop_labour_permanent[p,w,z] - model.v_crop_labour_manager[p,w,z] + lcrppy.mach_labour_perm(model,p,z) <= 0
    model.con_labour_crop_perm = pe.Constraint(model.s_labperiods, ['perm'], model.s_season_types, rule = labour_crop_perm, doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')

    ######################
    #labour Sheep        #
    ######################
    ##labour sheep - can be done by anyone
    def labour_sheep_cas(model,p,w,z):
        return -model.v_sheep_labour_casual[p,w,z] - model.v_sheep_labour_permanent[p,w,z] - model.v_sheep_labour_manager[p,w,z] + suppy.sup_labour(model,p,z) + stkpy.stock_labour_anyone(model,p,z) <= 0
    model.con_labour_sheep_anyone = pe.Constraint(model.s_labperiods, ['any'], model.s_season_types, rule = labour_sheep_cas, doc='link between labour supply and requirement by sheep jobs for all labour sources')

    ##labour sheep - can be done by permanent and manager staff
    def labour_sheep_perm(model,p,w,z):
        return - model.v_sheep_labour_permanent[p,w,z] - model.v_sheep_labour_manager[p,w,z] + stkpy.stock_labour_perm(model,p,z) <= 0
    model.con_labour_sheep_perm = pe.Constraint(model.s_labperiods, ['perm'], model.s_season_types, rule = labour_sheep_perm, doc='link between labour supply and requirement by sheep jobs for perm labour sources')

    ##labour sheep - can be done by manager
    def labour_sheep_manager(model,p,w,z):
        return  - model.v_sheep_labour_manager[p,w,z] + stkpy.stock_labour_manager(model,p,z)   <= 0
    model.con_labour_sheep_manager = pe.Constraint(model.s_labperiods, ['mngr'], model.s_season_types, rule = labour_sheep_manager, doc='link between labour supply and requirement by sheep jobs for manager labour sources')

    #######################################
    #stubble & nap consumption at harvest #
    #######################################
    def harv_stub_nap_cons(model,p6,z):
        if any(model.p_nap_prop[p6,z] or model.p_harv_prop[p6,z,k] for k in model.s_crops):
            return sum(-paspy.pas_me(model,p6,f,z) + sum(model.p_harv_prop[p6,z,k]/(1-model.p_harv_prop[p6,z,k])
                                                         * model.v_stub_con[f,p6,z,k,s] * model.p_stub_md[f,p6,z,k,s] for k in model.s_crops for s in model.s_stub_cat)
                    +  model.p_nap_prop[p6,z]/(1-model.p_nap_prop[p6,z]) * paspy.nappas_me(model,p6,f,z) for f in model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip
    model.con_harv_stub_nap_cons = pe.Constraint(model.s_feed_periods, model.s_season_types, rule = harv_stub_nap_cons, doc='limit stubble and nap consumption in the period harvest occurs')

    ######################
    #stubble             #
    ###################### 

    def stubble_a(model,k,s,z):
        if model.p_rot_stubble[k,s] !=0:
            return   -crppy.rot_stubble(model,k,s,z) + macpy.stubble_penalty(model,k,s,z) + stubpy.stubble_req_a(model,z,k,s) <= 0
        else:
            return pe.Constraint.Skip
    model.con_stubble_a = pe.Constraint(model.s_crops, model.s_stub_cat, model.s_season_types, rule = stubble_a, doc='links rotation stubble production with consumption of cat A')
    
    ######################
    #sow landuse        #
    ###################### 
   
    ##links crop sow req with mach sow provide - no p set because model can optimise crop sowing time
    def cropsow_link(model,k,l,z):
        if type(crppy.cropsow(model,k,l,z)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no crop is being sown on given rotation
        else:
            return sum(-model.v_seeding_crop[p,k,l,z] for p in model.s_labperiods) + crppy.cropsow(model,k,l,z)  <= 0
    model.con_cropsow = pe.Constraint(model.s_crops, model.s_lmus, model.s_season_types, rule = cropsow_link, doc='link between mach sow provide and rotation crop sow require')
   
    ##links pasture sow req with mach sow provide - requires a p set because the timing of sowing pasture is not optimisable (pasture sowing can occur in any period so the user specifies the periods when a given pasture must be sown)
    ##pasture sow has separate constraint from crop rotations because pas sow has a p axis so that user can specify period when pasture is sown (pasture has no yield penalty so model doesnt optimise seeding time like it does for crop)
    def passow_link(model,p,k,l,z):
        if type(paspy.passow(model,p,k,l,z)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no pasture is being sown
        else:
            return -model.v_seeding_pas[p,k,l,z]  + paspy.passow(model,p,k,l,z) <= 0
    model.con_passow = pe.Constraint( model.s_labperiods, model.s_landuses, model.s_lmus, model.s_season_types, rule = passow_link, doc='link between mach sow provide and rotation pas sow require')

    ######################
    #harvest crops       #
    ###################### 
    ##links crop and mach pyomo together
    def harv(model,k,z):
        return  -macpy.harv_supply(model,k,z) + sum(crppy.rotation_yield_transfer(model,g,k,z)/1000 for g in model.s_grain_pools)  <= 0
    model.con_harv = pe.Constraint(model.s_harvcrops, model.s_season_types, rule = harv, doc='harvest constraint')


    ######################
    #harvest hay         #
    ###################### 
    ##links crop and mach pyomo together
    def harv(model,k,z):
        return  sum(-model.v_hay_made[z] + crppy.rotation_yield_transfer(model,g,k,z)/1000 for g in model.s_grain_pools)  <= 0
    model.con_makehay = pe.Constraint(model.s_haycrops, model.s_season_types, rule = harv, doc='make hay constraint')

    #############################
    #yield income & transfer    #
    #############################
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint.
    def grain_transfer(model,g,k,z):
        return -crppy.rotation_yield_transfer(model,g,k,z) + macpy.late_seed_penalty(model,g,k,z) + sum(model.v_sup_con[z,k,g,f,p6]*1000 for f in model.s_feed_pools for p6 in model.s_feed_periods)\
               - model.v_buy_grain[z,k,g]*1000 + model.v_sell_grain[z,k,g]*1000 <=0
    model.con_grain_transfer = pe.Constraint(model.s_grain_pools, model.s_crops, model.s_season_types, rule=grain_transfer, doc='constrain grain transfer between rotation and sup feeding')
    
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    def grain_income(model,c,z):
        return sum(model.v_sell_grain[z,k,g] * model.p_grain_price[k,c,g] - model.v_buy_grain[z,k,g]* model.p_buy_grain_price[k,c,g] for k in model.s_crops for g in model.s_grain_pools)
    
    ######################
    #feed                #
    ###################### 
    ##green grazing on crop paddock before seeding
    def poc(model,f,l,z):
        return -macpy.ha_days_pasture_crop_paddocks(model,f,l,z) * model.p_poc_con[f,l,z] + sum(model.v_poc[v,f,l,z] for v in model.s_feed_pools) <=0
    model.con_poc_available = pe.Constraint(model.s_feed_periods, model.s_lmus, model.s_season_types, rule=poc, doc='constraint between poc available and consumed')

    ######################
    #  ME                #
    ######################
    def me(model,p6,f,z):
        return -paspy.pas_me(model,p6,f,z) - paspy.nappas_me(model,p6,f,z) - suppy.sup_me(model,p6,f,z) - stubpy.stubble_me(model,p6,f,z) \
               + stkpy.stock_me(model,p6,f,z) - mvf.mvf_me(model,p6,f) <=0
    model.con_me = pe.Constraint(model.s_feed_periods, model.s_feed_pools, model.s_season_types, rule=me, doc='constraint between me available and consumed')

    ######################
    #Vol                 #
    ######################
    def vol(model,p6,f,z):
        return paspy.pas_vol(model,p6,f,z) + suppy.sup_vol(model,p6,f,z) + stubpy.stubble_vol(model,p6,f,z) - stkpy.stock_pi(model,p6,f,z) \
               + mvf.mvf_vol(model,p6,f) <=0
    model.con_vol = pe.Constraint(model.s_feed_periods, model.s_feed_pools, model.s_season_types, rule=vol, doc='constraint between me available and consumed')

    ######################
    #cashflow constraints#
    ######################

    def cash_flow(model,i,z):
        '''
        Returns
        -------
        Constraint
            combines all cashflow functions from each module and includes debit and credit to form constraint.
            for each cashflow period dollar flow must be greater than 0. this is accomplished by taking a loan from the bank (if there is more exp than income) or depositing money in the bank.
            the money withdrawn or deposited in the bank (debit or credit) is then carried over to the next period.
            the debit and credit carried over is multiplied by j because there is no carry over in the first period (there may be a better way to do it though)
            Carryover basically represents interest free cash at the start of the year. It requires cash from ND and provides in JF.

        '''
        c = sinp.general['cashflow_periods']
        ##j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        ##this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period)
        j = [1] * len(c)
        j[0] = 0
        #todo Revisit the interest calculation at some stage because it didn't tally with the back of envelope estimate by $1000
        return (-grain_income(model,c[i], z) + crppy.rotation_cost(model,c[i], z) + labpy.labour_cost(model,c[i], z)
                + macpy.mach_cost(model,c[i], z) + suppy.sup_cost(model,c[i], z) + model.p_overhead_cost[c[i]]
                - stkpy.stock_cashflow(model,c[i], z)
                - model.v_debit[c[i]] + model.v_credit[c[i]]  + model.v_debit[c[i-1]] * fin.debit_interest() - model.v_credit[c[i-1]] * fin.credit_interest() * j[i] #mul by j so that credit in ND doesnt provide into JF otherwise it will be unbounded because it will get interest
                ) <= 0
    model.con_cashflow = pe.Constraint(range(len(model.s_cashflow_periods)), model.s_season_types, rule=cash_flow, doc='cashflow')

    ######################
    #dep                 #
    ######################
    def dep(model,z):
        return  macpy.total_dep(model,z) + suppy.sup_dep(model,z) - model.v_dep[z] <=0
    model.con_dep = pe.Constraint(model.s_season_types, rule=dep, doc='tallies depreciation from all activities so it can be transferred to objective')
    
    ######################
    #asset               #
    ######################
    def asset(model,z):
        return (suppy.sup_asset(model,z) + macpy.mach_asset(model) + stkpy.stock_asset(model,z)) * uinp.finance['opportunity_cost_capital']  \
                - model.v_asset[z] <=0
    model.con_asset = pe.Constraint(model.s_season_types, rule=asset, doc='tallies asset from all activities so it can be transferred to objective to represent ROE')
    
    ######################
    #Min ROE             #
    ######################
    def minroe(model,z):
        return (sum(crppy.rotation_cost(model,c,z)  + labpy.labour_cost(model,c,z) + macpy.mach_cost(model,c,z)
                    + suppy.sup_cost(model,c,z) for c in model.s_cashflow_periods) + stkpy.stock_cost(model,z)) * fin.f_min_roe() \
                - model.v_minroe[z] <=0
    model.con_minroe = pe.Constraint(model.s_season_types, rule=minroe, doc='tallies total expenditure to ensure minimum roe is met')
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #objective
    #######################################################################################################################################################
    #######################################################################################################################################################
    '''
    maximise credit in the last period of cashflow (rather than indexing directly with ND$FLOW, i index with the last name in the cashflow periods in case cashflow periods change) 
    minus dep (variable and fixed)
    '''
    def profit(model):
        c = sinp.general['cashflow_periods']
        i = len(c) - 1 # minus one because index starts from 0
        return model.v_credit[c[i]]-model.v_debit[c[i]] - sum(model.v_dep[z] + model.v_minroe[z] + model.v_asset[z] for z in model.s_season_types)#have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.

    model.profit = pe.Objective(rule=profit, sense=pe.maximize)
    # model.profit.pprint()


    #######################################################################################################################################################
    #######################################################################################################################################################
    #solve
    #######################################################################################################################################################
    #######################################################################################################################################################


    ##sometimes if there is a bug when solved it is good to write lp here - because the code doesn't run to the other place where lp written
    model.write('Output/test.lp',io_options={'symbolic_solver_labels':True}) #comment this out when not debugging

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
        solver.options['tmlim'] = 100 #limit solving time to 100sec in case solver stalls.
    solver_result = solver.solve(model, tee=True)  #turn to true for solver output - may be useful for troubleshooting
    try: #to handle infeasible (there is no profit component when infeasible)
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
            f.write("Solver Status: {0}" .format(solver_result.solver.termination_condition))
    else:  # Something else is wrong - solver may have stalled.
        print('***Solver Status: error (other)***')
        ###save infeasible file
        with open('Output/infeasible/%s.txt' % trial_name,'w') as f:
            f.write("Solver Status: {0}" .format(solver_result.solver.termination_condition))

    return obj