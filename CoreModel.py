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

from pyomo.environ import *
import sys

#MUDAS modules - should only be pyomo modules
from CreateModel import *
import CropPyomo as crppy
import MachPyomo as macpy
import FinancePyomo #not used but it needs to be imported so that it is run
import LabourPyomo as labpy 
import LabourFixedPyomo as lfixpy 
import LabourCropPyomo as lcrppy 
import PasturePyomo as paspy 
import Finance as fin

print('Status: running coremodel')


def coremodel_all():
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
    ##Fixed labour jobs that can be completed by anyone ie this constraint links labour fixed casual and perm and manager supply and requirment. 
    try:
        model.del_component(model.con_labour_fixed_casual_link)
    except AttributeError:
        pass
    def labour_fixed_casual(model,p):
        return model.v_fixed_labour_casual[p] + model.v_fixed_labour_permanent[p] + model.v_fixed_labour_manager[p] -  model.p_super_labour[p] - model.p_tax_labour[p] - model.p_bas_labour[p] >= 0
    model.con_labour_fixed_casual_link = Constraint(model.s_periods, rule = labour_fixed_casual, doc='link between labour supply and requirment by fixed jobs for casual and above')
    
    ##Fixed labour jobs that must be completed by the manager ie this constraint links labour fixed manager supply and requirment. 
    try:
        model.del_component(model.con_labour_fixed_manager_link)
    except AttributeError:
        pass
    def labour_fixed_manager(model,p):
        return model.v_fixed_labour_manager[p] -  model.p_planning_labour [p] - (model.p_learn_labour * model.v_learn_allocation[p]) >= 0
    model.con_labour_fixed_manager_link = Constraint(model.s_periods, rule = labour_fixed_manager, doc='link between labour supply and requirment by fixed jobs for manager')
    
    ######################
    #Labour crop         #
    ######################
    ##labour crop - can be done by anyone
    try:
        model.del_component(model.con_labour_crop_anyone)
    except AttributeError:
        pass
    def labour_crop(model,p):
        return model.v_crop_labour_casual[p] + model.v_crop_labour_permanent[p] + model.v_crop_labour_manager[p] - lcrppy.mach_labour(model,p)  >= 0
    model.con_labour_crop_anyone = Constraint(model.s_periods, rule = labour_crop, doc='link between labour supply and requirment by crop jobs for all labour sources')
    
    
    ######################
    #stubble             #
    ###################### 
    
    ######################
    #sow landuse        #
    ###################### 
    ##links crop & pasture sow req with mach sow provide
    try:
        model.del_component(model.con_sow_index_index_0)
        model.del_component(model.con_sow_index)
        model.del_component(model.con_sow)
    except AttributeError:
        pass
    def sow_link(model,p,k,l):
        return macpy.sow_supply(model,p,k,l) - crppy.cropsow(model,k, l) - paspy.cropsow(model,p,k,l) >= 0
    model.con_sow = Constraint(model.s_periods, model.s_landuses, model.s_lmus, rule = sow_link, doc='link between mach sow provide and rotation (crop and pas) sow require')

    ######################
    #harvest crops       #
    ###################### 
    ##links crop and mach pyomo together
    try:
        model.del_component(model.con_harv)
    except AttributeError:
        pass
    def harv(model,k):
        return  macpy.harv_supply(model,k) - sum(crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  >= 0
    model.con_harv = Constraint(model.s_harvcrops, rule = harv, doc='harvest constraint')


    ######################
    #harvest hay         #
    ###################### 
    ##links crop and mach pyomo together
    try:
        model.del_component(model.con_makehay)
    except AttributeError:
        pass
    def harv(model,k):
        return  sum(model.v_hay_made - crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  >= 0
    model.con_makehay = Constraint(model.s_haycrops, rule = harv, doc='make hay constraint')
    
    #############################
    #yield income & transfer    #
    #############################
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint. 
    try:
        model.del_component(model.con_grain_transfer)
    except AttributeError:
        pass
    def grain_transfer(model,g,k):
        return crppy.rotation_yield_transfer(model,g,k) - macpy.late_seed_penalty(model,g,k) + model.v_buy_grain[k,g]*1000 - model.v_sell_grain[k,g]*1000 >=0
    model.con_grain_transfer = Constraint(model.s_grain_pools, model.s_crops, rule=grain_transfer, doc='constrain grain transfer between rotation and sup feeding')
    
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    def yield_income(model,c):
        return sum(model.v_sell_grain[k,g] * model.p_grain_price[k,c,g] - model.v_buy_grain[k,g]* model.p_buy_grain_price[k,c,g] for k in model.s_crops for g in model.s_grain_pools)
    
    ######################
    #feed                #
    ###################### 
    ##green grazing on crop paddock before seeding
    try:
        model.del_component(model.con_poc_available_index_index_0)
        model.del_component(model.con_poc_available_index)
        model.del_component(model.con_poc_available)
    except AttributeError:
        pass
    def poc(model,f,l,t):
        return (macpy.ha_pasture_crop_paddocks(model,f,l) * paspy.model.p_poc_con[f,l,t])/1000 - sum(paspy.model.v_poc[e,f,l] for e in model.s_sheep_pools) >=0   #divide by 1000 converts to tonnes (maybe do this in pasture sheet before to keep this tidy)
    model.con_poc_available = Constraint(model.s_feed_periods, model.s_lmus, model.s_pastures, rule=poc, doc='constraint between poc available and consumed')

    ######################
    #  ME                #
    ###################### 
    def md(model,f,e):
        paspy.pas_md(e,f) 

    ######################
    #Vol                 #
    ###################### 
    def md(model,f,e):
        paspy.pas_vol(e,f)
        
    ######################
    #cashflow constraints#
    ######################    
    def cash_flow(model,i): 
        '''
        Returns
        -------
        Constraint
            combines all cashflow functions from each module and includes debit and credit to form constraint. 
            for each cashflow period dollar flow must be greater than 0. this is accomplished by taking a loan from the bank (if there is more exp than income) or depositing money in the bank. 
            the money withdrawn or deposited in the bank (debit or credit) is then carried over to the next period.
            the debit and credit carried over is multimpled by j because there is no carry over in the first period (there may be a better way to do it though)


        '''
        c = uinp.structure['cashflow_periods']
        #j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        #this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period) 
        j = [1] * len(c)
        j[0] = 0
        return (yield_income(model,c[i]) - crppy.rotation_cost(model,c[i])  - labpy.labour_cost(model,c[i]) - macpy.mach_cost(model,c[i]) +
                model.v_debit[c[i]] - model.v_credit[c[i]]  - model.v_debit[c[i-1]] * fin.debit_interest() * j[i]  + model.v_credit[c[i-1]] * fin.credit_interest() * j[i]) >= 0

    try:
        model.del_component(model.con_cashflow)
        model.del_component(model.con_cashflow_index)
    except AttributeError:
        pass
    model.con_cashflow = Constraint(range(len(model.s_cashflow_periods)), rule=cash_flow, doc='cashflow')
    
    
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #objective
    #######################################################################################################################################################
    #######################################################################################################################################################
    '''
    maximise credit in the last period of cashflow (rather than indexing directly with ND$FLOW, i index with the last name in the cashflow periods incase cashflow periods change) 
    minus dep (variable and fixed)
    '''
    def profit(model):
        c = uinp.structure['cashflow_periods']
        i = len(c) - 1 # minus one because index starts from 0
        return model.v_credit[c[i]]-model.v_debit[c[i]] - macpy.total_dep(model)  #have to include debit otherwise model selects lots of debit to increase credit, hence cant just maximise credit.
    try:
        model.del_component(model.profit)
    except AttributeError:
        pass
    model.profit = Objective(rule=profit, sense=maximize)
    # model.profit.pprint()
    
    
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #solve
    #######################################################################################################################################################
    #######################################################################################################################################################
    
    print('Status: writing...')
    model.write('test.lp',io_options={'symbolic_solver_labels':True})
    print('Status: solving...')
    model.dual = Suffix(direction=Suffix.IMPORT)
    model.rc = Suffix(direction=Suffix.IMPORT)
    model.slack = Suffix(direction=Suffix.IMPORT)
    results = SolverFactory('glpk').solve(model, tee=True)
    results.write() #need to write this somewhere
    # pyomo_postprocess(None, model, results) #not sure what this is
    # results.write(num=1) #not sure what the num does, if removed it still works the same, maybe this is if there are multiple model instances
    ##you can access the solution for individual variables doing this
    model.v_debit.pprint()
    model.v_credit.pprint()

    ##this prints the overall profit
    print("\nDisplaying Solution\n" + '-'*60)
    print(value(model.profit))
    
    ##this check if the solver is optimal - if infeasible or error the model will quit
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('solver optimal')# Do nothing when the solution in optimal and feasible
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print ('Solver Status: infeasible')
        sys.exit()
    else: # Something else is wrong
        print ('Solver Status: error')
        sys.exit()

    ##This writes variable with value greater than 1 to txt file 
    file = open('testfile.txt','w') 
    for v in model.component_objects(Var, active=True):
        file.write("Variable component object %s\n" %v)   #  \n makes new line
        for index in v:
            try:
                if v[index].value>0:
                    file.write ("   %s %s\n" %(index, v[index].value))
            except: pass 
    file.close()
    
    ##code below will access slacks on constraint
    # for c in model.component_objects(Constraint, active=True):
    #     print ("   Constraint",c)
    #     for index in c:
    #         print ("      ", index, model.dual[c[index]])
    
    
    
    
    
    
    
