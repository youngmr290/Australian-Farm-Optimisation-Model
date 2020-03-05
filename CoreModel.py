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
# from LabourPyomo import *
# from LabourFixedPyomo import *

# from LabourCropPyomo import *
from CreateModel import *

import CropPyomo as crppy
import MachPyomo as macpy
import FinancePyomo #not used but it needs to be imported so that it is run
import LabourPyomo as labpy 
import Finance as fin

print('Status: running coremodel')

coremodel_test_var=[]

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
    #labour fixed        #
    ######################
    #links labour fixed casual supply and requirment. ie fixed labour tasks that can be done by casual or any other staff (casual is the only supply but in the labourpyomo sheet there is a transfer variable that allows the managger and perm staff to complete casual jobs)
    def labour_fixed_casual(model,p):
        return model.fixed_labour_casual[p] -  model.p_super_labour[p] - model.p_tax_labour[p] - model.p_bas_labour[p] >= 0
    model.con_labour_fixed_casual_link = Constraint(model.s_periods, rule = labour_fixed_casual, doc='link between labour supply and requirment by fixed jobs for casual and above')
    
    
    
    #links labour fixed manager supply and requirment. ie fixed labour tasks that can be done only by manager.
    def labour_fixed_manager(model,p):
        return model.fixed_labour_manager[p] -  model.p_planning_labour [p] - model.p_learn_labour[p] >= 0
    model.con_labour_fixed_manager_link = Constraint(model.s_periods, rule = labour_fixed_manager, doc='link between labour supply and requirment by fixed jobs for manager')
    
    #labour crop - can be done by anyone
    
    
    ######################
    #stubble             #
    ###################### 
    
    
    #############################
    #reduction in yield income  #
    #############################
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint. 
    def yield_income(model,c):
        return sum(( macpy.late_seed_penalty(model,k)) * model.p_grain_price[c,k]/1000 for k in model.s_crops)
    
    ######################
    #feed                #
    ###################### 
    ##green grazing on crop paddock before seeding
    # def graze_pasture_crop_paddocks(model,f):
    #     return (ha_pasture_crop_paddocks(model,f,l) * foo on crop paddocks)/1000 - model.v_sheep_pascroppaddocks    #divide by 1000 converts to tonnes (maybe do this in pasture sheet before to keep this tidy)

    # ######################
    # #  ME                #
    # ###################### 
    # def sheep_me(model,f):
    #     model.v_sheep_pascroppaddocks

    # ######################
    # #Vol                 #
    # ###################### 
    # def sheep_vol(model,f):
    #     model.v_sheep_pascroppaddocks
        
    ######################
    #cashflow constraints#
    ######################    
    #combines all cashflow functions from each module and includes debit and credit to form constraint. 
    #for each cashflow period dollar flow must be greater than 0. this is accomplished by taking a loan from the bank (if there is more exp than income) or depositing money in the bank. 
    #the money withdrawn or deposited in the bank (debit or credit) is then carried over to the next period.
    #the debit and credit carried over is multimpled by j because there is no carry over in the first period (there may be a better way to do it though)
    def cash_flow(model,i): 
        c = uinp.structure['cashflow_periods']
        #j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        #this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period) 
        j = [1] * len(c)
        j[0] = 0
        return (yield_income(model,c[i]) - crppy.rotation_cost(model,c[i])  - labour_cost(model,c[i]) - mach_cost(model,c[i]) +
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
    
    # print('Status: writing...')
    model.write('test.lp',io_options={'symbolic_solver_labels':True})
    # print('Status: solving...')
    results = SolverFactory('glpk').solve(model, tee=True)
    # results.write() need to write this somewhere
    # print("\nDisplaying Solution\n" + '-'*60)
    # print(value(model.profit))
    # pyomo_postprocess(None, model, results) #not sure what this is
    # results.write(num=2) #not sure what the num does, if removed it still works the same, maybe this is if there are multiple model instances
    
    # model.v_debit.pprint()
    # model.v_credit.pprint()
    if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):
        print('solver optimal')# Do something when the solution in optimal and feasible
        coremodel_test_var.append(0)
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        print ('Solver Status: infeasible')#,  result.solver.status)
        coremodel_test_var.append(1)
        # print(coremodel_test_var)
        # sys.exit()
    else:
        # Something else is wrong
        print ('Solver Status: error')#,  result.solver.status)
        coremodel_test_var.append(1)

    ##writes variable to txt file to view
    file = open('testfile.txt','w') 
    for v in model.component_objects(Var, active=True):
        file.write("Variable component object %s\n" %v)   #  \n makes new line
        for index in v:
            try:
                if v[index].value>0:
                    file.write ("   %s %s\n" %(index, v[index].value))
            except: pass #print("error on   %s %s\n" %(index, v[index].value))
    file.close()
    
    
    
    
    
    
    
    
    
