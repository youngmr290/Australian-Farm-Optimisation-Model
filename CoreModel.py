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

import pyomo.environ as pe
import sys

#MUDAS modules - should only be pyomo modules
import UniversalInputs as uinp
from CreateModel import model
import CropPyomo as crppy
import MachPyomo as macpy
import FinancePyomo #not used but it needs to be imported so that it is run
import LabourPyomo as labpy 
import LabourFixedPyomo as lfixpy 
import LabourCropPyomo as lcrppy 
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
 
import Finance as fin


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
        model.del_component(model.con_labour_fixed_anyone)
    except AttributeError:
        pass
    def labour_fixed_casual(model,p):
        return -model.v_fixed_labour_casual[p] - model.v_fixed_labour_permanent[p] - model.v_fixed_labour_manager[p] +  model.p_super_labour[p] + model.p_tax_labour[p] + model.p_bas_labour[p] <= 0
    model.con_labour_fixed_anyone = pe.Constraint(model.s_periods, rule = labour_fixed_casual, doc='link between labour supply and requirment by fixed jobs for casual and above')
    
    ##Fixed labour jobs that must be completed by the manager ie this constraint links labour fixed manager supply and requirment. 
    try:
        model.del_component(model.con_labour_fixed_manager)
    except AttributeError:
        pass
    def labour_fixed_manager(model,p):
        return -model.v_fixed_labour_manager[p] +  model.p_planning_labour [p] + (model.p_learn_labour * model.v_learn_allocation[p]) <= 0
    model.con_labour_fixed_manager = pe.Constraint(model.s_periods, rule = labour_fixed_manager, doc='link between labour supply and requirment by fixed jobs for manager')
    
    ######################
    #Labour crop         #
    ######################
    ##labour crop - can be done by anyone
    try:
        model.del_component(model.con_labour_crop_anyone)
    except AttributeError:
        pass
    def labour_crop(model,p):
        return -model.v_crop_labour_casual[p] - model.v_crop_labour_permanent[p] - model.v_crop_labour_manager[p] + lcrppy.mach_labour(model,p)  <= 0
    model.con_labour_crop_anyone = pe.Constraint(model.s_periods, rule = labour_crop, doc='link between labour supply and requirment by crop jobs for all labour sources')
    
    ######################
    #Sheep crop          #
    ######################
    ##labour sheep - can be done by anyone
    try:
        model.del_component(model.con_labour_sheep_anyone)
    except AttributeError:
        pass
    def labour_crop(model,p):
        return -model.v_sheep_labour_casual[p] - model.v_sheep_labour_permanent[p] - model.v_sheep_labour_manager[p] + suppy.sup_labour(model,p)   <= 0
    model.con_labour_sheep_anyone = pe.Constraint(model.s_periods, rule = labour_crop, doc='link between labour supply and requirment by sheep jobs for all labour sources')
    
    ######################
    #stubble             #
    ###################### 
    try:
        model.del_component(model.con_harv_stub_cons_index)
        model.del_component(model.con_harv_stub_cons)
    except AttributeError:
        pass
    def harv_stub_cons(model,e,f):
        return -paspy.pas_md(model,e,f) + sum(model.p_harv_prop[f,k]/(1-model.p_harv_prop[f,k]) * model.v_stub_con[e,f,k,s] * model.p_stub_md[f,s,k] for k in model.s_crops for s in model.s_stub_cat)  <= 0
    model.con_harv_stub_cons = pe.Constraint(model.s_sheep_pools, model.s_feed_periods, rule = harv_stub_cons, doc='limit stubble consumption in the period harvest occurs')
  
    try:
        model.del_component(model.con_stubble_a_index)
        model.del_component(model.con_stubble_a)
    except AttributeError:
        pass
    def stubble_a(model,k,s):
        if model.p_rot_stubble[k,s] !=0:
            return   -crppy.rot_stubble(model,k,s) + macpy.stubble_penalty(model,k,s) + stubpy.stubble_req_a(model,k,s) <= 0
        else:
            return pe.Constraint.Skip
    model.con_stubble_a = pe.Constraint(model.s_crops, model.s_stub_cat, rule = stubble_a, doc='links rotation stubble production with consumption of cat A')
    
    ######################
    #sow landuse        #
    ###################### 
   
    ##links crop sow req with mach sow provide
    try:
        model.del_component(model.con_cropsow_index)
        model.del_component(model.con_cropsow)
    except AttributeError:
        pass
    def cropsow_link(model,k,l):
        return sum(-model.v_seeding_crop[p,k,l] for p in model.s_periods) + crppy.cropsow(model,k,l)  <= 0
    model.con_cropsow = pe.Constraint(model.s_crops, model.s_lmus, rule = cropsow_link, doc='link between mach sow provide and rotation crop sow require')
   
    ##links pasture sow req with mach sow provide
    try:
        model.del_component(model.con_passow_index_index_0)
        model.del_component(model.con_passow_index)
        model.del_component(model.con_passow)
    except AttributeError:
        pass
    def passow_link(model,p,k,l):
        return -model.v_seeding_pas[p,k,l]  + paspy.passow(model,p,k,l) <= 0
    model.con_passow = pe.Constraint( model.s_periods, model.s_pastures, model.s_lmus, rule = passow_link, doc='link between mach sow provide and rotation pas sow require')

    ######################
    #harvest crops       #
    ###################### 
    ##links crop and mach pyomo together
    try:
        model.del_component(model.con_harv)
    except AttributeError:
        pass
    def harv(model,k):
        return  -macpy.harv_supply(model,k) + sum(crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  <= 0
    model.con_harv = pe.Constraint(model.s_harvcrops, rule = harv, doc='harvest constraint')


    ######################
    #harvest hay         #
    ###################### 
    ##links crop and mach pyomo together
    try:
        model.del_component(model.con_makehay)
    except AttributeError:
        pass
    def harv(model,k):
        return  sum(-model.v_hay_made + crppy.rotation_yield_transfer(model,g,k)/1000 for g in model.s_grain_pools)  <= 0
    model.con_makehay = pe.Constraint(model.s_haycrops, rule = harv, doc='make hay constraint')
    
    #############################
    #yield income & transfer    #
    #############################
    ##combines rotation yield, on-farm sup feed and yield penalties from untimely sowing and crop grazing. Then passes to cashflow constraint. 
    try:
        model.del_component(model.con_grain_transfer_index)
        model.del_component(model.con_grain_transfer)
    except AttributeError:
        pass
    def grain_transfer(model,g,k):
        return -crppy.rotation_yield_transfer(model,g,k) + macpy.late_seed_penalty(model,g,k) + sum(model.v_sup_con[k,g,e,f]*1000 for e in model.s_sheep_pools for f in model.s_feed_periods)\
                - model.v_buy_grain[k,g]*1000 + model.v_sell_grain[k,g]*1000 <=0
    model.con_grain_transfer = pe.Constraint(model.s_grain_pools, model.s_crops, rule=grain_transfer, doc='constrain grain transfer between rotation and sup feeding')
    
    ##combined grain sold and purchased to get a $ amount which is added to the cashflow constrain
    def yield_income(model,c):
        return sum(model.v_sell_grain[k,g] * model.p_grain_price[k,c,g] - model.v_buy_grain[k,g]* model.p_buy_grain_price[k,c,g] for k in model.s_crops for g in model.s_grain_pools)
    
    ######################
    #feed                #
    ###################### 
    ##green grazing on crop paddock before seeding
    try:
        model.del_component(model.con_poc_available_index)
        model.del_component(model.con_poc_available)
    except AttributeError:
        pass
    def poc(model,f,l):
        return (-macpy.ha_pasture_crop_paddocks(model,f,l) * paspy.model.p_poc_con[f,l])/1000 + sum(paspy.model.v_poc[e,f,l] for e in model.s_sheep_pools) <=0   #divide by 1000 converts to tonnes (maybe do this in pasture sheet before to keep this tidy)
    model.con_poc_available = pe.Constraint(model.s_feed_periods, model.s_lmus, rule=poc, doc='constraint between poc available and consumed')

    ######################
    #  ME                #
    ###################### 
    def md(model,f,e):
        paspy.pas_md(e,f) + suppy.sup_md(model,e,f) + stubpy.stubble_md(model,e,f)

    ######################
    #Vol                 #
    ###################### 
    def vol(model,f,e):
        paspy.pas_vol(e,f) + suppy.sup_vol(model,e,f) + stubpy.stubble_vol(model,e,f)
        
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
        return (-yield_income(model,c[i]) + crppy.rotation_cost(model,c[i])  + labpy.labour_cost(model,c[i]) + macpy.mach_cost(model,c[i]) + suppy.sup_cost(model,c[i]) -
                model.v_debit[c[i]] + model.v_credit[c[i]]  + model.v_debit[c[i-1]] * fin.debit_interest() * j[i]  - model.v_credit[c[i-1]] * fin.credit_interest() * j[i]) <= 0

    try:
        model.del_component(model.con_cashflow)
        model.del_component(model.con_cashflow_index)
    except AttributeError:
        pass
    model.con_cashflow = pe.Constraint(range(len(model.s_cashflow_periods)), rule=cash_flow, doc='cashflow')
    
    ######################
    #dep                 #
    ###################### 
    try:
        model.del_component(model.con_dep)
    except AttributeError:
        pass
    def dep(model):
        return  macpy.total_dep(model) + suppy.sup_dep(model) - model.v_dep <=0   
    model.con_dep = pe.Constraint( rule=dep, doc='tallies depreciation from all activities so it can be transferd to objective')
    
    ######################
    #asset               #
    ###################### 
    try:
        model.del_component(model.con_asset)
    except AttributeError:
        pass
    def asset(model):
        return suppy.sup_asset(model) - model.v_asset <=0   
    model.con_asset = pe.Constraint( rule=asset, doc='tallies asset from all activities so it can be transferd to objective to represent ROE')
    
    ######################
    #Min ROE             #
    ###################### 
    try:
        model.del_component(model.con_minroe)
    except AttributeError:
        pass
    def minroe(model):
        return (sum(crppy.rotation_cost(model,c)  + labpy.labour_cost(model,c) + macpy.mach_cost(model,c) + suppy.sup_cost(model,c) for c in model.s_cashflow_periods) *uinp.finance['minroe']) \
                - model.v_minroe <=0   
    model.con_minroe = pe.Constraint(rule=minroe, doc='tallies total expenditure to ensure minimum roe is met')
    
    
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
        return model.v_credit[c[i]]-model.v_debit[c[i]] - model.v_dep - model.v_minroe - (model.v_asset * uinp.finance['opportunity_cost_capital'])  #have to include debit otherwise model selects lots of debit to increase credit, hence cant just maximise credit.
    try:
        model.del_component(model.profit)
    except AttributeError:
        pass
    model.profit = pe.Objective(rule=profit, sense=pe.maximize)
    # model.profit.pprint()
    
    
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #solve
    #######################################################################################################################################################
    #######################################################################################################################################################
    ##tells the solver you want duals and rc
    try:
        model.del_component(model.dual)
    except AttributeError:
        pass
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    try:
        model.del_component(model.rc)
    except AttributeError:
        pass
    model.rc = pe.Suffix(direction=pe.Suffix.IMPORT)
    ##solve - tee=True will print out solver information
    results = pe.SolverFactory('glpk').solve(model, tee=False) #turn to true for solver output - may be useful for troubleshooting
    ##this check if the solver is optimal - if infeasible or error the model will quit
    if (results.solver.status == pe.SolverStatus.ok) and (results.solver.termination_condition == pe.TerminationCondition.optimal):
        print('solver optimal')# Do nothing when the solution in optimal and feasible
    elif (results.solver.termination_condition == pe.TerminationCondition.infeasible):
        print ('Solver Status: infeasible')
        sys.exit()
    else: # Something else is wrong
        print ('Solver Status: error')
        sys.exit()

    
    
    
