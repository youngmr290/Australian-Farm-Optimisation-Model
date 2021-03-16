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

#AFO modules - should only be pyomo modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
from CreateModel import model
import CropPyomo as crppy
import MachPyomo as macpy
# import FinancePyomo #not used but it needs to be imported so that it is run
import LabourPyomo as labpy 
# import LabourFixedPyomo as lfixpy 
import LabourCropPyomo as lcrppy 
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
import StockPyomo as stkpy
 
import Finance as fin


def coremodel_all(params):
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
    try:
        model.del_component(model.con_labour_fixed_anyone_index_1)
        model.del_component(model.con_labour_fixed_anyone_index)
        model.del_component(model.con_labour_fixed_anyone)
    except AttributeError:
        pass
    def labour_fixed_casual(model,p,w):
        return -model.v_fixed_labour_casual[p,w] - model.v_fixed_labour_permanent[p,w] - model.v_fixed_labour_manager[p,w] +  model.p_super_labour[p] + model.p_tax_labour[p] + model.p_bas_labour[p] <= 0
    model.con_labour_fixed_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_fixed_casual, doc='link between labour supply and requirement by fixed jobs for casual and above')
    
    ##Fixed labour jobs that must be completed by the manager ie this constraint links labour fixed manager supply and requirement.
    try:
        model.del_component(model.con_labour_fixed_manager_index_1)
        model.del_component(model.con_labour_fixed_manager_index)
        model.del_component(model.con_labour_fixed_manager)
    except AttributeError:
        pass
    def labour_fixed_manager(model,p,w):
        return -model.v_fixed_labour_manager[p,w] +  model.p_planning_labour[p] + (model.p_learn_labour * model.v_learn_allocation[p]) <= 0
    model.con_labour_fixed_manager = pe.Constraint(model.s_labperiods, ['mngr'], rule = labour_fixed_manager, doc='link between labour supply and requirement by fixed jobs for manager')
    
    ######################
    #Labour crop         #
    ######################
    ##labour crop - can be done by anyone
    try:
        model.del_component(model.con_labour_crop_anyone_index_1)
        model.del_component(model.con_labour_crop_anyone_index)
        model.del_component(model.con_labour_crop_anyone)
    except AttributeError:
        pass
    def labour_crop_anyone(model,p,w):
        return -model.v_crop_labour_casual[p,w] - model.v_crop_labour_permanent[p,w] - model.v_crop_labour_manager[p,w] + lcrppy.mach_labour_anyone(model,p) <= 0
    model.con_labour_crop_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_crop_anyone, doc='link between labour supply and requirement by crop jobs for all labour sources')
    
    ##labour crop - can be done by perm and manager
    try:
        model.del_component(model.con_labour_crop_perm_index_1)
        model.del_component(model.con_labour_crop_perm_index)
        model.del_component(model.con_labour_crop_perm)
    except AttributeError:
        pass
    def labour_crop_perm(model,p,w):
        return - model.v_crop_labour_permanent[p,w] - model.v_crop_labour_manager[p,w] + lcrppy.mach_labour_perm(model,p) <= 0
    model.con_labour_crop_perm = pe.Constraint(model.s_labperiods, ['perm'], rule = labour_crop_perm, doc='link between labour supply and requirement by crop jobs for perm and manager labour sources')

    ######################
    #labour Sheep        #
    ######################
    ##labour sheep - can be done by anyone
    try:
        model.del_component(model.con_labour_sheep_anyone_index_1)
        model.del_component(model.con_labour_sheep_anyone_index)
        model.del_component(model.con_labour_sheep_anyone)
    except AttributeError:
        pass
    def labour_sheep_cas(model,p,w):
        return -model.v_sheep_labour_casual[p,w] - model.v_sheep_labour_permanent[p,w] - model.v_sheep_labour_manager[p,w] + suppy.sup_labour(model,p) + stkpy.stock_labour_anyone(model,p) <= 0
    model.con_labour_sheep_anyone = pe.Constraint(model.s_labperiods, ['any'], rule = labour_sheep_cas, doc='link between labour supply and requirement by sheep jobs for all labour sources')

    ##labour sheep - can be done by permanent and manager staff
    try:
        model.del_component(model.con_labour_sheep_perm_index_1)
        model.del_component(model.con_labour_sheep_perm_index)
        model.del_component(model.con_labour_sheep_perm)
    except AttributeError:
        pass
    def labour_sheep_perm(model,p,w):
        return - model.v_sheep_labour_permanent[p,w] - model.v_sheep_labour_manager[p,w] + stkpy.stock_labour_perm(model,p) <= 0
    model.con_labour_sheep_perm = pe.Constraint(model.s_labperiods, ['perm'], rule = labour_sheep_perm, doc='link between labour supply and requirement by sheep jobs for perm labour sources')

    ##labour sheep - can be done by manager
    try:
        model.del_component(model.con_labour_sheep_manager_index_1)
        model.del_component(model.con_labour_sheep_manager_index)
        model.del_component(model.con_labour_sheep_manager)
    except AttributeError:
        pass
    def labour_sheep_manager(model,p,w):
        return  - model.v_sheep_labour_manager[p,w] + stkpy.stock_labour_manager(model,p)   <= 0
    model.con_labour_sheep_manager = pe.Constraint(model.s_labperiods, ['mngr'], rule = labour_sheep_manager, doc='link between labour supply and requirement by sheep jobs for manager labour sources')

    #######################################
    #stubble & nap consumption at harvest #
    #######################################
    try:
        # model.del_component(model.con_harv_stub_nap_cons_index)
        model.del_component(model.con_harv_stub_nap_cons)
    except AttributeError:
        pass
    def harv_stub_nap_cons(model,f):
        if any(model.p_nap_prop[f] or model.p_harv_prop[f,k] for k in model.s_crops):
            return sum(-paspy.pas_me(model,v,f) + sum(model.p_harv_prop[f,k]/(1-model.p_harv_prop[f,k]) * model.v_stub_con[v,f,k,s] * model.p_stub_md[f,k,s] for k in model.s_crops for s in model.s_stub_cat)
                    +  model.p_nap_prop[f]/(1-model.p_nap_prop[f]) * paspy.nappas_me(model,v,f) for v in model.s_feed_pools) <= 0
        else:
            return pe.Constraint.Skip
    model.con_harv_stub_nap_cons = pe.Constraint(model.s_feed_periods, rule = harv_stub_nap_cons, doc='limit stubble and nap consumption in the period harvest occurs')

    ######################
    #stubble             #
    ###################### 
  
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
   
    ##links crop sow req with mach sow provide - no p set because model can optimise crop sowing time
    try:
        model.del_component(model.con_cropsow_index)
        model.del_component(model.con_cropsow)
    except AttributeError:
        pass
    def cropsow_link(model,k,l):
        if type(crppy.cropsow(model,k,l)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no crop is being sown on given rotation
        else:
            return sum(-model.v_seeding_crop[p,k,l] for p in model.s_labperiods) + crppy.cropsow(model,k,l)  <= 0
    model.con_cropsow = pe.Constraint(model.s_crops, model.s_lmus, rule = cropsow_link, doc='link between mach sow provide and rotation crop sow require')
   
    ##links pasture sow req with mach sow provide - requires a p set because the timing of sowing pasture is not optimisable (pasture sowing can occur in any period so the user specifies the periods when a given pasture must be sown)
    try:
        model.del_component(model.con_passow_index)
        model.del_component(model.con_passow)
    except AttributeError:
        pass
    def passow_link(model,p,k,l):
        if type(paspy.passow(model,p,k,l)) == int: #if crop sow param is zero this will be int (can't do if==0 because when it is not 0 it is a complex pyomo object which can't be evaluated)
            return pe.Constraint.Skip #skip constraint if no pasture is being sown
        else:
            return -model.v_seeding_pas[p,k,l]  + paspy.passow(model,p,k,l) <= 0
    model.con_passow = pe.Constraint( model.s_labperiods, model.s_landuses, model.s_lmus, rule = passow_link, doc='link between mach sow provide and rotation pas sow require')

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
        return -crppy.rotation_yield_transfer(model,g,k) + macpy.late_seed_penalty(model,g,k) + sum(model.v_sup_con[k,g,v,f]*1000 for v in model.s_feed_pools for f in model.s_feed_periods)\
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
        return -macpy.ha_pasture_crop_paddocks(model,f,l) * model.p_poc_con[f,l] + sum(model.v_poc[v,f,l] for v in model.s_feed_pools) <=0
    model.con_poc_available = pe.Constraint(model.s_feed_periods, model.s_lmus, rule=poc, doc='constraint between poc available and consumed')

    ######################
    #  ME                #
    ######################
    try:
        model.del_component(model.con_me_index)
        model.del_component(model.con_me)
    except AttributeError:
        pass
    def me(model,f,v):
        return -paspy.pas_me(model,v,f) - paspy.nappas_me(model,v,f) - suppy.sup_me(model,v,f) - stubpy.stubble_me(model,v,f) + stkpy.stock_me(model,v,f) <=0
    model.con_me = pe.Constraint(model.s_feed_periods, model.s_feed_pools, rule=me, doc='constraint between me available and consumed')

    ######################
    #Vol                 #
    ######################
    try:
        model.del_component(model.con_vol_index)
        model.del_component(model.con_vol)
    except AttributeError:
        pass
    def vol(model,f,v):
        return paspy.pas_vol(model,v,f) + suppy.sup_vol(model,v,f) + stubpy.stubble_vol(model,v,f) - stkpy.stock_pi(model,v,f) <=0
    model.con_vol = pe.Constraint(model.s_feed_periods, model.s_feed_pools, rule=vol, doc='constraint between me available and consumed')

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
            the debit and credit carried over is multiplied by j because there is no carry over in the first period (there may be a better way to do it though)
            Carryover basically represents interest free cash at the start of the year. It requires cash from ND and provides in JF. 

        '''
        c = sinp.general['cashflow_periods']
        ##j becomes a list which has 0 as first value and 1 after that. this is then indexed by i and multiplied by previous periods debit and credit.
        ##this means the first period doesn't include the previous debit or credit (because it doesn't exist, because it is the first period) 
        j = [1] * len(c)
        j[0] = 0
        return (-yield_income(model,c[i]) + crppy.rotation_cost(model,c[i]) + labpy.labour_cost(model,c[i]) + macpy.mach_cost(model,c[i]) + suppy.sup_cost(model,c[i]) + model.p_overhead_cost[c[i]]
                - stkpy.stock_cashflow(model,c[i])
                - model.v_debit[c[i]] + model.v_credit[c[i]]  + model.v_debit[c[i-1]] * fin.debit_interest() - model.v_credit[c[i-1]] * fin.credit_interest() * j[i] #mul by j so that credit in ND doesnt provide into JF otherwise it will be unbounded because it will get interest
                ) <= 0

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
    model.con_dep = pe.Constraint( rule=dep, doc='tallies depreciation from all activities so it can be transferred to objective')
    
    ######################
    #asset               #
    ###################### 
    try:
        model.del_component(model.con_asset)
    except AttributeError:
        pass
    def asset(model):
        return suppy.sup_asset(model) + macpy.mach_asset(model) + stkpy.stock_asset(model) - model.v_asset <=0   
    model.con_asset = pe.Constraint( rule=asset, doc='tallies asset from all activities so it can be transferred to objective to represent ROE')
    
    ######################
    #Min ROE             #
    ###################### 
    try:
        model.del_component(model.con_minroe)
    except AttributeError:
        pass
    def minroe(model):
        return (sum(crppy.rotation_cost(model,c)  + labpy.labour_cost(model,c) + macpy.mach_cost(model,c) + suppy.sup_cost(model,c) for c in model.s_cashflow_periods) + stkpy.stock_cost(model)) * fin.f_min_roe() \
                - model.v_minroe <=0   
    model.con_minroe = pe.Constraint(rule=minroe, doc='tallies total expenditure to ensure minimum roe is met')
    
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
        return model.v_credit[c[i]]-model.v_debit[c[i]] - model.v_dep - model.v_minroe - (model.v_asset * uinp.finance['opportunity_cost_capital'])  #have to include debit otherwise model selects lots of debit to increase credit, hence can't just maximise credit.
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

    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
    
        ##sometimes if there is a bug when solved it is good to write lp here - because the code doesn't run to the other place where lp written
        model.write('Output/test.lp',io_options={'symbolic_solver_labels':True}) #comment this out when not debugging

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
        results = pe.SolverFactory('glpk').solve(model, tee=True) #turn to true for solver output - may be useful for troubleshooting
        return results

    else:
        #todo include some error handling - eg can't run multiple TOL at the same time. Need different stage allocation for each tol. thus could use an if statement to pick the allocation used
        #todo each variable needs to assigned to stage - give error if that does not happen
        #todo can't allocate one variable to multiple stages
        #allocate labour provide variables to the last stage??
        #stage definitions differ for different nodes to the allocation will need to be based on the node as well eg for ealry break labour period 3 might be in stage 2 but for late break it might be in stage 3
        ##specify stages for variables when all variables go into the same stage
        root_vars=['v_quantity_perm[*]','v_quantity_manager[*]']
        root_sets=['FP9']
        stage1_node1_vars=[]
        stage1_node2_vars=[]
        stage1_node1_sets=['FP0', 'FP1','P']
        stage1_node2_sets=['FP0', 'FP1','P']
        stage2_vars=['v_phase_area[*]']
        stage2_sets=['FP2','FP3','FP4','FP5','FP6','FP7','FP8']
        stage3_vars=['v_sell_grain[*]', 'v_credit[*]', 'v_debit[*]', 'v_dep[*]', 'v_asset[*]', 'v_minroe[*]', 'v_buy_grain[*]'] #buy grain may not be in stage 3, i feel like you retain grain for the year ahead without knowing the type of season. but then the model will just counter by altering sale of grain.
        stage3_sets=['FP9']


        for v in model.component_objects(pe.Var,active=True):
            for index in v:
                print(v)
        def pysp_scenario_tree_model_callback():
            # Return a NetworkX scenario tree.
            g = networkx.DiGraph()

            ce1 = 'FirstStageCost'
            g.add_node("Root",
                       cost=ce1,
                       variables=["DevotedAcreage[*]"],
                       derived_variables=[])

            ce2 = 'SecondStageCost'
            g.add_node("BelowAverageScenario",
                       cost=ce2,
                       variables=["QuantitySubQuotaSold[*]",
                                  "QuantitySuperQuotaSold[*]",
                                  "QuantityPurchased[*]"],
                       derived_variables=[])
            g.add_edge("Root","BelowAverageScenario",weight=0.3333)

            g.add_node("AverageScenario",
                       cost=ce2,
                       variables=["QuantitySubQuotaSold[*]",
                                  "QuantitySuperQuotaSold[*]",
                                  "QuantityPurchased[*]"],
                       derived_variables=[])
            g.add_edge("Root","AverageScenario",weight=0.3333)

            g.add_node("AboveAverageScenario",
                       cost=ce2,
                       variables=["QuantitySubQuotaSold[*]",
                                  "QuantitySuperQuotaSold[*]",
                                  "QuantityPurchased[*]"],
                       derived_variables=[])
            g.add_edge("Root","AboveAverageScenario",weight=0.3334)

            return g

        def pysp_instance_creation_callback(scenario_name,node_names):
            instance = model.clone()
            ##stubble
            model.p_fp_transfer.store_values(params['stub'][scenario_name]['per_transfer'])
            model.p_a_req.store_values(params['stub'][scenario_name]['cat_a_st_req'])
            model.p_stub_vol.store_values(params['stub'][scenario_name]['vol'])
            model.p_stub_md.store_values(params['stub'][scenario_name]['md'])
            model.p_harv_prop.store_values(params['stub'][scenario_name]['cons_prop'])

            ##labour
            model.p_perm_hours.store_values(params['lab'][scenario_name]['permanent hours'])
            model.p_perm_supervision.store_values(params['lab'][scenario_name]['permanent supervision'])
            model.p_casual_cost.store_values(params['lab'][scenario_name]['casual_cost'])
            model.p_casual_hours.store_values(params['lab'][scenario_name]['casual hours'])
            model.p_casual_supervision.store_values(params['lab'][scenario_name]['casual supervision'])
            model.p_manager_hours.store_values(params['lab'][scenario_name]['manager hours'])
            model.p_casual_upper.store_values(params['lab'][scenario_name]['casual ub'])
            model.p_casual_lower.store_values(params['lab'][scenario_name]['casual lb'])

            ##labour crop
            model.p_prep_pack.store_values(params['crplab'][scenario_name]['prep_labour'])
            model.p_fert_app_hour_tonne.store_values(params['crplab'][scenario_name]['fert_app_time_t'])
            model.p_fert_app_hour_ha.store_values(params['crplab'][scenario_name]['fert_app_time_ha'])
            model.p_chem_app_lab.store_values(params['crplab'][scenario_name]['chem_app_time_ha'])
            model.p_variable_crop_monitor.store_values(params['crplab'][scenario_name]['variable_crop_monitor'])
            model.p_fixed_crop_monitor.store_values(params['crplab'][scenario_name]['fixed_crop_monitor'])

            ##labour fixed
            model.p_super_labour.store_values(params['labfx'][scenario_name]['super'])
            model.p_bas_labour.store_values(params['labfx'][scenario_name]['bas'])
            model.p_planning_labour.store_values(params['labfx'][scenario_name]['planning'])
            model.p_tax_labour.store_values(params['labfx'][scenario_name]['tax'])

            ##crop
            model.p_rotation_cost.store_values(params['crop'][scenario_name]['rot_cost'])
            model.p_rotation_yield.store_values(params['crop'][scenario_name]['rot_yield'])
            model.p_phasefert.store_values(params['crop'][scenario_name]['fert_req'])

            ##pasture
            model.p_germination.store_values(params['pas'][scenario_name]['p_germination_flrt'])
            model.p_foo_grn_reseeding.store_values(params['pas'][scenario_name]['p_foo_grn_reseeding_flrt'])
            model.p_foo_dry_reseeding.store_values(params['pas'][scenario_name]['p_foo_dry_reseeding_dflrt'])
            model.p_foo_end_grnha.store_values(params['pas'][scenario_name]['p_foo_end_grnha_goflt'])
            model.p_foo_start_grnha.store_values(params['pas'][scenario_name]['p_foo_start_grnha_oflt'])
            model.p_senesce_grnha.store_values(params['pas'][scenario_name]['p_senesce_grnha_dgoflt'])
            model.p_me_cons_grnha.store_values(params['pas'][scenario_name]['p_me_cons_grnha_vgoflt'])
            model.p_dry_mecons_t.store_values(params['pas'][scenario_name]['p_dry_mecons_t_vdft'])
            model.p_volume_grnha.store_values(params['pas'][scenario_name]['p_volume_grnha_goflt'])
            model.p_dry_volume_t.store_values(params['pas'][scenario_name]['p_dry_volume_t_dft'])
            model.p_dry_transfer_t.store_values(params['pas'][scenario_name]['p_dry_transfer_t_ft'])
            model.p_nap.store_values(params['pas'][scenario_name]['p_nap_dflrt'])
            model.p_nap_prop.store_values(params['pas'][scenario_name]['p_harvest_period_prop'])
            model.p_phase_area.store_values(params['pas'][scenario_name]['p_phase_area_flrt'])
            model.p_pas_sow.store_values(params['pas'][scenario_name]['p_pas_sow_plrk'])
            model.p_poc_vol.store_values(params['pas'][scenario_name]['p_poc_vol_f'])

            ##machine
            model.p_seed_days.store_values(params['mach'][scenario_name]['seed_days'])
            model.p_contractseeding_occur.store_values(params['mach'][scenario_name]['contractseeding_occur'])
            model.p_seeding_cost.store_values(params['mach'][scenario_name]['seeding_cost'])
            model.p_contract_seeding_cost.store_values(params['mach'][scenario_name]['contract_seed_cost'])
            model.p_harv_rate.store_values(params['mach'][scenario_name]['harv_rate_period'])
            model.p_harv_hrs_max.store_values(params['mach'][scenario_name]['max_harv_hours'])
            model.p_harv_cost.store_values(params['mach'][scenario_name]['harvest_cost'])
            model.p_contractharv_cost.store_values(params['mach'][scenario_name]['contract_harvest_cost'])
            model.p_yield_penalty.store_values(params['mach'][scenario_name]['yield_penalty'])
            model.p_seeding_grazingdays.store_values(params['mach'][scenario_name]['grazing_days'])

            ##sup feed
            model.p_sup_cost.store_values(params['sup'][scenario_name]['total_sup_cost'])
            model.p_sup_labour.store_values(params['sup'][scenario_name]['sup_labour'])

            ##stock
            model.p_nsires_req.store_values(params['stock'][scenario_name]['p_nsire_req_dams'])
            model.p_nsires_prov.store_values(params['stock'][scenario_name]['p_nsire_prov_sire'])
            model.p_progprov_dams.store_values(params['stock'][scenario_name]['p_progprov_dams'])
            model.p_progprov_offs.store_values(params['stock'][scenario_name]['p_progprov_offs'])
            model.p_numbers_prov_dams.store_values(params['stock'][scenario_name]['p_numbers_prov_dams'])
            model.p_numbers_provthis_dams.store_values(params['stock'][scenario_name]['p_numbers_provthis_dams'])
            model.p_numbers_prov_offs.store_values(params['stock'][scenario_name]['p_numbers_prov_offs'])
            model.p_mei_sire.store_values(params['stock'][scenario_name]['p_mei_sire'])
            model.p_mei_dams.store_values(params['stock'][scenario_name]['p_mei_dams'])
            model.p_mei_offs.store_values(params['stock'][scenario_name]['p_mei_offs'])
            model.p_pi_sire.store_values(params['stock'][scenario_name]['p_pi_sire'])
            model.p_pi_dams.store_values(params['stock'][scenario_name]['p_pi_dams'])
            model.p_pi_offs.store_values(params['stock'][scenario_name]['p_pi_offs'])
            model.p_cashflow_sire.store_values(params['stock'][scenario_name]['p_cashflow_sire'])
            model.p_cashflow_dams.store_values(params['stock'][scenario_name]['p_cashflow_dams'])
            model.p_cashflow_prog.store_values(params['stock'][scenario_name]['p_cashflow_prog'])
            model.p_cashflow_offs.store_values(params['stock'][scenario_name]['p_cashflow_offs'])
            model.p_cost_sire.store_values(params['stock'][scenario_name]['p_cost_sire'])
            model.p_cost_dams.store_values(params['stock'][scenario_name]['p_cost_dams'])
            model.p_cost_offs.store_values(params['stock'][scenario_name]['p_cost_offs'])
            model.p_asset_sire.store_values(params['stock'][scenario_name]['p_assetvalue_sire'])
            model.p_asset_dams.store_values(params['stock'][scenario_name]['p_assetvalue_dams'])
            model.p_asset_offs.store_values(params['stock'][scenario_name]['p_assetvalue_offs'])
            model.p_lab_anyone_sire.store_values(params['stock'][scenario_name]['p_labour_anyone_sire'])
            model.p_lab_perm_sire.store_values(params['stock'][scenario_name]['p_labour_perm_sire'])
            model.p_lab_manager_sire.store_values(params['stock'][scenario_name]['p_labour_manager_sire'])
            model.p_lab_anyone_dams.store_values(params['stock'][scenario_name]['p_labour_anyone_dams'])
            model.p_lab_perm_dams.store_values(params['stock'][scenario_name]['p_labour_perm_dams'])
            model.p_lab_manager_dams.store_values(params['stock'][scenario_name]['p_labour_manager_dams'])
            model.p_lab_anyone_offs.store_values(params['stock'][scenario_name]['p_labour_anyone_offs'])
            model.p_lab_perm_offs.store_values(params['stock'][scenario_name]['p_labour_perm_offs'])
            model.p_lab_manager_offs.store_values(params['stock'][scenario_name]['p_labour_manager_offs'])
            model.p_infra_sire.store_values(params['stock'][scenario_name]['p_infrastructure_sire'])
            model.p_infra_dams.store_values(params['stock'][scenario_name]['p_infrastructure_dams'])
            model.p_infra_offs.store_values(params['stock'][scenario_name]['p_infrastructure_offs'])
            model.p_dse_sire.store_values(params['stock'][scenario_name]['p_dse_sire'])
            model.p_dse_dams.store_values(params['stock'][scenario_name]['p_dse_dams'])
            model.p_dse_offs.store_values(params['stock'][scenario_name]['p_dse_offs'])
            model.p_cost_purch_sire.store_values(params['stock'][scenario_name]['p_purchcost_sire'])




            # seed_random(scenario_name)
            # # Note: Must sort the iteration order so that
            # #       data is generated deterministically
            # #       across runs. Iteration over dict and set
            # #       is very fickle in Python3.x
            # for key in sorted(RandomYield.keys()):
            #     instance.Yield[key] = random.normalvariate(*RandomYield[key])

            return instance

    concrete_tree = pysp_scenario_tree_model_callback()
    stsolver = rapper.StochSolver(None,tree_model=concrete_tree,fsfct=pysp_instance_creation_callback)
    ef_sol = stsolver.solve_ef('glpk',tee=True)
    print(ef_sol.solver.termination_condition)
    obj = stsolver.root_E_obj()
    print("Expecatation take over scenarios=",obj)
    for varname,varval in stsolver.root_Var_solution():  # doctest: +SKIP
        print(varname,str(varval))

    ##saves file to csv
    csvw.write_csv_soln(stsolver.scenario_tree,"solution")
    ##saves file to json
    jsonw.JSONSolutionWriter.write('',stsolver.scenario_tree,
                                   'ef')  # i don't know what the first arg does?? it needs to exist but can put any string without changing output

    ##load json back in
    with open('ef_solution.json') as f:
        data = json.load(f)

    
