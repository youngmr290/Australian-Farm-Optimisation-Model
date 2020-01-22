# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:07:19 2019

module: labour pyomo module - contains pyomo params, variables and constraits

Version Control:
Version     Date        Person  Change
1.1         11Dec19     M        update pyomo constrain rule structure - increase constrain building time by using smart (sparse) set indexing
1.2         16Jan20     M        removed yield from cashflow con - it is now seperate so it can be combined with yield penalty and crop grazing yield penalty before being multiplied by price.

Known problems:
Fixed   Date    ID by   Problem
     

@author: young
"""

#python modules
from pyomo.environ import *
#import pandas as pd
#import numpy as np
import timeit

#MUDAS modules
import Crop as crp
from CreateModel import *

print('Status:  running croppyomo')

def croppyomo_local():
    
    #########
    #param  #
    #########    
    try:
        model.del_component(model.p_rotation_cost)
    except AttributeError:
        pass
    model.p_rotation_cost = Param(model.s_phases,model.s_lmus,model.s_cashflow_periods, initialize=crp.rot_cost(),default=0, doc='total cashflow for 1 unit of rotation')
   
    try:
        model.del_component(model.p_rotation_yield)
    except AttributeError:
        pass
    model.p_rotation_yield = Param(model.s_phases, model.s_lmus, initialize=crp.rot_yield(), default = 0.0, doc='grain production for all crops for 1 unit of rotation')
    
    try:
        model.del_component(model.p_grain_price)
    except AttributeError:
        pass
    model.p_grain_price = Param(model.s_cashflow_periods, model.s_crops,  initialize=crp.grain_price(),default = 0.0, doc='farm gate price per tonne of each grain')
    try:
        model.del_component(model.stubble)
    except AttributeError:
        pass
    model.p_stubble = Param(model.s_crops, initialize=crp.stubble_production(), default = 0.0, doc='stubble produced / kg grain harvested')
    #model.rotation_cashflow.pprint()
    
    
    
    
    
      
    
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #local constraints
    #######################################################################################################################################################
    #######################################################################################################################################################
    


#######################################################################################################################################################
#######################################################################################################################################################
#functions used in global constraints
#######################################################################################################################################################
#######################################################################################################################################################

##############
#yield       #
##############
##total grain transfer for each crop, seperated so it can be combined with untimely sowing and crop grazing penalty before converting to cashflow 
## slightly more complicated because i have split the rotation set into history and current crop - this is so i get just the grain transfer for each crop which is compatible with yield penalty activities.

def rotation_yield_transfer(model,k):
    return sum(sum(model.p_rotation_yield[h,k,l]*model.v_phase_area[h,k,l] for h in model.s_rotconstraints if ((h)+(k,)+(l,)) in model.p_rotation_yield and model.p_rotation_yield[h,k,l] != 0)for l in model.s_lmus) #+ model.x[k] >=0 #

# model.x = Var(model.s_crops, bounds=(0,None), doc='delets - used for testing')
# model.j = Constraint(model.s_crops, rule=rotation_yield_transfer, doc='')
# model.j.pprint()

#####################################
# functions used to define cashflow #
#####################################

'''
To add:
    - chem cost
    - mach requirement ie sow and harv
    -stub cons
    -seed cost
    -insurance
    -pas provide by each phase
'''

def rotation_cost(model,c):
    return sum(sum(model.p_rotation_cost[r,l,c]*model.v_phase_area[r,l] for r in model.s_phases if model.p_rotation_cost[r,l,c] != 0) for l in model.s_lmus )#+ model.x[c] >=0 #0.10677s
#model.j = Constraint(model.cashflow_periods, rule=rotation_cashflow, doc='')
   
    


















# ########
    # # Area #
    # ########
    # #area of rotation on a given soil can't be more than the amount on that soil available on farm
    # def area_rule(model, l):
    #   return sum(model.area_phase[r,l] for r in model.phases) <= model.area[l] 
    # model.area_con = Constraint(model.lmus, rule=area_rule, doc='rotation area constraint')
    
    # #####################
    # # lo bound rotation #
    # #####################
    # #area of rotation on a given soil can't be more than the amount on that soil available on farm
    # def rot_lo_bound(model, r1,r2,r3,r4,r5, l):
    #   return model.area_phase[r1,r2,r3,r4,r5,l] >= model.lo[r1,r2,r3,r4,r5] 
    # model.rot_bound = Constraint(model.phases, model.lmus, rule=rot_lo_bound, doc='lo bound for the number of each phase')
    
    # ##build and define rotation constraint 1 - used to ensure that the each rotation provides and requires one or more histories
    # ##alternative method (a1 - michael)
    # def rot_phase_link(model,l,h1,h2,h3,h4):
    #     return sum(model.area_phase[r,l]*model.rot_phase_link[r,h1,h2,h3,h4] for r in model.phases if ((r)+(h1,)+(h2,)+(h3,)+(h4,)) in model.rot_phase_link)<=0
    # model.rotation_con1 = Constraint(model.lmus,model.rot_constraints, rule=rot_phase_link, doc='rotation phases constraint')
    
    # # ##build and define rotation constraint 2 - used to ensure that the history provided by a rotation is used by another rotation (because one rotation can provide multiple histories)
    # # def rot_phase_link2(model,l,h1,h2,h3,h4):
    # #     return sum(model.area_phase[r,l]*model.rot_phase_link2[r,h1,h2,h3,h4] for r in model.phases if ((r)+(h1,)+(h2,)+(h3,)+(h4,)) in model.rot_phase_link2)<=0
    # # model.rotation_con2 = Constraint(model.lmus,model.rot_constraints2, rule=rot_phase_link2, doc='rotation phases constraint')


#    model.del_component(model.j)
#print(timeit.timeit(test3,number=10)/10)
#model.j.pprint()
    # return rotation_income(model,c)#- fert_cost(model,c)#- fert_app_cost(model,c) - stubble_handling_cost(model,c) #



# def rot_phase_link(model,l,h1):
#     print(h1)
#     return sum(model.area_phase[r,l]*model.rot_phase_link[r,h1] for r in model.phases if ((r)+(h1,)) in model.rot_phase_link)<=0
# model.jj = Constraint(model.lmus,model.rot_constraints, rule=rot_phase_link, doc='')

# model.jj.pprint()
#model.del_component(model.jj)
#model.del_component(model.jj_index)









#old method below:

#'''
##constrains
#'''
#########
## Area #
#########
##area of rotation on a given soil can't be more than the amount on that soil available on farm
#def area_rule(model, l):
#  return sum(model.area_phase[l,r] for r in model.phases) <= model.area[l] 
#model.area_con = Constraint(model.lmus, rule=area_rule, doc='rotation area constraint')
#
######################################
## functions used to define cashflow #
######################################
##cost of fert application - 1) per tonne 2) per ha 
#def fert_app_cost(model,c):
#    per_t_cost = sum(sum(sum(model.phasefert[r,l,n]*model.area_phase[l,r]*(model.fert_app_cost_tonne[c,n]/1000)  for r in model.phases)for l in model.lmus)for n in model.fert_type ) 
#    per_ha_cost = sum(sum(model.area_phase[l,r]*(model.fert_app_cost_ha[r,c,l]/1000) for r in model.phases) for l in model.lmus)   
#    return per_t_cost + per_ha_cost 
#
##cost of fert for rotations for a give cashflow period (c), 
#def fert_cost(model,c):
#    return sum(sum(sum(model.phasefert[r,l,n]*model.area_phase[l,r] for r in model.phases) for l in model.lmus)*(model.fert_type_cost[c,n]/1000) for n in model.fert_type ) 
#
##cost of stubble handling for each phase for a give cashflow period (c), 
#def stubble_handling_cost(model,c):
#    return sum(sum(model.stubble_handling_prob[r,l]*model.area_phase[l,r]* model.stubble_handling_cost[c] for r in model.phases)for l in model.lmus) 
#
##costs of rotations excluding fert for a give cashflow period (c)
##def rotation_cost(model,c):
##    return sum(sum(model.rotationcost[i,j,c]*model.x[i,j] for j in model.rotations)for i in model.lmus) 
##rotation income (grain income) for a give cashflow period (c)
#def rotation_income(model,c):
#    return sum(sum(model.rotationyield[r,l]*model.area_phase[l,r]*(model.grainincome[r,c]/1000) for r in model.phases)for l in model.lmus) 
#
##overall cashflow function (sums up functions above, this could all just be one but i am thinking this looks more simple)
#def rotation_cashflow(model,c):
#    return rotation_income(model,c)- fert_cost(model,c)- fert_app_cost(model,c) - stubble_handling_cost(model,c) #+ model.x[c] >=0
##model.rotation_con = Constraint(model.cashflow_periods, rule=rotation_cashflow, doc='')
#

    # #calling a function multiple times takes time. call it once and assign result to a unique variable. 
    # #local variables are easier for pyton to locate
    # # a=crp.phase_yields()
    # # b=crp.fert()
    # # c=crp.stubble_handling_prob()
    # # d=crp.grain_price()
    # #e=crp.fert_cost()
    # # f=mac.fert_app_t()
    # # g=mac.fert_app_ha()
    # h=crp.rot_phase_mps()
    # g=crp.rot_phase_mps2()
    # #'''
    # ##define parameters



# #cost of fert application - 1) per tonne 2) per ha 
# def fert_app_cost(model,c):
#     #per_t_cost = sum(model.phasefert[rln[:-2],rln[-2],rln[-1]]*model.area_phase[rln[:-2],rln[-2]]*(model.fert_app_cost_tonne[c,rln[-1]]/1000) for rln in model.phasefert if ((c,)+(rln[-1],)) in model.fert_type_cost) 
#     per_ha_cost = sum(model.area_phase[rlc[:-2],rlc[-2]]*(model.fert_app_cost_ha[rlc[:-2],rlc[-2],c]/1000) for rlc in model.fert_app_cost_ha if ((rlc[-1],)+(rlc[-1],)+(c,)) in model.fert_app_cost_ha) 
#     return per_t_cost + per_ha_cost 

#cost of fert for rotations for a give cashflow period (c), 
#def tesft():
    
# def fert_cost(model,c):
#     return sum(model.rotation_cost[rlc[:-2],rlc[-2],c]*model.area_phase[rlc[:-2],rlc[-2]] for rlc in model.rotation_cost  in model.rotation_cost if (rlc[:-2]+(rlc[-2],)+(c,)))+ model.x[c] >=0
# model.rotation_con = Constraint(model.cashflow_periods, rule=fert_cost, doc='')
#model.rotation_con.pprint()
#     model.del_component(model.rotation_con)
# print(timeit.timeit(tesft,number=10)/10)

# #cost of stubble handling for each phase for a give cashflow period (c), 
# def stubble_handling_cost(model,c):
#     return sum(model.stubble_handling_prob[rl[:-1],rl[-1]]*model.area_phase[rl[:-1],rl[-1]]*(model.stubble_handling_cost[c]) for rl in model.stubble_handling_prob if c in model.stubble_handling_cost) 


#rotation income (grain income) for a give cashflow period (c)
#def test2():
# def rotation_income(model,c):
#     return sum(model.rotationyield[rl[:-1],rl[-1]]*model.area_phase[rl[:-1],rl[-1]]*(model.grainincome[c,rl[-2]]/1000) for rl in model.rotationyield if ((c,)+(rl[-2],)) in model.grainincome)#+ model.x[c] >=0 #0.0677s
  #  model.rotation_con2 = Constraint(model.cashflow_periods, rule=rotation_income, doc='')
   # model.del_component(model.rotation_con2)
#print(timeit.timeit(test2,number=10)/10)
#model.rotation_cashflow.pprint()
#total costs and income from rotations - transfered to core model

# def test2():
#     def rotation_cashflow(model,c):
#         return sum(model.rotation_cashflow[rl[:-2],rl[-2],c]*model.area_phase[rl[:-2],rl[-2]] for rl in model.rotation_cashflow if (rl[:-2]+(rl[-2],)+(c,)) in model.rotation_cashflow)+ model.x[c] >=0 #0.12677s
#     model.j = Constraint(model.cashflow_periods, rule=rotation_cashflow, doc='')
#     model.del_component(model.j)
# print(timeit.timeit(test2,number=10)/10)
    

    # # model.rotation_cost= Param(crp.f.keys(), initialize=crp.f)
    # #model.rotation_cost.pprint()
    # # model.rotation_cashflow = Param(crp.rot_cashflow().keys(), initialize=crp.rot_cashflow(), doc='total cashflow for 1 unit of rotation')
    # # model.phasefert = Param(b.keys(), initialize=b, doc='fert required by 1 unit of phase')
    # # model.stubble_handling_prob = Param(c.keys(), initialize=c, doc='probability of each phase that requires stubble handling')
    # # model.stubble_handling_cost = Param(model.cashflow_periods, initialize=mac.stubble_cost_ha(),default = 0.0, doc='cost to handle 1ha of stubble')
    # model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    # model.lo = Param(model.phases, initialize=crp.lo_bound, doc='lo bound of the number of ha of rot_phase') 
    
    # #model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    # # model.grainincome = Param(d.keys(), initialize=d, doc='farm gate price per tonne of each grain')
    # # model.fert_type_cost = Param(e.keys(), initialize=e, doc='price per tonne of each fert')
    # # model.fert_app_cost_tonne = Param(f.keys(), initialize= f, doc='cost of fert application per tonne of each fert (filling up and driving to paddock cost)')
    # # model.fert_app_cost_ha = Param(g.keys(), initialize= g, doc='cost of fert application per ha of each fert (driving around paddock cost)')
    # model.rot_phase_link= Param(h.keys(), initialize=h, doc='link between rotation history and current rotation')
    # model.rot_phase_link2= Param(g.keys(), initialize=g, doc='link between rotation history2 and current rotation')
    # #model.rot_phase_link.pprint() #need to fix the key for the dict so that it gets the correct set indexing in the param
    # '''
    # #define parameters, this method of defining params is 20-30% faster, doesn't need to calc the .keys method as above.
    # '''
    
    #requirment parameters for rotations are read in from csv but to complete model i have hand inputted some parameters
    #    model.rotationyield = Param(model.phases, model.lmus, initialize=crp.phase_yields(), default = 0.0, doc='grain production for all crops for 1 unit of rotation')
    #    model.phasefert = Param(model.phases, model.lmus, model.fert_type, initialize=crp.fert(), default = 0.0, doc='fert required by 1 unit of phase')
    #    model.stubble_handling_prob = Param(model.phases, model.lmus, initialize=crp.stubble_handling_prob(), default = 0.0, doc='probability of each phase that requires stubble handling')
    #    model.stubble_handling_cost = Param(model.cashflow_periods, initialize=mac.stubble_cost_ha(),default = 0.0, doc='cost to handle 1ha of stubble')
    #    model.area = Param(model.lmus, initialize=gi.general_input['lmu_area'], doc='available area on farm for each soil') #alternate way to initialise a parameter
    #    model.grainincome = Param(model.cashflow_periods, model.crops, initialize=crp.grain_price(),default = 0.0, doc='farm gate price per tonne of each grain')
    #    model.fert_type_cost = Param(model.cashflow_periods, model.fert_type, initialize=crp.fert_cost(), default = 0.0, doc='price per tonne of each fert')
    #    model.fert_app_cost_tonne = Param(model.cashflow_periods, model.fert_type, initialize= mac.fert_app_t(), default = 0.0, doc='cost of fert application per tonne of each fert (filling up and driving to paddock cost)')
    #    model.fert_app_cost_ha = Param(model.phases, model.cashflow_periods, model.lmus, initialize= mac.fert_app_ha(), default = 0.0, doc='cost of fert application per ha of each fert (driving around paddock cost)')
    #    model.stubble = Param(model.crops, initialize=crp.stubble_production(), default = 0.0, doc='stubble produced / kg grain harvested')
    