# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:50:11 2020

Version Control:
Version     Date        Person  Change

Known problems:
Fixed   Date    ID by   Problem
     

@author: young
"""

#python modules
from pyomo.environ import *

#MUDAS modules
import RotationPhases as rps
import PropertyInputs as pinp
from CreateModel import *

print('Status:  running rotationpyomo')

def rotationpyomo():
    
    ##calling a function multiple times takes time. call it once and assign result to a unique variable. 
    ##local variables are easier for pyton to locate
    

    ####################
    #define parameters #
    ####################
    
    try:
        model.del_component(model.p_area)
    except AttributeError:
        pass
    model.p_area = Param(model.s_lmus, initialize=pinp.general['lmu_area'].squeeze().to_dict(), doc='available area on farm for each soil')
    
    try:
        model.del_component(model.p_lo)
    except AttributeError:
        pass
    model.p_lo = Param(model.s_phases, initialize=rps.lo_bound, doc='lo bound of the number of ha of rot_phase') 
    
    try:
        model.del_component(model.p_rotphaselink)
        model.del_component(model.p_rotphaselink_index)
    except AttributeError:
        pass
    model.p_rotphaselink= Param(rps.rot_con1.keys(), initialize=rps.rot_con1, doc='link between rotation history and current rotation')
    try:
        model.del_component(model.p_rotphaselink2)
        model.del_component(model.p_rotphaselink2_index)
    except AttributeError:
        pass
    model.p_rotphaselink2= Param(rps.rot_con2.keys(), initialize=rps.rot_con2, doc='link between rotation history2 and current rotation')
    
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #local constraints
    #######################################################################################################################################################
    #######################################################################################################################################################
    
    ########
    # Area #
    ########
    #area of rotation on a given soil can't be more than the amount on that soil available on farm
    try:
        model.del_component(model.con_area)
    except AttributeError:
        pass
    def area_rule(model, l):
      return sum(model.v_phase_area[r,l] for r in model.s_phases) <= model.p_area[l] 
    model.con_area = Constraint(model.s_lmus, rule=area_rule, doc='rotation area constraint')
    
    ######################
    #rotation constraints#
    ######################
    ##build and define rotation constraint 1 - used to ensure that the each rotation provides and requires one or more histories
    ##alternative method (a1 - michael)
    try:
        model.del_component(model.con_rotationcon1)
        model.del_component(model.con_rotationcon1_index)
    except AttributeError:
        pass
    # def rot_phase_link(model,l,h1,h2,h3,h4):
    #     return sum(model.v_phase_area[r,l]*model.p_rotphaselink[r,h1,h2,h3,h4] for r in model.s_phases if ((r)+(h1,)+(h2,)+(h3,)+(h4,)) in model.p_rotphaselink)<=0
    # model.con_rotationcon1 = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_phase_link, doc='rotation phases constraint')
    def rot_phase_link(model,l,h):
        return sum(model.v_phase_area[r,l]*model.p_rotphaselink[r,h] for r in model.s_phases if ((r,)+(h,)) in model.p_rotphaselink)<=0
    model.con_rotationcon1 = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_phase_link, doc='rotation phases constraint')
    
    ##build and define rotation constraint 2 - used to ensure that the history provided by a rotation is used by another rotation (because one rotation can provide multiple histories)
    try:
        model.del_component(model.con_rotationcon2)
        model.del_component(model.con_rotationcon2_index)
    except AttributeError:
        pass
    def rot_phase_link2(model,l,h):
        return sum(model.v_phase_area[r,l]*model.p_rotphaselink2[r,h] for r in model.s_phases if ((r,)+(h,)) in model.p_rotphaselink2)<=0
    model.con_rotationcon2 = Constraint(model.s_lmus, model.s_rotconstraints2, rule=rot_phase_link2, doc='rotation phases constraint2')
    # model.con_rotationcon2.pprint()

    #####################
    # lo bound rotation #
    #####################
    #area of rotation on a given soil can't be more than the amount on that soil available on farm
    try:
        model.del_component(model.con_rotation_lobound)
        model.del_component(model.con_rotation_lobound_index)
    except AttributeError:
        pass
    def rot_lo_bound(model, r, l):
      return model.v_phase_area[r,l] >= model.p_lo[r] 
    model.con_rotation_lobound = Constraint(model.s_phases, model.s_lmus, rule=rot_lo_bound, doc='lo bound for the number of each phase')
    

#######################################################################################################################################################
#######################################################################################################################################################
#variables - don't need to be included in the function that is re-run
#######################################################################################################################################################
#######################################################################################################################################################
try:
    model.del_component(model.v_phase_area)
    model.del_component(model.v_phase_area_index)
except AttributeError:
    pass
##Amount of each phase on each soil, Positive Variable.
model.v_phase_area = Var(model.s_phases, model.s_lmus, bounds=(0,None), doc='number of ha of each phase')



