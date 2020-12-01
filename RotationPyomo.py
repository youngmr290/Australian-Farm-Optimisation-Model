# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 10:50:11 2020

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete incase we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem
     

@author: young
"""

#python modules
from pyomo.environ import *

#MUDAS modules
import RotationPhases as rps
from CreateModel import *

def rotation_precalcs(params, report):
    rps.rot_params(params)
    rps.report_landuses_phases(report)
    
def rotationpyomo(params):
    ####################
    #define parameters #
    ####################
    try:
        model.del_component(model.p_area)
    except AttributeError:
        pass
    model.p_area = Param(model.s_lmus, initialize=params['lmu_area'], doc='available area on farm for each soil')
    

    ##only build this param if it doesn't exist already ie the rotation link never changes
    try:
        if model.p_rotphaselink:
            pass
    except AttributeError:
        model.p_rotphaselink= Param(params['rot_con1'].keys(), initialize=params['rot_con1'], doc='link between rotation history and current rotation')
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #local constraints
    #######################################################################################################################################################
    #######################################################################################################################################################
    ######################
    #rotation constraints#
    ######################
    ##only build this con if it doesn't exist already ie the rotation link never changes
    try:
        if model.con_rotationcon1:
            pass
    except AttributeError:
        def rot_phase_link(model,l,h):
            ##skip constraint if the history is not used by any of the rotations. This only happens for the continuous rotations because there is not constraint for them because they provide and require themselves, so if no other rotation use the continuous history and error is thrown because constraint is built from nothing. But cant remove continuos histories becasue they can be used by other roations eg AAAAAa has history of AAAAA this history can be used by AAAAAb rotation.
            if any(params['hist']==h):
                return sum(model.v_phase_area[r,l]*model.p_rotphaselink[r,h] for r in model.s_phases if ((r,)+(h,)) in model.p_rotphaselink)<=0
            else:
                return Constraint.Skip
        model.con_rotationcon1 = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_phase_link, doc='rotation phases constraint')

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

#######################################################################################################################################################
#######################################################################################################################################################
#Main rotation param and constraint - only needs to be built once
#######################################################################################################################################################
#######################################################################################################################################################

# try:
#     model.del_component(model.p_rotphaselink2)
#     model.del_component(model.p_rotphaselink2_index)
# except AttributeError:
#     pass
# model.p_rotphaselink2= Param(rps.rot_con2.keys(), initialize=rps.rot_con2, doc='link between rotation history2 and current rotation')
   
######################
#rotation constraints#
######################
##build and define rotation constraint 1 - used to ensure that the each rotation provides and requires one or more histories
##alternative method (a1 - michael)

# ##build and define rotation constraint 2 - used to ensure that the history provided by a rotation is used by another rotation (because one rotation can provide multiple histories)
# try:
#     model.del_component(model.con_rotationcon2)
#     model.del_component(model.con_rotationcon2_index)
# except AttributeError:
#     pass
# def rot_phase_link2(model,l,h):
#     return sum(model.v_phase_area[r,l]*model.p_rotphaselink2[r,h] for r in model.s_phases if ((r,)+(h,)) in model.p_rotphaselink2)<=0
# model.con_rotationcon2 = Constraint(model.s_lmus, model.s_rotconstraints2, rule=rot_phase_link2, doc='rotation phases constraint2')





