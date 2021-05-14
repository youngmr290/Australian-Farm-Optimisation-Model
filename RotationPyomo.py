"""
author: young

"""

#python modules
from pyomo.environ import *

#AFO modules
import RotationPhases as rps
from CreateModel import *

def rotation_precalcs(params, report):
    '''
    Call rotation precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    rps.rot_params(params)
    rps.landuses_phases(params,report)
    
def rotationpyomo(params):
    ##################################################
    #variables that need to be built each iteration #
    ##################################################
    try:
        model.del_component(model.v_root_hist)
        model.del_component(model.v_root_hist_index)
    except AttributeError:
        pass
    if not pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1: #only needed for dsp version.
        model.v_root_hist = Var(model.s_rotconstraints, model.s_lmus, bounds=(0,None),doc='rotation history provided in the root stage')

    ####################
    #define parameters #
    ####################
    try:
        model.del_component(model.p_area)
    except AttributeError:
        pass
    model.p_area = Param(model.s_lmus, initialize=params['lmu_area'], doc='available area on farm for each soil')
    
    try:
        model.del_component(model.p_landuse_area)
        model.del_component(model.p_landuse_area_index)
    except AttributeError:
        pass
    model.p_landuse_area = Param(model.s_phases, model.s_landuses, initialize=params['phases_rk'], doc='landuse in each phase')


    # ##only build this param if it doesn't exist already ie the rotation link never changes
    # try:
    #     if model.p_rotphaselink:
    #         pass
    # except AttributeError:
    #     model.p_rotphaselink= Param(params['rot_con1'].keys(), initialize=params['rot_con1'], doc='link between rotation history and current rotation')
    
    ##only build this param if it doesn't exist already ie the rotation link never changes
    try:
        if model.p_hist_prov and model.p_hist_req:
            pass
    except AttributeError:
        model.p_hist_prov = Param(params['hist_prov'].keys(), initialize=params['hist_prov'], default=0, doc='history provided by  each rotation') #use keys instead of sets to reduce size of param
        model.p_hist_req = Param(params['hist_req'].keys(), initialize=params['hist_req'], default=0, doc='history required by  each rotation') #use keys instead of sets to reduce size of param

    #######################################################################################################################################################
    #######################################################################################################################################################
    #local constraints
    #######################################################################################################################################################
    #######################################################################################################################################################
    ######################
    #rotation constraints#
    ######################

    #todo i might not need the root hist variable and whatnot with new season structure

    '''
    For steady state model each rotation requires and provides a rotation history.
    For DSP the process is slight more complicated because the history that provides the rotations must be the same for
     each season. because each season needs to start in a common place. Therefore a history variable is created which
     can be assigned to the root stage. This means an additional constraint is required.
    Note: the DSP structure will work fine for steady state however just increases the size, but for debugging you can 
     use the DSP structure with the steady state model (just comment out the steady state stuff)'''

    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1:
        try:
            model.del_component(model.con_root_hist) #if running the steady state model we don't need the dsp rotation constraints
            model.del_component(model.con_root2rotation) #if running the steady state model we don't need the dsp rotation constraints
        except AttributeError:
            pass

        ##only build this con if it doesn't exist already ie the rotation link never changes
        try:
            if model.con_rotationcon1:
                pass
        except AttributeError:
            def rot_phase_link(model,l,h):
                return sum(model.v_phase_area[r,l]*model.p_hist_prov[r,h] for r in model.s_phases if ((r,)+(h,)) in params['hist_prov'].keys()) \
                           + sum(model.v_phase_area[r,l]*model.p_hist_req[r,h] for r in model.s_phases if ((r,)+(h,)) in params['hist_req'].keys())<=0
            model.con_rotationcon1 = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_phase_link, doc='rotation phases constraint')

    else:
        try:
            model.del_component(model.con_rotationcon1)  # if running the dsp model we don't need the steady state rotation constraints
        except AttributeError:
            pass

        ##only build this con if it doesn't exist already ie the rotation link never changes
        try:
            if model.con_root_hist and model.con_root2rotation:
                pass
        except AttributeError:

            ##constraint for history provided to history root. This is only required in the stochastic model so that each season starts from a common place.
            def rot_hist(model,l,h):
                return model.v_root_hist[h,l] + sum(model.v_phase_area[r,l]*model.p_hist_prov[r,h]
                            for r in model.s_phases if ((r,)+(h,)) in params['hist_prov'].keys())<=0
            model.con_rot_hist = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_hist, doc='constraint between rotation history provided and root history')

            ##constraint for history provided to history root. This is only required in the stochastic model so that each season starts from a common place.
            def rot_phase_link(model,l,h):
                return - model.v_root_hist[h,l] + sum(model.v_phase_area[r,l]*model.p_hist_req[r,h]
                            for r in model.s_phases if ((r,)+(h,)) in params['hist_req'].keys())<=0
            model.con_root2rotation = Constraint(model.s_lmus, model.s_rotconstraints, rule=rot_phase_link, doc='constraint between rotation history root and rotation')


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





