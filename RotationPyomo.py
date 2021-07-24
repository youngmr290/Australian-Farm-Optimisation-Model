"""
author: young

"""

#python modules
from pyomo.environ import *
import numpy as np

#AFO modules
import RotationPhases as rps
import PropertyInputs as pinp

def rotation_precalcs(params, report):
    '''
    Call rotation precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    rps.rot_params(params)
    rps.landuses_phases(params,report)
    
def f1_rotationpyomo(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    #############
    #variables  #
    #############
    ##Amount of each phase on each soil, Positive Variable.
    model.v_phase_area = Var(model.s_season_types, model.s_phases,model.s_lmus, bounds=(0,None),doc='number of ha of each phase')

    if not pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1: #only needed for dsp version.
        model.v_root_hist = Var(model.s_rotconstraints, model.s_lmus, bounds=(0,None),doc='rotation history provided in the root stage')

    ####################
    #define parameters #
    ####################
    model.p_area = Param(model.s_lmus, initialize=params['lmu_area'], doc='available area on farm for each soil')
    
    model.p_landuse_area = Param(model.s_phases, model.s_landuses, initialize=params['phases_rk'], doc='landuse in each phase')

    ##only build this param if it doesn't exist already ie the rotation link never changes
    model.p_hist_prov = Param(params['hist_prov'].keys(), initialize=params['hist_prov'], default=0, doc='history provided by  each rotation') #use keys instead of sets to reduce size of param
    model.p_hist_req = Param(params['hist_req'].keys(), initialize=params['hist_req'], default=0, doc='history required by  each rotation') #use keys instead of sets to reduce size of param

    ###################
    #call constraints #
    ###################
    f_con_rotation(params, model)
    f_con_area(model)



#######################################################################################################################################################
#######################################################################################################################################################
#local constraints
#######################################################################################################################################################
#######################################################################################################################################################
######################
#rotation constraints#
######################
##rotation constraints are usually the same each loop. but if the lmu mask changes they need to be built again
##thus they are just built each loop. Maybe this could be changed if running lots of rotations.

    #todo i might not need the root hist variable and whatnot with new season structure

def f_con_rotation(params, model):
    '''
    Creates the constraint between history provided and required for each rotation phase on each LMU.

    The rotation constraints are to ensure that the rotation phases that are selected in the optimisation can
    be arranged into an actual rotation. All phases except the continuous rotations require at least one other
    phase. Eg. a canola(z)-wheat(w) rotation would be generated from 2 phases  Y Y N E z and Y Y E N w. To represent
    this requires ensuring that each rotaion phase selected has a preceding phase that has landuses in the same
    order as the target rotation phase (except for year 0). This is called the history required and history required.

    For steady state model each rotation requires and provides a rotation history.
    For DSP the process is slight more complicated because the history that provides the rotations must be the same for
    each season. because each season needs to start in a common place. Therefore a history variable is created which
    can be assigned to the root stage. This means an additional constraint is required.

    .. note:: the DSP structure will work fine for steady state however just increases the size, but for debugging you can
        use the DSP structure with the steady state model (just comment out the steady state stuff)

    '''

    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1:

        ##steady state rotation constraint
        def rot_phase_link(model,l,h,z):
            return sum(model.v_phase_area[z,r,l]*model.p_hist_prov[r,h] for r in model.s_phases if ((r,)+(h,)) in params['hist_prov'].keys()) \
                       + sum(model.v_phase_area[z,r,l]*model.p_hist_req[r,h] for r in model.s_phases if ((r,)+(h,)) in params['hist_req'].keys())<=0
        model.con_rotationcon1 = Constraint(model.s_lmus, model.s_rotconstraints, model.s_season_types, rule=rot_phase_link, doc='rotation phases constraint')

    else:

        ##DSP rotation constraint
        ##constraint for history provided to history root. This is only required in the stochastic model so that each season starts from a common place.
        def rot_hist(model,l,h,z):
            return model.v_root_hist[h,l] + sum(model.v_phase_area[z,r,l]*model.p_hist_prov[r,h]
                        for r in model.s_phases if ((r,)+(h,)) in params['hist_prov'].keys())<=0
        model.con_rot_hist = Constraint(model.s_lmus, model.s_rotconstraints, model.s_season_types, rule=rot_hist, doc='constraint between rotation history provided and root history')

        ##constraint for history provided to history root. This is only required in the stochastic model so that each season starts from a common place.
        def rot_phase_link(model,l,h,z):
            return - model.v_root_hist[h,l] + sum(model.v_phase_area[z,r,l]*model.p_hist_req[r,h]
                        for r in model.s_phases if ((r,)+(h,)) in params['hist_req'].keys())<=0
        model.con_root2rotation = Constraint(model.s_lmus, model.s_rotconstraints, model.s_season_types, rule=rot_phase_link, doc='constraint between rotation history root and rotation')


########
# Area #
########
def f_con_area(model):
    '''
    Creates the constraint between farm area and rotation area on each LMU.

    Constrains the maximum area of all rotations on each lmu by the area of each LMU on the modelled property.
    The area of rotation on a given soil can't be more than the amount of that soil available on the farm.
    '''

    def area_rule(model, l, z):
      return sum(model.v_phase_area[z,r,l] for r in model.s_phases) <= model.p_area[l]
    model.con_area = Constraint(model.s_lmus, model.s_season_types, rule=area_rule, doc='rotation area constraint')
    


