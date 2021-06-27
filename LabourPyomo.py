# -*- coding: utf-8 -*-
"""
author: young

.. note:: Labour is uncondensed in pyomo. We have code which builds constraints for casual, perm and manager
    (rather than using a worker level set). The transfer labour variables do have a worker set which
    indicates what level of job is being done. The level is fixed for each constraint. Eg manager can do
    casual jobs, in con_sheep_anyone (sheep jobs that can be done by anyone) w (worker set) is fixed to ‘casual’.
    Eventually all the labour constraints could be condensed so there is one constraint for all the worker levels.
    This would require using the worker set as a set that is passed into the constraint. Would also need a
    param which indicates which level of working each source supplies eg casual needs to provide 0 manager
    level jobs.

    #todo add a manager and permanent labour DV that has a (small) positive OBJ value to use the slack.
    The value could be very small most of the time (sufficient to pick up the slack).
    But it could also be used as a control variable to allow SA on what jobs don't get done if you have minimum return on labour.
    i.e. if I'm not making at least $20/hr then "I'm not getting out of bed"

"""

#python modules
from pyomo.environ import *

#AFO modules
import Labour as lab
import PropertyInputs as pinp
import StructuralInputs as sinp

def lab_precalcs(params, r_vals):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    lab.labour_general(params, r_vals)
    lab.perm_cost(params, r_vals)
    lab.manager_cost(params, r_vals)
    params['min_perm'] = pinp.labour['min_perm'] 
    params['max_perm'] = pinp.labour['max_perm']
    params['min_managers'] = pinp.labour['min_managers'] 
    params['max_managers'] = pinp.labour['max_managers']
                    


def labpyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''
    ############
    # variable #
    ############

    # Casual supervision
    model.v_casualsupervision_perm = Var(model.s_labperiods,bounds=(0,None),
                                         doc='hours of perm labour used for supervision of casual')
    model.v_casualsupervision_manager = Var(model.s_labperiods,bounds=(0,None),
                                            doc='hours of manager labour used for supervision of casual')

    # Amount of casual. Casual labour can be optimised for each period
    model.v_quantity_casual = Var(model.s_labperiods,bounds=(0,None),
                                  doc='number of casual labour used in each labour period')

    # Amount of permanent labour.
    max_perm = pinp.labour['max_perm'] if pinp.labour['max_perm'] != 'inf' else None  # if none convert to python None
    model.v_quantity_perm = Var(bounds=(pinp.labour['min_perm'],max_perm),
                                doc='number of permanent labour used in each labour period')

    # Amount of manager labour
    max_managers = pinp.labour['max_managers'] if pinp.labour[
                                                      'max_managers'] != 'inf' else None  # if none convert to python None
    model.v_quantity_manager = Var(bounds=(pinp.labour['min_managers'],max_managers),
                                   doc='number of manager/owner labour used in each labour period')

    # manager pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    model.v_sheep_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                       doc='manager labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    model.v_crop_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='manager labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    model.v_fixed_labour_manager = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                       doc='manager labour used by fixed activities in each labour period for each different worker level')

    # permanent pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    model.v_sheep_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                         doc='permanent labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    model.v_crop_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                        doc='permanent labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    model.v_fixed_labour_permanent = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                         doc='permanent labour used by fixed activities in each labour period for each different worker level')

    # casual pool
    # labour for sheep activities (this variable transfers labour from source to sink)
    model.v_sheep_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='casual labour used by sheep activities in each labour period for each different worker level')

    # labour for crop activities (this variable transfers labour from source to sink)
    model.v_crop_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                     doc='casual labour used by crop activities in each labour period for each different worker level')

    # labour for fixed activities (this variable transfers labour from source to sink)
    model.v_fixed_labour_casual = Var(model.s_labperiods,model.s_worker_levels,bounds=(0,None),
                                      doc='casual labour used by fixed activities in each labour period for each different worker level')


    #########
    #param  #
    #########

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    ##called here , used below to generate params
    model.p_perm_hours = Param(model.s_labperiods, initialize= params[season]['permanent hours'], mutable=True, doc='hours worked by a permanent staff in each period')
    
    model.p_perm_supervision = Param(model.s_labperiods, initialize= params[season]['permanent supervision'], mutable=True, doc='hours of supervision required by a permanent staff in each period')
    
    model.p_perm_cost = Param(model.s_cashflow_periods, initialize = params['perm_cost'], default = 0.0, doc = 'cost of a permanent staff for 1 yr')
    
    model.p_casual_cost = Param(model.s_labperiods, model.s_cashflow_periods,  initialize = params[season]['casual_cost'], mutable=True, default = 0.0, doc = 'cost of a casual staff for each labour period')
    
    model.p_casual_hours = Param(model.s_labperiods, initialize= params[season]['casual hours'], mutable=True, doc='hours worked by a casual staff in each period')
    
    model.p_casual_supervision = Param(model.s_labperiods, initialize= params[season]['casual supervision'], mutable=True, doc='hours of supervision required by a casual staff in each period')
    
    model.p_manager_hours = Param(model.s_labperiods, initialize= params[season]['manager hours'], mutable=True, doc='hours worked by a manager in each period')
    
    model.p_manager_cost = Param(model.s_cashflow_periods, initialize = params['manager_cost'], doc = 'cost of a manager for 1 yr')
    
    model.p_casual_upper = Param(model.s_labperiods, initialize = params[season]['casual ub'], mutable=True,  doc = 'casual availability upper bound')
    
    model.p_casual_lower = Param(model.s_labperiods, initialize = params[season]['casual lb'], mutable=True, doc = 'casual availability lower bound')

    ###############################
    #call local constraints       #
    ###############################
    f_con_casual_bounds(model)
    f_con_casual_supervision(model)
    f_con_labour_transfer_manager(model)
    f_con_labour_transfer_permanent(model)
    f_con_labour_transfer_casual(model)


###############################
#local constraints            #
###############################
def f_con_casual_bounds(model):
    '''
    Optional constraint to bound the level of casual staff in each period.
    '''
    #this can't be done with variable bounds because it's not a constant value for each period (seeding and harv may differ)
    def casual_labour_availability(model, p):
        return  (model.p_casual_lower[p], model.v_quantity_casual[p], model.p_casual_upper[p]) #pyomos way of: lower <= x <= upper
    model.con_casual_bounds = Constraint(model.s_labperiods, rule = casual_labour_availability, doc='bounds the casual labour in each period')

def f_con_casual_supervision(model):
    '''
    Casual labourers require a certain amount of supervision per period. Supervision can be provided
    by either permanent or manager staff. This constraint ensures ensures that the supervision requirement
    is met.
    '''
    ##casual supervision - can be done by either perm or manager
    def transfer_casual_supervision(model,p):
        return -model.v_casualsupervision_manager[p] - model.v_casualsupervision_perm[p] + (model.p_casual_supervision[p] * model.v_quantity_casual[p]) <= 0
    model.con_casual_supervision = Constraint(model.s_labperiods, rule = transfer_casual_supervision, doc='casual require supervision from perm or manager')

def f_con_labour_transfer_manager(model):
    '''Transfer manager labour to livestock, cropping, fixed and supervising activities.'''
    #manager, this is a little more complex because also need to subtract the supervision hours off of the manager supply of workable hours
    def labour_transfer_manager(model,p):
        return -(model.v_quantity_manager * model.p_manager_hours[p]) + (model.v_quantity_perm * model.p_perm_supervision[p]) + model.v_casualsupervision_manager[p]      \
        + sum(model.v_sheep_labour_manager[p,w] + model.v_crop_labour_manager[p,w] + model.v_fixed_labour_manager[p,w] for w in model.s_worker_levels)  <= 0
    model.con_labour_transfer_manager = Constraint(model.s_labperiods, rule = labour_transfer_manager, doc='labour from manager to sheep and crop and fixed')

def f_con_labour_transfer_permanent(model):
    '''Transfer permanent labour to livestock, cropping, fixed and supervising activities.'''
    #permanent
    def labour_transfer_permanent(model,p):
        return -(model.v_quantity_perm * model.p_perm_hours[p]) + model.v_casualsupervision_perm[p]  \
        + sum(model.v_sheep_labour_permanent[p,w] + model.v_crop_labour_permanent[p,w] + model.v_fixed_labour_permanent[p,w] for w in model.s_worker_levels if w in sinp.general['worker_levels'][0:-1]) <= 0 #if statement just to remove unnecessary activities from lp output
    model.con_labour_transfer_permanent = Constraint(model.s_labperiods, rule = labour_transfer_permanent, doc='labour from permanent staff to sheep and crop and fixed')

def f_con_labour_transfer_casual(model):
    '''Transfer casual labour to livestock, cropping and fixed activities.'''
    #casual note perm and manager can do casual tasks - variables may need to change name so to be less confusing
    def labour_transfer_casual(model,p):
        return -(model.v_quantity_casual[p] *  model.p_casual_hours[p])  \
            + sum(model.v_sheep_labour_casual[p,w] + model.v_crop_labour_casual[p,w] + model.v_fixed_labour_casual[p,w] for w in model.s_worker_levels if w in sinp.general['worker_levels'][0])  <= 0  #if statement just to remove unnecessary activities from lp output
    model.con_labour_transfer_casual = Constraint(model.s_labperiods, rule = labour_transfer_casual, doc='labour from casual staff to sheep and crop and fixed')



#######################
#labour cost function #
#######################

#sum the cost of perm, casual and manager labour. When i tried to do it all in one function it didn't work (it should be possible though )
def casual(model,c):
    return sum( model.v_quantity_casual[p] * model.p_casual_cost[p,c] for p in model.s_labperiods) 
def perm(model,c):
    return model.v_quantity_perm * model.p_perm_cost[c] 
def manager(model,c):
    return model.v_quantity_manager * model.p_manager_cost[c] 
def labour_cost(model,c):
    '''
    Calculate the total cost of the selected labour activities.

    Used in global constraint (con_cashflow). See CorePyomo
    '''
    return casual(model,c) + perm(model,c) + manager(model,c)





