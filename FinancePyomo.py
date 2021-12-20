"""

author: young

"""
#python modules
from pyomo.environ import *

#AFO modules
import Finance as fin
import PropertyInputs as pinp

def fin_precalcs(params, r_vals):
    '''
    Call finance precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    fin.f1_fin_params(params, r_vals)



def f1_finpyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    ############
    #variables #
    ############

    ##credit for a given time period (time period defined by cashflow set)
    model.v_credit = Var(model.s_sequence_year, model.s_sequence, model.s_c1, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'amount of net positive cashflow in a given period')
    ##debit for a given time period (time period defined by cashflow set)
    model.v_debit = Var(model.s_sequence_year, model.s_sequence, model.s_c1, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'amount of net negative cashflow in a given period')
    ##working capital credit for a given time period (time period defined by cashflow set)
    model.v_wc_credit = Var(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'amount of net positive working capital in a given period')
    ##working capital for a given time period (time period defined by cashflow set)
    model.v_wc_debit = Var(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'amount of net negative working capital in a given period')
    ##dep
    model.v_dep = Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'transfers total dep to objective')
    ##dep
    model.v_asset = Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'transfers total value of asset to objective to ensure opportunity cost is represented')
    ##minroe
    model.v_minroe = Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_season_types, bounds = (0.0, None), doc = 'total expenditure, used to ensure min return is met')

    ####################
    #params            #
    ####################
    model.p_overhead_cost = Param(model.s_season_periods, model.s_season_types, initialize = params['overheads_cost'], doc = 'cost of overheads each period')

    model.p_overhead_wc = Param(model.s_enterprises, model.s_season_periods, model.s_season_types, initialize = params['overheads_wc'], doc = 'wc of overheads each period')

    model.p_prob_c1 = Param(model.s_c1, initialize = params['prob_c1'], doc = 'probability of each price scenario')

    #########################
    #call Local constrain   #
    #########################
    f_con_overdraw(params, model)



############
#Contraints#
############
def f_con_overdraw(params, model):
    '''
    Constrains the level of overdraw in each cashflow period.

    This ensures the model draws a realistic level of money from the bank. The user can specify the
    maximum overdraw level.
    '''
    ##debit can't be more than a specified amount ie farmers will draw a maximum from the bank throughout yr
    def overdraw(model,q,s,c0,p7,z):
        return model.v_wc_debit[q,s,c0,p7,z] <= params['overdraw']
    model.con_overdraw = Constraint(model.s_sequence_year, model.s_sequence, model.s_enterprises, model.s_season_periods, model.s_season_types, rule=overdraw, doc='overdraw limit')

