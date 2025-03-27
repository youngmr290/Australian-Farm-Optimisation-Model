#python modules
from pyomo import environ as pe

#AFO modules
from . import Trees as tree

def tree_precalcs(params, r_vals):
    '''
    Call tree precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''

    tree.f1_trees(params,r_vals)

    
def f1_treepyomo_local(params, model):
    ''' Builds pyomo variables, parameters'''

    ############
    # variable #
    ############
    #TODO need to hook this up later. Then have bnd so that we can correctly represent the tree interactions with adjacent land.
    model.v_tree_area_l = pe.Var(model.s_lmus, bounds=(0,None),
                               doc='hectares of trees on each land management unit')

    #########
    #param  #
    ######### 

    ##cost
    model.p_tree_cashflow_p7z = pe.Param(model.s_season_periods, model.s_season_types, initialize=params['p_tree_cashflow_p7z'], default = 0.0, mutable=True, doc='net cashflow from tree plantations')
    
    ##wc
    model.p_tree_wc_c0p7z = pe.Param(model.s_enterprises, model.s_season_periods, model.s_season_types, initialize=params['p_tree_wc_c0p7z'], default = 0.0, mutable=True, doc='net working capital gain/loss from tree plantation')



#######################################################################################################################################################
#######################################################################################################################################################
#functions for core model
#######################################################################################################################################################
#######################################################################################################################################################
def f_tree_cashflow(model,p7,z):
    '''
    Calculate the total net cashflow incured from tree plantings.

    Used in global constraint (con_profit). See CorePyomo
    '''

    return model.p_tree_cashflow_p7z[p7,z]

def f_tree_wc(model,c0,p7,z):
    '''
    Calculate the total wc of tree plantings.

    Used in global constraint (con_workingcap). See CorePyomo
    '''

    return model.p_tree_wc_c0p7z[c0,p7,z]
