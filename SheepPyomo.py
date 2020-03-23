# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:03:35 2020

@author: John
"""

#python modules
from pyomo.environ import *

#MUDAS modules
# from CreateModel import *
import SheepSim as ssim

def sheep_pyomo_local():
    ssim.sheep_sim()
    # ssim.create_masks()

    ######################
    ### setup parameters #
    ######################
    # try:
    #     model.del_component(model.p_the_parameter)
    # except AttributeError:
    #     pass
    # model.p_the_parameter = Param(model.s_component, initialize=ssim.function_that_return_the_parameters(), default = 0.0, doc='what the parameters are')

    #####################
    ### setup variables #
    #####################

    ########################
    ### set up constraints #
    ########################
    ## def constraint_function:
    ## ^ or define outside the main function and just call here
    ## call constraint function
    #model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')

##################################
### setup core model constraints #
##################################
## def constraint_function:
## ^ or define outside the main function and just call here
## call constraint function
#model.constrain_name = Constraint(model.s_whatever_combinations, rule=constraint_function, doc='constrain whatever it does')
