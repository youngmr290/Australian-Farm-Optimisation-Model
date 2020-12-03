'''
THis is where you can set up constraints on individual variables
withing a set of variables. This will act as bounds.

bounds are all controlled from this module
'''

import numpy as np


import Functions as fun
from CreateModel import model
import pyomo.environ as pe

bounds_inc=True


def boundarypyomo_local():
    if bounds_inc:
        print('bounds implemented - make sure they are correct')

        ###########################
        # initialise numpy array  #
        ###########################
        '''this array should be the same size as the variable it will bound'''

        ##keys to build arrays
        arrays = [model.s_phases, model.s_lmus]
        index_rl = fun.cartesian_product_simple_transpose(arrays)


        ##build array
        rot_lobound_rl = np.zeros((len(model.s_phases), len(model.s_lmus)))


        ###########################
        # set bound               #
        ###########################
        '''set the bound here, can do this like assigning any value to numpy.
            These could be adjusted with SA values if you want to alter the bounds for different trials
            - The forced sale or retain of drys is controled by livestock generator inputs'''

        rot_lobound_rl[0,2] = 0 #sets all rotations to a min of 1ha.
        # force_sale_drys
        # fore retain drys
        # force retention - done in generator this is an input


        #################################
        # ravel and zip bound and index #
        #################################
        ##rotation
        rot_lobound = rot_lobound_rl.ravel()
        tup_rl = tuple(map(tuple, index_rl))
        rot_lobound = dict(zip(tup_rl, rot_lobound))



        #################################
        # build the constraint          #
        #################################
        try:
            model.del_component(model.con_rotation_lobound)
            model.del_component(model.con_rotation_lobound_index)
        except AttributeError:
            pass
        def rot_lo_bound(model, r, l):
            return model.v_phase_area[r, l] >= rot_lobound[r,l]
        model.con_rotation_lobound = pe.Constraint(model.s_phases, model.s_lmus, rule=rot_lo_bound,
                                                doc='lo bound for the number of each phase')
