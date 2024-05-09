# -*- coding: utf-8 -*-
"""
Functions within this module are executed for each trial, after the user sensitivites have been applied, to validate
the inputs.
"""
import numpy as np

from . import Functions as fun
from . import Sensitivity as sen
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import PropertyInputs as pinp

def f_input_value_test():
    '''test data type, and range eg int between 0-5.'''
    pass

def f_input_shape_test():
    '''test input shape.'''
    #todo there are some input tests for stock in universal.xlsx that could be moved here.
    pass

def f_input_logic_test():
    '''logic tests of inputs'''
    # pasture.py
    # 1. pasture foo must be greater than 0 after it passes through conversion function.
    # foo_grazplan = np.maximum(0, np.minimum(foo, cu3[2] + cu3[0] * foo + cu3[1] * legume)) #this should be greater than 0

    ##if legume bnd is on then some legume crops must be included
    crop_landuse_mask_k1 = np.logical_and(pinp.general['i_crop_landuse_exists_k1'], pinp.general['i_crop_landuse_inc_k1'])
    legume_included = np.array([x in sinp.landuse['P'] for x in sinp.landuse['C'][crop_landuse_mask_k1]], dtype=bool)
    if sen.sav['bnd_total_legume_area_percent']!='-' and not legume_included.any():
        raise ValueError('No legume land uses are included yet the legume area bound is on')
