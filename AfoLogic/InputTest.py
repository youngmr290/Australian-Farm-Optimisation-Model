# -*- coding: utf-8 -*-
"""
Functions within this module are executed for each trial, after the user sensitivites have been applied, to validate
the inputs.
"""


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
    pass