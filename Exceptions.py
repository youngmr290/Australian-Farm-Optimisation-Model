# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

Add model exceptions here

"""

# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class TrialError(Error):
    """Raised when reported trial doesn't exist"""
    pass


class ArithError(Error):
    """Raised when user is preforming arithmetic on invalid axis"""
    pass


class AxisError(Error):
    """Raised when incorrect axis exist"""
    pass
