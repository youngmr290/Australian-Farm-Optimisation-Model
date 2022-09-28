# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 09:58:05 2020

@author: young

Add model exceptions here
This module can't import other AFO modules

"""

# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class TrialError(Error):
    """Raised when reported trial doesn't exist"""
    pass


class ArithError(Error):
    """Raised when user has incorrect specifications for performing arithmetic"""
    pass


class AxisError(Error):
    """Raised when incorrect axis exist"""
    pass


class Exp(Error):
    """Raised when user has specified duplicate trials in exp.xls"""
    pass


class LabourPeriodError(Error):
    """Raised when user has NOT included all season nodes in labour periods"""
    pass

class FeedPeriodError(Error):
    """Raised when user has NOT included all season nodes in labour periods"""
    pass

class FVPError(Error):
    """Raised when multiple fvps are on the same date"""
    pass

class ParamError(Error):
    """Raised when building param if index is not the same size as the param"""
    pass
