# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 21:26:59 2019

module: finance module

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

@author: young
"""
##python modules
import pandas as pd
import numpy as np

##AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp

'''
interest
'''

##########################
#debit & credit interest #
##########################

#If it's compound interest, which it generally is, take the annual interest rate (r) and raise it to the reciprocal of 12 to get your monthly rate.
#Why? Because there are 12 months in a year, and compound interest means exponential growth. Taking an exponent accounts for this.
#Converting yeary compound r to some shorter period m, use the following formula:
#[(1 + r)^(1/m)] - 1

#convert pa interest into per cashflow period
def debit_interest():
    return (1 + uinp.finance['debit_interest']) ** (1 / len(sinp.general['cashflow_periods']))


def credit_interest():
    return (1 + uinp.finance['credit_interest']) ** (1 / len(sinp.general['cashflow_periods']))


#################
#overheads      #
#################
def overheads(params, r_vals):
    overheads=pinp.general['overheads'] 
    overheads = overheads.squeeze().sum()/ len(sinp.general['cashflow_periods'])
    overheads = dict.fromkeys(sinp.general['cashflow_periods'], overheads)
    params['overheads'] = overheads
    r_vals['overheads'] = pd.Series(overheads)

#################
#Min ROE        #
#################
def f_min_roe():
    ##the default inputs for min roe are different for steady-state and stochastic version.
    ##but one SAV controls both inputs. So steady-state and stochastic can fairly be compared.
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        min_roe = uinp.finance['minroe']
    else:
        min_roe = uinp.finance['minroe_dsp']
    return min_roe



#################
# report vals   #
#################

def finance_rep(r_vals):
    keys_c = sinp.general['cashflow_periods']
    r_vals['keys_c'] = keys_c
    r_vals['opportunity_cost_capital'] = uinp.finance['opportunity_cost_capital']



