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

#MUDAS modules
import UniversalInputs as uinp
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
    return (1 + uinp.finance['debit_interest']) ** (1 / len(uinp.structure['cashflow_periods']))


def credit_interest():
    return (1 + uinp.finance['credit_interest']) ** (1 / len(uinp.structure['cashflow_periods']))


#################
#overheads      #
#################
def overheads():
    overheads=pinp.general['overheads'] 
    return overheads.squeeze().sum()/ len(uinp.structure['cashflow_periods'])    







