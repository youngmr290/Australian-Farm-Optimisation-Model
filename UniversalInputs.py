# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:06:06 2019

module: universal module - contains all the core input data - usually held constant/doesn't change between regions or farms' 


Version Control:
Version     Date        Person  Change
1.1         25Dec19     John    structure['phase_len'] = 5 (rather than 4)
1.2         27Dec19     MRY     moved rotation input data from crop to here
1.3         13Jan20     MRY     changed input.py to universal - and added other bits such as price, interest rates and mach options

Known problems:
Fixed   Date    ID by   Problem
1.2     25Dec19 John    The phase description in inputs are the full word whereas in the rotation phase it is just the letter.

@author: young
"""
#my plan
# these inputs are initally typed in
# the data is stored as a variable ie labour cost 
# then when the model is solving it can change labour cost to a different value specified in the exp section using kv's somehow.
# there is a different input sheet for each region/farm

import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta

import Functions as fun

#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#read in excel
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
price = fun.xl_all_named_ranges("Universal.xlsx","Price") 

##Finance inputs
finance = fun.xl_all_named_ranges("Universal.xlsx","Finance") 

##mach inputs - general 
mach_general = fun.xl_all_named_ranges("Universal.xlsx","Mach General") 

##feed inputs 
feed_inputs = fun.xl_all_named_ranges("Universal.xlsx","Feed Budget") 

##############
#mach options#
##############
##create a dict to store all options - this allows the user to select an option
machine_options_dict={} 
machine_options_dict['mach_1'] = fun.xl_all_named_ranges("Universal.xlsx","Mach 1") 



#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#general - used to determine model structure (these will stay in python to keep seperate from excel inputs which can be adjusted by any user)
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################

##create an empty dict to store all structure inputs 
structure = dict()

###############
#crops        #
###############
##the number of previous land uses considered for crop inputs - when this changes yeild input and fert and chem will need to be expended to include the extra years previous land use
structure['num_prev_phase']=1

###############
# cashflow    #
###############
##the number of these can change as long as each period is of equal length.
structure['cashflow_periods']=['JF$FLOW','MA$FLOW','MJ$FLOW','JA$FLOW','SO$FLOW','ND$FLOW']

#######
#sheep#
#######
##pools
structure['sheep_pools']=['pool1', 'pool2', 'pool3', 'pool4']


########################
#period                #
########################
##Length of standard labour period, must be an integer that 12 is divisible by
structure['labour_period_len'] = relativedelta(months=1)


##############
#phases      #        
##############

#number of phases analysed ie rotation length if you will (although not really a rotation)
structure['phase_len'] = 5 

#rotation phases and constraints read in from excel - used in crop module
structure['rotations'] = fun.xl_all_named_ranges("Rotation phases complete.xlsx","Phases") 


###############
#crops        #
###############
#update the pasture list as well! (just below)
#structure['phases'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay','fodder','annual']
#structure['stubble_crops'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay'] #used in stubble calcs in crop and stubble modules
#structure['pastures'] = ['annual'] #add any pasture phases to this list. this list is used to determine if a phase requires reseeding because it has cont crop before it. ie if none of this list is before it then it needs reseeding
structure['G']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
   , 'a', 'ar', 'a3', 'a4', 'a5'
   , 's', 'sr', 's3', 's4', 's5'
   , 'm', 'm3', 'm4'
   , 'u', 'u3', 'u4', 'u5'
   , 'x', 'x3', 'x4', 'x5'
   , 'j', 't'} #all landuses
structure['C']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #all crops
structure['N']={'z','r'} #canolas
structure['P']={'f','i', 'k', 'l', 'v'} #pulses
structure['E']={'b', 'h', 'o', 'of', 'w'} #cereals
structure['D']={'b', 'h', 'o', 'of', 'w', 'f', 'i', 'k', 'l', 'v'} #non canola crops (ie E & P)
structure['OF']={'of'} #oats fodder
structure['A']={'a', 'ar','a3', 'a4', 'a5', 's', 'sr','s3', 's4', 's5', 'm','m3', 'm4'} #annual
structure['A3']={'a3', 's3', 'm3'} #3rd yr pasture
structure['A4']={'a4', 's4', 'm4'} #4th yr pasture
structure['A5']={'a5', 's5', 'm5'} #5th yr pasture
structure['A3P']={'a3', 'a4', 'a5', 's3', 's4', 's5', 'm3', 'm4'} #3+ pastures
structure['AR']={'ar'} #resown annual
structure['M']={'m'} #manipulated pasture
structure['M3']={'m3'} #3rd yr manipulated pasture
structure['M4']={'m4'} #4th yr manipulated pasture
structure['S']={'s'} #spray topped pasture
structure['S3']={'s3'} #3rd yr spray topped pasture
structure['S4']={'s4'} #4th yr spray topped pasture
structure['S5']={'s5'} #5th yr spray topped pasture
structure['U']={'u', 'u3', 'u4', 'u5', 'x', 'x3', 'x4', 'x5'} #lucerne
structure['U3']={'u3'} #3rd yr lucerne
structure['U4']={'u4'} #4th yr lucerne
structure['U5']={'u5'} #5th yr lucerne
structure['X']={'x'}    
structure['X3']={'x3'} #3rd yr lucerne (monoculture)
structure['X4']={'x4'} #4th yr lucerne (monoculture)
structure['X5']={'x5'} #5th yr lucerne (monoculture)
structure['T']={'j', 't'} #tedera
structure['J']={'j'} #tedera manipulate


'''make each landuse a set so the issuperset func works'''
structure['b']={'b'}
structure['h']={'h'}
structure['o']={'o'}
structure['of']={'of'}
structure['w']={'w'}
structure['f']={'f'}
structure['i']={'i'}
structure['k']={'k'}
structure['l']={'l'}
structure['v']={'v'}
structure['z']={'z'}
structure['r']={'r'}
structure['a']={'a'}
structure['ar']={'ar'}
structure['a3']={'a3'}
structure['a4']={'a4'}
structure['a5']={'a5'}
structure['s']={'s'}
structure['sr']={'sr'}
structure['s3']={'s3'}
structure['s4']={'s4'}
structure['s5']={'s5'}
structure['m']={'m'}
structure['m3']={'m3'}
structure['m4']={'m4'}
structure['u']={'u'}
structure['u3']={'u3'}
structure['u4']={'u4'}
structure['u5']={'u5'}
structure['x']={'x'}
structure['x3']={'x3'}
structure['x4']={'x4'}
structure['x5']={'x5'}
structure['j']={'j'}
structure['t']={'t'}







#########################################################################################################################################################################################################
#########################################################################################################################################################################################################
#universal functions that use data from above
#########################################################################################################################################################################################################
#########################################################################################################################################################################################################


#Function that just uses inout inputs but is used in multiple other pre-calc modules 
#defined here to limit imorting pre calc modules in other precalc modules
def cols():
    #this is used to make a list of the relevent column numbers used in merge function, to specify the columns that are being matched - it will change if inputs specifying number of phases changes
    cols = []
    for i in reversed(range(structure['num_prev_phase']+1)):
        cols.append(structure['phase_len']-1-i)
    return cols


