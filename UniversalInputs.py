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
import numpy as np
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
n_feed_pools        = 4             # number of feed pools (by quality groups)   ^ Add this to Universal.xlsx


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

#rotation phases and constraints read in from excel
structure['phases'] = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header= None, index_col = 0).T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc



###############
#crops        #
###############
#update the pasture list as well! (just below)
#structure['phases'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay','fodder','annual']
#structure['stubble_crops'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay'] #used in stubble calcs in crop and stubble modules
#structure['pastures'] = ['annual'] #add any pasture phases to this list. this list is used to determine if a phase requires reseeding because it has cont crop before it. ie if none of this list is before it then it needs reseeding
structure['A']={'a', 'ar','a3', 'a4', 'a5', 's', 'sr','s3', 's4', 's5', 'm','m3', 'm4', 'm5'} #annual
structure['A1']={'a', 'a3', 'a4', 'a5', 's','s3', 's4', 's5', 'm','m3', 'm4', 'm5'} #annual - special set used when determining if a rotatin provides a rotation because in yr1 we dont want ar to provide an A bevause we need to distinguish beteween them
structure['A3']={'a3'}#, 's3', 'm3'} #3rd yr pasture
structure['A5']={'a5'}#, 's5','m5'} #5th yr pasture
structure['A4']={'a4'}#, 's4', 'm4'} #4th yr pasture
# structure['A3P']={'a3', 'a4', 'a5', 's3', 's4', 's5', 'm3', 'm4'} #3+ pastures
structure['AR']={'ar'} #resown annual
structure['C']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #all crops, used in stubble and mach (not used for rotations)
# structure['D']={'b', 'h', 'o', 'of', 'w', 'f', 'i', 'k', 'l', 'v'} #non canola crops (ie E & P)
structure['E']={'b', 'h', 'o', 'of', 'w'} #cereals
structure['E1']={'b', 'h', 'o', 'w'} #cereal - special set used when determining if a rotatin provides a rotation because in yr1 we dont want OF to provide an E bevause we need to distinguish beteween them
structure['G']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'a', 'ar', 'a3', 'a4', 'a5'
                , 's', 'sr', 's3', 's4', 's5'
                , 'm', 'm3', 'm4', 'm5'
                , 'u', 'ur', 'u3', 'u4', 'u5'
                , 'x', 'xr', 'x3', 'x4', 'x5'
                , 'j', 't', 'jr', 'tr'} #all landuses
# structure['H']={'h', 'of'} #non harvested cereals
structure['J']={'j', 'jr'} #tedera
structure['M']={'m', 'm3', 'm4','m5'} #manipulated pasture
structure['M3']={'m3'} #3rd yr manipulated pasture
structure['M4']={'m4'} #4th yr manipulated pasture
structure['M5']={'m5'} #4th yr manipulated pasture
structure['N']={'z','r'} #canolas
structure['OF']={'of'} #oats fodder
structure['P']={'f','i', 'k', 'l', 'v'} #pulses
structure['S']={'s','sr', 's3', 's4', 's5'} #spray topped pasture
structure['SR']={'sr'} #spray topped pasture
structure['S3']={'s3'} #3rd yr spray topped pasture
structure['S4']={'s4'} #4th yr spray topped pasture
structure['S5']={'s5'} #5th yr spray topped pasture
structure['T']={'t', 'tr'} #tedera
structure['U']={'u', 'ur', 'u3', 'u4', 'u5'} #lucerne
structure['U3']={'u3'} #3rd yr lucerne
structure['U4']={'u4'} #4th yr lucerne
structure['U5']={'u5'} #5th yr lucerne
structure['X']={'x', 'xr', 'x3', 'x4', 'x5'} #lucerne
structure['X3']={'x3'} #3rd yr lucerne (monoculture)
structure['X4']={'x4'} #4th yr lucerne (monoculture)
structure['X5']={'x5'} #5th yr lucerne (monoculture)
structure['Y']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
                , 'u', 'u3', 'u4', 'u5'
                , 'x', 'x3', 'x4', 'x5'
                , 'j', 't', 'jr', 'tr'} #anything not A


'''make each landuse a set so the issuperset func works'''
structure['a']={'a'}
structure['ar']={'ar'}
structure['a3']={'a3'}
structure['a4']={'a4'}
structure['a5']={'a5'}
structure['b']={'b'}
structure['f']={'f'}
structure['h']={'h'}
structure['j']={'j'}
structure['jr']={'jr'}
structure['l']={'l'}
structure['m']={'m'}
structure['m3']={'m3'}
structure['m4']={'m4'}
structure['m5']={'m5'}
structure['o']={'o'}
structure['of']={'of'}
structure['r']={'r'}
structure['s']={'s'}
structure['sr']={'sr'}
structure['s3']={'s3'}
structure['s4']={'s4'}
structure['s5']={'s5'}
structure['t']={'t'}
structure['tr']={'tr'}
structure['u']={'u'}
structure['ur']={'ur'}
structure['u3']={'u3'}
structure['u4']={'u4'}
structure['u5']={'u5'}
structure['w']={'w'}
structure['x']={'x'}
structure['xr']={'xr'}
structure['x3']={'x3'}
structure['x4']={'x4'}
structure['x5']={'x5'}
structure['z']={'z'}











# phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U4','U5','u4','u5']))] #only U or U3 ufter U
#     phases = phases[~(np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U', 'U3','U5','u','ur','u3','u5']))] #pasture 4 muxt come ufter pasture 3
#     phases = phases[~(np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 4
#     phases = phases[~(np.isin(phases[:,i], ['U5'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 5
#     phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U3','u3']))] #cant have U3 after anything except U
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+2], ['U3','u3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
#     except IndexError: pass
#     phases = phases[~(~np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U4','u4']))] #cant have U4 after anything except U3
#     phases = phases[~(~np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U5','u5']))] #cant have U5 after anything except U4
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U'])&~np.isin(phases[:,i+2], ['U3','u3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
#     except IndexError: pass

#     ##Manipulated Lucerne
#     phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X4','X5','x4','x5']))] #only U or U3 ufter U
#     phases = phases[~(np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X','X3','X5','x','xr','x3','x5']))] #pasture 4 muxt come ufter pasture 3
#     phases = phases[~(np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 4
#     phases = phases[~(np.isin(phases[:,i], ['X5'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 5
#     phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X3','x3']))] #cant have U3 after anything except U
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+2], ['X3','x3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
#     except IndexError: pass
#     phases = phases[~(~np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X4','x4']))] #cant have U4 after anything except U3
#     phases = phases[~(~np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X5','x5']))] #cant have U5 after anything except U4
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X'])&~np.isin(phases[:,i+2], ['X3','x3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
#     except IndexError: pass

# #Lucerne
#
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


