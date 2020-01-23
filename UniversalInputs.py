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
# structure['rotations'] = fun.xl_all_named_ranges("Rotation phases complete.xlsx","Phases") 

##after how many yrs is annual resown
structure['resow_a'] = 4


###############
#crops        #
###############
#update the pasture list as well! (just below)
#structure['phases'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay','fodder','annual']
#structure['stubble_crops'] = ['wheat','barley','oats','tcanola','rcanola','faba', 'lupins','hay'] #used in stubble calcs in crop and stubble modules
#structure['pastures'] = ['annual'] #add any pasture phases to this list. this list is used to determine if a phase requires reseeding because it has cont crop before it. ie if none of this list is before it then it needs reseeding
structure['A']={'a', 'ar','a3', 'a4', 'a5', 's', 'sr','s3', 's4', 's5', 'm','m3', 'm4'} #annual
structure['A3']={'a3', 's3', 'm3'} #3rd yr pasture
structure['A4']={'a4', 's4', 'm4'} #4th yr pasture
structure['A5']={'a5', 's5', 'm5'} #5th yr pasture
# structure['A3P']={'a3', 'a4', 'a5', 's3', 's4', 's5', 'm3', 'm4'} #3+ pastures
structure['AR']={'ar'} #resown annual
structure['C']={'b', 'h', 'o', 'of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'} #all crops
structure['D']={'b', 'h', 'o', 'of', 'w', 'f', 'i', 'k', 'l', 'v'} #non canola crops (ie E & P)
structure['E']={'b', 'h', 'o', 'of', 'w'} #cereals
# structure['G']={'b', 'h', 'o','of', 'w', 'f','i', 'k', 'l', 'v', 'z','r'
#                , 'a', 'ar', 'a3', 'a4', 'a5'
#                , 's', 'sr', 's3', 's4', 's5'
#                , 'm', 'm3', 'm4'
#                , 'u', 'u3', 'u4', 'u5'
#                , 'x', 'x3', 'x4', 'x5'
#                , 'j', 't', 'jr', 'tr'} #all landuses
structure['H']={'h', 'of'} #non harvested cereals
structure['M']={'m', 'm3', 'm4'} #manipulated pasture
structure['M3']={'m3'} #3rd yr manipulated pasture
structure['M4']={'m4'} #4th yr manipulated pasture
structure['N']={'z','r'} #canolas
structure['OF']={'of'} #oats fodder
structure['P']={'f','i', 'k', 'l', 'v'} #pulses
structure['S']={'s', 's3', 's4', 's5'} #spray topped pasture
structure['S3']={'s3'} #3rd yr spray topped pasture
structure['S4']={'s4'} #4th yr spray topped pasture
structure['S5']={'s5'} #5th yr spray topped pasture
structure['T']={'j', 't', 'jr', 'tr'} #tedera
structure['U']={'u', 'ur', 'u3', 'u4', 'u5', 'x', 'xr', 'x3', 'x4', 'x5'} #lucerne
structure['U3']={'u3'} #3rd yr lucerne
structure['U4']={'u4'} #4th yr lucerne
structure['U5']={'u5'} #5th yr lucerne
structure['X']={'x'}    
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

yr0 = np.array(['b', 'h', 'o','of', 'w', 'f', 'l', 'z','r'
               , 'a', 'ar', 'a3', 'a4', 'a5'
               , 's', 'sr', 's3', 's4', 's5'
               , 'm', 'm3', 'm4'
               , 'u', 'ur', 'u3', 'u4', 'u5'
               , 'x', 'xr', 'x3', 'x4', 'x5'
               , 'j', 't', 'jr', 'tr'])
yr1 = np.array(['ar', 'sr'
       ,'E', 'N', 'P'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr2 = np.array(['E', 'N', 'P'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr3 = np.array(['E', 'N', 'P'
       , 'A', 'A3', 'A4', 'A5'
       , 'S', 'S3', 'S4', 'S5'
       , 'M', 'M3', 'M4'
       , 'U', 'U3', 'U4', 'U5'
       , 'X', 'X3', 'X4', 'X5'
       , 'T', 'J'])
yr4 = np.array(['A','Y'])

arrays=[yr4,yr3,yr2,yr1,yr0]
phases=fun.cartesian_product_simple_transpose(arrays)

for i in range(np.size(phases,1)-1):
##drop rules 1; unprofitable
    phases = phases[~(np.isin(phases[:,i], ['N'])&np.isin(phases[:,i+1], ['N','r','z']))] #no cont canola
    phases = phases[~(np.isin(phases[:,i], ['P'])&np.isin(phases[:,i+1], ['P','l','f']))] #no cont pulse
    phases = phases[~(np.isin(phases[:,i], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5','U','U3','U4','U5','T','T3','T4','T5'])&np.isin(phases[:,i+1], ['P','l','f']))] #no pulse after pasture
    phases = phases[~(np.isin(phases[:,i], ['S','S3','S4','S5'])&np.isin(phases[:,i+1], ['A', 'A3', 'A4', 'A5','M','M3','M4','S','S3','S4','S5','a','ar','a3','a4','a5','s','sr','s3','s4','s5','m','m3','m4']))] #no pasture after spraytoped 
    phases = phases[~(np.isin(phases[:,i], ['M','M3','M4'])&np.isin(phases[:,i+1], ['A', 'A3', 'A4', 'A5','M','M3','M4','a','ar','a3','a4','a5','m','m3','m4']))] #no pasture after spraytoped or manipulated

##drop rules 2; logical    
    ##Annual 
    phases = phases[~(np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+1], ['A4','A5','M4','S4','S5','a4','a5','s4','s5','m4']))] #only A or A3 after A
    phases = phases[~(np.isin(phases[:,i], ['A3'])&np.isin(phases[:,i+1], ['A', 'A3','A5','M','M3','S','S3','S5','a','ar','a3','a5','s','sr','s3','s5','m','m3']))] #pasture 4 must come after pasture 3
    phases = phases[~(np.isin(phases[:,i], ['A4'])&np.isin(phases[:,i+1], ['A', 'A3','A4','M','M3','M4','S','S3','S4','a','ar','a3','a4','s','sr','s3','s4','m','m3','m4']))] #pasture 5 must come after pasture 4
    phases = phases[~(np.isin(phases[:,i], ['A5'])&np.isin(phases[:,i+1], ['A', 'A3','A4','M','M3','M4','S','S3','S4','a','ar','a3','a4','s','sr','s3','s4','m','m3','m4']))] #pasture 5 must come after pasture 5
    phases = phases[~(~np.isin(phases[:,i], ['A','ar','sr'])&np.isin(phases[:,i+1], ['A3','S3','M3','a3','m3','s3']))] #cant have A3 after anything except A  (have used a double negitive here)
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(~np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+2], ['A3','S3','M3','a3','m3','s3']))] #cant have A3 after anything except A A (goes with the rule above)
    except IndexError: pass
    phases = phases[~(~np.isin(phases[:,i], ['A3'])&np.isin(phases[:,i+1], ['A4','S4','M4','a4','s4','m4']))] #cant have A4 after anything except A3  (have used a double negitive here)
    phases = phases[~(~np.isin(phases[:,i], ['A4'])&np.isin(phases[:,i+1], ['A5','S5','a5','s5']))] #cant have A5 after anything except A4  (have used a double negitive here)
    try: #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['A'])&np.isin(phases[:,i+1], ['A'])&~np.isin(phases[:,i+2], ['A3','S3','M3','a3','m3','s3']))] #can only have A3 after A A (have used a double negitive here)
    except IndexError: pass
    
    ##Lucerne
    phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U4','U5','u4','u5']))] #only U or U3 ufter U
    phases = phases[~(np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U', 'U3','U5','u','ur','u3','u5']))] #pasture 4 muxt come ufter pasture 3
    phases = phases[~(np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 4
    phases = phases[~(np.isin(phases[:,i], ['U5'])&np.isin(phases[:,i+1], ['U', 'U3','U4','u','ur','u3','u4']))] #pasture 5 muxt come ufter pasture 5
    phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U3','u3']))] #cant have U3 after anything except U 
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(~np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+2], ['U3','u3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
    except IndexError: pass
    phases = phases[~(~np.isin(phases[:,i], ['U3'])&np.isin(phases[:,i+1], ['U4','u4']))] #cant have U4 after anything except U3  
    phases = phases[~(~np.isin(phases[:,i], ['U4'])&np.isin(phases[:,i+1], ['U5','u5']))] #cant have U5 after anything except U4  
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['U'])&np.isin(phases[:,i+1], ['U'])&~np.isin(phases[:,i+2], ['U3','u3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
    except IndexError: pass

    ##Manipulated Lucerne
    phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X4','X5','x4','x5']))] #only U or U3 ufter U
    phases = phases[~(np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X','X3','X5','x','xr','x3','x5']))] #pasture 4 muxt come ufter pasture 3
    phases = phases[~(np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 4
    phases = phases[~(np.isin(phases[:,i], ['X5'])&np.isin(phases[:,i+1], ['X','X3','X4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 5
    phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X3','x3']))] #cant have U3 after anything except U 
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(~np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+2], ['X3','x3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
    except IndexError: pass
    phases = phases[~(~np.isin(phases[:,i], ['X3'])&np.isin(phases[:,i+1], ['X4','x4']))] #cant have U4 after anything except U3  
    phases = phases[~(~np.isin(phases[:,i], ['X4'])&np.isin(phases[:,i+1], ['X5','x5']))] #cant have U5 after anything except U4  
    try:  #used for conditions that are concerned with more than two yrs
        phases = phases[~(np.isin(phases[:,i], ['X'])&np.isin(phases[:,i+1], ['X'])&~np.isin(phases[:,i+2], ['X3','x3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
    except IndexError: pass


##rules where we are interested in all yrs - done this way so that if the len of the phase changes this can remain the same

##if history is all crop/T/U you must resow annual
j=np.size(phases,1)-1-structure['resow_a'] #difference between history len and resow len (if 0 it means that all the history must be a crop for pasture to be resown)
reseed_index=~np.isin(phases[:,j], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5']) #just to create an array that can be added to.
for i in range(np.size(phases,1)-1):
    i+=j #to adjust for the diff between hist len and resown len
    if i <= np.size(phases,1)-1:
        reseed_index &= ~np.isin(phases[:,i], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5'])# checks if all the yrs of history are not anual
reseed_index &= np.isin(phases[:,np.size(phases,1)-1], ['a', 's','m']) #checks if yr0 is not resown annual
phases = phases[~reseed_index] #if there is an annual after cont crop it must be resown

##can't have reseeded pasture in yr0 if annual before
not_reseed_index=np.isin(phases[:,j], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5']) #just to create an array that can be added to.
for i in range(np.size(phases,1)-1):
    i+=j #to adjust for the diff between hist len and resown len
    if i <= np.size(phases,1)-1:
        not_reseed_index +=  np.isin(phases[:,i], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5'])# checks if any (or) of the years in the history are not pasture (will result in true if the phase needed resowing)
not_reseed_index &= np.isin(phases[:,np.size(phases,1)-1], ['ar', 'sr']) #checks if yr0 is not resown annual
phases = phases[~not_reseed_index] #if there is an annual in the history you don't need to reseed

##can't have reseeded pasture in yr1 if annual before -couldn't combine with above because i only want to loop through the columns up to the second last one
not_reseed_index2=np.isin(phases[:,j], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5']) #just to create an array that can be added to.
for i in range(np.size(phases,1)-1):
    i+=j #to adjust for the diff between hist len and resown len
    if i <= np.size(phases,1)-2:
        not_reseed_index2 += np.isin(phases[:,i], ['ar', 'sr','A','A3','A4','A5','M','M3','M4','S','S3','S4','S5'])# checks if any (or) of the years in the history are not pasture
not_reseed_index2 &= np.isin(phases[:,np.size(phases,1)-2], ['ar', 'sr']) #checks if yr1 is resown annual
phases = phases[~not_reseed_index2] #if there is an annual in the history you don't need to reseed


# #Lucerne
#     phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['U4','U5','X4','X5','u4','u5','x4','x5']))] #only U or U3 ufter U
#     phases = phases[~(np.isin(phases[:,i], ['U3','X3'])&np.isin(phases[:,i+1], ['U', 'U3','U5','X','X3','X5','u','ur','u3','u5','x','xr','x3','x5']))] #pasture 4 muxt come ufter pasture 3
#     phases = phases[~(np.isin(phases[:,i], ['U4','X4'])&np.isin(phases[:,i+1], ['U', 'U3','U4','X','X3','X4','u','ur','u3','u4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 4
#     phases = phases[~(np.isin(phases[:,i], ['U5','X5'])&np.isin(phases[:,i+1], ['U', 'U3','U4','X','X3','X4','u','ur','u3','u4','x','xr','x3','x4']))] #pasture 5 muxt come ufter pasture 5
#     phases = phases[~(~np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['U3','X3','u3','x3']))] #cant have U3 after anything except U 
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(~np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+2], ['U3','X3','u3','x3']))] #cant have U3 ufter unything except U U (this is the second part to the rule above)
#     except IndexError: pass
#     phases = phases[~(~np.isin(phases[:,i], ['U3','X3'])&np.isin(phases[:,i+1], ['U4','X4','u4','x4']))] #cant have U4 after anything except U3  
#     phases = phases[~(~np.isin(phases[:,i], ['U4','X4'])&np.isin(phases[:,i+1], ['U5','X5','u5','x5']))] #cant have U5 after anything except U4  
#     try:  #used for conditions that are concerned with more than two yrs
#         phases = phases[~(np.isin(phases[:,i], ['U','X'])&np.isin(phases[:,i+1], ['U','X'])&~np.isin(phases[:,i+2], ['U3','X3','u3','x3']))] #can only huve U3 ufter U U (huve uxed u double negitive here)
#     except IndexError: pass
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


