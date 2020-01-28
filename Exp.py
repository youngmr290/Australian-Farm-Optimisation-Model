# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: exriment module - this is the module that runs everything and controls kv's

@author: young
"""
#import datetime
import pandas as pd
from pyomo.environ import *

import RotationPhases as rps #this won't be here when sa is hooked up properly
import Sensitivity as sen 
import CoreModel as core
import CropPyomo as crppy
import RotationPyomo as rotpy 


#######################
#key options          #
#######################
#eg flk structure, mach option etc - this is set the default, can be changed in runs via saa
        
        

#########################
#Exp loop               #
#########################

#read in data (already manipulated)
# exp_data = pd.read_excel('C:/Users/young/Dropbox/Michael/python/MUDAS 2.0/exp test.xlsx')
# for row in range(len(exp_data)):
#     for column in exp_data:
#         value = exp_data.loc[exp_data.index[row], column]
#         indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in x.split(',')) #creats a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
#         sen[column][indices]=value
# indices=slice(*map(lambda x: int(x.strip()) if x.strip() else None, x.split(':'))) #doesnt work with multi dimension because only creates one slice

con_error=[]
##create loop for exp
for i in rps.lo_bound.keys():#(range(1)): #maybe this should be looping through an exp sheet in excel
    print('Starting exp loop')
    #define any SA - this module will have to import sensitivity module then other module will import sensitivity therefore sensitivity shouldn't inport pre calc modules
    rps.lo_bound[i]=0.1 #('A', 'C', 'N', 'OF', 'm')
    

    #call core model function, must call them in the correct order (core must be last)
    rotpy.rotationpyomo()
    crppy.croppyomo_local()
    core.coremodel_all()
    if core.coremodel_test_var[-1]==1:
        con_error.append(i)
    
    
    # crp.lo_bound[i]=0
    
    
    
    
    
    
    
    
    
    
    