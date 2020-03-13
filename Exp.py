# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: exriment module - this is the module that runs everything and controls kv's

@author: young
"""
#import datetime
import pandas as pd
from pyomo.environ import *
import time

import UniversalInputs as uinp
import PropertyInputs as pinp 
import Sensitivity as sen 
import RotationPyomo as rotpy 
import CropPyomo as crppy
import MachPyomo as macpy
import FinancePyomo as finpy
import LabourFixedPyomo as lfixpy 
import LabourPyomo as labpy 
import LabourCropPyomo as lcrppy 
import PasturePyomo as paspy 
import CoreModel as core

print('running exp')
#######################
#key options          #
#######################
#eg flk structure, mach option etc - this is set the default, can be changed in runs via saa
        







#########################
#Exp loop               #
#########################
#^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
##read in exp log 
exp_data = pd.read_excel('exp.xlsx',index_col=[0,1], header=[0,1,2,3])
start_time1 = time.time()
run=0 #counter to work out average time per loop
for row in range(len(exp_data)):
    ##start timer for each loop
    start_time = time.time()
    ##check to make sure user wants to run this trial
    if exp_data.index[row][0] == False:
        continue
    run+=1
    for dic,key1,key2,indx in exp_data:
        ##extract current value
        value = exp_data.loc[exp_data.index[row], (dic,key1,key2,indx)]
        ##checks if both slice and key2 exists
        if not ('Unnamed' in indx  or 'Unnamed' in key2):
            indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(',')) #creats a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
            if dic == 'sam':
                sen.sam[(key1,key2)][indices]=value
            elif dic == 'saa':
                sen.saa[(key1,key2)][indices]=value
            elif dic == 'sap':
                sen.sap[(key1,key2)][indices]=value

        ##checks if just slice exists
        elif not 'Unnamed' in indx:
            indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(',')) #creats a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
            if dic == 'sam':
                sen.sam[key1][indices]=value
            elif dic == 'saa':
                sen.saa[key1][indices]=value
            elif dic == 'sap':
                sen.sap[key1][indices]=value
        ##checks if just key2 exists
        elif not 'Unnamed' in key2:
            if dic == 'sam':
                sen.sam[(key1,key2)]=value
            elif dic == 'saa':
                sen.saa[(key1,key2)]=value
            elif dic == 'sap':
                sen.sap[(key1,key2)]=value


    ##call sa functions - assigns sa variables to relevant inputs
    uinp.univeral_inp_sa()
    pinp.property_inp_sa()
    ##call core model function, must call them in the correct order (core must be last)
    rotpy.rotationpyomo()
    crppy.croppyomo_local()
    macpy.machpyomo_local()
    finpy.finpyomo_local()
    lfixpy.labfxpyomo_local()
    labpy.labpyomo_local()
    lcrppy.labcrppyomo_local()
    paspy.paspyomo_local()
    core.coremodel_all()
    
    ##last step is to print the time for the current trial to run
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)

end_time1 = time.time()
print('total trials completed: ', run)
print("average time taken for each loop: ", (end_time1 - start_time1)/run)

# ##the stuff below will be superseeded with stuff above 

# con_error=[]
# con_correct=[]
# ##create loop for exp
# # for i in rps.lo_bound.keys():#(range(1)): #maybe this should be looping through an exp sheet in excel
# # # for i in (range(1)): #maybe this should be looping through an exp sheet in excel
# # #     print('Starting exp loop')
# # #     #define any SA - this module will have to import sensitivity module then other module will import sensitivity therefore sensitivity shouldn't inport pre calc modules
# #     rps.lo_bound[i]=0.1 #('A', 'C', 'N', 'OF', 'm')
    

# #     #call core model function, must call them in the correct order (core must be last)
# #     rotpy.rotationpyomo()
# #     crppy.croppyomo_local()
# #     core.coremodel_all()
# #     if core.coremodel_test_var[-1]==1:
# #         con_error.append(i)
# #     print(core.coremodel_test_var[-1])
    
# #     rps.lo_bound[i]=0
# import random
# for p in range(1):
        
#     # for j,i in zip(range(2000,2054), rps.lo_bound.keys()):#(range(1)): #maybe this should be looping through an exp sheet in excel
#     bounded_rots=[]
#     for i in random.sample(range(2441),1):#(range(1)): #maybe this should be looping through an exp sheet in excel
#     # # # for i in (range(1)): #maybe this should be looping through an exp sheet in excel
#     # # #     print('Starting exp loop')
#     # # #     #define any SA - this module will have to import sensitivity module then other module will import sensitivity therefore sensitivity shouldn't inport pre calc modules
        
#         k=list(rps.lo_bound.keys())[i] #('A', 'C', 'N', 'OF', 'm')
#         bounded_rots.append(k)
#         # rps.lo_bound['GX3NPo']=6 #('A', 'C', 'N', 'OF', 'm')
#         rps.lo_bound['GA5ENw']=6 #('A', 'C', 'N', 'OF', 'm')
#         # rps.lo_bound['GNEPb']=6 #('A', 'C', 'N', 'OF', 'm')
#     # print(rps.lo_bound)
    
#     #call core model function, must call them in the correct order (core must be last)
#         rotpy.rotationpyomo()
#         print('rps done')
#         crppy.croppyomo_local()
#         print('crop done')
#         core.coremodel_all()
#         print('cpre done')
#         if core.coremodel_test_var[-1]==1:
#             con_error.append(k)
#         else: 
#             con_correct.append(k)
#         print(core.coremodel_test_var[-1])
    
#         rps.lo_bound[k]=0
    
    
    
    
    
    
    
    
    
    
    