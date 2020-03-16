# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: exriment module - this is the module that runs everything and controls kv's

@author: young
"""
import pandas as pd
from pyomo.environ import *
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

start=time.time()
from CreateModel import *
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

#######################
#key options          #
#######################
#eg flk structure, mach option etc - this is set the default, can be changed in runs via saa
        
    
    
    

   



#########################
#Exp loop               #
#########################
#^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?

##read in exp and drop all false runs ie runs not being run this time
exp_data = pd.read_excel('exp.xlsx',index_col=[0,1], header=[0,1,2,3])
exp_data=exp_data.loc[True] #alternative ... exp_data.iloc[exp_data.index.get_level_values(0)index.levels[0]==True]
   
def exp(row):
    print('running exp: ',exp_data.index[row] )
    ##start timer for each loop
    start_time = time.time()
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
     
      ##need to save results to a dict here - include the trial name as the dict name or key.. probably need to return the dict at the end of the function so it can be joined with other processors
     
    #last step is to print the time for the current trial to run
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)
    var = model.component_objects(Var, active=True)
    return {str(v):{s:v[s].value for s in v} for v in a }     #creates dict with variable in it. This is tricky since pyomo returns a generator object




##3 - works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each itteration returns a dict and the multiprocess stuff returns a list)
def main():
      # Define the dataset
    inputs = (list(range(len(exp_data)))) 
    dataset = inputs

    # Output the dataset
    print ('Dataset: ' + str(dataset))

    # number of agents (processes) should be min of the num of cpus or trial
    agents = min(multiprocessing.cpu_count(),len(inputs))
    with multiprocessing.Pool(processes=agents) as pool:
        result = pool.map(exp, dataset)
    return result
if __name__ == '__main__':
    a=main() #returns a list is the same order of exp
    end=time.time()
    print('total time',end-start)





    
    
    
    
    
    