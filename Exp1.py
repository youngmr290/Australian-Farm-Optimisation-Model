# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: exriment module - this is the module that runs everything and controls kv's

@author: young
"""


import pandas as pd
import pyomo.environ as pe
import time
import math
import os.path
from datetime import datetime
import multiprocessing
import pickle as pkl
import sys

start=time.time()
from CreateModel import model
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
import SupFeedPyomo as suppy
import StubblePyomo as stubpy
import CoreModel as core


##read in exp and drop all false runs ie runs not being run this time
exp_data = pd.read_excel('exp.xlsx',index_col=[0,1,2], header=[0,1,2,3])
exp_data=exp_data.loc[True] #alternative ... exp_data.iloc[exp_data.index.get_level_values(0)index.levels[0]==True]
start_time1 = time.time()

if __name__ == '__main__':
    ##try to load in results file to dict, if it doesn't exist then create a new dict
    try:
        with open('pkl_results', "rb") as f:
            var_results = pkl.load(f)
    except FileNotFoundError:
        var_results={}
    ##prints out start status - number of trials to run, date and time exp.xl was last saved and output summary  
    print('Number of trials to run: ',len(exp_data))
    print('Number of full solutions: ',sum(exp_data.index[row][0] == True for row in range(len(exp_data))))
    print('Exp.xlsx last saved: ',datetime.fromtimestamp(round(os.path.getmtime("Exp.xlsx"))))

#########################
#Exp loop               #
#########################
#^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
def exp(row):
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
    suppy.suppyomo_local()
    stubpy.stubpyomo_local()
    results=core.coremodel_all() #required to access the solver status
     

    ##need to save results to a dict here - include the trial name as the dict name or key.. probably need to return the dict at the end of the function so it can be joined with other processors
    
    ##check if user wants full solution
    if exp_data.index[row][0] == True:
        ##make lp file
        model.write('%s.lp' %exp_data.index[row][1],io_options={'symbolic_solver_labels':True})
        
        ##write rc and dual to txt file
        with open('Rc and Duals - %s.txt' %exp_data.index[row][1],'w') as f:
            f.write('RC\n')        
            for v in model.component_objects(pe.Var, active=True):
                f.write("Variable %s\n" %v)   #  \n makes new line
                for index in v:
                    try:
                        print("      ", index, model.rc[v[index]], file=f)
                    except: pass 
            f.write('Dual\n')   #this can be used in search to find the start of this in the txt file     
            for c in model.component_objects(pe.Constraint, active=True):
                f.write("Constraint %s\n" %c)   #  \n makes new line
                for index in c:
                    # try:
                    print("      ", index, model.dual[c[index]], file=f)
                    # except: pass 
        
        ##prints what you see from pprint to txt file - you can see the slack on constraints but not the rc or dual
        with open('full model - %s.txt' %exp_data.index[row][1], 'w') as f:
            f.write("My description of the instance!\n")
            model.display(ostream=f)
    
    ##This writes variable with value greater than 1 to txt file, the file is overwritten each time - used to check stuff out each iteration if you want 
    file = open('variable summary.txt','w') 
    file.write('Trial: %s\n'%exp_data.index[row][1]) #the first line is the name of the trial
    for v in model.component_objects(pe.Var, active=True):
        file.write("Variable %s\n" %v)   #  \n makes new line
        for index in v:
            try:
                if v[index].value>0:
                    file.write ("   %s %s\n" %(index, v[index].value))
            except: pass 
    file.close()
    
       
    ##this prints stuff for each trial - trial name, overall profit
    print("\nDisplaying Solution for trial: %s\n" %exp_data.index[row][1] , '-'*60,'\n%s' %pe.value(model.profit))
    ##this check if the solver is optimal - if infeasible or error the model will quit
    if (results.solver.status == pe.SolverStatus.ok) and (results.solver.termination_condition == pe.TerminationCondition.optimal):
        print('solver optimal')# Do nothing when the solution in optimal and feasible
    elif (results.solver.termination_condition == pe.TerminationCondition.infeasible):
        print ('Solver Status: infeasible')
        sys.exit()
    else: # Something else is wrong
        print ('Solver Status: error')
        sys.exit()
    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    processes = multiprocessing.cpu_count()
    total_batches = math.ceil(len(exp_data) / processes ) 
    current_batch = math.ceil( (row+1) / processes ) #add 1 because python starts at 0
    remaining_batches = total_batches - current_batch
    time_taken = time.time() - start_time1
    batch_time = time_taken / current_batch
    time_remaining = remaining_batches * batch_time
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)
    print('Time remaining: %s' %time_remaining)
    
    #last step is to print the time for the current trial to run
    variables = model.component_objects(pe.Var, active=True)
    return {str(v):{s:v[s].value for s in v} for v in variables }     #creates dict with variable in it. This is tricky since pyomo returns a generator object


##3 - works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each itteration returns a dict and the multiprocess stuff returns a list)
def main():
      # Define the dataset
    inputs = (list(range(len(exp_data))))
    dataset = inputs

    # number of agents (processes) should be min of the num of cpus or trial
    agents = min(multiprocessing.cpu_count(),len(inputs))
    with multiprocessing.Pool(processes=agents) as pool:
        result = pool.map(exp, dataset)
    return result
if __name__ == '__main__':
    results=main() #returns a list is the same order of exp
    ##turn list of dicts into nested dict with trial name as key
    for result, trial_row in zip(results,range(len(results))):
        var_results[exp_data.index[trial_row][1]] = results[trial_row] 
    with open('pkl_results', "wb") as f:
        pkl.dump(var_results, f)

    end=time.time()
    print('total time',end-start)










