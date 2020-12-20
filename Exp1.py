# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: experiment module - this is the module that runs everything and controls kv's

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
import numpy as np
from random import randrange

#report the clock time that the experiment was started
print("Experiment commenced at: ", time.ctime())
start=time.time()

from CreateModel import model
import CreateModel as crtmod
import BoundsPyomo as bdypy
import UniversalInputs as uinp
import PropertyInputs as pinp
import Sensitivity as sen
import Functions as fun
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
import StockPyomo as spy
import CorePyomo as core

start_time1 = time.time()


#########################
#load pickle            # 
#########################
##try to load in params dict, if it doesn't exist then create a new dict
try:
    with open('pkl_params.pkl', "rb") as f:
        params = pkl.load(f)
except FileNotFoundError:
    params={}
prev_params = params.copy() #make a copy to compare with
##try to load in Previous Exp.xlsx file to dict, if it doesn't exist then create a new dict
try:
    with open('pkl_exp.pkl', "rb") as f:
        prev_exp = pkl.load(f)
except FileNotFoundError:
    prev_exp=pd.DataFrame()

if __name__ == '__main__':
    ##try to load in results file to dict, if it doesn't exist then create a new dict - isn't used by multiprocess therefore only needs to be loaded with main
    try:
        with open('pkl_lp_vars.pkl', "rb") as f:
            lp_vars = pkl.load(f)
    except FileNotFoundError:
        lp_vars={}
    ##try to load in results file to dict, if it doesn't exist then create a new dict
    try:
        with open('pkl_r_vals.pkl', "rb") as f:
            r_vals = pkl.load(f)
    except FileNotFoundError:
        r_vals={}

    
#########################
#load exp               # 
#########################
##read in exp and drop all false runs ie runs not being run this time
exp_data = fun.f_read_exp()
exp_data = exp_data.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx
exp_data1=exp_data.copy() #copy made so that the run col can be added - the original df is used to allocate sa values (would cause an error if run col existed but i cant drop it because it is used to determine if the trial is run)



##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xlsx has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial

exp_data1 = fun.f_run_required(prev_exp, exp_data1)


#########################
#Exp loop               #
#########################
#^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
def exp(row):
    ##sleep for random length of time. This is to offset processes with a goal of spreading the RAM load
    time.sleep(randrange(80))

    ##start timer for each loop
    start_time = time.time()
    for dic,key1,key2,indx in exp_data:
         ##extract current value
         value = exp_data.loc[exp_data.index[row], (dic,key1,key2,indx)].squeeze() #squeeze to remove exp header/index leaving just the value
         ##checks if both slice and key2 exists
         if not ('Unnamed' in indx or 'Unnamed' in key2):
             indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(
                 ','))  # creates a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
             if dic == 'sam':
                 sen.sam[(key1, key2)][indices] = value
             elif dic == 'saa':
                 sen.saa[(key1, key2)][indices] = value
             elif dic == 'sap':
                 sen.sap[(key1, key2)][indices] = value
             elif dic == 'sar':
                 sen.sar[(key1, key2)][indices] = value
             elif dic == 'sat':
                 sen.sat[(key1, key2)][indices] = value
             elif dic == 'sav':
                 sen.sav[(key1, key2)][indices] = value

         ##checks if just slice exists
         elif not 'Unnamed' in indx:
             indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(
                 ','))  # creates a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
             if dic == 'sam':
                 sen.sam[key1][indices] = value
             elif dic == 'saa':
                 sen.saa[key1][indices] = value
             elif dic == 'sap':
                 sen.sap[key1][indices] = value
             elif dic == 'sar':
                 sen.sar[key1][indices] = value
             elif dic == 'sat':
                 sen.sat[key1][indices] = value
             elif dic == 'sav':
                 sen.sav[key1][indices] = value
         ##checks if just key2 exists
         elif not 'Unnamed' in key2:
             if dic == 'sam':
                 sen.sam[(key1, key2)] = value
             elif dic == 'saa':
                 sen.saa[(key1, key2)] = value
             elif dic == 'sap':
                 sen.sap[(key1, key2)] = value
             elif dic == 'sar':
                 sen.sar[(key1, key2)] = value
             elif dic == 'sat':
                 sen.sat[(key1, key2)] = value
             elif dic == 'sav':
                 sen.sav[(key1, key2)] = value
         ##if just key1 exists
         else:
             if dic == 'sam':
                 sen.sam[key1] = value
             elif dic == 'saa':
                 sen.saa[key1] = value
             elif dic == 'sap':
                 sen.sap[key1] = value
             elif dic == 'sar':
                 sen.sar[key1] = value
             elif dic == 'sat':
                 sen.sat[key1] = value
             elif dic == 'sav':
                 sen.sav[key1] = value

    ##call sa functions - assigns sa variables to relevant inputs
    uinp.univeral_inp_sa()
    pinp.property_inp_sa()
    ##create empty dicts - have to do it here because need the trial as the first key, so whole trial can be compared when determining if pyomo needs to be run
    ###params
    params={}
    params['pas']={}
    params['rot']={}
    params['crop']={}
    params['mach']={}
    params['fin']={}
    params['labfx']={}
    params['lab']={}
    params['crplab']={}
    params['sup']={}
    params['stub']={}
    params['stock']={}
    ###report values
    r_vals={}
    r_vals['pas']={}
    r_vals['rot']={}
    r_vals['crop']={}
    r_vals['mach']={}
    r_vals['fin']={}
    r_vals['labfx']={}
    r_vals['lab']={}
    r_vals['crplab']={}
    r_vals['sup']={}
    r_vals['stub']={}
    r_vals['stock']={}
    ##call precalcs
    paspy.paspyomo_precalcs(params['pas'],r_vals['pas'])
    rotpy.rotation_precalcs(params['rot'],r_vals['rot'])
    crppy.crop_precalcs(params['crop'],r_vals['crop'])
    macpy.mach_precalcs(params['mach'],r_vals['mach'])
    finpy.fin_precalcs(params['fin'],r_vals['fin'])
    lfixpy.labfx_precalcs(params['labfx'],r_vals['labfx'])
    labpy.lab_precalcs(params['lab'],r_vals['lab'])
    lcrppy.crplab_precalcs(params['crplab'],r_vals['crplab'])
    suppy.sup_precalcs(params['sup'],r_vals['sup'])
    stubpy.stub_precalcs(params['stub'],r_vals['stub'])
    spy.stock_precalcs(params['stock'],r_vals['stock'])

    ##does pyomo need to be run?
    ##check if the two dicts are the same, it is possible that the current dict has less keys than the previous dict eg if a value becomes nan (because you removed the cell in excel inputs) and when it is stacked it disappears (this is very unlikely though so not going to test for it since this step is already slow)
    try: #try required in case the key (trial) doesn't exist in the old dict, if this is the case pyomo must be run
        run_pyomo_params=fun.findDiff(params, prev_params[exp_data.index[row][2]])
    except KeyError:
        run_pyomo_params= True
    ##determine if pyomo should run, note if pyomo doesn't run there will be no ful solution (they are the same as before so no need)
    lp_vars={} #create empty dict to return if pyomo isnt run
    if run_pyomo_params or exp_data1.loc[exp_data1.index[row],'runpyomo'].squeeze():
        ##call core model function, must call them in the correct order (core must be last)
        crtmod.sets() #certain sets have to be updated each iteration of exp
        rotpy.rotationpyomo(params['rot'])
        crppy.croppyomo_local(params['crop'])
        macpy.machpyomo_local(params['mach'])
        finpy.finpyomo_local(params['fin'])
        lfixpy.labfxpyomo_local(params['labfx'])
        labpy.labpyomo_local(params['lab'])
        lcrppy.labcrppyomo_local(params['crplab'])
        paspy.paspyomo_local(params['pas'])
        suppy.suppyomo_local(params['sup'])
        stubpy.stubpyomo_local(params['stub'])
        spy.stockpyomo_local(params['stock'])
        ###bounds-this must be done last because it uses sets built in some of the other modules
        bdypy.boundarypyomo_local()
        results=core.coremodel_all() #have to do this so i can access the solver status
 
        ##check if user wants full solution
        if exp_data.index[row][1] == True:
            ##make lp file
            model.write('Output\%s.lp' %exp_data.index[row][2],io_options={'symbolic_solver_labels':True})  #file name has to have capital
            
            ##write rc and dual to txt file
            with open('Output\Rc and Duals - %s.txt' %exp_data.index[row][2],'w') as f:  #file name has to have capital
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
            with open('Output\Full model - %s.txt' %exp_data.index[row][2], 'w') as f:  #file name has to have capital
                f.write("My description of the instance!\n")
                model.display(ostream=f)
    
        ##This writes variable with value greater than 1 to txt file, the file is overwritten each time - used to check stuff out each iteration if you want 
        file = open('Output\Variable summary.txt','w') #file name has to have capital
        file.write('Trial: %s\n'%exp_data.index[row][2]) #the first line is the name of the trial
        for v in model.component_objects(pe.Var, active=True):
            file.write("Variable %s\n" %v)   #  \n makes new line
            for index in v:
                try:
                    if v[index].value>0:
                        file.write ("   %s %s\n" %(index, v[index].value))
                except: pass 
        file.close()
        ##this prints stuff for each trial - trial name, overall profit
        print("\nDisplaying Solution for trial: %s\n" %exp_data.index[row][2] , '-'*60,'\n%s' %pe.value(model.profit))
        ##this check if the solver is optimal - if infeasible or error the model will quit
        if (results.solver.status == pe.SolverStatus.ok) and (results.solver.termination_condition == pe.TerminationCondition.optimal):
            print('solver optimal')# Do nothing when the solution in optimal and feasible
        elif (results.solver.termination_condition == pe.TerminationCondition.infeasible):
            print ('Solver Status: infeasible')
            sys.exit()
        else: # Something else is wrong
            print ('Solver Status: error')
            sys.exit()
        ##store profit
        r_vals['profit'] = pe.value(model.profit)
        #last step is to print the time for the current trial to run
        variables = model.component_objects(pe.Var, active=True)
        lp_vars = {str(v):{s:v[s].value for s in v} for v in variables }     #creates dict with variable in it. This is tricky since pyomo returns a generator object
    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    dataset = list(np.flatnonzero(np.array(exp_data.index.get_level_values(0)) * np.array(exp_data1['run']))) #gets the ordinal index values for the trials the user wants to run that are not up to date
    processes = multiprocessing.cpu_count()
    total_batches = math.ceil(len(dataset) / processes )
    current_batch = math.ceil( (dataset.index(row)+1) / processes ) #add 1 because python starts at 0
    remaining_batches = total_batches - current_batch
    time_taken = time.time() - start_time1
    batch_time = time_taken / current_batch
    time_remaining = remaining_batches * batch_time
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)
    print('Time remaining: %s' %time_remaining)

    return lp_vars, params, r_vals

##3 - works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each iteration returns a dict and the multiprocess stuff returns a list)
def main():
    ## Define the dataset - trials that require at least the precalcs done (user wants it run and it is out of date)
    dataset = list(np.flatnonzero(np.array(exp_data.index.get_level_values(0)) * np.array(exp_data1['run']))) #gets the ordinal index values for the trials the user wants to run that are not up to date
    ##prints out start status - number of trials to run, date and time exp.xl was last saved and output summary  
    print('Number of trials to run: ',len(dataset))
    print('Number of full solutions: ',sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data))))
    print('Exp.xlsx last saved: ',datetime.fromtimestamp(round(os.path.getmtime("Exp.xlsx"))))
    ##start multiprocessing
    agents = min(multiprocessing.cpu_count(),len(dataset)) # number of agents (processes) should be min of the num of cpus or trial
    with multiprocessing.Pool(processes=agents) as pool:
        result = pool.map(exp, dataset)
    ##update run require status - trials just run are now up to date for both pyomo and precalcs - all trials that the user wanted to run are now up to date (even if they didn't run because they were already up to date)
    exp_data1.loc[exp_data1.index[dataset],['run']] = False
    exp_data1.loc[exp_data1.index[dataset],['runpyomo']] = False
    ##return pyomo results and params dict
    return dataset, result, exp_data1

if __name__ == '__main__':
    dataset, results, exp_data1 = main() #returns a list is the same order of exp
    ##turn list of dicts into nested dict with trial name as key
    for trial_row, result, res_num in zip(dataset,results,range(len(results))):
        if any(results[res_num][0]):  # only do this if pyomo was run and the dict contains values
            lp_vars[exp_data.index[trial_row][2]] = results[res_num][0]
        params[exp_data.index[trial_row][2]] = results[res_num][1] 
        r_vals[exp_data.index[trial_row][2]] = results[res_num][2] 
    ##drop results into pickle file
    with open('pkl_lp_vars.pkl', "wb") as f:
        pkl.dump(lp_vars, f)
    with open('pkl_params.pkl', "wb") as f:
        pkl.dump(params, f)
    with open('pkl_r_vals.pkl', "wb") as f:
        pkl.dump(r_vals, f)
    with open('pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f)


    end=time.time()
    print('total time',end-start)







