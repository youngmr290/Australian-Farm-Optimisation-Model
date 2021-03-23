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
import os
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
import BoundsPyomo as bndpy
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

##used to get status on multiprocessing
# import logging
# logger = multiprocessing.log_to_stderr(logging.DEBUG)

## the upper limit of number of processes (concurrent trials) based on the memory capacity of this machine
maximum_processes = 8  # available memory / value determined by size of the model being run (~5GB for the small model)

start_time1 = time.time()


    
#########################
#load exp               # 
#########################
##read in exp and drop all false runs ie runs not being run this time
exp_data = fun.f_read_exp()
exp_data = exp_data.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xls
exp_data1=exp_data.copy() #copy made so that the run col can be added - the original df is used to allocate sa values (would cause an error if run col existed but i can't drop it because it is used to determine if the trial is run)



##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xls has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial

exp_data1 = fun.f_run_required(exp_data1)

##plk a copy of exp in case the code crashes before the end. (this is tracks if a trial needed to be run)
if __name__ == '__main__':
    try:
        with open('pkl/pkl_exp.pkl', "wb") as f:
            pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)
    except FileNotFoundError:
        os.mkdir('pkl')
        with open('pkl/pkl_exp.pkl', "wb") as f:
            pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)


#########################
#Exp loop               #
#########################
#^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
def exp(row):  # called with command: pool.map(exp, dataset)
    ##sleep for random length of time. This is to offset processes with a goal of spreading the RAM load
    # time.sleep(randrange(30))

    ##can use logger to get status on multiprocessing
    # logger.info('Received {}'.format(row))
    ##start timer for each loop
    start_time = time.time()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    run = row - list(exp_data.index.get_level_values(0))[:row].count(False)
    print(time.ctime()," : Starting trial %d, %s" %(run, trial_name))

    ##updaye sensitivity values
    fun.f_update_sen(row,exp_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)

    ##call sa functions - assigns sa variables to relevant inputs
    uinp.universal_inp_sa()
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
    ev = {} #dict to store ev params from StockGenerator to be used in pasture
    ##call precalcs
    rotpy.rotation_precalcs(params['rot'],r_vals['rot'])
    crppy.crop_precalcs(params['crop'],r_vals['crop'])
    macpy.mach_precalcs(params['mach'],r_vals['mach'])
    finpy.fin_precalcs(params['fin'],r_vals['fin'])
    lfixpy.labfx_precalcs(params['labfx'],r_vals['labfx'])
    labpy.lab_precalcs(params['lab'],r_vals['lab'])
    lcrppy.crplab_precalcs(params['crplab'],r_vals['crplab'])
    suppy.sup_precalcs(params['sup'],r_vals['sup'])
    stubpy.stub_precalcs(params['stub'],r_vals['stub'])
    spy.stock_precalcs(params['stock'],r_vals['stock'],ev)
    paspy.paspyomo_precalcs(params['pas'],r_vals['pas'],ev)

    ##does pyomo need to be run? In exp1 pyomo is always run because creating params file take up lots of time, RAM and disc space
    ###read in prev params for trial
    ##try to load in params dict, if it doesn't exist then create a new dict
    # try:
    #     with open('pkl/pkl_params_{0}.pkl'.format(trial_name),"rb") as f:
    #         prev_params = pkl.load(f)
    # except FileNotFoundError:
    #     prev_params = {}
    # ##check if the two dicts are the same, it is possible that the current dict has less keys than the previous dict eg if a value becomes nan (because you removed the cell in excel inputs) and when it is stacked it disappears (this is very unlikely though so not going to test for it since this step is already slow)
    # try: #try required in case the key (trial) doesn't exist in the old dict, if this is the case pyomo must be run
    #     run_pyomo_params=fun.findDiff(params, prev_params)
    # except KeyError:
    #     run_pyomo_params = True
    run_pyomo_params = True
    ##determine if pyomo should run, note if pyomo doesn't run there will be no full solution (they are the same as before so no need)
    lp_vars={} #create empty dict to return if pyomo isn't run. If dict is empty it doesnt overwrite the previous main lp_vars dict66
    if run_pyomo_params:
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
        bndpy.boundarypyomo_local()
        results=core.coremodel_all(params) #have to do this so i can access the solver status

        ##This writes variable summary each iteration with generic file name - it is overwritten each iteration and is created so the run progress can be monitored
        fun.write_variablesummary(model, row, exp_data, 1)

        ##check if user wants full solution
        if exp_data.index[row][1] == True:
            ##make lp file
            model.write('Output/%s.lp' %trial_name,io_options={'symbolic_solver_labels':True})  #file name has to have capital

            ##This writes variable summary for full solution (same file as the temporary version created above)
            fun.write_variablesummary(model, row, exp_data)

            ##write rc and dual to txt file
            with open('Output/Rc and Duals - %s.txt' %trial_name,'w') as f:  #file name has to have capital
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
            # with open('Output/Full model - %s.txt' %trial_name, 'w') as f:  #file name has to have capital
            #     f.write("My description of the instance!\n")
            #     model.display(ostream=f)

        ##this prints stuff for each trial - trial name, overall profit
        print("\nDisplaying Solution for trial: %s\n" %trial_name , '-'*60,'\n%s' %pe.value(model.profit))
        ##this check if the solver is optimal - if infeasible or error the model will quit
        if (results.solver.status == pe.SolverStatus.ok) and (results.solver.termination_condition == pe.TerminationCondition.optimal):
            print('solver optimal')# Do nothing when the solution in optimal and feasible
        elif (results.solver.termination_condition == pe.TerminationCondition.infeasible):
            print ('Solver Status: infeasible')
            sys.exit()
        else: # Something else is wrong
            print ('Solver Status: error')
            sys.exit()
        #last step is to print the time for the current trial to run
        variables = model.component_objects(pe.Var, active=True)
        lp_vars = {str(v):{s:v[s].value for s in v} for v in variables }     #creates dict with variable in it. This is tricky since pyomo returns a generator object
        ##store profit
        lp_vars['profit'] = pe.value(model.profit)

    ##pickle trial info
    if any(lp_vars):  # only do this if pyomo was run and the dict contains values
        with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"wb") as f:
            pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"wb") as f:
        pkl.dump(r_vals,f,protocol=pkl.HIGHEST_PROTOCOL)
    # with open('pkl/pkl_params_{0}.pkl'.format(trial_name),"wb") as f:  #pkl_params must be pickled last because it is used to determine if model crashed but the current trial was complete prior to crash
    #     pkl.dump(params,f,protocol=pkl.HIGHEST_PROTOCOL)

    ##track the successful execution of trial - so we don't update a trial that didn't finish
    trials_successfully_run = row

    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    dataset = list(np.flatnonzero(np.array(exp_data.index.get_level_values(0)) * np.array(exp_data1['run']))) #gets the ordinal index values for the trials the user wants to run that are not up to date
    processes = min(multiprocessing.cpu_count(), len(dataset), maximum_processes)
    total_batches = math.ceil(len(dataset) / processes )
    current_batch = math.ceil( (dataset.index(row)+1) / processes ) #add 1 because python starts at 0
    remaining_batches = total_batches - current_batch
    time_taken = time.time() - start_time1
    batch_time = time_taken / current_batch
    time_remaining = remaining_batches * batch_time
    end_time = time.time()
    print("total time taken this loop: ", end_time - start_time)
    print('Time remaining: %s' %time_remaining)

    return trials_successfully_run

##3 - works when run through anaconda prompt - if 9 runs and 8 processors, the first processor to finish, will start the 9th run
#using map it returns outputs in the order they go in ie in the order of the exp
##the result after the different processes are done is a list of dicts (because each iteration returns a dict and the multiprocess stuff returns a list)
def main():
    ## Define the dataset - trials that require at least the precalcs done (user wants it run and it is out of date)
    dataset = list(np.flatnonzero(np.array(exp_data.index.get_level_values(0)) * np.array(exp_data1['run']))) #gets the ordinal index values for the trials the user wants to run that are not up to date
    ##prints out start status - number of trials to run, date and time exp.xl was last saved and output summary  
    print('Number of trials to run: ',len(dataset))
    print('Number of full solutions: ',sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data))))
    print('Exp.xls last saved: ',datetime.fromtimestamp(round(os.path.getmtime("exp.xlsm"))))
    ##start multiprocessing
    ### number of agents (processes) should be min of the num of cpus, number of trials or the user specified limit due to memory capacity
    agents = min(multiprocessing.cpu_count(), len(dataset), maximum_processes)
    with multiprocessing.Pool(processes=agents) as pool:
        trials_successfully_run = pool.map(exp, dataset)

    ##update run require status - trials just run are now up to date for both pyomo and precalcs - all trials that the user wanted to run are now up to date (even if they didn't run because they were already up to date)
    exp_data1.loc[exp_data1.index[trials_successfully_run],['run']] = False
    exp_data1.loc[exp_data1.index[trials_successfully_run],['runpyomo']] = False
    ##return pyomo results and params dict
    return exp_data1

if __name__ == '__main__':
    exp_data1 = main() #returns a list is the same order of exp
    # ##turn list of dicts into nested dict with trial name as key
    # for trial_row, result, res_num in zip(dataset,results,range(len(results))):
    #     if any(results[res_num][0]):  # only do this if pyomo was run and the dict contains values
    #         lp_vars[exp_data.index[trial_row][2]] = results[res_num][0]
    #     params[exp_data.index[trial_row][2]] = results[res_num][1]
    #     r_vals[exp_data.index[trial_row][2]] = results[res_num][2]
    ##drop results into pickle file
    # with open('pkl_lp_vars.pkl', "wb") as f:
    #     pkl.dump(lp_vars, f, protocol=pkl.HIGHEST_PROTOCOL)
    # with open('pkl_params.pkl', "wb") as f:
    #     pkl.dump(params, f, protocol=pkl.HIGHEST_PROTOCOL)
    # with open('pkl_r_vals.pkl', "wb") as f:
    #     pkl.dump(r_vals, f, protocol=pkl.HIGHEST_PROTOCOL)
    with open('pkl/pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)


    end=time.time()
    print('total time',end-start)







