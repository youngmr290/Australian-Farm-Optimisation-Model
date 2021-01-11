# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: experiment module - this is the module that runs everything and controls kv's

@author: young
"""
#import datetime
import pandas as pd
import numpy as np
import pyomo.environ as pe
import time
import os.path
import sys
from datetime import datetime
import pickle as pkl

#report the clock time that the experiment was started
print("Experiment commenced at: ", time.ctime())

import CreateModel as crtmod
import BoundsPyomo as bndpy
from CreateModel import model
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

force_run=True #force precalcs to be run


# #########################
# #load pickle            #
# #########################
# ##try to load in lp variable dict, if it doesn't exist then create a new dict
# try:
#     with open('pkl_lp_vars.pkl', "rb") as f:
#         lp_vars = pkl.load(f)
# except FileNotFoundError:
#     lp_vars={}
#
# ##try to load in params dict, if it doesn't exist then create a new dict
# try:
#     with open('pkl_params.pkl', "rb") as f:
#         params = pkl.load(f)
# except FileNotFoundError:
#     params={}
# prev_params = params.copy() #make a copy to compare with
#
# ##try to load in results file to dict, if it doesn't exist then create a new dict
# try:
#     with open('pkl_r_vals.pkl', "rb") as f:
#         r_vals = pkl.load(f)
# except FileNotFoundError:
#     r_vals={}
#
# ##try to load in Previous Exp.xlsx file to dict, if it doesn't exist then create a new dict
# try:
#     with open('pkl_exp.pkl', "rb") as f:
#         prev_exp = pkl.load(f)
# except FileNotFoundError:
#     prev_exp=pd.DataFrame()
    
#########################
#Exp loop               # #^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
#########################
##read in exp log 

exp_data = fun.f_read_exp()
exp_data = exp_data.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx
exp_data1=exp_data.copy() #copy made so that the run and runpyomo cols can be added - the original df is used to allocate sa values (would cause an error if run col existed but i cant drop it because it is used to determine if the trial is run)


##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xlsx has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial

exp_data1 = fun.f_run_required(exp_data1)

##plk a copy of exp incase the code crashes before the end. (this is tracks if a trial needed to be run)
if __name__ == '__main__':
    with open('pkl/pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)


##print out number of trials to run
total_trials=sum(exp_data.index[row][0] == True for row in range(len(exp_data)))
print('Number of trials to run: ',total_trials)
print('Number of full solutions: ',sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data))))
print('exp.xlsx last saved: ',datetime.fromtimestamp(round(os.path.getmtime("exp.xlsx"))))
start_time1 = time.time()
run=0 #counter to work out average time per loop
for row in range(len(exp_data)):
    ##start timer for each loop
    start_time = time.time()

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][2]

    ##check to make sure user wants to run this trial - note pyomo is never run without precalcs being run (this could possibly be change by making a more custom function to check only precalc module time and then altering the 'continue' call below)
    if exp_data1.index[row][0] == False or (exp_data1.loc[exp_data1.index[row],'run'].squeeze()==False and force_run==False):
        continue
    # print('precalcs',exp_data1.index[row][2])
    exp_data1.loc[exp_data1.index[row],('run', '', '', '')] = False
    run+=1

    ##update sensitivity values
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
    ##call precalcs
    precalc_start = time.time()
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
    precalc_end = time.time()
    print('precalcs: ', precalc_end - precalc_start)
    
    
    ##does pyomo need to be run?
    ##check if the two dicts are the same, it is possible that the current dict has less keys than the previous dict eg if a value becomes nan (because you removed the cell in excel inputs) and when it is stacked it disappears (this is very unlikely though so not going to test for it since this step is already slow)
    ##try to load in params dict, if it doesn't exist then create a new dict
    try:
        with open('pkl/pkl_params_{0}.pkl'.format(trial_name),"rb") as f:
            prev_params = pkl.load(f)
    except FileNotFoundError:
        prev_params = {}
    ##check if the two dicts are the same, it is possible that the current dict has less keys than the previous dict eg if a value becomes nan (because you removed the cell in excel inputs) and when it is stacked it disappears (this is very unlikely though so not going to test for it since this step is already slow)
    try: #try required in case the key (trial) doesn't exist in the old dict, if this is the case pyomo must be run
        run_pyomo_params=fun.findDiff(params, prev_params)
    except KeyError:
        run_pyomo_params= True
    lp_vars={} #create empty dict to return if pyomo isn't run. If dict is empty it doesnt overwrite the previous main lp_vars
    ##determine if pyomo should run, note if pyomo doesn't run there will be no ful solution (they are the same as before so no need)
    if run_pyomo_params or exp_data1.loc[exp_data1.index[row],'runpyomo'].squeeze():
        # print('run pyomo')
        ##call pyomo model function, must call them in the correct order (core must be last)
        pyomocalc_start = time.time()
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
        ###if re-run update runpyomo to false
        exp_data1.loc[exp_data1.index[row], ('runpyomo', '', '', '')] = False
        ###bounds-this must be done last because it uses sets built in some of the other modules
        bndpy.boundarypyomo_local()

        pyomocalc_end = time.time()
        print('localpyomo: ', pyomocalc_end - pyomocalc_start)
        results=core.coremodel_all() #have to do this so i can access the solver status
        print('corepyomo: ',time.time() - pyomocalc_end)

        ##check if user wants full solution
        if exp_data.index[row][1] == True:
            ##make lp file
            model.write('Output/%s.lp' %exp_data.index[row][2], io_options={'symbolic_solver_labels':True})  #file name has to have capital
               
            ##write rc and dual to txt file
            with open('Output/Rc and Duals - %s.txt' %exp_data.index[row][2],'w') as f:  #file name has to have capital
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
            # with open('Output/Full model - %s.txt' %exp_data.index[row][2], 'w') as f:  #file name has to have capital
            #     f.write("My description of the instance!\n")
            #     model.display(ostream=f)
        
            ##This writes variable with value greater than 1 to txt file - used to check stuff out each iteration if you want
            file = open('Output/Variable summary %s.txt' %exp_data.index[row][2],'w') #file name has to have capital
            file.write('Trial: %s\n'%exp_data.index[row][2]) #the first line is the name of the trial
            file.write('{0} profit: {1}\n'.format(exp_data.index[row][2], pe.value(model.profit))) #the second line is profit
            for v in model.component_objects(pe.Var, active=True):
                file.write("Variable %s\n" %v)   #  \n makes new line
                for index in v:
                    try:
                        if v[index].value>0.0001:
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
        ##store pyomo variable output as a dict
        variables=model.component_objects(pe.Var, active=True)
        lp_vars['%s'%exp_data.index[row][2]]={str(v):{s:v[s].value for s in v} for v in variables}    #creates dict with variable in it. This is tricky since pyomo returns a generator object
        ##store profit
        lp_vars[exp_data.index[row][2]]['profit'] = pe.value(model.profit)

    ##pickle trial info
    if any(lp_vars):  # only do this if pyomo was run and the dict contains values
        with open('pkl/pkl_lp_vars_{0}.pkl'.format(trial_name),"wb") as f:
            pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)
    with open('pkl/pkl_params_{0}.pkl'.format(trial_name),"wb") as f:
        pkl.dump(params,f,protocol=pkl.HIGHEST_PROTOCOL)
    with open('pkl/pkl_r_vals_{0}.pkl'.format(trial_name),"wb") as f:
        pkl.dump(r_vals,f,protocol=pkl.HIGHEST_PROTOCOL)

    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    trials_to_go = total_trials - run
    time_taken= time.time()
    average_time = (time_taken- start_time1)/run
    remaining = trials_to_go * average_time
    print("total time taken this loop: ", time_taken - start_time)
    print('Time remaining: %s' %remaining)



##drop results into pickle file
with open('pkl/pkl_exp.pkl', "wb") as f:
    pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)

end_time1 = time.time()
print('total trials completed: ', run)
try:
    print("average time taken for each loop: ", (end_time1 - start_time1)/run)
except ZeroDivisionError: pass
    
