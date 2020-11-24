# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: exriment module - this is the module that runs everything and controls kv's

@author: young
"""
#import datetime
import pandas as pd
import pyomo.environ as pe
import time
import os.path
import glob
from datetime import datetime
import pickle as pkl
import sys


import CreateModel as crtmod #need bot
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
import StockPyomo as spy
import CorePyomo as core
import Report as rep


#########################
#load pickle            # 
#########################
##try to load in lp variable dict, if it doesn't exist then create a new dict
try:
    with open('pkl_lp_vars.pkl', "rb") as f:
        lp_vars = pkl.load(f)
except FileNotFoundError:
    lp_vars={}

##try to load in params dict, if it doesn't exist then create a new dict
try:
    with open('pkl_params.pkl', "rb") as f:
        params = pkl.load(f)
except FileNotFoundError:
    params={}
prev_params = params.copy() #make a copy to compare with

##try to load in results file to dict, if it doesn't exist then create a new dict
try:
    with open('pkl_r_vals.pkl', "rb") as f:
        r_vals = pkl.load(f)
except FileNotFoundError:
    r_vals={}

##try to load in Previous Exp.xlsx file to dict, if it doesn't exist then create a new dict
try:
    with open('pkl_exp.pkl', "rb") as f:
        prev_exp = pkl.load(f)
except FileNotFoundError:
    prev_exp=pd.DataFrame()
    
#########################
#Exp loop               # #^maybe there is a cleaner way to do some of the stuff below ie a way that doesn't need as many if statements?
#########################
##read in exp log 
exp_data = pd.read_excel('exp.xlsx',index_col=[0,1,2], header=[0,1,2,3])
exp_data = exp_data.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx
exp_data1=exp_data.copy() #copy made so that the run col can be added - the origional df is used to allocate sa values (would cause an error if run col existed but i cant drop it because it is used to determine if the trial is run)
exp_data1['run']=False
exp_data1['runpyomo']=False

##have any sa cols been added or removed, are the values the same, has the py code changed since last run?
###get a list of all sa cols (including the name of the trial because two trial may have the same values but have a different name)
keys_hist = list(prev_exp.reset_index().columns[2:-1].values) #:-1 so that the runpyomo col doesn't effect the run pre calcs.
keys_current = list(exp_data1.reset_index().columns[2:-1].values)
sorted_list = sorted(glob.iglob('*.py'), key=os.path.getctime)
####if only report.py has been updated precalcs don't need to be re-run therefore newest is equal to the newest py file that isn't report
if sorted_list[-1] != 'Repoprt.py':
    newest = sorted_list[-1]
else: newest = sorted_list[-2]
newest_pyomo = max(glob.iglob('*pyomo.py'), key=os.path.getctime)

###if pyomo code has not been updated then check each row and see if it needs to be run - if it is a new trial it will need to be run or if it needed to be run last time but didn't get run because user specified not to run that exp (this is done by comparing the prev_exp with current exp that has all false)
try: #incase pkl_exp doesn't exist
    if os.path.getmtime("pkl_exp") >= os.path.getmtime(newest_pyomo):
        i1 = prev_exp.reset_index().set_index([('level_2', '', '', ''),('runpyomo', '', '', '')]).index #have to reset index because the name of the trial is going to be included in the new index so it must first be dropped from current index
        i2 = exp_data1.reset_index().set_index([('level_2', '', '', ''),('runpyomo', '', '', '')]).index
        exp_data1.loc[~i2.isin(i1),'runpyomo'] = True
    ###if pyomo code has been updated then all pyomo must be updated
    else: exp_data1['runpyomo']=True
except FileNotFoundError: exp_data1['runpyomo']=True
###if headers are the same,pyomo code is the same and the excel inputs are the same then test if the values in exp.xlxs are the same
try: #incase pkl_exp doesn't exist
    if keys_current==keys_hist and os.path.getmtime("pkl_exp") >= os.path.getmtime(newest) and os.path.getmtime("pkl_exp") >= os.path.getmtime("Universal.xlsx") and os.path.getmtime("pkl_exp") >= os.path.getmtime("Property.xlsx"):
        ###check if each exp has the same values in exp.xlsx as last time it was run.
        i3 = prev_exp.reset_index().set_index(keys_hist).index #have to reset index because the name of the trial is going to be included in the new index so it must first be dropped from current index
        i4 = exp_data1.reset_index().set_index(keys_current).index
        exp_data1.loc[~i4.isin(i3),'run'] = True
    ###if headers are different or py code has changed then all trials need to be re-run
    else: exp_data1['run']=True
except FileNotFoundError: exp_data1['run']=True

##print out number of trials to run
total_trials=sum(exp_data.index[row][0] == True for row in range(len(exp_data)))
print('Number of trials to run: ',total_trials)
print('Number of full solutions: ',sum(exp_data.index[row][1] == True for row in range(len(exp_data))))
print('Exp.xlsx last saved: ',datetime.fromtimestamp(round(os.path.getmtime("Exp.xlsx"))))
start_time1 = time.time()
run=0 #counter to work out average time per loop
for row in range(len(exp_data)):
    ##start timer for each loop
    start_time = time.time()
    ##check to make sure user wants to run this trial - note pyomo is never run without precalcs being run (this could possibly be change by making a more custom functin to check only precalc module time and then altering the 'continue' call below)
    if exp_data1.index[row][0] == False or exp_data1.loc[exp_data1.index[row],'run'].squeeze()==False:
        continue
    # print('precalcs',exp_data1.index[row][2])
    exp_data1.loc[exp_data1.index[row],'run'] = False
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
            elif dic == 'sar':
                sen.sar[(key1,key2)][indices]=value
            elif dic == 'sat':
                sen.sat[(key1,key2)][indices]=value
            elif dic == 'sav':
                sen.sav[(key1,key2)][indices]=value

        ##checks if just slice exists
        elif not 'Unnamed' in indx:
            indices = tuple(slice(*(int(i) if i else None for i in part.strip().split(':'))) for part in indx.split(',')) #creats a slice object from a string - note slice objects are not inclusive ie to select the first number it should look like [0:1]
            if dic == 'sam':
                sen.sam[key1][indices]=value
            elif dic == 'saa':
                sen.saa[key1][indices]=value
            elif dic == 'sap':
                sen.sap[key1][indices]=value
            elif dic == 'sar':
                sen.sar[key1][indices]=value
            elif dic == 'sat':
                sen.sat[key1][indices]=value
            elif dic == 'sav':
                sen.sav[key1][indices]=value
        ##checks if just key2 exists
        elif not 'Unnamed' in key2:
            if dic == 'sam':
                sen.sam[(key1,key2)]=value
            elif dic == 'saa':
                sen.saa[(key1,key2)]=value
            elif dic == 'sap':
                sen.sap[(key1,key2)]=value
            elif dic == 'sar':
                sen.sar[(key1,key2)]=value
            elif dic == 'sat':
                sen.sat[(key1,key2)]=value
            elif dic == 'sav':
                sen.sav[(key1,key2)]=value
        ##if just key1 exists
        else:
            if dic == 'sam':
                sen.sam[key1]=value
            elif dic == 'saa':
                sen.saa[key1]=value
            elif dic == 'sap':
                sen.sap[key1]=value
            elif dic == 'sar':
                sen.sar[key1]=value
            elif dic == 'sat':
                sen.sat[key1]=value
            elif dic == 'sav':
                sen.sav[key1]=value

    ##call sa functions - assigns sa variables to relevant inputs
    uinp.univeral_inp_sa()
    pinp.property_inp_sa()
    ##create empty dicts - have to do it here because need the trial as the first key, so whole trial can be compared when determining if pyomo needs to be run
    ###params
    params[exp_data.index[row][2]]={}
    params[exp_data.index[row][2]]['pas']={}
    params[exp_data.index[row][2]]['rot']={}
    params[exp_data.index[row][2]]['crop']={}
    params[exp_data.index[row][2]]['mach']={}
    params[exp_data.index[row][2]]['fin']={}
    params[exp_data.index[row][2]]['labfx']={}
    params[exp_data.index[row][2]]['lab']={}
    params[exp_data.index[row][2]]['crplab']={}
    params[exp_data.index[row][2]]['sup']={}
    params[exp_data.index[row][2]]['stub']={}
    params[exp_data.index[row][2]]['stock']={}
    ###report values
    r_vals[exp_data.index[row][2]]={}
    r_vals[exp_data.index[row][2]]['pas']={}
    r_vals[exp_data.index[row][2]]['stock']={}
    ##call precalcs
    precalc_start = time.time()
    paspy.paspyomo_precalcs(params[exp_data.index[row][2]]['pas'],r_vals[exp_data.index[row][2]]['pas'])
    rotpy.rotation_precalcs(params[exp_data.index[row][2]]['rot'])
    crppy.crop_precalcs(params[exp_data.index[row][2]]['crop'])
    macpy.mach_precalcs(params[exp_data.index[row][2]]['mach'])
    finpy.fin_precalcs(params[exp_data.index[row][2]]['fin'])
    lfixpy.labfx_precalcs(params[exp_data.index[row][2]]['labfx'])
    labpy.lab_precalcs(params[exp_data.index[row][2]]['lab'])
    lcrppy.crplab_precalcs(params[exp_data.index[row][2]]['crplab'])
    suppy.sup_precalcs(params[exp_data.index[row][2]]['sup'])
    stubpy.stub_precalcs(params[exp_data.index[row][2]]['stub'])
    spy.stock_precalcs(params[exp_data.index[row][2]]['stock'], r_vals[exp_data.index[row][2]]['stock'])
    precalc_end = time.time()
    print('precalcs: ', precalc_end - precalc_start)
    
    
    ##does pyomo need to be run?
    ##check if two param dicts are the same.
    def findDiff(d1, d2):
        a=False
        for k in d1:
            # if a != True: #this stops it looping through the rest of the keys once it finds a difference
                if (k not in d2): #check if the key in current params is in previous params dict.
                    # print('DIFFERENT')
                    a = True
                    return a
                else:
                    if type(d1[k]) is dict:
                        # print('going level deeper',k)
                        a=findDiff(d1[k],d2[k])
                        # print(k,a)
                    else:
                        if d1[k] != d2[k]: #if keys are the same, check if the values are the same
                            # print('DIFFERENT',k)
                            a=(True)
                            return a
            # else: return a
        return a
    ##check if the two dicts are the same, it is possible that the current dict has less keys than the previous dict eg if a value becomes nan (because you removed the cell in exvel inputs) and when it is stacked it disapears (this is very unlikely though so not going to test for it since this step is already slow)
    try: #try required incase the key (trial) doesn't exist in the old dict, if this is the case pyomo must be run
        run_pyomo_params=findDiff(params[exp_data.index[row][2]], prev_params[exp_data.index[row][2]])
    except KeyError:
        run_pyomo_params= True
    ##determine if pyomo should run, note if pyomo doesn't run there will be no ful solution (they are the same as before so no need)
    if run_pyomo_params or exp_data1.loc[exp_data1.index[row],'runpyomo'].squeeze():
        # print('run pyomo')
        ###if re-run update runpyomo to false 
        exp_data1.loc[exp_data1.index[row],'runpyomo'] = False
        ##call pyomo model function, must call them in the correct order (core must be last)
        precalc_start = time.time()
        crtmod.sets() #certain sets have to be updated each iteration of exp
        rotpy.rotationpyomo(params[exp_data.index[row][2]]['rot'])
        crppy.croppyomo_local(params[exp_data.index[row][2]]['crop'])
        macpy.machpyomo_local(params[exp_data.index[row][2]]['mach'])
        finpy.finpyomo_local(params[exp_data.index[row][2]]['fin'])
        lfixpy.labfxpyomo_local(params[exp_data.index[row][2]]['labfx'])
        labpy.labpyomo_local(params[exp_data.index[row][2]]['lab'])
        lcrppy.labcrppyomo_local(params[exp_data.index[row][2]]['crplab'])
        paspy.paspyomo_local(params[exp_data.index[row][2]]['pas'])
        suppy.suppyomo_local(params[exp_data.index[row][2]]['sup'])
        stubpy.stubpyomo_local(params[exp_data.index[row][2]]['stub'])
        spy.stockpyomo_local(params[exp_data.index[row][2]]['stock'])
        precalc_end = time.time()
        print('localpyomo: ', precalc_end - precalc_start)
        results=core.coremodel_all() #have to do this so i can access the solver status
        print('corepyomo: ',time.time() - precalc_end)
        ##check if user wants full solution
        if exp_data.index[row][1] == True:
            ##make lp file
            model.write('%s.lp' %exp_data.index[row][2],io_options={'symbolic_solver_labels':True})
               
            ##write rc and dual to txt file
            with open('Rc and Duals - %s.txt' %exp_data.index[row][2],'w') as f:
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
            with open('full model - %s.txt' %exp_data.index[row][2], 'w') as f:
                f.write("My description of the instance!\n")
                model.display(ostream=f)
        
        ##This writes variable with value greater than 1 to txt file - used to check stuff out each iteration if you want 
        file = open('variable summary.txt','w') 
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
    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    trials_to_go = total_trials - run 
    time_taken= time.time()
    average_time = (time_taken- start_time1)/run
    remaining = trials_to_go * average_time
    print("total time taken this loop: ", time_taken - start_time)
    print('Time remaining: %s' %remaining)
    

    ##store pyomo variable output as a dict
    variables=model.component_objects(pe.Var, active=True)
    lp_vars['%s'%exp_data.index[row][2]]={str(v):{s:v[s].value for s in v} for v in variables }    #creates dict with variable in it. This is tricky since pyomo returns a generator object

##drop results into pikle file
with open('pkl_lp_vars.pkl', "wb") as f:
    pkl.dump(lp_vars, f)
with open('pkl_params.pkl', "wb") as f:
    pkl.dump(params, f)
with open('pkl_r_vals.pkl', "wb") as f:
    pkl.dump(r_vals, f)
with open('pkl_exp.pkl', "wb") as f:
    pkl.dump(exp_data1, f)

end_time1 = time.time()
print('total trials completed: ', run)
try:
    print("average time taken for each loop: ", (end_time1 - start_time1)/run)
except ZeroDivisionError: pass
    
#############################
# Reports and intermidiates #
#############################
inter={}
##create intermidiates for each trial
for row in range(len(exp_data)):
    ##check to make sure user wants to run this trial
    if exp_data1.index[row][0] == True:
        inter[exp_data.index[row][2]]={}
        rep.intermediates(inter[exp_data.index[row][2]], r_vals[exp_data.index[row][2]], lp_vars[exp_data.index[row][2]])  
##create reports
rep.report1(inter)
    