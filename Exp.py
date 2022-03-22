# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 21:00:13 2019

module: experiment module - this is the module that runs everything and controls kv's

@author: young
"""
#import datetime
import numpy as np
import pyomo.environ as pe
import time
import os.path
import json
from datetime import datetime
import pickle as pkl

##used to trace memory
# import tracemalloc
# tracemalloc.start(10)
# snapshots = []


import CreateModel as crtmod
import BoundsPyomo as bndpy
import StructuralInputs as sinp
import UniversalInputs as uinp
import PropertyInputs as pinp
import Sensitivity as sen
import Functions as fun
import RotationPyomo as rotpy
import Phase as phs
import PhasePyomo as phspy
import MachPyomo as macpy
import FinancePyomo as finpy
import LabourFixedPyomo as lfixpy
import LabourPyomo as labpy
import LabourPhasePyomo as lphspy
import PasturePyomo as paspy
import SupFeedPyomo as suppy
import CropResiduePyomo as stubpy
import StockPyomo as spy
import CorePyomo as core
import MVF as mvf
import CropGrazingPyomo as cgzpy
import SeasonPyomo as zgenpy
import FeedSupplyStock as fsstk

#report the clock time that the experiment was started
print(f'Experiment commenced at: {time.ctime()}')
start = time.time()

##settings
force_run=True #force precalcs to be run
run_pyomo = True #do you want pyomo to run (default is True but if testing reports it can be useful to only run the precalcs)
directory_path = os.path.dirname(os.path.abspath(__file__)) #path of directory - required when exp is run from a different location (eg in the web app)

#########################
#Exp loop               #
#########################
##read in exp.xl and determine which trials are in the experiment group.
exp_data, exp_group_bool = fun.f_read_exp()
exp_data1=exp_data.copy() #copy made so that the run and runpyomo cols can be added - the original df is used to allocate sa values (would cause an error if run col existed but i can't drop it because it is used to determine if the trial is run)


##check if precalcs and pyomo need to be recalculated.
##precalcs are rerun if
##  1. exp.xls has changed
##  2. any python module has been updated
##  3. the trial needed to be run last time but the user opted not to run that trial
exp_data1 = fun.f_run_required(exp_data1)

if __name__ == '__main__':
    ##check Output and pkl folders exist for outputs. If not create.
    if os.path.isdir('pkl'):
        pass
    else:
        os.mkdir('pkl')
    if os.path.isdir('Output'):
        pass
    else:
        os.mkdir('Output')
    if os.path.isdir('Output/infeasible'):
        pass
    else:
        os.mkdir('Output/infeasible')

    ##plk a copy of exp. Used next time the model is run to identify which trials are up to date.
    with open('pkl/pkl_exp.pkl', "wb") as f:
        pkl.dump(exp_data1, f, protocol=pkl.HIGHEST_PROTOCOL)

##cut exp_data based on the experiment group
exp_data = fun.f_group_exp(exp_data, exp_group_bool)
exp_data1 = fun.f_group_exp(exp_data1, exp_group_bool)


##print out number of trials to run
total_trials = sum(exp_data.index[row][0] == True for row in range(len(exp_data)))
print(f'Number of trials to run: {total_trials}')
print(f'Number of full solutions: {sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data)))}')
print(f'exp.xls last saved: {datetime.fromtimestamp(round(os.path.getmtime("exp.xlsx")))}')
start_time1 = time.time()
run=0 #counter to work out average time per loop
for row in range(len(exp_data)):
    ##start timer for each loop
    start_time = time.time()

    ##check the rotations and inputs align - this means rotation method can be controlled using a SA
    phs.f1_rot_check()

    ##check to make sure user wants to run this trial - note pyomo is never run without precalcs being run (this could possibly be change by making a more custom function to check only precalc module time and then altering the 'continue' call below)
    if exp_data1.index[row][0] == False or (exp_data1.loc[exp_data1.index[row],'run_req'].squeeze()==False and force_run==False):
        continue   # move to next row of the trial

    ##get trial name - used for outputs
    trial_name = exp_data.index[row][3]
    trial_description = f'{run} {trial_name}'
    print(f'\n{trial_description}, Starting trial at: {time.ctime()}')
    if run_pyomo != True:
        print("\n **** Pyomo is turned off... are you sure? ****\n")

    ##tally trials run
    run+=1

    ##update sensitivity values
    fun.f_update_sen(row,exp_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav
                     ,sen.sam_inp,sen.saa_inp,sen.sap_inp,sen.sar_inp,sen.sat_inp,sen.sav_inp)

    ##call sa functions - assigns sa variables to relevant inputs
    sinp.f_structural_inp_sa()
    uinp.f_universal_inp_sa()
    pinp.f_property_inp_sa()
    ##expand p6 axis to include nodes
    sinp.f1_expand_p6()
    pinp.f1_expand_p6()

    ##create empty dicts - have to do it here because need the trial as the first key, so whole trial can be compared when determining if pyomo needs to be run
    ###params
    params={}
    params['zgen']={}
    params['rot']={}
    params['crop']={}
    params['crpgrz']={}
    params['mach']={}
    params['fin']={}
    params['labfx']={}
    params['lab']={}
    params['crplab']={}
    params['sup']={}
    params['stock']={}
    params['stub']={}
    params['pas']={}
    ###report values
    r_vals={}
    r_vals['zgen']={}
    r_vals['rot']={}
    r_vals['crop']={}
    r_vals['crpgrz']={}
    r_vals['mach']={}
    r_vals['fin']={}
    r_vals['labfx']={}
    r_vals['lab']={}
    r_vals['crplab']={}
    r_vals['sup']={}
    r_vals[ 'stock']={}
    r_vals['stub']={}
    r_vals['pas']={}
    nv = {} #dict to store nv params from stockgen to be used in pasture
    pkl_fs_info = {} #dict to store info required to pkl feedsupply

    ##call precalcs
    precalc_start = time.time()
    zgenpy.season_precalcs(params['zgen'],r_vals['zgen'])
    rotpy.rotation_precalcs(params['rot'],r_vals['rot'])
    phspy.crop_precalcs(params['crop'],r_vals['crop'])
    macpy.mach_precalcs(params['mach'],r_vals['mach'])
    finpy.fin_precalcs(params['fin'],r_vals['fin'])
    lfixpy.labfx_precalcs(params['labfx'],r_vals['labfx'])
    labpy.lab_precalcs(params['lab'],r_vals['lab'])
    lphspy.crplab_precalcs(params['crplab'],r_vals['crplab'])
    suppy.sup_precalcs(params['sup'],r_vals['sup'])
    spy.stock_precalcs(params['stock'],r_vals['stock'],nv,pkl_fs_info)
    cgzpy.cropgraze_precalcs(params['crpgrz'],r_vals['crpgrz'], nv) #cropgraze must be after stock because it uses nv dict which is populated in stock.py
    stubpy.stub_precalcs(params['stub'],r_vals['stub'], nv) #stub must be after stock because it uses nv dict which is populated in stock.py
    paspy.paspyomo_precalcs(params['pas'],r_vals['pas'], nv) #pas must be after stock because it uses nv dict which is populated in stock.py
    precalc_end = time.time()
    print(f'{trial_description}, total time for precalcs: {precalc_end - precalc_start:.2f} finished at {time.ctime()}')

    
    ##determine if pyomo should run, note if pyomo doesn't run there will be no full solution (they are the same as before so no need)
    if run_pyomo:
        ##call pyomo model function, must call them in the correct order (core must be last)
        pyomocalc_start = time.time()
        model = pe.ConcreteModel() #create pyomo model - done each loop because memory was being leaked when just deleting and re adding the components.
        crtmod.sets(model, nv) #certain sets have to be updated each iteration of exp - has to be first since other modules use the sets
        zgenpy.f1_seasonpyomo_local(params['zgen'], model)  # has to be first since builds params used in other modules
        rotpy.f1_rotationpyomo(params['rot'], model)
        phspy.f1_croppyomo_local(params['crop'], model)
        macpy.f1_machpyomo_local(params['mach'], model)
        finpy.f1_finpyomo_local(params['fin'], model)
        lfixpy.f1_labfxpyomo_local(params['labfx'], model)
        labpy.f1_labpyomo_local(params['lab'], model)
        lphspy.f1_labcrppyomo_local(params['crplab'], model)
        paspy.f1_paspyomo_local(params['pas'], model)
        suppy.f1_suppyomo_local(params['sup'], model)
        cgzpy.f1_cropgrazepyomo_local(params['crpgrz'], model)
        stubpy.f1_stubpyomo_local(params['stub'], model)
        spy.f1_stockpyomo_local(params['stock'], model)
        mvf.f1_mvf_pyomo(model)
        ###bounds-this must be done last because it uses sets built in some of the other modules
        bndpy.f1_boundarypyomo_local(params, model)

        pyomocalc_end = time.time()
        print(f'{trial_description}, time for localpyomo: {pyomocalc_end - pyomocalc_start:.2f} finished at {time.ctime()}')
        obj = core.coremodel_all(trial_name, model)
        print(f'{trial_description}, time for corepyomo: {time.time() - pyomocalc_end:.2f} finished at {time.ctime()}')

        ##This writes variable summary each iteration with generic file name - it is overwritten each iteration and is created so the run progress can be monitored
        fun.write_variablesummary(model, row, exp_data, obj, 1)

        ##check if user wants full solution
        if exp_data.index[row][1] == True:
            ##make lp file
            model.write(os.path.join(directory_path, 'Output/%s.lp' %trial_name), io_options={'symbolic_solver_labels':True})  #file name has to have capital

            ##This writes variable summary for full solution (same file as the temporary version created above)
            fun.write_variablesummary(model, row, exp_data, obj)

            ##prints what you see from pprint to txt file - you can see the slack on constraints but not the rc or dual
            with open(os.path.join(directory_path, 'Output/Full model - %s.txt' %trial_name), 'w') as f:  #file name has to have capital
                f.write("My description of the instance!\n")
                model.display(ostream=f)

            ##write rc, duals and slacks to txt file. Duals are slow to write so that option must be turn on
            write_duals = True
            with open(os.path.join(directory_path, 'Output/Rc and Duals - %s.txt' %trial_name),'w') as f:  #file name has to have capital
                f.write('RC\n')
                for v in model.component_objects(pe.Var, active=True):
                    f.write("Variable %s\n" %v)
                    for index in v:
                        try: #in case variable has no index
                            print("      ", index, model.rc[v[index]], file=f)
                        except: pass
                f.write('Slacks (no entry means no slack)\n')  # this can be used in search to find the start of this in the txt file
                for c in model.component_objects(pe.Constraint,active=True):
                    f.write("Constraint %s\n" % c)
                    for index in c:
                        if c[index].lslack() != 0 and c[index].lslack() != np.inf:
                            print("  L   ",index,c[index].lslack(),file=f)
                        if c[index].uslack() != 0 and c[index].lslack() != np.inf:
                            print("  U   ",index,c[index].uslack(),file=f)
                if write_duals:
                    f.write('Dual\n')   #this can be used in search to find the start of this in the txt file
                    for c in model.component_objects(pe.Constraint, active=True):
                        f.write("Constraint %s\n" %c)
                        for index in c:
                            print("      ", index, model.dual[c[index]], file=f)

        ##store pyomo variable output as a dict
        variables=model.component_objects(pe.Var, active=True)
        lp_vars = {str(v):{s:v[s].value for s in v} for v in variables}     #creates dict with variable in it. This is tricky since pyomo returns a generator object
        ##store profit
        lp_vars['profit'] = obj
        ##store mvf rc
        lp_vars['mvf'] = {}
        for v in model.component_objects(pe.Var, active=True):
            if str(v)=='v_mvf':
                for index in v:
                    lp_vars['mvf'][index] = model.rc[v[index]]

        ##pickle lp info - only if pyomo is run
        with open(os.path.join(directory_path, 'pkl/pkl_lp_vars_{0}.pkl'.format(trial_name)),"wb") as f:
            pkl.dump(lp_vars,f,protocol=pkl.HIGHEST_PROTOCOL)
    ##pickle report values - every time a trial is run (even if pyomo not run)
    with open(os.path.join(directory_path, 'pkl/pkl_r_vals_{0}.pkl'.format(trial_name)),"wb") as f:
        pkl.dump(r_vals,f,protocol=pkl.HIGHEST_PROTOCOL)

    ##call function to store optimal feedsupply
    fsstk.f1_pkl_feedsupply(lp_vars,r_vals,pkl_fs_info)

    ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
    trials_to_go = total_trials - run
    average_time = (time.time() - start_time1) / run   #time since the start of experiment
    remaining = trials_to_go * average_time
    finish_time_expected = time.time() + remaining
    print(f'{trial_description}, total time taken this loop: {time.time() - start_time:.2f}')   #time since start of this loop
    print(f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m (at {time.ctime()})')

end = time.time()
print(f'Experiment completed at: {time.ctime()}, total trials completed: {run}')
try:
    print(f'average time taken for each loop: {(end - start) / run:.2f}') #average time since start of experiment
except ZeroDivisionError: pass


##code to track memory. Add this in at the end of exp loop
    # import gc
    # gc.collect()
    # snapshots.append(tracemalloc.take_snapshot())
    # if len(snapshots) > 1:
    #
    #     stats = snapshots[-1].compare_to(snapshots[-2], 'filename')
    #     # stats = snapshots[-1].compare_to(snapshots[-2],'traceback')
    #
    #     for stat in stats[:10]:
    #         print(stat)
    #         # for line in stat.traceback.format():
    #         #     print(line)



