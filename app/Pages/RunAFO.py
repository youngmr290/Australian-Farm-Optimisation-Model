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
import pandas as pd
from datetime import datetime
import pickle as pkl


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
import ReportControl as rep


def f_exp_app(exp_data, n_trials=1, region='GSM'):
    #report the clock time that the experiment was started
    print(f'Experiment commenced at: {time.ctime()}')
    start = time.time()

    ##list of trial names
    trials = exp_data.index[0:n_trials][3]

    ##empty dict to store results
    exp_lp_vars = {}
    exp_r_vals = {}


    ##print out number of trials to run
    total_trials = sum(exp_data.index[row][0] == True for row in range(len(exp_data)))
    print(f'Number of trials to run: {total_trials}')
    print(f'Number of full solutions: {sum((exp_data.index[row][1] == True) and (exp_data.index[row][0] == True) for row in range(len(exp_data)))}')
    print(f'exp.xls last saved: {datetime.fromtimestamp(round(os.path.getmtime("exp.xlsx")))}')
    start_time1 = time.time()

    for row in range(n_trials):
        ##start timer for each loop
        start_time = time.time()

        ##check the rotations and inputs align - this means rotation method can be controlled using a SA
        phs.f1_rot_check()

        ##get trial name - used for outputs
        trial_name = exp_data.index[row][3]
        trial_description = f'{row} {trial_name}'
        print(f'\n{trial_description}, Starting trial at: {time.ctime()}')

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


        ##store pyomo variable output as a dict
        variables=model.component_objects(pe.Var, active=True)
        lp_vars = {str(v):{s:v[s].value for s in v} for v in variables}     #creates dict with variable in it. This is tricky since pyomo returns a generator object
        ##store profit
        lp_vars['profit'] = obj


        ##add to output of previous trials
        exp_lp_vars[trials[n_trials]] = lp_vars
        exp_r_vals[trials[n_trials]] = r_vals

        ##determine expected time to completion - trials left multiplied by average time per trial &time for current loop
        trials_to_go = total_trials - row
        average_time = (time.time() - start_time1) / row   #time since the start of experiment
        remaining = trials_to_go * average_time
        finish_time_expected = time.time() + remaining
        print(f'{trial_description}, total time taken this loop: {time.time() - start_time:.2f}')   #time since start of this loop
        print(f'{trial_description}, Expected finish time: \033[1m{time.ctime(finish_time_expected)}\033[0m (at {time.ctime()})')

    ##report
    stacked_summary, stacked_ffcfw_dams, stacked_ffcfw_offs = rep.f_report(processor=1, trials=trials, app_lp_vars=exp_lp_vars, app_r_vals=exp_r_vals)




    end = time.time()
    print(f'Experiment completed at: {time.ctime()}, total trials completed: {run}')
    try:
        print(f'average time taken for each loop: {(end - start) / run:.2f}') #average time since start of experiment
    except ZeroDivisionError: pass


    return





