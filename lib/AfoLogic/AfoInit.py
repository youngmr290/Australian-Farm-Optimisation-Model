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
import sys
from datetime import datetime
import multiprocessing
import pickle as pkl
import json
import numpy as np

##used to trace memory
# import tracemalloc
# tracemalloc.start(10)
# snapshots = []

from . import InputTest as inptest
from . import CreateModel as crtmod
from . import BoundsPyomo as bndpy
from . import StructuralInputs as sinp
from . import UniversalInputs as uinp
from . import PropertyInputs as pinp
from . import Sensitivity as sen
from . import Functions as fun
from . import RotationPyomo as rotpy
from . import Phase as phs
from . import PhasePyomo as phspy
from . import MachPyomo as macpy
from . import FinancePyomo as finpy
from . import LabourFixedPyomo as lfixpy
from . import LabourPyomo as labpy
from . import LabourPhasePyomo as lphspy
from . import PasturePyomo as paspy
from . import SupFeedPyomo as suppy
from . import CropResiduePyomo as stubpy
from . import StockPyomo as spy
from . import CorePyomo as core
from . import MVF as mvf
from . import CropGrazingPyomo as cgzpy
from . import SeasonPyomo as zgenpy
from . import FeedSupplyStock as fsstk
from . import SaltbushPyomo as slppy


#########################
#Exp loop               #
#########################

def exp(solver_method, user_data, property, trial_name, trial_description, sinp_defaults, uinp_defaults, pinp_defaults,
        d_rot_info, cat_propn_s1_ks2, pkl_fs, print_debug_output, a_lmuregion_lmufarmer=None, mp_lp_vars_path='pkl/pkl_lp_vars_MP Initial position.pkl'):

    ##can use logger to get status on multiprocessing
    # logger.info('Received {}'.format(row))

    ##select property and reset default inputs for the current trial. Must occur first.
    sinp.f_select_n_reset_sinp(sinp_defaults)
    sinp.f_landuse_sets()
    uinp.f_select_n_reset_uinp(uinp_defaults)
    pinp.f_select_n_reset_pinp(property, pinp_defaults)

    ##for the web app need to update LMU axis
    if a_lmuregion_lmufarmer is not None:
        pinp.f_farmer_lmu_adj(a_lmuregion_lmufarmer)

    ##update sensitivity values
    sen.create_sa()
    fun.f_update_sen(user_data,sen.sam,sen.saa,sen.sap,sen.sar,sen.sat,sen.sav)

    ##call sa functions - assigns sa variables to relevant inputs
    sinp.f_structural_inp_sa(sinp_defaults)
    uinp.f_universal_inp_sa(uinp_defaults)
    pinp.f_property_inp_sa(pinp_defaults)

    ##expand p6 axis to include nodes
    sinp.f1_expand_p6()
    pinp.f1_expand_p6()

    ##check the rotations and inputs align - this means rotation method can be controlled using a SA
    d_rot_info = pinp.f1_phases(d_rot_info)

    ##preform inputs tests
    inptest.f_input_logic_test()
    inptest.f_input_shape_test()
    inptest.f_input_value_test()

    ##mask lmu
    pinp.f1_mask_lmu()

    ##mask land use
    pinp.f1_mask_landuse()
    uinp.f1_mask_landuse()

    if sinp.structuralsa['model_is_MP']:
        ###allow user to specify the trial name of the MP setup run
        if sen.sav['MP_setup_trial_name'] != "-":
            mp_lp_vars_path = "pkl/pkl_lp_vars_{0}.pkl".format(sen.sav['MP_setup_trial_name'])
        with open(mp_lp_vars_path, "rb") as f:
            MP_lp_vars = pkl.load(f)
    else:
        MP_lp_vars = {}

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
    params['slp']={}
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
    r_vals['stock']={}
    r_vals['slp']={}
    r_vals['stub']={}
    r_vals['pas']={}
    nv = {} #dict to store nv params from StockGenerator to be used in pasture
    pkl_fs_info = {}  # dict to store info required to pkl feedsupply

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
    spy.stock_precalcs(params['stock'],r_vals['stock'],nv,pkl_fs_info, pkl_fs)
    suppy.sup_precalcs(params['sup'],r_vals['sup'], nv) #sup must be after stock because it uses nv dict which is populated in stock.py
    cgzpy.cropgraze_precalcs(params['crpgrz'],r_vals['crpgrz'], nv) #cropgraze must be after stock because it uses nv dict which is populated in stock.py
    slppy.saltbush_precalcs(params['slp'],r_vals['slp'], nv) #saltbush must be after stock because it uses nv dict which is populated in stock.py
    stubpy.stub_precalcs(params['stub'],r_vals['stub'], nv, cat_propn_s1_ks2) #stub must be after stock because it uses nv dict which is populated in stock.py
    paspy.paspyomo_precalcs(params['pas'],r_vals['pas'], nv) #pas must be after stock because it uses nv dict which is populated in stock.py
    precalc_end = time.time()
    print(f'{trial_description}, total time for precalcs: {precalc_end - precalc_start:.2f} finished at {time.ctime()}')

    ##call core model function, must call them in the correct order (core must be last)
    pyomocalc_start = time.time()
    model = pe.ConcreteModel() #create pyomo model - done each loop because memory was being leaked when just deleting and re adding the components.
    crtmod.sets(model, nv) #certain sets have to be updated each iteration of exp - has to be first since other modules use the sets
    zgenpy.f1_seasonpyomo_local(params['zgen'], model) #has to be first since builds params used in other modules
    rotpy.f1_rotationpyomo(params['rot'], model, MP_lp_vars)
    phspy.f1_croppyomo_local(params['crop'], model)
    macpy.f1_machpyomo_local(params['mach'], model)
    finpy.f1_finpyomo_local(params['fin'], model)
    lfixpy.f1_labfxpyomo_local(params['labfx'], model)
    labpy.f1_labpyomo_local(params['lab'], model)
    lphspy.f1_labcrppyomo_local(params['crplab'], model)
    paspy.f1_paspyomo_local(params['pas'], model, MP_lp_vars)
    suppy.f1_suppyomo_local(params['sup'], model)
    cgzpy.f1_cropgrazepyomo_local(params['crpgrz'], model)
    slppy.f1_saltbushpyomo_local(params['slp'], model, MP_lp_vars)
    stubpy.f1_stubpyomo_local(params['stub'], model, MP_lp_vars)
    spy.f1_stockpyomo_local(params['stock'], model, MP_lp_vars)
    mvf.f1_mvf_pyomo(model)
    ###bounds-this must be done last because it uses sets built in some of the other modules
    bndpy.f1_boundarypyomo_local(params, model)
    pyomocalc_end = time.time()
    print(f'{trial_description}, time for localpyomo: {pyomocalc_end - pyomocalc_start:.2f} finished at {time.ctime()}')
    profit, obj, trial_infeasible = core.coremodel_all(trial_name, model, solver_method, nv, print_debug_output, MP_lp_vars)
    print(f'{trial_description}, time for corepyomo: {time.time() - pyomocalc_end:.2f} finished at {time.ctime()}')

    ##build lp_vars
    variables=model.component_objects(pe.Var, active=True)
    lp_vars = {str(v):{s:v[s].value for s in v} for v in variables}     #creates dict with variable in it. This is tricky since pyomo returns a generator object
    ##store profit and obj
    lp_vars['profit'] = profit
    lp_vars['utility'] = obj
    ##store mvf rc
    lp_vars['mvf'] = {}
    for v in model.component_objects(pe.Var, active=True):
        if str(v)=='v_mvf':
            for index in v:
                try:
                    lp_vars['mvf'][index] = model.rc[v[index]]
                except:
                    lp_vars['mvf'][index] = 0  #if model doesn't solve then RC might not exist so replace with 0


    return model, profit, trial_infeasible, lp_vars, r_vals, pkl_fs_info, d_rot_info

