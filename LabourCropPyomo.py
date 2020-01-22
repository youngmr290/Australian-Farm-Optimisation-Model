# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:13:44 2019

module - labour crop pyomo stuff

@author: young
"""
#python modules
from pyomo.environ import *

#MUDAS modules
from MachPyomo import *
from CreateModel import *
from LabourCropInputs import *
import LabourCrop as lc


'''
#define parameters
'''
#harvest helper time per crop
model.harv_helper = Param(model.s_crops, initialize=crop_labour_input['harvest_helper'], default = 0.0, doc='harvest helper time per crop')
#mach prep and pack time 
model.prep_pack = Param(model.s_periods, initialize=lc.prep_labour(), default = 0.0, doc='harvest helper time per crop')

model.fert_app_hour_tonne = Param(model.s_periods, model.s_fert_type, initialize= lc.fert_app_time_t(), default = 0.0, doc='time required for fert application per tonne of each fert (filling up and driving to paddock cost)')
model.fert_app_hour_ha = Param(model.s_phases, model.s_lmus, model.s_periods, initialize= lc.fert_app_time_ha(), default = 0.0, doc='time required for fert application per ha of each fert (driving around paddock cost)')


'''
constraints and functions used to make core model constraints
'''
###################################
#functions for core model         #
###################################
#labour req by seeding and harv, includes helper time
def mach_labour(model,p):
    seed_labour = sum(sum(model.mach_days_seeding[p, k, l] for k in model.s_crops) for l in model.s_lmus)        \
    * mach_input_data_general['daily_seed_hours'] *(1 + crop_labour_input['seeding_helper'])
    harv_labour = sum(model.hours_harv[p,k] * (1 + model.harv_helper[k])  for k in model.s_periods)  
    prep_labour = model.prep_pack[p]
    return seed_labour + harv_labour + prep_labour

#hours of fert application - 1) per tonne 2) per ha 
def fert_app_labour(model,p):
    per_t_time = sum(sum(sum(model.phasefert[r,l,n]*model.num_phase[l,r]*(model.fert_app_hour_tonne[p,n]/1000)  for r in model.s_rotations)for l in model.s_lmus)for n in model.s_fert_type ) 
    per_ha_time = sum(sum(model.num_phase[l,r]*(model.fert_app_hour_ha[r,l,p]/1000) for r in model.s_phases) for l in model.s_lmus)   
    return per_t_time + per_ha_time 
