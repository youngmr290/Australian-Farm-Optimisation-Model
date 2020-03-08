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
import LabourCrop as lcrp
import PropertyInputs as pinp


def labcrppyomo_local():
    #########
    #param  #
    #########    
    try:
        model.del_component(model.p_harv_helper)
    except AttributeError:
        pass
    model.p_harv_helper = Param(model.s_crops, initialize=pinp.labour['harvest_helper'].squeeze().to_dict(), default = 0.0, doc='harvest helper time per crop')

    try:
        model.del_component(model.p_prep_pack)
    except AttributeError:
        pass
    model.p_prep_pack = Param(model.s_periods, initialize=lcrp.prep_labour(), default = 0.0, doc='harvest helper time per crop')
    
    try:
        model.del_component(model.p_fert_app_hour_tonne_index)
        model.del_component(model.p_fert_app_hour_tonne)
    except AttributeError:
        pass
    model.p_fert_app_hour_tonne = Param(model.s_periods, model.s_fert_type, initialize= lcrp.fert_app_time_t(), default = 0.0, doc='time required for fert application per tonne of each fert (filling up and driving to paddock cost)')
 
    try:
        model.del_component(model.p_fert_app_hour_ha_index_index_0)
        model.del_component(model.p_fert_app_hour_ha_index)
        model.del_component(model.p_fert_app_hour_ha)
    except AttributeError:
        pass
    model.p_fert_app_hour_ha = Param(model.s_phases, model.s_lmus, model.s_periods, initialize= lcrp.fert_app_time_ha(), default = 0.0, doc='time required for fert application per ha of each fert (driving around paddock cost)')
    
    try:
        model.del_component(model.p_chem_app_lab_index_index_0)
        model.del_component(model.p_chem_app_lab_index)
        model.del_component(model.p_chem_app_lab)
    except AttributeError:
        pass
    model.p_chem_app_lab = Param(model.s_phases, model.s_lmus, model.s_periods, initialize= lcrp.chem_app_time_ha(), default = 0.0, doc='time required for chem application per ha (hr/ha)')


###################################
#functions for core model         #
###################################
#labour req by 
def mach_labour(model,p):
    '''
    Parameters
    ----------

    p : Set
        Period set from pyomo.

    Returns
    -------
    Pyomo function for core model
        All landuse labour;
        1- seeding and harv, includes helper time
        2- fert application, per tonne & per ha 
        3- chem application
    '''
    seed_labour = sum(sum(model.v_seeding_machdays[p, k, l] for k in model.s_crops) for l in model.s_lmus)        \
    * pinp.mach['daily_seed_hours'] *(1 + pinp.labour['seeding_helper'])
    harv_labour = sum(model.v_harv_hours[p,k] * (1 + model.p_harv_helper[k])  for k in model.s_harvcrops)  
    prep_labour = model.p_prep_pack[p]
    fert_t_time = sum(sum(sum(model.p_phasefert[r,l,n]*model.v_phase_area[r,l]*(model.p_fert_app_hour_tonne[p,n]/1000)  for r in model.s_phases)for l in model.s_lmus)for n in model.s_fert_type ) 
    fert_ha_time = sum(sum(model.v_phase_area[r,l]*(model.p_fert_app_hour_ha[r,l,p]/1000) for r in model.s_phases) for l in model.s_lmus)   
    chem_time = sum(sum(model.v_phase_area[r,l]*(model.p_chem_app_lab[r,l,p]/1000) for r in model.s_phases) for l in model.s_lmus)   
    return seed_labour + harv_labour + prep_labour + fert_t_time + fert_ha_time + chem_time









