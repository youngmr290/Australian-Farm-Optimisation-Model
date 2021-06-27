
"""
author: young
"""
#python modules
import pyomo.environ as pe

#AFO modules
import LabourCrop as lcrp
import PropertyInputs as pinp


def crplab_precalcs(params, r_vals):
    '''
    Call crop labour precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    lcrp.f_labcrop_params(params, r_vals)

def labcrppyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    #########
    #param  #
    #########

    ##used to index the season key in params
    season = pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]

    model.p_harv_helper = pe.Param(model.s_crops, initialize=params['harvest_helper'], default = 0.0, doc='harvest helper time per crop')
    
    model.p_daily_seed_hours = pe.Param(initialize=params['daily_seed_hours'], default = 0.0, doc='machine hours per day of seeding ie labour required per mach day')
    
    model.p_seeding_helper = pe.Param( initialize=params['seeding_helper'], default = 0.0, doc='proportion of time helper is needed for seeding')

    model.p_prep_pack = pe.Param(model.s_labperiods, initialize=params[season]['prep_labour'], default = 0.0, mutable=False, doc='labour for preparation and packing up for seeding and harv')
    
    model.p_fert_app_hour_tonne = pe.Param(model.s_labperiods, model.s_fert_type, initialize= params[season]['fert_app_time_t'], default = 0.0, mutable=False, doc='time required for fert application per tonne of each fert (filling up and driving to paddock cost)')
 
    model.p_fert_app_hour_ha = pe.Param(model.s_phases, model.s_lmus, model.s_labperiods, initialize= params[season]['fert_app_time_ha'], default = 0.0, mutable=False, doc='time required for fert application per ha of each fert (driving around paddock cost)')
    
    model.p_chem_app_lab = pe.Param(model.s_phases, model.s_lmus, model.s_labperiods, initialize= params[season]['chem_app_time_ha'], default = 0.0, mutable=False, doc='time required for chem application per ha (hr/ha)')

    model.p_variable_crop_monitor = pe.Param(model.s_phases, model.s_labperiods, initialize= params[season]['variable_crop_monitor'], default = 0.0, mutable=False, doc='time required for crop monitoring (hr/ha)')

    model.p_fixed_crop_monitor = pe.Param(model.s_labperiods, initialize= params[season]['fixed_crop_monitor'], default = 0.0, mutable=False, doc='fixed time required for crop monitoring (hr/period)')


###################################
#functions for core model         #
###################################
#labour req by 
def mach_labour_anyone(model,p):
    '''
    Calculate the total labour required by anyone for fertilising, spraying, seeding, harvest, preparation,
    packing and monitoring.

    Used in global constraint (con_labour_anyone). See CorePyomo

    '''
    seed_labour = sum(sum(model.v_seeding_machdays[p, k, l] for k in model.s_landuses) for l in model.s_lmus)        \
    * model.p_daily_seed_hours *(1 + model.p_seeding_helper)
    harv_labour = sum(model.v_harv_hours[p,k] * (1 + model.p_harv_helper[k])  for k in model.s_harvcrops)
    prep_labour = model.p_prep_pack[p]
    fert_t_time = sum(sum(sum(model.p_phasefert[r,l,n]*model.v_phase_area[r,l]*(model.p_fert_app_hour_tonne[p,n]/1000)  for r in model.s_phases
                              if pe.value(model.p_phasefert[r,l,n]) != 0)for l in model.s_lmus)for n in model.s_fert_type )
    fert_ha_time = sum(sum(model.v_phase_area[r,l]*(model.p_fert_app_hour_ha[r,l,p]) for r in model.s_phases
                           if pe.value(model.p_fert_app_hour_ha[r,l,p]) != 0) for l in model.s_lmus)
    chem_time = sum(sum(model.v_phase_area[r,l]*(model.p_chem_app_lab[r,l,p]) for r in model.s_phases
                        if pe.value(model.p_chem_app_lab[r,l,p]) != 0) for l in model.s_lmus)
    return seed_labour + harv_labour + prep_labour + fert_t_time + fert_ha_time + chem_time


#labour req by
def mach_labour_perm(model,p):
    '''
    Calculate the total labour required by permanent staff for fertilising, spraying, seeding, harvest, preparation,
    packing and monitoring.

    Used in global constraint (con_labour_perm). See CorePyomo

    '''
    fixed_monitor_time = model.p_fixed_crop_monitor[p]
    variable_monitor_time = sum(model.p_variable_crop_monitor[r,p] * model.v_phase_area[r,l]  for r in model.s_phases for l in model.s_lmus
                                if pe.value(model.p_variable_crop_monitor[r,p]) != 0)
    return variable_monitor_time + fixed_monitor_time









