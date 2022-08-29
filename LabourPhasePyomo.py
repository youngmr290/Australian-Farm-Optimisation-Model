"""
author: young
"""
#python modules
import pyomo.environ as pe

#AFO modules
import LabourPhase as lphs


def crplab_precalcs(params, r_vals):
    '''
    Call crop labour precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    lphs.f1_labcrop_params(params, r_vals)

def f1_labcrppyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''

    #########
    #param  #
    #########

    model.p_harv_helper = pe.Param(model.s_crops, initialize=params['harvest_helper'], default = 0.0, doc='harvest helper time per crop')
    
    model.p_daily_seed_hours = pe.Param(initialize=params['daily_seed_hours'], default = 0.0, doc='machine hours per day of seeding ie labour required per mach day')
    
    model.p_seeding_helper = pe.Param( initialize=params['seeding_helper'], default = 0.0, doc='proportion of time helper is needed for seeding')

    model.p_prep_pack = pe.Param(model.s_labperiods, model.s_season_types, initialize=params['prep_labour'], default = 0.0, mutable=False, doc='labour for preparation and packing up for seeding and harv')

    model.p_fert_app_hour_tonne = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                           model.s_season_periods, initialize= params['fert_app_time_t'], default = 0.0,
                                           mutable=False, doc='time required for fert application per tonne of each fert (filling up and driving to paddock cost)')
 
    model.p_increment_fert_app_hour_tonne = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                           model.s_season_periods, initialize= params['increment_fert_app_time_t'], default = 0.0,
                                           mutable=False, doc='time required for fert application per tonne of each fert (filling up and driving to paddock cost)')

    model.p_fert_app_hour_ha = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                        model.s_season_periods, initialize= params['fert_app_time_ha'], default = 0.0,
                                        mutable=False, doc='time required for fert application per ha of each fert (driving around paddock cost)')
    
    model.p_increment_fert_app_hour_ha = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                        model.s_season_periods, initialize= params['increment_fert_app_time_ha'], default = 0.0,
                                        mutable=False, doc='time required for fert application per ha of each fert (driving around paddock cost)')

    model.p_chem_app_lab = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                    model.s_season_periods, initialize= params['chem_app_time_ha'], default = 0.0,
                                    mutable=False, doc='time required for chem application per ha (hr/ha)')

    model.p_increment_chem_app_lab = pe.Param(model.s_phases, model.s_season_types, model.s_lmus, model.s_labperiods,
                                    model.s_season_periods, initialize= params['increment_chem_app_time_ha'], default = 0.0,
                                    mutable=False, doc='time required for chem application per ha (hr/ha)')

    model.p_variable_crop_monitor = pe.Param(model.s_season_periods, model.s_labperiods, model.s_season_types, model.s_phases,
                                             initialize= params['variable_crop_monitor'], default = 0.0, mutable=False, doc='time required for crop monitoring (hr/ha)')

    model.p_increment_variable_crop_monitor = pe.Param(model.s_season_periods, model.s_labperiods, model.s_season_types, model.s_phases,
                                             initialize= params['increment_variable_crop_monitor'], default = 0.0, mutable=False, doc='time required for crop monitoring (hr/ha)')

    model.p_fixed_crop_monitor = pe.Param(model.s_labperiods, model.s_season_types, initialize= params['fixed_crop_monitor'], default = 0.0, mutable=False, doc='fixed time required for crop monitoring (hr/period)')


###################################
#functions for core model         #
###################################
def f_mach_labour_anyone(model,q,s,p,z):
    '''
    Calculate the total labour required by anyone for fertilising, spraying, seeding, harvest, preparation,
    packing and monitoring.

    Used in global constraint (con_labour_anyone). See CorePyomo

    '''
    seed_labour = sum(model.v_seeding_machdays[q,s,z,p,k,l] for k in model.s_landuses for l in model.s_lmus)        \
    * model.p_daily_seed_hours *(1 + model.p_seeding_helper)
    harv_labour = sum(model.v_harv_hours[q,s,z,p,k] * (1 + model.p_harv_helper[k]) for k in model.s_crops)
    prep_labour = model.p_prep_pack[p,z]
    fert_t_time = sum(model.v_phase_area[q,s,p7,z,r,l]*model.p_fert_app_hour_tonne[r,z,l,p,p7]
                      + model.v_phase_change_increase[q,s,p7,z,r,l]*model.p_increment_fert_app_hour_tonne[r,z,l,p,p7]
                      for r in model.s_phases for l in model.s_lmus for p7 in model.s_season_periods
                      if pe.value(model.p_fert_app_hour_tonne[r,z,l,p,p7]) != 0 or pe.value(model.p_increment_fert_app_hour_tonne[r,z,l,p,p7]) != 0)
    fert_ha_time = sum(model.v_phase_area[q,s,p7,z,r,l]*model.p_fert_app_hour_ha[r,z,l,p,p7]
                       + model.v_phase_change_increase[q,s,p7,z,r,l]*model.p_increment_fert_app_hour_ha[r,z,l,p,p7]
                       for r in model.s_phases for l in model.s_lmus for p7 in model.s_season_periods
                       if pe.value(model.p_fert_app_hour_ha[r,z,l,p,p7]) != 0 or pe.value(model.p_increment_fert_app_hour_ha[r,z,l,p,p7]) != 0)
    chem_time = sum(model.v_phase_area[q,s,p7,z,r,l]*model.p_chem_app_lab[r,z,l,p,p7]
                    + model.v_phase_change_increase[q,s,p7,z,r,l]*model.p_increment_chem_app_lab[r,z,l,p,p7]
                    for r in model.s_phases for l in model.s_lmus for p7 in model.s_season_periods
                    if pe.value(model.p_chem_app_lab[r,z,l,p,p7]) != 0 or pe.value(model.p_increment_chem_app_lab[r,z,l,p,p7]) != 0)
    return seed_labour + harv_labour + prep_labour + fert_t_time + fert_ha_time + chem_time


def f_mach_labour_perm(model,q,s,p5,z):
    '''
    Calculate the total labour required by permanent staff for fertilising, spraying, seeding, harvest, preparation,
    packing and monitoring.

    Used in global constraint (con_labour_perm). See CorePyomo

    '''
    fixed_monitor_time = model.p_fixed_crop_monitor[p5,z]
    variable_monitor_time = sum(model.p_variable_crop_monitor[p7,p5,z,r] * model.v_phase_area[q,s,p7,z,r,l]
                                + model.p_increment_variable_crop_monitor[p7,p5,z,r] * model.v_phase_change_increase[q,s,p7,z,r,l]
                                for r in model.s_phases for l in model.s_lmus for p7 in model.s_season_periods
                                if pe.value(model.p_variable_crop_monitor[p7,p5,z,r]) != 0 or pe.value(model.p_increment_variable_crop_monitor[p7,p5,z,r]) != 0)
    return variable_monitor_time + fixed_monitor_time









