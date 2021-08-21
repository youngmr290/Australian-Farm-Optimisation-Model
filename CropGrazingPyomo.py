"""

author: young


"""

# python modules
import pyomo.environ as pe

# AFO modules
import PropertyInputs as pinp
import CropGrazing as cgz


def cropgraze_precalcs(params, r_vals, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        cgz.f1_cropgraze_params(params, r_vals, nv)


def f1_cropgrazepyomo_local(params,model):
    ''' Builds pyomo variables, parameters and constraints'''

    if pinp.cropgraze['i_cropgrazing_inc']:
        ############
        # variable #
        ############
        model.v_grazecrop_ha = pe.Var(model.s_crops, model.s_season_types, model.s_lmus, bounds=(0,None),
                                      doc='hectares of crop grazed')

        model.v_tonnes_crop_consumed = pe.Var(model.s_feed_pools, model.s_crops, model.s_feed_periods, model.s_season_types,
                                              bounds=(0,None), doc='tonnes of crop consumed by livestock')

        #########
        # param #
        #########
        model.p_cropgrazing_area = pe.Param(model.s_phases, model.s_crops, model.s_lmus,
                                         initialize=params['grazecrop_area_rkl'], default=0, mutable=False,
                                         doc='area of crop grazing provided by 1ha of rotation')

        model.p_crop_foo_provided = pe.Param(model.s_crops, model.s_feed_periods, model.s_season_types, model.s_lmus,
                                         initialize=params['crop_foo_provided_kp6zl'], default=0, mutable=False,
                                         doc='Grazeable FOO provided by 1ha of rotation')

        model.p_crop_foo_required = pe.Param(model.s_crops, initialize=params['crop_foo_required_k'], default=0, mutable=False,
                                         doc='FOO required for livestock to consume 1t of crop feed (this accounts for wastage)')

        model.p_cropgraze_yield_penalty = pe.Param(model.s_crops, model.s_season_types, model.s_lmus,
                                         initialize=params['yield_penalty_kzl'], default=0, mutable=False,
                                         doc='yield penalty (kg) from grazing 1ha of crop rotation')

        model.p_cropgraze_stubble_penalty = pe.Param(model.s_crops, model.s_season_types, model.s_lmus,
                                         initialize=params['stubble_penalty_kzl'], default=0, mutable=False,
                                         doc='stubble penalty (kg) from grazing 1ha of crop rotation')

        model.p_crop_md = pe.Param(model.s_feed_pools, model.s_crops, model.s_feed_periods, model.s_season_types, model.s_lmus,
                                         initialize=params['crop_md_fkp6zl'], default=0, mutable=False,
                                         doc='Energy provided from 1t of crop grazing')

        model.p_crop_vol = pe.Param(model.s_feed_pools, model.s_crops, model.s_feed_periods, model.s_season_types, model.s_lmus,
                                         initialize=params['crop_vol_kp6zl'], default=0, mutable=False,
                                         doc='Volume required to consume 1t of crop grazing')

        ###################################
        #call local constraints           #
        ###################################
        f_con_crop_foo_transfer(model)



def f_con_crop_foo_transfer(model):
    '''
    Transfer FOO from grazing 1ha of crop to the livestock crop consumption activity.
    '''
    def crop_foo_transfer(model,k,p6,z):
        return sum(- model.v_grazecrop_ha[k,z,l] * model.p_crop_foo_provided[k,p6,z,l] for l in model.s_lmus)    \
               + sum(model.v_tonnes_crop_consumed[f,k,p6,z] * model.p_crop_foo_required[k] for f in model.s_feed_pools) <=0
    model.con_crop_foo_transfer = pe.Constraint(model.s_crops, model.s_feed_periods, model.s_season_types, rule=crop_foo_transfer,
                                                doc='transfer FOO from the grazing grazing 1ha activity to the consumption activity')




###################################
#functions for core pyomo         #
###################################

def f_grazecrop_yield_penalty(model,g,k,z):
    '''
    Calculate the yield penalty from grazing crops.

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        return sum(model.v_grazecrop_ha[k,z,l] * model.p_cropgraze_yield_penalty[k,z,l]
                                 for l in model.s_lmus) * model.p_grainpool_proportion[k,g]
    else:
        return 0

def f_grazecrop_stubble_penalty(model,k,z):
    '''
    Calculate the stubble penalty from grazing crops.

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        return sum(model.v_grazecrop_ha[k,z,l] * model.p_cropgraze_stubble_penalty[k,z,l]
                                 for l in model.s_lmus)
    else:
        return 0



##stubble md
def f_grazecrop_me(model,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of stubble.

    Used in global constraint (con_me). See CorePyomo
    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        return sum(model.v_tonnes_crop_consumed[f,k,p6,z] * model.p_crop_md[f,k,p6,z,l] for k in model.s_crops for l in model.s_lmus)
    else:
        return 0



##stubble vol
def f_grazecrop_vol(model,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of stubble.

    Used in global constraint (con_vol). See CorePyomo
    '''
    if pinp.cropgraze['i_cropgrazing_inc']:
        return sum(model.v_tonnes_crop_consumed[f,k,p6,z] * model.p_crop_vol[f,k,p6,z,l] for k in model.s_crops for l in model.s_lmus)
    else:
        return 0
