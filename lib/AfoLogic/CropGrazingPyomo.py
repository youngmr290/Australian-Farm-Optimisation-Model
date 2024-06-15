"""

author: young


"""

# python modules
import pyomo.environ as pe

# AFO modules
from . import PropertyInputs as pinp
from . import CropGrazing as cgz


def cropgraze_precalcs(params, r_vals, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    cgz.f1_cropgraze_params(params, r_vals, nv)


def f1_cropgrazepyomo_local(params,model):
    ''' Builds pyomo variables, parameters and constraints'''

    ############
    # variable #
    ############
    model.v_tonnes_crop_consumed = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_pools, model.s_crops, model.s_feed_periods,
                                          model.s_labperiods, model.s_season_types,  model.s_lmus,bounds=(0,None),
                                          doc='tonnes of crop consumed by livestock in a p6 that was sown in a p5 (p5 axis tracks when the crop being grazed was sown)')

    model.v_tonnes_crop_transfer = pe.Var(model.s_sequence_year, model.s_sequence, model.s_crops, model.s_lmus, model.s_feed_periods, model.s_labperiods, model.s_season_types,
                                          bounds=(0,None), doc='tonnes of crop DM transferred to next feed period (p5 axis tracks when the crop being grazed was sown)')

    #########
    # param #
    #########
    # model.p_cropgrazing_area = pe.Param(model.s_phases, model.s_crops, model.s_lmus,
    #                                  initialize=params['grazecrop_area_rkl'], default=0, mutable=False,
    #                                  doc='area of crop grazing provided by 1ha of rotation')

    model.p_crop_DM_provided = pe.Param(model.s_crops, model.s_feed_periods, model.s_labperiods, model.s_season_types, model.s_lmus,
                                     model.s_season_types, initialize=params['crop_DM_provided_kp6p5z8lz9'], default=0, mutable=False,
                                     doc='Grazeable FOO provided by 1ha of rotation')

    # model.p_crop_DM_reduction = pe.Param(model.s_crops, model.s_feed_periods, model.s_labperiods, model.s_season_types, model.s_lmus,
    #                                  initialize=params['DM_reduction_kp6p5zl'], default=0, mutable=False,
    #                                  doc='Reduction in DM due to sowing timing (late sowing means less growth)')

    model.p_crop_DM_required = pe.Param(model.s_crops, model.s_feed_periods, model.s_labperiods, model.s_season_types,
                                        initialize=params['crop_DM_required_kp6p5z'], default=0, mutable=False,
                                        doc='FOO required for livestock to consume 1t of crop feed (this accounts for wastage)')

    model.p_transfer_exists = pe.Param(model.s_feed_periods, model.s_labperiods, model.s_season_types,
                                     initialize=params['transfer_exists_p6p5z'], default=0, mutable=False,
                                     doc='transfer exists into current feed period')

    model.p_cropgrazing_can_occur_kl = pe.Param(model.s_crops, model.s_lmus,
                                     initialize=params['propn_area_grazed_kl'], default=0, mutable=False,
                                     doc='max proportion of area for each crop and lmu, that can be grazed')

    model.p_cropgraze_biomass_penalty = pe.Param(model.s_crops, model.s_feed_periods, model.s_season_types,
                                               initialize=params['biomass_reduction_propn_kp6z'], default=0, mutable=False,
                                               doc='yield penalty as a proportion of crop consumed')

    # model.p_cropgraze_stubble_penalty = pe.Param(model.s_crops, model.s_feed_periods, model.s_season_types,
    #                                              initialize=params['stubble_reduction_propn_kp6z'], default=0, mutable=False,
    #                                              doc='stubble penalty as a proportion of crop consumed')

    model.p_crop_md = pe.Param(model.s_feed_pools, model.s_crops, model.s_feed_periods, model.s_labperiods, model.s_season_types, model.s_lmus,
                                     initialize=params['crop_md_fkp6p5zl'], default=0, mutable=False,
                                     doc='Energy provided from 1t of crop grazing for each time of sowing (p5 period)')

    model.p_crop_vol = pe.Param(model.s_feed_pools, model.s_crops, model.s_feed_periods, model.s_labperiods, model.s_season_types, model.s_lmus,
                                     initialize=params['crop_vol_kp6p5zl'], default=0, mutable=False,
                                     doc='Volume required to consume 1t of crop grazing for each time of sowing (p5 period)')

    model.co2e_cropgraze_kp6z = pe.Param(model.s_crops, model.s_feed_periods, model.s_season_types,
                                     initialize=params['co2e_cropgraze_kp6z'], default=0, mutable=False,
                                     doc='kgs of co2e produced per tonne of crop grazing')

    ###################################
    #call local constraints           #
    ###################################
    f_con_crop_DM_transfer(model)



def f_con_crop_DM_transfer(model):
    '''
    Links seeding with the ability to graze crops.

    Each period following seeding provides feed (if it is within the crop grazing window) which can be consumed
    in the given period. DM that is not consumed is transferred into the following feed period (without
    changing the growth rate of the crop).

    Note: there is no corresponding 'between' constraint because there is no carry over across the break.
    '''
    def crop_DM_transfer(model,q,s,k,l,p6,p5,z9):
        p6s = list(model.s_feed_periods)[list(model.s_feed_periods).index(p6) - 1]  #previous feedperiod - have to convert to a list first because indexing of an ordered set starts at 1
        if pe.value(model.p_wyear_inc_qs[q, s]) and model.p_crop_DM_required[k,p6,p5,z9]!=0:
            return - sum(model.v_contractseeding_ha[q,s,z8,p5,k,l] * model.p_can_sow[p5,z9,k] * model.p_crop_DM_provided[k,p6,p5,z8,l,z9]
                         + model.v_seeding_machdays[q,s,z8,p5,k,l] * model.p_seeding_rate[k,l] * model.p_can_sow[p5,z9,k] * model.p_crop_DM_provided[k,p6,p5,z8,l,z9]
                         for z8 in model.s_season_types) \
                   + sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z9,l] * model.p_crop_DM_required[k,p6,p5,z9] for f in model.s_feed_pools) \
                   - sum(model.v_tonnes_crop_transfer[q,s,k,l,p6s,p5,z8] * 1000 * model.p_transfer_exists[p6s,p5,z8]
                         * model.p_parentz_provwithin_fp[p6s,z8,z9] for z8 in model.s_season_types)        \
                   + model.v_tonnes_crop_transfer[q,s,k,l,p6,p5,z9]*1000 \
                   <=0
        else:
            return pe.Constraint.Skip

        # return sum(- model.v_grazecrop_ha[p7,k,z9,l] * model.p_crop_DM_provided[p7,k,p6,z9,l]
        #            for l in model.s_lmus for p7 in model.s_season_periods)    \
        #      + sum(model.v_tonnes_crop_consumed[f,k,p6,z9] * model.p_crop_DM_required[k] for f in model.s_feed_pools) \
        #      - sum(model.v_tonnes_crop_transfer[k,p6s,z8]*1000*model.p_transfer_exists[p6,z8] #meant to be p6 in transfer_exists because that states if crop can be grazing in current p6 (if not then don't transfer last periods dm)
        #            * model.p_parentz_provwithin_fp[p6s,z8,z9] for z8 in model.s_season_types)        \
        #      + model.v_tonnes_crop_transfer[k,p6,z9]*1000 \
        #      + sum(model.p_crop_DM_reduction[k,p6,p5,z9,l] * model.v_contractseeding_ha[z9,p5,k,l]
        #            for p5 in model.s_labperiods for l in model.s_lmus) \
        #      + sum(model.p_crop_DM_reduction[k,p6,p5,z9,l] * model.p_seeding_rate[k,l] * model.v_seeding_machdays[z9,p5,k,l]
        #            for p5 in model.s_labperiods for l in model.s_lmus) <=0

    model.con_crop_DM_transfer = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_crops, model.s_lmus, model.s_feed_periods, model.s_labperiods, model.s_season_types, rule=crop_DM_transfer,
                                                doc='transfer FOO from the grazing grazing 1ha activity to the consumption activity')




###################################
#functions for core pyomo         #
###################################

def f_grazecrop_biomass_penalty(model,q,s,p7,k,l,z):
    '''
    Calculate the yield penalty from grazing crops (kg).

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_cropgraze_biomass_penalty[k,p6,z] * model.p_a_p6_p7[p7,p6,z] * 1000
               for f in model.s_feed_pools for p6 in model.s_feed_periods for p5 in model.s_labperiods)

# def f_grazecrop_stubble_penalty(model,q,s,p7,k,z):
#     '''
#     Calculate the stubble penalty from grazing crops (kg).
#
#     Used in global constraint (con_grain_transfer). See CorePyomo
#     '''
#     if pinp.cropgraze['i_cropgrazing_inc']:
#         return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z] * model.p_cropgraze_stubble_penalty[k,p6,z] * model.p_a_p6_p7[p7,p6,z] * 1000
#                    for f in model.s_feed_pools for p6 in model.s_feed_periods for p5 in model.s_labperiods)
#     else:
#         return 0



##md
def f_grazecrop_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of green crop.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_crop_md[f,k,p6,p5,z,l]
               for k in model.s_crops for l in model.s_lmus for p5 in model.s_labperiods)



##vol
def f_grazecrop_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of green crop.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_crop_vol[f,k,p6,p5,z,l]
               for k in model.s_crops for l in model.s_lmus for p5 in model.s_labperiods)


def f_grazecrop_emissions(model,q,s,p6,z):
    '''
    Calculate the total emissions linked to consumption of green crop.

    Used in global constraint (con_emissions). See BoundPyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.co2e_cropgraze_kp6z[k,p6,z]
               for f in model.s_feed_pools for k in model.s_crops for l in model.s_lmus for p5 in model.s_labperiods)
