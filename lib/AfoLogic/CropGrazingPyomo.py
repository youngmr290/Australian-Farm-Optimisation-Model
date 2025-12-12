"""

author: young


"""

# python modules
import pyomo.environ as pe

# AFO modules
from . import PropertyInputs as pinp
from . import CropGrazing as cgz
from . import PyomoFunctions as fpy


def cropgraze_precalcs(params, r_vals, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    cgz.f1_cropgraze_params(params, r_vals, nv)


def f1_cropgrazepyomo_local(params,model):
    ''' Builds pyomo variables, parameters and constraints'''

    ########################
    # active index sets    #
    ########################
    ##Build sets that only have active items based on masks built in the precalcs.

    model.s_cropgraze_base_kp6p5zl = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl']
    )
    model.s_cropgraze_base_kp6p5z = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl'], drop_axes=(4,)
    )
    model.s_cropgraze_base_kp6z = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl'], drop_axes=(2,4)
    )
    model.s_cropgraze_base_kl = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl'], drop_axes=(1,2,3)
    )
    model.s_cropgraze_base_qskp6p5zl = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl'],
        prefix_sets=[model.s_active_qs]
    )
    model.s_cropgraze_base_qsfkp6p5zl = fpy.build_active_set(
        base_index=params['idx_cropgraze_base_kp6p5zl'],
        prefix_sets=[model.s_active_qs, model.s_feed_pools]
    )
    model.s_cropgraze_transfer_qskp6p5zl = fpy.build_active_set(
        base_index=params['idx_cropgraze_transfer_kp6p5zl'],
        prefix_sets=[model.s_active_qs]
    )
    model.s_cropgraze_base_kp6p5z8lz9 = fpy.build_active_set(
        base_index=params['idx_cropgraze_provide_kp6p5z8lz9']
    )


    ############
    # variable #
    ############
    model.v_tonnes_crop_consumed = pe.Var(model.s_cropgraze_base_qsfkp6p5zl, bounds=(0,None),
                                          doc='tonnes of crop consumed by livestock in a p6 that was sown in a p5 (p5 axis tracks when the crop being grazed was sown which impacts ME and Vol)')

    model.v_tonnes_crop_transfer = pe.Var(model.s_cropgraze_transfer_qskp6p5zl, bounds=(0,None),
                                          doc='tonnes of crop DM transferred to next feed period (p5 axis tracks when the crop being grazed was sown which impacts ME and Vol)')

    #########
    # param #
    #########
    model.p_crop_DM_provided = pe.Param(model.s_cropgraze_base_kp6p5z8lz9,
                                        initialize=params['crop_DM_provided_kp6p5z8lz9'], default=0, mutable=False,
                                        doc='Grazeable FOO provided by 1ha of rotation')

    model.p_crop_DM_required = pe.Param(model.s_cropgraze_base_kp6p5z,
                                        initialize=params['crop_DM_required_kp6p5z'], default=0, mutable=False,
                                        doc='FOO required for livestock to consume 1t of crop feed (this accounts for wastage)')

    model.p_propn_area_grazed_kl = pe.Param(model.s_cropgraze_base_kl,
                                     initialize=params['propn_area_grazed_kl'], default=0, mutable=False,
                                     doc='proportion of area for each crop and lmu, that can be crop grazed')

    model.p_cropgraze_biomass_penalty = pe.Param(model.s_cropgraze_base_kp6z,
                                               initialize=params['biomass_reduction_propn_kp6z'], default=0, mutable=False,
                                               doc='yield penalty as a proportion of crop consumed')

    model.p_crop_md = pe.Param(model.s_feed_pools * model.s_cropgraze_base_kp6p5zl,
                                     initialize=params['crop_md_fkp6p5zl'], default=0, mutable=False,
                                     doc='Energy provided from 1t of crop grazing for each time of sowing (p5 period)')

    model.p_crop_vol = pe.Param(model.s_feed_pools * model.s_cropgraze_base_kp6p5zl,
                                     initialize=params['crop_vol_kp6p5zl'], default=0, mutable=False,
                                     doc='Volume required to consume 1t of crop grazing for each time of sowing (p5 period)')

    model.co2e_cropgraze_kp6z = pe.Param(model.s_cropgraze_base_kp6z,
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
    p6_list = list(model.s_feed_periods)
    prev_p6 = {p6_list[i]: p6_list[i - 1] for i in range(len(p6_list))}
    def crop_DM_transfer(model,q,s,k,p6,p5,z9,l):
        p6s = prev_p6[p6]  #previous feedperiod
        return (
            # DM provided by seeding activities (rotation and machinery days)
            - sum(
                model.v_contractseeding_ha[q,s,z8,p5,k,l] * model.p_can_sow[p5,z9,k] * model.p_crop_DM_provided[k,p6,p5,z8,l,z9]
                 + model.v_seeding_machdays[q,s,z8,p5,k,l] * model.p_seeding_rate[k,l] * model.p_can_sow[p5,z9,k] * model.p_crop_DM_provided[k,p6,p5,z8,l,z9]
                 for z8 in model.s_season_types if (k, p6, p5, z8, l) in model.s_cropgraze_base_kp6p5zl
            )
            # DM consumed by livestock across all feed pools
            + sum(
                model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z9,l] * model.p_crop_DM_required[k,p6,p5,z9]
                for f in model.s_feed_pools
            )
            # DM carried in from previous feed period (if transfer exists)
            - sum(
                model.v_tonnes_crop_transfer[q,s,k,p6s,p5,z8,l] * 1000
                * model.p_parentz_provwithin_fp[p6s,z8,z9]
                for z8 in model.s_season_types
                if (q, s, k, p6s, p5, z8, l) in model.s_cropgraze_transfer_qskp6p5zl
            )
            # DM transferred out to next feed period
           + (
                model.v_tonnes_crop_transfer[q,s,k,p6,p5,z9,l] * 1000
                if (q, s, k, p6, p5, z9, l) in model.s_cropgraze_transfer_qskp6p5zl
                else 0
            )
           <=0
        )

    model.con_crop_DM_transfer = pe.Constraint(
        model.s_cropgraze_base_qskp6p5zl, rule=crop_DM_transfer,
        doc='transfer FOO from the grazing grazing 1ha activity to the consumption activity'
    )




###################################
#functions for core pyomo         #
###################################

def f_grazecrop_biomass_penalty(model,q,s,p7,k,l,z):
    '''
    Calculate the yield penalty from grazing crops (kg).

    Used in global constraint (con_grain_transfer). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_cropgraze_biomass_penalty[k,p6,z] * model.p_a_p6_p7[p7,p6,z] * 1000
               for f in model.s_feed_pools
               for (k_, p6, p5, z_, l_) in model.s_cropgraze_base_kp6p5zl
               if k_ == k and l_ == l and z_ == z)




##md
def f_grazecrop_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of green crop.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_crop_md[f,k,p6,p5,z,l]
               for (k, p6_, p5, z_, l) in model.s_cropgraze_base_kp6p5zl
               if p6_ == p6 and z_ == z)




##vol
def f_grazecrop_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of green crop.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.p_crop_vol[f,k,p6,p5,z,l]
               for (k, p6_, p5, z_, l) in model.s_cropgraze_base_kp6p5zl
               if p6_ == p6 and z_ == z)


def f_grazecrop_emissions(model,q,s,p6,z):
    '''
    Calculate the total emissions linked to consumption of green crop.

    Used in global constraint (con_emissions). See BoundPyomo
    '''
    return sum(model.v_tonnes_crop_consumed[q,s,f,k,p6,p5,z,l] * model.co2e_cropgraze_kp6z[k,p6,z]
               for f in model.s_feed_pools
               for (k, p6_, p5, z_, l) in model.s_cropgraze_base_kp6p5zl
               if p6_ == p6 and z_ == z)
