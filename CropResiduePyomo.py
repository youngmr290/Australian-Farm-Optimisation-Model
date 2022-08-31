# -*- coding: utf-8 -*-
"""
author: young
"""

#python modules
import pyomo.environ as pe

#AFO modules
import CropResidue as stub
import PropertyInputs as pinp

def stub_precalcs(params, r_vals, nv):
    '''
    Call crop precalc functions.

    :param params: dictionary which stores all arrays used to populate pyomo parameters.
    :param report: dictionary which stores all report values.

    '''
    stub.crop_residue_all(params, r_vals, nv)
    
    
    
def f1_stubpyomo_local(params, model):
    ''' Builds pyomo variables, parameters and constraints'''
    ###################
    # variable         #
    ###################
    ##stubble consumption
    model.v_stub_con = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods, model.s_feed_pools,
                              model.s_crops, model.s_stub_cat, model.s_biomass_uses, bounds=(0.0,None),
                              doc='consumption of 1t of stubble')
    ##stubble transfer
    model.v_stub_transfer = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_feed_periods,
                                   model.s_crops, model.s_stub_cat, model.s_biomass_uses, bounds=(0.0,None),
                                   doc='transfer of 1t of stubble to following period - 1t of stubble at the start of the period that is not consumed but is decayed')

    # model.v_stub_harv = pe.Var(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_season_types, model.s_crops, bounds=(0.0,None),
    #                                doc='total stubble at harvest. Used to transfer to stubble constraint')

    # model.v_stub_debit = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_season_periods, model.s_crops, bounds=(0,None),
    #                             doc='tonnes of total stub in debt (will need to be provided from harvest)')

    # model.v_stub_credit = pe.Var(model.s_sequence_year, model.s_sequence, model.s_season_periods, model.s_crops, model.s_stub_cat, model.s_season_types, bounds=(0,None),
    #                             doc='tonnes of total stub in credit (can be used for feeding)')



    ####################
    #define parameters #
    ####################
    # model.p_rot_stubble = pe.Param(model.s_phases, model.s_crops, model.s_lmus, model.s_season_periods, model.s_season_types,
    #                                initialize=params['rot_stubble'], default=0.0, doc='stubble produced per ha of each rotation')

    model.p_harv_prop = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, initialize=params['cons_prop'],
                                 default = 0.0, mutable=False, doc='proportion of the way through each fp harvest occurs (0 if harv does not occur in given period)')
    
    model.p_stub_md = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, initialize=params['md'],
                               default = 0.0, mutable=False, doc='md from 1t of each stubble categories for each crop')

    model.p_stub_vol = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, initialize=params['vol'],
                                default = 0.0, mutable=False, doc='amount of intake volume required by 1t of each stubble category for each crop')
    
    model.p_a_prov = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, model.s_biomass_uses, initialize=params['cat_a_prov'],
                             default = 0.0, mutable=False, doc='cat A stubble provided at harvest from 1t of stubble')
    
    model.p_biomass2residue = pe.Param(model.s_crops, model.s_lmus, model.s_biomass_uses, initialize=params['biomass2residue_kls2'],
                             default = 0.0, mutable=False, doc='conversion of biomass to crop residue for each biomass use (harvesting as normal, baling for hay and grazing as fodder)')

    model.p_bc_prov = pe.Param(model.s_crops, model.s_stub_cat, model.s_biomass_uses, initialize=params['transfer_prov'], default = 0.0,
                               doc='stubble B provided from 1t of cat A and stubble C provided from 1t of cat B')
    
    model.p_bc_req = pe.Param(model.s_crops, model.s_stub_cat, model.s_biomass_uses, initialize=params['transfer_req'], default = 0.0,
                              doc='stubble required from the row inorder to consume cat B or cat C')
    
    model.p_stub_transfer_prov = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, initialize=params['stub_transfer_prov'],
                                   default = 0.0, mutable=False, doc='stubble available for consumption. Transferred in from last period or harvest.')
    
    model.p_stub_transfer_req = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, initialize=params['stub_transfer_req'],
                                   default = 0.0, mutable=False, doc='stubble required for transfer to the next period')


    ########################
    #call local constraint #
    ########################
    f_con_cropresidue_within(model)
    f_con_cropresidue_between(model)



###################
#local constraint #
###################
def f_con_cropresidue_within(model):
    ''' Links the consumption of a given category with the provision of another category or the transfer of
    stubble to the following period. E.g. category A consumption provides category B. Category B can either be
    consumed (hence providing category C) or transferred to the following period.
    '''
    ##stubble transfer from category to category and period to period
    ##s2 required because cat propn can vary across s2
    def cropresidue_transfer_within(model,q,s,p6,z9,k,sc,s2):
        if pe.value(model.p_mask_childz_within_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s]) and pe.value(model.p_stub_transfer_req[p6,z9,k]): #p_stub_transfer_req included to remove constraints when stubble doesn't exist
            sc_prev = list(model.s_stub_cat)[list(model.s_stub_cat).index(sc)-1] #previous stubble cat - used to transfer from current cat to the next, list is required because indexing of an ordered set starts at 1 which means index of 0 chucks error
            p6_prev = list(model.s_feed_periods)[list(model.s_feed_periods).index(p6)-1] #have to convert to a list first because indexing of an ordered set starts at 1
            return  - sum(model.v_stub_transfer[q,s,z8,p6_prev,k,sc,s2] * model.p_stub_transfer_prov[p6_prev,z8,k]
                          * model.p_parentz_provwithin_fp[p6_prev,z8,z9] for z8 in model.s_season_types)  \
                    - sum(model.v_use_biomass[q,s,p7,z9,k,l,s2] * model.p_a_p6_p7[p7,p6,z9] * model.p_biomass2residue[k,l,s2]
                          for p7 in model.s_season_periods for l in model.s_lmus) * model.p_a_prov[p6,z9,k,sc,s2] \
                    + model.v_stub_transfer[q,s,z9,p6,k,sc,s2] * model.p_stub_transfer_req[p6,z9,k] \
                    + sum(-model.v_stub_con[q,s,z9,p6,f,k,sc_prev,s2] * model.p_bc_prov[k,sc_prev,s2]
                          + model.v_stub_con[q,s,z9,p6,f,k,sc,s2] * model.p_bc_req[k,sc,s2]
                          for f in model.s_feed_pools) <=0
        else:
            return pe.Constraint.Skip
    model.con_cropresidue_within = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_season_types,
                                             model.s_crops, model.s_stub_cat, model.s_biomass_uses, rule=cropresidue_transfer_within, doc='stubble transfer between feed periods and stubble transfer between categories.')


def f_con_cropresidue_between(model):
    ''' Links the consumption of a given category with the provision of another category, or the transfer of
    stubble to the following period. E.g. category A consumption provides category B. Category B can either be
    consumed (hence providing category C) or transferred to the following period.
    '''
    ##stubble transfer from category to category and period to period
    ##s2 required because cat propn can vary across s2
    def cropresidue_transfer_between(model,q,s9,p6,z9,k,sc,s2):
        if pe.value(model.p_mask_childz_between_fp[p6,z9]) and pe.value(model.p_wyear_inc_qs[q,s9]) and pe.value(model.p_stub_transfer_req[p6,z9,k]): #p_stub_transfer_req included to remove constraints when stubble doesn't exist
            sc_prev = list(model.s_stub_cat)[list(model.s_stub_cat).index(sc)-1] #previous stubble cat - used to transfer from current cat to the next, list is required because indexing of an ordered set starts at 1 which means index of 0 chucks error
            p6_prev = list(model.s_feed_periods)[list(model.s_feed_periods).index(p6)-1] #have to convert to a list first because indexing of an ordered set starts at 1
            q_prev = list(model.s_sequence_year)[list(model.s_sequence_year).index(q) - 1]
            return  - sum(model.v_stub_transfer[q_prev,s8,z8,p6_prev,k,sc,s2] * model.p_stub_transfer_prov[p6_prev,z8,k]
                          * model.p_parentz_provbetween_fp[p6_prev,z8,z9] * model.p_sequence_prov_qs8zs9[q_prev,s8,z8,s9]
                          + model.v_stub_transfer[q_prev,s8,z8,p6_prev,k,sc,s2] * model.p_stub_transfer_prov[p6_prev,z8,k]
                          * model.p_parentz_provbetween_fp[p6_prev,z8,z9] * model.p_endstart_prov_qsz[q_prev,s8,z8]
                          for z8 in model.s_season_types for s8 in model.s_sequence if pe.value(model.p_wyear_inc_qs[q_prev,s8])!=0)  \
                    - sum(model.v_use_biomass[q,s9,p7,z9,k,l,s2] * 1000 * model.p_a_p6_p7[p7,p6,z9] * model.p_biomass2residue[k,l,s2]
                          for p7 in model.s_season_periods for l in model.s_lmus) * model.p_a_prov[p6,z9,k,sc,s2] \
                    + model.v_stub_transfer[q,s9,z9,p6,k,sc,s2] * model.p_stub_transfer_req[p6,z9,k] \
                    + sum(-model.v_stub_con[q,s9,z9,p6,f,k,sc_prev,s2] * model.p_bc_prov[k,sc_prev,s2]
                          + model.v_stub_con[q,s9,z9,p6,f,k,sc,s2] * model.p_bc_req[k,sc,s2]
                          for f in model.s_feed_pools) <=0
        else:
            return pe.Constraint.Skip
    model.con_cropresidue_between = pe.Constraint(model.s_sequence_year, model.s_sequence, model.s_feed_periods, model.s_season_types,
                                              model.s_crops, model.s_stub_cat, model.s_biomass_uses, rule=cropresidue_transfer_between, doc='stubble transfer between feed periods and stubble transfer between categories.')


###################
#constraint global#
###################
##stubble transfer from category to category and period to period
# def f_cropresidue_req_a(model,q,s,p7,z,k,sc):
#     '''
#     Calculate the total stubble required to consume the selected volume category A stubble in each period.
#
#     Used in global constraint (con_cropresidue_a). See CorePyomo
#     '''
#
#     return sum(model.v_stub_transfer[q,s,p6,z,k,sc] * model.p_a_req[p6,z,k,sc] * model.p_a_p6_p7[p7,p6,z]
#                for p6 in model.s_feed_periods if pe.value(model.p_a_req[p6,z,k,sc]) !=0)
#     # return sum(model.v_stub_con[q,s,f,p6,z,k,sc] * model.p_a_req[p6,z,k,sc] * model.p_a_p6_p7[p7,p6,z]
#     #            for f in model.s_feed_pools for p6 in model.s_feed_periods if pe.value(model.p_a_req[p6,z,k,sc]) !=0)


##stubble md
def f_cropresidue_me(model,q,s,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of stubble.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_stub_con[q,s,z,p6,f,k,sc,s2] * model.p_stub_md[f,p6,z,k,sc]
               for k in model.s_crops for sc in model.s_stub_cat for s2 in model.s_biomass_uses)
    
##stubble vol
def f_cropresidue_vol(model,q,s,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of stubble.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_stub_con[q,s,z,p6,f,k,sc,s2] * model.p_stub_vol[f,p6,z,k,sc]
               for k in model.s_crops for sc in model.s_stub_cat for s2 in model.s_biomass_uses)