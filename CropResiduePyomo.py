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
    model.v_stub_con = pe.Var(model.s_feed_pools, model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat,bounds=(0.0,None),
                              doc='consumption of 1t of stubble')
    ##stubble transfer
    model.v_stub_transfer = pe.Var(model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat,bounds=(0.0,None),
                                   doc='transfer of 1t of stubble to following period')

    ####################
    #define parameters #
    ####################
    model.p_rot_stubble = pe.Param(model.s_phases, model.s_crops, model.s_lmus, model.s_phase_periods, model.s_season_types,
                                   initialize=params['rot_stubble'], default=0.0, doc='stubble produced per ha of each rotation')

    model.p_harv_prop = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, initialize=params['cons_prop'],
                                 default = 0.0, mutable=False, doc='proportion of the way through each fp harvest occurs (0 if harv doesnt occur in given period)')
    
    model.p_stub_md = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, initialize=params['md'],
                               default = 0.0, mutable=False, doc='md from 1t of each stubble categories for each crop')

    model.p_stub_vol = pe.Param(model.s_feed_pools, model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, initialize=params['vol'],
                                default = 0.0, mutable=False, doc='amount of intake volume required by 1t of each stubble category for each crop')
    
    model.p_a_req = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, initialize=params['cat_a_st_req'],
                             default = 0.0, mutable=False, doc='stubble required in each feed periods in order to consume 1t of cat A')
    
    model.p_bc_prov = pe.Param(model.s_crops, model.s_stub_cat, initialize=params['transfer_prov'], default = 0.0,
                               doc='stubble B provided from 1t of cat A and stubble C provided from 1t of cat B')
    
    model.p_bc_req = pe.Param(model.s_crops, model.s_stub_cat, initialize=params['transfer_req'], default = 0.0,
                              doc='stubble required from the row inorder to consume cat B or cat C')
    
    model.p_fp_transfer = pe.Param(model.s_feed_periods, model.s_season_types, model.s_crops, initialize=params['per_transfer'],
                                   default = 0.0, mutable=False, doc='stubble cat B or cat C transferred to the next feed period')
    

    ########################
    #call local constraint #
    ########################
    f_con_stubble_bcd(model)



###################
#local constraint #
###################
def f_con_stubble_bcd(model):
    ''' Links the consumption of a given category with the provision of another category or the transfer of
    stubble to the following period. Eg category A consumption provides category B. Category B can either be
    consumed (hence providing cat C) or transferred to the following period.
    '''
    ##stubble transter from category to category and period to period
    def stubble_transfer(model,p6,z9,k,s):
        if s == 'a':# or model.p_bc_req[k,s]==0: #this constraint is only for cat b and c
            return pe.Param.Skip
        else:
            ss = list(model.s_stub_cat)[list(model.s_stub_cat).index(s)-1] #previous stubble cat - used to transfer from current cat to the next, list is required because indexing of an ordered set starts at 1 which means index of 0 chucks error
            p6s = list(model.s_feed_periods)[list(model.s_feed_periods).index(p6)-1] #have to convert to a list first because indexing of an ordered set starts at 1
            return  - sum(model.v_stub_transfer[p6s,z8,k,s] * model.p_fp_transfer[p6s,z8,k]
                          * model.p_parentchildz_transfer_fp[p6s,z8,z9] for z8 in model.s_season_types)  \
                    + model.v_stub_transfer[p6,z9,k,s] * 1000 \
                    + sum(-model.v_stub_con[f,p6,z9,k,ss] * model.p_bc_prov[k,ss] + model.v_stub_con[f,p6,z9,k,s] * model.p_bc_req[k,s]
                          for f in model.s_feed_pools) <=0
    model.con_stubble_bcd = pe.Constraint(model.s_feed_periods, model.s_season_types, model.s_crops, model.s_stub_cat, rule = stubble_transfer, doc='links rotation stubble production with consumption of cat A')

###################
#constraint global#
###################
##stubble transter from category to category and period to period
def f_stubble_req_a(model,z,k,s):
    '''
    Calculate the total stubble required to consume the selected volume category A stubble in each period.

    Used in global constraint (con_stubble_a). See CorePyomo
    '''

    return sum(model.v_stub_con[f,p6,z,k,s] * model.p_a_req[p6,z,k,s] for f in model.s_feed_pools for p6 in model.s_feed_periods if pe.value(model.p_a_req[p6,z,k,s]) !=0)


##stubble md
def f_stubble_me(model,p6,f,z):
    '''
    Calculate the total energy provided to each nv pool from the selected amount of stubble.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_stub_con[f,p6,z,k,s] * model.p_stub_md[f,p6,z,k,s] for k in model.s_crops for s in model.s_stub_cat)
    
##stubble vol
def f_stubble_vol(model,p6,f,z):
    '''
    Calculate the total volume required by each nv pool to consume the selected level of stubble.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_stub_con[f,p6,z,k,s] * model.p_stub_vol[f,p6,z,k,s] for k in model.s_crops for s in model.s_stub_cat)