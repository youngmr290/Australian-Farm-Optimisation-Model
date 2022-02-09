# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:19:20 2019

module: create the model. this then gets used in all sheets that use pyomo

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete in case we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem


@author: young
"""


#python modules
from pyomo.environ import *
import pandas as pd
import numpy as np

#AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp
import Periods as per
import SeasonalFunctions as zfun


'''
pyomo sets
'''
##define sets - sets are redefined for each exp in case they change due to SA
def sets(model, nv):

    #######################
    #season               #
    #######################
    ##season types - set only has one season if steady state model is being used
    if pinp.general['steady_state']:
        z_keys = [pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]]
    else:
        z_keys = pinp.general['i_z_idx'][pinp.general['i_mask_z']] #mask season types by the ones included
    model.s_season_types = Set(initialize=z_keys, doc='season types')

    ##season periods
    model.s_season_periods = Set(initialize=per.f_season_periods(keys=True),doc='season nodes')

    ##season sequence set
    len_q = pinp.general['i_len_q']
    model.s_sequence_year = Set(initialize=np.array(['q%s' % i for i in range(len_q)]), doc='season sequences')

    ##season sequence set
    len_z = len(z_keys)
    len_s = np.power(len_z, len_q - 1)
    model.s_sequence = Set(initialize=np.array(['s%s' % i for i in range(len_s)]), doc='season sequences')

    #######################
    #price                #
    #######################
    model.s_c1 = Set(initialize=np.array(['c1_%s' % i for i in range(uinp.price_variation['len_c1'])]), doc='price scenarios')

    #######################
    #labour               #
    #######################
    ##labour periods
    model.s_labperiods = Set(initialize=per.f_p_date2_df().index, doc='labour periods')

    ##worker levels - levels for the different jobs
    model.s_worker_levels  = Set(initialize=sinp.general['worker_levels'], doc='worker levels for the different jobs')


    #######################
    #enterprises          #
    #######################

    model.s_enterprises = Set(initialize=sinp.general['i_enterprises_c0'], doc='enterprises')

    #######################
    #stubble              #
    #######################
    #stubble categories -  ordered so to allow transferring between categories
    model.s_stub_cat = Set(ordered=True, initialize=pinp.stubble['i_stub_cat_idx'], doc='stubble categories')

    #######################
    #cropping related     #
    #######################
    #grain pools ie firsts and seconds
    model.s_grain_pools = Set(initialize=sinp.general['grain_pools'], doc='grain pools')

    ##biomass uses
    model.s_biomass_uses = Set(initialize=pinp.stubble['i_idx_s2'], doc='uses of phase biomass')

    ##types of crops
    model.s_crops = Set(initialize=sinp.landuse['C'], doc='crop types')


    ##all crops and each pasture landuse eg t, tr
    model.s_landuses = Set(initialize=sinp.landuse['All'], doc='landuses')

    ##different fert options - used in labourcroppyomo
    model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')

    ###########
    #rotation #
    ###########

    ##lmus
    lmu_mask = pinp.general['i_lmu_area'] > 0
    model.s_lmus = Set(initialize=pinp.general['i_lmu_idx'][lmu_mask],doc='defined the soil type a given rotation is on')

    ##phases
    model.s_phases = Set(initialize=sinp.f_phases().index,doc='rotation phases set')

    ##rotation con1 set
    s_rotcon1 = pd.read_excel('Rotation.xlsx',sheet_name='rotation con1 set',header=None,index_col=0,engine='openpyxl')
    model.s_rotconstraints = Set(initialize=s_rotcon1.index,doc='rotation constraints histories')

    # ##phases disaggregated - used in rot yield transfer
    # def phases_dis():
    #     phase=sinp.stock['phases'].copy()
    #     return phase.set_index(list(range(sinp.general['phase_len']))).index
    # model.s_phases_dis = Set(dimen=sinp.general['phase_len'], ordered=True, initialize=phases_dis(), doc='rotation phases disaggregated')
    # model.s_phases_dis.pprint()

    # model.s_rotconstraints.pprint()

    ##con2 set
    # s_rotcon2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2 set', header= None, index_col = 0)
    # model.s_rotconstraints2 = Set(initialize=s_rotcon2.index, doc='rotation constraints histories 2')



    #######################
    #mvf                  #
    #######################
    model.s_mvf_q = Set(initialize=pinp.mvf['i_q_idx'], doc='mvf levels')

    #######################
    #sheep                #
    #######################
    '''other sheep sets are built in stockpyomo.py'''
    ##all groups
    model.s_infrastructure = Set(initialize=uinp.sheep['i_h1_idx'], doc='core sheep infrastructure')

    ##feed pool
    keys_nv = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    model.s_feed_pools = Set(initialize=keys_nv, doc='nutritive value pools')

    ##dams
    model.s_nut_dams = Set(initialize=np.array(['n%s'%i for i in range(sinp.structuralsa['i_n1_matrix_len'])]), doc='Nutrition levels in each feed period for dams')

    ##offs
    model.s_sale_offs = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t3_len'])], doc='Sales within the year for offs')
    model.s_nut_offs = Set(initialize=np.array(['n%s'%i for i in range(sinp.structuralsa['i_n3_matrix_len'])]), doc='Nutrition levels in each feed period for offs')

    ##prog
    model.s_sale_prog = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t2_len'])], doc='Sales and transfers options for yatf')
    model.s_lw_prog = Set(initialize=['w%03d'%i for i in range(sinp.structuralsa['i_progeny_w2_len'])], doc='Standard LW patterns prog')


    #######################
    #pasture             #
    #######################
    ##pasture types
    model.s_pastures = Set(initialize=sinp.general['pastures'][pinp.general['pas_inc']],doc='feed periods')

    ##feed periods
    model.s_feed_periods = Set(ordered=True, initialize=pinp.period['i_fp_idx'], doc='feed periods') #must be ordered so it can be sliced in pasture pyomo to allow feed to be transferred between periods.

    ##pasture groups
    model.s_dry_groups = Set(initialize=sinp.general['dry_groups'], doc='dry feed pools')
    model.s_grazing_int = Set(initialize=sinp.general['grazing_int'], doc='grazing intensity in the growth/grazing activities')
    model.s_foo_levels = Set(initialize=sinp.general['foo_levels'], doc='FOO level in the growth/grazing activities')




















