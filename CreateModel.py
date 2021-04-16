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
# import Crop as crp
# import StockFunctions as sfun


'''
# Creation of a Concrete Model
'''
model = ConcreteModel()
model.report_timing=True #haven't actually been able to get this to do any thing yet?????

'''
pyomo sets
'''
##define sets - sets are redefined for each exp in case they change due to SA
def sets() :
    ##season types - set only has one season if steady state model is being used
    try:
        model.del_component(model.s_season_types)
    except AttributeError:
        pass
    if pinp.general['steady_state']:
        model.s_season_types = Set(initialize=[pinp.general['i_z_idx'][pinp.general['i_mask_z']][0]], doc='season types')
    else:
        model.s_season_types = Set(initialize=pinp.general['i_z_idx'][pinp.general['i_mask_z']], doc='season types') #mask season types by the ones included

    #labour periods
    try:
        model.del_component(model.s_labperiods)
    except AttributeError:
        pass
    model.s_labperiods = Set(initialize=per.p_date2_df().index, doc='labour periods')

    ##pasture types
    try:
        model.del_component(model.s_pastures)
    except AttributeError:
        pass
    model.s_pastures = Set(initialize=sinp.general['pastures'][pinp.general['pas_inc']],doc='feed periods')

    ##feed pool
    confinement_inc = np.maximum(np.max(pinp.sheep['i_nut_spread_n1'][0:sinp.stock['i_n1_len']]),
                                 np.max(pinp.sheep['i_nut_spread_n3'][
                                        0:sinp.stock[
                                            'i_n3_len']])) > 3  # if fs>3 then need to include confinment feeding
    ev_is_not_confinement_v = sinp.general['ev_is_not_confinement']
    ev_mask_v = np.logical_or(ev_is_not_confinement_v,confinement_inc)
    try:
        model.del_component(model.s_feed_pools)
    except AttributeError:
        pass
    model.s_feed_pools = Set(initialize=sinp.general['sheep_pools'][ev_mask_v],doc='nutritive value pools')
    print('tes')


#######################
#labour               #
#######################
##worker levels - levels for the different jobs
model.s_worker_levels  = Set(initialize=sinp.general['worker_levels'], doc='worker levels for the different jobs')



#######################
#cash                 #
#######################

#cashflow periods
model.s_cashflow_periods = Set(initialize=sinp.general['cashflow_periods'], doc='cashflow periods')

#######################
#stubble              #
#######################

#stubble categories -  ordered so to allow transferring between categories
model.s_stub_cat = Set(ordered=True, initialize=pinp.stubble['stub_cat_idx'], doc='stubble categories')

#######################
#cropping related     #
#######################
#grain pools ie firsts and seconds
model.s_grain_pools = Set(initialize=sinp.general['grain_pools'], doc='grain pools')

#landuses that are harvested - used in harv constraints and variables
model.s_harvcrops = Set(initialize=uinp.mach_general['contract_harvest_speed'].index, doc='landuses that are harvest')

##landuses that produce hay - used in hay constraints
model.s_haycrops = Set(ordered=False, initialize=sinp.landuse['Hay'], doc='landuses that make hay')

##types of crops
model.s_crops = Set(ordered=False, initialize=sinp.landuse['C'], doc='crop types')


##all crops and each pasture landuse eg t, tr
model.s_landuses = Set(ordered=False, initialize=sinp.landuse['All'], doc='landuses')

#soils
model.s_lmus = Set(initialize=pinp.general['lmu_area'].index, doc='defined the soil type a given rotation is on')
# model.s_lmus.pprint()

##different fert options - used in labourcroppyomo
model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')

###########
#rotation #
###########
##phases
model.s_phases = Set(initialize=sinp.phases['phases'].index, doc='rotation phases set')
# model.s_phases.pprint()

# ##phases disaggregated - used in rot yield transfer
# def phases_dis():
#     phase=sinp.stock['phases'].copy()
#     return phase.set_index(list(range(sinp.general['phase_len']))).index
# model.s_phases_dis = Set(dimen=sinp.general['phase_len'], ordered=True, initialize=phases_dis(), doc='rotation phases disaggregated')
# model.s_phases_dis.pprint()

##con1 set
s_rotcon1 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con1 set', header= None, index_col = 0, engine='openpyxl')
model.s_rotconstraints = Set(initialize=s_rotcon1.index, doc='rotation constraints histories')
# model.s_rotconstraints.pprint()

##con2 set
# s_rotcon2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2 set', header= None, index_col = 0)
# model.s_rotconstraints2 = Set(initialize=s_rotcon2.index, doc='rotation constraints histories 2')



#######################
#sheep                 #
#######################
##all groups
model.s_infrastructure = Set(initialize=uinp.sheep['i_h1_idx'], doc='core sheep infrastructure')
# model.s_co_conception = Set(initialize=, doc='carryover characteristics - conception')
# model.s_co_bw = Set(initialize=, doc='carryover characteristics - Birth weight')
# model.s_co_ww = Set(initialize=, doc='carryover characteristics - Weaning weight')
# model.s_co_cfw = Set(initialize=, doc='carryover characteristics - Clean fleece weight')
# model.s_co_fd = Set(initialize=, doc='carryover characteristics - Fibre diameter')
# model.s_co_min_fd = Set(initialize=, doc='carryover characteristics - Minimum fibre diameter')
# model.s_co_fl = Set(initialize=, doc='carryover characteristics - Fibre length')

##dams & offs



##sire ^don't have any sets at the moment
# model.s_sale_sire = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t0_len'])], doc='Sales within the year for sires')
# model.s_dvp_sire = Set(ordered=True, initialize=, doc='Decision variable periods for sires')
# model.s_nut_sire = Set(initialize=sinp.stock['i_n_idx_sire'], doc='Nutrition levels in each feed period for sires')
# model.s_lw_sire = Set(initialize=sinp.stock['i_w_idx_sire'], doc='Standard LW patterns sires')
# model.s_sire_periods = Set(initialize=, doc='sire capacity periods')

##dams
model.s_nut_dams = Set(initialize=np.array(['n%s'%i for i in range(sinp.stock['i_n1_matrix_len'])]), doc='Nutrition levels in each feed period for dams')
##offs
model.s_sale_offs = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t3_len'])], doc='Sales within the year for offs')
model.s_nut_offs = Set(initialize=np.array(['n%s'%i for i in range(sinp.stock['i_n3_matrix_len'])]), doc='Nutrition levels in each feed period for offs')
##prog
model.s_sale_prog = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t2_len'])], doc='Sales and transfers options for yatf')
model.s_lw_prog = Set(initialize=['w%03d'%i for i in range(sinp.stock['i_progeny_w2_len'])], doc='Standard LW patterns prog')




#######################
#pasture             #
#######################
##feed periods
model.s_feed_periods = Set(ordered=True, initialize=pinp.period['i_fp_idx'], doc='feed periods') #must be ordered so it can be sliced in pasture pyomo to allow feed to be transferred betweeen periods.
##pasture groups
model.s_dry_groups = Set(initialize=sinp.general['dry_groups'], doc='dry feed pools')
model.s_grazing_int = Set(initialize=sinp.general['grazing_int'], doc='grazing intensity in the growth/grazing activities')
model.s_foo_levels = Set(initialize=sinp.general['foo_levels'], doc='FOO level in the growth/grazing activities')




















