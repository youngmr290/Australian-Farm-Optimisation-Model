# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:19:20 2019

module: create the model. this then gets used in all sheets that use pyomo

Version Control:
Version     Date        Person  Change
1.1         22/02/202   MRY      commented out con2 as it is not needed - don't delete incase we are wrong and it is required.

Known problems:
Fixed   Date    ID by   Problem


@author: young
"""


#python modules
from pyomo.environ import *
import pandas as pd
import numpy as np

#MUDAS modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import Periods as per
import Crop as crp
import StockFunctions as sfun


'''
# Creation of a Concrete Model
'''
model = ConcreteModel()
model.report_timing=True #haven't actually been able to get this to do any thing yet?????

'''
pyomo sets
'''
##define sets that may change for different iterations of exp
def sets() :
    #######################
    #seasons              #
    #######################
    ##season types - set only has one season if steady state model is being used
    if pinp.general['steady_state']:
        model.s_season_types = Set(initialize=['season 1'], doc='season types')
    else:    
        model.s_season_types = Set(initialize=pinp.general['season_info'].index[pinp.general['season_info']['included']], doc='season types') #mask season types by the ones included




    
#######################
#labour               #
#######################

#labour periods
model.s_labperiods = Set(initialize=per.p_date2_df().index.astype(str)) #needs to be sring for livestock module


#######################
#cash                 #
#######################

#cashflow periods
model.s_cashflow_periods = Set(initialize=uinp.structure['cashflow_periods'], doc='cashflow periods')

#######################
#stubble              #
#######################

#stubble categories -  ordered so to allow transfering between categories
model.s_stub_cat = Set(ordered=True, initialize=pinp.stubble['stub_cat_qual'].columns, doc='stubble categories') 

#######################
#cropping related     #
#######################
#grain pools ie firsts and seconds
model.s_grain_pools = Set(initialize=uinp.structure['grain_pools'], doc='grain pools')

#landuses that are harvested - used in harv constraints and variables
model.s_harvcrops = Set(initialize=uinp.mach_general['contract_harvest_speed'].index, doc='landuses that are harvest')

##landuses that produce hay - used in hay constraints 
model.s_haycrops = Set(ordered=False, initialize=uinp.structure['Hay'], doc='landuses that make hay')

##types of crops
model.s_crops = Set(ordered=False, initialize=uinp.structure['C'], doc='crop types')

##all crops and the pasture types ie annual, tedera, lucerne (not a, a3, a4 etc)
model.s_landuses = Set(ordered=False, initialize=uinp.structure['All'], doc='landuses')

#soils
model.s_lmus = Set(initialize=pinp.general['lmu_area'].index, doc='defined the soil type a given rotation is on')
# model.s_lmus.pprint()

##different fert options - used in labourcroppyomo
model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')

###########
#rotation #
###########
##phases
model.s_phases = Set(initialize=uinp.structure['phases'].index, doc='rotation phases set') 
# model.s_phases.pprint()

# ##phases disagregated - used in rot yield transfer
# def phases_dis():
#     phase=uinp.structure['phases'].copy()
#     return phase.set_index(list(range(uinp.structure['phase_len']))).index
# model.s_phases_dis = Set(dimen=uinp.structure['phase_len'], ordered=True, initialize=phases_dis(), doc='rotation phases disagregated') 
# model.s_phases_dis.pprint()

##con1 set
s_rotcon1 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con1 set', header= None, index_col = 0)
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
model.s_sheep_pools = Set(initialize=uinp.structure['sheep_pools'], doc='nutritive value pools')
# model.s_co_conception = Set(initialize=, doc='carryover characteristics - conception')
# model.s_co_bw = Set(initialize=, doc='carryover characteristics - Birth weight')
# model.s_co_ww = Set(initialize=, doc='carryover characteristics - Weaning weight')
# model.s_co_cfw = Set(initialize=, doc='carryover characteristics - Clean fleece weight')
# model.s_co_fd = Set(initialize=, doc='carryover characteristics - Fibre diameter')
# model.s_co_min_fd = Set(initialize=, doc='carryover characteristics - Minimum fibre diameter')
# model.s_co_fl = Set(initialize=, doc='carryover characteristics - Fibre length')

##dams & offs
 

   
##sire ^dont have any sets at the moment
# model.s_sale_sire = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t0_len'])], doc='Sales within the year for sires')
# model.s_dvp_sire = Set(ordered=True, initialize=, doc='Decision variable periods for sires')
# model.s_nut_sire = Set(initialize=uinp.structure['i_n_idx_sire'], doc='Nutrition levels in each feed period for sires')
# model.s_lw_sire = Set(initialize=uinp.structure['i_w_idx_sire'], doc='Standard LW patterns sires')
# model.s_sire_periods = Set(initialize=, doc='sire capacity periods')

##dams
model.s_nut_dams = Set(initialize=uinp.structure['i_n_idx_dams'], doc='Nutrition levels in each feed period for dams')
model.s_lw_dams = Set(initialize=uinp.structure['i_w_idx_dams'], doc='Standard LW patterns damss')
##offs
model.s_sale_offs = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t3_len'])], doc='Sales within the year for offss')
model.s_nut_offs = Set(initialize=uinp.structure['i_n_idx_offs'], doc='Nutrition levels in each feed period for offs')
model.s_lw_offs = Set(initialize=uinp.structure['i_w_idx_offs'], doc='Standard LW patterns offs')
##prog
model.s_sale_prog = Set(initialize=['t%s'%i for i in range(pinp.sheep['i_t2_len'])], doc='Sales and transfers options for yatf')
model.s_lw_prog = Set(initialize=['lw%s'%i for i in range(uinp.structure['i_progeny_w2_len'])], doc='Standard LW patterns prog')




#######################
#pasture             #
#######################
##feed periods
model.s_feed_periods = Set(ordered=True, initialize=pinp.feed_inputs['feed_periods'].index[:-1].astype(str), doc='feed periods') #must be ordered so it can be sliced in pasture pyomo to allow feed to be transferred betweeen periods. Must be string for livestock
##pasture types
model.s_pastures = Set(initialize=uinp.structure['pastures'], doc='feed periods')
model.s_dry_groups = Set(initialize=uinp.structure['dry_groups'], doc='dry feed pools')
model.s_grazing_int = Set(initialize=uinp.structure['grazing_int'], doc='grazing intensity in the growth/grazing activities')
model.s_foo_levels = Set(initialize=uinp.structure['foo_levels'], doc='FOO level in the growth/grazing activities')




















