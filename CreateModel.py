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


'''
# Creation of a Concrete Model
'''
model = ConcreteModel()
model.report_timing=True #haven't actually been able to get this to do any thing yet?????

'''
pyomo sets
'''

#######################
#labour               #
#######################

#labour periods
model.s_labperiods = Set(initialize=per.p_date2_df().index)


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
model.s_haycrops = Set(initialize=uinp.structure['Hay'], doc='landuses that make hay')

##types of crops
model.s_crops = Set(initialize=uinp.structure['C'], doc='crop types')

##all crops and the pasture types ie annual, tedera, lucerne (not a, a3, a4 etc)
model.s_landuses = Set(initialize=uinp.structure['All'], doc='landuses')

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
model.infrastructure = Set(initialize=, doc='core sheep infrastructure')
model.s_sheep_pools = Set(initialize=uinp.structure['sheep_pools'], doc='nutritive value pools')
model.s_tol = Set(initialize=pinp.sheep['i_tol_idx'][pinp.sheep['i_mask_i']], doc='birth groups (times of lambing)')
##dams & offs
model.s_wean_times = Set(initialize=pinp.sheep['i_wean_times'][pinp.sheep['i_mask_a']], doc='timing of weaning each year. Input as the age of the yatf when weaned')
model.s_e_cycles = Set(initialize=np.array(['e' + str(i) for i in np.arange(max(pinp.sheep['i_wean_times']))]), doc='oestrus cycles') #make set names by joining e with the number of cycles
 
model.s_conception = Set(initialize=, doc='carryover characteristics - conception')
model.s_bw = Set(initialize=, doc='carryover characteristics - Birth weight')
model.s_ww = Set(initialize=, doc='carryover characteristics - Weaning weight')
model.s_cfw = Set(initialize=, doc='carryover characteristics - Clean fleece weight')
model.s_fd = Set(initialize=, doc='carryover characteristics - Fibre diameter')
model.s_min_fd = Set(initialize=, doc='carryover characteristics - Minimum fibre diameter')
model.s_fl = Set(initialize=, doc='carryover characteristics - Fibre length')

   
##sire
model.s_sale_sire = Set(initialize=, doc='Sales within the year for sires')
model.s_fvp_sire = Set(initialize=, doc='Feed variation periods for sires')
model.s_nut_sire = Set(initialize=uinp.structure['i_n_idx_sire'], doc='Nutrition levels in each feed period for sires')
model.s_lw_sire = Set(initialize=uinp.structure['i_w_idx_sire'], doc='Standard LW patterns sires')
model.s_gen_merit_sire = Set(initialize=uinp.parameter['i_gen_merit_sire'], doc='genetic merit of sires')
model.s_groups_sire = Set(initialize=pinp.sheep['i_groups_sire'], doc='geneotype groups of sires')
##dams
model.s_sale_dams = Set(initialize=, doc='Sales within the year for damss')
model.s_fvp_dams = Set(initialize=, doc='Feed variation periods for damss')
model.s_lsln_dams = Set(initialize=uinp.structure['i_lsln_idx_dams'], doc='litter size lactation number for dams')
model.s_nut_dams = Set(initialize=uinp.structure['i_n_idx_dams'], doc='Nutrition levels in each feed period for dams')
model.s_lw_dams = Set(initialize=uinp.structure['i_w_idx_dams'], doc='Standard LW patterns damss')
model.s_gen_merit_dams = Set(initialize=uinp.parameter['i_gen_merit_dams'], doc='genetic merit of dams')
model.s_groups_dams = Set(initialize=pinp.sheep['i_groups_dams'], doc='geneotype groups of dams')
##offs
model.s_sale_offs = Set(initialize=, doc='Sales within the year for offss')
model.s_fvp_offs = Set(initialize=, doc='Feed variation periods for offss')
model.s_btrt_offs = Set(initialize=uinp.structure['i_btrt_idx_offs'], doc='birth type and rear type for offs')
model.s_nut_offs = Set(initialize=uinp.structure['i_n_idx_offs'], doc='Nutrition levels in each feed period for offs')
model.s_lw_offs = Set(initialize=uinp.structure['i_n_idx_offs'], doc='Standard LW patterns offs')
model.s_damage_offs = Set(initialize=, doc='age of mother - offs')
model.s_gender_offs = Set(initialize=, doc='gender of offs')
model.s_gen_merit_offs = Set(initialize=uinp.parameter['i_gen_merit_offs'], doc='genetic merit of offs')
model.s_groups_offs = Set(initialize=pinp.sheep['i_groups_offs'], doc='geneotype groups of offs')



#######################
#pasture             #
#######################
##feed periods
model.s_feed_periods = Set(ordered=True, initialize=pinp.feed_inputs['feed_periods'].index[:-1], doc='feed periods') #must be ordered so it can be sliced in pasture pyomo to allow feed to be transferred betweeen periods.
##pasture types
model.s_pastures = Set(initialize=uinp.structure['pastures'], doc='feed periods')
model.s_dry_groups = Set(initialize=uinp.structure['dry_groups'], doc='dry feed pools')
model.s_grazing_int = Set(initialize=uinp.structure['grazing_int'], doc='grazing intensity in the growth/grazing activities')
model.s_foo_levels = Set(initialize=uinp.structure['foo_levels'], doc='FOO level in the growth/grazing activities')

#######################
#seasons              #
#######################
##season types - set only has one season if steady state model is being used
if pinp.general['steady_state']:
    model.s_season_types = Set(initialize='season 1', doc='season types')
else:    
    model.s_season_types = Set(initialize=pinp.general['season_info'].index[pinp.general['season_info']['included']], doc='season types') #mask season types by the ones included



















