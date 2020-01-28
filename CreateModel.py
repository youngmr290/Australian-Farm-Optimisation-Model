# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:19:20 2019

module: create the model. this then gets used in all sheets that use pyomo

@author: young
"""


#python modules
from pyomo.environ import *
import pandas as pd

#MUDAS modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import Periods as per
import Crop as crp
import StubbleInputs as sinp


#from Finance import *
#from LabourFixed import *
#from Mach import * 


print('Status:  running create model')


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
model.s_periods = Set(initialize=per.p_date2_df().index)


#######################
#cash                 #
#######################

#cashflow periods
model.s_cashflow_periods = Set(initialize=uinp.structure['cashflow_periods'], doc='cashflow periods')

#######################
#stubble              #
#######################

#stubble categories
model.s_stub_cat = Set(initialize=sinp.stubble_inputs['crop_stub']['w']['stub_cat_qual'].keys(), doc='stubble categories') 

#######################
#cropping related     #
#######################

#types of crops
model.s_crops = Set(initialize=uinp.structure['C'], doc='crop types')

#soils
model.s_lmus = Set(initialize=pinp.general['lmu_area'].index, doc='defined the soil type a given rotation is on')
# model.s_lmus.pprint()

###########
#rotation #
###########
# def rotation_set():
#     return crp.phases_df.set_index(list(range(uinp.structure['phase_len']))).index
# model.s_phases = Set(dimen=uinp.structure['phase_len'],initialize=rotation_set(), doc='rotation phases set') 
# def s_rotation_hist():
#     return pd.unique(uinp.structure['phases'].set_index(list(range(uinp.structure['phase_len']-1))).index)
# model.s_phaseshist = Set(dimen=uinp.structure['phase_len']-1,initialize=s_rotation_hist(), doc='rotation phase history set') #using for crop yield constraint
# def s_rotation():
#     return uinp.structure['phases'].set_index(list(range(uinp.structure['phase_len']))).index
# model.s_phases = Set(dimen=uinp.structure['phase_len'],initialize=s_rotation(), doc='rotation phases set') #have to have this as a multi dimensional set for crop yield constraint
model.s_phases = Set(initialize=uinp.structure['phases'].index, doc='rotation phases set') 
# model.s_phaseshist.pprint()
# model.s_phases.pprint()


# def phase_hist():
#     rot_constraints =pd.Series(uinp.structure['rotations']['constraints']).str.split(expand=True).dropna()
#     return rot_constraints.set_index([*range(uinp.structure['phase_len']-1)]).index
# model.s_rotconstraints = Set(dimen=uinp.structure['phase_len']-1,initialize=phase_hist(),doc='rotation constraints histories')
s_rotcon1 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con1 set', header= None, index_col = 0)
model.s_rotconstraints = Set(initialize=s_rotcon1.index, doc='rotation constraints histories')
# model.s_rotconstraints.pprint()

# def phase_hist2():
#     rot_constraints =pd.Series(uinp.structure['rotations']['constraints2']).str.split(expand=True).dropna()
#     return rot_constraints.set_index([*range(uinp.structure['phase_len']-1)]).index
# model.s_rotconstraints2 = Set(dimen=uinp.structure['phase_len']-1,initialize=phase_hist2(),doc='rotation constraints histories 2')
s_rotcon2 = pd.read_excel('Rotation.xlsx', sheet_name='rotation con2 set', header= None, index_col = 0)
model.s_rotconstraints2 = Set(initialize=s_rotcon2.index, doc='rotation constraints histories 2')


# def phase_hist():
#     rot_constraints =pd.Series(inp.structure['rotations']['constraints']).dropna()
#     return rot_constraints
# # model.rot_constraints = Set(dimen=rinp.rotation_data['phase_len']-1,initialize=phase_hist(),doc='rotation constraints histories')
# model.rot_constraints = Set(initialize=phase_hist(),doc='rotation constraints histories')
# model.rot_constraints.pprint()
#model.phases.pprint()
    
#     phases_hist=fun.phases(inp.structure['phases'],rinp.rotation_data['phase_len']-1) #all possible rotation histories (not to be confused with above)
#     return phases_hist.set_index(list(range(len(phases_hist.columns)))).index
# model.rot_hist = Set(dimen=3,initialize=phase_hist())

#different fert options - used in labourcroppyomo
model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')


#######################
#sheep                 #
#######################

#sheep pools
model.s_sheep_pools = Set(initialize=uinp.structure['sheep_pools'], doc='sheep pools')

#######################
#pasture             #
#######################

model.s_feed_periods = Set(initialize=pinp.feed_inputs['feed_periods'].index, doc='feed periods')






















