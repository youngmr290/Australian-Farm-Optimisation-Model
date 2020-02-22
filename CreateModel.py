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

##different fert options - used in labourcroppyomo
model.s_fert_type = Set(initialize=uinp.price['fert_cost'].index, doc='fertiliser options')
>>>>>>> master

###########
#rotation #
###########
##phases
model.s_phases = Set(initialize=uinp.structure['phases'].index, doc='rotation phases set') 
# model.s_phases.pprint()

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

#sheep pools
model.s_sheep_pools = Set(initialize=uinp.structure['sheep_pools'], doc='sheep pools')

#######################
#pasture             #
#######################
##feed periods
model.s_feed_periods = Set(initialize=pinp.feed_inputs['feed_periods'].index, doc='feed periods')






















