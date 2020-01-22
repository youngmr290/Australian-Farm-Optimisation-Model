# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:55:51 2019

module: labour pyomo module - contains pyomo params, variables and constraits

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

@author: young
"""

#python modules
from pyomo.environ import *

#MUDAS modules
from Labour import *
from CreateModel import *


'''
pyomo parameters
'''
#number of hours provided by a permanent staff in one period 
model.perm_hours = Param(model.s_periods, initialize= labour_periods['permanent hours'].to_dict(), doc='hours worked by a permanent staff in each period')

#number of hours provided by a permanent staff in one period 
model.perm_supervison = Param(model.s_periods, initialize= labour_periods['permanent supervision'].to_dict(), doc='hours of supervision required by a permanent staff in each period')

#cost of permanent
model.perm_cost = Param(model.s_cashflow_periods, initialize = perm_cost(), default = 0.0, doc = 'cost of a permanent staff for 1 yr')

#cost of casual
model.casual_cost = Param(model.s_periods, model.s_cashflow_periods,  initialize = casual_cost(), default = 0.0, doc = 'cost of a casual staff for each labour period')

#number of hours provided by a casual staff in one period 
model.casual_hours = Param(model.s_periods, initialize= labour_periods['casual hours'].to_dict(), doc='hours worked by a casual staff in each period')

#number of hours provided by a casual staff in one period 
model.casual_supervison = Param(model.s_periods, initialize= labour_periods['casual supervision'].to_dict(), doc='hours of supervision required by a casual staff in each period')

#number of hours provided by a farmer in each period 
model.farmer_hours = Param(model.s_periods, initialize= labour_periods['permanent hours'].to_dict(), doc='hours worked by a farmer in each period')

#cost of farmer
model.farmer_cost = Param(model.s_cashflow_periods, initialize = farmer_cost(), doc = 'cost of a farmer for 1 yr')

#casual labour upper bound per period
model.casual_upper = Param(model.s_periods, initialize = casual_bound('max'),  doc = 'casual availability upper bound')

#casual labour lower bound per period
model.casual_lower = Param(model.s_periods, initialize = casual_bound('min'), doc = 'casual availability lower bound')


'''
pyomo variables
'''
#Amount of casual. Casual labour can be optimised for each period 
model.quantity_casual = Var(model.s_periods, bounds = (0,None) , doc='number of casual labour used in each labour period')

#Amount of permanent labour. 
model.quantity_perm = Var(bounds=(labour_input_data['min number permanent labour'],labour_input_data['max number permanent labour']), doc='number of permanent labour used in each labour period')

#Amount of farmer labour 
model.quantity_manager = Var(bounds=(labour_input_data['min number owner labour'],labour_input_data['max number owner labour']), doc='number of farmer/owner labour used in each labour period')

#farmer pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.sheep_labour_manager = Var(model.s_periods, bounds = (0,None), doc='farmer labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.crop_labour_manager = Var(model.s_periods, bounds = (0,None), doc='farmer labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.fixed_labour_manager = Var(model.s_periods, bounds = (0,None), doc='farmer labour used by fixed activities in each labour period')

#permanent pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.sheep_labour_permanent = Var(model.s_periods, bounds = (0,None), doc='permanent labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.crop_labour_permanent = Var(model.s_periods, bounds = (0,None), doc='permanent labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.fixed_labour_permanent = Var(model.s_periods, bounds = (0,None), doc='permanent labour used by fixed activities in each labour period')

#casual pool
#labour for sheep activities (this variable transfers labour from source to sink)
model.sheep_labour_casual = Var(model.s_periods, bounds = (0,None), doc='casual labour used by sheep activities in each labour period')

#labour for crop activities (this variable transfers labour from source to sink)
model.crop_labour_casual = Var(model.s_periods, bounds = (0,None), doc='casual labour used by crop activities in each labour period')

#labour for fixed activities (this variable transfers labour from source to sink)
model.fixed_labour_casual = Var(model.s_periods, bounds = (0,None), doc='casual labour used by fixed activities in each labour period')

#transfer labour between pools
#transfer labour from manager to casual, because jobs are tasked to particular pools and jobs tasked to casual can be done by the manager or permanent staff 
model.manager_casual_labour_transfer = Var(model.s_periods, bounds = (0,None), doc='manager labour used to complete jobs tasked to casual')

#transfer labour from manager to permanent 
model.manager_permanent_labour_transfer = Var(model.s_periods, bounds = (0,None), doc='manager labour used to complete jobs tasked to permanent')

#transfer labour from permanent to casual 
model.permanent_casual_labour_transfer = Var(model.s_periods, bounds = (0,None), doc='permanent labour used to complete jobs tasked to casual')

'''
pyomo constraints
'''
#farmer, this is a little more complex because also need to subtract the supervision hours off of the farmer supply of workable hours
#also subtracts off the amount of transfer labour which is labour used for jobs allocated to casual and permanent staff.
def labour_transfer_manager(model,p):
    return (model.quantity_manager * model.farmer_hours[p] - model.perm_supervison[p] * model.quantity_perm - model.casual_supervison[p] * model.quantity_casual[p]      \
    - model.manager_casual_labour_transfer[p] - model.manager_permanent_labour_transfer[p])- model.sheep_labour_manager[p] - model.crop_labour_manager[p] - model.fixed_labour_manager[p]  >= 0
model.labour_transfer_manager = Constraint(model.periods, rule = labour_transfer_manager, doc='labour from farmer to sheep and crop and fixed')

#permanent 
def labour_transfer_permanent(model,p):
    return model.quantity_perm *  model.perm_hours[p] + model.manager_permanent_labour_transfer[p] - model.permanent_casual_labour_transfer[p]   \
    - model.sheep_labour_permanent[p] - model.crop_labour_permanent[p] - model.fixed_labour_permanent[p] >= 0
model.labour_transfer_permanent = Constraint(model.periods, rule = labour_transfer_permanent, doc='labour from permanent staff to sheep and crop and fixed')

#casual note perm and manager can do casual tasks - variables may need to change name so to be less confusing
def labour_transfer_casual(model,p):
    return model.quantity_casual[p] *  model.casual_hours[p] + model.manager_casual_labour_transfer[p] + model.permanent_casual_labour_transfer[p] \
    - model.sheep_labour_casual[p] - model.crop_labour_casual[p] - model.fixed_labour_casual[p]  >= 0
model.labour_transfer_casual = Constraint(model.periods, rule = labour_transfer_casual, doc='labour from casual staff to sheep and crop and fixed')


#######################
#labour cost function #
#######################

#sum the cost of perm, casual and farmer labour. When i tried to do it all in one function it didn't work (it should be possible though )
def casual(model,c):
    return -sum( model.quantity_casual[p] * model.casual_cost[p,c] for p in model.s_periods) 
def perm(model,c):
    return -model.quantity_perm * model.perm_cost[c] 
def farmer(model,c):
    return -model.quantity_manager * model.farmer_cost[c] 
def labour_cost(model,c):
    return casual(model,c) + perm(model,c) + farmer(model,c)

###############################
#local constraints            #
###############################
#to constrain the amount of casual labour in each period
#this can't be done with variable bounds because it's not a constant value for each period (seeding and harv may differ)
def casual_labour_availability(model, p):
    return  (model.casual_lower[p], model.quantity_casual[p], model.casual_upper[p])
model.casual_bounds = Constraint(model.periods, rule = casual_labour_availability, doc='bounds the casual labour in each period')





#model.pprint()
