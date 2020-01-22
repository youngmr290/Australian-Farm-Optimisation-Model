# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:55:58 2019

module: crop input module - contains all input data likely to vary for different regions or farms

Version Control:
Version     Date        Person  Change
1.1         6jan20      MRY     Updated formatting 



     
     
@author: young
"""
#python modules
import datetime
import pandas as pd
import numpy as np


#other midas modules
import Functions as fun 


##################
#crop dictionary #
##################

##create empty dict
crop_input = dict()


#########################
#excel read in          #
#########################

# crop_input['excel_ranges'] = fun.xl_all_named_ranges("Property.xlsx","MRY Rotation") 


############################################################################################################################################################
############################################################################################################################################################
#periods
############################################################################################################################################################
############################################################################################################################################################


####################
# seeding periods  #
####################
'''
seed periods - you can add seed periods here as long as it is in succession to these periods (ie it happens straight after)
first period occurs a set number of days after season break
dry seeding would need its own seed periods - if labour was to be increased for \
dry seeding as it is for wet seeding it would also need to be included in the labour module that determine the labour periods
'''
##period length (days)
# crop_input['seed_period_lengths'] = [21, 5, 5, 5] #first number represents penalty free days

##number of days after season break that seeding can begin
# crop_input['seeding_after_season_start'] = datetime.timedelta(days = 10)


# ####################
# # harvest periods  #
# ####################
# '''
# flexible enough to add or remove harv period (dates can also be changed) - this affects mach sheet and labour periods
# '''
# #early harvest start date
# crop_input['harv_date'] = datetime.datetime(2019,11,25)

# #length of each harv period
# crop_input['harv_period_lengths'] = [10, 27]

# ##date to start harvesting a crop.
# ##at the moment it will get the max number of days of the period the date falls into (ie even if the date is half way through the first harv period it will still be given the full period harv days)
# ##later it could be improved to get a proportion depending on the exact date
# crop_input['start_harvest_crops'] = {        'b'    : datetime.datetime(2019,12,5),   \
#                                                          'w'    : datetime.datetime(2019,12,5), \
#                                                          'o'    : datetime.datetime(2019,12,5),   \
#                                                          'l'    : datetime.datetime(2019,11,25), \
#                                                          'z'    : datetime.datetime(2019,11,25),  \
#                                                          'r'    : datetime.datetime(2019,11,25),  \
#                                                          'f'    : datetime.datetime(2019,12,5),  \
#                                                          'h'   : datetime.datetime(2019,12,5)} #hay date is just used in stubble/stubble sim sheet, this date may need to change

    
############################################################################################################################################################
############################################################################################################################################################
#Grain yield stuff
############################################################################################################################################################
############################################################################################################################################################
    
####################
# yield penalty    #
####################
    
# ##kg/ha/day for late seeding
# crop_input['yield_penalty'] = {        'b'    : 20,    \
#                                             'w'    : 25,    \
#                                             'o'    : 15,       \
#                                             'l'    : 25,   \
#                                             'z'    : 30,   \
#                                             'r'    : 30,   \
#                                             'f'   : 25}

# ####################
# # grain prices     #
# ####################

# ##grain price $/t
# crop_input['grain_price'] =  {'b' : {'firsts' : 295
#                                           ,'seconds':265}
#                             ,'w' : {'firsts' : 295
#                                         ,'seconds':280}
#                             ,'o' : {'firsts' : 235}
#                             ,'l' : {'firsts' : 305}
#                             ,'z' : {'firsts' : 550}
#                             ,'r' : {'firsts' : 550}
#                             ,'f' : {'firsts' : 350}
#                             ,'h' : {'firsts' : 150}}

# ##proportion of harvested grain that goes first and second class (%) - must equal 1
# crop_input['fisrt_seconds_prop'] =  {'b' : {'firsts' : 0.7
#                                                  ,'seconds':0.3}
#                                     ,'w' : {'firsts' : 0.8
#                                                 ,'seconds':0.2}
#                                     ,'o' : {'firsts' : 1}
#                                     ,'l' : {'firsts' : 1}
#                                     ,'z' : {'firsts' : 1}
#                                     ,'r' : {'firsts' : 1}
#                                     ,'f' : {'firsts' : 1}
#                                     ,'h' : {'firsts' : 1}}

##############
#cartage $/t #
##############

# ##flagfall ($/t)
# crop_input['flagfall'] = 2 
# ##rail cartage cost, koji-alb ($/t)
# crop_input['rail_cartage_cost'] = 13.59 
# ##road cartage distance (km)  
# crop_input['road_cartage_distance'] = 25
##cost for road cartage ($/km/t) - driven by desiel price (should note down the desiel price when calibrating)                           
# crop_input['cartage_km_cost'] =  {'b' : 0.15
#                             ,'w' : 0.15
#                             ,'o' : 0.15
#                             ,'l' : 0.15
#                             ,'z' : 0.12
#                             ,'r' : 0.12
#                             ,'f' : 0.15
#                             ,'h' : 0.50}
                            
# ##cbh fees & gov levy ($/t) - updated 2019 using cbh data - this number includes; receical fees, BAMA fee and Canola testing fee. Gov levies also included - I have recodred the full info in the calabration section of dropbox
# crop_input['grain_tolls'] =  {'b' : 13.45
#                             ,'w' : 12.05
#                             ,'o' : 12.85
#                             ,'l' : 12.05
#                             ,'z' : 21.95
#                             ,'r' : 21.95
#                             ,'f' : 15.85
#                             ,'h' : 1.5}

# ##cash period allocation
# crop_input['grain_income_date'] = datetime.datetime(2019,12,15)
# crop_input['grain_income_length'] = datetime.timedelta(days = 5) #increasing this can make the income split over multiple periods
                

# '''
# Re-seeded pasture
# '''
# ###############################
# #reseeded pasture extra inputs#   
# ###############################
# #the extra fert that reseeded pas gets. note this is sumed to the inputted amount the std pas already gets
# #make sure fert is spelt the same as everywhere else
# crop_input['reseed_pas'] =  {'mop':0,
#                              'super':50}

'''
fert
'''

#######
#fert#
######
# ##proportion of fert spread with spreader - generally 1 unless applied at harvest
# ##this is just used to determine fert app cost per tonne - app cost per ha is accounted for using passes input, however i couldn't think of a good way to use it for the tonne cost
# crop_input['spreader_proportion'] =  {'agflow' : 0
#                             ,'ureamop' : 1 
#                             ,'ns' : 1
#                             ,'urea' : 1 
#                             ,'super' : 1
#                             ,'mop' : 1
#                             ,'lime' : 1}
# ##fert density t/m3
# crop_input['fert_density'] =  {'agflow' : 0.97
#                             ,'ureamop' : 0.86
#                             ,'ns' : 0.85
#                             ,'urea' : 0.7 
#                             ,'super' : 1.15
#                             ,'mop' : 1.1
#                             ,'lime' : 1.3}
# ##fert cost $/t
# crop_input['fert_cost'] =  {'agflow' : 624
#                             ,'ureamop' : 455
#                             ,'ns' : 401
#                             ,'urea' : 440
#                             ,'super' : 339
#                             ,'mop' : 541
#                             ,'lime' : 15}
##cartage $/t
# crop_input['fert_cartage_cost'] = 23

###################
#fert application #
###################

# ##date of application for each fert, use a range  - if the fert is applied twice you can just increase the range
# crop_input['fert_app'] = ({'date': {'agflow' : datetime.datetime(2019,5,15)
#                                     ,'ureamop' : datetime.datetime(2019,6,1)
#                                     ,'ns' : datetime.datetime(2019,7,7)
#                                     ,'urea' : datetime.datetime(2019,7,7)
#                                     ,'super' : datetime.datetime(2019,2,15)
#                                     ,'mop' : datetime.datetime(2019,7,1)
#                                     ,'lime' : datetime.datetime(2019,4,1)}
#                           ,'length': {'agflow' : datetime.timedelta(days = 5)
#                                     ,'ureamop' : datetime.timedelta(days = 5)
#                                     ,'ns' : datetime.timedelta(days = 5)
#                                     ,'urea' : datetime.timedelta(days =5)
#                                     ,'super' : datetime.timedelta(days = 90) #often spread between feb and march because it doesn't leach much
#                                     ,'mop' : datetime.timedelta(days = 5)
#                                     ,'lime' : datetime.timedelta(days = 5)}})


###################                       
#stubble handling #
###################
#handling threashold (t/ha) - yeild at which stubble must be handeled.                           
# crop_input['stubble_threashold'] =  {'b' : 4.2
#                                     ,'w' : 3.5
#                                     ,'o' : 3.5
#                                     #,'lupins' : 0  #din't include crop with 0 otherwise div0 error. not including makes the value turn to nan in the calc so it is accounted for that way
#                                     ,'z' : 2.3
#                                     ,'r' : 2.3}
#                                     #,'faba' : 0
#                                     #,'hay' : 0}








