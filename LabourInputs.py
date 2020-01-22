# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 19:42:26 2019


module: labour input module - contains all input data likely to vary for different regions or farms

extra info: - this could eventually interact with a user interface
            - interacts with kv's

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
     
@author: young
"""
import datetime
#################################
#define dict for labour  inputs #
#################################

labour_input_data = dict()


########################
#leave and hours worked#
########################

#sick leave permanent staff, doesn't need to be in timedata format
labour_input_data['sick_leave_permanent'] = 10

#post harvest leave permanent staff (usually takes a holiday 14days in jan and 14 days in feb)
labour_input_data['leave_permanent_start_date'] = datetime.datetime(2019,1,15)
labour_input_data['leave_permanent'] = datetime.timedelta(days = 28)

#post harvest leave farmer (usually takes a holiday 14days in feb and 14 days in mar)
labour_input_data['leave_manager_start_date'] = datetime.datetime(2019,2,15)
labour_input_data['leave_manager'] = datetime.timedelta(days = 28)

#labour efficency casual - amount of time to supervise = 25% normally and 15% during seeding and harvest
labour_input_data['casual_efficienct'] = {  'normal'	: 0.25, \
                                      'during harvest and seeding' : 0.15}

#labour efficency permanent - amount of time to supervise = 7% normally and 2% during seeding and harvest
labour_input_data['permanent_efficienct'] = {  'normal'	: 0.07, \
                                        'during harvest and seeding' : 0.02}

#hours worked by casual 
labour_input_data['casual_hours'] = {       'weekdays'	: 8, \
                                     'weekends'	: 0, \
                                     'seeding'	: 8, \
                                     'harvest'   : 8}

#hours worked by permanent 
labour_input_data['permanent_hours'] = {        'weekdays'	: 8, \
                                         'weekends'	: 0, \
                                         'seeding'	: 9, \
                                         'harvest'   : 9}

#hours worked by farmer 
labour_input_data['farmer_hours'] = {           'weekdays'	: 9,   \
                                         'weekends'	: 4.5, \
                                         'seeding'	: 10,  \
                                         'harvest'   : 9}


##########################
#number of staff (bounds)#
##########################

#number of owner staff (usually 1)
labour_input_data['max number owner labour'] = 1
labour_input_data['min number owner labour'] = 1

#number of permanent staff, note this can't be 0 if other staff require supervision
labour_input_data['max number permanent labour'] = 1
labour_input_data['min number permanent labour'] = 0

#number of causal staff normal periods
labour_input_data['max number casual labour normal'] = 1
labour_input_data['min number casual labour normal'] = 0.0

#number of casual staff harvest and seeding
labour_input_data['max number casual labour harv seed'] = 1
labour_input_data['min number casual labour harv seed'] = 0


#######
#cost #
#######

#farmer cost per yr
labour_input_data['farmer_cost'] = 80000

#permanent cost per yr (before super)
labour_input_data['permanent_cost'] = 80000
#permanent super
labour_input_data['permanent_super'] = 0.09
#permanent workers compensation
labour_input_data['permanent_workers_comp'] = 0.035
#permanent LS leave
labour_input_data['permanent_ls_leave'] = 0.023

#casual cost per hour (before super)
labour_input_data['casual_cost'] = 28
#casual super
labour_input_data['casual_super'] = 0.09
#casual workers compensation
labour_input_data['casual_workers_comp'] = 0.035

