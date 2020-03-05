# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:47:33 2019

module: labour module

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)
     
     
@author: young
"""
#python modules
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

#MUDAS modules
from Inputs import * 
from LabourInputs import *
from CropInputs import *
import Periods as per


 
'''
labour periods and length
'''
###################################################################
# make a df containing labour availability for each labour period #
###################################################################

labour_periods = per.p_dates_df() 

#
#period length (days)
def labour_p_length():
    for i in range(len(labour_periods['date'])-1):
        days = labour_periods.loc[i+1,'date'] - labour_periods.loc[i,'date']
        labour_periods.loc[i,'days'] = days
labour_p_length()
        
#leave farmer, this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
def farmer_leave():
    for i in range(len(labour_periods['date'])-1):
        #if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
        if labour_periods.loc[i + 1 , 'date'] < labour_input_data['leave_manager_start_date'] or labour_periods.loc[i , 'date'] > labour_input_data['leave_manager_start_date'] + labour_input_data['leave_manager']:
           labour_periods.loc[i , 'farmer leave'] = datetime.timedelta(days = 0) 
        #if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
        elif labour_periods.loc[i , 'date'] < labour_input_data['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] > labour_input_data['leave_manager_start_date'] + labour_input_data['leave_manager']:
            labour_periods.loc[i , 'farmer leave'] = labour_input_data['leave_manager']
        #if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
        elif labour_periods.loc[i , 'date'] < labour_input_data['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] < labour_input_data['leave_manager_start_date'] + labour_input_data['leave_manager']:
            labour_periods.loc[i , 'farmer leave'] = labour_periods.loc[i + 1 , 'date'] - labour_input_data['leave_manager_start_date']            
        #if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
        elif labour_periods.loc[i , 'date'] > labour_input_data['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] > labour_input_data['leave_manager_start_date'] + labour_input_data['leave_manager']:
            labour_periods.loc[i , 'farmer leave'] = labour_input_data['leave_manager_start_date'] + labour_input_data['leave_manager'] - labour_periods.loc[i , 'date']             
farmer_leave()

#leave permanent, very similar to above but also includes sick leave (10days/year split over each period), this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
def permanent_leave():
    for i in range(len(labour_periods['date'])-1):
        #if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
        if labour_periods.loc[i + 1 , 'date'] < labour_input_data['leave_permanent_start_date'] or labour_periods.loc[i , 'date'] > labour_input_data['leave_permanent_start_date'] + labour_input_data['leave_permanent']:
           labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[i , 'days'] * (labour_input_data['sick_leave_permanent']/365)
        #if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
        elif labour_periods.loc[i , 'date'] < labour_input_data['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] > labour_input_data['leave_permanent_start_date'] + labour_input_data['leave_permanent']:
            labour_periods.loc[i , 'permanent leave'] = labour_input_data['leave_permanent'] + labour_periods.loc[i , 'days'] * (labour_input_data['sick_leave_permanent']/365)
        #if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
        elif labour_periods.loc[i , 'date'] < labour_input_data['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] < labour_input_data['leave_permanent_start_date'] + labour_input_data['leave_permanent']:
            labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[i + 1 , 'date'] - labour_input_data['leave_permanent_start_date']  + labour_periods.loc[i , 'days'] * (labour_input_data['sick_leave_permanent']/365)          
        #if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
        elif labour_periods.loc[i , 'date'] > labour_input_data['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] > labour_input_data['leave_permanent_start_date'] + labour_input_data['leave_permanent']:
            labour_periods.loc[i , 'permanent leave'] = labour_input_data['leave_permanent_start_date'] + labour_input_data['leave_permanent'] - labour_periods.loc[i , 'date']  + labour_periods.loc[i , 'days'] * (labour_input_data['sick_leave_permanent']/365)          
permanent_leave()

#function to determine possible labour days worked by the farmer during the week and on weekend in a given labour perios
def farmer_work_days():
    #available days in the period minus leave multiplied by fraction of weekdays
    labour_periods['farmer weekdays'] = (labour_periods['days'] - labour_periods['farmer leave']) * 5/7
     #available days in the period minus leave multiplied by fraction of weekend days
    labour_periods['farmer weekend'] = (labour_periods['days'] - labour_periods['farmer leave']) * 2/7
farmer_work_days()

#function to determine possible labour days worked by permanent staff during the week and on weekend in a given labour perios
def permanent_work_days():
    #available days in the period minus leave multiplied by fraction of weekdays
    labour_periods['permanent weekdays'] = (labour_periods['days'] - labour_periods['permanent leave']) * 5/7
    #available days in the period minus leave multiplied by fraction of weekend days
    labour_periods['permanent weekend'] = (labour_periods['days'] - labour_periods['permanent leave']) * 2/7
permanent_work_days()

#function to determine possible labour days worked by casual staff during the week and on weekend in a given labour perios
def casual_work_days():
    #available days in the period multiplied by fraction of weekdays
    labour_periods['casual weekdays'] = labour_periods['days'] * 5/7
    #available days in the period multiplied by fraction of weekend days
    labour_periods['casual weekend'] = labour_periods['days'] * 2/7
casual_work_days()

#function to work out total hours available in each period for farmer (owner)
def owner_hours():
    #loops through each period date
    for i in labour_periods['date']:
        #checks if the date is a seed period
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths']):
            #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
            labour_periods.loc[labour_periods['date']==i , 'farmer hours'] = labour_periods.loc[labour_periods['date']==i , 'farmer weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['seeding'] + labour_periods.loc[labour_periods['date']==i , 'farmer weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['seeding'] 
        elif i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'farmer hours'] = labour_periods.loc[labour_periods['date']==i , 'farmer weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['harvest'] + labour_periods.loc[labour_periods['date']==i , 'farmer weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['harvest']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'farmer hours'] = labour_periods.loc[labour_periods['date']==i , 'farmer weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['weekdays'] + labour_periods.loc[labour_periods['date']==i , 'farmer weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['farmer_hours']['weekends']  
owner_hours()

#function to work out total hours available in each period for permanent staff
def permanent_hours():
    #loops through each period date
    for i in labour_periods['date']:
        #checks if the date is a seed period
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths']):
            #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding then multiply by efficiency to take into account supervision time
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['seeding'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['seeding'] 
        elif i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['harvest'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['harvest']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['weekdays'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['permanent_hours']['weekends']  
permanent_hours()

#function to work out total hours available in each period for casual staff
def casual_hours():
    #loops through each period date
    for i in labour_periods['date']:
        #checks if the date is a seed period
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths']):
            #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['seeding'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['seeding'] 
        elif i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['harvest'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['harvest']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['weekdays'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * labour_input_data['casual_hours']['weekends']  
casual_hours()

#function to work out the number of hours of supervision needed by permanent staff
def permanent_supervision():
    #loops through each period date
    for i in labour_periods['date']:
        #checks if the date is a seed period
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths']) \
        or i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            #multiplys number of permanent hours in a given period by the percentage of supervision required for harvest and seeding
            labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
            * labour_input_data['permanent_efficienct']['during harvest and seeding'] 
        else:
            #multiplys number of permanent hours in a given period by the percentage of supervision required for normal activities
            labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
            * labour_input_data['permanent_efficienct']['normal']   
permanent_supervision()

#function to work out the number of hours of supervision needed by casual staff
def casual_supervision():
    #loops through each period date
    for i in labour_periods['date']:
        #checks if the date is a seed period
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths'])   \
        or i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            #multiplys number of casual hours in a given period by the percentage of supervision required for harvest and seeding
            labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
            * labour_input_data['casual_efficienct']['during harvest and seeding'] 
        else:
            #multiplys number of casual hours in a given period by the percentage of supervision required for normal activities
            labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
            * labour_input_data['casual_efficienct']['normal']   
casual_supervision()

#function to determine bounds for casual labour, this is needed because casual labour requirments may be different during seeding and harvest compared to the rest
#makes a constraint
#enter 'min' or 'max'
#when the last row of the period df is dropped it also removes last row of the array produced here
def casual_bound(b):
    #create empty list for upper and lower bounds
    cas_b = np.zeros(shape=(len(labour_periods)))
    #loops through each period date
    for idx, i in enumerate(labour_periods['date']):
        #checks if the date is a seed period or harvest date
        if i in per.period_dates(per.wet_seeding_start_date(),crop_input['seed_period_lengths']) \
        or i in per.period_dates(crop_input['harv_date'],crop_input['harv_period_lengths']):
            #appends upper and lower bounds during seeding dates
            cas_b[idx] = (labour_input_data['%s number casual labour harv seed'% b] )
        else:
            #appends upper and lower bounds during normal periods
            cas_b[idx] = (labour_input_data['%s number casual labour normal'% b] )
    return dict(enumerate(cas_b))
 

#inserts data into the cashflow column of the labour df.
#could be simplified using cashflow func made in finance module
def labour_match_cashflow_periods():
    #index for cash periods
    i = 0
    #lowest date of the current cashflow period (ie start of year)
    lower = labour_periods.loc[1,'date'] + relativedelta(day=1,month=1)
    #upper date of the current cashflow period (ie end of year)
    upper = lower + relativedelta(days=(365 / len(input_data['cashflow_periods'])))
    while i < len(input_data['cashflow_periods']):
        cash_period = input_data['cashflow_periods'][i]
        for j in labour_periods.index:
            if lower <= labour_periods.loc[j,'date'] < upper - relativedelta(days=(15)): #added this bit to make sure it didn't pick up a period that just started
                labour_periods.loc[j, 'cashflow'] = cash_period
            elif lower <= labour_periods.loc[j,'date'] < upper :
                labour_periods.loc[j, 'cashflow'] = input_data['cashflow_periods'][i+1]
        i += 1
        lower += relativedelta(days=(365 / len(input_data['cashflow_periods'])))
        upper += relativedelta(days=(365 / len(input_data['cashflow_periods']))) 
labour_match_cashflow_periods()

#permanent cost per cashflow period - wage plus super plus workers comp and leave ls (multipled by wage because super and others are %)
def perm_cost():
    return (labour_input_data['permanent_cost'] + labour_input_data['permanent_cost'] * labour_input_data['permanent_super'] \
    + labour_input_data['permanent_cost'] * labour_input_data['permanent_workers_comp'] + labour_input_data['permanent_cost'] * labour_input_data['permanent_ls_leave']) / len(input_data['cashflow_periods'])

#casual cost per cashflow period- wage plus super plus workers comp (multipled by wage because super and others are %)
#has to return a dict with labour periods and cashflow periods as a key because it is used as a paramerter with those sets
#differect to perm and manager because they are at a fixed level throughout the year ie same number of perm staff all yr.
def casual_cost():
    #cost of casual for each labour period
    labour_periods['casual_cost'] = labour_periods['casual hours'] * (labour_input_data['casual_cost'] + labour_input_data['casual_cost'] * labour_input_data['casual_super'] + labour_input_data['casual_cost'] * labour_input_data['casual_workers_comp']) 
    #return a dictionary that has the labour cost for each period and the cashflow period
    data_set = labour_periods[['cashflow']]
    keys= list(data_set.itertuples(index=True, name=None))[0:-1] 
    values= labour_periods['casual_cost'][0:-1] 
    return dict(zip(keys,values))
#l=casual_cost()
    
#farmer cost per cashflow period
def farmer_cost():
    return labour_input_data['farmer_cost'] / len(input_data['cashflow_periods'])

# drop last row, because it has na because it only contains the end date, therefore not a period
labour_periods.drop(labour_periods.tail(1).index,inplace=True) 













