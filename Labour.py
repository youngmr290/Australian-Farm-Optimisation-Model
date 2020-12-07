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
# from LabourInputs import *
import PropertyInputs as pinp
import UniversalInputs as uinp
import Periods as per
import Functions as fun

 
'''
labour periods and length
'''
###################################################################
# make a df containing labour availability for each labour period #
###################################################################



def labour_general(params,r_vals):
    '''
    Returns
    -------
    None.
        Wraps labour into function.
        1- calcs day and length of each labour period
        2- calcs the leave for staff in each period
        3- calcs max days that can be worked per periods (on the weekend and weekdays)
        4- calcs the hours available in each period
        5- calcs supervision requirement for casual and permanent staff

    '''
    labour_periods = per.p_dates_df() 
    for i in range(len(labour_periods['date'])-1):
        ##period length (days)
        days = labour_periods.loc[i+1,'date'] - labour_periods.loc[i,'date']
        labour_periods.loc[i,'days'] = days
        
        ##leave manager, this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
        ###if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
        if labour_periods.loc[i + 1 , 'date'] < pinp.labour['leave_manager_start_date'] or labour_periods.loc[i , 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
           labour_periods.loc[i , 'manager leave'] = datetime.timedelta(days = 0) 
        ###if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
        elif labour_periods.loc[i , 'date'] < pinp.labour['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
            labour_periods.loc[i , 'manager leave'] = datetime.timedelta(days = pinp.labour['leave_manager'])
        ###if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
        elif labour_periods.loc[i , 'date'] < pinp.labour['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] < pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
            labour_periods.loc[i , 'manager leave'] = labour_periods.loc[i + 1 , 'date'] - pinp.labour['leave_manager_start_date']            
        ###if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
        elif labour_periods.loc[i , 'date'] > pinp.labour['leave_manager_start_date'] and labour_periods.loc[i + 1 , 'date'] > pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']):
            labour_periods.loc[i , 'manager leave'] = pinp.labour['leave_manager_start_date'] + datetime.timedelta(days = pinp.labour['leave_manager']) - labour_periods.loc[i , 'date']             
        
        ##leave permanent, very similar to above but also includes sick leave (10days/year split over each period), this only works if leave is taken in one chuck, if it were taken in two lots this would have to be altered
        ###if the end of labour period i is before leave begins or labour period starts after leave finished then there is 0 leave for that period
        if labour_periods.loc[i + 1 , 'date'] < pinp.labour['leave_permanent_start_date'] or labour_periods.loc[i , 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
           labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
        ###if labour i period starts before leave starts and leave finishes before the labour period finished then that period gets all the leave.
        elif labour_periods.loc[i , 'date'] < pinp.labour['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
            labour_periods.loc[i , 'permanent leave'] = datetime.timedelta(days = pinp.labour['leave_permanent']) + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)
        ###if labour i period starts before leave starts and leave finishes after the labour period finished then that period gets leave from the start date of leave to the end of the labour period.
        elif labour_periods.loc[i , 'date'] < pinp.labour['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] < pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
            labour_periods.loc[i , 'permanent leave'] = labour_periods.loc[i + 1 , 'date'] - pinp.labour['leave_permanent_start_date']  + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)          
        ###if labour i period starts after leave starts and leave finishes before the labour period finished then that period gets leave from the beginning on the labour period to the end date of leave.
        elif labour_periods.loc[i , 'date'] > pinp.labour['leave_permanent_start_date'] and labour_periods.loc[i + 1 , 'date'] > pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']):
            labour_periods.loc[i , 'permanent leave'] = pinp.labour['leave_permanent_start_date'] + datetime.timedelta(days = pinp.labour['leave_permanent']) - labour_periods.loc[i , 'date']  + labour_periods.loc[i , 'days'] * (pinp.labour['sick_leave_permanent']/365)          

    ##determine possible labour days worked by the manager during the week and on weekend in a given labour periods
    ###available days in the period minus leave multiplied by fraction of weekdays
    labour_periods['manager weekdays'] = (labour_periods['days'] - labour_periods['manager leave']) * 5/7
    ###available days in the period minus leave multiplied by fraction of weekend days
    labour_periods['manager weekend'] = (labour_periods['days'] - labour_periods['manager leave']) * 2/7

    ##determine possible labour days worked by permanent staff during the week and on weekend in a given labour perios
    ###available days in the period minus leave multiplied by fraction of weekdays
    labour_periods['permanent weekdays'] = (labour_periods['days'] - labour_periods['permanent leave']) * 5/7
    ###available days in the period minus leave multiplied by fraction of weekend days
    labour_periods['permanent weekend'] = (labour_periods['days'] - labour_periods['permanent leave']) * 2/7

    ##function to determine possible labour days worked by casual staff during the week and on weekend in a given labour perios
    ###available days in the period multiplied by fraction of weekdays
    labour_periods['casual weekdays'] = labour_periods['days'] * 5/7
    ###available days in the period multiplied by fraction of weekend days
    labour_periods['casual weekend'] = labour_periods['days'] * 2/7

    ##get cashflow period dates and names - used in the following loop
    p_dates = per.cashflow_periods()['start date']#get cashflow period dates
    p_name = per.cashflow_periods()['cash period']#gets the period name
    
    for i in labour_periods['date']: #loops through each period date
        ##work out total hours available in each period for manager (owner)
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']): #checks if the date is a seed period
            labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Manager'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
        elif i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Manager']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'manager hours'] = labour_periods.loc[labour_periods['date']==i , 'manager weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekdays', 'Manager'] + labour_periods.loc[labour_periods['date']==i , 'manager weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekends', 'Manager']  
        ##work out total hours available in each period for permanent staff
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']): #checks if the date is a seed period
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Permanent'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding 
        elif i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Permanent']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'permanent hours'] = labour_periods.loc[labour_periods['date']==i , 'permanent weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekdays', 'Permanent'] + labour_periods.loc[labour_periods['date']==i , 'permanent weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekends', 'Permanent']  
        ##work out total hours available in each period for casual staff
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']): #checks if the date is a seed period
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['seeding', 'Casual'] #convert the datetime into a float by dividing number of days by 1 day, then multiply by number of hours that can be worked during seeding.
        elif i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']):
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['harvest', 'Casual']  
        else:
            labour_periods.loc[labour_periods['date']==i , 'casual hours'] = labour_periods.loc[labour_periods['date']==i , 'casual weekdays'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekdays', 'Casual'] + labour_periods.loc[labour_periods['date']==i , 'casual weekend'] / datetime.timedelta(days=1) \
            * pinp.labour['daily_hours'].loc['weekends', 'Casual']  

        ##work out the number of hours of supervision needed by permanent staff
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']) \
        or i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']): #checks if the date is a seed period or harvest period
            ###multiplys number of permanent hours in a given period by the percentage of supervision required for harvest and seeding
            labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
            * pinp.labour['labour_eff'].loc['seedingharv', 'Permanent'] 
        else:
            ###multiplys number of permanent hours in a given period by the percentage of supervision required for normal activities
            labour_periods.loc[labour_periods['date']==i , 'permanent supervision'] = labour_periods.loc[labour_periods['date']==i , 'permanent hours'] \
            * pinp.labour['labour_eff'].loc['normal', 'Permanent']   

        ##function to work out the number of hours of supervision needed by casual staff
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths'])   \
        or i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']): #checks if the date is a seed period or harvest period
            ###multiplys number of casual hours in a given period by the percentage of supervision required for harvest and seeding
            labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
            * pinp.labour['labour_eff'].loc['seedingharv', 'Casual'] 
        else:
            ###multiplys number of casual hours in a given period by the percentage of supervision required for normal activities
            labour_periods.loc[labour_periods['date']==i , 'casual supervision'] = labour_periods.loc[labour_periods['date']==i , 'casual hours'] \
            * pinp.labour['labour_eff'].loc['normal', 'Casual']   

        ##determine bounds for casual labour, this is needed because casual labour requirments may be different during seeding and harvest compared to the rest
        ###upper bound
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']) \
        or i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']): #checks if the date is a seed period or harvest date
            labour_periods.loc[labour_periods['date']==i , 'casual ub'] =  (pinp.labour['max_casual_seedharv'] )
        else:
            labour_periods.loc[labour_periods['date']==i , 'casual ub'] = (pinp.labour['max_casual'] )
        ###lower bound
        if i in per.period_dates(per.wet_seeding_start_date(),pinp.crop['seed_period_lengths']) \
        or i in per.period_dates(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']): #checks if the date is a seed period or harvest date
            labour_periods.loc[labour_periods['date']==i , 'casual lb'] =  (pinp.labour['min_casual_seedharv'] )
        else:
            labour_periods.loc[labour_periods['date']==i , 'casual lb'] = (pinp.labour['min_casual'] )
       
        ##determine cashflow period each labour period alines with
        labour_periods.loc[labour_periods['date']==i , 'cashflow'] = fun.period_allocation(p_dates,p_name,i)

    ##cost of casual for each labour period - wage plus super plus workers comp (multipled by wage because super and others are %)
    ##differect to perm and manager because they are at a fixed level throughout the year ie same number of perm staff all yr.
    labour_periods['casual_cost'] = labour_periods['casual hours'] * (uinp.price['casual_cost'] + uinp.price['casual_cost'] * uinp.price['casual_super'] + uinp.price['casual_cost'] * uinp.price['casual_workers_comp']) 
    ## drop last row, because it has na because it only contains the end date, therefore not a period
    labour_periods.drop(labour_periods.tail(1).index,inplace=True) 
    ##create dicts for pyomo
    params['permanent hours'] = labour_periods['permanent hours'].to_dict()
    params['permanent supervision'] = labour_periods['permanent supervision'].to_dict()
    params['casual_cost'] = dict(zip(enumerate(labour_periods['cashflow']),labour_periods['casual_cost']))
    r_vals['casual_cost'] = pd.Series(params['casual_cost'])
    params['casual hours'] = labour_periods['casual hours'].to_dict()
    params['casual supervision'] = labour_periods['casual supervision'].to_dict()
    params['manager hours'] = labour_periods['manager hours'].to_dict()
    params['casual ub'] = labour_periods['casual ub'].to_dict()
    params['casual lb'] = labour_periods['casual lb'].to_dict()
    r_vals['keys_p5'] = per.p_date2_df().index.astype('object')

# t_labour_periods=labour_general()



#permanent cost per cashflow period - wage plus super plus workers comp and leave ls (multipled by wage because super and others are %)
def perm_cost(params, r_vals):
    perm_cost = (uinp.price['permanent_cost'] + uinp.price['permanent_cost'] * uinp.price['permanent_super'] \
    + uinp.price['permanent_cost'] * uinp.price['permanent_workers_comp'] + uinp.price['permanent_cost'] * uinp.price['permanent_ls_leave']) / len(uinp.structure['cashflow_periods'])
    perm_cost=dict.fromkeys(uinp.structure['cashflow_periods'], perm_cost)
    params['perm_cost']=perm_cost
    r_vals['perm_cost']=pd.Series(perm_cost)


#manager cost per cashflow period
def manager_cost(params, r_vals):
    manager_cost = uinp.price['manager_cost'] / len(uinp.structure['cashflow_periods'])
    manager_cost=dict.fromkeys(uinp.structure['cashflow_periods'], manager_cost)
    params['manager_cost']=manager_cost
    r_vals['manager_cost']=pd.Series(manager_cost)











