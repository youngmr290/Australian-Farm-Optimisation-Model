# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:25:08 2019

module: builds a df with period definitions
        this is one of the first modules calculated and therefore should only import input modules

        note: there must be a better way to do this!!

Version Control:
Version     Date        Person  Change
   1.1      25Dec19     John    altered the import staements to import as inp. & ci.

Known problems:
Fixed   Date    ID by   Problem


@author: young
"""
#python modules
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta



#MUDAS modules
# from Inputs import *
# from CropInputs import *
import UniversalInputs as uinp
import PropertyInputs as pinp



#####################################
#define dates of cashflow periods   #
#####################################

def cashflow_periods():
    #index for cash periods
    i = 0
    #empty list
    cashflow_period_dates = []
    #start date of the current cashflow period (ie start of year)
    start = pinp.feed_inputs['feed_periods'].loc[1,'date'] + relativedelta(day=1,month=1)
    cashflow_period_dates.append(start)
    date = start
    #upper date of the current cashflow period (ie end of year)
    #upper = lower + relativedelta(days=(365 / len(inp.structure['cashflow_periods'])))
    while i < len(uinp.structure['cashflow_periods']):
        cash_period_length = relativedelta(days=(365 / len(uinp.structure['cashflow_periods'])))
        date += cash_period_length
        cashflow_period_dates.append(date)
        i += 1
    #made df this way so the columns could be diff len
    cashflow_dates = pd.DataFrame({'start date' : pd.Series(cashflow_period_dates),'cash period' : pd.Series(uinp.structure['cashflow_periods'])})
    return cashflow_dates


'''
labour periods and length
'''
################################
# make a df containing  period #
################################




#function to determine seeding start - starts a specified number of days after season break
#also used in mach sheet
def wet_seeding_start_date():
    #wet seeding starts a specified number of days after season break
    return pinp.feed_inputs['feed_periods'].loc[0,'date'] +  datetime.timedelta(days = pinp.crop['seeding_after_season_start'])


#this function requires start date and length of each period (as a list) and spits out the start dates of each period
#used to determine harv and seed dates for period func below
def period_dates(start, length):
    #create empty list
    dates=[]
    perioddate = start
    #appends start date to lisr
    dates.append(perioddate)
    #loop used to append the rest of the seeding dates to list, doesnt include last seed period length because i only want start dates of seed periods
    for i in length[:-1]:
        perioddate += datetime.timedelta(days = i.astype(np.float64)) #for some reason the days must be a float64 otherwise you get an error (timedelta is seems only to be compatible with float64)
        dates.append(perioddate)
    return dates

#function to determine the end date of something (ie mach periods)
#also used in mach sheet
def period_end_date(start, length):
    #gets the last date from periods funct then adds the length of last period
    return period_dates(start,length)[-1] + datetime.timedelta(days = length[-1].astype(np.float64))
#print(period_end_date(wet_seeding_start_date(),ci.crop_input['seed_period_lengths']))


#This function determines the start dates of the labour periods. generally each period begins at the start of the month except seeding and harvest periods (which need to be seperate because the labour force works more hours during those periods)
def p_dates_df():
    periods = pd.DataFrame(columns=['date'])
    #create empty list of dates to be filled by this function
    period_start_dates = []
    #determine the start of the first period, this references feed periods so it has the same yr.
    start_date_period_1 = pinp.feed_inputs['feed_periods'].loc[1,'date'] + relativedelta(day=1,month=1)
    #end date of all labour periods, simply one yr after start date.
    date_last_period = start_date_period_1 + relativedelta(years=1)
    #start point for the loop counter.
    date = start_date_period_1
    #loop that runs until the loop counter reached the end date.
    while date <= date_last_period:
        #if not a seed period then
        if date < wet_seeding_start_date() or date > period_end_date(wet_seeding_start_date(),pinp.crop['seed_period_lengths']):
            #if not a harvest period then just simply add 1 month and append that date to the list
            if date < pinp.crop['harv_date'] or date > period_end_date(pinp.crop['harv_date'],pinp.crop['harv_period_lengths']):
                period_start_dates.append(date)
                date += uinp.structure['labour_period_len']
            #if harvest period then append the harvest dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
            else:
                start = pinp.crop['harv_date']
                length = pinp.crop['harv_period_lengths']
                for i in range(len(period_dates(start, length))):
                    period_start_dates.append(period_dates(start, length)[i])
                #end period can't be included in harvest date function above because then when that function is used to determine labour hours available in each period the period following harvest will also get more hours.
                period_start_dates.append(period_end_date(start, length))
                date = period_end_date(start, length) + uinp.structure['labour_period_len'] + relativedelta(day=1)
        #if seed period then append the seed dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
        else:
            start = wet_seeding_start_date()
            length = pinp.crop['seed_period_lengths']
            for i in range(len(period_dates(start, length))):
                period_start_dates.append(period_dates(start, length)[i])
            period_start_dates.append(period_end_date(start, length))
            date = period_end_date(start, length) + uinp.structure['labour_period_len'] + relativedelta(day=1)
    #add the list of dates to the labour dataframe
    periods['date']=period_start_dates
    return periods

# drop last row, because it only contains the end date, this version of the df is used for creating the period set and when determining labour allocation
def p_date2_df():
    periods=p_dates_df()
    return periods.drop(periods.tail(1).index)


