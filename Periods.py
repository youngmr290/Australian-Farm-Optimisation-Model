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



#AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
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
    start = sinp.general['i_date_assetvalue']
    cashflow_period_dates.append(start)
    date = start
    #upper date of the current cashflow period (ie end of year)
    #upper = lower + relativedelta(days=(365 / len(inp.structure['cashflow_periods'])))
    while i < len(sinp.general['cashflow_periods']):
        cash_period_length = relativedelta(days=(365 / len(sinp.general['cashflow_periods'])))
        date += cash_period_length
        cashflow_period_dates.append(date)
        i += 1
    #made df this way so the columns could be diff len
    cashflow_dates = pd.DataFrame({'start date' : pd.Series(cashflow_period_dates),'cash period' : pd.Series(sinp.general['cashflow_periods'])})
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
    seeding_after_season_start_z = pinp.f_seasonal_inp(pinp.period['seeding_after_season_start'], numpy=True, axis=0)
    seeding_after_season_start_z = (seeding_after_season_start_z * 24).astype('timedelta64[h]')
    seeding_after_season_start_z = seeding_after_season_start_z.astype(datetime.datetime)
    # seeding_after_season_start_z = pd.to_timedelta(seeding_after_season_start_z,unit='D')
    ##wet seeding starts a specified number of days after season break
    return f_feed_periods()[0] +  seeding_after_season_start_z
    # return f_feed_periods().iloc[0].squeeze() +  datetime.timedelta(days = pinp.period['seeding_after_season_start'])


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


#This function determines the start dates of the labour periods. generally each period begins at the start of the month except seeding and harvest periods (which need to be separate because the labour force works more hours during those periods)
def p_dates_df():
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        ##put season inputs through season input function
        harv_date = pinp.f_seasonal_inp(pinp.period['harv_date'],numpy=True,axis=0)[0]
        seed_period_lengths = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'],numpy=True,axis=1)[...,0]
        harv_period_lengths = pinp.f_seasonal_inp(pinp.period['harv_period_lengths'],numpy=True,axis=1)[...,0]
        wet_seeding_start = wet_seeding_start_date()[0]

        ##calc period
        keys_z = pinp.f_keys_z()
        periods = pd.DataFrame(columns=keys_z)
        #create empty list of dates to be filled by this function
        period_start_dates = []
        #determine the start of the first period, this references feed periods so it has the same yr.
        start_date_period_0 = f_feed_periods()[0,0] + relativedelta(day=1,month=1,hour=0, minute=0, second=0, microsecond=0)
        #end date of all labour periods, simply one yr after start date.
        date_last_period = start_date_period_0 + relativedelta(years=1)
        #start point for the loop counter.
        date = start_date_period_0
        #loop that runs until the loop counter reached the end date.
        while date <= date_last_period:
            #if not a seed period then
            if date < wet_seeding_start or date > period_end_date(wet_seeding_start,seed_period_lengths):
                #if not a harvest period then just simply add 1 month and append that date to the list
                if date < harv_date or date > period_end_date(harv_date,harv_period_lengths):
                    period_start_dates.append(date)
                    date += relativedelta(months=sinp.general['labour_period_len'])
                #if harvest period then append the harvest dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
                else:
                    start = harv_date
                    length = harv_period_lengths
                    for i in range(len(period_dates(start, length))):
                        period_start_dates.append(period_dates(start, length)[i])
                    #end period can't be included in harvest date function above because then when that function is used to determine labour hours available in each period the period following harvest will also get more hours.
                    period_start_dates.append(period_end_date(start, length))
                    date = period_end_date(start, length) + relativedelta(months=sinp.general['labour_period_len']) + relativedelta(day=1)
            #if seed period then append the seed dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
            else:
                start = wet_seeding_start
                length = seed_period_lengths
                for i in range(len(period_dates(start, length))):
                    period_start_dates.append(period_dates(start, length)[i])
                period_start_dates.append(period_end_date(start, length))
                date = period_end_date(start, length) + relativedelta(months=sinp.general['labour_period_len']) + relativedelta(day=1)
        #add the list of dates to the labour dataframe
        periods[keys_z[0]]=period_start_dates
        ##modify index
        index = ['P%02d' % i for i in range(len(periods))]
        periods.index = index
    else:
        periods = pinp.period['i_dsp_lp']
        ##make df
        index = ['P%02d' % i for i in range(len(periods))]
        cols = pinp.general['i_z_idx'][1:] #need to slice off 'typical' becasue no labour period inputs for typical becasue it is automatically generated
        periods = pd.DataFrame(periods, index=index, columns=cols)
        ##apply season mask
        mask_z = pinp.general['i_mask_z'][1:] #need to slice off 'typical' becasue no labour period inputs for typical becasue it is automatically generated
        periods = periods.loc[:, mask_z]
    return periods

# drop last row, because it only contains the end date, this version of the df is used for creating the period set and when determining labour allocation
def p_date2_df():
    periods=p_dates_df()
    return periods.drop(periods.tail(1).index)

# print(p_date2_df())

###############
#feed periods #
###############
def f_feed_periods(option=0):
    '''
    :param option: int:
        0 = return feed period date
        1 = return feed period length (days)
    '''
    # idx = pd.IndexSlice
    # fp = pinp.period['i_dsp_fp']
    # fp = fp.T.set_index(['period'],append=True).T

    ## return array of fp dates
    if option==0:
        # fp = fp.loc[:, idx[:, 'date']]
        fp = pinp.period['i_dsp_fp_date']
        fp = pinp.f_seasonal_inp(fp, numpy=True, axis=1)
        return fp
    ## return length
    else:
        # fp = fp.loc[:fp.index[-2], idx[:, 'length']] #last row not included becasue that only contains the end date of last period
        fp = pinp.period['i_dsp_fp_len']
        fp = pinp.f_seasonal_inp(fp, numpy=True, axis=1)
        return fp

    #     if pinp.general['steady_state']:
    #         # fp = pinp.period['feed_periods']
    #
    #         n_fp = fp.values.astype(np.int64)
    #         np.average(n_fp, axis=1, weights=z_prob)
    #         n_fp = pd.to_datetime(n_fp.mean(axis=1))
    #         fp = pd.DataFrame(n_fp, index=fp.index, columns=fp.columns[0])
    #

    #
    #     else:
    #         fp = pinp.period['i_dsp_fp']
    #         fp = fp.T.set_index(['period'], append=True).T
    #         ##apply season mask - more complicated becasue masking levl 0 of multilevel df
    #         fp = fp.loc[:, idx[:, pinp.general['i_mask_z']]]

def f_fp_baseyr():
    ##feed period data - need to convert all dates to the same year
    fp_start_p6z  = f_feed_periods().astype('datetime64')
    base_year = fp_start_p6z[0,0].astype('datetime64[Y]').astype(int) + 1970
    condition_date = datetime.datetime(year=base_year+1, month=1, day=1)
    fp_start_p6z[fp_start_p6z>condition_date] = fp_start_p6z[fp_start_p6z>condition_date] - np.timedelta64(365, 'D')
    return fp_start_p6z


