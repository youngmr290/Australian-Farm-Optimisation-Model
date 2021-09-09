# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 15:25:08 2019

module: builds a df with period definitions
        this is one of the first modules calculated and therefore should only import input modules

        note: there must be a better way to do this!!

Version Control:
Version     Date        Person  Change
   1.1      25Dec19     John    altered the import statements to import as inp. & ci.

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
import Functions as fun

na = np.newaxis

#####################################
#define dates of cashflow periods   #
#####################################

def f_cashflow_periods(pandas=False):
    '''cashflow periods begin at the minimum of the break of season or the cashflow date.'''
    ##create c0 axis
    date_cashflow_stock = np.average(pinp.sheep['i_date_cashflow_stock_i'][i_mask_i])
    date_cashflow_crop = np.array([pinp.crop['i_date_cashflow_crop']])
    cashflow_date_c0 = np.concatenate([date_cashflow_stock, date_cashflow_crop])

    p_std_p6z = pinp.period['i_dsp_fp_date'].astype('datetime64')

    ###add node dates as feed peirods if dsp
    if pinp.general['i_inc_node_periods'] or np.logical_not(pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1):
        date_node_mz = pinp.general['i_date_node_zm'].astype('datetime64').T
        date_node_c0mz = date_node_mz + (np.timedelta64(365, 'D') * (date_node_mz < cashflow_date_c0[:,na,na]))
        breaks_of_season_b = np.unique(p_std_p6z[0,:]) #need all the different breaks
        breaks_of_season_b = breaks_of_season_b + (np.timedelta64(365, 'D') * (breaks_of_season_b < cashflow_date_c0[:,na,na])) #adjust the year
        cashflow_dates_c0p7z = np.concatenate([cashflow_date_c0[:,na,na], breaks_of_season_b[:,na], date_node_mz[1:]]) #[1:] becasue first node is break of season which already exists in fp array.
        cashflow_dates_c0p7z = np.sort(cashflow_dates_c0p7z, axis=1)
    else: #if nodes are not added then the adjusted fps are the same as the std fp.
        cashflow_date_c0p7z = cashflow_date_c0.astype('datetime64')[:,na,na]

    #make df
    if pandas:
        cashflow_date_c0p7z = pd.DataFrame(cashflow_date_c0p7z, index=, columns=)
    return cashflow_date_c0p7z


'''
labour periods and length
'''
################################
# make a df containing  period #
################################




#function to determine seeding start - starts a specified number of days after season break
#also used in mach sheet
def f_wet_seeding_start_date():
    seeding_after_season_start_z = pinp.f_seasonal_inp(pinp.period['seeding_after_season_start'], numpy=True, axis=0)
    seeding_after_season_start_z = (seeding_after_season_start_z * 24).astype('timedelta64[h]')
    seeding_after_season_start_z = seeding_after_season_start_z.astype(datetime.datetime)
    # seeding_after_season_start_z = pd.to_timedelta(seeding_after_season_start_z,unit='D')
    ##wet seeding starts a specified number of days after season break
    return f_feed_periods()[0] +  seeding_after_season_start_z
    # return f_feed_periods().iloc[0].squeeze() +  datetime.timedelta(days = pinp.period['seeding_after_season_start'])


#this function requires start date and length of each period (as a list) and spits out the start dates of each period
#used to determine harv and seed dates for period func below
def f_period_dates(start, length):
    #create empty list
    dates=[]
    perioddate = start
    #appends start date to list
    dates.append(perioddate)
    #loop used to append the rest of the seeding dates to list, doesnt include last seed period length because i only want start dates of seed periods
    for i in length[:-1]:
        perioddate += datetime.timedelta(days = i.astype(np.float64)) #for some reason the days must be a float64 otherwise you get an error (timedelta is seems only to be compatible with float64)
        dates.append(perioddate)
    return dates

#function to determine the end date of something (ie mach periods)
#also used in mach sheet
def f_period_end_date(start, length):
    #gets the last date from periods function then adds the length of last period
    return f_period_dates(start,length)[-1] + datetime.timedelta(days = length[-1].astype(np.float64))
#print(f_period_end_date(f_wet_seeding_start_date(),ci.crop_input['seed_period_lengths']))


#This function determines the start dates of the labour periods. generally each period begins at the start of the month except seeding and harvest periods (which need to be separate because the labour force works more hours during those periods)
def f_p_dates_df():
    if pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1:
        ##put season inputs through season input function
        harv_date = pinp.f_seasonal_inp(pinp.period['harv_date'],numpy=True,axis=0)[0]
        seed_period_lengths = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'],numpy=True,axis=1)[...,0]
        harv_period_lengths = pinp.f_seasonal_inp(pinp.period['harv_period_lengths'],numpy=True,axis=1)[...,0]
        wet_seeding_start = f_wet_seeding_start_date()[0]

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
            if date < wet_seeding_start or date > f_period_end_date(wet_seeding_start,seed_period_lengths):
                #if not a harvest period then just simply add 1 month and append that date to the list
                if date < harv_date or date > f_period_end_date(harv_date,harv_period_lengths):
                    period_start_dates.append(date)
                    date += relativedelta(months=sinp.general['labour_period_len'])
                #if harvest period then append the harvest dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
                else:
                    start = harv_date
                    length = harv_period_lengths
                    for i in range(len(f_period_dates(start, length))):
                        period_start_dates.append(f_period_dates(start, length)[i])
                    #end period can't be included in harvest date function above because then when that function is used to determine labour hours available in each period the period following harvest will also get more hours.
                    period_start_dates.append(f_period_end_date(start, length))
                    date = f_period_end_date(start, length) + relativedelta(months=sinp.general['labour_period_len']) + relativedelta(day=1)
            #if seed period then append the seed dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
            else:
                start = wet_seeding_start
                length = seed_period_lengths
                for i in range(len(f_period_dates(start, length))):
                    period_start_dates.append(f_period_dates(start, length)[i])
                period_start_dates.append(f_period_end_date(start, length))
                date = f_period_end_date(start, length) + relativedelta(months=sinp.general['labour_period_len']) + relativedelta(day=1)
        #add the list of dates to the labour dataframe
        periods[keys_z[0]]=period_start_dates
        ##modify index
        index = ['P%02d' % i for i in range(len(periods))]
        periods.index = index
    else:
        periods = pinp.period['i_dsp_lp']
        ##make df
        index = ['P%02d' % i for i in range(len(periods))]
        cols = pinp.general['i_z_idx'][1:] #need to slice off 'typical' because no labour period inputs for typical because it is automatically generated
        periods = pd.DataFrame(periods, index=index, columns=cols)
        ##apply season mask
        mask_z = pinp.general['i_mask_z'][1:] #need to slice off 'typical' because no labour period inputs for typical because it is automatically generated
        periods = periods.loc[:, mask_z]
    return periods

# drop last row, because it only contains the end date, this version of the df is used for creating the period set and when determining labour allocation
def f_p_date2_df():
    periods=f_p_dates_df()
    return periods.drop(periods.tail(1).index)

# print(f_p_date2_df())

###############
#feed periods #
###############
def f_feed_periods(option=0):
    '''
    :param option: int:
        0 = return feed period date
        1 = return feed period length (days)
        2 = return association between std feed periods and node adjusted feed periods
    '''
    ##calc feed period dates from inputs plus adjust for node dates.
    fp_std_p6z = pinp.period['i_dsp_fp_date'].astype('datetime64')

    ##adjust end date of the last period (needs to be the date of the latest break so that pasture season junction has the correct length of the final fp)
    #todo add this when doing season stuff.

    ###add node dates as feed peirods if dsp
    if pinp.general['i_inc_node_periods'] or np.logical_not(pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1):
        date_node_mz = pinp.general['i_date_node_zm'].astype('datetime64').T
        date_node_mz = date_node_mz + (np.timedelta64(365, 'D') * (date_node_mz < fp_std_p6z[0,:]))
        fp_p6z = np.concatenate([fp_std_p6z, date_node_mz[1:]]) #[1:] becasue first node is break of season which already exists in fp array.
        fp_p6z = np.sort(fp_p6z, axis=0)
    else: #if nodes are not added then the adjusted fps are the same as the std fp.
        fp_p6z = pinp.period['i_dsp_fp_date'].astype('datetime64')

    ###return association between fp inputs and fp after node adjustment (before handling z axis)
    if option==2:
        ###build association between fp inputs and fp after adjustment (for steady state this is simply 1:1 association)
        a_p6std_p6z = fun.searchsort_multiple_dim(fp_std_p6z, fp_p6z, 1, 1, side='right')-1
        return a_p6std_p6z[:-1,:] #drop the last period since that is just the end of the final fp (not a real period)

    ###handle z axis
    fp_p6z = pinp.f_seasonal_inp(fp_p6z, numpy=True, axis=1)

    ### return array of fp dates
    if option==0:
        return fp_p6z

    ### return length
    if option==1:
        fp_len = (fp_p6z[1:,:] - fp_p6z[:-1,:]).astype('timedelta64[D]')
        return fp_len


    # else:
    #     # fp = fp.loc[:fp.index[-2], idx[:, 'length']] #last row not included because that only contains the end date of last period
    #     fp = pinp.period['i_dsp_fp_len']
    #     fp = pinp.f_seasonal_inp(fp, numpy=True, axis=1)
    #     return fp

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
    #         ##apply season mask - more complicated because masking level 0 of multilevel df
    #         fp = fp.loc[:, idx[:, pinp.general['i_mask_z']]]



