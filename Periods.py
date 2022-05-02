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
import SeasonalFunctions as zfun
import Exceptions as exc

na = np.newaxis

#############################
#define dates of cashflow   #
#############################
def f_cashflow_date():
    '''cashflow date.'''
    ##create c0 axis
    cash_date = pinp.sheep['i_date_cashflow_stock_i'][pinp.sheep['i_mask_i']]
    date_cashflow_stock = cash_date.mean(keepdims=True).astype(int) #take mean in case multiple tol included
    date_cashflow_crop = np.array([pinp.crop['i_date_cashflow_crop']])
    cashflow_date_c0 = np.concatenate([date_cashflow_stock, date_cashflow_crop]) #have to stack this in the same order as the enterprise input in sinp.
    return cashflow_date_c0

def f_peak_debt_date():
    date_peakdebt_stock = pinp.sheep['i_date_peakdebt_stock_i'][pinp.sheep['i_mask_i']]
    date_peakdebt_stock = date_peakdebt_stock.mean(keepdims=True).astype(int) #take mean in case multiple tol included
    date_peakdebt_crop = np.array([pinp.crop['i_date_peakdebt_crop']])
    peakdebt_date_c0 = np.concatenate([date_peakdebt_stock,date_peakdebt_crop])
    return peakdebt_date_c0


################################
#labour periods and length     #
################################
#function to determine seeding start - starts a specified number of days after season break
#also used in mach sheet
def f_wet_seeding_start_date():
    seeding_after_season_start_z = zfun.f_seasonal_inp(pinp.period['seeding_after_season_start'], numpy=True, axis=0)
    seeding_after_season_start_z = seeding_after_season_start_z
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    # seeding_after_season_start_z = seeding_after_season_start_z.astype(datetime.datetime)
    # seeding_after_season_start_z = pd.to_timedelta(seeding_after_season_start_z,unit='D')
    ##wet seeding starts a specified number of days after season break
    return season_break_z +  seeding_after_season_start_z
    # return f_feed_periods().iloc[0].squeeze() +  datetime.timedelta(days = pinp.period['seeding_after_season_start'])

#this function requires start date and length of each period (as a list) and spits out the start dates of each period
#used to determine harv and seed dates for period func below
def f_period_dates(start, length):
    #create empty list
    dates=[]
    perioddate = start
    #appends start date to list
    dates.append(perioddate)
    #loop used to append the rest of the seeding dates to list, doesn't include last seed period length because i only want start dates of seed periods
    for i in length[:-1]:
        perioddate += i
        dates.append(perioddate)
    return dates

#function to determine the end date of something (ie mach periods)
#also used in mach sheet
def f_period_end_date(start, length):
    #gets the last date from periods function then adds the length of last period
    return f_period_dates(start,length)[-1] + length[-1]
#print(f_period_end_date(f_wet_seeding_start_date(),ci.crop_input['seed_period_lengths']))


def f_p_dates_df():
    periods = pinp.period['i_dsp_lp']
    ##make df
    index = ['P%02d' % i for i in range(len(periods))]
    cols = pinp.general['i_z_idx'] #need to slice off 'typical' because no labour period inputs for typical because it is automatically generated
    periods = pd.DataFrame(periods, index=index, columns=cols)
    ##apply season mask
    periods = zfun.f_seasonal_inp(periods, axis=1)

    ##seeding and harv periods
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'],numpy=True,axis=0)
    seed_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['seed_period_lengths'],numpy=True,axis=1)
    harv_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['harv_period_lengths'],numpy=True,axis=1)
    wet_seeding_start_z = f_wet_seeding_start_date()
    seeding_periods_pz = np.cumsum(np.concatenate([wet_seeding_start_z[na,:], seed_period_lengths_pz]), axis=0)
    harv_periods_pz = np.cumsum(np.concatenate([harv_start_z[na,:], harv_period_lengths_pz]), axis=0)
    seed_and_harv_periods_pz = np.concatenate([seeding_periods_pz,harv_periods_pz])

    ##For the weighted average steady state model the periods are adjusted so that seeding and harv peirods are a labour period
    if pinp.general['steady_state'] and np.count_nonzero(pinp.general['i_mask_z'])!=1:
        periods = periods.round(0) #round periods to nearest day
        ###make all lp at least 7 days
        for p in range(len(periods)):
            if p==0:
                pass
            else:
                periods.iloc[p] = max(periods.iloc[p].values, periods.iloc[p-1].values+7)
        ###make sure last period is 364 days after first
        periods.iloc[-1] = periods.iloc[0] + 364
        ###lookup each seeding and harv period and set the closest lp to that date
        for i in seed_and_harv_periods_pz.squeeze(): #squeeze to remove singleton z
            ###build a mask which is the closest period
            mask = (periods - i).abs() == (periods - i).abs().min()
            periods[mask] = i

        # ##calc period
        # keys_z = zfun.f_keys_z()
        # periods = pd.DataFrame(columns=keys_z)
        # #create empty list of dates to be filled by this function
        # period_start_dates = []
        # #determine the start of the first period, this references feed periods so it has the same yr.
        # start_date_period_0 = min(dry_seeding_start, pinp.general['i_date_node_zm'][0,0])
        # #end date of all labour periods, simply one yr after start date.
        # date_last_period = start_date_period_0 + 364
        # #start point for the loop counter.
        # date = start_date_period_0
        # #loop that runs until the loop counter reached the end date.
        # while date < date_last_period:
        #     #if not a seed period then
        #     if date < wet_seeding_start or date > f_period_end_date(wet_seeding_start,seed_period_lengths):
        #         #if not a harvest period then just simply add 1 month and append that date to the list
        #         if date < harv_date or date > f_period_end_date(harv_date,harv_period_lengths):
        #             period_start_dates.append(date)
        #             date += 30
        #         #if harvest period then append the harvest dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
        #         else:
        #             start = harv_date
        #             length = harv_period_lengths
        #             for i in range(len(f_period_dates(start, length))):
        #                 period_start_dates.append(f_period_dates(start, length)[i])
        #             #end period can't be included in harvest date function above because then when that function is used to determine labour hours available in each period the period following harvest will also get more hours.
        #             period_start_dates.append(f_period_end_date(start, length))
        #             date = f_period_end_date(start, length) + 33# - f_period_end_date(start, length)%30
        #     #if seed period then append the seed dates to the list and adjust the loop counter (date) to the start of the following time period (time period is determined by standard period length in the input sheet).
        #     else:
        #         # period_start_dates.append(dry_seeding_start) #add dry seeding period before wet seeding periods.
        #         start = wet_seeding_start
        #         length = seed_period_lengths
        #         for i in range(len(f_period_dates(start, length))):
        #             period_start_dates.append(f_period_dates(start, length)[i])
        #         period_start_dates.append(f_period_end_date(start, length))
        #         date = f_period_end_date(start, length) + 30 - f_period_end_date(start, length)%30
        # #add last period end date
        # period_start_dates.append(date_last_period)
        # #add the list of dates to the labour dataframe
        # periods[keys_z[0]]=period_start_dates
        # ##modify index
        # index = ['P%02d' % i for i in range(len(periods))]
        # periods.index = index

    ##if dsp check the nodes have been included
    if not pinp.general['steady_state'] and np.count_nonzero(pinp.general['i_mask_z']) != 1:
        ##error check: node dates must be included in the lab periods
        date_node_zm = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'], numpy=True, axis=0)
        if np.all(np.any(periods.values[:,:,na]==date_node_zm, axis=0)):
            pass
        else:
            raise exc.LabourPeriodError('''Season nodes are not all included in labour periods''')

    ##check that seeding and harv periods begin at the start of a labour period
    if np.all(np.any(periods.values[:,na,:]==seed_and_harv_periods_pz, axis=0)):
        pass
    else:
        raise exc.LabourPeriodError('''Seeding or harv periods do not begin/end at the start of a labour period.''')

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
    Feed periods are 364 days long (exactly 52 weeks). In the inputs each fp is rounded to a week then
    converted to a day of the year. This means all feed periods begin at the start of a week. This is done so that fp and
    DVPs start on the same day.

    :param option: int:
        0 = return feed period date
        1 = return feed period length (days)
        2 = return association between std feed periods and node adjusted feed periods
    '''
    ##calc feed period dates from inputs plus adjust for node dates.
    fp_std_p6z = pinp.period['i_dsp_fp_date']

    ###add node dates as feed periods if dsp
    if pinp.general['i_inc_node_periods'] or np.logical_not(pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z'])==1):
        date_node_mz = pinp.general['i_date_node_zm'].T
        date_node_mz = date_node_mz + 364 * (date_node_mz < fp_std_p6z[0,:])
        fp_p6z = np.concatenate([fp_std_p6z, date_node_mz])
        t_fp_p6z = zfun.f_seasonal_inp(fp_p6z, numpy=True, axis=1) #apply z mask so that duplication removing below only looks at the active seasons.
        ###remove duplicate periods
        duplicate_mask_p6 = []
        for p6 in range(t_fp_p6z.shape[0]):  # maybe there is a way to do this without a loop.
            duplicate_mask_p6.append(np.all(np.any(t_fp_p6z[p6,...] == t_fp_p6z[0:p6,...], axis=0, keepdims=True)))
        fp_p6z = fp_p6z[np.logical_not(duplicate_mask_p6)]
        fp_p6z = np.sort(fp_p6z, axis=0)
    else: #if nodes are not added then the adjusted fps are the same as the std fp.
        fp_p6z = pinp.period['i_dsp_fp_date']

    ###return association between fp inputs and fp after node adjustment (before handling z axis)
    if option==2:
        ###build association between fp inputs and fp after adjustment (for steady state this is simply 1:1 association)
        a_p6std_p6z = fun.searchsort_multiple_dim(fp_std_p6z, fp_p6z, 1, 1, side='right')-1
        return a_p6std_p6z[:-1,:] #drop the last period since that is just the end of the final fp (not a real period)

    ###handle z axis
    fp_p6z = zfun.f_seasonal_inp(fp_p6z, numpy=True, axis=1)

    ### return array of fp dates
    if option==0:
        return fp_p6z

    ### return length
    if option==1:
        fp_len = (fp_p6z[1:,:] - fp_p6z[:-1,:])
        return fp_len


#################
#season periods #
#################
def f_season_periods(keys=False):
    '''
    :param keys: Boolean if True this returns the m keys
    :param periods: Boolean if True this returns the m period dates
    '''
    date_node_zp7 = zfun.f_seasonal_inp(pinp.general['i_date_node_zm'],numpy=True,axis=0)
    ##if steady state then p7 axis is singleton (start and finish at the break of season).
    ## all node are included even in steady state model if user overwrites.
    if np.logical_not(pinp.general['i_inc_node_periods']) and (pinp.general['steady_state'] or np.count_nonzero(pinp.general['i_mask_z']) == 1):
        date_node_zp7 = date_node_zp7[:,0:1]
        ###add end date of last node period - required for the allocation function
        end_zp7 = date_node_zp7[:,0:1] + 364  # increment the first date by 1yr so it becomes the end date for the last period
        date_season_node_p7z = np.concatenate([date_node_zp7,end_zp7],axis=1).T  # put p7 in pos 0 because that how the allocation function requires
    ##if DSP then all season node included plus a node for dry seeding
    else:
        ###add end date of last node period - required for the allocation function
        end_zp7 = date_node_zp7[:,0:1] + 364  # increment the first date by 1yr so it becomes the end date for the last period
        date_season_node_p7z = np.concatenate([date_node_zp7,end_zp7],
                                            axis=1).T  # put p7 in pos 0 because that how the allocation function requires

    len_p7 = date_season_node_p7z.shape[0] - 1  # minus one because end date is not a period

    ##return keys if wanted
    if keys:
        keys_p7 = np.array(['zm%s' % i for i in range(len_p7)]) #this is zm because if it were p7 then it gets confusing once the period number is added e.g. p70 (p7[0])
        return keys_p7
    else:
        return date_season_node_p7z


