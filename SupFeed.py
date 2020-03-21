# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:12:20 2020

@author: young
"""
#python modules
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import relativedelta as rdelta

#midas modules
import UniversalInputs as uinp
import Functions as fun
import Periods as per
import PropertyInputs as pinp
import UniversalInputs as uinp
import Crop as crp
import Mach as mac
import FeedBudget as fdb



########################
#off farm grain price  #
########################

def buy_grain_price():
    '''
    Returns
    -------
    Dict.
        purchase price of grain from neighbour for sup feeding
        Price includes:
        -transaction
        -cartage cost
    '''
    ##purchase price from neighbour is farm gate price plus transaction and transport
    price_df = crp.farmgate_grain_price()
    cartage=uinp.price['sup_cartage']
    transaction_fee=uinp.price['sup_transaction']
    price_df = price_df + cartage + transaction_fee
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    cols = pd.MultiIndex.from_product([allocation.index, price_df.columns])
    price_df = price_df.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    return  price_df.mul(allocation,axis=1,level=0).stack([0,1])

def sup_cost():
    ##calculate the insurance/dep/asset value per yr for the silos
    silo_info = pinp.supfeed['storage_type']
    silo_info.loc['dep'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/silo_info.loc['life']
    silo_info.loc['insurance'] = silo_info.loc['price'] * uinp.finance['equip_insurance']
    silo_info.loc['asset'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/2 #calculate the average value of the asset - used in the asset ROE constrinate
    ##using the capacity of each silo for each grain determine the costs per tonne foe each grain
    grain_info=uinp.supfeed['grain_density'].T.reset_index() #reindex so it can be combined with silo df
    grain_info=grain_info.set_index(['index','silo type']).T
    grain_info.loc['capacity'] =  grain_info.loc['density'].mul(silo_info.loc['capacity'] , level=1)
    grain_info.loc['dep'] =  silo_info.loc['dep'].div(grain_info.loc['capacity'] , level=1)
    grain_info.loc['cost'] =  (silo_info.loc['insurance'] + silo_info.loc['other']).div(grain_info.loc['capacity'] , level=1) #variable cost = insurance + other (cleaning silo etc)
    grain_info.loc['asset'] =  silo_info.loc['asset'].div(grain_info.loc['capacity'] , level=1)
    ##drop silo type index
    grain_info=grain_info.droplevel(1,axis=1)
    ##data to determine cash period
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    ##determine cost of feeding in each feed period and cashflow period
    feeding_cost = mac.sup_mach_cost()
    start_df  = pinp.feed_inputs['feed_periods'].loc[:len(pinp.feed_inputs['feed_periods'])-2,'date']
    start_df  =start_df.apply(lambda x: x.replace(year=p_dates[0].year)) #this is required because feed period dates are split over two yrs which causes and error when trying to determine which cashflow period each feed period date falls into
    length_df = pinp.feed_inputs['feed_periods'].loc[:len(pinp.feed_inputs['feed_periods'])-2,'length'].astype('timedelta64[D]') 
    allocation=fun.period_allocation2(start_df, length_df, p_dates, p_name)
    cols = pd.MultiIndex.from_product([allocation.columns, feeding_cost.index])
    allocation = allocation.reindex(cols,axis=1,level=0)
    feeding_cost = allocation.mul(feeding_cost, axis=1, level=1)
    ##determine cost of storage in each feed period and cashflow period
    start = pinp.supfeed['storage_cost_date']
    allocation=fun.period_allocation(p_dates,p_name,start)
    storage_cost = grain_info.loc['cost']
    indx = pd.MultiIndex.from_product([[allocation],start_df.index, storage_cost.index])
    storage_cost = storage_cost.reindex(indx,axis=0,level=2)
    ##total cost = feeding cost plus storage cost
    total_sup_cost=feeding_cost.add(storage_cost.unstack([1,2]),axis=1, fill_value=0).stack([0,1]).to_dict()
    ##dep
    storage_dep = grain_info.loc['dep']
    indx = pd.MultiIndex.from_product([start_df.index, storage_dep.index])
    storage_dep = storage_dep.reindex(indx,axis=0,level=1).to_dict()
    ##asset
    storage_asset = grain_info.loc['asset']
    indx = pd.MultiIndex.from_product([start_df.index, storage_asset.index])
    storage_asset = storage_asset.reindex(indx,axis=0,level=1).to_dict()
    ##return cost, dep and asset value
    return total_sup_cost, storage_dep, storage_asset
    
def sup_md_vol():
    sup_md_vol = uinp.supfeed['sup_md_vol']    
    ##calc vol
    ###convert md to dmd
    dmd=(sup_md_vol.loc['energy']/1000+2)/17 #rearanged version of dmd to md formula in feed budget
    ###use max(1,...) to make it the same as midas - this increases lupin vol slightly from what the equation returns
    vol_kg=np.maximum(1,1/fdb.ri_quality(dmd,0))
    ###convert vol per kg to per tonne fed - have to adjust for the actual dry matter content and wastage
    vol_tonne=vol_kg*1000*sup_md_vol.loc['prop consumed']/100*sup_md_vol.loc['dry matter content']/100
    ##calc ME
    md_tonne=sup_md_vol.loc['energy']*sup_md_vol.loc['prop consumed']/100*sup_md_vol.loc['dry matter content']/100
    return vol_tonne.to_dict(),md_tonne.to_dict()
    
def sup_labour():
    ##time to fill up
    fill_df= pinp.supfeed['time_fill_feeder']
    fill_time = (fill_df.loc['drive time']+fill_df.loc['fill time'])/fill_df.loc['capacity']
    ##time to empty feeder
    empty_df=pinp.supfeed['empty_rate'].T.reset_index() #reindex so it can be combined with silo df
    empty_df=empty_df.set_index(['index','date']).T
    ##convert to hr/m3 for lupins and hr/bale for hay
    grain_density= uinp.supfeed['grain_density'].T.reset_index() #reindex so it can be combined with different grains
    grain_density=grain_density.set_index(['index','silo type']).squeeze()
    empty_df[('grain','empty rate lupins')]=1/(empty_df[('grain','empty rate lupins')]*60*60/1000/grain_density.loc['l','grain']) #convert from kg/sec lupins to hr/m3 (which is the same for all grains). First convert kg/sec to t/hr then divid by density
    empty_df[('hay','empty rate')]=empty_df[('hay','empty rate')]/60 #convert min/bale to hr/bale
    ##combine time to fill and empty then convert to per tonne for each grain
    empty_df=empty_df.droplevel(1, axis=1)
    fill_empty = empty_df.add(fill_time, axis=1)
    fill_empty_tonne=fill_empty.reindex(grain_density.index, axis=1, level=1).div(grain_density).droplevel(1,axis=1)
    ##calc time between paddocks
    ###convert lupin rate fed to mj/hd/d
    feedrate=pinp.supfeed['feed_rate']
    mj=feedrate['feed rate']/1000000*uinp.supfeed['sup_md_vol'].loc['energy', 'l'] #divide by 1000000 because convert g to tonnes because energy is in mj/tonne
    ###determine how many mj are feed to each paddock each time feeding occurs ie total mj per week divided by frequency of feeding per week
    mj_mob_per_trip = mj * feedrate['mob size'] * 7 / pinp.supfeed['feed_freq']
    ###time per mj. this is just the time to drive between two paddocks divided by the mj fed
    time_mj=pinp.supfeed['time_between_pad']/mj_mob_per_trip
    ###convert to time per tonne - multile time per mj by energy content of each grain.
    energy = uinp.supfeed['sup_md_vol'].loc['energy']
    time_mj = pd.concat([time_mj]*len(energy), keys=energy.index, axis=1)
    transport_tonne=time_mj.mul(energy,axis=1)
    ##add transport with filling and emptying
    total_time=transport_tonne+fill_empty_tonne
    ##determine time in each labour period
    ###determine the time taken to feed a tonne of feed in each labour perior - this depends on the allocation of the labour periods into the entered sup feed dates
    p_dates=list(total_time.index)
    p_dates.append(total_time.index[0] + rdelta.relativedelta(years=1))
    p_name = p_dates #use the dates as index so i can have a similar inex to the time df
    start_df = per.p_date2_df()['date']
    date_full=list(per.p_dates_df()['date'])
    length_df =  pd.DataFrame([date_full[i+1]-date_full[i] for i in range(len(date_full)-1)])
    allocation=fun.period_allocation2(start_df, length_df, p_dates, p_name)
    ###mul allocation by the actual time taken and sum for each labour period ie if a labour period is split between two sup feed periods the time taken to feed in that given labour period is a comnination of the time taken in each sup period
    total_time=total_time.stack()
    time_lab_period= allocation.reindex(total_time.index, level=0).mul(total_time, axis=0).sum(axis=0, level=1)
    ##link feed periods to labour periods, ie determine the proportion of each feed period in each labour period so the time taken to sup feed can be divided up accordingly
    p_dates = per.p_dates_df()['date']
    p_name = per.p_dates_df().index
    fp=pinp.feed_inputs['feed_periods'].iloc[:-1]  #don't want the end date of the last period included
    start_df = fp['date'].apply(lambda x: x.replace(year=p_dates[0].year))
    length_df =  fp['length'].astype('timedelta64[D]') 
    allocation=fun.period_allocation2(start_df, length_df, p_dates, p_name)
    ###get the time taken in each labour period to feed 1t of feed in each feed period
    time_lab_period=time_lab_period.stack()
    time_lab_feed_period=allocation.reindex(time_lab_period.index, level=1).mul(time_lab_period, axis=0)
    return time_lab_feed_period.stack().to_dict()
    


