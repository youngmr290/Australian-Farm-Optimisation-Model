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

#AFO modules
import UniversalInputs as uinp
import Functions as fun
import Periods as per
import PropertyInputs as pinp
import UniversalInputs as uinp
import Crop as crp
import Mach as mac
import StockFunctions as sfun

na = np.newaxis

########################
#off farm grain price  #
########################

def f_buy_grain_price(r_vals):
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
    price_df = crp.f_farmgate_grain_price()
    cartage=uinp.price['sup_cartage']
    transaction_fee=uinp.price['sup_transaction']
    price_df = price_df + cartage + transaction_fee
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    allocation = allocation.fillna(0)
    cols = pd.MultiIndex.from_product([allocation.index, price_df.columns])
    price_df = price_df.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    buy_grain_price = price_df.mul(allocation,axis=1,level=0)
    r_vals['buy_grain_price'] = buy_grain_price.T
    return buy_grain_price.stack([0,1])

def f_sup_cost(r_vals):
    #todo there could be a limitation here. We are assuming the silo is only filled once each year - the cost of the silo per tonne of sup is calculated based on the silos capacity, if the silo is fill multiple times this will overestimate the cost.
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
    grain_info.loc['asset'] =  silo_info.loc['asset'].div(grain_info.loc['capacity'], level=1)
    grain_info=grain_info.droplevel(1,axis=1) #drop silo type index

    ##data to determine cash period
    cashflow_df = per.cashflow_periods()
    p_dates = cashflow_df['start date']
    p_dates_c = p_dates.values #np version
    p_name_c = cashflow_df['cash period'].values[:-1]

    ##feed period data - need to convert all dates to the same year
    start_p6z = fun.f_baseyr(per.f_feed_periods())[:-1,:]
    length_p6z = per.f_feed_periods(option=1).astype('timedelta64[D]')

    ##deterimine cashflow allocation
    alloc_cpz=fun.range_allocation_np(p_dates_c[...,na], start_p6z, length_p6z, True)[:-1] #drop last c row because it is just the end date of last period.
    alloc_cpz = alloc_cpz.reshape(alloc_cpz.shape[0], -1)
    keys_z = pinp.f_keys_z()
    keys_p6 = pinp.period['i_fp_idx']
    cols = pd.MultiIndex.from_product([keys_p6, keys_z])
    alloc_cpz = pd.DataFrame(alloc_cpz, index=p_name_c, columns=cols)

    ##determine cost of feeding in each feed period and cashflow period
    feeding_cost_k = mac.sup_mach_cost()
    alloc_cpz = alloc_cpz.stack(0)
    cols = pd.MultiIndex.from_product([alloc_cpz.columns, feeding_cost_k.index])
    alloc_cpz = alloc_cpz.reindex(cols,axis=1,level=0)
    feeding_cost_cpzk = alloc_cpz.mul(feeding_cost_k, axis=1, level=1)
    start = np.datetime64(pinp.supfeed['storage_cost_date'])
    alloc_c = np.logical_and(p_dates_c[:-1]<=start, start<p_dates_c[1:])
    storage_cost_k = grain_info.loc['cost'].values
    storage_cost_ck = storage_cost_k * alloc_c[:,na]
    storage_cost_ck = pd.DataFrame(storage_cost_ck, index=p_name_c, columns=grain_info.columns)

    ##total cost = feeding cost plus storage cost
    feeding_cost_cpzk = feeding_cost_cpzk.stack().unstack(1)
    total_sup_cost = feeding_cost_cpzk.add(storage_cost_ck.stack().sort_index(),axis=0).stack(1)
    r_vals['total_sup_cost_ckp6_z'] = total_sup_cost

    ##dep
    storage_dep = grain_info.loc['dep']
    ##asset
    storage_asset = grain_info.loc['asset']
    ##return cost, dep and asset value
    return total_sup_cost, storage_dep, storage_asset
    
    
def f_sup_md_vol():
    ##calc vol
    sup_md_vol = uinp.supfeed['sup_md_vol']    
    ###convert md to dmd
    dmd=(sup_md_vol.loc['energy']/1000).apply(fun.md_to_dmd)
    ##calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        rq = sfun.f_rq_cs(dmd,0)
    ###use max(1,...) to make it the same as midas - this increases lupin vol slightly from what the equation returns
    vol_kg=np.maximum(1,1/rq)
    ###convert vol per kg to per tonne fed - have to adjust for the actual dry matter content and wastage
    vol_tonne=vol_kg*1000*sup_md_vol.loc['prop consumed']*sup_md_vol.loc['dry matter content']
    ##calc ME
    md_tonne=sup_md_vol.loc['energy']*sup_md_vol.loc['prop consumed']*sup_md_vol.loc['dry matter content']
    ##load into params dict for pyomo
    return vol_tonne, md_tonne
    
    
def f_sup_labour():
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
    lp_dates_p5z = per.p_dates_df()
    start_p8 = total_time.index.values
    end_p8 = np.roll(start_p8, -1)
    end_p8[-1] = end_p8[-1] + np.timedelta64(365, 'D') #increment the first date by 1yr so it becomes the end date for the last period
    len_p8 = end_p8 - start_p8
    shape_p5zp8 = lp_dates_p5z.shape + start_p8.shape
    alloc_p5zp8 = fun.range_allocation_np(lp_dates_p5z.values[...,na], start_p8, len_p8, shape=shape_p5zp8)[:-1]

    ##combine allocation with the labour time
    total_time_p8k = total_time.values
    total_time_p5zp8k = alloc_p5zp8[...,na] * total_time_p8k
    total_time_p5zk = np.sum(total_time_p5zp8k, axis=-2)

    ##link feed periods to labour periods, ie determine the proportion of each feed period in each labour period so the time taken to sup feed can be divided up accordingly
    start_p6z = fun.f_baseyr(per.f_feed_periods())[:-1,:]
    length_p6z = per.f_feed_periods(option=1).astype('timedelta64[D]')
    shape_p5p6z = (lp_dates_p5z.shape[0],) + length_p6z.shape
    alloc_p5p6z = fun.range_allocation_np(lp_dates_p5z.values[:,na,:], start_p6z, length_p6z, True, shape=shape_p5p6z)[:-1]

    ##allocate time to labour period for each feed period - get the time taken in each labour period to feed 1t of feed in each feed period
    total_time_p5p6zk = total_time_p5zk[:,na,...] * alloc_p5p6z[...,na]

    ##build df
    total_time_p5p6zk = total_time_p5p6zk.reshape(total_time_p5p6zk.shape[0],-1)
    keys_z = pinp.f_keys_z()
    keys_p6 = pinp.period['i_fp_idx']
    cols = pd.MultiIndex.from_product([keys_p6, keys_z, total_time.columns])
    total_time_p5zk = pd.DataFrame(total_time_p5p6zk, index=lp_dates_p5z.index[:-1], columns=cols).stack([0,2])
    return total_time_p5zk


    # ###get the time taken in each labour period to feed 1t of feed in each feed period
    # time_lab_period=time_lab_period.stack()
    # time_lab_feed_period=allocation.reindex(time_lab_period.index, level=1).mul(time_lab_period, axis=0)
    # params['sup_labour'] = time_lab_feed_period.stack().to_dict()
    

##collates all the params
def f_sup_params(params,r_vals):
    total_sup_cost, storage_dep, storage_asset = f_sup_cost(r_vals)
    vol_tonne, md_tonne = f_sup_md_vol()
    sup_labour = f_sup_labour()
    buy_grain_price = f_buy_grain_price(r_vals)


    ##create non seasonal params
    params['storage_dep'] = storage_dep.to_dict()
    params['storage_asset'] = storage_asset.to_dict()
    params['vol_tonne'] = vol_tonne.to_dict()
    params['md_tonne'] = md_tonne.to_dict()
    params['buy_grain_price'] = buy_grain_price.to_dict()

    ##create season params in loop
    keys_z = pinp.f_keys_z()
    for z in range(len(keys_z)):
        ##create season key for params dict
        scenario = keys_z[z]
        params[scenario] = {}
        params[scenario]['total_sup_cost'] = total_sup_cost[scenario].to_dict()
        params[scenario]['sup_labour'] = sup_labour[scenario].to_dict()

