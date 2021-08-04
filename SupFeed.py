"""

author: young

Supplementary feeding is the supply of additional feed, primarily grain and hay, to livestock.
Supplementary feeding is commonly used to help meet liveweight targets during Summer and Autumn
months when pasture is limiting and enhance lamb diet for increase growth prior to sale. Additionally,
feeding supplement can be used as a tactic to allow pastures to be deferred. Grain and hay are the
primary supplements fed and hence represented in the model.

Grain and hay for supplementary feeding can either be grown on farm or purchased from another
farmer at farm gate price (net price of a product after selling costs have been subtracted)
plus the transaction and transport costs.

.. note:: Other grains can be added as supplements. Just remember to add their inputs in the mach
    sheet in universal.xlsx for each machine option and  the sup sheet in both universal.xlsx and property.xlsx.



"""
#python modules
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import relativedelta as rdelta

#AFO modules
import Functions as fun
import FeedsupplyFunctions as fsfun
import Periods as per
import PropertyInputs as pinp
import UniversalInputs as uinp
import Phase as phs
import Mach as mac
import Sensitivity as sen

na = np.newaxis

########################
#off farm grain price  #
########################

def f_buy_grain_price(r_vals):
    '''

    Cost to purchase a tonne of supplement off farm.

    Purchase price of grain off farm is slightly different to the selling price. The purchase price is
    equal to the price the selling farmer would receive had they sold to market (ie farm gate price)
    plus a transaction fee and transport cost.

    '''
    ##purchase price from neighbour is farm gate price plus transaction and transport
    price_df = phs.f_farmgate_grain_price()
    cartage=uinp.price['sup_cartage']
    transaction_fee=uinp.price['sup_transaction']
    price_df = price_df + cartage + transaction_fee
    ##calc farm gate grain price for each cashflow period - accounts for tolls and other fees
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
    '''

    Machinery, storage and depreciation costs incurred to feed 1t of supplement.

    Grain for supplementary feeding is generally stored in large on farm silos and hay in a hay shed.
    There are both variable cost and depreciation costs associated with the storage of supplementary
    feed. The variable cost represents the expenditure on insurance, silo preparation, insect management
    and grain shrinkage/loss. The depreciation represents the yearly reduction in the value of the silos
    and hay sheds.

    In the short term storage costs are fixed, that is, the farmer incurs the same variable and depreciation
    cost of storage independent of the amount of supplement fed. Additionally, the farmer is limited
    to only feed as much supplement as the storage capacity. However, AFO is built to evaluate
    the medium term where storage capacity can be varied. To account for this the cost
    of the storage is divided by the storage capacity returning the storage cost per tonne of supplement.
    This cost then applies to each tonne of supplement fed.

    The machinery cost to feed a tonne of supplement is added in this function however it
    is calculated in Mach.py (see Mach.py for details on machinery cost to feed supplement).

    '''

    #todo there could be a limitation here. We are assuming the silo is only filled once each year - the cost of the silo per tonne of sup is calculated based on the silos capacity, if the silo is fill multiple times this will overestimate the cost.
    ##calculate the insurance/dep/asset value per yr for the silos
    silo_info = pinp.supfeed['storage_type']
    silo_info.loc['dep'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/silo_info.loc['life']
    silo_info.loc['insurance'] = silo_info.loc['price'] * uinp.finance['equip_insurance']
    silo_info.loc['asset'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/2 #calculate the average value of the asset - used in the asset ROE constraint
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
    length_p6z = per.f_feed_periods(option=1)

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
    '''
    M/D and DM content of each supplement are known inputs.
    Unlike stubble and pasture, the quantity of supplementary feed consumed (the decision variables)
    are expressed including moisture content (i.e. as fed). Therefore M/D must be adjusted by the DM
    content of the feed.
    The volume of supplementary feed is calculated based on the quality of the feed. So, lower quality
    supplements (like oats) will substitute more for pasture than high quality supplements (like lupins).
    It is assumed that the availability of supplementary feed is high and that supplement is consumed as
    the first component of the animals diet. Furthermore, it is assumed that if sufficient levels of
    supplement are offered then all the sheep’s diet will be provided by supplementary feeding.
    This representation does not include an effect of high protein supplements (like lupins) overcoming
    a protein deficiency and therefore acting as a 'true' supplement and increasing intake. If this was represented it
    would likely make low rate lupin supplementation optimal in early summer/autumn to overcome a protein deficiency.
    '''
    #todo review the volume of supplementary feed, especially the max RI==1 and non-inclusion of protein supplementation.
    ##calc vol
    sup_md_vol = uinp.supfeed['sup_md_vol']    
    ###convert md to dmd
    dmd=(sup_md_vol.loc['energy'] / 1000).apply(fsfun.md_to_dmd)
    ##calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        rq = fsfun.f_rq_cs(dmd, 0)
    ###use max(1,...) to make it the same as MIDAS - this increases lupin vol slightly from what the equation returns
    ###do not calculate ra (relative availability) because assume that supplement has high availability
    vol_kg = np.maximum(1, 1 / rq)
    ###convert vol per kg to per tonne fed - have to adjust for the actual dry matter content and wastage
    vol_tonne = vol_kg * 1000 * sup_md_vol.loc['prop consumed'] * sup_md_vol.loc['dry matter content']
    vol_tonne = vol_tonne / (1 + sen.sap['pi'])
    ##calc ME (note: value in dict is MJ/t of DM, so doesn't need to be multiplied by 1000)
    md_tonne = sup_md_vol.loc['energy'] * sup_md_vol.loc['prop consumed'] * sup_md_vol.loc['dry matter content']
    ##load into params dict for pyomo
    return vol_tonne, md_tonne
    
    
def f_sup_labour():
    '''
    The labour required to feed sheep one tonne of supplement is calculated as the time spent
    traveling to and from the silo, filling the sheep feeder, emptying the feeder, and transporting
    between paddocks. The transport time to and from the silo and the time to fill up are inputs which
    are divided by the capacity of the feeder to return the hours per tonne. The rate of emptying
    the feeder at eight different times of the year are inputs which determine the time taken to feed
    a tonne of supplement at different times of the year. The time spent traveling between paddocks
    (not to the silo) is calculated based on the estimated feed rate per sheep (g/hd/d), the estimated
    number of sheep in each paddock and the average time taken to travel to the next paddock.

    To improve the accuracy between different grains, the time taken to fill and empty the feeder rate is
    calculated in cubic meter units (because a m3 is the same for all grain) and then converted to hr/tonne
    in the last step. However m3/hr is not an input many can relate to, so in the effort of making the model
    easier to calibrate the inputs are entered in a more common format for a specific grain and then converted
    to m3. For example the inputs used to determine time to empty the feeder are the feeding rate of lupins
    in kg/sec, this is then adjusted to m3/hr using the density of lupins.
    The time spent traveling between paddocks (not to the silo) is calculated slightly differently. Driving
    from one paddock to the next takes a given amount of time. This time is then allocated to each megajoule
    being fed. This time is associated with the energy fed because farmers have a target for how many megajoules
    to feed before going to the next paddock. Using energy also allows all grains to be compared on an equal
    playing field, because the target is the same for each feed.

    .. note:: The main limitations to this method are if the estimated rates of feeding are wrong compared
        to the model solution, it is a difficult limitation to avoid because the inputs are not aligned with
        the activities ie there is no activity that is the rate of feeding sheep so we have to make an estimated
        link between feeding rate and the total tonnes feed.

    '''
    ##time to fill up
    fill_df= pinp.supfeed['time_fill_feeder']
    fill_time = (fill_df.loc['drive time']+fill_df.loc['fill time'])/fill_df.loc['capacity']
    ##time to empty feeder
    empty_df=pinp.supfeed['empty_rate'].T.reset_index() #reindex so it can be combined with silo df
    empty_df=empty_df.set_index(['index','date']).T
    ##convert to hr/m3 for lupins and hr/bale for hay
    grain_density= uinp.supfeed['grain_density'].T.reset_index() #reindex so it can be combined with different grains
    grain_density=grain_density.set_index(['index','silo type']).squeeze()
    empty_df[('grain','empty rate lupins')]=1/(empty_df[('grain','empty rate lupins')]*60*60/1000/grain_density.loc['l','grain']) #convert from kg/sec lupins to hr/m3 (which is the same for all grains). First convert kg/sec to t/hr then divide by density
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
    ###convert to time per tonne - multiply time per mj by energy content of each grain.
    energy = uinp.supfeed['sup_md_vol'].loc['energy']
    time_mj = pd.concat([time_mj]*len(energy), keys=energy.index, axis=1)
    transport_tonne=time_mj.mul(energy,axis=1)
    ##add transport with filling and emptying
    total_time=transport_tonne+fill_empty_tonne
    ##determine time in each labour period
    ###determine the time taken to feed a tonne of feed in each labour period - this depends on the allocation of the labour periods into the entered sup feed dates
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
    length_p6z = per.f_feed_periods(option=1)
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

    ##create season params
    params['total_sup_cost'] = total_sup_cost.stack().to_dict()
    params['sup_labour'] = sup_labour.stack().to_dict()

