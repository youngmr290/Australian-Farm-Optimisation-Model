# -*- coding: utf-8 -*-
"""

author: young

The functions in this module calculate the rate at which machinery can be used, the cost incurred
and the timeliness of certain jobs.

The model has been designed to accommodate a range of machine options. Inputs that are dependent
on machine size are grouped into their own input sheet, this section can be duplicated as many times
as necessary to represent different machine sizes and or amount of machinery. The model user can
then select which machine option they want in the analysis.
The user inputs the desired machinery option in property input section. If the user wants
to test the impact of different machine options they can alter the input using a saa sensitivity variable.

All machine option inputs (located in Universal.xlsx) are calibrated to represent cropping on a standard LMU.
In GSM it is the loamy sand, with granite outcropping where Wandoo (white gum) is a common tree.
To allow for the differences between the calibration and the actual LMUs there are LMU adjustment
factors in Property.xlsx.

To reduce space and complexity the model currently only represents machinery activities for seeding and
harvest (all other machinery usage is directly converted to a cost linked with another activity).
Therefore the hours of machinery use outside of seeding and harvest are not tracked. Hence
machinery use outside of seeding and harvest does not incur any variable
depreciation cost. To minimise this error, the rate of variable depreciation can be increased slightly.

Crops and pasture can be sown dry (i.e. seeding occurs before the opening rains that trigger the commencement
of the growing season) or wet (seeding occurs after the opening rains) :cite:p:`RN119`. Dry seeding can
be useful in weather-years with a late start to the growing season, maximising the growing season and water
utilisation because the crop can germinate as soon as the first rains come. Dry seeding begins on
a date specified by the user and can be performed until the opening rains. Wet seeding occurs a set number
of days after the opening rains to allow time for weed germination and application of a knockdown spray.
Wet seeding can only occur during a proportion of the time because some of
the time it is either too dry or too wet to seed. The timeliness of sowing can have a large impact on
crop yields AFO accounts for this using a yield penalty (see yield penalty function for more info).

AFO also represents the ability to hire contract workers to complete harvest and seeding. This can be
useful for large operations, farms with poor machinery or farms with poor labour availability.

"""

#python modules
import pandas as pd
import datetime as dt
import numpy as np



#AFO modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StructuralInputs as sinp
import Periods as per
import Functions as fun
import SeasonalFunctions as zfun
import Finance as fin
import RotationPhases as rps

na = np.newaxis

################
#fuel price    #
################
#fuel price = price - rebate
def fuel_price():
    '''Price of fuel including rebate.'''
    return uinp.price['diesel'] - uinp.price['diesel_rebate']



#######################################################################################################################################################
#######################################################################################################################################################
#feeding supplement
#######################################################################################################################################################
#######################################################################################################################################################
def sup_mach_cost():
    '''
    
    Cost of machinery to feed 1t of each grain to sheep.

    The task of feeding the supplement has a cost reflecting the fuel and machine repairs and maintenance.
    This is a variable cost incurred for each tonne of supplement fed. The cost is calculated simply from
    inputs which state the litres required and repairs and maintenance cost to feed 1 tonne of each supplement.

    .. note:: This method of calculating the cost is satisfactory but it could be improved by linking
        the cost of feeding to the distance travelled which is controlled by the rate of feeding (g/hd),
        sheep numbers and the stocking rate (ie lower stocking rate means sheep are spread over a larger area).
        Rate of feeding could be calculated from the tonnes for feed fed (activity), the frequency of feeding
        (input) and the number of sheep (activity). Then somehow adjusted for the spread of sheep over the
        property using the stocking rate. Or maybe it could be calculated more like the labour required.

    '''
    sup_cost=uinp.mach[pinp.mach['option']]['sup_feed'].copy() #need this so it doesnt alter inputs
    ##add fuel cost
    sup_cost['litres']=sup_cost['litres'] * fuel_price()
    ##sum the cost of r&m and fuel - note that rm is in the sup_cost df
    return sup_cost.sum(axis=1)

#######################################################################################################################################################
#######################################################################################################################################################
#seeding
#######################################################################################################################################################
#######################################################################################################################################################

#######################
#seed days per period #
#######################
##create a copy of periods df - so it doesn't alter the original period df that is used for labour stuff
# mach_periods = per.f_p_dates_df()#periods.copy()

def f_seed_days():
    '''
    Determines the number of wet and dry seeding days in each period.
    '''
    dry_seed_start = np.datetime64(pinp.crop['dry_seed_start'])
    mach_periods = per.f_p_dates_df()
    start_pz = np.maximum(dry_seed_start, mach_periods.values[:-1])
    end_pz = mach_periods.values[1:]
    length_pz = np.maximum(0,(end_pz - start_pz).astype('timedelta64[D]').astype(int))
    days = pd.DataFrame(length_pz, index=mach_periods.index[:-1], columns=mach_periods.columns)
    return days

def f_contractseeding_occurs():
    '''
    This function just sets the period when contract seeding must occur (period when wet seeding begins).
    Contract seeding is not hooked up to yield penalty because if your going to hire someone you will hire
    them at the optimum time. Contract seeding is hooked up to poc so this param stops the model having late seeding.
    '''
    contract_start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
    mach_periods = per.f_p_dates_df()
    start_pz = mach_periods.values[:-1]
    end_pz = mach_periods.values[1:]
    contractseeding_occur_pz = np.logical_and(start_pz <= contract_start_z, contract_start_z < end_pz)
    contractseeding_occur_pz = pd.DataFrame(contractseeding_occur_pz,index=mach_periods.index[:-1],columns=mach_periods.columns)
    return contractseeding_occur_pz
    # params['contractseeding_occur'] = (mach_periods==contract_start).squeeze().to_dict()


def f_poc_grazing_days():
    '''
    Grazing days provided by wet seeding activity (days/ha sown in each mach period/feed period).

    This section represents the grazing achieved on crop paddocks before seeding. The longer seeding is
    delayed the more grazing is achieved. The calculations also account for a gap between when pasture
    germinates (the pasture break) and when seeding can begin (the seeding break). Dry seeding doesn't
    provide any grazing, so the calculations are only done using the wet seeding days. The calculations
    allow for destocking a certain number of days before seeding (termed the defer period) to allow the pasture
    leaf area to increase so that the knock down spray is effective.

    Grazing days is the sum of, the area grazed multiplied by the number of days of grazing. As such it is
    dependent on both the area that is being grazed and the duration of grazing. The number of grazing days
    is calculated for each seeding decision variable in each machinery period (p5). However, the number
    of grazing days must be calculated for each feed period (p6) so that the
    feed supply will align with the feed demand of the animals (that are calculated for feed periods).

    The grazing days associated with the seeding decision variable (per hectare sown) is made up of two parts:
    A 'rectangular' component which represents the area being grazed each day from the break of season up
    until the destocking date for the beginning of the machine period.
    A 'triangle' component which represents the grazing during the seeding period. The area grazed each day
    diminishes associated with destocking the area that is soon to be sown. For example at the start
    of the period all area can be grazed but by the end of the period once all the area has been destocked
    no grazing can occur. These 2 components must then be allocated to the feed periods.

    'Rectangular' component: The base of the rectangle is defined by the later of the break and the
    start of the feed period through to the earlier of the end of the feed period or the start of
    the machine period.

    Triangular component: The base of the triangle starts at the latter of the start of the feed period,
    the break of season and the start of the machinery period minus the defer period. It ends at the
    earlier of the end of the feed period and the end of the machinery period minus the defer period.
    This triangular component is then allocated to the feed periods and this is calculated using a height
    at the start and a height at the end to calculate an average height. The average 'height' is the average
    area grazed per day for that feed period, when multiplied by the number of days is the number of grazing
    days for that feed period.
    The height at the start is 100% of the area to be sown reduced by (1 / the length of the machinery period)
    for each day after the start of the machinery period. This reduction is to represent that the definition of the
    decision variable is 1 hectare sown, spread evenly over the seeding period, so if the seeding period is 10 days
    long one tenth of the area is sown each day. The end height is the area per day multiplied by the number of
    days prior to the end of the machinery period.

    The above returns the number of grazing days provided by ‘1ha of seeding in each machine period spread across
    the duration of the machine period’, which can then just be multiplied by the rate of seeding and the number
    of days seeded in each period to get the total number of grazing days. This last step happens in pyomo.

    The assumption is that; seeding is done evenly throughout a given period. In reality this is wrong eg if a
    period is 5 days long but the farmer only has to sow 20ha they will do it on the first day of the period not
    4ha each day of the period. Therefore, the calculation slightly overestimates the amount of grazing achieved.

    See poc section in google doc for diagram.
    '''
    ##inputs
    date_feed_periods = per.f_feed_periods()
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    date_end_p5z = mach_periods.values[1:]
    seed_days_p5z = f_seed_days().values
    defer_period = np.array([pinp.crop['poc_destock']]).astype('timedelta64[D]') #days between seeding and destocking
    season_break_z = date_start_p6z[0]

    ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
    base_p6p5z = (np.minimum(date_end_p6z[:,na,:], date_start_p5z - defer_period) \
                  - np.maximum(season_break_z, date_start_p6z[:,na,:]))/ np.timedelta64(1, 'D')
    height_p5z = 1
    poc_grazing_days_rect_p6p5z = np.maximum(0, base_p6p5z * height_p5z)

    ##grazing days triangular component (for p5) and allocation to feed periods (p6)
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(season_break_z, date_start_p5z - defer_period))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z - defer_period)
    base_p6p5z = (end_p6p5z - start_p6p5z)/ np.timedelta64(1, 'D')
    height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z - defer_period) - start_p6p5z)/ np.timedelta64(1, 'D')
                                                    , seed_days_p5z))
    height_end_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z - defer_period) - end_p6p5z)/ np.timedelta64(1, 'D')
                                                    , seed_days_p5z))
    poc_grazing_days_tri_p6p5z = np.maximum(0,base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

    ##total grazing days & convert to df
    total_poc_grazing_days_p6p5z = poc_grazing_days_tri_p6p5z + poc_grazing_days_rect_p6p5z

    total_poc_grazing_days_p6p5z = total_poc_grazing_days_p6p5z.reshape(total_poc_grazing_days_p6p5z.shape[0], -1)
    keys_z = zfun.f_keys_z()
    cols = pd.MultiIndex.from_product([mach_periods.index[:-1], keys_z])
    total_poc_grazing_days = pd.DataFrame(total_poc_grazing_days_p6p5z, index=pinp.period['i_fp_idx'], columns=cols)
    return total_poc_grazing_days.stack(0)

#################################################
#seeding ha/day for each crop on each lmu  type #
#################################################

def f_seed_time_lmus():
    '''
    Time taken to direct drill 1ha of the base crop on each lmu.

    Seeding rate is calculated from speed travelled, seeder width and seeding efficiency (efficiency is
    to account for overlap, turning around and filling up time). Seeding rate can vary for each LMU
    due to varying speeds associated with more or less force being required to pull the seeding equipment
    through the soil. The rate inputs are set for the base LMU and then adjusted by a
    user defined LMU factor.
    '''
    ##mask lmu input
    lmu_mask = pinp.general['i_lmu_area'] > 0
    seeder_speed_lmu_adj = pinp.mach['seeder_speed_lmu_adj'][lmu_mask]
    ##first turn seeding speed on each lmu to df so it can be manipulated (it was entered as a dict in machinputs)
    speed_lmu_df = seeder_speed_lmu_adj * uinp.mach[pinp.mach['option']]['seeder_speed_base']
    ##convert speed to rate of direct drill for wheat on each lmu type (hr/ha)
    rate_direct_drill = 1 / (speed_lmu_df * uinp.mach[pinp.mach['option']]['seeding_eff'] * uinp.mach[pinp.mach['option']]['seeder_width'] / 10)
    return rate_direct_drill

def f_overall_seed_rate(r_vals):
    '''
    Hectares of each crop that can be sown per day on each LMU.

    Seeding rate per day is a product of the hours seeding can occur per day and the seeding rate
    per hectare. Seeding rate per hectare for each crop on each LMU is calculated by adjusting the
    LMU seeding rate (see f_seed_time_lmus) by a crop adjustment factor.

    '''
    #convert seed time (hr/ha) to rate of direct drill per day (ha/day)
    seed_rate_lmus = 1 / f_seed_time_lmus().squeeze() * pinp.mach['daily_seed_hours']
    #adjusts the seeding rate (ha/day) for each different crop depending on its seeding speed vs wheat

    seedrate_df = pd.concat([uinp.mach[pinp.mach['option']]['seeder_speed_crop_adj']]*len(seed_rate_lmus),axis=1) #expands df for each lmu
    seedrate_df.columns = seed_rate_lmus.index #rename columns to lmu so i can mul
    seedrate_df=seedrate_df.mul(seed_rate_lmus)
    r_vals['seeding_rate'] = seedrate_df
    return seedrate_df.stack()

    
  

#################################################
#seeding 1 ha  cost                             #
#################################################
##the cost of seeding 1ha is the same for all crops, but the seeding rate (ha/day) may differ from crop to crop
##cost of seeding 1ha is dependant on the lmu


def fuel_use_seeding():
    '''
    Fuel use L/ha used by tractor to seed on each lmu.
    '''
    ##mask lmu input
    lmu_mask = pinp.general['i_lmu_area'] > 0
    seeding_fuel_lmu_adj = pinp.mach['seeding_fuel_lmu_adj'][lmu_mask]
    ##determine fuel use on base lmu (draft x tractor factor)
    base_lmu_seeding_fuel = uinp.mach[pinp.mach['option']]['draft_seeding'] * uinp.mach[pinp.mach['option']]['fuel_adj_tractor']
    ##determine fuel use on all soils by adjusting s5 fuel use with input adjustment factors
    df_seeding_fuel_lmu = base_lmu_seeding_fuel * seeding_fuel_lmu_adj
    #second multiply base cost by adj, to produce df with seeding fuel use for each lmu (L/ha)
    return df_seeding_fuel_lmu 
    
def tractor_cost_seeding():
    '''
    Cost of running tractor for seeding.

    Tractor cost includes fuel, oil, grease and r&m. Oil, grease repairs and maintenance are calculated
    as a factor of fuel cost (see fuel used function for calculation of fuel used for seeding).

    '''
    fuel_used= fuel_use_seeding() 
    ##Tractor r&m during seeding for each lmu
    r_m_cost = fuel_used * fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    ##determine the $ cost of tractor fuel 
    tractor_fuel_cost = fuel_used * fuel_price()
    ##determine the $ cost of tractor oil and grease for seeding
    tractor_oil_cost = fuel_used * fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    return r_m_cost + tractor_fuel_cost + tractor_oil_cost


def maint_cost_seeder():
    '''
    Cost to repair and maintain air seeder ($/ha).

    The seeder cost can vary depending on the LMU because different soil types wear out the cropping
    gear at different rates. The cost inputs are set for the base LMU and then adjusted by a
    user defined LMU factor.
    '''
    ##mask lmu input
    lmu_mask = pinp.general['i_lmu_area'] > 0
    tillage_maint_lmu_adj = pinp.mach['tillage_maint_lmu_adj'][lmu_mask]
    ##equals r&m on base lmu x lmu adj factor
    tillage_lmu_df = uinp.mach[pinp.mach['option']]['tillage_maint'] * tillage_maint_lmu_adj
    return  tillage_lmu_df

def f1_seed_cost_alloc():
    '''
    labour period allocation for seeding costs.

    All seeding costs for a seeding activity must be incurred in the current season node eg if seeding happens in node 1
    the costs must be incurred in node 1, to ensure no seasons get free seeding. This happens by default because labour
    periods include season nodes.
    '''
    ##inputs
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    p5_start_zp5 = per.f_p_dates_df().values[:-1].T #.T because needs to match other shapes in cash function call below

    ##calc interest and allocate to cash period - needs to be numpy
    seeding_cost_allocation_c0p7zp5, seeding_wc_allocation_c0p7zp5 = fin.f_cashflow_allocation(p5_start_zp5, enterprise='crp', z_pos=-2)
    ###convert to df
    new_index_c0p7zp5 = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5])
    seeding_cost_allocation_c0p7zp5 = pd.Series(seeding_cost_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    seeding_wc_allocation_c0p7zp5 = pd.Series(seeding_wc_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    return seeding_cost_allocation_c0p7zp5, seeding_wc_allocation_c0p7zp5


def f1_seeding_phaseperiod_allocation():
    '''Allocation of seeding costs into phase periods'''

    fert_info = pinp.crop['fert_info']
    fert_date_n = fert_info['app_date'].values
    fert_length_n = fert_info['app_len'].values.astype('timedelta64[D]')
    alloc_mzn = rps.f1_rot_period_alloc(fert_date_n[na,na,:], fert_length_n[na,na,:], z_pos=-2)
    ###convert to df
    keys_z = zfun.f_keys_z()
    keys_m = per.f_phase_periods(keys=True)
    new_index_mzn = pd.MultiIndex.from_product([keys_m, keys_z, fert_info.index])
    alloc_mzn = pd.Series(alloc_mzn.ravel(), index=new_index_mzn)
    return alloc_mzn


def f_seeding_cost(r_vals):
    '''
    Combines all the machinery costs required to seed 1 hectare of each crop and allocates the cost
    to a cashflow period.

    Total machinery cost of seeding includes tractor costs (see tractor_cost_seeding) and
    seeder maintenance (see maint_cost_seeder).
    '''
    ##Total cost seeding on each lmu $/ha.
    seeding_cost_l = tractor_cost_seeding() + maint_cost_seeder()
    seeding_cost_l = seeding_cost_l.squeeze()

    ##gets the cost allocation (includes interest)
    seeding_cost_allocation_c0p7zp5, seeding_wc_allocation_c0p7zp5 = f1_seed_cost_alloc()

    ##reindex with lmu so alloc can be mul with seeding_cost_l
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, seeding_cost_l.index])
    seeding_cost_allocation_c0p7zp5l = seeding_cost_allocation_c0p7zp5.reindex(columns)
    seeding_wc_allocation_c0p7zp5l = seeding_wc_allocation_c0p7zp5.reindex(columns)

    ##mul costs and allocation
    seeding_cost_c0p7zp5l = seeding_cost_allocation_c0p7zp5l.mul(seeding_cost_l, level=4)
    seeding_wc_c0p7zp5l = seeding_wc_allocation_c0p7zp5l.mul(seeding_cost_l, level=4)
    r_vals['seeding_cost'] = seeding_cost_c0p7zp5l
    return seeding_cost_c0p7zp5l, seeding_wc_c0p7zp5l

def f_contract_seed_cost(r_vals):
    '''
    Contract seeding cost in each cashflow period. Currently, contract cost is the same for all lmus and crops.
    '''
    ##gets the cost allocation (includes interest)
    seeding_cost_allocation_c0p7zp5, seeding_wc_allocation_c0p7zp5 = f1_seed_cost_alloc()

    ##cost to contract seed 1ha
    seed_cost = uinp.price['contract_seed_cost']
    contract_seeding_cost_c0p7zp5 = seeding_cost_allocation_c0p7zp5 * seed_cost
    contract_seeding_wc_c0p7zp5 = seeding_wc_allocation_c0p7zp5 * seed_cost
    r_vals['contractseed_cost'] = contract_seeding_cost_c0p7zp5
    return contract_seeding_cost_c0p7zp5, contract_seeding_wc_c0p7zp5

########################################
#late seeding & dry seeding penalty    #
########################################

def f_sowing_timeliness_penalty(stub=False):
    '''
    Calculates the yield penalty in each mach period due to wet sowing timeliness- kg/ha/period/crop.

    The timeliness of sowing can have a large impact on crop yields. AFO accounts for this using a
    yield penalty.

    Late sowing receives a yield reduction because the crop has less time to mature (e.g. shorter
    growing season) and grain filling often occurs during hotter drier conditions :cite:p:`RN121, RN122`.
    The user can specify the length of time after the beginning of wet seeding that no penalty applies
    after that a penalty is applied. The yield reduction is cumulative per day, so the longer sowing is
    delayed the larger the yield reduction.

    Yield penalty reduces grain available to sell and reduces stubble production.

    The assumption is that seeding is done evenly throughout a given period. In reality this is wrong eg if a
    period is 5 days long but the farmer only has to sow 20ha they will do it on the first day of the period not
    4ha each day of the period. Therefore, the calculation overestimates the yield penalty.

    .. note:: There are also risks associated with dry sowing such as less effective weed control (i.e. crops germinate at
        the same time as the weeds so you miss out on a knock down spray opportunity), poor crop emergence (if
        opening rains are spasmodic patchy crop germination is possible and early crop vigour may be absent without
        adequate follow up rain) and increased chance of frost :cite:p:`RN119`. These risks are represented
        in the model via the yield inputs because dry sown crop are separate landuses.

    :param stub: boolean: set to True when calculating yield penalty for stubble penalty.

    '''
    ##inputs
    seed_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    wet_seeding_penalty_k_z = zfun.f_seasonal_inp(pinp.crop['yield_penalty_wet'], axis=1)

    ##adjust seeding penalty - crops that are not harvested eg fodder dont have yield penalty. But do have a stubble penalty
    if stub:
        ###if calculating yield penalty for stubble then include all crop (eg include fodders)
        pass
    else:
        ###if calculating yield penalty for grain transfer then only include harvested crops (eg dont include fodders)
        proportion_grain_harv_k = pd.Series(pinp.stubble['proportion_grain_harv'], index=pinp.stubble['i_stub_landuse_idx'])
        wet_seeding_penalty_k_z = wet_seeding_penalty_k_z.mul(proportion_grain_harv_k>0, axis=0)

    ##general info
    mach_periods = per.f_p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]

    ##wet seeding penalty - penalty = average penalty of period (= (start day + end day) / 2 * penalty)
    seed_start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
    penalty_free_days_z = seed_period_lengths_pz[0].astype('timedelta64[D]')
    start_day_pz = 1 + (mach_periods_start_pz - (seed_start_z + penalty_free_days_z))/ np.timedelta64(1, 'D')
    end_day_pz = (mach_periods_end_pz - (seed_start_z + penalty_free_days_z))/ np.timedelta64(1, 'D')
    wet_penalty_pzk = (start_day_pz + end_day_pz)[...,na] / 2 * wet_seeding_penalty_k_z.T.values
    wet_penalty_pzk = np.clip(wet_penalty_pzk, 0, np.inf)

    ##add m axis - needed so yield penalty can be combined with phase yield
    alloc_mp5z = rps.f1_rot_period_alloc(mach_periods_start_pz[na,:,:], z_pos=-1)
    penalty_mp5zk = wet_penalty_pzk * alloc_mp5z[...,na]

    ##put into df
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = wet_seeding_penalty_k_z.index
    keys_m = per.f_phase_periods(keys=True)
    cols_mp5zk = pd.MultiIndex.from_product([keys_m, keys_p5, keys_z, keys_k])
    penalty = pd.Series(penalty_mp5zk.ravel(), index=cols_mp5zk)
    return penalty


def f_stubble_penalty():
    '''
    Calculates the stubble penalty in each mach period (wet and dry seeding) due to sowing timeliness- kg/ha/period/crop.
    '''
    import CropResidue as stub
    yield_penalty_mp5zk = f_sowing_timeliness_penalty(stub=True) #late sowing yield reduction kg/ha/period
    stub_production_k = stub.f_cropresidue_production() #stubble production per kg of grain yield
    stub_penalty = yield_penalty_mp5zk.mul(stub_production_k, level=-1)
    return stub_penalty

#######################################################################################################################################################
#######################################################################################################################################################
#harv / hay making
#######################################################################################################################################################
#######################################################################################################################################################


#################################################
#harvesting                                     #
#################################################

def harv_time_ha():
    '''
    Harvest rate for each crop (hr/ha).

    Harvest rate is calculated as a product of harvesting speed (for the base crop), header width and
    field efficiency (overlap, turning and reduced speed when unloading). Harvest rate is assumed
    to be the same on all LMUs however, it does vary for different crops because harvest biomass
    impacts harvest speed.

    Note: Has to be kept as a separate function because it is used in multiple places.

    '''
    harv_speed = uinp.mach[pinp.mach['option']]['harvest_speed']
    ##work rate hr/ha, determined from speed, size and eff
    return 10/ (harv_speed * uinp.mach[pinp.mach['option']]['harv_eff'] * uinp.mach[pinp.mach['option']]['harvester_width'])



def f_harv_rate_period():
    '''
    Harv rate (t/hr) in each harvest period for each crop.

    Tonnes harvested per hour in each period for each crop is calculated from the harvest rate per
    hectare (see harv_time_ha) and the average crop yield. The rate is then set to 0 if a crop can not
    be harvested in a period.

    Harvesting can begin on different dates depending on the crop. For example, in Western Australia the
    harvest of canola often begins before cereals :cite:p:`RN89`. To represent this AFO represents harvest in
    multiple periods determined by the user. Each crop has an inputted harvest start date which determines
    the harvest period for that crop. There is no penalty for late harvesting, however, to capture the
    timeliness of completion, harvest can only occur in the dedicated harvest periods.

    '''
    ##season inputs through function
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype(np.datetime64) #when the first crop begins to be harvested (eg when harv periods start)
    harv_period_lengths_z = np.sum(zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done
    start_harvest_crops = pinp.crop['start_harvest_crops']
    start_harvest_crops_kz = zfun.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1).astype(np.datetime64) #start harvest for each crop

    ##harv occur - note: some crops are not harvested in the early harv period
    mach_periods = per.f_p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]
    harv_occur_pkz = np.logical_and(mach_periods_start_pz[:,na,:] < harv_end_z,
                                    mach_periods_end_pz[:,na,:] > start_harvest_crops_kz)

    ##add m axis - needed so harv req can be combined with phase yield
    alloc_mp5z = rps.f1_rot_period_alloc(mach_periods_start_pz[na,:,:], z_pos=-1)
    harv_occur_mpkz = harv_occur_pkz * alloc_mp5z[...,na,:]

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = start_harvest_crops.index
    keys_m = per.f_phase_periods(keys=True)

    index_mpkz = pd.MultiIndex.from_product([keys_m, keys_p5, keys_k, keys_z])
    harv_occur = pd.Series(harv_occur_mpkz.ravel(), index=index_mpkz)

    ##Grain harvested per hr (t/hr) for each crop.
    harv_rate = (uinp.mach_general['harvest_yield'] * (1 / harv_time_ha())).squeeze()

    ##combine harv rate and harv_occur
    harv_rate_period = harv_occur.mul(harv_rate, level=2)
    return harv_rate_period


#adds the max number of harv hours for each crop for each period to the df  
def f_max_harv_hours():
    '''
    Maximum hours that can be spent harvesting in a given period per crop gear compliment.

    Grain moisture content can often exceed the accepted threshold during the harvesting period, particularly
    overnight and early in the morning due to higher humidity and overnight dew :cite:p:`RN118`. Growers without
    drying facilities must wait until the grain moisture is acceptable resulting in a restricted number of
    harvest hours per day. To capture this the model has a user defined input that controls the maximum number
    of harvest hours per day per harvester.

    '''

    ##inputs
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0)
    harv_period_lengths_z = np.sum(zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z.astype('datetime64') + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done

    ##does any harvest occur in given period
    mach_periods_start_pz = per.f_p_dates_df()[:-1]
    mach_periods_end_pz = per.f_p_dates_df()[1:]
    harv_occur_pz = np.logical_and(harv_start_z <= mach_periods_start_pz, mach_periods_start_pz < harv_end_z)

    ##max harv hour per period
    days_pz = (mach_periods_end_pz.values - mach_periods_start_pz.values)/ np.timedelta64(1, 'D')
    max_hours_pz = days_pz * harv_occur_pz * pinp.mach['daily_harvest_hours']
    return max_hours_pz


def f1_harv_cost_alloc():
    '''allocation of harvest cost into cashflow period'''

    ##inputs
    p5_start_zp5 = per.f_p_dates_df().values[:-1].T  # .T because needs to match other shapes in cash function call below
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]

    ##calc interest and allocate to cash period - needs to be numpy
    harv_cost_allocation_c0p7zp5, harv_wc_allocation_c0p7zp5 = fin.f_cashflow_allocation(p5_start_zp5, enterprise='crp', z_pos=-2)
    ###convert to df
    new_index_c0p7zp5 = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z,keys_p5])
    harv_cost_allocation_c0p7zp5 = pd.Series(harv_cost_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    harv_wc_allocation_c0p7zp5 = pd.Series(harv_wc_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    return harv_cost_allocation_c0p7zp5, harv_wc_allocation_c0p7zp5


def f_harvest_cost(r_vals):
    '''
    Cost of harvest in each cashflow period ($/hr).

    The cost of harvesting for one hour is the same for each crop. However, the tonnes harvested
    per hour varies for different crops resulting in a different harvest cost per tonne.
    Harvest cost includes fuel, oil, grease and r&m. Oil, grease repairs and maintenance are calculated
    as a factor of fuel cost.

    '''
    ##allocation
    harv_cost_allocation_c0p7zp5, harv_wc_allocation_c0p7zp5 = f1_harv_cost_alloc()

    ##cost
    ##fuel used L/hr - same for each crop
    fuel_used = uinp.mach[pinp.mach['option']]['harv_fuel_consumption']
    ##determine cost of fuel and oil and grease $/ha
    fuel_cost_hr = fuel_used * fuel_price()
    oil_cost_hr = fuel_used * uinp.mach[pinp.mach['option']]['oil_grease_factor_harv'] * fuel_price()
    ##determine fuel and oil cost per hr
    fuel_oil_cost_hr = fuel_cost_hr + oil_cost_hr
    ##return fuel and oil cost plus r & m ($/hr)
    cost_harv = fuel_oil_cost_hr + uinp.mach[pinp.mach['option']]['harvest_maint']
    harv_cost_k = cost_harv.squeeze()
    
    ##reindex with lmu so alloc can be mul with harv_cost
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns_c0p7zp5k = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, harv_cost_k.index])
    harv_cost_allocation_c0p7zp5k = harv_cost_allocation_c0p7zp5.reindex(columns_c0p7zp5k)
    harv_wc_allocation_c0p7zp5k = harv_wc_allocation_c0p7zp5.reindex(columns_c0p7zp5k)

    ##mul costs and allocation
    harv_cost_c0p7zp5k = harv_cost_allocation_c0p7zp5k.mul(harv_cost_k, level=-1)
    harv_wc_c0p7zp5k = harv_wc_allocation_c0p7zp5k.mul(harv_cost_k, level=-1)

    r_vals['harvest_cost'] = harv_cost_c0p7zp5k
    return harv_cost_c0p7zp5k, harv_wc_c0p7zp5k


#########################
#contract harvesting    #
#########################

def f_contract_harv_rate():
    '''
    Grain harvested per hr by contractor (t/hr).
    '''
    ##season inputs through function
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0).astype(np.datetime64) #when the first crop begins to be harvested (eg when harv periods start)
    harv_period_lengths_z = np.sum(zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z + harv_period_lengths_z.astype('timedelta64[D]') #when all harv is done
    start_harvest_crops = pinp.crop['start_harvest_crops']
    start_harvest_crops_kz = zfun.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1).astype(np.datetime64) #start harvest for each crop

    ##harv occur - note: some crops are not harvested in the early harv period
    mach_periods = per.f_p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]
    harv_occur_pkz = np.logical_and(mach_periods_start_pz[:,na,:] < harv_end_z,
                                    mach_periods_end_pz[:,na,:] > start_harvest_crops_kz)

    ##add m axis - needed so machinery can be linked with phases (machinery just has a p5 axis)
    alloc_mp5z = rps.f1_rot_period_alloc(mach_periods_start_pz[na,:,:], z_pos=-1)
    harv_occur_mpkz = harv_occur_pkz * alloc_mp5z[...,na,:]

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = start_harvest_crops.index
    keys_m = per.f_phase_periods(keys=True)

    index_mpkz = pd.MultiIndex.from_product([keys_m, keys_p5, keys_k, keys_z])
    harv_occur = pd.Series(harv_occur_mpkz.ravel(), index=index_mpkz)

    ##Grain harvested per hr (t/hr) for each crop.
    yield_approx = uinp.mach_general['harvest_yield'] #these are the yields the contract harvester is calibrated to - they are used to convert time/ha to t/hr
    harv_speed = uinp.mach_general['contract_harvest_speed']
    ###work rate hr/ha, determined from speed, size and eff
    contract_harv_time_ha = 10 / (harv_speed * uinp.mach_general['contract_harvester_width'] * uinp.mach_general['contract_harv_eff'])
    ###overall t/hr
    harv_rate = (yield_approx * (1 / contract_harv_time_ha)).squeeze()

    ##combine harv rate and harv_occur
    harv_rate = harv_occur.mul(harv_rate, level=2)

    return harv_rate
#print(contract_harv_rate())


def f_contract_harvest_cost(r_vals):
    '''
    Cost of contract harvest in each cashflow period ($/hr).
    '''
    ##allocation
    harv_cost_allocation_c0p7zp5, harv_wc_allocation_c0p7zp5 = f1_harv_cost_alloc()

    ##contract harv cost
    contract_harv_cost_k = uinp.price['contract_harv_cost'].squeeze() #contract harvesting cost for each crop ($/hr)
    
    ##reindex with lmu so alloc can be mul with harv_cost
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns_c0p7zp5k = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, contract_harv_cost_k.index])
    contract_harv_cost_allocation_c0p7zp5k = harv_cost_allocation_c0p7zp5.reindex(columns_c0p7zp5k)
    contract_harv_wc_allocation_c0p7zp5k = harv_wc_allocation_c0p7zp5.reindex(columns_c0p7zp5k)

    ##mul costs and allocation
    contract_harv_cost_c0p7zp5k = contract_harv_cost_allocation_c0p7zp5k.mul(contract_harv_cost_k, level=-1)
    contract_harv_wc_c0p7zp5k = contract_harv_wc_allocation_c0p7zp5k.mul(contract_harv_cost_k, level=-1)

    r_vals['contract_harvest_cost'] = contract_harv_cost_c0p7zp5k
    return contract_harv_cost_c0p7zp5k, contract_harv_wc_c0p7zp5k


#########################
#make hay               #
#########################
def f_hay_making_cost():
    '''
    Cost to make hay ($/t).

    Typically, hay making is completed by contract workers and generally hay is not a large component of a
    farming system. Therefore, currently contract hay making is the only option represented in AFO. There
    is a cost for mowing ($/ha), bailing ($/t) and carting ($/t).

    Note: Currently it is assumed that hay is allocated into the same cashflow periods in all seasons.
    '''
    ##cost allocation
    hay_start = np.array([pinp.crop['hay_making_date']]).astype('datetime64')
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_m = per.f_phase_periods(keys=True)
    ###call allocation/interset function - needs to be numpy
    hay_cost_allocation_c0p7z,hay_wc_allocation_c0p7z = fin.f_cashflow_allocation(hay_start, enterprise='crp', z_pos=-1)
    ###convert to df
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    hay_cost_allocation_c0p7z = pd.Series(hay_cost_allocation_c0p7z.ravel(),index=new_index_c0p7z)
    hay_wc_allocation_c0p7z = pd.Series(hay_wc_allocation_c0p7z.ravel(),index=new_index_c0p7z)


    ##hay making cost $/t
    mow_cost =  uinp.price['contract_mow_hay'] / uinp.mach_general['approx_hay_yield']
    bail_cost =  uinp.price['contract_bail'] 
    cart_cost = uinp.price['cart_hay']
    total_cost = mow_cost + bail_cost + cart_cost

    ##calc interest and allocate to cash period
    hay_cost_c0p7z = hay_cost_allocation_c0p7z * total_cost
    hay_wc_c0p7z = hay_wc_allocation_c0p7z * total_cost

    ##m allocation - hay can only be made in the season stage where the cost is incurred
    alloc_mz = rps.f1_rot_period_alloc(hay_start[na], z_pos=-1)
    index_mz = pd.MultiIndex.from_product([keys_m,keys_z])
    hay_made_prov_mz = pd.Series(alloc_mz.ravel(),index=index_mz)

    return hay_cost_c0p7z, hay_wc_c0p7z, hay_made_prov_mz

#######################################################################################################################################################
#######################################################################################################################################################
#stubble handling
#######################################################################################################################################################
#######################################################################################################################################################

##############################
#cost per ha stubble handling#
##############################
def f_stubble_cost_ha():
    '''
    Tractor cost to handle stubble for 1 ha.

    Stubble handling cost per hectare includes tractor costs and rack costs. Tractor costs consist of fuel, oil,
    grease and r&m. Rack costs consist of just repairs and maintenance. This cost is adjusted for rotation
    phase and LMU cost in Phase.py.

    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + stubble rake(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['stubble_maint']
    return cost
#cc=stubble_cost_ha()

#######################################################################################################################################################
#######################################################################################################################################################
#fert
#######################################################################################################################################################
#######################################################################################################################################################

###########################
#fert application time   # used in labour crop also, defined here because it uses inputs from the different mach options which are consolidated at the top of this sheet
###########################

#time taken to spread 1ha (not including driving to and from paddock and filling up)
# hr/ha= 10/(width*speed*efficiency)
def time_ha():
    '''
    Time taken to fertilise 1ha.

    This is dependent on the spread width (e.g. lighter fertilisers have a narrower spread meaning more
    distance must be travelled to cover 1 ha in fertiliser)
    '''
    width_df = uinp.mach[pinp.mach['option']]['spreader_width']
    return 10/(width_df*uinp.mach[pinp.mach['option']]['spreader_speed']*uinp.mach[pinp.mach['option']]['spreader_eff'])

#time taken to driving to and from paddock and filling up
# hr/cubic m = ((ave distance to paddock *2)/speed + fill up time)/ spreader capacity  # *2 because to and from paddock
def time_cubic():
    '''Time taken to fill up spreader and drive to and from paddock.

    This represents the time driving to and from the paddock and filling up. This is dependent on the
    the density of the fertiliser (e.g. more time would be required filling and traveling to
    spread 1 tonne of a lower density fertiliser).

    '''
    return (((pinp.mach['ave_pad_distance'] *2)
              /uinp.mach[pinp.mach['option']]['spreader_speed'] + uinp.mach[pinp.mach['option']]['time_fill_spreader'])
              /uinp.mach[pinp.mach['option']]['spreader_cap'])
     

###################
#application cost # *remember that lime application only happens every 4 yrs - accounted for in the passes inputs
################### *used in crop pyomo
#this is split into two sections - new feature of AFO
# 1- cost to drive around 1ha
# 2- cost per cubic metre ie to represent filling up and driving to and from paddock

def spreader_cost_hr():
    '''
    Cost to spread each fertiliser for one hour.

    Spreading cost per hour includes tractor costs and spreader costs. Tractor costs consist of fuel, oil,
    grease and r&m. Spreader costs consist of just repairs and maintenance.

    Used to determine both fertiliser application cost per hour and per ha.
    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + spreader(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['spreader_maint']
    return cost

def fert_app_cost_ha():
    '''

    Fertiliser application cost part 1: Application cost per hectare.

    The cost of applying fertilising is calculated in two parts. Part 1 is the cost per hectare
    for each fertiliser which represents the time taken spreading fertiliser in the paddock (see time_ha)
    and the cost of fertilising per hour (see spreader_cost_hr)

    '''
    return spreader_cost_hr() * time_ha().stack().droplevel(1)

def fert_app_cost_t():
    '''
    Fertiliser application cost part 2: time required per tonne.

    The cost of applying fertilising is calculated in two parts. Part 2 is the cost per tonne
    for each fertiliser which represents the time taken driving to and from the paddock and filling
    up (see time_cubic) and the cost of fertilising per hour (see spreader_cost_hr)

    '''
    spreader_proportion = pinp.crop['fert_info']['spreader_proportion']
    conversion = pinp.crop['fert_info']['fert_density']
    ##spreader cost per hr is multiplied by the full time required to drive to and from paddock and fill up even though during filling up the tractor is just sitting, but this accounts for the loader time/cost
    cost_cubic = spreader_cost_hr() * time_cubic()
    ##mulitiplied by a factor (spreader_proportion) 0 or 1 if the fert is applied at seeding (or a fraction if applied at both seeding and another time)
    cost_t = cost_cubic / conversion * spreader_proportion #convert from meters cubed to tonne - divide by conversion (density) because lighter ferts require more filling up time per tonne
    return cost_t
#e=fert_app_t()
    
#######################################################################################################################################################
#######################################################################################################################################################
#chem
#######################################################################################################################################################
#######################################################################################################################################################

###########################
#chem application time   # used in labour crop, defined here because it uses inputs from the different mach options which are consolidated at the top of this sheet
###########################

##time taken to spray 1ha (use efficiency input to allow for driving to and from paddock and filling up)
## hr/ha= 10/(width*speed*efficiency)
def spray_time_ha():
    '''
    Time taken to spray 1ha.

    This is dependent on the sprayer width, speed and field efficiency (accounts for overlap,
    filling up time and turing).

    '''

    width_df = uinp.mach[pinp.mach['option']]['sprayer_width']
    return 10/(width_df*uinp.mach[pinp.mach['option']]['sprayer_speed']*uinp.mach[pinp.mach['option']]['sprayer_eff'])

   

###################
#application cost # 
################### *used in crop pyomo

def chem_app_cost_ha():
    '''
    Chemical application cost per hectare.

    The cost of spraying per hectare is calculated based on the time to spray a hectare (see spray_time_ha)
    and the cost to spray per hour. Spraying cost per hour includes tractor costs and sprayer costs.
    Tractor costs consist of fuel, oil, grease and r&m. Sprayer costs consist of just repairs and maintenance.

    Typically, all spraying is done at the same speed and the rate of application is simply
    controlled by the chemical concentration in the spray. Thus, the cost of spraying is the same for
    all chemicals and on all LMUs.

    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price()
    tractor_rm = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['repair_maint_factor_tractor']
    tractor_oilgrease = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price() * uinp.mach[pinp.mach['option']]['oil_grease_factor_tractor']
    ##cost/hr= tractor costs + sprayer(r&m) 
    cost = tractor_fuel + tractor_rm + tractor_oilgrease + uinp.mach[pinp.mach['option']]['sprayer_maint']
    return cost



       
#######################################################################################################################################################
#######################################################################################################################################################
#Dep
#######################################################################################################################################################
#######################################################################################################################################################

#########################
#mach value             #
#########################
##harvest machine cost
def harvest_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'harvest allocation'])
    total_value = value * pinp.mach['number_harv_gear']
    return total_value


##value of gear used for seed. This is used to calculate the variable depreciation linked to seeding activity
def f_seeding_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'seeding allocation'])
    total_value = value * pinp.mach['number_seeding_gear']
    return total_value

##total machine value - used to calc asset value, fixed dep and insurance
def f_total_clearing_value():
    harv_value = harvest_gear_clearing_value()
    seed_value = f_seeding_gear_clearing_value()
    other_value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'remaining allocation'])
    total_clearing_value = harv_value + seed_value + other_value
    ##all is incurred in the last m1 period (although it could occur in any period it doesnt make a difference)
    keys_m1 = per.f_season_periods(keys=True)
    total_clearing_value = pd.Series(total_clearing_value,index=keys_m1[-1:])
    return total_clearing_value

#########################
#fixed depreciation     #
#########################


#total value of crop gear x dep rate x number of crop gear
def f_fix_dep():
    '''Fixed depreciation on machinery

    Fixed depreciation captures obsolescence costs and is incurred every year independent of equipment uses.
    It is simply calculated based on the total clearing sale value of equipment and the fixed rate of depreciation.
    '''
    fixed_dep = f_total_clearing_value() * uinp.finance['fixed_dep']
    return fixed_dep


####################################
#variable seeding depreciation     #
####################################

def f_seeding_dep():
    '''
    Average variable dep for seeding $/ha.

    Variable depreciation is use depreciation and is dependent on the number of hours the equipment is used.
    Seeding depreciation is calculated based on the time taken to sow 1ha of each crop on each LMU and
    the rate of depreciation per hour of seeding.

    The rate of depreciation per hour is calculated based 'typical' scenario, which simplifies the
    calibration process. The user enters the percentage of depreciation incurred for sowing `x` hectares
    of crop. This is converted to a dollar cost per hour based on the seeding rate and the machinery value.

    '''
    ##inputs
    seed_rate = f_seed_time_lmus().squeeze()
    ##first determine the approx time to seed all the crop - which is equal to dep area x average seeding rate (hr/ha)
    average_seed_rate = seed_rate.mean()
    seeding_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_seed_rate
    ##second, determine dep per hour - equal to crop gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    dep_hourly = seeding_gear_clearing_value * dep_rate / seeding_time
    ##third, convert to dep per ha for each soil type - equals cost per hr x seeding rate per hr
    dep_ha = dep_hourly * seed_rate
    ##allocate season period based on mach/labour period - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    alloc_m1p5z = zfun.f1_z_period_alloc(date_start_p5z[na,...], z_pos=-1)
    ###make df
    keys_p5 = mach_periods.index[:-1]
    keys_z = zfun.f_keys_z()
    keys_m1 = per.f_season_periods(keys=True)
    index_m1p5z = pd.MultiIndex.from_product([keys_m1,keys_p5,keys_z])
    alloc_m1p5z = pd.Series(alloc_m1p5z.ravel(), index=index_m1p5z)
    index_m1p5zl = pd.MultiIndex.from_product([keys_m1,keys_p5,keys_z,dep_ha.index])
    alloc_m1p5zl = alloc_m1p5z.reindex(index_m1p5zl)
    return alloc_m1p5zl.mul(dep_ha, level=-1)


####################################
#variable harvest depreciation     #
####################################

def f_harvest_dep():
    '''
    Average variable dep for harvesting $/hr.

    Variable depreciation is use depreciation and is dependent on the number of hours the equipment is used.
    The harvest activity is represented in hours so the variable depreciation is simply the rate of depreciation
    per hour of seeding.

    The rate of depreciation per hour is calculated based 'typical' scenario, which simplifies the
    calibration process. The user enters the percentage of depreciation incurred for harvesting `x` hectares
    of crop. This is converted to a dollar cost per hour based on the harvest rate and the machinery value.

    '''
    ##first determine the approx time to harvest all the crop - which is equal to dep area x average harvest rate (hr/ha)
    average_harv_rate = harv_time_ha().squeeze().mean()
    average_harv_time = uinp.mach[pinp.mach['option']]['dep_area'] * average_harv_rate 
    ##second, determine dep per hour - equal to harv gear value x dep % / seeding time
    dep_rate = uinp.mach[pinp.mach['option']]['variable_dep'] - uinp.finance['fixed_dep']
    dep_hourly = harvest_gear_clearing_value() * dep_rate / average_harv_time
    ##allocate season period based on mach/labour period - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    alloc_m1p5z = zfun.f1_z_period_alloc(date_start_p5z[na,...], z_pos=-1)
    ###make df
    keys_p5 = mach_periods.index[:-1]
    keys_z = zfun.f_keys_z()
    keys_m1 = per.f_season_periods(keys=True)
    index_m1p5z = pd.MultiIndex.from_product([keys_m1,keys_p5,keys_z])
    alloc_m1p5z = pd.Series(alloc_m1p5z.ravel(), index=index_m1p5z)

    return alloc_m1p5z * dep_hourly


#######################################################################################################################################################
#######################################################################################################################################################
#insurance on all gear
#######################################################################################################################################################
#######################################################################################################################################################
def f_insurance(r_vals):
    '''

    Cost of insurance for all machinery.

    Machinery insurance cost is calculated as a percentage of the total machinery value.
    The cost is incurred on day 1.

    '''
    ##determine the insurance paid
    value_all_mach = f_total_clearing_value().squeeze()
    insurance_cost = value_all_mach * uinp.finance['equip_insurance']
    
    ##determine cash period
    start = np.array([uinp.mach_general['insurance_date']]).astype('datetime64')
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    ###call allocation/interset function - needs to be numpy
    insurance_cost_allocation_c0p7z,insurance_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='crp', z_pos=-1)
    ###convert to df
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    insurance_cost_allocation_c0p7z = pd.Series(insurance_cost_allocation_c0p7z.ravel(),index=new_index_c0p7z)
    insurance_wc_allocation_c0p7z = pd.Series(insurance_wc_allocation_c0p7z.ravel(),index=new_index_c0p7z)
    
    ##calc interest and allocate to cash period
    insurance_cost_c0p7z = insurance_cost_allocation_c0p7z * insurance_cost
    insurance_wc_c0p7z = insurance_wc_allocation_c0p7z * insurance_cost

    r_vals['mach_insurance'] = insurance_cost_c0p7z
    return insurance_cost_c0p7z.to_dict(), insurance_wc_c0p7z.to_dict()


#######################################################################################################################################################
#######################################################################################################################################################
#params
#######################################################################################################################################################
#######################################################################################################################################################

##collates all the params
def f_mach_params(params,r_vals):
    seed_days = f_seed_days().stack()
    contractseeding_occur = f_contractseeding_occurs().stack()
    seedrate = f_overall_seed_rate(r_vals)
    seeding_cost, seeding_wc = f_seeding_cost(r_vals)
    contract_seed_cost, contract_seed_wc = f_contract_seed_cost(r_vals)
    harv_rate_period = f_harv_rate_period()
    contract_harv_rate = f_contract_harv_rate()
    max_harv_hours = f_max_harv_hours().stack()
    harvest_cost, harvest_wc = f_harvest_cost(r_vals)
    contract_harvest_cost, contract_harvest_wc = f_contract_harvest_cost(r_vals)
    hay_making_cost, hay_making_wc, hay_made_prov_mz  = f_hay_making_cost()
    yield_penalty = f_sowing_timeliness_penalty()
    stubble_penalty = f_stubble_penalty()
    poc_grazing_days = f_poc_grazing_days().stack()
    fixed_dep = f_fix_dep()
    harv_dep = f_harvest_dep()
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    seeding_dep = f_seeding_dep()
    insurance_cost, insurance_wc = f_insurance(r_vals)
    mach_asset_value = f_total_clearing_value()

    ##add inputs that are params to dict
    params['number_seeding_gear'] = pinp.mach['number_seeding_gear']
    params['number_harv_gear'] = pinp.mach['number_harv_gear']
    params['seeding_occur'] = pinp.mach['seeding_occur']

    ##create non seasonal params
    params['seed_rate'] = seedrate.to_dict()
    params['contract_harv_rate'] = contract_harv_rate.to_dict()
    params['fixed_dep'] = fixed_dep.to_dict()
    params['harv_dep'] = harv_dep.to_dict()
    params['seeding_gear_clearing_value'] = seeding_gear_clearing_value
    params['seeding_dep'] = seeding_dep.to_dict()
    params['mach_asset_value'] = mach_asset_value.to_dict()

    ##create season params
    params['seed_days'] = seed_days.to_dict()
    params['contractseeding_occur'] = contractseeding_occur.to_dict()
    params['seeding_cost'] = seeding_cost.to_dict()
    params['seeding_wc'] = seeding_wc.to_dict()
    params['contract_seed_cost'] = contract_seed_cost.to_dict()
    params['contract_seed_wc'] = contract_seed_wc.to_dict()
    params['harv_rate_period'] = harv_rate_period.to_dict()
    params['harvest_cost'] = harvest_cost.to_dict()
    params['harvest_wc'] = harvest_wc.to_dict()
    params['hay_making_cost'] = hay_making_cost.to_dict()
    params['hay_making_wc'] = hay_making_wc.to_dict()
    params['hay_made_prov_mz'] = hay_made_prov_mz.to_dict()
    params['max_harv_hours'] = max_harv_hours.to_dict()
    params['contract_harvest_cost'] = contract_harvest_cost.to_dict()
    params['contract_harvest_wc'] = contract_harvest_wc.to_dict()
    params['yield_penalty'] = yield_penalty.to_dict()
    params['stubble_penalty'] = stubble_penalty.to_dict()
    params['poc_grazing_days'] = poc_grazing_days.to_dict()
    params['insurance'] = insurance_cost
    params['insurance_wc'] = insurance_wc


