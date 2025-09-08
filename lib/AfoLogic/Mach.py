# -*- coding: utf-8 -*-
"""

author: young

The functions in this module calculate the rate at which machinery can be used, the cost incurred,
and the timeliness of certain jobs.

The model has been designed to accommodate a range of machine options. Inputs that are dependent
on machine size are grouped into their own input sheet, this section can be duplicated as many times
as necessary to represent different machine sizes and/ or amount of machinery. The model user can
then select which machine option they want in the analysis.
The user inputs the desired machinery option in property input section. If the user wants
to test the impact of different machine options they can alter the input using a saa sensitivity variable.

All machine option inputs (located in Universal.xlsx) are calibrated to represent cropping on a standard LMU.
In GSW it is the loamy sand, with granite outcropping where Wandoo (white gum) is a common tree.
To allow for the differences between the calibration and the actual LMUs there are LMU adjustment
factors in Property.xlsx.

To reduce space and complexity the model currently only represents machinery activities for seeding and
harvest (all other machinery usage is directly converted to a cost linked with another activity).
Therefore the hours of machinery use outside of seeding and harvest are not tracked. Hence,
machinery use outside of seeding and harvest does not incur any variable
depreciation cost. To minimise this error, the rate of variable depreciation can be increased slightly.

Crops and pasture can be sown dry (i.e. seeding occurs before the opening rains that trigger the commencement
of the growing season) or wet (seeding occurs after the opening rains) :cite:p:`RN119`. Dry seeding can
be useful in weather-years with a late start to the growing season, maximising the growing season and water
utilisation because the crop can germinate as soon as the first rains come. Dry seeding begins on
a date specified by the user, and can be performed until the opening rains. Wet seeding occurs a set number
of days after the opening rains to allow time for weed germination and application of a knockdown spray.
Wet seeding can only occur during a proportion of the time because some of
the time it is either too dry or too wet to seed. The timeliness of sowing can have a large impact on
crop yields. AFO accounts for this using a yield penalty (see yield penalty function for more info).

AFO also represents the ability to hire contract workers to complete harvest and seeding. This can be
useful for large operations, farms with poor machinery or farms with poor labour availability.

"""

#python modules
import pandas as pd
import datetime as dt
import numpy as np



#AFO modules
from . import UniversalInputs as uinp
from . import PropertyInputs as pinp
from . import StructuralInputs as sinp
from . import Periods as per
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import EmissionFunctions as efun
from . import Finance as fin

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

    The task of feeding the supplement has a cost reflecting the fuel, and machine repairs and maintenance.
    This is a variable cost incurred for each tonne of supplement fed. The cost is calculated simply from
    inputs which state the litres required and repairs and maintenance cost to feed 1 tonne of each supplement.

    .. note:: This method of calculating the cost is satisfactory, but it could be improved by linking
        the cost of feeding to the distance travelled which is controlled by the rate of feeding (g/hd),
        sheep numbers and the stocking rate (ie lower stocking rate means sheep are spread over a larger area).
        Rate of feeding could be calculated from the tonnes for feed fed (activity), the frequency of feeding
        (input) and the number of sheep (activity). Then somehow adjusted for the spread of sheep over the
        property using the stocking rate. Or maybe it could be calculated more like the labour required.

    '''
    sup_cost=uinp.mach[pinp.mach['option']]['sup_feed'].copy() #need this so it doesn't alter inputs
    ##add fuel cost
    fuel_used_k = sup_cost['litres']
    sup_cost['litres']=fuel_used_k * fuel_price()
    ##sum the cost of r&m and fuel - note that rm is in the sup_cost df
    return sup_cost.sum(axis=1), fuel_used_k

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
    Determines the max number of seeding days in each p5 period (basically just the length of p5).
    f_sow_prov in the phase module determines what crops/pastures can sown in each p5 period.
    '''
    mach_periods = per.f_p_dates_df()
    start_pz = mach_periods.values[:-1]
    end_pz = mach_periods.values[1:]
    length_pz = np.maximum(0,(end_pz - start_pz))
    days = pd.DataFrame(length_pz, index=mach_periods.index[:-1], columns=mach_periods.columns)
    return days

def f_contractseeding_occurs():
    '''
    #todo i think this can be removed because contract seeding is hooked up to the yield penalty.
    This function just sets the period when contract seeding must occur (period when wet seeding begins).
    Contract seeding is not hooked up to yield penalty because if you're going to hire someone you will hire
    them at the optimum time. Contract seeding is hooked up to poc so this param stops the model having late seeding
    (contract seeding must occur in the first seeding period).
    '''
    contract_start_z = per.f_wet_seeding_start_date()
    mach_periods = per.f_p_dates_df()
    start_pz = mach_periods.values[:-1]
    end_pz = mach_periods.values[1:]
    contractseeding_occur_pz = np.logical_and(start_pz <= contract_start_z, contract_start_z < end_pz)
    contractseeding_occur_pz = pd.DataFrame(contractseeding_occur_pz,index=mach_periods.index[:-1],columns=mach_periods.columns)
    return contractseeding_occur_pz * 1
    # params['contractseeding_occur'] = (mach_periods==contract_start).squeeze().to_dict()


def f_poc_grazing_days():
    '''
    Grazing days provided by wet seeding activity (days/ha sown in each mach period/feed period).

    This section represents the grazing achieved on crop paddocks before seeding since the previous node. The longer seeding is
    delayed the more grazing is achieved. The calculations also account for a gap between when pasture
    germinates (the pasture break) and when seeding can begin (the seeding break). Dry seeding doesn't
    provide any grazing, so the calculations are only done using the wet seeding days. The calculations
    allow for destocking a certain number of days before seeding (termed the defer period) to allow the pasture
    leaf area to increase so that the knock down spray is effective.

    Grazing days is the sum of the area grazed multiplied by the number of days of grazing. As such it is
    dependent on both the area that is being grazed and the duration of grazing. The number of grazing days
    is calculated for each seeding decision variable in each machinery period (p5). However, the number
    of grazing days must be calculated for each feed period (p6) so that the
    feed supply will align with the feed demand of the animals (that are calculated for feed periods).

    The grazing days associated with the seeding decision variable (per hectare sown) is made up of two parts:
    A 'rectangular' component which represents the area being grazed each day from the break of season or the start of
    the most recent season period up until the destocking date for the beginning of the machine period.
    The rectangle component must consider the season period because poc only exists when a crop phase has been selected.
    In the prior season period if the model is waiting to make a landuse decision it can select a temporary pasture (a2)
    which provides feed. Therefore, to aviod double counting poc is only provided in the current season period. This
    doesnt effect the triangle component below because the triangle component is based on the current p5 which has to be in the current season period.
    A 'triangle' component which represents the grazing during the seeding period. The area grazed each day
    diminishes associated with destocking the area that is soon to be sown. For example, at the start
    of the period all area can be grazed but by the end of the period once all the area has been destocked
    no grazing can occur. These 2 components must then be allocated to the feed periods.

    'Rectangular' component: The base of the rectangle is defined by the later of the break, the start of the season period
    and the start of the feed period through to the earlier of the end of the feed period or the start of
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

    The assumption is that; seeding is done evenly throughout a given period. In reality this is wrong e.g. if a
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
    defer_period = np.array([pinp.crop['poc_destock']]) #days between seeding and destocking
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'],numpy=True)
    wet_seeding_start_z = per.f_wet_seeding_start_date()
    date_p7z = per.f_season_periods()

    ##calc the most recent node date for each p5
    a_p7prev_p5z = fun.searchsort_multiple_dim(date_p7z, date_start_p5z, 1, 1, side='right') - 1
    p7prev_date_p5z = np.take_along_axis(date_p7z, a_p7prev_p5z, axis=0)

    ##calc wet seeding days
    start_pz = np.maximum(wet_seeding_start_z, date_start_p5z)
    seed_days_p5z = np.maximum(0,(date_end_p5z - start_pz))

    ##grazing days rectangle component (for p5) and allocation to feed periods (p6)
    rec_base_p6p5z = (np.minimum(date_end_p6z[:,na,:], date_start_p5z - defer_period) \
                  - np.maximum(season_break_z, np.maximum(date_start_p6z[:,na,:], p7prev_date_p5z)))
    height_p5z = 1
    poc_grazing_days_rect_p6p5z = np.maximum(0, rec_base_p6p5z * height_p5z)

    ##grazing days triangular component (for p5) and allocation to feed periods (p6)
    start_p6p5z = np.maximum(date_start_p6z[:,na,:], np.maximum(season_break_z, date_start_p5z - defer_period))
    end_p6p5z = np.minimum(date_end_p6z[:,na,:], date_end_p5z - defer_period)
    tri_base_p6p5z = (end_p6p5z - start_p6p5z)
    height_start_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z - defer_period) - start_p6p5z)
                                                    , seed_days_p5z))
    height_end_p6p5z = np.maximum(0, fun.f_divide(((date_end_p5z - defer_period) - end_p6p5z)
                                                    , seed_days_p5z))
    poc_grazing_days_tri_p6p5z = np.maximum(0,tri_base_p6p5z * (height_start_p6p5z + height_end_p6p5z) / 2)

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
    Rate of direct drill for each crop on each lmu (hr/ha).

    The base seeding rate per machine hour (ha/machine hr) is an input which accounts for machine size, speed,
    overlap and turning around. Seeding rate is converted from a rate per machine hour
    to a rate per seeding activity hour by adjusting by an efficiency factor that accounts for the proportion of seeding
    when seed is not going into the ground (e.g. moving paddocks or filling up)

    Seeding rate can vary for each LMU
    due to varying speeds associated with more or less force being required to pull the seeding equipment
    through the soil. The rate inputs are set for the base LMU and then adjusted by a
    user defined LMU factor.
    '''
    ##mask lmu input
    base_seeding_rate = uinp.mach[pinp.mach['option']]['seeding_rate_base']
    seeding_rate_lmu_adj = pinp.mach['seeding_rate_lmu_adj'].squeeze(axis=1)

    ##adjust for lmu
    rate_l = base_seeding_rate * seeding_rate_lmu_adj

    ##adjust for crop
    seeding_rate_crop_adj_k_l = pd.concat([uinp.mach[pinp.mach['option']]['seeding_rate_crop_adj']]*len(rate_l),axis=1) #expands df for each lmu
    seeding_rate_crop_adj_k_l.columns = rate_l.index #rename columns to lmu so i can mul
    rate_direct_drill_k_l=seeding_rate_crop_adj_k_l.mul(rate_l)

    ##convert from ha/hr to hr/ha
    rate_direct_drill_k_l = 1 / rate_direct_drill_k_l

    ##adjust for the time when seed is not being put in the ground due to moving paddocks or filling up.
    rate_direct_drill_k_l = rate_direct_drill_k_l / (1 - pinp.mach['seeding_downtime_frac'])

    return rate_direct_drill_k_l

def f_overall_seed_rate(r_vals):
    '''
    Hectares of each crop that can be sown per day on each LMU.

    Seeding rate per day is a product of the hours seeding can occur per day and the seeding rate
    per hectare (see f_seed_time_lmus).

    '''
    ##convert seed time (hr/ha) to rate of direct drill per day (ha/day)
    rate_direct_drill_k_l = f_seed_time_lmus()
    daily_rate_direct_drill_k_l = 1 / rate_direct_drill_k_l * pinp.mach['daily_seed_hours']

    ##store r_vals
    fun.f1_make_r_val(r_vals,daily_rate_direct_drill_k_l,'seeding_rate')
    return daily_rate_direct_drill_k_l.stack()

    
  

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
    seeding_fuel_lmu_adj = pinp.mach['seeding_fuel_lmu_adj']
    ##determine fuel use on base lmu (l/ha)
    base_lmu_seeding_fuel = uinp.mach[pinp.mach['option']]['fuel_seeding'] / uinp.mach[pinp.mach['option']]['seeding_rate_base']
    ##determine fuel use on all soils by adjusting s5 fuel use with input adjustment factors
    df_seeding_fuel_lmu = base_lmu_seeding_fuel * seeding_fuel_lmu_adj
    #second multiply base cost by adj, to produce df with seeding fuel use for each lmu (L/ha)
    return df_seeding_fuel_lmu 
    
def f1_seed_cost_alloc():
    '''
    Labour period allocation for seeding costs.

    All seeding costs for a seeding activity must be incurred in the current season node e.g. if seeding happens in node 1
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
    seeding_cost_allocation_p7zp5, seeding_wc_allocation_c0p7zp5 = fin.f_cashflow_allocation(p5_start_zp5, enterprise='crp', z_pos=-2)
    ###convert to df
    new_index_p7zp5 = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p5])
    seeding_cost_allocation_p7zp5 = pd.Series(seeding_cost_allocation_p7zp5.ravel(), index=new_index_p7zp5)
    new_index_c0p7zp5 = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5])
    seeding_wc_allocation_c0p7zp5 = pd.Series(seeding_wc_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    return seeding_cost_allocation_p7zp5, seeding_wc_allocation_c0p7zp5


# def f1_seeding_phaseperiod_allocation():
#     '''Allocation of seeding costs into phase periods'''
#
#     fert_info = pinp.crop['fert_info']
#     fert_date_n = fert_info['app_date'].values
#     fert_length_n = fert_info['app_len'].values
#     alloc_p7zn = zfun.f1_z_period_alloc(fert_date_n[na,na,:], fert_length_n[na,na,:], z_pos=-2)
#     ###convert to df
#     keys_z = zfun.f_keys_z()
#     keys_p7 = per.f_season_periods(keys=True)
#     new_index_p7zn = pd.MultiIndex.from_product([keys_p7, keys_z, fert_info.index])
#     alloc_p7zn = pd.Series(alloc_p7zn.ravel(), index=new_index_p7zn)
#     return alloc_p7zn


def f_seeding_cost(r_vals):
    '''
    Machinery costs required to seed 1 hectare of each crop on each LMU ($/ha).

    Cost is broken into fuel cost and repairs and maintenance cost. Repairs and maintenance includes both the tractor and the seeder.
    It is entered as an input for the base soil type and then adjusted for different crops and and soils by a user defined scalar.

    '''
    ##fuel cost ($/ha)
    fuel_cost_l = fuel_use_seeding() * fuel_price()

    ##seeder r&m ($/ha)
    base_rm = uinp.mach[pinp.mach['option']]['tillage_maint'] /  uinp.mach[pinp.mach['option']]['seeding_rate_base']
    tillage_maint_lmu_adj = pinp.mach['tillage_maint_lmu_adj']
    ##equals r&m on base lmu x lmu adj factor
    rm_cost_l = base_rm * tillage_maint_lmu_adj

    ##Total cost seeding on each lmu $/ha.
    seeding_cost_l = fuel_cost_l + rm_cost_l
    seeding_cost_l = seeding_cost_l.squeeze(axis=1)

    ##gets the cost allocation (includes interest)
    seeding_cost_allocation_p7zp5, seeding_wc_allocation_c0p7zp5 = f1_seed_cost_alloc()

    ##reindex with lmu so alloc can be mul with seeding_cost_l
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p5, seeding_cost_l.index])
    seeding_cost_allocation_p7zp5l = seeding_cost_allocation_p7zp5.reindex(columns)
    columns = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, seeding_cost_l.index])
    seeding_wc_allocation_c0p7zp5l = seeding_wc_allocation_c0p7zp5.reindex(columns)

    ##mul costs and allocation
    seeding_cost_p7zp5l = seeding_cost_allocation_p7zp5l.mul(seeding_cost_l, level=3)
    seeding_wc_c0p7zp5l = seeding_wc_allocation_c0p7zp5l.mul(seeding_cost_l, level=4)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, seeding_cost_p7zp5l, 'seeding_cost', mask_season_p7z[:,:,na,na], z_pos=-3)
    return seeding_cost_p7zp5l, seeding_wc_c0p7zp5l

def f_contract_seed_cost(r_vals):
    '''
    Contract seeding cost in each cashflow period. Currently, contract cost is the same for all lmus and crops.
    '''
    ##gets the cost allocation (includes interest)
    seeding_cost_allocation_p7zp5, seeding_wc_allocation_c0p7zp5 = f1_seed_cost_alloc()

    ##cost to contract seed 1ha
    seed_cost = uinp.price['contract_seed_cost']
    contract_seeding_cost_p7zp5 = seeding_cost_allocation_p7zp5 * seed_cost
    contract_seeding_wc_c0p7zp5 = seeding_wc_allocation_c0p7zp5 * seed_cost

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, contract_seeding_cost_p7zp5, 'contractseed_cost', mask_season_p7z[:,:,na], z_pos=-2)
    return contract_seeding_cost_p7zp5, contract_seeding_wc_c0p7zp5

########################################
#late seeding & dry seeding penalty    #
########################################

def f_sowing_timeliness_penalty(r_vals):
    '''
    Calculates the biomass penalty in each mach period due to effective germination date - kg/ha/period/crop.

    The timelines of sowing can have a large impact on crop biomass. AFO accounts for this using a
    biomass penalty. Late sowing receives a biomass reduction because the crop has less time to mature (e.g. shorter
    growing season) and grain filling often occurs during hotter, drier conditions :cite:p:`RN121, RN122`.
    Early sowing can also incur a yield penalty mainly due to frost.

    The biomass reduction is cumulative per day, so the longer sowing is
    delayed the larger the biomass reduction. Biomass penalty reduces grain available to sell and
    reduces stubble production. The penalty can be customised for each LMU because frost effects can be altered by
    the LMU topography and soil type. For example, sandy soils are more affected by frost because the lower
    moisture holding capacity reduces the heat buffering from the soil.

    The assumption is that seeding is done evenly throughout a given period. In reality this is wrong e.g. if a
    period is 5 days long but the farmer only has to sow 20ha they will do it on the first day of the period not
    4ha each day of the period. Therefore, the calculation overestimates the biomass penalty. The error can be
    reduced by having shorter p5 periods.

    .. note:: The penalty represented in this function is based on germination date. Therefore, dry seeding occurs the same penalty
        as sowing at the break of season. Other risks associated with dry sowing such as less effective weed control (i.e. crops germinate at
        the same time as the weeds so you miss out on a knock down spray opportunity), poor crop emergence (if
        opening rains are spasmodic patchy crop germination is possible and early crop vigour may be absent without
        adequate follow up rain) :cite:p:`RN119` are represented
        in the model via the biomass/yield inputs because dry sown crop are separate landuses.

    '''
    ##inputs
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    seeding_penalty_k_p = pinp.crop['seeding_yield_penalty']
    seeding_penalty_lmu_scalar_l_p = pinp.crop['seeding_penalty_lmu_scalar_lp']
    seeding_penalty_scalar_z_k = zfun.f_seasonal_inp(pinp.crop['seeding_penalty_scalar_kz'], axis=1).T
    mach_periods = per.f_p_dates_df()
    mach_periods_start_p5z = mach_periods.values[:-1]
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = sinp.general['i_idx_k1']
    keys_p7 = per.f_season_periods(keys=True)
    keys_l = pinp.general['i_lmu_idx']
    len_z = len(keys_z)
    len_k = len(keys_k)
    prob_z = zfun.f_z_prob()


    ##calc the yield penalty for each day of the year
    date_p = seeding_penalty_k_p.columns #inputs dates
    seeding_penalty_zkp = seeding_penalty_k_p.values * seeding_penalty_scalar_z_k.values[:,:,na]
    doy_p0 = np.arange(364)
    a_p_p0 = fun.f_next_prev_association(date_p,doy_p0, 1, "right")
    seeding_penalty_zkp0 = seeding_penalty_zkp[:,:,a_p_p0]

    ##accumulated yield penalty from break - yield relative to sowing at the break (this could be a positive number if sowing at the break is not optimal ie due to frost)
    adj_seeding_penalty_zkp0 = seeding_penalty_zkp0 * (doy_p0>=season_break_z[:,na,na])
    cum_seeding_penalty_zkp0 = np.cumsum(adj_seeding_penalty_zkp0, axis=-1)

    ##yield penalty relative to sowing at the optimum time.
    seeding_penalty_zkp0 = cum_seeding_penalty_zkp0 - np.max(cum_seeding_penalty_zkp0, axis=-1, keepdims=True)

    ##the penalty before the start of dry sowing is equal to the penalty at the end of the year
    seeding_penalty_zkp0[:,:,doy_p0 < pinp.crop['dry_seed_start']] = seeding_penalty_zkp0[:,:,-1:]

    ##sowing before the break (dry seeding) has the same sowing penalty as sowing on the first day of wet seeding.
    ## note the penalty is the weighted average of all seasons that have not broken (because of the season clustering).
    ## dry seeding may occur a yield penalty due to more weed pressure or less seed vigour due to time in the ground but this penalty is handled in the yield input
    ### 1) Break-day penalty for each weather-year and crop:
    break_penalty_zk = seeding_penalty_zkp0[np.arange(len_z)[:, None], np.arange(len_k)[None, :], season_break_z[:, None].astype(int)]

    ### 2) For each calendar day p0, find which weather-years have NOT yet broken by that day:
    #    mask M[p0, z] = True if date_break_z[z] >= doy_p0[p0]
    M_p0z = (season_break_z[None, :] >= doy_p0[:, None])  # shape [P0, Z]

    ### 3) Build per-day weights from prob_z over only the not-yet-broken years; normalize each day.
    weights_p0z = M_p0z * prob_z[None, :]  # shape [P0, Z]
    weights_sums = weights_p0z.sum(axis=1, keepdims=True)  # shape [P0, 1]
    # Avoid divide-by-zero; if sum==0 (all years already broken), keep weights at 0
    weights_norm_p0z = np.divide(
        weights_p0z,
        np.where(weights_sums == 0.0, 1.0, weights_sums),
    )

    ### 4) Compute the per-day, probability-weighted average of break-day penalties across z':
    avg_break_penalty_kp0 = np.einsum('pz,zk->pk', weights_norm_p0z, break_penalty_zk).T  # [K, P0]

    ### 5) Assign this average to all *dry* days for each z.
    for z in np.arange(len_z):
        dry_mask_p0 = np.logical_and(
            doy_p0 >= pinp.crop['dry_seed_start'],
            doy_p0 < season_break_z[z]
        )  # shape [P0]
        # broadcast assign: left side [K, #dry_days] ← right side [K, #dry_days]
        seeding_penalty_zkp0[z][:, dry_mask_p0] = avg_break_penalty_kp0[:, dry_mask_p0]

    ##scale for lmu (l by p)
    seeding_penalty_lmu_scalar_lp0 = seeding_penalty_lmu_scalar_l_p.values[:,a_p_p0]
    seeding_penalty_zklp0 = seeding_penalty_zkp0[:,:,na,:] * seeding_penalty_lmu_scalar_lp0

    ###calc the average penalty for each p5 period (kg/ha)
    alloc_p5zp0 = fun.f_range_allocation_np(mach_periods.values[:,:,na], doy_p0)[:-1,:,:] #remove last p5 period it was just the end date.
    seeding_penalty_p5zkl = fun.f_weighted_average(seeding_penalty_zklp0, weights=alloc_p5zp0[:,:,na,na,:], axis=-1)

    ##convert to positive number because pyomo expecting a positive number as the penalty.
    seeding_penalty_p5zkl = np.abs(seeding_penalty_p5zkl)

    ##convert from yield penalty to biomass penalty
    harvest_index_k = uinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because yield penalty is inputted as the harvestable grain
    seeding_penalty_p5zkl = seeding_penalty_p5zkl / harvest_index_k[:,na]

    ##add p7 axis - needed so yield penalty can be combined with phase yield
    alloc_p7p5z = zfun.f1_z_period_alloc(mach_periods_start_p5z[na,:,:], z_pos=-1)
    penalty_p7p5zkl = seeding_penalty_p5zkl * alloc_p7p5z[...,na,na]

    ##put into df
    cols_p7p5zkl = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_z, keys_k, keys_l])
    penalty_p7p5zkl = pd.Series(penalty_p7p5zkl.ravel(), index=cols_p7p5zkl)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)

    fun.f1_make_r_val(r_vals, penalty_p7p5zkl, 'sowing_yield_penalty_p7p5zkl', mask_season_p7z[:, na, :, na, na], z_pos=-3)
    return penalty_p7p5zkl


# def f_stubble_penalty():
#     '''
#     Calculates the stubble penalty in each mach period (wet and dry seeding) due to sowing timeliness- kg/ha/period/crop.
#     '''
#     import CropResidue as stub
#     yield_penalty_p7p5zk = f_sowing_timeliness_penalty(stub=True) #late sowing yield reduction kg/ha/period
#     stub_production_k = stub.f_cropresidue_production() #stubble production per kg of grain yield
#     stub_penalty = yield_penalty_p7p5zk.mul(stub_production_k, level=-1)
#     return stub_penalty

#######################################################################################################################################################
#######################################################################################################################################################
#harv / hay making
#######################################################################################################################################################
#######################################################################################################################################################


#################################################
#harvesting                                     #
#################################################

def f_harv_rate_period():
    '''
    Harv rate (t/hr) in each harvest period for each crop.

    Tonnes harvested per hour in each period for each crop is calculated from the harvest rate per
    hectare (see harv_time_ha) and the average crop yield. The rate is then set to 0 if a crop can not
    be harvested in a period. Note, harv_time_ha includes a factor that accounts for the time when grain is not being
    harvested due to moving paddocks or waiting for the chaser bin.

    Harvesting can begin on different dates depending on the crop. For example, in Western Australia the
    harvest of canola often begins before cereals :cite:p:`RN89`. To represent this AFO represents harvest in
    multiple periods determined by the user. Each crop has an inputted harvest start date which determines
    the harvest period for that crop. There is no penalty for late harvesting, however, to capture the
    timeliness of completion, harvest can only occur in the dedicated harvest periods.

    '''
    ##season inputs through function
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0) #when the first crop begins to be harvested (e.g. when harv periods start)
    harv_period_lengths_z = np.sum(zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z + harv_period_lengths_z #when all harv is done
    start_harvest_crops = pinp.crop['start_harvest_crops']
    start_harvest_crops_kz = zfun.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1) #start harvest for each crop

    ##harv occur - note: some crops are not harvested in the early harv period
    mach_periods = per.f_p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]
    harv_occur_pkz = np.logical_and(mach_periods_start_pz[:,na,:] < harv_end_z,
                                    mach_periods_end_pz[:,na,:] > start_harvest_crops_kz)

    ##add p7 axis - needed so harv req can be combined with phase yield
    alloc_p7p5z = zfun.f1_z_period_alloc(mach_periods_start_pz[na,:,:], z_pos=-1)
    harv_occur_p7pkz = harv_occur_pkz * alloc_p7p5z[...,na,:]

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = start_harvest_crops.index
    keys_p7 = per.f_season_periods(keys=True)
    index_p7pkz = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_k, keys_z])
    harv_occur = pd.Series(harv_occur_p7pkz.ravel(), index=index_p7pkz)

    ##Grain harvested per harvest activity hr (t/hr) for each crop. The efficiency adjustment factor converts from harvest rate per rotor hour to harvest rate per activity hour.
    harv_rate = uinp.mach[pinp.mach['option']]['harvest_rate'].squeeze() * (1 - pinp.mach['harv_downtime_frac'])

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
    harv_end_z = harv_start_z + harv_period_lengths_z #when all harv is done

    ##does any harvest occur in given period
    mach_periods_start_pz = per.f_p_dates_df()[:-1]
    mach_periods_end_pz = per.f_p_dates_df()[1:]
    harv_occur_pz = np.logical_and(harv_start_z <= mach_periods_start_pz, mach_periods_start_pz < harv_end_z)

    ##max harv hour per period
    days_pz = (mach_periods_end_pz.values - mach_periods_start_pz.values)
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
    harv_cost_allocation_p7zp5, harv_wc_allocation_c0p7zp5 = fin.f_cashflow_allocation(p5_start_zp5, enterprise='crp', z_pos=-2)
    ###convert to df
    new_index_p7zp5 = pd.MultiIndex.from_product([keys_p7,keys_z,keys_p5])
    harv_cost_allocation_p7zp5 = pd.Series(harv_cost_allocation_p7zp5.ravel(), index=new_index_p7zp5)
    new_index_c0p7zp5 = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z,keys_p5])
    harv_wc_allocation_c0p7zp5 = pd.Series(harv_wc_allocation_c0p7zp5.ravel(), index=new_index_c0p7zp5)
    return harv_cost_allocation_p7zp5, harv_wc_allocation_c0p7zp5


def f_harvest_cost(r_vals):
    '''
    Cost of harvest in each cashflow period ($/hr).

    The cost of harvesting for one hour is the same for each crop. However, the tonnes harvested
    per hour varies for different crops resulting in a different harvest cost per tonne.
    Harvest cost includes fuel, oil, grease and r&m. Oil, grease repairs and maintenance are calculated
    as a factor of fuel cost.

    '''
    ##allocation
    harv_cost_allocation_p7zp5, harv_wc_allocation_c0p7zp5 = f1_harv_cost_alloc()

    ##cost
    ##fuel used L/hr - same for each crop
    fuel_used = uinp.mach[pinp.mach['option']]['harv_fuel_consumption']
    fuel_cost_hr = fuel_used * fuel_price()
    ##harvester r&m $/hr
    cost_harv_rm = uinp.mach[pinp.mach['option']]['harvest_maint']
    harvest_maint_scalar_k = uinp.mach[pinp.mach['option']]['harvest_maint_scalar']
    rm_cost_harv_k = harvest_maint_scalar_k * cost_harv_rm
    ##cost (fuel + r&m)
    cost_harv = (fuel_cost_hr + rm_cost_harv_k) * (1 - pinp.mach['harv_downtime_frac']) #convert from $/rotor hr to $/activity hr.
    ##cost of other machinery linked to harvest - this is input in $/ activity hour. It does not need to be adjusted by harv downtime.
    truck_chaser_rm = uinp.mach[pinp.mach['option']]['truck_chaser_rm']
    ##return fuel and oil cost plus r & m ($/hr)
    harv_cost_k = (cost_harv + truck_chaser_rm).squeeze(axis=1)
    
    ##reindex with lmu so alloc can be mul with harv_cost
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns_p7zp5k = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p5, harv_cost_k.index])
    harv_cost_allocation_p7zp5k = harv_cost_allocation_p7zp5.reindex(columns_p7zp5k)
    columns_c0p7zp5k = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, harv_cost_k.index])
    harv_wc_allocation_c0p7zp5k = harv_wc_allocation_c0p7zp5.reindex(columns_c0p7zp5k)

    ##mul costs and allocation
    harv_cost_p7zp5k = harv_cost_allocation_p7zp5k.mul(harv_cost_k, level=-1)
    harv_wc_c0p7zp5k = harv_wc_allocation_c0p7zp5k.mul(harv_cost_k, level=-1)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, harv_cost_p7zp5k, 'harvest_cost', mask_season_p7z[:,:,na,na], z_pos=-3)
    return harv_cost_p7zp5k, harv_wc_c0p7zp5k


#########################
#contract harvesting    #
#########################

def f_contract_harv_rate():
    '''
    Grain harvested per hr by contractor (t/rotor hr).

    Contract harvest hours activity is rotor hours (i.e. it doesn’t include a factor for filling up/moving)
    because the cost is input in $/rotor hr.

    '''
    ##season inputs through function
    harv_start_z = zfun.f_seasonal_inp(pinp.period['harv_date'], numpy=True, axis=0) #when the first crop begins to be harvested (e.g. when harv periods start)
    harv_period_lengths_z = np.sum(zfun.f_seasonal_inp(pinp.period['harv_period_lengths'], numpy=True, axis=1), axis=0)
    harv_end_z = harv_start_z + harv_period_lengths_z #when all harv is done
    start_harvest_crops = pinp.crop['start_harvest_crops']
    start_harvest_crops_kz = zfun.f_seasonal_inp(start_harvest_crops.values, numpy=True, axis=1) #start harvest for each crop

    ##harv occur - note: some crops are not harvested in the early harv period
    mach_periods = per.f_p_dates_df()
    mach_periods_start_pz = mach_periods.values[:-1]
    mach_periods_end_pz = mach_periods.values[1:]
    harv_occur_pkz = np.logical_and(mach_periods_start_pz[:,na,:] < harv_end_z,
                                    mach_periods_end_pz[:,na,:] > start_harvest_crops_kz)

    ##add p7 axis - needed so machinery can be linked with phases (machinery just has a p5 axis)
    alloc_p7p5z = zfun.f1_z_period_alloc(mach_periods_start_pz[na,:,:], z_pos=-1)
    harv_occur_p7pkz = harv_occur_pkz * alloc_p7p5z[...,na,:]

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p5 = mach_periods.index[:-1]
    keys_k = start_harvest_crops.index
    keys_p7 = per.f_season_periods(keys=True)

    index_p7pkz = pd.MultiIndex.from_product([keys_p7, keys_p5, keys_k, keys_z])
    harv_occur = pd.Series(harv_occur_p7pkz.ravel(), index=index_p7pkz)

    ##Grain harvested per rotor hr (t/hr) for each crop.
    harv_rate = uinp.mach_general['contract_harvest_rate'].squeeze()

    ##combine harv rate and harv_occur
    harv_rate = harv_occur.mul(harv_rate, level=2)

    return harv_rate
#print(contract_harv_rate())


def f_contract_harvest_cost(r_vals):
    '''
    Cost of contract harvest in each cashflow period ($/rotor hr).
    '''
    ##allocation
    harv_cost_allocation_p7zp5, harv_wc_allocation_c0p7zp5 = f1_harv_cost_alloc()

    ##contract harv cost
    contract_harv_cost_k = uinp.price['contract_harv_cost'].squeeze(axis=1) #contract harvesting cost for each crop ($/hr)
    
    ##reindex with lmu so alloc can be mul with harv_cost
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p5 = per.f_p_dates_df().index[:-1]
    columns_p7zp5k = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p5, contract_harv_cost_k.index])
    contract_harv_cost_allocation_p7zp5k = harv_cost_allocation_p7zp5.reindex(columns_p7zp5k)
    columns_c0p7zp5k = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p5, contract_harv_cost_k.index])
    contract_harv_wc_allocation_c0p7zp5k = harv_wc_allocation_c0p7zp5.reindex(columns_c0p7zp5k)

    ##mul costs and allocation
    contract_harv_cost_p7zp5k = contract_harv_cost_allocation_p7zp5k.mul(contract_harv_cost_k, level=-1)
    contract_harv_wc_c0p7zp5k = contract_harv_wc_allocation_c0p7zp5k.mul(contract_harv_cost_k, level=-1)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, contract_harv_cost_p7zp5k, 'contract_harvest_cost', mask_season_p7z[:,:,na,na], z_pos=-3)
    return contract_harv_cost_p7zp5k, contract_harv_wc_c0p7zp5k


#########################
#make hay               #
#########################
def f_hay_making_cost():
    '''
    Cost to make hay ($/t).

    Typically, hay making is completed by contract workers and generally hay is not a large component of a
    farming system. Therefore, currently contract hay making is the only option represented in AFO. There
    is a cost for mowing ($/ha), baling ($/t) and carting ($/t).

    Note: Currently it is assumed that hay is allocated into the same cashflow periods in all seasons.
    '''
    ##cost allocation
    hay_start = np.array([pinp.crop['hay_making_date']])
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    ###call allocation/interset function - needs to be numpy
    hay_cost_allocation_p7z,hay_wc_allocation_c0p7z = fin.f_cashflow_allocation(hay_start, enterprise='crp', z_pos=-1)
    ###convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7,keys_z])
    hay_cost_allocation_p7z = pd.Series(hay_cost_allocation_p7z.ravel(),index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    hay_wc_allocation_c0p7z = pd.Series(hay_wc_allocation_c0p7z.ravel(),index=new_index_c0p7z)


    ##hay making cost $/t
    mow_cost =  uinp.price['contract_mow_hay'] / uinp.mach_general['approx_hay_yield']
    bail_cost =  uinp.price['contract_bail'] 
    cart_cost = uinp.price['cart_hay']
    total_cost = mow_cost + bail_cost + cart_cost

    ##calc interest and allocate to cash period
    hay_cost_p7z = hay_cost_allocation_p7z * total_cost
    hay_wc_c0p7z = hay_wc_allocation_c0p7z * total_cost

    ##m allocation - hay can only be made in the season stage where the cost is incurred
    alloc_p7z = zfun.f1_z_period_alloc(hay_start[na], z_pos=-1)
    index_p7z = pd.MultiIndex.from_product([keys_p7,keys_z])
    hay_made_prov_p7z = pd.Series(alloc_p7z.ravel(),index=index_p7z)

    return hay_cost_p7z, hay_wc_c0p7z, hay_made_prov_p7z

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
    ##cost/hs= fuel costs + r&m
    cost = tractor_fuel + uinp.mach[pinp.mach['option']]['stubble_maint']
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
    time_n = 10/(width_df*uinp.mach[pinp.mach['option']]['spreader_speed']*uinp.mach[pinp.mach['option']]['spreader_eff']).squeeze()
    ##mulitiplied by a factor (spreader_proportion) 0 or 1 if the fert is applied at seeding (or a fraction if applied at both seeding and another time)
    spreader_proportion = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']]).squeeze()
    time_n = time_n.mul(spreader_proportion)
    return time_n

#time taken to driving to and from paddock and filling up
# hr/cubic m = ((ave distance to paddock *2)/speed + fill up time)/ spreader capacity  # *2 because to and from paddock
def time_tonne():
    '''Time taken to fill up spreader and drive to and from paddock (hr/t).

    This represents the time driving to and from the paddock and filling up. This is dependent on
    the density of the fertiliser (e.g. more time would be required filling and traveling to
    spread 1 tonne of a lower density fertiliser).

    '''
    ##calc time taken to fill up spreader and drive to and from paddock
    time_cubic = (((pinp.mach['ave_pad_distance'] *2) /uinp.mach[pinp.mach['option']]['spreader_speed']
                   + uinp.mach[pinp.mach['option']]['time_fill_spreader'])
                  /uinp.mach[pinp.mach['option']]['spreader_cap'])

    ##convert from meters cubed to tonne - divide by conversion (density) because lighter ferts require more filling up time per tonne
    fert_density_n1 = uinp.general['i_fert_info_n1']['fert_density']
    a_ferttype_k_n = pinp.crop['i_a_ferttype_k_n']
    fert_density_kn = a_ferttype_k_n.replace(fert_density_n1)

    ##convert from m3 to tonnes
    time_kn = time_cubic / fert_density_kn

    ##mulitiplied by a factor (spreader_proportion) 0 or 1 if the fert is applied at seeding (or a fraction if applied at both seeding and another time)
    spreader_proportion_n = pd.DataFrame([pinp.crop['fert_info']['spreader_proportion']]).squeeze()
    time_kn = time_kn.mul(spreader_proportion_n, level=1)
    return time_kn

###################
#application cost # *remember that lime application only happens every 4 yrs - accounted for in the passes inputs
################### *used in crop pyomo

def spreader_cost_hr():
    '''
    Cost to spread each fertiliser for one hour.

    Spreading cost per hour includes tractor costs and spreader costs. Tractor costs consist of fuel, oil,
    grease and r&m. Spreader costs consist of just repairs and maintenance.

    Used to determine both fertiliser application cost per hour and per ha.
    '''
    ##tractor fuel cost
    tractor_fuel = uinp.mach[pinp.mach['option']]['spreader_fuel']*fuel_price()
    ##cost/hr= tractor costs + spreader(r&m)
    cost = tractor_fuel + uinp.mach[pinp.mach['option']]['spreader_maint']
    return cost

#######################################################################################################################################################
#######################################################################################################################################################
#chem
#######################################################################################################################################################
#######################################################################################################################################################

###########################
#chem application time   # used in labour crop, defined here because it uses inputs from the different mach options which are consolidated at the top of this sheet
###########################

def spray_time_ha():
    '''
    Time taken to spray 1ha (including filling up etc).

    This is dependent on the sprayer width, speed and field efficiency (accounts for overlap,
    filling up time and turing).

    '''
    ##rate per machine hour (ha/hr)
    spraying_rate = uinp.mach[pinp.mach['option']]['spraying_rate']
    ##convert to hr/ha
    spraying_rate = 1 / spraying_rate
    ##adjust for time spent filling up (mach hr/ha to activity hr/ha)
    spraying_rate = spraying_rate / (1-pinp.mach['spray_downtime_frac'])
    return spraying_rate

   

###################
#application cost # 
################### *used in crop pyomo

def spraying_cost_hr():
    '''
    Chemical application cost per hour of spraying activity (includes filling up).

    The cost of spraying per hectare is calculated based on the time to spray a hectare (see spray_time_ha)
    and the cost to spray per hour. Spraying cost per hour includes tractor costs and sprayer costs.
    Tractor costs consist of fuel, oil, grease and r&m. Sprayer costs consist of just repairs and maintenance.

    Typically, all spraying is done at the same speed and the rate of application is simply
    controlled by the chemical concentration in the spray. Thus, the cost of spraying is the same for
    all chemicals and on all LMUs.

    '''
    ##tractor costs = fuel + r&m + oil&grease
    tractor_fuel = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']*fuel_price()
    ##cost/machine hr= tractor costs + sprayer(r&m)
    cost = tractor_fuel + uinp.mach[pinp.mach['option']]['sprayer_maint']
    ##convert to cost per spraying activity hr
    cost = cost * (1 - pinp.mach['spray_downtime_frac'])
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
    return value


##value of gear used for seed. This is used to calculate the variable depreciation linked to seeding activity
def f_seeding_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'seeding allocation'])
    return value

def f_spray_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'spraying allocation'])
    return value

def f_spread_gear_clearing_value():
    value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'] * uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'spreading allocation'])
    return value

##total machine value - used to calc asset value, fixed dep and insurance
def f_total_clearing_value():
    total_clearing_value = sum(uinp.mach[pinp.mach['option']]['clearing_value'].loc[:,'value'])
    ##all is incurred in the last p7 period (although it could occur in any period it doesn't make a difference)
    keys_p7 = per.f_season_periods(keys=True)
    total_clearing_value = pd.Series(total_clearing_value,index=keys_p7[-1:])
    return total_clearing_value

#########################
#fixed depreciation     #
#########################


#total value of crop gear x dep rate x number of crop gear
def f_fix_dep(r_vals):
    '''Fixed depreciation on machinery

    Fixed depreciation captures obsolescence costs and is incurred every year independent of equipment uses.
    It is simply calculated based on the total clearing sale value of equipment and the fixed rate of depreciation.
    '''
    fixed_dep_p7 = f_total_clearing_value() * uinp.finance['fixed_dep']
    fun.f1_make_r_val(r_vals, fixed_dep_p7, 'fixed_dep_p7')
    return fixed_dep_p7


####################################
#variable seeding depreciation     #
####################################

def f_seeding_dep(r_vals):
    '''
    Average variable dep for seeding $/ha.

    Variable depreciation is use depreciation and is dependent on the number of hours the equipment is used.
    Seeding depreciation is calculated based on the time taken to sow 1ha of each crop on each LMU and
    the rate of depreciation per hour of seeding.

    The rate of depreciation per hour is calculated based 'typical' scenario, which simplifies the
    calibration process. The user enters the percentage of depreciation incurred for sowing `x` hectares
    of crop. This is converted to a dollar cost per hour based on the seeding rate and the machinery value.

    '''
    ##variable depn rate is input as a percent depn in all harvest gear per machine hour (%/machine hr).
    dep_rate_per_hr = uinp.mach_general['i_variable_dep_hr_seeding']
    ###convert from rotor hours to harvest activity hours
    dep_rate_per_hr = dep_rate_per_hr * (1 - pinp.mach['seeding_downtime_frac'])

    ##determine dep per hour - equal to crop gear value x depn %
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    dep_hourly = seeding_gear_clearing_value * dep_rate_per_hr

    ##convert to dep per ha for each soil type - equals cost per hr x seeding rate (hrs per ha)
    rate_direct_drill_k_l = f_seed_time_lmus()
    dep_ha_kl = dep_hourly * rate_direct_drill_k_l.stack()

    ##allocate season period based on mach/labour period - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    alloc_p7p5z = zfun.f1_z_period_alloc(date_start_p5z[na,...], z_pos=-1)
    ###make df
    keys_p5 = mach_periods.index[:-1]
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    keys_k = sinp.general['i_idx_k']
    keys_l = rate_direct_drill_k_l.columns
    index_p7p5z = pd.MultiIndex.from_product([keys_p7,keys_p5,keys_z])
    alloc_p7p5z = pd.Series(alloc_p7p5z.ravel(), index=index_p7p5z)
    index_p7p5zkl = pd.MultiIndex.from_product([keys_p7,keys_p5,keys_z,keys_k,keys_l])
    alloc_p7p5zkl = alloc_p7p5z.reindex(index_p7p5zkl)

    ##allocate dep to p7
    rate_direct_drill_p7p5zkl = alloc_p7p5zkl.unstack([-2, -1]).mul(dep_ha_kl, axis=1).stack([0,1])

    ##r_vals store
    fun.f1_make_r_val(r_vals, dep_ha_kl, 'seeding_dep_ha_kl')
    fun.f1_make_r_val(r_vals, uinp.mach[pinp.mach['option']]['number_of_seeders'], 'number_seeding_gear')


    return rate_direct_drill_p7p5zkl


####################################
#variable harvest depreciation     #
####################################

def f_harvest_dep(r_vals):
    '''
    Average variable dep for harvesting $/activity hr.

    Variable depreciation is use depreciation and is dependent on the number of hours the equipment is used.
    The harvest activity is represented in hours so the variable depreciation is simply the rate of depreciation
    per hour of harvest.

    '''
    ##variable depn rate is input as a percent depn in all harvest gear per rotor hour (%/rotor hr).
    dep_rate_per_hr = uinp.mach_general['i_variable_dep_hr_harv']
    ###convert from rotor hours to harvest activity hours
    dep_rate_per_hr = dep_rate_per_hr * (1 - pinp.mach['harv_downtime_frac'])

    ##determine dep per hour - equal to harv gear value x dep %
    dep_hourly = harvest_gear_clearing_value() * dep_rate_per_hr

    ##allocate season period based on mach/labour period - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    mach_periods = per.f_p_dates_df()
    date_start_p5z = mach_periods.values[:-1]
    alloc_p7p5z = zfun.f1_z_period_alloc(date_start_p5z[na,...], z_pos=-1)

    ##make df
    keys_p5 = mach_periods.index[:-1]
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    index_p7p5z = pd.MultiIndex.from_product([keys_p7,keys_p5,keys_z])
    alloc_p7p5z = pd.Series(alloc_p7p5z.ravel(), index=index_p7p5z)

    ##store r_vals
    fun.f1_make_r_val(r_vals, dep_hourly, 'harv_dep_hourly')
    fun.f1_make_r_val(r_vals, uinp.mach[pinp.mach['option']]['number_of_harvesters'], 'number_of_harvesters')

    return alloc_p7p5z * dep_hourly


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
    start = np.array([uinp.mach_general['insurance_date']])
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    ###call allocation/interset function - needs to be numpy
    insurance_cost_allocation_p7z,insurance_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='crp', z_pos=-1)
    ###convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7,keys_z])
    insurance_cost_allocation_p7z = pd.Series(insurance_cost_allocation_p7z.ravel(),index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0,keys_p7,keys_z])
    insurance_wc_allocation_c0p7z = pd.Series(insurance_wc_allocation_c0p7z.ravel(),index=new_index_c0p7z)
    
    ##calc interest and allocate to cash period
    insurance_cost_p7z = insurance_cost_allocation_p7z * insurance_cost
    insurance_wc_c0p7z = insurance_wc_allocation_c0p7z * insurance_cost

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, insurance_cost_p7z, 'mach_insurance', mask_season_p7z, z_pos=-1)
    return insurance_cost_p7z.to_dict(), insurance_wc_c0p7z.to_dict()


#########################
#emissions              #
#########################
def f_seeding_harv_fuel_emissions(r_vals):
    '''
    Counts emissions from fuel use for harvest and seeding.

    Harvest is linked to personal and contract harvest activities.

    Seeding is linked to personal and contract seeding activities.

    :return:
    '''

    ##v_harv_hours and v_contractharv_hours
    harv_fuel_used = uinp.mach[pinp.mach['option']]['harv_fuel_consumption'] #fuel used L/hr - same for each crop
    harv_fuel_used = np.array([harv_fuel_used])
    ###convert to emissions
    co2_harv_fuel_co2e, ch4_harv_fuel_co2e, n2o_harv_fuel_co2e = efun.f_fuel_emissions(harv_fuel_used)
    total_co2e_fuel_harv = co2_harv_fuel_co2e + ch4_harv_fuel_co2e + n2o_harv_fuel_co2e

    ##v_seeding_machdays and v_contractseeding_ha
    seeding_fuel_ha_l = fuel_use_seeding().squeeze(axis=1)
    ###convert to emissions
    co2_seeding_fuel_co2e_l, ch4_seeding_fuel_co2e_l, n2o_seeding_fuel_co2e_l = efun.f_fuel_emissions(seeding_fuel_ha_l)
    total_co2e_fuel_seeding_l = co2_seeding_fuel_co2e_l + ch4_seeding_fuel_co2e_l + n2o_seeding_fuel_co2e_l

    ##store r_vals
    fun.f1_make_r_val(r_vals, total_co2e_fuel_seeding_l, 'co2e_seeding_fuel_l')
    fun.f1_make_r_val(r_vals, total_co2e_fuel_harv, 'co2e_harv_fuel')

    return total_co2e_fuel_seeding_l, total_co2e_fuel_harv



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
    hay_making_cost, hay_making_wc, hay_made_prov_p7z  = f_hay_making_cost()
    biomass_penalty = f_sowing_timeliness_penalty(r_vals)
    # stubble_penalty = f_stubble_penalty()
    poc_grazing_days = f_poc_grazing_days().stack()
    fixed_dep = f_fix_dep(r_vals)
    harv_dep = f_harvest_dep(r_vals)
    seeding_gear_clearing_value = f_seeding_gear_clearing_value()
    seeding_dep = f_seeding_dep(r_vals)
    insurance_cost, insurance_wc = f_insurance(r_vals)
    mach_asset_value = f_total_clearing_value()
    total_co2e_fuel_seeding_l, total_co2e_fuel_harv = f_seeding_harv_fuel_emissions(r_vals)

    ##add inputs that are params to dict
    params['number_seeding_gear'] = uinp.mach[pinp.mach['option']]['number_of_seeders']
    params['number_harv_gear'] = uinp.mach[pinp.mach['option']]['number_of_harvesters']
    params['seeding_delays'] = pinp.mach['seeding_delays']
    params['harv_delays'] = pinp.mach['harv_delays']

    ##create non seasonal params
    params['seed_rate'] = seedrate.to_dict()
    params['contract_harv_rate'] = contract_harv_rate.to_dict()
    params['fixed_dep'] = fixed_dep.to_dict()
    params['harv_dep'] = harv_dep.to_dict()
    params['seeding_gear_clearing_value'] = seeding_gear_clearing_value
    params['seeding_dep'] = seeding_dep.to_dict()
    params['mach_asset_value'] = mach_asset_value.to_dict()
    params['co2e_fuel_seeding_l'] = total_co2e_fuel_seeding_l.to_dict()
    params['co2e_fuel_harv'] = total_co2e_fuel_harv


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
    params['hay_made_prov_p7z'] = hay_made_prov_p7z.to_dict()
    params['max_harv_hours'] = max_harv_hours.to_dict()
    params['contract_harvest_cost'] = contract_harvest_cost.to_dict()
    params['contract_harvest_wc'] = contract_harvest_wc.to_dict()
    params['biomass_penalty'] = biomass_penalty.to_dict()
    # params['stubble_penalty'] = stubble_penalty.to_dict()
    params['poc_grazing_days'] = poc_grazing_days.to_dict()
    params['insurance'] = insurance_cost
    params['insurance_wc'] = insurance_wc


