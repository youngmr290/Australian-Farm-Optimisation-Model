"""

author: young

The phase module is driven by the inputs [#i]_ for yield production, fertiliser and chemical
requirements for each rotation phase on each LMU. For pasture phases this module only generates data for
fertiliser, chemical and seed (if resown) requirement. Growth, consumption etc is generated in the
pasture module. AFO can then optimise the area of each rotation
on each LMU. AFO does not currently simulate the biology of crop plant growth under different technical
management. Thus, the model is unable to optimise technical aspects of cropping such as timing and
level of controls [#j]_. However, the user has the capacity to do this more manually by altering the
inputs (more like the technique of simulation modelling) or by including additional land uses
which represent varying levels of production and controls. When determining the inputs for each
rotation the user must consider the rotation history. The rotation history can influence the soil
fertility, weed burden and disease and pest levels. These factors impact the potential yield and
the optimal level of controls.

There are two methods that can be used to generate cropping inputs for the model:

#. Manually enter the inputs for selected rotation phases:

    The user can manually input the fertiliser and chemical requirement of given phases in a rotation
    and the resulting yield. To do this accurately requires an in-depth knowledge of cropping in the
    location being modelled. Thus, the process is often done in collaboration with a consultant or
    specialist in the field. This input method can be limiting if the user is hoping to include a
    large number of rotation phases or landuses that are not well established in the given location
    because it can be difficult to determine accurate inputs.

#. Generate using simulation modelling:

    APSIM is a whole farm simulation model widely used in Australia. APSIM has detailed modules which
    use robust relationships to simulate plant growth. The parameters used in APSIM can be altered to
    represent plant growth in many different situations. For example, different soil conditions. A-F-O users
    can use APSIM to generate yield, fertiliser requirement, number of fertiliser applications, chemical
    requirement and number of chemical applications for a wide range of rotations.

.. [#i] Inputs – AFO parameters.
.. [#j] Controls – chemicals and fertilisers.


"""

#python modules
import pandas as pd
import numpy as np
import timeit
import datetime as dt
import sys

#AFO modules
import UniversalInputs as uinp
import StructuralInputs as sinp
import PropertyInputs as pinp
import Functions as fun
import Periods as per
import Mach as mac

  
def f1_mask_lmu(df, axis):
    lmu_mask = pinp.general['i_lmu_area'] > 0
    if axis==0:
        df = df.loc[lmu_mask]
    if axis==1:
        df = df.loc[:, lmu_mask]
    return df


def f1_rot_check():
    ##check that the rotations match the inputs. If not then re-run rotation generation. If still not the same
    # quit and leave error message (most likely the user needs to re-run APSIM)
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_yields = pinp.crop['yields']
    else:
        ### AusFarm ^need to complete
        base_yields

    phases_df = sinp.f_phases()

    if len(phases_df) != len(base_yields):
        ##if the rotations don't match inputs then rerun rotation generation.
        import RotGeneration
        RotGeneration.f_rot_gen()

        ##read in newly generated rotations and see if the inputs now match
        phases_df = sinp.f_phases()

        ##if they still don't match then the user will need to either re-run simulation model (apsim) or change rotgeneration to line up with the rotations that have been simulated.
        if len(phases_df) != len(base_yields):
            print('''WARNING: Rotations don't match inputs.
                   Things to check: 
                   1. if you have generated new rotations have you re-run AusFarm?
                   2. the named ranges in for the user defined rotations and inputs are all correct''')
            sys.exit()



########################
#price                 #
########################

def f_farmgate_grain_price(r_vals={}):
    '''

    Calculates the grain price received by the farmer.

    The farm gate grain price [#]_ is calculated for each grain pool. Different grain pools are included to
    represent different grain qualities. Depending on the grain variety used, the season and the farmers skill
    the grain produced will change quality and hence receive a different price. The price received by the farmer is the
    market price received less any selling costs. The selling costs includes the transport cost which are
    dependent on the location of the modelled farm, and the selling fees which often includes receival
    and testing fees, and government levies.

    The market price is inputted for three different price percentiles. This is extrapolated to return
    the market price for the specified percentile.

    .. [#] Farm gate price – price received by the farmer after fees.

    '''
    ##inputs
    grain_price_info_df = uinp.price['grain_price_info'] #grain info
    percentile_price_df = uinp.price['grain_price'] #grain price for 3 different percentiles
    grain_price_percentile = uinp.price['grain_price_percentile'] #price percentile to use

    ##extrapolate price for the selected percentile (can go beyond the data input range)
    grain_price_firsts = pd.Series()
    for k in percentile_price_df.index:
        grain_price_firsts[k] = fun.np_extrap(np.array([grain_price_percentile]), percentile_price_df.columns, percentile_price_df.loc[k].values)[0] #returns as one value in an array thus take [0]
    ##seconds price
    grain_price_seconds = grain_price_firsts * (1-grain_price_info_df['seconds_discount'])

    ##gets the price of firsts and seconds for each grain
    price_df = pd.DataFrame(columns=['firsts','seconds'])
    price_df['firsts'] = grain_price_firsts
    price_df['seconds'] = grain_price_seconds

    ##determine cost of selling
    cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance']
            + pinp.general['rail_cartage'] + uinp.price['flagfall'])
    tols= grain_price_info_df['grain_tolls']
    total_fees= cartage+tols
    farmgate_price = price_df.sub(total_fees, axis=0).clip(0)
    r_vals['farmgate_price'] = farmgate_price
    return farmgate_price


def f_grain_price(r_vals):

    '''
    Allocates grain price into a cashflow period and stores parameter data for pyomo.

    :return: Dict of farm gate price received for each grain in each cashflow period.
        
    '''
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.f_cashflow_periods()['start date']
    p_name = per.f_cashflow_periods()['cash period']
    farm_gate_price=f_farmgate_grain_price(r_vals)
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    cols = pd.MultiIndex.from_product([allocation.index, farm_gate_price.columns])
    farm_gate_price = farm_gate_price.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    grain_price =  farm_gate_price.mul(allocation,axis=1,level=0)
    r_vals['grain_price'] =  grain_price.T
    return grain_price.stack([0,1])
# a=grain_price()

#########################
#yield                  #
#########################
def f_rot_yield(for_stub=False):
    '''
    Calculates the yield for each rotation. Accounting for LMU, arable area, frost and harvested proportion.

    The crop yield for each rotation phase, on the base LMU [#]_, before frost and harvested proportion adjustment,
    is entered as an input. The yield is inputted assuming seeding was completed at the optimal time.
    The base yield inputs are read in from either the simulation output or
    from Property.xl depending on what the user has specified to do. The yield input is dependant on the
    rotation history and hence accounts for the level of soil fertility, weed burden, disease prominence,
    and how the current land use is affected by the existing levels of each in the rotation.

    To extrapolate the inputs from the base LMU to the other LMUs an LMU adjustment factor is
    applied which determines the yield on each other LMU as a proportion of the base LMU. The LMU adjustment
    factor accounts for the variation in yield on different LMUs when management is the same.

    The decision variable represented in the model is the yield per hectare on a given LMU. To account for
    the fact that LMUs are rarely 100% arable due to patches of rocks, gully’s, waterlogged area and uncleared
    trees the yield is adjusted by the arable proportion. (eg if wheat yields 4 t/ha on LMU5 and LMU5 is 80%
    arable then 1 unit of the decision variable will yield 3.2t of wheat).

    Crop yield can also be adversely impacted by frost during the plants flowing stage :cite:p:`RN144`. Thus,
    the yield of each rotation phase is adjusted by a frost factor. The frost factor can be customised for each
    crop which is required because different crops flower at different times, changing the impact probability of
    frost yield reduction. Frost factor can be customised for each LMU because frost effects can be altered by
    the LMU topography and soil type. For example, sandy soils are more affected by frost because the lower
    moisture holding capacity reduces the heat buffering from the soil.

    .. note:: Potentially frost can be accounted for in the inputs (particularly if the simulation model accounts
        for frost). The LMU yield factor must then capture the difference of frost across LMUS.

    Furthermore, as detailed in the machinery chapter, sowing timeliness can also impact yield. Dry sowing tends [#]_
    to incur a yield reduction due to forgoing an initial knockdown spray. While later sowing incurs a yield
    loss due to a reduced growing season. Additionally, during the harvesting process a small proportion of grain
    is split/spilt. This is accounted for by adjusting the yield by a harvest proportion factor.

    .. [#] Base LMU – standardise LMU to which other LMUs are compared against.
    .. [#] Dry sowing may not incur a yield penalty in seasons with a late break.

    :param for_stub: Boolean set to true when calculating the yield that is used to calculate total stubble production.
    :return: Dataframe of rotation yields - passed to pyomo and used to calc grain insurance & stubble handling cost

    '''
    ##read phases
    phases_df = sinp.f_phases()

    ##read in base yields
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_yields = pinp.crop['yields']
        base_yields = pinp.f_seasonal_inp(base_yields, axis=1)
        base_yields = base_yields.set_index([phases_df.index, phases_df.iloc[:,-1]])
    else:
        ### AusFarm ^need to add code for ausfarm inputs
        base_yields
    base_yields = base_yields.stack()

    ##colate other info
    yields_lmus = f1_mask_lmu(pinp.crop['yield_by_lmu'], axis=1) #soil yield factor
    seeding_rate = pinp.crop['seeding_rate'].mul(pinp.crop['own_seed'],axis=0)#seeding rate adjusted by if the farmer is using their own seed from last yr
    frost = f1_mask_lmu(pinp.crop['frost'], axis=1)  #frost
    proportion_grain_harv = pd.Series(pinp.stubble['proportion_grain_harv'], index=pinp.stubble['i_stub_landuse_idx'])
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0) #read in arable area df
    ##calculate yield - base yield * arable area * harv_propn * frost * lmu factor - seeding rate
    yield_arable_by_soil = yields_lmus.mul(arable) #mul arable area to the the lmu factor (easy because dfs have the same axis's).
    yields=yield_arable_by_soil.reindex(base_yields.index, axis=0, level=1).mul(base_yields,axis=0) #reindes and mul with base yields
    if for_stub:
        ###return yield for stubble before accounting for frost, seed rate and harv propn
        return yields
    else:
        frost_harv_factor = (1-frost).mul(proportion_grain_harv, axis=0) #mul these two fisrt because they have same index so its easy.
        yields=frost_harv_factor.reindex(yields.index, axis=0, level=1).mul(yields,axis=0) #reindes and mul with base yields
        seeding_rate=seeding_rate.reindex(yields.index, axis=0, level=1) #minus seeding rate
        yields=yields.sub(seeding_rate,axis=0).clip(lower=0) #we don't want negative yields so clip at 0 (if any values are neg they become 0)
        return yields.stack()


def f_grain_pool_proportions():
    '''Calculate the proportion of grain in each pool.

    The total adjusted yield is split into two pools (firsts and seconds) to represent the grain that does
    and does not meet the quality specifications. Grain that does not meet the specifications is downgraded
    and sold for a discount. Each grain pool is represented as a separate grain transfer constraint, providing
    the option for the model to optimise the grain outcome. For example, the model has the option to sell high
    quality grain (firsts) to market and retain the lower quality grain (seconds) for livestock feed.

    '''
    prop = uinp.price['grain_price_info'][['prop_firsts','prop_seconds']]
    prop.columns = sinp.general['grain_pools']
    return prop.stack()


#######
#fert #    
#######    
'''
1) determines fert cost allocation 
2) fert requirement for each rot phase
3) cost of fert for each rotation 
4) application cost per kg and application cost per ha 
    -per tonne; represents the difference in application time based on fert density - represents the filling up and traveling to the paddock time, ie it would require more filling and traveling time to spread 1t of a lighter (less dense) fert.
    -per ha; represents the time to spread 1ha - this depends how far each fert is chucked out of the spreader
5) sum together to get overall fert cost
'''




def f1_fert_cost_allocation():
    '''

    :return: Dataframe with the allocation of each fertiliser cost into cashflow periods.

    '''
    start_df = pinp.crop['fert_info']['app_date'] #needed for allocation func
    length_df = pinp.crop['fert_info']['app_len'].astype('timedelta64[D]') #needed for allocation func
    p_dates = per.f_cashflow_periods()['start date'] #needed for allocation func
    p_name = per.f_cashflow_periods()['cash period'] #needed for allocation func
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)
# t_allocation=f1_fert_cost_allocation()


def f_fert_req():
    '''
    Fert required by 1ha of each rotation phase (kg/ha) after accounting for arable area.

    The fertiliser requirement is broken into two sections. Firstly, fixed fertiliser which is the
    amount of each fertiliser that applies to a land use independent of the phase history (e.g. lime
    which is typically applied routinely irrelevant of the land use history). Secondly, variable
    fertiliser which is applied to a rotation phase based on both the current land use and the
    history. This method is necessary because crops have varying nutrient requirements, have
    varying methods to obtain nutrients from the soil and leave the soil in varying states (e.g.
    pulse crop are able to fix nitrogen which typically reduces their requirement for external
    nitrogen and also leaves the soil with more nitrogen for the following years).

    For both fixed and variable applications the fertiliser requirement for each rotation phase, for
    the base LMU is entered by the user or obtained from the simulation output for each rotation.
    The fertiliser requirement is then adjusted by an LMU factor and the arable area factor.

    '''
    ##read phases
    phases_df = sinp.f_phases()

    ##read in fert by soil
    fert_by_soil = f1_mask_lmu(pinp.crop['fert_by_lmu'], axis=1)
    ##read in fert
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_fert = pinp.crop['fert']
        base_fert = base_fert.T.set_index(['fert'], append=True).T
        base_fert = pinp.f_seasonal_inp(base_fert, axis=1)
        base_fert=base_fert.set_index([phases_df.index,phases_df.iloc[:,-1]])
    else:        
        ### AusFarm ^need to add code for ausfarm inputs
        base_fert
        base_fert = pd.DataFrame(base_fert, index = phases_df.iloc[:,-1])  #make the rotation and current landuse the index
    ##rename index
    base_fert.index.rename(['rot','landuse'],inplace=True)
    ##add the fixed fert - currently this does not have season axis so need to reindex to add season axis
    fixed_fert = pinp.crop['fixed_fert']
    keys_z = pinp.f_keys_z()
    columns = pd.MultiIndex.from_product([keys_z, fixed_fert.columns])
    fixed_fert = fixed_fert.reindex(columns, axis=1, level=1)
    base_fert = pd.merge(base_fert, fixed_fert, how='left', left_on='landuse', right_index = True)
    ##add cont pasture fert req
    base_fert = f_cont_pas(base_fert.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    base_fert = base_fert.droplevel(0,axis=0)
    ## adjust the fert req for each rotation by lmu
    fert_by_soil = fert_by_soil.stack() #read in fert by soil
    fert=base_fert.stack(level=0).mul(fert_by_soil,axis=1,level=0).stack()
    ##account for arable area
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0) #read in arable area df
    fert=fert.mul(arable, axis=0, level=2) #add arable to df
    return fert


def f_fert_passes():
    '''
    Hectares of fertilising required over arable area.

    For both fixed and variable fertiliser the number of applications for each rotation phase
    is entered by the user or obtained from the simulation output. The
    number of fertiliser applications is the same for all LMUs because it is assumed that the rate
    of application varies rather than the frequency of application.

    '''
    ##read phases
    phases_df = sinp.f_phases()

    ####read in passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        fert_passes = pinp.crop['fert_passes']
        fert_passes = fert_passes.T.set_index(['passes'], append=True).T
        fert_passes = pinp.f_seasonal_inp(fert_passes, axis=1)
        fert_passes = fert_passes.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the rotation and current landuse the index
    else:
        ### AusFarm
        fert_passes
        fert_passes = pd.DataFrame(fert_passes, index = phases_df.iloc[:,-1])  #make the current landuse the index
    ##rename index
    fert_passes.index.rename(['rot','landuse'],inplace=True)
    ####add the fixed fert
    fixed_fert_passes = pinp.crop['fixed_fert_passes']
    keys_z = pinp.f_keys_z()
    columns = pd.MultiIndex.from_product([keys_z, fixed_fert_passes.columns])
    fixed_fert_passes = fixed_fert_passes.reindex(columns, axis=1, level=1)
    fert_passes = pd.merge(fert_passes, fixed_fert_passes, how='left', left_on='landuse', right_index = True)
    ##add cont pasture fert passes
    fert_passes = f_cont_pas(fert_passes.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    fert_passes = fert_passes.droplevel(0, axis=0)
    ##adjust fert passes by arable area
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0)
    index = pd.MultiIndex.from_product([fert_passes.index, arable.index])
    fert_passes = fert_passes.reindex(index, axis=0,level=0)
    fert_passes=fert_passes.mul(arable,axis=0,level=1)
    return fert_passes.stack(level=0).swaplevel(1,2, axis=0) #stack season axis and swap the order so season is level==1


def f_fert_cost(r_vals):
    '''
    Cost of fertilising the arable areas. Includes the fertiliser cost and the application cost.

    The cost of fertilising is made up from the cost of the fertilisers its self, the cost getting
    the fertiliser delivered to the farm and the machinery cost of application (detailed in the machinery section).
    The cost is incurred in the cashflow period when it is applied. The assumption is that fertilizer is
    purchased shortly before application because farmers wait to see how the year unfolds before locking
    in a fertiliser plan.

    Fertiser application cost is broken into two components (detailed in the machinery section).

        #. Application cost per tonne ($/rotation)
        #. Application cost per ha ($/rotation)

    :return: Dataframe of fertiliser costs. Summed with other cashflow items at the end of the module

    '''
    ##call functions and read inputs used within this function
    fertreq = f_fert_req()
    allocation = f1_fert_cost_allocation()
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost 
    ##calc cost of actual fertiliser
    total_cost = allocation.mul(cost+transport).stack() #total cost = fert cost and transport. Here we also account for cost allocation
    phase_fert_cost=fertreq.mul(total_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg, sum the cost of all the ferts
    r_vals['phase_fert_cost'] = phase_fert_cost
    ##aplication cost per tonne
    application_cost = allocation.mul(mac.fert_app_cost_t()).stack() #mul app cost per tonne with fert cost allocation
    fert_app_cost_t=fertreq.mul(application_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg
    ##app cost per ha 
    ###call passes function (it has to be a separate function because it is used in crplabour.py as well
    fert_passes = f_fert_passes()
    ###add the cost for each pass
    fert_cost_ha = allocation.mul(mac.fert_app_cost_ha()).stack() #cost for 1 pass for each fert.
    fert_app_cost_ha = fert_passes.mul(fert_cost_ha,axis=1,level=1).sum(axis=1, level=0)
    r_vals['fert_app_cost'] = fert_app_cost_ha + fert_app_cost_t
    ##combine all costs - fert, app per ha and app per tonne    
    fert_cost_total= pd.concat([phase_fert_cost,fert_app_cost_t, fert_app_cost_ha],axis=1).sum(axis=1,level=0) #must include level so that all cols don't sum, had to switch this from .add to concat because for some reason on multiple iterations of the model add stoped working
    return fert_cost_total

def f_nap_fert_req():
    '''
    Fert applied to non arable pasture area.

    Fertiliser is applied to non-arable area in a pasture phases, it is not applied to
    non-arable pasture in a crop phase because the non-arable pasture in a crop phase is not able to
    be grazed until the end of the year, by which time it is rank and therefore it is a waste of
    money to fertilise. Fertiliser rate for non-arable areas can be adjusted separately to the arable area.

    '''
    ##read phases and add empty header level
    phases_df2 = sinp.f_phases()
    phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns,['']])  # make the df multi index so that when it merges with other df below the indexs remanin separate (otherwise it turn into a one leveled tuple)
    ##adj arable
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0)  # read in arable area df
    fertreq_na = pinp.crop['nap_fert'].reset_index().set_index(['fert','landuse'])
    fertreq_na = fertreq_na.mul(1 - arable)
    ##add cont pasture fert req
    fertreq_na = f_cont_pas(fertreq_na.unstack(0))
    ##merge with full df
    fertreq_na = pd.merge(phases_df2, fertreq_na, how='left', left_on=sinp.end_col(), right_index = True) #merge with all the phases, requires because different phases have different application passes
    fertreq_na = fertreq_na.drop(list(range(sinp.general['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return fertreq_na

def f_nap_fert_passes():
    '''
    Hectares of fertilising required over non arable area.

    '''
    ##read phases and add empty header level
    phases_df2 = sinp.f_phases()
    phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns,['']])  # make the df multi index so that when it merges with other df below the indexs remanin separate (otherwise it turn into a one leveled tuple)

    ##passes over non arable pasture area (only for pasture phases because for pasture the non arable areas also receive fert)
    passes_na = pinp.crop['nap_passes'].reset_index().set_index(['fert','landuse'])
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0) #need to adjust for only non arable area
    passes_na= passes_na.mul(1-arable) #adjust for the non arable area
    ##add cont pasture fert req
    passes_na = f_cont_pas(passes_na.unstack(0))
    ##merge with full df
    passes_na = pd.merge(phases_df2, passes_na, how='left', left_on=sinp.end_col(), right_index = True) #merge with all the phases, requires because different phases have different application passes
    passes_na = passes_na.drop(list(range(sinp.general['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return passes_na

def f_nap_fert_cost(r_vals):
    '''
    Cost of fertilising the non arable areas. Includes the fertiliser cost and the application cost.

    .. note:: Currently setup so that only pasture phases get fertiliser on the non arable areas
        hence it needs to be a separate function.

    '''
    allocation = f1_fert_cost_allocation()
    ##fert cost
    fertreq = f_nap_fert_req()
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost
    total_cost = allocation.mul(cost+transport).stack() #total cost = fert cost and transport. Here we also account for cost allocation
    fert_cost = fertreq.mul(total_cost, axis=1, level=1)/1000  #div by 1000 to convert to $/kg
    r_vals['nap_phase_fert_cost'] = fert_cost.sum(axis=1, level=0) #sum all fertilisers
    ##application cost per tonne
    app_cost_t = fertreq.mul(mac.fert_app_cost_t(), axis=1)/1000  #div by 1000 to convert to $/kg
    ##application cost per ha
    passes = f_nap_fert_passes()
    app_cost_ha = passes.mul(mac.fert_app_cost_ha(), axis=1) #cost for 1 pass for each fert.
    ##total application cost in each cash period
    total_app_cost = (app_cost_t+app_cost_ha).mul(allocation.stack(), axis=1, level=1)
    r_vals['nap_fert_app_cost'] = total_app_cost.sum(axis=1, level=0) #sum all fertilisers
    ##total fert and app cost
    nap_fert_cost = fert_cost+total_app_cost
    nap_fert_cost = nap_fert_cost.sum(axis=1, level=0) #mul app cost per tonne with fert cost allocation
    ##currently the non-arable fert inputs are the same for each season type. but need season axis so it can combine with other costs. So simply reindex to include season.
    nap_fert_cost = nap_fert_cost.unstack()
    keys_z = pinp.f_keys_z()
    index = pd.MultiIndex.from_product([nap_fert_cost.index, keys_z])
    nap_fert_cost = nap_fert_cost.reindex(index,axis=0, level=0)
    return nap_fert_cost.stack()

def f1_total_fert_req():
    '''returns the total fert req after accounting for arable area.
       this is used in the LabourCropPyomo'''
    fertreq_arable = f_fert_req()
    ##fert required on the non arable areas - only for pasture phases. Currently no season axis so need to reindex
    fert_na = f_nap_fert_req()
    fert_na = fert_na.unstack().reindex(fertreq_arable.unstack().index, axis=0).stack()
    ##add fert for arable area and fert for nonarable area
    fert_total = pd.concat([fertreq_arable, fert_na], axis=1).sum(axis=1, level=0)
    fert_req = fert_total.stack()
    return fert_req


#######################
#stubble handeling    #
#######################
'''
cost now associated with current yr because it requires knpwing the yield of the crop - hence under the new rotation phases that contain sets it isn;t possible to use the previous landuse to determine cost.
As a general rule stubble handling is not required if:
-the land use is a legume crop, pasture or lucerne
Other general rules about stubble handling:
- once cereal crops are big enough to yield more than 3.5t of grain/ha their stubble residue starts to become an issue when sowing the next crop
- once canola crops are big enough to yield more than 2.3t of grain/ha their stubble residue starts to become an issue when sowing the next crop
- wheat stubble is the most problematic
- barley stubble tends to be less problematic because: barley crops tend to get harvested at a lower height;  barley stubble seems to break down quicker than wheat; and livestock tend to graze barley stubbles harder.
Limitations with the way stubble is handled in this Table:
    The biggest factor determining of whether stubble will require handling is the amount of stubble present, which largely depends on season type (other factors include width between crop rows, what type of seeding setup is used, how low crop was harvested, if the header was equipped with a straw chopper). All of this makes the requirement for stubble handling hard to represent in a steady state model. Hence the probability-based approach adopted in this Table.
    This probability is calculated by dividing the average yield for that LMU by the critical grain yield. This means the likelihood of stubble handling being req’d increases for higher yielding soil types etc, which is logical. However, the probability isn’t accurately linked to the likelihood that stubble will actually require handling. For instance, just because the average steady-state wht yield of a LMU is 1.75t/ha doesn’t necessarily mean that the wheat stubble on that LMU will need handling 1.75/3.5 = 50% of the time.
    So in summary, these probabilities are fairly crude...
    additionally this new structure assumes that even if the preceeding landuse is pasture the current phase will still get handling cost (wasn't able to find an alternative way)
frost is not included because that doesn't reduce biomass
'''

def f_phase_stubble_cost(r_vals):
    '''
    Cost to handle stubble for 1 ha.

    The stubble handling cost per hectare is calculated based on machinery usage (see Mach.py).
    The cost is then adjusted by the probability of each rotation phase requiring handling which is
    determined by the ration of the crop yield to the critical threshold.

    General rules regarding stubble handling:

    #. Stubble handling is not required if the land use is a legume crop, pasture or lucerne
    #. Once cereal crops are big enough to yield more than 3.5t of grain/ha their stubble residue
       starts to become an issue when sowing the next crop.
    #. Wheat stubble is the most problematic
    #. Barley stubble tends to be less problematic because barley crops tend to get harvested at a lower
       height, barley stubble seems to break down quicker than wheat and livestock tend to graze barley stubbles harder.
    #. Once canola crops are big enough to yield more than 2.3t of grain/ha their stubble residue
       starts to become an issue when sowing the next crop.

    The biggest factor determining whether stubble will require handling is the amount of stubble present,
    which largely depends on season type (other factors include width between crop rows, what type of seeding
    setup is used, how low crop was harvested, if the header was equipped with a straw chopper). All of this
    makes the requirement for stubble handling hard to represent in a steady state model. Hence the probability-based
    approach adopted in this Table.
    This probability is calculated by dividing the average yield for that LMU by the critical grain yield.
    This means the likelihood of stubble handling being req’d increases for higher yielding soil types etc, which
    is logical. However, the probability isn’t accurately linked to the likelihood that stubble will actually
    require handling. For instance, just because the average steady-state wht yield of a LMU is 1.75t/ha doesn’t
    necessarily mean that the wheat stubble on that LMU will need handling 1.75/3.5 = 50% of the time.
    This structure assumes that even if the preceding landuse is pasture the current phase will still get
    handling cost this is because the rotation phases use sets so you can’t determine exactly which land use is
    before or after the current phase.


    .. note:: An improvement could be to include harvest index in the calculation of handling probability. Also,
        with seasonal variation represented the probability is not required.

    .. note:: arable area accounted for in the yield (it is the same as accounting for it at the end
        ie yield x 0.8 / threshold x cost == yield / threshold x cost x 0.8)
    '''
    ##first calculate the probability of a rotation phase needing stubble handling
    base_yields = f_rot_yield(for_stub=True).stack()
    stub_handling_threshold = pd.Series(pinp.stubble['stubble_handling'], index=pinp.crop['start_harvest_crops'].index)*1000  #have to convert to kg to match base yield
    probability_handling = base_yields.div(stub_handling_threshold, level = 1) #divide here then account for lmu factor next - because either way is mathematically sound and this saves some manipulation.
    probability_handling = probability_handling.droplevel(1).unstack()
    ##add the cost - this needs to be flexible because the cost may be over multiple periods
    stub_cost_alloc=mac.stubble_cost_ha().squeeze(axis=1)
    cols = pd.MultiIndex.from_product([probability_handling.columns, stub_cost_alloc.index])  
    handling_cost = probability_handling.reindex(cols,axis =1 , level = 0).mul(stub_cost_alloc,axis=1,level=1)
    stub_cost = handling_cost.stack([0])
    r_vals['stub_cost'] = stub_cost
    return stub_cost
# t_stubcost=f_phase_stubble_cost()

#print(timeit.timeit(fert_cost,number=10)/10)
#########################
#chemical               #
#########################

def f1_chem_cost_allocation():
    '''

    :return Dataframe with the allocation of each chemical cost into cashflow periods.

    '''
    start_df = pinp.crop['chem_info']['app_date'] #needed for allocation func
    length_df = pinp.crop['chem_info']['app_len'].astype('timedelta64[D]') #needed for allocation func
    p_dates = per.f_cashflow_periods()['start date'] #needed for allocation func
    p_name = per.f_cashflow_periods()['cash period'] #needed for allocation func
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)
# t_allocation=f1_chem_cost_allocation()
    
def f_chem_application():
    '''

    Number of applications of each chemical option for each rotation.

    AFO represents a customisable number of spraying options (e.g. pre seeding knock down, pre-emergent
    and post emergent) and the chemical cost of each. The number of applications of each spraying option for each
    rotation phase is entered by the user or obtained from the simulation output .
    The number of applications is dependent on the rotation
    history and the current landuse. This is because the initial levels of weed burden and fungi and
    pest presence are impacted by previous landuses. Furthermore, chemicals can be specific for certain
    crops and have varying levels of effectiveness for different weeds and diseases which levels are
    impacted by previous landuses. Therefore, the number of applications of each spray option is highly
    dependent on the rotation phase.

    Similar to fertiliser, the number of chemical applications is the
    same for all LMUs because it is assumed that the spray rate varies rather than the frequency of
    application. However, the area sprayed is adjusted by the arable proportion for each LMU. The assumption
    is that non arable areas do not receive any spray.

    '''
    ##read in chem passes
    ##read phases
    phases_df = sinp.f_phases()

    if pinp.crop['user_crop_rot']:
        ### User defined
        base_chem = pinp.crop['chem']
        base_chem = base_chem.T.set_index(['chem'], append=True).T
        base_chem = pinp.f_seasonal_inp(base_chem, axis=1)
        base_chem = base_chem.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the current landuse the index
    else:
        ### AusFarm ^need to add code for ausfarm inputs
        base_chem
        base_chem = pd.DataFrame(base_chem, index = [phases_df.index, phases_df.iloc[:,-1]])  #make the current landuse the index
    ##rename index
    base_chem.index.rename(['rot','landuse'],inplace=True)
    ##add cont pasture fert req
    base_chem = f_cont_pas(base_chem.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    base_chem = base_chem.droplevel(0, axis=0)
    ##arable area.
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0)
    #adjust chem passes by arable area
    index = pd.MultiIndex.from_product([base_chem.index, arable.index])
    base_chem = base_chem.reindex(index, axis=0,level=0)
    base_chem=base_chem.mul(arable,axis=0,level=1)
    return base_chem.stack(level=0).swaplevel(1,2, axis=0) #stack season axis and swap order

def f_chem_cost(r_vals):
    '''

    Calculates the cost of spraying for each rotation phase on each LMU.

    To simplify the input process, the cost of each spray option on the base LMU for each land use is
    entered as an input by the user. This saves the step of going from a volume of chemical to a cost
    which means AFO does not need to represent the large array of chemical options available. Making it
    easier to keep AFO up to date. The chemical cost of each spray option is adjustable by an LMU
    factor because the spray rate may vary for according to LMU (e.g. a higher chemical concentration
    may be used on LMU5 vs LMU1).

    The cost of spraying chemicals is made up from the cost of the chemical itself and the machinery
    cost of application (detailed in the machinery section). The
    chemical cost is incurred in the cashflow period when it is applied. The assumption is that
    chemical is purchased shortly before application because farmers wait to see how the year unfolds
    before locking in a spraying plan.


    :return: Total cost of chemical and application for each rotation phase - summed with other cashflow
        items at the end of this section.

    '''
    ##read phases
    phases_df = sinp.f_phases()

    ##read in necessary bits and adjust indexed
    i_chem_cost = pinp.crop['chem_cost'].sort_index()
    chem_by_soil = f1_mask_lmu(pinp.crop['chem_by_lmu'], axis=1) #read in chem by soil
    ##add cont pasture to chem cost array
    i_chem_cost = f_cont_pas(i_chem_cost)
    ##number of applications for each rotation
    chem_applications = f_chem_application()
    ##determine the total chemical cost of each rotation. eg: chem cost per application * number of applications
    index = pd.MultiIndex.from_arrays([phases_df.iloc[:,-1], phases_df.index], names=['landuse','rot']) #add phase letter to index so it can be merged with the cost per application for each phase
    t_chem_applications = chem_applications.unstack(level=(1,2)).reindex(index, axis=0, level=1).stack(level=(1,2)) #reindex so the array has same axis so it can be multiplied
    ###reindex cost and mul with number of applications.
    i_chem_cost = i_chem_cost.reindex(t_chem_applications.index, axis=0, level=0)
    chem_cost = t_chem_applications.mul(i_chem_cost).droplevel(0)
    ## adjust the chem cost for each rotation by lmu
    chem_by_soil1 = chem_by_soil.stack()
    chem_cost=chem_cost.unstack().mul(chem_by_soil1,axis=1).stack()
    ##application cost
    app_cost_ha = chem_applications * mac.chem_app_cost_ha()
    ##add cashflow periods and sum across each chem - have to do this to both chem cost and application so i can report them separately
    c_chem_allocation = f1_chem_cost_allocation().stack()
    chem_cost = chem_cost.mul(c_chem_allocation, axis=1,level=1).sum(axis=1, level=0)#first stack is required so that reindexing can occur (ie can't reindex a multi index with a multi index)
    app_cost_ha = app_cost_ha.mul(c_chem_allocation, axis=1,level=1).sum(axis=1, level=0)#first stack is required so that reindexing can occur (ie can't reindex a multi index with a multi index)
    r_vals['chem_cost'] = chem_cost
    r_vals['chem_app_cost_ha'] = app_cost_ha
    ##add application cost and chem cost
    total_cost = chem_cost.add(app_cost_ha)
    return total_cost


#########################
#misc cost              #
#########################
def f_seedcost(r_vals):
    '''

    Seed costs includes:

        - seed treatment
        - raw seed cost (incurred if seed is purchased as apposed to using last yrs seed)
        - crop insurance
        - arable area
    '''
    ##read phases and add two empty col levels
    phases_df3 = sinp.f_phases()
    phases_df3.columns = pd.MultiIndex.from_product([phases_df3.columns,[''],['']])  # make the df multi index so that when it merges with other df below the indexs remanin separate (otherwise it turn into a one leveled tuple)

    ##seasonal inputs
    seed_period_lengths = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    i_z_idx = pinp.f_keys_z()
    ##inputs
    seeding_rate = pinp.crop['seeding_rate']
    seeding_cost = pinp.crop['seed_info']['Seed cost'] #this is 0 if the seed is sourced from last yrs crop ie cost is accounted for by minusing from the yield
    grading_cost = pinp.crop['seed_info']['Grading'] 
    percent_graded = pinp.crop['seed_info']['Percent Graded'] 
    cost1 = pinp.crop['seed_info']['Cost1'] #cost ($/l) for dressing 1 
    cost2 = pinp.crop['seed_info']['Cost2'] #cost ($/l) for dressing 2 
    rate1 = pinp.crop['seed_info']['Rate1'] #rate (ml/100g) for dressing 1
    rate2 = pinp.crop['seed_info']['Rate2'] #rate (ml/100g) for dressing 2
    percent_dressed = pinp.crop['seed_info']['percent dressed'] #rate (ml/100g) for dressing 2
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0)
    ##adjust for arable area.
    seeding_rate = seeding_rate.mul(arable, axis=1)
    ##overall seed grading cost per tonne
    cost = grading_cost * percent_graded
    ##add seed cost
    cost = cost + seeding_cost
    ##add dressing 1 - need to convert to $/t first
    cost = cost + (cost1*rate1/100 * percent_dressed)
    ##add dressing 2 - need to convert to $/t first
    cost = cost + (cost2*rate2/100 * percent_dressed)
    ##account for seeding rate to determine actual cost (divide by 1000 to convert cost to kg)
    phase_cost = seeding_rate.mul(cost/1000,axis=0)
    ##add cost for cont pasture
    phase_cost = f_cont_pas(phase_cost)
    ##cost allocation
    start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
    length_z = np.sum(seed_period_lengths, axis=0).astype('timedelta64[D]')
    p_dates_c = per.f_cashflow_periods()['start date'].values
    p_name_c = per.f_cashflow_periods()['cash period'].iloc[:-1]
    allocation_cz = fun.range_allocation_np(p_dates_c[...,None], start_z, length_z, True)[:-1,...] #drop last row because that is just the end date of last period
    allocation_cz = pd.DataFrame(allocation_cz, index=p_name_c, columns=i_z_idx).stack()
    ##mul cost by allocation - need to align column headers first
    columns = pd.MultiIndex.from_product([phase_cost.columns, p_name_c, i_z_idx])
    phase_cost = phase_cost.reindex(columns, axis=1, level=0)
    phase_cost = phase_cost.stack(level=0).mul(allocation_cz, axis=1).unstack()
    ##merge
    rot_cost = pd.merge(phases_df3, phase_cost, how='left', left_on=sinp.end_col(), right_index = True)
    seedcost = rot_cost.drop(list(range(sinp.general['phase_len'])), axis=1).stack([1,2])
    r_vals['seedcost'] = seedcost
    return seedcost

def f_insurance(r_vals):
    '''
    Crop insurance cost.

    Crop insurance is typically based off the farmers estimation of yield in mid spring.
    This is not going to exactly be equal to final yield but it is closer than using the average yield.
    The small amount of error in this assumption will have little impact due to the small magnitude of
    financial impact of insurance.

    .. note:: arable area is already counted for by the yield calculation.
    '''
    ##first need to combine each grain pool to get average price
    grain_pool_proportions = f_grain_pool_proportions()
    ave_price = f_farmgate_grain_price().mul(grain_pool_proportions.unstack()).sum(axis=1)
    insurance=ave_price*uinp.price['grain_price_info']['insurance']/100  #div by 100 because insurance is a percent
    rot_insurance = f_rot_yield().mul(insurance, axis=0, level = 1)/1000 #divide by 1000 to convert yield to tonnes
    rot_insurance = rot_insurance.droplevel(1).unstack()
    ##cost allocation
    start = uinp.price['crp_insurance_date']
    p_dates = per.f_cashflow_periods()['start date']
    p_name = per.f_cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start)
    ##add cashflow period to col index
    rot_insurance.columns = pd.MultiIndex.from_product([rot_insurance.columns, [allocation]])
    insurance_cost = rot_insurance.stack([0])
    r_vals['insurance_cost'] = insurance_cost
    return insurance_cost




#########################
#total rot cost         #
#########################
'''
adds up all the different cashflows for pyomo.
includes
-fert cost (fert & application)
-stubble handling cost
- chem cost (inc application)
-seed cost
-crop insurance cost
'''
def f1_rot_cost(r_vals):
    '''collates all the rotation costs'''
    cost = pd.concat([f_fert_cost(r_vals),f_nap_fert_cost(r_vals),f_chem_cost(r_vals),f_seedcost(r_vals), f_insurance(r_vals),f_phase_stubble_cost(r_vals)],axis=1).sum(axis=1,level=0)
    return cost


#################
#sow            #
#################
    
def f_phase_sow_req():
    '''
    Area of seeding required for 1ha of each rotation.

    This accounts for arable area and includes any seeding (wet or dry or pasture).

    '''
    ##read phases
    phases_df = sinp.f_phases()

    ##sow = arable area
    arable = f1_mask_lmu(pinp.crop['arable'].squeeze(), axis=0)
    seeding_landuses = uinp.mach[pinp.mach['option']]['seeder_speed_crop_adj'].index
    phasesow = arable.reindex(pd.MultiIndex.from_product([seeding_landuses, arable.index]), axis=0, level=1)
    ##merge to rot phases
    phasesow = pd.merge(phases_df, phasesow.unstack(), how='left', left_on=sinp.end_col(), right_index = True)
    ##add current crop to index
    phasesow.set_index(sinp.end_col(), append=True, inplace=True)
    phase_sow = phasesow.drop(list(range(sinp.general['phase_len']-1)), axis=1).stack()
    return phase_sow

def f_sow_prov():
    '''
    Creates provide param for wet and dry sowing activities:

        - Area of wet seeding provided by 1ha of the wet seeding activity.
        - Area of dry seeding provided by 1ha of the dry seeding activity.

    This accounts for period and crop eg wet seeding activity only provides sowing to crop after the break.

    '''
    ##machine periods
    labour_period_p5z = per.f_p_dates_df()
    labour_period_start_p5z = labour_period_p5z.values[:-1]
    labour_period_end_p5z = labour_period_p5z.values[1:]

    ##wet sowing periods
    seed_period_lengths_pz = pinp.f_seasonal_inp(pinp.period['seed_period_lengths'],numpy=True,axis=1)
    wet_seed_start_z = per.f_wet_seeding_start_date().astype(np.datetime64)
    wet_seed_len_z = np.sum(seed_period_lengths_pz, axis=0).astype('timedelta64[D]')
    wet_seed_end_z = wet_seed_start_z + wet_seed_len_z
    period_is_wetseeding_p5z = (labour_period_start_p5z < wet_seed_end_z) * (labour_period_end_p5z > wet_seed_start_z)

    ##dry sowing periods
    dry_seed_start_z = np.datetime64(pinp.crop['dry_seed_start'])
    date_feed_periods = per.f_feed_periods().astype('datetime64')
    date_start_p6z = date_feed_periods[:-1]
    season_break_z = date_start_p6z[0]
    period_is_dryseeding_p5z = (labour_period_start_p5z < season_break_z) * (labour_period_end_p5z > dry_seed_start_z)

    ##make df
    keys_z = pinp.f_keys_z()
    keys_p5 = labour_period_p5z.index[:-1]
    wetseeding_prov_p5z = pd.DataFrame(period_is_wetseeding_p5z, index=keys_p5, columns=keys_z) * 1 # *1 to convert bool to number
    dryseeding_prov_p5z = pd.DataFrame(period_is_dryseeding_p5z, index=keys_p5, columns=keys_z) * 1 # *1 to convert bool to number

    ##add wet sow landuse axis
    dry_sown_landuses = sinp.landuse['dry_sown']
    wet_sown_landuses = sinp.landuse['C'] - dry_sown_landuses #can subtract sets to return differences
    wetseeding_prov_p5kz = wetseeding_prov_p5z.reindex(pd.MultiIndex.from_product([keys_p5, wet_sown_landuses]),axis=0, level=0)
    dryseeding_prov_p5kz = dryseeding_prov_p5z.reindex(pd.MultiIndex.from_product([keys_p5, dry_sown_landuses]),axis=0, level=0)
    return wetseeding_prov_p5kz.stack(), dryseeding_prov_p5kz.stack()



#########
#params #
#########
##collates all the params
def f1_crop_params(params,r_vals):
    cost = f1_rot_cost(r_vals).stack()
    yields = f_rot_yield()
    propn = f_grain_pool_proportions()
    grain_price = f_grain_price(r_vals)
    phasesow_req = f_phase_sow_req()
    wetseeding_prov_p5kz, dryseeding_prov_p5kz = f_sow_prov()

    ##create params
    params['grain_pool_proportions'] = propn.to_dict()
    params['grain_price'] = grain_price.to_dict()
    params['phase_sow_req'] = phasesow_req.to_dict()
    params['wet_sow_prov'] = wetseeding_prov_p5kz.to_dict()
    params['dry_sow_prov'] = dryseeding_prov_p5kz.to_dict()
    params['rot_cost'] = cost.to_dict()
    params['rot_yield'] = yields.to_dict()




#################
#continuous pas #
#################

def f_cont_pas(cost_array):
    '''
    Calculates the cost for continuous pasture that is resown a proportion of the time. eg tc (cont tedera)
    the cost of cont pasture is a combination of the cost of normal and resown eg tc = t + tr (weighted by the frequency of resowing)
    This function requires the index to be the landuse with no other levels. You can use unstack to ensure landuse is the only index.
    Generally this function is applied early in the cost process (before landuse has been dropped)
    Cont pasture only needs to exist if the phase has been included in the rotation.

    .. note:: if a new pasture is added which has a continuous option that is resown occasionally it will need to be added to this function.

    :param cost_array: df with the cost of the corresponding resown landuse. This array will be returned with the addition of the continuous pasture landuse
    '''
    ##read phases
    phases_df = sinp.f_phases()

    pastures = sinp.general['pastures'][pinp.general['pas_inc']]
    ##if cont tedera is in rotation list and tedera is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['tc'])) and 'tedera' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time tc and jc are resown - this is used as a weighting to determine the input costs
        tc_idx = germ_df.iloc[:,-3].isin(['tc']) #checks current phase for tc
        tc_frequency = germ_df.loc[tc_idx,'resown'] #get frequency of resowing tc
        ##create mask for normal tedera and resown tedera
        bool_t = cost_array.index.isin(['t'])
        bool_tr = cost_array.index.isin(['tr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_t)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            t_cost = 0
        else:
            t_cost=(cost_array[bool_t]*(1-tc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_tr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            tr_cost = 0
        else:
            tr_cost=(cost_array[bool_tr]*(tc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        tc_cost = t_cost + tr_cost
        ##assign to df as new col
        cost_array.loc['tc', :] = tc_cost

    ##if cont tedera is in rotation list and tedera is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['jc'])) and 'tedera' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time jc and jc are resown - this is used as a weighting to determine the input costs
        jc_idx = germ_df.iloc[:,-3].isin(['jc']) #checks current phase for jc
        jc_frequency = germ_df.loc[jc_idx,'resown'] #get frequency of resowing jc
        ##create mask for normal tedera and resown tedera
        bool_j = cost_array.index.isin(['j'])
        bool_jr = cost_array.index.isin(['jr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_j)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            j_cost = 0
        else:
            j_cost=(cost_array[bool_j]*(1-jc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_jr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            jr_cost = 0
        else:
            jr_cost=(cost_array[bool_jr]*(jc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        jc_cost = j_cost + jr_cost
        ##assign to df as new col
        cost_array.loc['jc', :] = jc_cost

    ##if cont lucerne is in rotation list and lucerne is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['uc'])) and 'lucerne' in pastures:
        germ_df = pinp.pasture_inputs['lucerne']['GermPhases']
        ##determine the proportion of the time uc and xc are resown - this is used as a weighting to determine the input costs
        uc_idx = germ_df.iloc[:,-3].isin(['uc']) #checks current phase for uc
        uc_frequency = germ_df.loc[uc_idx,'resown'] #get frequency of resowing uc
        ##create mask for normal tedera and resown tedera
        bool_u = cost_array.index.isin(['u'])
        bool_ur = cost_array.index.isin(['ur'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_u)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            u_cost = 0
        else:
            u_cost=(cost_array[bool_u]*(1-uc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_ur)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            ur_cost = 0
        else:
            ur_cost=(cost_array[bool_ur]*(uc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        uc_cost = u_cost + ur_cost
        ##assign to df as new col
        cost_array.loc['uc', :] = uc_cost

    ##if cont lucerne is in rotation list and lucerne is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['xc'])) and 'lucerne' in pastures:
        germ_df = pinp.pasture_inputs['lucerne']['GermPhases']
        ##determine the proportion of the time xc and xc are resown - this is used as a weighting to determine the input costs
        xc_idx = germ_df.iloc[:,-3].isin(['xc']) #checks current phase for xc
        xc_frequency = germ_df.loc[xc_idx,'resown'] #get frequency of resowing xc
        ##create mask for normal tedera and resown tedera
        bool_x = cost_array.index.isin(['x'])
        bool_xr = cost_array.index.isin(['xr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_x)==0: #check if any of the phases in the input array had x, if not then the cost is 0
            x_cost = 0
        else:
            x_cost=(cost_array[bool_x]*(1-xc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_xr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            xr_cost = 0
        else:
            xr_cost=(cost_array[bool_xr]*(xc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        xc_cost = x_cost + xr_cost
        ##assign to df as new col
        cost_array.loc['xc', :] = xc_cost

    return cost_array

