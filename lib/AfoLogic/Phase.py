"""

author: young

The phase module calculates the costs and crop production of each phase. For pasture phases, production is
calculated in the pasture module.

The inputs [#i]_ include phase yield (for pasture this is 0), fertiliser and chemical requirements,
frost damage, seeding rates, soil type, residue management, proportion of arable area, commodity prices,
fees and levies. To accurately reflect rotation history, soil type and weather effects on phase production
and management, many of these user inputs vary by rotation history, soil type and weather conditions.
For example, the rotation phase history can influence the soil fertility, weed burden and disease and pest
levels. These factors impact the potential yield and the level of fertiliser and chemicals.

Each phase provides a certain amount of biomass based on the inputs above. Accordingly, AFO can optimise
the area of each rotation phase on each LMU and the best way to utilise the biomass of each rotation phase.
Biomass can be either harvested for grain, baled for hay or grazed as standing fodder. AFO does not currently
simulate the biology of crop plant growth under different technical management. Thus, AFO does not optimise
technical aspects of cropping such as timing and level of fertiliser applications. However, the user has
the capacity to do this manually by altering the inputs (à la management of inputs in simulation modelling)
or by including additional land uses which represent varying levels of inputs and production.


There are two methods that can be used to generate cropping inputs for the model:

#. Manually enter the inputs for selected rotation phases:

    The user can manually input the fertiliser and chemical requirements and resulting yields of each rotation phase.
    To do this accurately requires an in-depth knowledge of cropping in the
    location being modelled. Thus, the process is often done in collaboration with a consultant or
    specialist in the field. This input method can be limiting if the user is hoping to include a
    large number of rotation phases or landuses that are not well established in the given location
    because it can be difficult to determine accurate inputs.

#. Generate using simulation modelling:

    APSIM is a whole farm simulation model widely used in Australia. APSIM has detailed modules which
    use robust relationships to simulate plant growth. The parameters used in APSIM can be altered to
    represent plant growth in many different situations. For example, different rainfall or soil conditions. AFO users
    can use APSIM to generate yield of crops in a given rotation under a specified fertiliser and chemical regime.

The crop management decisions that are optimised can include:

    1.	Area of each rotation phase on each soil type in each weather year depending on paddock history.
    2.	Area of each crop harvested, baled or grazed.
    3.	Contractual services for seeding or harvesting.
    4.	Labour allocation.
    5.	Time of sowing.

The model can also represent and compare (but not optimise in a single model solution):

    1.	Fertiliser application rate and timing.
    2.	Chemical application rate and timing.
    3.	Seeding rate.
    4.	Alternative cultivars.
    5.	Seeding technology.


.. [#i] Inputs – AFO parameters.


"""

#python modules
import pandas as pd
import numpy as np
import timeit
import datetime as dt
import os

#AFO modules - cant import pasture or stubble or cropgrazing
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import PropertyInputs as pinp
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import EmissionFunctions as efun
from . import Periods as per
from . import Finance as fin
from . import Mach as mac
from . import RotationPhases as rps
from . import Sensitivity as sen
from . import relativeFile
from . import Trees as tree

####################
#general functions #
####################

na = np.newaxis

def f1_sim_inputs(sheet=None, index=None, header=None):
    ###build path this way so the file can be access even if AFO is run from another directory eg readthedocs or web app.
    property = pinp.general['i_property_id']
    return pd.read_excel(relativeFile.findExcel("SimInputs_{0}.xlsx".format(property)), sheet_name=sheet, index_col=index, header=header, engine='openpyxl')


########################
#price                 #
########################

def f_farmgate_grain_price(k_type, r_vals={}):
    '''

    Calculates the grain price received by the farmer.

    The farm gate grain price [#]_ is calculated for each grain pool. Different grain pools are included to
    represent different grain qualities. Depending on the grain variety used, and the season and the farmers skill,
    the grain produced will change quality and hence receive a different price. The price received by the farmer is the
    market price received less any selling costs. The selling costs includes the transport costs which are
    dependent on the location of the modelled farm, and the selling fees which often includes receival
    and testing fees, and government levies.

    The market price is inputted for three different price percentiles. This is extrapolated to return
    the market price for the specified percentile.

    .. [#] Farm gate price – price received by the farmer after fees.

    '''
    ##inputs
    len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
    grain_price_info_df = uinp.price['grain_price_info'] #grain info
    percentile_price_k_s2p = uinp.price['grain_price'] #grain price for 5 different percentiles
    grain_price_percentile = uinp.price['grain_price_percentile'] #price percentile to use
    grain_price_scalar_c1_z = zfun.f_seasonal_inp(uinp.price_variation['grain_price_scalar_c1z'],numpy=False, axis=1, level=0)
    ##k masks
    if k_type=="k1":
        mask_k4 = grain_price_info_df["is_crop"].values #this is mask_k1_is_k4
        mask_k = pinp.crop_landuse_mask_k1 #this is mask_k1
    if k_type=="k3":
        mask_k4 = grain_price_info_df["is_supp"].values  #ís_supp_k4
        mask_k = uinp.supfeed['i_supp_inc_k3'].squeeze().values  #supp_inc_k3

    ##apply masks
    grain_price_info_df = grain_price_info_df.iloc[mask_k4,:].iloc[mask_k,:] #grain info
    percentile_price_k_s2p = percentile_price_k_s2p.iloc[mask_k4,:].iloc[mask_k,:] #grain price for 5 different percentiles
    sav_grainp_k= sen.sav['grainp_k'][mask_k4][mask_k]
    sav_hayp_k = sen.sav['hayp_k'][mask_k4][mask_k]
    sam_grainp_k = sen.sam['grainp_k'][mask_k4][mask_k]
    sam_q_grain_price_scalar_qk = sen.sam['q_grain_price_scalar_Qk'][0:len_q, mask_k4][:,mask_k]

    ##extrapolate price for the selected percentile (can go beyond the data input range)
    percentile_price_ks2_p = percentile_price_k_s2p.stack(0)
    grain_price_firsts_ks2 = pd.Series(index=percentile_price_ks2_p.index, dtype='float64')
    for ks2 in percentile_price_ks2_p.index:
        grain_price_firsts_ks2[ks2] = fun.np_extrap(np.array([grain_price_percentile]), percentile_price_ks2_p.columns,
                                                    percentile_price_ks2_p.loc[ks2].values)[0] #returns as one value in an array thus take [0]

    ##apply price SAV and SAM (SAV first)
    grain_price_firsts_k_s2 = grain_price_firsts_ks2.unstack()
    grain_price_firsts_k_s2.loc[:,'Harv'] = fun.f_sa(grain_price_firsts_k_s2.loc[:,'Harv'], sav_grainp_k, 5)
    grain_price_firsts_k_s2.loc[:,'Bale'] = fun.f_sa(grain_price_firsts_k_s2.loc[:,'Bale'], sav_hayp_k, 5)
    ###apply sam with k axis
    grain_price_firsts_k_s2 = fun.f_sa(grain_price_firsts_k_s2, sam_grainp_k, 0, pandas=True, axis=0)
    grain_price_firsts_ks2 = grain_price_firsts_k_s2.stack()

    ##seconds price
    grain_price_seconds_ks2 = grain_price_firsts_ks2.mul(1-grain_price_info_df['seconds_discount'], level=0)

    ##get the price of firsts and seconds for each grain
    price_df = pd.DataFrame(columns=['firsts','seconds'])
    ###populate df
    price_df['firsts'] = grain_price_firsts_ks2
    price_df['seconds'] = grain_price_seconds_ks2

    ##calc grain price before selling cost - for crop_summary report
    propn_firsts_k = grain_price_info_df[['prop_firsts']].squeeze()
    propn_seconds_k = grain_price_info_df[['prop_seconds']].squeeze()
    ave_price_ks2 = grain_price_firsts_ks2.mul(propn_firsts_k, level=0) + grain_price_seconds_ks2.mul(propn_seconds_k.squeeze(), level=0)
    fun.f1_make_r_val(r_vals,ave_price_ks2.unstack(),'farmgate_price')

    ##determine cost of selling
    cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance']
             + grain_price_info_df['incur_rail_cost']*pinp.general['rail_cartage']
             + uinp.price['flagfall'])
    tolls= grain_price_info_df['grain_tolls']
    total_fees= cartage+tolls
    farmgate_price_ks2_g = price_df.sub(total_fees, axis=0, level=0).clip(0)

    ##scale by c1 & z
    keys_z = zfun.f_keys_z()
    keys_c1 = grain_price_scalar_c1_z.index
    new_index_c1zg = pd.MultiIndex.from_product([keys_c1,keys_z,farmgate_price_ks2_g.columns])
    farmgate_price_ks2g_c1z = farmgate_price_ks2_g.reindex(new_index_c1zg, axis=1, level=2).stack()
    farmgate_price_ks2g_c1z = farmgate_price_ks2g_c1z.mul(grain_price_scalar_c1_z.stack(), axis=1)
    farmgate_price_ks2gc1_z = farmgate_price_ks2g_c1z.stack(0)

    ##add q axis
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    keys_k = grain_price_info_df.index
    q_grain_price_scalar_q_k = pd.DataFrame(sam_q_grain_price_scalar_qk, index = keys_q, columns=keys_k) #have to slice len_q because SAM was initiliased with a big number (because q is unknown because it can be changed by SA)
    farmgate_price_s2gc1z_qk = farmgate_price_ks2gc1_z.stack().unstack(0).mul(q_grain_price_scalar_q_k.stack(), axis=1, level=1)
    farmgate_price_ks2gc1_qz = farmgate_price_s2gc1z_qk.stack(1).unstack(-2).reorder_levels([-1,0,1,2], axis=0)
    ##return
    return farmgate_price_ks2gc1_qz


def f_grain_price(r_vals):

    '''
    Allocates grain price into a cashflow period and stores parameter data for pyomo.

    :return: Dict of farm gate price received for each grain in each cashflow period.

    '''
    ##get grain price - accounts for tolls and other fees
    farmgate_price_ks2gc1_qz=f_farmgate_grain_price(k_type="k1", r_vals=r_vals)

    ##allocate farm gate grain price for each cashflow period and calc interest
    start = np.array([pinp.crop['i_grain_income_date']])
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    grain_cost_allocation_p7z, grain_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='crp', z_pos=-1)

    ##convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    grain_income_allocation_p7z = pd.Series(grain_cost_allocation_p7z.ravel(), index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z])
    grain_wc_allocation_c0p7z = pd.Series(grain_wc_allocation_c0p7z.ravel(), index=new_index_c0p7z)

    farmgate_price_ks2gc1q_z = farmgate_price_ks2gc1_qz.stack(0)
    # cols_p7zg = pd.MultiIndex.from_product([keys_p7, keys_z, farm_gate_price_k_g.columns])
    # grain_income_allocation_p7zg = grain_income_allocation_p7z.reindex(cols_p7zg, axis=1)#adds level to header so i can mul in the next step
    # grain_price =  farm_gate_price_k_g.mul(grain_income_allocation_p7zg,axis=1, level=-1)
    grain_price_ks2gc1q_p7z = farmgate_price_ks2gc1q_z.mul(grain_income_allocation_p7z,axis=1, level=-1)
    grain_price_ks2gc1_qp7z = grain_price_ks2gc1q_p7z.unstack(-1).reorder_levels([2,0,1], axis=1)
    # cols_c0p7zg = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, farm_gate_price_k_g.columns])
    # grain_wc_allocation_c0p7zg = grain_wc_allocation_c0p7z.reindex(cols_c0p7zg, axis=1)#adds level to header so i can mul in the next step
    # grain_price_wc =  farm_gate_price_k_g.mul(grain_wc_allocation_c0p7zg,axis=1, level=-1)
    grain_price_wc_ks2gc1q_c0p7z = farmgate_price_ks2gc1q_z.mul(grain_wc_allocation_c0p7z,axis=1, level=-1)
    grain_price_wc_ks2gc1_qc0p7z = grain_price_wc_ks2gc1q_c0p7z.unstack(-1).reorder_levels([3,0,1,2], axis=1)

    ##average c1 axis for wc and report
    c1_prob = uinp.price_variation['prob_c1']
    r_grain_price_ks2g_qp7z = grain_price_ks2gc1_qp7z.mul(c1_prob, axis=0, level=-1).groupby(axis=0, level=[0,1,2]).sum()
    grain_price_wc_ks2g_qc0p7z = grain_price_wc_ks2gc1_qc0p7z.mul(c1_prob, axis=0, level=-1).groupby(axis=0, level=[0,1,2]).sum()

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, r_grain_price_ks2g_qp7z, 'grain_price', mask_season_p7z, z_pos=-1)
    grain_price_qp7zgks2c1 = grain_price_ks2gc1_qp7z.unstack([2,0,1,3]).sort_index()
    grain_price_wc_qc0p7zgks2 = grain_price_wc_ks2g_qc0p7z.unstack([2,0,1]).sort_index()
    return grain_price_qp7zgks2c1, grain_price_wc_qc0p7zgks2
# a=grain_price()

#########################
#biomass                #
#########################
def f_rot_biomass(for_stub=False, for_insurance=False, r_vals=None):
    '''
    Calculates the biomass for each rotation. Accounting for LMU and arable area.

    The crop yield for each rotation phase, on the base LMU [#]_, at the optimal time of seeding and before harvested proportion adjustment
    (spilt/split grain), is entered as an input. The yield is inputted assuming seeding was completed at the optimal time.
    The base yield inputs are read in from either the simulation output or
    from Property.xl depending on what the user has specified to do. The yield input is dependent on the
    rotation history and hence accounts for the level of soil fertility, weed burden, disease prominence,
    and how the current land use is affected by the existing levels of each in the rotation. Biomass is calculated
    as a function of yield and harvest index. Yield is the input rather than biomass because that is easier to
    relate to and thus determine inputs. However, it is converted to biomass so that the optimisation has
    the option to tactically deviate from the overall strategy. For example, the model may select a barley phase at the
    beginning of the year with the expectation of harvesting it for saleable grain. However, based on spring conditions
    the model may choose to either cut the crop for hay or use it as standing fodder. To
    allow these tactics to be represented requires a common starting point which has been defined as phase biomass.
    Biomass can either be harvested for grain, cut for hay or grazed as standing fodder. The biomass is estimated at
    harvest time and is adjusted downwards for 'baling' because a crop is baled prior to harvest. Grazing of standing
    fodder is assumed to occur at or after harvest.

    Dry sown land uses can have a lower yield input to represent the
    risks associated with dry sowing such as less effective weed control (i.e. crops germinate at
    the same time as the weeds so you miss out on a knock down spray opportunity), poor crop emergence (if
    opening rains are spasmodic patchy crop germination is possible and early crop vigour may be absent without
    adequate follow up rain) :cite:p:`RN119`. Dry sowing may not incur a yield penalty in seasons with a late break or
    if more/better herbicide is applied.

    To extrapolate the inputs from the base LMU to the other LMUs an LMU adjustment factor is
    applied which determines the yield on each other LMU as a proportion of the base LMU. The LMU adjustment
    factor accounts for the variation in yield on different LMUs when management is the same.

    The decision variable represented in the model is the biomass per hectare on a given LMU at harvest. To account for
    the fact that LMUs are rarely 100% arable due to patches of rocks, gully’s, waterlogged area and uncleared
    trees the yield is adjusted by the arable proportion. (e.g. if wheat yields 4 t/ha on LMU5 and LMU5 is 80%
    arable then 1 unit of the decision variable will yield 3.2t of wheat).

    Furthermore, as detailed in the machinery chapter, sowing timeliness can also impact yield. For example,
    sowing to early may incur a yield reduction due to increased frost or forgoing an initial knockdown spray.
    While later sowing incurs a yield loss due to a reduced growing season.

    .. [#] Base LMU – standardise LMU to which other LMUs are compared against.

    :param for_stub: Boolean set to true when calculating the yield that is used to calculate total stubble production.
    :return: Dataframe of rotation yields - passed to pyomo and used to calc grain insurance & stubble handling cost

    '''
    ##read phases
    phases_df = pinp.phases_r
    mask_r = pinp.rot_mask_r
    keys_k = sinp.general['i_idx_k1']

    ##read in base yields
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_yields = pinp.crop['yields']
        base_yields = base_yields.loc[mask_r,:]
        base_yields_rk_z = base_yields.set_index([phases_df.index, phases_df.iloc[:,-1]])
    else:
        ###Sim version
        base_yields_rk_y = f1_sim_inputs(sheet='Yield', index=[0,1], header=0)
        base_yields_rk_y = base_yields_rk_y.loc[mask_r,:]
        ###Convert y to z
        season_group_yz = f1_sim_inputs(sheet='SeasonGroup', index=0, header=0).stack()
        base_yields_rk_z = base_yields_rk_y.mul(season_group_yz, axis=1, level=0).replace(0, np.nan).groupby(axis=1, level=1).mean().replace(np.nan, 0)
    ###rename index
    base_yields_rk_z.index.rename(['rot','landuse'],inplace=True)

    ##Mask z axis
    base_yields_rk_z = zfun.f_seasonal_inp(base_yields_rk_z, axis=1)

    ##apply sam with k axis
    crop_yield_k = pd.Series(sen.sam['crop_yield_k'][pinp.crop_landuse_mask_k1], index=keys_k)
    base_yields_rk_z = base_yields_rk_z.mul(crop_yield_k, axis=0, level=1).fillna(0)
    base_yields_rkz = base_yields_rk_z.stack()

    ##save expected yield r_val used for web app report
    if r_vals is not None:
        base_yields_k_z = base_yields_rk_z.groupby(level=1).mean()
        fun.f1_make_r_val(r_vals, base_yields_k_z, 'base_yields_k_z')

    ##collate other info
    tree_production_scalar_l = tree.f_adjacent_land_production_scalar()
    biomass_lmus_scalar_k_l = pinp.crop['yield_by_lmu'] #soil yield factor
    soil_production_scalar_zl = zfun.f_seasonal_inp(pinp.crop['i_soil_production_z_l'].T,numpy=False).unstack() #soil scalar by weather-year (to account for the fact that the relative performance of a soil is weather related)
    arable_l = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx']) #read in arable area df
    harvest_index_k = uinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because yield is inputted as the harvestable grain
    harvest_index_k = pd.Series(harvest_index_k, index=sinp.general['i_idx_k1'])
    propn_baled_k = pd.Series(uinp.stubble['i_propn_baled_k'], index=sinp.general['i_idx_k1']) #Proportion of biomass at baling that is baled (at point of baling - not including respiration losses).
    growth_scalar_k = pd.Series(uinp.stubble['i_growth_scalar_k'], index=sinp.general['i_idx_k1']) #Biomass at baling relative to biomass at harvest (if not baled). To account for growth from date of baling to harvest.
    propn_conserved_k = pd.Series(uinp.stubble['i_propn_conserved_k'], index=sinp.general['i_idx_k1']) #Proportion of baled biomass available to feed out. To allow for losses due to respiration during drying.
    is_baled_k = sinp.general['i_is_baled_k'] #is the land use baled normally

    ##convert to biomass at grain harvest time
    ###convert yield from hay baled at baling time to grain at harvest time
    baled_factor_k = 1/propn_baled_k/growth_scalar_k/propn_conserved_k * harvest_index_k
    baled_factor_k.iloc[:] = fun.f_update(baled_factor_k.values, 1, np.logical_not(is_baled_k)) #if landuse is not baled then set factor to 1 (mean that the yield inputted is grain at harvest time).
    grain_yields_rkz = base_yields_rkz.div(baled_factor_k, level=1)
    ###calc biomass from grain harvested
    base_biomass_rkz = grain_yields_rkz.div(harvest_index_k, level=1)

    ##calculate biomass - base biomass * arable area * harv_propn * frost * lmu factor * tree_prod_scalar - seeding rate
    biomass_lmus_scalar_kz_l = biomass_lmus_scalar_k_l.mul(soil_production_scalar_zl, axis=1, level=1).stack(0) * tree_production_scalar_l
    biomass_arable_by_soil_kz_l = biomass_lmus_scalar_kz_l.mul(arable_l) #mul arable area to the lmu factor (easy because dfs have the same axis's).
    biomass_rkz_l=biomass_arable_by_soil_kz_l.unstack().reindex(base_biomass_rkz.unstack().index, axis=0, level=1).stack().mul(base_biomass_rkz,axis=0).fillna(0) #reindes and mul with base biomass
    biomass_rkl_z = biomass_rkz_l.stack().unstack(2)

    ##add rotation period axis - if a rotation exists at the beginning of harvest it provides grain and requires harvesting.
    harv_start_date_z = zfun.f_seasonal_inp(pinp.period['harv_date'],numpy=True,axis=0) #this could be changed to include landuse axis.
    alloc_p7z = zfun.f1_z_period_alloc(harv_start_date_z[na,...], z_pos=-1)
    ###convert to df
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    alloc_p7z = pd.Series(alloc_p7z.ravel(), index=new_index_p7z)
    ###mul m allocation with cost
    biomass_rkl_p7z = biomass_rkl_z.mul(alloc_p7z, axis=1,level=1)

    ##add q axis
    len_q = sinp.structuralsa['i_len_q']  # number of years in MP model
    keys_q = np.array(['q%s' % i for i in range(len_q)])
    q_crop_yield_scalar_q_k = pd.DataFrame(sen.sam['q_crop_yield_scalar_Qk'][0:len_q, pinp.crop_landuse_mask_k1],
                                            index=keys_q, columns=keys_k)  # have to slice len_q because SAM was initiliased with a big number (because q is unknown because it can be changed by SA)
    ###mul - bit convoluted because we dont want r and k on different axes because that increase array size a lot.
    biomass_rk_p7zl = biomass_rkl_p7z.unstack(2)
    q_crop_yield_scalar_rkq = q_crop_yield_scalar_q_k.reindex(biomass_rk_p7zl.index, axis=1).fillna(1).unstack()
    biomass_rkq_p7zl = biomass_rk_p7zl.reindex(q_crop_yield_scalar_rkq.index, axis=0)
    biomass_rkq_p7zl = biomass_rkq_p7zl.mul(q_crop_yield_scalar_rkq, axis=0)
    biomass_rkqp7zl = biomass_rkq_p7zl.stack([0,1,2])
    biomass_qrklzp7 = biomass_rkqp7zl.reorder_levels([2,0,1,5,4,3])

    if for_insurance or for_stub:
        ###use q[0] (average yield) because that saves a bit of complexity without losing much accuracy.
        biomass_rklzp7 = biomass_qrklzp7.unstack(0).iloc[:,0]
        ###return biomass for stubble before accounting for frost, seed rate and harv propn
        biomass_rklz = biomass_rklzp7.unstack(-1).sum(axis=1)
        return biomass_rklz
    else:
        ###biomass for pyomo biomass param
        return biomass_qrklzp7

def f_biomass2product(r_vals=None):
    '''Relationship between biomass and saleable product. Where saleable product is either grain or hay.

    Biomass is related to product through harvest index, harvest proportion and biomass scalar.
    Harvest index is the amount of the target product (grain or hay) per unit of biomass at harvest (which is the unit of the biomass DV).
    Harvest proportion accounts for grain that is split/spilt during the harvesting process.
    Biomass scalar is the total biomass production from the area baled net of respiration losses relative
    to biomass at harvest if not baled. This is to account for difference in biomass between harvest and baling time.

    Crop yield can also be adversely impacted by frost during the plants flowing stage :cite:p:`RN144`.
    However, the impact of frost varies based on sowing date. Thus, frost is accounted for in the seeding penalty in Mach.py.

    '''
    ##inputs
    harvest_index_ks2 = uinp.stubble['i_harvest_index_ks2']
    biomass_scalar_ks2 = uinp.stubble['i_biomass_scalar_ks2']
    propn_grain_harv_ks2 = uinp.stubble['i_propn_grain_harv_ks2']

    ##calc biomass to product scalar - adjusted for frost
    biomass2product_ks2 = harvest_index_ks2 * propn_grain_harv_ks2 * biomass_scalar_ks2

    ##store rval before converting to pandas
    if r_vals is not None:
        fun.f1_make_r_val(r_vals, biomass2product_ks2, 'biomass2product_ks2')

    ##convert to pandas
    keys_k = sinp.general['i_idx_k1']
    keys_s2 = uinp.stubble['i_idx_s2']
    index_ks2 = pd.MultiIndex.from_product([keys_k, keys_s2])
    biomass2product_ks2 = pd.Series(biomass2product_ks2.ravel(), index=index_ks2)
    return biomass2product_ks2


def f_grain_pool_proportions():
    '''Calculate the proportion of grain in each pool.

    The total adjusted yield is split into two pools (firsts and seconds) to represent the grain that does
    and does not meet the quality specifications. Grain that does not meet the specifications is downgraded
    and sold for a discount. Each grain pool is represented as a separate grain transfer constraint, providing
    the option for the model to optimise the grain outcome. For example, the model has the option to sell high
    quality grain (firsts) to market and retain the lower quality grain (seconds) for livestock feed.

    '''
    grain_price_info_df = uinp.price['grain_price_info']
    mask_k4 = grain_price_info_df["is_crop"].values  # this is mask_k1_is_k4
    prop = grain_price_info_df[['prop_firsts','prop_seconds']].iloc[mask_k4, :].iloc[pinp.crop_landuse_mask_k1,:]
    prop.columns = sinp.general['grain_pools']
    return prop.stack()


#######
#fert #    
#######    

def f1_fert_cost_allocation():
    '''

    :return: Dataframe with the allocation of each fertiliser cost into cashflow periods.

    '''
    start_df = pinp.crop['fert_info']['app_date'] #needed for allocation func
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    ##calc interest and allocate to cash period - needs to be numpy
    fert_cost_allocation_p7zn, fert_wc_allocation_c0p7zn = fin.f_cashflow_allocation(start_df.values[na,:], enterprise='crp', z_pos=-2, is_phase_cost=True)
    ###convert to df
    new_index_p7zn = pd.MultiIndex.from_product([keys_p7, keys_z, start_df.index])
    fert_cost_allocation_p7zn = pd.Series(fert_cost_allocation_p7zn.ravel(), index=new_index_p7zn)
    new_index_c0p7zn = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, start_df.index])
    fert_wc_allocation_c0p7zn = pd.Series(fert_wc_allocation_c0p7zn.ravel(), index=new_index_c0p7zn)
    return fert_cost_allocation_p7zn, fert_wc_allocation_c0p7zn
# t_allocation=f1_fert_cost_allocation()


def f_fert_passes():
    '''
    Hectares of fertilising required over arable area.

    For both fixed and variable fertiliser the number of applications for each rotation phase
    is entered by the user or obtained from the simulation output. The
    number of fertiliser applications is the same for all LMUs because it is assumed that the rate
    of application varies rather than the frequency of application.

    '''
    ##read phases
    phases_df = pinp.phases_r
    mask_r = pinp.rot_mask_r

    ####read in passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        fert_passes = pinp.crop['fert_passes']
        ###mask r - need to do this now so that phases_df line up because phases_df has already been masked
        fert_passes = fert_passes.loc[mask_r,:]
        fert_passes_rk_zn = fert_passes.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the rotation and current landuse the index
        ###nap fert scalar
        nap_fert_scalar_r = pd.Series(pinp.crop['i_nap_fert_scalar_r'][mask_r], phases_df.index)

    else:
        ###Sim version
        fert_passes_rk_zn = f1_sim_inputs(sheet='No Fert Applications', index=[0,1], header=[0,1])
        fert_passes_rk_zn = fert_passes_rk_zn.loc[mask_r,:]
        ###nap fert scalar
        nap_fert_scalar_r = f1_sim_inputs(sheet='NAP Fert Scalar', index=[0], header=[0]).squeeze()
        nap_fert_scalar_r = nap_fert_scalar_r.loc[mask_r]

    ###Mask z axis
    fert_passes_rk_zn = zfun.f_seasonal_inp(fert_passes_rk_zn, axis=1, level=0)
    ###rename index
    fert_passes_rk_zn.index.rename(['rot','landuse'],inplace=True)

    ##fixed fert (fert that is applied to every phase) - currently this does not have season axis so need to reindex to add season axis
    fert_passes_rkz_n = fert_passes_rk_zn.stack(level=0)
    lime_passes_yearly = pinp.crop['i_lime_freq'] #proportion of lime application each year (cost is allocated equally across each year of rotation)
    fixed_fert_passes_rkz_n = pd.DataFrame(lime_passes_yearly, index=fert_passes_rkz_n.index, columns=['lime'], dtype=float)
    ###set fixed fert to 0 for PNC phases
    try: #incase 'a2' land use is mask out
        fixed_fert_passes_rkz_n.loc[(slice(None), 'a2', slice(None))] = 0
    except KeyError:
        pass
    fert_passes_rkz_n = pd.concat([fert_passes_rkz_n, fixed_fert_passes_rkz_n], axis=1).groupby(axis=1, level=0).sum()

    ##calculate fertiliser on non arable pasture paddocks (non-arable crop paddocks don't get crop (see function docs))
    nap_fert_passes_rkz_n = fert_passes_rkz_n.mul(nap_fert_scalar_r, axis=0, level=0)

    ##apply sam with k & n axis - without unstacking k (need to keep r & k paired to reduce size)
    keys_n = uinp.general['i_fert_idx']
    keys_k = sinp.general['i_idx_k1']
    keys_k2 = sinp.general['i_idx_k2']
    crop_fert_passes_k_n = pd.DataFrame(sen.saa['crop_fert_passes_kn'][pinp.crop_landuse_mask_k1,:], index=keys_k, columns=keys_n)
    pas_fert_passes_k_n = pd.DataFrame(sen.saa['pas_fert_passes_kn'][pinp.pas_landuse_mask_k2,:], index=keys_k2, columns=keys_n)
    fert_passes_saa_k_n = pd.concat([crop_fert_passes_k_n, pas_fert_passes_k_n])
    fert_passes_saa_krz_n = fert_passes_saa_k_n.reindex(fert_passes_rkz_n.index, axis=0, level=1)
    fert_passes_rkz_n = fert_passes_rkz_n.add(fert_passes_saa_krz_n)
    nap_fert_passes_rkz_n = nap_fert_passes_rkz_n.add(fert_passes_saa_krz_n)
    fert_passes_rkz_n[fert_passes_rkz_n < 0] = 0  # stop passes going negitive (incase saa is not correct)
    nap_fert_passes_rkz_n[nap_fert_passes_rkz_n < 0] = 0  # stop passes going negitive (incase saa is not correct)

    ##drop landuse from index
    fert_passes_rz_n = fert_passes_rkz_n.droplevel(1,axis=0)
    nap_fert_passes_rz_n = nap_fert_passes_rkz_n.droplevel(1,axis=0)


    # ####add the fixed fert
    # fixed_fert_passes_k_n = pinp.crop['fixed_fert_passes']
    # keys_z = zfun.f_keys_z()
    # columns = pd.MultiIndex.from_product([keys_z, fixed_fert_passes_k_n.columns])
    # fixed_fert_passes_k_zn = fixed_fert_passes_k_n.reindex(columns, axis=1, level=1)
    # fert_passes_rk_zn = pd.merge(fert_passes_rk_zn, fixed_fert_passes_k_zn, how='left', left_on='landuse', right_index = True)
    # ##drop landuse from index
    # fert_passes_r_zn = fert_passes_rk_zn.droplevel(1, axis=0)

    ##adjust fert passes by arable area
    arable_l = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
    # fert_passes_rz_n = fert_passes_r_zn.stack(0)
    col_nl = pd.MultiIndex.from_product([fert_passes_rz_n.columns, arable_l.index])
    fert_passes_rz_nl = fert_passes_rz_n.reindex(col_nl, axis=1,level=0)
    nap_fert_passes_rz_nl = nap_fert_passes_rz_n.reindex(col_nl, axis=1,level=0)
    fert_passes_rz_nl=fert_passes_rz_nl.mul(arable_l,axis=1,level=1)
    nap_fert_passes_rz_nl=nap_fert_passes_rz_nl.mul(1-arable_l,axis=1,level=1)
    total_fert_passes_rz_nl = fert_passes_rz_nl.fillna(0).stack(1) + nap_fert_passes_rz_nl.fillna(0).stack(1)
    return total_fert_passes_rz_nl.sort_index()


def f1_fertilising_time():
    '''
    Determines the time (hr/ha) spent fertilising for each rotation.

    Based on:

    1. per tonne; represents the difference in application time based on fert density - represents the filling up and
       traveling to the paddock time, ie it would require more filling and traveling time to spread 1t of a lighter (less dense) fert.
    2. per ha; represents the time to spread 1ha - this depends on how far each fert is chucked out of the spreader

    This is used to calculate machinery application cost, labour requirement and
    variable machinery depreciation associated with fertilising.
    '''
    ##fert passes - arable (arable area accounted for in passes function)
    total_passes_rzln = f_fert_passes().stack()

    ##time linked to tonnage
    ###fert used on each rotation phase - calculated based on inputted estimate of kgs of fert applied per pass
    fert_req_n = pinp.crop['fert_info']['expected_rate']/1000 #convert to tonnes
    fert_total_rzln = total_passes_rzln.mul(fert_req_n, level=-1)
    ###time per tonne
    time_n = mac.time_tonne()
    ###total time
    time_t_rzln = fert_total_rzln.mul(time_n, level=-1)

    ##time linked to spreading a hectare in paddock
    ###time taken to cover 1ha while spreading
    time_ha_n = mac.time_ha()
    ###total time
    time_ha_rzln = total_passes_rzln.mul(time_ha_n, level=-1)

    return time_t_rzln + time_ha_rzln


def f_fert_cost(r_vals):
    '''
    Cost of fertilising. Includes the fertiliser cost and the application cost.

    Different land uses have varying nutrient requirements,
    varying methods of obtaining nutrients from the soil and leave the soil in varying states (e.g.
    pulse crop fix nitrogen which typically reduces their requirement for external
    nitrogen and leaves the soil with a higher nitrogen content for the following year/s).
    To accommodate these factors the fertiliser requirement section is broken into three sections. Firstly,
    fixed fertiliser, which is the fertiliser that is applied to all rotation phases (e.g. lime
    which is typically applied routinely, irrelevant of the land use history). Secondly, land use specific fertiliser,
    which is applied based on the current land use independent of the phase history. Lastly, rotation phase specific
    fertiliser, which is applied based on both the current land use and the
    history.

    Fertiliser rate for non-arable areas are calculated as a scalar of the arable area.
    The standard assumption is that fertiliser is only applied to non-arable area in pasture phases
    It is not applied to the non-arable area in the crop phase because the non-arable pasture in a crop phase is
    only grazed at the end of the year when the pasture is rank and therefore fertiliser is not cost-effective.

    The cost of fertilising is made up from the cost of the fertilisers, the cost getting
    the fertiliser delivered to the farm and the machinery cost of application (detailed in the machinery section).
    The costs are incurred in the cashflow period when it is applied. The assumption is that fertiliser is
    purchased shortly before application because farmers wait to see how the year unfolds before locking
    in a fertiliser plan.

    :return: Dataframe of fertiliser costs. Summed with other cashflow items at the end of the module

    '''
    ##read in necessary stuff
    phases_df = pinp.phases_r
    mask_r = pinp.rot_mask_r
    fert_by_soil = pinp.crop['fert_by_lmu']
    fert_cost_allocation_p7zn, fert_wc_allocation_c0p7zn = f1_fert_cost_allocation()
    fert_cost_allocation_z_p7n = fert_cost_allocation_p7zn.unstack(1).T
    fert_wc_allocation_z_c0p7n = fert_wc_allocation_c0p7zn.unstack(2).T

    ##read in rotation phase specific fert - fert that can vary by phase
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_fert = pinp.crop['fert']
        ###mask r - need to do this now so that phases_df line up because phases_df has already been masked
        base_fert = base_fert.loc[mask_r,:]
        ###add landuse to index
        base_fert_rk_zn = base_fert.set_index([phases_df.index,phases_df.iloc[:,-1]])
        ###nap fert scalar
        nap_fert_scalar_r = pd.Series(pinp.crop['i_nap_fert_scalar_r'][mask_r], phases_df.index)
    else:
        ###Sim version
        base_fert_rk_zn = f1_sim_inputs(sheet='Fert Applied', index=[0,1], header=[0,1])
        ###mask r
        base_fert_rk_zn = base_fert_rk_zn.loc[mask_r,:]
        ###nap fert scalar
        nap_fert_scalar_r = f1_sim_inputs(sheet='NAP Fert Scalar', index=[0], header=[0]).squeeze()
        nap_fert_scalar_r = nap_fert_scalar_r.loc[mask_r]

    ###Mask z axis
    base_fert_rk_zn = zfun.f_seasonal_inp(base_fert_rk_zn, axis=1, level=0)
    ###rename index
    base_fert_rk_zn.index.rename(['rot','landuse'],inplace=True)

    ##fixed fert (fert that is applied to every phase) - currently this does not have season axis so need to reindex to add season axis
    base_fert_rkz_n = base_fert_rk_zn.stack(level=0)
    lime_cost_yearly = pinp.crop['i_lime'] * pinp.crop['i_lime_freq'] #workout the cost of liming each year (cost is allocated equally across each year of rotation)
    fixed_fert_rkz_n = pd.DataFrame(lime_cost_yearly, index=base_fert_rkz_n.index, columns=['lime'], dtype=float)
    ###set fixed fert to 0 for PNC phases
    try: #incase 'a2' land use is mask out
        fixed_fert_rkz_n.loc[(slice(None), 'a2', slice(None))] = 0
    except KeyError:
        pass
    base_fert_rkz_n = pd.concat([base_fert_rkz_n, fixed_fert_rkz_n], axis=1).groupby(axis=1, level=0).sum()

    ##apply sam with k & n axis - without unstacking k (need to keep r & k paired to reduce size)
    keys_n = uinp.general['i_fert_idx']
    keys_k = sinp.general['i_idx_k1']
    keys_k2 = sinp.general['i_idx_k2']
    crop_fert_k_n = pd.DataFrame(sen.sam['crop_fert_kn'][pinp.crop_landuse_mask_k1,:], index=keys_k, columns=keys_n)
    pas_fert_k_n = pd.DataFrame(sen.sam['pas_fert_kn'][pinp.pas_landuse_mask_k2,:], index=keys_k2, columns=keys_n)
    fert_sam_k_n = pd.concat([crop_fert_k_n, pas_fert_k_n])
    fert_sam_krz_n = fert_sam_k_n.reindex(base_fert_rkz_n.index, axis=0, level=1)
    base_fert_rkz_n = base_fert_rkz_n.mul(fert_sam_krz_n)
    ###drop landuse from index
    base_fert_rz_n = base_fert_rkz_n.droplevel(1,axis=0)

    ## adjust the fert cost for each rotation by lmu
    fert_by_soil_nl = fert_by_soil.stack() #read in fert by soil
    fert_rz_nl = base_fert_rz_n.mul(fert_by_soil_nl,axis=1,level=0)

    ##calculate fertiliser on non arable pasture paddocks (non-arable crop paddocks dont get crop (see function docs))
    nap_fert_rz_nl = fert_rz_nl.mul(nap_fert_scalar_r, axis=0, level=0)

    ##account for arable area
    arable_l = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
    fert_rz_nl = fert_rz_nl.mul(arable_l, axis=1, level=1) #add arable to df
    nap_fert_rz_nl = nap_fert_rz_nl.mul(1-arable_l, axis=1, level=1) #add arable to df
    phase_fert_cost_rzl_n = fert_rz_nl.fillna(0).stack(1) + nap_fert_rz_nl.fillna(0).stack(1)

    ##adjust for interest and p7 period
    phase_fert_cost_rzl_p7n = phase_fert_cost_rzl_n.reindex(fert_cost_allocation_z_p7n.columns, axis=1, level=1)
    phase_fert_cost_rl_p7nz = phase_fert_cost_rzl_p7n.unstack(1)
    phase_fert_cost_rl_p7z = phase_fert_cost_rl_p7nz.mul(fert_cost_allocation_z_p7n.unstack(), axis=1).groupby(axis=1, level=(0,2)).sum()  # sum the cost of all the ferts (have to do that after allocation and interest because ferts are applied at different times)
    phase_fert_wc_rzl_c0p7n = phase_fert_cost_rzl_n.reindex(fert_wc_allocation_z_c0p7n.columns, axis=1, level=2)
    phase_fert_wc_rl_c0p7nz = phase_fert_wc_rzl_c0p7n.unstack(1)
    phase_fert_wc_rl_c0p7z = phase_fert_wc_rl_c0p7nz.mul(fert_wc_allocation_z_c0p7n.unstack(), axis=1).groupby(axis=1, level=(0,1,3)).sum()  # sum the cost of all the ferts (have to do that after allocation and interest because ferts are applied at different times)

    ##aplication cost
    fert_time_rzl_n = f1_fertilising_time().unstack()
    fert_app_cost_rzl_n = fert_time_rzl_n * mac.spreader_cost_hr()
    fert_app_cost_rzl_p7n = fert_app_cost_rzl_n.reindex(fert_cost_allocation_z_p7n.columns, axis=1, level=1)
    fert_app_cost_rl_p7nz = fert_app_cost_rzl_p7n.unstack(1)
    fert_app_cost_rl_p7z = fert_app_cost_rl_p7nz.mul(fert_cost_allocation_z_p7n.unstack(), axis=1).groupby(axis=1, level=(0,2)).sum()  # sum the cost of all the ferts (have to do that after allocation and interest because ferts are applied at different times)
    fert_app_wc_rzl_c0p7n = fert_app_cost_rzl_n.reindex(fert_wc_allocation_z_c0p7n.columns, axis=1, level=2)
    fert_app_wc_rl_c0p7nz = fert_app_wc_rzl_c0p7n.unstack(1)
    fert_app_wc_rl_c0p7z = fert_app_wc_rl_c0p7nz.mul(fert_wc_allocation_z_c0p7n.unstack(), axis=1).groupby(axis=1, level=(0,1,3)).sum()  # sum the cost of all the ferts (have to do that after allocation and interest because ferts are applied at different times)

    ##combine all costs - fert, app per ha and app per tonne
    fert_cost_total = phase_fert_cost_rl_p7z + fert_app_cost_rl_p7z

    ##combine all wc - fert, app per ha and app per tonne
    fert_wc_total = phase_fert_wc_rl_c0p7z + fert_app_wc_rl_c0p7z

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, phase_fert_cost_rl_p7z, 'phase_fert_cost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(phase_fert_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'phase_fert_cost_increment', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, fert_app_cost_rl_p7z, 'fert_app_cost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(fert_app_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'fert_app_cost_increment', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, fert_time_rzl_n, 'fert_time_rzl_n')
    return fert_cost_total, fert_wc_total


#######################
#stubble handling    #
#######################
'''
cost now associated with current yr because it requires knowing the yield of the crop - hence under the new rotation phases that contain sets it isn;t possible to use the previous landuse to determine cost.
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
    additionally this new structure assumes that even if the preceding landuse is pasture the current phase will still get handling cost (wasn't able to find an alternative way)
frost is not included because that doesn't reduce biomass
'''

def f1_stubble_handling_prob():
    ##calculate the probability of a rotation phase needing stubble handling
    base_biomass_rkl_z = f_rot_biomass(for_stub=True).unstack()
    ###convert to grain
    harvest_index_k = uinp.stubble['i_harvest_index_ks2'][:,0] #select the harvest s2 slice because stubble handling is based on harvestable grain yield
    harvest_index_k = pd.Series(harvest_index_k, index=sinp.general['i_idx_k1'])
    base_yields_rkl_z = base_biomass_rkl_z.mul(harvest_index_k, axis=0, level=1)
    stub_handling_threshold = pd.Series(pinp.stubble['stubble_handling'], index=sinp.general['i_idx_k1'], dtype=float)*1000  #have to convert to kg to match base yield
    probability_handling_rkl_z = base_yields_rkl_z.div(stub_handling_threshold, axis=0, level=1) #divide here then account for lmu factor next - because either way is mathematically sound and this saves some manipulation.
    probability_handling_rl_z = probability_handling_rkl_z.droplevel(1)
    return probability_handling_rl_z

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
    This structure assumes that even if the preceding landuse is pasture, the current phase will still get
    handling costs. This is because the rotation phases use sets so you can’t determine exactly which land use is
    before or after the current phase.


    .. note:: An improvement could be to include harvest index in the calculation of handling probability. Also,
        with seasonal variation represented the probability is not required.

    .. note:: arable area accounted for in the yield (it is the same as accounting for it at the end
        ie yield x 0.8 / threshold x cost == yield / threshold x cost x 0.8)
    '''
    ##call mach func to get mach cost
    stub_cost=mac.f_stubble_cost_ha()

    ##calculate the probability of a rotation phase needing stubble handling
    probability_handling_rl_z = f1_stubble_handling_prob()

    ##adjust the cost of stubble handling by the probability of needing to handle stubble
    stub_cost_rl_z = probability_handling_rl_z * stub_cost

    ##allocate the cash period and calc interest and working capital
    start = np.array([pinp.mach['stub_handling_date']]) #needed for allocation func
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    stub_cost_allocation_p7z, stub_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='crp', z_pos=-1, is_phase_cost=True)
    ###convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    stub_cost_allocation_p7z = pd.Series(stub_cost_allocation_p7z.ravel(), index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z])
    stub_wc_allocation_c0p7z = pd.Series(stub_wc_allocation_c0p7z.ravel(), index=new_index_c0p7z)
    ###mul cost and allocation
    rot_stub_cost_rl_p7z = stub_cost_rl_z.mul(stub_cost_allocation_p7z, axis=1, level=1)
    rot_stub_wc_rl_c0p7z = stub_cost_rl_z.mul(stub_wc_allocation_c0p7z, axis=1, level=2)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, rot_stub_cost_rl_p7z, 'stub_cost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(rot_stub_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'stub_cost_increment', mask_season_p7z, z_pos=-1)

    return rot_stub_cost_rl_p7z, rot_stub_wc_rl_c0p7z
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
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    ##calc interest and allocate to cash period - needs to be numpy
    chem_cost_allocation_p7zn, chem_wc_allocation_c0p7zn = fin.f_cashflow_allocation(start_df.values[na,:], enterprise='crp', z_pos=-2, is_phase_cost=True)
    ###convert to df
    new_index_p7zn = pd.MultiIndex.from_product([keys_p7, keys_z, start_df.index])
    chem_cost_allocation_p7zn = pd.Series(chem_cost_allocation_p7zn.ravel(), index=new_index_p7zn)
    new_index_c0p7zn = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, start_df.index])
    chem_wc_allocation_c0p7zn = pd.Series(chem_wc_allocation_c0p7zn.ravel(), index=new_index_c0p7zn)
    return chem_cost_allocation_p7zn, chem_wc_allocation_c0p7zn
# t_allocation=f1_chem_cost_allocation()
    
def f_chem_application():
    '''

    Number of applications of each chemical option for each rotation.

    The number of applications at each spraying time (e.g. pre-seeding knock down, pre-emergent,
    post-emergent and fungicide) for each rotation phase is entered by the user or obtained from the simulation output .
    When determining the number of applications, the user must consider both the rotation
    history and the current landuse. The rotation history is important because it impacts the initial levels of
    weed and disease burden and fungi. The current landuse is important because different landuses are susceptible
    to different weeds and diseases. Furthermore, chemicals can be specific for certain
    crops and have varying levels of effectiveness for different weeds and diseases.

    Similar to fertiliser, the number of chemical applications is the
    same for all LMUs because it is assumed that the spray rate varies rather than the frequency of
    application. However, the area sprayed is adjusted by the arable proportion for each LMU. The assumption
    is that non-arable areas are not sprayed.

    '''
    ##read phases
    phases_df = pinp.phases_r
    mask_r = pinp.rot_mask_r

    ##read in chem passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        chem_passes = pinp.crop['chem']
        ###mask r - need to do this now so that phases_df line up because phases_df has already been masked
        chem_passes = chem_passes.loc[mask_r,:]
        chem_passes_rk_zn = chem_passes.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the rotation and current landuse the index
    else:
        ###Sim version
        chem_passes_rk_zn = f1_sim_inputs(sheet='No Chem Applications', index=[0,1], header=[0,1])
        chem_passes_rk_zn = chem_passes_rk_zn.loc[mask_r,:]
    ###Mask z axis
    chem_passes_rk_zn = zfun.f_seasonal_inp(chem_passes_rk_zn, axis=1, level=0)
    ##apply sam with k & n axis - without unstacking k (need to keep r & k paired to reduce size)
    keys_n = uinp.general['i_chem_idx']
    keys_k = sinp.general['i_idx_k1']
    keys_k2 = sinp.general['i_idx_k2']
    crop_chem_passes_k_n = pd.DataFrame(sen.saa['crop_chem_passes_kn1'][pinp.crop_landuse_mask_k1,:], index=keys_k, columns=keys_n)
    pas_chem_passes_k_n = pd.DataFrame(sen.saa['pas_chem_passes_kn1'][pinp.pas_landuse_mask_k2,:], index=keys_k2, columns=keys_n)
    chem_passes_saa_k_n = pd.concat([crop_chem_passes_k_n, pas_chem_passes_k_n])
    chem_passes_rkz_n = chem_passes_rk_zn.stack(0)
    chem_passes_saa_krz_n = chem_passes_saa_k_n.reindex(chem_passes_rkz_n.index, axis=0, level=1)
    chem_passes_rkz_n = chem_passes_rkz_n.add(chem_passes_saa_krz_n)
    chem_passes_rkz_n[chem_passes_rkz_n<0] = 0 #stop passes going negitive (incase saa is not correct)
    ###drop landuse from index
    chem_passes_rz_n = chem_passes_rkz_n.droplevel(1, axis=0)
    ##adjust chem passes by arable area
    arable_l = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
    col_nl = pd.MultiIndex.from_product([chem_passes_rz_n.columns, arable_l.index])
    chem_passes_rz_nl = chem_passes_rz_n.reindex(col_nl, axis=1,level=0)
    chem_passes_rz_nl=chem_passes_rz_nl.mul(arable_l,axis=1,level=1)
    return chem_passes_rz_nl.stack(1).sort_index()

def f1_spraying_time():
    '''
    Determines the time (hr/ha) spent spraying for each rotation (including filling up).

    This is used to calculate machinery application cost, labour requirement and
    variable machinery depreciation associated with fertilising.
    '''

    ##passes - arable (arable area accounted for in passes function)
    total_passes_rzln = f_chem_application().stack()
    ##time taken to cover 1ha while spraying include a factor for filling up.
    time_ha = mac.spray_time_ha()
    ##total time accounting for number of applications.
    time_ha_rzln = total_passes_rzln * time_ha

    return time_ha_rzln


def f_chem_cost(r_vals):
    '''

    Calculates the cost of spraying for each rotation phase on each LMU.

    The total cost of spraying is made up from the cost of the chemical (herbicide and fungicide) and the machinery
    cost of application (detailed in the machinery section). The
    chemical cost is incurred in the cashflow period when it is applied. The assumption is that
    chemical is purchased shortly before application because farmers wait to see how the year unfolds
    before locking in a spraying plan.

    To simplify the input process, the total cost of herbicide and fungicide on the base LMU for each land use is
    entered as an input by the user. This saves the step of going from a volume of chemical to a cost
    which means AFO does not need to represent the large array of chemical options available. Making it
    easier to keep AFO up to date. The chemical cost is adjusted by an LMU
    factor to account for the cost of herbicide/fungicide changing for different LMUs (e.g. a more expensive
    chemical package may be used on heavier soils compared to deep sand).

    :return: Total cost of chemical and application for each rotation phase - summed with other cashflow
        items at the end of this section.

    '''
    ##read phases
    phases_df = pinp.phases_r
    mask_r = pinp.rot_mask_r

    ##read in necessary bits and adjust indexed
    # i_chem_cost = pinp.crop['chem_cost'].sort_index() #cost of fungicide per application
    n_alloc = pinp.crop['chem_info']['chem_cost_alloc'] #allocation of chem cost to category (knockdown, pre-em and post-em). This is used for cashflow allocation.
    chem_by_soil = pinp.crop['chem_by_lmu'].squeeze() #read in chem by soil
    chem_cost_allocation_p7zn, chem_wc_allocation_c0p7zn = f1_chem_cost_allocation()
    chem_cost_allocation_z_p7n = chem_cost_allocation_p7zn.unstack(1).T
    chem_wc_allocation_z_c0p7n = chem_wc_allocation_c0p7zn.unstack(2).T

    ##total chem cost
    if pinp.crop['user_crop_rot']:
        ### User defined
        chem_cost = pinp.crop['chem_cost']
        chem_cost = chem_cost.loc[mask_r, :]
        chem_cost_rk_zn = chem_cost.set_index(
            [phases_df.index, phases_df.iloc[:, -1]])  # make the rotation and current landuse the index
    else:
        ###Sim version
        chem_cost_rk_zn = f1_sim_inputs(sheet='Total Chem Cost', index=[0,1], header=[0,1])
        chem_cost_rk_zn = chem_cost_rk_zn.loc[mask_r,:]
    ###Mask z axis
    chem_cost_rk_zn = zfun.f_seasonal_inp(chem_cost_rk_zn, axis=1, level=0)
    ###apply SAM
    keys_k = sinp.general['i_idx_k1']
    keys_k2 = sinp.general['i_idx_k2']
    crop_chem_k = pd.Series(sen.sam['crop_chem_k'][pinp.crop_landuse_mask_k1], index=keys_k)
    pas_chem_k = pd.Series(sen.sam['pas_chem_k'][pinp.pas_landuse_mask_k2], index=keys_k2)
    chem_sam_k = pd.concat([crop_chem_k, pas_chem_k])
    chem_cost_rk_zn = chem_cost_rk_zn.mul(chem_sam_k, axis=0, level=1)
    ###drop landuse from index
    chem_cost_r_zn = chem_cost_rk_zn.droplevel(1, axis=0)
    ### sum herbicide and fungicide cost
    chem_cost_r_z = chem_cost_r_zn.groupby(axis=1, level=0).sum()
    ###arable area.
    arable = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
    ###adjust by arable area and allocate to category (category allocation only affects cashflow timing hence interest)
    index_zln = pd.MultiIndex.from_product([chem_cost_r_z.columns, arable.index, n_alloc.index])
    chem_cost_r_zln = chem_cost_r_z.reindex(index_zln, axis=1,level=0)
    chem_cost_r_zln=chem_cost_r_zln.mul(arable,axis=1,level=1)
    chem_cost_r_zln=chem_cost_r_zln.mul(n_alloc,axis=1,level=2)
    ### adjust the chem cost for each rotation by lmu
    chem_cost_rzl_n=chem_cost_r_zln.mul(chem_by_soil,axis=1, level=1).stack([0,1])
    ###adjust for interest and p7 period
    phase_chem_cost_rzl_p7n = chem_cost_rzl_n.reindex(chem_cost_allocation_z_p7n.columns,axis=1,level=1)
    phase_chem_cost_rl_p7nz = phase_chem_cost_rzl_p7n.unstack(1)
    phase_chem_cost_rl_p7z = phase_chem_cost_rl_p7nz.mul(chem_cost_allocation_z_p7n.unstack(), axis=1).groupby(axis=1, level=(0,2)).sum()  # sum the cost of all the chem
    phase_chem_wc_rzl_c0p7n = chem_cost_rzl_n.reindex(chem_wc_allocation_z_c0p7n.columns,axis=1,level=2)
    phase_chem_wc_rl_c0p7nz = phase_chem_wc_rzl_c0p7n.unstack(1)
    phase_chem_wc_rl_c0p7z = phase_chem_wc_rl_c0p7nz.mul(chem_wc_allocation_z_c0p7n.unstack(), axis=1).groupby(axis=1, level=(0,1,3)).sum()  # sum the cost of all the chem

    ##application cost - only a per ha component
    time_ha_rzl_n = f1_spraying_time().unstack()
    chem_app_cost_rzl_n = time_ha_rzl_n * mac.spraying_cost_hr()
    ###adjust of interest and p7 period
    chem_app_cost_rzl_p7n = chem_app_cost_rzl_n.reindex(chem_cost_allocation_z_p7n.columns, axis=1, level=1)
    chem_app_cost_rl_p7nz = chem_app_cost_rzl_p7n.unstack(1)
    chem_app_cost_rl_p7z = chem_app_cost_rl_p7nz.mul(chem_cost_allocation_z_p7n.unstack(), axis=1).groupby(axis=1, level=(0,2)).sum()  # sum the cost of all the chems
    chem_app_wc_rzl_c0p7n = chem_app_cost_rzl_n.reindex(chem_wc_allocation_z_c0p7n.columns, axis=1, level=2)
    chem_app_wc_rl_c0p7nz = chem_app_wc_rzl_c0p7n.unstack(1)
    chem_app_wc_rl_c0p7z = chem_app_wc_rl_c0p7nz.mul(chem_wc_allocation_z_c0p7n.unstack(), axis=1).groupby(axis=1, level=(0,1,3)).sum()  # sum the cost of all the chems

    ##add application cost and chem cost
    total_cost = phase_chem_cost_rl_p7z + chem_app_cost_rl_p7z

    ##add application wc and chem wc
    total_wc = phase_chem_wc_rl_c0p7z + chem_app_wc_rl_c0p7z

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, phase_chem_cost_rl_p7z, 'chem_cost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(phase_chem_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'chem_cost_increment', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, chem_app_cost_rl_p7z, 'chem_app_cost_ha', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(chem_app_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'chem_app_cost_ha_increment', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, time_ha_rzl_n, 'chem_time_ha_rzl_n')
    return total_cost, total_wc


#########################
#misc cost              #
#########################
def f_seedcost(r_vals):
    '''

    Seed costs includes:

        - seed treatment
        - raw seed cost (this assumes that farmers purchase seed rather than using seed from last years harvest)
        - crop insurance
        - arable area
    '''
    ##read phases and add two empty col levels
    phases_df2 = pinp.phases_r.copy()
    phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns,[''],['']])  # make the df multi index so that when it merges with other df below the indexs remaining separate (otherwise it turn into a one leveled tuple)
    phases_df3 = pinp.phases_r.copy()
    phases_df3.columns = pd.MultiIndex.from_product([phases_df3.columns,[''],[''],['']])  # make the df multi index so that when it merges with other df below the indexs remaining separate (otherwise it turn into a one leveled tuple)

    ##seasonal inputs
    seed_period_lengths = zfun.f_seasonal_inp(pinp.period['seed_period_lengths'], numpy=True, axis=1)
    ##inputs
    seeding_rate = pinp.crop['seeding_rate']
    seeding_cost = pinp.crop['seed_info']['Seed cost']
    grading_cost = pinp.crop['seed_info']['Grading'] 
    percent_graded = pinp.crop['seed_info']['Percent Graded'] 
    cost1 = pinp.crop['seed_info']['Cost1'] #cost ($/l) for dressing 1 
    cost2 = pinp.crop['seed_info']['Cost2'] #cost ($/l) for dressing 2 
    rate1 = pinp.crop['seed_info']['Rate1'] #rate (ml/100g) for dressing 1
    rate2 = pinp.crop['seed_info']['Rate2'] #rate (ml/100g) for dressing 2
    percent_dressed = pinp.crop['seed_info']['percent dressed'] #rate (ml/100g) for dressing 2
    arable = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
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
    seed_cost = seeding_rate.mul(cost/1000,axis=0)
    ##cost allocation
    start_z = per.f_wet_seeding_start_date()
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    seed_cost_allocation_p7z, seed_wc_allocation_c0p7z = fin.f_cashflow_allocation(start_z, enterprise='crp', z_pos=-1, is_phase_cost=True)
    ###convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    seed_cost_allocation_p7z = pd.Series(seed_cost_allocation_p7z.ravel(), index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z])
    seed_wc_allocation_c0p7z = pd.Series(seed_wc_allocation_c0p7z.ravel(), index=new_index_c0p7z)

    ##mul cost by allocation - need to align column headers first
    columns_p7zl = pd.MultiIndex.from_product([keys_p7, keys_z, seed_cost.columns])
    seed_cost_kl_p7z = seed_cost.reindex(columns_p7zl, axis=1, level=2).stack()
    seed_cost_kl_p7z = seed_cost_kl_p7z.mul(seed_cost_allocation_p7z, axis=1)
    columns_c0p7zl = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, seed_cost.columns])
    seed_wc_kl_c0p7z = seed_cost.reindex(columns_c0p7zl, axis=1, level=3).stack()
    seed_wc_kl_c0p7z = seed_wc_kl_c0p7z.mul(seed_wc_allocation_c0p7z, axis=1)

    ##merge to rotation df
    phase_seed_cost_r_p7zl = pd.merge(phases_df2, seed_cost_kl_p7z.unstack(), how='left', left_on=sinp.end_col(), right_index = True)
    phase_seed_wc_r_c0p7zl = pd.merge(phases_df3, seed_wc_kl_c0p7z.unstack(), how='left', left_on=sinp.end_col(), right_index = True)
    seedcost_rl_p7z = phase_seed_cost_r_p7zl.drop(list(range(sinp.general['phase_len'])), axis=1).stack()
    seed_wc_rl_c0p7z = phase_seed_wc_r_c0p7zl.drop(list(range(sinp.general['phase_len'])), axis=1).stack()

    ## multiple by seeding freq
    seeding_freq_r = pd.Series(pinp.seeding_freq_r, index=phases_df2.index)
    seedcost_rl_p7z = seedcost_rl_p7z.mul(seeding_freq_r, axis=0, level=0)
    seed_wc_rl_c0p7z = seed_wc_rl_c0p7z.mul(seeding_freq_r, axis=0, level=0)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, seedcost_rl_p7z, 'seedcost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(seedcost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'seedcost_increment', mask_season_p7z, z_pos=-1)

    return seedcost_rl_p7z, seed_wc_rl_c0p7z

def f_insurance(r_vals):
    '''
    Crop insurance cost.

    Crop insurance is typically based off the farmers estimation of yield in mid spring (hence active z axis).
    This is not going to exactly be equal to final yield, but it is closer than using the average yield.
    The small amount of error in this assumption will have little impact due to the small magnitude of
    financial impact of insurance.

    .. note:: arable area is already counted for by the yield calculation.
    '''
    ##weight c1 to get average price
    c1_prob = uinp.price_variation['prob_c1']
    farmgate_price_ks2gc1_qz = f_farmgate_grain_price(k_type="k1")
    farmgate_price_ks2g_qz = farmgate_price_ks2gc1_qz.mul(c1_prob,axis=0,level=-1).groupby(axis=0,level=[0,1,2]).sum()
    ##combine each grain pool to get average price
    grain_pool_proportions_kg = f_grain_pool_proportions()
    ###take q[0] under assumption insurance is based on average price.
    farmgate_price_kg_zs2 = farmgate_price_ks2g_qz.stack(1).iloc[:,0].unstack([-1,1])
    ave_price_k_zs2 = farmgate_price_kg_zs2.mul(grain_pool_proportions_kg, axis=0).groupby(axis=0, level=0).sum()
    ##calc insurance cost per tonne
    grain_price_info_df = uinp.price['grain_price_info']
    mask_k4 = grain_price_info_df["is_crop"].values  # this is mask_k1_is_k4
    insurance_k1 = grain_price_info_df['insurance'].iloc[mask_k4].iloc[pinp.crop_landuse_mask_k1]
    insurance_k_zs2 = ave_price_k_zs2.mul(insurance_k1/100, axis=0)  #div by 100 because insurance is a percent
    insurance_ks2z = insurance_k_zs2.stack([1,0])
    ##calc phase product for each s2 option then select the s2 slice with maximum insurance cost (maximum because that would most likely be the expected s2 option)
    biomass_rklz = f_rot_biomass(for_insurance=True)
    biomass2product_ks2 = f_biomass2product()
    yields_rlz_ks2 = biomass_rklz.unstack(1).reindex(biomass2product_ks2.index, axis=1, level=0).mul(biomass2product_ks2, axis=1)
    yields_rl_ks2z = yields_rlz_ks2.unstack(2)
    yields_rl_ks2z = yields_rl_ks2z.reindex(insurance_ks2z.index, axis=1).mul(insurance_ks2z, axis=1)/1000 #divide by 1000 to convert yield to tonnes
    yields_rl_kz = yields_rl_ks2z.groupby(axis=1, level=[0,2]).max()
    rot_insurance_rl_z = yields_rl_kz.stack(0).droplevel(axis=0, level=-1)
    ##cost allocation
    start = np.array([uinp.price['crp_insurance_date']])
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    insurance_cost_allocation_p7z, insurance_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='crp', z_pos=-1, is_phase_cost=True)
    ###convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    insurance_cost_allocation_p7z = pd.Series(insurance_cost_allocation_p7z.ravel(), index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z])
    insurance_wc_allocation_c0p7z = pd.Series(insurance_wc_allocation_c0p7z.ravel(), index=new_index_c0p7z)

    ##add cashflow period to col index
    rot_insurance_cost_rl_p7z = rot_insurance_rl_z.mul(insurance_cost_allocation_p7z, axis=1, level=1)
    rot_insurance_wc_rl_c0p7z = rot_insurance_rl_z.mul(insurance_wc_allocation_c0p7z, axis=1, level=2)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, rot_insurance_cost_rl_p7z, 'insurance_cost', mask_season_p7z, z_pos=-1)
    fun.f1_make_r_val(r_vals, rps.f_v_phase_increment_adj(rot_insurance_cost_rl_p7z.stack([0,1]).sort_index()
                                                          ,p7_pos=-2,z_pos=-1), 'insurance_cost_increment', mask_season_p7z, z_pos=-1)

    ##take crp slice of c0 to reduce param size.
    return rot_insurance_cost_rl_p7z, rot_insurance_wc_rl_c0p7z


###################################################
# variable spraying and spreading depreciation     #
###################################################

def f_spraying_spreading_dep(r_vals):
    '''
    Average variable dep for seeding $/ha.

    Variable depreciation is use depreciation and is dependent on the number of hours the equipment is used.

    '''
    ##spraying
    ###variable depn rate is input as a percent depn in all spray gear per machine hour (%/machine hr).
    spray_dep_rate_per_hr = uinp.mach_general['i_variable_dep_hr_spraying']
    ####convert from rotor hours to spray activity hours (no dep when filling up)
    spray_dep_rate_per_hr = spray_dep_rate_per_hr * (1 - pinp.mach['spray_downtime_frac'])
    ###determine dep per hour - equal to crop gear value x depn %
    spray_gear_clearing_value = mac.f_spray_gear_clearing_value()
    spray_dep_hourly = spray_gear_clearing_value * spray_dep_rate_per_hr
    ###convert to dep per ha
    spray_time_rzln = f1_spraying_time()
    spray_dep_ha_rzln = spray_dep_hourly * spray_time_rzln
    ###allocate to season period based on application date - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    chem_info = pinp.crop['chem_info']
    chem_date_n = chem_info['app_date'].values
    chem_length_n = chem_info['app_len'].values
    alloc_p7zn = zfun.f1_z_period_alloc(chem_date_n[na,na,:], chem_length_n, z_pos=-2, is_phase_param=True) 
    ###make df
    keys_n = uinp.general['i_chem_idx']
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    index_p7zn = pd.MultiIndex.from_product([keys_p7, keys_z, keys_n])
    alloc_p7zn = pd.Series(alloc_p7zn.ravel(), index=index_p7zn)
    alloc_p7n_z = alloc_p7zn.unstack(1)
    spray_dep_ha_rzl_p7n = spray_dep_ha_rzln.unstack().reindex(alloc_p7zn.unstack(1).index, axis=1, level=1)
    spray_dep_ha_rl_p7nz = spray_dep_ha_rzl_p7n.unstack(1).mul(alloc_p7n_z.stack(),axis=1)
    spray_dep_ha_rl_p7z = spray_dep_ha_rl_p7nz.groupby(axis=1, level=(0,2)).sum() #sum n axis. not needed after alllocation to p7 has been made.
    
    
    ##spreading
    ###variable depn rate is input as a percent depn in all spreading gear per machine hour (%/machine hr).
    spread_dep_rate_per_hr = uinp.mach_general['i_variable_dep_hr_spreading']
    ###determine dep per hour - equal to crop gear value x depn %
    spread_gear_clearing_value = mac.f_spread_gear_clearing_value()
    spread_dep_hourly = spread_gear_clearing_value * spread_dep_rate_per_hr
    ###convert to dep per ha
    spread_time_rzln = f1_fertilising_time()
    spread_dep_ha_rzln = spread_dep_hourly * spread_time_rzln
    ###allocate to season period based on application date - so that depreciation can be linked to seeding activity and transferred as seasons uncluster
    fert_info = pinp.crop['fert_info']
    fert_date_n = fert_info['app_date'].values
    fert_length_n = fert_info['app_len'].values
    alloc_p7zn = zfun.f1_z_period_alloc(fert_date_n[na,na,:], fert_length_n, z_pos=-2, is_phase_param=True) 
    ###make df
    keys_n = uinp.general['i_fert_idx']
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    index_p7zn = pd.MultiIndex.from_product([keys_p7, keys_z, keys_n])
    alloc_p7zn = pd.Series(alloc_p7zn.ravel(), index=index_p7zn)
    alloc_p7n_z = alloc_p7zn.unstack(1)
    spread_dep_ha_rzl_p7n = spread_dep_ha_rzln.unstack().reindex(alloc_p7zn.unstack(1).index, axis=1, level=1)
    spread_dep_ha_rl_p7nz = spread_dep_ha_rzl_p7n.unstack(1).mul(alloc_p7n_z.stack(),axis=1)
    spread_dep_ha_rl_p7z = spread_dep_ha_rl_p7nz.groupby(axis=1, level=(0,2)).sum() #sum n axis. not needed after alllocation to p7 has been made.

    ##combine spray and spread
    spreader_sprayer_dep_p7zlr = spread_dep_ha_rl_p7z.add(spray_dep_ha_rl_p7z).unstack([1,0])

    ##create params for v_phase_change_increase
    ## costs for v_phase_change_increase activities are incurred in the season period when the activity is selected
    ## however the interest is calculated as if the cost was incurred at the normal time (this is because interest
    ## is calculated for each separate cost in the functions above).
    increment_spreader_sprayer_dep_p7zlr = rps.f_v_phase_increment_adj(spreader_sprayer_dep_p7zlr,p7_pos=-4,z_pos=-3)


    ##rvals
    fun.f1_make_r_val(r_vals, spray_dep_hourly, 'spray_dep_hourly')
    fun.f1_make_r_val(r_vals, spread_dep_hourly, 'spread_dep_hourly')


    return spreader_sprayer_dep_p7zlr, increment_spreader_sprayer_dep_p7zlr


#########################
#emissions              #
#########################
def f1_rot_fuel_emissions(r_vals):
    '''
    Counts fuel used for 1 ha of each rotation. Accounts for:

        - spraying
        - spreading
        - stubble handling

    Spraying, spreading and stubble handling are connected to v_phase_increase. Which means even if a phase is changed before all of the emissions are incurred
    they are still included. If this becomes a limitation then the parameter need to be allocated to p7 periods and
    linked to v_phase activity.

    :return:
    '''
    ##spreading
    spread_time_rzl = f1_fertilising_time().groupby(level=[0,1,2]).sum() #sum n
    spreading_fuel_hr = uinp.mach[pinp.mach['option']]['spreader_fuel']
    spreading_fuel_ha_rzl = spread_time_rzl * spreading_fuel_hr

    ##spraying
    spray_time_rzl = f1_spraying_time().groupby(level=[0,1,2]).sum() #sum n
    spraying_fuel_hr = uinp.mach[pinp.mach['option']]['sprayer_fuel_consumption']
    spraying_fuel_ha_rzl = spray_time_rzl * spraying_fuel_hr

    ##stubble handling
    probability_handling_rzl = f1_stubble_handling_prob().unstack().stack([0,1])
    stubble_handling_fuel_ha = uinp.mach[pinp.mach['option']]['stubble_fuel_consumption']
    stubble_handling_fuel_ha_rzl = probability_handling_rzl * stubble_handling_fuel_ha

    ##combine
    stubble_handling_fuel_ha_rzl = stubble_handling_fuel_ha_rzl.reindex(spreading_fuel_ha_rzl.index).fillna(0)
    phase_fuel_ha_rzl = spreading_fuel_ha_rzl.add(spraying_fuel_ha_rzl).add(stubble_handling_fuel_ha_rzl)
    phase_fuel_ha_zrl = phase_fuel_ha_rzl.unstack([0,2]).stack([0,1]) #reorder levels to match v_phase_increment

    ##convert to emissions
    co2_phase_fuel_co2e_zrl, ch4_phase_fuel_co2e_zrl, n2o_phase_fuel_co2e_zrl = efun.f_fuel_emissions(phase_fuel_ha_zrl)
    total_co2e_phase_fuel_zrl = co2_phase_fuel_co2e_zrl + ch4_phase_fuel_co2e_zrl + n2o_phase_fuel_co2e_zrl

    ##store r_vals - store as nupy
    len_z = len(zfun.f_keys_z())
    len_r = len(total_co2e_phase_fuel_zrl.index.levels[1])
    len_l = len(total_co2e_phase_fuel_zrl.index.levels[2])
    fun.f1_make_r_val(r_vals, total_co2e_phase_fuel_zrl.values.reshape(len_z, len_r, len_l), 'co2e_phase_fuel_zrl')

    return total_co2e_phase_fuel_zrl

def f1_rot_fert_emissions(r_vals):
    '''
    Calcs totoal co2e emissions for fertilising. See EmissionsFunctions.py for more details.

    fert emissions are connected to v_phase_increase. Which means even if a phase is changed before all of the emissions are incurred
    they are still included. If this becomes a limitation then the parameter need to be allocated to p7 periods and
    linked to v_phase activity.

    :param r_vals:
    :return:
    '''
    ##call emission function
    co2e_fert_k = efun.f_fert_emissions()

    ##convert k to r
    keys_k = sinp.general['i_idx_k']
    phases_df = pinp.phases_r
    landuse_r = phases_df.iloc[:, -1].values
    a_k_rk = landuse_r[:, na] == keys_k
    co2e_fert_r = np.sum(co2e_fert_k * a_k_rk, axis=1)

    ##save r_val
    fun.f1_make_r_val(r_vals, co2e_fert_r, 'co2e_fert_r')

    ##make df
    keys_r = np.array(phases_df.index).astype('str')
    co2e_fert_r = pd.Series(co2e_fert_r, keys_r)
    return co2e_fert_r

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

    fert_cost, fert_wc = f_fert_cost(r_vals)
    chem_cost, chem_wc = f_chem_cost(r_vals)
    seedcost, seedwc = f_seedcost(r_vals)
    insurance_cost, insurance_wc = f_insurance(r_vals)
    phase_stubble_cost, phase_stubble_wc = f_phase_stubble_cost(r_vals)

    #note if any array has dtype object then pandas throws error (No axis named 1 for object type Series)
    cost_rl_p7z = pd.concat([fert_cost, chem_cost, seedcost, insurance_cost, phase_stubble_cost],axis=1).groupby(axis=1,level=(0,1)).sum()
    wc_rl_c0p7z = pd.concat([fert_wc, chem_wc, seedwc, insurance_wc, phase_stubble_wc],axis=1).groupby(axis=1,level=(0,1,2)).sum()

    ##stack
    cost_p7zlr = cost_rl_p7z.unstack([1,0]).sort_index()
    wc_c0p7zlr = wc_rl_c0p7z.unstack([1,0]).sort_index()

    ##create params for v_phase_change_increase
    ## costs for v_phase_change_increase activities are incurred in the season period when the activity is selected
    ## however the interest is calculated as if the cost was incurred at the normal time (this is because interest
    ## is calculated for each separate cost in the functions above).
    increment_cost_p7zlr = rps.f_v_phase_increment_adj(cost_p7zlr,p7_pos=-4,z_pos=-3)
    increment_wc_c0p7zlr = rps.f_v_phase_increment_adj(wc_c0p7zlr,p7_pos=-4,z_pos=-3)

    return cost_p7zlr, increment_cost_p7zlr, wc_c0p7zlr, increment_wc_c0p7zlr


#################
#sow            #
#################
    
def f_phase_sow_req():
    '''
    Area of seeding required for 1ha of each rotation.

    This accounts for arable area and includes any seeding (wet or dry or pasture).

    '''
    ##read phases
    phases_df = pinp.phases_r
    ##adjust arable area
    arable = pd.Series(pinp.general['arable'], pinp.general['i_lmu_idx'])
    ##sow = arable area * frequency
    keys_l = arable.index
    seeding_freq_r = pinp.seeding_freq_r
    arable_l = arable.values
    sow_req_rl = seeding_freq_r[:,na] * arable_l
    ##make df
    keys_rk = pd.MultiIndex.from_arrays([phases_df.index,phases_df.iloc[:,-1]])
    phasesow = pd.DataFrame(sow_req_rl, index=keys_rk, columns=keys_l).stack()
    return phasesow

def f_sow_prov():
    '''
    Creates provide param for wet and dry sowing activities:

        - Area of wet seeding provided by 1ha of the wet seeding activity.
        - Area of dry seeding provided by 1ha of the dry seeding activity.

    This accounts for period and crop e.g. wet seeding activity only provides sowing to crop after the break.

    This also stop seeding (dry and wet) in a false break between the identification and the real break.
    This is because the soil is a "little bit wet". So in the areas that are wet enough the seed will germinate
    and in other areas it won't. So this will lead to a patchy crop establishment and make crop management difficult
    later in the season due to the variation in crop stages.
    This only effects crop because pasture seeding timing is an input with a z axis so it can be altered there if desired.

    '''
    ##machine periods
    labour_period_p5z = per.f_p_dates_df()
    labour_period_start_p5z = labour_period_p5z.values[:-1]
    labour_period_end_p5z = labour_period_p5z.values[1:]

    ##general info
    keys_k = sinp.general['i_idx_k']
    keys_z = zfun.f_keys_z()
    keys_p5 = labour_period_p5z.index[:-1]
    keys_p7 = per.f_season_periods(keys=True)
    dry_sown_landuses = sinp.landuse['dry_sown']
    wet_sown_landuses = set(sinp.general['i_idx_k1']) - dry_sown_landuses #can subtract sets to return differences
    dry_sown_landuses = dry_sown_landuses | {"ms"}
    false_brk_identification_z = zfun.f_seasonal_inp(pinp.general['i_false_brk_identification_z'],numpy=True,axis=0)
    false_brk_followuprains_z = zfun.f_seasonal_inp(pinp.general['i_false_brk_followuprains_z'],numpy=True,axis=0)

    ##determine which periods crop can't be sown because it is a false break (this doesnt effect seasons with no false brk)
    ## any p5 period that the false break goes through cant be seeded even if the false brk only partially covers a period.
    ## to avoid any misrepresentation ensure the false brk timing inputs line up with p5 periods
    z_is_false_break_z = false_brk_identification_z < false_brk_followuprains_z
    p5_is_false_break_p5z = np.logical_and(false_brk_identification_z >= labour_period_start_p5z, false_brk_followuprains_z > labour_period_start_p5z)
    p5z_is_false_break_p5z = np.logical_and(z_is_false_break_z, p5_is_false_break_p5z)
    p5z_isnot_during_false_break_p5z = np.logical_not(p5z_is_false_break_p5z)

    ##wet sowing periods
    seed_period_lengths_pz = zfun.f_seasonal_inp(pinp.period['seed_period_lengths'],numpy=True,axis=1)
    wet_seed_start_z = per.f_wet_seeding_start_date()
    wet_seed_len_z = np.sum(seed_period_lengths_pz, axis=0)
    wet_seed_end_z = wet_seed_start_z + wet_seed_len_z
    period_is_wetseeding_p5z = (labour_period_start_p5z < wet_seed_end_z) * (labour_period_end_p5z > wet_seed_start_z) \
                               * p5z_isnot_during_false_break_p5z
    ###add k axis
    period_is_wetseeding_p5zk = period_is_wetseeding_p5z[...,na] * np.sum(keys_k[:,na] == list(wet_sown_landuses), axis=-1)

    ##dry sowing periods - dry seeding occurs up until the start of the season (dry seeding is sowing without knock down spray)
    ##currently we are saying that dry sowning cant occur between the brk of season and wet seeding start. This may or may not be correct (not if dry seeding occurs after the brk of the season it doesnt need to happen in the children seasons).
    dry_seed_start = pinp.crop['dry_seed_start']
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'],numpy=True)
    period_is_dryseeding_p5z = (labour_period_start_p5z < season_break_z) * (labour_period_end_p5z > dry_seed_start)\
                               * p5z_isnot_during_false_break_p5z
    ###add k axis
    if not sinp.structuralsa['i_differentiate_wet_dry_seeding']: #in the web app all land uses can be dry sown (this is a simplification to save seperate representation of dry sown land uses.)
        dry_sown_landuses = sinp.general['i_idx_k1']
    period_is_dryseeding_p5zk = period_is_dryseeding_p5z[...,na] * np.sum(keys_k[:,na] == list(dry_sown_landuses), axis=-1)

    ##pasture seeding
    pastures = sinp.general['pastures'][pinp.general['pas_inc_t']]
    zt = (len(keys_z),len(pastures))
    i_reseeding_date_start_zt = np.zeros(zt, dtype = 'float64')
    i_reseeding_date_end_zt = np.zeros(zt, dtype = 'float64')
    for t,pasture in enumerate(pastures):
        i_reseeding_date_start_zt[...,t] = zfun.f_seasonal_inp(pinp.pasture_inputs[pasture]['Date_Seeding'],numpy=True)
        i_reseeding_date_end_zt[...,t] = zfun.f_seasonal_inp(pinp.pasture_inputs[pasture]['pas_seeding_end'],numpy=True)
    period_is_passeeding_p5zt = (labour_period_start_p5z[:,:,na] < i_reseeding_date_end_zt) * (labour_period_end_p5z[:,:,na] > i_reseeding_date_start_zt)
    ###convert t axis to k
    kt = (len(keys_k), len(pastures))
    resown_kt = np.zeros(kt)
    for t,pasture in enumerate(pastures):
        pasture_landuses = list(sinp.landuse['pasture_sets'][pasture])
        resown_kt[:,t] = np.in1d(keys_k, pasture_landuses)  #resown if landuse is a pasture and is a sown landuse
    period_is_passeeding_p5zk = np.sum(resown_kt * period_is_passeeding_p5zt[:,:,na,:], -1) #sum t axis - t is counted for in the k axis

    ##combine wet, dry and pas
    period_is_seeding_p5zk = np.minimum(1,period_is_wetseeding_p5zk + period_is_dryseeding_p5zk + period_is_passeeding_p5zk)

    ##add p7 axis - needed so machinery can be linked with phases (machinery just has a p5 axis)
    alloc_p7p5z = zfun.f1_z_period_alloc(labour_period_start_p5z[na,:,:], z_pos=-1)
    sow_prov_p7p5zk = period_is_seeding_p5zk * alloc_p7p5z[...,na]

    ##make df
    index_p7p5zk = pd.MultiIndex.from_product([keys_p7,keys_p5,keys_z,keys_k])
    index_p5zk = pd.MultiIndex.from_product([keys_p5,keys_z,keys_k])
    sow_prov_p7p5zk = pd.Series(sow_prov_p7p5zk.ravel(), index=index_p7p5zk)
    can_sow_p5zk = pd.Series(period_is_seeding_p5zk.ravel(), index=index_p5zk)
    return sow_prov_p7p5zk, can_sow_p5zk


#########
#params #
#########
##collates all the params
def f1_crop_params(params,r_vals):
    cost, increment_cost, wc, increment_wc = f1_rot_cost(r_vals)
    spreader_sprayer_dep_p7zlr, increment_spreader_sprayer_dep_p7zlr = f_spraying_spreading_dep(r_vals)
    biomass = f_rot_biomass(r_vals=r_vals)
    biomass2product_ks2 = f_biomass2product(r_vals)
    propn = f_grain_pool_proportions()
    grain_price, grain_wc = f_grain_price(r_vals)
    phasesow_req = f_phase_sow_req()
    sow_prov_p7p5zk, can_sow_p5zk = f_sow_prov()
    total_co2e_phase_fuel_zrl = f1_rot_fuel_emissions(r_vals)
    co2e_fert_r = f1_rot_fert_emissions(r_vals)

    ##create params
    params['grain_pool_proportions'] = propn.to_dict()
    params['grain_price'] = grain_price.to_dict()
    params['grain_wc'] = grain_wc.to_dict()
    params['phase_sow_req'] = phasesow_req.to_dict()
    params['sow_prov'] = sow_prov_p7p5zk.to_dict()
    params['can_sow_p5zk'] = can_sow_p5zk.to_dict()
    params['rot_cost'] = cost.to_dict()
    params['increment_rot_cost'] = increment_cost.to_dict()
    params['rot_wc'] = wc.to_dict()
    params['increment_rot_wc'] = increment_wc.to_dict()
    params['rot_biomass'] = biomass[biomass!=0].to_dict() #only save non-zero params to save space.
    params['biomass2product_ks2'] = biomass2product_ks2.to_dict()
    params['spreader_sprayer_dep_p7zlr'] = spreader_sprayer_dep_p7zlr.to_dict()
    params['increment_spreader_sprayer_dep_p7zlr'] = increment_spreader_sprayer_dep_p7zlr.to_dict()
    params['co2e_phase_fuel_zrl'] = total_co2e_phase_fuel_zrl.to_dict()
    params['co2e_phase_fert_r'] = co2e_fert_r.to_dict()



##cont pas are now just included in the inputs.
# #################
# #continuous pas #
# #################
#
# def f_cont_pas(cost_array):
#     '''
#     Calculates the cost for continuous pasture that is resown a proportion of the time. E.g. tc (cont tedera)
#     the cost of cont pasture is a combination of the cost of normal and resown e.g. tc = t + tr (weighted by the frequency of resowing)
#     This function requires the index to be the landuse with no other levels. You can use unstack to ensure landuse is the only index.
#     Generally this function is applied early in the cost process (before landuse has been dropped)
#     Cont pasture only needs to exist if the phase has been included in the rotation.
#
#     .. note:: if a new pasture is added which has a continuous option that is resown occasionally it will need to be added to this function.
#
#     :param cost_array: df with the cost of the corresponding resown landuse. This array will be returned with the addition of the continuous pasture landuse
#     '''
#     ##read phases
#     phases_df = pinp.f1_phases()
#
#     pastures = sinp.general['pastures'][pinp.general['pas_inc_t']]
#     ##if cont tedera is in rotation list and tedera is included in the pasture modules then generate the inputs for it
#     if any(phases_df.iloc[:,-1].isin(['tc'])) and 'tedera' in pastures:
#         germ_df = pinp.pasture_inputs['tedera']['GermPhases']
#         ##determine the proportion of the time tc and jc are resown - this is used as a weighting to determine the input costs
#         tc_idx = germ_df.iloc[:,-3].isin(['tc']) #checks current phase for tc
#         tc_frequency = germ_df.loc[tc_idx,'resown'] #get frequency of resowing tc
#         ##create mask for normal tedera and resown tedera
#         bool_t = cost_array.index.isin(['t'])
#         bool_tr = cost_array.index.isin(['tr'])
#         ##create new param - average all phases.
#         if np.count_nonzero(bool_t)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             t_cost = 0
#         else:
#             t_cost=(cost_array[bool_t]*(1-tc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         if np.count_nonzero(bool_tr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             tr_cost = 0
#         else:
#             tr_cost=(cost_array[bool_tr]*(tc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         ##add weighted average of the resown and normal phase
#         tc_cost = t_cost + tr_cost
#         ##assign to df as new col
#         cost_array.loc['tc', :] = tc_cost
#
#     ##if cont tedera is in rotation list and tedera is included in the pasture modules then generate the inputs for it
#     if any(phases_df.iloc[:,-1].isin(['jc'])) and 'tedera' in pastures:
#         germ_df = pinp.pasture_inputs['tedera']['GermPhases']
#         ##determine the proportion of the time jc and jc are resown - this is used as a weighting to determine the input costs
#         jc_idx = germ_df.iloc[:,-3].isin(['jc']) #checks current phase for jc
#         jc_frequency = germ_df.loc[jc_idx,'resown'] #get frequency of resowing jc
#         ##create mask for normal tedera and resown tedera
#         bool_j = cost_array.index.isin(['j'])
#         bool_jr = cost_array.index.isin(['jr'])
#         ##create new param - average all phases.
#         if np.count_nonzero(bool_j)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             j_cost = 0
#         else:
#             j_cost=(cost_array[bool_j]*(1-jc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         if np.count_nonzero(bool_jr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             jr_cost = 0
#         else:
#             jr_cost=(cost_array[bool_jr]*(jc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         ##add weighted average of the resown and normal phase
#         jc_cost = j_cost + jr_cost
#         ##assign to df as new col
#         cost_array.loc['jc', :] = jc_cost
#
#     ##if cont lucerne is in rotation list and lucerne is included in the pasture modules then generate the inputs for it
#     if any(phases_df.iloc[:,-1].isin(['uc'])) and 'lucerne' in pastures:
#         germ_df = pinp.pasture_inputs['lucerne']['GermPhases']
#         ##determine the proportion of the time uc and xc are resown - this is used as a weighting to determine the input costs
#         uc_idx = germ_df.iloc[:,-3].isin(['uc']) #checks current phase for uc
#         uc_frequency = germ_df.loc[uc_idx,'resown'] #get frequency of resowing uc
#         ##create mask for normal tedera and resown tedera
#         bool_u = cost_array.index.isin(['u'])
#         bool_ur = cost_array.index.isin(['ur'])
#         ##create new param - average all phases.
#         if np.count_nonzero(bool_u)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             u_cost = 0
#         else:
#             u_cost=(cost_array[bool_u]*(1-uc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         if np.count_nonzero(bool_ur)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             ur_cost = 0
#         else:
#             ur_cost=(cost_array[bool_ur]*(uc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         ##add weighted average of the resown and normal phase
#         uc_cost = u_cost + ur_cost
#         ##assign to df as new col
#         cost_array.loc['uc', :] = uc_cost
#
#     ##if cont lucerne is in rotation list and lucerne is included in the pasture modules then generate the inputs for it
#     if any(phases_df.iloc[:,-1].isin(['xc'])) and 'lucerne' in pastures:
#         germ_df = pinp.pasture_inputs['lucerne']['GermPhases']
#         ##determine the proportion of the time xc and xc are resown - this is used as a weighting to determine the input costs
#         xc_idx = germ_df.iloc[:,-3].isin(['xc']) #checks current phase for xc
#         xc_frequency = germ_df.loc[xc_idx,'resown'] #get frequency of resowing xc
#         ##create mask for normal tedera and resown tedera
#         bool_x = cost_array.index.isin(['x'])
#         bool_xr = cost_array.index.isin(['xr'])
#         ##create new param - average all phases.
#         if np.count_nonzero(bool_x)==0: #check if any of the phases in the input array had x, if not then the cost is 0
#             x_cost = 0
#         else:
#             x_cost=(cost_array[bool_x]*(1-xc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         if np.count_nonzero(bool_xr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
#             xr_cost = 0
#         else:
#             xr_cost=(cost_array[bool_xr]*(xc_frequency[0])).mean(axis=0) #get average cost of each t phase
#         ##add weighted average of the resown and normal phase
#         xc_cost = x_cost + xr_cost
#         ##assign to df as new col
#         cost_array.loc['xc', :] = xc_cost
#
#     return cost_array

