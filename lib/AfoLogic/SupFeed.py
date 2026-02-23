"""

author: young

Supplementary feeding is the supply of additional feed to livestock, primarily grain and hay (which are both
represented in the model). Supplementary feeding is commonly used to help meet production targets such as
lamb growth rates prior to sale, or to fill the feed gap to allow higher stocking rates during the summer
and autumn months when pastures and crop residues are limiting. Additionally, feeding supplement can be
used as a tactic to allow pastures to be deferred early in the growing season which increases subsequent
pasture growth rates through increasing leaf area index.

AFO represents a range of supplements including, oats, lupins and hay. Grain and hay as supplementary
feeds can either be grown on farm or purchased from another farmer at a farm-gate price (i.e. net price
of a product after selling costs have been subtracted) plus the transaction and transport costs.
Supplementary feeding incurs a depreciation cost associated with storage infrastructure and variable
costs associated with insurance, silo preparation, insect management, grain shrinkage/losses and machinery
usage when feeding the supplement. The costs are calculated per tonne to allow for variations in grain
density and amounts fed. Supplementary feeding also incurs a labour requirement for time spent traveling
to and from the silo, filling the sheep feeder, emptying the feeder, and transporting between paddocks.

.. note:: Other grains can be added as supplements. Just remember to add their inputs in the mach
    sheet in universal.xlsx for each machine option and the sup sheet in both universal.xlsx and property.xlsx.



"""
#python modules
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import relativedelta as rdelta

#AFO modules
from . import Functions as fun
from . import SeasonalFunctions as zfun
from . import FeedsupplyFunctions as fsfun
from . import EmissionFunctions as efun
from . import Periods as per
from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Phase as phs
from . import Mach as mac
from . import Sensitivity as sen
from . import Finance as fin

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
    farmgate_price_k3s2gc1_qz = phs.f_farmgate_grain_price(k_type="k3")
    cartage=uinp.price['sup_cartage']
    transaction_fee=uinp.price['sup_transaction']
    buy_price_k3s2gc1_qz = farmgate_price_k3s2gc1_qz + cartage + transaction_fee

    ##allocate farm gate grain price for each cashflow period and calc interest
    start = np.array([pinp.crop['i_grain_income_date']])
    keys_p7 = per.f_season_periods(keys=True)
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    grain_cost_allocation_p7z, grain_wc_allocation_c0p7z = fin.f_cashflow_allocation(start, enterprise='stk', z_pos=-1)

    ##convert to df
    new_index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    grain_income_allocation_p7z = pd.Series(grain_cost_allocation_p7z.ravel(), index=new_index_p7z)
    new_index_c0p7z = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z])
    grain_wc_allocation_c0p7z = pd.Series(grain_wc_allocation_c0p7z.ravel(), index=new_index_c0p7z)

    # cols_p7zg = pd.MultiIndex.from_product([keys_p7, keys_z, price_k_g.columns])
    # grain_income_allocation_p7zg = grain_income_allocation_p7z.reindex(cols_p7zg, axis=1)#adds level to header so i can mul in the next step
    # cols_c0p7zg = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, price_k_g.columns])
    # grain_wc_allocation_c0p7zg = grain_wc_allocation_c0p7z.reindex(cols_c0p7zg, axis=1)#adds level to header so i can mul in the next step
    # buy_grain_price =  price_k_g.mul(grain_income_allocation_p7zg,axis=1, level=-1)
    # buy_grain_price_wc =  price_k_g.mul(grain_wc_allocation_c0p7zg,axis=1, level=-1)
    buy_price_k3s2gc1q_z = buy_price_k3s2gc1_qz.stack(0)
    buy_grain_price_k3s2gc1q_p7z =  buy_price_k3s2gc1q_z.mul(grain_income_allocation_p7z,axis=1, level=-1)
    buy_grain_price_k3s2gc1_qp7z = buy_grain_price_k3s2gc1q_p7z.unstack(-1).reorder_levels([2, 0, 1], axis=1)
    buy_grain_price_wc_k3s2gc1q_c0p7z =  buy_price_k3s2gc1q_z.mul(grain_wc_allocation_c0p7z,axis=1, level=-1)
    buy_grain_price_wc_k3s2gc1_qc0p7z = buy_grain_price_wc_k3s2gc1q_c0p7z.unstack(-1).reorder_levels([3, 0, 1, 2], axis=1)

    ##average c1 axis for wc and report
    c1_prob = uinp.price_variation['prob_c1']
    buy_grain_price_wc_k3s2g_qc0p7z = buy_grain_price_wc_k3s2gc1_qc0p7z.mul(c1_prob, axis=0, level=-1).groupby(axis=0, level=[0,1,2]).sum()
    r_buy_grain_price_k3s2g_qp7z = buy_grain_price_k3s2gc1_qp7z.mul(c1_prob, axis=0, level=-1).groupby(axis=0, level=[0,1,2]).sum()

    ##buy grain period - purchased grain can only provide into the grain transfer constraint in the phase period when it is purchased (otherwise it will get free grain)
    alloc_p7z = zfun.f1_z_period_alloc(start[na], z_pos=-1)
    index_p7z = pd.MultiIndex.from_product([keys_p7, keys_z])
    buy_grain_prov_p7z = pd.Series(alloc_p7z.ravel(), index=index_p7z)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, r_buy_grain_price_k3s2g_qp7z, 'buy_grain_price', mask_season_p7z, z_pos=-1)
    return buy_grain_price_k3s2gc1_qp7z.unstack([2,0,1,3]).sort_index(), buy_grain_price_wc_k3s2g_qc0p7z.unstack([2,0,1]).sort_index(), buy_grain_prov_p7z

#todo supp feeding in confinement should incur some capital cost.
def f_sup_feeding_cost(r_vals, nv):
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
    This cost then applies to each tonne of supplement fed. The storage cost is calculated assuming
    that all supplement fed for the year is stored on farm. If the capacity of the silos is less than
    the supplement fed (meaning additional supplement is purchased part way through the year) then the
    storage cost will be overestimated. However, discussions with farm consultants suggest that farmers
    store enough supplement for the whole year and extra to handle a poor year.

    The machinery cost to feed a tonne of supplement is added in this function however it
    is calculated in Mach.py (see Mach.py for details on machinery cost to feed supplement).

    '''

    ##calculate the insurance/dep/asset value per yr for the silos
    silo_info = pinp.supfeed['storage_type']
    silo_info.loc['dep'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/silo_info.loc['life']
    silo_info.loc['insurance'] = silo_info.loc['price'] * uinp.finance['equip_insurance']
    silo_info.loc['asset'] = (silo_info.loc['price'] - silo_info.loc['salvage value'])/2 #calculate the average value of the asset - used in the asset ROE constraint
    ##using the capacity of each silo for each grain determine the costs per tonne foe each grain
    grain_info=uinp.supfeed['grain_density'].T.reset_index() #reindex so it can be combined with silo df
    grain_info=grain_info.set_index(['index','silo type']).T.astype(float)
    grain_info.loc['capacity'] =  grain_info.loc['density'].mul(silo_info.loc['capacity'] , level=1)
    grain_info.loc['dep'] =  silo_info.loc['dep'].div(grain_info.loc['capacity'] , level=1)
    grain_info.loc['cost'] =  (silo_info.loc['insurance'] + silo_info.loc['other']).div(grain_info.loc['capacity'] , level=1) #variable cost = insurance + other (cleaning silo etc)
    grain_info.loc['asset'] =  silo_info.loc['asset'].div(grain_info.loc['capacity'], level=1)
    grain_info=grain_info.droplevel(1,axis=1) #drop silo type index

    len_nv = nv['len_nv']
    keys_f = np.array(['nv{0}'.format(i) for i in range(nv['len_nv'])])
    keys_p7 = per.f_season_periods(keys=True)
    keys_k3 = grain_info.columns
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_z = zfun.f_keys_z()
    keys_p6 = pinp.period['i_fp_idx']
    # ##data to determine cash period
    # cashflow_df = per.f_cashflow_periods()
    # p_dates = cashflow_df['start date']
    # p_dates_c = p_dates.values #np version
    # p_name_c = cashflow_df['cash period'].values[:-1]

    ##determine cost of feeding in each feed period and cashflow period
    feeding_cost_k3, fuel_used_k3 = mac.sup_mach_cost()
    storage_cost_k3 = grain_info.loc['cost']

    ##confinement costs
    confinement_infra = pinp.supfeed['i_confinement_infra'].squeeze()
    ###fixed depreciation
    confinement_dep = (confinement_infra.loc['price'] - confinement_infra.loc['salvage value']) / confinement_infra.loc['life']
    ###variable r & m
    confinement_rm = pinp.supfeed['i_confinement_infra_rm']

    ##adjust feeding cost for confinement.
    nv_is_confinement_f = np.full(len_nv, False)
    nv_is_confinement_f[-1] = nv['confinement_inc'] #if confinement is included the last nv pool is confinement.
    confinement_feeding_cost_factor = uinp.supfeed['i_confinement_feeding_cost_factor'] #cost of feeding for confinement vs in paddock.
    cost_factor_f = fun.f_update(1, confinement_feeding_cost_factor, nv_is_confinement_f)
    cost_factor_f = pd.Series(cost_factor_f, index=keys_f)
    index_fk3 = pd.MultiIndex.from_product([keys_f, keys_k3])
    cost_factor_fk3 = cost_factor_f.reindex(index_fk3, level=0)
    feeding_cost_fk3 = feeding_cost_k3.mul(cost_factor_fk3, level=1)
    fuel_used_fk3 = fuel_used_k3.mul(cost_factor_fk3, level=1)
    ###variable confinemnet infra r&m costs only occur for confinement nv pool
    confinement_rm_f = confinement_rm * nv_is_confinement_f
    confinement_rm_f = pd.Series(confinement_rm_f, index=keys_f)
    feeding_cost_fk3 = feeding_cost_fk3.add(confinement_rm_f, level=0)
    ###fixed confinemnet infra depreciation is only incured if confinement is included
    confinement_dep = confinement_dep * nv['confinement_inc']


    ##feeding cost allocaion
    start_p6z = per.f_feed_periods()[:-1,:]
    sup_cost_allocation_p7zp6, sup_wc_allocation_c0p7zp6 = fin.f_cashflow_allocation(start_p6z.T, enterprise='stk', z_pos=-2)
    ###convert to df
    new_index_p7zp6 = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p6])
    sup_cost_allocation_p7zp6 = pd.Series(sup_cost_allocation_p7zp6.ravel(), index=new_index_p7zp6)
    new_index_c0p7zp6 = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p6])
    sup_wc_allocation_c0p7zp6 = pd.Series(sup_wc_allocation_c0p7zp6.ravel(), index=new_index_c0p7zp6)
    ###reindex
    cols_p7zp6k3 = pd.MultiIndex.from_product([keys_p7, keys_z, keys_p6, keys_k3])
    sup_cost_allocation_p7zp6k3 = sup_cost_allocation_p7zp6.reindex(cols_p7zp6k3)
    cols_c0p7zp6k3 = pd.MultiIndex.from_product([keys_c0, keys_p7, keys_z, keys_p6, keys_k3])
    sup_wc_allocation_c0p7zp6k3 = sup_wc_allocation_c0p7zp6.reindex(cols_c0p7zp6k3)

    ##adjust cost for allocation and interest
    feeding_cost_p7zp6k3_f = sup_cost_allocation_p7zp6k3.unstack(-1).mul(feeding_cost_fk3, axis=1).stack(1)
    feeding_wc_c0p7zp6k3_f = sup_wc_allocation_c0p7zp6k3.unstack(-1).mul(feeding_cost_fk3, axis=1).stack(1)
    storage_cost_p7zp6k3 = sup_cost_allocation_p7zp6k3.mul(storage_cost_k3, level=3)
    storage_wc_c0p7zp6k3 = sup_wc_allocation_c0p7zp6k3.mul(storage_cost_k3, level=4)

    ##total cost = feeding cost plus storage cost
    total_sup_cost_p7zp6k3f = feeding_cost_p7zp6k3_f.add(storage_cost_p7zp6k3, axis=0).stack()
    total_sup_wc_c0p7zp6k3f = feeding_wc_c0p7zp6k3_f.add(storage_wc_c0p7zp6k3, axis=0).stack()

    ##storage dep
    storage_dep_k3 = grain_info.loc['dep']
    ##asset
    storage_asset_k3 = grain_info.loc['asset']
    ##allocate both dep and asset to season periods so it can be transferred as seasons unfold
    alloc_p7p6z = zfun.f1_z_period_alloc(start_p6z[na,...], z_pos=-1)
    ###make df
    index_p7p6z = pd.MultiIndex.from_product([keys_p7,keys_p6,keys_z])
    alloc_p7p6z = pd.Series(alloc_p7p6z.ravel(), index=index_p7p6z)
    index_p7p6zk3 = pd.MultiIndex.from_product([keys_p7,keys_p6,keys_z,keys_k3])
    alloc_p7p6zk3 = alloc_p7p6z.reindex(index_p7p6zk3)
    storage_dep_p7p6zk3 = alloc_p7p6zk3.mul(storage_dep_k3, level=-1)
    storage_asset_p7p6zk3 = alloc_p7p6zk3.mul(storage_asset_k3, level=-1)

    ##emissions from fuel used to feed sup
    co2_fuel_co2e_fk3, ch4_fuel_co2e_fk3, n2o_fuel_co2e_fk3 = efun.f_fuel_emissions(fuel_used_fk3)
    co2e_fuel_fk3 =  co2_fuel_co2e_fk3 + ch4_fuel_co2e_fk3 + n2o_fuel_co2e_fk3
    ###build df
    index_fk3 = pd.MultiIndex.from_product([keys_f, keys_k3])
    co2e_fuel_fk3 = pd.Series(co2e_fuel_fk3.ravel(), index=index_fk3)

    ##store r_vals
    ###make z8 mask - used to uncluster
    date_season_node_p7z = per.f_season_periods()[:-1,...] #slice off end date p7
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z,z_pos=-1,mask=True)
    ###store
    fun.f1_make_r_val(r_vals, total_sup_cost_p7zp6k3f, 'total_sup_cost_p7zp6k3f', mask_season_p7z[:,:,na,na,na], z_pos=-4)
    fun.f1_make_r_val(r_vals, co2e_fuel_fk3, 'co2e_sup_fuel_fk3')

    ##return cost, dep and asset value
    return total_sup_cost_p7zp6k3f, total_sup_wc_c0p7zp6k3f, storage_dep_p7p6zk3, storage_asset_p7p6zk3, confinement_dep, co2e_fuel_fk3


def f_sup_md_vol(r_vals, nv):
    '''
    M/D and DM content of each supplement are known inputs.
    Unlike stubble and pasture, the quantity of supplementary feed consumed (the decision variables)
    are expressed including moisture content (i.e. as fed). Therefore M/D must be adjusted by the DM
    content of the feed.
    The volume of supplementary feed is calculated based on the quality of the feed. So, lower quality
    supplements (like oats) will substitute more for pasture than high quality supplements (like lupins).
    In the generator it is assumed that the availability of supplementary feed is high and that supplement is consumed
    as the first component of the animals diet. Furthermore, it is assumed that if sufficient levels of
    supplement are offered then all the sheep’s diet will be provided by supplementary feeding.
    In pyomo, the optimisation can select a combination of paddock feed and supplement that meets the volume
    constraint while providing sufficient ME intake for the animals.
    The volume required for supplement is adjusted (to 80%) because this is necessary to make the substitution rate
    in the matrix (using volumes) align with the substitution rate calculated using the GrazPlan selection routine.
    The current representation does not include an effect of high protein supplements (like lupins) overcoming
    a protein deficiency, and therefore acting as a 'true' supplement and increasing intake. If this was represented it
    would likely make low rate lupin supplementation optimal in early summer/autumn to overcome a protein deficiency.

    A portion of supplement fed in the paddock is wasted and hence not consumed. This is accounted for by
    reducing the ME and Vol of the supplement decision variable (the supplement decision variable is the supplement fed
    not the supplement consumed by livestock). In confinement, it is assumed that no supplement is wasted.

    .. note:: Supplement M/D does not go through f_effective_mei because the quantity of Sup feed can be controlled
              so the animals achieve their target weight profile and aren't gaining then losing weight.
    '''
    #todo review the volume of supplementary feed, especially the max RI==1 and non-inclusion of protein supplementation.
    ##inputs
    sup_md_vol = uinp.supfeed['sup_md_vol']
    energy_k3 = sup_md_vol.loc['energy'].values
    dry_matter_content_k3 = sup_md_vol.loc['dry matter content'].values
    prop_consumed_k3 = sup_md_vol.loc['prop consumed'].values
    prop_consumed_confinement_k3 = sup_md_vol.loc['prop consumed confinement'].values

    ##adjust wastage for confinement (no wastage in confinement).
    len_nv = nv['len_nv']
    nv_is_confinement_f = np.full(len_nv, False)
    nv_is_confinement_f[-1] = nv['confinement_inc'] #if confinement is included the last nv pool is confinement.
    prop_consumed_fk3 = fun.f_update(prop_consumed_k3, prop_consumed_confinement_k3, nv_is_confinement_f[:,na])

    ##calc vol
    ###convert md to dmd
    dmd_k3 = fsfun.f1_md_to_dmd(energy_k3/1000)
    ###calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        rq_k3 = fsfun.f_rq_cs(dmd_k3, 0)
    ###use max(1,...) to make it the same as MIDAS - this increases lupin vol slightly from what the equation returns
    ###do not calculate ra (relative availability) because assume that supplement has high availability
    ###Scale supplement volume to 0.8. This value was selected to best fit the substitution rate based on
    ###feed volumes with the substitution rate calculated using the GrazPlan diet selection routine.
    vol_kg_k3 = np.maximum(1, 1 / rq_k3) * 0.8
    ###convert vol per kg to per tonne fed - have to adjust for the actual dry matter content and wastage
    vol_tonne_fk3 = vol_kg_k3 * 1000 * prop_consumed_fk3 * dry_matter_content_k3
    vol_tonne_fk3 = vol_tonne_fk3 / (1 + sen.sap['pi'])

    ##calc ME (note: value in dict is MJ/t of DM, so doesn't need to be multiplied by 1000)
    md_tonne_fk3 = energy_k3 * prop_consumed_fk3 * dry_matter_content_k3

    ##apply season mask
    date_start_p6z = per.f_feed_periods()[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z,z_pos=-1,mask=True)
    vol_tonne_fk3p6z = vol_tonne_fk3[:,:,na,na] * mask_fp_z8var_p6z
    md_tonne_fk3p6z = md_tonne_fk3[:,:,na,na] * mask_fp_z8var_p6z

    ##build df
    keys_z = zfun.f_keys_z()
    keys_p6 = pinp.period['i_fp_idx']
    keys_f = np.array(['nv{0}'.format(i) for i in range(nv['len_nv'])])
    index = pd.MultiIndex.from_product([keys_f, sup_md_vol.columns, keys_p6, keys_z])
    vol_tonne_fk3p6z = pd.Series(vol_tonne_fk3p6z.ravel(), index=index)
    md_tonne_fk3p6z = pd.Series(md_tonne_fk3p6z.ravel(), index=index)

    ##store r_vals
    fun.f1_make_r_val(r_vals, md_tonne_fk3p6z, 'md_tonne_fk3p6z', mask_fp_z8var_p6z, z_pos=-1)

    return vol_tonne_fk3p6z, md_tonne_fk3p6z
    
 
def f_sup_emissions(r_vals, nv):
    '''
    Livestock emissions liked to consuming 1t of supplement.

    Note the supplement activity is 1t of grain with moisture therefore adjust intake for dry matter content
    because emission equations are based on dry matter.

    '''

    ##inputs
    sup_md_vol = uinp.supfeed['sup_md_vol']
    md_k3 = sup_md_vol.loc['energy'].values #MJ/kg DM
    dmd_k3 = fsfun.f1_md_to_dmd(md_k3 / 1000)
    dry_matter_content_k3 = sup_md_vol.loc['dry matter content'].values
    cp_k3 = sup_md_vol.loc['cp'].values
    prop_consumed_k3 = sup_md_vol.loc['prop consumed'].values
    prop_consumed_confinement_k3 = sup_md_vol.loc['prop consumed confinement'].values
    ###adjust wastage for confinement (less wastage in confinement).
    len_nv = nv['len_nv']
    nv_is_confinement_f = np.full(len_nv, False)
    nv_is_confinement_f[-1] = nv['confinement_inc'] #if confinement is included the last nv pool is confinement.
    prop_consumed_fk3 = fun.f_update(prop_consumed_k3, prop_consumed_confinement_k3, nv_is_confinement_f[:,na])


    ##livestock methane emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_ch4_sup_fk3 = efun.f_stock_ch4_feed_nir(1000 * dry_matter_content_k3 * prop_consumed_fk3, dmd_k3)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        stock_ch4_sup_fk3 = efun.f_stock_ch4_feed_bc(1000 * dry_matter_content_k3 * prop_consumed_fk3, md_k3 / 1000) #div 1000 to convert to kg

    ##livestock nitrous oxide emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        stock_n2o_sup_fk3 = efun.f_stock_n2o_feed_nir(1000 * dry_matter_content_k3 * prop_consumed_fk3, dmd_k3, cp_k3)

    co2e_sup_fk3 = stock_ch4_sup_fk3 * uinp.emissions['i_ch4_gwp_factor'] + stock_n2o_sup_fk3 * uinp.emissions['i_n2o_gwp_factor']

    ##build df
    keys_f = np.array(['nv{0}'.format(i) for i in range(nv['len_nv'])])
    index_fk3 = pd.MultiIndex.from_product([keys_f, sup_md_vol.columns])
    co2e_sup_fk3 = pd.Series(co2e_sup_fk3.ravel(), index=index_fk3)
    stock_n2o_sup_fk3 = pd.Series(stock_n2o_sup_fk3.ravel(), index=index_fk3)
    stock_ch4_sup_fk3 = pd.Series(stock_ch4_sup_fk3.ravel(), index=index_fk3)

    ##store report vals
    fun.f1_make_r_val(r_vals,stock_n2o_sup_fk3,'stock_n2o_sup_fk3')
    fun.f1_make_r_val(r_vals,stock_ch4_sup_fk3,'stock_ch4_sup_fk3')

    return co2e_sup_fk3

def f_sup_labour(nv):
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
    easier to calibrate, the inputs are entered in a more common format for a specific grain and then converted
    to m3. For example, the inputs used to determine time to empty the feeder are the feeding rate of lupins
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
    empty_df=empty_df.set_index(['index','DOY']).T
    ##convert to hr/m3 for lupins and hr/bale for hay
    grain_density= uinp.supfeed['grain_density'].T.reset_index() #reindex so it can be combined with different grains
    grain_density=grain_density.set_index(['index','silo type']).squeeze()
    empty_df[('grain','empty rate lupins')]=1/(empty_df[('grain','empty rate lupins')]*60*60/1000/0.72) #convert from kg/sec lupins (density of lupins is 0.72 - this has to be hard coded incase lupins are masked out of the land uses) to hr/m3 (which is the same for all grains). First convert kg/sec to t/hr then divide by density
    empty_df[('hay','empty rate')]=empty_df[('hay','empty rate')]/60 #convert min/bale to hr/bale
    ##combine time to fill and empty then convert to per tonne for each grain
    empty_df=empty_df.droplevel(1, axis=1)
    fill_empty = empty_df.add(fill_time, axis=1)
    fill_empty_tonne=fill_empty.reindex(grain_density.index, axis=1, level=1).div(grain_density).droplevel(1,axis=1)
    ##calc time between paddocks
    ###convert lupin rate fed to mj/hd/d
    feedrate=pinp.supfeed['feed_rate']
    mj=feedrate['feed rate']/1000000*13300 #13300 is energy of 1t lupins. Divide by 1000000 because convert g to tonnes because energy is in mj/tonne
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
    lp_dates_p5z = per.f_p_dates_df()
    np_lp_dates_p5z = lp_dates_p5z.values
    lp_start_p5z = np_lp_dates_p5z[:-1]
    lp_end_p5z = np_lp_dates_p5z[1:]
    lp_len_p5z = lp_end_p5z - lp_start_p5z
    dates_p8 = total_time.index.values
    end = dates_p8[0] + 364 #increment the first date by 1yr so it becomes the end date for the last period
    dates_p8 = np.concatenate([dates_p8,np.array([end])])
    shape_p8p5z = dates_p8.shape + lp_start_p5z.shape
    ####allocate labour periods into p8 periods
    alloc_p8p5z = fun.f_range_allocation_np(dates_p8[:,na,na], lp_start_p5z, lp_len_p5z, shape=shape_p8p5z)[:-1]
    alloc_p5zp8 = np.moveaxis(alloc_p8p5z,0,-1)

    ##combine allocation with the labour time
    total_time_p8k3 = total_time.values
    total_time_p5zp8k3 = alloc_p5zp8[...,na] * total_time_p8k3
    total_time_p5zk3 = np.sum(total_time_p5zp8k3, axis=-2)

    ##link feed periods to labour periods, ie determine the proportion of each feed period in each labour period so the time taken to sup feed can be divided up accordingly
    start_p6z = per.f_feed_periods()[:-1,:]
    length_p6z = per.f_feed_periods(option=1)
    shape_p5p6z = (lp_dates_p5z.shape[0],) + length_p6z.shape
    alloc_p5p6z = fun.f_range_allocation_np(lp_dates_p5z.values[:,na,:], start_p6z, length_p6z, shape=shape_p5p6z)[:-1]

    ##allocate time to labour period for each feed period - get the time taken in each labour period to feed 1t of feed in each feed period
    total_time_p5p6zk3 = total_time_p5zk3[:,na,...] * alloc_p5p6z[...,na]

    ##adjust feeding labour for confinement.
    len_nv = nv['len_nv']
    nv_is_confinement_f = np.full(len_nv, False)
    nv_is_confinement_f[-1] = nv['confinement_inc'] #if confinement is included the last nv pool is confinement.
    confinement_feeding_cost_factor = uinp.supfeed['i_confinement_feeding_labour_factor'] #cost of feeding for confinement vs in paddock.
    cost_factor_f = fun.f_update(1, confinement_feeding_cost_factor, nv_is_confinement_f)
    total_time_p5p6zk3f = total_time_p5p6zk3[...,na] * cost_factor_f

    ##apply season mask
    date_start_p6z = per.f_feed_periods()[:-1]
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    total_time_p5p6zk3f = total_time_p5p6zk3f * mask_fp_z8var_p6z[...,na,na]

    ##build df
    keys_z = zfun.f_keys_z()
    keys_p5 = lp_dates_p5z.index[:-1]
    keys_p6 = pinp.period['i_fp_idx']
    keys_k3 = uinp.supfeed['grain_density'].columns
    keys_f = np.array(['nv{0}'.format(i) for i in range(nv['len_nv'])])
    index = pd.MultiIndex.from_product([keys_p5, keys_p6, keys_z, keys_k3, keys_f])
    total_time_p5p6zk3f = pd.Series(total_time_p5p6zk3f.ravel(), index=index)
    return total_time_p5p6zk3f


def f1_a_p6_p7():
    '''
    Association between p6 and p7. Used to link supplement consumed in each p6 with grain transfer which has p7 periods.
    '''
    start_p6z = per.f_feed_periods()[:-1,:]
    alloc_p7p6z = zfun.f1_z_period_alloc(start_p6z[na,:,:], z_pos=-1)

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p7 = per.f_season_periods(keys=True)
    keys_p6 = pinp.period['i_fp_idx']
    new_index_p7p6z = pd.MultiIndex.from_product([keys_p7, keys_p6, keys_z])
    alloc_p7p6z = pd.Series(alloc_p7p6z.ravel(), index=new_index_p7p6z)
    return alloc_p7p6z


def f1_sup_s2_ks2(r_vals):
    '''
    Association between supplement feed and s2.

    This param is required because v_sup does not have a s2 axis but product transfer does.
    An s2 set could be added to supplement but at the time of building it was deemed not to be worth the effort.
    The lack of s2 axis means that each crop only has one s2 slice that provides supp. E.g. barely only provides
    grain supplement. Therefore barely that was tactically baled for hay can't be fed as supplement. It can
    only be sold. This is not a big limitation because the model can just sell the barley hay and purchase some oat hay
    to feed.
    '''
    sup_s2_k3_s2 = uinp.supfeed['i_sup_s2_ks2']
    fun.f1_make_r_val(r_vals, sup_s2_k3_s2, 'sup_s2_k_s2')
    return sup_s2_k3_s2.stack()


def f1_sup_selectivity():
    '''
    Selectivity param of paddock feed to supplement when trail feeding.

    If sheep are being fed grain in the paddock they will still each some pasture. This function builds the
    parameter which represents the selectivity. This param is multiplied by the feed volume.
    The default is one meaning that the stock eat an equivalent volume of paddock feed as they do supplement.
    Using volume is good because it accounts for the impact of foo and quality on the selectivity e.g. if the
    FOO is low stock won't have to consume as much because the volume is higher.

    This constraint only acts for the green feed periods because we dont want the model to be able to defer pastures
    by feeding supplement to stock that are not in confinement. This is because if there is green feed the sheep will
    eat some green feed even if they are being fed supplement.

    '''

    i_max_sup_selectivity = uinp.supfeed['i_max_sup_selectivity']

    ##adjust for p6 - only want this constraint to be active when grn feed exists.
    ###create dry and green pasture exists mask
    ###in the late brk season dry feed can occur in fp0&1.
    feed_period_dates_p6z = per.f_feed_periods()
    index_p6 = np.arange(feed_period_dates_p6z.shape[0]-1)
    season_break_z = zfun.f_seasonal_inp(pinp.general['i_break'], numpy=True)
    idx_fp_start_gs_z = fun.searchsort_multiple_dim(feed_period_dates_p6z, season_break_z, 1, 0, side='right') - 1
    i_end_of_gs_z = zfun.f_seasonal_inp(pinp.pasture_inputs['annual']['EndGS'], numpy=True)
    mask_greenfeed_exists_p6z = np.logical_or(np.logical_and(index_p6[:,na]>=idx_fp_start_gs_z, index_p6[:, na] <= i_end_of_gs_z),       #green exists in the period which is the end of growing season hence <=
                                              np.logical_and(i_end_of_gs_z < idx_fp_start_gs_z,
                                                             np.logical_or(index_p6[:,na]>=idx_fp_start_gs_z, index_p6[:,na]<=i_end_of_gs_z)))   #this handles if green feed starts mid fp and wraps around to start fps.
    max_sup_selectivity_p6z = fun.f_update(i_max_sup_selectivity, 1, np.logical_not(mask_greenfeed_exists_p6z)) #can have as much sup as it wants when it is not green

    ##make df
    keys_z = zfun.f_keys_z()
    keys_p6 = pinp.period['i_fp_idx']
    index_p6z = pd.MultiIndex.from_product([keys_p6, keys_z])
    max_sup_selectivity_p6z = pd.Series(max_sup_selectivity_p6z.ravel(), index=index_p6z)

    return max_sup_selectivity_p6z


##collates all the params
def f_sup_params(params,r_vals, nv):
    total_sup_cost, total_sup_wc, storage_dep, storage_asset, confinement_dep, co2e_fuel_fk = f_sup_feeding_cost(r_vals, nv)
    vol_tonne, md_tonne = f_sup_md_vol(r_vals, nv)
    co2e_sup_consumption_fk = f_sup_emissions(r_vals, nv)
    sup_labour = f_sup_labour(nv)
    buy_grain_price, buy_grain_wc, buy_grain_prov_p7z = f_buy_grain_price(r_vals)
    a_p6_p7 = f1_a_p6_p7()
    sup_s2_ks2 = f1_sup_s2_ks2(r_vals)
    max_sup_selectivity_p6z = f1_sup_selectivity()


    ##create non seasonal params
    params['confinement_dep'] = confinement_dep
    params['storage_dep'] = storage_dep.to_dict()
    params['storage_asset'] = storage_asset.to_dict()
    params['vol_tonne'] = vol_tonne.to_dict()
    params['md_tonne'] = md_tonne.to_dict()
    params['buy_grain_price'] = buy_grain_price.to_dict()
    params['buy_grain_wc'] = buy_grain_wc.to_dict()
    params['buy_grain_prov_p7z'] = buy_grain_prov_p7z.to_dict()

    ##create season params
    params['co2e_sup_fk'] = co2e_sup_consumption_fk.add(co2e_fuel_fk).to_dict()
    params['total_sup_cost'] = total_sup_cost.to_dict()
    params['total_sup_wc'] = total_sup_wc.to_dict()
    params['sup_labour'] = sup_labour.to_dict()
    params['a_p6_p7'] = a_p6_p7.to_dict()
    params['sup_s2_ks2'] = sup_s2_ks2.to_dict()
    params['max_sup_selectivity_p6z'] = max_sup_selectivity_p6z.to_dict()

