"""
author: young

Salt land pastures (SLP) are a novel feed source that consists of saltbushes and a grazable pasture understory. SLP
establishment requires labour, specific machinery and a significant financial outlay, however it comes with numerous
characteristics which make it attractive for certain situations. These characteristics include:

    1.	Saline tolerance and can therefore be established on land management units that would have had very low productivity.
    2.	Draw-down of the water table (Barrett-Lennard and Malcolm, 1999). This drawdown allows salts to be flushed from the topsoil of the moderately saline land, thereby creating growing conditions more suited to higher productivity annual pastures and perhaps leads to long term rehabilitation of the area.
    3.	Edible leaf for livestock consumption.
    4.	Livestock shelter – shelter provided by shrubs can be used by stock at vulnerable times such as lambing which helps increase animal survival.
    5.	Increased wool growth due to additional nutrients provided by grazing saltbush.
    6.	Reduced erosion risk due to the wind protection provided by the saltbushes year-round.

The salt land pasture land use is a combination of saltbush and understory. The saltbush module represents the
saltbush component and the understory component is calculated in the pasture module.

The saltbush module includes the:

    •	cost of salt land pasture establishment and maintenance.
    •	productivity of saltbush during the year based on grazing management.
    •	feed value of saltbush during the year.
    •	diet selectivity of salt bush versus understory.
    •	impact of salt consumption on animal intake.
    •	livestock emissions linked to the consumption of saltbush.


"""
import numpy as np
import numpy_financial as npf

from . import PropertyInputs as pinp
from . import UniversalInputs as uinp
from . import StructuralInputs as sinp
from . import Periods as per
from . import Functions as fun
from . import FeedsupplyFunctions as fsfun
from . import SeasonalFunctions as zfun
from . import EmissionFunctions as efun
from . import Finance as fin

na = np.newaxis

def f_saltbush_precalcs(params, r_vals, nv):


    ###########################
    # salt land pasture area  #
    ###########################
    ##inputs
    sb_landuse = pinp.saltbush['i_sb_landuse']
    phases_rotn_df = pinp.phases_r

    ##determine which phases provide slp
    phase_is_slp_r = phases_rotn_df.iloc[:,-1] == sb_landuse

    ###########################
    # salt land pasture costs #
    ###########################
    ##inputs
    rate = uinp.finance['i_interest']
    sb_estab_costs = pinp.saltbush['i_saltbush_estab_cost']
    sb_prob_success = pinp.saltbush['i_saltbush_success']
    sb_life = pinp.saltbush['i_saltbush_life']
    us_estab_costs = pinp.saltbush['i_understory_estab_cost']
    us_prob_success = pinp.saltbush['i_understory_success']
    us_life = pinp.saltbush['i_understory_life']
    slp_estab_date = np.array([pinp.saltbush['i_sb_estab_date']])

    ##calc cost per year of salt land pasture
    sb_annual_cost = npf.pmt(rate, sb_life, -(sb_estab_costs / sb_prob_success))
    us_annual_cost = npf.pmt(rate, us_life, -(us_estab_costs / us_prob_success))
    slp_total_cost = sb_annual_cost + us_annual_cost

    ##p7 allocation
    stub_cost_allocation_p7z, stub_wc_allocation_c0p7z = fin.f_cashflow_allocation(slp_estab_date, enterprise='stk', z_pos=-1)
    slp_estab_cost_p7z = slp_total_cost * stub_cost_allocation_p7z
    slp_estab_wc_c0p7z = slp_total_cost * stub_wc_allocation_c0p7z

    ###########################
    # saltbush productivity   #
    ###########################
    ##inputs
    sb_expected_foo_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_expected_foo_zp6'], numpy=True, axis=0) #g/stem
    sb_expected_growth_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_expected_growth_zp6'], numpy=True, axis=0) #g/stem/day
    sb_growth_reduction_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_growth_reduction_zp6'], numpy=True, axis=0) #% per day
    sb_stems_per_ha = pinp.saltbush['i_sbstemspha'] #saltbush stems/ha
    lmu_mask = pinp.general['i_lmu_area'] > 0
    sb_lmu_scalar_l = pinp.saltbush['i_sb_lmu_scalar'][lmu_mask]

    date_feed_periods = per.f_feed_periods()
    date_start_p6z = date_feed_periods[:-1]
    date_end_p6z = date_feed_periods[1:]
    len_zp6 = date_end_p6z.T - date_start_p6z.T

    ##saltbush max growth (note max growth occurs when saltbush is full grazed).
    ## feed per plant grows to a maximum and then net growth stops.
    ## So the net growth rate reduces with higher Foo in an asymptotic way. Therefore AFOs representation is not perfect
    ## but it is reasonable. See google doc for a method to improve this.
    max_growth_per_bush_zp6 = sb_expected_growth_zp6 + sb_expected_foo_zp6 * sb_growth_reduction_zp6 #g/stem/d
    max_growth_per_ha_zp6 = (max_growth_per_bush_zp6/1000) * len_zp6 * sb_stems_per_ha #kg/ha/period

    ##Adjust growth for average saltbush production over life of stand
    sb_ave_prodn = pinp.saltbush['i_sb_ave_prodn'] #potential of production achieved in each year
    max_growth_per_ha_zp6 = max_growth_per_ha_zp6 * sb_ave_prodn

    ##adjust for lmu
    max_growth_per_ha_zp6l = max_growth_per_ha_zp6[...,na] * sb_lmu_scalar_l

    ##loss in prodn of saltbush when ungrazed (%/period)
    prodn_loss_zp6 = 1 - (1 - sb_growth_reduction_zp6) ** len_zp6
    transfer_prov_zp6 = (1-prodn_loss_zp6) * 1000


    ###########################
    # saltbush feed value     #
    ###########################
    ##inputs
    sb_ash_content_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_ash_content_zp6'], numpy=True, axis=0)
    sb_omd = pinp.saltbush['i_sb_omd']*100

    ##nv stuff
    len_nv = nv['len_nv']
    nv_is_not_confinement_f = np.full(len_nv, True)
    nv_is_not_confinement_f[-1] = np.logical_not(nv['confinement_inc']) #if confinement is included the last nv pool is confinement.

    ##SB ME
    sb_domd_zp6 = sb_omd*(1-sb_ash_content_zp6)
    sb_me_zp6 = ((0.18*sb_domd_zp6)-1.8)*1000 #mj/t - because of the salt we can't use the same dmd2me function in other feed modules.
    ###SLP cannot be grazed in the confinement pool hence me is 0
    sb_me_zp6f = sb_me_zp6[...,na] * nv_is_not_confinement_f

    ##SB vol
    sb_dmd_zp6 = sb_domd_zp6 + (sb_ash_content_zp6-0.04)*100
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        sb_rq_zp6 = fsfun.f_rq_cs(sb_dmd_zp6, legume=0)
    ### calc relative availability - this is always 1
    sb_ra_zp6 = 1
    ###combine ra and rq
    sb_ri_zp6 = fsfun.f_rel_intake(sb_rq_zp6, sb_ra_zp6, legume=0)
    sb_vol_zp6 = fun.f_divide(1000, sb_ri_zp6)  # 1000 to convert to vol per tonne
    sb_vol_zp6f = sb_vol_zp6[:,:,na] * nv_is_not_confinement_f #me from SLP is 0 in the confinement pool


    ###########################
    # saltbush selectivity    #
    ###########################
    ##The ratio of the volume of saltbush consumed to the total volume of feed consumed by stock grazing saltbush and
    ## understory is based on 2006 data from Hayley Norman (pers comm.). The data is manipulated to include relative
    ## intake of the understory. The r squared of the relationship is 0.93. More detail is included in the AFO calibration
    ## directory in the spreadsheet "Saltland pasture diet selection prediction Hayley Norman-June2006.xlsx
    sb_selectivity_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_selectivity_zp6'], numpy=True, axis=0)

    ###########################
    # saltbush animal effects #
    ###########################
    ##inputs
    slp_diet_propn_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_slp_diet_propn_zp6'], numpy=True, axis=0)
    sb_animal_salt_threshold = pinp.saltbush['i_sb_animal_salt_threshold']
    sb_animal_salt_slope = pinp.saltbush['i_sb_animal_salt_slope']
    sb_salt_content_normal_feed = pinp.saltbush['i_sb_salt_content_normal_feed']
    sb_animal_typical_intake = pinp.saltbush['i_sb_animal_typical_intake']

    ##% salt in SLP diet
    slp_diet_salt_content_zp6 = sb_ash_content_zp6 * sb_selectivity_zp6 + sb_salt_content_normal_feed * (1-sb_selectivity_zp6)

    ##% salt in total diet (required to handle when the sheep is not grazing slp for the whole period)
    diet_salt_content_zp6 = slp_diet_salt_content_zp6 * slp_diet_propn_zp6 + sb_salt_content_normal_feed * (1-slp_diet_propn_zp6)

    ##scale volume for anti-nutritional effect due to salt in saltbush
    ###propn of max intake
    total_intake_redction_salt_zp6 = 1 - np.minimum(1, 1 - (diet_salt_content_zp6 - sb_animal_salt_threshold) * sb_animal_salt_slope / sb_animal_typical_intake)
    ###saltbush vol is scalled to reflect total diet intake reduction
    sb_vol_scalar_salt_zp6 = total_intake_redction_salt_zp6 / sb_selectivity_zp6 / slp_diet_propn_zp6
    sb_vol_zp6f = sb_vol_zp6f * (1+sb_vol_scalar_salt_zp6)[:,:,na]


    ###########################
    # emissions               #
    ###########################
    ##inputs
    sb_cp_zp6 = zfun.f_seasonal_inp(pinp.saltbush['i_sb_cp_zp6'], numpy=True, axis=0)

    ##livestock methane emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 0:  # National Greenhouse Gas Inventory Report
        ch4_sb_zp6 = efun.f_stock_ch4_feed_nir(1000, sb_dmd_zp6)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][12, 0] == 1:  #Baxter and Claperton
        ch4_sb_zp6 = efun.f_stock_ch4_feed_bc(1000, sb_me_zp6/1000) #have to divide by 1000 because it was ME/t

    ##livestock nitrous oxide emissions linked to the consumption of 1t of saltbush - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][13, 0] == 0:  # National Greenhouse Gas Inventory Report
        n2o_sb_zp6 = efun.f_stock_n2o_feed_nir(1000, sb_dmd_zp6, sb_cp_zp6)

    co2e_sb_zp6 = ch4_sb_zp6 * uinp.emissions['i_ch4_gwp_factor'] + n2o_sb_zp6 * uinp.emissions['i_n2o_gwp_factor']

    ###########################
    # create params           #
    ###########################

    ##make season mask
    mask_fp_z8var_p6z = zfun.f_season_transfer_mask(date_start_p6z, z_pos=-1, mask=True)
    mask_fp_z8var_zp6 = mask_fp_z8var_p6z.T
    date_season_node_p7z = per.f_season_periods()[:-1, ...]
    mask_season_p7z = zfun.f_season_transfer_mask(date_season_node_p7z, z_pos=-1, mask=True)

    ##apply mask
    slp_estab_cost_p7z = slp_estab_cost_p7z * mask_season_p7z
    slp_estab_wc_c0p7z = slp_estab_wc_c0p7z * mask_season_p7z
    max_growth_per_ha_zp6l = max_growth_per_ha_zp6l * mask_fp_z8var_zp6[:,:,na]
    transfer_prov_zp6 = transfer_prov_zp6 * mask_fp_z8var_zp6
    sb_selectivity_zp6 = sb_selectivity_zp6 * mask_fp_z8var_zp6
    sb_me_zp6f = sb_me_zp6f * mask_fp_z8var_zp6[:,:,na]
    sb_vol_zp6f = sb_vol_zp6f * mask_fp_z8var_zp6[:,:,na]
    co2e_sb_zp6 = co2e_sb_zp6 * mask_fp_z8var_zp6

    ##make key arrays
    ###keys
    keys_l = pinp.general['i_lmu_idx'][lmu_mask]
    keys_r  = np.array(phases_rotn_df.index).astype('str')
    keys_c0 = sinp.general['i_enterprises_c0']
    keys_p7 = per.f_season_periods(keys=True)
    keys_p6 = pinp.period['i_fp_idx']
    keys_f = np.array(['nv{0}' .format(i) for i in range(nv['len_nv'])])
    keys_z = zfun.f_keys_z()
    ###make arrays from keys
    arrays_p7z = [keys_p7, keys_z]
    arrays_c0p7z = [keys_c0, keys_p7, keys_z]
    arrays_zp6l = [keys_z, keys_p6, keys_l]
    arrays_zp6 = [keys_z, keys_p6]
    arrays_zp6f = [keys_z, keys_p6, keys_f]


    ##make params
    params['phase_slp_area_r'] = fun.f1_make_pyomo_dict(phase_is_slp_r*1, [keys_r])
    params['slp_estab_cost_p7z'] = fun.f1_make_pyomo_dict(slp_estab_cost_p7z, arrays_p7z)
    params['slp_estab_wc_c0p7z'] = fun.f1_make_pyomo_dict(slp_estab_wc_c0p7z, arrays_c0p7z)
    params['max_growth_per_ha_zp6l'] = fun.f1_make_pyomo_dict(max_growth_per_ha_zp6l, arrays_zp6l)
    params['transfer_prov_zp6'] = fun.f1_make_pyomo_dict(transfer_prov_zp6, arrays_zp6)
    params['sb_me_zp6f'] = fun.f1_make_pyomo_dict(sb_me_zp6f, arrays_zp6f)
    params['sb_vol_zp6f'] = fun.f1_make_pyomo_dict(sb_vol_zp6f, arrays_zp6f)
    params['sb_selectivity_zp6'] = fun.f1_make_pyomo_dict(sb_selectivity_zp6, arrays_zp6)
    params['co2e_sb_zp6'] = fun.f1_make_pyomo_dict(co2e_sb_zp6, arrays_zp6)

    ##make r_vals
    fun.f1_make_r_val(r_vals,sb_me_zp6f,'sb_me_zp6f',mask_fp_z8var_zp6[...,na],z_pos=-2)
    fun.f1_make_r_val(r_vals,slp_estab_cost_p7z,'slp_estab_cost_p7z',mask_season_p7z,z_pos=-1)
    ###emissions
    fun.f1_make_r_val(r_vals,ch4_sb_zp6,'ch4_sb_zp6',mask_fp_z8var_zp6,z_pos=-2)
    fun.f1_make_r_val(r_vals,n2o_sb_zp6,'n2o_sb_zp6',mask_fp_z8var_zp6,z_pos=-2)






