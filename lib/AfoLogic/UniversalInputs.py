"""

These are inputs that are expected to remain constant between regions and properties, this includes:

* Prices of inputs
* Value of outputs (grain, wool ,meat)
* Interest rates & depreciation rates
* Machinery options
* Sheep parameters and definition of genotypes

author: young
"""

##python modules
import pickle as pkl
import numpy as np
import pandas as pd
import copy
import os.path

from . import Functions as fun
from . import PropertyInputs as pinp
try:
    from Inputs import TreeUniversalInputs as tinp #raw version
except ImportError:
    from module.afo.Inputs import TreeUniversalInputs as tinp  # Web app case


def f_reshape_uinp_defaults(uinp_defaults):
    ##lengths
    len_h1 = uinp_defaults["sheep_inp"]['i_husb_muster_infrastructurereq_h1h4'].shape[-1]
    len_h4 = uinp_defaults["sheep_inp"]['i_h4_len']
    len_h6 = uinp_defaults["sheep_inp"]['i_husb_muster_requisites_prob_h6h4'].shape[-1]
    len_l2 = uinp_defaults["sheep_inp"]['i_husb_muster_labourreq_l2h4'].shape[-1]
    len_p4 = uinp_defaults["sheep_inp"]['i_salep_months_priceadj_s7s9p4'].shape[-1]
    len_s5 = uinp_defaults["sheep_inp"]['i_s5_len']
    len_s6 = uinp_defaults["sheep_inp"]['i_salep_score_scalar_s7s5s6'].shape[-1]
    len_s7 = uinp_defaults["sheep_inp"]['i_s7_len']
    len_s9 = uinp_defaults["sheep_inp"]['i_s9_len']


    ###shapes
    h4h1 = (len_h4, len_h1)
    h4h6 = (len_h4, len_h6)
    h4l2 = (len_h4, len_l2)
    s7s9p4 = (len_s7, len_s9, len_p4)
    s7s5s6 = (len_s7, len_s5, len_s6)
    s7s5 = (len_s7, len_s5)
    cb0 = (uinp_defaults["parameters_inp"]['i_cb0_len'], uinp_defaults["parameters_inp"]['i_cb0_len2'],-1)
    ce = (uinp_defaults["parameters_inp"]['i_ce_len'], uinp_defaults["parameters_inp"]['i_ce_len2'],-1)
    cl0 = (uinp_defaults["parameters_inp"]['i_cl0_len'], uinp_defaults["parameters_inp"]['i_cl0_len2'],-1)
    cl1 = (uinp_defaults["parameters_inp"]['i_cl1_len'], uinp_defaults["parameters_inp"]['i_cl1_len2'],-1)
    cu1 = (uinp_defaults["parameters_inp"]['i_cu1_len'], uinp_defaults["parameters_inp"]['i_cu1_len2'],-1)
    cu2 = (uinp_defaults["parameters_inp"]['i_cu2_len'], uinp_defaults["parameters_inp"]['i_cu2_len2'],-1)
    cu6 = (uinp_defaults["parameters_inp"]['i_cu6_len'], uinp_defaults["parameters_inp"]['i_cu6_len2'],-1)
    cx = (uinp_defaults["parameters_inp"]['i_cx_len'], uinp_defaults["parameters_inp"]['i_cx_len2'],-1)

    ###price
    uinp_defaults["price_inp"]['grain_price'] = uinp_defaults["price_inp"]['grain_price'].T.set_index(['percentile'], append=True).T.astype(float) #convert to float because array was initialised with string as well therefore it is an object type.

    ###stock
    uinp_defaults["sheep_inp"]['i_salep_months_priceadj_s7s9p4'] = np.reshape(uinp_defaults["sheep_inp"]['i_salep_months_priceadj_s7s9p4'], s7s9p4)
    uinp_defaults["sheep_inp"]['i_salep_score_scalar_s7s5s6'] = np.reshape(uinp_defaults["sheep_inp"]['i_salep_score_scalar_s7s5s6'], s7s5s6)
    uinp_defaults["sheep_inp"]['i_salep_weight_scalar_s7s5s6'] = np.reshape(uinp_defaults["sheep_inp"]['i_salep_weight_scalar_s7s5s6'], s7s5s6)
    uinp_defaults["sheep_inp"]['i_salep_weight_range_s7s5'] = np.reshape(uinp_defaults["sheep_inp"]['i_salep_weight_range_s7s5'], s7s5)
    uinp_defaults["sheep_inp"]['i_husb_muster_requisites_prob_h6h4'] = np.reshape(uinp_defaults["sheep_inp"]['i_husb_muster_requisites_prob_h6h4'], h4h6)
    uinp_defaults["sheep_inp"]['i_husb_muster_labourreq_l2h4'] = np.reshape(uinp_defaults["sheep_inp"]['i_husb_muster_labourreq_l2h4'], h4l2)
    uinp_defaults["sheep_inp"]['i_husb_muster_infrastructurereq_h1h4'] = np.reshape(uinp_defaults["sheep_inp"]['i_husb_muster_infrastructurereq_h1h4'], h4h1)
    uinp_defaults["parameters_inp"]['i_cb0_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cb0_c2'], cb0)
    uinp_defaults["parameters_inp"]['i_cb0_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cb0_y'], cb0)
    uinp_defaults["parameters_inp"]['i_ce_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_ce_c2'], ce)
    uinp_defaults["parameters_inp"]['i_ce_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_ce_y'], ce)
    uinp_defaults["parameters_inp"]['i_cl0_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cl0_c2'], cl0)
    uinp_defaults["parameters_inp"]['i_cl0_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cl0_y'], cl0)
    uinp_defaults["parameters_inp"]['i_cl1_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cl1_c2'], cl1)
    uinp_defaults["parameters_inp"]['i_cl1_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cl1_y'], cl1)
    uinp_defaults["parameters_inp"]['i_cu1_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu1_c2'], cu1)
    uinp_defaults["parameters_inp"]['i_cu1_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu1_y'], cu1)
    uinp_defaults["parameters_inp"]['i_cu2_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu2_c2'], cu2)
    uinp_defaults["parameters_inp"]['i_cu2_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu2_y'], cu2)
    uinp_defaults["parameters_inp"]['i_cu6_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu6_c2'], cu6)
    uinp_defaults["parameters_inp"]['i_cu6_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cu6_y'], cu6)
    uinp_defaults["parameters_inp"]['i_cx_c2'] = np.reshape(uinp_defaults["parameters_inp"]['i_cx_c2'], cx)
    uinp_defaults["parameters_inp"]['i_cx_y'] = np.reshape(uinp_defaults["parameters_inp"]['i_cx_y'], cx)

    ##pasture
    uinp_defaults["pastparameters_inp"]['i_cu3_c4'] = uinp_defaults["pastparameters_inp"]['i_cu3_c4'].reshape(uinp_defaults["pastparameters_inp"]['i_cu3_len'], uinp_defaults["pastparameters_inp"]['i_cu3_len2'], -1)
    uinp_defaults["pastparameters_inp"]['i_cu4_c4'] = uinp_defaults["pastparameters_inp"]['i_cu4_c4'].reshape(uinp_defaults["pastparameters_inp"]['i_cu4_len'], uinp_defaults["pastparameters_inp"]['i_cu4_len2'], -1)


#######################
#reset defaults       #
#######################
def f_select_n_reset_uinp(uinp_defaults):
    ##occurs for each trial
    ##create a copy of each input dict - so that the base inputs remain unchanged
    ##the copy created is the one used in the actual modules
    ###NOTE: if an input sheet is added remember to add it to the dict reset in f_sa() below.
    global general
    global price
    global finance
    global mach_general
    global supfeed
    global stubble
    global emissions
    global crop
    global sheep
    global parameters
    global pastparameters
    global mach
    global price_variation
    global tree
    general = copy.deepcopy(uinp_defaults["general_inp"])
    price = copy.deepcopy(uinp_defaults["price_inp"])
    finance = copy.deepcopy(uinp_defaults["finance_inp"])
    mach_general = copy.deepcopy(uinp_defaults["mach_general_inp"])
    supfeed = copy.deepcopy(uinp_defaults["sup_inp"])
    stubble = copy.deepcopy(uinp_defaults["stubble_inp"])
    emissions = copy.deepcopy(uinp_defaults["emissions_inp"])
    crop = copy.deepcopy(uinp_defaults["crop_inp"])
    sheep = copy.deepcopy(uinp_defaults["sheep_inp"])
    parameters = copy.deepcopy(uinp_defaults["parameters_inp"])
    pastparameters = copy.deepcopy(uinp_defaults["pastparameters_inp"])
    mach = copy.deepcopy(uinp_defaults["machine_options_dict_inp"])
    price_variation = copy.deepcopy(uinp_defaults["price_variation_inp"])
    tree = copy.deepcopy(tinp.tree_inputs)

#######################
#apply SA             #
#######################
def f_universal_inp_sa(uinp_defaults):
    '''
    Applies sensitivity adjustment to each input.
    This function gets called at the beginning of each loop in the RunAfoRaw_Multi.py module.

    SA order is: sav, sam, sap, saa, sat, sar.
    So that multiple SA can be applied to one input.

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    from . import Sensitivity as sen

    ##general
    ###SAV
    general['i_utility_method'] = fun.f_sa(general['i_utility_method'], sen.sav['utility_method'], 5)
    general['i_cara_risk_coef'] = fun.f_sa(general['i_cara_risk_coef'], sen.sav['cara_risk_coef'], 5)
    general['i_crra_risk_coef'] = fun.f_sa(general['i_crra_risk_coef'], sen.sav['crra_risk_coef'], 5)
    general['i_a_expo_risk_coef'] = fun.f_sa(general['i_a_expo_risk_coef'], sen.sav['expo_risk_coef'], 5)

    ##finance
    ###SAV
    finance['i_working_capital_constraint_included'] = fun.f_sa(finance['i_working_capital_constraint_included'], sen.sav['working_cap_constraint_included'], 5)  #control inclusion of work cap constraint in corepyomo
    finance['minroe'] = fun.f_sa(finance['minroe'], sen.sav['minroe'], 5)  #value for minroe (same sav as below)
    finance['minroe_dsp'] = fun.f_sa(finance['minroe_dsp'], sen.sav['minroe'], 5)  #value for minroe (same sav as above)
    finance['i_interest'] = fun.f_sa(finance['i_interest'], sen.sav['interest_rate'], 5)  #value for bank interest rate
    finance['opportunity_cost_capital'] = fun.f_sa(finance['opportunity_cost_capital'], sen.sav['opp_cost_capital'], 5)  #value for opportunity cost of capital
    finance['fixed_dep'] = fun.f_sa(finance['fixed_dep'], sen.sav['fixed_dep_rate'], 5)  #value for fixed rate of machinery depreciation per year
    finance['equip_insurance'] = fun.f_sa(finance['equip_insurance'], sen.sav['equip_insurance_rate'], 5)  #value for machinery insurance (as a propn of total value)

    ##price
    ###sav
    price['grain_price_percentile'] = fun.f_sa(price['grain_price_percentile'],sen.sav['grain_percentile'], 5)
    price['manager_cost'] = fun.f_sa(price['manager_cost'], sen.sav['manager_cost'], 5)
    price['permanent_cost'] = fun.f_sa(price['permanent_cost'], sen.sav['permanent_cost'], 5)
    price['casual_cost'] = fun.f_sa(price['casual_cost'], sen.sav['casual_cost'], 5)

    ##Mach general
    mach_general['i_variable_dep_hr_seeding'] = fun.f_sa(mach_general['i_variable_dep_hr_seeding'], sen.sav['variable_dep_hr_seeding'], 5)  #value for variable rate of machinery depreciation per year if cropping 1000ha
    mach_general['i_variable_dep_hr_harv'] = fun.f_sa(mach_general['i_variable_dep_hr_harv'], sen.sav['variable_dep_hr_harv'], 5)  #value for variable rate of machinery depreciation per year if cropping 1000ha
    mach_general['i_variable_dep_hr_spraying'] = fun.f_sa(mach_general['i_variable_dep_hr_spraying'], sen.sav['variable_dep_hr_spraying'], 5)  #value for variable rate of machinery depreciation per year if cropping 1000ha
    mach_general['i_variable_dep_hr_spreading'] = fun.f_sa(mach_general['i_variable_dep_hr_spreading'], sen.sav['variable_dep_hr_spreading'], 5)  #value for variable rate of machinery depreciation per year if cropping 1000ha

    ##Mach
    for option in mach: #all pasture inputs are adjusted even if a given pasture is not included
        ###sav
        mach[option]['clearing_value'].loc[:,'value'] = fun.f_sa(np.array(mach[option]['clearing_value'].loc[:,'value']), sen.sav[('clearing_value',option)], 5) #use np so f_update does the dtype correctly
        mach[option]['number_of_seeders'] = fun.f_sa(mach[option]['number_of_seeders'], sen.sav[('number_seeders',option)], 5)
        mach[option]['seeding_rate_base'] = fun.f_sa(mach[option]['seeding_rate_base'], sen.sav[('seeding_rate_base',option)], 5)
        mach[option]['fuel_seeding'] = fun.f_sa(mach[option]['fuel_seeding'], sen.sav[('seeding_fuel_use',option)], 5)
        mach[option]['tillage_maint'] = fun.f_sa(mach[option]['tillage_maint'], sen.sav[('seeding_rm',option)], 5)
        mach[option]['number_of_harvesters'] = fun.f_sa(mach[option]['number_of_harvesters'], sen.sav[('number_harvesters',option)], 5)
        mach[option]['harvest_rate'].iloc[:,0] = fun.f_sa(np.array(mach[option]['harvest_rate'].iloc[:,0]), sen.sav[('harvest_rate',option)], 5) #use np so f_update does the dtype correctly
        mach[option]['harv_fuel_consumption'] = fun.f_sa(mach[option]['harv_fuel_consumption'], sen.sav[('harvester_fuel_use',option)], 5)
        mach[option]['harvest_maint'] = fun.f_sa(mach[option]['harvest_maint'], sen.sav[('harvester_rm',option)], 5)
        mach[option]['truck_chaser_rm'] = fun.f_sa(mach[option]['truck_chaser_rm'], sen.sav[('truck_n_chaser_rm',option)], 5)
        mach[option]['spraying_rate'] = fun.f_sa(mach[option]['spraying_rate'], sen.sav[('spraying_rate',option)], 5)
        mach[option]['sprayer_fuel_consumption'] = fun.f_sa(mach[option]['sprayer_fuel_consumption'], sen.sav[('spraying_fuel',option)], 5)
        mach[option]['sprayer_maint'] = fun.f_sa(mach[option]['sprayer_maint'], sen.sav[('spraying_rm',option)], 5)
        mach[option]['spreader_cap'] = fun.f_sa(mach[option]['spreader_cap'], sen.sav[('spreader_cap',option)], 5)
        mach[option]['spreader_width'].iloc[:,0] = fun.f_sa(np.array(mach[option]['spreader_width'].iloc[:,0]), sen.sav[('spreader_width',option)], 5) #use np so f_update does the dtype correctly
        mach[option]['spreader_speed'] = fun.f_sa(mach[option]['spreader_speed'], sen.sav[('spreading_speed',option)], 5)
        mach[option]['spreader_eff'] = fun.f_sa(mach[option]['spreader_eff'], sen.sav[('spreading_eff',option)], 5)
        mach[option]['spreader_fuel'] = fun.f_sa(mach[option]['spreader_fuel'], sen.sav[('spreading_fuel',option)], 5)
        mach[option]['spreader_maint'] = fun.f_sa(mach[option]['spreader_maint'], sen.sav[('spreading_rm',option)], 5)
        ###sam
        ###sap
        ###saa
        ###sat
        ###sar

    ##supfeed
    ###sav
    supfeed['i_max_sup_selectivity'] = fun.f_sa(supfeed['i_max_sup_selectivity'], sen.sav['max_sup_selectivity'], 5)
    supfeed['i_sup_selectivity_included'] = fun.f_sa(supfeed['i_sup_selectivity_included'], sen.sav['inc_sup_selectivity'], 5)
    supfeed['i_confinement_feeding_cost_factor'] = fun.f_sa(supfeed['i_confinement_feeding_cost_factor'], sen.sav['confinement_feeding_cost_factor'], 5)
    supfeed['i_confinement_feeding_labour_factor'] = fun.f_sa(supfeed['i_confinement_feeding_labour_factor'], sen.sav['confinement_feeding_labour_factor'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##trees
    ###sav
    tree["controls"]['plantation_structure'] = fun.f_sa(tree["controls"]['plantation_structure'], sen.sav['plantation_structure'], 5)
    tree["controls"]['include_livestock_shelter'] = fun.f_sa(tree["controls"]['include_livestock_shelter'], sen.sav['include_livestock_shelter'], 5)
    tree["controls"]['include_adjacent_pad_interaction'] = fun.f_sa(tree["controls"]['include_adjacent_pad_interaction'], sen.sav['include_adjacent_pad_interaction'], 5)
    tree["controls"]['include_carbon_credit'] = fun.f_sa(tree["controls"]['include_carbon_credit'], sen.sav['include_carbon_credit'], 5)
    tree["controls"]['include_biodiversity_credit'] = fun.f_sa(tree["controls"]['include_biodiversity_credit'], sen.sav['include_biodiversity_credit'], 5)
    tree["controls"]['include_harvesting'] = fun.f_sa(tree["controls"]['include_harvesting'], sen.sav['include_harvesting'], 5)
    ###sam
    tree['carbon_price'] = fun.f_sa(tree['carbon_price'], sen.sam['price_carbon_credit'])
    tree['biomass_price'] = fun.f_sa(tree['biomass_price'], sen.sam['price_tree_biomass'])
    tree[f"plantation_structure_{tree['controls']['plantation_structure']}"]['biodiversity_value'] = fun.f_sa(tree[f"plantation_structure_{tree['controls']['plantation_structure']}"]['biodiversity_value'], sen.sam['price_biodiversity_credit'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##sheep
    ###SAV
    sheep['i_eqn_compare'] = fun.f_sa(sheep['i_eqn_compare'], sen.sav['eqn_compare'], 5)  #determines if both equation systems are being run and compared
    sheep['i_eqn_used_g0_q1p7'] = fun.f_sa(sheep['i_eqn_used_g0_q1p7'], sen.sav['eqn_used_g0_q1p7'], 5)  #determines which equation systems are being run
    sheep['i_eqn_used_g1_q1p7'] = fun.f_sa(sheep['i_eqn_used_g1_q1p7'], sen.sav['eqn_used_g1_q1p7'], 5)  #determines which equation systems are being run
    sheep['i_eqn_used_g2_q1p7'] = fun.f_sa(sheep['i_eqn_used_g2_q1p7'], sen.sav['eqn_used_g2_q1p7'], 5)  #determines which equation systems are being run
    sheep['i_eqn_used_g3_q1p7'] = fun.f_sa(sheep['i_eqn_used_g3_q1p7'], sen.sav['eqn_used_g3_q1p7'], 5)  #determines which equation systems are being run
    sheep['i_woolp_mpg_percentile'] = fun.f_sa(sheep['i_woolp_mpg_percentile'], sen.sav['woolp_mpg_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_woolp_fdprem_percentile'] = fun.f_sa(sheep['i_woolp_fdprem_percentile'], sen.sav['woolp_fdprem_percentile'], 5) #replaces the std percentile input with the sa value
    sheep['i_salep_price_max_s7'] = fun.f_sa(sheep['i_salep_price_max_s7'], sen.sav['salep_max_s7'], 5)
    sheep['i_salep_percentile'] = fun.f_sa(sheep['i_salep_percentile'], sen.sav['salep_percentile'], 5) #Value for percentile for all sale grids
    sheep['i_salep_months_priceadj_s7s9p4'] = fun.f_sa(sheep['i_salep_months_priceadj_s7s9p4'], sen.sav['salep_month_adjust_s7s9p4'], 5)  #Value for the premiums and discounts by month
    sheep['i_salep_price_type_s7'] = fun.f_sa(sheep['i_salep_price_type_s7'], sen.sav['salep_price_type_s7'], 5)
    sheep['ia_s8_s7'] = fun.f_sa(sheep['ia_s8_s7'], sen.sav['salep_score_type_s7'], 5)
    sheep['i_sale_ffcfw_min'] = fun.f_sa(sheep['i_sale_ffcfw_min'], sen.sav['sale_ffcfw_min'], 5) #Value for min ffcfw for each grid
    sheep['i_sale_ffcfw_max'] = fun.f_sa(sheep['i_sale_ffcfw_max'], sen.sav['sale_ffcfw_max'], 5) #Value for max ffcfw for each grid
    sheep['i_mobsize_scalar_l0'] = fun.f_sa(sheep['i_mobsize_scalar_l0'], sen.sav['mobsize_scalar_l0'], 5) #Relative mobsize at lambing across the b1 axis
    ###SAM
    sheep['i_husb_operations_contract_cost_h2'] = fun.f_sa(sheep['i_husb_operations_contract_cost_h2'],sen.sam['husb_cost_h2'])
    sheep['i_husb_operations_muster_propn_h2'] = fun.f_sa(sheep['i_husb_operations_muster_propn_h2'], sen.sam['husb_mustering_h2'])
    sheep['i_husb_operations_labourreq_l2h2'] = fun.f_sa(sheep['i_husb_operations_labourreq_l2h2'],sen.sam['husb_labour_l2h2'])
    sheep['i_salep_price_max_s7'] = fun.f_sa(sheep['i_salep_price_max_s7'], sen.sam['salep_max_s7'])
    sheep['i_salep_months_priceadj_s7s9p4'] = fun.f_sa(sheep['i_salep_months_priceadj_s7s9p4'],sen.sam['salep_month_adjust_s7s9p4'])
    ###SAP
    ###SAA
    sheep['i_husb_operations_contract_cost_h2'] = fun.f_sa(sheep['i_husb_operations_contract_cost_h2'],sen.saa['husb_cost_h2'], 2)
    sheep['i_husb_operations_labourreq_l2h2'] = fun.f_sa(sheep['i_husb_operations_labourreq_l2h2'],sen.saa['husb_labour_l2h2'], 2)
    ###SAT
    sheep['i_salep_weight_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_weight_scalar_s7s5s6'], sen.sat['salep_weight_scalar'], 3, 1, 0) #Scalar for LW impact across grid 1 (sat adjusted)
    sheep['i_salep_score_scalar_s7s5s6'] = fun.f_sa(sheep['i_salep_score_scalar_s7s5s6'], sen.sat['salep_score_scalar'], 3, 1, 0) #Scalar for score impact across the grid (sat adjusted)
    ###SAR

    ##parameters (c2 genotype sensitivity)
    ###SAV - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_srw_c2'] = fun.f_sa(parameters['i_srw_c2'].astype(float), sen.sav['srw_c2'], 5) #genotype srw
    parameters['i_sfw_c2'] = fun.f_sa(parameters['i_sfw_c2'].astype(float), sen.sav['sfw_c2'], 5) #genotype sfw
    parameters['i_sfd_c2'] = fun.f_sa(parameters['i_sfd_c2'].astype(float), sen.sav['sfd_c2'], 5) #genotype sfd
    parameters['i_cn_c2'] = fun.f_sa(parameters['i_cn_c2'].astype(float), sen.sav['cn_c1c2'], 5) #normal growth parameters
    parameters['i_ci_c2'] = fun.f_sa(parameters['i_ci_c2'].astype(float), sen.sav['ci_c1c2'], 5) #potential intake parameters
    parameters['i_cl_c2'] = fun.f_sa(parameters['i_cl_c2'].astype(float), sen.sav['cl_c1c2'], 5) #lactation parameters
    parameters['i_cp_c2'] = fun.f_sa(parameters['i_cp_c2'].astype(float), sen.sav['cp_c1c2'], 5) #pregnancy parameters
    parameters['i_cw_c2'] = fun.f_sa(parameters['i_cw_c2'].astype(float), sen.sav['cw_c1c2'], 5) #wool growth parameters
    parameters['i_cg_c2'] = fun.f_sa(parameters['i_cg_c2'].astype(float), sen.sav['cg_c1c2'], 5) #growth parameters
    parameters['i_cd_c2'] = fun.f_sa(parameters['i_cd_c2'].astype(float), sen.sav['cd_c1c2'], 5) #mortaltiuy parameters
    parameters['i_ck_c2'] = fun.f_sa(parameters['i_ck_c2'].astype(float), sen.sav['ck_c1c2'], 5) #energy efficiency parameters
    parameters['i_cl0_c2'] = fun.f_sa(parameters['i_cl0_c2'].astype(float), sen.sav['cl0_c1c2'], 5) #genotype litter size params
    parameters['i_cu2_c2'] = fun.f_sa(parameters['i_cu2_c2'].astype(float), sen.sav['cu2_c1c2'], 5) #Murdoch Uni parameters
    parameters['i_cu6_c2'] = fun.f_sa(parameters['i_cu6_c2'].astype(float), sen.sav['cu6_c1c2'], 5) #Murdoch Uni parameters
    parameters['i_ce_c2'][0,...] = fun.f_sa(parameters['i_ce_c2'][0,...].astype(float), sen.sav['bnd_twice_dry_propn'], 5) #propn of twice drys
    ###SAM - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_ci_c2'] = fun.f_sa(parameters['i_ci_c2'].astype(float),sen.sam['ci_c1c2'])
    parameters['i_cl_c2'] = fun.f_sa(parameters['i_cl_c2'].astype(float),sen.sam['cl_c1c2'])    #lactation parameters
    parameters['i_cm_c2'] = fun.f_sa(parameters['i_cm_c2'].astype(float),sen.sam['cm_c1c2'])
    parameters['i_sfw_c2'] = fun.f_sa(parameters['i_sfw_c2'].astype(float),sen.sam['sfw_c2'])
    parameters['i_muscle_target_c2'] = fun.f_sa(parameters['i_muscle_target_c2'].astype(float), sen.sam['muscle_target_c2'])
    ###SAP
    ###SAA - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_nlb_c2'] = fun.f_sa(parameters['i_nlb_c2'].astype(float),sen.saa['nlb_c2'], 2)
    parameters['i_sfd_c2'] = fun.f_sa(parameters['i_sfd_c2'].astype(float),sen.saa['sfd_c2'], 2)
    parameters['i_srw_c2'] = fun.f_sa(parameters['i_srw_c2'].astype(float),sen.saa['srw_c2'], 2)
    parameters['i_cg_c2'] = fun.f_sa(parameters['i_cg_c2'].astype(float), sen.saa['cg_c1c2'], 2) #genotype growth params
    parameters['i_ck_c2'] = fun.f_sa(parameters['i_ck_c2'].astype(float), sen.saa['ck_c1c2'], 2) #genotype efficiency params
    parameters['i_cl0_c2'] = fun.f_sa(parameters['i_cl0_c2'].astype(float), sen.saa['cl0_c1c2'], 2) #genotype litter size params
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float), sen.saa['scan_std_c2'], 2) #genotype scanning percent params
    ###SAT
    parameters['i_cb0_c2'] = fun.f_sa(parameters['i_cb0_c2'].astype(float), sen.sat['cb0_c2'], 3, 1) #genotype BTRT params (sat -ve values allowed)

    ##parameters (overall sensitivity - carried out after the c2 genotype sa)
    ###SAM - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float), sen.sam['rr']) #genotype scanning percent params
    ###SAA - these have to be converted to float so that the blank column becomes nan rather that None
    parameters['i_scan_std_c2'] = fun.f_sa(parameters['i_scan_std_c2'].astype(float)
                                           , sen.saa['rr'] * (parameters['i_scan_std_c2'] > 0), 2)  #no change if original value was zero
    parameters['i_lss_std_c2'] = fun.f_sa(parameters['i_lss_std_c2'].astype(float), sen.saa['lss'], 2)
    parameters['i_lstw_std_c2'] = fun.f_sa(parameters['i_lstw_std_c2'].astype(float), sen.saa['lstw'], 2)
    parameters['i_lstr_std_c2'] = fun.f_sa(parameters['i_lstr_std_c2'].astype(float), sen.saa['lstr'], 2)

    ##average c1 axis if price variation is not included
    general['i_c1_variation_included'] = fun.f_sa(general['i_c1_variation_included'], sen.sav['inc_c1_variation'], 5)
    if not general['i_c1_variation_included']:
        price_variation['grain_price_scalar_c1z'] = pd.DataFrame(price_variation['grain_price_scalar_c1z'].mul(price_variation['prob_c1'], axis=0).sum(axis=0),
                                                                 columns=price_variation['grain_price_scalar_c1z'].index[0:1]).T
        price_variation['meat_price_scalar_c1z'] = np.sum(price_variation['meat_price_scalar_c1z'] * price_variation['prob_c1'].values[:,None], axis=0, keepdims=True)
        price_variation['wool_price_scalar_c1z'] = np.sum(price_variation['wool_price_scalar_c1z'] * price_variation['prob_c1'].values[:,None], axis=0, keepdims=True)
        price_variation['prob_c1'] = pd.Series(price_variation['prob_c1'].sum(), index=price_variation['prob_c1'].index[0:1])
        price_variation['len_c1'] = len(price_variation['prob_c1'])


def f1_mask_landuse():
    ##price (note grain_price with k4 axis gets masked later)
    pinp.f1_do_mask_landuse(price, "contract_harv_cost", landuse_axis_type="crop", landuse_axis=0)

    ##sup feed
    supp_mask_k3 = supfeed['i_supp_inc_k3'].squeeze().values
    supfeed["sup_md_vol"] = supfeed["sup_md_vol"].loc[:, supp_mask_k3]
    supfeed["grain_density"] = supfeed["grain_density"].loc[:, supp_mask_k3]
    supfeed["i_sup_s2_ks2"] = supfeed["i_sup_s2_ks2"].loc[supp_mask_k3,:]

    ##stub
    pinp.f1_do_mask_landuse(stubble, "i_growth_scalar_k", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_propn_baled_k", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_propn_conserved_k", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_biomass_scalar_ks2", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_harvest_index_ks2", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_propn_grain_harv_ks2", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "quantity_decay", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "quality_deterioration", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "trampling", landuse_axis_type="crop", landuse_axis=0)
    pinp.f1_do_mask_landuse(stubble, "i_t_inc_s2kt", landuse_axis_type="crop", landuse_axis=1)

    ##emissions
    pinp.f1_do_mask_landuse(emissions, "i_Rbg", landuse_axis_type="crop", landuse_axis=0)

    ##mach general
    pinp.f1_do_mask_landuse(mach_general, "contract_harvest_rate", landuse_axis_type="crop", landuse_axis=0)

    for option in mach: #all mach inputs are adjusted even if not used
        ###sav
        pinp.f1_do_mask_landuse(mach[option], "seeding_rate_crop_adj", landuse_axis_type="all", landuse_axis=0)
        pinp.f1_do_mask_landuse(mach[option], "harvest_rate", landuse_axis_type="crop", landuse_axis=0)
        pinp.f1_do_mask_landuse(mach[option], "harvest_maint_scalar", landuse_axis_type="crop", landuse_axis=0)
        mach[option]["sup_feed"] = mach[option]["sup_feed"].loc[supp_mask_k3, :]
