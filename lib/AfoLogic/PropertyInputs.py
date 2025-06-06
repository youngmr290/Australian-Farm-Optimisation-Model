"""

Inputs specific to a property (or region) including:

* Crop production
* Pasture production
* Labour
* Supplementary feeding
* Stubble
* Farm level finance ie capital level
* Farm level machinery ie soil adjustment factors for seeding efficiency

author: young
"""
##python modules
import pickle as pkl
import os.path
import numpy as np
import pandas as pd
import copy
import sys

##AFO modules
from . import Functions as fun
from . import StructuralInputs as sinp
from . import relativeFile
try:
    from Inputs import TreePropertyInputs as tinp  # Local case
except ImportError:
    from module.afo.Inputs import TreePropertyInputs as tinp  # Web app case

na = np.newaxis

def f_reshape_pinp_defaults(pinp_defaults, sinp_defaults):
    ##occurs once when inputs are read
    for property in pinp_defaults:
        ###lengths
        len_d = len(pinp_defaults[property]['sheep_inp']['i_d_idx'])
        len_h2 = pinp_defaults[property]['sheep_inp']['i_h2_len']
        len_h5 = pinp_defaults[property]['sheep_inp']['i_h5_len']
        len_h7 = pinp_defaults[property]['sheep_inp']['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
        len_i = pinp_defaults[property]['sheep_inp']['i_i_len']
        len_j2 = pinp_defaults[property]['feedsupply_inp']['i_j2_len']
        len_k = len(sinp_defaults["general_inp"]['i_idx_k1'])
        len_k0 = pinp_defaults[property]['sheep_inp']['i_k0_len']
        len_k1 = pinp_defaults[property]['sheep_inp']['i_k1_len']
        len_k2 = pinp_defaults[property]['sheep_inp']['i_k2_len']
        len_k3 = pinp_defaults[property]['sheep_inp']['i_k3_len']
        len_k4 = pinp_defaults[property]['sheep_inp']['i_k4_len']
        len_k5 = pinp_defaults[property]['sheep_inp']['i_k5_len']
        len_l = len(pinp_defaults[property]['general_inp']['i_lmu_area'])
        len_o = pinp_defaults[property]['sheep_inp']['i_o_len']
        len_p6 = len(pinp_defaults[property]['period_inp']['i_fp_idx'])
        len_r1 = pinp_defaults[property]['feedsupply_inp']['i_r1_len']
        len_s = pinp_defaults[property]['sheep_inp']['i_s_len'] #s = shear
        len_sc = sinp_defaults["stock_inp"]['i_len_s'] #sc = scan
        len_t3 = pinp_defaults[property]['sheep_inp']['i_t3_len']
        len_x = pinp_defaults[property]['sheep_inp']['i_x_len']
        len_z = len(pinp_defaults[property]['general_inp']['i_mask_z'])


        ###shapes
        zp6 = (len_z, len_p6)
        zkp6 = (len_z, len_k, len_p6)
        zp6l = (len_z, len_p6, len_l)
        zp6j0 = (len_z, len_p6, -1)
        zp6i = (len_z, len_p6, len_i)
        h2h5h7 = (len_h2, len_h5, len_h7)
        iog = (len_i, len_o, -1)
        idg = (len_i, len_d, -1)
        isxg = (len_i, len_s, len_x, -1)
        izg = (len_i, len_z, -1)
        ik0g = (len_i, len_k0, -1)
        ik1g = (len_i, len_k1, -1)
        isk2g = (len_i, len_sc, len_k2, -1)
        ik3g = (len_i, len_k3, -1)
        ik4g = (len_i, len_k4, -1)
        ik5g = (len_i, len_k5, -1)
        t3Sg = (len_t3, len_s+1, -1) #capital S to indicate this is special e.g. not normal because +1
        r1j2P = (len_r1, len_j2, -1) #capital p to indicate this is just the remaining length of input, it is p axis but input is longer
        r1p6z = (len_r1, len_p6, len_z)

        ##crop
        crop_inp = pinp_defaults[property]['crop_inp']
        crop_inp['fert'] = crop_inp['fert'].T.set_index(['fert'], append=True).T.astype(float) #set headers
        crop_inp['fert_passes'] = crop_inp['fert_passes'].T.set_index(['passes'], append=True).T.astype(float) #set headers
        crop_inp['chem_cost'] = crop_inp['chem_cost'].T.set_index(['chem'], append=True).T.astype(float) #set headers
        crop_inp['chem'] = crop_inp['chem'].T.set_index(['chem'], append=True).T.astype(float) #set headers

        ###pasture
        for t,pasture in enumerate(sinp_defaults["general_inp"]['pastures'][pinp_defaults[property]['general_inp']['i_pastures_exist']]):
            inp = pinp_defaults[property]['pasture_inp'][pasture]
            inp['DigRednSenesce'] = np.reshape(inp['DigRednSenesce'], zp6)
            inp['DigDryAve'] = np.reshape(inp['DigDryAve'], zp6)
            inp['DigDryRange'] = np.reshape(inp['DigDryRange'], zp6)
            inp['FOODryH'] = np.reshape(inp['FOODryH'], zp6)
            inp['DigSpread'] = np.reshape(inp['DigSpread'], zp6)
            inp['ErosionLimit'] = np.reshape(inp['ErosionLimit'], zp6l)
            inp['LowFOO'] = np.reshape(inp['LowFOO'], zp6l)
            inp['MedFOO'] = np.reshape(inp['MedFOO'], zp6l)
            inp['LowPGR'] = np.reshape(inp['LowPGR'], zp6l)
            inp['MedPGR'] = np.reshape(inp['MedPGR'], zp6l)
            inp['DigGrn'] = np.reshape(inp['DigGrn'], zp6l)

        ###saltbush
        saltbush_inp = pinp_defaults[property]['saltbush_inp']
        saltbush_inp['i_sb_expected_foo_zp6'] = np.reshape(saltbush_inp['i_sb_expected_foo_zp6'], zp6)
        saltbush_inp['i_sb_expected_growth_zp6'] = np.reshape(saltbush_inp['i_sb_expected_growth_zp6'], zp6)
        saltbush_inp['i_sb_growth_reduction_zp6'] = np.reshape(saltbush_inp['i_sb_growth_reduction_zp6'], zp6)
        saltbush_inp['i_sb_ash_content_zp6'] = np.reshape(saltbush_inp['i_sb_ash_content_zp6'], zp6)
        saltbush_inp['i_sb_selectivity_zp6'] = np.reshape(saltbush_inp['i_sb_selectivity_zp6'], zp6)
        saltbush_inp['i_slp_diet_propn_zp6'] = np.reshape(saltbush_inp['i_slp_diet_propn_zp6'], zp6)
        saltbush_inp['i_sb_cp_zp6'] = np.reshape(saltbush_inp['i_sb_cp_zp6'], zp6)

        ###stock
        sheep_inp = pinp_defaults[property]['sheep_inp']
        sheep_inp['i_pasture_stage_p6z'] = np.reshape(sheep_inp['i_pasture_stage_p6z'], zp6)
        sheep_inp['i_legume_p6z'] = np.reshape(sheep_inp['i_legume_p6z'], zp6)
        sheep_inp['i_supplement_zp6'] = np.reshape(sheep_inp['i_supplement_zp6'], zp6)
        sheep_inp['i_paststd_foo_zp6j0'] = np.reshape(sheep_inp['i_paststd_foo_zp6j0'], zp6j0)
        sheep_inp['i_paststd_dmd_zp6j0'] = np.reshape(sheep_inp['i_paststd_dmd_zp6j0'], zp6j0)
        sheep_inp['i_mobsize_sire_zp6i'] = np.reshape(sheep_inp['i_mobsize_sire_zp6i'], zp6i)
        sheep_inp['i_mobsize_dams_zp6i'] = np.reshape(sheep_inp['i_mobsize_dams_zp6i'], zp6i)
        sheep_inp['i_mobsize_offs_zp6i'] = np.reshape(sheep_inp['i_mobsize_offs_zp6i'], zp6i)
        sheep_inp['i_density_p6z'] = np.reshape(sheep_inp['i_density_p6z'], zp6)
        sheep_inp['i_husb_operations_triggerlevels_h5h7h2'] = np.reshape(sheep_inp['i_husb_operations_triggerlevels_h5h7h2'], h2h5h7)
        sheep_inp['i_date_born1st_iog2'] = np.reshape(sheep_inp['i_date_born1st_iog2'], iog)
        sheep_inp['i_date_born1st_idg3'] = np.reshape(sheep_inp['i_date_born1st_idg3'], idg)
        sheep_inp['i_sire_propn_oig1'] = np.reshape(sheep_inp['i_sire_propn_oig1'], iog)
        sheep_inp['i_date_shear_sixg0'] = np.reshape(sheep_inp['i_date_shear_sixg0'], isxg)
        sheep_inp['i_date_shear_sixg1'] = np.reshape(sheep_inp['i_date_shear_sixg1'], isxg)
        sheep_inp['i_date_shear_sixg3'] = np.reshape(sheep_inp['i_date_shear_sixg3'], isxg)
        sheep_inp['ia_r1_zig0'] = np.reshape(sheep_inp['ia_r1_zig0'], izg)
        sheep_inp['ia_r1_zig1'] = np.reshape(sheep_inp['ia_r1_zig1'], izg)
        sheep_inp['ia_r1_zig3'] = np.reshape(sheep_inp['ia_r1_zig3'], izg)
        sheep_inp['ia_r2_ik0g1'] = np.reshape(sheep_inp['ia_r2_ik0g1'], ik0g)
        sheep_inp['ia_r2_ik1g1'] = np.reshape(sheep_inp['ia_r2_ik1g1'], ik1g)
        sheep_inp['ia_r2_isk2g1'] = np.reshape(sheep_inp['ia_r2_isk2g1'], isk2g)
        sheep_inp['ia_r2_ik0g3'] = np.reshape(sheep_inp['ia_r2_ik0g3'], ik0g)
        sheep_inp['ia_r2_ik3g3'] = np.reshape(sheep_inp['ia_r2_ik3g3'], ik3g)
        sheep_inp['ia_r2_ik4g3'] = np.reshape(sheep_inp['ia_r2_ik4g3'], ik4g)
        sheep_inp['ia_r2_ik5g3'] = np.reshape(sheep_inp['ia_r2_ik5g3'], ik5g)
        sheep_inp['i_sales_age_tsg3'] = np.reshape(sheep_inp['i_sales_age_tsg3'], t3Sg)
        sheep_inp['i_target_weight_tsg3'] = np.reshape(sheep_inp['i_target_weight_tsg3'], t3Sg)
        sheep_inp['ia_i_idg2'] = np.reshape(sheep_inp['ia_i_idg2'], idg)

        ###feedsupply
        feedsupply_inp = pinp_defaults[property]['feedsupply_inp']
        feedsupply_inp['i_feedsupply_options_r1j2p'] = np.reshape(feedsupply_inp['i_feedsupply_options_r1j2p'], r1j2P)
        feedsupply_inp['i_confinement_options_r1p6z'] = np.reshape(feedsupply_inp['i_confinement_options_r1p6z'], r1p6z)


#######################
#reset defaults       #
#######################
def f_select_n_reset_pinp(property, pinp_defaults):
    ##occurs for each trial
    ##create a copy of each input dict - so that the base inputs remain unchanged
    ##the copy created is the one used in the actual modules

    print('Using property: {0}'.format(property))

    ##needs to be global so inputs can be accessed outside this function
    global general
    global labour
    global crop
    global emissions
    global cropgraze
    global saltbush
    global mach
    global stubble
    global finance
    global period
    global supfeed
    global sheep
    global feedsupply
    global mvf
    global pasture_inputs
    global tree
    general = copy.deepcopy(pinp_defaults[property]['general_inp'])
    labour = copy.deepcopy(pinp_defaults[property]['labour_inp'])
    crop = copy.deepcopy(pinp_defaults[property]['crop_inp'])
    emissions = copy.deepcopy(pinp_defaults[property]['emissions_inp'])
    cropgraze = copy.deepcopy(pinp_defaults[property]['cropgraze_inp'])
    saltbush = copy.deepcopy(pinp_defaults[property]['saltbush_inp'])
    mach = copy.deepcopy(pinp_defaults[property]['mach_inp'])
    stubble = copy.deepcopy(pinp_defaults[property]['stubble_inp'])
    finance = copy.deepcopy(pinp_defaults[property]['finance_inp'])
    period = copy.deepcopy(pinp_defaults[property]['period_inp'])
    supfeed = copy.deepcopy(pinp_defaults[property]['sup_inp'])
    sheep = copy.deepcopy(pinp_defaults[property]['sheep_inp'])
    feedsupply = copy.deepcopy(pinp_defaults[property]['feedsupply_inp'])
    mvf = copy.deepcopy(pinp_defaults[property]['mvf_inp'])
    pasture_inputs = copy.deepcopy(pinp_defaults[property]['pasture_inp'])
    tree = copy.deepcopy(tinp.region_tree_inputs[property])


########################
#adjust lmu for web app#
########################
def f_farmer_lmu_adj(a_lmuregion_lmufarmer):
    '''
    After collecting data from a couple of farmers I have found that, for the web app, we need to make the LMU/soil type structure
    more flexible so farmers can make their own LMUS.

    To achieve this, the user creates their own LMUs and selects which regional LMU most closely aligns. This is done
    so that the default values for their custom LMU are as realistic as possible. The user can then go and adjust the
    inputs in the web app that have an LMU axis to allign the production correctly.

    However, because the web app doesnt have every input is an LMU axis this function exists to create the users lmu
    (for inputs that don't exist in the app the user lmu data will be equal to the nearest regional lmu, as specified by
    the user).

    Has to occur before SA are applied because the SAs from web app have already been adjusted.
    '''
    ##create a global dict with a flag for each input with LMU axis that is adjusted. This allows us to perform a check
    ##that each input has been included here. The check occurs when the input is masked for lmu axis.
    global lmu_flag
    lmu_flag={}

    ##general
    lmu_flag['i_lmu_idx']=True #manually add this because we don't want to adjust this input
    fun.f1_lmuregion_to_lmufarmer(general, "i_lmu_area", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(general, "i_non_cropable_area_l", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(general, "arable", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)

    ##crop
    fun.f1_lmuregion_to_lmufarmer(crop, "yield_by_lmu", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(crop, "i_soil_production_z_l", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(crop, "seeding_penalty_lmu_scalar_lp", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(crop, "fert_by_lmu", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(crop, "chem_by_lmu", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(crop, "seeding_rate", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)

    ##cropgraze
    fun.f1_lmuregion_to_lmufarmer(cropgraze, "i_cropgrowth_lmu_factor_kl", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(cropgraze, "i_cropgraze_propn_area_grazed_kl", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)

    ##saltbush
    fun.f1_lmuregion_to_lmufarmer(saltbush, "i_sb_lmu_scalar", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)

    ##pasture
    for pasture in sinp.general['pastures'][general['i_pastures_exist']]:
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "i_pasture_coverage", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "GermScalarLMU", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "FaG_LMU", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "LowFOO", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "LowPGR", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "MedFOO", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "MedPGR", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "i_soil_production_zl", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "DigGrn", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "POCCons", a_lmuregion_lmufarmer, lmu_axis=1, lmu_flag=lmu_flag)
        fun.f1_lmuregion_to_lmufarmer(pasture_inputs[pasture], "ErosionLimit", a_lmuregion_lmufarmer, lmu_axis=2, lmu_flag=lmu_flag)

    ##machine
    fun.f1_lmuregion_to_lmufarmer(mach, "seeding_fuel_lmu_adj", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(mach, "tillage_maint_lmu_adj", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(mach, "seeding_rate_lmu_adj", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    
    ##tree
    fun.f1_lmuregion_to_lmufarmer(tree, "tree_fert_soil_scalar", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(tree, "estimated_area_trees_l", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(tree, "lmu_growth_scalar_l", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)
    fun.f1_lmuregion_to_lmufarmer(tree, "lmu_carbon_scalar_l", a_lmuregion_lmufarmer, lmu_axis=0, lmu_flag=lmu_flag)


#######################
#apply SA             #
#######################
def f_property_inp_sa(pinp_defaults):
    '''

    Applies sensitivity adjustment to each input. After the sensitivities are applied, when using the DSP model, inputs
    with a feed period index are expanded to account for additional feed periods that are added due to season nodes.
    This function gets called at the beginning of each loop in the RunAfoRaw_Multi.py module

    SA order is: sav, sam, sap, saa, sat, sar.
    So that multiple SA can be applied to one input.

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    from . import Sensitivity as sen

    ##special sav to change node dates for MP model
    if np.any(sen.sav['date_node_p7']!='-'):
        ###slice p7 so that inputs match new p7 definition
        len_p7 = len(sen.sav['date_node_p7'])
        general['i_date_node_zm'] = general['i_date_node_zm'][:,0:len_p7] #slice p7
        general['i_phase_can_increase_kp7'] = general['i_phase_can_increase_kp7'][:,0:len_p7]
        general['i_phase_can_reduce_kp7'] = general['i_phase_can_reduce_kp7'][:,0:len_p7]
        general['i_node_is_fvp'] = general['i_node_is_fvp'][0:len_p7]
        ###update node dates
        general['i_date_node_zm'][...] = sen.sav['date_node_p7']

    ##general
    ###sav
    general['i_mask_z'] = fun.f_sa(general['i_mask_z'], sen.sav['mask_z'], 5)
    general['i_season_propn_z'] = fun.f_sa(general['i_season_propn_z'], sen.sav['prob_z'], 5)
    general['i_node_is_fvp'] = fun.f_sa(general['i_node_is_fvp'], sen.sav['node_is_fvp'][0:len(general['i_node_is_fvp'])], 5)
    general['i_phase_can_increase_kp7'] = fun.f_sa(general['i_phase_can_increase_kp7'], sen.sav['phase_can_increase_kp7'][:,0:len(general['i_node_is_fvp'])], 5)
    general['i_phase_can_reduce_kp7'] = fun.f_sa(general['i_phase_can_reduce_kp7'], sen.sav['phase_can_reduce_kp7'][:,0:len(general['i_node_is_fvp'])], 5)
    labour['max_managers'] = fun.f_sa(labour['max_managers'], sen.sav['manager_ub'], 5)
    labour['min_managers'] = fun.f_sa(labour['min_managers'], sen.sav['manager_lo'], 5)
    labour['max_perm'] = fun.f_sa(labour['max_perm'], sen.sav['perm_ub'], 5)
    labour['min_perm'] = fun.f_sa(labour['min_perm'], sen.sav['perm_lo'], 5)
    labour['max_casual'] = fun.f_sa(labour['max_casual'], sen.sav['casual_ub'], 5)
    labour['min_casual'] = fun.f_sa(labour['min_casual'], sen.sav['casual_lo'], 5)
    labour['max_casual_seedharv'] = fun.f_sa(labour['max_casual_seedharv'], sen.sav['seedharv_casual_ub'], 5)
    labour['min_casual_seedharv'] = fun.f_sa(labour['min_casual_seedharv'], sen.sav['seedharv_casual_lo'], 5)
    general['i_lmu_area'] = fun.f_sa(general['i_lmu_area'], sen.sav['lmu_area_l'], 5)
    general['i_non_cropable_area_l'] = fun.f_sa(general['i_non_cropable_area_l'], sen.sav['non_cropable_area_l'], 5)
    general['arable'] = fun.f_sa(general['arable'], sen.sav['lmu_arable_propn_l'], 5)
    general['i_crop_landuse_inc_k1'] = fun.f_sa(general['i_crop_landuse_inc_k1'], sen.sav['crop_landuse_inc_k1'], 5)
    general['i_pas_landuse_inc_k2'] = fun.f_sa(general['i_pas_landuse_inc_k2'], sen.sav['pas_landuse_inc_k2'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##finance
    ###sav
    finance['capital_limit'] = fun.f_sa(finance['capital_limit'], sen.sav['capital_limit'], 5)
    general['i_overheads'] = fun.f_sa(general['i_overheads'], sen.sav['overheads'], 5)

    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##crop
    ###sav
    crop['user_crop_rot'] = fun.f_sa(crop['user_crop_rot'], sen.sav['pinp_rot'], 5)
    ####r axis SAVs need special handling because rotations can be added in the web app therefore len_r needs to be dynamic
    if (sen.sav['user_rotphases'] != '-').all():
        ####rotation info gets complete overwrite when coming from web app
        index = [''.join(x) for x in sen.sav['user_rotphases'][1:].astype(str)]
        web_app_rots = pd.DataFrame(sen.sav['user_rotphases'][1:,1:], index=sen.sav['user_rotphases'][1:,0], columns=sen.sav['user_rotphases'][0,1:])
        crop['fixed_rotphases'] = web_app_rots
        crop['i_user_rot_inc_r'] = sen.sav['rot_inc_R'].astype(bool)
        crop['i_seeding_freq_r'] = sen.sav['sowing_freq_R'].astype(float)
        crop['i_nap_fert_scalar_r'] = 1 - crop['i_seeding_freq_r'] #assumption is that non arable area is only fertilised for non-resown phases.
        crop['yields'] = pd.DataFrame(sen.sav['yield_Rz'], index=index, columns=crop['yields'].columns, dtype=float)
        crop['fert'] = pd.DataFrame(sen.sav['fert_R_nz'], index=index, columns=crop['fert'].columns, dtype=float)
        crop['fert_passes'] = pd.DataFrame(sen.sav['fert_passes_R_nz'], index=index, columns=crop['fert_passes'].columns, dtype=float)
        crop['chem_cost'] = pd.DataFrame(sen.sav['chem_R_nz'], index=index, columns=crop['chem_cost'].columns, dtype=float)
        crop['chem'] = pd.DataFrame(sen.sav['chem_passes_R_nz'], index=index, columns=crop['chem'].columns, dtype=float)
    else:
        len_r = len(crop['fixed_rotphases'])
        crop['i_user_rot_inc_r'] = fun.f_sa(crop['i_user_rot_inc_r'][0:len_r], sen.sav['rot_inc_R'][0:len_r], 5)
        crop['i_seeding_freq_r'] = fun.f_sa(crop['i_seeding_freq_r'][0:len_r], sen.sav['sowing_freq_R'][0:len_r], 5)
        crop['yields'] = fun.f_sa(crop['yields'][0:len_r], sen.sav['yield_Rz'][0:len_r,:], 5, pandas=True)
        crop['fert'] = fun.f_sa(crop['fert'][0:len_r], sen.sav['fert_R_nz'][0:len_r,:], 5, pandas=True)
        crop['fert_passes'] = fun.f_sa(crop['fert_passes'][0:len_r], sen.sav['fert_passes_R_nz'][0:len_r,:], 5, pandas=True)
        crop['chem_cost'] = fun.f_sa(crop['chem_cost'][0:len_r], sen.sav['chem_R_nz'][0:len_r], 5, pandas=True)
        crop['chem'] = fun.f_sa(crop['chem'][0:len_r], sen.sav['chem_passes_R_nz'][0:len_r,:],5, pandas=True)
    crop['i_lime'] = fun.f_sa(crop['i_lime'], sen.sav['lime_cost'], 5)
    crop['i_lime_freq'] = fun.f_sa(crop['i_lime_freq'], sen.sav['liming_freq'], 5)
    crop['yield_by_lmu'] = fun.f_sa(crop['yield_by_lmu'], sen.sav['lmu_yield_adj_kl'], 5)
    crop['fert_by_lmu'] = fun.f_sa(crop['fert_by_lmu'], sen.sav['lmu_fert_adj_nl'], 5)
    crop['chem_by_lmu'] = fun.f_sa(crop['chem_by_lmu'], sen.sav['lmu_chem_adj_l'], 5)
    ###sam
    crop['seeding_yield_penalty'] = fun.f_sa(crop['seeding_yield_penalty'], sen.sam['sowing_penalty'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##machinery
    ###sav
    mach['option'] = fun.f_sa(mach['option'], sen.sav['mach_option'], 5)
    mach['daily_seed_hours'] = fun.f_sa(mach['daily_seed_hours'], sen.sav['daily_seed_hours'], 5)
    mach['seeding_downtime_frac'] = fun.f_sa(mach['seeding_downtime_frac'], sen.sav['seeding_downtime_frac'], 5)
    mach['seeding_delays'] = fun.f_sa(mach['seeding_delays'], sen.sav['seeding_delays'], 5)
    mach['daily_harvest_hours'] = fun.f_sa(mach['daily_harvest_hours'], sen.sav['daily_harvest_hours'], 5)
    mach['harv_downtime_frac'] = fun.f_sa(mach['harv_downtime_frac'], sen.sav['harv_downtime_frac'], 5)
    mach['harv_delays'] = fun.f_sa(mach['harv_delays'], sen.sav['harv_delays'], 5)
    mach['spray_downtime_frac'] = fun.f_sa(mach['spray_downtime_frac'], sen.sav['spray_downtime_frac'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##crop grazing
    ###sav
    cropgraze['i_cropgrazing_inc'] = fun.f_sa(cropgraze['i_cropgrazing_inc'], sen.sav['cropgrazing_inc'], 5)
    cropgraze['i_cropgraze_propn_area_grazed_kl'] = fun.f_sa(cropgraze['i_cropgraze_propn_area_grazed_kl'], sen.sav['cropgraze_propn_area_grazed_kl'], 5)
    cropgraze['i_cropgraze_yield_reduction_k'] = fun.f_sa(cropgraze['i_cropgraze_yield_reduction_k'], sen.sav['cropgraze_yield_penalty_k'], 5)
    ###sam
    cropgraze['i_cropgraze_yield_reduction_k'] = fun.f_sa(cropgraze['i_cropgraze_yield_reduction_k'], sen.sam['cropgraze_yield_penalty'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##trees
    ###sav
    tree['estimated_area_trees_l'] = fun.f_sa(tree['estimated_area_trees_l'], sen.sav['estimated_area_trees_l'], 5)
    ###sap
    ###saa
    ###sat
    ###sar

    ##salt land pasture
    ###sav
    saltbush['i_saltbush_estab_cost'] = fun.f_sa(saltbush['i_saltbush_estab_cost'], sen.sav['saltbush_estab_cost'], 5)
    saltbush['i_understory_estab_cost'] = fun.f_sa(saltbush['i_understory_estab_cost'], sen.sav['understory_estab_cost'], 5)
    saltbush['i_saltbush_success'] = fun.f_sa(saltbush['i_saltbush_success'], sen.sav['saltbush_success'], 5)
    saltbush['i_understory_success'] = fun.f_sa(saltbush['i_understory_success'], sen.sav['understory_success'], 5)
    saltbush['i_saltbush_life'] = fun.f_sa(saltbush['i_saltbush_life'], sen.sav['saltbush_life'], 5)
    saltbush['i_understory_life'] = fun.f_sa(saltbush['i_understory_life'], sen.sav['understory_life'], 5)
    saltbush['i_sb_omd'] = fun.f_sa(saltbush['i_sb_omd'], sen.sav['sb_omd'], 5)
    ###sam
    saltbush['i_sb_expected_growth_zp6'] = fun.f_sa(saltbush['i_sb_expected_growth_zp6'], sen.sam['sb_growth'])
    saltbush['i_sb_lmu_scalar'] = fun.f_sa(saltbush['i_sb_lmu_scalar'], sen.sam['sb_growth_l'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##pasture
    ###sav
    crop['i_poc_inc'] = fun.f_sa(crop['i_poc_inc'], sen.sav['poc_inc'], 5)
    general['pas_inc_t'] = fun.f_sa(general['pas_inc_t'], sen.sav['pas_inc_t'], 5)
    for t, pasture in enumerate(sinp.general['pastures']):
        general['pas_inc_t'][t] &= any(x in sinp.landuse['pasture_sets'][pasture] for x in sinp.general['i_idx_k2'][general['i_pas_landuse_inc_k2']]) #exclude pasture type if there is no active landuses.

    for pasture in sinp.general['pastures'][general['pas_inc_t']]:
        ###sav
        ###SAM
        pasture_inputs[pasture]['GermStd'] = fun.f_sa(pasture_inputs[pasture]['GermStd'], sen.sam[('germ',pasture)])
        pasture_inputs[pasture]['GermScalarLMU'] = fun.f_sa(pasture_inputs[pasture]['GermScalarLMU'], sen.sam[('germ_l',pasture)])
        pasture_inputs[pasture]['ErosionLimit'] = fun.f_sa(pasture_inputs[pasture]['ErosionLimit'], sen.sam[('conservation_limit_f',pasture)][...,na])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_zp6',pasture)][...,na])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_zp6',pasture)][...,na])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['DigDryAve'] = (pasture_inputs[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)]
                                                + np.max(pasture_inputs[pasture]['DigDryAve'],axis=1) * (1 - sen.sam[('dry_dmd_decline',pasture)]))
        pasture_inputs[pasture]['DigSpread'] = fun.f_sa(pasture_inputs[pasture]['DigSpread'], sen.sam[('grn_dmd_range_f',pasture)])
#        pasture_inputs[pasture]['DigDeclineFOO'] = fun.f_sa(pasture_inputs[pasture]['DigDeclineFOO'], sen.sam[('grn_dmd_declinefoo_f',pasture)])
        pasture_inputs[pasture]['DigRednSenesce'] = fun.f_sa(pasture_inputs[pasture]['DigRednSenesce'], sen.sam[('grn_dmd_senesce_f',pasture)])

        ###sap
        ###saa
        pasture_inputs[pasture]['GermStd'] = fun.f_sa(pasture_inputs[pasture]['GermStd'], sen.saa[('germ',pasture)], 2)
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.saa[('pgr',pasture)], 2)
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.saa[('pgr_zp6',pasture)][...,na], 2)
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.saa[('pgr_l',pasture)], 2)
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.saa[('pgr',pasture)], 2)
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.saa[('pgr_zp6',pasture)][...,na], 2)
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.saa[('pgr_l',pasture)], 2)
        ###sat
        ###sar

    ##sheep
    ###SAV
    # sheep['i_chill_adj'] = fun.f_sa(sheep['i_chill_adj'], sen.sav['chill_variation'], 5)
    sheep['i_date_shear_sixg0'] = fun.f_sa(sheep['i_date_shear_sixg0'], sen.sav['date_shear_isxg0'], 5)
    sheep['i_date_shear_sixg1'] = fun.f_sa(sheep['i_date_shear_sixg1'], sen.sav['date_shear_isxg1'], 5)
    sheep['i_date_shear_sixg3'] = fun.f_sa(sheep['i_date_shear_sixg3'], sen.sav['date_shear_isxg3'], 5)
    sheep['i_mask_i'] = fun.f_sa(sheep['i_mask_i'], sen.sav['TOL_inc'], 5)
    sheep['i_g3_inc'] = fun.f_sa(sheep['i_g3_inc'], sen.sav['g3_included'],5)
    sheep['a_c2_c0'] = fun.f_sa(sheep['a_c2_c0'], sen.sav['genotype'],5)
    sheep['i_scan_og1'] = fun.f_sa(sheep['i_scan_og1'], sen.sav['scan_og1'],5)
    sheep['i_dry_sales_forced_o'] = fun.f_sa(sheep['i_dry_sales_forced_o'], sen.sav['bnd_drys_sold_o'],5)
    sheep['i_dry_retained_forced_o'] = fun.f_sa(sheep['i_dry_retained_forced_o'], sen.sav['bnd_drys_retained_o'],5)
    #### The expected proportion retained at scanning or birth is a 2-step calc.
    #### 1. Update with own SAV and then 2. override if either of the dry management options is forced
    sheep['i_drys_retained_scan_est_o'] = fun.f_sa(sheep['i_drys_retained_scan_est_o'], sen.sav['est_drys_retained_scan_o'], 5)
    sheep['i_drys_retained_birth_est_o'] = fun.f_sa(sheep['i_drys_retained_birth_est_o'], sen.sav['est_drys_retained_birth_o'], 5)
    ##### If sale of drys is forced then estimated proportion of drys retained is 0 otherwise the proportion remains as per the input.
    sheep['i_drys_retained_scan_est_o'] = fun.f_update(sheep['i_drys_retained_scan_est_o'], 0, sheep['i_dry_sales_forced_o'])
    sheep['i_drys_retained_birth_est_o'] = fun.f_update(sheep['i_drys_retained_birth_est_o'], 0, sheep['i_dry_sales_forced_o'])
    #### If retain drys is forced (True) then estimated proportion of drys retained is 1 otherwise the proportion remains as per the input.
    sheep['i_drys_retained_scan_est_o'] = fun.f_update(sheep['i_drys_retained_scan_est_o'], 1, sheep['i_dry_retained_forced_o'])
    sheep['i_drys_retained_birth_est_o'] = fun.f_update(sheep['i_drys_retained_birth_est_o'], 1, sheep['i_dry_retained_forced_o'])

    sheep['ia_r1_zig1'] = fun.f_sa(sheep['ia_r1_zig1'], sen.sav['r1_izg1'],5)
    sheep['ia_r2_ik0g1'] = fun.f_sa(sheep['ia_r2_ik0g1'], sen.sav['r2_ik0g1'],5)
    sheep['ia_r2_isk2g1'] = fun.f_sa(sheep['ia_r2_isk2g1'], sen.sav['r2_isk2g1'],5)
    sheep['ia_r1_zig3'] = fun.f_sa(sheep['ia_r1_zig3'], sen.sav['r1_izg3'],5)
    sheep['ia_r2_ik0g3'] = fun.f_sa(sheep['ia_r2_ik0g3'], sen.sav['r2_ik0g3'],5)
    feedsupply['i_feedsupply_adj_options_r2p'] = fun.f_sa(feedsupply['i_feedsupply_adj_options_r2p'], sen.sav['feedsupply_adj_r2p'], 5)   #SAV before SAA allows the Property.xl inputs to be overwritten with 0 and then add SAA values from exp.xl

    ###sam
    ###sap
    ###saa
    sheep['ia_r1_zig1'] = fun.f_sa(sheep['ia_r1_zig1'], sen.saa['r1_izg1'], 2).astype('int')
    sheep['ia_r2_isk2g1'] = fun.f_sa(sheep['ia_r2_isk2g1'], sen.saa['r2_isk2g1'], 2).astype('int')
    sheep['ia_r1_zig3'] = fun.f_sa(sheep['ia_r1_zig3'], sen.saa['r1_izg3'], 2).astype('int')
    sheep['ia_r2_ik5g3'] = fun.f_sa(sheep['ia_r2_ik5g3'], sen.saa['r2_ik5g3'], 2).astype('int')
    sheep['i_date_born1st_iog2'] = fun.f_sa(sheep['i_date_born1st_iog2'], sen.saa['date_born1st_iog'], 2)
    #sheep['i_date_born1st_idg3'] = fun.f_sa(sheep['i_date_born1st_idg3'], sen.saa['date_born1st_iog'], 2)
    feedsupply['i_feedsupply_options_r1j2p'] = fun.f_sa(feedsupply['i_feedsupply_options_r1j2p'], sen.saa['feedsupply_r1jp'], 2)
    feedsupply['i_feedsupply_adj_options_r2p'] = fun.f_sa(feedsupply['i_feedsupply_adj_options_r2p'], sen.saa['feedsupply_adj_r2p'], 2)
    ###sat
    ###sar
    feedsupply['i_feedsupply_options_r1j2p'] = fun.f_sa(feedsupply['i_feedsupply_options_r1j2p']
                                                , sen.sar['feedsupply_r1jp'], 4, value_min = 0.0, target = 13.0)

    ##mask out unrequired nodes dates - nodes are removed if there are double ups (note this used to remove nodes where no season was identified but there are cases when we want a node but no identification eg EWW)
    ## includes the masked out season in the test below. This is to remove randomness if comparing with a different season mask. If a season is removed we dont want the number of node periods to change.
    ## has to be here because if affects two inputs so cant put it in f_season_periods.
    ###test for duplicate
    duplicate_mask_m = []
    for m in range(general['i_date_node_zm'].shape[1]):  # maybe there is a way to do this without a loop.
        duplicate_mask_m.append(np.all(np.any(general['i_date_node_zm'][:,m:m+1] == general['i_date_node_zm'][:,0:m],axis=1,keepdims=True)))
    duplicate_mask_m = np.logical_not(duplicate_mask_m)
    mask_m = duplicate_mask_m
    ###if steady state and nodes are not included then mask out node period (except p7[0])
    if np.logical_not(sinp.structuralsa['i_inc_node_periods']) and (
            sinp.structuralsa['steady_state'] or np.count_nonzero(general['i_mask_z']) == 1):
        mask_m[1:] = False
    ###mask inputs
    general['i_date_node_zm'] = general['i_date_node_zm'][:,mask_m]
    general['i_node_is_fvp'] = general['i_node_is_fvp'][mask_m]
    general['i_phase_can_increase_kp7'] = general['i_phase_can_increase_kp7'][:,mask_m]
    general['i_phase_can_reduce_kp7'] = general['i_phase_can_reduce_kp7'][:,mask_m]

##############################
# handle inputs with p6 axis #
##############################
def f1_expand_p6():
    '''
    When using DSP, expand inputs with a p6 axis for each season node.
    Has to be a separate function to the sa because values altered in SA impact a_p6std_p6z
    '''
    ##have to import it here since sen.py imports this module
    from . import Periods as per

    ###get association between the input fp and the node adjusted fp
    a_p6std_p6z = per.f_feed_periods(option=2)
    a_p6std_zp6 = a_p6std_p6z.T
    ###apply association
    ####feedsupply
    feedsupply['i_confinement_options_r1p6z'] = np.take_along_axis(feedsupply['i_confinement_options_r1p6z'], a_p6std_p6z[na,:,:], axis=1)

    ####stock
    sheep['i_legume_p6z'] = np.take_along_axis(sheep['i_legume_p6z'], a_p6std_zp6, axis=1)
    sheep['i_supplement_zp6'] = np.take_along_axis(sheep['i_supplement_zp6'], a_p6std_zp6, axis=1)
    sheep['i_paststd_foo_zp6j0'] = np.take_along_axis(sheep['i_paststd_foo_zp6j0'], a_p6std_zp6[...,na], axis=1)
    sheep['i_paststd_dmd_zp6j0'] = np.take_along_axis(sheep['i_paststd_dmd_zp6j0'], a_p6std_zp6[...,na], axis=1)
    sheep['i_pasture_stage_p6z'] = np.take_along_axis(sheep['i_pasture_stage_p6z'], a_p6std_zp6, axis=1)
    sheep['i_density_p6z'] = np.take_along_axis(sheep['i_density_p6z'], a_p6std_zp6, axis=1)
    sheep['i_mobsize_sire_zp6i'] = np.take_along_axis(sheep['i_mobsize_sire_zp6i'], a_p6std_zp6[...,na], axis=1)
    sheep['i_mobsize_dams_zp6i'] = np.take_along_axis(sheep['i_mobsize_dams_zp6i'], a_p6std_zp6[...,na], axis=1)
    sheep['i_mobsize_offs_zp6i'] = np.take_along_axis(sheep['i_mobsize_offs_zp6i'], a_p6std_zp6[...,na], axis=1)
    sheep['i_dse_group_dp6z'] = np.take_along_axis(sheep['i_dse_group'][:,:,na], a_p6std_p6z[na,:,:], axis=1)
    sheep['i_wg_propn_p6z'] = np.take_along_axis(sheep['i_wg_propn_p6'][:,na], a_p6std_p6z, axis=0)

    ####crop grazing
    cropgraze['i_cropgraze_yield_reduction_scalar_zp6'] = np.take_along_axis(cropgraze['i_cropgraze_yield_reduction_scalar_zp6'], a_p6std_zp6, axis=1)
    cropgraze['i_crop_dmd_kp6z'] = np.take_along_axis(cropgraze['i_crop_dmd_kp6'][...,na], a_p6std_p6z[na,...], axis=1)
    cropgraze['i_crop_growth_zp6'] = np.take_along_axis(cropgraze['i_crop_growth_zp6'], a_p6std_zp6[:,:], axis=1)
    cropgraze['i_cropgraze_consumption_factor_zp6'] = np.take_along_axis(cropgraze['i_cropgraze_consumption_factor_zp6'], a_p6std_zp6, axis=1)

    ###saltbush
    saltbush['i_sb_expected_foo_zp6'] = np.take_along_axis(saltbush['i_sb_expected_foo_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_expected_growth_zp6'] = np.take_along_axis(saltbush['i_sb_expected_growth_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_growth_reduction_zp6'] = np.take_along_axis(saltbush['i_sb_growth_reduction_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_ash_content_zp6'] = np.take_along_axis(saltbush['i_sb_ash_content_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_selectivity_zp6'] = np.take_along_axis(saltbush['i_sb_selectivity_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_slp_diet_propn_zp6'] = np.take_along_axis(saltbush['i_slp_diet_propn_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_cp_zp6'] = np.take_along_axis(saltbush['i_sb_cp_zp6'], a_p6std_zp6, axis=1)

    ####pasture
    for pasture in sinp.general['pastures'][general['pas_inc_t']]:
        pasture_inputs[pasture]['POCCons'] = np.take_along_axis(pasture_inputs[pasture]['POCCons'][:,:,na], a_p6std_p6z[:,na,:], axis=0)
        pasture_inputs[pasture]['i_pasture_stage_p6z'] = np.take_along_axis(pasture_inputs[pasture]['i_pasture_stage_p6z'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['DigRednSenesce'] = np.take_along_axis(pasture_inputs[pasture]['DigRednSenesce'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['DigDryAve'] = np.take_along_axis(pasture_inputs[pasture]['DigDryAve'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['DigDryRange'] = np.take_along_axis(pasture_inputs[pasture]['DigDryRange'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['FOODryH'] = np.take_along_axis(pasture_inputs[pasture]['FOODryH'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['GermScalarFP'] = np.take_along_axis(pasture_inputs[pasture]['GermScalarFP'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['CPGrn'] = np.take_along_axis(pasture_inputs[pasture]['CPGrn'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['CPDry'] = np.take_along_axis(pasture_inputs[pasture]['CPDry'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['DigPOC'] = np.take_along_axis(pasture_inputs[pasture]['DigPOC'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['FOOPOC'] = np.take_along_axis(pasture_inputs[pasture]['FOOPOC'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['DigSpread'] = np.take_along_axis(pasture_inputs[pasture]['DigSpread'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['PGRScalarH'] = np.take_along_axis(pasture_inputs[pasture]['PGRScalarH'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['BaseLevelInput'] = np.take_along_axis(pasture_inputs[pasture]['BaseLevelInput'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['ErosionLimit'] = np.take_along_axis(pasture_inputs[pasture]['ErosionLimit'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['SenescePropn'] = np.take_along_axis(pasture_inputs[pasture]['SenescePropn'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['SenesceEOS'] = np.take_along_axis(pasture_inputs[pasture]['SenesceEOS'], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['LowFOO'] = np.take_along_axis(pasture_inputs[pasture]['LowFOO'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['MedFOO'] = np.take_along_axis(pasture_inputs[pasture]['MedFOO'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['LowPGR'] = np.take_along_axis(pasture_inputs[pasture]['LowPGR'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['MedPGR'] = np.take_along_axis(pasture_inputs[pasture]['MedPGR'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['DigGrn'] = np.take_along_axis(pasture_inputs[pasture]['DigGrn'], a_p6std_zp6[:,:,na], axis=1)
        pasture_inputs[pasture]['MaintenanceEff'] = np.take_along_axis(pasture_inputs[pasture]['MaintenanceEff'][:,na,:], a_p6std_p6z[:,:,na], axis=0)

        ###end of growing season and period when dry feed exists need special handling because they are fp pointer
        index_p6z = np.arange(len(a_p6std_p6z))[:,na]
        pasture_inputs[pasture]['EndGS'] = np.max((a_p6std_p6z == pasture_inputs[pasture]['EndGS']) * index_p6z, axis=0)
        pasture_inputs[pasture]['i_dry_exists'] = np.max((a_p6std_p6z == pasture_inputs[pasture]['i_dry_exists']) * index_p6z, axis=0) \
                                                  - (np.count_nonzero(a_p6std_p6z == pasture_inputs[pasture]['i_dry_exists'], axis=0) -1) #have to minus count non zero in case an extra fp is added in the period dry pas becomes available. because in this case we still want to point at the first period it become available (opposite to end of gs)

    ###crop residue
    stubble['i_fp_end_stub_z'] = np.max((a_p6std_p6z == stubble['i_fp_end_stub_z']) * index_p6z, axis=0)

    ###fp index needs special handling because it isn't just expanded it is rebuilt
    period['i_fp_idx'] = ['fp%02d'%i for i in range(len(a_p6std_p6z))]



##################
#mask lmu inputs #
##################
##function to do the masking
def f1_do_mask_lmu(dict, key, lmu_axis):
    ##check input with lmu axis was adjusted (if running from the web app)
    if 'lmu_flag' in globals():
        if not key in lmu_flag:
            raise UserWarning("{0} has not been through f_farmer_lmu_adj".format(key))

    ##mask lmu
    if isinstance(dict[key], pd.DataFrame) or isinstance(dict[key], pd.Series):
        if lmu_axis == 0:
            dict[key] = dict[key].loc[lmu_mask]
        elif lmu_axis == 1:
            dict[key] = dict[key].loc[:, lmu_mask]
    else:
        dict[key] = np.compress(lmu_mask, dict[key], lmu_axis)


def f1_mask_lmu():
    ##make the mask - this is a global input because used to mask incode SAVs in Bounds.py
    global lmu_mask
    lmu_mask = general['i_lmu_area'] > 0

    
    ##general
    f1_do_mask_lmu(general, "i_lmu_idx", lmu_axis=0)
    f1_do_mask_lmu(general, "i_lmu_area", lmu_axis=0)
    f1_do_mask_lmu(general, "i_non_cropable_area_l", lmu_axis=0)
    f1_do_mask_lmu(general, "arable", lmu_axis=0)

    ##crop
    f1_do_mask_lmu(crop, "yield_by_lmu", lmu_axis=1)
    f1_do_mask_lmu(crop, "i_soil_production_z_l", lmu_axis=1)
    f1_do_mask_lmu(crop, "seeding_penalty_lmu_scalar_lp", lmu_axis=0)
    f1_do_mask_lmu(crop, "fert_by_lmu", lmu_axis=1)
    f1_do_mask_lmu(crop, "chem_by_lmu", lmu_axis=1)
    f1_do_mask_lmu(crop, "seeding_rate", lmu_axis=1)

    ##cropgraze
    f1_do_mask_lmu(cropgraze, "i_cropgrowth_lmu_factor_kl", lmu_axis=1)
    f1_do_mask_lmu(cropgraze, "i_cropgraze_propn_area_grazed_kl", lmu_axis=1)

    ##saltbush
    f1_do_mask_lmu(saltbush, "i_sb_lmu_scalar", lmu_axis=0)

    ##pasture
    for pasture in sinp.general['pastures'][general['i_pastures_exist']]:
        f1_do_mask_lmu(pasture_inputs[pasture], "i_pasture_coverage", lmu_axis=0)
        f1_do_mask_lmu(pasture_inputs[pasture], "GermScalarLMU", lmu_axis=1)
        f1_do_mask_lmu(pasture_inputs[pasture], "FaG_LMU", lmu_axis=0)
        f1_do_mask_lmu(pasture_inputs[pasture], "LowFOO", lmu_axis=2)
        f1_do_mask_lmu(pasture_inputs[pasture], "LowPGR", lmu_axis=2)
        f1_do_mask_lmu(pasture_inputs[pasture], "MedFOO", lmu_axis=2)
        f1_do_mask_lmu(pasture_inputs[pasture], "MedPGR", lmu_axis=2)
        f1_do_mask_lmu(pasture_inputs[pasture], "DigGrn", lmu_axis=2)
        f1_do_mask_lmu(pasture_inputs[pasture], "i_soil_production_zl", lmu_axis=1)
        f1_do_mask_lmu(pasture_inputs[pasture], "POCCons", lmu_axis=1)
        f1_do_mask_lmu(pasture_inputs[pasture], "ErosionLimit", lmu_axis=2)

    ##machine
    f1_do_mask_lmu(mach, "seeding_fuel_lmu_adj", lmu_axis=0)
    f1_do_mask_lmu(mach, "tillage_maint_lmu_adj", lmu_axis=0)
    f1_do_mask_lmu(mach, "seeding_rate_lmu_adj", lmu_axis=0)
    
    ##tree
    f1_do_mask_lmu(tree, "tree_fert_soil_scalar", lmu_axis=0)
    f1_do_mask_lmu(tree, "estimated_area_trees_l", lmu_axis=0)
    f1_do_mask_lmu(tree, "lmu_growth_scalar_l", lmu_axis=0)
    f1_do_mask_lmu(tree, "lmu_carbon_scalar_l", lmu_axis=0)

def f1_do_mask_landuse(dict, key, landuse_axis_type, landuse_axis):
    '''
    This function mask the land use axis. Note, sometimes the crop/pasture landuses are
    input separate and sometimes together.
    :param dict: input dictionary.
    :param key:  dictionary key to find input.
    :param landuse_axis_type: "all" - all land uses, "crop" - crop land uses, "pas" - pasture land uses.
    :param landuse_axis: axis position
    :return:
    '''

    ##select the masking
    if landuse_axis_type=="all":
        k_mask = all_landuse_mask_k
    elif landuse_axis_type=="crop":
        k_mask = crop_landuse_mask_k1
    elif landuse_axis_type=="pas":
        k_mask = pas_landuse_mask_k2

    ##mask landuse
    if isinstance(dict[key], pd.DataFrame) or isinstance(dict[key], pd.Series):
        if landuse_axis == 0:
            dict[key] = dict[key].loc[k_mask]
        elif landuse_axis == 1:
            dict[key] = dict[key].loc[:, k_mask]
    else:
        dict[key] = np.compress(k_mask, dict[key], landuse_axis)

def f1_mask_landuse():
    ##make the mask - this is a global input because used to mask incode SAVs in Bounds.py
    global crop_landuse_mask_k1
    global pas_landuse_mask_k2
    global all_landuse_mask_k
    crop_landuse_mask_k1 = np.logical_and(general['i_crop_landuse_exists_k1'], general['i_crop_landuse_inc_k1'])
    if not any(crop_landuse_mask_k1):
        crop_landuse_mask_k1[0] =True #need at least one active crop so that code works. Note this crop is still excluded in the rotations.
    pas_landuse_mask_k2 = np.logical_and(general['i_pas_landuse_exists_k2'], general['i_pas_landuse_inc_k2'])
    ###create the k mask for the full land use array. Needs to be ordered correctly
    ####concat the crop and pasture mask
    t_mask_k = np.concatenate([crop_landuse_mask_k1, pas_landuse_mask_k2], axis=0)
    t_idx_k = np.concatenate([sinp.general['i_idx_k1'], sinp.general['i_idx_k2']], axis=0)
    ####order mask_k to match the input order (alphebetic)
    a_tk_k = np.where(sinp.general['i_idx_k'][:, na] == t_idx_k)[1]
    all_landuse_mask_k = t_mask_k[a_tk_k]

    ##general
    f1_do_mask_landuse(general, "i_phase_can_increase_kp7", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(general, "i_phase_can_reduce_kp7", landuse_axis_type="all", landuse_axis=0)

    ##crop
    f1_do_mask_landuse(crop, "seeding_yield_penalty", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(crop, "seeding_penalty_scalar_kz", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(crop, "start_harvest_crops", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(crop, "yield_by_lmu", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(crop, "seeding_rate", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(crop, "seed_info", landuse_axis_type="all", landuse_axis=0)

    ##emmisions
    f1_do_mask_landuse(emissions, "i_burn_propn_k", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(emissions, "i_nitrogen_applied_k", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(emissions, "i_propn_Urea", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(emissions, "i_lime_applied_k", landuse_axis_type="all", landuse_axis=0)

    ##cropgrazing
    f1_do_mask_landuse(cropgraze, "i_cropgraze_propn_area_grazed_kl", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(cropgraze, "i_crop_growth_landuse_scalar_k", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(cropgraze, "i_cropgrowth_lmu_factor_kl", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(cropgraze, "i_cropgraze_wastage", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(cropgraze, "i_cropgraze_yield_reduction_k", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(cropgraze, "i_crop_dmd_kp6z", landuse_axis_type="crop", landuse_axis=0)

    ##labour
    f1_do_mask_landuse(labour, "harvest_helper", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(labour, "variable_crop_monitoring", landuse_axis_type="crop", landuse_axis=0)

    ##stub
    f1_do_mask_landuse(stubble, "stubble_handling", landuse_axis_type="crop", landuse_axis=0)

    ##Index names - sinp
    f1_do_mask_landuse(sinp.general, "i_idx_k", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(sinp.general, "i_idx_k1", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(sinp.general, "i_idx_k2", landuse_axis_type="pas", landuse_axis=0)
    f1_do_mask_landuse(sinp.general, "i_is_baled_k", landuse_axis_type="crop", landuse_axis=0)
    f1_do_mask_landuse(sinp.general, "i_landuse_is_dual", landuse_axis_type="all", landuse_axis=0)
    f1_do_mask_landuse(sinp.general, "i_history4_req", landuse_axis_type="all", landuse_axis=0)



##############
#phases      #
##############
def f1_phases(d_rot_info):
    '''
    Rotation info.

    When using pinp rotations they are all included but if using full rotations then a mask is applied.
    The user can control the mask in SimInputs_{property}.xl.
    
    :param mask_r: If True the function returns the rot included mask.
    :param check: If True the function returns nothing - it is just used to check the correct rotations exist in rot.xl.
    '''
    ##set to global so they can be accessed in other module without having to pass into each function
    global phases_r
    global rot_req
    global rot_prov
    global s_rotcon1
    global rot_mask_r
    global seeding_freq_r

    phases_r = d_rot_info["phases_r"]
    rot_req = d_rot_info["rot_req"]
    rot_prov = d_rot_info["rot_prov"]
    s_rotcon1 = d_rot_info["s_rotcon1"]

    ###add variable that is the number of yrs in the rot phases
    if phases_r is not None:
        sinp.general['phase_len'] = len(phases_r.columns)

    ##check that the rotations match the inputs. If not then re-run rotation generation. If still not the same
    ## quit and leave error message (most likely the user needs to re-run APSIM).
    if crop['user_crop_rot']:
        ### User defined
        base_yields = crop['yields']
    else:
        ### Simulation version
        ###build path this way so the file can be access even if AFO is run from another directory eg readthedocs or web app.
        property = general['i_property_id']
        xl_path = relativeFile.findExcel("SimInputs_{0}.xlsx".format(property))
        base_yields = pd.read_excel(xl_path, sheet_name='Yield', index_col=0, header=0, engine='openpyxl')
    ###if the rotations don't match inputs then rerun rotation generation.
    if len(phases_r) != len(base_yields) or any(base_yields.index!=phases_r.index):
        ###maybe there is a better way to do this rather than using if statements.
        property = general['i_property_id']
        if property=="EWW":
            from . import RotGeneration_EWW as RotGeneration
        elif property=="GSW":
            from . import RotGeneration_GSW as RotGeneration
        elif property=="CWW":
            from . import RotGeneration_CWW as RotGeneration
        else:
            from . import RotGeneration
        d_rot_info = RotGeneration.f_rot_gen(crop['user_crop_rot'])
        phases_r = d_rot_info["phases_r"]
        rot_req = d_rot_info["rot_req"]
        rot_prov = d_rot_info["rot_prov"]
        s_rotcon1 = d_rot_info["s_rotcon1"]
        ###update len rot
        sinp.general['phase_len'] = len(phases_r.columns)

        ##if they still don't match then the user will need to either re-run simulation model (apsim) or change rotgeneration to line up with the rotations that have been simulated.
        if len(phases_r) != len(base_yields) or any(base_yields.index!=phases_r.index):
            print('''WARNING: Rotations don't match inputs.
                   Things to check: 
                   1. if you are using full rotation have you generated the full inputs for the selected property
                   2. if you have generated new rotations have you re-generated the full inputs (ie re-run simulation)?
                   3. the named ranges in for the user defined rotations and inputs are all correct''')
            sys.exit()

    ##rotation mask - read in from excel
    if crop['user_crop_rot']:
        rot_mask_r = crop['i_user_rot_inc_r']
        seeding_freq_r = crop['i_seeding_freq_r']
    else:
        ###build path this way so the file can be access even if AFO is run from another directory eg readthedocs or web app.
        property = general['i_property_id']
        xl_path = relativeFile.findExcel("SimInputs_{0}.xlsx".format(property))
        rot_mask_r = pd.read_excel(xl_path, sheet_name='RotMask', index_col=0, header=0, engine='openpyxl').squeeze().values
        seeding_freq_r = pd.read_excel(xl_path, sheet_name='SeedFreq', index_col=0, header=0, engine='openpyxl').squeeze().values

    ##add user landuse mask to rot mask (allows users to remove all phases with a given landuse)
    crop_landuse_mask_k1 = np.logical_and(general['i_crop_landuse_exists_k1'], general['i_crop_landuse_inc_k1'])
    pas_landuse_mask_k2 = np.logical_and(general['i_pas_landuse_exists_k2'], general['i_pas_landuse_inc_k2'])
    crop_landuse_mask_r = np.sum((phases_r.iloc[:,-1].values[:,na]==sinp.general['i_idx_k1']) * crop_landuse_mask_k1, axis=1)
    pas_landuse_mask_r = np.sum((phases_r.iloc[:,-1].values[:,na]==sinp.general['i_idx_k2']) * pas_landuse_mask_k2, axis=1)
    landuse_mask_r = np.logical_or(crop_landuse_mask_r, pas_landuse_mask_r)
    rot_mask_r = np.logical_and(rot_mask_r, landuse_mask_r)

    ##apply mask
    phases_r_not_masked = phases_r #save version without mask so that rot generator doesnt get run everytime the landuse mask changes.
    phases_r = phases_r.loc[rot_mask_r,:]
    seeding_freq_r = seeding_freq_r[rot_mask_r]
    rot_req_not_masked = rot_req
    rot_prov_not_masked = rot_prov
    mask_req = rot_req[0].isin(phases_r.index)
    rot_req = rot_req.loc[mask_req,:]
    mask_prov = rot_prov[0].isin(phases_r.index)
    rot_prov = rot_prov.loc[mask_prov,:]

    return {"phases_r": phases_r_not_masked, "rot_req": rot_req_not_masked, "rot_prov": rot_prov_not_masked, "s_rotcon1": s_rotcon1}

