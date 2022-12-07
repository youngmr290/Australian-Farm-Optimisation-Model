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
import Functions as fun
import StructuralInputs as sinp

na = np.newaxis

##############
#read inputs #
##############
##determine which properties are used in current exp
pinp_req = fun.f_read_exp(pinp_req=True)

##read in inputs
inputs={}
for property in pinp_req:
    inputs[property] = {}
    ##build path this way so that readthedocs can read correctly.
    directory_path = os.path.dirname(os.path.abspath(__file__))
    property_xl_path = os.path.join(directory_path, "Property_{0}.xlsx".format(property))
    property_pkl_path = os.path.join(directory_path, "pkl_property_{0}.pkl".format(property))
    try:
        if os.path.getmtime(property_xl_path) > os.path.getmtime(property_pkl_path):
            inputs_from_pickle = False
        else:
            inputs_from_pickle = True
            print('Reading property {0} inputs from pickle'.format(property), end=' ', flush=True)
    except FileNotFoundError:
        inputs_from_pickle = False

    ##if inputs are not read from pickle then they are read from excel and written to pickle
    if inputs_from_pickle == False:
        print('Reading property {0} inputs from Excel'.format(property), end=' ', flush=True)
        with open(property_pkl_path, "wb") as f:
            inputs[property]['general_inp'] = fun.xl_all_named_ranges(property_xl_path,"General", numpy=True)
            pkl.dump(inputs[property]['general_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['labour_inp'] = fun.xl_all_named_ranges(property_xl_path,"Labour")
            pkl.dump(inputs[property]['labour_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['crop_inp'] = fun.xl_all_named_ranges(property_xl_path,"Crop")
            pkl.dump(inputs[property]['crop_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['cropgraze_inp'] = fun.xl_all_named_ranges(property_xl_path,"CropGrazing", numpy=True)
            pkl.dump(inputs[property]['cropgraze_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['saltbush_inp'] = fun.xl_all_named_ranges(property_xl_path,"Saltbush", numpy=True)
            pkl.dump(inputs[property]['saltbush_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['mach_inp'] = fun.xl_all_named_ranges(property_xl_path,"Mach")
            pkl.dump(inputs[property]['mach_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['stubble_inp'] = fun.xl_all_named_ranges(property_xl_path,"CropResidue", numpy=True)
            pkl.dump(inputs[property]['stubble_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['finance_inp'] = fun.xl_all_named_ranges(property_xl_path,"Finance")
            pkl.dump(inputs[property]['finance_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['period_inp'] = fun.xl_all_named_ranges(property_xl_path,"Periods", numpy=True) #automatically read in the periods as dates
            pkl.dump(inputs[property]['period_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['sup_inp'] = fun.xl_all_named_ranges(property_xl_path,"Sup Feed")
            pkl.dump(inputs[property]['sup_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['sheep_inp']  = fun.xl_all_named_ranges(property_xl_path, 'Sheep', numpy=True)
            pkl.dump(inputs[property]['sheep_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['feedsupply_inp']  = fun.xl_all_named_ranges(property_xl_path, 'FeedSupply', numpy=True)
            pkl.dump(inputs[property]['feedsupply_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['mvf_inp']  = fun.xl_all_named_ranges(property_xl_path, 'MVEnergy', numpy=True)
            pkl.dump(inputs[property]['mvf_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

            inputs[property]['pasture_inp']=dict()
            for pasture in sinp.general['pastures'][inputs[property]['general_inp']['i_pastures_exist']]:
                inputs[property]['pasture_inp'][pasture] = fun.xl_all_named_ranges(property_xl_path, pasture, numpy=True)
            pkl.dump(inputs[property]['pasture_inp'], f, protocol=pkl.HIGHEST_PROTOCOL)

    ##else the inputs are read in from the pickle file
    ##note this must be in the same order as above
    else:
        with open(property_pkl_path, "rb") as f:
            inputs[property]['general_inp'] = pkl.load(f)

            inputs[property]['labour_inp'] = pkl.load(f)

            inputs[property]['crop_inp'] = pkl.load(f)

            inputs[property]['cropgraze_inp'] = pkl.load(f)

            inputs[property]['saltbush_inp'] = pkl.load(f)

            inputs[property]['mach_inp'] = pkl.load(f)

            inputs[property]['stubble_inp'] = pkl.load(f)

            inputs[property]['finance_inp'] = pkl.load(f)

            inputs[property]['period_inp'] = pkl.load(f)

            inputs[property]['sup_inp'] = pkl.load(f)

            inputs[property]['sheep_inp'] = pkl.load(f)

            inputs[property]['feedsupply_inp'] = pkl.load(f)

            inputs[property]['mvf_inp'] = pkl.load(f)

            inputs[property]['pasture_inp'] = pkl.load(f)

    print('- finished')

    ##reshape required inputs
    ###lengths
    len_d = len(inputs[property]['sheep_inp']['i_d_idx'])
    len_h2 = inputs[property]['sheep_inp']['i_h2_len']
    len_h5 = inputs[property]['sheep_inp']['i_h5_len']
    len_h7 = inputs[property]['sheep_inp']['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
    len_i = inputs[property]['sheep_inp']['i_i_len']
    len_j2 = inputs[property]['feedsupply_inp']['i_j2_len']
    len_k = len(sinp.landuse['C'])
    len_k0 = inputs[property]['sheep_inp']['i_k0_len']
    len_k1 = inputs[property]['sheep_inp']['i_k1_len']
    len_k2 = inputs[property]['sheep_inp']['i_k2_len']
    len_k3 = inputs[property]['sheep_inp']['i_k3_len']
    len_k4 = inputs[property]['sheep_inp']['i_k4_len']
    len_k5 = inputs[property]['sheep_inp']['i_k5_len']
    len_l = len(inputs[property]['general_inp']['i_lmu_area'])
    len_o = inputs[property]['sheep_inp']['i_o_len']
    len_p6 = len(inputs[property]['period_inp']['i_fp_idx'])
    len_r1 = inputs[property]['feedsupply_inp']['i_r1_len']
    len_s = inputs[property]['sheep_inp']['i_s_len'] #s = shear
    len_sc = sinp.stock['i_len_s'] #sc = scan
    len_t3 = inputs[property]['sheep_inp']['i_t3_len']
    len_x = inputs[property]['sheep_inp']['i_x_len']
    len_z = len(inputs[property]['general_inp']['i_mask_z'])


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


    ###pasture
    for t,pasture in enumerate(sinp.general['pastures'][inputs[property]['general_inp']['i_pastures_exist']]):
        inp = inputs[property]['pasture_inp'][pasture]
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

    ###crop grazing
    cropgraze_inp = inputs[property]['cropgraze_inp']
    cropgraze_inp['i_crop_growth_zkp6'] = np.reshape(cropgraze_inp['i_crop_growth_zkp6'], zkp6)

    ###saltbush
    saltbush_inp = inputs[property]['saltbush_inp']
    saltbush_inp['i_sb_expected_foo_zp6'] = np.reshape(saltbush_inp['i_sb_expected_foo_zp6'], zp6)
    saltbush_inp['i_sb_expected_growth_zp6'] = np.reshape(saltbush_inp['i_sb_expected_growth_zp6'], zp6)
    saltbush_inp['i_sb_growth_reduction_zp6'] = np.reshape(saltbush_inp['i_sb_growth_reduction_zp6'], zp6)
    saltbush_inp['i_sb_ash_content_zp6'] = np.reshape(saltbush_inp['i_sb_ash_content_zp6'], zp6)
    saltbush_inp['i_sb_selectivity_zp6'] = np.reshape(saltbush_inp['i_sb_selectivity_zp6'], zp6)
    saltbush_inp['i_slp_diet_propn_zp6'] = np.reshape(saltbush_inp['i_slp_diet_propn_zp6'], zp6)

    ###stock
    sheep_inp = inputs[property]['sheep_inp']
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
    feedsupply_inp = inputs[property]['feedsupply_inp']
    feedsupply_inp['i_feedsupply_options_r1j2p'] = np.reshape(feedsupply_inp['i_feedsupply_options_r1j2p'], r1j2P)
    feedsupply_inp['i_confinement_options_r1p6z'] = np.reshape(feedsupply_inp['i_confinement_options_r1p6z'], r1p6z)

def f_select_pinp(property):
    ##create a copy of each input dict - so that the base inputs remain unchanged
    ##the copy created is the one used in the actual modules

    print('Using property: {0}'.format(property))

    ##needs to be global so inputs can be accessed outside this function
    global general
    global labour
    global crop
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
    general = copy.deepcopy(inputs[property]['general_inp'])
    labour = copy.deepcopy(inputs[property]['labour_inp'])
    crop = copy.deepcopy(inputs[property]['crop_inp'])
    cropgraze = copy.deepcopy(inputs[property]['cropgraze_inp'])
    saltbush = copy.deepcopy(inputs[property]['saltbush_inp'])
    mach = copy.deepcopy(inputs[property]['mach_inp'])
    stubble = copy.deepcopy(inputs[property]['stubble_inp'])
    finance = copy.deepcopy(inputs[property]['finance_inp'])
    period = copy.deepcopy(inputs[property]['period_inp'])
    supfeed = copy.deepcopy(inputs[property]['sup_inp'])
    sheep = copy.deepcopy(inputs[property]['sheep_inp'])
    feedsupply = copy.deepcopy(inputs[property]['feedsupply_inp'])
    mvf = copy.deepcopy(inputs[property]['mvf_inp'])
    pasture_inputs = copy.deepcopy(inputs[property]['pasture_inp'])


#######################
#apply SA             #
#######################
def f_property_inp_sa():
    '''

    Applies sensitivity adjustment to each input. After the sensitivities are applied, when using the DSP model, inputs
    with a feed period index are expanded to account for additional feed periods that are added due to season nodes.
    This function gets called at the beginning of each loop in the exp.py module

    SA order is: sav, sam, sap, saa, sat, sar.
    So that multiple SA can be applied to one input.

    :return: None.

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen

    ##general
    ###sav
    general['steady_state'] = fun.f_sa(general['steady_state'], sen.sav['steady_state'], 5)
    general['i_mask_z'] = fun.f_sa(general['i_mask_z'], sen.sav['mask_z'], 5)
    general['i_inc_node_periods'] = fun.f_sa(general['i_inc_node_periods'], sen.sav['inc_node_periods'], 5)
    general['i_len_q'] = fun.f_sa(general['i_len_q'], sen.sav['seq_len'], 5)
    labour['max_managers'] = fun.f_sa(labour['max_managers'], sen.sav['manager_ub'], 5)
    labour['max_casual'] = fun.f_sa(labour['max_casual'], sen.sav['casual_ub'], 5)
    labour['max_casual_seedharv'] = fun.f_sa(labour['max_casual_seedharv'], sen.sav['seedharv_casual_ub'], 5)
    general['i_lmu_area'] = fun.f_sa(general['i_lmu_area'], sen.sav['lmu_area_l'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##finance
    ###sav
    finance['capital_limit'] = fun.f_sa(finance['capital_limit'], sen.sav['capital_limit'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##crop
    ###sav
    crop['user_crop_rot'] = fun.f_sa(crop['user_crop_rot'], sen.sav['pinp_rot'], 5)
    ###sam
    crop['yields'] = fun.f_sa(crop['yields'], sen.sam['all_rot_yield'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##machinery
    ###sav
    mach['option'] = fun.f_sa(mach['option'], sen.sav['mach_option'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##crop grazing
    ###sav
    cropgraze['i_cropgrazing_inc'] = fun.f_sa(cropgraze['i_cropgrazing_inc'], sen.sav['cropgrazing_inc'], 5)
    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##salt land pasture
    ###sav
    saltbush['i_saltbush_inc'] = fun.f_sa(saltbush['i_saltbush_inc'], sen.sav['slp_inc'], 5)
    ###sam
    saltbush['i_sb_expected_growth_zp6'] = fun.f_sa(saltbush['i_sb_expected_growth_zp6'], sen.sam['sb_growth'])
    ###sap
    ###saa
    ###sat
    ###sar

    ##pasture
    ###sav
    crop['i_poc_inc'] = fun.f_sa(crop['i_poc_inc'], sen.sav['poc_inc'], 5)
    general['pas_inc'] = fun.f_sa(general['pas_inc'], sen.sav['pas_inc_t'], 5)

    for pasture in sinp.general['pastures'][general['pas_inc']]: #all pasture inputs are adjusted even if a given pasture is not included
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
    sheep['i_mask_i'] = fun.f_sa(sheep['i_mask_i'], sen.sav['TOL_inc'], 5)
    sheep['i_g3_inc'] = fun.f_sa(sheep['i_g3_inc'], sen.sav['g3_included'],5)
    sheep['a_c2_c0'] = fun.f_sa(sheep['a_c2_c0'], sen.sav['genotype'],5)
    sheep['i_scan_og1'] = fun.f_sa(sheep['i_scan_og1'], sen.sav['scan_og1'],5)
    sheep['i_dry_sales_forced_o'] = fun.f_sa(sheep['i_dry_sales_forced_o'], sen.sav['bnd_drys_sold_o'],5)
    sheep['i_dry_retained_forced_o'] = fun.f_sa(sheep['i_dry_retained_forced_o'], sen.sav['bnd_drys_retained_o'],5)
    ### The expected proportion retained at scanning or birth is a 3-step calc. Update with own SAV and then override if either of the dry management options is forced
    sheep['i_drys_retained_scan_est_o'] = fun.f_sa(sheep['i_drys_retained_scan_est_o'], sen.sav['est_drys_retained_scan_o'], 5)
    ### If sale of drys is forced then proportion of drys retained is 0, so need to convert a True in the SAV to False (which converts to 0).
    bnd_drys_sold_o = sen.sav['bnd_drys_sold_o'].copy()
    bnd_drys_sold_o[bnd_drys_sold_o == True] = False  # '0'
    sheep['i_drys_retained_scan_est_o'] = fun.f_sa(sheep['i_drys_retained_scan_est_o'], bnd_drys_sold_o,5)
    ### If retain drys is forced (True) then proportion of drys retained is 1 so can use the SAV[] (True == 1)
    sheep['i_drys_retained_scan_est_o'] = fun.f_sa(sheep['i_drys_retained_scan_est_o'], sen.sav['bnd_drys_retained_o'],5)
    sheep['i_drys_retained_birth_est_o'] = fun.f_sa(sheep['i_drys_retained_birth_est_o'], sen.sav['est_drys_retained_birth_o'], 5)
    sheep['i_drys_retained_birth_est_o'] = fun.f_sa(sheep['i_drys_retained_birth_est_o'], bnd_drys_sold_o,5)
    sheep['i_drys_retained_birth_est_o'] = fun.f_sa(sheep['i_drys_retained_birth_est_o'], sen.sav['bnd_drys_retained_o'],5)
    sheep['ia_r1_zig1'] = fun.f_sa(sheep['ia_r1_zig1'], sen.sav['r1_izg1'],5)
    sheep['ia_r2_ik0g1'] = fun.f_sa(sheep['ia_r2_ik0g1'], sen.sav['r2_ik0g1'],5)
    sheep['ia_r2_isk2g1'] = fun.f_sa(sheep['ia_r2_isk2g1'], sen.sav['r2_isk2g1'],5)
    sheep['ia_r1_zig3'] = fun.f_sa(sheep['ia_r1_zig3'], sen.sav['r1_izg3'],5)
    sheep['ia_r2_ik0g3'] = fun.f_sa(sheep['ia_r2_ik0g3'], sen.sav['r2_ik0g3'],5)
    sheep['i_sr_constraint_t'] = fun.f_sa(sheep['i_sr_constraint_t'], sen.sav['bnd_sr_t'],5)
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

    ##mask out unrequired nodes dates - nodes are removed if there are double ups or if a season is not identified at the node (and node is not used as fvp)
    ## includes the masked out season in the test below. This is to remove randomness if comparing with a different season mask. If a season is removed we dont want the number of node periods to change.
    ## has to be here because if affects two inputs so cant put it in f_season_periods.
    ###test for duplicate
    duplicate_mask_m = []
    for m in range(general['i_date_node_zm'].shape[1]):  # maybe there is a way to do this without a loop.
        duplicate_mask_m.append(np.all(np.any(general['i_date_node_zm'][:,m:m+1] == general['i_date_node_zm'][:,0:m],axis=1,keepdims=True)))
    duplicate_mask_m = np.logical_not(duplicate_mask_m)
    ###test if any season is identified at the node
    import SeasonalFunctions as zfun #have to import here since zfun imports pinp.
    mask_zm = np.logical_or(general['i_date_initiate_z'][:,na]==general['i_date_node_zm'], general['i_node_is_fvp'])
    mask_m = np.any(mask_zm, axis=0)
    mask_m = np.logical_and(duplicate_mask_m, mask_m)
    ###if steady state and nodes are not included then mask out node period (except p7[0])
    if np.logical_not(general['i_inc_node_periods']) and (
            general['steady_state'] or np.count_nonzero(general['i_mask_z']) == 1):
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
    import Periods as per

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
    sheep['i_dse_group'] = np.take_along_axis(sheep['i_dse_group'][:,:,na], a_p6std_p6z[na,:,:], axis=1)
    sheep['i_wg_propn_p6z'] = np.take_along_axis(sheep['i_wg_propn_p6'][:,na], a_p6std_p6z, axis=0)

    ####crop grazing
    cropgraze['i_cropgraze_yield_reduction_kp6z'] = np.take_along_axis(cropgraze['i_cropgraze_yield_reduction_kp6'][...,na], a_p6std_p6z[na,...], axis=1)
    cropgraze['i_crop_dmd_kp6z'] = np.take_along_axis(cropgraze['i_crop_dmd_kp6'][...,na], a_p6std_p6z[na,...], axis=1)
    cropgraze['i_crop_growth_zkp6'] = np.take_along_axis(cropgraze['i_crop_growth_zkp6'], a_p6std_zp6[:,na,:], axis=2)
    cropgraze['i_cropgraze_consumption_factor_zp6'] = np.take_along_axis(cropgraze['i_cropgraze_consumption_factor_zp6'], a_p6std_zp6, axis=1)

    ###saltbush
    saltbush['i_sb_expected_foo_zp6'] = np.take_along_axis(saltbush['i_sb_expected_foo_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_expected_growth_zp6'] = np.take_along_axis(saltbush['i_sb_expected_growth_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_growth_reduction_zp6'] = np.take_along_axis(saltbush['i_sb_growth_reduction_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_ash_content_zp6'] = np.take_along_axis(saltbush['i_sb_ash_content_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_sb_selectivity_zp6'] = np.take_along_axis(saltbush['i_sb_selectivity_zp6'], a_p6std_zp6, axis=1)
    saltbush['i_slp_diet_propn_zp6'] = np.take_along_axis(saltbush['i_slp_diet_propn_zp6'], a_p6std_zp6, axis=1)

    ####pasture
    for pasture in sinp.general['pastures'][general['pas_inc']]:
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



##############
#phases      #
##############
def f1_phases(mask_r=False, check=False):
    '''
    Rotation info.

    When using pinp rotations they are all included but if using full rotations then a mask is applied.
    The user can control the mask in SimInputs_{property}.xl.
    
    :param mask_r: If True the function returns the rot included mask.
    :param check: If True the function returns nothing - it is just used to check the correct rotations exist in rot.xl.
    '''
    ##rotation phases - read in from excel
    try:
        phases_r = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header=None, index_col=0, engine='openpyxl').T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc
    except FileNotFoundError:
        import RotGeneration
        RotGeneration.f_rot_gen(crop['user_crop_rot'])
        phases_r = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header=None, index_col=0, engine='openpyxl').T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc

    ###add variable that is the number of yrs in the rot phases
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
        directory_path = os.path.dirname(os.path.abspath(__file__))
        xl_path = os.path.join(directory_path, "SimInputs_{0}.xlsx".format(property))
        base_yields = pd.read_excel(xl_path, sheet_name='Yield', index_col=0, header=0, engine='openpyxl')
    ###if the rotations don't match inputs then rerun rotation generation.
    if len(phases_r) != len(base_yields) or any(base_yields.index!=phases_r.index):
        import RotGeneration
        RotGeneration.f_rot_gen(crop['user_crop_rot'])

        ###read in newly generated rotations and see if the inputs now match
        phases_r = pd.read_excel('Rotation.xlsx', sheet_name='rotation list', header=None, index_col=0, engine='openpyxl').T.reset_index(drop=True).T  #reset the col headers to std ie 0,1,2 etc
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

    ##if calling this function to check the rots then simply return nothing once the tests have been passed
    if check:
        return

    ##rotation mask - read in from excel
    if crop['user_crop_rot']:
        rot_mask_r = crop['i_user_rot_inc_r']
    else:
        ###build path this way so the file can be access even if AFO is run from another directory eg readthedocs or web app.
        property = general['i_property_id']
        directory_path = os.path.dirname(os.path.abspath(__file__))
        xl_path = os.path.join(directory_path, "SimInputs_{0}.xlsx".format(property))
        rot_mask_r = pd.read_excel(xl_path, sheet_name='RotMask', index_col=0, header=0, engine='openpyxl').squeeze().values

    ##if using the full list then apply the rot mask
    if mask_r:
        return rot_mask_r

    ##apply mask
    phases_r = phases_r.loc[rot_mask_r,:]

    return phases_r

