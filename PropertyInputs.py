"""

Inputs specific to a property (or region) including:

* Crop production
* Pasture production
* Labour
* Supplementary feeding
* Stubble
* Farm level finance ie overdraw level
* Farm level machinery ie soil adjustment factors for seeding efficiency

author: young
"""
##python modules
import pickle as pkl
import os.path
import numpy as np
import pandas as pd
import copy

##AFO modules
import Functions as fun
import StructuralInputs as sinp

na = np.newaxis

##############
#read inputs #
##############
##build path this way so that readthedocs can read correctly.
directory_path = os.path.dirname(os.path.abspath(__file__))
property_xl_path = os.path.join(directory_path, "Property.xlsx")
property_pkl_path = os.path.join(directory_path, "pkl_property.pkl")
try:
    if os.path.getmtime(property_xl_path) > os.path.getmtime(property_pkl_path):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print('Reading property inputs from pickle', end=' ', flush=True)
except FileNotFoundError:      
    inputs_from_pickle = False


##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading property inputs from Excel', end=' ', flush=True)
    with open(property_pkl_path, "wb") as f:
        general_inp = fun.xl_all_named_ranges(property_xl_path,"General", numpy=True)
        pkl.dump(general_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        rep_inp = fun.xl_all_named_ranges(property_xl_path,"Report Settings")
        pkl.dump(rep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        labour_inp = fun.xl_all_named_ranges(property_xl_path,"Labour")
        pkl.dump(labour_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        crop_inp = fun.xl_all_named_ranges(property_xl_path,"Crop")
        pkl.dump(crop_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        cropgraze_inp = fun.xl_all_named_ranges(property_xl_path,"CropGrazing", numpy=True)
        pkl.dump(cropgraze_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        mach_inp = fun.xl_all_named_ranges(property_xl_path,"Mach")
        pkl.dump(mach_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        stubble_inp = fun.xl_all_named_ranges(property_xl_path,"CropResidue", numpy=True)
        pkl.dump(stubble_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        finance_inp = fun.xl_all_named_ranges(property_xl_path,"Finance")
        pkl.dump(finance_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        period_inp = fun.xl_all_named_ranges(property_xl_path,"Periods", numpy=True) #automatically read in the periods as dates
        pkl.dump(period_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sup_inp = fun.xl_all_named_ranges(property_xl_path,"Sup Feed")
        pkl.dump(sup_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sheep_inp  = fun.xl_all_named_ranges(property_xl_path, 'Sheep', numpy=True)
        pkl.dump(sheep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        feedsupply_inp  = fun.xl_all_named_ranges(property_xl_path, 'FeedSupply', numpy=True)
        pkl.dump(feedsupply_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        mvf_inp  = fun.xl_all_named_ranges(property_xl_path, 'MVEnergy', numpy=True)
        pkl.dump(mvf_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        pasture_inp=dict()
        for pasture in sinp.general['pastures'][sinp.general['pastures_exist']]:
            pasture_inp[pasture] = fun.xl_all_named_ranges(property_xl_path, pasture, numpy=True)
        pkl.dump(pasture_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(property_pkl_path, "rb") as f:
        general_inp = pkl.load(f)

        rep_inp = pkl.load(f)

        labour_inp = pkl.load(f)

        crop_inp = pkl.load(f)

        cropgraze_inp = pkl.load(f)

        mach_inp = pkl.load(f)

        stubble_inp = pkl.load(f)

        finance_inp = pkl.load(f)

        period_inp = pkl.load(f)

        sup_inp = pkl.load(f)

        sheep_inp  = pkl.load(f)
        
        feedsupply_inp  = pkl.load(f)

        mvf_inp  = pkl.load(f)

        pasture_inp = pkl.load(f)

print('- finished')

##reshape required inputs
###lengths
len_d = len(sheep_inp['i_d_idx'])
len_h2 = sheep_inp['i_h2_len']
len_h5 = sheep_inp['i_h5_len']
len_h7 = sheep_inp['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
len_i = sheep_inp['i_i_len']
len_j2 = feedsupply_inp['i_j2_len']
len_k = len(sinp.landuse['C'])
len_k0 = sheep_inp['i_k0_len']
len_k1 = sheep_inp['i_k1_len']
len_k2 = sheep_inp['i_k2_len']
len_k3 = sheep_inp['i_k3_len']
len_k4 = sheep_inp['i_k4_len']
len_k5 = sheep_inp['i_k5_len']
len_l = len(general_inp['i_lmu_area'])
len_o = sheep_inp['i_o_len']
len_p6 = len(period_inp['i_fp_idx'])
len_r1 = feedsupply_inp['i_r1_len']
len_s = sheep_inp['i_s_len'] #s = shear
len_sc = sinp.stock['i_len_s'] #sc = scan
len_t3 = sheep_inp['i_t3_len']
len_x = sheep_inp['i_x_len']
len_z = len(general_inp['i_mask_z'])


###shapes
zp6 = (len_z, len_p6)
zkp6 = (len_z, len_k, len_p6)
zp6l = (len_z, len_p6, len_l)
zp6j0 = (len_z, len_p6, -1)
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
t3Sg = (len_t3, len_s+1, -1) #capital S to indicate this is special eg not normal because +1
r1j2P = (len_r1, len_j2, -1) #capital p to indicate this is just the remaining length of input, it is p axis but input is longer
r1p6z = (len_r1, len_p6, len_z)


###pasture
for t,pasture in enumerate(sinp.general['pastures'][sinp.general['pastures_exist']]):
    inp = pasture_inp[pasture]
    inp['DigRednSenesce'] = np.reshape(inp['DigRednSenesce'], zp6)
    inp['DigDryAve'] = np.reshape(inp['DigDryAve'], zp6)
    inp['DigDryRange'] = np.reshape(inp['DigDryRange'], zp6)
    inp['FOODryH'] = np.reshape(inp['FOODryH'], zp6)
    inp['LowFOO'] = np.reshape(inp['LowFOO'], zp6l)
    inp['MedFOO'] = np.reshape(inp['MedFOO'], zp6l)
    inp['LowPGR'] = np.reshape(inp['LowPGR'], zp6l)
    inp['MedPGR'] = np.reshape(inp['MedPGR'], zp6l)
    inp['DigGrn'] = np.reshape(inp['DigGrn'], zp6l)

###crop grazing
cropgraze_inp['i_crop_growth_zkp6'] = np.reshape(cropgraze_inp['i_crop_growth_zkp6'], zkp6)

###stock
sheep_inp['i_pasture_stage_p6z'] = np.reshape(sheep_inp['i_pasture_stage_p6z'], zp6)
sheep_inp['i_legume_p6z'] = np.reshape(sheep_inp['i_legume_p6z'], zp6)
sheep_inp['i_supplement_zp6'] = np.reshape(sheep_inp['i_supplement_zp6'], zp6)
sheep_inp['i_paststd_foo_zp6j0'] = np.reshape(sheep_inp['i_paststd_foo_zp6j0'], zp6j0)
sheep_inp['i_paststd_dmd_zp6j0'] = np.reshape(sheep_inp['i_paststd_dmd_zp6j0'], zp6j0)
sheep_inp['i_density_p6z'] = np.reshape(sheep_inp['i_density_p6z'], zp6)
sheep_inp['i_husb_operations_triggerlevels_h5h7h2'] = np.reshape(sheep_inp['i_husb_operations_triggerlevels_h5h7h2'], h2h5h7)
sheep_inp['i_date_born1st_oig2'] = np.reshape(sheep_inp['i_date_born1st_oig2'], iog)
sheep_inp['i_date_born1st_idg3'] = np.reshape(sheep_inp['i_date_born1st_idg3'], idg)
sheep_inp['i_sire_propn_oig1'] = np.reshape(sheep_inp['i_sire_propn_oig1'], iog)
sheep_inp['i_date_shear_sixg0'] = np.reshape(sheep_inp['i_date_shear_sixg0'], isxg)
sheep_inp['i_date_shear_sixg1'] = np.reshape(sheep_inp['i_date_shear_sixg1'], isxg)
sheep_inp['i_date_shear_sixg3'] = np.reshape(sheep_inp['i_date_shear_sixg3'], isxg)
sheep_inp['ia_r1_zig0'] = np.reshape(sheep_inp['ia_r1_zig0'], izg)
sheep_inp['ia_r1_zig1'] = np.reshape(sheep_inp['ia_r1_zig1'], izg)
sheep_inp['ia_r1_zig3'] = np.reshape(sheep_inp['ia_r1_zig3'], izg)
sheep_inp['ia_r2_k0ig1'] = np.reshape(sheep_inp['ia_r2_k0ig1'], ik0g)
sheep_inp['ia_r2_k1ig1'] = np.reshape(sheep_inp['ia_r2_k1ig1'], ik1g)
sheep_inp['ia_r2_sk2ig1'] = np.reshape(sheep_inp['ia_r2_sk2ig1'], isk2g)
sheep_inp['ia_r2_ik0g3'] = np.reshape(sheep_inp['ia_r2_ik0g3'], ik0g)
sheep_inp['ia_r2_ik3g3'] = np.reshape(sheep_inp['ia_r2_ik3g3'], ik3g)
sheep_inp['ia_r2_ik4g3'] = np.reshape(sheep_inp['ia_r2_ik4g3'], ik4g)
sheep_inp['ia_r2_ik5g3'] = np.reshape(sheep_inp['ia_r2_ik5g3'], ik5g)
sheep_inp['i_sales_offset_tsg3'] = np.reshape(sheep_inp['i_sales_offset_tsg3'], t3Sg)
sheep_inp['i_target_weight_tsg3'] = np.reshape(sheep_inp['i_target_weight_tsg3'], t3Sg)
sheep_inp['i_shear_prior_tsg3'] = np.reshape(sheep_inp['i_shear_prior_tsg3'], t3Sg)
sheep_inp['ia_i_idg2'] = np.reshape(sheep_inp['ia_i_idg2'], idg)
feedsupply_inp['i_feedsupply_options_r1j2p'] = np.reshape(feedsupply_inp['i_feedsupply_options_r1j2p'], r1j2P)
feedsupply_inp['i_confinement_options_r1p6z'] = np.reshape(feedsupply_inp['i_confinement_options_r1p6z'], r1p6z)

##create a copy of each input dict - so that the base inputs remain unchanged
##the copy created is the one used in the actual modules
###NOTE: if an input sheet is added remember to add it to the dict reset in f_sa() below.
general = copy.deepcopy(general_inp)
rep = copy.deepcopy(rep_inp)
labour = copy.deepcopy(labour_inp)
crop = copy.deepcopy(crop_inp)
cropgraze = copy.deepcopy(cropgraze_inp)
mach = copy.deepcopy(mach_inp)
stubble = copy.deepcopy(stubble_inp)
finance = copy.deepcopy(finance_inp)
period = copy.deepcopy(period_inp)
supfeed = copy.deepcopy(sup_inp)
sheep = copy.deepcopy(sheep_inp)
feedsupply = copy.deepcopy(feedsupply_inp)
mvf = copy.deepcopy(mvf_inp)
pasture_inputs = copy.deepcopy(pasture_inp)


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

    ##reset inputs to base at the start of each trial before applying SA - old method was to update the SA based on the _inp dict but that doesnt work well when multiple SA on the same variable.
    fun.f_dict_reset(general, general_inp)
    fun.f_dict_reset(rep, rep_inp)
    fun.f_dict_reset(labour, labour_inp)
    fun.f_dict_reset(crop, crop_inp)
    fun.f_dict_reset(cropgraze, cropgraze_inp)
    fun.f_dict_reset(mach, mach_inp)
    fun.f_dict_reset(stubble, stubble_inp)
    fun.f_dict_reset(finance, finance_inp)
    fun.f_dict_reset(period, period_inp)
    fun.f_dict_reset(supfeed, sup_inp)
    fun.f_dict_reset(sheep, sheep_inp)
    fun.f_dict_reset(feedsupply, feedsupply_inp)
    fun.f_dict_reset(mvf, mvf_inp)
    fun.f_dict_reset(pasture_inputs, pasture_inp)

    ##general
    ###sav
    general['steady_state'] = fun.f_sa(general['steady_state'], sen.sav['steady_state'], 5)
    general['i_inc_node_periods'] = fun.f_sa(general['i_inc_node_periods'], sen.sav['inc_node_periods'], 5)

    ###sam
    ###sap
    ###saa
    ###sat
    ###sar

    ##pasture
    ###sav
    general['pas_inc'] = fun.f_sa(general['pas_inc'], sen.sav['pas_inc'], 5)

    for pasture in sinp.general['pastures'][general['pas_inc']]: #all pasture inputs are adjusted even if a given pasture is not included
        ###sav
        ###SAM
        pasture_inputs[pasture]['GermStd'] = fun.f_sa(pasture_inputs[pasture]['GermStd'], sen.sam[('germ',pasture)])
        pasture_inputs[pasture]['GermScalarLMU'] = fun.f_sa(pasture_inputs[pasture]['GermScalarLMU'], sen.sam[('germ_l',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_f',pasture)][...,na])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_f',pasture)][...,na])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['DigDryAve'] = (pasture_inputs[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)]
                                                + np.max(pasture_inputs[pasture]['DigDryAve'],axis=1) * (1 - sen.sam[('dry_dmd_decline',pasture)]))
        pasture_inputs[pasture]['DigSpread'] = fun.f_sa(pasture_inputs[pasture]['DigSpread'], sen.sam[('grn_dmd_range_f',pasture)])
#        pasture_inputs[pasture]['DigDeclineFOO'] = fun.f_sa(pasture_inputs[pasture]['DigDeclineFOO'], sen.sam[('grn_dmd_declinefoo_f',pasture)])
        pasture_inputs[pasture]['DigRednSenesce'] = fun.f_sa(pasture_inputs[pasture]['DigRednSenesce'], sen.sam[('grn_dmd_senesce_f',pasture)])

        ###sap
        ###saa
        ###sat
        ###sar

    ##sheep
    ###SAV
    sheep['i_mask_i'] = fun.f_sa(sheep['i_mask_i'], sen.sav['TOL_inc'], 5)
    sheep['i_g3_inc'] = fun.f_sa(sheep['i_g3_inc'], sen.sav['g3_included'],5)
    sheep['a_c2_c0'] = fun.f_sa(sheep['a_c2_c0'], sen.sav['genotype'],5)
    sheep['i_scan_og1'] = fun.f_sa(sheep['i_scan_og1'], sen.sav['scan_og1'],5)
    sheep['i_dry_sales_forced_o'] = fun.f_sa(sheep['i_dry_sales_forced_o'], sen.sav['bnd_drys_sold_o'],5)
    sheep['i_dry_retained_forced'] = fun.f_sa(sheep['i_dry_retained_forced'], sen.sav['bnd_drys_retained_o'],5)
    sheep['ia_r1_zig1'] = fun.f_sa(sheep['ia_r1_zig1'], sen.sav['r1_izg1'],5)
    sheep['ia_r2_sk2ig1'] = fun.f_sa(sheep['ia_r2_sk2ig1'], sen.sav['r2_isk2g1'],5)
    sheep['ia_r1_zig3'] = fun.f_sa(sheep['ia_r1_zig3'], sen.sav['r1_izg3'],5)
    sheep['i_sr_constraint_t'] = fun.f_sa(sheep['i_sr_constraint_t'], sen.sav['bnd_sr_t'],5)

    ###sam
    ###sap
    ###saa
    sheep['ia_r1_zig1'] = fun.f_sa(sheep['ia_r1_zig1'], sen.saa['r1_izg1'], 2).astype('int')
    sheep['ia_r2_sk2ig1'] = fun.f_sa(sheep['ia_r2_sk2ig1'], sen.saa['r2_isk2g1'], 2).astype('int')
    sheep['ia_r1_zig3'] = fun.f_sa(sheep['ia_r1_zig3'], sen.saa['r1_izg3'], 2).astype('int')
    sheep['ia_r2_ik5g3'] = fun.f_sa(sheep['ia_r2_ik5g3'], sen.saa['r2_ik5g3'], 2).astype('int')
    feedsupply['i_feedsupply_options_r1j2p'] = fun.f_sa(feedsupply['i_feedsupply_options_r1j2p'], sen.saa['feedsupply_r1jp'], 2)
    feedsupply['i_feedsupply_adj_options_r2p'] = fun.f_sa(feedsupply['i_feedsupply_adj_options_r2p'], sen.saa['feedsupply_adj_r2p'], 2)
    ###sat
    ###sar

    ##report controls
    ###SAV
    rep['i_store_nv_rep'] = fun.f_sa(rep['i_store_nv_rep'], sen.sav['nv_inc'], 5)
    rep['i_store_lw_rep'] = fun.f_sa(rep['i_store_lw_rep'], sen.sav['lw_inc'], 5)
    rep['i_store_ffcfw_rep'] = fun.f_sa(rep['i_store_ffcfw_rep'], sen.sav['ffcfw_inc'], 5)
    rep['i_store_on_hand_mort'] = fun.f_sa(rep['i_store_on_hand_mort'], sen.sav['onhand_mort_p_inc'], 5)
    rep['i_store_mort'] = fun.f_sa(rep['i_store_mort'], sen.sav['mort_inc'], 5)

    ##mask out unrequired nodes dates - nodes are removed if there are double ups or if a season is not identified at the node (and node is not used as fvp)
    ## has to be here because if affects two inputs so cant put it in f_season_periods.
    ###test for duplicate
    duplicate_mask_m = []
    for m in range(general['i_date_node_zm'].shape[1]):  # maybe there is a way to do this without a loop.
        duplicate_mask_m.append(np.all(np.any(general['i_date_node_zm'][:,m:m+1] == general['i_date_node_zm'][:,0:m],axis=1,keepdims=True)))
    duplicate_mask_m = np.logical_not(duplicate_mask_m)
    ###test if any season is identified at the node
    import SeasonalFunctions as zfun #have to import here since zfun imports pinp.
    mask_zm = np.logical_or(general['i_date_initiate_z'][:,na]==general['i_date_node_zm'], general['i_node_is_fvp'])
    mask_zm = zfun.f_seasonal_inp(mask_zm,numpy=True,axis=0)
    mask_m = np.any(mask_zm, axis=0)
    mask_m = np.logical_and(duplicate_mask_m, mask_m)
    general['i_date_node_zm'] = general['i_date_node_zm'][:,mask_m]
    general['i_node_is_fvp'] = general['i_node_is_fvp'][mask_m]

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
    sheep['i_mobsize_sire_p6zi'] = np.take_along_axis(sheep['i_mobsize_sire_p6i'][:,na,:], a_p6std_p6z[:,:,na], axis=0)
    sheep['i_mobsize_dams_p6zi'] = np.take_along_axis(sheep['i_mobsize_dams_p6i'][:,na,:], a_p6std_p6z[:,:,na], axis=0)
    sheep['i_mobsize_offs_p6zi'] = np.take_along_axis(sheep['i_mobsize_offs_p6i'][:,na,:], a_p6std_p6z[:,:,na], axis=0)
    sheep['i_dse_group'] = np.take_along_axis(sheep['i_dse_group'][:,:,na], a_p6std_p6z[na,:,:], axis=1)
    sheep['i_wg_propn_p6z'] = np.take_along_axis(sheep['i_wg_propn_p6'][:,na], a_p6std_p6z, axis=0)

    ####crop grazing
    cropgraze['i_cropgraze_yield_reduction_kp6z'] = np.take_along_axis(cropgraze['i_cropgraze_yield_reduction_kp6'][...,na], a_p6std_p6z[na,...], axis=1)
    cropgraze['i_crop_dmd_kp6z'] = np.take_along_axis(cropgraze['i_crop_dmd_kp6'][...,na], a_p6std_p6z[na,...], axis=1)
    cropgraze['i_crop_growth_zkp6'] = np.take_along_axis(cropgraze['i_crop_growth_zkp6'], a_p6std_zp6[:,na,:], axis=2)
    cropgraze['i_cropgraze_consumption_factor_zp6'] = np.take_along_axis(cropgraze['i_cropgraze_consumption_factor_zp6'], a_p6std_zp6, axis=1)


    ####pasture
    for pasture in sinp.general['pastures'][general['pas_inc']]:
        pasture_inputs[pasture]['POCCons'] = np.take_along_axis(pasture_inputs[pasture]['POCCons'][:,:,na], a_p6std_p6z[:,na,:], axis=0)
        pasture_inputs[pasture]['DigRednSenesce'] = np.take_along_axis(pasture_inputs[pasture]['DigRednSenesce'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['DigDryAve'] = np.take_along_axis(pasture_inputs[pasture]['DigDryAve'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['DigDryRange'] = np.take_along_axis(pasture_inputs[pasture]['DigDryRange'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['FOODryH'] = np.take_along_axis(pasture_inputs[pasture]['FOODryH'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['GermScalarFP'] = np.take_along_axis(pasture_inputs[pasture]['GermScalarFP'], a_p6std_zp6, axis=1)
        pasture_inputs[pasture]['CPGrn'] = np.take_along_axis(pasture_inputs[pasture]['CPGrn'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['CPDry'] = np.take_along_axis(pasture_inputs[pasture]['CPDry'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['DigPOC'] = np.take_along_axis(pasture_inputs[pasture]['DigPOC'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['FOOPOC'] = np.take_along_axis(pasture_inputs[pasture]['FOOPOC'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['DigSpread'] = np.take_along_axis(pasture_inputs[pasture]['DigSpread'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['PGRScalarH'] = np.take_along_axis(pasture_inputs[pasture]['PGRScalarH'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['BaseLevelInput'] = np.take_along_axis(pasture_inputs[pasture]['BaseLevelInput'][:,na], a_p6std_p6z, axis=0)
        pasture_inputs[pasture]['ErosionLimit'] = np.take_along_axis(pasture_inputs[pasture]['ErosionLimit'][:,:,na], a_p6std_p6z[:,na,:], axis=0)
        pasture_inputs[pasture]['SenescePropn'] = np.take_along_axis(pasture_inputs[pasture]['SenescePropn'][:,na], a_p6std_p6z, axis=0)
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
                                                  - (np.count_nonzero(a_p6std_p6z == pasture_inputs[pasture]['i_dry_exists'], axis=0) -1) #have to minus count non zero in case an extra fp is added in the period dry pas become available. because in this case we still want to point at the first period it become available (opposite to end of gs)

    ###fp index needs special handling because it isn't just expanded it is rebuilt
    period['i_fp_idx'] = ['fp%02d'%i for i in range(len(a_p6std_p6z))]




