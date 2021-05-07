# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:03:30 2020

@author: young
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
try:
    if os.path.getmtime("Property.xlsx") > os.path.getmtime("pkl_property.pkl"):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print('Reading property inputs from pickle', end=' ', flush=True)
except FileNotFoundError:      
    inputs_from_pickle = False


filename= 'pkl_property.pkl'
##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    print('Reading property inputs from Excel', end=' ', flush=True)
    with open(filename, "wb") as f:
        general_inp = fun.xl_all_named_ranges("Property.xlsx","General")
        pkl.dump(general_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        rep_inp = fun.xl_all_named_ranges("Property.xlsx","Report Settings")
        pkl.dump(rep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        labour_inp = fun.xl_all_named_ranges("Property.xlsx","Labour")
        pkl.dump(labour_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        crop_inp = fun.xl_all_named_ranges("Property.xlsx","Crop")
        pkl.dump(crop_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        mach_inp = fun.xl_all_named_ranges("Property.xlsx","Mach")
        pkl.dump(mach_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        stubble_inp = fun.xl_all_named_ranges("Property.xlsx","Stubble", numpy=True)
        pkl.dump(stubble_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        finance_inp = fun.xl_all_named_ranges("Property.xlsx","Finance")
        pkl.dump(finance_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        period_inp = fun.xl_all_named_ranges("Property.xlsx","Periods", numpy=True) #automatically read in the periods as dates
        pkl.dump(period_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sup_inp = fun.xl_all_named_ranges("Property.xlsx","Sup Feed")
        pkl.dump(sup_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sheep_inp  = fun.xl_all_named_ranges('Property.xlsx', 'Sheep', numpy=True)
        pkl.dump(sheep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        feedsupply_inp  = fun.xl_all_named_ranges('Property.xlsx', 'FeedSupply', numpy=True)
        pkl.dump(feedsupply_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        pasture_inp=dict()
        for pasture in sinp.general['pastures'][sinp.general['pastures_exist']]:
            pasture_inp[pasture] = fun.xl_all_named_ranges('Property.xlsx', pasture, numpy=True)
        pkl.dump(pasture_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(filename, "rb") as f:
        general_inp = pkl.load(f)

        rep_inp = pkl.load(f)

        labour_inp = pkl.load(f)

        crop_inp = pkl.load(f)

        mach_inp = pkl.load(f)

        stubble_inp = pkl.load(f)

        finance_inp = pkl.load(f)

        period_inp = pkl.load(f)

        sup_inp = pkl.load(f)

        sheep_inp  = pkl.load(f)
        
        feedsupply_inp  = pkl.load(f)

        pasture_inp = pkl.load(f)

print('- finished')
##reshape required inputs
###lengths
len_d = len(sheep_inp['i_d_idx'])
len_h2 = sheep_inp['i_h2_len']
len_h5 = sheep_inp['i_h5_len']
len_h7 = sheep_inp['i_husb_operations_triggerlevels_h5h7h2'].shape[-1]
len_i = sheep_inp['i_i_len']
len_j0 = feedsupply_inp['i_j0_len']
len_k0 = sheep_inp['i_k0_len']
len_k1 = sheep_inp['i_k1_len']
len_k2 = sheep_inp['i_k2_len']
len_k3 = sheep_inp['i_k3_len']
len_k4 = sheep_inp['i_k4_len']
len_k5 = sheep_inp['i_k5_len']
len_l = len(general_inp['lmu_area'])
len_o = sheep_inp['i_o_len']
len_p6 = len(period_inp['i_fp_idx'])
len_r1 = feedsupply_inp['i_r1_len']
len_s = sheep_inp['i_s_len']
len_t3 = sheep_inp['i_t3_len']
len_x = sheep_inp['i_x_len']
len_z = len(general_inp['i_mask_z'])


###shapes
zp6 = (len_z, len_p6)
zp6l = (len_z, len_p6, len_l)
zp6j0 = (len_z, len_p6, len_j0)
h2h5h7 = (len_h2, len_h5, len_h7)
iog = (len_i, len_o, -1)
idg = (len_i, len_d, -1)
isxg = (len_i, len_s, len_x, -1)
izg = (len_i, len_z, -1)
ik0g = (len_i, len_k0, -1)
ik1g = (len_i, len_k1, -1)
ik2g = (len_i, len_k2, -1)
ik3g = (len_i, len_k3, -1)
ik4g = (len_i, len_k4, -1)
ik5g = (len_i, len_k5, -1)
t3Sg = (len_t3, len_s+1, -1) #capital S to indicate this is special eg not normal because +1
r1j0P = (len_r1, len_j0, -1) #capital p to indicate this is just the remaining length of input, it is p axis but input is longer


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

###stock
sheep_inp['i_pasture_stage_p6z'] = np.reshape(sheep_inp['i_pasture_stage_p6z'], zp6)
sheep_inp['i_legume_p6z'] = np.reshape(sheep_inp['i_legume_p6z'], zp6)
sheep_inp['i_paststd_foo_p6zj0'] = np.reshape(sheep_inp['i_paststd_foo_p6zj0'], zp6j0)
sheep_inp['i_paststd_dmd_p6zj0'] = np.reshape(sheep_inp['i_paststd_dmd_p6zj0'], zp6j0)
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
sheep_inp['ia_r2_k2ig1'] = np.reshape(sheep_inp['ia_r2_k2ig1'], ik2g)
sheep_inp['ia_r2_ik0g3'] = np.reshape(sheep_inp['ia_r2_ik0g3'], ik0g)
sheep_inp['ia_r2_ik3g3'] = np.reshape(sheep_inp['ia_r2_ik3g3'], ik3g)
sheep_inp['ia_r2_ik4g3'] = np.reshape(sheep_inp['ia_r2_ik4g3'], ik4g)
sheep_inp['ia_r2_ik5g3'] = np.reshape(sheep_inp['ia_r2_ik5g3'], ik5g)
sheep_inp['i_sales_offset_tsg3'] = np.reshape(sheep_inp['i_sales_offset_tsg3'], t3Sg)
sheep_inp['i_target_weight_tsg3'] = np.reshape(sheep_inp['i_target_weight_tsg3'], t3Sg)
sheep_inp['i_shear_prior_tsg3'] = np.reshape(sheep_inp['i_shear_prior_tsg3'], t3Sg)
sheep_inp['ia_i_idg2'] = np.reshape(sheep_inp['ia_i_idg2'], idg)
feedsupply_inp['i_feedoptions_r1pj0'] = np.reshape(feedsupply_inp['i_feedoptions_r1pj0'], r1j0P)




##create a copy of each input dict - this means there is always a copy of the original inputs (the second copy has SA applied to it)
##the copy created is the one used in the actual modules
general=copy.deepcopy(general_inp)
rep=copy.deepcopy(rep_inp)
labour= copy.deepcopy(labour_inp)
crop= copy.deepcopy(crop_inp)
mach= copy.deepcopy(mach_inp)
stubble= copy.deepcopy(stubble_inp)
finance= copy.deepcopy(finance_inp)
period= copy.deepcopy(period_inp)
supfeed= copy.deepcopy(sup_inp)
sheep= copy.deepcopy(sheep_inp)
feedsupply=copy.deepcopy(feedsupply_inp)
pasture_inputs=copy.deepcopy(pasture_inp)

#######################
#apply SA             #
#######################
def property_inp_sa():
    '''

    Returns
    -------
    None.

    Applies sensitivity adjustment to each input.
    This function gets called at the beginning of each loop in the exp.py module

    '''
    ##have to import it here since sen.py imports this module
    import Sensitivity as sen
    ##general
    ###sav
    general['steady_state'] = fun.f_sa(general_inp['steady_state'], sen.sav['steady_state'], 5)


    ##pasture
    ###sav
    general['pas_inc'] = fun.f_sa(general_inp['pas_inc'], sen.sav['pas_inc'], 5)

    for pasture in sinp.general['pastures'][general['pas_inc']]: #all pasture inputs are adjusted even if a given pasture is not included
        ###SAM
        pasture_inputs[pasture]['GermStd'] = fun.f_sa(pasture_inp[pasture]['GermStd'], sen.sam[('germ',pasture)])
        pasture_inputs[pasture]['GermScalarLMU'] = fun.f_sa(pasture_inp[pasture]['GermScalarLMU'], sen.sam[('germ_l',pasture)])
        ##Do the PGR sensitivity on LowPGR & MedPGR sequentially building on the previous value. Alter code if lines are deleted
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inp[pasture]['LowPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_f',pasture)][...,na])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inputs[pasture]['LowPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inp[pasture]['MedPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_f',pasture)][...,na])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inputs[pasture]['MedPGR'], sen.sam[('pgr_l',pasture)])
        pasture_inputs[pasture]['DigDryAve'] = (pasture_inp[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)]
                                                + np.max(pasture_inp[pasture]['DigDryAve'],axis=1) * (1 - sen.sam[('dry_dmd_decline',pasture)]))
        pasture_inputs[pasture]['DigSpread'] = fun.f_sa(pasture_inp[pasture]['DigSpread'], sen.sam[('grn_dmd_range_f',pasture)])
        pasture_inputs[pasture]['DigDeclineFOO'] = fun.f_sa(pasture_inp[pasture]['DigDeclineFOO'], sen.sam[('grn_dmd_declinefoo_f',pasture)])
        pasture_inputs[pasture]['DigRednSenesce'] = fun.f_sa(pasture_inp[pasture]['DigRednSenesce'], sen.sam[('grn_dmd_senesce_f',pasture)])

    ##sheep
    ###SAV
    sheep['i_mask_i'] = fun.f_sa(sheep_inp['i_mask_i'], sen.sav['TOL_inc'], 5)
    sheep['i_g3_inc'] = fun.f_sa(sheep_inp['i_g3_inc'], sen.sav['g3_included'],5)
    sheep['a_c2_c0'] = fun.f_sa(sheep_inp['a_c2_c0'], sen.sav['genotype'],5)
    sheep['i_scan_og1'] = fun.f_sa(sheep_inp['i_scan_og1'], sen.sav['scan_og1'],5)
    sheep['i_dry_sales_forced'] = fun.f_sa(sheep_inp['i_dry_sales_forced'], sen.sav['bnd_drys_sold'],5)
    sheep['i_dry_retained_forced'] = fun.f_sa(sheep_inp['i_dry_retained_forced'], sen.sav['bnd_drys_retained'],5)
    sheep['ia_r1_zig1'] = fun.f_sa(sheep_inp['ia_r1_zig1'], sen.sav['r1_izg1'],5)
    sheep['ia_r1_zig3'] = fun.f_sa(sheep_inp['ia_r1_zig3'], sen.sav['r1_izg3'],5)
    sheep['i_sr_constraint_t'] = fun.f_sa(sheep_inp['i_sr_constraint_t'], sen.sav['bnd_sr_t'],5)

    ##report controls
    ###SAV
    rep['i_store_fec_rep'] = fun.f_sa(rep_inp['i_store_fec_rep'], sen.sav['fec_inc'], 5)
    rep['i_store_lw_rep'] = fun.f_sa(rep_inp['i_store_lw_rep'], sen.sav['lw_inc'], 5)
    rep['i_store_ffcfw_rep'] = fun.f_sa(rep_inp['i_store_ffcfw_rep'], sen.sav['ffcfw_inc'], 5)
    rep['i_store_on_hand'] = fun.f_sa(rep_inp['i_store_on_hand'], sen.sav['onhand_p_inc'], 5)
    rep['i_store_mort'] = fun.f_sa(rep_inp['i_store_mort'], sen.sav['mort_inc'], 5)


def f_z_prob():
    '''calc the prob of each active season'''
    ##season mask - controls which seasons are included
    z_mask = general['i_mask_z']
    ##adjust season prob accounting for the seasons which are not included
    z_prob = np.array(general['i_season_propn_z'])
    z_prob = z_prob[z_mask]
    z_prob = z_prob / sum(z_prob)
    return z_prob


def f_seasonal_inp(inp, numpy=False, axis=0, level=0):
    '''
    This function adjust treats the seasonal axis.
    Either returning inputs for the static model or DSP.

    For numpy z axis can be any but for pandas z must be column (if you want to change this the function will need mods).

    For the static model the inputs which are seasonally effected can be generated by 2 methods:
        1. Take the weighted average of the inputs for different seasons
        2. Take the first slice of the z axis which is the user defined 'typical' season.
    :param axis: the axis number where z is
    :param level: for pandas which level is season (if multi index)
    :return:
    '''
    ##season mask - controls which seasons are included
    z_mask = general['i_mask_z']
    ##calc season prob accounting for the seasons which are not included
    z_prob = f_z_prob()

    if numpy:
        ##mask the season types
        inp = np.compress(z_mask, inp, axis)

        ##weighted average if steady state
        if general['steady_state']:
            try:  # in case array is datearray
                inp = np.expand_dims(np.average(inp, axis=axis, weights=z_prob), axis)
            except TypeError:
                n_inp = inp.astype("datetime64[ns]").astype(np.int64)
                n_inp = np.expand_dims(np.average(n_inp, axis=axis, weights=z_prob), axis)
                n_inp = n_inp.astype("datetime64[ns]")
                inp = n_inp.astype('M8[us]').astype('O') #converts to datetime

    else:
        ##mask the season types
        keys_z = general['i_z_idx'][z_mask]
        if inp.columns.nlevels > 2: #if statement required because can't convert one element to tuple
            slc_none = tuple([slice(None)] * (inp.columns.nlevels - 1)) #makes a slice(none) for each column level except season.
            inp = inp.loc[:, (keys_z, slc_none)]
        elif inp.columns.nlevels > 1:
            slc_none = slice(None)
            inp = inp.loc[:, (keys_z, slc_none)]
        else:
            inp = inp.loc[:,z_mask]

        ##weighted average if steady state
        if general['steady_state']:
            try: #in case df is datearray
                z_prob = pd.Series(z_prob, index=keys_z)
                if axis==0:
                    sum_level = list(range(inp.index.nlevels))
                else:
                    sum_level = list(range(inp.columns.nlevels))

                del sum_level[level]
                if sum_level == []:
                    inp = inp.mul(z_prob, axis=axis, level=level).sum(axis=axis)
                    inp = pd.concat([inp],keys=[keys_z[0]],axis=axis) #add z0 index key
                else:
                    inp = inp.mul(z_prob, axis=axis, level=level).sum(axis=axis, level=sum_level)
                    inp = pd.concat([inp],keys=[keys_z[0]],axis=axis) #add z0 index key
                    col_level_order = sum_level[:]
                    col_level_order.insert(level,0)
                    inp = inp.reorder_levels(col_level_order, axis=axis)
            except TypeError:
                #this won't work if columns have two levels (would need to reshape into multi d numpy do the average then reshape to 2-d)
                n_inp = inp.values.astype(np.int64)
                n_inp = np.average(n_inp, axis=axis, weights=z_prob)
                n_inp = n_inp.astype("datetime64[ns]")
                n_inp = n_inp.astype('M8[us]').astype('O') #converts to datetime
                col = pd.MultiIndex.from_tuples([inp.columns[0]])
                inp = pd.DataFrame(n_inp, index=inp.index, columns=col)
    return inp

def f_keys_z():
    '''returns the index/keys for z axis'''
    if general['steady_state']:
        keys_z = np.array([general['i_z_idx'][general['i_mask_z']][0]]).astype('str')
    else:
        keys_z = general['i_z_idx'][general['i_mask_z']].astype('str')
    return keys_z

