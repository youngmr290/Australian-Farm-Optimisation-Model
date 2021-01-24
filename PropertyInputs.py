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

##Midas modules
import Functions as fun
import UniversalInputs as uinp

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

        stubble_inp = fun.xl_all_named_ranges("Property.xlsx","Stubble")
        pkl.dump(stubble_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        finance_inp = fun.xl_all_named_ranges("Property.xlsx","Finance")
        pkl.dump(finance_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        period_inp = fun.xl_all_named_ranges("Property.xlsx","Periods") #automatically read in the periods as dates
        pkl.dump(period_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sup_inp = fun.xl_all_named_ranges("Property.xlsx","Sup Feed") #automatically read in the periods as dates
        pkl.dump(sup_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        sheep_inp  = fun.xl_all_named_ranges('Property.xlsx', 'Sheep', numpy=True)
        pkl.dump(sheep_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        feedsupply_inp  = fun.xl_all_named_ranges('Property.xlsx', 'FeedSupply', numpy=True) #^think this will get combined with the input sheet above
        pkl.dump(feedsupply_inp, f, protocol=pkl.HIGHEST_PROTOCOL)

        pasture_inp=dict()
        for pasture in uinp.structure['pastures'][uinp.structure['pastures_exist']]:#all pasture inputs are adjusted even if a given pasture is not included
            pasture_inp[pasture] = fun.xl_all_named_ranges('Property.xlsx', pasture)
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

##create a copy of each input dict - this means there is always a copy of the original inputs (the second copy has SA applied to it)
##the copy created is the one used in the actual modules
general=general_inp.copy()
rep=rep_inp.copy()
labour=labour_inp.copy()
crop=crop_inp.copy()
mach=mach_inp.copy()
stubble=stubble_inp.copy()
finance=finance_inp.copy()
period=period_inp.copy()
supfeed=sup_inp.copy()
sheep=sheep_inp.copy()
feedsupply=feedsupply_inp.copy()
# sheep_management=sheep_management_inp.copy()
# sheep_regions=sheep_regions_inp.copy()
pasture_inputs=pasture_inp.copy()

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
    ##pasture
    general['pas_inc'] = fun.f_sa(general_inp['pas_inc'],sen.sav['pas_inc'],5)
    for pasture in uinp.structure['pastures'][general['pas_inc']]: #all pasture inputs are adjusted even if a given pasture is not included
        ###SAM
        pasture_inputs[pasture]['GermStd'] = fun.f_sa(pasture_inp[pasture]['GermStd'], sen.sam[('germ',pasture)])
        pasture_inputs[pasture]['GermScalarLMU'] = fun.f_sa(pasture_inp[pasture]['GermScalarLMU'], sen.sam[('germ_l',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inp[pasture]['LowPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inp[pasture]['MedPGR'], sen.sam[('pgr',pasture)])
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inp[pasture]['LowPGR'], sen.sam[('pgr_f',pasture)], pandas=True, axis=0)
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inp[pasture]['MedPGR'], sen.sam[('pgr_f',pasture)], pandas=True, axis=0)
        pasture_inputs[pasture]['LowPGR'] = fun.f_sa(pasture_inp[pasture]['LowPGR'], sen.sam[('pgr_l',pasture)], pandas=True, axis=1)
        pasture_inputs[pasture]['MedPGR'] = fun.f_sa(pasture_inp[pasture]['MedPGR'], sen.sam[('pgr_l',pasture)], pandas=True, axis=1)
        pasture_inputs[pasture]['DigDryAve'] = pasture_inp[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)] \
                                                + max(pasture_inp[pasture]['DigDryAve']) * (1 - sen.sam[('dry_dmd_decline',pasture)])
        pasture_inputs[pasture]['DigSpread'] = fun.f_sa(pasture_inp[pasture]['DigSpread'], sen.sam[('grn_dmd_range_f',pasture)])
        pasture_inputs[pasture]['DigDeclineFOO'] = fun.f_sa(pasture_inp[pasture]['DigDeclineFOO'], sen.sam[('grn_dmd_declinefoo_f',pasture)])
        pasture_inputs[pasture]['DigRednSenesce'] = fun.f_sa(pasture_inp[pasture]['DigRednSenesce'], sen.sam[('grn_dmd_senesce_f',pasture)])
    ##sheep
    ###SAV
    sheep['i_mask_i'] = fun.f_sa(sheep_inp['i_mask_i'], sen.sav['TOL_inc'], 5)
    sheep['i_g3_inc'] = fun.f_sa(sheep_inp['i_g3_inc'], sen.sav['g3_included'],5)
    sheep['i_scan_og1'] = fun.f_sa(sheep_inp['i_scan_og1'], sen.sav['scan_og1'],5)
    sheep['i_sai_lw_dams_owi'] = fun.f_sa(sheep_inp['i_sai_lw_dams_owi'], sen.sav['nut_mask_dams'],5)
    sheep['i_sai_lw_offs_swix'] = fun.f_sa(sheep_inp['i_sai_lw_offs_swix'], sen.sav['nut_mask_offs'],5)

    ##report controls
    ###SAV
    rep['i_store_fec_rep'] = fun.f_sa(rep_inp['i_store_fec_rep'], sen.sav['fec_inc'], 5)


def f_seasonal_inp(inp, numpy=False, axis=0):
    '''
    This function adjust the seasonal inputs.
    Either returning inputs for the static model or DSP.
    For the static model the inputs which are seasnally effected can be generated by 2 methods:
        1. Take the weighted average of the inputs for differnt seasons
        2. Take the fisrt slice of the z axis which is the user defined 'typical' season.
    :return:
    '''
    ##season mask - contros which seasons are included
    z_mask = general['i_mask_z']
    ##adjust season prob accounting for the seasons which are not included
    z_prob = np.array(general['i_season_propn_z'])
    z_prob = z_prob[z_mask]
    z_prob = z_prob / sum(z_prob)

    if numpy:
        ##mask the season types
        inp = np.compress(z_mask, inp, axis)

        ##weighted average if steady state
        if general['steady_state']:
            inp = np.average(inp, axis=axis, weights=z_prob)

    else:
        ##mask the season types
        if inp.columns.nlevels > 2: #if statement required becasue cant convert one element to tuple
            slc_none = tuple([slice(None)] * (inp.columns.nlevels - 1)) #makes a slice(none) for each column level except season.
        else:
            slc_none = slice(None)
        inp = inp.loc[:, (z_mask, slc_none)]

        ##weighted average if steady state
        if general['steady_state']:
            try: #incase df is datearray
                inp = inp.mul(z_prob, axis=1, level=0).sum(axis=1, level=1)
            except TypeError:
                #this wont work if columns have two levels (would need to reshape into multi d numpy do the average then reshape to 2-d)
                n_inp = inp.values.astype(np.int64)
                n_inp = np.average(n_inp, axis=1, weights=z_prob)
                n_inp = n_inp.astype("datetime64[ns]")
                n_inp = n_inp.astype('M8[us]').astype('O') #converts to datetime
                col = pd.MultiIndex.from_tuples([inp.columns[0]])
                inp = pd.DataFrame(n_inp, index=inp.index, columns=col)

    return inp


