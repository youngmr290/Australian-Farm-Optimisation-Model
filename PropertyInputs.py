# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:03:30 2020

@author: young
"""
##python modules
import pickle as pkl
import os.path

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
        print( 'Reading property inputs from pickle')
except FileNotFoundError:      
    inputs_from_pickle = False


filename= 'pkl_property.pkl'
##if inputs are not read from pickle then they are read from excel and written to pickle
if inputs_from_pickle == False:
    with open(filename, "wb") as f:
        general_inp = fun.xl_all_named_ranges("Property.xlsx","General")
        pkl.dump(general_inp, f)

        labour_inp = fun.xl_all_named_ranges("Property.xlsx","Labour")
        pkl.dump(labour_inp, f)

        crop_inp = fun.xl_all_named_ranges("Property.xlsx","Crop")
        pkl.dump(crop_inp, f)

        mach_inp = fun.xl_all_named_ranges("Property.xlsx","Mach")
        pkl.dump(mach_inp, f)

        stubble_inp = fun.xl_all_named_ranges("Property.xlsx","Stubble")
        pkl.dump(stubble_inp, f)

        finance_inp = fun.xl_all_named_ranges("Property.xlsx","Finance")
        pkl.dump(finance_inp, f)

        feed_inp = fun.xl_all_named_ranges("Property.xlsx","Feed Budget") #automatically read in the periods as dates
        pkl.dump(feed_inp, f)

        sup_inp = fun.xl_all_named_ranges("Property.xlsx","Sup Feed") #automatically read in the periods as dates
        pkl.dump(sup_inp, f)

        sheep_inp  = fun.xl_all_named_ranges('Inputs parameters.xlsm', 'Property', numpy=True)
        pkl.dump(sheep_inp, f)

        feedsupply_inp  = fun.xl_all_named_ranges('Inputs parameters.xlsm', 'FeedSupply', numpy=True) #^think this will get combined with the input sheet above
        pkl.dump(feedsupply_inp, f)
        
        ##^the sheep one below might be able to be removed (remeber to delete in the sections below as well)
        # sheep_management_inp  = fun.xl_all_named_ranges('Property.xlsx', 'Sheep Management', numpy=True)
        # pkl.dump(sheep_management_inp, f)

        # sheep_regions_inp  = fun.xl_all_named_ranges('Property.xlsx', 'Sheep Regions', numpy=True)
        # pkl.dump(sheep_regions_inp, f)

        pasture_inp=dict()
        for pasture in uinp.structure['pastures']:
            pasture_inp[pasture] = fun.xl_all_named_ranges('Property.xlsx', pasture)
        pkl.dump(pasture_inp, f)

##else the inputs are read in from the pickle file
##note this must be in the same order as above
else:
    with open(filename, "rb") as f:
        general_inp = pkl.load(f)

        labour_inp = pkl.load(f)

        crop_inp = pkl.load(f)

        mach_inp = pkl.load(f)

        stubble_inp = pkl.load(f)

        finance_inp = pkl.load(f)

        feed_inp = pkl.load(f)

        sup_inp = pkl.load(f)

        sheep_inp  = pkl.load(f)
        
        feedsupply_inp  = pkl.load(f)
        
        # sheep_management_inp  = pkl.load(f)

        # sheep_regions_inp  = pkl.load(f)

        pasture_inp = pkl.load(f)

##create a copy of each input dict - this means there is always a copy of the origional inputs (the second copy has SA applied to it)
##the copy created is the one used in the actuall modules
general=general_inp.copy()
labour=labour_inp.copy()
crop=crop_inp.copy()
mach=mach_inp.copy()
stubble=stubble_inp.copy()
finance=finance_inp.copy()
feed_inputs=feed_inp.copy()
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
    #mach['approx_hay_yield']=mach_inp['approx_hay_yield']+sen.saa['variable'] #just an example, this can be deleted
    ##pasture will have to be added in a loop
    for pasture in uinp.structure['pastures']:
        ###SAM
        pasture_inputs[pasture]['GermStd'] = pasture_inp[pasture]['GermStd'] * sen.sam[('germ',pasture)]
        pasture_inputs[pasture]['GermScalarLMU'] = pasture_inp[pasture]['GermScalarLMU'] * sen.sam[('germ_l',pasture)]
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'] * sen.sam[('pgr',pasture)]
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'] * sen.sam[('pgr',pasture)]
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'].mul(sen.sam[('pgr_f',pasture)], axis=0)
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'].mul(sen.sam[('pgr_f',pasture)], axis=0)
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'].mul(sen.sam[('pgr_l',pasture)], axis=1)
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'].mul(sen.sam[('pgr_l',pasture)], axis=1)
        pasture_inputs[pasture]['DigDryAve'] = pasture_inp[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)] \
                                                + max(pasture_inp[pasture]['DigDryAve']) * (1 - sen.sam[('dry_dmd_decline',pasture)])
        pasture_inputs[pasture]['DigSpread'] = pasture_inp[pasture]['DigSpread'] * sen.sam[('grn_dmd_range_f',pasture)]
        pasture_inputs[pasture]['DigDeclineFOO'] = pasture_inp[pasture]['DigDeclineFOO'] * sen.sam[('grn_dmd_declinefoo_f',pasture)]
        pasture_inputs[pasture]['DigRednSenesce'] = pasture_inp[pasture]['DigRednSenesce'] * sen.sam[('grn_dmd_senesce_f',pasture)]
    ##sheep
    ###SAV
    sheep['i_eqn_compare'] = sen.sav['eqn_compare'] if not sen.sav['eqn_compare'] == '-' else sheep_inp['i_eqn_compare']
    sheep['i_mask_i'][0] = sen.sav['TOL_1_inc'] if not sen.sav['TOL_1_inc'] == '-' else sheep_inp['i_mask_i'][0] 
    sheep['i_mask_i'][1] = sen.sav['TOL_2_inc'] if not sen.sav['TOL_2_inc'] == '-' else sheep_inp['i_mask_i'][1] 
    sheep['i_includedg3_inc'][1] = sen.sav['g2_BM_included'] if not sen.sav['g2_BM_included'] == '-' else sheep_inp['i_includedg3_inc'][1] 
    sheep['i_includedg3_inc'][2] = sen.sav['g2_BM_included'] if not sen.sav['g2_BM_included'] == '-' else sheep_inp['i_includedg3_inc'][2] 
    sheep['i_includedg3_inc'][3] = sen.sav['g2_BM_included'] if not sen.sav['g2_BM_included'] == '-' else sheep_inp['i_includedg3_inc'][3] 



