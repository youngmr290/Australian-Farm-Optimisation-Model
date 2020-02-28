# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:03:30 2020

@author: young
"""


import Functions as fun
import UniversalInputs as uinp 



general_inp = fun.xl_all_named_ranges("Property.xlsx","General")
general=general_inp.copy()

crop_inp = fun.xl_all_named_ranges("Property.xlsx","Crop")
crop=crop_inp.copy()

mach_inp = fun.xl_all_named_ranges("Property.xlsx","Mach")
mach=mach_inp.copy()
stubble_inp = fun.xl_all_named_ranges("Property.xlsx","Stubble")
stubble=stubble_inp.copy()

finance_inp = fun.xl_all_named_ranges("Property.xlsx","Finance")
finance=finance_inp.copy()

feed_inp = fun.xl_all_named_ranges("Property.xlsx","Feed Budget") #automatically read in the periods as dates
feed_inputs=feed_inp.copy()

sheep_management_inp  = fun.xl_all_named_ranges('Property.xlsx', ['Management'])
sheep_management=mach_inp.copy()

sheep_regions_inp  = fun.xl_all_named_ranges('Property.xlsx', ['Regions'])
sheep_regions=sheep_regions_inp.copy()

pasture_inp=dict()
for pasture in uinp.structure['pastures']:
    pasture_inp[pasture] = fun.xl_all_named_ranges('Property.xlsx', pasture)
    # if pasture == 'annual':
    #     t_exceldata = pasture[pasture]
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
        pasture_inputs[pasture]['GermStd'] = pasture_inp[pasture]['GermStd'] * sen.sam[('germ',pasture)]
        pasture_inputs[pasture]['GermScalarLMU'] = pasture_inp[pasture]['GermScalarLMU'] * sen.sam[('germ_l',pasture)]
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'] * sen.sam[('pgr',pasture)]
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'] * sen.sam[('pgr',pasture)]
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'].mul(sen.sam[('pgr_f',pasture)], axis=0) 
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'].mul(sen.sam[('pgr_f',pasture)], axis=0)
        pasture_inputs[pasture]['LowPGR'] = pasture_inp[pasture]['LowPGR'].mul(sen.sam[('pgr_l',pasture)], axis=1)
        pasture_inputs[pasture]['MedPGR'] = pasture_inp[pasture]['MedPGR'].mul(sen.sam[('pgr_l',pasture)], axis=1)
        pasture_inputs[pasture]['DigDryAve'] = pasture_inp[pasture]['DigDryAve'] * sen.sam[('dry_dmd_decline',pasture)] # ^is this correct?
        pasture_inputs[pasture]['DigSpread'] = pasture_inp[pasture]['DigSpread'].mul(sen.sam[('grn_dmd_range_f',pasture)], axis=0)
        pasture_inputs[pasture]['DigDeclineFOO'] = pasture_inp[pasture]['DigDeclineFOO'].mul(sen.sam[('grn_dmd_declinefoo_f',pasture)], axis=0)
        pasture_inputs[pasture]['DigDeclineFOO'] = pasture_inp[pasture]['DigDeclineFOO'].mul(sen.sam[('grn_dmd_declinefoo_l',pasture)], axis=1)
        pasture_inputs[pasture]['DigRednSenesce'] = pasture_inp[pasture]['DigRednSenesce'].mul(sen.sam[('grn_dmd_senesce_f',pasture)], axis=0)




  



     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     