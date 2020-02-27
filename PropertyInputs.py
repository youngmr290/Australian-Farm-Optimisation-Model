# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:03:30 2020

@author: young
"""


import Functions as fun
import Sensitivity as sen 
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

n_pasture_types     = len(pastures)  #^should be done in pasfunctions           # Annual, Lucerne, Tedera  ^Add this to Property.xlsx (maybe in General) as a list of the pastures to include & this is the length.
pasture=dict()
for pasture in uinp.structure['pastures']:
    pasture[pasture] = fun.xl_all_named_ranges('Property.xlsx', [pasture])
    # if pasture == 'annual':
    #     t_exceldata = pasture[pasture]
pasture_inputs=pasture.copy()        
        
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
    #mach['approx_hay_yield']=mach_inp['approx_hay_yield']+sen.saa['variable'] #just an example, this can be deleted
    ##pasture will have to be added in a loop
    for pasture in uinp.structure['pastures']:
        pasture_inputs[pasture]['input name'] = sen.sam['sa name']['this will be a slice to select the correct section of the sa variable to match the current pasture']
