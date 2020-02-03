# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 21:03:30 2020

@author: young
"""


import Functions as fun

##use an if statement to say if the variable already exists don't read it in - will mean clearing all variable each time you make a change in excel inputs
general = fun.xl_all_named_ranges("Property.xlsx","General")
crop = fun.xl_all_named_ranges("Property.xlsx","Crop")
mach = fun.xl_all_named_ranges("Property.xlsx","Mach")
stubble = fun.xl_all_named_ranges("Property.xlsx","Stubble")
finance = fun.xl_all_named_ranges("Property.xlsx","Finance")
feed_inputs = fun.xl_all_named_ranges("Property.xlsx","Feed Budget") #automatically read in the periods as dates

n_pasture_types     = 3             # Annual, Lucerne, Tedera  ^Add this to Property.xlsx