# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:43:43 2020

Run this module if you want to validate your inputs.
Add to this module when you find inputs that give error if calculated wrong.

need to plan how this will work:
    1. run certain functions, then apply sa and run functions again - this is done for each exp. (could be hard if an input causes error after multiple function)
    2. could run everything except pyomo first - this is quicker so if any errors/warnings pop up you can fix them. Then run pyomo. maybe precalcs too slow
    3. just look through input manually and come up with rules to avoid bad inputs
@author: young
"""
#todo maybe these tests are just built into the code where relevant eg whats been done for fvp/dvp
pasture.py
1. pasture foo must be greate than 0 after it passes through conversion function.
foo_shears = np.maximum(0, np.minimum(foo, cu3[2] + cu3[0] * foo + cu3[1] * legume)) #this should be greater than 0
