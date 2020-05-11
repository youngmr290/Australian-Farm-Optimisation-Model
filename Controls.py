# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 09:57:03 2020

This module can be used to control things that will effect the whole execution of the code ie it could be used to determine what part of the solution is printed each itteration

@author: young
"""


##############
# pickle info#
##############
##used to determine if the inputs are read in from excel or pickle file
import os.path
try:
    if os.path.getmtime("Property.xlsx") > os.path.getmtime("pkl_property.pkl") or os.path.getmtime("Universal.xlsx") > os.path.getmtime("pkl_universal.pkl"):
        inputs_from_pickle = False 
    else: 
        inputs_from_pickle = True
        print( 'Reading inputs from pickle')
except FileNotFoundError:      
    inputs_from_pickle = False
##can i add an if statement to check if inputs were modified after last run ie if date of univ.xlsx > textfile.txt then read inputs from excel else from pickle
