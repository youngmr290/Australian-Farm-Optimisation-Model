# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:26 2020

@author: John
"""

from timeit import default_timer as timer
import datetime as dt

import SheepSimFunctions as sfun
import PropertyInputs as pinp
import feedbudget as fdb

time_list = [] ; time_was = []
time_list.append(timer()) ; time_was.append("start")


sfun.read_excel()                         # read inputs from Excel file and map to the python variables
time_list.append(timer()) ; time_was.append("init & read inputs from Excel")

