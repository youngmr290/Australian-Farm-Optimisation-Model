# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John
"""

'''
import functions from other modules
'''
# import datetime as dt
# import timeit
import pandas as pd
import numpy as np

# from numba import jit

import PropertyInputs as pinp
import FeedBudget as fdb
import UniversalInputs as uinp
import Functions as fun
import Periods as per
import Sensitivity as sen


########################
#constants required    #
########################
## define some parameters required to size arrays.
# n_feed_pools        = uinp.n_feed_pools
# n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1

#^ put the values as list in universal.xlsx and define n by length of the list
n_groups_ewes = 4       # genotype groups of ewes
n_groups_offspring = 5  # genotype groups and growth profile
n_groups_rams = 3       # genotypes of rams
n_birth_dates = 1       # birth dates for the seed animals (1 unless doing a TOL analysis or 8 month joinings)
n_litter_size = 5       # Dry, single, twin, triplet, not mated
n_lactation_number = 5  # dry, single, twin, triplet, in utero
n_btrt = 6              # 11, 22, 21, 33, 32, 31
n_dam_ages = 3          # yearling, maiden, adult
n_sexes = 2             # ewe, wether

# length_f  = np.array(pinp.feed_inputs['feed_periods'].loc[:n_feed_periods-1,'length']) # converted to np. to get @jit working

pbixy =  (n_sim_periods, n_birth_dates, n_groups_ewes, n_litter_size, n_lactation_number)
pbjsbd = (n_sim_periods, n_birth_dates, n_groups_offspring, n_sexes, n_btrt, n_dam_ages)
pk =    (n_sim_periods, n_groups_rams)   #^ is the b dimension required


def read_excel():
    '''Instantiate variables required and map inputs from excel to the arrays'''
    ## set global on all variables required outside this function
    #global n_sim_periods   #not sure that htis is required as well as p_index_p
    global p_index_p

    ### -define the vessels that will store the input data that require pre-defining

    i_me_maintenance_eft            = np.zeros(eft,    dtype = np.float64)  # M/D level for target LW pattern
    c_pgr_gi_scalar_gft             = np.zeros(gft,    dtype = np.float64)  # numpy array of pgr scalar =f(startFOO) for grazing intensity (due to impact of FOO changing during the period)

    ## map the Excel data into the numpy arrays
    i_variable name_indices              = propertydata['ExcelName']

    ## Some one time data manipulation for the inputs just read
    n_sim_periods = int(uinp.i_oldest_animal * uinp.n_sim_period_year)
    p_index_p = np.arange(n_sim_periods)

def set_sim_periods():
    '''Define the dates for the simulation periods

    start_date = pinp.feed_inputs['feed_periods']['date'][0]
    period_date_p = start_date + index_p * (365 / n_sim_periods_year)

