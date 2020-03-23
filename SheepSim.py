# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John
"""

'''
import functions from other modules
'''
# import datetime as dt
# import timeit as ti
import pandas as pd
import numpy as np

# from numba import jit

# import FeedBudget as fdb
# import Functions as fun
# import Periods as per
# import PropertyInputs as pinp
# import Sensitivity as sen
import SheepSimFunctions as sfun
# import UniversalInputs as uinp


############################
### _constants required    #
############################
## define some parameters required to size arrays.
# n_feed_pools        = uinp.n_feed_pools
# n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1

#^ put the values as lists in universal.xlsx (SheepDefinitions!)
# then define n by length of the list
n_groups_rams = 3       # genotypes of rams
n_groups_ewes = 4       # genotype groups of ewes
n_groups_offspring = 5  # genotype groups and growth profile
n_groups_lambing = 1    # lambing groups for the seed animals (1 unless doing a TOL analysis or 8 month joinings)
n_litter_size = 5       # Dry, single, twin, triplet, not mated
n_lactation_number = 5  # dry, single, twin, triplet, in utero
n_btrt = 6              # 11, 22, 21, 33, 32, 31
n_dam_ages = 3          # yearling, maiden, adult
n_sexes = 3             # ram, ewe, wether
n_max_ecycles = 2       # max number of estrus cycles they are joined

i_sim_periods_year = 52 # ^uinp.n_sim_periods_year   will be in structure dict now
i_oldest_animal= 6.5    # ^uinp.i_oldest_animal

# period_date_p = sfun.sim_periods(i_birth_dates, i_sim_periods_year, i_oldest_animal)
# ^ replace 5 lines below with above function
n_sim_periods = int(i_oldest_animal * i_sim_periods_year)
start_date = np.datetime64('2019-01-01','D')#^ pinp.feed_inputs['feed_periods']['date'][0],'D')
step = pd.to_timedelta(365.25 / n_sim_periods_year,'D')
finish_date = start_date + step * n_sim_periods
period_date_p = np.arange(start_date, finish_date, step, dtype='datetime64[D]')
p_index_p = np.arange(n_sim_periods)

pi      = (n_sim_periods,
           n_groups_rams)
pjexyl  = (n_sim_periods,
           n_groups_ewes,
           n_max_ecycles,
           n_litter_size,
           n_lactation_number,
           n_groups_lambing)
pjwexyl = (n_sim_periods,
           n_groups_ewes,
           n_sexes,
           n_max_ecycles,
           n_litter_size,
           n_lactation_number,
           n_groups_lambing)
pkdwebl = (n_sim_periods,
           n_groups_offspring,
           n_dam_ages,
           n_sexes,
           n_max_ecycles,
           n_btrt,
           n_groups_lambing)

def sheep_sim():
    """
    A function to wrap the simulation that can be called by SheepPyomo.
    Called after the sensitivty variables have been updated.
    It populates the arrays by looping through

    Returns
    -------
    None.

    """
    ####################################
    ### initialise arrays & map inputs #
    ####################################
    '''Instantiate the arrays: required because inputs are mapped to sub-arrays
    and map from input dictionaries to the arrays
    '''

    ## -define the vessels that will store the input data that require pre-defining
    ## # see documetation for a description of each variable
    i_variable_pjexyl       = np.zeros(pjexyl,    dtype = 'float64')

    ## map the sensitivity adjusted Excel data into the numpy arrays
    i_variablename_indices  = pinp.propertydata['ExcelName']

    ## Some one time data manipulation for the inputs just read
    start_year = np.min(birth_date_jl)
    ## might need to test and rebase the year for the other animal groups

    ###############################
    ### fixed period calculations #
    ###############################
    '''Calculations for which the inputs do not depend on previous periods
    See spreadsheet: Group independent and Age,Date,Timing'''
    date_p = sfun.sim_periods(start_year, i_sim_periods_year, i_oldest_animal)

    ##########################################
    ### Initialise then loop through periods #
    ##########################################
    ## initialise the arrays for the first period #
    lw_ffcf = i.weaning_wt
    mw =
    aw =
    bw =
    cfw =
    fd =
    fl =

    ## Loop through each period (p) for ewes
    for p in range(n_sim_periods):
        if p <>0:  # don't carry this out with p=0
            cr[p,...] = sfun.conception(z_end[p-1,...], rc_end[p-1,...])
            ## or pass lw_cfff_end and nw_end & srw and calculate z and rc
            mr[p,...] = sfun.mortality(rc[p-1,...])
            tem[p,...], dmr[p,...], lmr[p,...] = sfun.ewe_mortality()
            .... = sfun.numbers(....)
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        mei = sfun.intake(....)
        mem = sfun.energy(....)
        mep = sfun.pregnancy(....)
        mel = sfun.lactation(....)
        dcfw, new = sfun.wool_growth(....)
        cfw[p,...] = cfw[p-1,...] + dcfw
        wmax = np.maximum(lw_ffcf,axis=0)
        lwc = sfun.lw_change(mei, mem, mep, mel, new, wmax, zf1, zf2)
        .... = sfun.end_values

    # repeat loop for rams & then for offspring
    # these don't require
    sfun.conception
    sfun.ewe_mortality()
    sfun.pregnancy(....)
    sfun.lactation(....)

def sheep_parameters():
    """Parameter generation for the pyomo variables


    Returns
    -------
    an array of parameters for pyomo.
    Rows are the constraints.
    Columns are variables.
    """
parameters = np.zeros((len(output_required),len(activities0)), dtype = 'float64')
    # Loop through the number of variables
    for a in activites:
        ### create array masks  for the pyomo variable
        ''' For each pyomo variable create a mask that represents the animals
        The arrays can then be summed across the axes for that mask '''
        mask = sfun.create_mask(i_activity_definition)

        ### apply each mask to each simulation output
        #output_required is a list of the arrays that are required as parameters
        for n, o in enumerate(output_required):
            parameters[n,a] = np.sum(o[mask])
return parameters

''' Or to allow one function call per constraint this function could
generate the array and then multiple functions that just return the
required row of the array.'''