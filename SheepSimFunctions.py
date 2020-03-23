# -*- coding: utf-8 -*-
"""
All the calculation function for the simulation
Created on Sat Feb 29 08:05:09 2020

@author: John
"""

import numpy as np
import pandas as pd

### -numbers & weight functions
def sim_periods(start_year, periods_per_year, oldest_animal):
    '''Define the dates for the simulation periods.
    Starts on 1 Jan of the year with the earliest birthdate.

    Parameters:
    start_year = int: year to start simulation. Derived from the birth dates
    periods_per_year = int:

    Returns:
    array of period dates (1D periods)
    '''
    n_sim_periods = int(i_oldest_animal * i_sim_periods_year)
    start_date = np.datetime64(start_year+'01-01','D')   #^ want this to return 1 Jan YYYY
    step = pd.to_timedelta(365.25 / n_sim_periods_year,'D')
    finish_date = start_date + step * n_sim_periods
    period_date_p = np.arange(start_date, finish_date, step, dtype='datetime64[D]')
    return period_date_p


### _intake functions
def potential_intake():
    #do potential intake calculations

### _energy functions


### _conception functions()
    #increment number of rams required for the ewes that are being joined this period
    n_rams_il = np.sum(n_pjexyl[this period is joining for e=0 &
                                this period is in the window for lambing period i joining
                                these ewes are mated to ram group i], axis(3,4))  \
                * ram joining percentage_oj