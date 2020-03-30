# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 16:19:34 2019

Module: FeedBudget - Contains relationships and formulas that are common across the modules that do feed budgeting (pastures, stubble & livestock)

key: green section title is major title
     '#' around a title is a minor section title
     std '#' comment about a given line of code

Version Control:
Version     Date        Person  Change
   1.1      29Dec19     JMY     Added the date for the begining of the P0 in the next year (for period length calculation for the full 10 periods)
   1.2      30Dec19     JMY     Added calculation of length of the feed periods (to be used in PastureFunctions)
                                Altered 'import datetime' to 'import datetime as dt'
                                Converted all variable names with RI to ri
                                Updated the RI functions to handle numpy array as an input

Known problems:
Fixed   Date    ID by   Problem



@author: John
"""
##python modules
import datetime as dt
import numpy as np
# import pandas as pd
import math


##Midas modules
import PropertyInputs as pinp
import UniversalInputs as uinp

################
##feed periods #
################

# feed_inputs = dict()
# #start dates - strp function convert the string to a date, could be entered as a date using std date time formatting
# feed_inputs['feed_periods'] = {  0	: dt.datetime.strptime('24/4/2019',  '%d/%m/%Y'),
#                                  1  : dt.datetime.strptime('15/5/2019',  '%d/%m/%Y'),
#                                  2	: dt.datetime.strptime('12/6/2019',  '%d/%m/%Y'),
#                                  3	: dt.datetime.strptime(' 7/8/2019',  '%d/%m/%Y'),
#                                  4	: dt.datetime.strptime('25/9/2019',  '%d/%m/%Y'),
#                                  5	: dt.datetime.strptime('30/10/2019', '%d/%m/%Y'),
#                                  6	: dt.datetime.strptime('27/11/2019', '%d/%m/%Y'),
#                                  7	: dt.datetime.strptime('22/1/2020',  '%d/%m/%Y'),
#                                  8	: dt.datetime.strptime('12/3/2020',  '%d/%m/%Y'),
#                                  9	: dt.datetime.strptime(' 9/4/2020',  '%d/%m/%Y'),
#                                  10	: dt.datetime.strptime('24/4/2020',  '%d/%m/%Y')}



'''
define standard feed budgeting coefficients
'''
ria:float   = uinp.feed_inputs['ria']     # coefficient for RI Quantity calculations. Exponential coefficient for rate of eating
rib:float   = uinp.feed_inputs['rib']     # coefficient for RI Quantity calculations. Multiplier for time spent eating
rik:float   = uinp.feed_inputs['rik']     # coefficient for RI Quantity calculations. Expontential coefficient for time spent eating
# rifoo:float = pinp.feed_inputs['rifoo']   # coefficient for RI Quantity calculations. Extra foo measured in MIDAS region, compared to region used to calabrate intake equation. #rifoo is defined in pas module for each pasture and it defaults to 0 for stubble
rih:float   = uinp.feed_inputs['rih']     # coefficient for RI Quality calculations. Coefficient to alter sensitivity to dmd for introduced and native species.
rig:float   = uinp.feed_inputs['rig']     # coefficient for RI Quality calculations. Multiplier for impact of proportion of legume
rid:float   = uinp.feed_inputs['rid']     # coefficient for RI Quality calculations. Digestability for max intake.

'''
define standard feed budgeting functions
'''
def dmd_to_md(dmd):
    '''define a function to return M/D from DMD

    dmd can be either a percentage or a decimal
    returns M/D in MJ of ME per kg of DM
    dmd can be a numpy array or a scalar (not sure if it handles lists and data frames)

    ^ this could be expanded to include forage (0.172 * dmd - 1.7)
       and supplement (.133 * dmd + 23.4 ee + 1.32)
       using an extra 'type' input that is default 'herbage'
    '''
    try:
        if (dmd <= 1).all() : dmd *= 100 # if dmd is a list or an array and is a decimal then convert to percentage (in excel 80% is 0.8 in python)
    except:
        if dmd <= 1:          dmd *= 100 # if dmd is a scalar and is a decimal then convert to percentage   ^ alternative would be to convert scalar values to a list (if dmd isinstance not list: dmd=[dmd]) or perhaps type is float]
    return 0.17 * dmd - 2                # formula 1.13C from SCA 1990 pg 9

'''
define ri functions
'''
#################
## quality      #
#################
##there are two ways to use this function loop through your dmd data and enter one at a time into this funct
##or use .apply method to apply it to every value in a df or series - this method is faster, i have used it in stubble module (this method removes the need for a loop)
def ri_quality(dmd,clover_propn): #requires the clover prop and dmd of the feed, this is used in the stub module if an example is needed
    min_ri=0.01
    try:
        if (dmd >= 1).any() : dmd /= 100 # if dmd is a list or an array and is a percentage then convert to decimal (in excel 80% is 0.8 in python)
    except:
        if dmd >= 1:          dmd /= 100 # if dmd is a scalar and is a percentage then convert to decimal   ^ alternative would be to convert scalar values to a list (if dmd isinstance not list: dmd=[dmd]) or perhaps type is float]
    try:
        ri = float(1-rih*(rid-dmd)+rig*clover_propn) #formula 6.8 from SCA 1990 pg 218  ^could be updated with formula from Sheep Explorer
        return max(min_ri, ri)
    except:   # handle a numpy array
        ri = 1-rih*(rid-dmd)+rig*clover_propn                   #formula 6.8 from SCA 1990 pg 218  ^could be updated with formula from Sheep Explorer
        # print('ri_qual',ri)
        return np.maximum(min_ri, ri)                                                #set the maximum value to 1 and the minimum to 0

#################
## availability #
#################
##made up of a combination of grazing time and bite size. foori is accounting for difference in foo estimate of model region cf region equation is calibrated for.
def ri_availability(foo,rifoo=0):
    min_ri = 0.01
    try:
        ri_avail = (1-math.exp(-ria*(foo - rifoo)/1000))*(1+rib*math.exp(-rik*((foo-rifoo)/1000)**2))    #formula 6.7 from SCA 1990 pg 216  ^could be updated with formula from Sheep Explorer
        ri_avail = min(1,max(min_ri,ri_avail))
    except:   # handle a numpy array
        ri_avail =  (1-  np.exp(-ria*(foo - rifoo)/1000))*(1+rib*  np.exp(-rik*((foo-rifoo)/1000)**2))
        ri_avail = ri_avail.clip(min_ri,1)
        # print('ri_avail',ri_avail)
    return ri_avail
##################
## effective mei #
##################
def effective_mei(dmi, md, threshold, ri=1, eff_above=0.5):
    """Calculate MEI and scale for reduced efficiency if above animal requirements.

    Parameters
    ----------
    dmi       : value or array - Dry matter intake (kg).
    md        : value or array - M/D of the feed (MJ of ME / kg of DM).
    threshold : value or array - Diet quality (ME/Vol) required by animals.
    ri        : value or array, optional (1.0)     - Relative intake (quality and quantity).
    eff_above : value or array, optional (0.5) - Efficiency.
    that energy is used if above required quality and animals are gaining then losing weight.

    If inputs are provided in arrays then they must be braodcastable.

    Returns
    -------
    ME avaialable to the animal to meet their ME requirements, from the quantity of DM consumed.

    """
    fec = md * ri
    fec_effective  = np.minimum(fec, threshold + (fec - threshold) * eff_above)
    md_effective = fec_effective / ri
    mei_effective = dmi * md_effective
    return mei_effective


##check the feed_allocation function with a random date & length

##method 1 - put all the things straight into the func
## s=fun.period_allocation(list(feed_inputs['feed_periods'].values()),feed_inputs['feed_periods'].keys(),dt.datetime(2019,5,20),datetime.timedelta(days=20))

##method 2 - define the different args then input a simplified var into fun - this looks neater
##doesn't have to be done if a func but reduces variable name space congestion
# def test():
#     p_dates=list(feed_inputs['feed_periods'].values())
#     print(p_dates)
#     p_name=feed_inputs['feed_periods'].keys()
#     start=dt.datetime(2019,5,20)
#     length=datetime.timedelta(days=20)
#     return fun.period_allocation(p_dates,p_name,start,length)

# print(test())






