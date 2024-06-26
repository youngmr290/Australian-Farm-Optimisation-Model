# -*- coding: utf-8 -*-
"""
In addition to the labour requirement described in the other sections of the model, there is a fixed
labour requirement which reflects the labour required for administration tasks such as BAS, tax and
pay roll, farm planning and upskill activities such as attending conferences or field days.

    
@author: young
"""
#python modules
import pandas as pd
import math
import numpy as np
# from dateutil.relativedelta import relativedelta

#AFO modules
from . import PropertyInputs as pinp
from . import Periods as per
from . import Functions as fun
from . import SeasonalFunctions as zfun

na = np.newaxis

def fixed(params):
    ###inputs
    labour_period = per.f_p_dates_df()
    lp_p5z = labour_period.values
    keys_p5 = labour_period.index[:-1]
    keys_z = zfun.f_keys_z()

    ##z8 mask
    maskz8_p5z = zfun.f_season_transfer_mask(lp_p5z[:-1,:],z_pos=-1,mask=True) #slice off end date

    ##super
    super_dates_p8 = pinp.labour['super'].index.values
    super_length_p8 = pinp.labour['super']['days'].values
    super_labour_p8 = pinp.labour['super']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], super_dates_p8, super_length_p8)[:-1,:,:]
    super_p5z = np.sum(alloc_p5zp8 * super_labour_p8, axis=-1) #get rid of p8 axis
    super_p5z = super_p5z * maskz8_p5z
    super = pd.DataFrame(super_p5z, index=keys_p5, columns=keys_z)

    ##bas
    bas_dates_p8 = pinp.labour['bas'].index.values
    bas_length_p8 = pinp.labour['bas']['days'].values
    bas_labour_p8 = pinp.labour['bas']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], bas_dates_p8, bas_length_p8)[:-1,:,:]
    bas_p5z = np.sum(alloc_p5zp8 * bas_labour_p8, axis=-1) #get rid of p8 axis
    bas_p5z = bas_p5z * maskz8_p5z
    bas = pd.DataFrame(bas_p5z, index=keys_p5, columns=keys_z)

    ##planning
    planning_dates_p8 = pinp.labour['planning'].index.values
    planning_length_p8 = pinp.labour['planning']['days'].values
    planning_labour_p8 = pinp.labour['planning']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], planning_dates_p8, planning_length_p8)[:-1,:,:]
    planning_p5z = np.sum(alloc_p5zp8 * planning_labour_p8, axis=-1) #get rid of p8 axis
    planning_p5z = planning_p5z * maskz8_p5z
    planning = pd.DataFrame(planning_p5z, index=keys_p5, columns=keys_z)

    ##tax
    tax_dates_p8 = pinp.labour['tax'].index.values
    tax_length_p8 = pinp.labour['tax']['days'].values
    tax_labour_p8 = pinp.labour['tax']['hours'].values
    alloc_p5zp8 = fun.f_range_allocation_np(lp_p5z[...,na], tax_dates_p8, tax_length_p8)[:-1,:,:]
    tax_p5z = np.sum(alloc_p5zp8 * tax_labour_p8, axis=-1) #get rid of p8 axis
    tax_p5z = tax_p5z * maskz8_p5z
    tax = pd.DataFrame(tax_p5z, index=keys_p5, columns=keys_z)

    ##create params
    params['super'] = super.stack().to_dict()
    params['bas'] = bas.stack().to_dict()
    params['planning'] = planning.stack().to_dict()
    params['tax'] = tax.stack().to_dict()





    















