# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 08:57:13 2019

module: labour fixed and learn module 

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
     
formatting; try to avoid capitals (reduces possible mistakes in future)

note: the different aspects of fixed labour can be allocated to different labour pools
      currently i have labour learn and planning in the farmer and permanent pool and the rest in all pools. ie a farmer or permanent staff can only
      provide the time to learn however anyone can provide time to complete tax stuff.
    
    
@author: young
"""
#python modules
import pandas as pd
import math
import numpy as np
# from dateutil.relativedelta import relativedelta

#AFO modules
import PropertyInputs as pinp
import Periods as per
import Functions as fun

na = np.newaxis

def fixed(params):
    ###inputs
    labour_period = per.p_dates_df()
    lp_p5z = labour_period.values
    keys_p5 = labour_period.index[:-1]
    keys_z = pinp.f_keys_z()

    ##super
    super_dates_p8 = pinp.labour['super'].index.values
    super_length_p8 = pinp.labour['super']['days'].values.astype('timedelta64[D]')
    super_labour_p8 = pinp.labour['super']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], super_dates_p8, super_length_p8, True)[:-1,:,:]
    super_p5z = np.sum(alloc_p5zp8 * super_labour_p8, axis=-1) #get rid of p8 axis
    super = pd.DataFrame(super_p5z, index=keys_p5, columns=keys_z)

    ##bas
    bas_dates_p8 = pinp.labour['bas'].index.values
    bas_length_p8 = pinp.labour['bas']['days'].values.astype('timedelta64[D]')
    bas_labour_p8 = pinp.labour['bas']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], bas_dates_p8, bas_length_p8, True)[:-1,:,:]
    bas_p5z = np.sum(alloc_p5zp8 * bas_labour_p8, axis=-1) #get rid of p8 axis
    bas = pd.DataFrame(bas_p5z, index=keys_p5, columns=keys_z)

    ##planning
    planning_dates_p8 = pinp.labour['planning'].index.values
    planning_length_p8 = pinp.labour['planning']['days'].values.astype('timedelta64[D]')
    planning_labour_p8 = pinp.labour['planning']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], planning_dates_p8, planning_length_p8, True)[:-1,:,:]
    planning_p5z = np.sum(alloc_p5zp8 * planning_labour_p8, axis=-1) #get rid of p8 axis
    planning = pd.DataFrame(planning_p5z, index=keys_p5, columns=keys_z)

    ##tax
    tax_dates_p8 = pinp.labour['tax'].index.values
    tax_length_p8 = pinp.labour['tax']['days'].values.astype('timedelta64[D]')
    tax_labour_p8 = pinp.labour['tax']['hours'].values
    alloc_p5zp8 = fun.range_allocation_np(lp_p5z[...,na], tax_dates_p8, tax_length_p8, True)[:-1,:,:]
    tax_p5z = np.sum(alloc_p5zp8 * tax_labour_p8, axis=-1) #get rid of p8 axis
    tax = pd.DataFrame(tax_p5z, index=keys_p5, columns=keys_z)


    ##create params
    params['super'] = super.stack().to_dict()
    params['bas'] = bas.stack().to_dict()
    params['planning'] = planning.stack().to_dict()
    params['tax'] = tax.stack().to_dict()





    















