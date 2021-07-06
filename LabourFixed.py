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
    keys_p5 = labour_period.index[:-1]
    keys_z = pinp.f_keys_z()
    labour_period_start_p5z = labour_period.values[:-1]
    labour_period_end_p5z = labour_period.values[1:]

    ##super
    super_dates_p8 = pinp.labour['super'].index.values
    super_labour_p8 = pinp.labour['super'].squeeze().values
    alloc_p5zp8 = np.logical_and(labour_period_start_p5z[...,na] <= super_dates_p8,
                                 super_dates_p8 < labour_period_end_p5z[...,na])
    super_p5z = np.sum(alloc_p5zp8 * super_labour_p8, axis=-1) #get rid of p8 axis
    super = pd.DataFrame(super_p5z, index=keys_p5, columns=keys_z)

    ##bas
    bas_dates_p8 = pinp.labour['bas'].index.values
    bas_labour_p8 = pinp.labour['bas'].squeeze().values
    alloc_p5zp8 = np.logical_and(labour_period_start_p5z[...,na] <= bas_dates_p8,
                                 bas_dates_p8 < labour_period_end_p5z[...,na])
    bas_p5z = np.sum(alloc_p5zp8 * bas_labour_p8, axis=-1) #get rid of p8 axis
    bas = pd.DataFrame(bas_p5z, index=keys_p5, columns=keys_z)

    ##planning
    planning_dates_p8 = pinp.labour['planning'].index.values
    planning_labour_p8 = pinp.labour['planning'].squeeze().values
    alloc_p5zp8 = np.logical_and(labour_period_start_p5z[...,na] <= planning_dates_p8,
                                 planning_dates_p8 < labour_period_end_p5z[...,na])
    planning_p5z = np.sum(alloc_p5zp8 * planning_labour_p8, axis=-1) #get rid of p8 axis
    planning = pd.DataFrame(planning_p5z, index=keys_p5, columns=keys_z)

    ##tax
    tax_dates_p8 = pinp.labour['tax'].index.values
    tax_labour_p8 = pinp.labour['tax'].squeeze().values
    alloc_p5zp8 = np.logical_and(labour_period_start_p5z[...,na] <= tax_dates_p8,
                                 tax_dates_p8 < labour_period_end_p5z[...,na])
    tax_p5z = np.sum(alloc_p5zp8 * tax_labour_p8, axis=-1) #get rid of p8 axis
    tax = pd.DataFrame(tax_p5z, index=keys_p5, columns=keys_z)


    ##create params
    params['super'] = super.stack().to_dict()
    params['bas'] = bas.stack().to_dict()
    params['planning'] = planning.stack().to_dict()
    params['tax'] = tax.stack().to_dict()





    















