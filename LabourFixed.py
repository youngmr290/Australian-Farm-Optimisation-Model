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
from dateutil.relativedelta import relativedelta

#MUDAS modules
import PropertyInputs as pinp
import Periods as per
'''
this can all be updated to the new labour format using the labour allocation function to 
determine the period allocation

'''


def fixed():
    ##copy labour periods df
    labour_periods_fx=per.p_dates_df() 
    ##add blank column for the following fixed labour activities - this is populated below
    labour_periods_fx['tax'] = np.nan
    labour_periods_fx['planning'] = np.nan
    labour_periods_fx['bas'] = np.nan
    labour_periods_fx['super'] = np.nan
    for i in labour_periods_fx.index:
            ##add time required for tax to df in corresponding labour periods
            for j in pinp.labour['tax'].index:
               # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
               if labour_periods_fx.loc[i,'date'] <= j < labour_periods_fx.loc[i + 1 ,'date']:
                   labour_periods_fx.loc[i,'tax'] = pinp.labour['tax'].loc[j,'hours']
            ##add time required for planning to df in corresponding labour periods
            for j in pinp.labour['planning'].index:
               # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
               if labour_periods_fx.loc[i,'date'] <= j < labour_periods_fx.loc[i + 1 ,'date']:
                   labour_periods_fx.loc[i,'planning'] = pinp.labour['planning'].loc[j,'hours']
            ##add time required for bas to df in corresponding labour periods
            for j in pinp.labour['bas'].index:
               # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
               if labour_periods_fx.loc[i,'date'] <= j < labour_periods_fx.loc[i + 1 ,'date']:
                   labour_periods_fx.loc[i,'bas'] = pinp.labour['bas'].loc[j,'hours']
            #add time required for paying staff and super and workers comp to df in corresponding labour periods
            for j in pinp.labour['super'].index:
               # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
               if labour_periods_fx.loc[i,'date'] <= j < labour_periods_fx.loc[i + 1 ,'date']:
                   labour_periods_fx.loc[i,'super'] = pinp.labour['super'].loc[j,'hours']
    ## drop last row, because it has na because it only contains the end date, therefore not a period
    labour_periods_fx.drop(labour_periods_fx.tail(1).index,inplace=True) 
    ##fill blanks with 0
    labour_periods_fx[['bas','super','planning','tax']]= labour_periods_fx[['bas','super','planning','tax']].fillna(0)
    return labour_periods_fx
# fixed()



    















