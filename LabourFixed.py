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

#AFO modules
import PropertyInputs as pinp
import Periods as per
import Functions as fun



def fixed(params):
    params['super'] = fun.df_period_total(per.p_dates_df()['date'],per.p_dates_df().index,pinp.labour['super'])
    params['bas'] = fun.df_period_total(per.p_dates_df()['date'],per.p_dates_df().index,pinp.labour['bas'])
    params['planning'] = fun.df_period_total(per.p_dates_df()['date'],per.p_dates_df().index,pinp.labour['planning'])
    params['tax'] = fun.df_period_total(per.p_dates_df()['date'],per.p_dates_df().index,pinp.labour['tax'])

# fixed()



    















