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


#MUDAS modules
from LabourFixedInputs import *
import Labour as lab
import PropertyInputs as pinp
'''
this can all be updated to the new labour format using the labour allocation function to 
determine the period allocation

'''


#add time required for tax to df in corresponding labour periods
def tax():
    lab.labour_periods['tax'] = np.nan
    #add the end date of the last labour period to the df. So that i + 1 doesn't give an error on the last period
    lab.labour_periods.loc[len(lab.labour_periods.index),'date']= pinp.feed_inputs['feed_periods'][1] + relativedelta(day=1,month=1,years=1)
    for i in lab.labour_periods.index:
        #stops it picking up the last row which is nan (except for date)
        if not math.isnan(i):
             for j in labour_fixed_input_data['labour_tax']:
                # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
                if lab.labour_periods.loc[i,'date'] <= j < lab.labour_periods.loc[i + 1 ,'date']:
                    lab.labour_periods.loc[i,'tax'] = labour_fixed_input_data['labour_tax'][j]
    # drop last row, because it has na because it only contains the end date, therefore not a period
    lab.labour_periods.drop(lab.labour_periods.tail(1).index,inplace=True) 
tax()        

#add time required for planning to df in corresponding labour periods
def planning():
    lab.labour_periods['planning'] = np.nan
    #add the end date of the last labour period to the df. So that i + 1 doesn't give an error on the last period
    lab.labour_periods.loc[len(lab.labour_periods.index),'date']= pinp.feed_inputs['feed_periods'][1] + relativedelta(day=1,month=1,years=1)
    for i in lab.labour_periods.index:
        #stops it picking up the last row which is nan (except for date)
        if not math.isnan(i):
             for j in labour_fixed_input_data['labour_planning']:
                # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
                if lab.labour_periods.loc[i,'date'] <= j < lab.labour_periods.loc[i + 1 ,'date']:
                    lab.labour_periods.loc[i,'planning'] = labour_fixed_input_data['labour_planning'][j]
    # drop last row, because it has na because it only contains the end date, therefore not a period
    lab.labour_periods.drop(lab.labour_periods.tail(1).index,inplace=True) 
planning()     

#add time required for bas to df in corresponding labour periods
def bas():
    lab.labour_periods['bas'] = np.nan
    #add the end date of the last labour period to the df. So that i + 1 doesn't give an error on the last period
    lab.labour_periods.loc[len(lab.labour_periods.index),'date']= pinp.feed_inputs['feed_periods'][1] + relativedelta(day=1,month=1,years=1)
    for i in lab.labour_periods.index:
        #stops it picking up the last row which is nan (except for date)
        if not math.isnan(i):
             for j in labour_fixed_input_data['labour_bas']:
                # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
                if lab.labour_periods.loc[i,'date'] <= j < lab.labour_periods.loc[i + 1 ,'date']:
                    lab.labour_periods.loc[i,'bas'] = labour_fixed_input_data['labour_bas'][j]
    # drop last row, because it has na because it only contains the end date, therefore not a period
    lab.labour_periods.drop(lab.labour_periods.tail(1).index,inplace=True) 
bas() 

#add time required for paying staff and super and workers comp to df in corresponding labour periods
def super_wc():
    lab.labour_periods['super'] = np.nan
    #add the end date of the last labour period to the df. So that i + 1 doesn't give an error on the last period
    lab.labour_periods.loc[len(lab.labour_periods.index),'date']= pinp.feed_inputs['feed_periods'][1] + relativedelta(day=1,month=1,years=1)
    for i in lab.labour_periods.index:
        #stops it picking up the last row which is nan (except for date)
        if not math.isnan(i):
             for j in labour_fixed_input_data['labour_super']:
                # check if the date falls into period i, if so add the number of tax hours to the tax column corresponding to that period.
                if lab.labour_periods.loc[i,'date'] <= j < lab.labour_periods.loc[i + 1 ,'date']:
                    lab.labour_periods.loc[i,'super'] = labour_fixed_input_data['labour_super'][j]
    # drop last row, because it has na because it only contains the end date, therefore not a period
    lab.labour_periods.drop(lab.labour_periods.tail(1).index,inplace=True) 
super_wc()

#fill blanks with 0
def fill_blanks():
    lab.labour_periods[['bas','super','planning','tax']]= lab.labour_periods[['bas','super','planning','tax']].fillna(0)
fill_blanks()
    















