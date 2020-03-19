# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:12:20 2020

@author: young
"""
#python modules
import pandas as pd
import datetime as dt

#midas modules
import UniversalInputs as uinp
import Functions as fun
import Periods as per


########################
#off farm grain price  #
########################

def buy_grain_price():
    '''
    Returns
    -------
    Dict.
        purchase price of grain from neighbour for sup feeding
        Price includes:
        -transaction
        -cartage cost
    '''
    grain_price_info_df=uinp.price['grain_price'] #create a copy of grain price df so you dont have to reference input module each time
    ##multiplies the price and proportion of firsts and seconds for each grain, then sum to get overall price
    price_df = grain_price_info_df[['firsts','seconds']]
    cartage=uinp.price['sup_cartage']
    transaction_fee=uinp.price['sup_transaction']
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    cols = pd.MultiIndex.from_product([allocation.index, price_df.columns])
    price_df = price_df.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    return  price_df.mul(allocation,axis=1,level=0).stack([0,1])
