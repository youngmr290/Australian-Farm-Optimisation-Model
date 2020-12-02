"""
Created on Thu Dec 2 09:35:26 2020

@author: Young
"""

import pickle as pkl
import pandas as pd

import Report as rep


##read in exp
exp_data = pd.read_excel('exp.xlsx',index_col=[0,1,2], header=[0,1,2,3])
exp_data = exp_data.sort_index() #had to sort to stop performance warning, this means runs may not be executed in order of exp.xlsx


##load pickle
with open('pkl_lp_vars.pkl', "rb") as f:
    lp_vars = pkl.load(f)
with open('pkl_r_vals.pkl', "rb") as f:
    r_vals = pkl.load(f)


##call the reporting
inter={}
##create intermidiates for each trial
for row in range(len(exp_data)):
    ##check to make sure user wants to run this trial
    if exp_data.index[row][0] == True:
        inter[exp_data.index[row][2]]={}
        rep.intermediates(inter[exp_data.index[row][2]], r_vals[exp_data.index[row][2]], lp_vars[exp_data.index[row][2]])
##create reports
rep.dse(inter, r_vals)
rep.profitloss_table(inter)
