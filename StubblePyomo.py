# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:10:55 2019

@author: young
"""

#python modules
import pyomo.environ as pe

#MUDAS modules
import Stubble as stub
from CreateModel import model

def stubpyomo_local():
    ####################
    #define parameters #
    ####################
    ##call the stubble function
    c_stub=stub.stubble_all()
    try:
        model.del_component(model.p_harv_prop_index)
        model.del_component(model.p_harv_prop)
    except AttributeError:
        pass
    model.p_harv_prop = pe.Param(model.s_feed_periods, model.s_crops, initialize=c_stub[0], default = 0.0, doc='proportion of the way through each fp harvest occurs (0 if harv doesnt occur in given period)')
    
    try:
        model.del_component(model.p_stub_md_index_index_0)
        model.del_component(model.p_stub_md_index)
        model.del_component(model.p_stub_md)
    except AttributeError:
        pass
    model.p_stub_md = pe.Param(model.s_feed_periods, model.s_stub_cat, model.s_crops, initialize=c_stub[1], default = 0.0, doc='md from 1t of each stubble categories for each crop')
    
    #^this param has inf values - this may cause a problem, if it does do a replace in the stubble sheet line 151
    try:
        model.del_component(model.p_stub_vol_index_index_0)
        model.del_component(model.p_stub_vol_index)
        model.del_component(model.p_stub_vol)
    except AttributeError:
        pass
    model.p_stub_vol = pe.Param(model.s_feed_periods, model.s_stub_cat, model.s_crops, initialize=c_stub[2], default = 0.0, doc='amount of intake volume required by 1t of each stubble category for each crop')
    
    try:
        model.del_component(model.p_a_req_index_index_0)
        model.del_component(model.p_a_req_index)
        model.del_component(model.p_a_req)
    except AttributeError:
        pass
    model.p_a_req = pe.Param(model.s_stub_cat, model.s_feed_periods, model.s_crops, initialize=c_stub[3], default = 0.0, doc='stubble required in each feed periods in order to consume 1t of cat A')
    
    try:
        model.del_component(model.p_bc_prov_index)
        model.del_component(model.p_bc_prov)
    except AttributeError:
        pass
    model.p_bc_prov = pe.Param(model.s_stub_cat, model.s_crops, initialize=c_stub[4], default = 0.0, doc='stubble B provided from 1t of cat A and stubble C provided from 1t of cat B')
    
    try:
        model.del_component(model.p_bc_req_index)
        model.del_component(model.p_bc_req)
    except AttributeError:
        pass
    model.p_bc_req = pe.Param(model.s_stub_cat, model.s_crops, initialize=c_stub[5], default = 0.0, doc='stubble required from the row inorder to consume cat B or cat C')
    
    try:
        model.del_component(model.p_fp_transfer_index)
        model.del_component(model.p_fp_transfer)
    except AttributeError:
        pass
    model.p_fp_transfer = pe.Param(model.s_feed_periods, model.s_crops, initialize=c_stub[6], default = 0.0, doc='stubble cat B or cat C transferred to the next feed period')
    

    ###################
    #local constraint #
    ###################
    ##stubble transter from category to category and period to period
    try:
        model.del_component(model.con_stubble_bcd_index_index_0)
        model.del_component(model.con_stubble_bcd_index)
        model.del_component(model.con_stubble_bcd)
    except AttributeError:
        pass
    def stubble_transfer(model,f,k,s):
        ss = list(model.s_stub_cat)[list(model.s_stub_cat).index(s)-1] #stubble cat plus one - used to transfer from current cat to the next, list is required because indexing of an ordered set starts at 1 which means index of 0 chucks error 
        fs = list(model.s_feed_periods)[f-1] #have to convert to a list first beacuse indexing of an ordered set starts at 1
        return  -model.v_stub_transfer[fs,k,s]*1000  + model.p_fp_transfer[f,k]*model.v_stub_transfer[f,k,s] \
                    - sum(model.v_stub_con[e,f,k,ss] * model.p_bc_prov[s,k] + model.v_stub_con[e,f,k,s] * model.p_bc_req[s,k] for e in model.s_sheep_pools) <=0
    model.con_stubble_bcd = pe.Constraint(model.s_feed_periods, model.s_crops, model.s_stub_cat, rule = stubble_transfer, doc='links rotation stubble production with consumption of cat A')



###################
#variable         #
###################
##stubble consumption
model.v_stub_con = pe.Var(model.s_sheep_pools, model.s_feed_periods, model.s_crops, model.s_stub_cat, bounds = (0.0, None), doc = 'consumption of stubble')
##stubble transfer
model.v_stub_transfer = pe.Var(model.s_feed_periods, model.s_crops, model.s_stub_cat, bounds = (0.0, None), doc = 'transfer of 1t of stubble to following period')

###################
#constraint global#
###################
##stubble transter from category to category and period to period
def stubble_req_a(model,k,s):
    return sum(model.v_stub_con[e,f,k,s] * model.p_a_req[s,f,k] for e in model.s_sheep_pools for f in model.s_feed_periods if model.p_a_req[s,f,k] !=0) 


##stubble md
def stubble_md(model,e,f):
    return sum(model.v_stub_con[e,f,k,s] * model.p_stub_md[f,s,k] for k in model.s_crops for s in model.s_stub_cat)
    
##stubble vol
def stubble_vol(model,e,f):
    return sum(model.v_stub_con[e,f,k,s] * model.p_stub_vol[f,s,k] for k in model.s_crops for s in model.s_stub_cat)