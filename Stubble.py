# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 09:10:54 2019

Version Control:
Version     Date        Person  Change
1.1         29Dec19     JMY     Altered feed periods in FeedBudget, therefore had to remove last row in the fp df used in this module
1.2         4Jan20      MRY     Documented me and vol section and changed definitions of dmd and md
1.3         6Jan20      MRY     clip the vol and me df at 0 (don't want negative me and negative vol could lead to potential issue) & add vol = 1000/(ri_quality * availability) to be used in pyomo


Known problems:
Fixed   Date        ID by   Problem
       4Jan20       MRY     ^ protein stuff needs to be added still, when the emissions are done
1.3    4Jan20       MRY     ^ need to clip the vol and me df at 0 (don't want negative me and negative vol could lead to potential issue)
       4Jan20       JMY     ^ could possibly convert this numpy to make it more simple 
1.3    4Jan20       JMY     ^need to add vol = 1000/(ri_quality * availability) to be used in pyomo




Stubble:

Total Grain = HI * (above ground) biomass
Leaf + Stem = (1-HI) * biomass
Harvested grain = (1 - spilt%) * Total grain
Spilt grain = spilt% * Total grain
Stubble = Leaf + Stem + Spilt grain

Spilt grain as a proportion of the stubble = (HI * spilt %) / (1 - HI(1 - spilt%))

@author: young
"""
#python modules
import numpy as np

import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')

#midas modules
import Functions as fun
import StockFunctions as sfun
import PropertyInputs as pinp
import UniversalInputs as uinp
import Crop as crp
import Sensitivity as SA


def stubble_all(params):
    '''
    Wraps all of stubble into a function that is called in pyomo 
    
    Returns; multiple dicts
        - stubble transfer
        - md and vol
        - harv con limit
    '''

    #create df with feed periods
    fp = pinp.feed_inputs['feed_periods'].iloc[:-1].copy() #removes last row, some reason i need to copy so that i don't get settingcopywarning
    cat_a_st_req=pd.DataFrame()
    cat_b_st_prov=pd.DataFrame() #provide tr ie cat A provides stub b tr - done straight as a dict because value doesn't change for different periods
    cat_b_st_req=pd.DataFrame() #requirement for tr ie cat B requires stub b tr - done straight as a dict because value doesn't change for different periods, this would have to change if trampling was different for periods
    cat_c_st_prov=pd.DataFrame()
    cat_c_st_req=pd.DataFrame()
    per_transfer=pd.DataFrame()
    dmd=pd.DataFrame(columns=pd.MultiIndex.from_product([pinp.stubble['harvest_index'].index,pinp.stubble['stub_cat_qual']]))
    md=pd.DataFrame(columns=pd.MultiIndex.from_product([pinp.stubble['harvest_index'].index,pinp.stubble['stub_cat_qual']]))
    ri_quality=pd.DataFrame(index=fp.index, columns=pd.MultiIndex.from_product([pinp.stubble['harvest_index'].index,pinp.stubble['stub_cat_qual']]))
    ri_availability=pd.DataFrame(index=fp.index, columns=pd.MultiIndex.from_product([pinp.stubble['harvest_index'].index,pinp.stubble['stub_cat_qual']]))
    vol=pd.DataFrame(index=fp.index, columns=pd.MultiIndex.from_product([pinp.stubble['harvest_index'].index,pinp.stubble['stub_cat_qual']]))
    cons_prop=pd.DataFrame(index=fp.index)


    ##create mask which is stubble available. Stubble is available from the period harvest starts to the beginning of the following growing season.
    ##if the end date of the fp is after harvest then stubble is available.
    harv_date = pinp.crop['harv_date']
    mask_stubble_exists = pinp.feed_inputs['feed_periods'].loc['FP1':, 'date'] > harv_date  #need to use the full fp array that has the end date of the last period.
    mask_stubble_exists = mask_stubble_exists.values #convert to numpy

    #########################
    #dmd deterioration      #
    #########################
    stubble_per_grain = crp.stubble_production() #produces dict with stubble production per kg of yield for each grain used in the ri.availability section
    for crop in pinp.stubble['harvest_index'].index:
        #add column that is days since harvest in each period (calced from the end date of the period)
        days_since_harv=[]
        for period_num in range(len(fp)):
            if period_num < len(fp)-1:  #because im using the end date of the period, hence the start of the next p, there is an issue for the last period in the df - the last period should ref the date of the first period
                feed_period_date = fp.loc['FP%s' %(period_num+1),'date']
                harv_start = pinp.crop['start_harvest_crops'].loc[crop,'date']
                if feed_period_date < harv_start:
                    days_since_harv.append(365 + (feed_period_date - harv_start).days) #add a yr because dates before harvest wont have access to stubble until next yr
                else:  days_since_harv.append((feed_period_date - harv_start).days)
            else: 
                period = fp.index[0]
                feed_period_date = fp.loc[period,'date']
                harv_start = pinp.crop['start_harvest_crops'].loc[crop,'date']
                if feed_period_date < harv_start:
                    days_since_harv.append(365 + (feed_period_date - harv_start).days) #add a yr because dates before harvest wont have access to stubble until next yr
                else:  days_since_harv.append((feed_period_date - harv_start).days)
        fp.loc[:,'days_%s' %crop]=days_since_harv
        #add the quantity decline % for each period - used in transfer constraints, need to average the number of days in the period of interest
        quant_decline=[]
        for period_num in range(len(fp)):
                if period_num == 0:
                     if fp.loc['FP%s' %period_num,'date'] < harv_start < fp.loc['FP%s' %(period_num+1),'date']:
                         days=fp.loc['FP%s' %period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
                     else: days=fp.loc[fp.index[len(fp)-1],'days_%s' %crop]+(fp.loc['FP%s' %period_num,'days_%s' %crop]-fp.loc[fp.index[len(fp)-1],'days_%s' %crop])/2#divid by two to get the average days on the period to work out average deterioration
                elif fp.loc['FP%s' %period_num,'date'] < harv_start < fp.loc['FP%s' %(period_num+1),'date']:
                    days=fp.loc['FP%s' %period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
                else: days=fp.loc['FP%s' %(period_num-1),'days_%s' %crop]+(fp.loc['FP%s' %period_num,'days_%s' %crop]-fp.loc['FP%s' %(period_num-1),'days_%s' %crop])/2
                quant_decline.append(1-(1-pinp.stubble['quantity_deterioration'].loc[crop,'deterioration']/100)**days)
        fp.loc[:,'quant_decline_%s' %crop] = quant_decline
        #add dmd for each component in each period for each crop
        for component, dmd_harv in zip(pinp.stubble['component_dmd'],pinp.stubble['component_dmd'].loc[crop]):
            dmd_component=[]
            day=[]
            for period_num in range(len(fp)):
                if period_num == 0:
                      if fp.loc['FP%s' %period_num,'date'] < harv_start < fp.loc['FP%s' %(period_num+1),'date']:
                          days=fp.loc['FP%s' %period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
                      else: days=fp.loc[fp.index[len(fp)-1],'days_%s' %crop]+(fp.loc['FP%s' %period_num,'days_%s' %crop]-fp.loc[fp.index[len(fp)-1],'days_%s' %crop])/2#divid by two to get the average days on the period to work out average deterioration
                elif fp.loc['FP%s' %period_num,'date'] < harv_start < fp.loc['FP%s' %(period_num+1),'date']:
                    days=fp.loc['FP%s' %period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
                else: days=fp.loc['FP%s' %(period_num-1),'days_%s' %crop]+(fp.loc['FP%s' %period_num,'days_%s' %crop]-fp.loc['FP%s' %(period_num-1),'days_%s' %crop])/2
                day.append(days)
                deterioration_factor = pinp.stubble['quality_deterioration'].loc[crop,component]
                dmd_component.append((1-(deterioration_factor*days)/100)*dmd_harv)
            fp.loc[:,'%s_%s_dmd' %(crop ,component)]=dmd_component #important as the column names are used in the sim (objective)
        
          
        ###############
        # M/D & vol   #      
        ###############
        '''
        This section creates a df that contains the M/D for each stubble category for each crop and 
        the equivalent for vol. This is used by live stock.
        
        1) read in stubble component composition, calculated by the sim (stored in an excel file)
        2) converts total dmd to dmd of category 
        3) calcs ri quantity and availability 
        4) calcs the md of each stubble category (dmd to MD)
        
        '''    
        stub_cat_component_proportion=pd.read_excel('stubble sim.xlsx',sheet_name=crop,header=None, engine='openpyxl')
        ##quality of each category in each period - multiply quality by proportion of components
        num_stub_cat = len( pinp.stubble['component_dmd'].loc[crop]) #determine number of cats
        comp_dmd_period=fp.iloc[:,-num_stub_cat:] #selects just the dmd from fp df for the crop of interest
        stub_cat_component_proportion.index=comp_dmd_period.columns #makes index = to column names so df mul can be done (quicker than using a loop)
        j=0 #used as scalar for stub available for in each cat below.
        base_yields = base_yields = crp.f_base_yield()  #todo is this correct?
        for cat_inx, cat_name in zip(range(len(pinp.stubble['stub_cat_qual'].loc[crop])), pinp.stubble['stub_cat_qual']):
            dmd.loc[:,(crop,cat_name)]=comp_dmd_period.mul(stub_cat_component_proportion[cat_inx]).sum(axis=1) #dmd by stub category (a, b, c, d)
            ##calc relative quality before converting dmd to md - note that the equation system used is the one selected for dams in p1 - currently only cs function exists 
            if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
                ri_quality.loc[:,(crop,cat_name)]= dmd.loc[:,(crop,cat_name)].apply(sfun.f_rq_cs, args=(pinp.stubble['clover_propn_in_sward_stubble'],))
            ##ri availability - first calc stubble foo (stub available) this is the average from all rotations because we just need one value for foo
            ###try calc the base yield for each crop but if the crop is not one of the rotation phases then assign the average foo (this is only to stop error. it doesnt matter because the crop doesnt exist so the stubble is never used)
            try:
                stub_foo_harv = base_yields.loc[crop].mean() * stubble_per_grain[(crop,'a')]
            except KeyError: #if the crop is not in any of the rotations assign average foo to stop error - this is not used so could assign any value.
                stub_foo_harv = base_yields.mean()
            stubble_foo = stub_foo_harv * (1 - fp['quant_decline_%s' %crop]) * (1 - j)
            if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used - note that the equation system used is the one selected for dams in p1
                ri_availability.loc[:,(crop,cat_name)] = sfun.f_ra_cs(stubble_foo.values, pinp.stubble['i_hf'])
            ##combine ri quality and ri availability to calc overall vol (potential intake)
            vol.loc[:,(crop,cat_name)]=(1/(ri_availability.loc[:,(crop,cat_name)].mul(ri_quality.loc[:,(crop,cat_name)]))*1000/(1+SA.sap['pi']))#.replace(np.inf, 10000) #this produces some inf when ri_quality is 0 (1/0)  
            j+= pinp.stubble['stub_cat_prop'].loc[crop,cat_name]
            #now convert dmd to M/D
            md.loc[:,(crop,cat_name)]=dmd.loc[:,(crop,cat_name)].apply(fun.dmd_to_md).mul(1000).clip(lower=0) #mul to convert to tonnes    
    
        ###########
        #trampling#
        ###########
        #for now this is just a single number however the input could be changed to per period, then convert this to a df or array, if this is changed some of the dict below would need to be dfs the stacked - so they account for period
        tramp_effect=[]
        stub_cat_prop = [] #used in next section but easy to add here in the same loop
        for cat in pinp.stubble['stub_cat_prop']:
            tramp_effect.append(pinp.stubble['trampling'].loc[crop,'trampling']/100 * pinp.stubble['stub_cat_prop'].loc[crop,cat])
            stub_cat_prop.append(pinp.stubble['stub_cat_prop'].loc[crop,cat])
        
        ################################
        # allow access to next category#   #^this is a little inflexible ie you would need to add or remove code if a stubble cat was added or removed
        ################################
        
        cat_a_st_req.loc[:,crop]= (1/(1-fp['quant_decline_%s' %crop]))*(1+tramp_effect[0])*(1/stub_cat_prop[0])*1000 #*1000 - to convert to tonnes
        cat_b_st_prov.loc[crop,'prov'] = stub_cat_prop[1]/stub_cat_prop[0]*1000
        cat_b_st_req.loc[crop,'req'] = 1000*(1+tramp_effect[1])
        cat_c_st_prov.loc[crop,'prov'] = stub_cat_prop[2]/stub_cat_prop[1]*1000
        cat_c_st_req.loc[crop,'req'] = 1000*(1+tramp_effect[2])
        
        
        ##############################
        #transfers between periods   #   
        ##############################
        ##transfer a given cat to the next period. 
        per_transfer.loc[:,crop]= fp.loc[:,'quant_decline_%s' %crop]*1000 + 1000 
    
       
        
        ###############
        #harvest p con# stop sheep consuming more than possible because harvest is not at the start of the period
        ###############
        #how far through each period does harv start? note 0 for each period harv doesn't start in. Used to calc stub consumption limit in harv period
        #^could just use the period allocation function or period_proportion_np 
        prop=[]
        for period_num in range(len(fp)):
            if period_num == len(fp)-1:
                prop.append(1-(fp.loc['FP%s' %period_num,'days_%s' %crop] / ((fp.loc['FP0','date'] - fp.loc['FP%s' %period_num,'date']).days+365)))
            else: prop.append(1-(fp.loc['FP%s' %period_num,'days_%s' %crop] / (fp.loc['FP%s' %(period_num+1),'date'] - fp.loc['FP%s' %period_num,'date']).days))
        cons_prop.loc[:,crop]= prop
        cons_prop.loc[:,crop] =cons_prop.loc[:,crop].clip(0)
        #cons_limit[crop] =  (cons_limit[crop]/(1-cons_limit[crop]))*1000 ^not needed anymore because only need to proportion of time that sheep can't graze rather than the volume because p7con is now based on me not vol

    ###################################
    ##collate all the bits for pyomo  #
    ###################################

    ##return everything as a dict which is accessed in pyomo
    ###p7con to dict for pyomo
    cons_prop=cons_prop.stack().to_dict()
    per_transfer = per_transfer.mul(mask_stubble_exists, axis=0)
    per_transfer=per_transfer.stack().to_dict()
    ###add category to transfer 'require' params ie consuming 1t of stubble B requires 1.002t from the constraint (0.002 accounts for trampling)
    cat_a_st_req=pd.concat([cat_a_st_req], keys='a')
    cat_b_st_req=pd.concat([cat_b_st_req], keys='b')
    cat_a_st_req=cat_a_st_req.stack().to_dict()
    cat_c_st_req=pd.concat([cat_c_st_req], keys='c')
    transfer_req=pd.concat([cat_b_st_req,cat_c_st_req]).squeeze().to_dict()
    ###add category to transfer 'provide' params ie transferring 1t from current period to the next - this accounts for deterioration
    cat_b_st_prov=pd.concat([cat_b_st_prov], keys='b')
    cat_c_st_prov=pd.concat([cat_c_st_prov], keys='c')
    transfer_prov=pd.concat([cat_b_st_prov,cat_c_st_prov]).squeeze().to_dict()
    ###md & vol #todo when confinement pool is added set stubble md to zero because stubble can't be grazed in confinement
    #### Stubble doesn't include calculation of effective mei because stubble is generally low quality feed with a wide variation in quality within the sward.
    #### Therefore, there is scope to alter average diet quality by altering the grazing time and the proportion of the stubble consumed.
    md = md.mul(mask_stubble_exists, axis=0)
    md.columns=pd.MultiIndex.from_tuples(md) #converts to multi index so stacking will have crop and cat as separate keys
    md = md.stack().stack().to_dict()    #for pyomo
    vol = vol.mul(mask_stubble_exists, axis=0)
    vol.columns=pd.MultiIndex.from_tuples(vol) #converts to multi index so stacking will have crop and cat as separate keys
    vol = vol.stack().stack().to_dict()    #for pyomo

    ##load params to dict for pyomo
    params['cons_prop']=cons_prop
    params['md']=md
    params['vol']=vol
    params['cat_a_st_req']=cat_a_st_req
    params['transfer_prov']=transfer_prov
    params['transfer_req']=transfer_req
    params['per_transfer']=per_transfer
    
    
    return cons_prop, md, vol,cat_a_st_req,transfer_prov,transfer_req,per_transfer #return multiple dicts that can be accessed in pyomo    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    