# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:28:11 2019

Version Control:
Version     Date        Person  Change
   1.1      29Dec19     John    fb.feed_inputs['feed_periods'] was altered to include the beginning of the next period (year +1)
                                so references to create fp were shorten fp=fp[:-1]


this module determines the proportion of each stubble component (grain, leaf etc) in each stub categ (a,b,c,d)
The module requires access to stubble inputs module
The module writes the answers to and excel book whick is referenced by midas - this means this module only needs to be run if you make changes to stub

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
import pandas as pd
import numpy as np
from scipy.optimize import minimize

#midas modules
import StubbleInputs as si
import Inputs as inp
import CropInputs as ci
import FeedBudget as fb

#this first section is just to get the ddm of the diff components in the first period 
#this is also used in the stubble module but i wanted to keep it seperate from here so i have just re-used code

fp = pd.DataFrame.from_dict(fb.feed_inputs['feed_periods'],orient='index', columns=['date'])
fp=fp[:-1]

# Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('stubble sim.xlsx', engine='xlsxwriter')
    
for crop in si.stubble_inputs['crop_stub'].keys():
    
    
    #add column that is days since harvest in each period (calced from the end date of the period)
    days_since_harv=[]
    for period_num in fp.index:
        if period_num < len(fp)-1:  #because im using the end date of the period, hence the start of the next p, there is an issue for the last period in the df - the last period should ref the date of the first period
            feed_period_date = fp.loc[period_num+1,'date']
            harv_start = ci.crop_input['start_harvest_crops'][crop]
            if feed_period_date < harv_start:
                days_since_harv.append(365 + (feed_period_date - harv_start).days) #add a yr because dates before harvest wont have access to stubble until next yr
            else:  days_since_harv.append((feed_period_date - harv_start).days)
        else: 
            period_num = fp.index[0]
            feed_period_date = fp.loc[period_num,'date']
            harv_start = ci.crop_input['start_harvest_crops'][crop]
            if feed_period_date < harv_start:
                days_since_harv.append(365 + (feed_period_date - harv_start).days) #add a yr because dates before harvest wont have access to stubble until next yr
            else:  days_since_harv.append((feed_period_date - harv_start).days)
    fp['days_%s' %crop]=days_since_harv
    #add dmd for each component in each period for each crop
    for component, dmd_harv in si.stubble_inputs['crop_stub'][crop]['component_dmd'].items():
        dmd_component=[]
        for period_num in fp.index:
            if period_num == 0:
                 if fp.loc[period_num,'date'] < harv_start < fp.loc[period_num+1,'date']:
                     days=fp.loc[period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
                 else: days=fp.loc[len(fp)-1,'days_%s' %crop]+(fp.loc[period_num,'days_%s' %crop]-fp.loc[len(fp)-1,'days_%s' %crop])/2#divid by two to get the average days on the period to work out average deterioration
            elif fp.loc[period_num,'date'] < harv_start < fp.loc[period_num+1,'date']:
                days=fp.loc[period_num,'days_%s' %crop]/2  #divid by two to get the average days on the period to work out average deterioration
            else: days=fp.loc[period_num-1,'days_%s' %crop]+(fp.loc[period_num,'days_%s' %crop]-fp.loc[period_num-1,'days_%s' %crop])/2
            deterioration_factor = si.stubble_inputs['crop_stub'][crop]['quality_deterioration'][component]
            dmd_component.append((1-(deterioration_factor*days)/100)*dmd_harv)
        fp['%s_%s_dmd' %(crop ,component)]=dmd_component #important as the column names are used in the sim (objective)
 
    
    ######
    #sim #
    ######   
        
    #proportin of grain is determined by hi and harv proportion
    #first determine the proportion of grain per straw and leaf
    #then convert to grain proportin of stubble (note stubble definition includes grain but HI is just the ratio of biomass to grain harvested ie doesn't include grain spilt)
    def grain_prop():
        hi= si.stubble_inputs['crop_stub'][crop]['harvest_index']
        harv_prop = si.stubble_inputs['crop_stub'][crop]['proportion_grain_harv']
        split_grain_straw = hi*(1-harv_prop)/(1-hi)*100
        return split_grain_straw/(1+split_grain_straw/100)

          
    #quantity of each stubble component at harvest  
    def stubble_sim(x):
        #variables to be solved for (this is new to this version of the sim)
        z,w,q, g, b, s, c = x
        
        
        component_proportion={'grain' : grain_prop() 
        ,'blade' : z
        ,'sheath': w
        ,'chaff' : q}
        
        #might be worth making this a proper constrain, either make stem a variable then then con is the sum of all == 100
        component_proportion['stem'] = 100- (z+w+q+grain_prop())
        
        
        #variables to be solved for (this is new to this version of the sim)
        grazing_pref_component={'grain' :  g
        ,'blade' : b
        ,'sheath': s
        ,'chaff' : c
        ,'stem' :1}
        #sim length
        sim_length = int(100/si.stubble_inputs['step_size'])
        #number of components
        number_of_components = len(component_proportion)
        #numpy array for each stubble section used in sim
        stubble_availability=np.zeros([number_of_components,sim_length])
        weighted_availability=np.zeros([number_of_components+1,sim_length]) #extra one for a total tally which is required
        consumption=np.zeros([number_of_components,sim_length])
        cumulative_consumption=np.zeros([number_of_components,sim_length])
        #fill in each numpy array one step at a time. have to fill in each step for each array one at a time because the arrays are linked therefore each array used values from another
        for step in range(sim_length):
            #stubble availability
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(number_of_components)):
                if step == 0:
                    stubble_availability[component_num, step]=proportion
                elif stubble_availability[component_num, step-1] - consumption[component_num, step-1]<=0:
                    stubble_availability[component_num, step]=0
                else: stubble_availability[component_num, step]=stubble_availability[component_num, step-1] - consumption[component_num, step-1]
            #weighted availability
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                weighted_availability[component_num, step] = stubble_availability[component_num,step] * grazing_pref_component[component]
            weighted_availability[5, step] = weighted_availability[:,step].sum() 
            #consumption per time step (consumption of each component %)
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                if weighted_availability[number_of_components,step] <= 0:
                    consumption[component_num, step] = 0
                else:
                    consumption[component_num, step] = (si.stubble_inputs['step_size'] 
                    / weighted_availability[number_of_components,step] * weighted_availability[component_num, step] )
            #cumulative comsumption
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                cumulative_consumption[component_num, step]= consumption[component_num].sum()
     
        #determine the actual proportion of componets consumed in each stubble category
        si.stubble_inputs['crop_stub'][crop]['stub_cat_prop']['d']= (1 - (si.stubble_inputs['crop_stub'][crop]['stub_cat_prop']['a']
        +si.stubble_inputs['crop_stub'][crop]['stub_cat_prop']['b']+si.stubble_inputs['crop_stub'][crop]['stub_cat_prop']['c']))
        num_stub_cat = len( si.stubble_inputs['crop_stub'][crop]['stub_cat_prop'])
        categ_sizes = si.stubble_inputs['crop_stub'][crop]['stub_cat_prop'].values()
        cumulative_cat_size=[]
        for i,j in zip(categ_sizes,range(num_stub_cat)):
            if j > 0: 
                cumulative_cat_size.append(cumulative_cat_size[-1]+i)
            else: cumulative_cat_size.append(i)
        #create numpy to store stubble dets that go into the rest of the stubble calcs
        stub_cat_component_proportion = np.zeros([number_of_components,num_stub_cat])
        for cat_num, cum_cat_size, cat_size in zip(range(num_stub_cat), cumulative_cat_size, categ_sizes):
            for component in range(number_of_components):
                #ammount of a component consumed in a given category
                if cat_num == 0: #if not cat A then need to subtract off the consumed amount in the periods before
                    comp_consumed = cumulative_consumption[component,int(cum_cat_size*100-1)]  #multiplied by 100 to convert the percent to int. it is then use to index the steps in the numpy arrays above, minus 1 because indexing starts from 0
                else: comp_consumed = (cumulative_consumption[component,int(cum_cat_size*100-1)] #use the cat list so that i can determine the the consumption of a component at in the cat before
                    - cumulative_consumption[component,int(list(cumulative_cat_size)[cat_num-1]*100-1)])
                stub_cat_component_proportion[component, cat_num] = comp_consumed/cat_size/100
        return stub_cat_component_proportion
    
    def objective(x):
        #select the dmd from the period of interest (depends on what date the experimental data is for) generally the preiod of harvest ie p7. this is because you want the the categories to be calibrated with the right proportions of each component to match the quality when target qualitied were collected. This is generally not exactly at harvest therefore you need the whole period dmd to account for deterioration
        harv_start = ci.crop_input['start_harvest_crops'][crop]
        component_dmd=[]
        for comp in si.stubble_inputs['crop_stub'][crop]['component_dmd']:
            m=fp['date'].ge(harv_start) #finds dates that are after harvest (ge->=) - in the next step it selects the first max. this give the row indx for the first date after harv, then minus 1 to give the period harv is in
            dmd = fp.loc[m.idxmax()-1,'%s_%s_dmd'%(crop ,comp)]
            component_dmd.append(dmd)
        #multiplies the component dmd by the proportion of that component consumed in each cat
        #this determines the overall dmd of that cat.
        #the objective func minimised the diff between the value above and the inputted value of cat dmd
        component_dmd = np.array(component_dmd, dtype=float)
        cat_a_dmd=stubble_sim(x)[:,0]
        a=np.dot(cat_a_dmd,component_dmd)
        cat_b_dmd=stubble_sim(x)[:,1]
        b=np.dot(cat_b_dmd,component_dmd)
        cat_c_dmd=stubble_sim(x)[:,2]
        c=np.dot(cat_c_dmd,component_dmd)
        cat_d_dmd=stubble_sim(x)[:,3]
        d=np.dot(cat_d_dmd,component_dmd)
        cat_a_target = si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['a']
        cat_b_target = si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['b']
        cat_c_target = si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['c']
        cat_d_target = si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['d']
        
        return ((a-cat_a_target)**2+(b-cat_b_target)**2+(c-cat_c_target)**2+(d-cat_d_target)**2)
    #initial guesses    
    x0 = np.ones(7)
    # bounds on variables
    bndspositive = (10, 100.0) #qualtity of other components must be greater than 10%
    no_upbnds = (1, 1.0e10) #pref has to be greater than stem
    if crop in ('rcanola', 'tcanola', 'lupins', 'faba'):   #because these crops only have 4 stubble components ie no sheath
        var_bound = (0,0)
    else: var_bound = (10,100)
    bnds = (bndspositive, var_bound, bndspositive, no_upbnds, no_upbnds, no_upbnds, no_upbnds)
    #may have to change around the solver (method) to get the best solution
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x
    stub_cat_component_proportion = pd.DataFrame(stubble_sim(x)) 
    stub_cat_component_proportion.to_excel(writer, sheet_name=crop,index=False,header=False)
    #################################################
    #post calcs to make sure everything looks good  #
    #################################################
    
    #check the component proportion
    component_proportion={'grain' : grain_prop() 
        ,'blade' : x[0]
        ,'sheath': x[1]
        ,'chaff' : x[2]
        ,'stem': 100- (x[0]+x[1]+x[2]+grain_prop())}
    grazing_pref_component={'grain' :  x[3]
        ,'blade' : x[4]
        ,'sheath': x[5]
        ,'chaff' : x[6]
,'stem' :1}
    
    def cat_ddm(x):
        #func from above used to determine the final ddm of each cat
        harv_start = ci.crop_input['start_harvest_crops'][crop]
        component_dmd=[]
        for comp in si.stubble_inputs['crop_stub'][crop]['component_dmd']:
            m=fp['date'].ge(harv_start) #finds dates that are after harvest - in the next step it selects the first max. this give the row indx for the first date after harv, then minus 1 to give the period harv is in
            dmd = fp.loc[m.idxmax()-1,'%s_%s_dmd'%(crop ,comp)]
            component_dmd.append(dmd)
        #multiplies the component dmd by the proportion of that component consumed in each cat
        #this determines the overall dmd of that cat.
        #the objective func minimised the diff between the value above and the inputted value of cat dmd
        component_dmd = np.array(component_dmd, dtype=float)
        cat_a_dmd=stubble_sim(x)[:,0]
        a=np.dot(cat_a_dmd,component_dmd)
        cat_b_dmd=stubble_sim(x)[:,1]
        b=np.dot(cat_b_dmd,component_dmd)
        cat_c_dmd=stubble_sim(x)[:,2]
        c=np.dot(cat_c_dmd,component_dmd)
        cat_d_dmd=stubble_sim(x)[:,3]
        d=np.dot(cat_d_dmd,component_dmd)
        return(a,b,c,d)
        
    print('-'*10)
    print(crop)
    print('component proportions : ',component_proportion.values()) #dict values, check to make sure they look sensible
    print('graxing pref : ',grazing_pref_component.values()) #dict values, check to make sure they look sensible
    print('cat ddm : ',cat_ddm(x))
    print('Target cat ddm : ',si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['a'],si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['b'], si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['c'],si.stubble_inputs['crop_stub'][crop]['stub_cat_qual']['d'])
    print('objective : ',objective(x))
    
writer.save()    
    
    