# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:28:11 2019

Version Control:
Version     Date        Person  Change
   1.1      29Dec19     John    fb.period['feed_periods'] was altered to include the beginning of the next period (year +1)
                                so references to create fp were shorten fp=fp[:-1]

^update this with CSIRO (dean thomas) experiment data. - will hopefully simplify the sim
^if the sim becomes quick enough then it can be added into the exp loop ie can be adjusted by sen values ie like sheep sim.

this module determines the proportion of each stubble component (grain, leaf etc) in each stub categ (a,b,c,d)
The module requires access to stubble inputs module
The module writes the answers to an excel book which is referenced by AFO - this means this module only needs to be run if you make changes to stub

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

#AFO modules
import PropertyInputs as pinp

# Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in excel
writer = pd.ExcelWriter('stubble sim.xlsx', engine='xlsxwriter')
    

##inputs
hi_k = pinp.stubble['harvest_index']
index_k = pinp.stubble['i_stub_landuse_idx']
proportion_grain_harv_k = pinp.stubble['proportion_grain_harv']
stub_cat_prop_ks1 = pinp.stubble['stub_cat_prop']

##calc the dmd of each component at the point when category dmd was calibrated
deterioration_factor_ks0 = pinp.stubble['quality_deterioration']
days_since_harv = pinp.stubble['i_calibration_offest']
dmd_component_harv_ks0 = pinp.stubble['component_dmd'] #dmd at harvest
dmd_component_ks0 = ((1 - deterioration_factor_ks0) ** days_since_harv) * dmd_component_harv_ks0


for crp in range(len(index_k)):
    ######
    #sim #
    ######   
        
    def grain_prop():
        '''calc grain propn in stubble

        HI = total grain / total biomass (total biomass includes grain as well)
        stubble = leaf and stalk plus split grain
        '''
        hi = hi_k[crp]
        harv_prop = proportion_grain_harv_k[crp]
        splitgrain_propn_totalbiomass = hi*(1-harv_prop) #split grain as a propn of total biomass
        leafstalk_propn_totalbiomass = (1-hi) #leaf & stalk as a propn of total biomass
        stubble_propn_totalbiomass = splitgrain_propn_totalbiomass + leafstalk_propn_totalbiomass #stubble as a propn of total biomass
        return splitgrain_propn_totalbiomass/stubble_propn_totalbiomass * 100 #split grain as propn of stubble

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
        sim_length = int(100/pinp.stubble['step_size'])
        #number of components
        number_of_components = len(component_proportion)
        #numpy array for each stubble section used in sim
        stubble_availability=np.zeros([number_of_components,sim_length])
        weighted_availability=np.zeros([number_of_components+1,sim_length]) #extra one for a total tally which is required
        consumption=np.zeros([number_of_components,sim_length])
        cumulative_consumption=np.zeros([number_of_components,sim_length])
        #fill in each numpy array one step at a time. have to fill in each step for each array one at a time because the arrays are linked therefore each array used values from another
        for step in range(sim_length):
            #stubble availability (at the start of the sim this is component propn it then decreases depending on which components are consumed)
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(number_of_components)):
                if step == 0:
                    stubble_availability[component_num, step]=proportion
                elif stubble_availability[component_num, step-1] - consumption[component_num, step-1]<=0:
                    stubble_availability[component_num, step]=0
                else: stubble_availability[component_num, step]=stubble_availability[component_num, step-1] - consumption[component_num, step-1]
            #weighted availability (weight by consumption preference)
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                weighted_availability[component_num, step] = stubble_availability[component_num,step] * grazing_pref_component[component]
            weighted_availability[5, step] = weighted_availability[:,step].sum() 
            #consumption per time step (consumption of each component %)
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                if weighted_availability[number_of_components,step] <= 0:
                    consumption[component_num, step] = 0
                else:
                    consumption[component_num, step] = (pinp.stubble['step_size'] 
                    / weighted_availability[number_of_components,step] * weighted_availability[component_num, step] )
            #cumulative comsumption
            for component, proportion,component_num in zip(component_proportion.keys(),component_proportion.values(),range(len(component_proportion))):
                cumulative_consumption[component_num, step]= consumption[component_num].sum()
     
        #determine the proportion of each component in each category
        num_stub_cat = stub_cat_prop_ks1.shape[1]
        categ_sizes = stub_cat_prop_ks1[crp,:]
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
                    comp_consumed = cumulative_consumption[component,round(cum_cat_size*100-1)]  #multiplied by 100 to convert the percent to int. it is then use to index the steps in the numpy arrays above, minus 1 because indexing starts from 0
                else: comp_consumed = (cumulative_consumption[component,round(cum_cat_size*100-1)] #use the cat list so that i can determine the the consumption of a component at in the cat before
                    - cumulative_consumption[component,round(list(cumulative_cat_size)[cat_num-1]*100-1)])
                stub_cat_component_proportion[component, cat_num] = comp_consumed/cat_size/100
        return stub_cat_component_proportion
    
    def objective(x):
        #multiplies the component dmd by the proportion of that component consumed in each cat
        #this determines the overall dmd of that cat.
        #the objective func minimised the diff between the value above and the inputted value of cat dmd
        # component_dmd = np.array(component_dmd, dtype=float)
        cat_a_component_propn=stubble_sim(x)[:,0]
        a=np.dot(cat_a_component_propn,dmd_component_ks0[crp,:])
        cat_b_component_propn=stubble_sim(x)[:,1]
        b=np.dot(cat_b_component_propn,dmd_component_ks0[crp,:])
        cat_c_component_propn=stubble_sim(x)[:,2]
        c=np.dot(cat_c_component_propn,dmd_component_ks0[crp,:])
        cat_d_component_propn=stubble_sim(x)[:,3]
        d=np.dot(cat_d_component_propn,dmd_component_ks0[crp,:])
        cat_a_target = pinp.stubble['stub_cat_qual'][crp,0]
        cat_b_target = pinp.stubble['stub_cat_qual'][crp,1]
        cat_c_target = pinp.stubble['stub_cat_qual'][crp,2]
        cat_d_target = pinp.stubble['stub_cat_qual'][crp,3]
        
        return ((a-cat_a_target)**2+(b-cat_b_target)**2+(c-cat_c_target)**2+(d-cat_d_target)**2)
    #initial guesses    
    x0 = np.ones(7)
    # bounds on variables
    bndspositive = (0, 100.0) #qualtity of other components must be greater than 10%
    no_upbnds = (1, 1.0e10) #pref has to be greater than stem
    if index_k[crp] in ('r', 'z', 'l', 'f'):   #because these crops only have 4 stubble components ie no sheath
        var_bound = (0,10) #still need to give optimisation some room to move otherwise it gives bad solution.
    else: var_bound = (0,100)
    bnds = (bndspositive, var_bound, bndspositive, no_upbnds, no_upbnds, no_upbnds, no_upbnds)
    #may have to change around the solver (method) to get the best solution
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds)
    x = solution.x
    stub_cat_component_proportion = pd.DataFrame(stubble_sim(x)) 
    stub_cat_component_proportion.to_excel(writer, sheet_name=index_k[crp],index=False,header=False)
    
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
        #multiplies the component dmd by the proportion of that component consumed in each cat
        #this determines the overall dmd of that cat.
        #the objective func minimised the diff between the value above and the inputted value of cat dmd
        cat_a_dmd=stubble_sim(x)[:,0]
        a=np.dot(cat_a_dmd,dmd_component_ks0[crp,:])
        cat_b_dmd=stubble_sim(x)[:,1]
        b=np.dot(cat_b_dmd,dmd_component_ks0[crp,:])
        cat_c_dmd=stubble_sim(x)[:,2]
        c=np.dot(cat_c_dmd,dmd_component_ks0[crp,:])
        cat_d_dmd=stubble_sim(x)[:,3]
        d=np.dot(cat_d_dmd,dmd_component_ks0[crp,:])
        return(a,b,c,d)
        
    print('-'*100)
    print(index_k[crp])
    print('component proportions at harv : ',component_proportion.values()) #dict values, check to make sure they look sensible
    print('graxing pref : ',grazing_pref_component.values()) #dict values, check to make sure they look sensible
    print('cat ddm : ',cat_ddm(x))
    print('Target cat ddm : ',pinp.stubble['stub_cat_qual'][crp,0],pinp.stubble['stub_cat_qual'][crp,1], pinp.stubble['stub_cat_qual'][crp,2],pinp.stubble['stub_cat_qual'][crp,3])
    print('objective : ',objective(x))
    
writer.save()    
    
    