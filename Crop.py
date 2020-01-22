# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:05:00 2019

module: crop/rotation module

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
 
Version Control:
Version     Date        Person  Change
1.1         27Dec19     MRY     Updated rotation phase function, and crop functions (individual costs seperated and one funct to sum them all at the bottom)
1.1         28Dec19     MRY     alterd stubble handling to work with new rotation system (stubble handling cost is now associated with the current phase rather than the previous phase)
1.2         16Jan20     M       seperated grain price from yield - it is now seperate so it can be combined with yield penalty and crop grazing yield penalty before being multiplied by price.

Known problems:
Fixed   Date        ID by   Problem
        28/12/19    MRY     No input optimisation (solution is to convert crop into a simulation)
        28/12/19    MRY     All lmus recieve a fert app cost per ha even if the fert applied to that lmu is 0 (generally not a problem bevause all lmus recieve some fert. and the per tonne cost accounts for variable lmu rates)
        28/12/19    MRY     Stubble handeling cost still aplies even if the next phase would be pasture (difficult to fix but only a minor issue)

    
Things to add to this module: some input optimisation

assumption: reseeded pas is 100% arable (this means reseeded pas costs slightly more to sow than reality) - it is difficult to seperate from normal pas and fert application is 100% anyway the only thing not 100% is the seeding.

@author: young
"""

#python modules
import pandas as pd
import numpy as np
import timeit
import datetime as dt


#MUDAS modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import StubbleInputs as si
import Functions as fun
import Periods as per
import Mach as mac

print('Status:  running crop')





   
#print(timeit.timeit(test,number=1))
                
        



########################
#phases                #
########################
##makes a df of all possible rotation phases
phases_df =pd.Series(uinp.structure['rotations']['rot_phase']).str.split(expand=True).dropna()
phases_df2=phases_df.copy() #make a copy so that it doesn't alter the phases df that exists outside this func
phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns, ['']])  #make the df multi index so that when it merges with other df below the indexs remanin seperate (otherwise it turn into a one leveled tuple)



#########################
#yield                  #
#########################

def rot_yield():
    '''
    Returns
    ----------
    Dict for pyomo
        Grain yield for each rotation.
        Yield includes:
        -arable area
        -seeding rate (if farmers use thier own seed)
        -lmu factor
        -frost
    '''
    ##combines yield and grain price
    base_yields = pinp.crop['yield'].stack().swaplevel()*1000  #get base yields #swap levels - switches the order of the multi level index, so the yield lines up with the current yr not previos yr #stack to convert to a 1d dataframe so it can be merged using its index
    yields_lmus = pinp.crop['yield_by_lmu'] #soil yield factor
    seeding_rate = pinp.crop['seeding_rate'] #seeding rate
    frost = pinp.crop['frost'] #frost
    ##calculate yield - base yield * arable area * frost * lmu factor - seeding rate
    arable = pinp.crop['arable'] #read in arable area df
    yield_arable_by_soil=yields_lmus.mul(arable).mul(1-frost) #mul arable area to the the lmu factor (easy because dfs have the same axis's). THen mul frost
    yields=yield_arable_by_soil.reindex(base_yields.index, axis=0, level=1).mul(base_yields,axis=0, level=1) #reindes and mul with base yields
    seeding_rate=seeding_rate.reindex(yields.index, axis=0, level=1) #minus seeding rate
    yields=yields.sub(seeding_rate,axis=0, level=1).clip(lower=0) #we don't want negitive yields so clip at 0 (if any values are neg they become 0)
    yields = pd.merge(phases_df,yields, how='left', left_on=uinp.cols(), right_index = True)
    return yields.set_index(list(range(uinp.structure['phase_len']))).stack().to_dict()
#a=yield_income()

def grain_price():
    '''
    Returns
    -------
    Dict.
        Farm gate price for each grain
        Price includes:
        -offspec grain
        -cartage cost
        -other fees ie cbh and levies
    '''
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    grain_price_info_df=uinp.price['grain_price'] #create a copy of grain price df so you dont have to reference input module each time
    ##multiplies the price and proportion of firsts and seconds for each grain, then sum to get overall price
    price_df = pd.np.multiply(grain_price_info_df[['firsts','seconds']], grain_price_info_df[['prop_firsts','prop_seconds']]).sum(axis=1)
    cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance'] 
            + pinp.general['rail_cartage'] + uinp.price['flagfall'])
    tols= grain_price_info_df['grain_tolls']
    total_fees= cartage+tols
    farm_gate_price=(price_df-total_fees)
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period')
    allocation_cols = pd.MultiIndex.from_product([allocation.columns, farm_gate_price.index])
    allocation = allocation.reindex(allocation_cols, axis=1,level=0)#adds level to header so i can mul in the next step
    return  allocation.mul(farm_gate_price,axis=1,level=1).droplevel(0, axis=1).stack().to_dict()

#######
#fert #    
#######    
'''
1) determines fert cost allocation 
2) fert requirment for each rot phase
3) cost of fert for each rotation 
4) application cost per kg and application cost per ha 
    -per tonne; represents the difference in application time based on fert density - represents the filling up and traveling to the paddock time, ie it would require more filling and traveling time to spread 1t of a lighter (less dense) fert.
    -per ha; represents the time to spread 1ha - this depends how far each fert is chucked out of the spreader
5) sum together to get overall fert cost
'''

def fert_cost_allocation():
    '''
    Returns
    ----------
    Dataframe; multiplied with fert cashflows 
        Determines the cashflow period allocation for costs associated with each different fert
    '''
    start_df = pinp.crop['fert_info']['app_date'] #needed for allocation func
    length_df = pinp.crop['fert_info']['app_len'].astype('timedelta64[D]') #needed for allocation func
    p_dates = per.cashflow_periods()['start date'] #needed for allocation func
    p_name = per.cashflow_periods()['cash period'] #needed for allocation func
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)
# allocation=fert_cost_allocation()

def fert_req():
    '''
    Returns
    ----------
    Dataframe; used to calc fert cost in the next function 
        Fert required by 1ha of each phases (kg/ha)
    '''
    arable = pinp.crop['arable'] #read in arable area df
    fert = pinp.crop['fert'].reset_index().pivot(index='fert',columns='index').T  #read in and convert input fert df to a shape that can be merged to the phases df
    fert_by_soil = pinp.crop['fert_by_lmu'].stack() #read in fert by soil
    arable2=arable.reindex(fert_by_soil.index, axis=1, level=1)
    fert1=fert.reindex(fert_by_soil.index, axis=1, level=0).mul(fert_by_soil)
    fert1=fert1.mul(arable2,axis=0,level=1) #add arable to df
    fert = pd.merge(phases_df2, fert1, how='left', left_on=uinp.cols(), right_index = True) #merge fert with phases
    return fert.set_index(list(range(uinp.structure['phase_len']))).stack()
# fert=fert_req()
  
def fert_cost():
    '''
    Returns
    ----------
    Dataframe; summed with other fert cashflow items at the end of this section 
        Calcs fert cost for each rotation phase 
        - per tonne - inc cartage  for each cashflow period
    '''
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost 
    total_cost = fert_cost_allocation().mul(cost+transport).stack() #mul fert cost and transport with fert cost allocation
    ##now combine with fert cost per tonne with fert requirment of each phase
    fertreq = fert_req().reindex(total_cost.index, axis=1, level=1)
    phase_fert_cost_t=fertreq.mul(total_cost/1000).sum(axis=1, level=0) #div by 1000 to convert to $/kg, sum the cost of all the ferts
    return phase_fert_cost_t

def phase_fert_app_cost():  
    '''
    Application cost per tonne ($/rotation)
        
    Returns
    ----------
    Dataframe; summed with other fert cashflow items at the end of this section 
    '''
    allocation = fert_cost_allocation()
    application_cost = allocation.mul(mac.fert_app_cost_t()).stack() #mul app cost per tonne with fert cost allocation
    ##now combine with fert cost per tonne with fert requirment of each phase
    fertreq = fert_req().reindex(application_cost.index, axis=1, level=1)
    fert_app_cost_t=fertreq.mul(application_cost/1000).sum(axis=1, level=0) #div by 1000 to convert to $/kg
    ##now add fert app cost per ha 
    arable = pinp.crop['arable'] #read in arable area df
    passes = pinp.crop['passes'].reset_index().pivot(index='fert',columns='index').T #passes over each ha for each fert type
    arable3=arable.reindex(passes.index, axis=0, level=1).stack() #reindex so it can be mul with passes
    passes=passes.reindex(arable3.index).T.mul(arable3).T
    fert_cost = allocation.mul(mac.fert_app_cost_ha()).stack() #cost for 1 pass for each fert.
    fert_cost = passes.reindex(fert_cost.index, axis=1,level=1).mul(fert_cost) #total cost 
    fert_cost=fert_cost.sum(level=[0], axis=1).replace(0, np.nan).unstack() #sum each fert cost - cost doesn't need to be seperated by fert type once joined with passes #sum nan returns 0 therefore i need to convert 0 back to nan so that they are dropped when stacking to reduce dict size.
    phase_fert_cost_ha = pd.merge(phases_df2, fert_cost, how='left', left_on=uinp.cols(), right_index = True) #merge with all the phases, requires because different phases have different application passes
    phase_fert_cost_ha = phase_fert_cost_ha.set_index(list(range(uinp.structure['phase_len']))).stack([1])
    fert_cost_total= fert_app_cost_t.add(phase_fert_cost_ha, fill_value=0) #fill_value replaces any values that don't exist in both df with 0. This avoide getting nan if a cost exists in only one df.
    return fert_cost_total

def total_phase_fert_cost():
    '''
    Sum off application cost and actual fert cost
        
    Returns
    ----------
    Dataframe; summed with other cashflow items at the end of the module 
    '''
    fert_cost_total= fert_cost().add(phase_fert_app_cost(), fill_value=0) #fill_value replaces any values that don't exist in both df with 0. This avoide getting nan if a cost exists in only one df.
    return fert_cost_total

#######################
#stubble handeling    #
#######################
'''
cost now associated with current yr because it requires knpwing the yield of the crop - hence under the new rotation phases that contain sets it isn;t possible to use the previous landuse to determine cost.
As a general rule stubble handling is not required if:
-the land use is a legume crop, pasture or lucerne
Other general rules about stubble handling:
- once cereal crops are big enough to yield more than 3.5t of grain/ha their stubble residue starts to become an issue when sowing the next crop
- once canola crops are big enough to yield more than 2.3t of grain/ha their stubble residue starts to become an issue when sowing the next crop
- wheat stubble is the most problematic
- barley stubble tends to be less problematic because: barley crops tend to get harvested at a lower height;  barley stubble seems to break down quicker than wheat; and livestock tend to graze barley stubbles harder.
Limitations with the way stubble is handled in this Table:
    The biggest factor determining of whether stubble will require handling is the amount of stubble present, which largely depends on season type (other factors include width between crop rows, what type of seeding setup is used, how low crop was harvested, if the header was equipped with a straw chopper). All of this makes the requirement for stubble handling hard to represent in a steady state model. Hence the probability-based approach adopted in this Table.
    This probability is calculated by dividing the average yield for that LMU by the critical grain yield. This means the likelihood of stubble handling being req’d increases for higher yielding soil types etc, which is logical. However, the probability isn’t accurately linked to the likelihood that stubble will actually require handling. For instance, just because the average steady-state wht yield of a LMU is 1.75t/ha doesn’t necessarily mean that the wheat stubble on that LMU will need handling 1.75/3.5 = 50% of the time.
    So in summary, these probabilities are fairly crude...
    additionally this new structure assumes that even if the preceeding landuse is pasture the current phase will still get handling cost (wasn't able to find an alternative way)
frost is not included because that doesn't reduce biomass
'''

def phase_stubble_cost():
    '''first calculate the probability of a rotation phase needing stubble handling'''
    base_yields = pinp.crop['yield'].T#.stack().swaplevel()*1000
    stub_handling_threashold = pd.Series(pinp.stubble['stubble_handling']['stubble_threashold'])
    probability_handling = (base_yields/stub_handling_threashold).stack() #divide here then account for arable and lmu factor next - because either way is mathematically sound and this saves some manipulation.
    yields_lmus = pinp.crop['yield_by_lmu']
    arable = pinp.crop['arable'] #read in arable area df
    arable_by_soil=yields_lmus.mul(arable) #this is the lmu yield factor mul lmu arable factor
    probability_handling_lmu=arable_by_soil.reindex(probability_handling.index, axis=0, level=1).mul(probability_handling,axis=0, level=1)
    '''add the cost of stubble handling''' #there must be a better way that doesn't require soo much reindexing etc
    probability_handling_lmu.columns = pd.MultiIndex.from_product([probability_handling_lmu.columns, ['allocation']])  
    stub_cost_alloc=mac.stubble_cost_ha()
    stub_cost_alloc=stub_cost_alloc.reindex(probability_handling_lmu.columns, axis=1,level=1)
    stub_cost_alloc=stub_cost_alloc.droplevel(1, axis=1).stack()
    probability_handling_lmu=probability_handling_lmu.droplevel(1, axis=1).reindex(stub_cost_alloc.index, axis=1,level=1)
    stub_cost=probability_handling_lmu.mul(stub_cost_alloc)
    '''add to full phase df'''   
    phases_stub_cost = pd.merge(phases_df2,stub_cost, how='left', left_on=uinp.cols(), right_index = True) #[i-1 for i in uinp.cols()] little for loop is used so that merge is done based on the previous phase. since that is the yield we are interested in for stub 
    return phases_stub_cost.set_index(list(range(uinp.structure['phase_len']))).stack([1])
    
   

#print(timeit.timeit(fert_cost,number=10)/10)
#########################
#chemical               #
#########################

#app cost is the same for all lmus even if there is a scaler on rate applied because same amount of water is applied just less chem. 


#########################
#total rot cashflow     #
#########################
'''
adds up all the different cashflows for pyomo.
includes
- grain income
-fert cost (fert & application)
-stubble handling cost
-
'''
def rot_cost():
    total_cost = total_phase_fert_cost().add(phase_stubble_cost(), fill_value=0)
    return total_cost.stack().to_dict()
# jj=rot_cost()

#########################
#stubble                #
#########################
#stubble produced per kg grain harvested
def stubble_production():
    '''stubble produced by each rotation phase'''
    stubble = {}
    for crop in si.stubble_inputs['crop_stub'].keys():
        harv_index = si.stubble_inputs['crop_stub'][crop]['harvest_index']
        proportion_harvested = si.stubble_inputs['crop_stub'][crop]['proportion_grain_harv']
        stubble[crop] = 1/(harv_index * proportion_harvested)-1
    return stubble
#print (stubble_production())









