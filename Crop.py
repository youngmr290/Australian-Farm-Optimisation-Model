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


@author: young
"""

#python modules
import pandas as pd
import numpy as np
import timeit
import datetime as dt
import sys

#MUDAS modules
import UniversalInputs as uinp
import PropertyInputs as pinp
import Functions as fun
import Periods as per
import Mach as mac

  
               
        



########################
#phases                #
########################
##makes a df of all possible rotation phases
phases_df =uinp.structure['phases']
phases_df2=phases_df.copy() #make a copy so that it doesn't alter the phases df that exists outside this func
phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns, ['']])  #make the df multi index so that when it merges with other df below the indexs remanin seperate (otherwise it turn into a one leveled tuple)

##check that the rotations match the inputs. If not then quit and leave error message
if pinp.crop['user_crop_rot']:
    ### User defined
    base_yields = pinp.crop['yields']
else:        
    ### AusFarm ^need to complete
    base_yields

if len(phases_df) != len(base_yields): 
    print ('''Rotations dont match inputs.
           Things to check: 
           1. if you have generated new rotations have you re-run AusFarm?
           2. if you added new rotations in the user defined section have you re-run the rotation generator?
           3. the named ranges in for the user defined rotations and inputs are all correct''')
    sys.exit()



########################
#price                 #
########################

def farmgate_grain_price(*args):
    '''
    

    Parameters
    ----------
    *args : Boolean
        If True is entered it will calculate the grain price for the sim ie including capital sets.

    Returns
    -------
    Dataframe - used below and to calculate insurance and sup feed purchase price. If args is True then it returns the overall price accounting for proportion, this is used in sim 
                Price includes:
                -offspec grain
                -cartage cost
                -other fees ie cbh and levies
    '''
    if args:
        ##first concat the df with capital sets and the df with actual crop landuses
        grain_price_info_df=pd.concat([uinp.price['grain_price_grouped'], uinp.price['grain_price']]) #create a copy of grain price df so you dont have to reference input module each time
        ##multiplies the price and proportion of firsts and seconds for each grain, then sum to get overall price
        price_df = np.sum(np.multiply(grain_price_info_df[['firsts','seconds']], grain_price_info_df[['prop_firsts','prop_seconds']]),axis=1)
        cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance'] 
                + pinp.general['rail_cartage'] + uinp.price['flagfall'])
        tols= grain_price_info_df['grain_tolls']
        total_fees= cartage+tols
    else:
        grain_price_info_df=uinp.price['grain_price'] #create a copy of grain price df so you dont have to reference input module each time
        ##delete 'of' as it is not sold
        grain_price_info_df.drop('of')
        ##gets the price of firsts and seconds for each grain
        price_df = grain_price_info_df[['firsts','seconds']]
        ##determine cost of selling
        cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance'] 
                + pinp.general['rail_cartage'] + uinp.price['flagfall'])
        tols= grain_price_info_df['grain_tolls']
        total_fees= cartage+tols
    return price_df.sub(total_fees, axis=0).clip(0)


def grain_price(params):
    '''
    Returns
    -------
    Dict.
        Farm gate price for each grain - allocated into cash periods
        
    '''
    ##calc farm gate grain price for each cashflow period - accounts for tols and other fees
    start = uinp.price['grain_income_date']
    length = dt.timedelta(days=uinp.price['grain_income_length'])
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    farm_gate_price=farmgate_grain_price()
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    cols = pd.MultiIndex.from_product([allocation.index, farm_gate_price.columns])
    farm_gate_price = farm_gate_price.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    params['grain_price'] =  farm_gate_price.mul(allocation,axis=1,level=0).stack([0,1]).to_dict()
# a=grain_price()

##function to determine the proportion of grain in each pool 
def grain_pool_proportions(params):
    prop = uinp.price['grain_price'][['prop_firsts','prop_seconds']]
    prop.columns=['firsts','seconds']
    params['grain_pool_proportions'] = dict(prop.stack())

#########################
#yield                  #
#########################
def f_base_yield():
    '''base yield - used in stubble as well to calc foo for relativie availability'''

    ##read in yields
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_yields = pinp.crop['yields']
    else:
        ### AusFarm ^need to add code for ausfarm inputs
        base_yields
    base_yields = pd.Series(base_yields, index = phases_df.iloc[:,-1])
    return base_yields

def rot_yield(params=False):
    '''
    Returns
    ----------
    Dataframe - passed to pyomo and used to calc insurance & stubble handling
        Grain yield for each rotation.
        Yield includes:
        -arable area
        -seeding rate (if farmers use thier own seed)
        -lmu factor
        -frost
    '''
    ##base yields
    base_yields = f_base_yield()
    ##colate other info
    yields_lmus = pinp.crop['yield_by_lmu'] #soil yield factor
    seeding_rate = pinp.crop['seeding_rate'].mul(pinp.crop['own_seed'],axis=0)#seeding rate adjusted by if the farmer is using their own seed from last yr
    frost = pinp.crop['frost'] #frost
    arable = pinp.crop['arable'].stack().droplevel(0) #read in arable area df
    ##calculate yield - base yield * arable area * frost * lmu factor - seeding rate
    yield_arable_by_soil=yields_lmus.mul(arable).mul(1-frost) #mul arable area to the the lmu factor (easy because dfs have the same axis's). THen mul frost
    yields=yield_arable_by_soil.reindex(base_yields.index, axis=0).mul(base_yields,axis=0) #reindes and mul with base yields
    seeding_rate=seeding_rate.reindex(yields.index, axis=0) #minus seeding rate
    yields=yields.sub(seeding_rate,axis=0).clip(lower=0) #we don't want negitive yields so clip at 0 (if any values are neg they become 0)
    ##add the rotation to index - current landuse is also part of the index it is required for pyomo, also used in the insurance calc to multiply on.   
    yields.set_index(phases_df.index, append=True, inplace=True)
    yields.index = yields.index.swaplevel(0, 5)
    if params:
        params['rot_yield'] = yields.stack().to_dict()
    else:
        return yields.stack()
    # return rot_yields.set_index(list(range(uinp.structure['phase_len']))).stack() #need to use the multiindex to create a multidimensional param for pyomo so i can split it down when indexing
# a=rot_yield().to_dict()


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
# t_allocation=fert_cost_allocation()


def fert_req():
    '''
    Returns
    ----------
    Dataframe; used to calc fert cost in the next function
        Fert required by 1ha of each phases (kg/ha) after accounting for arable area

    '''
    ##read in chem by soil
    fert_by_soil = pinp.crop['fert_by_lmu'] 
    ##read in fert
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_fert = pinp.crop['fert'].reset_index()
        base_fert=base_fert.set_index([phases_df.iloc[:,-1]])
    else:        
        ### AusFarm ^need to add code for ausfarm inputs
        base_fert
        base_fert = pd.DataFrame(base_fert, index = phases_df.iloc[:,-1])  #make the current landuse the index
    ##add the full rotation index eg EEEEEb - done here because merge sorts the order, has to be a col because otherwise cant merge
    base_fert['rot']=phases_df.index
    ##add the fixed fert 
    fixed_fert = pinp.crop['fixed_fert']
    base_fert = pd.merge(base_fert, fixed_fert, how='left', left_index=True, right_index = True) 
    ##replace index
    base_fert.set_index('rot', inplace=True)
    ## adjust the fert req for each rotation by lmu
    fert_by_soil = fert_by_soil.stack() #read in fert by soil
    fert=base_fert.mul(fert_by_soil,axis=1,level=0).stack()
    ##account for arable area
    arable = pinp.crop['arable'].squeeze() #read in arable area df
    fert=fert.mul(arable,axis=0,level=1) #add arable to df
    return fert
# fert=fert_req()

def f_fert_passes():
    '''passes over arable area'''
    ####read in passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        fert_passes = pinp.crop['fert_passes'].reset_index()
        fert_passes = fert_passes.set_index([phases_df.iloc[:,-1]])  #make the current landuse the index
    else:
        ### AusFarm
        fert_passes
        fert_passes = pd.DataFrame(fert_passes, index = phases_df.iloc[:,-1])  #make the current landuse the index
    ##add the full rotation index eg EEEEEb - done here because merge sorts the order, has to be a col because otherwise cant merge
    fert_passes['rot']=phases_df.index
    ####add the fixed fert
    fixed_fert_passes = pinp.crop['fixed_fert_passes']
    fert_passes = pd.merge(fert_passes, fixed_fert_passes, how='left', left_index=True, right_index = True)
    ##replace index
    fert_passes.set_index('rot', inplace=True)
    ##adjust fert passes by arable area
    arable = pinp.crop['arable'].squeeze()
    index = pd.MultiIndex.from_product([fert_passes.index, arable.index])
    fert_passes = fert_passes.reindex(index, axis=0,level=0)
    fert_passes=fert_passes.mul(arable,axis=0,level=1)
    return fert_passes


def fert_cost(r_vals):
    '''
    Returns
    ----------
    Dataframe; summed with other fert cashflow items at the end of this section 
        Calcs;
        - cost of actual fertiliser for each rotation phase (including transport)
        - Application cost per tonne ($/rotation)
        - Application cost per ha ($/rotation)
    '''
    ##call functions and read inputs used within this function
    fertreq = fert_req()
    allocation = fert_cost_allocation()
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost 
    ##calc cost of actual fertiliser
    total_cost = allocation.mul(cost+transport).stack() #total cost = fert cost and transport. Here we also account for cost allocation
    phase_fert_cost_t=fertreq.mul(total_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg, sum the cost of all the ferts
    ##aplication cost per tonne
    application_cost = allocation.mul(mac.fert_app_cost_t()).stack() #mul app cost per tonne with fert cost allocation
    fert_app_cost_t=fertreq.mul(application_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg
    ##app cost per ha 
    ###call passes function (it has to be a seperate function because it is used in crplabour.py as well
    fert_passes = f_fert_passes()
    ###add the cost for each pass
    fert_cost_ha = allocation.mul(mac.fert_app_cost_ha()).stack() #cost for 1 pass for each fert.
    fert_app_cost_ha = fert_passes.mul(fert_cost_ha,axis=1,level=1).sum(axis=1, level=0) 
    ##combine all costs - fert, app per ha and app per tonne    
    fert_cost_total= pd.concat([phase_fert_cost_t,fert_app_cost_t, fert_app_cost_ha],axis=1).sum(axis=1,level=0) #must include level so that all cols dont sum, had to switch this from .add to concat because for some reason on multiple itterations of the model add stoped working
    return fert_cost_total

def f_nap_fert_req():
    '''fert applied to non arable pasture area'''
    arable = pinp.crop['arable'].squeeze()  # read in arable area df
    fertreq_na = pinp.crop['nap_fert']
    fertreq_na = fertreq_na.mul(1 - arable)
    arr=[list(uinp.structure['All_pas']),list(fertreq_na.index)] #create multi index from lmu and pasture landuse code
    inx = pd.MultiIndex.from_product(arr)
    fertreq_na = fertreq_na.reindex(inx,axis=0,level=1)
    fertreq_na = pd.merge(phases_df2, fertreq_na.unstack(), how='left', left_on=uinp.cols()[-1], right_index = True) #merge with all the phases, requires because different phases have different application passes
    fertreq_na = fertreq_na.drop(list(range(uinp.structure['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return fertreq_na

def f_nap_fert_passes():
    '''hectares spread on non arable area'''
    ##passes over non arable pasture area (only for pasture phases becasue for pasture the non arable areas also recieve fert)
    passes_na = pinp.crop['nap_passes']
    arable = pinp.crop['arable'].squeeze() #eed to adjust for only non arable area
    passes_na= passes_na.mul(1-arable) #adjust for the non arable area
    arr=[list(uinp.structure['All_pas']),list(passes_na.index)] #create multi index from lmu and pasture landuse code
    inx = pd.MultiIndex.from_product(arr)
    passes_na = passes_na.reindex(inx,axis=0,level=1)
    passes_na = pd.merge(phases_df2, passes_na.unstack(), how='left', left_on=uinp.cols()[-1], right_index = True) #merge with all the phases, requires because different phases have different application passes
    passes_na = passes_na.drop(list(range(uinp.structure['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return passes_na

def nap_fert_cost():
    '''
    
    Returns
    -------
    Dataframe- to be added to total costs at the end.
        Fert applied to non arable pasture - currently setup so that only pasture phases get fert on the non arable areas hence it needs to be a seperate function.
    '''
    allocation = fert_cost_allocation()
    ##fert cost
    fertreq = f_nap_fert_req()
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost
    total_cost = allocation.mul(cost+transport).stack() #total cost = fert cost and transport. Here we also account for cost allocation
    fert_cost_t = fertreq.mul(total_cost, axis=1, level=1)/1000  #div by 1000 to convert to $/kg
    ##application cost per tonne
    app_cost_t = fertreq.mul(mac.fert_app_cost_t(), axis=1)/1000  #div by 1000 to convert to $/kg
    ##application cost per ha
    passes = f_nap_fert_passes()
    app_cost_ha = passes.mul(mac.fert_app_cost_ha(), axis=1) #cost for 1 pass for each fert.
    ##total application cost in each cash period
    total_app_cost = (app_cost_t+app_cost_ha).mul(allocation.stack(), axis=1, level=1)
    ##total fert and app cost
    nap_fert_cost = fert_cost_t+total_app_cost
    nap_fert_cost = nap_fert_cost.sum(axis=1, level=0) #mul app cost per tonne with fert cost allocation
    return nap_fert_cost

def total_fert_req(param):
    '''returns the total fert req after accounting for arable area.
       this is used in the LabourCropPyomo'''
    fertreq_arable = fert_req()
    ##fert required on the non arable areas - only for pasture phases, so need to add pasture as index
    fert_na = f_nap_fert_req()
    ##add fert for arable area and fert for nonarable area
    fert_total = pd.concat([fertreq_arable, fert_na], axis=1).sum(axis=1, level=0)
    param['fert_req'] = fert_total.stack().to_dict()


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
    '''
    Yields
    ------
    Dataframe
        Cost to handle stubble for each rotation - summed with other costs in final function below.
        *note - arable area accounted for in the yield (it is the same as accounting for it at the end ie yield x 0.8 / threshold x cost == yield / threshold x cost x 0.8)
    '''
    ##first calculate the probability of a rotation phase needing stubble handling
    base_yields = rot_yield()
    stub_handling_threashold = pd.Series(pinp.stubble['stubble_handling']['stubble_threshold'])*1000  #have to convert to kg to match base yield
    probability_handling = base_yields.div(stub_handling_threashold, level = 1) #divide here then account for lmu factor next - because either way is mathematically sound and this saves some manipulation.
    probability_handling = probability_handling.droplevel(1).unstack()
    ##add the cost - this needs to be flexible because the cost may be over multiple periods
    stub_cost_alloc=mac.stubble_cost_ha().squeeze(axis=1)
    cols = pd.MultiIndex.from_product([probability_handling.columns, stub_cost_alloc.index])  
    handling_cost = probability_handling.reindex(cols,axis =1 , level = 0).mul(stub_cost_alloc,axis=1,level=1)
    return handling_cost.stack([0])
# t_stubcost=phase_stubble_cost()   

#print(timeit.timeit(fert_cost,number=10)/10)
#########################
#chemical               #
#########################

def chem_cost_allocation():
    '''
    Returns
    ----------
    Dataframe; multiplied with chem cashflows 
        Determines the cashflow period allocation for costs associated with each different chem
    '''
    start_df = pinp.crop['chem_info']['app_date'] #needed for allocation func
    length_df = pinp.crop['chem_info']['app_len'].astype('timedelta64[D]') #needed for allocation func
    p_dates = per.cashflow_periods()['start date'] #needed for allocation func
    p_name = per.cashflow_periods()['cash period'] #needed for allocation func
    return fun.period_allocation2(start_df, length_df, p_dates, p_name)
# t_allocation=chem_cost_allocation()
    
def chem_application():
    '''number of chemical applications for each rotation'''
    ##read in chem passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_chem = pinp.crop['chem'].reset_index()
        base_chem = base_chem.set_index([phases_df.iloc[:,-1]])  #make the current landuse the index
    else:
        ### AusFarm ^need to add code for ausfarm inputs
        base_chem
        base_chem = pd.DataFrame(base_chem, index = phases_df.iloc[:,-1])  #make the current landuse the index
    ##add proper rotation index - need to add this here because the multiplication step re-orders the df
    base_chem['rot']=phases_df.index
    return base_chem

def chem_cost():  
    '''
    Returns
    ----------
    Dataframe: Total cost of chemical and application - summed with other cashflow items at the end of this section 
        -Calcs the raw chem cost for each rotation phase (ie doesn't include application)
        -Application cost per ha ($/rotation)
        -Arable area accounted here
    '''
    ##read in neccessary bits and adjust indexed
    i_chem_cost = pinp.crop['chem_cost']
    chem_by_soil = pinp.crop['chem_by_lmu'] #read in chem by soil
    ##number of applications for each rotation
    base_chem = chem_application()
    ##adjust for the cost eg cost per application * number of applications. This is a bit messy because the order of the df changes therefore need to add the full rotation name, but for merge it can be an index so i add it as a col then merge then add it as index. ^maybe there is a cleaner way to do this.
    merge_chem = base_chem.merge(i_chem_cost, left_index=True, right_index=True, how='left')
    merge_chem.set_index('rot', inplace=True)
    chem_cost = merge_chem.iloc[:,0:len(i_chem_cost.columns)].values * merge_chem.iloc[:,len(i_chem_cost.columns):].values
    chem_cost = pd.DataFrame(chem_cost, index = merge_chem.index, columns = i_chem_cost.columns )
    ## adjust the chem cost for each rotation by lmu
    chem_by_soil1 = chem_by_soil.stack()
    chem_cost=chem_cost.mul(chem_by_soil1,axis=1,level=0)
    ##application cost
    base_chem.set_index('rot', inplace=True)
    app_cost_ha = base_chem * mac.chem_app_cost_ha()
    # app_cost_ha.set_index(phases_df.index, append=True, inplace=True)
    app_cost_ha = app_cost_ha.reindex(chem_cost.columns, axis=1,level=0) #reindex to get lmu 
    ##add application cost and chem cost
    total_cost = chem_cost.add(app_cost_ha)
    ##add cashflow periods and sum across each chem
    c_chem_allocation = chem_cost_allocation().stack()
    chem_cost = total_cost.stack().reindex(c_chem_allocation.index, axis=1,level=1).mul(c_chem_allocation, axis=1).sum(axis=1, level=0)#first stack is required so that reindexing can occur (ie cant reindex a multi index with a multi index)
    return chem_cost


#########################
#misc cost              #
#########################
def seedcost():
    '''
    Returns
    ----------
    Dataframe to add with other rotation costs
        Misc costs
        - seed treatment
        - raw seed cost (incurred if seed is purchased as aposed to using last yrs seed)
        - crop insurance
        *note - arable area accounted for in the final function that sums all costs
    '''
    seeding_rate = pinp.crop['seeding_rate']
    seeding_cost = pinp.crop['seed_info']['Seed cost'] #this is 0 if the seed is sourced from last yrs crop ie cost is accounted for by minusing from the yield
    grading_cost = pinp.crop['seed_info']['Grading'] 
    percent_graded = pinp.crop['seed_info']['Percent Graded'] 
    cost1 = pinp.crop['seed_info']['Cost1'] #cost ($/l) for dressing 1 
    cost2 = pinp.crop['seed_info']['Cost2'] #cost ($/l) for dressing 2 
    rate1 = pinp.crop['seed_info']['Rate1'] #rate (ml/100g) for dressing 1
    rate2 = pinp.crop['seed_info']['Rate2'] #rate (ml/100g) for dressing 2
    percent_dressed = pinp.crop['seed_info']['percent dressed'] #rate (ml/100g) for dressing 2
    ##overall seed grading cost per tonne
    cost = grading_cost * percent_graded
    ##add seed cost
    cost = cost + seeding_cost
    ##add dressing 1 - need to convert to $/t first
    cost = cost + (cost1*rate1/100 * percent_dressed)
    ##add dressing 2 - need to convert to $/t first
    cost = cost + (cost2*rate2/100 * percent_dressed)
    ##account for seeding rate to determine actual cost (divide by 1000 to convert cost to kg)
    phase_cost = seeding_rate.mul(cost/1000,axis=0)
    ##cost allocation
    start = per.wet_seeding_start_date()
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start)
    ##add cashflow period to col index
    phase_cost.columns = pd.MultiIndex.from_product([phase_cost.columns, [allocation]])
    ##merge
    rot_cost = pd.merge(phases_df2, phase_cost, how='left', left_on=uinp.cols()[-1], right_index = True)
    return rot_cost.drop(list(range(uinp.structure['phase_len'])), axis=1).stack([0])

def insurance():
    '''
    Returns
    ----------
    Dataframe to add with other rotation costs
        Misc costs
        - crop insurance
        *note - arable area is already counted for by the yield calculation
    '''
    ##first need to combine each grain pool to get average price
    ave_price=np.multiply(farmgate_grain_price(),uinp.price['grain_price'][['prop_firsts','prop_seconds']]).sum(axis=1)#np multiply doen't look at the column names and indexs
    insurance=ave_price*uinp.price['grain_price']['insurance']/100  #div by 100 because insurance is a percent
    rot_insurance = rot_yield().mul(insurance, axis=0, level = 1)/1000 #divide by 1000 to convert yield to tonnes    
    rot_insurance = rot_insurance.droplevel(1).unstack()
    ##merge - required to get the agregated phase index
    # phases_df.index.rename('Index', inplace=True) #have to rename index so i can do the next step otherwise col and index had the same name '0'
    # rot_cost = pd.merge(phases_df, rot_insurance.unstack(), how='left', left_on=[*range(uinp.structure['phase_len'])], right_index = True)
    # rot_cost = pd.merge(phases_df, rot_insurance.unstack(), how='left', left_index=True, right_index = True)
    ##cost allocation
    start = uinp.price['crp_insurance_date']
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start)
    ##add cashflow period to col index
    rot_insurance.columns = pd.MultiIndex.from_product([rot_insurance.columns, [allocation]])
    return rot_insurance.stack([0])




#########################
#total rot cost         #
#########################
'''
adds up all the different cashflows for pyomo.
includes
-fert cost (fert & application)
-stubble handling cost
- chem cost (inc application)
-seed cost
-crop insurance cost
'''
def rot_cost(params, r_vals):
    cost = pd.concat([fert_cost(r_vals),nap_fert_cost(),chem_cost(),seedcost()],axis=1).sum(axis=1,level=0)

    #^remove after checkign it works
    ##adjust for arable area
    # arable = pinp.crop['arable'].squeeze() #read in arable area df
    # cost=cost.mul(arable, axis=0, level=1)

    ##add insurance & stubble - it has already been adjusted by arable area because of the yield
    cost = pd.concat([cost, insurance(),phase_stubble_cost()],axis=1).sum(axis=1,level=0)#, fill_value=0)

    # ##add non arable pasture cost
    # nap_cost = nap_fert_cost()
    # cost = pd.concat([cost, nap_cost],axis=1).sum(axis=1,level=0)#, fill_value=0)

    ##add cont tedera costs = combination of resown and normal
    if 'tedera' in uinp.structure['pastures']:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ###determine the proportion of the time tc and jc are resown - this is used as a weighting to determine the input costs
        tc_inx = germ_df.iloc[:,-3].isin(['tc']) #checks current phase for tc
        tc_frequency = germ_df.loc[tc_inx,'resown'] #get frequency of resowing tc
        jc_inx = germ_df.iloc[:,-3].isin(['jc']) #checks current phase for resown jc
        jc_frequency = germ_df.loc[jc_inx,'resown'] #get frequency of resowing jc
        ###combine the costs of t and tr with the frequency of tc resowing to get the cost for tc
        #^what about lmu? what if landuse below doesnt exist and need to use a different one... would be better to average all landuses on a given lmu...
        t_cost=cost.loc[['YYEETt']]*(1-tc_frequency[0])
        tr_cost=cost.loc[['YYEEEtr']]*tc_frequency[0]
        tc_cost = np.add(t_cost , tr_cost)
        cost.loc[['tctctctctctc']] = tc_cost.values
        ###combine the costs of j and jr with the frequency of tc resowing to get the cost for tc
        j_cost=cost.loc[['YYEEJj']]*(1-jc_frequency[0])
        jr_cost=cost.loc[['YYEEEjr']]*jc_frequency[0]
        jc_cost = np.add(j_cost , jr_cost)
        cost.loc[['jcjcjcjcjcjc']] = jc_cost.values
    ##add cont lucerne costs
    if 'lucerne' in uinp.structure['pastures']:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ###determine the proportion of the time uc and xc are resown - this is used as a weighting to determine the input costs
        uc_inx = germ_df.iloc[:,-3].isin(['uc']) #checks current phase for uc
        uc_frequency = germ_df.loc[uc_inx,'resown'] #get frequency of resowing
        xc_inx = germ_df.iloc[:,-3].isin(['xc']) #checks current phase for resown pasture
        xc_frequency = germ_df.loc[xc_inx,'resown'] #get frequency of resowing
        ###combine the costs of t and tr with the frequency of uc resowing to get the cost for uc
        u_cost=cost.loc[['YYEEUu']]*(1-uc_frequency[0])
        ur_cost=cost.loc[['YYEEEur']]*uc_frequency[0]
        uc_cost = np.add(u_cost , ur_cost)
        cost.loc[['ucucucucucuc']] = uc_cost.values
        ###combine the costs of j and jr with the frequency of xc resowing to get the cost for xc
        x_cost=cost.loc[['YYEEXx']]*(1-xc_frequency[0])
        xr_cost=cost.loc[['YYEEExr']]*xc_frequency[0]
        xc_cost = np.add(x_cost , xr_cost)
        cost.loc[['xcxcxcxcxcxc']] = xc_cost.values

    params['rot_cost'] = cost.stack().to_dict()

# jj=rot_cost()

#########################
#stubble                #
#########################
#stubble produced per kg grain harvested, used in stubble.py as well
def stubble_production(*params):
    '''stubble produced by each rotation phase
        kgs of dry matter'''
    stubble = pd.DataFrame()
    for crop in pinp.stubble['harvest_index'].index:
        harv_index = pinp.stubble['harvest_index'].loc[crop,'hi']
        proportion_harvested = pinp.stubble['proportion_grain_harv'].loc[crop,'prop']
        stubble.loc[crop,'a'] = 1/(harv_index * proportion_harvested)-1 #subtract 1 to account for the tonne of grain that was harvested
    if params:
        params[0]['stubble_production'] = stubble.stack().to_dict()
    else: return stubble.stack().to_dict()
#print (stubble_production())


#################
#sow            #
#################
    
def crop_sow(params):
    '''
    Returns
    -------
    Dict for pyomo.
        Crop sow (wet or dry or spring) requirment for each rot phase

    '''
    ##sow = arable area
    arable = pinp.crop['arable']
    cropsow = arable.reindex(pd.MultiIndex.from_product([uinp.structure['C'],arable.index]), axis=0, level=1).droplevel(1)
    ##merge to rot phases
    cropsow = pd.merge(phases_df, cropsow, how='left', left_on=uinp.cols()[-1], right_index = True)
    ##add current crop to index
    cropsow.set_index(uinp.cols()[-1], append=True, inplace=True)
    params['crop_sow'] = cropsow.drop(list(range(uinp.structure['phase_len']-1)), axis=1).stack().to_dict()


