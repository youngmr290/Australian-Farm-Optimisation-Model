# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:05:00 2019

module: crop/rotation module

key: green section title is major title 
     '#' around a title is a minor section title
     std '#' comment about a given line of code
 
Version Control:
Version     Date        Person  Change
1.1         27Dec19     MRY     Updated rotation phase function, and crop functions (individual costs separated and one funct to sum them all at the bottom)
1.1         28Dec19     MRY     alterd stubble handling to work with new rotation system (stubble handling cost is now associated with the current phase rather than the previous phase)
1.2         16Jan20     M       separated grain price from yield - it is now separate so it can be combined with yield penalty and crop grazing yield penalty before being multiplied by price.

Known problems:
Fixed   Date        ID by   Problem
        28/12/19    MRY     No input optimisation (solution is to convert crop into a simulation)
        28/12/19    MRY     All lmus receive a fert app cost per ha even if the fert applied to that lmu is 0 (generally not a problem bevause all lmus receive some fert. and the per tonne cost accounts for variable lmu rates)
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

#AFO modules
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
phases_df2.columns = pd.MultiIndex.from_product([phases_df2.columns, ['']])  #make the df multi index so that when it merges with other df below the indexs remanin separate (otherwise it turn into a one leveled tuple)

##check that the rotations match the inputs. If not then quit and leave error message
if pinp.crop['user_crop_rot']:
    ### User defined
    base_yields = pinp.crop['yields']
else:        
    ### AusFarm ^need to complete
    base_yields

if len(phases_df) != len(base_yields): 
    print ('''Rotations don't match inputs.
           Things to check: 
           1. if you have generated new rotations have you re-run AusFarm?
           2. if you added new rotations in the user defined section have you re-run the rotation generator?
           3. the named ranges in for the user defined rotations and inputs are all correct''')
    sys.exit()



########################
#price                 #
########################

def f_farmgate_grain_price(r_vals={}):
    '''

    Returns
    -------
    Dataframe - used below and to calculate insurance and sup feed purchase price.
                Price includes:
                -offspec grain
                -cartage cost
                -other fees ie cbh and levies
    '''
    grain_price_info_df=uinp.price['grain_price'] #create a copy of grain price df so you don't have to reference input module each time
    ##gets the price of firsts and seconds for each grain
    price_df = grain_price_info_df[['firsts','seconds']]
    ##determine cost of selling
    cartage=(grain_price_info_df['cartage_km_cost']*pinp.general['road_cartage_distance']
            + pinp.general['rail_cartage'] + uinp.price['flagfall'])
    tols= grain_price_info_df['grain_tolls']
    total_fees= cartage+tols
    farmgate_price = price_df.sub(total_fees, axis=0).clip(0)
    r_vals['farmgate_price'] = farmgate_price
    return farmgate_price


def grain_price(params, r_vals):
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
    farm_gate_price=f_farmgate_grain_price(r_vals)
    allocation=fun.period_allocation(p_dates, p_name, start, length).set_index('period').squeeze()
    cols = pd.MultiIndex.from_product([allocation.index, farm_gate_price.columns])
    farm_gate_price = farm_gate_price.reindex(cols, axis=1,level=1)#adds level to header so i can mul in the next step
    grain_price =  farm_gate_price.mul(allocation,axis=1,level=0)
    params['grain_price'] =  farm_gate_price.mul(allocation,axis=1,level=0).stack([0,1]).to_dict()
    r_vals['grain_price'] =  grain_price.T
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
    yields=yields.sub(seeding_rate,axis=0).clip(lower=0) #we don't want negative yields so clip at 0 (if any values are neg they become 0)
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
2) fert requirement for each rot phase
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
        base_fert=base_fert.set_index([phases_df.index,phases_df.iloc[:,-1]])
    else:        
        ### AusFarm ^need to add code for ausfarm inputs
        base_fert
        base_fert = pd.DataFrame(base_fert, index = phases_df.iloc[:,-1])  #make the rotation and current landuse the index
    ##rename index
    base_fert.index.rename(['rot','landuse'],inplace=True)
    ##add the fixed fert
    fixed_fert = pinp.crop['fixed_fert']
    base_fert = pd.merge(base_fert, fixed_fert, how='left', left_on='landuse', right_index = True)
    ##add cont pasture fert req
    base_fert = f_cont_pas(base_fert.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    base_fert = base_fert.droplevel(0,axis=0)
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
        fert_passes = fert_passes.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the rotation and current landuse the index
    else:
        ### AusFarm
        fert_passes
        fert_passes = pd.DataFrame(fert_passes, index = phases_df.iloc[:,-1])  #make the current landuse the index
    ##rename index
    fert_passes.index.rename(['rot','landuse'],inplace=True)
    ####add the fixed fert
    fixed_fert_passes = pinp.crop['fixed_fert_passes']
    fert_passes = pd.merge(fert_passes, fixed_fert_passes, how='left', left_on='landuse', right_index = True)
    ##add cont pasture fert passes
    fert_passes = f_cont_pas(fert_passes.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    fert_passes = fert_passes.droplevel(0, axis=0)
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
    phase_fert_cost=fertreq.mul(total_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg, sum the cost of all the ferts
    r_vals['phase_fert_cost'] = phase_fert_cost
    ##aplication cost per tonne
    application_cost = allocation.mul(mac.fert_app_cost_t()).stack() #mul app cost per tonne with fert cost allocation
    fert_app_cost_t=fertreq.mul(application_cost/1000,axis=1,level=1).sum(axis=1, level=0) #div by 1000 to convert to $/kg
    ##app cost per ha 
    ###call passes function (it has to be a separate function because it is used in crplabour.py as well
    fert_passes = f_fert_passes()
    ###add the cost for each pass
    fert_cost_ha = allocation.mul(mac.fert_app_cost_ha()).stack() #cost for 1 pass for each fert.
    fert_app_cost_ha = fert_passes.mul(fert_cost_ha,axis=1,level=1).sum(axis=1, level=0)
    r_vals['fert_app_cost'] = fert_app_cost_ha
    ##combine all costs - fert, app per ha and app per tonne    
    fert_cost_total= pd.concat([phase_fert_cost,fert_app_cost_t, fert_app_cost_ha],axis=1).sum(axis=1,level=0) #must include level so that all cols don't sum, had to switch this from .add to concat because for some reason on multiple iterations of the model add stoped working
    return fert_cost_total

def f_nap_fert_req():
    '''fert applied to non arable pasture area'''
    arable = pinp.crop['arable'].squeeze()  # read in arable area df
    fertreq_na = pinp.crop['nap_fert'].reset_index().set_index(['fert','landuse'])
    fertreq_na = fertreq_na.mul(1 - arable)
    ##add cont pasture fert req
    fertreq_na = f_cont_pas(fertreq_na.unstack(0))
    ##merge with full df
    fertreq_na = pd.merge(phases_df2, fertreq_na, how='left', left_on=uinp.cols()[-1], right_index = True) #merge with all the phases, requires because different phases have different application passes
    fertreq_na = fertreq_na.drop(list(range(uinp.structure['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return fertreq_na

def f_nap_fert_passes():
    '''hectares spread on non arable area'''
    ##passes over non arable pasture area (only for pasture phases because for pasture the non arable areas also receive fert)
    passes_na = pinp.crop['nap_passes'].reset_index().set_index(['fert','landuse'])
    arable = pinp.crop['arable'].squeeze() #eed to adjust for only non arable area
    passes_na= passes_na.mul(1-arable) #adjust for the non arable area
    ##add cont pasture fert req
    passes_na = f_cont_pas(passes_na.unstack(0))
    ##merge with full df
    passes_na = pd.merge(phases_df2, passes_na, how='left', left_on=uinp.cols()[-1], right_index = True) #merge with all the phases, requires because different phases have different application passes
    passes_na = passes_na.drop(list(range(uinp.structure['phase_len'])), axis=1, level=0).stack([0]) #drop the segregated landuse cols
    return passes_na

def nap_fert_cost(r_vals):
    '''
    
    Returns
    -------
    Dataframe- to be added to total costs at the end.
        Fert applied to non arable pasture - currently setup so that only pasture phases get fert on the non arable areas hence it needs to be a separate function.
    '''
    allocation = fert_cost_allocation()
    ##fert cost
    fertreq = f_nap_fert_req()
    cost=uinp.price['fert_cost'].squeeze()
    transport=uinp.price['fert_cartage_cost']  #transport cost
    total_cost = allocation.mul(cost+transport).stack() #total cost = fert cost and transport. Here we also account for cost allocation
    fert_cost = fertreq.mul(total_cost, axis=1, level=1)/1000  #div by 1000 to convert to $/kg
    r_vals['nap_phase_fert_cost'] = fert_cost.sum(axis=1, level=0) #sum all fertilisers
    ##application cost per tonne
    app_cost_t = fertreq.mul(mac.fert_app_cost_t(), axis=1)/1000  #div by 1000 to convert to $/kg
    ##application cost per ha
    passes = f_nap_fert_passes()
    app_cost_ha = passes.mul(mac.fert_app_cost_ha(), axis=1) #cost for 1 pass for each fert.
    ##total application cost in each cash period
    total_app_cost = (app_cost_t+app_cost_ha).mul(allocation.stack(), axis=1, level=1)
    r_vals['nap_fert_app_cost'] = total_app_cost.sum(axis=1, level=0) #sum all fertilisers
    ##total fert and app cost
    nap_fert_cost = fert_cost+total_app_cost
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

def phase_stubble_cost(r_vals):
    '''
    Yields
    ------
    Dataframe
        Cost to handle stubble for each rotation - summed with other costs in final function below.
        *note - arable area accounted for in the yield (it is the same as accounting for it at the end ie yield x 0.8 / threshold x cost == yield / threshold x cost x 0.8)
    '''
    ##first calculate the probability of a rotation phase needing stubble handling
    base_yields = rot_yield()
    stub_handling_threshold = pd.Series(pinp.stubble['stubble_handling'], index=pinp.crop['start_harvest_crops'].index)*1000  #have to convert to kg to match base yield
    probability_handling = base_yields.div(stub_handling_threshold, level = 1) #divide here then account for lmu factor next - because either way is mathematically sound and this saves some manipulation.
    probability_handling = probability_handling.droplevel(1).unstack()
    ##add the cost - this needs to be flexible because the cost may be over multiple periods
    stub_cost_alloc=mac.stubble_cost_ha().squeeze(axis=1)
    cols = pd.MultiIndex.from_product([probability_handling.columns, stub_cost_alloc.index])  
    handling_cost = probability_handling.reindex(cols,axis =1 , level = 0).mul(stub_cost_alloc,axis=1,level=1)
    stub_cost = handling_cost.stack([0])
    r_vals['stub_cost'] = stub_cost
    return stub_cost
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
    
def f_chem_application():
    '''number of applications for each chemical for each rotation
        Arable area accounted for here
    '''
    ##read in chem passes
    if pinp.crop['user_crop_rot']:
        ### User defined
        base_chem = pinp.crop['chem'].reset_index()
        base_chem = base_chem.set_index([phases_df.index, phases_df.iloc[:,-1]])  #make the current landuse the index
    else:
        ### AusFarm ^need to add code for ausfarm inputs
        base_chem
        base_chem = pd.DataFrame(base_chem, index = [phases_df.index, phases_df.iloc[:,-1]])  #make the current landuse the index
    ##rename index
    base_chem.index.rename(['rot','landuse'],inplace=True)
    ##add cont pasture fert req
    base_chem = f_cont_pas(base_chem.unstack(0)).stack() #unstack for function then stack
    ##drop landuse from index
    base_chem = base_chem.droplevel(0, axis=0)
    ##arable area.
    arable = pinp.crop['arable'].squeeze()
    #adjust chem passes by arable area
    index = pd.MultiIndex.from_product([base_chem.index, arable.index])
    base_chem = base_chem.reindex(index, axis=0,level=0)
    base_chem=base_chem.mul(arable,axis=0,level=1)
    return base_chem

def f_chem_cost(r_vals):
    '''
    Returns
    ----------
    Dataframe: Total cost of chemical and application - summed with other cashflow items at the end of this section 
        -Calcs the raw chem cost for each rotation phase (ie doesn't include application)
        -Application cost per ha ($/rotation)
        -Arable area accounted for in application funciton above
    '''
    ##read in necessary bits and adjust indexed
    i_chem_cost = pinp.crop['chem_cost']
    chem_by_soil = pinp.crop['chem_by_lmu'] #read in chem by soil
    ##add cont pasture to chem cost array
    i_chem_cost = f_cont_pas(i_chem_cost)
    ##number of applications for each rotation
    chem_applications = f_chem_application()
    ##determine the total chemical cost of each rotation. eg: chem cost per application * number of applications
    index = pd.MultiIndex.from_arrays([phases_df.iloc[:,-1], phases_df.index]) #add phase letter to index so it can be merged with the cost per application for each phase
    t_chem_applications = chem_applications.unstack().reindex(index, axis=0, level=0).stack() #reindex so the array has same axis so it can be multiplied
    ###Merge the phase cost with the rotation application number then mul to get final cost. ^i couldnt get reinexing to work so i have ended up merging the mul instead (i think there should be a way to reindex i_cost so it is the same as chem_applications then mul).
    merge_chem = t_chem_applications.merge(i_chem_cost, left_on=[5], right_index=True, how='left')
    chem_cost = merge_chem.iloc[:,0:len(i_chem_cost.columns)].values * merge_chem.iloc[:,len(i_chem_cost.columns):].values
    chem_cost = pd.DataFrame(chem_cost, index=chem_applications.index, columns = i_chem_cost.columns)
    ## adjust the chem cost for each rotation by lmu
    chem_by_soil1 = chem_by_soil.stack()
    chem_cost=chem_cost.unstack().mul(chem_by_soil1,axis=1,level=0).stack()
    ##application cost
    app_cost_ha = chem_applications * mac.chem_app_cost_ha()
    ##add cashflow periods and sum across each chem - have to do this to both chem cost and application so i can report them separately
    c_chem_allocation = chem_cost_allocation().stack()
    chem_cost = chem_cost.mul(c_chem_allocation, axis=1,level=1).sum(axis=1, level=0)#first stack is required so that reindexing can occur (ie cant reindex a multi index with a multi index)
    app_cost_ha = app_cost_ha.mul(c_chem_allocation, axis=1,level=1).sum(axis=1, level=0)#first stack is required so that reindexing can occur (ie cant reindex a multi index with a multi index)
    r_vals['chem_cost'] = chem_cost
    r_vals['chem_app_cost_ha'] = app_cost_ha
    ##add application cost and chem cost
    total_cost = chem_cost.add(app_cost_ha)
    return chem_cost


#########################
#misc cost              #
#########################
def seedcost(r_vals):
    '''
    Returns
    ----------
    Dataframe to add with other rotation costs
        Misc costs includes:
        - seed treatment
        - raw seed cost (incurred if seed is purchased as aposed to using last yrs seed)
        - crop insurance
        - arable area
    '''
    ##inputs
    seeding_rate = pinp.crop['seeding_rate']
    seeding_cost = pinp.crop['seed_info']['Seed cost'] #this is 0 if the seed is sourced from last yrs crop ie cost is accounted for by minusing from the yield
    grading_cost = pinp.crop['seed_info']['Grading'] 
    percent_graded = pinp.crop['seed_info']['Percent Graded'] 
    cost1 = pinp.crop['seed_info']['Cost1'] #cost ($/l) for dressing 1 
    cost2 = pinp.crop['seed_info']['Cost2'] #cost ($/l) for dressing 2 
    rate1 = pinp.crop['seed_info']['Rate1'] #rate (ml/100g) for dressing 1
    rate2 = pinp.crop['seed_info']['Rate2'] #rate (ml/100g) for dressing 2
    percent_dressed = pinp.crop['seed_info']['percent dressed'] #rate (ml/100g) for dressing 2
    arable = pinp.crop['arable'].squeeze()
    ##adjust for arable area.
    seeding_rate = seeding_rate.mul(arable, axis=1)
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
    ##add cost for cont pasture
    phase_cost = f_cont_pas(phase_cost)
    ##cost allocation
    start = per.wet_seeding_start_date()
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start)
    ##add cashflow period to col index
    phase_cost.columns = pd.MultiIndex.from_product([phase_cost.columns, [allocation]])
    ##merge
    rot_cost = pd.merge(phases_df2, phase_cost, how='left', left_on=uinp.cols()[-1], right_index = True)
    seedcost = rot_cost.drop(list(range(uinp.structure['phase_len'])), axis=1).stack([0])
    r_vals['seedcost'] = seedcost
    return seedcost

def insurance(r_vals):
    '''
    Returns
    ----------
    Dataframe to add with other rotation costs
        Misc costs
        - crop insurance
        *note - arable area is already counted for by the yield calculation.
    '''
    ##first need to combine each grain pool to get average price
    ave_price=np.multiply(f_farmgate_grain_price(),uinp.price['grain_price'][['prop_firsts','prop_seconds']]).sum(axis=1)#np multiply doen't look at the column names and indexs
    insurance=ave_price*uinp.price['grain_price']['insurance']/100  #div by 100 because insurance is a percent
    rot_insurance = rot_yield().mul(insurance, axis=0, level = 1)/1000 #divide by 1000 to convert yield to tonnes    
    rot_insurance = rot_insurance.droplevel(1).unstack()
    ##cost allocation
    start = uinp.price['crp_insurance_date']
    p_dates = per.cashflow_periods()['start date']
    p_name = per.cashflow_periods()['cash period']
    allocation=fun.period_allocation(p_dates, p_name, start)
    ##add cashflow period to col index
    rot_insurance.columns = pd.MultiIndex.from_product([rot_insurance.columns, [allocation]])
    insurance_cost = rot_insurance.stack([0])
    r_vals['insurance_cost'] = insurance_cost
    return insurance_cost




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
    cost = pd.concat([fert_cost(r_vals),nap_fert_cost(r_vals),f_chem_cost(r_vals),seedcost(r_vals), insurance(r_vals),phase_stubble_cost(r_vals)],axis=1).sum(axis=1,level=0)
    params['rot_cost'] = cost.stack().to_dict()

# jj=rot_cost()

#########################
#stubble                #
#########################
#stubble produced per kg grain harvested, used in stubble.py as well
def stubble_production(params=False):
    '''stubble produced by each rotation phase
        kgs of dry matter'''
    stubble = pd.DataFrame(index=pinp.crop['start_harvest_crops'].index, columns=['a'])
    stubble['a'] = 1 / (pinp.stubble['harvest_index'] * pinp.stubble['proportion_grain_harv']) - 1  # subtract 1 to account for the tonne of grain that was harvested
    if params:
        params['stubble_production'] = stubble.stack().to_dict()
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
        Crop sow (wet or dry or spring) requirement for each rot phase

    '''
    ##sow = arable area
    arable = pinp.crop['arable']
    cropsow = arable.reindex(pd.MultiIndex.from_product([uinp.structure['C'],arable.index]), axis=0, level=1).droplevel(1)
    ##merge to rot phases
    cropsow = pd.merge(phases_df, cropsow, how='left', left_on=uinp.cols()[-1], right_index = True)
    ##add current crop to index
    cropsow.set_index(uinp.cols()[-1], append=True, inplace=True)
    params['crop_sow'] = cropsow.drop(list(range(uinp.structure['phase_len']-1)), axis=1).stack().to_dict()



#################
#continuous pas #
#################

def f_cont_pas(cost_array):
    '''
    calculates the cost for continuos pasture that is resown a proportion of the time. eg tc (cont tedera)
    the cost of cont pasture is a combination of the cost of normal and resown eg tc = t + tr (weighted by the frequency of resowing)
    This function requires the index to be the landuse with no other levels. You can use unstack to ensure landuse is the only index.
    Generally this function is applied early in the cost process (before landuse has been dropped)
    Cont pasture only needs to exist if the phase has been included in the rotation.

    Note: if a new pasture is added which has a continuos option that is resown occasionally it will need to be added to this function.

    :param
    cost_array - df with landues axis. this array will be returned with the addition of the continuos pasture landuse
    '''
    pastures = uinp.structure['pastures'][pinp.general['pas_inc']]
    ##if cont tedera is in rotion list and tedera is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['tc'])) and 'tedera' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time tc and jc are resown - this is used as a weighting to determine the input costs
        tc_idx = germ_df.iloc[:,-3].isin(['tc']) #checks current phase for tc
        tc_frequency = germ_df.loc[tc_idx,'resown'] #get frequency of resowing tc
        ##create mask for normal tedera and resown tedera
        bool_t = cost_array.index.isin(['t'])
        bool_tr = cost_array.index.isin(['tr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_t)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            t_cost = 0
        else:
            t_cost=(cost_array[bool_t]*(1-tc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_tr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            tr_cost = 0
        else:
            tr_cost=(cost_array[bool_tr]*(tc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        tc_cost = t_cost + tr_cost
        ##assign to df as new col
        cost_array.loc['tc', :] = tc_cost

    ##if cont tedera is in rotion list and tedera is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['jc'])) and 'tedera' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time jc and jc are resown - this is used as a weighting to determine the input costs
        jc_idx = germ_df.iloc[:,-3].isin(['jc']) #checks current phase for jc
        jc_frequency = germ_df.loc[jc_idx,'resown'] #get frequency of resowing jc
        ##create mask for normal tedera and resown tedera
        bool_j = cost_array.index.isin(['j'])
        bool_jr = cost_array.index.isin(['jr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_j)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            j_cost = 0
        else:
            j_cost=(cost_array[bool_j]*(1-jc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_jr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            jr_cost = 0
        else:
            jr_cost=(cost_array[bool_jr]*(jc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        jc_cost = j_cost + jr_cost
        ##assign to df as new col
        cost_array.loc['jc', :] = jc_cost

    ##if cont lucerne is in rotion list and lucerne is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['uc'])) and 'lucerne' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time uc and xc are resown - this is used as a weighting to determine the input costs
        uc_idx = germ_df.iloc[:,-3].isin(['uc']) #checks current phase for uc
        uc_frequency = germ_df.loc[uc_idx,'resown'] #get frequency of resowing uc
        ##create mask for normal tedera and resown tedera
        bool_u = cost_array.index.isin(['u'])
        bool_ur = cost_array.index.isin(['ur'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_u)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            u_cost = 0
        else:
            u_cost=(cost_array[bool_u]*(1-uc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_ur)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            ur_cost = 0
        else:
            ur_cost=(cost_array[bool_ur]*(uc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        uc_cost = u_cost + ur_cost
        ##assign to df as new col
        cost_array.loc['uc', :] = uc_cost

    ##if cont lucerne is in rotion list and lucerne is included in the pasture modules then generate the inputs for it
    if any(phases_df.iloc[:,-1].isin(['xc'])) and 'lucerne' in pastures:
        germ_df = pinp.pasture_inputs['tedera']['GermPhases']
        ##determine the proportion of the time xc and xc are resown - this is used as a weighting to determine the input costs
        xc_idx = germ_df.iloc[:,-3].isin(['xc']) #checks current phase for xc
        xc_frequency = germ_df.loc[xc_idx,'resown'] #get frequency of resowing xc
        ##create mask for normal tedera and resown tedera
        bool_x = cost_array.index.isin(['x'])
        bool_xr = cost_array.index.isin(['xr'])
        ##create new param - average all phases.
        if np.count_nonzero(bool_x)==0: #check if any of the phases in the input array had x, if not then the cost is 0
            x_cost = 0
        else:
            x_cost=(cost_array[bool_x]*(1-xc_frequency[0])).mean(axis=0) #get average cost of each t phase
        if np.count_nonzero(bool_xr)==0: #check if any of the phases in the input array had t, if not then the cost is 0
            xr_cost = 0
        else:
            xr_cost=(cost_array[bool_xr]*(xc_frequency[0])).mean(axis=0) #get average cost of each t phase
        ##add weighted average of the resown and normal phase
        xc_cost = x_cost + xr_cost
        ##assign to df as new col
        cost_array.loc['xc', :] = xc_cost

    return cost_array

