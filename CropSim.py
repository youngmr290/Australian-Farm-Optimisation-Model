# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:22:21 2020

@author: young
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

##midas
import UniversalInputs as uinp
import Crop as crp

##define lengths
n_sigmoid_params = 3 #this depends on the function that describes the relationship
n_yield_params = 2 #this depends on the function that describes the relationship
n_weedcontrols = len(uinp.crop['weed_params'])
n_fungicontrols = len(uinp.crop['fungi_control_params'])
n_controls = n_weedcontrols + n_fungicontrols #add other controls as they exist ie fert
n_fungi = len(uinp.crop['initial_fungus'].squeeze())
n_fungilanduses = len(uinp.crop['fungi_landuse_params']) #just the landuse sets required for fungi, currently only the major capital sets are required
n_weeds = len(uinp.crop['initial_weed'].squeeze())
n_weedseed_groups = len(uinp.crop['initial_weed'])
n_croplanduses = len(uinp.crop['weed_yield_params'])
n_phases = len(uinp.structure['phases'])
n_years = len(uinp.structure['phases'].columns)
n_pas = len(uinp.crop['pasture_control'])

##define empty arrays
control_bnds_ko = np.zeros((n_croplanduses,n_controls),  dtype = 'float64')
crop_control_levels_ryo = np.zeros((n_phases,n_years,n_controls),  dtype = 'float64')
cost_control_o = np.zeros(n_controls,  dtype = 'float64')
fungibank_die_ku = np.zeros((n_fungilanduses,n_fungi),  dtype = 'float64')
fungibank_ryu = np.zeros((n_phases,n_years,n_fungi),  dtype = 'float64')
germ_ew = np.zeros((n_weeds,n_weedseed_groups),  dtype = 'float64')
pas_control_to = np.zeros((n_pas,n_controls),  dtype = 'float64')
seedbank_rew = np.zeros((n_phases,n_weedseed_groups,n_weeds),  dtype = 'float64')
seed_die_ew = np.zeros((n_weedseed_groups,n_weeds),  dtype = 'float64')
seed_set_max_ew = np.zeros((n_weedseed_groups,n_weeds),  dtype = 'float64')
seed_set_min_w = np.zeros((n_weeds),  dtype = 'float64')
seed_set_slope_w = np.zeros((n_weeds),  dtype = 'float64')
seed_soften_w = np.zeros(n_weeds,  dtype = 'float64')
seedbank_value_ew = np.zeros((n_weedseed_groups,n_weeds),  dtype = 'float64')
yield_rev_k = np.zeros(n_croplanduses,  dtype = 'float64')
t_weed_density_ryw = np.zeros((n_phases,n_years,n_weeds),  dtype = 'float64')
t_seed_set_soft_ryw = np.zeros((n_phases,n_years,n_weeds),  dtype = 'float64')
t_seed_set_hard_ryw = np.zeros((n_phases,n_years,n_weeds),  dtype = 'float64')
t_seed_bank_soft_ryw = np.zeros((n_phases,n_years,n_weeds),  dtype = 'float64')
t_seed_bank_hard_ryw = np.zeros((n_phases,n_years,n_weeds),  dtype = 'float64')
t_yield_ry = np.zeros((n_phases,n_years),  dtype = 'float64')

##make phases df into a array of sets so we can test of the landues sets contain the phase set
phases = np.array(uinp.structure['phases'])
for x in range(n_years):
    for xx in range(n_phases):
            phases[xx,x]={phases[xx,x]} 
##build phase arrays for pasture and crop- the reason for doing it twice is just so the inputs can be split up which keeps them looking more clean.
###first have to build a list of the supersets each set in the list becomes a slice in the landuse axis
sets = []
for k in uinp.crop['weed_yield_params'].index:
    sets.append(uinp.structure[k])
phases_crop_ryk = phases[...,np.newaxis]<=sets
###remove true from generalised landuse slice in yr0 - ie for yr0=b, there will be a true in both the slice of E and b but i only want it in the b slice.
gen_landuse_slice = len(uinp.crop['max_yield_gen'])
phases_crop_ryk[:,-1,0:gen_landuse_slice] = False
##create the pasture phase array
###first have to build a list of the supersets each set in the list becomes a slice in the landuse axis
sets = []
for k in uinp.crop['pasture_control'].index:
    sets.append(uinp.structure[k])
phases_pas_ryt = phases[...,np.newaxis]<=sets
##create the fungi landuse phase array, a seperate array must be built because the fungi bank params have a simplified landuse index to reduce inputs
###first have to build a list of the supersets each set in the list becomes a slice in the landuse axis
sets = []
for k in uinp.crop['fungi_landuse_params'].index:
    sets.append(uinp.structure[k])
phases_fungi_ryk = phases[...,np.newaxis]<=sets

##populate arrays
control_bnds_ko = np.array(pd.concat([uinp.crop['weed_control_bnds'],uinp.crop['fungi_control_bnds']],axis=1))
cost_control_o[...] = np.concatenate([uinp.price['herbicide'],uinp.price['fungicide']]) #will have to concat other control costs here too
germ_ew[...] = uinp.crop['weed_seed_germ']
seedbank_rew[...] = uinp.crop['initial_weed'] #divide by 100 to make scale match the function ^this divide should be removed and just factored in the function
fungibank_ryu[:,0,:] = uinp.crop['initial_fungus'] #divide by 100 to make scale match the function ^this divide should be removed and just factored in the function
fungi_params_oup = uinp.crop['fungi_control_params'].values.reshape(n_fungicontrols,n_fungi,n_sigmoid_params)
fungi_host_params_kup = uinp.crop['fungi_landuse_params'].values.reshape(n_fungilanduses,n_fungi,n_sigmoid_params)
fungi_host_params_ryup = np.sum(phases_fungi_ryk[...,np.newaxis,np.newaxis] * fungi_host_params_kup, axis =2)
fungi_die_ryu = np.sum(phases_fungi_ryk[...,np.newaxis] * fungibank_die_ku, axis =2)
fungi_yield_params_kup = uinp.crop['fungi_yield_params'].values.reshape(n_croplanduses,n_fungi,n_yield_params)
fungi_yield_params_ryup = np.sum(phases_crop_ryk[...,np.newaxis,np.newaxis] * fungi_yield_params_kup, axis =2)
seed_die_ew[...] = uinp.crop['seeds_die'] #divide by 100 to make scale match the function ^this divide should be removed and just factored in the function
seed_set_max_ew[...] = uinp.crop['seed_set']
seed_set_min_w[...] = uinp.crop['seedset_min']
seed_set_slope_w[...] = uinp.crop['seedset_slope']
seed_soften_w[...] = uinp.crop['seeds_soften']
seedbank_value_ew[...] = uinp.crop['seedbank_value']
weed_params_owp = uinp.crop['weed_params'].values.reshape(n_weedcontrols,n_weeds,n_sigmoid_params)
weed_yield_params_kwp = uinp.crop['weed_yield_params'].values.reshape(n_croplanduses,n_weeds,n_yield_params)
weed_yield_params_rywp = np.sum(phases_crop_ryk[...,np.newaxis,np.newaxis] * weed_yield_params_kwp, axis =2)
pas_control_to[...] = uinp.crop['pasture_control']
phases_pas_ryto = phases_pas_ryt[...,np.newaxis] * pas_control_to[np.newaxis,np.newaxis,...] #add control level to pas phase array then sum on pasture axis to return the control level per phase per rotation
pas_control_levels_ryo = np.sum(phases_pas_ryto,axis=2)
yield_max_k=pd.concat([uinp.crop['max_yield_gen'],uinp.crop['max_yield_spec']]) #make array for yield
yield_rev_k[...] = crp.farmgate_grain_price(True)
ymax_ry = np.sum(phases_crop_ryk * yield_max_k.values.flatten(),axis=2)

##build functions for optimisation.
def weed_density(seedbank_ew,germ_ew,x,a_ow,b_ow,c_ow,ax):
    '''
    Parameters
    ----------
    seedbank_ew : float array
        weed seed bank.
    germ_ew : float
        germination of weeds.
    a : float
        midpoint.
    x : float
        level of control.
    b : float
        logistic growth.
    c : float
        proportion of weeds still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.

    Returns
    -------
    2d array of float (1 number per weed, per landuse)
        weed density after a given level of control is applied.
    '''
    ##determine weed kill proportion for each control option
    prop_kill_ow = c_ow + (1-c_ow)/(1+np.exp(b_ow*(x[...,None]-a_ow)))
    ##multiply across control option to get overall weed kill proportion
    prop_kill_w = np.prod(prop_kill_ow, axis=ax) 
    # print('prop : ',prop_kill_w)
    ##calc total plants germinated 
    germ_w = np.sum(seedbank_ew*germ_ew, axis=ax)
    return germ_w * prop_kill_w

def fungi_density(fungibank_u,x,a_ou,b_ou,c_ou,ax):
    '''
    Parameters
    ----------
    fungibank_u : float array
        level of fungi at the beginning of the given year.
    a : float
        midpoint.
    x : float
        level of control.
    b : float
        logistic growth.
    c : float
        proportion of fungi still remaining at very high level of control, ie 0.1 means that the maximum control of fungi is 0.9.

    Returns
    -------
    1d array of float 
        fungi level after a given level of all controls are applied.
    '''
    ##determine fungi kill proportion for each control option
    prop_kill_ou = c_ou + (1-c_ou)/(1+np.exp(b_ou*(x[...,None]-a_ou)))
    ##multiply across control option to get overall fungi kill proportion
    prop_kill_u = np.prod(prop_kill_ou, axis=ax) 
    # print('prop : ',prop_kill_u)
    ##calc total plants germinated 
    return fungibank_u * prop_kill_u


def yield_n_state(fungibank_u,seedbank_ew,germ_ew,x,a_ow,a_ou,b_ow,b_ou,c_ow,c_ou,d_kw,d_ku,k_kw,k_ku,ymax,ax):
    '''
    Parameters
    ----------
    fungibank_u : float array
        fungi at the start of a given season.
    seedbank_ew : float array
        weed seed bank.
    germ_ew : float
        germination of weeds.
    x : float array
        level of control.
    a : float array
        midpoint of control kill.
    b : float array
        logistic growth of control kill.
    c : float array
        proportion of state variable still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.
    d : float array
        maximum yield loss.
    k : float array
        logistic growth of yield loss at different state variable densities.
    ymax : float array
        max yield attainable for give landuse.

    Returns
    -------
    float
        yield of given lanuse and the seedset.

    '''
    # print( 'yield remaining',(1-(d+(-d/np.exp(k*weed_density(s,g,x,a,b,c))))))
    # print('yield: ', np.prod((1-(d+(-d/np.exp(k*weed_density(s,g,x,a,b,c))))),axis=0))
    fungi_density_u = fungi_density(fungibank_u,x[...,n_weedcontrols:(n_fungicontrols+n_weedcontrols)],a_ou,b_ou,c_ou,ax)
    weed_density_w = weed_density(seedbank_ew,germ_ew,x[...,:n_weedcontrols],a_ow,b_ow,c_ow,ax)
    ##when passing the whole rotation array to this function there needs to be a new axis, this is the best way i could think of achieving this given ax is already being passed in
    if ax==0:
        seed_set_plant = seed_set_max_ew * np.exp(np.sum(weed_density_w) * seed_set_slope_w) + seed_set_min_w 
        seed_set = weed_density_w * seed_set_plant #^seed set per plant should become a function of total plant density, not just weed density
    else:
        seed_set_plant_rew = seed_set_max_ew * np.exp(np.sum(weed_density_w[:,np.newaxis,:],axis=2,keepdims=True) * seed_set_slope_w) + seed_set_min_w 
        seed_set = weed_density_w[:,np.newaxis,:] * seed_set_plant_rew  #^seed set per plant should become a function of total plant density, not just weed density
    ##determine yield
    yield_remaining_weeds = np.prod(1-(d_kw+(-d_kw/np.exp(k_kw*weed_density_w))),axis =ax)
    yield_remaining_fungi = np.prod(1-(d_ku+(-d_ku/np.exp(k_ku*fungi_density_u))),axis =ax)
    overall_yield = ymax * yield_remaining_weeds * yield_remaining_fungi
    return overall_yield, seed_set, weed_density_w,fungi_density_u

    
def objective(x,row,k_slice):
    '''
    Parameters
    ----------
    x : float array
        level of control.
    row : int
          rotation slice 
    Returns
    -------
    objective
        point where max revenue from yield is most greater than control cost.
    '''
    ##determine all params for functions
    seedbank_ew = seedbank_rew[row,...] 
    fungibank_u = fungibank_ryu[row,yr,...] 
    a_ow = weed_params_owp[:,:,0]
    a_ou = fungi_params_oup[:,:,0]
    b_ow = weed_params_owp[:,:,1]
    b_ou = fungi_params_oup[:,:,1]
    c_ow = weed_params_owp[:,:,2]
    c_ou = fungi_params_oup[:,:,2]
    d_kw = weed_yield_params_rywp[row,yr,:,0]
    d_ku = fungi_yield_params_ryup[row,yr,:,0]
    k_kw = weed_yield_params_rywp[row,yr,:,1]
    k_ku = fungi_yield_params_ryup[row,yr,:,1]
    ymax = ymax_ry[row,yr]
    ax=0 #this is the axis being summed or multiplied along - it is inputted because it is different when calculating the overall results for the whole roataion vertorily
    control_cost = cost_control_o * x
    crp_yield, seedset_ew, weed_density, fungi_level = yield_n_state(fungibank_u,seedbank_ew,germ_ew,x,a_ow,a_ou,b_ow,b_ou,c_ow,c_ou,d_kw,d_ku,k_kw,k_ku,ymax,ax)
    yield_revenue = yield_rev_k[k_slice] * crp_yield - np.sum(seedset_ew * seedbank_value_ew)
    seedvalue = np.sum(seedset_ew * seedbank_value_ew)
    # print('revenue: ',yield_revenue)
    # print('cost: ',control_cost)
    return -(yield_revenue - np.sum(control_cost) - seedvalue)

##initial guesses    
x0 = np.ones(n_weedcontrols) #len of controls ie one for each control
x0 = [5]*n_controls #len of controls ie one for each control
x0 = [5,5,3,3,3,3,3] #len of controls ie one for each control


def minimise(row):
    row=row[0]
    ##check if any of the crop landuse axis values are True ie is the current phase a crop. If it is then optimise the control levels
    k_slice = phases_crop_ryk[row,yr,:]
    if k_slice.any():
        ##create bounds from input array
        upper = control_bnds_ko[k_slice,:].flatten()
        lower = np.zeros(n_controls)
        bnds=list(zip(lower,upper))
        # minimize(objective, x0,args=4000, method='L-BFGS-B', bounds=bnds, tol=1e-2) #may have to change around the solver (method) to get the best solution - time it and see what is best
        crop_control_levels_ryo[row,yr,:] = minimize(objective, x0,args=(row,k_slice), method='L-BFGS-B', bounds=bnds, tol=1e-2).x #may have to change around the solver (method) to get the best solution - time it and see what is best
    
for yr in range(n_years):
# yr=6    ##determine all params for vector function
    print(yr)
    a_ow = weed_params_owp[:,:,0]
    a_ou = fungi_params_oup[:,:,0]
    b_ow = weed_params_owp[:,:,1]
    b_ou = fungi_params_oup[:,:,1]
    c_ow = weed_params_owp[:,:,2]
    c_ou = fungi_params_oup[:,:,2]
    d_kw = weed_yield_params_rywp[:,yr,:,0]
    d_ku = fungi_yield_params_ryup[:,yr,:,0]
    k_kw = weed_yield_params_rywp[:,yr,:,1]
    k_ku = fungi_yield_params_ryup[:,yr,:,1]
    ymax = ymax_ry[:,yr]
    ax=1
    row=pd.DataFrame(range(len(phases)))
    np.apply_along_axis(minimise, 1, row)
    ##combine optimised control levels and the inputted ones for pasture.
    rotation_control_levels_ro = pas_control_levels_ryo[:,yr,:] + crop_control_levels_ryo[:,yr,:]
    ##using the optimised level of control calc the yield and seed bank that is used in next yr calcs.
    cropyield,seedset_rew,weed_density_rw,fungi_level_ru=yield_n_state(fungibank_ryu[:,yr,:],seedbank_rew,germ_ew,rotation_control_levels_ro,a_ow,a_ou,b_ow,b_ou,c_ow,c_ou,d_kw,d_ku,k_kw,k_ku,ymax,ax)
    ##calc seedbank for next yr - do this outside of the objective to save it from being done multiple times.
    ###soft
    seedbank_rew[:,0,:] = (seedset_rew[:,0,:] + seedbank_rew[:,1,:] * seed_soften_w) * seed_die_ew[0,:]
    ###hard
    seedbank_rew[:,1,:] = (seedset_rew[:,1,:] + seedbank_rew[:,1,:] *(1 - seed_soften_w)) * seed_die_ew[1,:]
    ##calc fungi level for following yr = fungi level after control + landuse factor
    a_ru = fungi_host_params_ryup[:,yr,:,0]
    b_ru = fungi_host_params_ryup[:,yr,:,1]
    c_ru = fungi_host_params_ryup[:,yr,:,2]
    die_prop_ru = fungi_die_ryu[:,yr,:]
    ###fungi level after host landuses adjustment
    fungi_level_ru = np.maximum(fungi_level_ru,c_ru/(1+np.exp(-b_ru*(fungi_level_ru-a_ru))))
    ###fungi level after non-host adjustment
    fungibank_ryu[:,yr+1,:] = fungi_level_ru * die_prop_ru
    ##build some arrays that contain data for testing purposes
    ###weed density
    t_weed_density_ryw[:,yr,:] = weed_density_rw
    ###seed set soft
    t_seed_set_soft_ryw[:,yr,:] = seedset_rew[:,0,:]
    ###seed set hard
    t_seed_set_hard_ryw[:,yr,:] = seedset_rew[:,1,:]
    ###seedbank soft
    t_seed_bank_soft_ryw[:,yr,:] = seedbank_rew[:,0,:]
    ###seedbank hard
    t_seed_bank_hard_ryw[:,yr,:] = seedbank_rew[:,1,:]
    ###yield
    t_yield_ry[:,yr] = cropyield

##build df to write
phases_yield_df =uinp.structure['phases'].copy()
phases_chem_df =uinp.structure['phases'].copy()
phases_fert_df =uinp.structure['phases'].copy()
phases_yield_df['yield'] = cropyield
phases_chem_df['chem'] = rotation_control_levels_ro[:,0:n_weedcontrols+n_fungcontrols]
phases_fert_df['fert'] = rotation_control_levels_ro[:,n_weedcontrols+n_fungcontrols:]
##start writing
writer = pd.ExcelWriter('crop sim.xlsx', engine='xlsxwriter')
phases_yield_df.to_excel(writer, sheet_name='yield',index=True,header=False)
phases_fert_df.to_excel(writer, sheet_name='fert',index=True,header=False)
phases_chem_df.to_excel(writer, sheet_name='chem',index=True,header=False)
##finish writing and save
writer.save()


'''
Notes
1- L-BFGS-B solver is faster than SLSQP and Nelder-Mead
2- Accurate starting guesses halves the solving time
3- allter the tollerance for a good trade off between time and accuracy
4- all methods i could make work are similar time, if the solving could be done similtaneously for each rotation that would speed it up a lot.


# import timeit

# f = lambda p: minimize(objective, x0,p, method='L-BFGS-B', bounds=bnds, tol=1e-2).x #may have to change around the solver (method) to get the best solution - time it and see what is best
# vf = np.vectorize(f)
# b=np.ones(5000,dtype='int')
# def method1():
#     return np.fromiter((f(xi) for xi in b), float, count=len(b))
# print(timeit.timeit(method1,number=1))

# def method2():
#     return [f(xi) for xi in b]
# print(timeit.timeit(method2,number=1))

# def method3():
#     return [minimize(objective, x0,p, method='L-BFGS-B', bounds=bnds, tol=1e-2) for p in b]
# print(timeit.timeit(method3,number=1))

# def minimise(row):
#     if not pasture:
#         minimize(objective, x0,args=row, method='L-BFGS-B', bounds=bnds, tol=1e-2).x #may have to change around the solver (method) to get the best solution - time it and see what is best
#     ##calc the weedseed bank for all landuses using inputted control level ie only pasture should get a level (maybe they all get a level but take min)
    

# def method4():
#     return np.array(list(map(minimise, b)))
# print(timeit.timeit(method4,number=1))

# def method5():
#     return f(b)
# print(timeit.timeit(method5,number=1))

# b=pd.DataFrame(b)
# def method6():
#     return np.apply_along_axis(minimise, 1, b)
# print(timeit.timeit(method6,number=1))

'''




























