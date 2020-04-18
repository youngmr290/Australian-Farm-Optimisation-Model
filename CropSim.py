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

##define lengths
n_control_params = 3 #this depends on the function that describes the relationship
n_yield_params = 2 #this depends on the function that describes the relationship
n_weedcontrols = len(uinp.crop['weed_params'])
n_controls = n_weedcontrols #add other controls as they exist ie fert
n_weeds = len(uinp.crop['initial'].squeeze())
n_weedseed_groups = len(uinp.crop['initial'])
n_landuses = len(uinp.crop['yield_params'])
n_phases = len(uinp.structure['phases'])
n_years = len(uinp.structure['phases'].columns)
n_pas = len(uinp.crop['pasture_control'])
##define empty arrays
crop_control_levels_ryo = np.zeros((n_phases,n_years,n_controls),  dtype = 'float64')
pas_control_to = np.zeros((n_pas,n_controls),  dtype = 'float64')
cost_control_o = np.zeros(n_controls,  dtype = 'float64')
germ_ew = np.zeros((n_weeds,n_weedseed_groups),  dtype = 'float64')
seedbank_rew = np.zeros((n_phases,n_weedseed_groups,n_weeds),  dtype = 'float64')
seed_set_ew = np.zeros((n_weeds,n_weedseed_groups),  dtype = 'float64')
seed_soften_w = np.zeros(n_weeds,  dtype = 'float64')
seedbank_value_ew = np.zeros((n_weedseed_groups,n_weeds),  dtype = 'float64')

##make phases df into a array of sets so we can test of the landues sets contain the phase set
phases = np.array(uinp.structure['phases'])
for x in range(n_years):
    for xx in range(n_phases):
            phases[xx,x]={phases[xx,x]} 
##build phase arrays for pasture and crop- the reason for doing it twice is just so the inputs can be split up which keeps them looking more clean.
###first have to build a list of the supersets each set in the list becomes a slice in the landuse axis
sets = []
for k in uinp.crop['yield_params'].index:
    sets.append(uinp.structure[k])
phases_crop_ryk = phases[...,np.newaxis]<=sets
##remove true from generalised landuse slice in yr0 - ie for yr0=b, there will be a true in both the slice of E and b but i only want it in the b slice.
gen_landuse_slice = len(uinp.crop['max_yield_gen'])
phases_crop_ryk[:,-1,0:gen_landuse_slice] = False
##create the pasture phase array
###first have to build a list of the supersets each set in the list becomes a slice in the landuse axis
sets = []
for k in uinp.crop['pasture_control'].index:
    sets.append(uinp.structure[k])
phases_pas_ryt = phases[...,np.newaxis]<=sets

seedbank_rew[...] = uinp.crop['initial']/100 #divide by 100 to make scale match the function ^this divide should be removed and just factored in the function
germ_ew[...] = uinp.crop['weed_seed_germ']
seed_set_ew[...] = uinp.crop['seed_set']
seed_soften_w[...] = uinp.crop['seeds_soften']
seedbank_value_ew[...] = uinp.crop['seedbank_value']
weed_params_owp = uinp.crop['weed_params'].values.reshape(n_weedcontrols,n_weeds,n_control_params)
control_bnds_ko = np.array(uinp.crop['control_bnds'])
##params used in objective
cost_control_o[...] = uinp.crop['chem_cost'] #will have to concat other control costs here too
yield_rev_t = 350
pas_control_to[...] = uinp.crop['pasture_control']
yield_params_kwp = uinp.crop['yield_params'].values.reshape(n_landuses,n_weeds,n_yield_params)
yield_params_rywp = np.sum(phases_crop_ryk[...,np.newaxis,np.newaxis] * yield_params_kwp, axis =2)
##add control level to pas phase array then sum on pasture axis to return the control level per phase per rotation
phases_pas_ryto = phases_pas_ryt[...,np.newaxis] * pas_control_to[np.newaxis,np.newaxis,...] 
pas_control_levels_ryo = np.sum(phases_pas_ryto,axis=2)
##make array for yield
cyield=pd.concat([uinp.crop['max_yield_gen'],uinp.crop['max_yield_spec']])
ymax_ry = np.sum(phases_crop_ryk * cyield.values.flatten(),axis=2)
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
    # np.subtract(x,a_ow)
    # print(a_ow)
    # print(b_ow)
    # print(c_ow)
    # print(x)
    # np.subtract(x,a_ow)
    prop_kill_ow = c_ow + (1-c_ow)/(1+np.exp(b_ow*(x[...,None]-a_ow)))
    ##multiply across control option to get overall weed kill proportion
    prop_kill_w = np.prod(prop_kill_ow, axis=ax) 
    ##calc total plants germinated
    germ_w = np.sum(seedbank_ew*germ_ew, axis=ax)
    # print('prop : ',prop_kill_w)
    return germ_w * prop_kill_w


def yield_n_seedset(seedbank_ew,germ_ew,x,a_ow,b_ow,c_ow,d_kw,k_kw,ymax,ax):
    '''
    Parameters
    ----------
    seedbank_ew : float array
        weed seed bank.
    germ_ew : float
        germination of weeds.
    x : float array
        level of control.
    a_ow : float array
        midpoint of weed kill.
    b_ow : float array
        logistic growth of weed kill.
    c_ow : float array
        proportion of weeds still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.
    d_kw : float array
        maximum yield loss.
    k_kw : float array
        logistic growth of yield loss at different weed densities.
    ymax : float array
        max yield attainable for give landuse.

    Returns
    -------
    float
        yield of given lanuse and the seedset.

    '''
    # print( 'yield remaining',(1-(d+(-d/np.exp(k*weed_density(s,g,x,a,b,c))))))
    # print('yield: ', np.prod((1-(d+(-d/np.exp(k*weed_density(s,g,x,a,b,c))))),axis=0))
    weed_density_w = weed_density(seedbank_ew,germ_ew,x,a_ow,b_ow,c_ow,ax)
    ##when passing the whole rotation array to this function there needs to be a new axis, this is the best way i could think of achieving this given ax is already being passed in
    if ax==0:
        seed_set = weed_density_w * seed_set_ew #^seed set per plant should become a function of total plant density
    else:
        seed_set = weed_density_w[:,np.newaxis,:] * seed_set_ew #^seed set per plant should become a function of total plant density
    return ymax*np.prod((1-(d_kw+(-d_kw/np.exp(k_kw*weed_density_w)))),axis=ax),seed_set

    
def objective(x,row):
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
    a_ow = weed_params_owp[:,:,0]
    b_ow = weed_params_owp[:,:,1]
    c_ow = weed_params_owp[:,:,2]
    d_kw = yield_params_rywp[row,yr,:,0]
    k_kw = yield_params_rywp[row,yr,:,1]
    ymax = ymax_ry[row,yr]

    ax=0 #this is the axis being summed or multiplied along - it is inputted because it is different when calculating the overall results for the whole roataion
    
    control_cost = cost_control_o * x
    crp_yield, seedset_ew = yield_n_seedset(seedbank_ew,germ_ew,x,a_ow,b_ow,c_ow,d_kw,k_kw,ymax,ax)
    yield_revenue = yield_rev_t * crp_yield - np.sum(seedset_ew * seedbank_value_ew)
    seedvalue = np.sum(seedset_ew * seedbank_value_ew)
    # print('revenue: ',yield_revenue)
    # print('cost: ',control_cost)
    return -(yield_revenue - np.sum(control_cost) - seedvalue)

##initial guesses    
x0 = np.ones(n_weedcontrols) #len of controls ie one for each control
x0 = [5]*n_weedcontrols #len of controls ie one for each control
x0 = [5,5,3,3] #len of controls ie one for each control


def minimise(row):
    row=row[0]
    ##check if any of the crop landuse axis values are True ie is the current phase a crop. If it is then optimise the control levels
    k_slice = phases_crop_ryk[row,yr,:]
    if k_slice.any():
        ##create bounds from input array
        upper = control_bnds_ko[k_slice,:].flatten()
        lower = np.zeros(n_controls)
        bnds=list(zip(lower,upper))
        minimize(objective, x0,args=4000, method='L-BFGS-B', bounds=bnds, tol=1e-2) #may have to change around the solver (method) to get the best solution - time it and see what is best
        crop_control_levels_ryo[row,yr,:] = minimize(objective, x0,args=row, method='L-BFGS-B', bounds=bnds, tol=1e-2).x #may have to change around the solver (method) to get the best solution - time it and see what is best
    
    

yr=0
##determine all params for vector function
a_ow = weed_params_owp[:,:,0]
b_ow = weed_params_owp[:,:,1]
c_ow = weed_params_owp[:,:,2]
d_kw = yield_params_rywp[:,yr,:,0]
k_kw = yield_params_rywp[:,yr,:,1]
ymax = ymax_ry[:,yr]
ax=1
b=pd.DataFrame(range(len(phases)))
np.apply_along_axis(minimise, 1, b)
##combine optimised control levels and the inputted ones for pasture.
rotation_control_levels_ro = pas_control_levels_ryo[:,yr,:] + crop_control_levels_ryo[:,yr,:]
##using the optimised level of control calc the yield and seed bank that is used in next yr calcs.
cropyield,seedset_rew=yield_n_seedset(seedbank_rew,germ_ew,rotation_control_levels_ro,a_ow,b_ow,c_ow,d_kw,k_kw,ymax,ax)
##calc seedbank for next yr - do this outside of the objective to save it from being done multiple times.
###soft
seedbank_rew[:,0,:] = seedset_rew[:,0,:] + seedbank_rew[:,1,:] * seed_soften_w
###hard
seedbank_rew[:,1,:] = seedset_rew[:,1,:] + seedbank_rew[:,1,:] *(1 - seed_soften_w)
    



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




























