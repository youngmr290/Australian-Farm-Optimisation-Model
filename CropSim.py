# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:22:21 2020

@author: young
"""
import math
import numpy as np
from scipy.optimize import minimize

##midas
import UniversalInputs as uinp


n_controls = len(uinp.crop['weed_params'])
n_weeds = len(uinp.crop['initial'].squeeze())
n_landuses = len(uinp.crop['yield_params'])
landuses = uinp.crop['yield_params'].index
phases = np.array(uinp.structure['phases'])
landuses = np.array(['z','r','w','f','o','b','l','h','of','i','k','v','annual','lucerne','tedera'])
vsets = np.vectorize(set)
##build the list of sets using a loop (use the index of the inputs as the key to the structure dict - will have to add some special sets in unversalinputs ie new A) 
sets = []
for k in uinp.crop['yield_params'].index:
    sets.append(uinp.structure[k])
phases = vsets(phases[...,np.newaxis])<=[{'A','a'},{'N','n'}]



initial_w = np.array(uinp.crop['initial']). flatten()
germ_w = np.array(uinp.crop['weed_seed_germ']). flatten()
seed_set_w = np.array(uinp.crop['seed_set']). flatten()
seedbank_value_w = np.array(uinp.crop['seedbank_value']). flatten()
weed_params_owp = uinp.crop['weed_params'].values.reshape(n_controls,n_weeds,3)
yield_params_kwp = uinp.crop['yield_params'].values.reshape(n_landuses,n_weeds,2)

def w_kill(a,b,c,x):
    '''
    Parameters
    ----------
    c : float
        proportion of weeds still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.
    b : float
        logistic growth.
    x : float
        level of control.
    a : float
        midpoint.

    Returns
    -------
    2d array of float (1 number per weed, per landuse)
        proportion of weeds remaining after control is applied.
    '''
    ##determine weed kill proportion for each control option
    prop_kill_wok = c + (1-c)/(1+np.exp(b*(x-a)))

    print('prop_kill_wok ',prop_kill_wok)
    print('prop_kill_wok ',np.prod(prop_kill_wok, axis=0))
    ##multiply across control option to get overall weed kill proportion
    return np.prod(prop_kill_wok, axis=0) 
    # return prop_kill_wok 

def weed_density(i,g,x,a,b,c):
    '''
    Parameters
    ----------
    i : float array
        initial weed seed bank.
    g : float
        germination of weeds.
    c : float array
        proportion of weeds still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.
    b : float array
        logistic growth of weed kill.
    x : float array
        level of control.
    a : float array
        midpoint of weed kill.

    Returns
    -------
    Array
        weed density for each weed.

    '''
    print('weed density: ', i * g * w_kill(a,b,c,x))
    return i * g * w_kill(a,b,c,x)

#added yield from increasing control. how to get this without derivative? ie biggest difference between total cost and revenue
# def extra_yield():

##calc the $ from yield    
def yield_n_seedset(i,g,x,a,b,c,d,k,ymax):
    '''
    Parameters
    ----------
    i : float array
        initial weed seed bank.
    g : float
        germination of weeds.
    c : float array
        proportion of weeds still remaining at very high level of control, ie 0.1 means that the maximum control of weeds is 0.9.
    b : float array
        logistic growth of weed kill.
    x : float array
        level of control.
    a : float array
        midpoint of weed kill.
    d : float array
        maximum yield loss.
    k : float array
        logistic growth of yield loss at different weed densities.
    ymax : float array
        max yield attainable for give landuse.

    Returns
    -------
    float
        yield of given lanuse.

    '''
    print( 'yield remaining',(1-(d+(-d/np.exp(k*weed_density(i,g,x,a,b,c))))))
    print('yield: ', np.prod((1-(d+(-d/np.exp(k*weed_density(i,g,x,a,b,c))))),axis=1))
    weed_density = weed_density(i,g,x,a,b,c)
    seed_set = weed_density * seed_set_w #^seed set per plant should become a function of total plant density
    return ymax*np.prod((1-(d+(-d/np.exp(k*weed_density)))),axis=1),seed_set


    
    
    
    
##solve for point where max revenue from yield is most greater than control cost

def objective(x):
    '''
    Parameters
    ----------
    x : float array
        level of control.
    Returns
    -------
    objective
        point where max revenue from yield is most greater than control cost.
    '''
    i = initial_w/100 #divide by 100 to make it the right scale for grapph
    g = germ_w
    # a = np.array([2,2])
    a_ow = weed_params_owp[:,:,0]
    # b_ow = np.array([1,1])
    b_ow = weed_params_owp[:,:,1]
    # c_ow = np.array([0.1,0.5])
    c_ow = weed_params_owp[:,:,2]
    # d = 0.7
    # k = 0.35
    d_lw = yield_params_kwp[:,:,0]
    k_lw = yield_params_kwp[:,:,1]
    ymax = 4
    cost_control_litre = 30
    yield_rev_t = 350
    control_cost = cost_control_litre * x
    yield, seedset = yield_n_seedset(i,g,x,a_ow,b_ow,c_ow,d_lw,k_lw,ymax)
    yield_revenue = yield_rev_t * yield + seedset_w * seedbank_value_w
    print('revenue: ',yield_revenue)
    print('cost: ',control_cost)
    return -(yield_revenue - np.sum(control_cost))


# def solve_control_level():
    #initial guesses    
x0 = np.ones(n_controls) #len of controls ie one for each control
# x0 = [0]
## bounds on variables
# bndspositive = (0, 100.0) #qualtity of other components must be greater than 10%
no_upbnds = (2, 1.0e10) 
bnds = [no_upbnds]*n_controls
solution = minimize(objective, x0, method='SLSQP', bounds=bnds) #may have to change around the solver (method) to get the best solution - time it and see what is best

x = solution.x
print('level of control ',x)


# objective(x)































