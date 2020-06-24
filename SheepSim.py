# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John
"""

'''
import functions from other modules
'''
# import datetime as dt
import pandas as pd
import numpy as np

# from numba import jit

# import FeedBudget as fdb
# import Functions as fun
# import Periods as per
import PropertyInputs as pinp
import SheepSimRoutines as sfun
import UniversalInputs as uinp


############################
### _constants required    #
############################
## define some parameters required to size arrays.
# n_feed_pools        = uinp.n_feed_pools
# n_feed_periods      = len(pinp.feed_inputs['feed_periods']) - 1

#^ put the values as lists in universal.xlsx (SheepDefinitions!)
# then define n by length of the list
n_animal_types = 2      # a: wool, meat
n_btrt = 6              # b: 11, 22, 21, 33, 32, 31
n_genotype_options = 5  # c: number in spreadhseet
n_dam_ages = 3          # d: yearling, maiden, adult
n_max_ecycles = 2       # e: max number of estrus cycles they are joined
n_feed_var_periods = 10 # f:
n_genotypes = 6         # g: B, M, T, BM, BT, BMT
n_g0 = 3                # g0: B, M, T
n_g1 = 2                # g1: BB, BM
n_g2 = 4                # g2: BBB, BBM, BBT, BMT
n_husbandry_class = 1   # h
n_groups_rams = 3       # i & g0: genotypes of rams
n_groups_ewes = 4       # j: genotype groups of ewes
n_groups_offspring = 5  # k: genotype groups and growth profile
n_groups_lambing = 1    # l: lambing groups for the seed animals (1 unless doing a TOL analysis or 8 month joinings)
n_months = 12           # m: Jan to Dec
n_feed_periods = 10     # n:
n_lambing_opps = 15     # o:
# n_sim_periods         # p: see below
n_labour_periods = 16   # q:
n_feed_variables        # r:
n_shearing_occs = 16    # s:
n_sale_times = 4        # t: weaner, backgrounded, finished, remainder
n_husbandry_options = 10# u:
n_genders = 3           # w: ewe, wether, ram
n_litter_size = 5       # x: Dry, single, twin, triplet, not mated
n_lactation_number = 5  # y: dry, single, twin, triplet, in utero
n_sexes = 3             # w: ram, ewe, wether
# n_sim_periods see below
n_labour_periods = 16   # q
        i_sim_periods_year = 52 # ^uinp.n_sim_periods_year   will be in structure dict now
i_oldest_animal= 6.5    # ^uinp.i_oldest_animal

birth_date_i = uinp.propertydata['ExcelName']   #Find the ExcelNames
birth_date_jl = uinp.propertydata['ExcelName']
birth_date_jel = uinp.propertydata['ExcelName']
birth_date_kel = uinp.propertydata['ExcelName']
## Some one time data manipulation for the inputs just read
start_year = np.min(birth_date_jl)
## might need to test and rebase the year for the other animal groups


### _define the periods
n_sim_periods, date_p, p_index_p, step \
        = sfun.sim_periods(start_year, i_sim_periods_year, i_oldest_animal)
### _array dimensions







###################################
### initialise global arrays      #
###################################
'''only create arrays that are used in sim and post processing.
'''
## Instantiate the globals arrays
## # these store the output of simulation and the parameters for pyomo
## # see documentation for a description of each variable




def simulation():
    """
    A function to wrap the simulation that can be called by SheepPyomo.

    Called after the sensitivty variables have been updated.
    It populates the arrays by looping through the time periods
    Globally define arrays are used to transfer results to sheep_paramters()

    Returns
    -------
    None.
    """
    ###################################
    ### reshape neccessary inputs     # 
    ###################################
    '''only >2 dim array'''


    ############################
    ### initialise arrays      #
    ############################
    '''only if assign with a slice'''
    ## Instantiate the arrays that are only required within this function
    ## mainly arrays that will store the input data that require pre-defining
    ## # see documentation for a description of each variable


    #####################################
    ### populate the association arrays #
    #####################################
    ## the association arrays relate the slices of one array with the slices of another array
    ##needs to be within the loop because the genotype inputs can change in exp.xlsx
# a_j_kd

##genotype option of the input genotypes
a_k2_g0[0] = pinp.sheep['i_genotype_b0']
a_k2_g0[1] = pinp.sheep['i_genotype_m0']
a_k2_g0[2] = pinp.sheep['i_genotype_t0	']		
##Animal type option of the input genotypes
a_k1_g0[0] = pinp.sheep['i_animaltype_b0']
a_k1_g0[1] = pinp.sheep['i_animaltype_m0']
a_k1_g0[2] = pinp.sheep['i_animaltype_t0	']		
##Maternal genotype of the dams
a_maternal_g0_g1
a_paternal_g0_g1
a_maternal_g1_g2
a_paternal_g0_g2


a_g_i
a_g_j
a_g0_jo
a_g2_jo
a_g_kd
a_g1_kd
a_g0_kd
a_w_i
a_w_j
a_w_kd
a_join_o_pjel
a_s_pil
a_s_pjl
a_s_pkdwl
a_join_p_ojel
a_day90_p_ojel
a_6weeks_p_ojel
a_lamb_p_ojel
a_wean_p_ojel

    ############################
    ### management calculations#
    ############################
    date_birth_c0 =
    date_birth_ic1 =
    date_birth_diec2 =
    date_birth_y_oiec1 = 
    startdate = np.minimum(date_birth_c0 , date_birth_ic1 , date_birth_diec2)			
    step = sfun.sim_periods(start_year, periods_per_year, oldest_animal)[3] #^need to populate function with actual args
    step_int = step.astype('timedelta64[D]').astype(int)			
    step_float = step.astype(float)/(24 * 60 * 60)			
    age_mated_oiec1 = i_age_join_oic1[:, :, na, …] + cf_gc1[4, 0, :] * (index_e + 0.5)			
    age_scan = age_mated_oiec1 + i_scan_day_oic1[:, :, na, …]			
    







    ###########################
    ### non-loop calculations #
    ###########################
    '''Calculations for which the inputs do not depend on previous periods
    See spreadsheet: Group independent and Age,Date,Timing'''

    doy_p =
    lgf_eff_p =
    dlf_eff_p =
    dlf_wool_p =
    chill_p =
    kw =
    kc =
    birth_date_i =
    birth_date_jl =
    birth_date_jel =
    birth_date_kel =
    age_pi =
    age_pjl =
    age_pjel =
    age_pkel =
    age_f_pjel =
    age_f_pjel =
    pimi_pjel =
    ra_pjel =
    age_y_adj_pjel =
    af_wool_pi =
    af_wool_pjl =
    af_wool_pjel =
    af_wool_pkel =
    mm_pjel =
    d_cfw_ave_pi =
    d_cfw_ave_pjl =
    d_cfw_ave_pjel =
    d_cfw_ave_pkel =
    nw_max_pi =
    nw_max_pjxyl =
    nw_max_pkdwebl =
    ### _feed inputs
    sfun.feed_inputs function


    ##########################################
    ### Calc standard feed supply for periods#
    ##########################################
    '''flow chart 5'''
    
    ##########################################
    ### Initialise then loop through periods #
    ##########################################
    ## initialise the arrays for the first period #
    lw_ffcf = i.weaning_wt
    mw = 0.7 * lw_ffcf
    aw = 0.2 * lw_ffcf
    bw = 0.1 * lw_ffcf
    cfw = 0.6 #cfw at weaning
    fd = 19 #fd at weaning
    fl = 10 #fl at weaning
    #set all arrays that are assigned using += to 0.

    ## Loop through each week of the simulation (p) for ewes
    ## # number of periods is a fixed value so I'm thinking a 'for' loop
    for p in range(n_sim_periods):
        if p != 0:  # only carry this out with p<>0
            ### _conception
            cr_ojexyl[mask] += sfun.conception(lw_ffcf[p,...], srw_j)[mask]
            # with a mask to a
            nlb_ojewbl += cr_ojexyl#convert conception in _xy format to _wb
            ### _mortality
            mr[p,...] = sfun.mortality(rc[p-1,...])
            tem[p,...], dmr[p,...], lmr[p,...] = sfun.ewe_mortality()
            nlw_ojewbl = nlb_ojewbl &
            ### _start numbers & weight
            number[p,...] = sfun.transfers(number[p-1,...], sales
                            , ewe_mortality, cr, lamb_mortality, ....)  #function call or in global
            number[p] = (number[p-1] - sales[p-1]) * (1 - mortality) ....
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        ### feed supply loop
        # this loop is only required if a LW target is specified for the animals
        # if there is a target then the loop needs to continue until
        # the feed supply has converged on a value that generates a liveweight
        # change close to the target
        # The loop needs to execute at least once, then repeat if there
        # is a target and the result is not close enough to the target
        if this period (p) is a new feed variation period (f) or a new MIDAS feed period (n):
            then feed_supply_jxyl = feed_supply_pjxyl[p,...]
            otherwise use feedsupply from last period (which was optimised for the target)
        Feed supply loop start
            # the loop will be a bit tricky because the target is for an array of values
            # and some parts of the array may be within the tolerance but other parts are not.
            # To further complicate it the target will often be associated with
            # the weighted average of a slice of the array rather than an individual
            # element.
            foo, dmd, supp = sfun.feed_supply(feed_supply_jxyl, foo_std, dmd_std)
            #'
            pi_jexyl = sfun.p_intake(rc, srw, rel_size)
            ri_jexyl = sfun.r_intake(foo, dmd, supp)
            mei_jexyl = pi_jexyl - np.newaxis(e, supp_jxyl) * ri_jexyl * nv_jexyl + newaxis(supp_jxyl) * supp_md
            p_mei_pjexyl[p,...] = mei_jexyl
            mem = sfun.energy(....)
            mep, cw = sfun.pregnancy(....)
            mel = sfun.lactation(....)
            dcfw, new = sfun.wool_growth(....)
            ebg, pg = sfun.lw_change(mei, mem, mep, mel, mew, mecold, wmax, zf1, zf2)
            lwc = ebg * (1)
            if there is a target and abs(lwc-target) > eps:
                update feed_supply
                #      feed supply is a number between 0 and 3. We could use a binary
                #      type process to converge on the feed supply. But given that
                #      the feed supply was calculated in the previous period and
                #      it should be close then maybe a step process might be quicker.
                #      The main advantage of the binary approach is that each element
                #      of the array should converge at a similar rate, whereas maybe
                #      not with the step approach
                #      Open to ideas here.
            loop if feed_supply was changed
        lw_ffcf_jexyl = lw_ffcf_start_jexyl + lwc_jexyl * step
        lw_ffcf_max_jexyl = np.maximum(lw_ffcf_jexyl, lw_ffcf_max_jexyl)
        aw_jexyl
        mw_jexyl
        bw_jexyl
        ww_jexyl
        gw_jexyl
        fw_end_jexyl
        cfw_jexyl = cfw_start_jexyl + dcfw * step
        fl_jexyl
        fd_min_jexyl
        fd_jexyl
        ldr_end_jexyl
        lb_end_jexyl
        lw_jexyl = lw_ffcf_jexyl + cw_jexyl + cfw_Jexyl
        r_lw_jexyl[p,...] = lw_jexyl


    # repeat loop for rams & then for offspring
    # these don't require conception, pregnancy, lactation and ewe mortality
    for p in range(n_sim_periods):
        if p <>0:  # only carry this out with p<>0
            ## or pass lw_cfff_end and nw_end & srw and calculate z and rc
            mr[p,...] = sfun.mortality(rc[p-1,...])   # offspring
            mr[p,...] = sfun.mortality(rc[p-1,...])   # rams
            .... = sfun.numbers(....)                 #offspring
            .... = sfun.numbers(....)                 #rams
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
            lw_ffcf[p,...], mw, aw, bw, zf1, zf2 = sfun.start_weight(lw_ffcf[p-1],...)
        Feed supply Loop for offspring
            #` mei and rc are not defined
            mei[p,...] = sfun.intake(rc, c_ci_gy, )
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values
        Feed supply Loop for rams #Probably will never need to loop this
            #because not specifying a target for the rams
            mei[p,...] = sfun.intake(....)
            mem = sfun.energy(....)
            dcfw, new = sfun.wool_growth(....)
            cfw = cfw_start + dcfw
            wmax = np.maximum(lw_ffcf,axis=0)
            lwc = sfun.lw_change(mei[p,...], mem, new, wmax, zf1, zf2)
            .... = sfun.end_values

def parameters():
    """Parameter generation for the pyomo variables


    Returns
    -------
    dictionaries for pyomo
    """
parameters = np.zeros((len(output_required),len(activities0)), dtype = 'float64')
    # Loop through the number of variables
    for a in activites:
        ### create array masks  for the pyomo variable
        ''' For each pyomo variable create a mask that represents the animals
        The arrays can then be summed across the axes for that mask '''
        mask = sfun.create_mask(i_activity_definition)

        ### apply each mask to each simulation output
        #output_required is a list of the arrays that are required as parameters
        for n, o in enumerate(output_required):
            parameters[n,a] = np.sum(o[mask])

return parameters

''' Or to allow one function call per constraint this function could
generate the array and then multiple functions that just return the
required row of the array.'''