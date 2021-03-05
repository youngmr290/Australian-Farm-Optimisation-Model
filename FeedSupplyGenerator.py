# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:35:18 2020

@author: John


A module of 2 functions that is called by FeedTest.py to calculate the FEC for increments of the feed supply inputs
and the allocation of generator periods to FVPs (for graphing).
The structure is based on the Sheep Generator but only calculates the feed inputs (FEC) for dams
 (because sheep class doesn't alter the values)
For sheep, FEC is the same for offspring because no effect of FOO & DMD on intake (small impact for cattle of bite size)
The FVPs are calculated for both dams and offspring.


"""


"""
import functions from other modules
"""
import numpy as np

import Functions as fun
import PropertyInputs as pinp
import StockFunctions as sfun
import UniversalInputs as uinp




def feed_generator():
    """
    A function that generates the feed energy concentration (FEC) for each feed supply value in each feed period.

    Called after the sensitivity variables have been updated.
    It populates the arrays by looping through the feed periods

    Returns
    -------
    None.
    """

    print("starting feed supply generator")

    ######################
    ##date               #
    ######################
    na=np.newaxis
    ## define the periods - default (dams and sires)
    n_sim_periods = 11  #number of feed periods + 1
    date_start_p = np.array(pinp.feed_inputs['feed_periods']['date']).astype('datetime64[D]')[:-1] #convert from df to numpy remove last date because that is the end date of the last period (not required)
    # date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(uinp.structure['i_p_pos']+1, 0)))


    ###################################
    ### axis len & index arrays       #
    ###################################
    ## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
    len_p = len(date_start_p)
    #len_n = 3
    len_w = 301

    index_p = np.arange(len_p)
    index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(index_p, axis=tuple(range(uinp.structure['i_p_pos'] + 1, 0)))
    # index_n = np.arange(len_n)
    # index_nwzida0e0b0xyg = np.expand_dims(index_n, axis=tuple(range(uinp.structure['i_n_pos'] + 1, 0)))
    index_w = np.arange(len_w)
    index_wzida0e0b0xyg = np.expand_dims(index_w, axis=tuple(range(uinp.structure['i_w_pos'] + 1, 0)))

    ###################################
    ### Create feedsupply             #
    ###################################

    # create a feed supply between 0 and 3 in 0.1 increments across the w axis.
    feedsupply = index_wzida0e0b0xyg / 100 #  + index_nwzida0e0b0xyg
    ## make the last entry 4, which is ad lib supplement in confinement
    feedsupply[-1] = 4



    ############################
    ### initialise arrays      #
    ############################
    '''only if assigned with a slice'''
    ##unique array shapes required to initialise arrays
    pw1 = (len_p, len_w)


    ##output variables for postprocessing & reporting
    dtype ='float64' #using 64 was getting slow

    ##dams
    ###arrays for postprocessing
    o_pi = np.zeros(pw1, dtype = dtype)
    o_mei = np.zeros(pw1, dtype = dtype)


    ###################################
    ###group independent              #  type(pinp.sheep['i_mask_z']).dtype
    ###################################
    ##legume proportion in each period
    legume_p6a1e1b1nwzida0e0b0xyg = fun.f_reshape_expand(pinp.sheep['i_legume_p6z'], pinp.sheep['i_z_pos'], pinp.sheep['i_p6_len'], pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (association section)
    ##estimated foo and dmd for the midas periods - apply z mask
    paststd_foo_p6a1e1b1j0wzida0e0b0xyg = fun.f_reshape_expand(pinp.sheep['i_paststd_foo_p6zj0'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], len_ax2=pinp.feedsupply['i_j0_len'], swap=True, ax1=1, ax2=2, left_pos2=uinp.structure['i_n_pos'], right_pos2=pinp.sheep['i_z_pos'], left_pos3=uinp.structure['i_p_pos'], right_pos3=uinp.structure['i_n_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (association section), axis order doesnt matter because sliced when used
    paststd_dmd_p6a1e1b1j0wzida0e0b0xyg = fun.f_reshape_expand(pinp.sheep['i_paststd_dmd_p6zj0'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], len_ax2=pinp.feedsupply['i_j0_len'], swap=True, ax1=1, ax2=2, left_pos2=uinp.structure['i_n_pos'], right_pos2=pinp.sheep['i_z_pos'], left_pos3=uinp.structure['i_p_pos'], right_pos3=uinp.structure['i_n_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (association section), axis order doesnt matter because sliced when used
    pasture_stage_p6a1e1b1j0wzida0e0b0xyg = fun.f_reshape_expand(pinp.sheep['i_pasture_stage_p6z'], pinp.sheep['i_z_pos'], len_ax0=pinp.sheep['i_p6_len'], len_ax1=pinp.sheep['i_z_len'], left_pos2=uinp.structure['i_p_pos'], right_pos2=pinp.sheep['i_z_pos'], condition = pinp.sheep['i_mask_z'], axis = pinp.sheep['i_z_pos']) #p6 axis converted to p axis later (association section)
    ##Calculation of rainfall distribution across the week - i_rain_distribution_m4m1 = how much rain falls on each day of the week sorted in order of quantity of rain. SO the most rain falls on the day with the highest rainfall.
    rain_m4a1e1b1nwzida0e0b0xygm1 = fun.f_reshape_expand(pinp.sheep['i_rain_m4'][...,na] * pinp.sheep['i_rain_distribution_m4m1'] * (7/30.4), uinp.structure['i_p_pos']-1,right_pos=-1) #-1 because p is -16 when m1 axis is included
    ##Mean daily temperature
    temp_ave_m4a1e1b1nwzida0e0b0xyg= fun.f_reshape_expand(pinp.sheep['i_temp_ave_m4'], uinp.structure['i_p_pos'])
    ##Mean daily maximum temperature
    temp_max_m4a1e1b1nwzida0e0b0xyg= fun.f_reshape_expand(pinp.sheep['i_temp_max_m4'], uinp.structure['i_p_pos'])
    ##Mean daily minimum temperature
    temp_min_m4a1e1b1nwzida0e0b0xyg= fun.f_reshape_expand(pinp.sheep['i_temp_min_m4'], uinp.structure['i_p_pos'])
    ##latitude

    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from c to g
    ###production params
    srw_female_yg0, srw_female_yg1, srw_female_yg2, srw_female_yg3 = sfun.f_c2g(uinp.parameters['i_srw_c2'], uinp.parameters['i_srw_y']) #srw of a female of the given genotype (this is the definition of the inputs)

    ###sim params
    ci_sire, ci_dams, ci_yatf, ci_offs = sfun.f_c2g(uinp.parameters['i_ci_c2'], uinp.parameters['i_ci_y'], uinp.parameters['i_ci_pos'], uinp.parameters['i_ci_len'])
    cl_sire, cl_dams, cl_yatf, cl_offs = sfun.f_c2g(uinp.parameters['i_cl_c2'], uinp.parameters['i_cl_y'], uinp.parameters['i_cl_pos'], uinp.parameters['i_cl_len'])
    cr_sire, cr_dams, cr_yatf, cr_offs = sfun.f_c2g(uinp.parameters['i_cr_c2'], uinp.parameters['i_cr_y'], uinp.parameters['i_cr_pos'], uinp.parameters['i_cr_len'])
    cu0_sire, cu0_dams, cu0_yatf, cu0_offs = sfun.f_c2g(uinp.parameters['i_cu0_c2'], uinp.parameters['i_cu0_y'], uinp.parameters['i_cu0_pos'], uinp.parameters['i_cu0_len'])
    cx_sire, cx_dams, cx_yatf, cx_offs = sfun.f_c2g(uinp.parameters['i_cx_c2'], uinp.parameters['i_cx_y'], uinp.parameters['i_cx_pos'], uinp.parameters['i_cx_len'], uinp.parameters['i_cx_len2'])
    ##pasture params
    cu3 = uinp.pastparameters['i_cu3_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu3_len'], uinp.pastparameters['i_cu3_len2']).astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)
    cu4 = uinp.pastparameters['i_cu4_c4'][...,pinp.sheep['i_pasture_type']].reshape(uinp.pastparameters['i_cu4_len'], uinp.pastparameters['i_cu4_len2']).astype(float)#have to convert from object to float so it doesnt chuck error in np.exp (np.exp cant handle object arrays)



    ############################
    ### associations           #
    ############################
    ### Aim is to create an association array that points from period to opportunity
    ## The date_p array will be the start date for each period.
    ## A pointer (association) is required that points from the period to the previous or next lambing opportunity
    ## The Lambing opportunity is defined by the joining_date array (which has an 'o' axis)
    ## The previous lambing opportunity is the latest joining date that is less than the date at the end of the period (ie previous/current opportunity)
    ## # ('end of the period' so that if joining occurs during the period it is the previous
    ## The next lambing opportunity is the earliest joining date that is greater than or equal to the end date at the end of the period
    ##prev is just next - 1





    ##month of each period (0 - 11 not 1 -12 because this is association array)
    a_m4_p = date_start_p.astype('datetime64[M]').astype(int) % 12


    ############################
    ### apply associations     #
    ############################
    '''
    The association applied determines when the increment to the next opportunity will occur:
        eg if you use a_prev_joining the date in the p slice will increment at joining each time.
    
    '''
    ##feed period
    legume_pa1e1b1nwzida0e0b0xyg = legume_p6a1e1b1nwzida0e0b0xyg
    ##select which equation is used for the sheep sim functions for each period
    eqn_used_g1_q1p = uinp.sheep['i_eqn_used_g1_q1p7'][:, 0:1]
    ##convert foo and dmd for each feed period to each sim period
    paststd_foo_pa1e1b1j0wzida0e0b0xyg = paststd_foo_p6a1e1b1j0wzida0e0b0xyg
    paststd_dmd_pa1e1b1j0wzida0e0b0xyg = paststd_dmd_p6a1e1b1j0wzida0e0b0xyg
    pasture_stage_pa1e1b1j0wzida0e0b0xyg = pasture_stage_p6a1e1b1j0wzida0e0b0xyg
 #add g0 axis
    ##weather
    rain_pa1e1b1nwzida0e0b0xygm1 = rain_m4a1e1b1nwzida0e0b0xygm1[a_m4_p]
    temp_ave_pa1e1b1nwzida0e0b0xyg= temp_ave_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
    temp_max_pa1e1b1nwzida0e0b0xyg= temp_max_m4a1e1b1nwzida0e0b0xyg[a_m4_p]
    temp_min_pa1e1b1nwzida0e0b0xyg= temp_min_m4a1e1b1nwzida0e0b0xyg[a_m4_p]


    ###########################
    ##genotype calculations   #
    ###########################

    ###gender adjustment for srw
    srw_xyg1 = srw_female_yg1 * cx_dams[11, 1:2, ...]



    ############################
    ### Daily steps            #
    ############################
    ##definition for this is that the action eg weaning occurs at 12am on the given date. therefore is weaning occurs on day 150 the lambs are counted as weaned lambs on that day.
    ##This info determines the side with > or >=

    ##Impact of rainfall on 'cold' intake increment
    rain_intake_pa1e1b1nwzida0e0b0xyg1 = fun.f_weighted_average(np.maximum(0, 1 - rain_pa1e1b1nwzida0e0b0xygm1 / ci_dams[18, ..., na]),  weights=np.array([1]), axis = -1)




    #####################################
    ##expand feedsupply for all w slices#
    #####################################
    ##add p axis to feedsupply
    feedsupplyw_pa1e1b1nwzida0e0b0xyg1 = feedsupply * (index_pa1e1b1nwzida0e0b0xyg >= 0)


    ####################################
    ### initialise arrays for sim loop  # axis names not always track from now on because they change between p=0 and p=1
    ####################################
    ##dams
    temp_lc_dams = np.array([0.0]) #this is calculated in the chill function but it is required for the intake function so it is set to 0 for the first period.


    ######################
    ### sim engine       #
    ######################


    ## Loop through each week of the simulation (p) for dams.
    for p in range(n_sim_periods-1):   #-1 because assigns to [p+1] for start values

        ###################
        ##dependent start #
        ###################
        ##dams
        ###Relative condition (start)
        rc_start_dams = np.array([1])
        ###Relative size (start) - dams & sires
        relsize_start_dams = np.array([1])
        ###PI Size factor (for cattle)
        zf_dams = np.maximum(1, 1 + cr_dams[7, ...] - relsize_start_dams)




        ##feed supply loop
        ##this loop is only required if a LW target is specified for the animals
        ##if there is a target then the loop needs to continue until
        ##the feed supply has converged on a value that generates a liveweight
        ##change close to the target
        ##The loop needs to execute at least once, then repeat if there
        ##is a target and the result is not close enough to the target


        ##potential intake
        eqn_group = 4
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                pi_dams = sfun.f_potential_intake_cs(ci_dams, cl_dams, srw_xyg1, relsize_start_dams, rc_start_dams, temp_lc_dams
                                                   , temp_ave_pa1e1b1nwzida0e0b0xyg[p], temp_max_pa1e1b1nwzida0e0b0xyg[p]
                                                  , temp_min_pa1e1b1nwzida0e0b0xyg[p], rain_intake_pa1e1b1nwzida0e0b0xyg1[p])

        ###murdoch ^function doesnt exist yet, add args when it is built
        eqn_system = 1 # mu = 1
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                pi_dams = sfun.f_potential_intake_mu(srw_xyg1)

        ##feedsupply - calculated after pi because pi required for intake_s
        foo_dams, hf_dams, dmd_dams, intake_s_dams, md_herb_dams = \
            sfun.f_feedsupply(cu3, cu4, cr_dams, feedsupplyw_pa1e1b1nwzida0e0b0xyg1[p], paststd_foo_pa1e1b1j0wzida0e0b0xyg[p]
                              , paststd_dmd_pa1e1b1j0wzida0e0b0xyg[p], legume_pa1e1b1nwzida0e0b0xyg[p], pi_dams
                              , pasture_stage_pa1e1b1j0wzida0e0b0xyg[p], pinp.sheep['i_hr_scalar'], pinp.sheep['i_region']
                              , uinp.pastparameters['i_n_pasture_stage'], uinp.pastparameters['i_hd_std'])

        ##relative availability
        eqn_group = 5
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                ra_dams = sfun.f_ra_cs(foo_dams, hf_dams, cr_dams, zf_dams)

        eqn_system = 1 # Murdoch = 1
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                ra_dams = sfun.f_ra_mu(cu0_dams, foo_dams, hf_dams, zf_dams)


        ##relative quality/ingestibility
        eqn_group =  6
        eqn_system = 0 # CSIRO = 0
        if uinp.sheep['i_eqn_exists_q0q1'][eqn_group, eqn_system]:  # proceed with call & assignment if this system exists for this group
            ###dams
            eqn_used = (eqn_used_g1_q1p[eqn_group, 0] == eqn_system)
            if eqn_used:
                rq_dams = sfun.f_rq_cs(dmd_dams, legume_pa1e1b1nwzida0e0b0xyg[p], cr_dams, pinp.sheep['i_sf'])

        ##intake
        mei_dams, mei_solid_dams, intake_f_dams, md_solid_dams, mei_propn_milk_dams, mei_propn_herb_dams, mei_propn_supp_dams = \
            sfun.f_intake(cr_dams, pi_dams, ra_dams, rq_dams,  md_herb_dams, feedsupplyw_pa1e1b1nwzida0e0b0xyg1[p]
                          , intake_s_dams, pinp.sheep['i_md_supp'], legume_pa1e1b1nwzida0e0b0xyg[p])



        ######################################
        #store postprocessing and report vars#
        ######################################
        ###dams
        ### Return the n & w axes with everything else slice 0 (because Pandas can only handle a 2D array)
        pi = pi_dams[0,0,0,0,:,0,0,0,0,0,0,0,0,0]
        mei = mei_solid_dams[0,0,0,0,:,0,0,0,0,0,0,0,0,0]
        ###store output variables for the post processing
        o_pi[p] = pi
        o_mei[p] = mei

    # calculate feed energy concentration
    r_fec = o_mei / o_pi
    # print (r_fec)

    return r_fec, np.squeeze(feedsupply)



def period_generator():
    '''A function that creates the generator periods and feed variation periods 
    that are used as part of creating the feedsupply inputs'''

    ######################
    ##background vars    #
    ######################
    na=np.newaxis
    p_pos = uinp.structure['i_p_pos']
#    a1_pos = pinp.sheep['i_a1_pos']
    e1_pos = pinp.sheep['i_e1_pos']
#    b1_pos = uinp.parameters['i_b1_pos']
#    n_pos = uinp.structure['i_n_pos']
#    w_pos = uinp.structure['i_w_pos']
#    z_pos = pinp.sheep['i_z_pos']
    i_pos = pinp.sheep['i_i_pos']
    d_pos = uinp.parameters['i_d_pos']
    a0_pos = pinp.sheep['i_a0_pos']
    e0_pos = uinp.structure['i_e0_pos']
#    b0_pos = uinp.parameters['i_b0_pos']
    x_pos = uinp.parameters['i_x_pos']
#    y_pos = uinp.parameters['i_y_pos']

    ######################
    ##date               #
    ######################
    ## define the periods - default (dams and sires)
    sim_years = uinp.structure['i_age_max']
    sim_years_offs = min(uinp.structure['i_age_max_offs'], sim_years)
    n_sim_periods, date_start_p, date_end_p, p_index_p, step \
        = sfun.sim_periods(pinp.sheep['i_startyear'], uinp.structure['i_sim_periods_year'], sim_years)
    date_start_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_start_p, axis = tuple(range(p_pos+1, 0)))
#    date_end_pa1e1b1nwzida0e0b0xyg = np.expand_dims(date_end_p, axis = tuple(range(p_pos+1, 0)))
#    p_index_pa1e1b1nwzida0e0b0xyg = np.expand_dims(p_index_p, axis = tuple(range(p_pos+1, 0)))
    ## define the periods - offs - these make the p axis customisable for offs which means they can be smaller
    n_sim_periods_offs, offs_date_start_p, offs_date_end_p, p_index_offs_p, step \
        = sfun.sim_periods(pinp.sheep['i_startyear'], uinp.structure['i_sim_periods_year'], sim_years_offs)
    date_start_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_start_p, axis = tuple(range(p_pos+1, 0)))
#    date_end_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(offs_date_end_p, axis = tuple(range(p_pos+1, 0)))
#    p_index_pa1e1b1nwzida0e0b0xyg3 = np.expand_dims(p_index_offs_p, axis = tuple(range(p_pos+1, 0)))
#    mask_p_offs_p = p_index_p<=(n_sim_periods_offs-1)

    ###################################
    ## calculate masks                #
    ###################################
    ##masks required for initialising arrays
#    mask_sire_inc_g0 = np.any(uinp.structure['i_mask_g0g3'] * pinp.sheep['i_g3_inc'], axis =1)
#    mask_dams_inc_g1 = np.any(uinp.structure['i_mask_g1g3'] * pinp.sheep['i_g3_inc'], axis =1)
#    mask_offs_inc_g3 = np.any(uinp.structure['i_mask_g3g3'] * pinp.sheep['i_g3_inc'], axis =1)
    ##o/d mask - if dob is after the end of the sim then it is masked out -  the mask is created before the date of birth is adjusted to the start of a period however it is adjusted to the start of the next period so the mask wont cut out a birth event that actually would occur, additionally this is the birth of the first however the matrix sees the birth of average animal which is also later therefore if anything the mask will leave in unnecessary o slices
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = sfun.f_g2g(pinp.sheep['i_date_born1st_oig2'],'yatf',i_pos,pinp.sheep['i_i_len'],pinp.sheep['i_o_len'],swap=True,left_pos2=p_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]') #left2 = e1-1 because e1 needs to be included for the calculation following
    mask_o_dams = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_d_offs = np.max(date_born1st_oa1e1b1nwzida0e0b0xyg2<=date_end_p[-1], axis=tuple(range(p_pos+1, 0))) #compare each birth opp with the end date of the sim and make the mask - the mask is of the longest axis (ie to handle situations where say bbb and bbm have birth at different times so one has 6 opp and the other has 5 opp)
    mask_x = pinp.sheep['i_gender_propn_x']>0
    fvp_mask_dams = uinp.structure['i_fvp_mask_dams']


    # ###################################
    # ### axis len                      #
    # ###################################
    ## Final length of axis after any masks have been applied, used to initialise arrays and in code below (note: these are not used to reshape input array).
#    len_p = len(date_start_p)

    ###################################
    ### index arrays                  #
    ###################################
    index_e = np.arange(np.max(pinp.sheep['i_join_cycles_ig1']))
    index_e1b1nwzida0e0b0xyg = fun.f_reshape_expand(index_e, e1_pos)
    index_e0b0xyg = fun.f_reshape_expand(index_e, e0_pos)

    ##output variables for postprocessing & reporting
#    dtype='float32' #using 64 was getting slow

    ################################################
    #  management, age, date, timing inputs inputs #
    ################################################
    ##gender propn yatf
#    gender_propn_xyg = fun.f_reshape_expand(pinp.sheep['i_gender_propn_x'], x_pos, condition=mask_x, axis=0).astype(dtype)
    ##join
    join_cycles_ida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_join_cycles_ig1'],'dams',i_pos)[pinp.sheep['i_mask_i'],...]
    ##lamb and lost
#    gbal_oa1e1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_gbal_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
#    gbal_da0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_gbal_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied
    ##scanning
    scan_oa1e1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_scan_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
#    scan_da0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_scan_og1'],'dams',d_pos, condition=mask_d_offs, axis=d_pos) #need axis up to p so that p association can be applied
    ##post weaning management
#    wean_oa1e1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_wean_og1'],'dams',p_pos, condition=mask_o_dams, axis=p_pos) #need axis up to p so that p association can be applied
    ##association between offspring and sire/dam (used to determine the wean age of sire and dams based on the inputted wean age of offs)
#    a_g0_g1 = sfun.f_g2g(pinp.sheep['ia_g0_g1'],'dams')
#    a_g3_g0 = sfun.f_g2g(pinp.sheep['ia_g3_g0'],'sire')  # the sire association (pure bred B, M & T) are all based on purebred B because there are no pure bred M & T inputs
#    a_g3_g1 = sfun.f_g2g(pinp.sheep['ia_g3_g1'],'dams')  # if BMT exist then BBM exist and they will be in slice 1, therefore the association value doesn't need to be adjusted for "prior exclusions"
    ##age weaning- used to calc wean date and also to calc m1 stuff, sire and dams have no active a0 slice therefore just take the first slice
    age_wean1st_a0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_age_wean_a0g3'],'offs',a0_pos).astype('timedelta64[D]')[pinp.sheep['i_mask_a']]
#    age_wean1st_e0b0xyg0 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g0],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
#    age_wean1st_e0b0xyg1 = np.rollaxis(age_wean1st_a0e0b0xyg3[0, ...,a_g3_g1],0,age_wean1st_a0e0b0xyg3.ndim-1) #when you slice one slice of the array and also take multiple slices from another axis the axis with multiple slices jumps to the front therefore need to roll the g axis back to the end
    ##date first lamb is born - need to apply i mask to these inputs - make sure animals are born at beginning of gen period
#    date_born1st_ida0e0b0xyg0 = sfun.f_g2g(pinp.sheep['i_date_born1st_ig0'],'sire',i_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos).astype('datetime64[D]')
#    date_born1st_ida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_date_born1st_ig1'],'dams',i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos).astype('datetime64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2[mask_o_dams,...] #input read in in the mask section
    date_born1st_ida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_date_born1st_idg3'],'offs',d_pos,pinp.sheep['i_i_len'],uinp.parameters['i_d_len']
                                    , condition=pinp.sheep['i_mask_i'], axis=i_pos, condition2=mask_d_offs, axis2=d_pos).astype('datetime64[D]')
    ##mating
#    sire_propn_oa1e1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_sire_propn_oig1'],'dams',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_o_len']
#                                                   , swap=True, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos
#                                                   , condition2=mask_o_dams, axis2=p_pos)
#    sire_periods_p8g0 = sfun.f_g2g(pinp.sheep['i_sire_periods_p8g0'], 'sire', condition=pinp.sheep['i_mask_p8'], axis=0)
#    sire_periods_g0p8 = np.swapaxes(sire_periods_p8g0, 0, 1) #cant swap in function above because g needs to be in pos-1
    ##Shearing date - set to be on the last day of a sim period
    ###sire
#    date_shear_sida0e0b0xyg0 = sfun.f_g2g(pinp.sheep['i_date_shear_sixg0'],'sire',x_pos,pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len']
#                                          ,swap=True,left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_masksire_i'], axis=i_pos
#                                          )[...,0:1,:,:].astype('datetime64[D]') #slice x axis for only female
#    mask_shear_g0 = np.max(date_shear_sida0e0b0xyg0<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
#    date_shear_sida0e0b0xyg0 = date_shear_sida0e0b0xyg0[mask_shear_g0]
    ###dam
#    date_shear_sida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['i_date_shear_sixg1'],'dams',x_pos,pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len']
#                                          ,swap=True,left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos
#                                          )[...,1:2,:,:].astype('datetime64[D]') #slice x axis for only female
#    mask_shear_g1 = np.max(date_shear_sida0e0b0xyg1<=date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
#    date_shear_sida0e0b0xyg1 = date_shear_sida0e0b0xyg1[mask_shear_g1]
    ###off - shearing cant occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_date_shear_sixg3'],'offs',x_pos,pinp.sheep['i_i_len'], pinp.sheep['i_s_len'], pinp.sheep['i_x_len']
                                          ,swap=True,left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos
                                          , condition2=mask_x, axis2=x_pos).astype('datetime64[D]')
    mask_shear_g3 = np.max(date_shear_sida0e0b0xyg3<=offs_date_end_p[-1], axis=tuple(range(i_pos, 0))) #mask out shearing opps that occur after gen is done
    date_shear_sida0e0b0xyg3 = date_shear_sida0e0b0xyg3[mask_shear_g3]

    ############################
    ### feed supply inputs     #
    ############################
    ##feedsupply
    ###feedsupply option selected
#    a_r_zida0e0b0xyg0 = sfun.f_g2g(pinp.sheep['ia_r1_zig0'],'sire',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True, condition=pinp.sheep['i_masksire_i'], axis=i_pos)
#    a_r_zida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['ia_r1_zig1'],'dams',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True, condition=pinp.sheep['i_mask_i'], axis=i_pos)
#    a_r_zida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['ia_r1_zig3'],'offs',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_z_len'],swap=True, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    ###feed variation for dams
#    a_r2_k0e1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['ia_r2_k0ig1'],'dams',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k0_len'],swap=True,left_pos2=a1_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
#    a_r2_k1b1nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['ia_r2_k1ig1'],'dams',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k1_len'],swap=True,left_pos2=e1_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
#    a_r2_k2nwzida0e0b0xyg1 = sfun.f_g2g(pinp.sheep['ia_r2_k2ig1'],'dams',i_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k2_len'],swap=True,left_pos2=b1_pos,right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)  #add axis between g and i and i and b1
    ###feed variation for offs
#    a_r2_idk0e0b0xyg3 = sfun.f_g2g(pinp.sheep['ia_r2_ik0g3'],'offs',a0_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k0_len'],left_pos2=i_pos,right_pos2=a0_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
#    a_r2_ik3a0e0b0xyg3 = sfun.f_g2g(pinp.sheep['ia_r2_ik3g3'],'offs',d_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k3_len'], condition=pinp.sheep['i_mask_i'], axis=i_pos)
#    a_r2_ida0e0k4xyg3 = sfun.f_g2g(pinp.sheep['ia_r2_ik4g3'],'offs',b0_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k4_len'],left_pos2=i_pos,right_pos2=b0_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)  #add axis between g and b0 and b0 and i
#    a_r2_ida0e0b0k5yg3 = sfun.f_g2g(pinp.sheep['ia_r2_ik5g3'],'offs',x_pos,pinp.sheep['i_i_len'], pinp.sheep['i_k5_len'],left_pos2=i_pos,right_pos2=x_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)  #add axis between g and b0 and b0 and i

    ##std feed options
#    feedoptions_r1j0p = pinp.feedsupply['i_feedoptions_r1pj0'].reshape(pinp.feedsupply['i_r1_len'],pinp.feedsupply['i_j0_len'],pinp.feedsupply['i_feedoptions_r1pj0'].shape[-1])[...,0:len_p].astype(np.float) #slice off extra p periods so it is the same length as the sim periods
    ##feed variation
#    feedoptions_var_r2p = pinp.feedsupply['i_feedoptions_var_r2p'][:,0:len_p].astype(np.float) #slice off extra p periods so it is the same length as the sim periods
    ##an association between the k2 cluster (feed variation) and reproductive management (scanning, gbal & weaning).
#    a_k2_mlsb1 = uinp.structure['ia_k2_mlsb1'].reshape(uinp.structure['i_len_m'], uinp.structure['i_len_l'], uinp.structure['i_len_s'], uinp.structure['ia_k2_mlsb1'].shape[-1])

    ############
    #fvp inputs#
    ############
    fvp0_offset_ida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_fvp0_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp1_offset_ida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_fvp1_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    fvp2_offset_ida0e0b0xyg3 = sfun.f_g2g(pinp.sheep['i_fvp2_offset_ig3'], 'offs', i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)

    ############################
    ### sim param arrays       # '''csiro params '''
    ############################
    ##convert input params from c to g
    ###production params
    ###sim params
    cf_sire, cf_dams, cf_yatf, cf_offs = sfun.f_c2g(uinp.parameters['i_cf_c2'], uinp.parameters['i_cf_y'], uinp.parameters['i_cf_pos'], uinp.parameters['i_cf_len'])
    cp_sire, cp_dams, cp_yatf, cp_offs = sfun.f_c2g(uinp.parameters['i_cp_c2'], uinp.parameters['i_cp_y'], uinp.parameters['i_cp_pos'], uinp.parameters['i_cp_len'])


    ########################################
    #adjust input to align with gen periods#
    ########################################
    # 1. Adjust date born (average) = start period
    # 2. Calc date born1st from slice 0 subtract 8 days
    # 3. Calc wean date
    # 4 adjust wean date to occur at start period
    # 5 calc adjusted wean age

    ##calc and adjust date born average of group - convert from date of first lamb born to average date born of lambs in the first cycle
    ###sire
#    date_born_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
#    date_born_idx_ida0e0b0xyg0=sfun.f_next_prev_association(date_start_p, date_born_ida0e0b0xyg0, 0, 'left')
#    date_born_ida0e0b0xyg0 = date_start_p[date_born_idx_ida0e0b0xyg0]
    ###dams
#    date_born_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
#    date_born_idx_ida0e0b0xyg1=sfun.f_next_prev_association(date_start_p, date_born_ida0e0b0xyg1, 0, 'left')
#    date_born_ida0e0b0xyg1 = date_start_p[date_born_idx_ida0e0b0xyg1]
    ###yatf
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + (index_e1b1nwzida0e0b0xyg + 0.5) * cf_yatf[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle. e_index is to account for ewe cycles.
    date_born_idx_oa1e1b1nwzida0e0b0xyg2 =sfun.f_next_prev_association(date_start_p, date_born_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_born_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_born_idx_oa1e1b1nwzida0e0b0xyg2]
    ###offs
    date_born_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + (index_e0b0xyg + 0.5) * cf_offs[4, 0:1,:].astype('timedelta64[D]')	 #times by 0.5 to get the average birth date for all lambs because ewes can be conceived anytime within joining cycle
    date_born_idx_ida0e0b0xyg3=sfun.f_next_prev_association(offs_date_start_p, date_born_ida0e0b0xyg3, 0, 'left')
    date_born_ida0e0b0xyg3 = offs_date_start_p[date_born_idx_ida0e0b0xyg3]
    ##recalc date_born1st from the adjusted average birth date
#    date_born1st_ida0e0b0xyg0 = date_born_ida0e0b0xyg0 - 0.5 * cf_sire[4, 0:1,:].astype('timedelta64[D]')
#    date_born1st_ida0e0b0xyg1 = date_born_ida0e0b0xyg1 - 0.5 * cf_dams[4, 0:1,:].astype('timedelta64[D]')
    date_born1st_oa1e1b1nwzida0e0b0xyg2 = date_born_oa1e1b1nwzida0e0b0xyg2[:,:,0:1,...] - 0.5 * cf_yatf[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis
    date_born1st_ida0e0b0xyg3 = date_born_ida0e0b0xyg3[:,:,:,0:1,...] - 0.5 * cf_offs[4, 0:1,:].astype('timedelta64[D]') #take slice 0 of e axis

    ##calc wean date (weaning input is counting from the date of the first lamb)
#    date_weaned_ida0e0b0xyg0 = date_born1st_ida0e0b0xyg0 + age_wean1st_e0b0xyg0
#    date_weaned_ida0e0b0xyg1 = date_born1st_ida0e0b0xyg1 + age_wean1st_e0b0xyg1
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_born1st_oa1e1b1nwzida0e0b0xyg2 + age_wean1st_a0e0b0xyg3 #use off wean age
    date_weaned_ida0e0b0xyg3 = date_born1st_ida0e0b0xyg3 + age_wean1st_a0e0b0xyg3
    ###adjust weaning to occur at the beginning of generator period and recalc wean age
    ####sire
#    date_weaned_idx_ida0e0b0xyg0=sfun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg0, 0, 'left')
#    date_weaned_ida0e0b0xyg0 = date_start_p[date_weaned_idx_ida0e0b0xyg0]
#    age_wean1st_ida0e0b0xyg0 = date_weaned_ida0e0b0xyg0 - date_born1st_ida0e0b0xyg0
    ####dams
#    date_weaned_idx_ida0e0b0xyg1=sfun.f_next_prev_association(date_start_p, date_weaned_ida0e0b0xyg1, 0, 'left')
#    date_weaned_ida0e0b0xyg1 = date_start_p[date_weaned_idx_ida0e0b0xyg1]
#    age_wean1st_ida0e0b0xyg1 = date_weaned_ida0e0b0xyg1 -date_born1st_ida0e0b0xyg1
    ####yatf - this is the same as offs except without the offs periods (offs potentially have less periods)
    date_weaned_idx_oa1e1b1nwzida0e0b0xyg2=sfun.f_next_prev_association(date_start_p, date_weaned_oa1e1b1nwzida0e0b0xyg2, 0, 'left')
    date_weaned_oa1e1b1nwzida0e0b0xyg2 = date_start_p[date_weaned_idx_oa1e1b1nwzida0e0b0xyg2]
#    age_wean1st_oa1e1b1nwzida0e0b0xyg2 = date_weaned_oa1e1b1nwzida0e0b0xyg2 - date_born1st_oa1e1b1nwzida0e0b0xyg2
    ####offs
    date_weaned_idx_ida0e0b0xyg3=sfun.f_next_prev_association(offs_date_start_p, date_weaned_ida0e0b0xyg3, 0, 'left')
    date_weaned_ida0e0b0xyg3 = offs_date_start_p[date_weaned_idx_ida0e0b0xyg3]
    age_wean1st_ida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 - date_born1st_ida0e0b0xyg3

    ##Shearing date - set to be on the last day of a sim period
    ###sire
#    idx_sida0e0b0xyg0 = sfun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg0,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
#    date_shear_sa1e1b1nwzida0e0b0xyg0 = fun.f_reshape_expand(date_end_p[idx_sida0e0b0xyg0], p_pos, right_pos=i_pos)
    ###dam
#    idx_sida0e0b0xyg1 = sfun.f_next_prev_association(date_end_p, date_shear_sida0e0b0xyg1,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
#    date_shear_sa1e1b1nwzida0e0b0xyg1 = fun.f_reshape_expand(date_end_p[idx_sida0e0b0xyg1], p_pos, right_pos=i_pos)
    ###off - shearing cant occur as yatf because then need to shear all lambs (ie no scope to not shear the lambs that are going to be fed up and sold) because the offs decision variables for feeding are not linked to the yatf (which are in the dam decision variables)
    date_shear_sida0e0b0xyg3 = np.maximum(date_born1st_ida0e0b0xyg3 + age_wean1st_ida0e0b0xyg3, date_shear_sida0e0b0xyg3) #shearing must be after weaning.
    idx_sida0e0b0xyg3 = sfun.f_next_prev_association(offs_date_end_p, date_shear_sida0e0b0xyg3,0, 'left')#shearing occurs at the end of the next/current generator period therefore 0 offset
    date_shear_sa1e1b1nwzida0e0b0xyg3 = fun.f_reshape_expand(offs_date_end_p[idx_sida0e0b0xyg3], p_pos, right_pos=i_pos)


    ############################
    ## calc for associations   #
    ############################
    ##date joined (when the rams go in)
    date_joined_oa1e1b1nwzida0e0b0xyg1 = date_born1st_oa1e1b1nwzida0e0b0xyg2 - cp_dams[1,...,0:1,:].astype('timedelta64[D]') #take slice 0 from y axis because cp1 is not affected by genetic merit
    ##expand feed periods over all the years of the sim so that an association between sim period can be made.
#    feedperiods_p6 = np.array(pinp.feed_inputs['feed_periods']['date']).astype('datetime64[D]')[:-1] #convert from df to numpy remove last date because that is the end date of the last period (not required)
#    feedperiods_p6 = feedperiods_p6 + np.timedelta64(365,'D') * ((date_start_p[0].astype(object).year -1) - feedperiods_p6[0].astype(object).year) #this is to make sure the first sim period date is greater than the first feed period date.
#    feedperiods_p6 = np.ravel(feedperiods_p6  + (np.arange(np.ceil(sim_years +1)) * np.timedelta64(365,'D') )[...,na]) #expand then ravel to return 1d array of the feed period dates expanded the length of the sim. +1 because feed periods start and finish mid yr so add one to ensure they go to the end of the sim.

    ###################################
    # Feed variation period calcs dams#  ^note if type stuff is used then might need to add a special fvp that is the start of the simulation ie 1/1/19
    ###################################
    ##fvp/dvp types
#    season_vtype1=0
#    prejoin_vtype1 = 0 #todo once season dvp properly added need to add 1 to other dvp types
#    condense_vtype1 = prejoin_vtype1 #lw patterns are condensed (go back to 3) at prejoining
#    scan_vtype1 = 1
#    birth_vtype1 = 2
#    wean_ftype1 = 3
#    other_ftype1 = 4

    ##beginning - first day of generator
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = date_start_pa1e1b1nwzida0e0b0xyg[0:1]
    ##early pregnancy fvp start - The pre-joining accumulation of the dams from the previous reproduction cycle - this date must correspond to the start date of period
    prejoining_approx_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 - uinp.structure['prejoin_offset'] #approx date of prejoining - adjusted to be the start of a sim period in the next step
    idx = np.searchsorted(date_start_p, prejoining_approx_oa1e1b1nwzida0e0b0xyg1, 'right') - 1 #gets the sim period index for the period that prejoining occurs (eg prejoining fvp starts at the beginning of the sim period when prejoining approx occurs), side=right so that if the date is already the start of a period it remains in that period.
    prejoining_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx]
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = prejoining_oa1e1b1nwzida0e0b0xyg1
    ##late pregnancy fvp start - Scanning if carried out, day 90 from joining (ram in) if not scanned.
    late_preg_oa1e1b1nwzida0e0b0xyg1 = date_joined_oa1e1b1nwzida0e0b0xyg1 + join_cycles_ida0e0b0xyg1 * cf_dams[4, 0:1, :].astype('timedelta64[D]') + pinp.sheep['i_scan_day'][scan_oa1e1b1nwzida0e0b0xyg1].astype('timedelta64[D]')
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, late_preg_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when dams in late preg (eg late preg fvp starts at the beginning of the sim period when late preg occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ## lactation fvp start - average date of lambing (with e axis)
    lactation_date_oa1e1b1nwzida0e0b0xyg1 = date_born_oa1e1b1nwzida0e0b0xyg2
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, lactation_date_oa1e1b1nwzida0e0b0xyg1, 'right')-1 #gets the sim period index for the period when lactation starts (eg lactation fvp starts at the beginning of the sim period when lactation occurs), side=right so that if the date is already the start of a period it remains in that period.
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ##weaning
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = date_weaned_oa1e1b1nwzida0e0b0xyg2
    ##user defined fvp - rounded to nearest sim period
    fvp_other_yi = uinp.structure['i_fvp4_date_i'] + np.arange(np.ceil(sim_years))[:,na] * np.timedelta64(365,'D')
    fvp_other_ya1e1b1nwzida0e0b0xyg = fun.f_reshape_expand(fvp_other_yi, left_pos=i_pos, left_pos2=p_pos, right_pos2=i_pos, condition=pinp.sheep['i_mask_i'], axis=i_pos)
    idx_ya1e1b1nwzida0e0b0xyg = np.searchsorted(date_start_p, fvp_other_ya1e1b1nwzida0e0b0xyg, 'right')-1 #gets the sim period index for the period when season breaks (eg break of season fvp starts at the beginning of the sim period when season breaks), side=right so that if the date is already the start of a period it remains in that period.
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = date_start_p[idx_ya1e1b1nwzida0e0b0xyg]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[1:],
                               fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[1:], fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[1:]]) #create shape which has the max size, this is used for o array

    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,(fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_scan_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,(fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_birth_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,(fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_wean_start_oa1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,(fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))
    fvp_other_start_ya1e1b1nwzida0e0b0xyg = np.broadcast_to(fvp_other_start_ya1e1b1nwzida0e0b0xyg,(fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape))
    fvp_begin_start_ba1e1b1nwzida0e0b0xyg1 = np.broadcast_to(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,(fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape))

    ##create fvp type arrays. these are the same shape as the start arrays and are filled with the number corresponding to the fvp number
#    fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape),prejoin_vtype1)
#    fvp_scan_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_scan_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape),scan_vtype1)
#    fvp_birth_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_birth_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape),birth_vtype1)
#    fvp_wean_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_wean_start_oa1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape),wean_ftype1)
#    fvp_other_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_other_start_ya1e1b1nwzida0e0b0xyg.shape[0],)+tuple(shape),other_ftype1)
#    fvp_begin_type_va1e1b1nwzida0e0b0xyg1 = np.full((fvp_begin_start_ba1e1b1nwzida0e0b0xyg1.shape[0],)+tuple(shape),4)
    ##stack & mask which dvps are included
    fvp_date_all_f = np.array([fvp_begin_start_ba1e1b1nwzida0e0b0xyg1,fvp_prejoin_start_oa1e1b1nwzida0e0b0xyg1,
                               fvp_scan_start_oa1e1b1nwzida0e0b0xyg1,fvp_birth_start_oa1e1b1nwzida0e0b0xyg1,
                               fvp_wean_start_oa1e1b1nwzida0e0b0xyg1,fvp_other_start_ya1e1b1nwzida0e0b0xyg], dtype=object)
#    fvp_type_all_f = np.array([fvp_begin_type_va1e1b1nwzida0e0b0xyg1,fvp_prejoin_type_va1e1b1nwzida0e0b0xyg1,
#                               fvp_scan_type_va1e1b1nwzida0e0b0xyg1,fvp_birth_type_va1e1b1nwzida0e0b0xyg1,
#                               fvp_wean_type_va1e1b1nwzida0e0b0xyg1,fvp_other_type_va1e1b1nwzida0e0b0xyg1], dtype=object)
    fvp1_inc = np.concatenate([np.array([True]), fvp_mask_dams]) #True at start is to count for the period from the start of the sim (this is not included in fvp mask because it is not a real fvp as it doesnt occur each year)
    fvp_date_inc_f = fvp_date_all_f[fvp1_inc]
#    fvp_type_inc_f = fvp_type_all_f[fvp1_inc]
    fvp_start_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_date_inc_f,axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.concatenate(fvp_type_inc_f,axis=0)
    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg1, axis=0)
    fvp_date_start_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg1 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg1, ind, axis=0)


    ####################################
    # Feed variation period calcs offs #
    ####################################
    ##fvp's between weaning and first shearing - there will be 3 fvp's equally spaced between wean and first shearing (unless shearing occurs within 3 periods from weaning - if weaning and shearing are close the extra fvp are masked out in the stacking process below)
    ###0
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = date_start_pa1e1b1nwzida0e0b0xyg3[0:1]
    ###1
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b1_start_ba1e1b1nwzida0e0b0xyg3, side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ###2
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = date_weaned_ida0e0b0xyg3 + 2*(date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3)/3
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,side='right')-1 #makes sure fvp starts on the same date as sim period. (-1 get the start date of current period)
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ##fvp0 - date shearing plus 1 day because shearing is the last day of period
    fvp_0_start_oa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + np.maximum(1, fvp0_offset_ida0e0b0xyg3) #plus 1 at least 1 because shearing is the last day of the period and the fvp should start after shearing
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_0_start_oa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period. side=right so that if the date is already the start of a period it remains in that period.
    fvp_0_start_oa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ##fvp1 - date shearing (this is the first day of sim period)
    fvp_1_start_oa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + fvp1_offset_ida0e0b0xyg3
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_1_start_oa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_1_start_oa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_oa1e1b1nwzida0e0b0xyg]
    ##fvp2 - date shearing (this is the first day of sim period)
    fvp_2_start_oa1e1b1nwzida0e0b0xyg3 = date_shear_sa1e1b1nwzida0e0b0xyg3 + fvp2_offset_ida0e0b0xyg3
    idx_oa1e1b1nwzida0e0b0xyg = np.searchsorted(offs_date_start_p, fvp_2_start_oa1e1b1nwzida0e0b0xyg3, 'right')-1 #makes sure fvp starts on the same date as sim period, side=right so that if the date is already the start of a period it remains in that period.
    fvp_2_start_oa1e1b1nwzida0e0b0xyg3 = offs_date_start_p[idx_oa1e1b1nwzida0e0b0xyg]

    ##create shape which has max size of each fvp array. Exclude the first dimension because that can be different sizes because only the other dimensions need to be the same for stacking
    shape = np.maximum.reduce([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[1:], fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[1:]
                               ,fvp_0_start_oa1e1b1nwzida0e0b0xyg3.shape[1:], fvp_1_start_oa1e1b1nwzida0e0b0xyg3.shape[1:], fvp_2_start_oa1e1b1nwzida0e0b0xyg3.shape[1:]]) #create shape which has the max size, this is used for o array
    ##broadcast the start arrays so that they are all the same size (except axis 0 can be different size)
    fvp_b0_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b1_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_b2_start_ba1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,(fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_0_start_oa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_0_start_oa1e1b1nwzida0e0b0xyg3,(fvp_0_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_1_start_oa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_1_start_oa1e1b1nwzida0e0b0xyg3,(fvp_1_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))
    fvp_2_start_oa1e1b1nwzida0e0b0xyg3 = np.broadcast_to(fvp_2_start_oa1e1b1nwzida0e0b0xyg3,(fvp_2_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape))

#    fvp_b0_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_b0_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),0)
#    fvp_b1_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_b1_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),1)
#    fvp_b2_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_b2_start_ba1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),2)
#    fvp_0_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_0_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),0)
#    fvp_1_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_1_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),1)
#    fvp_2_type_va1e1b1nwzida0e0b0xyg3 = np.full((fvp_2_start_oa1e1b1nwzida0e0b0xyg3.shape[0],)+tuple(shape),2)

    ##stack - extra fvp only included if shearing occurs more than 3 periods after weaning)
    mask = (date_shear_sa1e1b1nwzida0e0b0xyg3[0:1] - date_weaned_ida0e0b0xyg3) < ((step.astype('timedelta64[D]')+1)*3)  #true if not enough gap between weaning and shearing for extra dvps.
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(([fvp_b0_start_ba1e1b1nwzida0e0b0xyg3, fvp_b1_start_ba1e1b1nwzida0e0b0xyg3, fvp_b2_start_ba1e1b1nwzida0e0b0xyg3,
                                                        fvp_0_start_oa1e1b1nwzida0e0b0xyg3,fvp_1_start_oa1e1b1nwzida0e0b0xyg3,fvp_2_start_oa1e1b1nwzida0e0b0xyg3]),axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.concatenate(([fvp_b0_type_va1e1b1nwzida0e0b0xyg3, fvp_b1_type_va1e1b1nwzida0e0b0xyg3, fvp_b2_type_va1e1b1nwzida0e0b0xyg3,
#                                                        fvp_0_type_va1e1b1nwzida0e0b0xyg3,fvp_1_type_va1e1b1nwzida0e0b0xyg3,fvp_2_type_va1e1b1nwzida0e0b0xyg3]),axis=0)
    ###if shearing is less than 3 sim periods after weaning then set the break fvp dates to the first date of the sim (so they arent used)
    mask, fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.broadcast_arrays(mask, fvp_start_fa1e1b1nwzida0e0b0xyg3)
    mask = mask.copy()
    mask[3:, ...] = False #only a mask for the 3 extra dvps at the beginning (all the others will not be altered hence set mask to false)
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = fvp_start_fa1e1b1nwzida0e0b0xyg3.copy() #have to create copy because can't assign to broadcasted array or something
    fvp_start_fa1e1b1nwzida0e0b0xyg3[mask] = offs_date_start_p[0]
#    fvp_type_fa1e1b1nwzida0e0b0xyg3[mask] = 0
    ##sort into date order
    ind=np.argsort(fvp_start_fa1e1b1nwzida0e0b0xyg3, axis=0)
    fvp_start_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_start_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)
#    fvp_type_fa1e1b1nwzida0e0b0xyg3 = np.take_along_axis(fvp_type_fa1e1b1nwzida0e0b0xyg3, ind, axis=0)

    ##feed variation period
    a_fvp_pa1e1b1nwzida0e0b0xyg1 = np.apply_along_axis(sfun.f_next_prev_association, 0, fvp_date_start_fa1e1b1nwzida0e0b0xyg1, date_end_p, 1,'right')
    a_fvp_pa1e1b1nwzida0e0b0xyg3 = np.apply_along_axis(sfun.f_next_prev_association, 0, fvp_start_fa1e1b1nwzida0e0b0xyg3, offs_date_end_p, 1,'right')

    return date_start_p, a_fvp_pa1e1b1nwzida0e0b0xyg1, a_fvp_pa1e1b1nwzida0e0b0xyg3