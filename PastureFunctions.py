##import core modules
import numpy as np

##import AFO modules
import PropertyInputs as pinp
import Periods as per

na = np.newaxis

#todo foo needs to go through the convert function like sheep does
### define a function that loops through feed periods to generate the foo profile for a specified germination and consumption
def calc_foo_profile(germination_flzt, dry_decay_fzt, length_of_periods_fz,
                     i_fxg_foo_oflzt, c_fxg_a_oflzt, c_fxg_b_oflzt, i_grn_senesce_eos_fzt,
                     grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt):
    '''
    Calculate the FOO level at the start of each feed period from the germination & sam on PGR provided

    Parameters
    ----------
    germination_flt     - An array[feed_period,lmu,type] : kg of green feed germinating in the period.
    dry_decay_ft        - An array[feed_period,type]     : decay rate of dry feed
    length_of_periods_f - An array[feed_period]          : days in each period
    Returns
    -------
    An array[feed_period,lmu,type]: foo at the start of the period.
    '''
    n_feed_periods = len(per.f_feed_periods()) - 1
    n_lmu = len(pinp.general['lmu_area'])
    n_pasture_types = germination_flzt.shape[-1]
    n_season = length_of_periods_fz.shape[-1]
    flzt = (n_feed_periods, n_lmu, n_season, n_pasture_types)
    ### reshape the inputs passed and set some initial variables that are required
    grn_foo_start_flzt   = np.zeros(flzt, dtype = 'float64')
    grn_foo_end_flzt     = np.zeros(flzt, dtype = 'float64')
    dry_foo_start_flzt   = np.zeros(flzt, dtype = 'float64')
    dry_foo_end_flzt     = np.zeros(flzt, dtype = 'float64')
    pgr_daily_l          = np.zeros(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array

    # grn_foo_end_flzt[-1,...] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    # dry_foo_end_flzt[-1,...] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    ### loop through the pasture types
    for t in range(n_pasture_types):   #^ is this loop required?
        ## loop through the feed periods and calculate the foo at the start of each period
        for f in range(n_feed_periods):
            grn_foo_start_flzt[f,:,:,t] = germination_flzt[f,:,:,t] + grn_foo_end_flzt[f-1,:,:,t]
            dry_foo_start_flzt[f,:,:,t] = dry_foo_end_flzt[f-1,:,:,t]
            ##loop season
            for z in range(n_season):
                ## alternative approach (a1)
                ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
                for l in [*range(n_lmu)]: #loop through lmu
                    idx = np.searchsorted(i_fxg_foo_oflzt[:,f,l,z,t], grn_foo_start_flzt[f,l,z,t], side='left')   # find where foo_start fits into the input data
                    pgr_daily_l[l] = (       c_fxg_a_oflzt[idx,f,l,z,t]
                                      +      c_fxg_b_oflzt[idx,f,l,z,t]
                                      * grn_foo_start_flzt[f,l,z,t])
                grn_foo_end_flzt[f,:,z,t] = (              grn_foo_start_flzt[f,:,z,t]
                                             * (1 - grn_senesce_startfoo_fzt[f,z,t])
                                             +                 pgr_daily_l
                                             *         length_of_periods_fz[f,z]
                                             * (1 -  grn_senesce_pgrcons_fzt[f,z,t])) \
                                            * (1 -     i_grn_senesce_eos_fzt[f,z,t])
                senescence_l = grn_foo_start_flzt[f,:,z,t]  \
                              +    pgr_daily_l * length_of_periods_fz[f,z]  \
                              -  grn_foo_end_flzt[f,:,z,t]
                dry_foo_end_flzt[f,:,z,t] = dry_foo_start_flzt[f,:,z,t] \
                                        * (1 - dry_decay_fzt[f,z,t]) \
                                        + senescence_l
    return grn_foo_start_flzt, dry_foo_start_flzt

def update_reseeding_foo(foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt,
                         resown_rt, period_zt, proportion_zt,
                         foo_arable_zt, foo_na_zt, propn_grn=1): #, dmd_dry=0):
    ''' Adjust p_foo parameters due to changes associated with reseeding: destocking pastures prior to spraying and restocking after reseeding

    period_t     - an array [type] : the first period affected by the destocking or subsequent restocking.
    proportion_t - an array [type] : the proportion of the FOO adjustment that occurs in the first period (the balance of the adjustment occurs in the subsequent period).
    foo_arable   - an array either [lmu] or [lmu,type] : change of FOO on arable area. A negative value if it is a FOO reduction due to destocking or positive if a FOO increase when restocked.
    foo_na       - an array either [lmu] or [lmu,type] : change of FOO on the non arable area.
    propn_grn    - a scalar or an array [type] : proportion of the change in feed available for grazing that is green.
    # dmd_dry      - a scalar or an array [lmu,type] : dmd of dry feed (if any).

    the FOO adjustments is spread between periods to allow for the pasture growth that can occur from the green feed
    and the amount of grazing available if the feed is dry
    If there is an adjustment to the dry feed then it is spread equally between the high & the low quality pools.
    '''
    ##base inputs
    n_feed_periods = len(per.f_feed_periods()) - 1
    len_t = np.count_nonzero(pinp.general['pas_inc'])
    n_lmu = len(pinp.general['lmu_area'])
    len_z = period_zt.shape[0]
    len_r = resown_rt.shape[0]
    lzt = (n_lmu,len_z,len_t)
    arable_l = np.array(pinp.crop['arable']).reshape(-1)
    ##create arrays
    foo_arable_lzt      = np.zeros(lzt, dtype = 'float64')             # create the array foo_arable_lt with the required shape - needed because different sized arrays are passed in
    foo_arable_lzt[...] = foo_arable_zt                                # broadcast foo_arable into foo_arable_lt (to handle foo_arable not having an lmu axis)
    foo_na_lzt          = np.zeros(lzt, dtype = 'float64')             # create the array foo_na_l with the required shape
    foo_na_lzt[...]     = foo_na_zt                                    # broadcast foo_na into foo_na_l (to handle foo_arable not having an lmu axis)
#    propn_grn_t         = np.ones(len_t, dtype = 'float64')            # create the array propn_grn_t with the required shape
#    propn_grn_t[:]      = propn_grn                                    # broadcast propn_grn into propn_grn_t (to handle propn_grn not having a pasture type axis)

    ### the arable foo allocated to the rotation phases
    foo_arable_lrzt = foo_arable_lzt[:,na,...]  \
                        * arable_l[:,na,na,na] \
                        * resown_rt[:,na,:]
    foo_arable_lrzt[np.isnan(foo_arable_lrzt)] = 0

    foo_na_lrz = np.sum(foo_na_lzt[:, na,:,0:1]
                   * (1-arable_l[:,na,na,na])
                   *    resown_rt[:, na,0:1], axis = -1)
    foo_na_lrz[np.isnan(foo_na_lrz)] = 0
    foo_change_lrzt         = foo_arable_lrzt
    foo_change_lrzt[...,0] += foo_na_lrz  #because all non-arable is pasture 0 (annuals)

    ##allocate into reseeding period - using advanced indexing
    period_lrzt = period_zt[na,na,...]
    next_period_lrzt = (period_lrzt+1) % n_feed_periods
    l_idx=np.arange(n_lmu)[:,na,na,na]
    r_idx=np.arange(len_r)[:,na,na]
    z_idx=np.arange(len_z)[:,na]
    t_idx=np.arange(len_t)
    foo_grn_reseeding_flrzt[period_lrzt,l_idx,r_idx,z_idx,t_idx]      = foo_change_lrzt *    proportion_zt  * propn_grn      # add the amount of green for the first period
    foo_grn_reseeding_flrzt[next_period_lrzt,l_idx,r_idx,z_idx,t_idx] = foo_change_lrzt * (1-proportion_zt) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
    foo_dry_reseeding_flrzt[period_lrzt,l_idx,r_idx,z_idx,t_idx]      = foo_change_lrzt *    proportion_zt  * (1-propn_grn) * 0.5  # assume 50% in high & 50% into low pool. for the first period
    foo_dry_reseeding_flrzt[next_period_lrzt,l_idx,r_idx,z_idx,t_idx] = foo_change_lrzt * (1-proportion_zt) * (1-propn_grn) * 0.5  # add the remainder to the next period (wrapped if past the 10th period)

    return foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt