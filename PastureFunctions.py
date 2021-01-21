##import core modules
import numpy as np

##import AFO modules
import PropertyInputs as pinp
import Periods as per

na = np.newaxis

#todo foo needs to go through the convert function like sheep does
### define a function that loops through feed periods to generate the foo profile for a specified germination and consumption
def calc_foo_profile(germination_flt, dry_decay_ft, length_of_periods_f,
                     i_fxg_foo_oflt, c_fxg_a_oflt, c_fxg_b_oflt, i_grn_senesce_eos_ft,
                     grn_senesce_startfoo_ft, grn_senesce_pgrcons_ft):
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
    n_pasture_types = np.count_nonzero(pinp.general['pas_inc'])
    ### reshape the inputs passed and set some initial variables that are required
    grn_foo_start_flt   = np.zeros_like(germination_flt, dtype = 'float64')
    grn_foo_end_flt     = np.zeros_like(germination_flt, dtype = 'float64')
    dry_foo_start_flt   = np.zeros_like(germination_flt, dtype = 'float64')
    dry_foo_end_flt     = np.zeros_like(germination_flt, dtype = 'float64')
    pgr_daily_l         = np.zeros(n_lmu,dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array

    grn_foo_end_flt[-1,:,:] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    dry_foo_end_flt[-1,:,:] = 0 # ensure foo_end[-1] is 0 because it is used in calculation of foo_start[0].
    ### loop through the pasture types
    for t in range(n_pasture_types):   #^ is this loop required?
        ## loop through the feed periods and calculate the foo at the start of each period
        for f in range(n_feed_periods):
            grn_foo_start_flt[f,:,t]      = germination_flt[f,:,t] + grn_foo_end_flt[f-1,:,t]
            dry_foo_start_flt[f,:,t]      =                          dry_foo_end_flt[f-1,:,t]
            ## alternative approach (a1)
            ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
            for l in [*range(n_lmu)]: #loop through lmu
                idx = np.searchsorted(i_fxg_foo_oflt[:,f,l,t], grn_foo_start_flt[f,l,t], side='left')   # find where foo_starts fits into the input data
                pgr_daily_l[l] = (c_fxg_a_oflt[idx,f,l,t]
                                  + c_fxg_b_oflt[idx,f,l,t]
                                  * grn_foo_start_flt[f,l,t])
            grn_foo_end_flt[f,:,t] = (grn_foo_start_flt[f,:,t]
                                      * (1 - grn_senesce_startfoo_ft[f,t])
                                      + pgr_daily_l * length_of_periods_f[f]
                                      * (1 -  grn_senesce_pgrcons_ft[f,t])) \
                                    * (1 - i_grn_senesce_eos_ft[f,t])
            senescence_l = grn_foo_start_flt[f,:,t]  \
                          +    pgr_daily_l * length_of_periods_f[f]  \
                          -  grn_foo_end_flt[f,:,t]
            dry_foo_end_flt[f,:,t] = dry_foo_start_flt[f,:,t] \
                                    * (1 - dry_decay_ft[f,t]) \
                                    + senescence_l
    return grn_foo_start_flt, dry_foo_start_flt

def update_reseeding_foo(foo_grn_reseeding_flrt, foo_dry_reseeding_flrt,
                         resown_rt, period_t, proportion_t,
                         foo_arable, foo_na, propn_grn=1): #, dmd_dry=0):
    ''' Update p_foo parameters with values for destocking & subsequent grazing (reseeding)

    period_t     - an array [type] : the first period affected by the destocking or subsequent grazing.
    proportion_t - an array [type] : the proportion of the period that has occurred prior to the destocking or subsequent grazing.
    foo_arable   - an array either [lmu] or [lmu,type] : foo on arable area.
    foo_na       - an array either [lmu] or [lmu,type] : foo on non arable area to be spread between the period and the subsequent period.
    propn_grn    - a scalar or an array [type] : proportion of the total feed available for grazing that is green.
    # dmd_dry      - a scalar or an array [lmu,type] : dmd of dry feed (if any).

    the adjustments are spread between periods to allow for the pasture growth that can occur from the green feed
    and the amount of grazing available if the feed is dry
    '''
    ##base inputs
    n_feed_periods = len(per.f_feed_periods()) - 1
    n_pasture_types = np.count_nonzero(pinp.general['pas_inc'])
    n_lmu = len(pinp.general['lmu_area'])
    arable_l = np.array(pinp.crop['arable']).reshape(-1)
    lt = (n_lmu,n_pasture_types)
    ##create arrays
    foo_arable_lt      = np.zeros(lt, dtype = 'float64')             # create the array foo_arable_lt with the required shape - needed because different sized arrays are passed in
    foo_arable_lt[...] = foo_arable                                  # broadcast foo_arable into foo_arable_lt (to handle foo_arable not having an lmu axis)
    foo_na_lt          = np.zeros(lt, dtype = 'float64')             # create the array foo_na_l with the required shape
    foo_na_lt[...]     = foo_na                                      # broadcast foo_na into foo_na_l (to handle foo_arable not having an lmu axis)
    propn_grn_t        = np.ones(n_pasture_types, dtype = 'float64') # create the array propn_grn_t with the required shape
    propn_grn_t[:]     = propn_grn                                   # broadcast propn_grn into propn_grn_t (to handle propn_grn not having an pasture type axis)

    ### the arable foo allocated to the rotation phases
    foo_arable_lrt = foo_arable_lt[:, na,:]  \
        * arable_l.reshape(-1,1,1)  \
        * resown_rt
    foo_arable_lrt[np.isnan(foo_arable_lrt)] = 0

    foo_na_lr = np.sum(foo_na_lt[:, na,:]
                       * (1-arable_l.reshape(-1,1,1))
                       * resown_rt, axis = -1)
    foo_na_lr[np.isnan(foo_na_lr)] = 0
    foo_change_lrt         = foo_arable_lrt
    foo_change_lrt[...,0] += foo_na_lr  #assuming all non-arable is pasture 0 (annuals)

    ### ^This loop might be able to be removed
    ### # 1. p_foo_grn_reseeding_flrt[period_t,:,:,*range(n_pasture_types)] += might work (see AdvanceIndexing.py & AdvIndex2.py)
    ### #  . p_foo_grn_reseeding_flrt[period_t,:,:,t_list]
    ### # 2. removed if on propn_grn_t and doing all the dry calculations even if prop_grn_t = 1. (makes the code look much neater)
    for t in range(n_pasture_types):
        proportion  = proportion_t[t]
        propn_grn   =  propn_grn_t[t]
        foo_change  = foo_change_lrt[...,t]
        period      =     period_t[t]
        next_period = (period+1) % n_feed_periods

        foo_grn_reseeding_flrt[period,:,:,t]      += foo_change *    proportion  * propn_grn      # add the amount of green for the first period
        foo_grn_reseeding_flrt[next_period,:,:,t] += foo_change * (1-proportion) * propn_grn  # add the remainder to the next period (wrapped if past the 10th period)
        foo_dry_reseeding_flrt[period,:,:,t]      += foo_change *    proportion  * (1-propn_grn) * 0.5  # assume 50% in high & 50% into low pool. for the first period
        foo_dry_reseeding_flrt[next_period,:,:,t] += foo_change * (1-proportion) * (1-propn_grn) * 0.5  # add the remainder to the next period (wrapped if past the 10th period)

    return foo_grn_reseeding_flrt, foo_dry_reseeding_flrt