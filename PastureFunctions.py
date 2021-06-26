##import core modules
import numpy as np

##import AFO modules
import PropertyInputs as pinp
import UniversalInputs as uinp
import Periods as per
import FeedsupplyFunctions as fsfun
import Functions as fun

na = np.newaxis

def f_germination(i_germination_std_zt, i_germ_scalar_lzt, germ_scalar_rt, i_germ_scalar_fzt
                  , pasture_rt, arable_l,  resown_rt, pastures, phase_germresow_df, i_phase_germ_dict):
    '''
    create an array called p_germination_flrt being the parameters to be passed to pyomo
    :param i_germination_std_zt:
    :param i_germ_scalar_lzt:
    :param germ_scalar_rt:
    :param i_germ_scalar_fzt:
    :param pasture_rt:
    :param arable_l:
    :param resown_rt:
    :param pastures:
    :param phase_germresow_df:
    :param i_phase_germ_dict:
    :return germination_flrzt:
    '''
    #todo currently all germination occurs in period 0, however, other code handles germination in other periods if the inputs & this code are changed

    for t, pasture in enumerate(pastures):
        phase_germresow_df['germ_scalar']=0 #set default to 0
        phase_germresow_df['resown']=False #set default to false
        ###loop through each combo of landuses and pastures (i_phase_germ), then check which rotations fall into each germ/resowing category. Then populate the rot phase df with the necessary germination and resowing param.
        for ix_row in i_phase_germ_dict[pasture].index:
            ix_bool = pd.Series(data=True,index=range(len(phase_germresow_df)))
            for ix_col in range(i_phase_germ_dict[pasture].shape[1]-2):    #-2 because two of the cols are germ and resowing
                c_set = sinp.landuse[i_phase_germ_dict[pasture].iloc[ix_row,ix_col]]
                ix_bool &= phase_germresow_df.loc[:,ix_col].reset_index(drop=True).isin(c_set) #had to drop index so that it would work (just said false when the index was different between series)
            ### maps the relevant germ scalar and resown bool to the rotation phase
            phase_germresow_df.loc[list(ix_bool),'germ_scalar'] = i_phase_germ_dict[pasture].iloc[ix_row, -2]  #have to make bool into a list for some reason it doesn't like a series
            phase_germresow_df.loc[list(ix_bool),'resown'] = i_phase_germ_dict[pasture].iloc[ix_row, -1]
        ### Convert germ and resow into a numpy - each pasture goes in a different slice
        germ_scalar_rt[:,t] = phase_germresow_df['germ_scalar'].to_numpy()    # extract the germ_scalar from the dataframe
        resown_rt[:,t] = phase_germresow_df['resown'].to_numpy()              # extract the resown boolean from the dataframe

    ## germination on the arable area of pasture paddocks based on std germ, rotation scalar, lmu scalar and distribution across periods
    arable_germination_flrzt = i_germination_std_zt                 \
                              *   i_germ_scalar_lzt[:, na, ...]     \
                              *      germ_scalar_rt[:, na, :]       \
                              *   i_germ_scalar_fzt[:, na, na, ...]
    arable_germination_flrzt[np.isnan(arable_germination_flrzt)]  = 0.0

    ## germination on the non arable area is the maximum germination across phases (continuous pasture) for the first pasture type (annuals)
    ### todo a potential error here when if the allocation of germination across periods varies by rotation phase (because taking max of each period)
    max_germination_flz = np.max(arable_germination_flrzt[..., 0], axis=2)  #use germination_flrzt because it includes any sensitivity that is carried out

    ## germination on the non arable area of pasture paddocks. Grows pasture type 0 that can be grazed during the growing season
    na_germination_flrz = max_germination_flz[..., na, :] * np.any(pasture_rt[:, na, :], axis = -1)
    ## set germination in first period to germination on arable area
    germination_flrzt = arable_germination_flrzt * arable_l[:, na, na, na]
    ## add germination on the non-arable area to the first pasture type
    germination_flrzt[..., 0] += na_germination_flrz * (1 - arable_l[:,na,na])
    return germination_flrzt, max_germination_flz


def f_reseeding(i_destock_date_zt, i_restock_date_zt, i_destock_foo_zt, grn_restock_foo_flzt, dry_restock_foo_flzt
                , resown_rt, feed_period_dates_fz
                , foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt, foo_na_destock_fzt
                , i_restock_fooscalar_lt, i_restock_foo_arable_t, dry_decay_period_fzt
                , i_fxg_foo_oflzt, c_fxg_a_oflzt, c_fxg_b_oflzt, i_grn_senesce_eos_fzt
                , grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt, length_fz, n_feed_periods
                , max_germination_flz, n_pasture_types):
    ##reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    ##Results are stored in p_...._reseeding
    #todo test the calculation of FOO on the resown area when the full set of rotation phases is included
    ## the green feed to remove from matrix when pasture is destocked.
    foo_arable_destock_zt = i_destock_foo_zt
    foo_na_destock_zt =  i_destock_foo_zt
    ## the periods from which to remove based on date destocked.
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na]  # which feed period does destocking occur & the proportion that destocking occurs during the period.
                                                          , i_destock_date_zt)
    ## the change (reduction) in green and dry FOO on the arable and non-arable areas when pasture is destocked for spraying prior to reseeding
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = update_reseeding_foo(foo_grn_reseeding_flrzt
                                                                            , foo_dry_reseeding_flrzt
                                                                            ,              resown_rt
                                                                            ,               period_zt
                                                                            ,       1 - proportion_zt
                                                                            ,  -foo_arable_destock_zt
                                                                            ,      -foo_na_destock_zt) # Assumes that all feed lost is green

    ##FOO on the arable area of each LMU when reseeded pasture is restocked (this is calculated from input values)
    foo_arable_restock_lt =  i_restock_fooscalar_lt * i_restock_foo_arable_t

    ## calc foo on non arable area when the area is restocked after reseeding
    ### FOO on non-arable areas at restocking equals foo at destocking plus any germination occurring in the destocked period plus growth from destocking to grazing
    #### FOO at destocking is an input, allocate the input to the destocking feed period
    foo_na_destock_fzt[period_zt, z_idx[:,na], t_idx] = foo_na_destock_zt

    #### the period from destocking to restocking (for germination and growth)
    destock_duration_zt = i_restock_date_zt - i_destock_date_zt
    shape_fzt = feed_period_dates_fz.shape + (i_destock_date_zt.shape[-1],)
    periods_destocked_fzt = fun.range_allocation_np(feed_period_dates_fz[...,na]
                                                    ,   i_destock_date_zt
                                                    , destock_duration_zt
                                                    ,     shape=shape_fzt)[0:n_feed_periods,...]
    days_each_period_fzt = periods_destocked_fzt * length_fz[..., na]
    #### period when restocking occurs and the proportion through the period that it occurs
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na], i_restock_date_zt)

    ### germination during destocked period (this is the germination of pasture type 1 but it includes a t axis because the destocked period can vary with pasture type)
    germination_destocked_flzt = max_germination_flz[..., na] * periods_destocked_fzt[:, na, ...]

    ### Calculate the FOO profile on the non arable area from destocking through to restocking
    #### need to loop through t because FOO at destocking and reseeding date can change
    for t in range(n_pasture_types):
        ### green FOO to start the profile is FOO at destocking plus germination that occurs during the destocking period
        #### assumes FOO at destocking of pasture type 0 on the non arable area is equivalent to the pasture itself.
        grn_foo_na_initial_flzt = foo_na_destock_fzt[:, na, :, t:t + 1] + germination_destocked_flzt[..., t: t+1]
        ##FOO at the end of the destocked period is calculated from the FOO profile from destocking to restocking
        grn_restock, dry_restock = f1_calc_foo_profile(grn_foo_na_initial_flzt  # axes are aligned in the function
                                                       , dry_decay_period_fzt[..., 0:1]
                                                       , days_each_period_fzt[...,t]
                                                       , i_fxg_foo_oflzt[..., 0:1]
                                                       , c_fxg_a_oflzt[..., 0:1]
                                                       , c_fxg_b_oflzt[..., 0:1]
                                                       , i_grn_senesce_eos_fzt[..., 0:1]
                                                       , grn_senesce_startfoo_fzt[..., 0:1]
                                                       , grn_senesce_pgrcons_fzt[..., 0:1])
        #### assign the growth to a variable to store all the pasture types
        grn_restock_foo_flzt[...,t:t+1] = grn_restock
        dry_restock_foo_flzt[...,t:t+1] = dry_restock
    ### combine dry and grn foo because the proportion of green at restocking is an input
    #### foo is calculated at the start of period, +1 to get end period FOO.
    foo_na_restock_lzt = grn_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx]   \
                        + dry_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx] #foo is calc at the start of period, +1 to get end period foo.

    ## increment the change in green and dry foo on the arable and non-arable areas when pasture is restocked after reseeding
    ### Note: the function call includes += for the green and dry foo variables

    ## combine the non-arable and arable foo to get the resulting foo in the green and dry pools when paddocks are restocked. Spread between periods based on date grazed. (arable proportion accounted for in function)
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = pfun.update_reseeding_foo(foo_grn_reseeding_flrzt  #axes aligned in function
                                                                               , foo_dry_reseeding_flrzt
                                                                               ,                 resown_rt
                                                                               ,               period_zt
                                                                               ,       1 - proportion_zt
                                                                               ,   foo_arable_restock_lt[:,na,:]
                                                                               ,       foo_na_restock_lzt
                                                                               , propn_grn=i_restock_grn_propn_t)

    ## split the change in dry FOO between the high & low quality FOO pools
    ### a 50% split assumes the dry feed removed at destocking and added at restocking is average quality.
    foo_dry_reseeding_dflrzt = np.stack([foo_dry_reseeding_flrzt * 0.5, foo_dry_reseeding_flrzt * 0.5], axis=0)
    return foo_grn_reseeding_flrzt, foo_dry_reseeding_dflrzt, periods_destocked_fzt


def f_pas_sow(i_reseeding_date_start_zt, i_reseeding_date_end_zt, resown_rt, arable_l, phases_rotn_df):
    ### sow param determination
    ### determine the labour periods pas seeding occurs
    i_seeding_length_zt = i_reseeding_date_end_zt - i_reseeding_date_start_zt
    period_dates_p5z = per.p_dates_df().values
    shape_p5zt = period_dates_p5z.shape + (i_seeding_length_zt.shape[-1],)
    reseeding_machperiod_p5zt  = fun.range_allocation_np(        period_dates_p5z[...,na]
                                                        ,i_reseeding_date_start_zt
                                                        ,      i_seeding_length_zt
                                                        , True,   shape=shape_p5zt)
    ### combine with rotation reseeding requirement
    pas_sown_lrt = resown_rt * arable_l[:, na, na]
    pas_sow_plrzt = pas_sown_lrt[...,na,:] * reseeding_machperiod_p5zt[:, na, na,...]
    pas_sow_plrz = np.sum(pas_sow_plrzt, axis=-1) #sum the t axis. the different pastures are tracked by the rotation.
    pas_sow_plrkz = pas_sow_plrz[..., na,:] * (keys_k[:, na]==phases_rotn_df.iloc[:,-1].values[:, na,na]) #add k (landuse axis) this is required for sow param
    return pas_sow_plrkz


def f_green_area(resown_rt, pasture_rt, periods_destocked_fzt, arable_l):
    ## area of green pasture being grazed and growing
    ### calculate the area (for all the phases) that is growing pasture for each feed period. The area can be 0 for a pasture phase if it has been destocked for reseeding.
    arable_phase_area_flrzt = (1 - (resown_rt[:,na,:] * periods_destocked_fzt[:, na, na, ...]))  \
                             * arable_l[:, na, na, na] * pasture_rt[:, na, :]
    phase_area_flrzt = arable_phase_area_flrzt
    ###pasture on the non-arable area is annual pasture only (first pasture type 0:1)
    na_phase_area_flrzt = np.sum((1 - (resown_rt[:,na,:] * periods_destocked_fzt[:, na, na, ...]))
                                        * (1 - arable_l[:, na, na, na]) * pasture_rt[:, na, :]
                                        , axis = -1, keepdims=True)
    phase_area_flrzt[..., 0:1] = phase_area_flrzt[..., 0:1] + na_phase_area_flrzt
    return phase_area_flrzt

def f_grn_pasture():
    return

def f_dry_pasture():
    return

#f_poc here


def f1_calc_foo_profile(germination_flzt, dry_decay_fzt, length_of_periods_fz
                        , i_fxg_foo_oflzt, c_fxg_a_oflzt, c_fxg_b_oflzt, i_grn_senesce_eos_fzt
                        , grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt):
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
    n_lmu = np.count_nonzero(pinp.general['lmu_area'])
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
    ##lmu mask
    lmu_mask_l = pinp.general['lmu_area'].squeeze().values > 0

    ##base inputs
    n_feed_periods = len(per.f_feed_periods()) - 1
    len_t = np.count_nonzero(pinp.general['pas_inc'])
    n_lmu = np.count_nonzero(pinp.general['lmu_area'])
    len_z = period_zt.shape[0]
    len_r = resown_rt.shape[0]
    lzt = (n_lmu,len_z,len_t)
    arable_l = pinp.crop['arable'].squeeze().values[lmu_mask_l]
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

def f_poc(cu3, cu4, i_poc_intake_daily_flt, i_poc_dmd_ft, i_poc_foo_ft, i_legume_zt, i_pasture_stage_p6z, ev_is_not_confinement_v):
    '''
    Calculate energy, volume and consumption parameters for pasture consumed on crop paddocks before seeding.

    The amount of pasture consumption that can occur on crop paddocks per hectare per day before seeding
    - adjusted for lmu and feed period
    The energy provided by the consumption of 1 tonne of pasture on crop paddocks.
    - adjusted for feed period
    The livestock intake volume required to consume 1 tonne of pasture on crop paddocks.
    - adjusted for feed period

    Pasture can be grazed on crop paddocks if seeding occurs after pasture germination. Grazing can occur between
    pasture germination and destocking. Destocking date occurs a certain number of days before seeding, this is to
    allow the pasture leaf area to grow so that the knock down spray is effective. The amount of pasture that can be
    consumed per day is a user defined input that can vary by LMU. The grazing days provide by each seeding activity
    are calculated in mach.py and depend on the time between the break of season and destocking prior to seeding.

    :param cu3: params used to convert foo for rel availability.
    :param cu4: params used to convert height for rel availability.
    :param i_poc_intake_daily_flt: maximum daily intake available from 1ha of pasture on crop paddocks
    :param i_poc_dmd_ft: average digestibility of pasture on crop paddocks.
    :param i_poc_foo_ft: average foo of pasture on crop paddocks.
    :param i_legume_zt: legume content of pasture.
    :param i_pasture_stage_p6z: maturity of the pasture (establishment or vegetative as defined by CSIRO)
    :param ev_is_not_confinement_v: boolean array stating which fev pools are not confinement feeding pools.
    :return:
        - poc_con_fl - tonnes of dry matter available per hectare per day on crop paddocks before seeding.
        - poc_md_vf - md per tonne of poc.
        - poc_vol_fz - volume required to consume 1 tonne of poc.
    '''
    ### poc is assumed to be annual hence the 0 slice in the last axis
    ## con
    poc_con_fl = i_poc_intake_daily_flt[..., 0] / 1000 #divide 1000 to convert to tonnes of foo per ha
    ## md per tonne
    poc_md_f = fsfun.dmd_to_md(i_poc_dmd_ft[..., 0]) * 1000 #times 1000 to convert to mj per tonne
    poc_md_vf = poc_md_f * ev_is_not_confinement_v[:,na] #me from pasture is 0 in the confinement pool

    ## vol
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        poc_ri_qual_fz = fsfun.f_rq_cs(i_poc_dmd_ft[..., na, 0], i_legume_zt[..., 0])

    ### adjust foo and calc hf
    i_poc_foo_fz, hf = fsfun.f_foo_convert(cu3, cu4, i_poc_foo_ft[:,na,0], i_pasture_stage_p6z, i_legume_zt[...,0], z_pos=-1)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1 - need to hook up mu function
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        poc_ri_quan_fz = fsfun.f_ra_cs(i_poc_foo_fz, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        poc_ri_quan_fz = fsfun.f_ra_mu(i_poc_foo_fz, hf)

    poc_ri_fz = fsfun.f_rel_intake(poc_ri_quan_fz, poc_ri_qual_fz, i_legume_zt[..., 0])
    poc_vol_fz = fun.f_divide(1000, poc_ri_fz)  # 1000 to convert to vol per tonne


    return poc_con_fl, poc_md_vf, poc_vol_fz