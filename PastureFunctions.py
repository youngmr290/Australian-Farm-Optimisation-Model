##import core modules
import numpy as np
import pandas as pd

##import AFO modules
import PropertyInputs as pinp
import UniversalInputs as uinp
import StructuralInputs as sinp
import Periods as per
import FeedsupplyFunctions as fsfun
import Functions as fun
import Sensitivity as sen


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

def f_reseeding(i_destock_date_zt, i_restock_date_zt, i_destock_foo_zt, i_restock_grn_propn_t, resown_rt
                , feed_period_dates_fz, foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt, foo_na_destock_fzt
                , i_restock_fooscalar_lt, i_restock_foo_arable_t, dry_decay_period_fzt
                , i_fxg_foo_oflzt, c_fxg_a_oflzt, c_fxg_b_oflzt, i_grn_senesce_eos_fzt
                , grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt, length_fz, n_feed_periods
                , max_germination_flz, t_idx, z_idx, l_idx):
    ##reseeding: generates the green & dry FOO that is lost and gained from reseeding pasture. It is stored in a numpy array (phase, lmu, feed period)
    ##Results are stored in p_...._reseeding
    #todo test the calculation of FOO on the resown area when the full set of rotation phases is included

    ## the green feed to remove from matrix when pasture is destocked.
    foo_arable_destock_zt = i_destock_foo_zt
    foo_na_destock_zt =  i_destock_foo_zt
    ## the periods from which to remove foo based on date destocked. Returns feed period destocking occurs & the proportion of the way that destocking occurs.
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na]
                                                          , i_destock_date_zt)
    ## the change (reduction) in green and dry FOO on the arable and non-arable areas when pasture is destocked for spraying prior to reseeding
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = f1_update_reseeding_foo(
        foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt, resown_rt, period_zt, 1 - proportion_zt, -foo_arable_destock_zt
        , -foo_na_destock_zt) # Assumes that all feed lost is green

    ##FOO on the arable area of each LMU when reseeded pasture is restocked (this is calculated from input values)
    foo_arable_restock_lt =  i_restock_fooscalar_lt * i_restock_foo_arable_t

    ## calc foo on non arable area when the area is restocked after reseeding
    ### FOO on non-arable areas at restocking equals foo at destocking plus any germination occurring in the destocked period plus growth from destocking to grazing
    #### FOO at destocking is an input, allocate the input to the destocking feed period
    foo_na_destock_fzt[period_zt, z_idx[:,na], t_idx] = foo_na_destock_zt

    #### the period from destocking to restocking (for germination and growth)
    destock_duration_zt = i_restock_date_zt - i_destock_date_zt
    shape_fzt = feed_period_dates_fz.shape + (i_destock_date_zt.shape[-1],)
    periods_destocked_fzt = fun.range_allocation_np(feed_period_dates_fz[...,na], i_destock_date_zt, destock_duration_zt
                                                    , shape=shape_fzt)[0:n_feed_periods,...]
    days_each_period_fzt = periods_destocked_fzt * length_fz[..., na]
    #### period when restocking occurs and the proportion through the period that it occurs
    period_zt, proportion_zt = fun.period_proportion_np(feed_period_dates_fz[...,na], i_restock_date_zt)

    ### germination during destocked period (this is the germination of pasture type 1 but it includes a t axis because the destocked period can vary with pasture type)
    germination_destocked_flzt = max_germination_flz[..., na] * periods_destocked_fzt[:, na, ...]

    ### Calculate the FOO profile on the non arable area from destocking through to restocking
    ### green FOO to start the profile is FOO at destocking plus germination that occurs during the destocking period
    #### assumes FOO at destocking of pasture type 0 on the non arable area is equivalent to the pasture itself.
    grn_foo_na_initial_flzt = foo_na_destock_fzt[:, na, ...] + germination_destocked_flzt
    ##FOO at the end of the destocked period is calculated from the FOO profile from destocking to restocking
    grn_restock_foo_flzt, dry_restock_foo_flzt = f1_calc_foo_profile(grn_foo_na_initial_flzt  # axes are aligned in the function
                                                                     , dry_decay_period_fzt[..., 0:1]
                                                                     , days_each_period_fzt
                                                                     , i_fxg_foo_oflzt[..., 0:1]
                                                                     , c_fxg_a_oflzt[..., 0:1]
                                                                     , c_fxg_b_oflzt[..., 0:1]
                                                                     , i_grn_senesce_eos_fzt[..., 0:1]
                                                                     , grn_senesce_startfoo_fzt[..., 0:1]
                                                                     , grn_senesce_pgrcons_fzt[..., 0:1])

    ### combine dry and grn foo because the proportion of green at restocking is an input
    #### foo is calculated at the start of period, +1 to get end period FOO.
    foo_na_restock_lzt = grn_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx]   \
                        + dry_restock_foo_flzt[period_zt+1,l_idx[:,na,na], z_idx[:,na], t_idx] #foo is calc at the start of period, +1 to get end period foo.

    ## increment the change in green and dry foo on the arable and non-arable areas when pasture is restocked after reseeding
    ### Note: the function call includes += for the green and dry foo variables

    ## combine the non-arable and arable foo to get the resulting foo in the green and dry pools when paddocks are restocked. Spread between periods based on date grazed. (arable proportion accounted for in function)
    ### the change in FOO on the nonarable area occurs in pasture type 0 (annuals) because it is assumed that other pasture species have not been established.
    ### Note: the arable proportion is accounted for in function
    foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt = f1_update_reseeding_foo(foo_grn_reseeding_flrzt  #axes aligned in function
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
    ### add k (landuse axis) - this is required for sow param
    keys_k = np.asarray(list(sinp.landuse['All']))
    pas_sow_plrkz = pas_sow_plrz[..., na,:] * (keys_k[:, na]==phases_rotn_df.iloc[:,-1].values[:, na,na])
    return pas_sow_plrkz


def f1_green_area(resown_rt, pasture_rt, periods_destocked_fzt, arable_l):
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


def f_erosion(i_lmu_conservation_flt, arable_l, pasture_rt):
    ############################################################
    ## erosion limit. The minimum FOO at the end of each period#
    ############################################################
    arable_erosion_flrt = i_lmu_conservation_flt[..., na,:]  \
                                    *  arable_l[:, na, na]  \
                                    * pasture_rt
    na_erosion_flr = np.sum(i_lmu_conservation_flt[..., na,:]
                                    *         (1-arable_l[:, na, na])
                                    *           pasture_rt
                                    , axis = -1)
    erosion_flrt = arable_erosion_flrt
    erosion_flrt[..., 0] = erosion_flrt[..., 0] + na_erosion_flr #non arable area is annual pasture thus add to the annual slice
    return erosion_flrt


def f_grn_pasture(cu3, cu4, i_fxg_foo_oflzt, i_fxg_pgr_oflzt, c_pgr_gi_scalar_gft, grn_foo_start_ungrazed_flzt
                  , i_foo_graze_propn_gt, grn_senesce_startfoo_fzt, grn_senesce_pgrcons_fzt, i_grn_senesce_eos_fzt
                  , i_base_ft, i_grn_trampling_ft, i_grn_dig_flzt, i_grn_dmd_range_ft, i_pasture_stage_p6z, i_legume_zt
                  , me_threshold_vfzt, i_me_eff_gainlose_ft, mask_greenfeed_exists_fzt, length_fz, ev_is_not_confinement_v):
    '''
    Pasture growth, consumption and senescence of green feed.
    :param cu3:
    :param cu4:
    :param i_fxg_foo_oflzt:
    :param i_fxg_pgr_oflzt:
    :param c_pgr_gi_scalar_gft:
    :param grn_foo_start_ungrazed_flzt:
    :param i_foo_graze_propn_gt:
    :param grn_senesce_startfoo_fzt:
    :param grn_senesce_pgrcons_fzt:
    :param i_grn_senesce_eos_fzt:
    :param i_base_ft:
    :param i_grn_trampling_ft:
    :param i_grn_dig_flzt:
    :param i_grn_dmd_range_ft:
    :param i_pasture_stage_p6z:
    :param i_legume_zt:
    :param me_threshold_vfzt:
    :param i_me_eff_gainlose_ft:
    :param mask_greenfeed_exists_fzt:
    :param length_fz:
    :param ev_is_not_confinement_v:
    :return:
    '''
    #

    ## green initial FOO for the 'grnha' decision variables
    foo_start_grnha_oflzt = i_fxg_foo_oflzt
    #    foo_start_grnha_oflzt = np.maximum(i_fxg_foo_oflzt, i_base_ft[:, na, na, :])  # to ensure that final foo can not be below the base level
    #FOO of the high FOO slice is the maximum of ungrazed foo and foo from the medium foo level
    max_foo_flzt = np.maximum(i_fxg_foo_oflzt[1, ...], grn_foo_start_ungrazed_flzt)
    #maximum accumulated along the feed periods axis, i.e. max to date
    foo_start_grnha_oflzt[2, ...] = np.maximum.accumulate(max_foo_flzt, axis=0)
    #masks out any green foo at the end of periods in which green pasture doesn't exist.
    foo_start_grnha_oflzt = foo_start_grnha_oflzt * mask_greenfeed_exists_fzt[:, na,...]

    ## green, pasture growth for the 'grnha' decision variables
    #todo revisit the effect of gi on PGR by basing the reduction on the effect of gi on average FOO (rather than c_pgr_gi_scalar)
    pgr_grnha_goflzt = (i_fxg_pgr_oflzt * length_fz[:, na, :, na]
                        * c_pgr_gi_scalar_gft[:, na, :, na, na, :] * mask_greenfeed_exists_fzt[:, na, ...])

    ## green, final foo from initial, pgr and senescence
    ### senescence during the period is senescence of the starting FOO and of the FOO that is added/reduced by growth/grazing
    senesce_period_grnha_goflzt = (foo_start_grnha_oflzt * grn_senesce_startfoo_fzt[:, na, ...]
                                   + pgr_grnha_goflzt * grn_senesce_pgrcons_fzt[:, na, ...])
    ### foo at end of period if ungrazed
    foo_end_ungrazed_grnha_oflzt = foo_start_grnha_oflzt + pgr_grnha_goflzt[0, ...] - senesce_period_grnha_goflzt[0, ...]
    ### foo at end of period with range of grazing intensity prior to eos senescence
    foo_endprior_grnha_goflzt = (foo_end_ungrazed_grnha_oflzt
                                 - (foo_end_ungrazed_grnha_oflzt - i_base_ft[:, na, na, :])
                                 * i_foo_graze_propn_gt[:, na, na, na, na, :])
    senesce_eos_grnha_goflzt = foo_endprior_grnha_goflzt * i_grn_senesce_eos_fzt[:, na, ...]
    foo_end_grnha_goflzt = foo_endprior_grnha_goflzt - senesce_eos_grnha_goflzt
    #apply mask to remove any green foo at the end of period in periods when green pas doesnt exist.
    foo_end_grnha_goflzt = foo_end_grnha_goflzt * mask_greenfeed_exists_fzt[:, na, ...]

    ## green, removal & dmi
    ### divide by (1 - grn_senesce_pgrcons) to allows for consuming feed reducing senescence
    removal_grnha_goflzt = np.maximum(0, (foo_start_grnha_oflzt * (1 - grn_senesce_startfoo_fzt[:, na, ...])
                                          + pgr_grnha_goflzt * (1 - grn_senesce_pgrcons_fzt[:, na, :])
                                          - foo_endprior_grnha_goflzt)
                                      / (1 - grn_senesce_pgrcons_fzt[:, na, :]))
    cons_grnha_t_goflzt = removal_grnha_goflzt / (1 + i_grn_trampling_ft[:, na, na, :])

    ## green, dmd & md from input values and impact of foo & grazing intensity
    ### sward digestibility is reduced with higher FOO (based on start FOO)
    ### diet digestibility is reduced with higher FOO if grazing intensity is greater than 25%
    #### Low FOO or low grazing intensity is input
    #### High FOO with 100% grazing is reduced by half the range in digestibility.
    #### Between low and high FOO it is a linear interpolation
    dmd_sward_start_grnha_oflzt = (i_grn_dig_flzt - i_grn_dmd_range_ft[:, na, na, :] / 2
                                   * fun.f_divide(foo_start_grnha_oflzt - foo_start_grnha_oflzt[0, ...]
                                                  , foo_start_grnha_oflzt[-1, ...] - foo_start_grnha_oflzt[0, ...]))
    #### Diet digestibility includes a linear interpolation of selectivity
    #### 0.25 is grazing intensity that gives diet quality == input value.
    dmd_diet_grnha_goflzt = (i_grn_dig_flzt - i_grn_dmd_range_ft[:, na, na, :] / 2
                             * fun.f_divide(foo_start_grnha_oflzt - foo_start_grnha_oflzt[0, ...]
                                            , foo_start_grnha_oflzt[-1, ...] - foo_start_grnha_oflzt[0, ...])
                             * (i_foo_graze_propn_gt[:, na, na, na, na, :] - 0.25) / (1 - 0.25))
    #### dmd of the sward after grazing is reduced due to removal of the high quality feed from selective grazing
    dmd_sward_end_grnha_goflzt = dmd_sward_start_grnha_oflzt - ((dmd_diet_grnha_goflzt - dmd_sward_start_grnha_oflzt)
                                                                * fun.f_divide(i_foo_graze_propn_gt[:, na, na, na, na, :]
                                                                    , 1 - i_foo_graze_propn_gt[:, na, na, na, na, :]))
    grn_md_grnha_goflzt = fsfun.dmd_to_md(dmd_diet_grnha_goflzt)

    ## green, mei & volume
    ###Average FOO is calculated using FOO at the end prior to EOS senescence (which assumes all pasture senesces after grazing)
    foo_ave_grnha_goflzt = (foo_start_grnha_oflzt + foo_endprior_grnha_goflzt) / 2
    ### pasture params used to convert foo for rel availability
    pasture_stage_flzt = i_pasture_stage_p6z[:, na, :, na]
    ### adjust foo and calc hf
    foo_ave_grnha_goflzt, hf = fsfun.f_foo_convert(cu3, cu4, foo_ave_grnha_goflzt, pasture_stage_flzt, i_legume_zt,
                                                   z_pos=-2)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][5, 0] == 0:  #csiro function used
        grn_ri_availability_goflzt = fsfun.f_ra_cs(foo_ave_grnha_goflzt, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5, 0] == 1:  #Murdoch function used
        grn_ri_availability_goflzt = fsfun.f_ra_mu(foo_ave_grnha_goflzt, hf)
    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6, 0] == 0:  #csiro function used
        grn_ri_quality_goflzt = fsfun.f_rq_cs(dmd_diet_grnha_goflzt, i_legume_zt)
    grn_ri_goflzt = fsfun.f_rel_intake(grn_ri_availability_goflzt, grn_ri_quality_goflzt, i_legume_zt)

    me_cons_grnha_vgoflzt = fsfun.f_effective_mei(cons_grnha_t_goflzt
                                                  , grn_md_grnha_goflzt
                                                  , me_threshold_vfzt[:, na, na, :, na, ...]
                                                  , grn_ri_goflzt
                                                  , i_me_eff_gainlose_ft[:, na, na, :])
    #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    me_cons_grnha_vgoflzt = me_cons_grnha_vgoflzt * mask_greenfeed_exists_fzt[:, na, ...]
    #me from pasture is 0 in the confinement pool
    me_cons_grnha_vgoflzt = me_cons_grnha_vgoflzt * ev_is_not_confinement_v[:, na, na, na, na, na, na]

    # parameters for the growth/grazing activities: Total volume of feed consumed from the hectare
    volume_grnha_goflzt = cons_grnha_t_goflzt / grn_ri_goflzt
    #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    volume_grnha_goflzt = volume_grnha_goflzt * mask_greenfeed_exists_fzt[:, na,...]
    return (me_cons_grnha_vgoflzt, volume_grnha_goflzt, foo_start_grnha_oflzt, foo_end_grnha_goflzt
           , senesce_period_grnha_goflzt, senesce_eos_grnha_goflzt, dmd_sward_end_grnha_goflzt, pgr_grnha_goflzt
           , foo_endprior_grnha_goflzt, cons_grnha_t_goflzt, foo_ave_grnha_goflzt, dmd_diet_grnha_goflzt)


def f_senescence(senesce_period_grnha_goflzt, senesce_eos_grnha_goflzt, dry_decay_period_fzt, dmd_sward_end_grnha_goflzt
                 , i_grn_dmd_senesce_redn_fzt, dry_dmd_dfzt, mask_greenfeed_exists_fzt):
    ## senescence from green to dry - green, total senescence for the period (available in the next period)
    ## the pasture that senesces at the eos is assumed to be senescing at the end of the growth period and doesn't decay
    ## the pasture that senesces during the period decays prior to being transferred
    ## the senesced feed that is available to stock is that which senesces at the end of the growing season (i.e. not during the growing season)
    senesce_total_grnha_goflzt = senesce_eos_grnha_goflzt + senesce_period_grnha_goflzt * (1 - dry_decay_period_fzt[:, na, ...])
    grn_dmd_senesce_goflzt = dmd_sward_end_grnha_goflzt + i_grn_dmd_senesce_redn_fzt[:, na, ...]
    # senescence to high pool. np.clip reduces the range of the dmd to the range of dmd in the dry feed pools
    senesce_propn_h_goflzt = np.clip((grn_dmd_senesce_goflzt - dry_dmd_dfzt[0,:, na, :])
                                      / (dry_dmd_dfzt[1,:, na,:] - dry_dmd_dfzt[0,:, na,:]), 0, 1)
    senesce_propn_l_dgoflzt = 1- senesce_propn_h_goflzt                       # senescence to low pool
    senesce_propn_dgoflzt = np.stack([senesce_propn_l_dgoflzt, senesce_propn_h_goflzt])
    senesce_grnha_dgoflzt = senesce_total_grnha_goflzt * senesce_propn_dgoflzt       # ^alternative in one array parameters for the growth/grazing activities: quantity of green that senesces to the high pool
    senesce_grnha_dgoflzt = senesce_grnha_dgoflzt * mask_greenfeed_exists_fzt[:, na, ...]  # apply mask - green pasture only senesces when green pas exists.
    return senesce_grnha_dgoflzt


def f_dry_pasture(cu3, cu4, i_dry_dmd_ave_fzt, i_dry_dmd_range_fzt, i_dry_foo_high_fzt, me_threshold_vfzt, i_me_eff_gainlose_ft, mask_dryfeed_exists_fzt
                  , i_pasture_stage_p6z, ev_is_not_confinement_v, i_legume_zt, n_feed_pools):
    #Consumption & deferment of dry feed.
    ## dry, dmd & foo of feed consumed
    ### do sensitivity adjustment for dry_dmd_input based on increasing/reducing the reduction in dmd from the maximum (starting value)
    dry_dmd_adj_fzt  = (i_dry_dmd_ave_fzt - np.max(i_dry_dmd_ave_fzt, axis=0)) * sen.sam['dry_dmd_decline','annual']
    dry_dmd_high_fzt = np.max(i_dry_dmd_ave_fzt, axis=0) + dry_dmd_adj_fzt + i_dry_dmd_range_fzt/2
    dry_dmd_low_fzt  = np.max(i_dry_dmd_ave_fzt, axis=0) + dry_dmd_adj_fzt - i_dry_dmd_range_fzt/2
    dry_dmd_dfzt     = np.stack((dry_dmd_low_fzt, dry_dmd_high_fzt), axis=0)    # create an array with a new axis 0 by stacking the existing arrays

    dry_foo_high_fzt = i_dry_foo_high_fzt * 3/4
    dry_foo_low_fzt  = i_dry_foo_high_fzt * 1/4                               # assuming half the foo is high quality and the remainder is low quality
    dry_foo_dfzt     = np.stack((dry_foo_low_fzt, dry_foo_high_fzt),axis=0)  # create an array with a new axis 0 by stacking the existing arrays

    ## dry, volume of feed consumed per tonne
    ### adjust foo and calc hf
    pasture_stage_fzt = i_pasture_stage_p6z[...,na]
    dry_foo_dfzt, hf = fsfun.f_foo_convert(cu3, cu4, dry_foo_dfzt, pasture_stage_fzt, i_legume_zt, z_pos=-2)
    ### calc relative availability - note that the equation system used is the one selected for dams in p1
    if uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==0: #csiro function used
        dry_ri_availability_dfzt = fsfun.f_ra_cs(dry_foo_dfzt, hf)
    elif uinp.sheep['i_eqn_used_g1_q1p7'][5,0]==1: #Murdoch function used
        dry_ri_availability_dfzt = fsfun.f_ra_mu(dry_foo_dfzt, hf)

    ### calc relative quality - note that the equation system used is the one selected for dams in p1 - currently only cs function exists
    if uinp.sheep['i_eqn_used_g1_q1p7'][6,0]==0: #csiro function used
        dry_ri_quality_dfzt = fsfun.f_rq_cs(dry_dmd_dfzt, i_legume_zt)
    dry_ri_dfzt = fsfun.f_rel_intake(dry_ri_availability_dfzt, dry_ri_quality_dfzt, i_legume_zt)  #set the minimum RI to 0.05

    dry_volume_t_dfzt = 1000 / dry_ri_dfzt                 # parameters for the dry feed grazing activities: Total volume of the tonne consumed
    dry_volume_t_dfzt = dry_volume_t_dfzt * mask_dryfeed_exists_fzt  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.

    ## dry, ME consumed per kg consumed
    dry_md_dfzt        = fsfun.dmd_to_md(dry_dmd_dfzt)
    dry_md_vdfzt       = np.stack([dry_md_dfzt] * n_feed_pools, axis = 0)
    ## convert to effective quality per tonne
    dry_mecons_t_vdfzt = fsfun.f_effective_mei( 1000                                    # parameters for the dry feed grazing activities: Total ME of the tonne consumed
                               ,          dry_md_vdfzt
                               ,     me_threshold_vfzt[:, na, ...]
                               ,           dry_ri_dfzt
                               , i_me_eff_gainlose_ft[:,na,:])
    dry_mecons_t_vdfzt = dry_mecons_t_vdfzt * mask_dryfeed_exists_fzt  #apply mask - this masks out any green foo at the end of period in periods when green pas doesnt exist.
    dry_mecons_t_vdfzt = dry_mecons_t_vdfzt * ev_is_not_confinement_v[:,na,na,na,na] #me from pasture is 0 in the confinement pool
    return dry_mecons_t_vdfzt, dry_volume_t_dfzt, dry_dmd_dfzt, dry_foo_dfzt


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
    pgr_daily_lt          = np.zeros((n_lmu, n_pasture_types), dtype=float)  #only required if using the ## loop on lmu. The boolean filter method creates the array

    ## loop through the feed periods and calculate the foo at the start of each period
    for f in range(n_feed_periods):
        grn_foo_start_flzt[f,:,:,:] = germination_flzt[f,:,:,:] + grn_foo_end_flzt[f-1,:,:,:]
        dry_foo_start_flzt[f,:,:,:] = dry_foo_end_flzt[f-1,:,:,:]
        ##loop season
        for z in range(n_season):
            ## alternative approach (a1)
            ## for pgr by creating an index using searchsorted (requires an lmu loop). ^ More readable than other but requires pgr_daily matrix to be predefined
            for l in [*range(n_lmu)]: #loop through lmu
                ###find where foo_start fits into the input data
                idx = fun.searchsort_multiple_dim(i_fxg_foo_oflzt[:,f,l,z,:], grn_foo_start_flzt[f,l,z,:], axis_a0=1, axis_v0=0, side='left')
                pgr_daily_lt[l] = (       c_fxg_a_oflzt[idx,f,l,z,:]
                                  +      c_fxg_b_oflzt[idx,f,l,z,:]
                                  * grn_foo_start_flzt[f,l,z,:])
            grn_foo_end_flzt[f,:,z,:] = (              grn_foo_start_flzt[f,:,z,:]
                                         * (1 - grn_senesce_startfoo_fzt[f,z,:])
                                         +                 pgr_daily_lt
                                         *         length_of_periods_fz[f,z]
                                         * (1 -  grn_senesce_pgrcons_fzt[f,z,:])) \
                                        * (1 -     i_grn_senesce_eos_fzt[f,z,:])
            senescence_l = grn_foo_start_flzt[f,:,z,:]  \
                          +    pgr_daily_lt * length_of_periods_fz[f,z]  \
                          -  grn_foo_end_flzt[f,:,z,:]
            dry_foo_end_flzt[f,:,z,:] = dry_foo_start_flzt[f,:,z,:] \
                                    * (1 - dry_decay_fzt[f,z,:]) \
                                    + senescence_l
    return grn_foo_start_flzt, dry_foo_start_flzt

def f1_update_reseeding_foo(foo_grn_reseeding_flrzt, foo_dry_reseeding_flrzt,
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

